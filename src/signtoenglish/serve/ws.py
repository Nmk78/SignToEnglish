"""`WS /stream`: one streaming session per socket (ARCHITECTURE.md, section 6.3).

    browser frame ──► receive loop ──► FrameQueue (bounded, drops the OLDEST when full)
                                           │
                       pose thread: JPEG decode ─► LiveExtractor.push ─► StreamingEngine.push
                                                                              │ jobs
                       the service's one decode worker ◄─────────────────────┘
                                           │ partial / final text
                       outbox ──► the socket

Protocol, as fixed here where the architecture document is silent (the web app's choices):

- Client, first, JSON: `{"type": "start", "width", "height", "client_fps"}`. Optional
  `"overload": "drop_oldest" | "wait"` (default `drop_oldest`). With `wait` the service
  stops reading the socket while its frame queue is full instead of dropping, which is what
  a file replay wants when it must lose no frame; a camera uses the default.
- Client, binary: 8 bytes, a signed 64-bit little-endian integer of whole milliseconds
  (capture time, or media time for a played file; strictly increasing), then one JPEG whose
  pixel size is the `width` and `height` of the start message.
- Client, JSON: `{"type": "stop"}`. The service finishes the frames it holds, closes the
  open segment, sends the last `final`, then a `summary`, then closes the socket (1000).
- Server, JSON:
  - `{"type": "partial" | "final", "segment_id", "text", "start_ms", "end_ms",
    "latency_ms", "decode_ms"}`, and `"closed_by"` on a final.
  - `{"type": "status", "fps_in", "queue_depth", "rtf", "signing", "frames_received",
    "frames_dropped", "frames_processed", "pose_fps"}` every status interval.
    `frames_dropped` is the number of frames dropped unprocessed since the session began
    because the service was behind.
  - `{"type": "dropped", "segment_id", "start_ms", "end_ms", "reason"}`: a stretch that was
    not translated (too short, no signing in view, or the decoder's backlog). Not in the
    architecture document; a client may ignore it.
  - `{"type": "error", "detail"}`: a refused message, a bad frame, or a failed decode. The
    session goes on unless the socket is then closed.
  - `{"type": "summary", ...}`: once, after `stop`: frame counts, per-stage timings, the
    engine's counters.

`latency_ms` is stream time from the end of the text's frames to the moment the text was
ready, where "now" in stream time is the newest received frame's timestamp plus the wall
time since it arrived. For a final this is the commit delay as the service sees it.

`rtf` is the real-time factor of the busier of the two stages, averaged over the last few
status intervals; under 1.0 the service keeps up:

- pose path: the compute time one frame takes (JPEG decode plus pose estimation) times the
  frames that arrived, per second of stream. Over 1.0 means frames arrive faster than they
  can be processed, and the oldest are being dropped;
- translation: decode seconds per second of stream. It stays under 1.0 by construction (a
  hop that comes due while the decoder is busy is skipped), so it shows how much of the
  decoder is in use.

The two stages run on two threads. The `summary` gives each one, and their sum, separately.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from collections import deque
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from signtoenglish.common.logging import get_logger
from signtoenglish.infer.streaming import DecodeJob, LiveExtractor, StreamEvent, StreamingEngine
from signtoenglish.serve.service import NotReadyError, Runtime, Service, Spread

log = get_logger(__name__)

TIMESTAMP_BYTES = 8
_JPEG_START = b"\xff\xd8"
#: The same kind of error is reported to the client at most this often.
_ERROR_REPEAT_S = 2.0
#: More unsent messages than this and the session is ended: the client is not reading.
_MAX_OUTBOX = 2000

CLOSE_NORMAL = 1000
CLOSE_POLICY = 1008
CLOSE_TRY_LATER = 1013


class FrameError(ValueError):
    """A binary message is not `8 timestamp bytes + JPEG`; the message says how."""


def parse_frame(message: bytes, max_bytes: int | None = None) -> tuple[int, bytes]:
    """Split one binary message into `(t_ms, jpeg bytes)`.

    The timestamp is a signed 64-bit little-endian integer of whole milliseconds."""
    if max_bytes is not None and len(message) > max_bytes:
        raise FrameError(f"frame message of {len(message)} bytes is over the {max_bytes} limit")
    if len(message) < TIMESTAMP_BYTES + 4:
        raise FrameError(
            f"frame message of {len(message)} bytes is too short: expected 8 timestamp bytes "
            "followed by a JPEG"
        )
    t_ms = int.from_bytes(message[:TIMESTAMP_BYTES], "little", signed=True)
    if t_ms < 0:
        raise FrameError(f"negative frame timestamp {t_ms}")
    jpeg = message[TIMESTAMP_BYTES:]
    if jpeg[:2] != _JPEG_START:
        raise FrameError("frame payload is not a JPEG (8 timestamp bytes must come first)")
    return t_ms, jpeg


class StartMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: Literal["start"]
    width: int = Field(gt=0, le=8192)
    height: int = Field(gt=0, le=8192)
    client_fps: float | None = Field(default=None, gt=0)
    overload: Literal["drop_oldest", "wait"] = "drop_oldest"


class FrameQueue[T]:
    """Frames waiting for pose estimation. Bounded: `put` on a full queue drops the OLDEST
    waiting frame and counts it, so the newest signing is what gets processed and memory
    cannot grow. One thread puts, one thread gets."""

    def __init__(self, max_frames: int) -> None:
        if max_frames < 1:
            raise ValueError(f"max_frames must be positive, got {max_frames}")
        self.max_frames = max_frames
        self._items: deque[T] = deque()
        self._cond = threading.Condition()
        self._closed = False
        self.dropped = 0
        self.max_depth = 0

    @property
    def depth(self) -> int:
        return len(self._items)

    @property
    def full(self) -> bool:
        return len(self._items) >= self.max_frames

    def put(self, item: T) -> int:
        """Add the newest frame. Returns how many waiting frames were dropped for it."""
        with self._cond:
            if self._closed:
                return 0
            dropped = 0
            while len(self._items) >= self.max_frames:
                self._items.popleft()
                dropped += 1
            self._items.append(item)
            self.dropped += dropped
            self.max_depth = max(self.max_depth, len(self._items))
            self._cond.notify()
            return dropped

    def get(self) -> T | None:
        """The oldest waiting frame; blocks while empty. None once closed and drained."""
        with self._cond:
            while not self._items and not self._closed:
                self._cond.wait()
            return self._items.popleft() if self._items else None

    def close(self, discard: bool = False) -> None:
        """No more frames. With `discard`, the waiting ones are thrown away uncounted."""
        with self._cond:
            self._closed = True
            if discard:
                self._items.clear()
            self._cond.notify_all()


def event_message(event: StreamEvent) -> dict[str, Any]:
    """An engine event as the JSON object sent to the client."""
    if event.type in ("partial", "final"):
        message: dict[str, Any] = {
            "type": event.type,
            "segment_id": event.segment_id,
            "text": event.text,
            "start_ms": event.start_ms,
            "end_ms": event.end_ms,
            "latency_ms": None if event.latency_ms is None else round(event.latency_ms),
            "decode_ms": None if event.decode_ms is None else round(event.decode_ms, 1),
        }
        if event.type == "final":
            message["closed_by"] = event.closed_by
        return message
    if event.type == "dropped":
        return {
            "type": "dropped",
            "segment_id": event.segment_id,
            "start_ms": event.start_ms,
            "end_ms": event.end_ms,
            "reason": event.reason,
        }
    return {
        "type": "error",
        "detail": event.reason or "decoding failed",
        "segment_id": event.segment_id,
    }


class StreamSession:
    """One open stream: its frame queue, its pose thread, its engine, and its counters."""

    def __init__(
        self,
        service: Service,
        runtime: Runtime,
        start: StartMessage,
        loop: asyncio.AbstractEventLoop,
        outbox: asyncio.Queue[dict[str, Any] | None],
    ) -> None:
        self.service = service
        self.start = start
        self.frame_size = (start.width, start.height)
        self.engine = StreamingEngine(runtime.stream_cfg, self.frame_size, runtime.max_frames)
        self.decoding = False
        self._extractor = LiveExtractor(runtime.core)
        self._loop = loop
        self._outbox = outbox
        self._queue: FrameQueue[tuple[int, bytes, float]] = FrameQueue(
            service.settings.frame_queue_max
        )
        self._room = asyncio.Event()
        self._pose_done = asyncio.Event()
        self._abandoned = False
        self._last_error: dict[str, float] = {}

        self.opened = time.perf_counter()
        self.frames_received = 0
        self.frames_rejected_bad = 0
        self.frames_rejected_non_increasing = 0
        self.frames_thinned_by_rate = 0
        self.frames_processed = 0
        self.first_t_ms: int | None = None
        self._last_t_ms: int | None = None
        self._last_recv = 0.0
        self.t_frame_decode = Spread()
        self.t_pose = Spread()
        self.t_queue_wait = Spread()
        self.t_frame_total = Spread()
        self.t_partial_decode = Spread()
        self.t_final_decode = Spread()
        self.t_commit_delay = Spread()
        self.frames_queued = 0
        self._window: deque[tuple[float, ...]] = deque(maxlen=service.settings.rtf_window_intervals)
        self._tick: tuple[float, ...] = (0.0,) * 7
        self._tick_at = self.opened

        self._thread = threading.Thread(target=self._pose_loop, name="s2e-pose", daemon=True)
        self._thread.start()
        service.register(self)

    # --- to the client -----------------------------------------------------------------------

    def send(self, message: dict[str, Any]) -> None:
        """Queue a message for the socket, from any thread, in call order."""
        if not self._abandoned:
            with contextlib.suppress(RuntimeError):  # the loop has gone away
                self._loop.call_soon_threadsafe(self._outbox.put_nowait, message)

    def error(self, kind: str, detail: str) -> None:
        """Report a problem, but not the same kind more than once every couple of seconds."""
        now = time.perf_counter()
        if now - self._last_error.get(kind, -1e9) >= _ERROR_REPEAT_S:
            self._last_error[kind] = now
            self.send({"type": "error", "detail": detail})

    def _forward(self, events: list[StreamEvent]) -> None:
        for event in events:
            self.send(event_message(event))

    # --- frames in (event loop) --------------------------------------------------------------

    async def receive_frame(self, message: bytes) -> None:
        self.frames_received += 1
        try:
            t_ms, jpeg = parse_frame(message, self.service.settings.max_frame_bytes)
        except FrameError as exc:
            self.frames_rejected_bad += 1
            self.error("frame", str(exc))
            return
        if self._last_t_ms is not None and t_ms <= self._last_t_ms:
            self.frames_rejected_non_increasing += 1
            self.error(
                "order",
                f"frame timestamp {t_ms} does not come after the previous one "
                f"({self._last_t_ms}); the frame was not used",
            )
            return
        if self.first_t_ms is None:
            self.first_t_ms = t_ms
        self._last_t_ms, self._last_recv = t_ms, time.perf_counter()
        if self.start.overload == "wait":
            while self._queue.full and not self._abandoned:
                self._room.clear()
                if self._queue.full:
                    await self._room.wait()
        self.frames_queued += 1
        self._queue.put((t_ms, jpeg, self._last_recv))

    def stream_now_ms(self) -> int | None:
        """Now, in the stream's own time: the newest received frame's timestamp plus the
        wall time since it arrived."""
        if self._last_t_ms is None:
            return None
        return self._last_t_ms + round(1000 * (time.perf_counter() - self._last_recv))

    @property
    def stream_ms(self) -> int:
        if self.first_t_ms is None or self._last_t_ms is None:
            return 0
        return self._last_t_ms - self.first_t_ms

    # --- pose thread -------------------------------------------------------------------------

    def _pose_loop(self) -> None:
        try:
            while (item := self._queue.get()) is not None:
                with contextlib.suppress(RuntimeError):
                    self._loop.call_soon_threadsafe(self._room.set)
                if self._abandoned:
                    break
                try:
                    self._process(*item)
                except Exception as exc:  # one bad frame must not end the stream
                    self.frames_rejected_bad += 1
                    self.error("pose", f"frame could not be processed: {exc}")
            if not self._abandoned:
                # `stop`: every held frame has been processed; close the open segment.
                self._forward(self.engine.flush())
                self.service.wake()
        finally:
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._pose_done.set)

    def _process(self, t_ms: int, jpeg: bytes, received: float) -> None:
        started = time.perf_counter()
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        decoded = time.perf_counter()
        if image is None:
            self.frames_rejected_bad += 1
            self.error("frame", "frame payload could not be decoded as a JPEG")
            return
        height, width = image.shape[:2]
        if (width, height) != self.frame_size:
            self.frames_rejected_bad += 1
            self.error(
                "size",
                f"frame is {width}x{height} but the start message said "
                f"{self.frame_size[0]}x{self.frame_size[1]}; the frame was not used",
            )
            return
        with self.service.pose_lock:
            pose_started = time.perf_counter()
            keypoints = self._extractor.push(image.astype(np.uint8, copy=False), t_ms)
            pose_ended = time.perf_counter()
        self.t_frame_decode.add(1000 * (decoded - started))
        if keypoints is None:  # thinned to the model frame rate
            self.frames_thinned_by_rate += 1
            return
        self.t_queue_wait.add(1000 * (started - received))
        self.t_pose.add(1000 * (pose_ended - pose_started))
        events = self.engine.push(
            keypoints.t_ms, keypoints.keypoints, keypoints.scores, keypoints.person_found
        )
        self.frames_processed += 1
        self.t_frame_total.add(1000 * (time.perf_counter() - received))
        self._forward(events)
        self.service.wake()

    # --- decode worker -----------------------------------------------------------------------

    def on_decoded(self, job: DecodeJob, decode_ms: float, events: list[StreamEvent]) -> None:
        (self.t_final_decode if job.kind == "final" else self.t_partial_decode).add(decode_ms)
        for event in events:
            if event.type == "final" and event.latency_ms is not None:
                self.t_commit_delay.add(event.latency_ms)
        self._forward(events)

    # --- reports -----------------------------------------------------------------------------

    @staticmethod
    def _factors(
        pose_path_ms: float, translate_ms: float, queued: float, done: float, stream_ms: float
    ) -> dict[str, float | None]:
        """Real-time factors from totals over some stretch of the stream (module docstring)."""
        if stream_ms <= 0:
            return {"pose_demand": None, "translation": None, "bottleneck": None}
        demand = (pose_path_ms / done) * queued / stream_ms if done else 0.0
        translation = translate_ms / stream_ms
        return {
            "pose_demand": round(demand, 3),
            "translation": round(translation, 3),
            "bottleneck": round(max(demand, translation), 3),
        }

    def _totals(self) -> tuple[float, ...]:
        return (
            self.t_frame_decode.total + self.t_pose.total,
            self.t_partial_decode.total + self.t_final_decode.total,
            float(self.frames_queued),
            float(self.t_frame_decode.n),
            float(self.stream_ms),
            float(self.frames_received),
            float(self.frames_processed),
        )

    def status(self) -> dict[str, Any]:
        now = time.perf_counter()
        totals, before = self._totals(), self._tick
        seconds = max(now - self._tick_at, 1e-6)
        self._tick, self._tick_at = totals, now
        delta = tuple(a - b for a, b in zip(totals, before, strict=True))
        self._window.append(delta[:5])
        recent = [sum(column) for column in zip(*self._window, strict=True)]
        return {
            "type": "status",
            "fps_in": round(delta[5] / seconds, 1),
            "queue_depth": self._queue.depth,
            "rtf": self._factors(*recent)["bottleneck"],
            "signing": self.engine.signing,
            "frames_received": self.frames_received,
            "frames_dropped": self._queue.dropped,
            "frames_processed": self.frames_processed,
            "pose_fps": round(delta[6] / seconds, 1),
        }

    def summary(self) -> dict[str, Any]:
        stream_ms = self.stream_ms

        def factor(total_ms: float) -> float | None:
            return round(total_ms / stream_ms, 3) if stream_ms else None

        translate_ms = self.t_partial_decode.total + self.t_final_decode.total
        overall = self._factors(*self._totals()[:5])
        return {
            "type": "summary",
            "frame_size": list(self.frame_size),
            "client_fps": self.start.client_fps,
            "overload": self.start.overload,
            "stream_ms": stream_ms,
            "wall_s": round(time.perf_counter() - self.opened, 2),
            "frames": {
                "received": self.frames_received,
                "dropped_behind": self._queue.dropped,
                "rejected_bad": self.frames_rejected_bad,
                "rejected_non_increasing": self.frames_rejected_non_increasing,
                "thinned_to_model_rate": self.frames_thinned_by_rate,
                "processed": self.frames_processed,
                "max_queue_depth": self._queue.max_depth,
                "queue_limit": self._queue.max_frames,
            },
            "timings_ms": {
                "frame_decode": self.t_frame_decode.summary(),
                "pose": self.t_pose.summary(),
                "queue_wait": self.t_queue_wait.summary(),
                "receive_to_keypoints": self.t_frame_total.summary(),
                "partial_decode": self.t_partial_decode.summary(),
                "final_decode": self.t_final_decode.summary(),
                "commit_delay": self.t_commit_delay.summary(),
            },
            # Compute seconds per second of stream, by stage; `pose_demand` and `bottleneck`
            # are explained in the module docstring. `sum` adds stages that run in parallel.
            "real_time_factor": {
                "frame_decode": factor(self.t_frame_decode.total),
                "pose": factor(self.t_pose.total),
                "translation": factor(translate_ms),
                "sum": factor(self.t_frame_decode.total + self.t_pose.total + translate_ms),
                "pose_demand": overall["pose_demand"],
                "bottleneck": overall["bottleneck"],
            },
            "engine": self.engine.stats.to_dict(),
        }

    # --- ending ------------------------------------------------------------------------------

    async def finish(self, timeout_s: float) -> bool:
        """After `stop`: process the frames still held, close the open segment, and wait for
        its text. True if everything was delivered within the time allowed."""
        deadline = time.perf_counter() + timeout_s
        self._queue.close()
        try:
            await asyncio.wait_for(self._pose_done.wait(), timeout_s)
        except TimeoutError:
            return False
        # No frame can arrive now, so no new job can appear. `busy` is read before
        # `decoding`: the worker raises `decoding` before it takes a job and lowers it only
        # after the job's events are queued for the socket.
        while self.engine.busy or self.decoding:
            if time.perf_counter() > deadline:
                return False
            await asyncio.sleep(0.01)
        return True

    def release(self) -> None:
        """End of the session, however it ended: stop the pose thread, drop held frames, and
        take the engine away from the decode worker. Safe to call twice."""
        self._abandoned = True
        self._queue.close(discard=True)
        self.service.unregister(self)
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._room.set)


def _origin_allowed(ws: WebSocket, allowed: tuple[str, ...]) -> bool:
    """Browsers always send Origin on a WebSocket; other clients (tests, `ws_replay`) do not."""
    origin = ws.headers.get("origin")
    return origin is None or "*" in allowed or origin in allowed


async def _send_loop(ws: WebSocket, outbox: asyncio.Queue[dict[str, Any] | None]) -> None:
    """The only writer to the socket. Ends at the None marker; after a send has failed (the
    client went away) it keeps draining and sends nothing."""
    alive = True
    while (message := await outbox.get()) is not None:
        if not alive:
            continue
        try:
            await ws.send_text(json.dumps(message, ensure_ascii=False))
        except Exception:  # the client went away mid-send
            alive = False


async def _status_loop(session: StreamSession, interval_s: float) -> None:
    while True:
        await asyncio.sleep(interval_s)
        session.send(session.status())


async def stream_endpoint(ws: WebSocket, service: Service) -> None:
    """Serve one `/stream` socket from the first message to the close."""
    settings = service.settings
    if not _origin_allowed(ws, settings.cors_origins):
        await ws.close(CLOSE_POLICY)
        return
    await ws.accept()

    async def refuse(detail: str) -> None:
        await ws.send_text(json.dumps({"type": "error", "detail": detail}))
        await ws.close(CLOSE_TRY_LATER)

    try:
        runtime = service.runtime()
    except NotReadyError as exc:
        await refuse(f"The translation service is not ready: {exc}")
        return
    if service.session_count >= settings.max_sessions:
        await refuse(
            f"The translation service already has {settings.max_sessions} open stream(s); "
            "close one and try again."
        )
        return

    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    sender = asyncio.create_task(_send_loop(ws, outbox))
    status_task: asyncio.Task[None] | None = None
    session: StreamSession | None = None
    close_code = CLOSE_NORMAL
    close_reason = ""

    def say(detail: str) -> None:
        loop.call_soon_threadsafe(outbox.put_nowait, {"type": "error", "detail": detail})

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1005))
            if outbox.qsize() > _MAX_OUTBOX:
                close_code, close_reason = CLOSE_POLICY, "client is not reading"
                break
            data = message.get("bytes")
            if data is not None:
                if session is None:
                    say("Frame received before the start message; it was not used.")
                else:
                    await session.receive_frame(data)
                continue
            try:
                body = json.loads(message.get("text") or "")
            except json.JSONDecodeError:
                say("Text message was not JSON.")
                continue
            kind = body.get("type") if isinstance(body, dict) else None
            if kind == "start":
                if session is not None:
                    say("This stream has already started; the second start message was ignored.")
                    continue
                try:
                    start = StartMessage.model_validate(body)
                except ValidationError as exc:
                    problems = "; ".join(
                        f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
                    )
                    say(f"Bad start message: {problems}")
                    continue
                session = StreamSession(service, runtime, start, loop, outbox)
                status_task = asyncio.create_task(
                    _status_loop(session, settings.status_interval_ms / 1000)
                )
            elif kind == "stop":
                if session is not None:
                    complete = await session.finish(settings.stop_timeout_s)
                    session.send(session.status())
                    session.send({**session.summary(), "complete": complete})
                    log.info("stream stopped: %s", json.dumps(session.summary()["frames"]))
                close_reason = "stopped"
                break
            else:
                say(f"Unknown message type {json.dumps(kind)}.")
    except WebSocketDisconnect:
        if session is not None:
            log.info("stream disconnected after %d frames", session.frames_received)
    finally:
        if status_task is not None:
            status_task.cancel()
        if session is not None:
            session.release()
        # The end marker is queued behind every message already handed over, so the socket
        # closes after the last `final` and the summary.
        loop.call_soon_threadsafe(outbox.put_nowait, None)
        await sender
        if ws.client_state == WebSocketState.CONNECTED:
            with contextlib.suppress(Exception):
                await ws.close(close_code, close_reason)
