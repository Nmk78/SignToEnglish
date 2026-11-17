"""What the service holds between requests: the loaded model, the pose estimator, the one
decode worker, and the open stream sessions.

- **Loading happens in the background.** The HTTP server answers `/health` at once with
  `not_ready` and the reason ("loading ..."); it turns `ready` only after the translator has
  loaded, the pose models have loaded, and a one-frame self-test has gone through pose
  estimation, normalization, and a real decode. A failure (a bad checkpoint path, a
  preprocessing mismatch, a missing file) leaves the service up and `not_ready` with the
  reason, and `/translate` and `/stream` refuse work with that reason. The old prototype
  started fine and then answered 503 without saying why; this is the fix.
- **One decoder.** Every translation, for every stream and every uploaded file, goes
  through `Service.decode`, which holds one lock: one clip at a time on this machine.
- **One pose estimator.** One `PoseCore` is shared; `pose_lock` serializes its use.
- **Same path as offline.** Uploaded files and stream segments are both cut by
  `infer.streaming.StreamingEngine` and translated by `infer.offline.OfflineTranslator`
  through `offline_decoder`. Nothing is normalized or extracted here.

Nothing in this module makes a network call.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from signtoenglish.common.config import PathsConfig
from signtoenglish.common.logging import get_logger
from signtoenglish.infer.loader import (
    LoadedModel,
    load_keypoint_config,
    load_model_config,
    load_translator,
)
from signtoenglish.infer.offline import OfflineTranslator
from signtoenglish.infer.streaming import (
    DecodeJob,
    PoseCore,
    StreamConfig,
    StreamEvent,
    StreamingEngine,
    iter_raw,
    keypoints_for_replay,
    load_pose_core,
    load_stream_config,
    offline_decoder,
    replay,
)
from signtoenglish.serve.settings import ServeSettings

log = get_logger(__name__)

#: A hop this long never comes due: an uploaded file needs its committed sentences only, so
#: no partial text is decoded for it. Boundaries do not depend on the hop.
_NO_PARTIALS_HOP_MS = 2**62
#: The self-test frame: blank, so no person is in it. Its text is discarded.
_SELF_TEST_SIZE = (640, 480)

ModelStatus = Literal["loading", "ready", "failed"]


class NotReadyError(RuntimeError):
    """The model is not loaded; the message is the reason `/health` reports."""


class UploadError(RuntimeError):
    """An uploaded file could not be read as video; the message says why."""


class Spread:
    """Running statistics of one timing, in milliseconds. Sum, count, and maximum are exact;
    the median and 90th percentile come from the most recent `keep` values."""

    def __init__(self, keep: int = 50_000) -> None:
        self._recent: deque[float] = deque(maxlen=keep)
        self.n = 0
        self.total = 0.0
        self.max = 0.0

    def add(self, value_ms: float) -> None:
        self._recent.append(value_ms)
        self.n += 1
        self.total += value_ms
        self.max = max(self.max, value_ms)

    def summary(self) -> dict[str, float] | None:
        if self.n == 0:
            return None
        recent = np.asarray(self._recent, dtype=np.float64)
        return {
            "n": self.n,
            "mean": round(self.total / self.n, 1),
            "median": round(float(np.median(recent)), 1),
            "p90": round(float(np.percentile(recent, 90)), 1),
            "max": round(self.max, 1),
        }


@dataclass(frozen=True, slots=True)
class Runtime:
    """Everything a request needs once the service is ready."""

    loaded: LoadedModel
    offline: OfflineTranslator
    core: PoseCore
    stream_cfg: StreamConfig
    load_s: float

    @property
    def max_frames(self) -> int:
        return int(self.loaded.model_cfg.translator.max_frames)

    def model_info(self) -> dict[str, Any]:
        translator = self.loaded.translator
        return {
            "checkpoint": translator.report.checkpoint_name,
            "checkpoint_id": self.loaded.checkpoint_id,
            "preprocessing_hash": self.loaded.preprocessing_hash,
            "device": str(translator.device),
            "fine_tuned": self.loaded.fine_tuned_from is not None,
        }


class DecodeClient(Protocol):
    """What the decode worker needs from a stream session (`serve.ws.StreamSession`)."""

    engine: StreamingEngine
    #: True from before the worker asks this session for a job until the job's events have
    #: been handed back. A stopping session waits for it to go False.
    decoding: bool

    def stream_now_ms(self) -> int | None: ...

    def on_decoded(self, job: DecodeJob, decode_ms: float, events: list[StreamEvent]) -> None: ...


class Service:
    def __init__(
        self,
        settings: ServeSettings,
        paths: PathsConfig,
        loader: Callable[[], LoadedModel] | None = None,
    ) -> None:
        self.settings = settings
        self.paths = paths
        self._loader = loader or self._load_from_settings
        self._status: ModelStatus = "loading"
        self._detail = "the service has not started loading the model yet"
        self._runtime: Runtime | None = None
        self._state_lock = threading.Lock()
        self._stopping = threading.Event()
        self._started = False
        self._decode_fn: Callable[[DecodeJob], str] | None = None
        #: One translation at a time, whoever asks.
        self._model_lock = threading.Lock()
        #: One user of the pose estimator at a time.
        self.pose_lock = threading.Lock()
        #: One uploaded file at a time.
        self._upload_lock = threading.Lock()
        self._sessions: list[DecodeClient] = []
        self._sessions_lock = threading.Lock()
        self._work = threading.Event()
        self._threads: list[threading.Thread] = []

    # --- life cycle --------------------------------------------------------------------------

    def start(self) -> None:
        """Begin loading in the background and start the decode worker. Returns at once."""
        if self._started:
            return
        self._started = True
        self._stopping.clear()
        with self._state_lock:
            self._status, self._detail = "loading", "loading the translation model"
        for target, name in ((self._load, "s2e-load"), (self._decode_loop, "s2e-decode")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        """Stop the decode worker and forget the sessions. The model is released with the
        object; a load still under way finishes in its thread and is discarded."""
        self._stopping.set()
        self._work.set()
        with self._sessions_lock:
            self._sessions.clear()
        for thread in self._threads:
            if thread.name == "s2e-decode":
                thread.join(timeout=30)
        self._threads.clear()
        self._started = False
        with self._state_lock:
            if self._status == "ready":
                self._status, self._detail = "failed", "the service was stopped"
            self._runtime = None
            self._decode_fn = None

    def _load_from_settings(self) -> LoadedModel:
        s = self.settings
        return load_translator(
            self.paths,
            load_model_config(s.translator_config),
            load_keypoint_config(s.keypoint_config),
            base_checkpoint=s.checkpoint,
            fine_tuned=s.fine_tuned,
            device=s.device,
        )

    def _set_detail(self, detail: str) -> None:
        with self._state_lock:
            if self._status == "loading":
                self._detail = detail

    def _load(self) -> None:
        started = time.perf_counter()
        try:
            stream_cfg = load_stream_config(self.settings.stream_config)
            loaded = self._loader()
            self._set_detail("loading the pose models")
            core = load_pose_core(self.paths, loaded.keypoint_cfg)
            offline = OfflineTranslator(loaded, self.paths)
            self._set_detail("running the one-frame self-test")
            self._self_test(core, offline)
        except Exception as exc:  # whatever went wrong is the reason /health reports
            log.error("model load failed: %s: %s", type(exc).__name__, exc)
            with self._state_lock:
                self._status = "failed"
                self._detail = f"{type(exc).__name__}: {exc}"
            return
        runtime = Runtime(loaded, offline, core, stream_cfg, time.perf_counter() - started)
        with self._state_lock:
            if self._stopping.is_set():
                return
            self._runtime = runtime
            self._decode_fn = offline_decoder(offline)
            self._status = "ready"
            self._detail = (
                f"model loaded in {runtime.load_s:.1f} s; one-frame self-test passed; "
                f"boundary detector: {stream_cfg.boundary.kind}"
            )
        log.info("ready: %s %s", runtime.model_info(), self._detail)

    @staticmethod
    def _self_test(core: PoseCore, offline: OfflineTranslator) -> None:
        """One blank frame through pose estimation, normalization, and a real decode. It
        proves the whole path runs on this machine; the text means nothing and is dropped."""
        width, height = _SELF_TEST_SIZE
        frame = core.process(np.zeros((height, width, 3), dtype=np.uint8), 0)
        result = offline.translate_arrays(
            frame.keypoints[None], frame.scores[None], _SELF_TEST_SIZE, clip_id="self-test"
        )
        if not isinstance(result.text, str):
            raise RuntimeError("self-test: the decoder did not return text")

    # --- state -------------------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._status == "ready"

    def runtime(self) -> Runtime:
        with self._state_lock:
            if self._status != "ready" or self._runtime is None:
                raise NotReadyError(self._detail)
            return self._runtime

    @property
    def session_count(self) -> int:
        with self._sessions_lock:
            return len(self._sessions)

    def health(self) -> dict[str, Any]:
        """The `/health` body (ARCHITECTURE.md 6.3), plus `checkpoint` (the base
        checkpoint's name) and `active_sessions`."""
        with self._state_lock:
            status, detail, runtime = self._status, self._detail, self._runtime
        info = runtime.model_info() if runtime is not None and status == "ready" else {}
        return {
            "status": "ready" if status == "ready" else "not_ready",
            "detail": detail,
            "checkpoint_id": info.get("checkpoint_id"),
            "preprocessing_hash": info.get("preprocessing_hash"),
            "device": info.get("device"),
            "checkpoint": info.get("checkpoint"),
            "active_sessions": self.session_count,
        }

    # --- decoding ----------------------------------------------------------------------------

    def decode(self, job: DecodeJob) -> str:
        """Translate one job. One at a time for the whole service."""
        decode = self._decode_fn
        if decode is None:
            raise NotReadyError(self._detail)
        with self._model_lock:
            return decode(job)

    def register(self, session: DecodeClient) -> None:
        with self._sessions_lock:
            self._sessions.append(session)

    def unregister(self, session: DecodeClient) -> None:
        with self._sessions_lock:
            if session in self._sessions:
                self._sessions.remove(session)

    def wake(self) -> None:
        """A session may have a job now."""
        self._work.set()

    def _decode_loop(self) -> None:
        """The one decode worker: takes the next job of each open session in turn."""
        while not self._stopping.is_set():
            self._work.clear()
            with self._sessions_lock:
                sessions = list(self._sessions)
            worked = False
            for session in sessions:
                session.decoding = True
                try:
                    job = session.engine.next_job()
                    if job is None:
                        continue
                    worked = True
                    started = time.perf_counter()
                    try:
                        text: str | None = self.decode(job)
                        detail = ""
                    except Exception as exc:  # one bad segment must not end the stream
                        text, detail = None, f"{type(exc).__name__}: {exc}"
                    decode_ms = 1000 * (time.perf_counter() - started)
                    if text is None:
                        events = session.engine.fail(job, detail)
                    else:
                        events = session.engine.complete(
                            job, text, decode_ms, now_ms=session.stream_now_ms()
                        )
                    session.on_decoded(job, decode_ms, events)
                finally:
                    session.decoding = False
            if not worked:
                self._work.wait(timeout=0.25)

    # --- uploaded files ----------------------------------------------------------------------

    def translate_file(self, path: Path) -> dict[str, Any]:
        """Translate an uploaded video: the same extraction (through the keypoint cache), the
        same boundary detector, and the same decoder as a stream, on the file's own
        timestamps. Blocking; call it from a worker thread.

        Returns the `/translate` body without the request-level timings."""
        runtime = self.runtime()
        with self._upload_lock:
            started = time.perf_counter()
            try:
                with self.pose_lock:
                    raw, keypoints_from = keypoints_for_replay(
                        path, self.paths, runtime.loaded.keypoint_cfg, core=runtime.core
                    )
            except Exception as exc:  # StreamError carries the reader's own message
                raise UploadError(str(exc)) from exc
            keypoints_ms = 1000 * (time.perf_counter() - started)

            cfg = runtime.stream_cfg.model_copy(update={"hop_ms": _NO_PARTIALS_HOP_MS})
            engine = StreamingEngine(cfg, raw.frame_size, runtime.max_frames)
            result = replay(engine, iter_raw(raw), self.decode, pace="instant")

        errors = [e.reason or "decode failed" for e in result.events if e.type == "error"]
        if errors:
            raise RuntimeError("; ".join(errors))
        finals = result.finals
        translate_ms = sum(result.final_decode_ms)
        video_ms = result.stream_ms
        return {
            "text": " ".join(e.text.strip() for e in finals if e.text.strip()),
            "segments": [
                {
                    "text": e.text,
                    "start_ms": e.start_ms,
                    "end_ms": e.end_ms,
                    "segment_id": e.segment_id,
                    "closed_by": e.closed_by,
                }
                for e in finals
            ],
            # Stretches the engine did not send to the model, and why (for example
            # "no_signing": no hands in view). Not part of the documented response.
            "skipped": [
                {"start_ms": e.start_ms, "end_ms": e.end_ms, "reason": e.reason}
                for e in result.events
                if e.type == "dropped"
            ],
            "timings": {
                "keypoints_ms": round(keypoints_ms, 1),
                "translate_ms": round(translate_ms, 1),
                "video_ms": video_ms,
                "n_frames": len(raw),
                "n_segments": len(finals),
                "real_time_factor": round((keypoints_ms + translate_ms) / video_ms, 3)
                if video_ms
                else 0.0,
            },
            "keypoints": keypoints_from,
            "boundary": runtime.stream_cfg.boundary.kind,
            "model": runtime.model_info(),
        }
