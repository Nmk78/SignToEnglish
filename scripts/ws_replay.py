"""Replay a video file over `WS /stream` the way the web app sends a camera, and write a
latency report. Run with `uv run --frozen python scripts/ws_replay.py VIDEO --report out.json`
against a running `s2e serve`.

What it does: reads the file's frames with their own timestamps (`preprocess.video_io`),
thins them to the send rate by timestamp, encodes each as JPEG, and sends
`8-byte little-endian millisecond timestamp + JPEG` messages, exactly the framing of
`web/src/lib/protocol.ts`. It then sends `stop`, waits for the service to close the socket,
and reports what came back and when.

`--pace realtime` (default) holds each frame back to its own timestamp on the wall clock,
so the service sees what a camera would give it, and may fall behind and drop frames.
`--pace wait` sends as fast as the service takes frames and asks it not to drop any
(`"overload": "wait"`); the text is then the text of every frame, and the latency figures
mean nothing and are left out.

The report measures, on this client's wall clock:

- time between text updates of a segment (partial or final after the previous text);
- commit delay: from sending the last frame of a sentence to receiving its `final`;
- and it copies the service's own summary: per-stage compute times (JPEG decode, pose
  estimation, translation), real-time factors, dropped frames, engine counters.

Everything is measured, nothing is asserted: the targets in PLAN.md Step 3.3 are printed
next to the numbers as met or missed. Output is experimental model text.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from signtoenglish.preprocess.video_io import VideoError, probe, read_frames

#: PLAN.md Step 3.3 metric targets [estimates].
TARGET_RTF = 1.0
TARGET_UPDATE_MS = 1500.0
TARGET_COMMIT_MS = 3000.0


def spread(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "median": round(float(np.median(arr)), 1),
        "p90": round(float(np.percentile(arr, 90)), 1),
        "max": round(float(arr.max()), 1),
    }


def frames_to_send(
    video: Path, send_fps: float | None, loop_to_ms: int | None
) -> Iterator[tuple[int, np.ndarray]]:
    """`(t_ms, BGR image)` of the frames to send: the file's own timestamps, thinned to
    `send_fps` by timestamp (the rule of `web/src/lib/framePump.ts`), and repeated with
    shifted timestamps until `loop_to_ms` of stream time when that is given."""
    interval = None if send_fps is None else 1000.0 / send_fps
    next_due: float | None = None
    shift = 0
    first: int | None = None
    while True:
        last: int | None = None
        step = 33
        for frame in read_frames(video):
            t = frame.t_ms + shift
            if first is None:
                first = t
            if loop_to_ms is not None and t - first >= loop_to_ms:
                return
            if last is not None:
                step = t - last
            last = t
            if interval is not None:
                if next_due is not None and t < next_due - 2:
                    continue
                due = t if next_due is None else next_due
                next_due = due + interval if due + interval >= t else t + interval
            yield t, frame.image
        if loop_to_ms is None or last is None or first is None:
            return
        # The next pass starts one frame interval after this one ended.
        shift = last + step - first


def hardware() -> str:
    def sysctl(name: str) -> str:
        try:
            done = subprocess.run(
                ["/usr/sbin/sysctl", "-n", name],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return done.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    chip = sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine()
    memory = sysctl("hw.memsize")
    gb = f", {int(memory) / 2**30:.0f} GB" if memory.isdigit() else ""
    return f"{chip}{gb}, {platform.system()} {platform.release()}"


def fetch_health(ws_url: str) -> dict[str, Any] | None:
    """`GET /health` of the service behind a ws:// URL; None if it cannot be read."""
    base = ws_url.replace("ws://", "http://", 1).replace("wss://", "https://", 1)
    url = base.rsplit("/", 1)[0] + "/health"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never a proxy
    try:
        with opener.open(url, timeout=5) as response:
            body: Any = json.loads(response.read())
    except urllib.error.HTTPError as exc:  # 503 still carries the body
        try:
            body = json.loads(exc.read())
        except ValueError:
            return None
    except (OSError, ValueError):
        return None
    return body if isinstance(body, dict) else None


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    info = probe(args.video)
    scale = min(1.0, args.max_width / info.width)
    size = (round(info.width * scale), round(info.height * scale))
    send_fps = None if args.send_fps == "source" else float(args.send_fps)
    loop_to_ms = int(60_000 * args.loop_to_minutes) if args.loop_to_minutes else None
    realtime = args.pace == "realtime"

    health = fetch_health(args.url)
    received: list[tuple[float, dict[str, Any]]] = []
    close: dict[str, Any] = {}

    with connect(args.url, max_size=None, open_timeout=20, close_timeout=5) as ws:

        def listen() -> None:
            try:
                for raw in ws:
                    message = json.loads(raw)
                    received.append((time.perf_counter(), message))
                    if args.quiet or message.get("type") not in ("partial", "final", "error"):
                        continue
                    if message["type"] == "error":
                        print(f"ERROR    {message.get('detail')}", flush=True)
                    elif message["type"] == "final" or args.show_partials:
                        begin, end = message["start_ms"] / 1000, message["end_ms"] / 1000
                        span = f"[{begin:8.2f}s -{end:8.2f}s]"
                        label = "FINAL   " if message["type"] == "final" else "partial "
                        print(
                            f"{label} {span} #{message['segment_id']}  {message['text']}",
                            flush=True,
                        )
            except ConnectionClosed:
                pass
            close["code"] = ws.close_code
            close["reason"] = ws.close_reason

        listener = threading.Thread(target=listen, daemon=True)
        listener.start()
        ws.send(
            json.dumps(
                {
                    "type": "start",
                    "width": size[0],
                    "height": size[1],
                    "client_fps": send_fps or info.nominal_fps or 30.0,
                    "overload": "drop_oldest" if realtime else "wait",
                }
            )
        )

        sent: list[tuple[int, float]] = []  # (timestamp on the wire, wall time sent)
        encode_ms: list[float] = []
        late_ms: list[float] = []
        jpeg_bytes = 0
        offset = 0
        t0: int | None = None
        wall0 = time.perf_counter()
        quality = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
        try:
            for t_ms, image in frames_to_send(args.video, send_fps, loop_to_ms):
                if t0 is None:
                    t0 = t_ms
                    wall0 = time.perf_counter()
                    if args.clock == "epoch":
                        offset = round(time.time() * 1000) - t_ms
                started = time.perf_counter()
                if scale < 1.0:
                    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode(".jpg", image, quality)
                if not ok:
                    raise RuntimeError(f"JPEG encoding failed at {t_ms} ms")
                payload = encoded.tobytes()
                encode_ms.append(1000 * (time.perf_counter() - started))
                if realtime:
                    wait = (t_ms - t0) / 1000 - (time.perf_counter() - wall0)
                    if wait > 0:
                        time.sleep(wait)
                    else:
                        late_ms.append(-1000 * wait)
                stamp = t_ms + offset
                ws.send(stamp.to_bytes(8, "little", signed=True) + payload)
                sent.append((stamp, time.perf_counter()))
                jpeg_bytes += len(payload)
        except ConnectionClosed:
            print("error: the service closed the connection while frames were being sent",
                  file=sys.stderr)  # fmt: skip
        else:
            ws.send(json.dumps({"type": "stop"}))
        stop_sent = time.perf_counter()
        listener.join(timeout=args.stop_timeout_s)
        closed_by_service = not listener.is_alive()
        closed_after_s = time.perf_counter() - stop_sent

    report = build_report(args, health, size, sent, received, encode_ms, late_ms, jpeg_bytes)
    report["close"] = {
        "closed_by_service": closed_by_service,
        "seconds_after_stop": round(closed_after_s, 2),
        **close,
    }
    ok = closed_by_service and report["service_summary"] is not None and bool(sent)
    return report, 0 if ok else 1


def build_report(
    args: argparse.Namespace,
    health: dict[str, Any] | None,
    size: tuple[int, int],
    sent: list[tuple[int, float]],
    received: list[tuple[float, dict[str, Any]]],
    encode_ms: list[float],
    late_ms: list[float],
    jpeg_bytes: int,
) -> dict[str, Any]:
    realtime = args.pace == "realtime"
    texts = [(at, m) for at, m in received if m.get("type") in ("partial", "final")]
    finals = [m for _, m in texts if m["type"] == "final"]
    summary = next((m for _, m in received if m.get("type") == "summary"), None)
    statuses = [m for _, m in received if m.get("type") == "status"]

    # Time between text updates of one segment, and from one segment's first frame to its
    # first text, on this client's clock.
    updates: list[float] = []
    last_text: dict[Any, float] = {}
    for at, message in texts:
        previous = last_text.get(message["segment_id"])
        if previous is not None:
            updates.append(1000 * (at - previous))
        last_text[message["segment_id"]] = at

    # Commit delay: from sending the last frame inside a segment to receiving its final.
    stamps = np.array([s for s, _ in sent], dtype=np.int64)
    commit: list[float] = []
    for at, message in texts:
        if message["type"] != "final" or not len(stamps):
            continue
        index = int(np.searchsorted(stamps, message["end_ms"], side="left")) - 1
        if index >= 0:
            commit.append(1000 * (at - sent[index][1]))

    stream_ms = int(stamps[-1] - stamps[0]) if len(stamps) > 1 else 0
    wall_s = sent[-1][1] - sent[0][1] if len(sent) > 1 else 0.0
    factors = (summary or {}).get("real_time_factor") or {}
    measured: dict[str, Any] = {
        "update_interval_ms": spread(updates) if realtime else None,
        "commit_delay_ms": spread(commit) if realtime else None,
        "service_latency_ms": spread(
            [float(m["latency_ms"]) for m in finals if m.get("latency_ms") is not None]
        )
        if realtime
        else None,
    }

    targets: dict[str, Any] = {}
    if realtime and summary is not None:
        # The busier of the two stages (pose path, translation); they run on two threads.
        # `sum` in `by_stage` adds them, for a reader who wants the stricter figure.
        bottleneck = factors.get("bottleneck")
        targets["real_time_factor_under_1.0"] = {
            "measured_bottleneck": bottleneck,
            "by_stage": factors,
            "met": None if bottleneck is None else bottleneck < TARGET_RTF,
            "met_for_sum_of_stages": None
            if factors.get("sum") is None
            else factors["sum"] < TARGET_RTF,
        }
        intervals = measured["update_interval_ms"]
        targets["text_update_at_least_every_1.5_s"] = {
            "measured_ms": intervals,
            "share_within_target": round(float(np.mean(np.asarray(updates) <= TARGET_UPDATE_MS)), 3)
            if updates
            else None,
            "met_at_median": None if intervals is None else intervals["median"] <= TARGET_UPDATE_MS,
            "met_at_p90": None if intervals is None else intervals["p90"] <= TARGET_UPDATE_MS,
            "met_at_max": None if intervals is None else intervals["max"] <= TARGET_UPDATE_MS,
        }
        delays = measured["commit_delay_ms"]
        targets["median_commit_delay_under_3_s"] = {
            "measured_ms": delays,
            "met": None if delays is None else delays["median"] < TARGET_COMMIT_MS,
        }

    return {
        "what": "ws_replay: a video file replayed over WS /stream (experimental model output)",
        "video": str(args.video),
        "url": args.url,
        "pace": args.pace,
        "clock": args.clock,
        "hardware": hardware(),
        "service": health,
        "sent": {
            "frames": len(sent),
            "frame_size": list(size),
            "send_fps_requested": args.send_fps,
            "send_fps_measured": round((len(sent) - 1) / wall_s, 2) if wall_s > 0 else None,
            "jpeg_quality": args.jpeg_quality,
            "mean_jpeg_bytes": round(jpeg_bytes / len(sent)) if sent else None,
            "stream_s": round(stream_ms / 1000, 2),
            "wall_s": round(wall_s, 2),
            "client_encode_ms": spread(encode_ms),
            "frames_sent_late": len(late_ms),
            "worst_lateness_ms": round(max(late_ms), 1) if late_ms else 0.0,
        },
        "received": {
            "partials": len(texts) - len(finals),
            "finals": len(finals),
            "errors": [m.get("detail") for _, m in received if m.get("type") == "error"],
            "segments_not_translated": [m for _, m in received if m.get("type") == "dropped"],
            "max_queue_depth_in_status": max(
                (m.get("queue_depth") or 0 for m in statuses), default=0
            ),
            "frames_dropped": statuses[-1].get("frames_dropped") if statuses else None,
        },
        "measured_on_client": measured,
        "targets": targets,
        "service_summary": summary,
        "finals": [
            {k: m.get(k) for k in ("segment_id", "start_ms", "end_ms", "text", "closed_by")}
            for m in finals
        ],
    }


def print_summary(report: dict[str, Any]) -> None:
    out = sys.stderr
    sent, got = report["sent"], report["received"]
    service = report["service"] or {}
    print("--- ws_replay ---", file=out)
    print(f"hardware: {report['hardware']}", file=out)
    print(
        f"service: {service.get('status')} checkpoint {service.get('checkpoint')} "
        f"id {str(service.get('checkpoint_id'))[:12]} preprocessing "
        f"{service.get('preprocessing_hash')} device {service.get('device')}",
        file=out,
    )
    print(
        f"sent {sent['frames']} frames {sent['frame_size'][0]}x{sent['frame_size'][1]} over "
        f"{sent['stream_s']} s of stream in {sent['wall_s']} s ({report['pace']}); "
        f"received {got['partials']} partials, {got['finals']} finals, "
        f"{len(got['errors'])} errors; frames dropped by the service: {got['frames_dropped']}",
        file=out,
    )
    summary = report["service_summary"]
    if summary is not None:
        print(f"service real-time factors: {json.dumps(summary['real_time_factor'])}", file=out)
        for stage in ("frame_decode", "pose", "partial_decode", "final_decode"):
            print(f"  {stage} ms: {json.dumps(summary['timings_ms'][stage])}", file=out)
    for name, target in report["targets"].items():
        verdicts = {k: v for k, v in target.items() if k.startswith("met")}
        print(f"target {name}: {json.dumps(verdicts)}", file=out)
        print(f"  measured: {json.dumps({k: v for k, v in target.items() if k not in verdicts})}",
              file=out)  # fmt: skip
    print(f"close: {json.dumps(report['close'])}", file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/stream")
    parser.add_argument("--pace", choices=["realtime", "wait"], default="realtime")
    parser.add_argument(
        "--send-fps",
        default="15",
        help='frames per second to send, or "source" for every frame (default 15, the web '
        "app's current placeholder)",
    )
    parser.add_argument("--jpeg-quality", type=int, default=80, help="default 80, as the web app")
    parser.add_argument("--max-width", type=int, default=1280, help="default 1280, as the web app")
    parser.add_argument(
        "--clock",
        choices=["media", "epoch"],
        default="media",
        help="media: the file's timestamps (play as live); epoch: shifted to now (webcam)",
    )
    parser.add_argument(
        "--loop-to-minutes", type=float, help="repeat the file until this much stream time"
    )
    parser.add_argument("--stop-timeout-s", type=float, default=60.0)
    parser.add_argument("--report", type=Path, help="write the report as JSON to this file")
    parser.add_argument("--show-partials", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="do not print text as it arrives")
    args = parser.parse_args(argv)
    if args.send_fps != "source":
        try:
            if float(args.send_fps) <= 0:
                raise ValueError
        except ValueError:
            parser.error('--send-fps must be a positive number or "source"')

    try:
        report, status = run(args)
    except (VideoError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(report)
    return status


if __name__ == "__main__":
    sys.exit(main())
