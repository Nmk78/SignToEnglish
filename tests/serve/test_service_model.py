"""The service with the real model, the real pose estimator, and real video.

One application for the whole module, built by `create_app` and driven through the test
client and, for `scripts/ws_replay.py`, through a real uvicorn server on a local port. The
translator is the test session's shared instance (OpenASL checkpoint, CPU), handed in
through `create_app(loader=...)` so that only one model is in memory on this machine;
everything after the load is the service's own code: pose models, self-test, readiness,
decode worker, sessions.

Replay material is local and git-ignored: the two fixture clips and a public-domain
102-second video of continuous ASL (`docs/DATA_LICENSES.md`). There are no continuous takes
with marked sentence ends yet, so "one segment per marked sentence" is NOT tested here
(Step 3.2). With the fixed-length detector the segments are arbitrary windows, and nothing
in this file says anything about translation quality.

No test asserts a speed. Waits have generous limits and exist only to end a hung test.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from signtoenglish.infer.offline import OfflineTranslator
from signtoenglish.infer.streaming import (
    StreamingEngine,
    iter_raw,
    keypoints_for_replay,
    load_stream_config,
    replay,
)
from signtoenglish.preprocess.extract import LiveExtractor
from signtoenglish.preprocess.schema import RawKeypoints
from signtoenglish.preprocess.video_io import read_frames
from signtoenglish.serve.app import create_app
from signtoenglish.serve.settings import ServeSettings

pytestmark = pytest.mark.model

CONTINUOUS = Path("commons_source") / "COVID-19_Prevention_Tips_in_ASL.webm"
JPEG_QUALITY = 80  # what the web app and ws_replay send
READY_TIMEOUT_S = 600
WAIT_TIMEOUT_S = 120


def _encode(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    assert ok
    return encoded.tobytes()


def _message(t_ms: int, jpeg: bytes) -> bytes:
    return t_ms.to_bytes(8, "little", signed=True) + jpeg


def _until(condition: Callable[[], bool], what: str, timeout_s: float = WAIT_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.02)


def _read_to_close(ws: Any) -> tuple[list[dict[str, Any]], int]:
    """Every message until the service closes the socket, and the close code."""
    messages: list[dict[str, Any]] = []
    while True:
        try:
            messages.append(ws.receive_json())
        except WebSocketDisconnect as closed:
            return messages, closed.code


@pytest.fixture(scope="module")
def app(model_for: Callable[..., Any], paths: Any) -> FastAPI:
    return create_app(ServeSettings(), paths=paths, loader=lambda: model_for("openasl", "cpu"))


@pytest.fixture(scope="module")
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        _until(
            lambda: (
                not test_client.get("/health").json()["detail"].startswith(("loading", "running"))
            ),
            "the model to load",
            READY_TIMEOUT_S,
        )
        yield test_client


@pytest.fixture(scope="module")
def offline(model_for: Callable[..., Any], paths: Any) -> OfflineTranslator:
    return OfflineTranslator(model_for("openasl", "cpu"), paths)


@pytest.fixture(scope="module")
def jpeg_frames(clip_path: Callable[[str], Path]) -> Callable[[str], list[tuple[int, bytes]]]:
    """A fixture clip as the browser would send it: every frame, its own timestamp, JPEG."""
    held: dict[str, list[tuple[int, bytes]]] = {}

    def get(stem: str) -> list[tuple[int, bytes]]:
        if stem not in held:
            held[stem] = [(f.t_ms, _encode(f.image)) for f in read_frames(clip_path(stem))]
        return held[stem]

    return get


@pytest.fixture(scope="module")
def offline_text_of_sent_frames(
    core: Any, offline: OfflineTranslator
) -> Callable[[list[tuple[int, bytes]], tuple[int, int]], str]:
    """Offline inference on exactly the pixels the service receives: decode each JPEG, run
    the shared extraction, and translate all frames as one clip with `infer/offline.py`."""

    def text(frames: list[tuple[int, bytes]], size: tuple[int, int]) -> str:
        extractor = LiveExtractor(core)
        out = []
        for t_ms, jpeg in frames:
            image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            keypoints = extractor.push(image.astype(np.uint8, copy=False), t_ms)
            if keypoints is not None:
                out.append(keypoints)
        return offline.translate_raw(RawKeypoints.from_frames(out, *size)).text

    return text


def _size_of(jpeg: bytes) -> tuple[int, int]:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    return int(image.shape[1]), int(image.shape[0])


# --- /health -------------------------------------------------------------------------------------


def test_health_is_ready_and_names_the_model(client: TestClient, model_for: Any) -> None:
    loaded = model_for("openasl", "cpu")
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert "self-test passed" in body["detail"]
    assert body["checkpoint_id"] == loaded.checkpoint_id
    assert body["preprocessing_hash"] == loaded.preprocessing_hash
    assert body["device"] == "cpu"
    assert body["checkpoint"] == "openasl"


# --- POST /translate -----------------------------------------------------------------------------


@pytest.mark.parametrize("stem", ["hands_visible", "hands_leave"])
def test_uploaded_single_clip_gives_the_offline_text(
    client: TestClient, offline: OfflineTranslator, clip_path: Callable[[str], Path], stem: str
) -> None:
    path = clip_path(stem)
    expected = offline.translate_file(path)
    with path.open("rb") as handle:
        response = client.post("/translate", files={"video": (path.name, handle)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert [s["text"] for s in body["segments"]] == [expected.text]
    assert body["text"] == expected.text.strip()
    segment = body["segments"][0]
    assert segment["start_ms"] >= 0 and segment["end_ms"] > segment["start_ms"]
    assert body["model"]["checkpoint_id"] == offline.loaded.checkpoint_id
    assert body["model"]["preprocessing_hash"] == offline.loaded.preprocessing_hash
    assert body["timings"]["n_frames"] == expected.n_frames
    assert all(isinstance(v, int | float) for v in body["timings"].values())
    assert {"total_ms", "keypoints_ms", "translate_ms"} <= set(body["timings"])


def test_uploaded_multi_sentence_file_gives_the_offline_replay_segments(
    client: TestClient,
    offline: OfflineTranslator,
    clip_path: Callable[[str], Path],
    need_file: Any,
    paths: Any,
    keypoint_cfg: Any,
    core: Any,
    model_cfg: Any,
) -> None:
    """With the fixed-length detector: the upload returns the segments an offline replay of
    the same file cuts, and each one's text is what `infer/offline.py` gives for that slice.
    (One segment per MARKED sentence needs marked takes and the Step 3.2 detector.)"""
    path = need_file(
        clip_path("hands_visible").parent / CONTINUOUS, "public-domain continuous ASL video"
    )
    with path.open("rb") as handle:
        response = client.post("/translate", files={"video": (path.name, handle)})
    assert response.status_code == 200, response.text
    body = response.json()

    raw, _ = keypoints_for_replay(path, paths, keypoint_cfg, core=core)
    engine = StreamingEngine(load_stream_config(), raw.frame_size, model_cfg.translator.max_frames)
    cuts = replay(engine, iter_raw(raw), None, pace="instant")  # boundaries only, no decoding
    expected_spans = [(e.start_ms, e.end_ms) for e in cuts.finals]
    assert len(expected_spans) > 5  # a multi-segment file
    assert [(s["start_ms"], s["end_ms"]) for s in body["segments"]] == expected_spans
    for segment in body["segments"]:
        direct = offline.translate_raw(raw, start_ms=segment["start_ms"], end_ms=segment["end_ms"])
        assert segment["text"] == direct.text
    assert body["text"] == " ".join(
        s["text"].strip() for s in body["segments"] if s["text"].strip()
    )
    assert len(body["skipped"]) == sum(1 for e in cuts.events if e.type == "dropped")
    assert body["timings"]["n_segments"] == len(expected_spans)


def test_unreadable_video_is_422_and_leaves_no_temporary_file(client: TestClient) -> None:
    before = set(Path(tempfile.gettempdir()).glob("s2e-upload-*"))
    response = client.post(
        "/translate", files={"video": ("broken.mp4", b"this is not a video at all" * 100)}
    )
    assert response.status_code == 422
    assert "could not be read as video" in response.json()["detail"]
    empty = client.post("/translate", files={"video": ("empty.mov", b"")})
    assert empty.status_code == 400 and "empty" in empty.json()["detail"]
    assert set(Path(tempfile.gettempdir()).glob("s2e-upload-*")) == before
    assert client.get("/health").json()["status"] == "ready"  # still serving


def test_video_without_signing_returns_no_text(
    client: TestClient, video_writer: Callable[..., Path], tmp_path: Path
) -> None:
    """An empty scene never reaches the model (the engine's guard), so no sentence is invented."""
    frames = [np.zeros((240, 320, 3), dtype=np.uint8)] * 45
    path = video_writer(tmp_path / "blank.mkv", frames, [i * 33 for i in range(45)])
    renamed = path.with_suffix(".mp4")  # the content is what is read, not the name
    path.rename(renamed)
    with renamed.open("rb") as handle:
        response = client.post("/translate", files={"video": ("blank.mp4", handle)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["text"] == "" and body["segments"] == []
    assert [s["reason"] for s in body["skipped"]] == ["no_signing"]


# --- WS /stream ----------------------------------------------------------------------------------


def test_clip_over_the_socket_gives_the_offline_text(
    client: TestClient,
    jpeg_frames: Callable[[str], list[tuple[int, bytes]]],
    offline_text_of_sent_frames: Callable[..., str],
) -> None:
    frames = jpeg_frames("hands_visible")
    size = _size_of(frames[0][1])
    with client.websocket_connect("/stream") as ws:
        ws.send_json(
            {
                "type": "start",
                "width": size[0],
                "height": size[1],
                "client_fps": 30,
                "overload": "wait",
            }
        )
        for t_ms, jpeg in frames:
            ws.send_bytes(_message(t_ms, jpeg))
        ws.send_json({"type": "stop"})
        messages, code = _read_to_close(ws)

    assert code == 1000
    assert not [m for m in messages if m["type"] == "error"]
    finals = [m for m in messages if m["type"] == "final"]
    assert [m["text"] for m in finals] == [offline_text_of_sent_frames(frames, size)]
    final = finals[0]
    assert final["start_ms"] == frames[0][0] and final["end_ms"] == frames[-1][0] + 1
    assert set(final) >= {"type", "segment_id", "text", "start_ms", "end_ms", "latency_ms"}
    # Committed text is last: no partial of a segment follows its final.
    kinds = [m["type"] for m in messages if m["type"] in ("partial", "final")]
    assert kinds[-1] == "final" and kinds.count("final") == 1
    assert all(m["segment_id"] == final["segment_id"] for m in messages if m["type"] == "partial")
    # The summary is the last message before the close and accounts for every frame.
    summary = messages[-1]
    assert summary["type"] == "summary" and summary["complete"] is True
    counts = summary["frames"]
    assert counts["received"] == counts["processed"] == len(frames)
    assert counts["dropped_behind"] == 0 and counts["max_queue_depth"] <= counts["queue_limit"]
    for stage in ("frame_decode", "pose", "final_decode", "receive_to_keypoints"):
        assert summary["timings_ms"][stage]["n"] > 0
    assert summary["engine"]["finals_emitted"] == 1
    _until(lambda: client.get("/health").json()["active_sessions"] == 0, "the session to end")


def test_disconnect_releases_the_session(
    client: TestClient, app: FastAPI, jpeg_frames: Callable[[str], list[tuple[int, bytes]]]
) -> None:
    frames = jpeg_frames("hands_visible")[:20]
    size = _size_of(frames[0][1])
    threads_before = {t.name for t in threading.enumerate()}
    with client.websocket_connect("/stream") as ws:
        ws.send_json({"type": "start", "width": size[0], "height": size[1], "client_fps": 30})
        for t_ms, jpeg in frames:
            ws.send_bytes(_message(t_ms, jpeg))
        _until(lambda: client.get("/health").json()["active_sessions"] == 1, "the session to open")
        # Leave without `stop`, mid-stream.
    _until(lambda: client.get("/health").json()["active_sessions"] == 0, "the session's release")
    assert app.state.service.session_count == 0
    _until(
        lambda: sum(t.name == "s2e-pose" for t in threading.enumerate()) == 0,
        "the session's pose thread to end",
    )
    assert "s2e-decode" in threads_before  # the one decode worker outlives sessions
    assert client.get("/health").json()["status"] == "ready"


def test_falling_behind_drops_the_oldest_frames_and_reports_the_count(client: TestClient) -> None:
    """A burst far faster than pose estimation: the queue stays at its limit, the oldest
    frames are dropped and counted, and every frame is accounted for."""
    jpeg = _encode(np.zeros((480, 640, 3), dtype=np.uint8))
    burst = 300
    with client.websocket_connect("/stream") as ws:
        ws.send_json({"type": "start", "width": 640, "height": 480, "client_fps": 30})
        for i in range(burst):
            ws.send_bytes(_message(1000 + 33 * i, jpeg))
        ws.send_json({"type": "stop"})
        messages, code = _read_to_close(ws)
    assert code == 1000
    summary = messages[-1]
    counts = summary["frames"]
    assert counts["received"] == burst
    assert counts["dropped_behind"] > 0
    assert counts["max_queue_depth"] <= counts["queue_limit"] == ServeSettings().frame_queue_max
    assert counts["received"] == (
        counts["dropped_behind"] + counts["processed"] + counts["thinned_to_model_rate"]
    )
    status = [m for m in messages if m["type"] == "status"][-1]
    assert status["frames_dropped"] == counts["dropped_behind"]
    assert status["queue_depth"] <= counts["queue_limit"]
    assert {"fps_in", "queue_depth", "rtf", "signing"} <= set(status)
    # A blank scene: nothing was sent to the model, so no text was invented.
    assert not [m for m in messages if m["type"] in ("partial", "final")]
    assert summary["engine"]["finals_queued"] == 0 and summary["engine"]["partials_requested"] == 0


def test_protocol_mistakes_get_an_error_and_the_stream_goes_on(
    client: TestClient, jpeg_frames: Callable[[str], list[tuple[int, bytes]]]
) -> None:
    frames = jpeg_frames("hands_visible")[:3]
    size = _size_of(frames[0][1])
    wrong_size = _encode(np.zeros((48, 64, 3), dtype=np.uint8))
    with client.websocket_connect("/stream") as ws:
        ws.send_bytes(_message(1, frames[0][1]))
        assert "before the start message" in ws.receive_json()["detail"]
        ws.send_text("not json")
        assert "not JSON" in ws.receive_json()["detail"]
        ws.send_json({"type": "dance"})
        assert "Unknown message type" in ws.receive_json()["detail"]
        ws.send_json({"type": "start", "width": 0, "height": 720})
        assert "Bad start message" in ws.receive_json()["detail"]
        ws.send_json({"type": "start", "width": size[0], "height": size[1], "overload": "wait"})
        ws.send_json({"type": "start", "width": size[0], "height": size[1]})
        assert "already started" in ws.receive_json()["detail"]
        ws.send_bytes(b"\x00\x01")
        assert "too short" in ws.receive_json()["detail"]
        ws.send_bytes(_message(100, frames[0][1]))
        ws.send_bytes(_message(100, frames[1][1]))  # timestamp does not advance
        assert "does not come after" in ws.receive_json()["detail"]
        ws.send_bytes(_message(133, wrong_size))
        ws.send_bytes(_message(166, frames[2][1]))
        ws.send_json({"type": "stop"})
        messages, code = _read_to_close(ws)
    assert code == 1000
    assert any("but the start message said" in m.get("detail", "") for m in messages)
    counts = messages[-1]["frames"]
    assert counts["received"] == 5
    assert counts["rejected_bad"] == 2 and counts["rejected_non_increasing"] == 1
    assert counts["processed"] == 2


# --- scripts/ws_replay.py against a real server ---------------------------------------------------


@pytest.fixture(scope="module")
def server_port(app: FastAPI, client: TestClient) -> Iterator[int]:
    """The same application behind a real uvicorn server on a free local port. `client`
    keeps the application started; this server does not run its lifespan a second time."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _until(lambda: server.started, "uvicorn to start")
    yield server.servers[0].sockets[0].getsockname()[1]
    server.should_exit = True
    thread.join(timeout=30)


def test_ws_replay_script_gives_the_offline_text_and_a_report(
    server_port: int,
    repo_root: Path,
    clip_path: Callable[[str], Path],
    jpeg_frames: Callable[[str], list[tuple[int, bytes]]],
    offline_text_of_sent_frames: Callable[..., str],
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    done = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "ws_replay.py"),
            str(clip_path("hands_leave")),
            "--url",
            f"ws://127.0.0.1:{server_port}/stream",
            "--pace",
            "wait",
            "--send-fps",
            "source",
            "--jpeg-quality",
            str(JPEG_QUALITY),
            "--report",
            str(report_path),
            "--stop-timeout-s",
            "300",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=540,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frames = jpeg_frames("hands_leave")
    assert report["sent"]["frames"] == len(frames)
    assert report["service"]["status"] == "ready" and report["service"]["device"] == "cpu"
    assert report["close"]["closed_by_service"] is True and report["close"]["code"] == 1000
    assert report["received"]["errors"] == [] and report["received"]["frames_dropped"] == 0
    expected = offline_text_of_sent_frames(frames, tuple(report["sent"]["frame_size"]))
    assert [f["text"] for f in report["finals"]] == [expected]
    assert report["service_summary"]["frames"]["processed"] == len(frames)
    # In "wait" pace nothing ran in real time, so no latency figure is reported.
    assert report["measured_on_client"]["commit_delay_ms"] is None and report["targets"] == {}
