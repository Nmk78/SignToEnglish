"""The service when the model cannot load, and request checks that never reach the model.

These run the real application (`create_app`) through the test client with no model: a
checkpoint path that does not exist is refused before any weights are read. The regression
they guard: the old prototype started fine with a bad model path and then answered 503
from `/translate` without `/health` saying anything was wrong.
"""

from __future__ import annotations

import io
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from signtoenglish.cli import build_parser
from signtoenglish.common.config import PathsConfig, load_paths
from signtoenglish.serve.app import create_app
from signtoenglish.serve.settings import ServeSettings

WEB_ORIGIN = "http://127.0.0.1:3000"


def _wait_until_decided(client: TestClient, timeout_s: float = 120) -> dict[str, Any]:
    """Poll `/health` until loading is over, one way or the other."""
    deadline = time.monotonic() + timeout_s
    while True:
        body: dict[str, Any] = client.get("/health").json()
        if not str(body["detail"]).startswith(("loading", "running", "the service has not")):
            return body
        assert time.monotonic() < deadline, f"still loading after {timeout_s} s: {body}"
        time.sleep(0.05)


@pytest.fixture(scope="module")
def missing_checkpoint(repo_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("runs") / "no-such-run" / "best.pt"


@pytest.fixture(scope="module")
def broken(repo_root: Path, missing_checkpoint: Path) -> Iterator[TestClient]:
    """The real app, pointed at a fine-tuned checkpoint that does not exist."""
    app = create_app(ServeSettings(fine_tuned=missing_checkpoint), paths=load_paths(repo_root))
    with TestClient(app) as client:
        _wait_until_decided(client)
        yield client


def _upload(name: str = "clip.mp4", data: bytes = b"not a video") -> dict[str, Any]:
    return {"video": (name, io.BytesIO(data), "application/octet-stream")}


def test_bad_checkpoint_path_gives_not_ready_with_the_reason(
    broken: TestClient, missing_checkpoint: Path
) -> None:
    response = broken.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert "checkpoint not found" in body["detail"] and str(missing_checkpoint) in body["detail"]
    assert body["checkpoint_id"] is None and body["preprocessing_hash"] is None
    assert body["device"] is None and body["active_sessions"] == 0
    # The fields ARCHITECTURE.md 6.3 names are all present.
    assert {"status", "detail", "checkpoint_id", "preprocessing_hash", "device"} <= set(body)


def test_translate_answers_503_with_the_reason_when_not_ready(
    broken: TestClient, missing_checkpoint: Path
) -> None:
    response = broken.post("/translate", files=_upload())
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "not ready" in detail and "checkpoint not found" in detail


def test_stream_says_why_and_closes_when_not_ready(broken: TestClient) -> None:
    with broken.websocket_connect("/stream") as ws:
        message = ws.receive_json()
        assert message["type"] == "error" and "checkpoint not found" in message["detail"]
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
        assert closed.value.code == 1013
    assert broken.get("/health").json()["active_sessions"] == 0


def test_missing_base_checkpoint_gives_not_ready(repo_root: Path, tmp_path: Path) -> None:
    """An assets folder without the weights: the failure is reported, the service stays up."""
    real = load_paths(repo_root)
    paths = PathsConfig(
        data_root=real.data_root,
        cache_root=tmp_path / "cache",
        runs_root=real.runs_root,
        assets_root=tmp_path / "no-assets",
        third_party_root=tmp_path / "no-third-party",
    )
    with TestClient(create_app(ServeSettings(), paths=paths)) as client:
        body = _wait_until_decided(client)
        assert body["status"] == "not_ready" and str(tmp_path) in body["detail"]
        assert client.post("/translate", files=_upload()).status_code == 503


def test_health_is_not_ready_while_loading(repo_root: Path) -> None:
    """Ready is never reported before the load has finished: here it never finishes."""
    import threading

    release = threading.Event()

    def never_loads() -> Any:
        release.wait(60)
        raise RuntimeError("released by the test")

    app = create_app(ServeSettings(), paths=load_paths(repo_root), loader=never_loads)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"
        assert response.json()["detail"].startswith("loading")
        assert client.post("/translate", files=_upload()).status_code == 503
        release.set()
        assert "released by the test" in _wait_until_decided(client)["detail"]


@pytest.mark.parametrize("name", ["notes.txt", "clip.avi", "clip", "clip.mp4.exe"])
def test_unsupported_file_type_is_415(broken: TestClient, name: str) -> None:
    response = broken.post("/translate", files=_upload(name))
    assert response.status_code == 415
    assert "MP4, WebM, or MOV" in response.json()["detail"]


def test_missing_video_field_is_422(broken: TestClient) -> None:
    response = broken.post("/translate", files={"file": ("clip.mp4", io.BytesIO(b"x"))})
    assert response.status_code == 422
    assert any("video" in map(str, error["loc"]) for error in response.json()["detail"])


def test_cors_allows_the_web_app_and_nobody_else(broken: TestClient) -> None:
    allowed = broken.get("/health", headers={"Origin": WEB_ORIGIN})
    assert allowed.headers["access-control-allow-origin"] == WEB_ORIGIN
    other = broken.get("/health", headers={"Origin": "http://example.com"})
    assert "access-control-allow-origin" not in other.headers
    preflight = broken.options(
        "/translate",
        headers={"Origin": WEB_ORIGIN, "Access-Control-Request-Method": "POST"},
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == WEB_ORIGIN
    # An error answer carries the header too, so the page can read why it failed.
    refused = broken.post("/translate", files=_upload(), headers={"Origin": WEB_ORIGIN})
    assert refused.status_code == 503
    assert refused.headers["access-control-allow-origin"] == WEB_ORIGIN


def test_stream_refuses_a_foreign_browser_origin(broken: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as refused,
        broken.websocket_connect("/stream", headers={"Origin": "http://example.com"}),
    ):
        pass
    assert refused.value.code == 1008


def test_no_interactive_api_pages(broken: TestClient) -> None:
    """They would load scripts from the internet; the demo runs offline."""
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert broken.get(path).status_code == 404


def test_serve_command_is_registered_and_binds_to_localhost_by_default() -> None:
    args = build_parser().parse_args(["serve"])
    assert (args.host, args.port) == ("127.0.0.1", 8000)


def test_serve_reads_no_environment_variables_and_has_no_network_client(repo_root: Path) -> None:
    """Settings come through `ServeSettings` and `load_paths` only, and the service makes no
    outbound calls: nothing under serve/ imports an HTTP client or touches the environment."""
    banned = re.compile(
        r"os\.environ|os\.getenv|getenv\(|import requests|import httpx|urllib\.request|"
        r"import socket\b|huggingface_hub"
    )
    for path in sorted((repo_root / "src" / "signtoenglish" / "serve").glob("*.py")):
        found = banned.findall(path.read_text(encoding="utf-8"))
        assert not found, f"{path.name}: {found}"
