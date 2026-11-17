"""The local service: `GET /health`, `POST /translate`, `WS /stream` (ARCHITECTURE.md 6.3).

`create_app(settings)` builds the application; `s2e serve` runs it with uvicorn. It binds to
localhost by default and makes no outbound network calls. The interactive API pages are
switched off because they load their scripts from the internet.

Carried over from the old prototype's service (REUSE.md, item 6): `/health` that explains
instead of crashing, and an upload handler that never trusts the client's file name and
always deletes its temporary file. Not carried over: enrollment, the conversation store,
gloss rewriting, confidence gates, and scattered environment lookups.

Status codes of `POST /translate` (multipart field `video`):

- 200 `{"text", "segments": [{"text", "start_ms", "end_ms", ...}], "timings", "model", ...}`
- 415 the file name does not end in .mp4, .webm, or .mov
- 503 the model is not loaded; `detail` is the reason `/health` gives
- 413 the upload is over the size limit; 400 it is empty
- 422 the file could not be read as video (or the `video` field is missing)
- 500 a decode failed

`GET /health` answers 200 when ready and 503 when not, with the same body shape either way.
"""

from __future__ import annotations

import contextlib
import tempfile
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from signtoenglish.common.config import PathsConfig, find_repo_root, load_paths
from signtoenglish.common.logging import get_logger
from signtoenglish.infer.loader import LoadedModel
from signtoenglish.serve.service import NotReadyError, Service, UploadError
from signtoenglish.serve.settings import ACCEPTED_VIDEO_SUFFIXES, ServeSettings
from signtoenglish.serve.ws import stream_endpoint

log = get_logger(__name__)

_COPY_CHUNK = 1024 * 1024


def _suffix_of(filename: str | None) -> str | None:
    """The accepted video suffix of an uploaded file's name, or None. Only the suffix of the
    client's name is ever used; the name itself never reaches the file system."""
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in ACCEPTED_VIDEO_SUFFIXES else None


async def _save_upload(upload: UploadFile, suffix: str, max_bytes: int) -> Path:
    """Copy the upload to a temporary file of our own naming. The caller deletes it."""
    size = 0
    with tempfile.NamedTemporaryFile(prefix="s2e-upload-", suffix=suffix, delete=False) as out:
        path = Path(out.name)
        try:
            while chunk := await upload.read(_COPY_CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, "Upload is too large.")
                out.write(chunk)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    if size == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "The uploaded file is empty.")
    return path


def create_app(
    settings: ServeSettings | None = None,
    *,
    paths: PathsConfig | None = None,
    loader: Callable[[], LoadedModel] | None = None,
) -> FastAPI:
    """Build the service. The model starts loading when the application starts up.

    `loader` replaces how the translator is obtained (default: from the settings, through
    `infer.loader.load_translator`). The tests use it to share one loaded model across the
    test session on this 16 GB machine; everything after it is the same code.
    """
    settings = settings or ServeSettings()
    service = Service(settings, paths or load_paths(find_repo_root()), loader)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        service.start()
        try:
            yield
        finally:
            await run_in_threadpool(service.stop)

    app = FastAPI(
        title="SignToEnglish local service",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> JSONResponse:
        body = service.health()
        return JSONResponse(body, status_code=200 if body["status"] == "ready" else 503)

    @app.post("/translate")
    async def translate(request: Request, video: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
        started = time.perf_counter()
        suffix = _suffix_of(video.filename)
        if suffix is None:
            raise HTTPException(
                415, f"Unsupported file {video.filename!r}. Use an MP4, WebM, or MOV video."
            )
        if not service.ready:
            raise HTTPException(
                503, f"The translation service is not ready: {service.health()['detail']}"
            )
        max_bytes = settings.max_upload_mb * 1024 * 1024
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > max_bytes + _COPY_CHUNK:
            raise HTTPException(413, "Upload is too large.")
        path = await _save_upload(video, suffix, max_bytes)
        saved = time.perf_counter()
        try:
            body = await run_in_threadpool(service.translate_file, path)
        except NotReadyError as exc:
            raise HTTPException(503, f"The translation service is not ready: {exc}") from exc
        except UploadError as exc:
            raise HTTPException(422, f"The file could not be read as video: {exc}") from exc
        except Exception as exc:
            log.exception("translate failed")
            raise HTTPException(500, f"Translation failed: {type(exc).__name__}: {exc}") from exc
        finally:
            path.unlink(missing_ok=True)
        body["timings"] = {
            "total_ms": round(1000 * (time.perf_counter() - started), 1),
            "receive_ms": round(1000 * (saved - started), 1),
            **body["timings"],
        }
        return body

    @app.websocket("/stream")
    async def stream(ws: WebSocket) -> None:
        await stream_endpoint(ws, service)

    return app
