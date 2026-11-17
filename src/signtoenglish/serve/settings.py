"""The service's settings: one typed, frozen object (ARCHITECTURE.md, section 3).

Nothing here reads an environment variable. `s2e serve` builds a `ServeSettings` from its
command-line flags; machine-specific paths still come from `common.config.load_paths`, which
is the one place environment variables are read. The translator, keypoint, and streaming
settings stay in their own YAML files; this object only says which files to load.

Every numeric default below is [assumed] until Wave 3 tunes it against the real web app.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from signtoenglish.common.config import FrozenConfig

#: Where the web app runs in development and at the show (`web/`, port 3000).
DEFAULT_CORS_ORIGINS = ("http://127.0.0.1:3000", "http://localhost:3000")
#: Upload types `POST /translate` accepts, by file extension (ARCHITECTURE.md, section 6.3).
ACCEPTED_VIDEO_SUFFIXES = (".mp4", ".webm", ".mov")


class ServeSettings(FrozenConfig):
    #: Localhost by default: the demo runs on one machine and needs no network.
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=0, le=65535)
    #: Browser origins allowed to call `/health` and `/translate` and to open `/stream`.
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS

    #: Base checkpoint name from the model config; None means its `default_checkpoint`.
    checkpoint: str | None = None
    #: A checkpoint written by this project (`runs/<id>/best.pt`), loaded on top of the base.
    fine_tuned: Path | None = None
    #: None means the model config's `device`.
    device: Literal["auto", "mps", "cpu"] | None = None
    #: Config files; None means the default under `configs/`.
    translator_config: Path | None = None
    keypoint_config: Path | None = None
    stream_config: Path | None = None

    #: A larger upload is refused with 413.
    max_upload_mb: int = Field(default=500, gt=0)
    #: A larger WebSocket frame message is refused.
    max_frame_bytes: int = Field(default=4_000_000, gt=0)
    #: Frames waiting for pose estimation, per session. When a frame arrives and this many
    #: are waiting, the OLDEST waiting frame is dropped and counted (ARCHITECTURE.md 6.3).
    frame_queue_max: int = Field(default=8, ge=1)
    #: How often a `status` message is sent on an open stream.
    status_interval_ms: int = Field(default=1000, gt=0)
    #: Over how many status intervals the `rtf` in a status message is averaged.
    rtf_window_intervals: int = Field(default=5, ge=1)
    #: Open streams at once. They share one pose estimator and one decoder.
    max_sessions: int = Field(default=2, ge=1)
    #: After `stop`: how long to wait for the last frames and the last decode before closing.
    stop_timeout_s: float = Field(default=30.0, gt=0)
