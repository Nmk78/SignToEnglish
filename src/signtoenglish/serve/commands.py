"""CLI command owned by `serve`: `s2e serve`. Heavy imports stay inside the handler."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _serve(args: argparse.Namespace) -> int:
    import uvicorn
    from pydantic import ValidationError

    from signtoenglish.common.config import ConfigError, find_repo_root, load_paths
    from signtoenglish.common.logging import setup_logging
    from signtoenglish.serve.app import create_app
    from signtoenglish.serve.settings import DEFAULT_CORS_ORIGINS, ServeSettings

    try:
        settings = ServeSettings(
            host=args.host,
            port=args.port,
            cors_origins=tuple(args.allow_origin) if args.allow_origin else DEFAULT_CORS_ORIGINS,
            checkpoint=args.checkpoint,
            fine_tuned=args.fine_tuned,
            device=args.device,
            translator_config=args.model_config,
            keypoint_config=args.keypoint_config,
            stream_config=args.stream_config,
            frame_queue_max=args.frame_queue_max,
            max_upload_mb=args.max_upload_mb,
        )
        paths = load_paths(find_repo_root())
    except (ValidationError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    setup_logging(args.log_level.upper())
    if settings.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"warning: binding to {settings.host}, not localhost. This is a personal, "
            "non-commercial demo and must not be offered as a hosted service (CLAUDE.md).",
            file=sys.stderr,
        )
    print(
        f"SignToEnglish service on http://{settings.host}:{settings.port} "
        "(experimental output; the model loads in the background, see /health)",
        file=sys.stderr,
    )
    uvicorn.run(
        create_app(settings, paths=paths),
        host=settings.host,
        port=settings.port,
        log_level=args.log_level,
        ws_max_size=settings.max_frame_bytes + 1024,
    )
    return 0


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser(
        "serve",
        help="run the local service: GET /health, POST /translate, WS /stream "
        "(experimental output)",
    )
    p.add_argument("--host", default="127.0.0.1", help="default: 127.0.0.1 (this machine only)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--allow-origin",
        action="append",
        help="browser origin allowed to call the service; repeatable "
        "(default: http://127.0.0.1:3000 and http://localhost:3000)",
    )
    p.add_argument("--checkpoint", help="base checkpoint name from the model config")
    p.add_argument(
        "--fine-tuned", type=Path, help="a checkpoint written by this project (runs/<id>/best.pt)"
    )
    p.add_argument("--device", choices=["auto", "mps", "cpu"], help="override the config's device")
    p.add_argument(
        "--frame-queue-max",
        type=int,
        default=8,
        help="frames that may wait for pose estimation before the oldest is dropped (default 8)",
    )
    p.add_argument("--max-upload-mb", type=int, default=500)
    p.add_argument("--stream-config", type=Path, help="default: configs/stream/default.yaml")
    p.add_argument("--model-config", type=Path, help="default: configs/model/unisign_pose.yaml")
    p.add_argument(
        "--keypoint-config", type=Path, help="default: configs/preprocess/keypoints.yaml"
    )
    p.add_argument("--log-level", choices=["debug", "info", "warning", "error"], default="info")
    p.set_defaults(func=_serve)
