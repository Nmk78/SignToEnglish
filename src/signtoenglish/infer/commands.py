"""CLI command owned by `infer`: `s2e translate`. Heavy imports stay inside the handler."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _translate(args: argparse.Namespace) -> int:
    from signtoenglish.common.config import ConfigError, find_repo_root, load_paths
    from signtoenglish.infer.loader import (
        CheckpointRefusedError,
        load_keypoint_config,
        load_model_config,
        load_translator,
    )
    from signtoenglish.infer.offline import OfflineError, OfflineTranslator
    from signtoenglish.models.translator import TranslatorError

    try:
        paths = load_paths(find_repo_root())
        model_cfg = load_model_config(args.model_config)
        keypoint_cfg = load_keypoint_config(args.keypoint_config)
        if args.pose_provider:
            keypoint_cfg = keypoint_cfg.model_copy(update={"pose_provider": args.pose_provider})
        loaded = load_translator(
            paths,
            model_cfg,
            keypoint_cfg,
            base_checkpoint=args.checkpoint,
            fine_tuned=args.fine_tuned,
            device=args.device,
        )
    except (ConfigError, CheckpointRefusedError, TranslatorError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    offline = OfflineTranslator(loaded, paths, use_cache=not args.no_cache)
    status = 0
    for video in args.videos:
        try:
            result = offline.translate_file(video, start_ms=args.start_ms, end_ms=args.end_ms)
        except (OfflineError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            status = 1
            continue
        if args.json:
            report = result.to_dict()
            report["model"] = {
                "checkpoint": loaded.translator.report.checkpoint_name,
                "checkpoint_id": loaded.checkpoint_id,
                "preprocessing_hash": loaded.preprocessing_hash,
                "device": loaded.translator.device,
            }
            print(json.dumps(report, ensure_ascii=False))
        else:
            print(result.text)
    return status


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser(
        "translate",
        help="translate one signed sentence per video file (experimental output)",
    )
    p.add_argument("videos", nargs="+", type=Path, help="video files, one signed sentence each")
    p.add_argument(
        "--checkpoint", help="base checkpoint name from the model config (default: its default)"
    )
    p.add_argument(
        "--fine-tuned", type=Path, help="a checkpoint written by this project (runs/<id>/best.pt)"
    )
    p.add_argument("--device", choices=["auto", "mps", "cpu"], help="override the config's device")
    p.add_argument("--start-ms", type=int, help="use frames from this time on (file timestamps)")
    p.add_argument("--end-ms", type=int, help="use frames before this time")
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="extract keypoints without reading or writing the cache",
    )
    p.add_argument(
        "--pose-provider",
        choices=["cpu", "coreml"],
        help="override pose_provider (a separate cache)",
    )
    p.add_argument(
        "--json", action="store_true", help="print one JSON object per video instead of the text"
    )
    p.add_argument(
        "--model-config", help="model config YAML (default: configs/model/unisign_pose.yaml)"
    )
    p.add_argument(
        "--keypoint-config",
        help="keypoint config YAML (default: configs/preprocess/keypoints.yaml)",
    )
    p.set_defaults(func=_translate)
