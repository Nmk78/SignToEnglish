"""CLI commands owned by `infer`: `s2e translate` and, added in Step 3.1, `s2e stream-replay`.
Heavy imports stay inside the handlers."""

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


def _stream_replay(args: argparse.Namespace) -> int:
    """Replay a video's keypoints through the streaming engine, printing text as it comes."""
    from signtoenglish.common.config import ConfigError, find_repo_root, load_paths
    from signtoenglish.infer.loader import (
        CheckpointRefusedError,
        load_keypoint_config,
        load_model_config,
        load_translator,
    )
    from signtoenglish.infer.offline import OfflineTranslator
    from signtoenglish.infer.streaming import (
        StreamError,
        StreamEvent,
        StreamingEngine,
        iter_raw,
        keypoints_for_replay,
        load_stream_config,
        memory_snapshot,
        offline_decoder,
        replay,
    )
    from signtoenglish.models.boundary import (
        BoundaryError,
        FixedLengthConfig,
        OracleConfig,
        parse_segments,
    )
    from signtoenglish.models.translator import TranslatorError

    try:
        paths = load_paths(find_repo_root())
        model_cfg = load_model_config(args.model_config)
        keypoint_cfg = load_keypoint_config(args.keypoint_config)
        stream_cfg = load_stream_config(args.stream_config)
        if args.segments:
            stream_cfg = stream_cfg.model_copy(
                update={"boundary": OracleConfig(segments=parse_segments(args.segments))}
            )
        elif args.length_ms:
            stream_cfg = stream_cfg.model_copy(
                update={"boundary": FixedLengthConfig(length_ms=args.length_ms)}
            )
        if args.hop_ms:
            stream_cfg = stream_cfg.model_copy(update={"hop_ms": args.hop_ms})
    except (ConfigError, BoundaryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        raw, keypoints_from = keypoints_for_replay(args.video, paths, keypoint_cfg)
    except (StreamError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    translate = None
    model_info: dict[str, object] = {"decoding": "disabled"}
    if not args.no_decode:
        try:
            loaded = load_translator(
                paths, model_cfg, keypoint_cfg, base_checkpoint=args.checkpoint, device=args.device
            )
        except (CheckpointRefusedError, TranslatorError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        translate = offline_decoder(OfflineTranslator(loaded, paths))
        model_info = {
            "decoding": "real",
            "checkpoint": loaded.translator.report.checkpoint_name,
            "checkpoint_id": loaded.checkpoint_id,
            "preprocessing_hash": loaded.preprocessing_hash,
            "device": loaded.translator.device,
        }

    def clock(ms: int) -> str:
        return f"{ms / 1000:8.2f}s"

    def show(event: StreamEvent) -> None:
        if args.json:
            print(json.dumps(event.to_dict(), ensure_ascii=False), flush=True)
            return
        span = f"[{clock(event.start_ms)} -{clock(event.end_ms)}]"
        text = event.text if translate is not None else "(decoding disabled)"
        if event.type == "final":
            print(f"FINAL    {span} #{event.segment_id}  {text}", flush=True)
        elif event.type == "partial":
            if not args.hide_partials:
                print(f"partial  {span} #{event.segment_id}  {text}", flush=True)
        else:
            print(f"{event.type:8} {span} #{event.segment_id}  ({event.reason})", flush=True)

    memory: list[dict[str, float]] = []
    engine = StreamingEngine(stream_cfg, raw.frame_size, model_cfg.translator.max_frames)
    every_ms = int(1000 * args.memory_every_s) if args.memory_every_s else None
    next_sample = [int(raw.timestamps_ms[0]) if len(raw) else 0]

    def sample(t_ms: int) -> None:
        if every_ms is None or t_ms < next_sample[0]:
            return
        next_sample[0] = t_ms + every_ms
        row = {
            "stream_s": round(t_ms / 1000, 1),
            **memory_snapshot(),
            "buffered_frames": float(engine.buffered_frames),
            "pending_finals": float(engine.pending_finals),
        }
        memory.append(row)
        if not args.json:
            print(f"memory   {json.dumps(row)}", flush=True)

    loop_to_ms = int(60_000 * args.loop_to_minutes) if args.loop_to_minutes else None
    result = replay(
        engine,
        iter_raw(raw, loop_to_ms=loop_to_ms),
        translate,
        pace=args.pace,
        assumed_decode_ms=args.assumed_decode_ms,
        on_event=show,
        on_frame=sample,
        keep_events=False,
    )
    summary = {
        "video": str(args.video),
        "keypoints": keypoints_from,
        "frames_in_file": len(raw),
        "model": model_info,
        "stream_config": stream_cfg.model_dump(mode="json"),
        "timing": result.timing_summary(),
        "stats": result.stats.to_dict(),
        "memory": memory,
    }
    if args.report:
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps({"type": "summary", **summary}, ensure_ascii=False))
    else:
        print("--- summary ---", file=sys.stderr)
        print(json.dumps({"timing": summary["timing"], "stats": summary["stats"]}, indent=2),
              file=sys.stderr)  # fmt: skip
    return 0


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    r = subparsers.add_parser(
        "stream-replay",
        help="replay a video through the streaming engine: partial and committed text as it "
        "goes (experimental output)",
    )
    r.add_argument("video", type=Path, help="a video file; its keypoints are cached")
    r.add_argument(
        "--pace",
        choices=["instant", "simulated", "realtime"],
        default="simulated",
        help="instant: decode every hop as if decoding took no time; simulated (default): "
        "live timing on the stream's own clock, without waiting; realtime: also wait, so "
        "text appears at the speed of the video",
    )
    r.add_argument(
        "--segments",
        help="oracle boundaries, START-END,START-END in milliseconds of the video's timestamps",
    )
    r.add_argument("--length-ms", type=int, help="fixed-length boundaries of this length")
    r.add_argument("--hop-ms", type=int, help="override the partial-text hop")
    r.add_argument(
        "--loop-to-minutes",
        type=float,
        help="repeat the video's keypoints until this much stream time has passed",
    )
    r.add_argument(
        "--no-decode",
        action="store_true",
        help="do not load the model: exercises buffer and scheduling only, prints no text",
    )
    r.add_argument(
        "--assumed-decode-ms",
        type=float,
        default=0.0,
        help="with --no-decode: how long each decode is taken to last, in stream time",
    )
    r.add_argument(
        "--memory-every-s", type=float, help="sample process memory this often (stream time)"
    )
    r.add_argument("--hide-partials", action="store_true", help="print committed text only")
    r.add_argument("--json", action="store_true", help="one JSON object per event, then a summary")
    r.add_argument("--report", type=Path, help="write the summary as JSON to this file")
    r.add_argument("--checkpoint", help="base checkpoint name from the model config")
    r.add_argument("--device", choices=["auto", "mps", "cpu"], help="override the config's device")
    r.add_argument("--stream-config", help="default: configs/stream/default.yaml")
    r.add_argument("--model-config", help="default: configs/model/unisign_pose.yaml")
    r.add_argument("--keypoint-config", help="default: configs/preprocess/keypoints.yaml")
    r.set_defaults(func=_stream_replay)

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
