"""CLI commands owned by `preprocess`: `s2e extract`, `s2e cache verify`, `s2e bench extract`.

Kept cheap to import; everything heavy is imported inside the handlers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path("configs") / "preprocess" / "keypoints.yaml"


def _setup(args: argparse.Namespace) -> tuple[Any, Any]:
    from signtoenglish.common.config import find_repo_root, load_config, load_paths
    from signtoenglish.preprocess.schema import KeypointConfig

    root = find_repo_root()
    paths = load_paths(root)
    config_path = Path(args.config) if args.config else root / DEFAULT_CONFIG
    cfg = load_config(config_path, KeypointConfig)
    return paths, cfg


def _sources(args: argparse.Namespace, data_root: Path) -> list[Any]:
    from signtoenglish.preprocess.cache import Source, read_manifest_sources

    sources: list[Any] = []
    for manifest in args.manifest or []:
        sources += read_manifest_sources(Path(manifest), data_root)
    for name in getattr(args, "files", None) or []:
        sources.append(Source(path=Path(name).resolve(), expected_sha256=None, origin="file"))
    return sources


def _extract(args: argparse.Namespace) -> int:
    from signtoenglish.preprocess.cache import KeypointCache, ManifestError, get_or_extract
    from signtoenglish.preprocess.extract import PoseCore

    paths, cfg = _setup(args)
    try:
        sources = _sources(args, paths.data_root)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not sources:
        print("error: nothing to extract; give --manifest or video files", file=sys.stderr)
        return 2
    cache = KeypointCache(paths.cache_root, cfg)
    cores: list[Any] = []

    def core() -> Any:  # built on the first cache miss only
        if not cores:
            cores.append(PoseCore(cfg, paths.assets_root))
        return cores[0]

    counts = {"cached": 0, "extracted": 0, "failed": 0}
    frames = 0
    seconds = 0.0
    for i, source in enumerate(sources, start=1):
        _, outcome = get_or_extract(source, cache, core)
        counts[outcome.status] += 1
        if outcome.status == "extracted":
            frames += outcome.frames
            seconds += outcome.seconds
        line = f"[{i}/{len(sources)}] {outcome.status:9s} {source.path}"
        if outcome.status == "failed":
            line += f"  ({outcome.detail}; from {source.origin})"
        elif outcome.status == "extracted":
            line += f"  {outcome.frames} frames in {outcome.seconds:.1f} s"
        print(line, flush=True)
    summary = {
        "preprocessing_hash": cache.hash,
        "cache_dir": str(cache.dir),
        **counts,
        "frames_extracted": frames,
        "fps_including_decode_and_hashing": round(frames / seconds, 2) if seconds else None,
    }
    print(json.dumps(summary))
    return 1 if counts["failed"] else 0


def _verify(args: argparse.Namespace) -> int:
    from signtoenglish.preprocess.cache import KeypointCache, ManifestError, verify_sources

    paths, cfg = _setup(args)
    try:
        sources = _sources(args, paths.data_root)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not sources:
        print("error: give at least one --manifest", file=sys.stderr)
        return 2
    cache = KeypointCache(paths.cache_root, cfg)
    results = verify_sources(sources, cache, check_sources=args.check_sources)
    bad = [r for r in results if r.status != "ok"]
    for r in bad:
        print(
            f"{r.status}: {r.source.path} "
            f"(sha256 {r.source.expected_sha256}; from {r.source.origin})"
        )
    print(
        json.dumps(
            {
                "preprocessing_hash": cache.hash,
                "cache_dir": str(cache.dir),
                "sources": len(results),
                "ok": len(results) - len(bad),
                "missing": sum(r.status == "missing" for r in bad),
                "corrupt": sum(r.status.startswith("corrupt") for r in bad),
                "source_problems": sum(r.status.startswith("source") for r in bad),
            }
        )
    )
    return 1 if bad else 0


def _bench(args: argparse.Namespace) -> int:
    import statistics
    import time

    from signtoenglish.preprocess.extract import PoseCore, RateReducer
    from signtoenglish.preprocess.video_io import read_frames

    paths, cfg = _setup(args)
    if args.provider:
        cfg = cfg.model_copy(update={"pose_provider": args.provider})
    start = time.perf_counter()
    core = PoseCore(cfg, paths.assets_root)
    init_s = time.perf_counter() - start
    reducer = RateReducer(cfg.model_rate_fps, cfg.rate_slack)
    per_frame: list[float] = []
    decoded = 0
    wall = time.perf_counter()
    for frame in read_frames(args.video):
        decoded += 1
        if not reducer.keep(frame.t_ms):
            continue
        t0 = time.perf_counter()
        core.process(frame.image, frame.t_ms)
        per_frame.append(time.perf_counter() - t0)
        if args.frames and len(per_frame) >= args.frames:
            break
    wall = time.perf_counter() - wall
    if len(per_frame) < 4:
        print("error: fewer than 4 frames processed", file=sys.stderr)
        return 2
    steady = per_frame[3:]  # the first frames include warm-up
    print(
        json.dumps(
            {
                "video": str(args.video),
                "providers": core.providers,
                "frames_decoded": decoded,
                "frames_processed": len(per_frame),
                "session_init_s": round(init_s, 2),
                "pose_fps_excluding_decode": round(len(steady) / sum(steady), 1),
                "frame_ms_median": round(1000 * statistics.median(steady), 1),
                "frame_ms_p95": round(1000 * sorted(steady)[int(0.95 * (len(steady) - 1))], 1),
                "fps_including_decode": round(len(per_frame) / wall, 1),
            }
        )
    )
    return 0


def _usage(parser: argparse.ArgumentParser) -> Any:
    def show(args: argparse.Namespace) -> int:
        parser.print_help()
        return 2

    return show


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    config_help = f"keypoint config YAML (default: {DEFAULT_CONFIG} in the repo)"

    p = subparsers.add_parser(
        "extract",
        help="extract keypoints for manifest sources or video files into the cache (resumable)",
    )
    p.add_argument("files", nargs="*", help="video files to extract")
    p.add_argument("--manifest", action="append", help="manifest JSONL file; may be repeated")
    p.add_argument("--config", help=config_help)
    p.set_defaults(func=_extract)

    cache = subparsers.add_parser("cache", help="keypoint cache maintenance")
    cache_sub = cache.add_subparsers(dest="cache_command", metavar="<action>")
    v = cache_sub.add_parser("verify", help="check every manifest source against the cache")
    v.add_argument(
        "--manifest", action="append", required=True, help="manifest JSONL file; may be repeated"
    )
    v.add_argument("--check-sources", action="store_true", help="also re-hash the video files")
    v.add_argument("--config", help=config_help)
    v.set_defaults(func=_verify)
    cache.set_defaults(func=_usage(cache))

    bench = subparsers.add_parser("bench", help="measurements on this machine")
    bench_sub = bench.add_subparsers(dest="bench_command", metavar="<what>")
    b = bench_sub.add_parser(
        "extract", help="pose extraction speed on one video (nothing is cached)"
    )
    b.add_argument("video", type=Path)
    b.add_argument("--frames", type=int, default=0, help="stop after this many processed frames")
    b.add_argument("--provider", choices=["cpu", "coreml"], help="override pose_provider")
    b.add_argument("--config", help=config_help)
    b.set_defaults(func=_bench)
    bench.set_defaults(func=_usage(bench))
