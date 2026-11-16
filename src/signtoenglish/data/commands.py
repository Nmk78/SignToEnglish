"""CLI commands owned by `data`: `s2e domain check`, `s2e manifest validate`,
`s2e splits build`. See `signtoenglish.cli` for the `commands.py` contract."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_SENTENCES = "domain/sentences.yaml"


def _domain_check(args: argparse.Namespace) -> int:
    from signtoenglish.common.config import ConfigError
    from signtoenglish.data.domain import check_domain, load_domain

    try:
        domain = load_domain(args.sentences)
    except ConfigError as exc:
        print(f"ERROR {exc}")
        return 1
    report = check_domain(domain)
    quiet_codes = {"asl_missing", "source_missing"} if args.brief else set()
    for issue in report.issues:
        if issue.code not in quiet_codes:
            print(issue)
    n_train = len(domain.with_role("train"))
    n_unseen = len(domain.with_role("unseen_test"))
    print(
        f"{args.sentences}: {len(domain.sentences)} sentences ({n_train} train, "
        f"{n_unseen} unseen_test), {len(domain.vocabulary)} vocabulary entries"
    )
    print(
        f"{len(report.errors)} error(s), {len(report.warnings)} warning(s); "
        f"{len(report.with_code('asl_missing'))} sentence(s) without an ASL version, "
        f"{len(report.with_code('source_missing'))} without a source"
    )
    if not report.ok:
        return 1
    return 1 if args.strict and report.warnings else 0


def _default_manifests(data_root: Path) -> list[Path]:
    from signtoenglish.data.manifest import MANIFEST_DIR

    return sorted((data_root / MANIFEST_DIR).glob("*.jsonl"))


def _manifest_validate(args: argparse.Namespace) -> int:
    from signtoenglish.common.config import ConfigError, load_paths
    from signtoenglish.data.domain import load_domain
    from signtoenglish.data.manifest import validate_manifests

    data_root = Path(args.data_root) if args.data_root else load_paths().data_root
    paths = [Path(p) for p in args.paths] or _default_manifests(data_root)
    if not paths:
        print(f"ERROR no manifests found under {data_root / 'manifests'}")
        return 1
    sentences = None
    if not args.no_sentences:
        try:
            sentences = load_domain(args.sentences).text_by_id()
        except ConfigError as exc:
            print(f"ERROR {exc}")
            return 1
    report = validate_manifests(
        paths,
        data_root,
        check_hash=not args.no_hash,
        check_media=not args.no_media,
        sentences=sentences,
    )
    for issue in report.issues:
        print(issue)
    for path in paths:
        print(f"checked {path}")
    print(
        f"{report.n_rows} row(s) in {report.n_files} file(s): "
        f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)"
    )
    print("VALID" if report.ok else "INVALID")
    return 0 if report.ok else 1


def _splits_build(args: argparse.Namespace) -> int:
    from signtoenglish.common.config import load_paths
    from signtoenglish.data.domain import check_domain, load_domain
    from signtoenglish.data.manifest import (
        TAKES,
        load_clip_rows,
        load_continuous_rows,
        manifest_path,
        validate_manifests,
    )
    from signtoenglish.data.splits import (
        DATASET,
        SplitConfig,
        SplitError,
        build_splits,
        require_unassigned,
        write_splits,
    )

    data_root = Path(args.data_root) if args.data_root else load_paths().data_root
    domain = load_domain(args.sentences)
    domain_report = check_domain(domain)
    if not domain_report.ok:
        for issue in domain_report.errors:
            print(issue)
        print("ERROR the sentence list has errors; fix them before building splits")
        return 1

    clips_path = manifest_path(data_root, DATASET, TAKES)
    continuous_path = manifest_path(data_root, DATASET, TAKES, "continuous")
    sources = [p for p in (clips_path, continuous_path) if p.is_file()]
    if clips_path not in sources:
        print(f"ERROR {clips_path} not found; record takes with `s2e record` first")
        return 1
    report = validate_manifests(sources, data_root, sentences=domain.text_by_id())
    if not report.ok:
        for manifest_issue in report.errors:
            print(manifest_issue)
        print("ERROR the recorded manifests are invalid; fix them before building splits")
        return 1

    cfg = SplitConfig(
        val_unseen_count=args.val_unseen,
        val_takes_per_sentence=args.val_takes,
        test_seen_takes_per_sentence=args.test_seen_takes,
        holdout_session=args.holdout_session,
        seed=args.seed,
    )
    clip_rows = load_clip_rows(clips_path)
    continuous_rows = load_continuous_rows(continuous_path) if continuous_path.is_file() else []
    try:
        require_unassigned([*clip_rows, *continuous_rows])
        result = build_splits(clip_rows, domain, cfg, continuous_rows)
    except SplitError as exc:
        print(f"ERROR {exc}")
        return 1
    for warning in result.warnings:
        print(f"WARNING {warning}")
    print(f"unseen sentences held for validation: {', '.join(result.val_unseen_ids) or '-'}")
    for name, rows in result.clips.items():
        print(
            f"{name:12s} {len(rows):5d} clip(s), "
            f"{len({r.sentence_id for r in rows}):4d} sentence(s)"
        )
    for name, crows in result.continuous.items():
        print(f"continuous {name:5s} {len(crows):3d} take(s)")
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    for path in write_splits(result, data_root):
        print(f"wrote {path}")
    return 0


def _help(parser: argparse.ArgumentParser):  # type: ignore[no-untyped-def]
    def show(_: argparse.Namespace) -> int:
        parser.print_help()
        return 2

    return show


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    domain = subparsers.add_parser("domain", help="check the domain sentence list")
    domain.set_defaults(func=_help(domain))
    domain_sub = domain.add_subparsers(metavar="<action>")
    p = domain_sub.add_parser(
        "check",
        help="validate the sentence list; warns for every sentence without an ASL version "
        "or a source",
    )
    p.add_argument("--sentences", default=DEFAULT_SENTENCES)
    p.add_argument("--brief", action="store_true", help="count missing ASL fields, do not list")
    p.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    p.set_defaults(func=_domain_check)

    manifest = subparsers.add_parser("manifest", help="validate manifests")
    manifest.set_defaults(func=_help(manifest))
    manifest_sub = manifest.add_subparsers(metavar="<action>")
    p = manifest_sub.add_parser(
        "validate",
        help="check manifests: ids, text, bounds, files, content hashes, video timing",
    )
    p.add_argument("paths", nargs="*", help="manifest files (default: all under the data root)")
    p.add_argument("--data-root", default=None, help="default: from configs/paths.local.yaml")
    p.add_argument("--sentences", default=DEFAULT_SENTENCES)
    p.add_argument("--no-sentences", action="store_true", help="skip the sentence-list check")
    p.add_argument("--no-hash", action="store_true", help="skip content hashes")
    p.add_argument("--no-media", action="store_true", help="skip opening the videos")
    p.set_defaults(func=_manifest_validate)

    splits = subparsers.add_parser("splits", help="build the domain splits")
    splits.set_defaults(func=_help(splits))
    splits_sub = splits.add_subparsers(metavar="<action>")
    p = splits_sub.add_parser(
        "build", help="assign recorded takes to train, val, test_seen, test_unseen"
    )
    p.add_argument("--sentences", default=DEFAULT_SENTENCES)
    p.add_argument("--data-root", default=None)
    p.add_argument("--val-unseen", type=int, default=5, help="unseen sentences held for val")
    p.add_argument("--val-takes", type=int, default=1, help="val takes per training sentence")
    p.add_argument("--test-seen-takes", type=int, default=1)
    p.add_argument("--holdout-session", default=None, help="session id kept whole for test_seen")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_splits_build)
