"""CLI commands owned by `common`. See `signtoenglish.cli` for the `commands.py` contract."""

from __future__ import annotations

import argparse


def _doctor(args: argparse.Namespace) -> int:
    from signtoenglish.common import doctor  # deferred: imports torch and onnxruntime

    return doctor.run(as_json=args.json)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser(
        "doctor", help="report Python, package versions, device, free disk, and ffmpeg"
    )
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.set_defaults(func=_doctor)
