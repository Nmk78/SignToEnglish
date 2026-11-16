"""CLI command owned by `record`: `s2e record`. See `signtoenglish.cli` for the contract."""

from __future__ import annotations

import argparse
import datetime
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from signtoenglish.record.recorder import UI, FrameSource

SHOW_CONDITIONS_TAG = "show_conditions"


def session_id_for(label: str, date: datetime.date | None = None) -> str:
    """`<YYYY-MM-DD>_<label>`: the recording date and a label for the conditions."""
    day = date or datetime.date.today()
    return f"{day.isoformat()}_{label}"


def run_record(
    args: argparse.Namespace,
    make_source: Callable[[], FrameSource] | None = None,
    make_ui: Callable[[], UI] | None = None,
    today: datetime.date | None = None,
) -> int:
    """The `s2e record` handler. `make_source` and `make_ui` default to the webcam and the
    OpenCV window; tests pass generated frames and scripted keys instead."""
    from signtoenglish.common.config import ConfigError, load_paths
    from signtoenglish.data.manifest import ManifestError
    from signtoenglish.record.prompts import load_prompts
    from signtoenglish.record.recorder import Recorder, RecorderError, RecorderSettings

    try:
        prompt_list = load_prompts(args.sentences)
    except ConfigError as exc:
        print(f"ERROR {exc}")
        return 1
    prompts = list(prompt_list.prompts)
    if args.role != "all":
        prompts = [p for p in prompts if p.role == args.role]
    if args.ids:
        wanted = [s.strip() for s in args.ids.split(",") if s.strip()]
        known = {p.id for p in prompts}
        unknown = [s for s in wanted if s not in known]
        if unknown:
            print(f"ERROR unknown sentence id(s): {', '.join(unknown)}")
            return 1
        prompts = [p for p in prompts if p.id in set(wanted)]
    if not prompts:
        print("ERROR no sentences match")
        return 1

    data_root = Path(args.data_root) if args.data_root else load_paths().data_root
    session_id = session_id_for(args.session, today)
    tags = list(
        dict.fromkeys([*(args.tag or []), *([SHOW_CONDITIONS_TAG] if args.show_conditions else [])])
    )
    settings = RecorderSettings(
        countdown_s=args.countdown,
        takes_per_sentence=args.takes,
        max_take_s=args.max_seconds,
        group_size=args.group_size,
        shuffle=args.order == "shuffled",
    )

    print(f"session {session_id}, signer {args.signer}, {len(prompts)} sentence(s)")
    if prompt_list.n_without_asl:
        print(
            f"WARNING {prompt_list.n_without_asl} sentence(s) in {args.sentences} have no ASL "
            "version yet. Run `s2e domain check` for the list. Record a sentence only once "
            "you sign it the same way every time (docs/recording_protocol.md)."
        )

    source: FrameSource | None = None
    ui: UI | None = None
    try:
        if make_source is None:
            from signtoenglish.record.camera import CameraSource

            camera = CameraSource(args.camera, args.width, args.height)
            rate = camera.measure_rate()
            print(
                f"camera {args.camera}: {camera.width}x{camera.height}, "
                f"measured {rate:.1f} frames/s"
            )
            if rate < 20:
                print(
                    "WARNING the camera delivers fewer than 20 frames/s; "
                    "add light or close other apps"
                )
            source = camera
        else:
            source = make_source()
        if make_ui is None:
            from signtoenglish.record.camera import CvWindow

            ui = CvWindow()
            print("Click the camera window so it receives your key presses.")
        else:
            ui = make_ui()

        recorder = Recorder(
            prompts=prompts,
            source=source,
            ui=ui,
            data_root=data_root,
            signer_id=args.signer,
            session_id=session_id,
            sign_language=prompt_list.sign_language,
            text_language=prompt_list.text_language,
            tags=tags,
            settings=settings,
        )
        summary = (
            recorder.run_continuous(args.count)
            if args.continuous
            else recorder.run_clips(args.count)
        )
    except (RecorderError, ManifestError) as exc:
        print(f"ERROR {exc}")
        return 1
    finally:
        if ui is not None:
            ui.close()
        if source is not None:
            source.close()

    kind = "continuous take(s)" if args.continuous else "take(s)"
    print(
        f"{len(summary.accepted)} {kind} saved, {summary.redone} redone, "
        f"{summary.discarded} dropped ({summary.ended})"
    )
    if summary.accepted:
        print(f"manifest: {summary.manifest}")
        print("check it with: s2e manifest validate")
    return 0


def _record(args: argparse.Namespace) -> int:
    return run_record(args)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser(
        "record",
        help="record prompted sentence takes, or continuous takes, from the webcam",
        description="Shows a sentence, counts down, records the webcam with real timestamps, "
        "and lets you accept or redo the take. Nothing is uploaded anywhere.",
    )
    p.add_argument("--sentences", default="domain/sentences.yaml")
    p.add_argument("--signer", required=True, help="short id for the person signing")
    p.add_argument(
        "--session",
        required=True,
        help="label for the conditions, e.g. desk-evening; today's date is added in front",
    )
    p.add_argument(
        "--continuous",
        action="store_true",
        help="record several sentences in a row, pressing SPACE at the end of each",
    )
    p.add_argument("--takes", type=int, default=3, help="takes per sentence in this session")
    p.add_argument("--count", type=int, default=None, help="stop after this many saved takes")
    p.add_argument("--group-size", type=int, default=4, help="sentences per continuous take")
    p.add_argument("--role", choices=["all", "train", "unseen_test"], default="all")
    p.add_argument("--ids", default=None, help="comma-separated sentence ids to record")
    p.add_argument("--order", choices=["shuffled", "file"], default="shuffled")
    p.add_argument("--show-conditions", action="store_true", help="tag takes show_conditions")
    p.add_argument("--tag", action="append", help="extra tag for every take (repeatable)")
    p.add_argument("--countdown", type=float, default=3.0, help="seconds")
    p.add_argument("--max-seconds", type=float, default=20.0, help="longest sentence take")
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--data-root", default=None, help="default: from configs/paths.local.yaml")
    p.set_defaults(func=_record)
