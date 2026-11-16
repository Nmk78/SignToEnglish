"""Prompted recording of sentence takes and continuous takes (ARCHITECTURE.md, section 5).

This collects parallel video and English text for sequence-to-sequence fine-tuning. It is
not a phrase-enrollment system: there is no label set and no classifier, and the model's
output stays free text scored on sentences it has never seen.

`Recorder` is the whole flow (prompt, countdown, capture, accept or redo, manifest row) and
depends on a camera and a window only through two small interfaces, `FrameSource` and
`UI`. `record/camera.py` implements them with the webcam and an OpenCV window; the tests
drive the same `Recorder` with generated frames and scripted keys and check the real files
it writes.

Time comes from frame timestamps only. The recorder never assumes a frame rate.

Keys (as `UI.poll_key` reports them): `space`, `enter`, `esc`, and single letters.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from signtoenglish.common.hashing import sha256_file
from signtoenglish.data.manifest import (
    OWN_RECORDING_LICENSE,
    TAKES,
    UNASSIGNED,
    ClipRow,
    ContinuousRow,
    Segment,
    append_row,
    load_clip_rows,
    load_continuous_rows,
    manifest_path,
)
from signtoenglish.record.prompts import Prompt
from signtoenglish.record.writer import ClipWriter

DATASET = "domain"
State = Literal["prompt", "countdown", "recording", "review"]


class RecorderError(RuntimeError):
    pass


@dataclass(frozen=True)
class Frame:
    image: NDArray[np.uint8]  # BGR
    t_ns: int  # capture time on a monotonic clock


class FrameSource(Protocol):
    def read(self) -> Frame | None:
        """Next frame with its capture time, or None when there are no more frames."""
        ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Overlay:
    """What the window shows on top of the live picture. Never written into the video."""

    state: State
    heading: str
    text: str
    detail: tuple[str, ...] = ()
    keys: str = ""
    big: str | None = None
    elapsed_ms: int | None = None
    notice: str | None = None


class UI(Protocol):
    def show(self, image: NDArray[np.uint8], overlay: Overlay) -> None: ...

    def poll_key(self) -> str | None: ...

    def replay(self, path: Path) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class RecorderSettings:
    countdown_s: float = 3.0
    takes_per_sentence: int = 3  # accepted takes per sentence wanted in this session
    max_take_s: float = 20.0
    min_take_s: float = 0.7
    group_size: int = 4  # sentences per continuous take
    max_continuous_s: float = 120.0
    min_segment_s: float = 0.7  # a second key press sooner than this is ignored
    tail_s: float = 1.0  # recorded after the last sentence end is marked
    shuffle: bool = True  # vary the sentence order between sessions


@dataclass
class Summary:
    accepted: list[str] = field(default_factory=list)  # row ids, in order
    redone: int = 0
    discarded: int = 0
    ended: str = "quit"  # quit | done | count_reached | source_ended
    manifest: Path | None = None


class _SourceEnded(Exception):
    pass


# --------------------------------------------------------------------------------------
# Choosing what to record next
# --------------------------------------------------------------------------------------


def choose_next(
    sentence_ids: Sequence[str],
    session_counts: Mapping[str, int],
    total_counts: Mapping[str, int],
    *,
    target: int,
    skipped: frozenset[str] | set[str] = frozenset(),
    rank: Mapping[str, int] | None = None,
) -> str | None:
    """The sentence to record next, or None when every sentence has `target` accepted
    takes in this session (or was skipped).

    Picks the sentence with the fewest takes in this session, then the fewest takes over
    all sessions, then by `rank` (the session's order). Recording one take at a time this
    way keeps the counts within one of each other at every point.
    """
    best: tuple[int, int, int] | None = None
    choice: str | None = None
    for position, sentence_id in enumerate(sentence_ids):
        if sentence_id in skipped or session_counts.get(sentence_id, 0) >= target:
            continue
        key = (
            session_counts.get(sentence_id, 0),
            total_counts.get(sentence_id, 0),
            rank[sentence_id] if rank is not None else position,
        )
        if best is None or key < best:
            best, choice = key, sentence_id
    return choice


def choose_group(
    sentence_ids: Sequence[str],
    use_counts: Mapping[str, int],
    size: int,
    rng: random.Random,
) -> list[str]:
    """Sentences for one continuous take: the least used so far, in a shuffled order."""
    order = list(sentence_ids)
    rng.shuffle(order)
    order.sort(key=lambda s: use_counts.get(s, 0))  # stable: ties keep the shuffled order
    group = order[: max(1, min(size, len(order)))]
    rng.shuffle(group)
    return group


def session_rank(sentence_ids: Sequence[str], session_id: str, shuffle: bool) -> dict[str, int]:
    order = list(sentence_ids)
    if shuffle:
        random.Random(f"order|{session_id}").shuffle(order)
    return {sentence_id: i for i, sentence_id in enumerate(order)}


# --------------------------------------------------------------------------------------
# The recorder
# --------------------------------------------------------------------------------------


def _check_label(name: str, value: str) -> str:
    if not value or any(not (c.isalnum() or c in "-_.") for c in value) or value.startswith("."):
        raise RecorderError(f"{name} {value!r} may contain only letters, digits, '-', '_', and '.'")
    return value


def _rate(writer: ClipWriter) -> float:
    """Frames per second of a finished take, from its own timestamps."""
    return (writer.n_frames - 1) * 1000 / writer.last_ms if writer.last_ms > 0 else 0.0


class Recorder:
    def __init__(
        self,
        *,
        prompts: Sequence[Prompt],
        source: FrameSource,
        ui: UI,
        data_root: str | Path,
        signer_id: str,
        session_id: str,
        sign_language: str,
        text_language: str,
        tags: Sequence[str] = (),
        settings: RecorderSettings | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        if not prompts:
            raise RecorderError("no sentences to record")
        self.prompts = {p.id: p for p in prompts}
        self.order = [p.id for p in prompts]
        self.source = source
        self.ui = ui
        self.data_root = Path(data_root)
        self.signer_id = _check_label("signer id", signer_id)
        self.session_id = _check_label("session id", session_id)
        self.sign_language = sign_language
        self.text_language = text_language
        self.tags = tuple(tags)
        self.settings = settings or RecorderSettings()
        self.log = log
        self.clip_manifest = manifest_path(self.data_root, DATASET, TAKES)
        self.continuous_manifest = manifest_path(self.data_root, DATASET, TAKES, "continuous")
        self._writer: ClipWriter | None = None
        self._notice: str | None = None

    # ---- shared pieces ----------------------------------------------------------------

    def _next(self) -> Frame:
        frame = self.source.read()
        if frame is None:
            raise _SourceEnded
        return frame

    def _step(self, overlay: Overlay) -> tuple[Frame, str | None]:
        """Read one frame, show it, and return it with the key pressed, if any."""
        frame = self._next()
        self.ui.show(frame.image, overlay)
        return frame, self.ui.poll_key()

    def _take_notice(self) -> str | None:
        notice, self._notice = self._notice, None
        return notice

    def _discard(self) -> None:
        if self._writer is not None:
            self._writer.discard()
            self._writer = None

    def _session_dir(self) -> str:
        return f"{DATASET}/{self.session_id}"

    def _free_take(self, used: int, stem: str) -> tuple[int, str]:
        """First take number above `used` whose file does not exist yet, and its path
        relative to the data root."""
        take = used + 1
        while (self.data_root / self._session_dir() / f"{stem}_take{take:02d}.mp4").exists():
            take += 1
        return take, f"{self._session_dir()}/{stem}_take{take:02d}.mp4"

    def _countdown(self, heading: str, text: str, detail: tuple[str, ...]) -> str:
        """Returns "go", "cancel", or "quit"."""
        start_ns: int | None = None
        while True:
            frame = self._next()
            if start_ns is None:
                start_ns = frame.t_ns
            remaining = self.settings.countdown_s - (frame.t_ns - start_ns) / 1e9
            if remaining <= 0:
                return "go"
            self.ui.show(
                frame.image,
                Overlay(
                    "countdown",
                    heading,
                    text,
                    detail,
                    keys="SPACE or ESC cancel",
                    big=str(math.ceil(remaining)),
                ),
            )
            key = self.ui.poll_key()
            if key == "q":
                return "quit"
            if key in ("space", "esc"):
                return "cancel"

    def _review(self, heading: str, text: str, detail: tuple[str, ...], path: Path) -> str:
        """Returns "accept", "redo", "skip", or "quit"."""
        keys = "A accept   R redo   P replay   N skip   Q quit (drops this take)"
        while True:
            _, key = self._step(Overlay("review", heading, text, detail, keys=keys))
            if key in ("a", "enter"):
                return "accept"
            if key == "r":
                return "redo"
            if key == "n":
                return "skip"
            if key == "q":
                return "quit"
            if key == "p":
                self.ui.replay(path)

    # ---- sentence takes ---------------------------------------------------------------

    def run_clips(self, count: int | None = None) -> Summary:
        """Record sentence takes until every sentence has the session's target, `count`
        takes were accepted, or the signer quits."""
        summary = Summary(manifest=self.clip_manifest)
        existing = load_clip_rows(self.clip_manifest) if self.clip_manifest.is_file() else []
        mine = [r for r in existing if r.signer_id == self.signer_id and r.sentence_id]
        total: Counter[str] = Counter(r.sentence_id or "" for r in mine)
        in_session = [r for r in mine if r.session_id == self.session_id]
        session: Counter[str] = Counter(r.sentence_id or "" for r in in_session)
        last_take: dict[str, int] = {}
        for r in in_session:
            last_take[r.sentence_id or ""] = max(last_take.get(r.sentence_id or "", 0), r.take)
        rank = session_rank(self.order, self.session_id, self.settings.shuffle)
        skipped: set[str] = set()
        target = self.settings.takes_per_sentence

        try:
            while True:
                if count is not None and len(summary.accepted) >= count:
                    summary.ended = "count_reached"
                    break
                sentence_id = choose_next(
                    self.order, session, total, target=target, skipped=skipped, rank=rank
                )
                if sentence_id is None:
                    summary.ended = "done"
                    break
                prompt = self.prompts[sentence_id]
                take, rel_path = self._free_take(
                    last_take.get(sentence_id, 0), f"{sentence_id}_{self.signer_id}"
                )
                done = sum(min(session[s], target) for s in self.order)
                heading = (
                    f"{sentence_id}   take {session[sentence_id] + 1} of {target}   "
                    f"session {done}/{target * len(self.order)}"
                )
                detail = (f"ASL: {prompt.asl}",) if prompt.asl else ()
                outcome = self._one_clip(prompt, heading, detail, take, rel_path)
                if outcome == "quit":
                    summary.ended = "quit"
                    break
                if outcome == "skip":
                    skipped.add(sentence_id)
                    summary.discarded += 1
                elif outcome == "skip_prompt":
                    skipped.add(sentence_id)
                elif outcome == "redo":
                    summary.redone += 1
                elif outcome == "discarded":
                    summary.discarded += 1
                elif outcome.startswith("accepted:"):
                    summary.accepted.append(outcome.removeprefix("accepted:"))
                    session[sentence_id] += 1
                    total[sentence_id] += 1
                    last_take[sentence_id] = take
        except _SourceEnded:
            summary.ended = "source_ended"
        finally:
            self._discard()
        return summary

    def _one_clip(
        self, prompt: Prompt, heading: str, detail: tuple[str, ...], take: int, rel_path: str
    ) -> str:
        settings = self.settings
        while True:
            frame, key = self._step(
                Overlay(
                    "prompt",
                    heading,
                    prompt.text,
                    detail,
                    keys="SPACE start   N skip this sentence   Q quit",
                    notice=self._take_notice(),
                )
            )
            if key == "q":
                return "quit"
            if key == "n":
                return "skip_prompt"
            if key != "space":
                continue
            # The encoder is opened now, so that its start-up pause falls in the countdown.
            writer = self._writer = ClipWriter(self.data_root / rel_path)
            writer.prepare(frame.image)
            started = self._countdown(heading, prompt.text, detail)
            if started == "quit":
                self._discard()
                return "quit"
            if started == "cancel":
                self._discard()
                continue
            break

        stop_reason = ""
        while True:
            frame = self._next()
            writer.write(frame.image, frame.t_ns)
            self.ui.show(
                frame.image,
                Overlay(
                    "recording",
                    heading,
                    prompt.text,
                    detail,
                    keys="SPACE stop   R start over   Q quit",
                    elapsed_ms=writer.last_ms,
                ),
            )
            key = self.ui.poll_key()
            if key == "q":
                self._discard()
                return "quit"
            if key in ("r", "esc"):
                self._discard()
                self._notice = "Take dropped. Same sentence again."
                return "redo"
            if key == "space":
                break
            if writer.last_ms >= settings.max_take_s * 1000:
                stop_reason = f"Stopped at {settings.max_take_s:g} s."
                break

        if writer.last_ms < settings.min_take_s * 1000 or writer.n_frames < 2:
            self._discard()
            self._notice = "Too short, take dropped. Same sentence again."
            return "discarded"
        partial = writer.close_partial()
        review_heading = (
            f"{heading}   recorded {writer.last_ms / 1000:.1f} s. {stop_reason}".strip()
        )
        verdict = self._review(review_heading, prompt.text, detail, partial)
        if verdict != "accept":
            self._discard()
            if verdict == "redo":
                self._notice = "Take dropped. Same sentence again."
            return verdict

        path = writer.promote()
        self._writer = None
        row = ClipRow(
            id=f"{self.session_id}_{prompt.id}_{self.signer_id}_t{take:02d}",
            dataset=DATASET,
            split=UNASSIGNED,
            source_path=rel_path,
            source_sha256=sha256_file(path),
            start_ms=None,
            end_ms=None,
            text=prompt.text,
            sentence_id=prompt.id,
            signer_id=self.signer_id,
            session_id=self.session_id,
            take=take,
            sign_language=self.sign_language,
            text_language=self.text_language,
            license=OWN_RECORDING_LICENSE,
            tags=self.tags,
        )
        append_row(self.clip_manifest, row)
        self.log(
            f"saved {rel_path}  {writer.n_frames} frames, {writer.last_ms / 1000:.1f} s, "
            f"{_rate(writer):.1f} frames/s measured"
            + (f", {writer.n_dropped} dropped" if writer.n_dropped else "")
        )
        self._notice = f"Saved {prompt.id} take {take}."
        return f"accepted:{row.id}"

    # ---- continuous takes -------------------------------------------------------------

    def run_continuous(self, count: int | None = None) -> Summary:
        """Record continuous takes: several sentences in a row, with a key press at the
        end of each one. Runs until `count` takes were accepted or the signer quits."""
        summary = Summary(manifest=self.continuous_manifest)
        existing = (
            load_continuous_rows(self.continuous_manifest)
            if self.continuous_manifest.is_file()
            else []
        )
        mine = [r for r in existing if r.signer_id == self.signer_id]
        used: Counter[str] = Counter(
            seg.sentence_id for r in mine for seg in r.segments if seg.sentence_id
        )
        last_take = max((r.take for r in mine if r.session_id == self.session_id), default=0)
        group: list[str] | None = None

        try:
            while True:
                if count is not None and len(summary.accepted) >= count:
                    summary.ended = "count_reached"
                    break
                take, rel_path = self._free_take(last_take, f"continuous_{self.signer_id}")
                if group is None:
                    rng = random.Random(f"group|{self.session_id}|{take}|{sum(used.values())}")
                    group = choose_group(self.order, used, self.settings.group_size, rng)
                outcome = self._one_continuous([self.prompts[s] for s in group], take, rel_path)
                if outcome == "quit":
                    summary.ended = "quit"
                    break
                if outcome == "redo":
                    summary.redone += 1
                elif outcome == "discarded":
                    summary.discarded += 1
                elif outcome in ("skip", "skip_prompt"):
                    if outcome == "skip":
                        summary.discarded += 1
                    for sentence_id in group:  # steer the next group to other sentences
                        used[sentence_id] += 1
                    group = None
                elif outcome.startswith("accepted:"):
                    summary.accepted.append(outcome.removeprefix("accepted:"))
                    used.update(group)
                    last_take = take
                    group = None
        except _SourceEnded:
            summary.ended = "source_ended"
        finally:
            self._discard()
        return summary

    def _one_continuous(self, group: Sequence[Prompt], take: int, rel_path: str) -> str:
        settings = self.settings
        n = len(group)
        heading = f"Continuous take {take}: {n} sentences, hands down between sentences"
        listing = tuple(f"{i + 1}. {p.text}" for i, p in enumerate(group))
        while True:
            frame, key = self._step(
                Overlay(
                    "prompt",
                    heading,
                    "Sign these in order. Press SPACE each time a sentence ends.",
                    listing,
                    keys="SPACE start   N other sentences   Q quit",
                    notice=self._take_notice(),
                )
            )
            if key == "q":
                return "quit"
            if key == "n":
                return "skip_prompt"
            if key != "space":
                continue
            writer = self._writer = ClipWriter(self.data_root / rel_path)
            writer.prepare(frame.image)
            started = self._countdown(heading, group[0].text, listing)
            if started == "quit":
                self._discard()
                return "quit"
            if started == "cancel":
                self._discard()
                continue
            break

        marks: list[int] = []
        flash: str | None = None
        while True:
            frame = self._next()
            writer.write(frame.image, frame.t_ns)
            current = len(marks)
            if current < n:
                text = group[current].text
                detail: tuple[str, ...] = (
                    (f"next: {group[current + 1].text}",) if current + 1 < n else ("last one",)
                )
                step_heading = f"Sentence {current + 1} of {n}"
                keys = "SPACE sentence ended   R start over   Q quit"
            else:
                text, detail, step_heading = "Done. Keep your hands down.", (), "Finishing"
                keys = ""
            self.ui.show(
                frame.image,
                Overlay(
                    "recording",
                    step_heading,
                    text,
                    detail,
                    keys=keys,
                    elapsed_ms=writer.last_ms,
                    notice=flash,
                ),
            )
            flash = None
            key = self.ui.poll_key()
            if key == "q":
                self._discard()
                return "quit"
            if key in ("r", "esc"):
                self._discard()
                self._notice = "Take dropped. Same sentences again."
                return "redo"
            if key == "space" and current < n:
                previous = marks[-1] if marks else 0
                if writer.last_ms - previous < settings.min_segment_s * 1000:
                    flash = "Key press ignored: too soon after the last one."
                else:
                    marks.append(writer.last_ms)
            if len(marks) == n and writer.last_ms >= marks[-1] + settings.tail_s * 1000:
                break
            if writer.last_ms >= settings.max_continuous_s * 1000:
                self._discard()
                self._notice = (
                    f"Stopped at {settings.max_continuous_s:g} s before every sentence end "
                    "was marked. Take dropped."
                )
                return "discarded"

        partial = writer.close_partial()
        verdict = self._review(
            f"Continuous take {take}: {writer.last_ms / 1000:.1f} s, {n} sentence ends marked",
            "Keep this take?",
            listing,
            partial,
        )
        if verdict != "accept":
            self._discard()
            if verdict == "redo":
                self._notice = "Take dropped. Same sentences again."
            return verdict

        path = writer.promote()
        self._writer = None
        starts = [0, *marks[:-1]]
        row = ContinuousRow(
            id=f"{self.session_id}_continuous_{self.signer_id}_t{take:02d}",
            dataset=DATASET,
            split=UNASSIGNED,
            source_path=rel_path,
            source_sha256=sha256_file(path),
            signer_id=self.signer_id,
            session_id=self.session_id,
            take=take,
            sign_language=self.sign_language,
            text_language=self.text_language,
            license=OWN_RECORDING_LICENSE,
            tags=self.tags,
            segments=tuple(
                Segment(sentence_id=p.id, text=p.text, start_ms=start, end_ms=end)
                for p, start, end in zip(group, starts, marks, strict=True)
            ),
        )
        append_row(self.continuous_manifest, row)
        self.log(
            f"saved {rel_path}  {writer.n_frames} frames, {writer.last_ms / 1000:.1f} s, "
            f"{_rate(writer):.1f} frames/s measured, "
            f"sentence ends at {', '.join(f'{m / 1000:.1f}' for m in marks)} s"
        )
        self._notice = f"Saved continuous take {take}."
        return f"accepted:{row.id}"
