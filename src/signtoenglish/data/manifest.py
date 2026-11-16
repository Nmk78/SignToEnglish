"""Manifests: one JSON object per line (ARCHITECTURE.md, section 4.2).

Two row kinds:

- `ClipRow`, one sentence clip per row, in `data/manifests/<dataset>_<split>.jsonl`.
- `ContinuousRow`, one recording of several sentences per row with a `segments` list, in
  `data/manifests/<dataset>_continuous_<split>.jsonl`.

Rows written by `s2e record` have not been through the split builder yet. They carry
`split: "unassigned"` and live in `domain_takes.jsonl` / `domain_continuous_takes.jsonl`
(`TAKES`). `data/splits.py` turns them into the per-split files.

Validation never stops at the first problem: `validate_manifests` returns every issue it
finds, each with a code, the file, and the line.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from signtoenglish.common.hashing import sha256_file

SPLITS: tuple[str, ...] = ("train", "val", "test_seen", "test_unseen", "test")
UNASSIGNED = "unassigned"
TAKES = "takes"  # file-name slot used for rows that have no split yet
MANIFEST_DIR = "manifests"
OWN_RECORDING_LICENSE = "own-recording"

# A sentence clip shorter or longer than this is suspicious, not invalid.
SHORT_CLIP_MS = 500
LONG_CLIP_MS = 30_000
# Slack when comparing marked bounds with the container's duration.
BOUNDS_TOLERANCE_MS = 50

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^\S+$")

Kind = Literal["clip", "continuous"]
Level = Literal["error", "warning"]


class ManifestError(ValueError):
    """A manifest cannot be read, or a row that is about to be written is invalid."""


class _Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ClipRow(_Row):
    """One sentence clip. Field meanings: ARCHITECTURE.md, section 4.2."""

    id: str
    dataset: str
    split: str
    source_path: str
    source_sha256: str
    start_ms: int | None
    end_ms: int | None
    text: str
    sentence_id: str | None
    signer_id: str
    session_id: str
    take: int
    sign_language: str
    text_language: str
    license: str
    tags: tuple[str, ...] = ()


class Segment(_Row):
    """One sentence inside a continuous take. Times are container times of the source."""

    sentence_id: str | None
    text: str
    start_ms: int
    end_ms: int


class ContinuousRow(_Row):
    """One recording of several sentences in a row, with marked sentence bounds."""

    id: str
    dataset: str
    split: str
    source_path: str
    source_sha256: str
    signer_id: str
    session_id: str
    take: int
    sign_language: str
    text_language: str
    license: str
    tags: tuple[str, ...] = ()
    segments: tuple[Segment, ...]


Row = ClipRow | ContinuousRow


@dataclass(frozen=True)
class Issue:
    level: Level
    code: str
    message: str
    path: str | None = None
    line: int | None = None
    row_id: str | None = None

    def __str__(self) -> str:
        where = self.path or "<rows>"
        if self.line is not None:
            where += f":{self.line}"
        who = f" [{self.row_id}]" if self.row_id else ""
        return f"{self.level.upper()} {self.code} {where}{who}: {self.message}"


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    n_files: int = 0
    n_rows: int = 0

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class VideoProbe:
    """Timing of a video file, read from its container without decoding."""

    n_frames: int
    first_ms: float
    last_ms: float
    duration_ms: float  # last frame's end minus first frame's start
    strictly_increasing: bool
    width: int
    height: int


# --------------------------------------------------------------------------------------
# File names
# --------------------------------------------------------------------------------------


def manifest_path(data_root: str | Path, dataset: str, split: str, kind: Kind = "clip") -> Path:
    """`<data_root>/manifests/<dataset>[_continuous]_<split>.jsonl`. `split` may be `TAKES`."""
    middle = "_continuous" if kind == "continuous" else ""
    return Path(data_root) / MANIFEST_DIR / f"{dataset}{middle}_{split}.jsonl"


def parse_manifest_name(path: str | Path) -> tuple[str, Kind, str] | None:
    """(dataset, kind, split slot) from a manifest file name, or None if it does not fit.

    The split slot is one of `SPLITS` or `TAKES`.
    """
    name = Path(path).name
    if not name.endswith(".jsonl"):
        return None
    stem = name[: -len(".jsonl")]
    for slot in sorted((*SPLITS, TAKES), key=len, reverse=True):
        suffix = "_" + slot
        if stem.endswith(suffix):
            head = stem[: -len(suffix)]
            if head.endswith("_continuous"):
                dataset, kind = head[: -len("_continuous")], "continuous"
            else:
                dataset, kind = head, "clip"
            if dataset:
                return dataset, kind, slot  # type: ignore[return-value]
    return None


def kind_of(path: str | Path) -> Kind:
    parsed = parse_manifest_name(path)
    if parsed is not None:
        return parsed[1]
    return "continuous" if "_continuous_" in Path(path).name else "clip"


# --------------------------------------------------------------------------------------
# Checks on one row, needing nothing but the row
# --------------------------------------------------------------------------------------


def _common_problems(row: Row) -> list[tuple[str, str]]:
    problems: list[tuple[str, str]] = []
    if not _ID.match(row.id):
        problems.append(("bad_field", f"id {row.id!r} is empty or contains whitespace"))
    for name in ("dataset", "signer_id", "session_id", "sign_language", "text_language", "license"):
        if not getattr(row, name).strip():
            problems.append(("bad_field", f"{name} is empty"))
    if row.split not in SPLITS and row.split != UNASSIGNED:
        problems.append(("bad_field", f"split {row.split!r} is not one of {SPLITS}"))
    if row.take < 1:
        problems.append(("bad_field", f"take is {row.take}, must be 1 or more"))
    if not _SHA256.match(row.source_sha256):
        problems.append(("bad_field", "source_sha256 is not 64 lowercase hex characters"))
    source = PurePosixPath(row.source_path)
    if (
        not row.source_path.strip()
        or source.is_absolute()
        or ".." in source.parts
        or "\\" in row.source_path
    ):
        problems.append(
            (
                "bad_path",
                f"source_path {row.source_path!r} must be relative to the data root, "
                "with forward slashes and no '..'",
            )
        )
    if any(not t.strip() for t in row.tags):
        problems.append(("bad_field", "tags contains an empty string"))
    return problems


def row_problems(row: Row) -> list[tuple[str, str]]:
    """(code, message) for everything wrong with `row` that the row alone shows."""
    problems = _common_problems(row)
    if isinstance(row, ClipRow):
        if not row.text.strip():
            problems.append(("empty_text", "text is empty"))
        if (row.start_ms is None) != (row.end_ms is None):
            problems.append(("bad_bounds", "start_ms and end_ms must both be set or both be null"))
        elif row.start_ms is not None and row.end_ms is not None:
            if row.start_ms < 0:
                problems.append(("bad_bounds", f"start_ms is negative ({row.start_ms})"))
            if row.end_ms <= row.start_ms:
                problems.append(
                    ("bad_bounds", f"end_ms {row.end_ms} is not after start_ms {row.start_ms}")
                )
        if row.sentence_id is not None and not row.sentence_id.strip():
            problems.append(("bad_field", "sentence_id is an empty string; use null"))
        return problems

    if not row.segments:
        problems.append(("bad_bounds", "segments is empty"))
    previous_end = 0
    for i, seg in enumerate(row.segments):
        if not seg.text.strip():
            problems.append(("empty_text", f"segment {i}: text is empty"))
        if seg.start_ms < 0:
            problems.append(("bad_bounds", f"segment {i}: start_ms is negative ({seg.start_ms})"))
        if seg.end_ms <= seg.start_ms:
            problems.append(
                (
                    "bad_bounds",
                    f"segment {i}: end_ms {seg.end_ms} is not after start_ms {seg.start_ms}",
                )
            )
        if i > 0 and seg.start_ms < previous_end:
            problems.append(
                (
                    "bad_bounds",
                    f"segment {i}: starts at {seg.start_ms} ms, before the previous segment "
                    f"ends ({previous_end} ms)",
                )
            )
        previous_end = max(previous_end, seg.end_ms)
    return problems


# --------------------------------------------------------------------------------------
# Reading and writing
# --------------------------------------------------------------------------------------


def _model_for(kind: Kind) -> type[ClipRow] | type[ContinuousRow]:
    return ContinuousRow if kind == "continuous" else ClipRow


def to_json_line(row: Row) -> str:
    return json.dumps(row.model_dump(mode="json"), ensure_ascii=False)


def read_manifest(
    path: str | Path, kind: Kind | None = None
) -> tuple[list[tuple[int, Row]], list[Issue]]:
    """Parse every line. Returns (line number, row) pairs and the lines that did not parse.

    Blank lines are skipped. This does not run `row_problems`; use `validate_manifests` for
    a full check or `load_manifest` to get rows or an exception.
    """
    path = Path(path)
    model = _model_for(kind or kind_of(path))
    rows: list[tuple[int, Row]] = []
    issues: list[Issue] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], [Issue("error", "unreadable", f"cannot read manifest: {exc}", str(path))]
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append((number, model.model_validate_json(line)))
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '<row>'}: {e['msg']}" for e in exc.errors()
            )
            issues.append(Issue("error", "parse_error", details, str(path), number))
    return rows, issues


def load_manifest(path: str | Path, kind: Kind | None = None) -> list[Row]:
    """Rows of a manifest. Raises `ManifestError` if any line fails to parse or is invalid
    on its own. A file that does not exist is an error; check before calling."""
    rows, issues = read_manifest(path, kind)
    for number, row in rows:
        issues.extend(
            Issue("error", code, message, str(path), number, row.id)
            for code, message in row_problems(row)
        )
    if issues:
        shown = "\n".join(str(i) for i in issues[:10])
        raise ManifestError(f"{path}: {len(issues)} problem(s):\n{shown}")
    return [row for _, row in rows]


def load_clip_rows(path: str | Path) -> list[ClipRow]:
    return [r for r in load_manifest(path, "clip") if isinstance(r, ClipRow)]


def load_continuous_rows(path: str | Path) -> list[ContinuousRow]:
    return [r for r in load_manifest(path, "continuous") if isinstance(r, ContinuousRow)]


def _require_valid(row: Row) -> None:
    problems = row_problems(row)
    if problems:
        raise ManifestError(
            f"refusing to write row {row.id!r}: " + "; ".join(f"{c}: {m}" for c, m in problems)
        )


def append_row(path: str | Path, row: Row) -> None:
    """Append one row and flush it to disk. Refuses a row that is invalid on its own."""
    _require_valid(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(to_json_line(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_manifest(path: str | Path, rows: Iterable[Row]) -> None:
    """Replace `path` with `rows`, written to a temporary name and renamed."""
    rows = list(rows)
    for row in rows:
        _require_valid(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(to_json_line(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


# --------------------------------------------------------------------------------------
# Video timing
# --------------------------------------------------------------------------------------


def probe_video(path: str | Path) -> VideoProbe:
    """Read frame presentation times from the container. Raises `ManifestError` if the
    file is not a readable video or has no frames. No frame rate is assumed anywhere."""
    import av  # deferred: heavy

    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ManifestError(f"{path}: no video stream")
            stream = container.streams.video[0]
            time_base = stream.time_base
            if time_base is None:
                raise ManifestError(f"{path}: video stream has no time base")
            width, height = stream.width, stream.height
            decode_order: list[tuple[float, float]] = []
            for packet in container.demux(stream):
                if packet.pts is None:
                    continue
                duration = float(packet.duration * time_base * 1000) if packet.duration else 0.0
                decode_order.append((float(packet.pts * time_base * 1000), duration))
    except ManifestError:
        raise
    except Exception as exc:  # PyAV raises several unrelated error types
        raise ManifestError(f"{path}: cannot read video: {exc}") from exc
    if not decode_order:
        raise ManifestError(f"{path}: video has no frames")
    frames = sorted(decode_order)
    times = [t for t, _ in frames]
    increasing = all(b > a for a, b in itertools.pairwise(times))
    last_duration = frames[-1][1]
    if last_duration <= 0 and len(times) > 1:
        last_duration = statistics.median(b - a for a, b in itertools.pairwise(times))
    return VideoProbe(
        n_frames=len(times),
        first_ms=times[0],
        last_ms=times[-1],
        duration_ms=times[-1] + last_duration - times[0],
        strictly_increasing=increasing,
        width=width,
        height=height,
    )


# --------------------------------------------------------------------------------------
# Full validation
# --------------------------------------------------------------------------------------


def _bounds(row: Row) -> list[tuple[int | None, int | None]]:
    if isinstance(row, ClipRow):
        return [(row.start_ms, row.end_ms)]
    return [(s.start_ms, s.end_ms) for s in row.segments]


def _overlap(a: tuple[int | None, int | None], b: tuple[int | None, int | None]) -> bool:
    a0, a1 = a
    b0, b1 = b
    if a0 is None or a1 is None or b0 is None or b1 is None:
        return True  # a row without bounds claims the whole file
    return a0 < b1 and b0 < a1


def validate_manifests(
    paths: Sequence[str | Path],
    data_root: str | Path,
    *,
    check_files: bool = True,
    check_hash: bool = True,
    check_media: bool = True,
    sentences: Mapping[str, str] | None = None,
    sentence_dataset: str = "domain",
) -> ValidationReport:
    """Validate manifest files together.

    Reports, without stopping at the first: lines that do not parse; duplicate ids (within
    and across the given files); empty text; bad bounds and durations; a split field that
    disagrees with the file name; missing source files; content-hash mismatches; unreadable
    video or non-increasing timestamps; bounds beyond the end of the video; and the same
    recording used by two rows. Recorder output (`unassigned` rows) and the split files
    built from it are compared only within their own stage, so both can be given at once.

    With `sentences` (sentence id -> English text), rows of
    `sentence_dataset` must name a known sentence and carry its text unchanged.

    `check_hash` and `check_media` need `check_files`.
    """
    data_root = Path(data_root)
    report = ValidationReport()
    seen_ids: dict[tuple[bool, str], str] = {}
    # recording identity -> rows using it, for the duplicate-recording check
    by_hash: dict[tuple[bool, str], list[tuple[str, int, Row]]] = {}
    file_cache: dict[str, tuple[bool, str | None, VideoProbe | None, str | None]] = {}

    def add(level: Level, code: str, message: str, path: Path, line: int, row: Row) -> None:
        report.issues.append(Issue(level, code, message, str(path), line, row.id))

    for raw_path in paths:
        path = Path(raw_path)
        report.n_files += 1
        if not path.is_file():
            report.issues.append(Issue("error", "unreadable", "manifest not found", str(path)))
            continue
        rows, parse_issues = read_manifest(path)
        report.issues.extend(parse_issues)
        parsed_name = parse_manifest_name(path)
        if parsed_name is None:
            report.issues.append(
                Issue(
                    "warning",
                    "file_name",
                    "file name is not <dataset>[_continuous]_<split>.jsonl; "
                    "the split and dataset fields cannot be checked against it",
                    str(path),
                )
            )
        for line, row in rows:
            report.n_rows += 1
            for code, message in row_problems(row):
                add("error", code, message, path, line, row)

            where = f"{path}:{line}"
            # Split files are built from the recorder's `unassigned` rows, so the same row
            # and recording legitimately appear once before and once after the split.
            stage = row.split == UNASSIGNED
            if (stage, row.id) in seen_ids:
                add(
                    "error",
                    "duplicate_id",
                    f"id already used at {seen_ids[stage, row.id]}",
                    path,
                    line,
                    row,
                )
            else:
                seen_ids[stage, row.id] = where

            if parsed_name is not None:
                dataset, _, slot = parsed_name
                expected = UNASSIGNED if slot == TAKES else slot
                if row.split != expected:
                    add(
                        "error",
                        "split_mismatch",
                        f"split is {row.split!r} but the file name says {expected!r}",
                        path,
                        line,
                        row,
                    )
                if row.dataset != dataset:
                    add(
                        "error",
                        "split_mismatch",
                        f"dataset is {row.dataset!r} but the file name says {dataset!r}",
                        path,
                        line,
                        row,
                    )

            if sentences is not None and row.dataset == sentence_dataset:
                pairs = (
                    [(row.sentence_id, row.text)]
                    if isinstance(row, ClipRow)
                    else [(s.sentence_id, s.text) for s in row.segments]
                )
                for sentence_id, text in pairs:
                    if sentence_id is None or sentence_id not in sentences:
                        add(
                            "error",
                            "unknown_sentence",
                            f"sentence_id {sentence_id!r} is not in the sentence list",
                            path,
                            line,
                            row,
                        )
                    elif sentences[sentence_id] != text:
                        add(
                            "error",
                            "text_mismatch",
                            f"text {text!r} differs from sentence {sentence_id}: "
                            f"{sentences[sentence_id]!r}",
                            path,
                            line,
                            row,
                        )

            by_hash.setdefault((stage, row.source_sha256), []).append((str(path), line, row))

            if not check_files or any(c == "bad_path" for c, _ in row_problems(row)):
                continue
            source = data_root / row.source_path
            key = str(source)
            if key not in file_cache:
                exists = source.is_file()
                digest = sha256_file(source) if exists and check_hash else None
                probe: VideoProbe | None = None
                probe_error: str | None = None
                if exists and check_media:
                    try:
                        probe = probe_video(source)
                    except ManifestError as exc:
                        probe_error = str(exc)
                file_cache[key] = (exists, digest, probe, probe_error)
            exists, digest, probe, probe_error = file_cache[key]
            if not exists:
                add("error", "missing_file", f"source file not found: {source}", path, line, row)
                continue
            if digest is not None and digest != row.source_sha256:
                add(
                    "error",
                    "hash_mismatch",
                    f"file SHA-256 is {digest}, the row says {row.source_sha256}",
                    path,
                    line,
                    row,
                )
            if probe_error is not None:
                add("error", "bad_media", probe_error, path, line, row)
            if probe is None:
                continue
            if probe.n_frames < 2 or probe.duration_ms <= 0:
                add(
                    "error",
                    "bad_duration",
                    f"video has {probe.n_frames} frame(s) and lasts {probe.duration_ms:.0f} ms",
                    path,
                    line,
                    row,
                )
                continue
            if not probe.strictly_increasing:
                add(
                    "error",
                    "bad_media",
                    "frame timestamps are not strictly increasing",
                    path,
                    line,
                    row,
                )
            video_end = probe.first_ms + probe.duration_ms
            for start, end in _bounds(row):
                if end is not None and end > video_end + BOUNDS_TOLERANCE_MS:
                    add(
                        "error",
                        "bad_bounds",
                        f"end_ms {end} is past the end of the video ({video_end:.0f} ms)",
                        path,
                        line,
                        row,
                    )
                if isinstance(row, ClipRow):
                    length = (
                        probe.duration_ms
                        if start is None or end is None
                        else min(end, video_end) - start
                    )
                    if length < SHORT_CLIP_MS:
                        add(
                            "warning",
                            "short_clip",
                            f"clip lasts {length:.0f} ms",
                            path,
                            line,
                            row,
                        )
                    elif length > LONG_CLIP_MS:
                        add(
                            "warning",
                            "long_clip",
                            f"clip lasts {length / 1000:.1f} s for one sentence",
                            path,
                            line,
                            row,
                        )

    for users in by_hash.values():
        if len(users) < 2:
            continue
        first_path, first_line, first = users[0]
        for other_path, other_line, other in users[1:]:
            same_file = other.source_path == first.source_path
            clash = (
                not same_file
                or isinstance(first, ContinuousRow)
                or isinstance(other, ContinuousRow)
                or any(_overlap(a, b) for a in _bounds(first) for b in _bounds(other))
            )
            if clash:
                reason = (
                    "the same part of the same recording"
                    if same_file
                    else f"a file with the same content as {first.source_path!r}"
                )
                report.issues.append(
                    Issue(
                        "error",
                        "duplicate_source",
                        f"uses {reason}, already used by {first.id!r} at {first_path}:{first_line}",
                        other_path,
                        other_line,
                        other.id,
                    )
                )
    return report
