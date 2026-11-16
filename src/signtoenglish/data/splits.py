"""Domain split builder and its guarantees (ARCHITECTURE.md, section 4.3).

Input: the rows `s2e record` wrote (split `unassigned`) and the domain sentence list.
Output: `train`, `val`, `test_seen`, `test_unseen` for sentence clips, and `val` / `test`
for continuous takes.

Guarantees, checked by `check_split_guarantees` on every build and by the tests:

1. No recording appears in two splits (by row id, by file path, and by content hash).
2. No `test_unseen` sentence appears in train or val, by sentence id or by text.
3. Every sign listed for a `test_unseen` sentence appears in at least three distinct
   sentences that have a take in train.

The builder is deterministic: the same rows, sentence list, and config give the same
splits. Build the splits once, after recording is finished; rebuilding after adding takes
can move a take between train, val, and test_seen.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from signtoenglish.common.config import FrozenConfig
from signtoenglish.data.domain import (
    MIN_TRAIN_SENTENCES_PER_SIGN,
    DomainSentences,
    text_key,
    uncovered_signs,
)
from signtoenglish.data.manifest import (
    UNASSIGNED,
    ClipRow,
    ContinuousRow,
    Row,
    manifest_path,
    write_manifest,
)

CLIP_SPLITS: tuple[str, ...] = ("train", "val", "test_seen", "test_unseen")
CONTINUOUS_SPLITS: tuple[str, ...] = ("val", "test")
DATASET = "domain"


class SplitError(ValueError):
    """The rows cannot be split, or the result breaks a guarantee."""


class SplitConfig(FrozenConfig):
    # How many `role: unseen_test` sentences are set aside for validation; the rest form
    # test_unseen. 5 of the 20 (user decision, 2026-11-16; ARCHITECTURE.md, section 4.3).
    val_unseen_count: int = 5
    val_takes_per_sentence: int = 1
    test_seen_takes_per_sentence: int = 1
    # If set, every take of a training sentence from this session goes to test_seen and
    # none to train or val ("from a session not used in train where possible").
    holdout_session: str | None = None
    min_train_sentences_per_sign: int = MIN_TRAIN_SENTENCES_PER_SIGN
    seed: int = 0


@dataclass
class SplitResult:
    clips: dict[str, list[ClipRow]]
    continuous: dict[str, list[ContinuousRow]]
    val_unseen_ids: tuple[str, ...]
    test_unseen_ids: tuple[str, ...]
    warnings: list[str] = field(default_factory=list)


def choose_val_unseen(domain: DomainSentences, cfg: SplitConfig) -> tuple[str, ...]:
    """The unseen sentences set aside for validation. Depends only on the sentence ids and
    the seed, so recording more takes never moves a sentence between val and test."""
    unseen = sorted(s.id for s in domain.sentences if s.role == "unseen_test")
    if cfg.val_unseen_count < 0 or cfg.val_unseen_count > len(unseen):
        raise SplitError(
            f"val_unseen_count is {cfg.val_unseen_count} but the list has {len(unseen)} "
            "unseen-test sentences"
        )
    return tuple(sorted(random.Random(cfg.seed).sample(unseen, cfg.val_unseen_count)))


def _order(row: Row) -> tuple[str, int, str]:
    # session ids start with the date, so this is recording order
    return (row.session_id, row.take, row.id)


def build_splits(
    clip_rows: Sequence[ClipRow],
    domain: DomainSentences,
    cfg: SplitConfig | None = None,
    continuous_rows: Sequence[ContinuousRow] = (),
) -> SplitResult:
    """Assign every row to a split and verify the guarantees. Raises `SplitError` when a
    row names an unknown sentence or a guarantee does not hold."""
    cfg = cfg or SplitConfig()
    if cfg.val_takes_per_sentence < 0 or cfg.test_seen_takes_per_sentence < 0:
        raise SplitError("takes per sentence cannot be negative")
    sentences = domain.by_id()
    val_unseen = set(choose_val_unseen(domain, cfg))
    test_unseen = {s.id for s in domain.sentences if s.role == "unseen_test"} - val_unseen
    warnings: list[str] = []

    by_sentence: dict[str, list[ClipRow]] = defaultdict(list)
    for row in clip_rows:
        if row.sentence_id is None or row.sentence_id not in sentences:
            raise SplitError(f"row {row.id!r}: sentence_id {row.sentence_id!r} is not in the list")
        if row.text != sentences[row.sentence_id].text:
            raise SplitError(
                f"row {row.id!r}: text differs from sentence {row.sentence_id} in the list"
            )
        by_sentence[row.sentence_id].append(row)

    assigned: dict[str, list[ClipRow]] = {name: [] for name in CLIP_SPLITS}
    sessions = {row.session_id for row in clip_rows}
    if cfg.holdout_session is not None and cfg.holdout_session not in sessions:
        raise SplitError(
            f"holdout_session {cfg.holdout_session!r} has no takes; sessions: {sorted(sessions)}"
        )

    for sentence_id in sorted(by_sentence):
        takes = sorted(by_sentence[sentence_id], key=_order)
        if sentence_id in val_unseen:
            assigned["val"].extend(takes)
            continue
        if sentence_id in test_unseen:
            assigned["test_unseen"].extend(takes)
            continue

        if cfg.holdout_session is not None:
            held = [r for r in takes if r.session_id == cfg.holdout_session]
            rest = [r for r in takes if r.session_id != cfg.holdout_session]
            n_val = min(cfg.val_takes_per_sentence, max(0, len(rest) - 1))
            n_test = 0
        else:
            held, rest = [], takes
            # Order of priority when takes are scarce: one for train, then val, then test.
            n_val = min(cfg.val_takes_per_sentence, max(0, len(rest) - 1))
            n_test = min(cfg.test_seen_takes_per_sentence, max(0, len(rest) - 1 - n_val))
        # The latest takes are held out: test_seen from the end, val just before them.
        n_train = len(rest) - n_val - n_test
        assigned["train"].extend(rest[:n_train])
        assigned["val"].extend(rest[n_train : n_train + n_val])
        assigned["test_seen"].extend(rest[n_train + n_val :])
        assigned["test_seen"].extend(held)

        if n_train == 0:
            warnings.append(f"{sentence_id}: no take left for train")
        if n_val < cfg.val_takes_per_sentence:
            warnings.append(
                f"{sentence_id}: {n_val} val take(s), wanted {cfg.val_takes_per_sentence}"
            )
        n_test_total = n_test + len(held)
        if n_test_total < cfg.test_seen_takes_per_sentence:
            warnings.append(
                f"{sentence_id}: {n_test_total} test_seen take(s), "
                f"wanted {cfg.test_seen_takes_per_sentence}"
            )

    for sentence in domain.sentences:
        if sentence.id not in by_sentence:
            warnings.append(f"{sentence.id}: no takes recorded")

    clips = {
        name: [row.model_copy(update={"split": name}) for row in sorted(rows, key=_order)]
        for name, rows in assigned.items()
    }

    # Continuous takes are for choosing and scoring the sentence-end detector, not for
    # adaptation. A take that contains a test_unseen sentence can only be a test take; the
    # others alternate between val and test in recording order.
    continuous: dict[str, list[ContinuousRow]] = {name: [] for name in CONTINUOUS_SPLITS}
    free: list[ContinuousRow] = []
    for crow in sorted(continuous_rows, key=_order):
        for seg in crow.segments:
            if seg.sentence_id is None or seg.sentence_id not in sentences:
                raise SplitError(
                    f"row {crow.id!r}: sentence_id {seg.sentence_id!r} is not in the list"
                )
        if any(seg.sentence_id in test_unseen for seg in crow.segments):
            continuous["test"].append(crow)
        else:
            free.append(crow)
    for i, crow in enumerate(free):
        continuous["val" if i % 2 == 0 else "test"].append(crow)
    continuous = {
        name: [row.model_copy(update={"split": name}) for row in sorted(rows, key=_order)]
        for name, rows in continuous.items()
    }

    result = SplitResult(
        clips=clips,
        continuous=continuous,
        val_unseen_ids=tuple(sorted(val_unseen)),
        test_unseen_ids=tuple(sorted(test_unseen)),
        warnings=warnings,
    )
    violations = check_split_guarantees(
        clips, domain, cfg.min_train_sentences_per_sign, continuous=continuous
    )
    if violations:
        raise SplitError(
            f"{len(violations)} split guarantee violation(s):\n" + "\n".join(violations)
        )
    return result


def check_split_guarantees(
    clips: Mapping[str, Sequence[ClipRow]],
    domain: DomainSentences,
    min_train_sentences_per_sign: int = MIN_TRAIN_SENTENCES_PER_SIGN,
    continuous: Mapping[str, Sequence[ContinuousRow]] | None = None,
) -> list[str]:
    """Every violation of the guarantees in the module docstring, as readable strings.

    Works on any splits, not only the builder's, so it can be run on manifests read back
    from disk.
    """
    violations: list[str] = []
    sentences = domain.by_id()

    # 1. No recording in two splits. Sentence clips and continuous takes are separate
    #    recordings, and one file must not serve both either.
    all_rows: list[tuple[str, Row]] = [(name, row) for name, rows in clips.items() for row in rows]
    for name, crows in (continuous or {}).items():
        all_rows.extend((f"continuous {name}", crow) for crow in crows)
    for label, attribute in (
        ("row id", "id"),
        ("file", "source_path"),
        ("recording (content hash)", "source_sha256"),
    ):
        seen: dict[str, tuple[str, str]] = {}
        for name, row in all_rows:
            key = str(getattr(row, attribute))
            if key in seen:
                first_split, first_id = seen[key]
                place = "twice in" if first_split == name else f"in {first_split} and"
                violations.append(
                    f"same {label} {place} {name}: {first_id!r} and {row.id!r} ({key})"
                )
            else:
                seen[key] = (name, row.id)

    for name, rows in clips.items():
        for row in rows:
            if row.split != name:
                violations.append(
                    f"row {row.id!r} is in {name} but its split field is {row.split!r}"
                )

    # 2. No test_unseen sentence in train or val, by id or by text.
    test_unseen_rows = clips.get("test_unseen", ())
    unseen_ids = {r.sentence_id for r in test_unseen_rows}
    unseen_texts = {text_key(r.text): r.sentence_id for r in test_unseen_rows}
    for name in ("train", "val"):
        for row in clips.get(name, ()):
            if row.sentence_id in unseen_ids:
                violations.append(
                    f"test_unseen sentence {row.sentence_id} has a take in {name}: {row.id!r}"
                )
            elif text_key(row.text) in unseen_texts:
                violations.append(
                    f"text of test_unseen sentence {unseen_texts[text_key(row.text)]} appears "
                    f"in {name}: {row.id!r} ({row.text!r})"
                )
    for crow in (continuous or {}).get("val", ()):
        for seg in crow.segments:
            if seg.sentence_id in unseen_ids or text_key(seg.text) in unseen_texts:
                violations.append(
                    f"test_unseen sentence {seg.sentence_id} is in continuous val take {crow.id!r}"
                )

    # Sentences never meant for training must not reach train, and test_unseen holds only them.
    for row in clips.get("train", ()):
        sentence = sentences.get(row.sentence_id or "")
        if sentence is None or sentence.role != "train":
            violations.append(f"train holds a take of a non-training sentence: {row.id!r}")
    for row in test_unseen_rows:
        sentence = sentences.get(row.sentence_id or "")
        if sentence is None or sentence.role != "unseen_test":
            violations.append(f"test_unseen holds a take of a training sentence: {row.id!r}")

    # 3. Sign coverage, counted over sentences that really have a take in train.
    train_ids = {r.sentence_id for r in clips.get("train", ()) if r.sentence_id}
    known_unseen = sorted(i for i in unseen_ids if i in sentences)
    short = uncovered_signs(domain, known_unseen, train_ids, min_train_sentences_per_sign)
    for sentence_id, signs in short.items():
        listing = ", ".join(f"{sign} ({n})" for sign, n in sorted(signs.items()))
        violations.append(
            f"test_unseen sentence {sentence_id}: signs in fewer than "
            f"{min_train_sentences_per_sign} training sentences with a take in train: {listing}"
        )
    for sentence_id in known_unseen:
        if not sentences[sentence_id].signs:
            violations.append(f"test_unseen sentence {sentence_id} lists no signs")
    return violations


def require_unassigned(rows: Sequence[Row]) -> None:
    """The builder starts from recorder output; refuse rows that already have a split."""
    wrong = [r.id for r in rows if r.split != UNASSIGNED]
    if wrong:
        raise SplitError(f"{len(wrong)} row(s) already have a split, for example {wrong[0]!r}")


def write_splits(result: SplitResult, data_root: str | Path) -> list[Path]:
    """Write `domain_<split>.jsonl` and `domain_continuous_<split>.jsonl`, each atomically.
    An empty split is written as an empty file so a stale one never survives a rebuild."""
    written: list[Path] = []
    for name in CLIP_SPLITS:
        path = manifest_path(data_root, DATASET, name)
        write_manifest(path, result.clips[name])
        written.append(path)
    for name in CONTINUOUS_SPLITS:
        path = manifest_path(data_root, DATASET, name, "continuous")
        write_manifest(path, result.continuous[name])
        written.append(path)
    return written
