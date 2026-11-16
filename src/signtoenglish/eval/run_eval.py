"""Evaluation harness, written against a `translate(batch) -> list[str]` callable.

The harness knows nothing about the model: the caller supplies a callable that takes a
batch of `EvalItem` (manifest rows) and returns one English string per item. Step 2.1
wraps `Translator.translate` plus the keypoint cache in such a callable.

Output (ARCHITECTURE.md, section 4.7), all under `run_dir`:

    eval/<split>/hyps.txt           one hypothesis per line, in item order
    eval/<split>/refs.txt           one reference per line, same order
    eval/<split>/predictions.jsonl  {"id", "sentence_id", "subset", "ref", "hyp", "exact"}
    eval/<split>/metrics.json       metrics, sacreBLEU signature, checkpoint id, config hash
    test_runs.log                   one JSON line per test-split scoring

Test discipline (ARCHITECTURE.md, section 7): a split named `test` or `test_*` is scored
once per checkpoint. A second scoring of the same split for the same checkpoint id raises
`TestSetAlreadyScoredError` unless `force=True`; every scoring, forced or not, is appended
to `test_runs.log`. `test_seen` and `test_unseen` are separate splits, scored and reported
separately and never pooled.

Manifest rows are read with only the fields this module needs (ARCHITECTURE.md, section
4.2). Full validation belongs to `data/manifest.py` (Track C).
"""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from signtoenglish.common.hashing import config_hash
from signtoenglish.eval.metrics import (
    TextMetrics,
    compute_metrics,
    metric_versions,
    normalize_for_match,
)

SCHEMA_VERSION = 1
TEST_LOG_NAME = "test_runs.log"
SUBSET_SEEN = "seen"
SUBSET_UNSEEN = "unseen"


class EvalError(RuntimeError):
    """The evaluation input or the translate callable broke the harness contract."""


class TestSetAlreadyScoredError(EvalError):
    """This checkpoint has already been scored on this test split (and `force` is off)."""

    __test__ = False  # not a pytest test class, despite the name


@dataclass(frozen=True)
class EvalItem:
    """One sentence clip to translate and score. `row` is the full manifest row."""

    id: str
    split: str
    text: str
    dataset: str
    sentence_id: str | None
    row: Mapping[str, Any]
    subset: str | None = None  # optional reporting group, e.g. "seen" / "unseen" inside val


Translate = Callable[[Sequence[EvalItem]], list[str]]


@dataclass(frozen=True)
class EvalResult:
    split: str
    checkpoint_id: str
    eval_dir: Path
    metrics: TextMetrics
    subsets: Mapping[str, TextMetrics]
    report: Mapping[str, Any]  # exactly what was written to metrics.json


def is_test_split(split: str) -> bool:
    return split == "test" or split.startswith("test_")


def item_from_row(row: Mapping[str, Any], where: str = "row") -> EvalItem:
    """Build an `EvalItem` from a manifest row, checking only the fields used here."""
    for key in ("id", "split", "text", "dataset"):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise EvalError(f"{where}: field {key!r} must be a non-empty string, got {value!r}")
    sentence_id = row.get("sentence_id")
    if sentence_id is not None and not isinstance(sentence_id, str):
        raise EvalError(f"{where}: field 'sentence_id' must be a string or null")
    return EvalItem(
        id=row["id"],
        split=row["split"],
        text=row["text"],
        dataset=row["dataset"],
        sentence_id=sentence_id,
        row=dict(row),
    )


def load_eval_items(manifest_path: str | Path, *, split: str | None = None) -> list[EvalItem]:
    """Read a manifest JSONL file. With `split`, every row must belong to that split."""
    path = Path(manifest_path)
    items: list[EvalItem] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            where = f"{path}:{lineno}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalError(f"{where}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise EvalError(f"{where}: a manifest row must be a JSON object")
            item = item_from_row(row, where)
            if item.id in seen_ids:
                raise EvalError(f"{where}: duplicate id {item.id!r}")
            if split is not None and item.split != split:
                raise EvalError(f"{where}: row is in split {item.split!r}, expected {split!r}")
            seen_ids.add(item.id)
            items.append(item)
    if not items:
        raise EvalError(f"{path}: no rows")
    return items


def label_seen_unseen(items: Iterable[EvalItem], train_items: Iterable[EvalItem]) -> list[EvalItem]:
    """Set `subset` to "seen" or "unseen" by whether the sentence occurs in training.

    A sentence counts as seen when its `sentence_id` is a training `sentence_id` or its
    text equals a training text after `normalize_for_match`. Used for the validation
    split, which mixes both kinds (ARCHITECTURE.md, section 4.3).
    """
    train = list(train_items)
    train_ids = {t.sentence_id for t in train if t.sentence_id is not None}
    train_texts = {normalize_for_match(t.text) for t in train}
    labelled = []
    for item in items:
        seen = item.sentence_id in train_ids or normalize_for_match(item.text) in train_texts
        labelled.append(replace(item, subset=SUBSET_SEEN if seen else SUBSET_UNSEEN))
    return labelled


def read_test_log(run_dir: str | Path) -> list[dict[str, Any]]:
    """Every record in `run_dir/test_runs.log`, oldest first. A missing file is empty."""
    path = Path(run_dir) / TEST_LOG_NAME
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            # An unreadable log must not silently permit a second look at the test set.
            raise EvalError(f"{path}:{lineno}: unreadable test log line: {exc}") from exc
        if not isinstance(record, dict) or "checkpoint_id" not in record or "split" not in record:
            raise EvalError(f"{path}:{lineno}: test log record lacks checkpoint_id or split")
        records.append(record)
    return records


def previous_test_scorings(run_dir: str | Path, checkpoint_id: str, split: str) -> list[str]:
    """Descriptions of earlier scorings of `split` for `checkpoint_id`; empty if none.

    Looks at `test_runs.log` and, as a second line of defence should the log have been
    lost, at an existing `eval/<split>/metrics.json` for the same checkpoint.
    """
    run_dir = Path(run_dir)
    found = [
        f"{TEST_LOG_NAME}: scored at {r.get('time_utc', '?')}"
        for r in read_test_log(run_dir)
        if r["checkpoint_id"] == checkpoint_id and r["split"] == split
    ]
    metrics_path = run_dir / "eval" / split / "metrics.json"
    if metrics_path.is_file():
        try:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvalError(f"{metrics_path}: unreadable: {exc}") from exc
        if isinstance(existing, dict) and existing.get("checkpoint_id") == checkpoint_id:
            found.append(f"{metrics_path}: created {existing.get('created_utc', '?')}")
    return found


def _refuse_if_scored(run_dir: Path, checkpoint_id: str, split: str, force: bool) -> bool:
    """Raise unless this test scoring may proceed. Returns True when it is a forced rerun."""
    previous = previous_test_scorings(run_dir, checkpoint_id, split)
    if previous and not force:
        raise TestSetAlreadyScoredError(
            f"checkpoint {checkpoint_id} was already scored on {split!r} "
            f"({'; '.join(previous)}). The test split is scored once per checkpoint; "
            "pass --force (force=True) to score it again, which is logged."
        )
    return bool(previous)


def _one_line(text: str) -> str:
    """Collapse whitespace so one sentence is one line in hyps.txt / refs.txt."""
    return " ".join(text.split())


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _append_log(run_dir: Path, record: Mapping[str, Any]) -> None:
    with (run_dir / TEST_LOG_NAME).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _hardware() -> dict[str, Any]:
    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


def _translate_all(translate: Translate, items: Sequence[EvalItem], batch_size: int) -> list[str]:
    hyps: list[str] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        out = translate(batch)
        if isinstance(out, str) or not isinstance(out, Sequence):
            raise EvalError(f"translate must return a list of strings, got {type(out).__name__}")
        if len(out) != len(batch):
            raise EvalError(
                f"translate returned {len(out)} strings for a batch of {len(batch)} "
                f"(items {start} to {start + len(batch) - 1})"
            )
        for item, hyp in zip(batch, out, strict=True):
            if not isinstance(hyp, str):
                raise EvalError(f"translate returned {type(hyp).__name__} for item {item.id!r}")
            hyps.append(_one_line(hyp))
    return hyps


def run_eval(
    translate: Translate,
    items: Sequence[EvalItem],
    *,
    split: str,
    run_dir: str | Path,
    checkpoint_id: str,
    config: BaseModel,
    batch_size: int = 8,
    force: bool = False,
    device: str | None = None,
    extra_hashes: Mapping[str, str] | None = None,
) -> EvalResult:
    """Translate `items`, score them, and write `run_dir/eval/<split>/`.

    `checkpoint_id` identifies the weights being scored (the checkpoint file's SHA-256).
    `config` is the resolved config the translation depends on (model and decoding); its
    `config_hash` is recorded. `extra_hashes` carries further identifiers to record, for
    example `{"preprocessing_hash": ...}`. For a test split, see the module docstring.
    """
    run_dir = Path(run_dir)
    items = list(items)
    if not checkpoint_id.strip():
        raise EvalError("checkpoint_id must not be empty")
    if batch_size < 1:
        raise EvalError(f"batch_size must be at least 1, got {batch_size}")
    if not items:
        raise EvalError(f"no items to evaluate for split {split!r}")
    wrong = sorted({i.split for i in items} - {split})
    if wrong:
        raise EvalError(f"items from split(s) {wrong} passed to an evaluation of {split!r}")
    if len({i.id for i in items}) != len(items):
        raise EvalError("duplicate item ids")

    cfg_hash = config_hash(config)
    is_test = is_test_split(split)
    forced_rerun = _refuse_if_scored(run_dir, checkpoint_id, split, force) if is_test else False

    started = time.perf_counter()
    hyps = _translate_all(translate, items, batch_size)
    translate_seconds = time.perf_counter() - started
    refs = [_one_line(i.text) for i in items]

    metrics = compute_metrics(hyps, refs)
    subsets: dict[str, TextMetrics] = {}
    for name in sorted({i.subset for i in items if i.subset is not None}):
        idx = [k for k, i in enumerate(items) if i.subset == name]
        subsets[name] = compute_metrics([hyps[k] for k in idx], [refs[k] for k in idx])

    created = datetime.now(UTC).isoformat(timespec="seconds")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "datasets": sorted({i.dataset for i in items}),
        "checkpoint_id": checkpoint_id,
        "config_hash": cfg_hash,
        "extra_hashes": dict(extra_hashes or {}),
        "n": len(items),
        "sacrebleu_signature": metrics.bleu.signature,
        "metrics": metrics.to_dict(),
        "subsets": {name: m.to_dict() for name, m in subsets.items()},
        "is_test_split": is_test,
        "forced_rerun": forced_rerun,
        "device": device,
        "hardware": _hardware(),
        "versions": metric_versions(),
        "batch_size": batch_size,
        "translate_seconds": round(translate_seconds, 3),
        "created_utc": created,
    }

    eval_dir = run_dir / "eval" / split
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / "metrics.json"
    metrics_path.unlink(missing_ok=True)  # metrics.json is written last and marks completeness
    _write_atomic(eval_dir / "hyps.txt", "".join(h + "\n" for h in hyps))
    _write_atomic(eval_dir / "refs.txt", "".join(r + "\n" for r in refs))
    predictions = [
        {
            "id": item.id,
            "sentence_id": item.sentence_id,
            "subset": item.subset,
            "ref": ref,
            "hyp": hyp,
            "exact": normalize_for_match(hyp) == normalize_for_match(ref),
        }
        for item, hyp, ref in zip(items, hyps, refs, strict=True)
    ]
    _write_atomic(
        eval_dir / "predictions.jsonl",
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions),
    )
    _write_atomic(metrics_path, json.dumps(report, indent=2, sort_keys=True) + "\n")

    if is_test:
        _append_log(
            run_dir,
            {
                "time_utc": created,
                "checkpoint_id": checkpoint_id,
                "split": split,
                "datasets": report["datasets"],
                "n": len(items),
                "config_hash": cfg_hash,
                "forced": force,
                "forced_rerun": forced_rerun,
                "bleu4": metrics.bleu.bleu4,
                "rouge_l_f1": metrics.rouge_l.f1,
                "exact_match": metrics.exact_match.rate,
                "sacrebleu_signature": metrics.bleu.signature,
                "eval_dir": f"eval/{split}",
            },
        )

    return EvalResult(
        split=split,
        checkpoint_id=checkpoint_id,
        eval_dir=eval_dir,
        metrics=metrics,
        subsets=subsets,
        report=report,
    )


def run_eval_splits(
    translate: Translate,
    items_by_split: Mapping[str, Sequence[EvalItem]],
    *,
    run_dir: str | Path,
    checkpoint_id: str,
    config: BaseModel,
    batch_size: int = 8,
    force: bool = False,
    device: str | None = None,
    extra_hashes: Mapping[str, str] | None = None,
) -> dict[str, EvalResult]:
    """Evaluate several splits, each on its own, for example `test_seen` and `test_unseen`.

    Every test split is checked for an earlier scoring before anything is translated, so
    a refusal leaves no split scored. Results are per split and are never pooled.
    """
    run_dir = Path(run_dir)
    for split in items_by_split:
        if is_test_split(split):
            _refuse_if_scored(run_dir, checkpoint_id, split, force)
    return {
        split: run_eval(
            translate,
            items,
            split=split,
            run_dir=run_dir,
            checkpoint_id=checkpoint_id,
            config=config,
            batch_size=batch_size,
            force=force,
            device=device,
            extra_hashes=extra_hashes,
        )
        for split, items in items_by_split.items()
    }


def _metrics_line(label: str, m: TextMetrics) -> str:
    return (
        f"  {label:<10} n={m.n:<4} "
        f"BLEU-1 {m.bleu.bleu1:6.2f}  BLEU-2 {m.bleu.bleu2:6.2f}  "
        f"BLEU-3 {m.bleu.bleu3:6.2f}  BLEU-4 {m.bleu.bleu4:6.2f}  "
        f"ROUGE-L {m.rouge_l.f1:6.2f}  "
        f"exact {m.exact_match.rate:6.2f}% ({m.exact_match.n_match}/{m.exact_match.n})"
    )


def format_report(results: Mapping[str, EvalResult]) -> str:
    """Plain-text report: one block per split, with split, checkpoint, signature, hardware."""
    lines: list[str] = []
    for split, result in results.items():
        r = result.report
        hw = r["hardware"]
        lines.append(f"split {split}  checkpoint {result.checkpoint_id}  config {r['config_hash']}")
        lines.append(_metrics_line("all", result.metrics))
        for name, m in result.subsets.items():
            lines.append(_metrics_line(name, m))
        lines.append(f"  sacreBLEU signature: {r['sacrebleu_signature']}")
        lines.append(
            f"  hardware: {hw['machine']} / {hw['platform']}  device: {r['device'] or 'not given'}"
        )
    return "\n".join(lines)
