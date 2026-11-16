"""The evaluation harness, end to end with a trivial translate callable.

The callables here are stand-ins for the model by design: this harness is model-free and
its gate (WORKERS.md, track D) is that it runs end to end with a trivial callable. The run
with the real model is deferred to PLAN.md Step 2.1.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from signtoenglish.common.config import FrozenConfig
from signtoenglish.common.hashing import config_hash
from signtoenglish.eval.metrics import compute_metrics
from signtoenglish.eval.run_eval import (
    TEST_LOG_NAME,
    EvalError,
    EvalItem,
    TestSetAlreadyScoredError,
    format_report,
    is_test_split,
    label_seen_unseen,
    load_eval_items,
    read_test_log,
    run_eval,
    run_eval_splits,
)

CKPT = "a" * 64
OTHER_CKPT = "b" * 64


class DecodeConfig(FrozenConfig):
    """Stands in for the resolved model and decoding config whose hash is recorded."""

    num_beams: int = 4
    max_new_tokens: int = 64


CFG = DecodeConfig()


def _row(id_: str, split: str, text: str, sentence_id: str | None) -> dict[str, object]:
    """A full manifest row as ARCHITECTURE.md section 4.2 defines it."""
    return {
        "id": id_,
        "dataset": "domain",
        "split": split,
        "source_path": f"domain/{id_}.mp4",
        "source_sha256": "0" * 64,
        "start_ms": None,
        "end_ms": None,
        "text": text,
        "sentence_id": sentence_id,
        "signer_id": "signer01",
        "session_id": "2026-11-16_desk",
        "take": 1,
        "sign_language": "ase",
        "text_language": "en",
        "license": "own-recording",
        "tags": [],
    }


ROWS = {
    "train": [
        _row("tr1", "train", "Hello, my name is Alex.", "s001"),
        _row("tr2", "train", "Nice to meet you.", "s002"),
    ],
    "val": [
        _row("v1", "val", "Hello, my name is Alex.", "s001"),
        _row("v2", "val", "Nice to meet you.", "s002"),
        _row("v3", "val", "Where do you live?", "s090"),
    ],
    "test_seen": [
        _row("ts1", "test_seen", "Hello, my name is Alex.", "s001"),
        _row("ts2", "test_seen", "Nice to meet you.", "s002"),
        _row("ts3", "test_seen", "Nice to meet you.", "s002"),
    ],
    "test_unseen": [
        _row("tu1", "test_unseen", "Where do you work?", "s101"),
        _row("tu2", "test_unseen", "What is your name?", "s102"),
    ],
}

# What the trivial "model" says for each clip id.
ANSWERS = {
    "v1": "hello my name is Alex",
    "v2": "Nice to meet you.",
    "v3": "Where do you work now?",
    "ts1": "hello my name is alex",
    "ts2": "Nice to meet you",
    "ts3": "Nice to see you.",
    "tu1": "Where do you work?",
    "tu2": "Hello my name",
}


class Lookup:
    """Trivial translate callable: looks the answer up by clip id and counts its calls."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, batch: Sequence[EvalItem]) -> list[str]:
        self.batch_sizes.append(len(batch))
        return [ANSWERS[item.id] for item in batch]


def _write_manifest(directory: Path, split: str) -> Path:
    path = directory / f"domain_{split}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in ROWS[split]), encoding="utf-8")
    return path


@pytest.fixture
def items(tmp_path: Path) -> dict[str, list[EvalItem]]:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    return {s: load_eval_items(_write_manifest(manifests, s), split=s) for s in ROWS}


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    path = tmp_path / "runs" / "run01"
    path.mkdir(parents=True)
    return path


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_is_test_split() -> None:
    assert [is_test_split(s) for s in ("test", "test_seen", "test_unseen")] == [True] * 3
    assert [is_test_split(s) for s in ("train", "val", "testing")] == [False] * 3


def test_manifest_rows_load_with_the_full_row_kept(items: dict[str, list[EvalItem]]) -> None:
    first = items["test_unseen"][0]
    assert (first.id, first.split, first.text, first.sentence_id, first.dataset) == (
        "tu1",
        "test_unseen",
        "Where do you work?",
        "s101",
        "domain",
    )
    assert first.row["source_path"] == "domain/tu1.mp4"  # what a real callable will need
    assert first.subset is None


def test_manifest_loading_rejects_bad_rows(tmp_path: Path) -> None:
    def write(*rows: object) -> Path:
        path = tmp_path / "m.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return path

    good = _row("a", "val", "Hello.", None)
    with pytest.raises(EvalError, match="duplicate id"):
        load_eval_items(write(good, good))
    with pytest.raises(EvalError, match="'text'"):
        load_eval_items(write({**good, "text": " "}))
    with pytest.raises(EvalError, match="expected 'test_seen'"):
        load_eval_items(write(good), split="test_seen")
    with pytest.raises(EvalError, match="no rows"):
        load_eval_items(write())
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(EvalError, match="invalid JSON"):
        load_eval_items(bad)


def test_validation_run_end_to_end_with_seen_and_unseen_subsets(
    items: dict[str, list[EvalItem]], run_dir: Path
) -> None:
    val = label_seen_unseen(items["val"], items["train"])
    assert [i.subset for i in val] == ["seen", "seen", "unseen"]

    translate = Lookup()
    result = run_eval(
        translate,
        val,
        split="val",
        run_dir=run_dir,
        checkpoint_id=CKPT,
        config=CFG,
        batch_size=2,
        device="cpu",
        extra_hashes={"preprocessing_hash": "feedfacefeedface"},
    )
    assert translate.batch_sizes == [2, 1]

    eval_dir = run_dir / "eval" / "val"
    assert result.eval_dir == eval_dir
    hyps, refs = _lines(eval_dir / "hyps.txt"), _lines(eval_dir / "refs.txt")
    assert hyps == [ANSWERS["v1"], ANSWERS["v2"], ANSWERS["v3"]]
    assert refs == ["Hello, my name is Alex.", "Nice to meet you.", "Where do you live?"]

    # This is the three-sentence fixture of test_metrics.py, so the hand-computed values
    # apply: BLEU-4 57.934, ROUGE-L 88.889, exact match 2 of 3.
    assert result.metrics == compute_metrics(hyps, refs)
    assert result.metrics.bleu.bleu4 == pytest.approx(57.934, abs=5e-3)
    assert result.metrics.rouge_l.f1 == pytest.approx(88.889, abs=5e-3)
    assert result.metrics.exact_match.n_match == 2
    assert result.subsets["seen"] == compute_metrics(hyps[:2], refs[:2])
    assert result.subsets["unseen"] == compute_metrics(hyps[2:], refs[2:])
    assert result.subsets["seen"].exact_match.rate == 100.0
    assert result.subsets["unseen"].exact_match.rate == 0.0

    report = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    assert report == result.report
    assert report["split"] == "val"
    assert report["checkpoint_id"] == CKPT
    assert report["config_hash"] == config_hash(CFG)
    assert report["extra_hashes"] == {"preprocessing_hash": "feedfacefeedface"}
    assert report["n"] == 3
    assert report["sacrebleu_signature"] == result.metrics.bleu.signature
    assert report["sacrebleu_signature"].endswith("version:2.6.0")
    assert report["metrics"]["bleu"]["bleu4"] == result.metrics.bleu.bleu4
    assert set(report["subsets"]) == {"seen", "unseen"}
    assert report["device"] == "cpu"
    assert report["hardware"]["machine"]
    assert report["versions"] == {"sacrebleu": "2.6.0", "rouge-score": "0.1.2"}
    assert report["is_test_split"] is False

    predictions = [json.loads(x) for x in _lines(eval_dir / "predictions.jsonl")]
    assert [p["id"] for p in predictions] == ["v1", "v2", "v3"]
    assert [p["exact"] for p in predictions] == [True, True, False]
    assert predictions[2] == {
        "id": "v3",
        "sentence_id": "s090",
        "subset": "unseen",
        "ref": "Where do you live?",
        "hyp": "Where do you work now?",
        "exact": False,
    }

    # Validation is not a test split: no log, and it can be run again freely.
    assert not (run_dir / TEST_LOG_NAME).exists()
    run_eval(Lookup(), val, split="val", run_dir=run_dir, checkpoint_id=CKPT, config=CFG)
    assert not (run_dir / TEST_LOG_NAME).exists()
    assert not list(eval_dir.glob("*.tmp"))


def test_seen_and_unseen_test_sets_are_scored_and_reported_separately(
    items: dict[str, list[EvalItem]], run_dir: Path
) -> None:
    results = run_eval_splits(
        Lookup(),
        {"test_seen": items["test_seen"], "test_unseen": items["test_unseen"]},
        run_dir=run_dir,
        checkpoint_id=CKPT,
        config=CFG,
        device="cpu",
    )
    assert list(results) == ["test_seen", "test_unseen"]
    seen, unseen = results["test_seen"], results["test_unseen"]

    # Each split has its own directory and its own numbers; nothing is pooled.
    for split, result in results.items():
        eval_dir = run_dir / "eval" / split
        assert result.eval_dir == eval_dir
        hyps, refs = _lines(eval_dir / "hyps.txt"), _lines(eval_dir / "refs.txt")
        assert hyps == [ANSWERS[i.id] for i in items[split]]
        assert refs == [i.text for i in items[split]]
        assert result.metrics == compute_metrics(hyps, refs)
        assert json.loads((eval_dir / "metrics.json").read_text())["split"] == split
    assert not (run_dir / "eval" / "test").exists()

    # test_seen: "hello my name is alex" and "Nice to meet you" match their references
    # after normalization, "Nice to see you." does not: 2 of 3.
    assert (seen.metrics.exact_match.n_match, seen.metrics.n) == (2, 3)
    # test_unseen: "Where do you work?" is exact, "Hello my name" is not: 1 of 2.
    assert (unseen.metrics.exact_match.n_match, unseen.metrics.n) == (1, 2)

    text = format_report(results)
    blocks = text.split("split ")[1:]
    assert len(blocks) == 2
    assert blocks[0].startswith("test_seen") and "(2/3)" in blocks[0]
    assert blocks[1].startswith("test_unseen") and "(1/2)" in blocks[1]
    for block in blocks:
        assert CKPT in block
        assert "sacreBLEU signature: nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp" in block
        assert "hardware:" in block and "device: cpu" in block

    log = read_test_log(run_dir)
    assert [(r["checkpoint_id"], r["split"]) for r in log] == [
        (CKPT, "test_seen"),
        (CKPT, "test_unseen"),
    ]


def test_test_runs_log_records_each_scoring(
    items: dict[str, list[EvalItem]], run_dir: Path
) -> None:
    result = run_eval(
        Lookup(),
        items["test_seen"],
        split="test_seen",
        run_dir=run_dir,
        checkpoint_id=CKPT,
        config=CFG,
    )
    raw = _lines(run_dir / TEST_LOG_NAME)
    assert len(raw) == 1
    record = json.loads(raw[0])
    assert record == read_test_log(run_dir)[0]
    assert record["checkpoint_id"] == CKPT
    assert record["split"] == "test_seen"
    assert record["n"] == 3
    assert record["config_hash"] == config_hash(CFG)
    assert record["forced"] is False and record["forced_rerun"] is False
    assert record["bleu4"] == result.metrics.bleu.bleu4
    assert record["rouge_l_f1"] == result.metrics.rouge_l.f1
    assert record["exact_match"] == result.metrics.exact_match.rate
    assert record["sacrebleu_signature"] == result.metrics.bleu.signature
    assert record["eval_dir"] == "eval/test_seen"
    assert record["time_utc"] == result.report["created_utc"]
    assert record["time_utc"].endswith("+00:00")


def test_second_test_run_for_the_same_checkpoint_is_refused_without_force(
    items: dict[str, list[EvalItem]], run_dir: Path
) -> None:
    kwargs = {"split": "test_seen", "run_dir": run_dir, "checkpoint_id": CKPT, "config": CFG}
    run_eval(Lookup(), items["test_seen"], **kwargs)  # type: ignore[arg-type]
    metrics_path = run_dir / "eval" / "test_seen" / "metrics.json"
    before = (metrics_path.read_bytes(), (run_dir / TEST_LOG_NAME).read_bytes())

    translate = Lookup()
    with pytest.raises(TestSetAlreadyScoredError, match="--force"):
        run_eval(translate, items["test_seen"], **kwargs)  # type: ignore[arg-type]
    # Refused before anything was translated or written.
    assert translate.batch_sizes == []
    assert (metrics_path.read_bytes(), (run_dir / TEST_LOG_NAME).read_bytes()) == before

    # The other test split and another checkpoint are separate scorings and are allowed.
    run_eval(
        Lookup(),
        items["test_unseen"],
        split="test_unseen",
        run_dir=run_dir,
        checkpoint_id=CKPT,
        config=CFG,
    )
    run_eval(Lookup(), items["test_seen"], **{**kwargs, "checkpoint_id": OTHER_CKPT})  # type: ignore[arg-type]
    assert len(read_test_log(run_dir)) == 3


def test_force_allows_a_second_test_run_and_logs_it(
    items: dict[str, list[EvalItem]], run_dir: Path
) -> None:
    kwargs = {"split": "test_seen", "run_dir": run_dir, "checkpoint_id": CKPT, "config": CFG}
    run_eval(Lookup(), items["test_seen"], **kwargs)  # type: ignore[arg-type]
    result = run_eval(Lookup(), items["test_seen"], force=True, **kwargs)  # type: ignore[arg-type]
    assert result.report["forced_rerun"] is True

    log = read_test_log(run_dir)
    assert len(log) == 2  # the first record is kept; the forced run is appended
    assert [(r["forced"], r["forced_rerun"]) for r in log] == [(False, False), (True, True)]
    # force on a first scoring is recorded as requested but is not a rerun.
    first = run_eval(
        Lookup(),
        items["test_unseen"],
        split="test_unseen",
        run_dir=run_dir,
        checkpoint_id=CKPT,
        config=CFG,
        force=True,
    )
    assert first.report["forced_rerun"] is False
    assert read_test_log(run_dir)[-1]["forced"] is True


def test_refusal_covers_all_requested_splits_before_any_is_scored(
    items: dict[str, list[EvalItem]], run_dir: Path
) -> None:
    run_eval(
        Lookup(),
        items["test_unseen"],
        split="test_unseen",
        run_dir=run_dir,
        checkpoint_id=CKPT,
        config=CFG,
    )
    translate = Lookup()
    with pytest.raises(TestSetAlreadyScoredError, match="test_unseen"):
        run_eval_splits(
            translate,
            {"test_seen": items["test_seen"], "test_unseen": items["test_unseen"]},
            run_dir=run_dir,
            checkpoint_id=CKPT,
            config=CFG,
        )
    assert translate.batch_sizes == []
    assert not (run_dir / "eval" / "test_seen").exists()  # test_seen was not spent
    assert len(read_test_log(run_dir)) == 1


def test_losing_the_log_does_not_reopen_the_test_set(
    items: dict[str, list[EvalItem]], run_dir: Path
) -> None:
    kwargs = {"split": "test_seen", "run_dir": run_dir, "checkpoint_id": CKPT, "config": CFG}
    run_eval(Lookup(), items["test_seen"], **kwargs)  # type: ignore[arg-type]
    (run_dir / TEST_LOG_NAME).unlink()
    with pytest.raises(TestSetAlreadyScoredError, match=r"metrics\.json"):
        run_eval(Lookup(), items["test_seen"], **kwargs)  # type: ignore[arg-type]


def test_unreadable_log_blocks_test_scoring(
    items: dict[str, list[EvalItem]], run_dir: Path
) -> None:
    (run_dir / TEST_LOG_NAME).write_text("not json\n", encoding="utf-8")
    translate = Lookup()
    with pytest.raises(EvalError, match="unreadable test log line"):
        run_eval(
            translate,
            items["test_seen"],
            split="test_seen",
            run_dir=run_dir,
            checkpoint_id=CKPT,
            config=CFG,
            force=True,
        )
    assert translate.batch_sizes == []


def test_translate_contract_is_enforced(items: dict[str, list[EvalItem]], run_dir: Path) -> None:
    kwargs = {"split": "val", "run_dir": run_dir, "checkpoint_id": CKPT, "config": CFG}
    with pytest.raises(EvalError, match="returned 1 strings for a batch of 3"):
        run_eval(lambda batch: ["x"], items["val"], **kwargs)  # type: ignore[arg-type]
    with pytest.raises(EvalError, match="NoneType for item 'v1'"):
        run_eval(lambda batch: [None] * len(batch), items["val"], **kwargs)  # type: ignore[arg-type,list-item]
    with pytest.raises(EvalError, match="list of strings"):
        run_eval(lambda batch: "hello", items["val"], **kwargs)  # type: ignore[arg-type,return-value]
    # A failed run leaves no metrics.json behind.
    assert not (run_dir / "eval" / "val" / "metrics.json").exists()


def test_bad_arguments_are_rejected(items: dict[str, list[EvalItem]], run_dir: Path) -> None:
    kwargs = {"run_dir": run_dir, "checkpoint_id": CKPT, "config": CFG}
    with pytest.raises(EvalError, match=r"split\(s\) \['val'\]"):
        run_eval(Lookup(), items["val"], split="test_seen", **kwargs)  # type: ignore[arg-type]
    with pytest.raises(EvalError, match="no items"):
        run_eval(Lookup(), [], split="val", **kwargs)  # type: ignore[arg-type]
    with pytest.raises(EvalError, match="batch_size"):
        run_eval(Lookup(), items["val"], split="val", batch_size=0, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(EvalError, match="checkpoint_id"):
        run_eval(Lookup(), items["val"], split="val", run_dir=run_dir, checkpoint_id="", config=CFG)
    # A mismatched test evaluation must not be logged as a scoring.
    assert read_test_log(run_dir) == []


def test_multiline_hypothesis_stays_on_one_line(
    items: dict[str, list[EvalItem]], run_dir: Path
) -> None:
    result = run_eval(
        lambda batch: ["Nice to\nmeet  you. "] * len(batch),
        items["val"],
        split="val",
        run_dir=run_dir,
        checkpoint_id=CKPT,
        config=CFG,
    )
    hyps = _lines(result.eval_dir / "hyps.txt")
    assert hyps == ["Nice to meet you."] * 3
    assert len(_lines(result.eval_dir / "refs.txt")) == 3
    # The files reproduce the reported metrics exactly.
    assert result.metrics == compute_metrics(hyps, _lines(result.eval_dir / "refs.txt"))


def test_config_change_changes_the_recorded_hash(
    items: dict[str, list[EvalItem]], run_dir: Path
) -> None:
    a = run_eval(
        Lookup(), items["val"], split="val", run_dir=run_dir, checkpoint_id=CKPT, config=CFG
    )
    b = run_eval(
        Lookup(),
        items["val"],
        split="val",
        run_dir=run_dir,
        checkpoint_id=CKPT,
        config=DecodeConfig(num_beams=1),
    )
    assert a.report["config_hash"] != b.report["config_hash"]
    assert len(a.report["config_hash"]) == 16
