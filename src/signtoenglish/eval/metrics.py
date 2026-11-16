"""Text metrics: sacreBLEU (BLEU-1 to BLEU-4 with signature), ROUGE-L, and exact match.

Conventions, fixed here and nowhere else (ARCHITECTURE.md, section 7):

- Every score is on a 0 to 100 scale.
- **BLEU** is corpus-level sacreBLEU on the untouched text (case-sensitive, sacreBLEU's
  default `13a` tokenizer, default `exp` smoothing, one reference per hypothesis).
  BLEU-n is the cumulative score: brevity penalty times the geometric mean of the 1-gram
  to n-gram precisions (`max_ngram_order=n`). The sacreBLEU signature does not encode the
  n-gram order, so it is the same string for all four and is recorded once.
- **ROUGE-L** is `rouge-score`'s sentence-level `rougeL` without stemming, averaged over
  sentences (mean F1, with mean precision and recall alongside). Its tokenizer lowercases
  and keeps only `[a-z0-9]` runs, which is adequate for English references. Nothing here
  touches NLTK data: `rouge_score` imports `nltk` at module import (code only), and calls
  `nltk.sent_tokenize`, which needs downloaded data, only for `rougeLsum` with
  `split_summaries=True`, which this module never uses.
- **Exact match** compares `normalize_for_match(hypothesis)` with
  `normalize_for_match(reference)`. That function is the single definition of the "light
  normalization (case and punctuation)"; BLEU and ROUGE-L do not use it.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any

from rouge_score import rouge_scorer
from sacrebleu.metrics.bleu import BLEU

MAX_NGRAM_ORDER = 4
_APOSTROPHES = frozenset("'\u2018\u2019\u02bc\uff07")


def normalize_for_match(text: str) -> str:
    """Light normalization for exact match: case and punctuation only.

    Steps, in order: Unicode NFKC; case-fold; delete apostrophes (so "I'm", "I\u2019m",
    and "Im" agree); replace every other Unicode punctuation character with a space (so
    "twenty-one" and "twenty one" agree); collapse whitespace runs and strip the ends.
    Words, their order, digits, and symbols such as "+" or "$" are left alone.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    out: list[str] = []
    for ch in text:
        if ch in _APOSTROPHES:
            continue
        out.append(" " if unicodedata.category(ch).startswith("P") else ch)
    return " ".join("".join(out).split())


@dataclass(frozen=True)
class BleuScores:
    """Cumulative corpus BLEU-1 to BLEU-4 (0 to 100) and what they were computed from."""

    bleu1: float
    bleu2: float
    bleu3: float
    bleu4: float
    signature: str
    brevity_penalty: float
    sys_len: int
    ref_len: int
    # Per order 1..4, from the BLEU-4 computation: clipped matches, hypothesis n-gram
    # totals, and the (smoothed, if a count was zero) precisions in percent.
    counts: tuple[int, ...]
    totals: tuple[int, ...]
    precisions: tuple[float, ...]


@dataclass(frozen=True)
class RougeLScores:
    """Mean sentence-level ROUGE-L (0 to 100)."""

    f1: float
    precision: float
    recall: float


@dataclass(frozen=True)
class ExactMatch:
    """Share of hypotheses equal to their reference after `normalize_for_match` (0 to 100)."""

    rate: float
    n_match: int
    n: int


@dataclass(frozen=True)
class TextMetrics:
    n: int
    bleu: BleuScores
    rouge_l: RougeLScores
    exact_match: ExactMatch

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form (tuples become lists)."""
        data = asdict(self)
        for key in ("counts", "totals", "precisions"):
            data["bleu"][key] = list(data["bleu"][key])
        return data


def _check_pairs(hypotheses: Sequence[str], references: Sequence[str]) -> None:
    if isinstance(hypotheses, str) or isinstance(references, str):
        raise TypeError("hypotheses and references must be sequences of strings, not one string")
    if len(hypotheses) != len(references):
        raise ValueError(f"{len(hypotheses)} hypotheses but {len(references)} references")
    if len(hypotheses) == 0:
        raise ValueError("cannot score an empty set of sentences")
    for name, seq in (("hypothesis", hypotheses), ("reference", references)):
        for i, s in enumerate(seq):
            if not isinstance(s, str):
                raise TypeError(f"{name} {i} is {type(s).__name__}, expected str")
    for i, ref in enumerate(references):
        if not ref.strip():
            raise ValueError(f"reference {i} is empty")


def bleu_scores(hypotheses: Sequence[str], references: Sequence[str]) -> BleuScores:
    """Corpus sacreBLEU, cumulative orders 1 to 4, one reference per hypothesis."""
    _check_pairs(hypotheses, references)
    hyps, refs = list(hypotheses), [list(references)]
    scores: list[float] = []
    signature = ""
    last: Any = None
    for order in range(1, MAX_NGRAM_ORDER + 1):
        metric = BLEU(max_ngram_order=order)
        last = metric.corpus_score(hyps, refs)
        scores.append(float(last.score))
        signature = str(metric.get_signature())
    return BleuScores(
        bleu1=scores[0],
        bleu2=scores[1],
        bleu3=scores[2],
        bleu4=scores[3],
        signature=signature,
        brevity_penalty=float(last.bp),
        sys_len=int(last.sys_len),
        ref_len=int(last.ref_len),
        counts=tuple(int(c) for c in last.counts),
        totals=tuple(int(t) for t in last.totals),
        precisions=tuple(float(p) for p in last.precisions),
    )


def rouge_l_scores(hypotheses: Sequence[str], references: Sequence[str]) -> RougeLScores:
    """Mean sentence-level ROUGE-L, no stemming."""
    _check_pairs(hypotheses, references)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    f1 = precision = recall = 0.0
    for hyp, ref in zip(hypotheses, references, strict=True):
        score = scorer.score(ref, hyp)["rougeL"]  # signature is (target, prediction)
        f1 += float(score.fmeasure)
        precision += float(score.precision)
        recall += float(score.recall)
    n = len(hypotheses)
    return RougeLScores(
        f1=100.0 * f1 / n, precision=100.0 * precision / n, recall=100.0 * recall / n
    )


def exact_match(hypotheses: Sequence[str], references: Sequence[str]) -> ExactMatch:
    _check_pairs(hypotheses, references)
    n_match = sum(
        normalize_for_match(hyp) == normalize_for_match(ref)
        for hyp, ref in zip(hypotheses, references, strict=True)
    )
    return ExactMatch(rate=100.0 * n_match / len(hypotheses), n_match=n_match, n=len(hypotheses))


def compute_metrics(hypotheses: Sequence[str], references: Sequence[str]) -> TextMetrics:
    """All three metrics for aligned hypotheses and references."""
    return TextMetrics(
        n=len(hypotheses),
        bleu=bleu_scores(hypotheses, references),
        rouge_l=rouge_l_scores(hypotheses, references),
        exact_match=exact_match(hypotheses, references),
    )


def metric_versions() -> dict[str, str]:
    """Installed versions of the metric libraries, for `metrics.json`."""
    return {
        "sacrebleu": metadata.version("sacrebleu"),
        "rouge-score": metadata.version("rouge-score"),
    }
