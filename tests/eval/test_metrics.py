"""Metric wrappers against values derived by hand (no number here comes from the libraries).

The three-sentence fixture:

    #  hypothesis                 reference
    1  hello my name is Alex      Hello, my name is Alex.
    2  Where do you work now?     Where do you live?
    3  Nice to meet you.          Nice to meet you.

BLEU (sacreBLEU defaults: case-sensitive, tokenizer 13a, which splits punctuation off)

    Tokens after 13a:
    1  hyp: hello my name is Alex            (5)   ref: Hello , my name is Alex .   (7)
    2  hyp: Where do you work now ?          (6)   ref: Where do you live ?         (5)
    3  hyp: Nice to meet you .               (5)   ref: Nice to meet you .          (5)

    Matched / total hypothesis n-grams (no n-gram repeats, so clipping changes nothing):
    order  pair 1                               pair 2                       pair 3   corpus
    1      4/5 (my name is Alex; "hello" is     4/6 (Where do you ?)         5/5      13/16
                not "Hello")
    2      3/4 (my name, name is, is Alex)      2/5 (Where do, do you)       4/4       9/13
    3      2/3 (my name is, name is Alex)       1/4 (Where do you)           3/3       6/10
    4      1/2 (my name is Alex)                0/3                          2/2       3/7

    Hypothesis length 5 + 6 + 5 = 16, reference length 7 + 5 + 5 = 17.
    16 < 17, so the brevity penalty is exp(1 - 17/16) = exp(-1/16) = 0.939413.
    Every corpus count is non-zero, so smoothing does not apply.

    BLEU-1 = 100 * BP * 13/16                                   = 76.327
    BLEU-2 = 100 * BP * (13/16 * 9/13)^(1/2) = 100 * BP * 3/4   = 70.456
    BLEU-3 = 100 * BP * (13/16 * 9/13 * 6/10)^(1/3)             = 65.406
    BLEU-4 = 100 * BP * (13/16 * 9/13 * 6/10 * 3/7)^(1/4)       = 57.934

ROUGE-L (rouge-score tokenizer: lowercase, keep [a-z0-9] runs; LCS = longest common
subsequence; P = LCS/hyp tokens, R = LCS/ref tokens, F1 = 2PR/(P+R))

    1  hello my name is alex  vs  hello my name is alex    LCS 5   P 5/5  R 5/5  F1 1
    2  where do you work now  vs  where do you live        LCS 3   P 3/5  R 3/4  F1 2/3
    3  nice to meet you       vs  nice to meet you         LCS 4   P 1    R 1    F1 1

    Mean F1 = (1 + 2/3 + 1)/3 = 8/9; mean P = 2.6/3; mean R = 2.75/3.

Exact match after `normalize_for_match`: pair 1 matches (case and punctuation only),
pair 2 does not, pair 3 does: 2 of 3.
"""

from __future__ import annotations

import math
import socket

import pytest

from signtoenglish.eval.metrics import (
    bleu_scores,
    compute_metrics,
    exact_match,
    metric_versions,
    normalize_for_match,
    rouge_l_scores,
)

HYPS = ["hello my name is Alex", "Where do you work now?", "Nice to meet you."]
REFS = ["Hello, my name is Alex.", "Where do you live?", "Nice to meet you."]

BP = math.exp(1 - 17 / 16)
P1, P2, P3, P4 = 13 / 16, 9 / 13, 6 / 10, 3 / 7


def test_bleu_matches_hand_computed_values() -> None:
    b = bleu_scores(HYPS, REFS)
    assert b.counts == (13, 9, 6, 3)
    assert b.totals == (16, 13, 10, 7)
    assert (b.sys_len, b.ref_len) == (16, 17)
    assert b.brevity_penalty == pytest.approx(BP, abs=1e-9)
    assert b.bleu1 == pytest.approx(100 * BP * P1, abs=1e-6)
    assert b.bleu2 == pytest.approx(100 * BP * (P1 * P2) ** (1 / 2), abs=1e-6)
    assert b.bleu3 == pytest.approx(100 * BP * (P1 * P2 * P3) ** (1 / 3), abs=1e-6)
    assert b.bleu4 == pytest.approx(100 * BP * (P1 * P2 * P3 * P4) ** (1 / 4), abs=1e-6)
    # The same values as decimals, worked out by hand to three places.
    assert b.brevity_penalty == pytest.approx(0.939413, abs=1e-6)
    assert b.bleu1 == pytest.approx(76.327, abs=5e-3)
    assert b.bleu2 == pytest.approx(70.456, abs=5e-3)
    assert b.bleu3 == pytest.approx(65.406, abs=5e-3)
    assert b.bleu4 == pytest.approx(57.934, abs=5e-3)


def test_bleu_signature_is_recorded_and_states_the_settings() -> None:
    sig = bleu_scores(HYPS, REFS).signature
    fields = dict(part.split(":", 1) for part in sig.split("|"))
    assert fields["nrefs"] == "1"
    assert fields["case"] == "mixed"
    assert fields["tok"] == "13a"
    assert fields["smooth"] == "exp"
    assert fields["version"] == metric_versions()["sacrebleu"] == "2.6.0"


def test_bleu_clips_repeated_ngrams() -> None:
    # hyp "the the the" (3 tokens) vs ref "the cat" (2 tokens): "the" occurs once in the
    # reference, so only 1 of the 3 hypothesis unigrams counts. The hypothesis is longer
    # than the reference, so BP = 1 and BLEU-1 = 100 * 1/3.
    b = bleu_scores(["the the the"], ["the cat"])
    assert b.counts[0] == 1
    assert b.totals[0] == 3
    assert b.brevity_penalty == 1.0
    assert b.bleu1 == pytest.approx(100 / 3, abs=1e-6)


def test_bleu_is_100_for_identical_text_and_is_case_sensitive() -> None:
    assert bleu_scores(REFS, REFS).bleu4 == pytest.approx(100.0)
    assert bleu_scores([r.lower() for r in REFS], REFS).bleu4 < 100.0


def test_rouge_l_matches_hand_computed_values() -> None:
    r = rouge_l_scores(HYPS, REFS)
    assert r.f1 == pytest.approx(100 * (1 + 2 / 3 + 1) / 3, abs=1e-6)
    assert r.precision == pytest.approx(100 * (1 + 3 / 5 + 1) / 3, abs=1e-6)
    assert r.recall == pytest.approx(100 * (1 + 3 / 4 + 1) / 3, abs=1e-6)
    assert r.f1 == pytest.approx(88.889, abs=5e-3)


def test_rouge_l_uses_subsequence_not_substring() -> None:
    # hyp "a b c d" vs ref "a x c d": LCS is "a c d" = 3 of 4 on both sides, F1 = 3/4.
    assert rouge_l_scores(["a b c d"], ["a x c d"]).f1 == pytest.approx(75.0)
    # No common token at all.
    assert rouge_l_scores(["one two"], ["three four"]).f1 == 0.0


def test_exact_match_matches_hand_computed_values() -> None:
    em = exact_match(HYPS, REFS)
    assert (em.n_match, em.n) == (2, 3)
    assert em.rate == pytest.approx(100 * 2 / 3)


def test_compute_metrics_bundles_the_three() -> None:
    m = compute_metrics(HYPS, REFS)
    assert m.n == 3
    assert m.bleu == bleu_scores(HYPS, REFS)
    assert m.rouge_l == rouge_l_scores(HYPS, REFS)
    assert m.exact_match == exact_match(HYPS, REFS)
    d = m.to_dict()
    assert d["bleu"]["counts"] == [13, 9, 6, 3]
    assert d["bleu"]["signature"] == m.bleu.signature
    assert d["exact_match"]["n_match"] == 2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello, my name is Alex.", "hello my name is alex"),
        ("  NICE   to\tmeet you!! ", "nice to meet you"),
        ("I'm fine.", "im fine"),
        ("I\u2019m fine.", "im fine"),  # typographic apostrophe
        ("Im fine", "im fine"),
        ("twenty-one", "twenty one"),
        ("twenty one", "twenty one"),
        ("\u201cWhere do you live?\u201d", "where do you live"),  # curly quotes
        ("\u00bfQu\u00e9?", "qu\u00e9"),  # non-ASCII punctuation goes, letters stay
        ("I am 21 years old.", "i am 21 years old"),
        ("?!.", ""),
        ("", ""),
    ],
)
def test_normalize_for_match(raw: str, expected: str) -> None:
    assert normalize_for_match(raw) == expected


def test_normalize_for_match_is_idempotent_and_keeps_word_differences() -> None:
    for text in [*HYPS, *REFS, "I\u2019m twenty-one!"]:
        once = normalize_for_match(text)
        assert normalize_for_match(once) == once
    # Only case and punctuation are forgiven: a different word or word order is a miss.
    assert normalize_for_match("Where do you work?") != normalize_for_match("Where do you live?")
    assert normalize_for_match("you do live where") != normalize_for_match("where do you live")
    assert exact_match(["an apple"], ["a apple"]).n_match == 0


def test_exact_match_does_not_count_an_empty_hypothesis() -> None:
    assert exact_match(["", "..."], ["Hello.", "Hello."]).n_match == 0


def test_empty_hypothesis_scores_zero_without_raising() -> None:
    m = compute_metrics([""], ["Nice to meet you."])
    assert m.bleu.bleu4 == 0.0
    assert m.rouge_l.f1 == 0.0
    assert m.exact_match.rate == 0.0


@pytest.mark.parametrize(
    ("hyps", "refs", "error"),
    [
        (["a"], ["a", "b"], ValueError),  # misaligned
        ([], [], ValueError),  # nothing to score
        (["a"], ["  "], ValueError),  # empty reference
        ("abc", ["abc"], TypeError),  # a bare string is not a list of sentences
        (["a"], [None], TypeError),
    ],
)
def test_bad_input_is_rejected(hyps: object, refs: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        compute_metrics(hyps, refs)  # type: ignore[arg-type]


def test_metrics_make_no_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing may be fetched at run time (for example NLTK data or sacreBLEU test sets)."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("metric code tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    m = compute_metrics(HYPS, REFS)
    assert m.exact_match.n_match == 2
