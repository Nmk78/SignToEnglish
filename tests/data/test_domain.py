"""Loader and validator for `domain/sentences.yaml`.

The test files written here use made-up placeholder strings for `asl` and `source`. They
are not ASL and are never shown to a signer.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from signtoenglish import cli
from signtoenglish.common.config import ConfigError
from signtoenglish.data.domain import (
    check_domain,
    load_domain,
    sign_coverage,
    text_key,
    uncovered_signs,
)

HEADER = "domain: test\nsign_language: ase\ntext_language: en\n"
SMALL = (
    HEADER
    + """
vocabulary:
  - {id: A}
  - {id: B}
  - {id: NO}
sentences:
  - {id: t1, role: train, text: "One.", signs: [A, B], asl: null, source: null}
  - {id: t2, role: train, text: "Two.", signs: [A, B, NO], asl: null, source: null}
  - {id: t3, role: train, text: "Three.", signs: [A, B], asl: null, source: null}
  - {id: u1, role: unseen_test, text: "Four.", signs: [B, A], asl: null, source: null}
"""
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "sentences.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _codes(tmp_path: Path, text: str, level: str = "error") -> list[str]:
    report = check_domain(load_domain(_write(tmp_path, text)))
    return sorted(i.code for i in report.issues if i.level == level)


def test_committed_sentence_list_loads_and_has_no_errors(repo_root: Path) -> None:
    domain = load_domain(repo_root / "domain" / "sentences.yaml")
    assert domain.sign_language == "ase" and domain.text_language == "en"
    assert len(domain.sentences) == 103
    assert len(domain.with_role("train")) == 83
    assert len(domain.with_role("unseen_test")) == 20
    assert len(domain.vocabulary) == 76
    report = check_domain(domain)
    assert report.errors == []


def test_committed_list_has_only_text_ids_and_signs(repo_root: Path) -> None:
    domain = load_domain(repo_root / "domain" / "sentences.yaml")
    assert all(type(v.id) is str and v.id for v in domain.vocabulary)
    assert all(type(s.id) is str and s.id for s in domain.sentences)
    assert all(type(sign) is str and sign for s in domain.sentences for sign in s.signs)
    vocabulary = {v.id for v in domain.vocabulary}
    assert {"YES", "NO"} <= vocabulary
    assert "NO" in domain.by_id()["t65"].signs and "YES" in domain.by_id()["t33"].signs
    # the shared loader agrees, now that YES and NO are quoted in the file
    from signtoenglish.common.config import load_yaml

    raw = load_yaml(repo_root / "domain" / "sentences.yaml")
    assert [v["id"] for v in raw["vocabulary"]] == [v.id for v in domain.vocabulary]
    assert [tuple(s["signs"]) for s in raw["sentences"]] == [s.signs for s in domain.sentences]


def test_an_unquoted_yes_or_no_typed_by_hand_is_still_a_sign_id(tmp_path: Path) -> None:
    domain = load_domain(_write(tmp_path, SMALL))  # SMALL writes NO without quotes
    assert "NO" in {v.id for v in domain.vocabulary} and "NO" in domain.by_id()["t2"].signs


def test_committed_list_meets_the_coverage_rule_counted_independently(repo_root: Path) -> None:
    domain = load_domain(repo_root / "domain" / "sentences.yaml")
    counts: Counter[str] = Counter()
    for sentence in domain.sentences:
        if sentence.role == "train":
            counts.update(set(sentence.signs))
    for sentence in domain.sentences:
        if sentence.role == "unseen_test":
            assert sentence.signs
            for sign in sentence.signs:
                assert counts[sign] >= 3, (sentence.id, sign, counts[sign])
    assert sign_coverage(domain) == counts


def test_no_text_is_shared_between_training_and_unseen_sentences(repo_root: Path) -> None:
    domain = load_domain(repo_root / "domain" / "sentences.yaml")
    train = {text_key(s.text) for s in domain.with_role("train")}
    unseen = {text_key(s.text) for s in domain.with_role("unseen_test")}
    assert train.isdisjoint(unseen)


def test_every_sentence_without_asl_or_source_gets_its_own_warning(repo_root: Path) -> None:
    """Today every `asl` and `source` in the committed list is empty: 103 + 103 warnings."""
    domain = load_domain(repo_root / "domain" / "sentences.yaml")
    report = check_domain(domain)
    empty_asl = [s.id for s in domain.sentences if not (s.asl or "").strip()]
    empty_source = [s.id for s in domain.sentences if not (s.source or "").strip()]
    assert [i.sentence_id for i in report.with_code("asl_missing")] == empty_asl
    assert [i.sentence_id for i in report.with_code("source_missing")] == empty_source
    assert all(i.level == "warning" for i in report.with_code("asl_missing"))


def test_warnings_follow_the_fields_sentence_by_sentence(tmp_path: Path) -> None:
    text = SMALL.replace(
        '{id: t1, role: train, text: "One.", signs: [A, B], asl: null, source: null}',
        '{id: t1, role: train, text: "One.", signs: [A, B], '
        'asl: "PLACEHOLDER-NOT-ASL", source: "placeholder source"}',
    ).replace(
        '{id: t2, role: train, text: "Two.", signs: [A, B, NO], asl: null, source: null}',
        '{id: t2, role: train, text: "Two.", signs: [A, B, NO], '
        'asl: "PLACEHOLDER-NOT-ASL", source: "   "}',
    )
    report = check_domain(load_domain(_write(tmp_path, text)))
    assert [i.sentence_id for i in report.with_code("asl_missing")] == ["t3", "u1"]
    assert [i.sentence_id for i in report.with_code("source_missing")] == ["t2", "t3", "u1"]
    assert report.ok


def test_small_list_is_clean_apart_from_missing_asl(tmp_path: Path) -> None:
    assert _codes(tmp_path, SMALL) == []
    assert set(_codes(tmp_path, SMALL, "warning")) == {"asl_missing", "source_missing"}


def test_duplicate_sentence_id(tmp_path: Path) -> None:
    text = SMALL.replace("id: t2,", "id: t1,")
    assert "duplicate_id" in _codes(tmp_path, text)


def test_duplicate_text_ignores_case_and_punctuation(tmp_path: Path) -> None:
    text = SMALL.replace('text: "Four."', 'text: "one"')
    report = check_domain(load_domain(_write(tmp_path, text)))
    assert [(i.code, i.sentence_id) for i in report.errors] == [("duplicate_text", "u1")]
    assert text_key("  Hello,   HOW are you?? ") == text_key("hello how are you")
    assert text_key("I don't know.") == text_key("I dont know")
    assert text_key("Yes.") != text_key("No.")


def test_empty_text(tmp_path: Path) -> None:
    assert "empty_text" in _codes(tmp_path, SMALL.replace('text: "Two."', 'text: "  "'))


def test_sign_missing_from_the_vocabulary(tmp_path: Path) -> None:
    text = SMALL.replace("signs: [A, B, NO]", "signs: [A, B, NO, ZZZ]")
    report = check_domain(load_domain(_write(tmp_path, text)))
    assert [(i.code, i.sentence_id) for i in report.errors] == [("unknown_sign", "t2")]


def test_unseen_sign_in_fewer_than_three_training_sentences(tmp_path: Path) -> None:
    text = SMALL.replace("signs: [B, A]", "signs: [B, NO]")  # NO is in one training sentence
    report = check_domain(load_domain(_write(tmp_path, text)))
    assert [(i.code, i.sentence_id) for i in report.errors] == [("coverage", "u1")]
    assert "NO (1)" in report.errors[0].message
    domain = load_domain(_write(tmp_path, text))
    assert uncovered_signs(domain, ["u1"]) == {"u1": {"NO": 1}}
    # counting only sentences that really have a training take makes it stricter
    assert uncovered_signs(domain, ["u1"], train_sentence_ids={"t1", "t2"}) == {
        "u1": {"B": 2, "NO": 1}
    }


def test_unseen_sentence_without_signs_is_an_error(tmp_path: Path) -> None:
    assert _codes(tmp_path, SMALL.replace("signs: [B, A]", "signs: []")) == ["no_signs"]


def test_vocabulary_warnings(tmp_path: Path) -> None:
    text = SMALL.replace("- {id: A}", "- {id: A, train_sentences: 7}\n  - {id: UNUSED}")
    warnings = _codes(tmp_path, text, "warning")
    assert "stale_annotation" in warnings and "unused_sign" in warnings


def test_malformed_files_raise_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="role"):
        load_domain(_write(tmp_path, SMALL.replace("role: unseen_test", "role: test")))
    with pytest.raises(ConfigError, match="glosses"):
        load_domain(_write(tmp_path, SMALL.replace("signs: [B, A]", "glosses: [B, A]")))
    with pytest.raises(ConfigError, match="duplicate key"):
        load_domain(_write(tmp_path, SMALL + "domain: again\n"))
    with pytest.raises(ConfigError, match="cannot read"):
        load_domain(tmp_path / "missing.yaml")


def test_cli_domain_check(
    repo_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    committed = str(repo_root / "domain" / "sentences.yaml")
    assert cli.main(["domain", "check", "--sentences", committed]) == 0
    out = capsys.readouterr().out
    assert out.count("WARNING asl_missing") == 103
    assert out.count("WARNING source_missing") == 103
    assert "103 sentence(s) without an ASL version, 103 without a source" in out
    assert cli.main(["domain", "check", "--sentences", committed, "--strict"]) == 1
    capsys.readouterr()

    broken = _write(tmp_path, SMALL.replace("id: t2,", "id: t1,"))
    assert cli.main(["domain", "check", "--sentences", str(broken)]) == 1
    assert "ERROR duplicate_id" in capsys.readouterr().out
