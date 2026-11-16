"""Split builder guarantees (ARCHITECTURE.md, section 4.3), on the committed sentence list.

Rows here are built in memory: the builder works on manifest rows and the sentence list
and never opens video. The end-to-end test that goes through real recorded files is
tests/record/test_end_to_end.py.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest

from signtoenglish.data.domain import DomainSentences, load_domain, text_key
from signtoenglish.data.manifest import (
    ClipRow,
    ContinuousRow,
    Segment,
    load_clip_rows,
    manifest_path,
    validate_manifests,
)
from signtoenglish.data.splits import (
    CLIP_SPLITS,
    SplitConfig,
    SplitError,
    build_splits,
    check_split_guarantees,
    choose_val_unseen,
    require_unassigned,
    write_splits,
)

SESSIONS = ("2026-11-20_desk", "2026-11-24_window", "2026-11-28_show")


@pytest.fixture(scope="module")
def domain(repo_root: Path) -> DomainSentences:
    return load_domain(repo_root / "domain" / "sentences.yaml")


def _row(domain: DomainSentences, sentence_id: str, session: str, take: int) -> ClipRow:
    name = f"{session}_{sentence_id}_me_t{take:02d}"
    return ClipRow(
        id=name,
        dataset="domain",
        split="unassigned",
        source_path=f"domain/{session}/{sentence_id}_me_take{take:02d}.mp4",
        source_sha256=hashlib.sha256(name.encode()).hexdigest(),
        start_ms=None,
        end_ms=None,
        text=domain.by_id()[sentence_id].text,
        sentence_id=sentence_id,
        signer_id="me",
        session_id=session,
        take=take,
        sign_language="ase",
        text_language="en",
        license="own-recording",
        tags=("show_conditions",) if session.endswith("show") else (),
    )


def _recorded(domain: DomainSentences, takes_per_session: int = 2) -> list[ClipRow]:
    """Every sentence, `takes_per_session` takes in each of three sessions."""
    return [
        _row(domain, s.id, session, take)
        for s in domain.sentences
        for session in SESSIONS
        for take in range(1, takes_per_session + 1)
    ]


def _continuous(domain: DomainSentences, take: int, sentence_ids: list[str]) -> ContinuousRow:
    name = f"{SESSIONS[0]}_continuous_me_t{take:02d}"
    return ContinuousRow(
        id=name,
        dataset="domain",
        split="unassigned",
        source_path=f"domain/{SESSIONS[0]}/continuous_me_take{take:02d}.mp4",
        source_sha256=hashlib.sha256(name.encode()).hexdigest(),
        signer_id="me",
        session_id=SESSIONS[0],
        take=take,
        sign_language="ase",
        text_language="en",
        license="own-recording",
        segments=tuple(
            Segment(
                sentence_id=s,
                text=domain.by_id()[s].text,
                start_ms=i * 4000,
                end_ms=(i + 1) * 4000,
            )
            for i, s in enumerate(sentence_ids)
        ),
    )


# ---- the three guarantees, each checked by code written here, not by the builder ---------


def test_no_recording_appears_in_two_splits(domain: DomainSentences) -> None:
    rows = _recorded(domain)
    result = build_splits(rows, domain)
    placed = [r for name in CLIP_SPLITS for r in result.clips[name]]
    assert len(placed) == len(rows)
    for attribute in ("id", "source_path", "source_sha256"):
        values = [getattr(r, attribute) for r in placed]
        assert len(set(values)) == len(values) == len(rows)
    assert {r.id for r in placed} == {r.id for r in rows}  # nothing lost, nothing invented
    for name in CLIP_SPLITS:
        assert all(r.split == name for r in result.clips[name])


def test_no_unseen_test_sentence_text_in_train_or_val(domain: DomainSentences) -> None:
    result = build_splits(_recorded(domain), domain)
    unseen_ids = {r.sentence_id for r in result.clips["test_unseen"]}
    unseen_texts = {text_key(r.text) for r in result.clips["test_unseen"]}
    assert len(unseen_ids) == 15
    for name in ("train", "val"):
        assert unseen_ids.isdisjoint({r.sentence_id for r in result.clips[name]})
        assert unseen_texts.isdisjoint({text_key(r.text) for r in result.clips[name]})
        assert unseen_texts.isdisjoint({r.text for r in result.clips[name]})
    # and nothing held out, for test or for validation, is ever trained on
    roles = {s.id: s.role for s in domain.sentences}
    assert {roles[r.sentence_id or ""] for r in result.clips["train"]} == {"train"}
    assert {roles[r.sentence_id or ""] for r in result.clips["test_unseen"]} == {"unseen_test"}


def test_every_unseen_test_sign_is_in_at_least_three_training_sentences(
    domain: DomainSentences,
) -> None:
    result = build_splits(_recorded(domain), domain)
    sentences = domain.by_id()
    train_sentence_ids = {r.sentence_id for r in result.clips["train"]}
    counts: Counter[str] = Counter()
    for sentence_id in train_sentence_ids:
        counts.update(set(sentences[sentence_id or ""].signs))
    checked = 0
    for sentence_id in {r.sentence_id for r in result.clips["test_unseen"]}:
        for sign in sentences[sentence_id or ""].signs:
            assert counts[sign] >= 3, (sentence_id, sign, counts[sign])
            checked += 1
    assert checked > 30


# ---- what goes where ----------------------------------------------------------------------


def test_split_contents(domain: DomainSentences) -> None:
    result = build_splits(_recorded(domain), domain)  # 6 takes per sentence
    per_split = {name: Counter(r.sentence_id for r in result.clips[name]) for name in CLIP_SPLITS}
    train_ids = [s.id for s in domain.with_role("train")]
    assert len(result.val_unseen_ids) == 5 and len(result.test_unseen_ids) == 15
    for sentence_id in train_ids:
        assert per_split["train"][sentence_id] == 4
        assert per_split["val"][sentence_id] == 1
        assert per_split["test_seen"][sentence_id] == 1
    for sentence_id in result.val_unseen_ids:
        assert per_split["val"][sentence_id] == 6
    for sentence_id in result.test_unseen_ids:
        assert per_split["test_unseen"][sentence_id] == 6
    assert set(per_split["test_seen"]) == set(train_ids)
    # held-out takes are the latest ones: test_seen comes from the last session
    assert {r.session_id for r in result.clips["test_seen"]} == {SESSIONS[-1]}
    assert result.warnings == []


def test_build_is_deterministic_and_order_independent(domain: DomainSentences) -> None:
    rows = _recorded(domain)
    first = build_splits(rows, domain)
    second = build_splits(list(reversed(rows)), domain)
    assert first.clips == second.clips
    assert first.val_unseen_ids == second.val_unseen_ids


def test_held_out_sentences_do_not_move_when_takes_are_added(domain: DomainSentences) -> None:
    cfg = SplitConfig()
    chosen = choose_val_unseen(domain, cfg)
    assert build_splits(_recorded(domain, 1), domain, cfg).val_unseen_ids == chosen
    assert build_splits(_recorded(domain, 3), domain, cfg).val_unseen_ids == chosen
    assert choose_val_unseen(domain, SplitConfig(seed=1)) != chosen
    assert len(choose_val_unseen(domain, SplitConfig(val_unseen_count=0))) == 0
    with pytest.raises(SplitError):
        choose_val_unseen(domain, SplitConfig(val_unseen_count=21))


def test_holdout_session_is_never_used_for_train_or_val(domain: DomainSentences) -> None:
    cfg = SplitConfig(holdout_session=SESSIONS[1])
    result = build_splits(_recorded(domain), domain, cfg)
    assert SESSIONS[1] not in {r.session_id for r in result.clips["train"]}
    held_in_val = [
        r
        for r in result.clips["val"]
        if r.session_id == SESSIONS[1] and r.sentence_id not in result.val_unseen_ids
    ]
    assert held_in_val == []
    assert {r.session_id for r in result.clips["test_seen"]} == {SESSIONS[1]}
    with pytest.raises(SplitError, match="no takes"):
        build_splits(_recorded(domain), domain, SplitConfig(holdout_session="2026-01-01_x"))


def test_scarce_takes_go_to_train_first_then_val_then_test(domain: DomainSentences) -> None:
    rows = [r for r in _recorded(domain) if r.sentence_id not in ("t01", "t02", "t03")]
    rows.append(_row(domain, "t01", SESSIONS[0], 1))
    rows += [_row(domain, "t02", SESSIONS[0], take) for take in (1, 2)]
    # coverage is not what this test is about; t03 has no takes at all here
    result = build_splits(rows, domain, SplitConfig(min_train_sentences_per_sign=0))
    where = {
        sentence_id: [name for name in CLIP_SPLITS for r in result.clips[name]
                      if r.sentence_id == sentence_id]
        for sentence_id in ("t01", "t02", "t03")
    }  # fmt: skip
    assert where == {"t01": ["train"], "t02": ["train", "val"], "t03": []}
    assert any(w.startswith("t03: no takes recorded") for w in result.warnings)
    assert any(w.startswith("t01:") and "val" in w for w in result.warnings)


# ---- the builder refuses, and the checker catches, broken splits --------------------------


def test_build_fails_when_training_takes_do_not_cover_the_unseen_signs(
    domain: DomainSentences,
) -> None:
    # Drop all takes of every training sentence that uses SCHOOL: unseen sentences that
    # need SCHOOL are then not covered by anything that is really trained on.
    uses_school = {s.id for s in domain.with_role("train") if "SCHOOL" in s.signs}
    assert len(uses_school) >= 3
    rows = [r for r in _recorded(domain) if r.sentence_id not in uses_school]
    with pytest.raises(SplitError, match=r"SCHOOL \(0\)"):
        build_splits(rows, domain, SplitConfig(val_unseen_count=0))


def test_unknown_sentence_or_changed_text_is_refused(domain: DomainSentences) -> None:
    good = _recorded(domain, 1)
    with pytest.raises(SplitError, match="not in the list"):
        build_splits([*good, good[0].model_copy(update={"id": "x", "sentence_id": "zz"})], domain)
    with pytest.raises(SplitError, match="text differs"):
        build_splits([*good, good[0].model_copy(update={"id": "x", "text": "Hi."})], domain)


def test_checker_catches_a_recording_in_two_splits(domain: DomainSentences) -> None:
    result = build_splits(_recorded(domain), domain)
    assert check_split_guarantees(result.clips, domain) == []
    leaked = result.clips["train"][0]

    by_copy = {k: list(v) for k, v in result.clips.items()}
    by_copy["test_seen"].append(leaked.model_copy(update={"split": "test_seen"}))
    assert len(check_split_guarantees(by_copy, domain)) == 3  # id, file, and content hash

    # the same recording under a new id and a new file name is still caught, by its hash
    renamed = {k: list(v) for k, v in result.clips.items()}
    renamed["val"].append(
        leaked.model_copy(update={"split": "val", "id": "new", "source_path": "domain/x/n.mp4"})
    )
    violations = check_split_guarantees(renamed, domain)
    assert len(violations) == 1 and "content hash" in violations[0]


def test_checker_catches_unseen_text_in_train_or_val(domain: DomainSentences) -> None:
    result = build_splits(_recorded(domain), domain)
    unseen = result.clips["test_unseen"][0]
    for target in ("train", "val"):
        # same sentence id
        broken = {k: list(v) for k, v in result.clips.items()}
        broken[target].append(
            unseen.model_copy(
                update={
                    "split": target,
                    "id": "leak",
                    "source_path": "domain/x/leak.mp4",
                    "source_sha256": "f" * 64,
                }
            )
        )
        assert any("test_unseen sentence" in v for v in check_split_guarantees(broken, domain))

        # same words under a training sentence's id, with different case and punctuation
        disguised = {k: list(v) for k, v in result.clips.items()}
        disguised[target].append(
            unseen.model_copy(
                update={
                    "split": target,
                    "id": "leak2",
                    "source_path": "domain/x/leak2.mp4",
                    "source_sha256": "e" * 64,
                    "sentence_id": "t01",
                    "text": unseen.text.upper().replace("?", "").replace(".", ""),
                }
            )
        )
        violations = check_split_guarantees(disguised, domain)
        assert any("text of test_unseen sentence" in v for v in violations), violations


def test_checker_catches_thin_sign_coverage(domain: DomainSentences) -> None:
    result = build_splits(_recorded(domain), domain)
    sign = domain.by_id()[result.test_unseen_ids[0]].signs[0]
    users = {s.id for s in domain.with_role("train") if sign in s.signs}
    thinned = {k: list(v) for k, v in result.clips.items()}
    thinned["train"] = [r for r in thinned["train"] if r.sentence_id not in users]
    violations = check_split_guarantees(thinned, domain)
    assert any(
        v.startswith(f"test_unseen sentence {result.test_unseen_ids[0]}") and f"{sign} (0)" in v
        for v in violations
    ), violations
    assert check_split_guarantees(thinned, domain, min_train_sentences_per_sign=0) == []


def test_rows_that_already_have_a_split_are_refused(domain: DomainSentences) -> None:
    rows = _recorded(domain, 1)
    require_unassigned(rows)
    with pytest.raises(SplitError, match="already have a split"):
        require_unassigned([rows[0].model_copy(update={"split": "train"})])


# ---- continuous takes ---------------------------------------------------------------------


def test_continuous_takes_with_test_unseen_sentences_only_go_to_test(
    domain: DomainSentences,
) -> None:
    cfg = SplitConfig()
    val_unseen = choose_val_unseen(domain, cfg)
    test_unseen = sorted({s.id for s in domain.with_role("unseen_test")} - set(val_unseen))
    takes = [
        _continuous(domain, 1, ["t01", "t02", "t03"]),
        _continuous(domain, 2, ["t04", test_unseen[0], "t05"]),
        _continuous(domain, 3, ["t06", val_unseen[0]]),
        _continuous(domain, 4, ["t07", "t08"]),
        _continuous(domain, 5, ["t09", "t10"]),
    ]
    result = build_splits(_recorded(domain), domain, cfg, takes)
    assert [r.take for r in result.continuous["val"]] == [1, 4]
    assert [r.take for r in result.continuous["test"]] == [2, 3, 5]
    assert all(r.split == "val" for r in result.continuous["val"])

    moved = {"val": [*result.continuous["val"], result.continuous["test"][0]], "test": []}
    violations = check_split_guarantees(result.clips, domain, continuous=moved)
    assert any("continuous val take" in v for v in violations)


# ---- written files ------------------------------------------------------------------------


def test_written_split_files_are_valid_manifests(domain: DomainSentences, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    takes = [_continuous(domain, 1, ["t01", "t02"]), _continuous(domain, 2, ["t03", "t04"])]
    result = build_splits(_recorded(domain), domain, continuous_rows=takes)
    written = write_splits(result, data_root)
    assert sorted(p.name for p in written) == [
        "domain_continuous_test.jsonl",
        "domain_continuous_val.jsonl",
        "domain_test_seen.jsonl",
        "domain_test_unseen.jsonl",
        "domain_train.jsonl",
        "domain_val.jsonl",
    ]
    report = validate_manifests(
        written, data_root, check_files=False, sentences=domain.text_by_id()
    )
    assert report.issues == []
    assert report.n_rows == len(domain.sentences) * 6 + 2
    # read back from disk, the guarantees still hold
    from_disk = {
        name: load_clip_rows(manifest_path(data_root, "domain", name)) for name in CLIP_SPLITS
    }
    assert check_split_guarantees(from_disk, domain) == []
    assert from_disk == result.clips
