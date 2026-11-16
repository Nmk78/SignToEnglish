"""The recording flow, driven end to end with generated frames and scripted keys.

Not covered here, because it needs a camera and a person: opening the webcam, the macOS
camera permission, the window, and real key presses (`record/camera.py`). The user's
recording trial covers those.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from signtoenglish.common.hashing import sha256_file
from signtoenglish.data.manifest import (
    load_clip_rows,
    load_continuous_rows,
    manifest_path,
    validate_manifests,
)
from signtoenglish.record.recorder import (
    RecorderError,
    RecorderSettings,
    choose_group,
    choose_next,
    session_rank,
)

SENTENCES = {
    "t01": "Hello, how are you?",
    "t02": "Where does your mother live?",
    "t03": "My sister is hungry.",
    "u01": "Where does your friend live?",
}
ONE_TAKE = [("prompt", 3, "space"), ("recording", 40, "space"), ("review", 2, "a")]
QUIT = [("prompt", 1, "q")]


def _videos(data_root: Path) -> list[str]:
    return sorted(p.relative_to(data_root).as_posix() for p in data_root.rglob("*.mp4"))


def _expected_ms(source, frames: list[int]) -> list[int]:
    first = source.times_ns[frames[0]]
    return [round((source.times_ns[i] - first) / 1_000_000) for i in frames]


# ---- one take, checked all the way down to the file -----------------------------------------


def test_accepted_take_is_a_real_video_with_the_capture_times(
    data_root: Path, make_recorder, tools
) -> None:
    recorder, source, ui = make_recorder(ONE_TAKE + QUIT, tags=("show_conditions",))
    summary = recorder.run_clips()

    assert summary.ended == "quit" and len(summary.accepted) == 1
    rows = load_clip_rows(manifest_path(data_root, "domain", "takes"))
    assert len(rows) == 1
    row = rows[0]
    assert row.id == summary.accepted[0]
    assert row.sentence_id in SENTENCES and row.text == SENTENCES[row.sentence_id]
    assert (row.dataset, row.split, row.license) == ("domain", "unassigned", "own-recording")
    assert (row.signer_id, row.session_id, row.take) == ("me", "2026-11-16_desk", 1)
    assert (row.sign_language, row.text_language) == ("ase", "en")
    assert row.start_ms is None and row.end_ms is None
    assert row.tags == ("show_conditions",)
    assert row.source_path == f"domain/2026-11-16_desk/{row.sentence_id}_me_take01.mp4"
    assert _videos(data_root) == [row.source_path]  # no partial file left behind
    assert row.source_sha256 == sha256_file(data_root / row.source_path)

    # The file holds exactly the frames that were on screen while recording, at the times
    # the source gave them: uneven spacing and the skipped-frame gap included.
    (recorded,) = ui.recording_runs()
    assert len(recorded) == 40
    times = tools.read_times_ms(data_root / row.source_path)
    assert times == _expected_ms(source, recorded)
    gaps = np.diff(times)
    assert gaps.min() >= 26
    assert len(set(gaps.tolist())) > 5  # really variable, not a nominal frame rate

    # The saved picture is the camera picture: not mirrored, nothing drawn on it.
    images = tools.read_images(data_root / row.source_path)
    for position in (0, 17, 39):
        wanted = tools.frame_image(recorded[position]).astype(int)
        assert np.abs(images[position].astype(int) - wanted).mean() < 3.0
        assert np.abs(images[position].astype(int) - wanted[:, ::-1]).mean() > 10.0

    report = validate_manifests(
        [manifest_path(data_root, "domain", "takes")], data_root, sentences=SENTENCES
    )
    assert report.issues == [] and report.n_rows == 1


def test_countdown_lasts_three_seconds_of_capture_time(data_root: Path, make_recorder) -> None:
    settings = RecorderSettings(countdown_s=3.0)
    recorder, source, ui = make_recorder(ONE_TAKE + QUIT, settings=settings)
    recorder.run_clips()
    start_key_frame = ui.keys[0][0]
    first_recorded = ui.recording_runs()[0][0]
    waited_s = (source.times_ns[first_recorded] - source.times_ns[start_key_frame]) / 1e9
    assert 3.0 <= waited_s < 3.2
    digits = [s.overlay.big for s in ui.shown if s.overlay.state == "countdown"]
    assert [d for i, d in enumerate(digits) if i == 0 or digits[i - 1] != d] == ["3", "2", "1"]
    # nothing is written before the countdown ends
    assert not any(s.overlay.elapsed_ms for s in ui.shown if s.overlay.state != "recording")


def test_prompt_shows_the_sentence_and_the_users_asl_line_when_present(
    data_root: Path, make_recorder, tools
) -> None:
    from signtoenglish.record.prompts import Prompt

    prompts = (Prompt("t01", "Hello, how are you?", "train", asl="PLACEHOLDER-NOT-ASL"),)
    recorder, _, ui = make_recorder(QUIT, prompts=prompts)
    recorder.run_clips()
    first = ui.shown[0].overlay
    assert first.state == "prompt" and first.text == "Hello, how are you?"
    assert first.detail == ("ASL: PLACEHOLDER-NOT-ASL",)
    assert "t01" in first.heading and "SPACE start" in first.keys

    recorder, _, ui = make_recorder(QUIT)
    recorder.run_clips()
    assert ui.shown[0].overlay.detail == ()  # the recorder invents nothing


# ---- accept, redo, skip, quit ---------------------------------------------------------------


def test_redo_drops_the_take_and_asks_for_the_same_sentence_again(
    data_root: Path, make_recorder
) -> None:
    plan = [
        ("prompt", 3, "space"), ("recording", 30, "space"), ("review", 2, "r"),
        ("prompt", 3, "space"), ("recording", 25, "space"), ("review", 2, "a"),
        *QUIT,
    ]  # fmt: skip
    recorder, _, ui = make_recorder(plan)
    summary = recorder.run_clips()
    assert (len(summary.accepted), summary.redone) == (1, 1)
    rows = load_clip_rows(manifest_path(data_root, "domain", "takes"))
    assert [r.take for r in rows] == [1]  # a dropped take does not use up a number
    assert _videos(data_root) == [rows[0].source_path]
    prompted = [s.overlay.heading.split()[0] for s in ui.shown if s.overlay.state == "prompt"]
    first_two = list(dict.fromkeys(prompted))[:2]
    assert first_two[0] == rows[0].sentence_id and first_two[1] != rows[0].sentence_id
    runs = ui.recording_runs()
    assert [len(r) for r in runs] == [30, 25]
    notices = [s.overlay.notice for s in ui.shown if s.overlay.notice]
    assert notices[0] == "Take dropped. Same sentence again."


def test_start_over_during_recording_drops_the_take(data_root: Path, make_recorder) -> None:
    plan = [("prompt", 3, "space"), ("recording", 10, "r"), *ONE_TAKE, *QUIT]
    recorder, _, _ = make_recorder(plan)
    summary = recorder.run_clips()
    assert (len(summary.accepted), summary.redone) == (1, 1)
    assert len(_videos(data_root)) == 1


def test_replay_shows_the_take_before_it_is_accepted(data_root: Path, make_recorder) -> None:
    plan = [("prompt", 3, "space"), ("recording", 30, "space"), ("review", 2, "p"),
            ("review", 2, "a"), *QUIT]  # fmt: skip
    recorder, _, ui = make_recorder(plan)
    recorder.run_clips()
    assert len(ui.replayed) == 1 and ui.replayed[0].name.endswith(".partial.mp4")
    assert len(_videos(data_root)) == 1


@pytest.mark.parametrize(
    "plan",
    [
        [("prompt", 3, "space"), ("recording", 30, "q")],
        [("prompt", 3, "space"), ("recording", 30, "space"), ("review", 2, "q")],
        [("prompt", 3, "space"), ("countdown", 2, "q")],
    ],
)
def test_quitting_leaves_no_file_and_no_row(data_root: Path, make_recorder, plan) -> None:
    recorder, _, _ = make_recorder(plan)
    summary = recorder.run_clips()
    assert summary.accepted == [] and summary.ended == "quit"
    assert _videos(data_root) == []
    assert not manifest_path(data_root, "domain", "takes").exists()


def test_camera_stopping_mid_take_leaves_no_file(data_root: Path, make_recorder) -> None:
    recorder, _, _ = make_recorder([("prompt", 3, "space")], n_frames=40)
    summary = recorder.run_clips()
    assert summary.ended == "source_ended" and summary.accepted == []
    assert _videos(data_root) == []


def test_cancelled_countdown_returns_to_the_prompt(data_root: Path, make_recorder) -> None:
    plan = [("prompt", 3, "space"), ("countdown", 2, "space"), *ONE_TAKE, *QUIT]
    recorder, _, ui = make_recorder(plan)
    summary = recorder.run_clips()
    assert len(summary.accepted) == 1 and len(ui.recording_runs()) == 1
    assert len(_videos(data_root)) == 1  # the cancelled take left nothing


def test_too_short_takes_are_dropped(data_root: Path, make_recorder) -> None:
    plan = [("prompt", 3, "space"), ("recording", 4, "space"), *QUIT]
    recorder, _, ui = make_recorder(plan)
    summary = recorder.run_clips()
    assert summary.accepted == [] and summary.discarded == 1
    assert _videos(data_root) == []
    assert any(s.overlay.notice and "Too short" in s.overlay.notice for s in ui.shown)
    assert not ui.frames_in("review")


def test_take_stops_itself_at_the_maximum_length(data_root: Path, make_recorder, tools) -> None:
    settings = RecorderSettings(countdown_s=0.2, max_take_s=1.5)
    plan = [("prompt", 3, "space"), ("review", 2, "a"), *QUIT]
    recorder, _, ui = make_recorder(plan, settings=settings)
    recorder.run_clips()
    (row,) = load_clip_rows(manifest_path(data_root, "domain", "takes"))
    times = tools.read_times_ms(data_root / row.source_path)
    assert 1500 <= times[-1] < 1650
    assert any("Stopped at 1.5 s" in s.overlay.heading for s in ui.shown)


def test_skipped_sentence_is_not_offered_again_in_the_session(
    data_root: Path, make_recorder
) -> None:
    plan = [("prompt", 2, "n"), *ONE_TAKE, *ONE_TAKE, *ONE_TAKE, *ONE_TAKE, *ONE_TAKE, *ONE_TAKE]
    recorder, _, ui = make_recorder(plan)
    summary = recorder.run_clips()
    skipped = ui.shown[0].overlay.heading.split()[0]
    rows = load_clip_rows(manifest_path(data_root, "domain", "takes"))
    assert summary.ended == "done"
    assert Counter(r.sentence_id for r in rows) == {s: 2 for s in SENTENCES if s != skipped}


# ---- balance, sessions, numbering ------------------------------------------------------------


def test_choose_next_keeps_counts_within_one_and_stops_at_the_target() -> None:
    ids = [f"s{i:02d}" for i in range(7)]
    session: Counter[str] = Counter()
    total: Counter[str] = Counter({"s00": 5, "s01": 5, "s02": 1})
    rank = session_rank(ids, "2026-11-16_desk", shuffle=True)
    picked = []
    while (choice := choose_next(ids, session, total, target=3, rank=rank)) is not None:
        picked.append(choice)
        session[choice] += 1
        total[choice] += 1
        assert max(session[s] for s in ids) - min(session[s] for s in ids) <= 1
    assert len(picked) == 21 and set(session.values()) == {3}
    # sentences with fewer takes overall come first in each round
    assert set(picked[5:7]) == {"s00", "s01"} and picked[4] == "s02"
    assert sorted(picked[:7]) == ids and sorted(picked[7:14]) == ids


def test_choose_next_honours_skips_and_order() -> None:
    ids = ["a", "b", "c"]
    assert choose_next(ids, {}, {}, target=1) == "a"  # file order without a rank
    assert choose_next(ids, {}, {}, target=1, skipped={"a"}) == "b"
    assert choose_next(ids, {"a": 1, "b": 1, "c": 1}, {}, target=1) is None
    assert choose_next(ids, {}, {}, target=0) is None
    assert choose_next(ids, {}, {"a": 2, "b": 1, "c": 2}, target=1) == "b"
    assert session_rank(ids, "x", shuffle=False) == {"a": 0, "b": 1, "c": 2}
    many = [f"s{i}" for i in range(30)]
    assert session_rank(many, "day1", True) == session_rank(many, "day1", True)
    assert session_rank(many, "day1", True) != session_rank(many, "day2", True)


def test_a_full_session_records_every_sentence_the_same_number_of_times(
    data_root: Path, make_recorder
) -> None:
    recorder, _, _ = make_recorder(ONE_TAKE * 8)
    summary = recorder.run_clips()
    assert summary.ended == "done" and len(summary.accepted) == 8
    rows = load_clip_rows(manifest_path(data_root, "domain", "takes"))
    assert Counter(r.sentence_id for r in rows) == dict.fromkeys(SENTENCES, 2)
    # balanced at every point, not just at the end: each sentence once before any twice
    assert sorted(r.sentence_id for r in rows[:4]) == sorted(SENTENCES)
    assert sorted((r.sentence_id, r.take) for r in rows) == sorted(
        (s, t) for s in SENTENCES for t in (1, 2)
    )
    assert len({r.id for r in rows}) == 8 and len(_videos(data_root)) == 8
    assert (
        validate_manifests(
            [manifest_path(data_root, "domain", "takes")], data_root, sentences=SENTENCES
        ).issues
        == []
    )


def test_resuming_a_session_continues_the_numbering_and_the_balance(
    data_root: Path, make_recorder
) -> None:
    first, _, _ = make_recorder(ONE_TAKE * 3 + QUIT)
    first.run_clips()
    done = [r.sentence_id for r in load_clip_rows(manifest_path(data_root, "domain", "takes"))]

    second, _, _ = make_recorder(ONE_TAKE * 2 + QUIT, seed=1)
    second.run_clips()
    rows = load_clip_rows(manifest_path(data_root, "domain", "takes"))
    assert rows[3].sentence_id not in done  # the sentence with no take yet comes first
    assert rows[4].take == 2 and rows[4].sentence_id in done
    assert len(_videos(data_root)) == 5


def test_a_later_session_starts_with_the_sentences_that_have_fewest_takes_overall(
    data_root: Path, make_recorder
) -> None:
    first, _, _ = make_recorder(ONE_TAKE * 2 + QUIT, session_id="2026-11-16_desk")
    first.run_clips()
    done = {r.sentence_id for r in load_clip_rows(manifest_path(data_root, "domain", "takes"))}

    second, _, _ = make_recorder(ONE_TAKE * 4 + QUIT, session_id="2026-11-20_show", seed=2)
    second.run_clips()
    rows = load_clip_rows(manifest_path(data_root, "domain", "takes"))
    later = [r for r in rows if r.session_id == "2026-11-20_show"]
    assert {r.sentence_id for r in later[:2]} == set(SENTENCES) - done
    assert all(r.take == 1 for r in later)  # take numbers are per session
    assert all(r.source_path.startswith("domain/2026-11-20_show/") for r in later)


def test_a_leftover_file_is_never_overwritten(data_root: Path, make_recorder) -> None:
    session_dir = data_root / "domain" / "2026-11-16_desk"
    session_dir.mkdir(parents=True)
    for sentence_id in SENTENCES:
        (session_dir / f"{sentence_id}_me_take01.mp4").write_bytes(b"left over from a crash")
    recorder, _, _ = make_recorder(ONE_TAKE + QUIT)
    recorder.run_clips()
    (row,) = load_clip_rows(manifest_path(data_root, "domain", "takes"))
    assert row.take == 2 and row.source_path.endswith("_me_take02.mp4")
    assert (session_dir / f"{row.sentence_id}_me_take01.mp4").read_bytes().startswith(b"left")


def test_labels_that_cannot_be_file_names_are_refused(make_recorder) -> None:
    for bad in ("../x", "a b", "", ".hidden", "a/b"):
        with pytest.raises(RecorderError):
            make_recorder(QUIT, session_id=bad)
        with pytest.raises(RecorderError):
            make_recorder(QUIT, signer_id=bad)


# ---- continuous takes ------------------------------------------------------------------------

CONTINUOUS = [
    ("prompt", 3, "space"),
    ("recording", 45, "space"),
    ("recording", 30, "space"),
    ("recording", 60, "space"),
    ("review", 2, "a"),
]


def test_continuous_take_marks_each_sentence_end_at_the_key_press(
    data_root: Path, make_recorder, tools
) -> None:
    settings = RecorderSettings(countdown_s=0.2, group_size=3, tail_s=0.5, min_segment_s=0.5)
    recorder, source, ui = make_recorder(CONTINUOUS + QUIT, settings=settings)
    summary = recorder.run_continuous()

    path = manifest_path(data_root, "domain", "takes", "continuous")
    assert path.name == "domain_continuous_takes.jsonl" and summary.manifest == path
    (row,) = load_continuous_rows(path)
    assert row.id == summary.accepted[0] and row.take == 1 and row.split == "unassigned"
    assert row.source_path == "domain/2026-11-16_desk/continuous_me_take01.mp4"
    assert _videos(data_root) == [row.source_path]

    (recorded,) = ui.recording_runs()
    times = tools.read_times_ms(data_root / row.source_path)
    assert times == _expected_ms(source, recorded)

    # each sentence end is the file time of the frame on screen when SPACE was pressed
    presses = [frame for frame, key in ui.keys if key == "space"][1:]
    marks = [times[recorded.index(frame)] for frame in presses]
    assert len(marks) == 3
    assert [s.end_ms for s in row.segments] == marks
    assert [s.start_ms for s in row.segments] == [0, marks[0], marks[1]]
    assert marks == sorted(marks) and marks[0] > 1000

    # the sentences shown, in the order shown, with their text unchanged
    shown_texts = list(
        dict.fromkeys(
            s.overlay.text
            for s in ui.shown
            if s.overlay.state == "recording" and s.overlay.heading.startswith("Sentence")
        )
    )
    assert [s.text for s in row.segments] == shown_texts
    assert all(SENTENCES[s.sentence_id] == s.text for s in row.segments)
    assert len({s.sentence_id for s in row.segments}) == 3

    # recording goes on for the tail after the last mark, then stops by itself
    assert 500 <= times[-1] - marks[-1] < 650

    assert validate_manifests([path], data_root, sentences=SENTENCES).issues == []
    raw = json.loads(path.read_text())
    assert set(raw["segments"][0]) == {"sentence_id", "text", "start_ms", "end_ms"}


def test_a_second_key_press_right_after_a_mark_is_ignored(data_root: Path, make_recorder) -> None:
    settings = RecorderSettings(countdown_s=0.2, group_size=2, tail_s=0.3, min_segment_s=0.7)
    plan = [
        ("prompt", 3, "space"),
        ("recording", 40, "space"),
        ("recording", 3, "space"),  # a bounce, 0.1 s later
        ("recording", 40, "space"),
        ("review", 2, "a"),
        *QUIT,
    ]
    recorder, _, ui = make_recorder(plan, settings=settings)
    recorder.run_continuous()
    (row,) = load_continuous_rows(manifest_path(data_root, "domain", "takes", "continuous"))
    assert len(row.segments) == 2
    assert all(s.end_ms - s.start_ms >= 700 for s in row.segments)
    assert any(s.overlay.notice and "ignored" in s.overlay.notice for s in ui.shown)


def test_continuous_redo_quit_and_unfinished_takes_leave_nothing(
    data_root: Path, make_recorder
) -> None:
    settings = RecorderSettings(
        countdown_s=0.2, group_size=2, tail_s=0.3, min_segment_s=0.5, max_continuous_s=4.0
    )
    plan = [
        ("prompt", 3, "space"), ("recording", 20, "r"),  # start over
        ("prompt", 3, "space"), ("recording", 30, "space"),  # one end marked, then time runs out
        ("prompt", 3, "space"), ("recording", 30, "space"), ("recording", 30, "space"),
        ("review", 2, "r"),  # redo at review
        ("prompt", 3, "space"), ("recording", 30, "q"),
    ]  # fmt: skip
    recorder, _, ui = make_recorder(plan, settings=settings)
    summary = recorder.run_continuous()
    assert summary.accepted == [] and (summary.redone, summary.discarded) == (2, 1)
    assert _videos(data_root) == []
    assert not manifest_path(data_root, "domain", "takes", "continuous").exists()
    assert any(
        s.overlay.notice and "before every sentence end" in s.overlay.notice for s in ui.shown
    )
    # the same sentences are offered again after a redo
    listings = list(
        dict.fromkeys(s.overlay.detail for s in ui.shown if s.overlay.state == "prompt")
    )
    assert len(listings) == 1


def test_continuous_takes_spread_over_the_sentences(data_root: Path, make_recorder) -> None:
    settings = RecorderSettings(countdown_s=0.2, group_size=2, tail_s=0.3, min_segment_s=0.5)
    one = [("prompt", 3, "space"), ("recording", 30, "space"), ("recording", 30, "space"),
           ("review", 2, "a")]  # fmt: skip
    recorder, _, _ = make_recorder(one * 4, settings=settings)
    summary = recorder.run_continuous(count=4)
    assert summary.ended == "count_reached"
    rows = load_continuous_rows(manifest_path(data_root, "domain", "takes", "continuous"))
    assert [r.take for r in rows] == [1, 2, 3, 4]
    used = Counter(s.sentence_id for r in rows for s in r.segments)
    assert used == dict.fromkeys(SENTENCES, 2)  # 4 sentences, 8 slots: each exactly twice
    assert (
        validate_manifests(
            [manifest_path(data_root, "domain", "takes", "continuous")],
            data_root,
            sentences=SENTENCES,
        ).issues
        == []
    )


def test_choose_group_prefers_the_least_used_sentences() -> None:
    ids = [f"s{i}" for i in range(10)]
    used = {s: 1 for s in ids[:6]}
    group = choose_group(ids, used, 4, random.Random(0))
    assert sorted(group) == ids[6:]
    assert len(choose_group(ids[:2], {}, 4, random.Random(0))) == 2
    orders = {tuple(choose_group(ids, {}, 4, random.Random(seed))) for seed in range(20)}
    assert len(orders) > 10  # the order and the choice among ties vary
