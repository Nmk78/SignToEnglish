"""Manifest formats and validation (ARCHITECTURE.md, section 4.2), on real video files."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from signtoenglish import cli
from signtoenglish.data.manifest import (
    ClipRow,
    ContinuousRow,
    ManifestError,
    Segment,
    append_row,
    load_clip_rows,
    load_continuous_rows,
    manifest_path,
    parse_manifest_name,
    probe_video,
    read_manifest,
    validate_manifests,
    write_manifest,
)

MakeVideo = Callable[..., str]
MakeClip = Callable[..., ClipRow]
MakeContinuous = Callable[..., ContinuousRow]

CLIP_FIELDS = [
    "id", "dataset", "split", "source_path", "source_sha256", "start_ms", "end_ms", "text",
    "sentence_id", "signer_id", "session_id", "take", "sign_language", "text_language",
    "license", "tags",
]  # fmt: skip
SENTENCES = {"t01": "Hello, how are you?", "t05": "The water is cold."}


def _codes(report: object) -> list[str]:
    return sorted(i.code for i in report.issues)  # type: ignore[attr-defined]


def _takes(data_root: Path) -> Path:
    return manifest_path(data_root, "domain", "takes")


def test_manifest_file_names() -> None:
    assert manifest_path("d", "domain", "train").as_posix() == "d/manifests/domain_train.jsonl"
    assert (
        manifest_path("d", "domain", "val", "continuous").as_posix()
        == "d/manifests/domain_continuous_val.jsonl"
    )
    assert parse_manifest_name("domain_train.jsonl") == ("domain", "clip", "train")
    assert parse_manifest_name("domain_test_unseen.jsonl") == ("domain", "clip", "test_unseen")
    assert parse_manifest_name("domain_test_seen.jsonl") == ("domain", "clip", "test_seen")
    assert parse_manifest_name("how2sign_test.jsonl") == ("how2sign", "clip", "test")
    assert parse_manifest_name("domain_continuous_val.jsonl") == ("domain", "continuous", "val")
    assert parse_manifest_name("domain_continuous_takes.jsonl") == (
        "domain", "continuous", "takes",
    )  # fmt: skip
    assert parse_manifest_name("notes.jsonl") is None
    assert parse_manifest_name("domain_train.json") is None


def test_clip_row_round_trip_has_exactly_the_documented_fields(
    data_root: Path, make_video: MakeVideo, make_clip_row: MakeClip
) -> None:
    sha = make_video("domain/s/a.mp4")
    row = make_clip_row("domain/s/a.mp4", sha, tags=("show_conditions",), text="Café? Yes.")
    path = _takes(data_root)
    append_row(path, row)
    raw = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert list(raw) == CLIP_FIELDS
    assert raw["text"] == "Café? Yes."  # untouched, not escaped away
    assert raw["start_ms"] is None and raw["tags"] == ["show_conditions"]
    assert load_clip_rows(path) == [row]


def test_continuous_row_round_trip(
    data_root: Path, make_video: MakeVideo, make_continuous_row: MakeContinuous
) -> None:
    sha = make_video("domain/s/c.mp4")
    row = make_continuous_row("domain/s/c.mp4", sha)
    path = manifest_path(data_root, "domain", "takes", "continuous")
    append_row(path, row)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert list(raw["segments"][0]) == ["sentence_id", "text", "start_ms", "end_ms"]
    assert "text" not in raw and "start_ms" not in raw
    assert load_continuous_rows(path) == [row]


def test_valid_manifests_pass_every_check(
    data_root: Path,
    make_video: MakeVideo,
    make_clip_row: MakeClip,
    make_continuous_row: MakeContinuous,
) -> None:
    for name in ("a", "b"):
        sha = make_video(f"domain/s/{name}.mp4")
        append_row(_takes(data_root), make_clip_row(f"domain/s/{name}.mp4", sha))
    sha = make_video("domain/s/c.mp4")
    continuous = manifest_path(data_root, "domain", "takes", "continuous")
    append_row(continuous, make_continuous_row("domain/s/c.mp4", sha))

    report = validate_manifests([_takes(data_root), continuous], data_root, sentences=SENTENCES)
    assert report.issues == []
    assert report.ok and report.n_rows == 3 and report.n_files == 2


def test_probe_reads_real_container_timestamps(data_root: Path, make_video: MakeVideo) -> None:
    times = (0, 33, 67, 100, 141, 166, 1000, 1034)  # a gap, as after dropped frames
    make_video("v.mp4", times_ms=times)
    probe = probe_video(data_root / "v.mp4")
    assert probe.n_frames == len(times)
    assert probe.first_ms == 0 and probe.last_ms == 1034
    assert probe.strictly_increasing
    assert probe.duration_ms > 1034
    assert (probe.width, probe.height) == (64, 48)


def test_duplicate_ids_within_and_across_files(
    data_root: Path, make_video: MakeVideo, make_clip_row: MakeClip
) -> None:
    rows = []
    for name in ("a", "b", "c"):
        sha = make_video(f"domain/s/{name}.mp4")
        rows.append(make_clip_row(f"domain/s/{name}.mp4", sha, id="same"))
    write_manifest(_takes(data_root), rows[:2])
    other = data_root / "manifests" / "domain_takes_copy.jsonl"
    write_manifest(other, rows[2:])
    report = validate_manifests([_takes(data_root), other], data_root)
    assert [i.code for i in report.errors] == ["duplicate_id", "duplicate_id"]
    assert not report.ok


def test_empty_text_is_reported_and_cannot_be_written(
    data_root: Path, make_video: MakeVideo, make_clip_row: MakeClip
) -> None:
    sha = make_video("domain/s/a.mp4")
    row = make_clip_row("domain/s/a.mp4", sha, text="   ")
    with pytest.raises(ManifestError, match="empty_text"):
        append_row(_takes(data_root), row)
    assert not _takes(data_root).exists()

    _takes(data_root).parent.mkdir(parents=True)
    _takes(data_root).write_text(json.dumps(row.model_dump(mode="json")) + "\n")
    report = validate_manifests([_takes(data_root)], data_root)
    assert _codes(report) == ["empty_text"]
    with pytest.raises(ManifestError):
        load_clip_rows(_takes(data_root))


@pytest.mark.parametrize(
    ("start", "end", "why"),
    [
        (100, 100, "is not after"),
        (300, 200, "is not after"),
        (-5, 200, "negative"),
        (None, 200, "both be set or both be null"),
        (0, 5_000, "past the end of the video"),
    ],
)
def test_bad_bounds(
    data_root: Path,
    make_video: MakeVideo,
    make_clip_row: MakeClip,
    start: int | None,
    end: int | None,
    why: str,
) -> None:
    sha = make_video("domain/s/a.mp4")
    row = make_clip_row("domain/s/a.mp4", sha, start_ms=start, end_ms=end)
    path = _takes(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row.model_dump(mode="json")) + "\n")
    report = validate_manifests([path], data_root)
    assert "bad_bounds" in [i.code for i in report.errors]
    assert any(why in i.message for i in report.errors)


def test_good_bounds_inside_a_longer_source(
    data_root: Path, make_video: MakeVideo, make_clip_row: MakeClip
) -> None:
    sha = make_video("domain/s/long.mp4", times_ms=tuple(range(0, 3000, 40)))
    path = _takes(data_root)
    rows = [
        make_clip_row("domain/s/long.mp4", sha, id="x1", start_ms=0, end_ms=1400),
        make_clip_row("domain/s/long.mp4", sha, id="x2", start_ms=1400, end_ms=2900),
    ]
    write_manifest(path, rows)
    assert validate_manifests([path], data_root).issues == []


def test_bad_durations(data_root: Path, make_video: MakeVideo, make_clip_row: MakeClip) -> None:
    one_frame = make_video("domain/s/one.mp4", times_ms=(0,))
    short = make_video("domain/s/short.mp4", times_ms=(0, 33, 66, 100))
    long = make_video("domain/s/long.mp4", times_ms=tuple(range(0, 31_000, 500)))
    path = _takes(data_root)
    write_manifest(
        path,
        [
            make_clip_row("domain/s/one.mp4", one_frame),
            make_clip_row("domain/s/short.mp4", short),
            make_clip_row("domain/s/long.mp4", long),
        ],
    )
    report = validate_manifests([path], data_root)
    assert {(i.row_id, i.level, i.code) for i in report.issues} == {
        ("one", "error", "bad_duration"),
        ("short", "warning", "short_clip"),
        ("long", "warning", "long_clip"),
    }
    assert not report.ok


def test_missing_file(data_root: Path, make_video: MakeVideo, make_clip_row: MakeClip) -> None:
    sha = make_video("domain/s/a.mp4")
    path = _takes(data_root)
    append_row(path, make_clip_row("domain/s/a.mp4", sha))
    (data_root / "domain/s/a.mp4").unlink()
    report = validate_manifests([path], data_root)
    assert _codes(report) == ["missing_file"]
    # without the file checks the row itself is still fine
    assert validate_manifests([path], data_root, check_files=False).issues == []


def test_content_hash_mismatch(
    data_root: Path, make_video: MakeVideo, make_clip_row: MakeClip
) -> None:
    sha = make_video("domain/s/a.mp4", seed=1)
    path = _takes(data_root)
    append_row(path, make_clip_row("domain/s/a.mp4", sha))
    make_video("domain/s/a.mp4", seed=2)  # the file changed after the row was written
    report = validate_manifests([path], data_root)
    assert _codes(report) == ["hash_mismatch"]
    assert validate_manifests([path], data_root, check_hash=False).issues == []


def test_a_file_that_is_not_a_video(data_root: Path, make_clip_row: MakeClip) -> None:
    from signtoenglish.common.hashing import sha256_file

    target = data_root / "domain/s/fake.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"this is not a video" * 100)
    path = _takes(data_root)
    append_row(path, make_clip_row("domain/s/fake.mp4", sha256_file(target)))
    report = validate_manifests([path], data_root)
    assert _codes(report) == ["bad_media"]


def test_lines_that_do_not_parse(data_root: Path, make_clip_row: MakeClip) -> None:
    good = make_clip_row("domain/s/a.mp4", "a" * 64).model_dump(mode="json")
    path = _takes(data_root)
    path.parent.mkdir(parents=True)
    lines = [
        "{not json",
        json.dumps({**good, "extra_field": 1}),
        json.dumps({**good, "take": "1"}),
        json.dumps({k: v for k, v in good.items() if k != "text"}),
        "",
        json.dumps(good),
    ]
    path.write_text("\n".join(lines) + "\n")
    rows, issues = read_manifest(path)
    assert [line for line, _ in rows] == [6]
    assert [(i.code, i.line) for i in issues] == [("parse_error", n) for n in (1, 2, 3, 4)]
    assert "extra_field" in issues[1].message and "take" in issues[2].message


def test_field_problems(data_root: Path, make_clip_row: MakeClip) -> None:
    path = _takes(data_root)
    path.parent.mkdir(parents=True)
    rows = [
        make_clip_row("../outside.mp4", "a" * 64, id="r1"),
        make_clip_row("/abs/a.mp4", "b" * 64, id="r2"),
        make_clip_row("domain/s/a.mp4", "XYZ", id="r3"),
        make_clip_row("domain/s/b.mp4", "c" * 64, id="r4", take=0),
        make_clip_row("domain/s/c.mp4", "d" * 64, id="r5", split="training"),
        make_clip_row("domain/s/d.mp4", "e" * 64, id="has space"),
    ]
    path.write_text("".join(json.dumps(r.model_dump(mode="json")) + "\n" for r in rows))
    report = validate_manifests([path], data_root, check_files=False)
    by_row = {i.row_id: i.code for i in report.errors if i.code != "split_mismatch"}
    assert by_row == {
        "r1": "bad_path", "r2": "bad_path", "r3": "bad_field", "r4": "bad_field",
        "r5": "bad_field", "has space": "bad_field",
    }  # fmt: skip


def test_split_field_must_agree_with_the_file_name(
    data_root: Path, make_clip_row: MakeClip
) -> None:
    train = manifest_path(data_root, "domain", "train")
    write_manifest(
        train,
        [
            make_clip_row("domain/s/a.mp4", "a" * 64, id="ok", split="train"),
            make_clip_row("domain/s/b.mp4", "b" * 64, id="wrong", split="val"),
            make_clip_row("domain/s/c.mp4", "c" * 64, id="other", split="train", dataset="x"),
        ],
    )
    report = validate_manifests([train], data_root, check_files=False)
    assert [(i.row_id, i.code) for i in report.errors] == [
        ("wrong", "split_mismatch"),
        ("other", "split_mismatch"),
    ]
    # recorder output carries `unassigned` and lives in the takes file
    takes = _takes(data_root)
    write_manifest(takes, [make_clip_row("domain/s/d.mp4", "d" * 64, split="train")])
    assert _codes(validate_manifests([takes], data_root, check_files=False)) == ["split_mismatch"]


def test_the_same_recording_used_twice(
    data_root: Path, make_video: MakeVideo, make_clip_row: MakeClip
) -> None:
    sha = make_video("domain/s/a.mp4", seed=7)
    copy = make_video("domain/s/copy_of_a.mp4", seed=7)  # same content, another name
    assert copy == sha
    path = _takes(data_root)
    write_manifest(
        path,
        [
            make_clip_row("domain/s/a.mp4", sha, id="first"),
            make_clip_row("domain/s/a.mp4", sha, id="again"),
            make_clip_row("domain/s/copy_of_a.mp4", copy, id="copied"),
        ],
    )
    report = validate_manifests([path], data_root)
    assert [(i.row_id, i.code) for i in report.errors] == [
        ("again", "duplicate_source"),
        ("copied", "duplicate_source"),
    ]


def test_sentence_list_cross_check(
    data_root: Path, make_clip_row: MakeClip, make_continuous_row: MakeContinuous
) -> None:
    path = _takes(data_root)
    write_manifest(
        path,
        [
            make_clip_row("domain/s/a.mp4", "a" * 64, id="ok"),
            make_clip_row("domain/s/b.mp4", "b" * 64, id="edited", text="hello how are you"),
            make_clip_row("domain/s/c.mp4", "c" * 64, id="unknown", sentence_id="zz9"),
            make_clip_row("domain/s/d.mp4", "d" * 64, id="none", sentence_id=None),
        ],
    )
    continuous = manifest_path(data_root, "domain", "takes", "continuous")
    write_manifest(
        continuous,
        [
            make_continuous_row(
                "domain/s/e.mp4",
                "e" * 64,
                segments=(Segment(sentence_id="t05", text="Water cold.", start_ms=0, end_ms=9),),
            )
        ],
    )
    report = validate_manifests(
        [path, continuous], data_root, check_files=False, sentences=SENTENCES
    )
    assert [(i.row_id, i.code) for i in report.errors] == [
        ("edited", "text_mismatch"),
        ("unknown", "unknown_sentence"),
        ("none", "unknown_sentence"),
        ("e", "text_mismatch"),
    ]


def test_continuous_segment_problems(
    data_root: Path, make_video: MakeVideo, make_continuous_row: MakeContinuous
) -> None:
    def seg(start: int, end: int, text: str = "The water is cold.") -> Segment:
        return Segment(sentence_id="t05", text=text, start_ms=start, end_ms=end)

    path = manifest_path(data_root, "domain", "takes", "continuous")
    path.parent.mkdir(parents=True)
    cases = {
        "empty": (),
        "overlap": (seg(0, 300), seg(200, 500)),
        "backwards": (seg(300, 100),),
        "blank": (seg(0, 300, " "),),
        "beyond": (seg(0, 300), seg(300, 60_000)),
        "fine": (seg(0, 300), seg(300, 600)),
    }
    lines = []
    for name, segments in cases.items():
        sha = make_video(f"domain/s/{name}.mp4")
        row = make_continuous_row(f"domain/s/{name}.mp4", sha, segments=segments)
        lines.append(json.dumps(row.model_dump(mode="json")))
    path.write_text("\n".join(lines) + "\n")
    report = validate_manifests([path], data_root)
    assert {(i.row_id, i.code) for i in report.errors} == {
        ("empty", "bad_bounds"),
        ("overlap", "bad_bounds"),
        ("backwards", "bad_bounds"),
        ("blank", "empty_text"),
        ("beyond", "bad_bounds"),
    }


def test_missing_manifest_is_an_error(data_root: Path) -> None:
    report = validate_manifests([data_root / "manifests" / "domain_train.jsonl"], data_root)
    assert _codes(report) == ["unreadable"]


def test_write_manifest_replaces_atomically_and_refuses_bad_rows(
    data_root: Path, make_clip_row: MakeClip
) -> None:
    path = manifest_path(data_root, "domain", "train")
    good = make_clip_row("domain/s/a.mp4", "a" * 64, split="train")
    write_manifest(path, [good])
    with pytest.raises(ManifestError):
        write_manifest(path, [good, make_clip_row("domain/s/b.mp4", "b" * 64, text="")])
    assert load_clip_rows(path) == [good]  # the old file survived
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_cli_manifest_validate(
    data_root: Path,
    make_video: MakeVideo,
    make_clip_row: MakeClip,
    capsys: pytest.CaptureFixture[str],
    repo_root: Path,
) -> None:
    sha = make_video("domain/s/a.mp4")
    append_row(_takes(data_root), make_clip_row("domain/s/a.mp4", sha))
    argv = [
        "manifest", "validate", "--data-root", str(data_root),
        "--sentences", str(repo_root / "domain" / "sentences.yaml"),
    ]  # fmt: skip
    assert cli.main(argv) == 0
    assert capsys.readouterr().out.strip().endswith("VALID")

    (data_root / "domain/s/a.mp4").unlink()
    assert cli.main(argv) == 1
    out = capsys.readouterr().out
    assert "missing_file" in out and out.strip().endswith("INVALID")
