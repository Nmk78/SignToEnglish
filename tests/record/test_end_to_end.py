"""`s2e record` -> `s2e manifest validate` -> `s2e splits build`, through the real command
handlers and real files, with generated frames standing in for the camera and scripted keys
for the signer. The camera, the window, and real key presses are left to the user's trial.

The sentence list here is a made-up miniature. Its `signs` are arbitrary labels for the
coverage rule, and its `asl` / `source` strings are placeholders, not ASL.
"""

from __future__ import annotations

import datetime
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from signtoenglish import cli
from signtoenglish.data.domain import load_domain
from signtoenglish.data.manifest import (
    load_clip_rows,
    load_continuous_rows,
    manifest_path,
    validate_manifests,
)
from signtoenglish.data.splits import CLIP_SPLITS, check_split_guarantees
from signtoenglish.record.commands import run_record, session_id_for

MINI = """
domain: mini
sign_language: ase
text_language: en
vocabulary:
  - {id: X1}
  - {id: X2}
  - {id: "NO"}
sentences:
  - {id: a1, role: train, text: "First sentence.", signs: [X1, X2], asl: "PLACEHOLDER", source: "placeholder"}
  - {id: a2, role: train, text: "Second sentence.", signs: [X1, X2, "NO"], asl: null, source: null}
  - {id: a3, role: train, text: "Third sentence.", signs: [X2, X1], asl: null, source: null}
  - {id: b1, role: unseen_test, text: "Fourth sentence?", signs: [X1, X2], asl: null, source: null}
  - {id: b2, role: unseen_test, text: "Fifth sentence?", signs: [X2], asl: null, source: null}
"""  # noqa: E501

TAKE = [("prompt", 3, "space"), ("recording", 35, "space"), ("review", 2, "a")]
CONTINUOUS_TAKE = [
    ("prompt", 3, "space"),
    ("recording", 35, "space"),
    ("recording", 35, "space"),
    ("review", 2, "a"),
]


@pytest.fixture
def sentences(tmp_path: Path) -> Path:
    path = tmp_path / "sentences.yaml"
    path.write_text(MINI, encoding="utf-8")
    return path


def _record(tmp_path, sentences, data_root, plan, extra, day, source_cls, ui_cls, seed=0):
    """Run the real `s2e record` handler with the argument parser's own namespace."""
    argv = [
        "record", "--sentences", str(sentences), "--signer", "me", "--data-root", str(data_root),
        "--countdown", "0.2", *extra,
    ]  # fmt: skip
    args = cli.build_parser().parse_args(argv)
    source = source_cls(seed=seed)
    ui = ui_cls(source, plan)
    code = run_record(args, make_source=lambda: source, make_ui=lambda: ui, today=day)
    # whatever was opened is closed again; bad arguments are refused before opening
    assert source.closed == ui.closed
    assert source.closed or code == 1
    return code, ui


def test_record_validate_and_build_splits(
    tmp_path: Path, sentences: Path, data_root: Path, capsys: pytest.CaptureFixture[str], tools
) -> None:
    GeneratedSource, ScriptedUI = tools.source_cls, tools.ui_cls
    day1, day2 = datetime.date(2026, 11, 20), datetime.date(2026, 11, 24)

    # Session 1: two takes of each of the five sentences.
    code, _ = _record(
        tmp_path, sentences, data_root, TAKE * 10, ["--session", "desk", "--takes", "2"],
        day1, GeneratedSource, ScriptedUI,
    )  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0
    assert "session 2026-11-20_desk, signer me, 5 sentence(s)" in out
    assert "WARNING 4 sentence(s)" in out and "s2e domain check" in out
    assert "10 take(s) saved, 0 redone, 0 dropped (done)" in out

    # Session 2, another day, tagged as show conditions: two more takes each.
    code, _ = _record(
        tmp_path, sentences, data_root, TAKE * 10,
        ["--session", "hall", "--takes", "2", "--show-conditions", "--tag", "jacket"],
        day2, GeneratedSource, ScriptedUI, seed=1,
    )  # fmt: skip
    assert code == 0
    capsys.readouterr()

    # Two continuous takes of two sentences each.
    code, _ = _record(
        tmp_path, sentences, data_root, CONTINUOUS_TAKE * 2,
        ["--session", "hall", "--continuous", "--group-size", "2", "--count", "2"],
        day2, GeneratedSource, ScriptedUI, seed=2,
    )  # fmt: skip
    assert code == 0
    assert "2 continuous take(s) saved" in capsys.readouterr().out

    takes = load_clip_rows(manifest_path(data_root, "domain", "takes"))
    assert len(takes) == 20
    assert Counter(r.session_id for r in takes) == {"2026-11-20_desk": 10, "2026-11-24_hall": 10}
    assert Counter(r.sentence_id for r in takes) == dict.fromkeys(["a1", "a2", "a3", "b1", "b2"], 4)
    assert {r.tags for r in takes if r.session_id.endswith("hall")} == {
        ("jacket", "show_conditions")
    }
    assert {r.tags for r in takes if r.session_id.endswith("desk")} == {()}
    continuous = load_continuous_rows(manifest_path(data_root, "domain", "takes", "continuous"))
    assert [len(r.segments) for r in continuous] == [2, 2]

    # `s2e manifest validate`: files, hashes, video timing, and the sentence list.
    validate = [
        "manifest",
        "validate",
        "--data-root",
        str(data_root),
        "--sentences",
        str(sentences),
    ]
    assert cli.main(validate) == 0
    out = capsys.readouterr().out
    assert "22 row(s) in 2 file(s): 0 error(s), 0 warning(s)" in out and out.strip().endswith(
        "VALID"
    )

    # `s2e splits build`, then the split manifests validate against the same real files.
    build = [
        "splits", "build", "--data-root", str(data_root), "--sentences", str(sentences),
        "--val-unseen", "1",
    ]  # fmt: skip
    assert cli.main(build) == 0
    capsys.readouterr()
    split_files = [manifest_path(data_root, "domain", name) for name in CLIP_SPLITS]
    split_files += [manifest_path(data_root, "domain", n, "continuous") for n in ("val", "test")]
    domain = load_domain(sentences)
    report = validate_manifests(split_files, data_root, sentences=domain.text_by_id())
    assert report.issues == [] and report.n_rows == 22

    clips = {name: load_clip_rows(manifest_path(data_root, "domain", name)) for name in CLIP_SPLITS}
    assert check_split_guarantees(clips, domain) == []
    assert {name: len(rows) for name, rows in clips.items()} == {
        "train": 6, "val": 3 + 4, "test_seen": 3, "test_unseen": 4,
    }  # fmt: skip
    hashes = [r.source_sha256 for rows in clips.values() for r in rows]
    assert len(set(hashes)) == 20
    unseen_texts = {r.text for r in clips["test_unseen"]}
    assert len(unseen_texts) == 1
    assert unseen_texts.isdisjoint({r.text for r in clips["train"] + clips["val"]})

    # The default validates the recorder's files and the split files together.
    assert cli.main(validate) == 0
    assert "44 row(s) in 8 file(s): 0 error(s), 0 warning(s)" in capsys.readouterr().out


def test_record_refuses_bad_arguments(
    tmp_path: Path, sentences: Path, data_root: Path, capsys: pytest.CaptureFixture[str], tools
) -> None:
    GeneratedSource, ScriptedUI = tools.source_cls, tools.ui_cls
    day = datetime.date(2026, 11, 20)
    code, _ = _record(
        tmp_path, sentences, data_root, [], ["--session", "desk", "--ids", "a1,zz"],
        day, GeneratedSource, ScriptedUI,
    )  # fmt: skip
    assert code == 1 and "unknown sentence id(s): zz" in capsys.readouterr().out
    code, _ = _record(
        tmp_path, sentences, data_root, [], ["--session", "bad label"],
        day, GeneratedSource, ScriptedUI,
    )  # fmt: skip
    assert code == 1 and "may contain only" in capsys.readouterr().out
    assert not data_root.exists()


def test_ids_and_role_filters_and_count(
    tmp_path: Path, sentences: Path, data_root: Path, capsys: pytest.CaptureFixture[str], tools
) -> None:
    GeneratedSource, ScriptedUI = tools.source_cls, tools.ui_cls
    day = datetime.date(2026, 11, 20)
    code, _ = _record(
        tmp_path, sentences, data_root, TAKE * 5,
        ["--session", "desk", "--role", "train", "--count", "3", "--order", "file"],
        day, GeneratedSource, ScriptedUI,
    )  # fmt: skip
    assert (
        code == 0
        and "3 take(s) saved, 0 redone, 0 dropped (count_reached)" in capsys.readouterr().out
    )
    rows = load_clip_rows(manifest_path(data_root, "domain", "takes"))
    assert [r.sentence_id for r in rows] == ["a1", "a2", "a3"]


def test_session_id_is_the_date_and_the_label() -> None:
    assert session_id_for("desk-evening", datetime.date(2026, 11, 16)) == "2026-11-16_desk-evening"
    today = datetime.date.today().isoformat()
    assert session_id_for("x") == f"{today}_x"


def test_record_command_is_registered_and_needs_signer_and_session() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["record", "--signer", "me", "--session", "desk"])
    assert callable(args.func)
    assert args.sentences == "domain/sentences.yaml" and args.takes == 3 and not args.continuous
    with pytest.raises(SystemExit):
        parser.parse_args(["record", "--signer", "me"])


def test_new_commands_do_not_import_heavy_libraries_at_startup() -> None:
    code = (
        "import sys; from signtoenglish import cli; p = cli.build_parser(); "
        "p.parse_args(['record', '--signer', 'a', '--session', 'b']); "
        "heavy = [m for m in ('cv2', 'av', 'numpy', 'torch') if m in sys.modules]; "
        "print(heavy); sys.exit(1 if heavy else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_recordings_and_manifests_are_git_ignored(repo_root: Path) -> None:
    for path in (
        "data/domain/2026-11-20_desk/t01_me_take01.mp4",
        "data/domain/2026-11-20_desk/t01_me_take01.partial.mp4",
        "data/manifests/domain_takes.jsonl",
        "data/manifests/domain_continuous_takes.jsonl",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", path], cwd=repo_root, check=False
        )
        assert result.returncode == 0, path
