"""Offline translation and the commands built on it, on the fixture clips with the real
model (OpenASL checkpoint, CPU, loaded once and shared)."""

from __future__ import annotations

import gzip
import json
import pickle
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from signtoenglish.cli import main
from signtoenglish.common.hashing import sha256_file
from signtoenglish.infer import loader
from signtoenglish.infer.offline import (
    OfflineError,
    OfflineTranslator,
    load_published_labels,
    load_published_pose,
    select_frames,
)
from signtoenglish.preprocess.schema import RawKeypoints

CLIPS = ("hands_visible", "hands_leave")


# --- frame selection (no files needed) -------------------------------------------------


def test_select_frames_keeps_everything_up_to_the_maximum() -> None:
    assert select_frames(0, 256).tolist() == []
    assert select_frames(150, 256).tolist() == list(range(150))
    assert select_frames(256, 256).tolist() == list(range(256))


@pytest.mark.parametrize("n", [257, 300, 512, 1000, 5000])
def test_select_frames_thins_long_inputs_evenly_and_repeatably(n: int) -> None:
    idx = select_frames(n, 256)
    assert len(idx) == 256 and idx[0] == 0 and idx[-1] == n - 1
    assert np.all(np.diff(idx) >= 1)  # increasing, no frame used twice
    steps = np.diff(idx)
    assert steps.max() - steps.min() <= 1  # evenly spaced
    assert np.array_equal(idx, select_frames(n, 256))  # the same every time
    with pytest.raises(ValueError, match="bad arguments"):
        select_frames(10, 0)


# --- downloaded pickles are read as data only ------------------------------------------


class _Exploit:
    def __reduce__(self) -> Any:
        import os

        return (os.system, ("echo should-never-run",))


def test_published_files_are_unpickled_as_data_only(tmp_path: Path) -> None:
    evil = tmp_path / "evil.pkl"
    evil.write_bytes(pickle.dumps({"keypoints": _Exploit(), "scores": []}))
    with pytest.raises(OfflineError, match="refusing to unpickle"):
        load_published_pose(evil)
    evil_labels = tmp_path / "labels.evil"
    evil_labels.write_bytes(gzip.compress(pickle.dumps({"a": _Exploit()})))
    with pytest.raises(OfflineError, match="refusing to unpickle"):
        load_published_labels(evil_labels)

    rng = np.random.default_rng(0)
    pose = {
        "keypoints": [rng.uniform(size=(1, 133, 2)) for _ in range(12)],
        "scores": [rng.uniform(size=(1, 133)).astype(np.float32) for _ in range(12)],
    }
    good = tmp_path / "good.pkl"
    good.write_bytes(pickle.dumps(pose))
    keypoints, scores = load_published_pose(good)
    assert keypoints.shape == (12, 133, 2) and scores.shape == (12, 133)
    np.testing.assert_array_equal(keypoints[3], pose["keypoints"][3][0])
    good.write_bytes(pickle.dumps({**pose, "start": 2, "end": 9}))  # the optional frame range
    assert load_published_pose(good)[0].shape == (7, 133, 2)
    good.write_bytes(pickle.dumps({"scores": []}))
    with pytest.raises(OfflineError, match="expected a dict"):
        load_published_pose(good)

    labels = tmp_path / "labels.test"
    samples = {
        "k1": {
            "name": "clip_one",
            "video_path": "clip_one.mp4",
            "text": "First sentence.",
            "gloss": [],
        },
        "k2": {"name": "clip_two", "video_path": "sub/clip_two.mp4", "text": "Second sentence."},
    }
    labels.write_bytes(gzip.compress(pickle.dumps(samples)))
    loaded = load_published_labels(labels)
    assert [(lab.name, lab.video_path, lab.text) for lab in loaded] == [
        ("clip_one", "clip_one.mp4", "First sentence."),
        ("clip_two", "sub/clip_two.mp4", "Second sentence."),
    ]
    labels.write_bytes(b"not gzip")
    with pytest.raises(OfflineError, match="not a gzip file"):
        load_published_labels(labels)


# --- the pipeline on real clips ----------------------------------------------------------


@pytest.fixture(scope="module")
def offline(
    model_for: Callable[..., Any], paths: Any, tmp_path_factory: pytest.TempPathFactory
) -> OfflineTranslator:
    cached_paths = paths.model_copy(update={"cache_root": tmp_path_factory.mktemp("offline_cache")})
    return OfflineTranslator(model_for("openasl", "cpu"), cached_paths)


def test_translate_file_gives_the_wrapper_output_and_uses_the_cache(
    offline: OfflineTranslator,
    clip_path: Callable[[str], Path],
    clip_inputs: Callable[[str], dict[str, np.ndarray]],
) -> None:
    translator = offline.loaded.translator
    for stem in CLIPS:
        first = offline.translate_file(clip_path(stem))
        expected_text, expected_ids = translator.translate_with_ids(clip_inputs(stem))[0]
        assert first.text == expected_text and list(first.token_ids) == expected_ids
        assert first.n_frames == first.n_frames_used == 150 and first.keypoints_from == "extracted"
        again = offline.translate_file(
            clip_path(stem), expected_sha256=sha256_file(clip_path(stem))
        )
        assert again.keypoints_from == "cache" and again.text == first.text
        report = first.to_dict()
        assert set(report) >= {"id", "text", "n_frames", "decode_ms"}  # ARCHITECTURE.md 4.6
        assert report["id"] == clip_path(stem).name and report["decode_ms"] > 0
        print(f"{stem}: {first.text!r} in {first.decode_ms:.0f} ms")


def test_time_slices_and_errors(
    offline: OfflineTranslator, clip_path: Callable[[str], Path], tmp_path: Path
) -> None:
    clip = clip_path("hands_leave")
    part = offline.translate_file(clip, start_ms=1000, end_ms=3000)
    assert part.n_frames == 60 and part.text.strip()
    with pytest.raises(OfflineError, match="no frames between"):
        offline.translate_file(clip, start_ms=9000, end_ms=9500)
    with pytest.raises(OfflineError, match="not found"):
        offline.translate_file(tmp_path / "absent.mp4")
    with pytest.raises(OfflineError, match="differs from the manifest"):
        offline.translate_file(clip, expected_sha256="0" * 64)
    uncached = OfflineTranslator(offline.loaded, offline._paths, use_cache=False)
    with pytest.raises(OfflineError, match="not found"):
        uncached.translate_file(tmp_path / "absent.mp4")


def test_inputs_over_256_frames_are_thinned_before_normalization(
    offline: OfflineTranslator, raw_of: Callable[[str], RawKeypoints]
) -> None:
    raw = raw_of("hands_visible")
    reps = 3  # 450 frames: the clip three times over, with increasing timestamps
    long = RawKeypoints(
        timestamps_ms=np.concatenate([raw.timestamps_ms + i * 6000 for i in range(reps)]),
        keypoints=np.concatenate([raw.keypoints] * reps),
        scores=np.concatenate([raw.scores] * reps),
        person_found=np.concatenate([raw.person_found] * reps),
        width=raw.width,
        height=raw.height,
    )
    a = offline.translate_raw(long, clip_id="long")
    b = offline.translate_raw(long, clip_id="long")
    assert a.n_frames == 450 and a.n_frames_used == 256
    assert a.token_ids == b.token_ids and a.text.strip()  # repeatable, unlike random sampling
    keep = select_frames(450, 256)
    by_hand = offline.translate_arrays(long.keypoints[keep], long.scores[keep], long.frame_size)
    assert by_hand.token_ids == a.token_ids


def write_published_set(
    folder: Path, raws: dict[str, RawKeypoints], texts: dict[str, str]
) -> tuple[Path, Path]:
    """The fixture clips' own keypoints, saved in the Uni-Sign authors' published layout,
    with made-up reference sentences. SYNTHETIC stand-in for the How2Sign files."""
    pose_dir = folder / "pose_format" / "test"
    pose_dir.mkdir(parents=True)
    samples = {}
    for name, raw in raws.items():
        size = np.array(raw.frame_size, dtype=np.float64)
        pose = {
            "keypoints": [(k.astype(np.float64) / size)[None] for k in raw.keypoints],
            "scores": [s[None] for s in raw.scores],
        }
        (pose_dir / f"{name}.pkl").write_bytes(pickle.dumps(pose))
        samples[name] = {"name": name, "video_path": f"{name}.mp4", "text": texts[name]}
    labels = folder / "labels.test"
    labels.write_bytes(gzip.compress(pickle.dumps(samples)))
    return pose_dir, labels


def test_published_format_gives_the_same_translation_as_our_own_keypoints(
    offline: OfflineTranslator, raw_of: Callable[[str], RawKeypoints], tmp_path: Path
) -> None:
    raws = {stem: raw_of(stem) for stem in CLIPS}
    pose_dir, _ = write_published_set(tmp_path, raws, dict.fromkeys(CLIPS, "x"))
    for stem in CLIPS:
        ours = offline.translate_raw(raws[stem], clip_id=stem)
        published = offline.translate_published(pose_dir / f"{stem}.pkl")
        assert published.keypoints_from == "published" and published.id == stem
        assert published.token_ids == ours.token_ids


# --- the commands ------------------------------------------------------------------------


@pytest.fixture
def reuse_loaded_model(
    offline: OfflineTranslator, monkeypatch: pytest.MonkeyPatch, repo_root: Path, tmp_path: Path
) -> Path:
    """Commands in these tests reuse the model that is already loaded instead of loading it
    again for every invocation (one command below does load it for real). Returns a
    temporary cache root."""
    loaded = offline.loaded

    def already_loaded(paths: Any, model_cfg: Any, keypoint_cfg: Any, **kwargs: Any) -> Any:
        assert kwargs.get("fine_tuned") is None
        assert kwargs.get("base_checkpoint") in (None, "openasl")
        return loaded

    monkeypatch.setattr(loader, "load_translator", already_loaded)
    monkeypatch.chdir(repo_root)
    monkeypatch.setenv("S2E_CACHE_ROOT", str(tmp_path / "cache"))
    return tmp_path / "cache"


def test_reproduce_command_runs_end_to_end_on_a_synthetic_published_set(
    reuse_loaded_model: Path,
    offline: OfflineTranslator,
    raw_of: Callable[[str], RawKeypoints],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The How2Sign reproduction check as one command, exercised on SYNTHETIC data: the two
    fixture clips' keypoints in the published layout and invented reference sentences. This
    proves the command, the file formats, the subset option and the test-split log. It says
    nothing about the How2Sign score, which needs the real files (deferred)."""
    raws = {f"{stem}_{i}": raw_of(stem) for stem in CLIPS for i in range(3)}
    own = {stem: offline.translate_raw(raw_of(stem)).text for stem in CLIPS}
    texts = {
        name: own[name.rsplit("_", 1)[0]] if name.endswith("_0") else "A different sentence."
        for name in raws
    }
    pose_dir, labels = write_published_set(tmp_path, raws, texts)
    run_dir = tmp_path / "run"
    args = ["reproduce", "--pose-dir", str(pose_dir), "--labels", str(labels)]
    args += ["--run-dir", str(run_dir), "--checkpoint", "openasl", "--device", "cpu"]

    assert main(args) == 0
    out = capsys.readouterr().out
    extra = json.loads(out.strip().splitlines()[-1])
    assert (
        extra["sentences_scored"] == 6
        and extra["is_subset"] is False
        and extra["checkpoint"] == "openasl"
    )
    assert set(extra["reference_style_bleu"]) >= {"bleu1", "bleu4", "definition"}
    eval_dir = run_dir / "eval" / "test"
    metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "signature" in json.dumps(metrics)  # the sacreBLEU signature is recorded
    hyps = (eval_dir / "hyps.txt").read_text(encoding="utf-8").splitlines()
    refs = (eval_dir / "refs.txt").read_text(encoding="utf-8").splitlines()
    assert len(hyps) == len(refs) == 6
    assert sum(h == r for h, r in zip(hyps, refs, strict=True)) == 2  # the two planted matches
    assert json.loads((eval_dir / "reproduction.json").read_text(encoding="utf-8")) == extra
    assert len((run_dir / "test_runs.log").read_text(encoding="utf-8").splitlines()) == 1

    # A test split is scored once per checkpoint: a second run is refused, --force is logged.
    assert main(args) == 1
    assert "already" in capsys.readouterr().err.lower()
    assert main([*args, "--force", "--subset", "4", "--seed", "7"]) == 0
    extra = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (
        extra["sentences_scored"] == 4 and extra["is_subset"] is True and extra["subset_seed"] == 7
    )
    assert len((run_dir / "test_runs.log").read_text(encoding="utf-8").splitlines()) == 2

    (pose_dir / "hands_leave_2.pkl").unlink()
    assert main([*args, "--force"]) == 2
    assert "pose files are missing" in capsys.readouterr().err


def test_eval_command_scores_a_manifest_split(
    reuse_loaded_model: Path,
    offline: OfflineTranslator,
    clip_path: Callable[[str], Path],
    raw_of: Callable[[str], RawKeypoints],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = tmp_path / "data"
    (data / "domain").mkdir(parents=True)
    rows = []
    own = offline.translate_raw(raw_of("hands_leave")).text
    for stem, text in (("hands_visible", "Wash your hands."), ("hands_leave", own)):
        target = data / "domain" / clip_path(stem).name
        shutil.copyfile(clip_path(stem), target)
        rows.append({
            "id": stem, "dataset": "domain", "split": "val", "source_path": f"domain/{target.name}",
            "source_sha256": sha256_file(target), "start_ms": None, "end_ms": None, "text": text,
            "sentence_id": None, "signer_id": "p1", "session_id": "s", "take": 1,
            "sign_language": "ase", "text_language": "en", "license": "own-recording", "tags": [],
        })  # fmt: skip
    manifest = tmp_path / "domain_val.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("S2E_DATA_ROOT", str(data))
    run_dir = tmp_path / "run"
    assert (
        main(["eval", "--manifest", str(manifest), "--split", "val", "--run-dir", str(run_dir)])
        == 0
    )
    assert "val" in capsys.readouterr().out
    predictions = [
        json.loads(line)
        for line in (run_dir / "eval" / "val" / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [p["id"] for p in predictions] == ["hands_visible", "hands_leave"]
    assert predictions[1]["hyp"] == own and predictions[1]["exact"] is True
    metrics = json.loads((run_dir / "eval" / "val" / "metrics.json").read_text(encoding="utf-8"))
    assert offline.loaded.checkpoint_id in json.dumps(metrics)
    assert offline.loaded.preprocessing_hash in json.dumps(metrics)
    assert not (run_dir / "test_runs.log").exists()  # val is not a test split
    assert (
        main(
            [
                "eval",
                "--manifest",
                str(tmp_path / "absent.jsonl"),
                "--split",
                "val",
                "--run-dir",
                str(run_dir),
            ]
        )
        == 2
    )
    capsys.readouterr()


def test_translate_command_loads_the_model_for_real(
    offline: OfflineTranslator,
    clip_path: Callable[[str], Path],
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`s2e translate` from start to finish, with its own model load (the shared one is
    freed first so two copies are never in memory)."""
    expected = offline.translate_file(clip_path("hands_leave")).text
    offline.loaded = None  # type: ignore[assignment]
    import gc

    gc.collect()
    monkeypatch.chdir(repo_root)
    monkeypatch.setenv("S2E_CACHE_ROOT", str(tmp_path / "cache"))
    clip = str(clip_path("hands_leave"))
    assert main(["translate", clip, "--device", "cpu", "--json"]) == 0
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["text"] == expected and report["n_frames"] == 150
    assert report["model"]["checkpoint"] == "openasl" and report["model"]["device"] == "cpu"
    assert len(report["model"]["preprocessing_hash"]) == 16
    assert main(["translate", str(tmp_path / "absent.mp4"), "--checkpoint", "nope"]) == 2
    assert "unknown checkpoint" in capsys.readouterr().err
