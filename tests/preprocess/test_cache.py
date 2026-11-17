"""Keypoint cache and the `s2e extract`, `s2e cache verify`, `s2e bench extract` commands.

The first half needs no model files. The second half runs real extraction on the fixture
clips into a temporary cache (see `conftest.py` for what happens where files are absent).
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from signtoenglish.cli import main
from signtoenglish.common.config import PathsConfig, load_config
from signtoenglish.common.hashing import sha256_file
from signtoenglish.preprocess import cache as cache_module
from signtoenglish.preprocess.cache import (
    KeypointCache,
    ManifestError,
    Source,
    get_or_extract,
    preprocessing_fingerprint,
    preprocessing_hash,
    read_manifest_sources,
    verify_sources,
)
from signtoenglish.preprocess.extract import PoseCore
from signtoenglish.preprocess.schema import (
    EXTRACTOR_VERSION,
    NUM_KEYPOINTS,
    KeypointConfig,
    PoseModelFile,
    RawKeypoints,
)
from signtoenglish.preprocess.video_io import ReadStats, VideoInfo

CFG = KeypointConfig()
SHA_A = "a" * 64
SHA_B = "b" * 64


def fake_raw(t: int = 7, seed: int = 0) -> RawKeypoints:
    rng = np.random.default_rng(seed)
    return RawKeypoints(
        timestamps_ms=np.arange(t, dtype=np.int64) * 33,
        keypoints=rng.uniform(0, 700, (t, NUM_KEYPOINTS, 2)).astype(np.float32),
        scores=rng.uniform(0, 1, (t, NUM_KEYPOINTS)).astype(np.float32),
        person_found=rng.uniform(size=t) > 0.2,
        width=1280,
        height=720,
    )


def fake_stats(t: int = 7) -> ReadStats:
    return ReadStats(
        decoded=t + 1,
        yielded=t,
        dropped_non_increasing=1,
        info=VideoInfo(1280, 720, 30.0, t + 1, 250, "h264"),
    )


def same(a: RawKeypoints, b: RawKeypoints) -> bool:
    return (
        all(
            getattr(a, n).tobytes() == getattr(b, n).tobytes()
            and getattr(a, n).dtype == getattr(b, n).dtype
            for n in ("timestamps_ms", "keypoints", "scores", "person_found")
        )
        and a.frame_size == b.frame_size
    )


def must_not_extract() -> PoseCore:
    raise AssertionError("the pose models were loaded, but this source should have been cached")


# --- the preprocessing hash ------------------------------------------------------------


def test_committed_config_file_is_the_default_config(repo_root: Path) -> None:
    loaded = load_config(repo_root / "configs" / "preprocess" / "keypoints.yaml", KeypointConfig)
    assert loaded == CFG
    assert preprocessing_hash(loaded) == preprocessing_hash(CFG)
    assert len(preprocessing_hash(CFG)) == 16


def _with_file(cfg: KeypointConfig, which: str, **changes: Any) -> KeypointConfig:
    return cfg.model_copy(update={which: getattr(cfg, which).model_copy(update=changes)})


@pytest.mark.parametrize(
    "changed",
    [
        CFG.model_copy(update={"model_rate_fps": 15.0}),
        CFG.model_copy(update={"model_rate_fps": None}),
        CFG.model_copy(update={"rate_slack": 0.1}),
        CFG.model_copy(update={"pose_provider": "coreml"}),
        _with_file(CFG, "detector", sha256="0" * 64),
        _with_file(CFG, "pose", sha256="1" * 64),
        _with_file(CFG, "detector", input_size=(640, 640)),
        _with_file(CFG, "pose", input_size=(288, 384)),
    ],
    ids=[
        "rate",
        "no-rate",
        "slack",
        "provider",
        "detector-file",
        "pose-file",
        "detector-size",
        "pose-size",
    ],
)
def test_every_extraction_setting_changes_the_key(changed: KeypointConfig) -> None:
    assert preprocessing_hash(changed) != preprocessing_hash(CFG)


def test_extractor_version_and_library_versions_change_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = preprocessing_hash(CFG)
    assert preprocessing_hash(CFG, extractor_version=EXTRACTOR_VERSION + 1) != base
    real = cache_module.version
    for package in ("rtmlib", "onnxruntime"):
        monkeypatch.setattr(
            cache_module, "version", lambda n, p=package: "99.0" if n == p else real(n)
        )
        assert preprocessing_hash(CFG) != base, package
    monkeypatch.setattr(cache_module, "version", real)
    assert preprocessing_hash(CFG) == base


def test_unrelated_settings_do_not_change_the_key(tmp_path: Path) -> None:
    moved = KeypointConfig(
        detector=PoseModelFile(
            relative_path="elsewhere/det.onnx",
            sha256=CFG.detector.sha256,
            input_size=CFG.detector.input_size,
        ),
        pose=PoseModelFile(
            relative_path="elsewhere/pose.onnx",
            sha256=CFG.pose.sha256,
            input_size=CFG.pose.input_size,
        ),
    )
    assert preprocessing_hash(moved) == preprocessing_hash(CFG)  # where a model file sits
    one, two = KeypointCache(tmp_path / "one", CFG), KeypointCache(tmp_path / "two", CFG)
    assert one.hash == two.hash and one.dir != two.dir  # where the cache sits
    fingerprint = json.dumps(preprocessing_fingerprint(CFG))
    assert "relative_path" not in fingerprint and "/" not in fingerprint.replace(
        "opencv-python", ""
    )


# --- entries on disk -------------------------------------------------------------------


def test_round_trip_layout_and_meta(tmp_path: Path) -> None:
    cache = KeypointCache(tmp_path, CFG)
    raw = fake_raw()
    path = cache.store(SHA_A, raw, fake_stats())
    assert path == tmp_path / "keypoints" / cache.hash / "aa" / f"{SHA_A}.npz"
    entry = cache.load(SHA_A)
    assert entry is not None and same(entry.raw, raw)
    meta = entry.meta
    for key in (
        "schema_version", "source_sha256", "width", "height", "nominal_fps", "n_source_frames",
        "n_kept_frames", "n_dropped_duplicate_timestamps", "model_rate_fps", "preprocessing_hash",
        "pose_models", "rtmlib_version", "onnxruntime_version", "execution_provider",
        "extractor_version", "created_utc",
    ):  # fmt: skip
        assert key in meta, key
    assert meta["n_source_frames"] == 8 and meta["n_kept_frames"] == 7
    assert meta["n_dropped_duplicate_timestamps"] == 1
    assert meta["execution_provider"] == {"detector": "cpu", "pose": "cpu"}
    assert meta["pose_models"]["pose"]["sha256"] == CFG.pose.sha256
    dir_meta = json.loads((cache.dir / "meta.json").read_text(encoding="utf-8"))
    assert dir_meta["preprocessing_hash"] == cache.hash
    assert dir_meta["fingerprint"] == preprocessing_fingerprint(CFG)
    assert cache.load(SHA_B) is None and cache.inspect(SHA_B)[1] == "missing"
    with pytest.raises(ValueError, match="SHA-256"):
        cache.path_for("not-a-hash")


def test_an_interrupted_write_leaves_nothing_that_reads_as_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = KeypointCache(tmp_path, CFG)
    final = cache.path_for(SHA_A)

    # 1. The process dies after the data is written but before the rename.
    real_replace = Path.replace

    def dying_replace(self: Path, target: Any) -> Any:
        if str(target).endswith(".npz"):
            raise KeyboardInterrupt("killed before the rename")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", dying_replace)
    with pytest.raises(KeyboardInterrupt):
        cache.store(SHA_A, fake_raw(), fake_stats())
    monkeypatch.setattr(Path, "replace", real_replace)
    assert not final.exists() and cache.load(SHA_A) is None

    # 2. A hard kill that could not clean up: a stray temporary file is never read.
    stray = final.with_name(f".{SHA_A}.12345.tmp")
    cache.store(SHA_B, fake_raw(), fake_stats())
    stray.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cache.path_for(SHA_B), stray)
    assert cache.load(SHA_A) is None and cache.inspect(SHA_A)[1] == "missing"

    # 3. A file cut short at the final name (a copy that stopped half way) reads as corrupt.
    whole = cache.path_for(SHA_B).read_bytes()
    for cut in (0, 100, len(whole) // 2, len(whole) - 40):
        final.write_bytes(whole[:cut])
        entry, status = cache.inspect(SHA_A)
        assert entry is None and status.startswith("corrupt"), (cut, status)

    # 4. After a real store the entry is whole again.
    cache.store(SHA_A, fake_raw(seed=3), fake_stats())
    loaded = cache.load(SHA_A)
    assert loaded is not None and same(loaded.raw, fake_raw(seed=3))


def test_an_entry_under_the_wrong_name_or_hash_is_not_accepted(tmp_path: Path) -> None:
    cache = KeypointCache(tmp_path, CFG)
    cache.store(SHA_A, fake_raw(), fake_stats())
    wrong_source = cache.path_for(SHA_B)
    wrong_source.parent.mkdir(parents=True)
    shutil.copyfile(cache.path_for(SHA_A), wrong_source)
    assert cache.inspect(SHA_B)[1].startswith("corrupt: written for source")
    other = KeypointCache(tmp_path, CFG.model_copy(update={"model_rate_fps": 15.0}))
    target = other.path_for(SHA_A)
    target.parent.mkdir(parents=True)
    shutil.copyfile(cache.path_for(SHA_A), target)
    assert other.inspect(SHA_A)[1].startswith("corrupt: written for preprocessing hash")


def manifest_row(name: str, sha: str, **extra: Any) -> str:
    row = {
        "id": f"take-{name}", "dataset": "domain", "split": "unassigned", "source_path": name,
        "source_sha256": sha, "start_ms": None, "end_ms": None, "text": "Hello.",
        "sentence_id": "s001", "signer_id": "p1", "session_id": "2026-11-17_a", "take": 1,
        "sign_language": "ase", "text_language": "en", "license": "own-recording", "tags": [],
    }  # fmt: skip
    return json.dumps({**row, **extra})


def test_manifest_reading_needs_only_two_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        "\n".join(
            [
                manifest_row("a.mp4", SHA_A),
                manifest_row(
                    "a.mp4", SHA_A, id="second-clip-in-the-same-file", start_ms=0, end_ms=900
                ),
                "",
                json.dumps({"source_path": "b.mov", "source_sha256": SHA_B, "segments": []}),
            ]
        ),
        encoding="utf-8",
    )
    sources = read_manifest_sources(manifest, tmp_path / "data")
    assert [(s.path.name, s.expected_sha256) for s in sources] == [
        ("a.mp4", SHA_A),
        ("b.mov", SHA_B),
    ]
    assert sources[0].path == tmp_path / "data" / "a.mp4" and sources[0].origin.endswith(
        "m.jsonl:1"
    )
    for bad, message in [
        ("{not json", "not JSON"),
        (json.dumps({"source_path": "a.mp4"}), "source_path and source_sha256"),
        (json.dumps({"source_path": "a.mp4", "source_sha256": "abc"}), "SHA-256"),
        (
            json.dumps({"source_path": "../a.mp4", "source_sha256": SHA_A}),
            "relative to the data root",
        ),
    ]:
        manifest.write_text(bad, encoding="utf-8")
        with pytest.raises(ManifestError, match=message):
            read_manifest_sources(manifest, tmp_path)
    with pytest.raises(ManifestError, match="cannot read"):
        read_manifest_sources(tmp_path / "absent.jsonl", tmp_path)


def test_verify_reports_missing_and_corrupt_entries(tmp_path: Path) -> None:
    cache = KeypointCache(tmp_path, CFG)
    cache.store(SHA_A, fake_raw(), fake_stats())
    cache.store(SHA_B, fake_raw(), fake_stats())
    sources = [Source(tmp_path / "a.mp4", SHA_A, "m:1"), Source(tmp_path / "b.mp4", SHA_B, "m:2")]
    assert [r.status for r in verify_sources(sources, cache)] == ["ok", "ok"]
    cache.path_for(SHA_A).unlink()
    cache.path_for(SHA_B).write_bytes(b"junk")
    statuses = [r.status for r in verify_sources(sources, cache)]
    assert statuses[0] == "missing" and statuses[1].startswith("corrupt")
    assert [r.status for r in verify_sources(sources, cache, check_sources=True)] == [
        "source missing"
    ] * 2


# --- real extraction through the cache -------------------------------------------------


@pytest.fixture(scope="module")
def warm(
    core: PoseCore, clip_path: Callable[[str], Path], tmp_path_factory: pytest.TempPathFactory
) -> tuple[KeypointCache, Path, str]:
    """A temporary cache holding `hands_leave`, extracted for real once."""
    cache = KeypointCache(tmp_path_factory.mktemp("cache"), CFG)
    clip = clip_path("hands_leave")
    entry, outcome = get_or_extract(Source(clip, None, "file"), cache, lambda: core)
    assert entry is not None and outcome.status == "extracted" and outcome.frames == 150
    assert outcome.source_sha256 is not None
    return cache, clip, outcome.source_sha256


def test_second_run_does_no_extraction_and_returns_identical_data(
    warm: tuple[KeypointCache, Path, str], raw_of: Callable[[str], RawKeypoints]
) -> None:
    cache, clip, sha = warm
    stored_bytes = cache.path_for(sha).read_bytes()
    entry, outcome = get_or_extract(Source(clip, None, "file"), cache, must_not_extract)
    assert outcome.status == "cached" and entry is not None
    assert same(entry.raw, raw_of("hands_leave"))  # equal to a direct extraction, bit for bit
    again, _ = get_or_extract(Source(clip, None, "file"), cache, must_not_extract)
    assert again is not None and same(again.raw, entry.raw)
    assert cache.path_for(sha).read_bytes() == stored_bytes  # and the file was not rewritten
    assert entry.meta["n_source_frames"] == 150 and entry.meta["nominal_fps"] == 30.0
    assert entry.meta["source_sha256"] == sha == sha256_file(clip)


def test_a_copied_and_retimestamped_source_still_hits_the_cache(
    warm: tuple[KeypointCache, Path, str], tmp_path: Path
) -> None:
    cache, clip, sha = warm
    copy = tmp_path / "renamed take (copy).mov"
    shutil.copyfile(clip, copy)
    os.utime(copy, (1_000_000_000, 1_000_000_000))  # a different modification time
    entry, outcome = get_or_extract(Source(copy, None, "file"), cache, must_not_extract)
    assert outcome.status == "cached" and outcome.source_sha256 == sha and entry is not None


def test_a_changed_setting_misses_the_cache_and_extracts_again(
    warm: tuple[KeypointCache, Path, str], paths: PathsConfig, tmp_path: Path
) -> None:
    cache, clip, sha = warm
    slower = CFG.model_copy(update={"model_rate_fps": 15.0})
    other = KeypointCache(cache.dir.parents[1], slower)
    assert other.dir != cache.dir and other.load(sha) is None
    entry, outcome = get_or_extract(
        Source(clip, None, "file"), other, lambda: PoseCore(slower, paths.assets_root)
    )
    assert outcome.status == "extracted" and entry is not None and len(entry.raw) == 75
    assert cache.load(sha) is not None and len(cache.load(sha).raw) == 150  # type: ignore[union-attr]


def test_a_source_that_does_not_match_its_manifest_hash_is_refused(
    warm: tuple[KeypointCache, Path, str],
) -> None:
    cache, clip, _ = warm
    entry, outcome = get_or_extract(Source(clip, SHA_A, "m.jsonl:1"), cache, must_not_extract)
    assert (
        entry is None
        and outcome.status == "failed"
        and "differs from the manifest" in outcome.detail
    )
    entry, outcome = get_or_extract(
        Source(clip.with_name("gone.mov"), None, "file"), cache, must_not_extract
    )
    assert entry is None and outcome.status == "failed" and "not found" in outcome.detail


# --- the commands ----------------------------------------------------------------------


@pytest.fixture
def cli_env(
    tmp_path: Path,
    repo_root: Path,
    paths: PathsConfig,
    monkeypatch: pytest.MonkeyPatch,
    clip_path: Callable[[str], Path],
    keypoint_cfg: KeypointConfig,
    need_file: Callable[[Path, str], Path],
) -> dict[str, Any]:
    """Commands run from the repo root with a temporary cache and a temporary data root
    holding copies of both clips and a manifest in the ARCHITECTURE.md 4.2 format."""
    need_file(paths.assets_root / keypoint_cfg.pose.relative_path, "pose model file (asset A6)")
    data = tmp_path / "data"
    (data / "domain").mkdir(parents=True)
    names = {}
    for stem in ("hands_visible", "hands_leave"):
        source = clip_path(stem)
        names[stem] = f"domain/{source.name}"
        shutil.copyfile(source, data / names[stem])
    manifest = tmp_path / "domain_takes.jsonl"
    manifest.write_text(
        "\n".join(
            manifest_row(name, sha256_file(data / name), id=stem) for stem, name in names.items()
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo_root)
    monkeypatch.setenv("S2E_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("S2E_DATA_ROOT", str(data))
    monkeypatch.setenv("S2E_ASSETS_ROOT", str(paths.assets_root))
    return {
        "data": data,
        "manifest": manifest,
        "names": names,
        "cache_root": tmp_path / "cache",
        "tmp": tmp_path,
    }


def last_json(capsys: pytest.CaptureFixture[str]) -> tuple[dict[str, Any], str]:
    out = capsys.readouterr().out
    return json.loads(out.strip().splitlines()[-1]), out


def test_extract_over_a_list_is_resumable_and_verify_reports_a_deleted_entry(
    cli_env: dict[str, Any], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data, manifest, names = cli_env["data"], str(cli_env["manifest"]), cli_env["names"]
    first, second = (str(data / names[s]) for s in ("hands_visible", "hands_leave"))
    broken = cli_env["tmp"] / "broken.mp4"
    broken.write_bytes(b"not a video at all" * 100)

    # Nothing is cached yet: verify says so and exits non-zero.
    assert main(["cache", "verify", "--manifest", manifest]) == 1
    summary, _ = last_json(capsys)
    assert summary["sources"] == 2 and summary["missing"] == 2 and summary["ok"] == 0

    # A run that gets through one file and hits a bad one: the good one is kept.
    assert main(["extract", first, str(broken)]) == 1
    summary, out = last_json(capsys)
    assert (summary["extracted"], summary["cached"], summary["failed"]) == (1, 0, 1)
    assert "failed" in out and "broken.mp4" in out and summary["frames_extracted"] == 150

    # Run again over the whole list from the manifest: only the missing file is extracted.
    loads: list[int] = []
    real_init = PoseCore.__init__

    def counting_init(self: PoseCore, *a: Any, **k: Any) -> None:
        loads.append(1)
        real_init(self, *a, **k)

    monkeypatch.setattr(PoseCore, "__init__", counting_init)
    assert main(["extract", "--manifest", manifest]) == 0
    summary, _ = last_json(capsys)
    assert (summary["extracted"], summary["cached"], summary["failed"]) == (1, 1, 0)
    assert len(loads) == 1

    # A third run does nothing and never loads the pose models.
    assert main(["extract", "--manifest", manifest, second]) == 0
    summary, _ = last_json(capsys)
    assert (summary["extracted"], summary["cached"], summary["failed"]) == (0, 3, 0)
    assert len(loads) == 1

    # The cache is clean; then one entry is deleted and another damaged.
    assert main(["cache", "verify", "--manifest", manifest, "--check-sources"]) == 0
    summary, _ = last_json(capsys)
    assert summary["ok"] == 2 and summary["missing"] == summary["corrupt"] == 0
    cache = KeypointCache(cli_env["cache_root"], KeypointConfig())
    assert summary["preprocessing_hash"] == cache.hash
    deleted = cache.path_for(sha256_file(Path(first)))
    deleted.unlink()
    assert main(["cache", "verify", "--manifest", manifest]) == 1
    summary, out = last_json(capsys)
    assert summary["missing"] == 1 and summary["ok"] == 1
    assert "missing: " in out and names["hands_visible"] in out
    damaged = cache.path_for(sha256_file(Path(second)))
    damaged.write_bytes(damaged.read_bytes()[:5000])
    assert main(["cache", "verify", "--manifest", manifest]) == 1
    summary, out = last_json(capsys)
    assert summary["missing"] == 1 and summary["corrupt"] == 1 and "corrupt" in out

    # Extract repairs both.
    assert main(["extract", "--manifest", manifest]) == 0
    assert main(["cache", "verify", "--manifest", manifest]) == 0


def test_extract_refuses_a_source_that_changed_since_the_manifest_was_written(
    cli_env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    changed = cli_env["data"] / cli_env["names"]["hands_leave"]
    with changed.open("ab") as fh:
        fh.write(b"\0")
    only_that_row = cli_env["tmp"] / "one.jsonl"
    only_that_row.write_text(
        cli_env["manifest"].read_text(encoding="utf-8").splitlines()[1] + "\n", encoding="utf-8"
    )
    assert main(["extract", "--manifest", str(only_that_row)]) == 1
    summary, out = last_json(capsys)
    assert (
        summary["failed"] == 1 and summary["extracted"] == 0 and "differs from the manifest" in out
    )
    assert main(["cache", "verify", "--manifest", str(only_that_row), "--check-sources"]) == 1
    assert "source changed" in capsys.readouterr().out


def test_commands_report_usage_errors(
    cli_env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["extract"]) == 2
    assert main(["extract", "--manifest", str(cli_env["tmp"] / "absent.jsonl")]) == 2
    assert main(["cache"]) == 2 and main(["bench"]) == 2
    capsys.readouterr()


def test_bench_extract_measures_throughput(
    cli_env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    clip = str(cli_env["data"] / cli_env["names"]["hands_visible"])
    assert main(["bench", "extract", clip, "--frames", "60"]) == 0
    report, _ = last_json(capsys)
    print("bench extract:", report)
    assert report["frames_processed"] == 60
    assert report["providers"]["pose"] == ["CPUExecutionProvider"]
    assert report["providers"]["detector"] == ["CPUExecutionProvider"]
    # Speed is measured and reported, never asserted: it halves when the machine is busy, and
    # a threshold here would fail `make check` for reasons that have nothing to do with the
    # code. The target (10 frames per second) is checked by hand with `s2e bench extract`.
    for field in (
        "pose_fps_excluding_decode",
        "fps_including_decode",
        "frame_ms_median",
        "frame_ms_p95",
    ):
        assert report[field] > 0, field
    assert report["frame_ms_p95"] >= report["frame_ms_median"]
    assert report["session_init_s"] >= 0 and report["frames_decoded"] >= 60
    assert not cli_env["cache_root"].exists()  # a benchmark never writes to the cache
