"""Fixtures for the preprocessing tests, and the rule for tests that need local files.

Some of these tests run the real pose estimator on real video and compare against the
Uni-Sign reference code. They need files that are never in git:

- pose model files under the assets root (`assets/`, or `S2E_ASSETS_ROOT`);
- the Uni-Sign checkout under the third-party root (`third_party/`, or `S2E_THIRD_PARTY_ROOT`);
- the fixture clips `hands_visible.*` and `hands_leave.*` under `tests/fixtures/clips/`
  (or `S2E_FIXTURE_CLIPS`);
- for one test, a Uni-Sign checkpoint and the mT5 tokenizer files under the assets root.

What happens when one is missing (`need()` below):

- If this checkout has an assets directory at all, or `S2E_REQUIRE_ASSETS=1` is set, the
  test FAILS with the path it wanted. A machine that is supposed to have the files must
  not pass by skipping.
- Otherwise (a clean clone, or a worktree that never had assets) the test is SKIPPED with a
  reason starting "LOCAL FILES ABSENT", which `pytest -ra` prints in the summary. A skip
  does not satisfy a phase gate; run gates with `S2E_REQUIRE_ASSETS=1`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from signtoenglish.common.config import PathsConfig, load_paths
from signtoenglish.preprocess.extract import PoseCore, extract_file
from signtoenglish.preprocess.schema import KeypointConfig, RawKeypoints

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CLI = Path(__file__).with_name("reference_cli.py")
CLIP_STEMS = ("hands_visible", "hands_leave")
CLIP_SUFFIXES = (".mp4", ".mov", ".webm", ".mkv")


def _paths() -> PathsConfig:
    return load_paths(REPO_ROOT)


def need(path: Path, what: str) -> Path:
    """Return `path` if it exists; otherwise fail or skip by the rule in the module docstring."""
    if path.exists():
        return path
    message = f"{what} not found at {path}"
    if os.environ.get("S2E_REQUIRE_ASSETS") == "1" or _paths().assets_root.is_dir():
        pytest.fail(f"REQUIRED LOCAL FILE MISSING: {message}", pytrace=False)
    pytest.skip(f"LOCAL FILES ABSENT: {message} (not a pass; see tests/preprocess/conftest.py)")


def need_tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    message = f"command-line tool `{name}` is not installed"
    if os.environ.get("S2E_REQUIRE_ASSETS") == "1":
        pytest.fail(f"REQUIRED TOOL MISSING: {message}", pytrace=False)
    pytest.skip(f"LOCAL FILES ABSENT: {message}")


@pytest.fixture(scope="session")
def paths() -> PathsConfig:
    return _paths()


@pytest.fixture(scope="session")
def keypoint_cfg() -> KeypointConfig:
    return KeypointConfig()


def _pose_files(cfg: KeypointConfig, paths: PathsConfig) -> tuple[Path, Path]:
    det = need(paths.assets_root / cfg.detector.relative_path, "detector model file (asset A6)")
    pose = need(paths.assets_root / cfg.pose.relative_path, "pose model file (asset A6)")
    return det, pose


@pytest.fixture(scope="session")
def core(keypoint_cfg: KeypointConfig, paths: PathsConfig) -> PoseCore:
    _pose_files(keypoint_cfg, paths)
    return PoseCore(keypoint_cfg, paths.assets_root)


@pytest.fixture(scope="session")
def clip_path() -> Callable[[str], Path]:
    folder = Path(os.environ.get("S2E_FIXTURE_CLIPS", REPO_ROOT / "tests" / "fixtures" / "clips"))

    def find(stem: str) -> Path:
        for suffix in CLIP_SUFFIXES:
            candidate = folder / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
        return need(folder / f"{stem}.mov", f"fixture clip `{stem}` (git-ignored, see docs)")

    return find


@pytest.fixture(scope="session")
def raw_of(core: PoseCore, clip_path: Callable[[str], Path]) -> Callable[[str], RawKeypoints]:
    """File-path extraction of a fixture clip, done once per session."""
    cache: dict[str, RawKeypoints] = {}

    def get(stem: str) -> RawKeypoints:
        if stem not in cache:
            cache[stem] = extract_file(clip_path(stem), core)
        return cache[stem]

    return get


@pytest.fixture(scope="session")
def uni_sign_dir(paths: PathsConfig) -> Path:
    root = paths.third_party_root / "Uni-Sign"
    need(root / "datasets.py", "Uni-Sign reference checkout (asset A3)")
    return root


@pytest.fixture(scope="session")
def run_reference() -> Callable[..., None]:
    """Run one `reference_cli.py` subcommand in a separate process."""

    def run(*args: str | Path) -> None:
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        done = subprocess.run(
            [sys.executable, str(REFERENCE_CLI), *map(str, args)],
            capture_output=True,
            text=True,
            env=env,
            timeout=540,
            check=False,
        )
        if done.returncode != 0:
            pytest.fail(
                f"reference_cli {args[0]} failed ({done.returncode}):\n{done.stderr[-3000:]}",
                pytrace=False,
            )

    return run


@pytest.fixture(scope="session")
def reference_of(
    keypoint_cfg: KeypointConfig,
    paths: PathsConfig,
    uni_sign_dir: Path,
    clip_path: Callable[[str], Path],
    run_reference: Callable[..., None],
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[str], dict[str, Path]]:
    """Reference pipeline output for a clip: its extraction (bundled rtmlib 0.0.13 on
    OpenCV-decoded frames) and its normalization of that. Run once per session."""
    out_dir = tmp_path_factory.mktemp("reference")
    cache: dict[str, dict[str, Path]] = {}

    def get(stem: str) -> dict[str, Path]:
        if stem not in cache:
            det, pose = _pose_files(keypoint_cfg, paths)
            extracted = out_dir / f"{stem}_extract.npz"
            inputs = out_dir / f"{stem}_inputs.npz"
            run_reference("extract", uni_sign_dir, clip_path(stem), det, pose, extracted)
            run_reference("normalize", uni_sign_dir, extracted, inputs)
            cache[stem] = {"extract": extracted, "inputs": inputs}
        return cache[stem]

    return get


def write_video(
    path: Path,
    frames: list[np.ndarray],
    pts_ms: list[int],
    codec: str = "mpeg4",
) -> Path:
    """Write BGR frames with explicit presentation times in milliseconds. Synthetic test
    input only; `mpeg4` in Matroska keeps a 1 ms time base, so any timing can be written."""
    assert len(frames) == len(pts_ms)
    height, width = frames[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=30)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        for image, t in zip(frames, pts_ms, strict=True):
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = t
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def ffprobe_times_ms(path: Path) -> list[float]:
    """Presentation time of every video frame according to ffprobe, in milliseconds."""
    ffprobe = need_tool("ffprobe")
    done = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "frame=best_effort_timestamp_time", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=120,
    )  # fmt: skip
    return [
        float(f["best_effort_timestamp_time"]) * 1000 for f in json.loads(done.stdout)["frames"]
    ]


# Helper functions are handed to tests as fixtures: with pytest's importlib import mode, test
# modules cannot import from this file directly.
@pytest.fixture(scope="session")
def video_writer() -> Callable[..., Path]:
    return write_video


@pytest.fixture(scope="session")
def ffprobe_ms() -> Callable[[Path], list[float]]:
    return ffprobe_times_ms


@pytest.fixture(scope="session")
def need_file() -> Callable[[Path, str], Path]:
    return need
