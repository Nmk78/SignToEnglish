"""Repo hygiene from CLAUDE.md: what git must ignore, and what it must not."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

MUST_BE_IGNORED = [
    "data/domain/s001_take1.mp4",
    "data/manifests/domain_train.jsonl",
    "cache/keypoints/abc/de/f.npz",
    "runs/r1/best.pt",
    "assets/unisign/openasl_pose_only_slt.pth",
    "third_party/Uni-Sign/models.py",
    "scratch/spike.py",
    "src/signtoenglish/__pycache__/cli.cpython-312.pyc",
    "anything/weights.pt",
    "anything/weights.pth",
    "anything/pose.onnx",
    "anything/model.safetensors",
    ".venv/bin/python",
    ".env",
    ".env.local",
    "web/node_modules/react/index.js",
    "web/.next/build-manifest.json",
    "configs/paths.local.yaml",
    "tests/fixtures/clips/hands_visible.mov",
    "tests/fixtures/anything.mp4",
    "docs/screenshot_clip.webm",
]

MUST_NOT_BE_IGNORED = [
    "src/signtoenglish/data/manifest.py",
    "src/signtoenglish/data/__init__.py",
    "src/signtoenglish/preprocess/cache.py",
    "tests/data/test_manifest.py",
    "tests/fixtures/tiny_keypoints.npy",
    "domain/sentences.yaml",
    "configs/paths.example.yaml",
    "configs/preprocess/keypoints.yaml",
    "pyproject.toml",
    "uv.lock",
    "web/package-lock.json",
    "docs/results.md",
]


def _ignored(repo_root: Path, path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path], cwd=repo_root, check=False
    )
    assert result.returncode in (0, 1), f"git check-ignore failed for {path}"
    return result.returncode == 0


@pytest.mark.parametrize("path", MUST_BE_IGNORED)
def test_local_artifacts_are_ignored(repo_root: Path, path: str) -> None:
    assert _ignored(repo_root, path)


@pytest.mark.parametrize("path", MUST_NOT_BE_IGNORED)
def test_source_and_planning_files_are_not_ignored(repo_root: Path, path: str) -> None:
    assert not _ignored(repo_root, path)


def test_no_weights_caches_videos_or_secrets_are_tracked(repo_root: Path) -> None:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    banned_suffixes = (
        ".pt", ".pth", ".onnx", ".safetensors", ".npz", ".pyc",
        ".mp4", ".mov", ".webm", ".mkv", ".avi",
    )  # fmt: skip
    banned_roots = ("data/", "cache/", "runs/", "assets/", "third_party/", "scratch/")
    offenders = [
        f
        for f in tracked
        if f.endswith(banned_suffixes)
        or f.startswith(banned_roots)
        or Path(f).name.startswith(".env")
    ]
    assert offenders == []
