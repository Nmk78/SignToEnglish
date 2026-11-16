"""Helpers for the data tests: real, tiny, generated video files and manifest rows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from signtoenglish.common.hashing import sha256_file
from signtoenglish.data.manifest import ClipRow, ContinuousRow, Segment

# Uneven on purpose: real cameras do not deliver evenly spaced frames.
DEFAULT_TIMES_MS = (0, 33, 67, 100, 141, 166, 233, 250, 301, 333, 400, 433, 467, 520, 600, 640)

MakeVideo = Callable[..., str]


def write_video(path: Path, times_ms: Sequence[int], seed: int = 0) -> None:
    """Encode one small generated frame per timestamp, with exactly those timestamps."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("libx264", options={"preset": "ultrafast", "bf": "0"})
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        for t in times_ms:
            image = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = t
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def make_video(data_root: Path) -> MakeVideo:
    """make_video(rel_path, times_ms=..., seed=...) -> SHA-256 of the file it wrote."""
    counter = iter(range(1, 10_000))

    def make(
        rel_path: str, times_ms: Sequence[int] = DEFAULT_TIMES_MS, seed: int | None = None
    ) -> str:
        path = data_root / rel_path
        write_video(path, times_ms, seed if seed is not None else next(counter))
        return sha256_file(path)

    return make


def clip_row(source_path: str, sha: str, **overrides: object) -> ClipRow:
    fields: dict[str, object] = {
        "id": Path(source_path).stem,
        "dataset": "domain",
        "split": "unassigned",
        "source_path": source_path,
        "source_sha256": sha,
        "start_ms": None,
        "end_ms": None,
        "text": "Hello, how are you?",
        "sentence_id": "t01",
        "signer_id": "s1",
        "session_id": "2026-11-16_desk",
        "take": 1,
        "sign_language": "ase",
        "text_language": "en",
        "license": "own-recording",
        "tags": (),
    }
    fields.update(overrides)
    return ClipRow(**fields)  # type: ignore[arg-type]


def continuous_row(source_path: str, sha: str, **overrides: object) -> ContinuousRow:
    fields: dict[str, object] = {
        "id": Path(source_path).stem,
        "dataset": "domain",
        "split": "unassigned",
        "source_path": source_path,
        "source_sha256": sha,
        "signer_id": "s1",
        "session_id": "2026-11-16_desk",
        "take": 1,
        "sign_language": "ase",
        "text_language": "en",
        "license": "own-recording",
        "tags": (),
        "segments": (
            Segment(sentence_id="t01", text="Hello, how are you?", start_ms=0, end_ms=250),
            Segment(sentence_id="t05", text="The water is cold.", start_ms=250, end_ms=600),
        ),
    }
    fields.update(overrides)
    return ContinuousRow(**fields)  # type: ignore[arg-type]


@pytest.fixture
def make_clip_row() -> Callable[..., ClipRow]:
    return clip_row


@pytest.fixture
def make_continuous_row() -> Callable[..., ContinuousRow]:
    return continuous_row
