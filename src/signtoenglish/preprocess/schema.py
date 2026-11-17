"""Data shapes and settings for preprocessing.

Everything here is dictated by the base model (Uni-Sign pose-only); the file and line
references are in `docs/base_model_notes.md`. Nothing in this module does any work.

Keypoints are the 133-point COCO-WholeBody layout, one person per frame. The base model
uses 69 of them in four groups (`PARTS`), and takes each group as `[T, V, 3]` arrays of
`(x, y, score)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from signtoenglish.common.config import FrozenConfig

#: Bump by hand whenever extraction output could change (ARCHITECTURE.md, section 3).
EXTRACTOR_VERSION = 1

NUM_KEYPOINTS = 133

PartName = Literal["body", "left", "right", "face_all"]

#: Group order used by the base model (`models.py:77`).
PARTS: tuple[PartName, ...] = ("body", "left", "right", "face_all")

#: Indices into the 133 keypoints for each group (`datasets.py:31-45`).
PART_INDICES: dict[PartName, tuple[int, ...]] = {
    "body": (0, 3, 4, 5, 6, 7, 8, 9, 10),
    "left": tuple(range(91, 112)),
    "right": tuple(range(112, 133)),
    "face_all": (*range(23, 40, 2), *range(83, 91), 53),
}

#: Position inside each group of the point the group is measured from. The body group is
#: box-normalized and has none. Hands use their first point, the face its last (point 53).
PART_REFERENCE: dict[PartName, int | None] = {
    "body": None,
    "left": 0,
    "right": 0,
    "face_all": 17,
}

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]
ImageArray = npt.NDArray[np.uint8]

#: Model inputs: one float32 array `[T, V, 3]` per group, in `PARTS` order.
ModelInputs = dict[PartName, FloatArray]
#: For each group, a bool array `[T, V]`: True where the keypoint reached the model as a
#: value, False where the base model's rule replaced it by `(0, 0, 0)`.
PartMask = dict[PartName, BoolArray]


@dataclass(frozen=True, slots=True)
class Frame:
    """One decoded video frame. `image` is `[H, W, 3]` uint8 in BGR order, which is what
    the pose estimator was fed in the reference pipeline (`demo/pose_extraction.py:25-36`).
    `t_ms` is the capture or container timestamp; it is never derived from a frame rate."""

    image: ImageArray
    t_ms: int


@dataclass(frozen=True, slots=True)
class FrameKeypoints:
    """Raw pose-estimator output for one frame, in pixels of the frame it was run on."""

    t_ms: int
    keypoints: FloatArray  # [133, 2]
    scores: FloatArray  # [133]
    person_found: bool


@dataclass(frozen=True, slots=True)
class RawKeypoints:
    """Raw keypoints for a run of frames: the in-memory form of a cache entry
    (ARCHITECTURE.md, section 4.4). Values are untouched estimator output."""

    timestamps_ms: IntArray  # [T]
    keypoints: FloatArray  # [T, 133, 2], pixels
    scores: FloatArray  # [T, 133]
    person_found: BoolArray  # [T]
    width: int
    height: int

    def __post_init__(self) -> None:
        t = self.timestamps_ms.shape[0]
        expected = {
            "timestamps_ms": ((t,), np.int64),
            "keypoints": ((t, NUM_KEYPOINTS, 2), np.float32),
            "scores": ((t, NUM_KEYPOINTS), np.float32),
            "person_found": ((t,), np.bool_),
        }
        for name, (shape, dtype) in expected.items():
            arr = getattr(self, name)
            if arr.shape != shape or arr.dtype != dtype:
                raise ValueError(
                    f"{name}: expected shape {shape} and dtype {np.dtype(dtype)}, "
                    f"got {arr.shape} and {arr.dtype}"
                )
        if t > 1 and not bool(np.all(np.diff(self.timestamps_ms) > 0)):
            raise ValueError("timestamps_ms must be strictly increasing")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"bad frame size {self.width}x{self.height}")

    def __len__(self) -> int:
        return int(self.timestamps_ms.shape[0])

    @property
    def frame_size(self) -> tuple[int, int]:
        """`(width, height)`, the order `normalize_keypoints` takes."""
        return (self.width, self.height)

    @staticmethod
    def from_frames(frames: list[FrameKeypoints], width: int, height: int) -> RawKeypoints:
        n = len(frames)
        return RawKeypoints(
            timestamps_ms=np.array([f.t_ms for f in frames], dtype=np.int64),
            keypoints=np.stack([f.keypoints for f in frames]).astype(np.float32)
            if n
            else np.zeros((0, NUM_KEYPOINTS, 2), np.float32),
            scores=np.stack([f.scores for f in frames]).astype(np.float32)
            if n
            else np.zeros((0, NUM_KEYPOINTS), np.float32),
            person_found=np.array([f.person_found for f in frames], dtype=np.bool_),
            width=width,
            height=height,
        )


class PoseModelFile(FrozenConfig):
    """One ONNX file under the assets root, pinned by content hash."""

    relative_path: str
    sha256: str
    input_size: tuple[int, int]  # (width, height), as rtmlib takes it


class KeypointConfig(FrozenConfig):
    """Extraction settings. Every field affects the output, so every field belongs in the
    preprocessing hash. Defaults are the base model's demo settings: rtmlib "lightweight"
    mode (`demo/online_inference.py:72-77`), approved as assets A6.

    `model_rate_fps`: frames arriving faster than this are thinned by timestamp before pose
    estimation. The base model has no frame rate of its own; it used every frame of its
    training videos. `None` keeps every frame.

    `pose_provider`: the detector always runs on CPU (CoreML cannot run it). CoreML for the
    pose model is faster but gives slightly different keypoints, so it is a different
    preprocessing hash and must not be mixed with CPU output (`docs/feasibility.md`, 6.4).
    """

    detector: PoseModelFile = PoseModelFile(
        relative_path="rtmlib/lightweight/yolox_tiny/yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx",
        sha256="ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250",
        input_size=(416, 416),
    )
    pose: PoseModelFile = PoseModelFile(
        relative_path="rtmlib/lightweight/rtmw_l_m/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx",
        sha256="68db326c5c6f3273456f803eae0faf5440478369f2c3ee67258d2f36512e8d76",
        input_size=(192, 256),
    )
    pose_provider: Literal["cpu", "coreml"] = "cpu"
    model_rate_fps: float | None = 30.0
    #: A frame is kept if it is no more than this fraction of a frame period early.
    rate_slack: float = 0.25


class NormalizeConfig(FrozenConfig):
    """Settings of the base model's normalization. They are the pretrained model's training
    values, not tuning knobs: changing one breaks parity with the checkpoint."""

    #: A keypoint with score <= this is replaced by (0, 0, 0) (`datasets.py:19,71,104`).
    score_threshold: float = 0.3
    #: Fewer valid body coordinates than this in the whole clip zeroes everything
    #: (`datasets.py:87`).
    min_valid_body_points: int = 4
