"""The one place keypoints are transformed into model inputs.

`normalize_keypoints` reproduces what the pretrained Uni-Sign model saw in training
(`load_part_kp` and `crop_scale`, `datasets.py:18-105` at the pinned commit; described in
`docs/base_model_notes.md`, section 2.3). It is an independent implementation of that
arithmetic, guarded by a parity test against the reference code. Do not "improve" it: a
change that breaks parity breaks the checkpoint.

What the base model's rule is, and is not:

- A keypoint whose score is at or below the threshold reaches the model as `(0, 0, 0)`.
  There is no interpolation and no separate presence input.
- The body group is mapped to [-1, 1] with ONE bounding box over the whole run of frames
  passed in. The same frame therefore normalizes differently inside a different segment.
  The live path calls this function on the open segment.
- Each hand is measured from its first point and the face from point 53, using that
  point's raw position even when its own score is below the threshold, and divided by the
  body box size. So a masked reference point still moves its group.
- `person_found` is not used. The reference never knew whether the detector found anyone;
  when it found no one, the pose model ran on the whole frame and the score rule did the
  rest. The argument is part of the contract so that a later, explicit decision to use it
  does not change every caller.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from signtoenglish.preprocess.schema import (
    NUM_KEYPOINTS,
    PART_INDICES,
    PART_REFERENCE,
    PARTS,
    BoolArray,
    ModelInputs,
    NormalizeConfig,
    PartMask,
)

_F64 = npt.NDArray[np.float64]


def normalize_keypoints(
    keypoints: npt.NDArray[np.floating],
    scores: npt.NDArray[np.floating],
    person_found: BoolArray,
    frame_size: tuple[int, int],
    cfg: NormalizeConfig,
) -> tuple[ModelInputs, PartMask]:
    """Turn raw keypoints into the base model's inputs.

    Args:
        keypoints: `[T, 133, 2]` pixel coordinates as the pose estimator returned them.
        scores: `[T, 133]` per-keypoint confidence.
        person_found: `[T]`; checked for shape only (see the module docstring).
        frame_size: `(width, height)` of the frames the keypoints were measured on.
        cfg: the base model's normalization settings.

    Returns:
        `(model_inputs, mask)`. `model_inputs[part]` is float32 `[T, V, 3]` of
        `(x, y, score)`; `mask[part]` is bool `[T, V]`, False exactly where the rule
        replaced a keypoint by zeros. Pure: the inputs are not modified, and the result
        depends only on the arguments.
    """
    t = keypoints.shape[0]
    if keypoints.shape != (t, NUM_KEYPOINTS, 2) or scores.shape != (t, NUM_KEYPOINTS):
        raise ValueError(
            f"expected keypoints [T, {NUM_KEYPOINTS}, 2] and scores [T, {NUM_KEYPOINTS}], "
            f"got {keypoints.shape} and {scores.shape}"
        )
    if person_found.shape != (t,):
        raise ValueError(f"expected person_found [T]={t}, got {person_found.shape}")
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError(f"bad frame size {frame_size}")

    # float64 throughout, as in the reference; scores are widened BEFORE the comparison so
    # that a float32 score just above the threshold compares the way it did in training.
    xy = keypoints.astype(np.float64) / np.array([width, height], dtype=np.float64)
    conf = scores.astype(np.float64)
    thr = float(cfg.score_threshold)

    inputs: ModelInputs = {}
    mask: PartMask = {}

    body_idx = list(PART_INDICES["body"])
    body, scale = _crop_scale(xy[:, body_idx], conf[:, body_idx], thr, cfg.min_valid_body_points)
    keep = (conf[:, body_idx] > thr) & (scale != 0.0)
    inputs["body"] = body.astype(np.float32)
    mask["body"] = keep

    for part in PARTS[1:]:
        idx = list(PART_INDICES[part])
        ref = PART_REFERENCE[part]
        assert ref is not None
        part_conf = conf[:, idx]
        out = np.zeros((t, len(idx), 3), dtype=np.float64)
        keep = np.zeros((t, len(idx)), dtype=np.bool_)
        if scale != 0.0:
            part_xy = xy[:, idx]
            out[..., :2] = (part_xy - part_xy[:, ref : ref + 1]) / scale
            out[..., 2] = part_conf
            out = np.clip(out, -1.0, 1.0)
            keep = part_conf > thr
            out[~keep] = 0.0
        inputs[part] = out.astype(np.float32)
        mask[part] = keep
    return inputs, mask


def _crop_scale(xy: _F64, conf: _F64, thr: float, min_valid: int) -> tuple[_F64, float]:
    """Body group: one box over all frames, longer side mapped to [-1, 1].

    Returns the `[T, V, 3]` result and the box size. Size 0 means "no usable body", in
    which case every group is all zeros.
    """
    out = np.zeros((*xy.shape[:2], 3), dtype=np.float64)
    valid = conf > thr
    if int(valid.sum()) < min_valid:
        return out, 0.0
    vx, vy = xy[..., 0][valid], xy[..., 1][valid]
    xmin, xmax, ymin, ymax = float(vx.min()), float(vx.max()), float(vy.min()), float(vy.max())
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0.0:
        return out, 0.0
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    out[..., :2] = ((xy - np.array([xs, ys])) / scale - 0.5) * 2
    out[..., 2] = conf
    out = np.clip(out, -1.0, 1.0)
    out[~valid] = 0.0
    return out, scale


def part_present(mask: PartMask, min_points: int = 1) -> dict[str, BoolArray]:
    """Per frame and group: did at least `min_points` keypoints reach the model?

    A reading aid for tests and for the sentence-end detector. It is not a model input.
    """
    return {part: m.sum(axis=1) >= min_points for part, m in mask.items()}
