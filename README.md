# SignToEnglish

A continuous ASL-to-English translation demo: camera in, English sentences out. It adapts an
existing pretrained sign language translation model to one small conversation domain and one
signer, and runs entirely on one Mac.

**Status:** scaffold only (Step 0.1 of `PLAN.md`). There is no model, no data, and no demo yet.
This file is completed in Step 4.2 with setup, measured results, and limits.

**Scope and terms:** a personal, non-commercial project-show demo. It is meant to work inside
the recorded domain with the demo signer and makes no claim outside it.

## Setup

Requires macOS on Apple Silicon, [`uv`](https://docs.astral.sh/uv/) 0.13.8, and `ffmpeg`.

```sh
uv sync --frozen     # creates .venv with Python 3.12 and the exact locked versions
make check           # ruff, mypy, pytest
make doctor          # Python, package versions, device, free disk, ffmpeg
```

Nothing is downloaded from a model hub: `HF_HUB_OFFLINE=1` is the default everywhere.

## Layout

See `ARCHITECTURE.md`. Planning and rules: `PLAN.md`, `CLAUDE.md`, `OPEN_QUESTIONS.md`,
`REUSE.md`, `WORKERS.md`.

## Credits

This project builds on the work below. No weights, model files, or third-party code are
included in this repository.

- **Base model:** Uni-Sign (pose-only translation checkpoints),
  <https://github.com/ZechengLi19/Uni-Sign>, paper <https://arxiv.org/abs/2501.15187>. Weights are tagged CC BY-NC 4.0 on the Hugging
  Face hub; the code repository declares no license.
- **Pose estimator:** RTMPose whole-body, run through `rtmlib` on ONNX Runtime.
- **Text model:** mT5-Base (`google/mt5-base`).

The licenses of `rtmlib`, the RTMPose model files, and mT5-Base have not been verified yet;
Step 0.2 reads them from primary sources and records them in `OPEN_QUESTIONS.md` and
`docs/DATA_LICENSES.md`.
