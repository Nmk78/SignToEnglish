# CLAUDE.md — SignToEnglish

Durable rules for every session in this repo. Read `STATUS.md` first for where the project stands and what is waiting on whom. Read `PLAN.md` for what to build next, `ARCHITECTURE.md` for how it is laid out, `OPEN_QUESTIONS.md` for decisions and their status, and `REUSE.md` before touching the old prototype.

## What this project is

A continuous ASL-to-English translation system: camera in, English sentences out, gloss-free and sentence-level. It is not an isolated-sign classifier. If a change turns the system into "classify signs, then stitch words together", it is out of scope.

Constraints set by the user on 2026-11-16:

- It is a **project-show demo**. It must work well inside one conversation domain with the demo signer; it does not have to work elsewhere, and it must not claim to.
- It starts from an **existing pretrained sign language translation model** and adapts it. Nothing is trained from scratch.
- Everything runs and trains on **this Mac** (Apple M5, 16 GB, no NVIDIA GPU). No rented or cloud compute.
- Domain data is **recorded for this project** with `s2e record`, by the demo signer (user decision 2026-11-17: the user does not record; other training data may be downloaded, each dataset approved by name, with a heads-up to the user before any download over 50 MB). That tool collects video and English sentences for sequence-to-sequence fine-tuning. It must not become a closed-set phrase classifier: model output stays free text, and evaluation always includes sentences never seen in training.

## Status gate

Check the status line at the top of `PLAN.md`. `READY` means implementation may proceed step by step from where the plan stands; it never waives a stop point below. If it says `DRAFT`, do not write code; ask the user about the open questions in `OPEN_QUESTIONS.md`.

## Stop points: ask the user first, every time

These need explicit approval in the current session. Approval given in an earlier session, or for a different item, does not carry over.

1. **Datasets.** Do not download, scrape, or request access to any dataset, in whole or in part, including "just a few samples" and including pre-extracted features or keypoints published by a dataset's authors.
2. **Pretrained weights and model files.** Do not download any weights or model asset that is not already listed as approved in `OPEN_QUESTIONS.md` (section "Approved assets"). This includes anything `transformers` would fetch implicitly; set `HF_HUB_OFFLINE=1` by default in tests and scripts and pass local paths.
3. **Training code.** Before writing or changing any training code (anything under `src/signtoenglish/train/`, a training config, a loss, or an optimizer setup), state the design in a few lines and wait for the user to confirm. Model definitions, datasets, and evaluation code do not need this confirmation, but the first training run of any new configuration does.
4. **Long or paid jobs.** Do not start a job expected to run longer than 10 minutes, and do not use anything that costs money or runs off this machine. Prepare the command, give a time estimate based on a measured small run, and let the user launch it.
5. **Test-set scoring.** The test split is scored once per final checkpoint. Ask before running it.
6. **Installing system tools** (for example `uv` or `ffmpeg`, neither of which was installed on this machine on 2026-11-16).

## License rules

- The pretrained base model's weights are CC BY-NC 4.0 and every public ASL dataset checked is non-commercial (see `OPEN_QUESTIONS.md`, Q2 and Q3). Until the user says otherwise, this is a **personal, non-commercial demo**. Do not add anything that implies commercial use: no hosted public demo, no paid API wrapper, no redistribution of base or adapted weights.
- The base model's code repository has no license. Keep it in the git-ignored `third_party/` folder at a pinned commit. Never copy its code into `src/`, never commit it, and never publish it. Only `models/translator.py` may import it.
- Never redistribute dataset videos, frames, or text. Never commit them, and never put dataset frames in docs, test fixtures, or screenshots. Derived caches stay local and are git-ignored.
- Recordings made for this project belong to the people in them. Keep them local and git-ignored, and do not use them outside this project without the signer's consent.
- Credit the base model, the pose estimator, and the text model in the README and on the demo page.
- Do not use weights from the old prototype. Its bundled classifier was CC BY-NC-SA 4.0, and all old checkpoints are excluded by `REUSE.md`.
- Before adding any pretrained model, dataset, or code copied from another repo, record its license, the URL where the license was read, and the date in the `Verification log` section of `OPEN_QUESTIONS.md`. If the license cannot be read from a primary source, say so there and ask the user.
- Code license and weights license can differ for the same project. Check both.
- Models trained here inherit the restrictions of their training data and of any non-commercial pretrained component. Write the resulting terms into each run's `model_card.md`.
- Do not copy `OLD/backend/.env` or any secret. No API keys belong in this repo; the system makes no external API calls at inference time.

## Stack

| Layer | Choice |
| --- | --- |
| Python | One minor version, pinned in `.python-version` and `pyproject.toml` (plan default: 3.12). The system Python is 3.14 and must not be used directly. |
| Environment | `uv`, with `pyproject.toml` and a committed `uv.lock`. |
| Base model | Uni-Sign pose-only translation checkpoint (pending Q2), adapted with low-rank adapters. |
| Keypoints | RTMPose whole-body through `rtmlib` on ONNX Runtime, because the base model was trained on it. MediaPipe is not used. |
| ML | PyTorch and Hugging Face `transformers`, 32-bit floats. Device order: MPS, then CPU. No CUDA-only code. |
| Text model | mT5-Base, as required by the base checkpoint. |
| Metrics | `sacrebleu` for BLEU, `rouge-score` for ROUGE-L. Always record the sacreBLEU signature. |
| Video I/O | PyAV or OpenCV for decoding, with real per-frame timestamps. `ffmpeg` for inspection only. |
| Service | FastAPI and uvicorn: REST for clips, WebSocket for streaming. |
| Config | YAML files validated by pydantic models. No Hydra, no scattered `os.getenv`. |
| Web | Next.js, React, TypeScript under `web/` (pending Q5). |
| Tests | `pytest`. Lint and format with `ruff`. Types with `pyright` or `mypy` on `src/`. |

This machine is an Apple M5 with 16 GB of unified memory and no NVIDIA GPU. It is the only machine. Code must run on MPS and CPU, and the demo must work without an internet connection.

## Pinned-version policy

- Python dependencies are exact pins (`==`) in `pyproject.toml`, resolved into `uv.lock`. No `>=`, `~=`, or unpinned entries.
- Node dependencies are exact versions in `web/package.json` (no `^` or `~`) with a committed `package-lock.json`. Install with `npm ci`.
- Model files and pretrained weights are pinned by content hash (SHA-256) and, for Hugging Face models, by revision id. The loader checks the hash and refuses a mismatch.
- A dependency bump is its own commit, with the reason in the message, and must re-run the full test suite. Bumping `rtmlib` or `onnxruntime`, or changing a pose model file, invalidates the keypoint cache by design.
- Do not pick versions from memory. Read the current versions from the package index at the time of pinning. Versions seen on 2026-11-16 are listed in `OPEN_QUESTIONS.md` as candidates only.

## Engineering rules

- **One preprocessing path.** Extraction lives in `preprocess/extract.py` and normalization in `preprocess/normalize.py`. Training, offline inference, and streaming all call these. Never write a second copy, and never normalize anywhere else.
- **Preprocessing must match the base model.** The normalization reproduces what the pretrained model saw in training, and a parity test against the reference pipeline guards it. Do not "improve" the normalization without that test passing.
- **Missing keypoints are handled explicitly** through the mask and the base model's documented rule, never by transforming placeholder zeros.
- **Real timestamps.** Never assume a frame rate. Frames carry their capture or container timestamp end to end.
- **Checkpoints know their preprocessing.** Each checkpoint stores the preprocessing config hash; inference refuses to run when the runtime config has a different hash.
- **Import direction:** `serve` → `infer` → `models`, `preprocess`. Nothing under `infer` or `serve` imports from `train`. No `sys.path` edits and no script importing another script.
- **Split discipline.** Choose epochs, checkpoints, decoding settings, and thresholds on validation. Never tune on test. No recording appears in two splits, and the text of an unseen-test sentence never appears in train or validation. Report the seen-sentence and unseen-sentence tests separately.
- **Best and last checkpoints are separate files.**
- **Tests must exercise the real thing:** real pose extraction on a real video and a real model forward pass and decode. A stub, mock, or demo mode does not satisfy a phase gate.
- **No LLM in the pipeline.** Decoder output is the system output unless Q6 is changed.
- **Not reintroduced:** pause-based sign splitting, isolated-sign classification, closed-set phrase classification, gloss-to-English rewriting. The sentence-end detector in Phase 3 detects a rest position between whole sentences and is scored against marked boundaries; it must not be used to cut individual signs.
- **Repo hygiene.** `.gitignore` covers `__pycache__/`, `*.pyc`, `.venv/`, `data/`, `cache/`, `runs/`, `assets/`, `third_party/`, `scratch/`, `*.pt`, `*.pth`, `*.onnx`, `*.safetensors`, `.env*`, `node_modules/`, `.next/`. No checkpoints, caches, or videos in git.

## Reporting rules

- In plans and write-ups, label each factual claim **verified** (read from a primary source, with URL and date), **inferred** (reasoned from verified facts), or **assumed** (not checked).
- Never invent a citation, benchmark number, or license. If it cannot be checked, say so.
- Report metrics with the split, the checkpoint id, the sacreBLEU signature, and the hardware. Report failed gates as failures, with the numbers.
- Time and cost estimates in `PLAN.md` are estimates until replaced by a measurement; when you measure one, update the file and change its label.

## Parallel sessions

When several sessions work at once, `WORKERS.md` is binding: one git worktree and branch per track, edit only the paths your track owns, and never edit `pyproject.toml`, `uv.lock`, `.gitignore`, or the planning files from a worker track. Only one session at a time may load the model or run pose extraction on this 16 GB machine.

## One session, one phase step

Each numbered step in `PLAN.md` is sized for one session and ends at a test gate. Do the step, run its acceptance tests, report the result, and stop. Do not start the next step in the same session without the user asking.
