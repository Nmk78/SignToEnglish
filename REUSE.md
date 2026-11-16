# REUSE.md — what to carry over from the old prototype

Old repo location (verified on disk, 2026-11-16):
`/Users/naymyokhant/Programming/Personal/Projects/SignLanguageTranslator-old`
(called `OLD/` below). It was moved from `~/Programming/Personal/SignLanguageTranslator`.

Every row in this file was checked by reading the old file in this planning session, unless it says otherwise. Line numbers refer to the old files as they are on disk today.

## Rules for copying

1. Copy code by rewriting it into the new module layout in `ARCHITECTURE.md`. Do not copy a file wholesale and patch it, because every file below mixes reusable logic with logic that is banned.
2. Never copy `OLD/backend/.env`. It contains live API keys for four LLM providers and a stale `ASL_PHRASE_MODEL_PATH`. It is not tracked by git there, and it must not enter this repo in any form.
3. Never copy anything under `OLD/backend/models/` or `OLD/backend/data/`.
4. Do not copy `.pyc` files, `node_modules/`, `.next/`, or `OLD/.git` (3.7 GB, mostly checkpoints).

## Files to reuse

### 1. Video extraction loop (structure only)

- **Old path:** `OLD/backend/app/recognizer.py`, `_extract_sequence` (lines 152–189). A second, diverged copy is in `OLD/backend/app/dataset_recognizer.py`, `_extract_positions` (lines 133–165).
- **New home:** `src/signtoenglish/preprocess/video_io.py` and `src/signtoenglish/preprocess/extract.py` (the only extraction module).
- **Why reuse:** the open, read, convert-color, run-estimator, collect loop and its "no readable frames" and "nothing detected" checks are correct.
- **Changes required:**
  - **The pose estimator changes.** Revision 2 of the plan uses RTMPose whole-body through `rtmlib`, because the pretrained base model was trained on it. The MediaPipe `HolisticLandmarker` calls are not carried over; only the loop structure is.
  - Replace the fake timestamp `frame_index * 33` (assumes 30 fps) with the real presentation timestamp of each frame, read from the container. Timestamps must be strictly increasing; drop duplicates and count them in the output metadata.
  - Return raw keypoints and confidence scores. Do not zero-fill and do not normalize inside extraction.
  - Take model paths from config, not from a module-level constant.
  - Expose two entry points over one core function: one for files and one for live frames. Offline and streaming must call the same per-frame code.
  - Merge the two old copies into one. Delete nothing in the old repo.

### 2. Shoulder-based normalization (the idea and its test, not the behavior)

- **Old path:** `OLD/backend/app/recognizer.py`, `_normalize_landmarks` (lines 126–150); test at `OLD/backend/tests/test_recognizer_contract.py`, `test_runtime_landmarks_match_the_checkpoint_body_relative_format` (lines 20–32).
- **New home:** `src/signtoenglish/preprocess/normalize.py` and `tests/preprocess/test_normalize.py`.
- **Why reuse:** the old test's style is the useful part: a worked numeric example (shifted subject, doubled shoulder width) that pins down the transform. The lesson from the old code also carries over.
- **Changes required:**
  - The transform itself is now dictated by the pretrained base model and is read from its code in Step 0.2. Do not carry over the shoulder-based formula unless that is what the base model uses.
  - This becomes the single normalization function. Training and inference both import it; nothing else may normalize keypoints.
  - Missing parts get an explicit mask channel. A missing hand is never run through the transform. The old inference path normalized zero-filled hands (so a missing hand became a nonzero offset), while the old training path multiplied by the validity mask and kept zeros (`OLD/backend/scripts/train_asl_citizen_model.py` lines 45–50). That mismatch is the audit finding this function exists to remove.
  - When shoulders are missing, mark the whole frame invalid through the mask. The old code silently returned the un-normalized frame (`dataset_recognizer.py` lines 124–130).

### 3. Disk cache keyed by source and config

- **Old path:** `OLD/backend/scripts/train_asl_citizen_model.py`, `cache_matches` (lines 54–62) and `prepare_cache` (lines 65 onward), plus `CachedDataset` (lines 136–155).
- **New home:** `src/signtoenglish/preprocess/cache.py` and `src/signtoenglish/data/dataset.py`.
- **Why reuse:** the pattern is right. A metadata file decides whether the cache is valid, features are written once, and the dataset memory-maps them.
- **Changes required:**
  - Key on a content hash of the source video and a hash of the extraction config, model file, and extractor version. The old key was file size, mtime, and one parameter (`frame_count`), which goes stale when code changes and breaks when a file is copied.
  - Store one cache file per source video with variable length, not one fixed-shape array per split. The old cache resampled every clip to 128 frames; sentence-level translation needs real durations.
  - Keep features in the cache. The old cache directory (`OLD/backend/data/asl-citizen-partial-cache/`) now holds only targets, indices, and metadata because the feature files were deleted, so the new cache needs an integrity check command that reports missing files.
  - Schema is defined in `ARCHITECTURE.md`.

### 4. Training resume and checkpoint logic

- **Old path:** `OLD/backend/scripts/train_dataset_model.py`, resume block (lines 209–232), per-epoch save (lines 284–304), finalize block (lines 310–329), `select_device` (line 32).
- **New home:** `src/signtoenglish/train/checkpoint.py`, `src/signtoenglish/train/loop.py`, `src/signtoenglish/common/device.py`.
- **Why reuse:** it saves optimizer and scheduler state, refuses to resume when the config differs, and has a separate finalize step. `select_device` already handles MPS with a CPU fallback, which matches this machine.
- **Changes required:**
  - Write two separate files: `last.pt` (model, optimizer, scheduler, step, RNG state, history) and `best.pt` (model weights and the validation metric that selected it). The old code stored the best weights inside the progress checkpoint and then overwrote the same path with the final model, so a finished run could not be resumed and the last-epoch weights were lost.
  - Select the best checkpoint on validation BLEU, not accuracy.
  - Write checkpoints atomically (temp file, then rename).
  - Save by optimizer step as well as by epoch, because epochs will be long.
  - Embed the resolved training config and the preprocessing config hash in every checkpoint.

### 5. Split discipline

- **Old path:** `OLD/backend/scripts/train_dataset_model.py`, finalize block (lines 310–329): temperature and threshold are chosen on validation, then test is scored once and written to a report.
- **New home:** `src/signtoenglish/eval/` and the rules in `CLAUDE.md`.
- **Why reuse:** this is the correct order of operations.
- **Changes required:** replace accuracy, temperature, and threshold with BLEU-4, ROUGE-L, and decoding settings (beam size, length penalty, maximum length) chosen on validation. Make the test-scoring command refuse to run twice for the same checkpoint unless forced, and record every test run in a log file.

### 6. FastAPI service shape and `/health`

- **Old path:** `OLD/backend/app/main.py`: app and CORS setup (lines 17–19), `/health` (lines 47–53), upload handling in `/translate` (lines 56–64, 96–97).
- **New home:** `src/signtoenglish/serve/app.py`.
- **Why reuse:** `/health` reports "configuration-needed" with a reason instead of crashing, which is the right pattern. The upload handler sanitizes the filename and always deletes the temp file.
- **Changes required:**
  - `/health` must load the model and report the checkpoint id, the preprocessing config hash, and the device. The audit found `/translate` returning 503 because of a path that did not exist while the service still started; `/health` must fail loudly for that case and a test must cover it.
  - `/translate` returns `{text, segments, timings, model}`. Remove gloss, confidence gate, alternatives, and conversation fields.
  - Add a WebSocket endpoint for streaming (protocol in `ARCHITECTURE.md`).
  - Remove `/enroll` (lines 100–121), `ConversationStore`, and `GlossTranslator`.
  - Read settings through one typed settings object, not scattered `os.getenv` calls.

### 7. Next.js demo shell

- **Old paths:** `OLD/src/app/page.tsx`, `OLD/src/app/layout.tsx`, `OLD/src/app/globals.css`, `OLD/package.json`, `OLD/package-lock.json`, `OLD/tsconfig.json`, `OLD/next.config.ts`, `OLD/postcss.config.mjs`, `OLD/next-env.d.ts`.
- **New home:** `web/` (same file names under `web/src/app/`).
- **Why reuse:** the camera permission flow, the upload control, the two-panel layout, and the styling work and are worth keeping. Whether to keep Next.js at all is question Q5 in `OPEN_QUESTIONS.md`; this row applies if the answer is "keep".
- **Changes required:**
  - `page.tsx`: keep `getUserMedia` setup (lines 57–65) and the upload path (lines 95–107). Replace `MediaRecorder` record-then-upload with a live loop that grabs frames from the video element and sends them over a WebSocket with capture timestamps. Add a caption area that shows committed text and the current unstable text differently.
  - Remove the enrollment panel and `enrollPhrase` (lines 141–162 and the "TEACH A COMPLETE PHRASE" block), the gloss/confidence/alternatives display, the fake `setTimeout` stage delays (lines 121 and 135), and the conversation history copy about context expiry.
  - Split the single 219-line component into a camera component, a caption component, and a connection hook.
  - `package.json`: pin exact versions (remove every `^`), rename the package, and review the `"license": "MIT"` field against Q3.
  - `layout.tsx`: change the title and description; they say "SignBridge" and "short ASL video segments".

### 8. Dev launcher script

- **Old path:** `OLD/scripts/dev-fresh.sh`.
- **New home:** `scripts/dev.sh`.
- **Why reuse:** it starts both services and cleans up both on Ctrl+C.
- **Changes required:** use the `uv` environment instead of `backend/.venv`, stop sourcing `.env` into the shell, and remove the `__pycache__` deletion (it existed because `.pyc` files were committed).

### 9. MediaPipe Holistic model file — no longer reused

- **Old path:** `OLD/backend/models/mediapipe/holistic_landmarker.task` (13,683,609 bytes, SHA-256 `e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8`, hash computed in the planning session).
- **Status:** not copied. Revision 2 of the plan uses RTMPose keypoints to match the pretrained base model, so MediaPipe is not a dependency. This entry stays only so the file can be found if the base model changes to one that needs MediaPipe (the SHuBERT fallback in `OPEN_QUESTIONS.md`, Q2, uses MediaPipe face and hand models, which are different files). Its license was not verified.

### 10. Tests worth porting

- **Old path:** `OLD/backend/tests/test_recognizer_contract.py`, lines 13–32 (landmark vector shape and normalization example). `OLD/backend/tests/test_api.py`, the client setup pattern (lines 15–36).
- **New home:** `tests/preprocess/`, `tests/serve/`.
- **Why reuse:** these two tests check real numeric behavior.
- **Changes required:** rewrite in pytest. Every other old test exercises the demo recognizer, the LLM providers, the conversation store, or enrollment, and is dropped. New tests must run real extraction on a real clip and a real (tiny) model; demo-mode tests do not count toward any gate.

## Do not reuse

| Old path | Reason |
| --- | --- |
| `OLD/backend/app/recognizer.py`, `_split_sign_segments` (lines 197–241) | Pause-based splitter. Over-segmented 46 of 60 single-sign clips in the audit. |
| `OLD/backend/app/recognizer.py`, `ASLCitizenRecognizer`, `build_asl_citizen_network`, `TorchScriptASLRecognizer`, `DemoASLRecognizer`, `create_recognizer` | Isolated-sign classifier, an adapter that always raises, a filename-matching demo mode, and an env-var chain that produced the 503. |
| `OLD/backend/models/asl-citizen-bigru-200/`, `OLD/backend/models/asl-citizen-200/` | Third-party weights, CC BY-NC-SA 4.0 per the old README. Not to be reused. |
| `OLD/backend/models/asl-186-pose-hands/`, `OLD/backend/models/asl-citizen-partial-pose-hands/` | Old checkpoints from isolated-sign training (about 115 MB under `models/`). |
| `OLD/backend/app/dataset_recognizer.py`, `OLD/backend/scripts/train_dataset_model.py` model code, `OLD/backend/scripts/train_asl_citizen_model.py` model code | 186-class BiLSTM classifier. Only the resume, cache, and device patterns listed above are reused. `train_asl_citizen_model.py` imports from the other script through `sys.path` and redefines `evaluate()`. |
| `OLD/backend/app/phrase_recognizer.py`, `OLD/backend/scripts/train_phrase_model.py`, `/enroll` in `main.py`, `OLD/backend/data/phrase-enrollments/` | Phrase enrollment system. The enrollment folder also holds five personal recordings. |
| `OLD/backend/app/translator.py`, `OLD/backend/app/conversation.py` | LLM gloss rewriter with a six-entry fallback table, and its context store. |
| `OLD/backend/app/models.py` | `Recognition` and `SignRecognizer` are gloss-and-confidence types with no use in a translation system. |
| `OLD/backend/data/demo-clips/` (60 clips) | Isolated-sign clips. Their source dataset and license are not recorded in the old repo, so they must not be copied here, even as test fixtures. |
| `OLD/backend/data/asl-citizen-partial-cache/` | Metadata only; the feature files are gone. |
| `OLD/backend/requirements.txt` | Every entry is an unpinned `>=` range and it includes `openai` and `google-genai`. Use it only as a list of package names to consider. |
| `OLD/backend/.env`, `OLD/backend/.env.example` | Secrets, and settings for removed features. |
| `OLD/README.md` | Describes the removed system. |
| `OLD/sign-language-ai-research-notes.md` | Not code. Read once for background (it lists SSVP-SLT, How2Sign, OpenASL and notes an interest in Myanmar Sign Language); nothing in it was treated as verified. |
