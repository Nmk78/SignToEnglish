# PLAN.md — SignToEnglish

**Status: READY (revision 2, 2026-11-16).** The user answered all blocking questions on 2026-11-16; decisions are recorded in `OPEN_QUESTIONS.md`. READY means implementation may start at Step 0.1. It does not approve any download, install, training run, or test-set scoring; each **STOP** below still needs the user's approval in the session that reaches it.

Claim labels: **[verified]** read from a primary source on 2026-11-16 (sources in `OPEN_QUESTIONS.md`, "Verification log"); **[inferred]** reasoned from verified facts; **[assumed]** not checked. Every time figure is an estimate unless it says "measured".

## 1. What changed in revision 2

The user set four constraints on 2026-11-16, after revision 1:

1. No rented GPU. Everything runs and trains on this machine (Apple M5, 16 GB unified memory, no NVIDIA GPU, 100 GB free disk; measured).
2. Scope is limited to one conversation domain.
3. Start from an existing pretrained model built for sign language translation.
4. The goal is a project-show demo that works well inside that domain. It does not have to be good everywhere.

Revision 1 planned to train a keypoint-plus-video-feature model on How2Sign with a rented GPU. That is replaced. Revision 2 adapts a pretrained ASL translation model to a small domain using sentences recorded for this project.

## 2. Evidence the plan rests on

1. **Training from scratch on what fits this machine does not work.** Keypoints into T5 trained on How2Sign alone scores 1.22 BLEU; adding DINOv2 crop features gives 2.4. [verified] So a pretrained translation model is required, as the user proposed.
2. **Pretrained ASL translation models with downloadable weights exist, but few.** Of the candidates checked [verified unless noted]:
   - **Uni-Sign, pose-only.** Full translation checkpoints for How2Sign and for OpenASL (1.19 GB each) are on the Hugging Face hub, tagged CC BY-NC 4.0. The paper reports 14.5 BLEU on How2Sign for the pose-only model. Input is whole-body keypoints from RTMPose, run through `rtmlib` on ONNX Runtime; the text model is mT5-Base. A video-to-text script exists but hardcodes CUDA and bfloat16. The GitHub code repository has no license file.
   - **SHuBERT with ByT5.** Weights and a complete video-to-text app are public, tagged MIT. The pipeline has five model stages (YOLO, MediaPipe, two DINOv2 models, a fairseq encoder, ByT5), and the translation checkpoint's training data and scores are undocumented.
   - **SSVP-SLT.** No How2Sign or YouTube-ASL checkpoints are published. The only public weights were trained on one news signer and the stated requirements are Linux and CUDA.
   - Others found were unlicensed, unreleased, or self-reported at under 2 BLEU.
3. **No dataset covers everyday conversation.** How2Sign is entirely instructional how-to video in ten categories, none conversational. No verified small-talk, classroom, or workplace sentence set was found. [verified for How2Sign; the search for others was not exhaustive] Domain data has to be recorded.
4. **All of these models translate one pre-cut sentence clip at a time.** [verified] Continuous operation and sentence boundaries are this project's work.
5. **General-purpose quality is low.** The best published How2Sign score is 15.5 BLEU. [verified] A pretrained model used as-is on a new signer, a webcam, and conversational content will be worse than that. [inferred] The demo's quality has to come from narrowing the domain and adapting to it.

## 3. The approach

- **Base model:** Uni-Sign pose-only, starting from the OpenASL or the How2Sign checkpoint, whichever does better on the domain validation set. Fallback: the SHuBERT pipeline. (Q2)
- **Keypoints:** RTMPose whole-body through `rtmlib`, because that is what the base model was trained on. This replaces MediaPipe Holistic from the original brief. The extraction design rules are unchanged: one module, cached, real timestamps, one normalization function shared by training and inference, explicit handling of missing parts.
- **Domain data:** a written sentence list for one domain (default: introductions and small talk), about 100 sentences over a vocabulary of about 76 signs (drafted in `domain/sentences.yaml`, user-approved 2026-11-16, ASL versions still to be filled in), each recorded several times by the user, who is learning these sentences in ASL for the project. (Q1, Q10, Q11)
- **Adaptation:** parameter-efficient fine-tuning of the pretrained model on those recordings, on this Mac.
- **Honest measurement:** the test set has two parts, new recordings of sentences seen in training, and sentences never seen in training that use only vocabulary seen in training. The second part is what shows the system translates rather than memorizes. Both are reported.
- **Continuous mode:** a rolling keypoint buffer, re-decoded every second for partial text, with committed text at sentence boundaries.
- **Still out:** isolated-sign classification, the pause-based sign splitter, phrase enrollment with a closed-set classifier, LLM rewriting, old checkpoints. The model stays a sequence-to-sequence translator whose output is free text.

What the demo will and will not be: inside the recorded domain, with the demo signer, it should produce correct sentences most of the time. Outside the domain, or with a different signer, it will fall back toward the pretrained model's general quality, which is poor. The demo UI and README must say so.

## 4. Compute on this Mac

Rows marked **measured** were measured in Step 0.3 on 2026-11-17 on this Mac (Apple M5, 16 GB, 32-bit floats; torch 2.15.0, onnxruntime 1.32.0, rtmlib 0.0.16) on the two 150-frame, 1280×720 fixture clips; details in `docs/feasibility.md`. The remaining figures are **[assumed]** until Steps 2.2 and 2.4 measure them.

| Job | Estimate | Measured in |
| --- | --- | --- |
| RTMPose whole-body extraction, lightweight mode | **measured:** 20.1–24.0 frames per second on CPU; 41.4–47.2 with the pose model on CoreML (detector on CPU). Performance mode: 5.5–8.5, below target, and its scores do not fit the base model's 0.3 rule | Step 0.3 |
| Translating one sentence (encode plus mT5-Base beam search, 4 beams, 150 frames) | **measured:** 0.54–1.29 s on MPS; 0.78–4.21 s on CPU, varying run to run. Model load 15–16 s; peak memory 4.6–6.4 GB | Step 0.3 |
| Extracting the domain recordings (about 1.5 hours of video) | about 2.3 h on CPU, about 1.1 h with CoreML [inferred from the Step 0.3 rates] | Step 2.2 |
| Fine-tuning with adapters on about 500 clips | 0.5–3 h per run | Step 2.4 |
| Disk | Under 15 GB: weights and pose files 3.1 GB (**measured**; the separate mT5 weight file is not needed), recordings about 5 GB, caches under 1 GB | — |

Memory is the main risk: mT5-Base has 582 M parameters [verified], about 2.3 GB in 32-bit floats [inferred]. Inference fits. Full fine-tuning may not fit in 16 GB alongside the system, so the plan uses adapters with the base weights frozen and measures peak memory in Step 2.4. [assumed]

## 5. Phases

Each numbered step is one session for the implementing model and ends at a test gate. **STOP** marks a point that needs the user's approval in that session (`CLAUDE.md`, "Stop points").

Steps can be run by several sessions in parallel. `WORKERS.md` defines the tracks, the file ownership, and the prompts, and makes three adjustments to the order below: the manifest, split builder, and recording tool are built early by their own track; the metrics and evaluation harness are built early by their own track; and Step 3.3 may start after Step 3.1 using the fixed-length boundary detector.

---

### Phase 0 — Environment, license check, feasibility

#### Step 0.1 — Repo scaffold and pinned environment

- **Entry criteria:** status is `READY`. **STOP:** the user approves installing `uv` and `ffmpeg`.
- **Deliverables:** `git init`; `.gitignore` per `CLAUDE.md`; package skeleton from `ARCHITECTURE.md`; `.python-version`, `pyproject.toml` with exact pins, committed `uv.lock`; `ruff`, a type checker, and `pytest` behind one `make check`; `scripts/doctor.py` reporting Python, package versions, device, free disk, and `ffmpeg`.
- **Acceptance tests:** `uv sync --frozen` works from a clean clone; `make check` passes; a test runs a tensor operation on MPS and an ONNX Runtime session on CPU; a test fails if any dependency is not an exact pin.
- **Metric target:** clean install and full check in under 10 minutes. [assumed]

#### Step 0.2 — License and asset check (stop point)

- **Entry criteria:** Step 0.1 passed.
- **Deliverables:**
  - Re-read from primary sources and log in `OPEN_QUESTIONS.md`: the Uni-Sign weights license on the hub; the status of the unlicensed Uni-Sign code (see Q3); licenses of `rtmlib`, the RTMPose ONNX model files it fetches, and `google/mt5-base`, none of which were verified in planning.
  - Read the Uni-Sign code and write `docs/base_model_notes.md`: which keypoints it uses, how it normalizes them, expected frame rate and maximum length, how missing keypoints and low-confidence scores are handled, how the pose encoder feeds mT5, and decoding settings. Everything in `ARCHITECTURE.md` marked "per base model" is filled in from this.
  - `docs/DATA_LICENSES.md` and the "Approved assets" table.
- **STOP:** the user approves each download by name: the two Uni-Sign checkpoints, mT5-Base, the RTMPose model files, and the code checkout. Nothing is downloaded before that.
- **Acceptance tests:** every asset used later has an approved row with license URL and date; open items are either closed or brought to the user.
- **Metric target:** none. This is a gate.

#### Step 0.3 — Feasibility spike: does the base model run here

- **Entry criteria:** Step 0.2 passed; assets present. The user has recorded two fixture clips of about 5 s (hands visible; hands leaving the frame), kept local and git-ignored. (Q11)
- **Deliverables:** a throwaway script under `scratch/` (not the final modules) that runs keypoint extraction and the Uni-Sign pose-only model end to end on this Mac, on CPU and on MPS, in 32-bit floats; `docs/feasibility.md` with what had to change to run without CUDA, frames per second for extraction, seconds per sentence for translation, peak memory, and the text produced for the fixture clips by both checkpoints.
- **Acceptance tests:** the script produces English text for both fixture clips on this machine; CPU and MPS give the same text for the same input, or the difference is explained.
- **Metric target** [assumed]: extraction at 10 frames per second or better and translation of a 6 s clip in under 3 s. **Decision gate:** if the model cannot be made to run, or is far slower than this, stop and return to the user with the fallback in Q2 before any further work. Update section 4 with the measurements.

---

### Phase 1 — Preprocessing and cache, with tests

#### Step 1.1 — Extraction and normalization

- **Entry criteria:** Phase 0 passed.
- **Deliverables:** `preprocess/video_io.py`, `preprocess/extract.py` (one per-frame core with a file entry point and a live-frame entry point), `preprocess/normalize.py`, `preprocess/schema.py`, per `ARCHITECTURE.md`.
- **Acceptance tests** (real extraction on the fixture clips unless noted):
  - Timestamps come from the container, are strictly increasing, and match `ffprobe` within 1 ms; a variable-frame-rate fixture keeps its real timestamps.
  - Frame-rate reduction to the model rate happens before pose estimation and picks the same frames in the file path and the live path.
  - File path and live path give identical arrays for the same frames.
  - Two runs give byte-identical output.
  - **Parity with the base model's own pipeline:** on the fixture clips, the model inputs produced by our extraction plus normalization match those produced by the Uni-Sign reference code within a stated tolerance, and the translated text is identical. This is the test that guards against the old training-and-inference mismatch, now between us and the pretrained model.
  - Missing or low-confidence hands follow the documented rule from `docs/base_model_notes.md`; in the hands-out-of-frame clip the presence mask is false for those frames, and changing the raw values of a masked keypoint does not change the normalized output.
  - `normalize_keypoints` is defined in exactly one module; a test searches `src/` for a second implementation.
  - A blank synthetic video does not raise and yields an all-false presence mask.
- **Metric target:** hands detected in at least 90% of frames of the hands-visible fixture. [assumed]

#### Step 1.2 — Cache, manifest, CLI

- **Entry criteria:** Step 1.1 passed.
- **Deliverables:** `preprocess/cache.py`, `data/manifest.py`, commands `s2e extract`, `s2e cache verify`, `s2e bench extract`.
- **Acceptance tests:** a second run does no extraction and returns identical data; changing any extraction setting, a pose model file hash, or the extractor version changes the cache key, while unrelated settings do not; a copied and re-timestamped source still hits the cache; an interrupted write leaves nothing that reads as valid; `s2e cache verify` reports a deleted entry; extraction over a file list is resumable.
- **Metric target:** measured extraction throughput recorded in section 4.

---

### Phase 2 — Offline baseline on pre-recorded clips

#### Step 2.1 — Model wrapper, evaluation harness, and reproduction check

- **Entry criteria:** Phase 1 passed. **STOP:** the user approves downloading the How2Sign test-split keypoints published in the Uni-Sign hub repository and the How2Sign translation CSVs (424 KB for test [verified]); both derive from How2Sign, which is CC BY-NC 4.0. If the user declines, this step skips the reproduction check and says so in the results.
- **Deliverables:** `models/translator.py` (loads the base checkpoint, runs on CPU or MPS in 32-bit floats, no CUDA-only code), `infer/offline.py`, `eval/metrics.py` (sacreBLEU with signature, ROUGE-L, exact match), `eval/run_eval.py`.
- **Acceptance tests:** forward and `generate` on a real cached clip on CPU and MPS; padding does not change outputs; metric wrappers reproduce hand-computed values on a fixture; a checkpoint whose preprocessing hash differs from the runtime config is refused.
- **Metric target:** BLEU on the How2Sign test split with the published keypoints of at least 13.0. [estimate, anchored on the paper's 14.5 for the pose-only model, verified; which BLEU implementation the paper used is not verified, so a gap of about a point is not alarming] A much lower score means the port is wrong; fix it before going on. If the full test split is too slow, score a fixed random subset of 500 sentences and report it as a subset.

#### Step 2.2 — Domain definition and recording tool

- **Entry criteria:** Step 1.2 passed; Q1 and Q11 answered.
- **Deliverables:**
  - `domain/sentences.yaml`: the sentence list for the chosen domain, with a vocabulary list. A draft exists (103 English sentences, 83 training and 20 held out, 76 concepts; approved by the user on 2026-11-16) with the ASL fields empty. Sentences are 2 to 5 signs, with 20 sentences held out entirely for the unseen-sentence test and every sign in those 20 also appearing in training sentences. Each sentence has an ASL version taken from a reliable source, recorded in the file with where it came from; the implementing model must not invent how a sentence is signed. A fluent signer's review is strongly recommended before recording.
  - `s2e record`: a local tool that shows a prompt, records the webcam with real timestamps, lets the signer accept or redo the take, and writes a manifest row. It also records continuous takes of several sentences in a row with sentence end times marked by a key press, for Phase 3.
  - `docs/recording_protocol.md`: framing, lighting, how many takes, sessions on different days and in different clothing, and at least one session in the lighting and background expected at the show.
  - The recorded dataset (the user's job: learning the sentences first, then about 2 to 3 hours of recording [assumed]): train, validation, and two test sets as defined in `ARCHITECTURE.md`.
- **Acceptance tests:** the split builder guarantees no recording appears in two splits, no unseen-test sentence text appears in train or validation, and every sign in the unseen-test sentences is covered by training sentences (checked against the vocabulary annotations); manifest validation passes; cache verify is clean; recordings are git-ignored.
- **Metric target:** at least 6 accepted takes per training sentence across at least 2 sessions. [assumed]

#### Step 2.3 — Zero-shot baseline on the domain

- **Entry criteria:** Steps 2.1 and 2.2 passed.
- **Deliverables:** validation scores for both pretrained checkpoints with no adaptation; decoding settings chosen on validation; `docs/results.md` started.
- **Acceptance tests:** results list BLEU, ROUGE-L, and exact-match rate with the sacreBLEU signature; the test sets are not touched in this step.
- **Metric target:** none imposed. This number is the "before" for the demo, and I expect it to be low. [assumed] It also decides which checkpoint is fine-tuned.

#### Step 2.4 — Domain adaptation

- **Entry criteria:** Step 2.3 passed. **STOP:** per `CLAUDE.md`, state the training design (which weights are frozen, adapter type and rank, loss, optimizer, schedule, augmentation, batch size, precision) and get the user's confirmation before writing training code. **STOP** before the single test-set scoring.
- **Deliverables:** `train/loop.py`, `train/checkpoint.py` (separate `last` and `best`, atomic writes, resume, refuse on config change), `s2e train`; the adapted model selected on validation; one scoring of both test sets; `runs/<id>/model_card.md`.
- **Acceptance tests:** overfit test on 16 clips reaches exact reproduction; resume test matches an uninterrupted run; `best` changes only when validation improves; a finished run can be resumed; the test sets were scored once (logged); peak memory during training is recorded.
- **Metric targets** [all estimates; no published anchor exists for this setup]:
  - Seen-sentence test (new recordings of training sentences): exact match of at least 80%.
  - Unseen-sentence test (new sentences from seen vocabulary): BLEU of at least 30 and at least half of the sentences judged correct in meaning by the user.
  - If the unseen-sentence result is poor while the seen-sentence result is high, the model has memorized sentences. Report that plainly; the demo is then limited to the recorded sentence list and must be described that way.

---

### Phase 3 — Streaming inference

#### Step 3.1 — Streaming engine on replayed keypoints

- **Entry criteria:** Step 2.1 passed (the unadapted model is enough to build the engine).
- **Deliverables:** `infer/streaming.py` (rolling buffer, partial text every hop, commit logic, boundary interface with a fixed-length fallback), `s2e stream-replay`, `eval/stream_eval.py`.
- **Acceptance tests:** with oracle boundaries, streaming output equals offline output sentence for sentence; committed text never changes; memory is bounded over a 30-minute replay; gaps, duplicates, and out-of-order timestamps are handled and counted; nothing under `infer` imports `train`.
- **Metric target:** oracle-boundary streaming scores equal offline scores on domain validation. [inferred]

#### Step 3.2 — Sentence boundaries

- **Entry criteria:** Step 3.1 passed; continuous takes recorded in Step 2.2; Q8 answered. **STOP** before writing any training code for a learned detector.
- **Deliverables:** a rest-position boundary detector (both hands at rest or out of the signing space for a minimum time, with hysteresis and a maximum-length forced cut), with thresholds chosen on validation continuous takes; optionally a small learned tagger trained on the continuous takes if the rule misses the target; integration into the engine.
- **Acceptance tests:** boundary F1 at ±1 s on the continuous validation takes is reported for the detector and for the fixed-length fallback, and the detector is better; it runs causally with at most 0.5 s look-ahead.
- **Metric target** [estimates]: boundary F1 of at least 0.85 when the signer returns hands to rest between sentences; end-to-end score on continuous takes of at least 85% of the oracle-boundary score.
- Note: this is a sentence-level rest detector with an explicit rest position, evaluated against marked boundaries. It is not the old pause-based splitter, which cut individual signs on motion dips and is not reused. The signer is told, in the protocol, to drop the hands between sentences.

#### Step 3.3 — Live path and latency

- **Entry criteria:** Step 3.2 passed.
- **Deliverables:** `serve/app.py` with `/health`, `POST /translate`, and `WS /stream` per `ARCHITECTURE.md`; per-stage timing; `scripts/ws_replay.py`. Video-file input is a required feature (user decision): `POST /translate` accepts an uploaded video (MP4, WebM, MOV) containing one or several sentences, runs it through the same extraction, boundary detection, and translation as the live path, and returns each sentence with its start and end time.
- **Acceptance tests:** `/health` is ready only after the model has loaded and reports checkpoint id, preprocessing hash, and device; a bad checkpoint path gives not-ready with the reason and a 503 from `/translate` (regression test for the old 503); a clip replayed over the socket gives the same final text as offline inference; an uploaded single-sentence video returns the same text as `infer/offline.py`, and an uploaded continuous take returns one segment per marked sentence within the boundary tolerance; an unreadable or unsupported file gets a clear 4xx error; a disconnect releases the session; a latency report is produced from a 10-minute replay on this Mac.
- **Metric target** [estimates]: real-time factor under 1.0; partial text at least every 1.5 s; median delay from sentence end to committed text under 3 s.

---

### Phase 4 — Demo

#### Step 4.1 — Web app with live captions

- **Entry criteria:** Step 3.3 passed; Q5 answered.
- **Deliverables:** `web/` app: camera preview with a framing guide, start and stop, committed captions and partial text shown differently, status line, a video-input mode (user requirement) with two options, "translate file" (upload, then show each sentence with its time range beside the video) and "play as live" (the chosen video plays in the page and its frames go through the same streaming path as the webcam, so captions appear as it plays), a visible list of what the demo covers (the domain and example sentences), a notice that output is experimental, exact-pinned `package.json`.
- **Acceptance tests:** `npm ci`, lint, type check, and build pass; a browser test with a video file as the camera shows committed text arriving; browser tests cover both video-input options with a recorded domain clip; refused camera permission and a dropped connection give clear messages.
- **Metric target:** frames sent at the configured rate within 10% for 5 minutes without memory growth in the tab. [assumed]

#### Step 4.2 — Show rehearsal and documentation

- **Entry criteria:** Step 4.1 passed.
- **Deliverables:** `scripts/dev.sh`; `README.md` (setup, how to run, measured results, limits, licenses and attributions); final `docs/results.md`; `docs/show_runbook.md` (setup checklist, lighting and framing, a rehearsed sentence sequence, what to do if the camera or model fails, and a recorded backup clip that can be played through upload mode); a rehearsal log of at least 30 live sentences from the domain with outputs written down as produced.
- **Acceptance tests:** from a clean clone, the README leads to a working live demo; the full test suite passes; the results document shows each target next to what was measured, with misses stated as misses.
- **Metric target** [estimate]: in the live rehearsal, at least 80% of in-domain sentences judged correct in meaning.

## 6. Risks

| Risk | Effect | Handling |
| --- | --- | --- |
| The signer is learning ASL for this project (Q10, decided) | Signing may be non-standard or inconsistent; the pretrained model helps less; memorizing the list is more likely | Small domain; ASL versions from a reliable source; fluent review if possible; record only once signing is consistent; the unseen-sentence test reports what happened; the demo is described as tuned to one learner |
| Uni-Sign will not run acceptably on this Mac | Base model changes | **Resolved in Step 0.3 (2026-11-17):** it runs in 32-bit floats with lightweight pose models and translation on MPS. CPU-only translation is borderline against the 3 s target |
| The Uni-Sign code has no license [verified] | Using it rests on the authors' evident intent, not a grant | Q3; keep the code out of this repo, do not redistribute, optionally ask the authors |
| Fine-tuning does not fit in 16 GB [assumed] | Slower or smaller adaptation | Adapters with frozen base; measured in Step 2.4; smaller batch with accumulation |
| The adapted model memorizes sentences instead of translating | Demo works only on the exact list | The unseen-sentence test measures this; reported either way |
| Show conditions differ from recording conditions (light, background, distance) | Live quality drops | Recording protocol includes varied sessions and one in show-like conditions; rehearsal in Step 4.2 |
| Model trained on one signer | Works for the demo signer only | Stated in the UI and README; add a second signer's recordings if one is available |
| Estimates in section 4 are wrong | Schedule | Steps 0.3, 1.2, and 2.4 replace them with measurements |
