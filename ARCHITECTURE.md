# ARCHITECTURE.md — SignToEnglish

Revision 2, 2026-11-16. This document fixes module boundaries and data formats. Values marked "default" can change through config. Items marked **per base model** are dictated by the pretrained model and are filled in from `docs/base_model_notes.md` in Step 0.2; they were not verified in planning.

Claim labels: **[verified]**, **[inferred]**, **[assumed]** as in `PLAN.md`.

## 1. Pipeline

```
          OFFLINE (recordings, fine-tuning, clip translation)      LIVE (webcam)

 video file ──► video_io.read_frames ─┐          browser frame + capture timestamp
                (real timestamps)     │                        │ WebSocket
                                      ▼                        ▼
                    reduce to the model frame rate (same function for both)
                                      │
                    extract.PoseCore.process(frame, t_ms)             ◄── one code path
                    (RTMPose whole-body through rtmlib / ONNX Runtime)
                                      │  raw keypoints + scores
                        ┌─────────────┴──────────────┐
                        ▼                            ▼
                  cache (disk)                 rolling buffer (memory)
                        └────────────┬───────────────┘
                                     ▼
                    normalize.normalize_keypoints                      ◄── one function,
                                     │  model inputs + mask               matches the base model
                                     ▼
                    models.Translator = pretrained pose encoder + mT5 (+ adapters)
                                     ▼
                                English text
                 offline: one sentence per clip        live: partial text every hop,
                                                       committed text at each sentence end
```

Three stages with hard boundaries:

1. **Preprocessing** turns pixels into cached keypoint arrays. It knows nothing about text, models, or splits.
2. **Training** (adapter fine-tuning) reads manifests and caches and writes checkpoints. It never reads video.
3. **Inference** loads a checkpoint and calls the same preprocessing functions on files or live frames. It never imports training code.

## 2. Repository layout

```
SignToEnglish/
├── CLAUDE.md  PLAN.md  ARCHITECTURE.md  OPEN_QUESTIONS.md  REUSE.md  README.md
├── pyproject.toml  uv.lock  .python-version  .gitignore
├── configs/
│   ├── preprocess/keypoints.yaml
│   ├── model/unisign_pose.yaml
│   ├── train/domain_adapter.yaml
│   ├── stream/default.yaml
│   └── paths.example.yaml               # copied to paths.local.yaml (git-ignored)
├── domain/
│   └── sentences.yaml                   # the domain sentence list and vocabulary (committed)
├── src/signtoenglish/
│   ├── common/        config.py  hashing.py  device.py  logging.py
│   ├── preprocess/    video_io.py  extract.py  normalize.py  cache.py  schema.py
│   ├── data/          manifest.py  domain.py  splits.py  dataset.py  collate.py
│   ├── record/        recorder.py                  # prompted recording tool
│   ├── models/        translator.py  adapters.py  boundary.py
│   ├── train/         loop.py  checkpoint.py
│   ├── eval/          metrics.py  run_eval.py  stream_eval.py
│   ├── infer/         offline.py  streaming.py  loader.py
│   ├── serve/         app.py  ws.py  settings.py
│   └── cli.py                           # `s2e` entry point
├── tests/             mirrors src/
│   └── fixtures/      user-recorded clips (git-ignored), tiny synthetic arrays (committed)
├── scripts/           doctor.py  dev.sh  ws_replay.py
├── docs/              DATA_LICENSES.md  base_model_notes.md  feasibility.md  results.md
│                      recording_protocol.md  show_runbook.md
├── web/               Next.js app
└── (git-ignored)  data/  cache/  runs/  assets/  third_party/  scratch/
```

`third_party/` holds the Uni-Sign code checkout at a pinned commit, if `models/translator.py` imports from it. It is never committed (`OPEN_QUESTIONS.md`, Q3). `assets/` holds weights and pose model files, each checked against a recorded SHA-256 on load.

### Import rules

| Package | May import | Must not import |
| --- | --- | --- |
| `common` | standard library, pydantic, numpy | anything else in the project |
| `preprocess` | `common` | `data`, `models`, `train`, `eval`, `infer`, `serve` |
| `data` | `common`, `preprocess.schema`, `preprocess.cache`, `preprocess.normalize` | `train`, `infer`, `serve` |
| `record` | `common`, `preprocess.video_io`, `data.manifest`, `data.domain` | `models`, `train`, `infer`, `serve` |
| `models` | `common`, the third-party model code | `train`, `data`, `serve` |
| `train` | `common`, `data`, `models`, `eval` | `infer`, `serve` |
| `eval` | `common`, `data`, `models`, `infer` | `train`, `serve` |
| `infer` | `common`, `preprocess`, `models` | `train`, `data.dataset`, `serve` |
| `serve` | `common`, `infer` | everything else |

A test walks the import graph and fails on a violation. Only `models/translator.py` may import the third-party model code, so replacing the base model touches one module plus `preprocess`. No `sys.path` edits outside that one loader, and no script importing another script.

## 3. Configuration

- YAML files under `configs/`, each loaded into a frozen pydantic model. Unknown keys are an error.
- Machine-specific paths live in `configs/paths.local.yaml` or environment variables read by one settings class. Paths never enter a hash.
- **Config hash:** SHA-256 of the canonical JSON of a config model, truncated to 16 hex characters.
- **Preprocessing hash:** hash of the keypoint config fields that affect output, the SHA-256 of each pose model file, the `rtmlib` and `onnxruntime` versions, the execution provider, and `EXTRACTOR_VERSION` (an integer bumped by hand whenever extraction output could change).
- Normalization settings belong to the model config. Normalization runs at load time and does not invalidate the cache.
- Every run directory stores the resolved config; every checkpoint embeds it with the preprocessing hash and the base checkpoint's SHA-256.
- `infer/loader.py` compares the checkpoint's preprocessing hash with the runtime config and refuses to start on a mismatch.

## 4. Data formats

### 4.1 Domain sentence list (`domain/sentences.yaml`)

```yaml
domain: small_talk
sign_language: ase          # ISO 639-3 for ASL
text_language: en
vocabulary:                 # one entry per sign the domain uses
  - id: HELLO
  - id: NAME
sentences:
  - id: s001
    text: "Hello, my name is Alex."
    signs: [HELLO, MY, NAME, fs-ALEX]     # intended signing, for coverage checks only
    role: train                            # train | unseen_test
```

The `signs` field is a planning aid for the signer and for vocabulary-coverage checks in the split builder. It is never a model input or target; the model is trained on video and English text only.

### 4.2 Manifest (one JSON object per line)

File: `data/manifests/<dataset>_<split>.jsonl`, one row per sentence clip.

| Field | Type | Notes |
| --- | --- | --- |
| `id` | string | Unique |
| `dataset` | string | `domain` or `how2sign` |
| `split` | string | `train`, `val`, `test_seen`, `test_unseen`, or `test`; `unassigned` for recorder output before splits are built (`domain_takes.jsonl`, `domain_continuous_takes.jsonl`) |
| `source_path` | string | Relative to the data root |
| `source_sha256` | string | Content hash of the video file |
| `start_ms`, `end_ms` | int or null | Bounds inside the source; null when the file is one sentence |
| `text` | string | English reference, untouched |
| `sentence_id` | string or null | Id in `domain/sentences.yaml` |
| `signer_id` | string | |
| `session_id` | string | Recording session (date and a label for the conditions) |
| `take` | int | |
| `sign_language`, `text_language` | string | `ase`, `en` |
| `license` | string | `own-recording`, or the dataset's license |
| `tags` | list of strings | For example `show_conditions` |

Continuous takes for Phase 3 use `data/manifests/domain_continuous_<split>.jsonl`: one row per recording with a `segments` list of `{sentence_id, text, start_ms, end_ms}`.

### 4.3 Domain splits (`data/splits.py`)

| Split | Contents | Purpose |
| --- | --- | --- |
| `train` | Takes of `role: train` sentences | Adaptation |
| `val` | Different takes of `role: train` sentences, plus takes of about 15 `unseen` sentences set aside for validation | Choosing checkpoints, decoding settings, thresholds |
| `test_seen` | Further takes of `role: train` sentences, from a session not used in train where possible | Does it work on sentences it has practised |
| `test_unseen` | Takes of the remaining `role: unseen_test` sentences, whose text never appears in train or val | Does it translate new sentences from known signs |

Rules enforced by tests: no recording in two splits; no `test_unseen` sentence text in train or val; every sign listed for a `test_unseen` sentence appears in at least three training sentences. Both test splits are scored once per final checkpoint.

### 4.4 Keypoint cache

Path: `cache/keypoints/<preprocessing_hash>/<source_sha256[:2]>/<source_sha256>.npz`, one file per source video, written to a temporary name and renamed. `meta.json` beside the hash directory holds the full config and component hashes.

| Key | Shape | Dtype | Content |
| --- | --- | --- | --- |
| `timestamps_ms` | [T] | int64 | Container presentation time of each kept frame |
| `keypoints` | [T, K, 2] | float32 | Pixel coordinates as the pose estimator returns them |
| `scores` | [T, K] | float32 | Per-keypoint confidence |
| `person_found` | [T] | bool | False when no person was detected |
| `meta` | scalar | JSON string | See below |

`K` is the whole-body keypoint count the pose model outputs; RTMPose whole-body models use the 133-point COCO-WholeBody layout [assumed; confirm in Step 0.2]. `meta`: `schema_version`, `source_sha256`, `width`, `height`, `nominal_fps`, `n_source_frames`, `n_kept_frames`, `n_dropped_duplicate_timestamps`, `model_rate_fps`, `preprocessing_hash`, pose model names and SHA-256 values, `rtmlib_version`, `onnxruntime_version`, `execution_provider`, `extractor_version`, `created_utc`.

Rules: values are raw (no normalization, smoothing, or resampling beyond frame-rate reduction); when several people are detected, the selection rule is the base model's (**per base model**) and is recorded; `s2e cache verify` checks every manifest row against the cache. Size is a few hundred megabytes for the whole domain dataset. [inferred]

### 4.5 Normalization contract

`normalize_keypoints(keypoints, scores, person_found, frame_size, cfg) -> (model_inputs, mask)` in `preprocess/normalize.py`. It is the only place keypoints are transformed, for fine-tuning, offline inference, and streaming.

- The transform reproduces the base model's training-time preprocessing exactly: which keypoints are used, how they are grouped (body, each hand, face), how each group is centered and scaled, and how low-confidence or missing keypoints are treated. **Per base model.**
- The parity test in Step 1.1 compares this function's output with the reference pipeline's on real clips.
- Missing parts: follow the base model's rule if it has one. Where it does not, a missing or low-confidence part is represented by the mask and by the neutral value the model saw in training for that case, never by transforming placeholder zeros. The chosen behavior is documented and tested, including a test that changing the raw values of a masked keypoint does not change the output.
- The function is pure and works on any slice of frames, so the live path calls it on the rolling buffer. If the base model uses whole-clip statistics, the live path computes them over the open segment, and the parity test covers that case.
- No fixed-length resampling unless the base model requires it.

### 4.6 Model input and output

`models/translator.py` wraps the pretrained model behind one interface:

```
Translator.translate(model_inputs, mask, decode_cfg) -> list[str]
Translator.forward(model_inputs, mask, labels) -> loss          # used only by train/
```

- Base: Uni-Sign pose-only, which is a pose encoder of about 10 M parameters feeding mT5-Base with 582 M. [verified]
- Runs on MPS or CPU in 32-bit floats. No CUDA-only calls and no bfloat16. [the released script hardcodes both, verified]
- `models/adapters.py` adds low-rank adapters and controls which parameters are trainable. An adapter checkpoint stores only the trained parameters plus the base checkpoint's SHA-256; loading checks that hash.
- Frame rate, maximum input length, and decoding defaults: **per base model**, overridable in config; decoding settings are chosen on validation and stored with the checkpoint.
- Offline output: `{"id", "text", "n_frames", "decode_ms"}`.

### 4.7 Checkpoints and runs

```
runs/<run_id>/
├── config.resolved.yaml
├── last.pt            # trainable weights, optimizer, scheduler, step, RNG states, history
├── best.pt            # trainable weights + {metric, value, step} that selected it
├── history.jsonl
├── eval/<split>/      # hyps.txt, refs.txt, metrics.json (with sacreBLEU signature)
├── test_runs.log      # every test scoring, with timestamp and checkpoint hash
└── model_card.md      # base model, data, licenses and attributions, results, limits
```

Writes are atomic. `best.pt` is selected on a validation metric fixed in the training config (default: BLEU on the validation set's unseen sentences, with exact match on seen sentences as a tie-break).

## 5. Recording tool (`record/`)

`s2e record --sentences domain/sentences.yaml --signer <id> --session <label>`:

- Shows the next sentence, counts down, records the webcam with real timestamps, and lets the signer accept or redo the take.
- Chooses what to record next so that takes per sentence stay balanced.
- Continuous mode shows several sentences in sequence; a key press marks each sentence end.
- Writes video files under `data/domain/` and appends manifest rows. Nothing is uploaded anywhere.

This tool collects parallel video and English text for sequence-to-sequence fine-tuning. It is not the old phrase-enrollment system: there is no per-phrase classifier and no label set, and the model's output remains free text scored on sentences it has not seen.

## 6. Streaming

### 6.1 Engine (`infer/streaming.py`)

```
push(frame keypoints, t_ms)
  → append to rolling buffer
  → boundary detector updates on the newest frames
  every HOP (default 1.0 s) while a segment is open:
      translate buffer[segment_start : now] → emit PARTIAL(text)
  when the detector reports a sentence end, or the segment reaches MAX_SEGMENT (default 16 s):
      translate buffer[segment_start : end] → emit FINAL(text); wait for the next segment start
```

- Partial text is display-only and may change. Final text never changes.
- A segment starts when the hands leave the rest position and ends when they return to it for at least `REST_MIN` (default 0.6 s). A segment shorter than `MIN_SEGMENT` (default 1.0 s) is dropped without decoding.
- The boundary detector is an interface with implementations `FixedLength` (baseline), `RestPosition` (default), and `LearnedBoundary` (optional small causal tagger). Thresholds are chosen on recorded continuous validation takes and the detector is scored by boundary F1.
- The engine is independent of transport: `s2e stream-replay` drives it from a file or cache entry, `serve/ws.py` from a socket. Tests use replay.
- One engine per session; the model is shared and decoding goes through one worker. If a partial decode is still running when the next hop is due, that hop is skipped.
- Timing is recorded per stage: frame decode, pose estimation, translation, total, and the delay from segment end to commit.

### 6.2 Frame and timing assumptions

| Item | Default | Basis |
| --- | --- | --- |
| Camera request | 1280×720 at 30 fps, accept what the camera gives, minimum 640×480 | [assumed] |
| Model frame rate | Per base model; reduced by timestamp before pose estimation in both paths | Keeps the estimator's input identical offline and live [inferred] |
| Frames sent by the browser | At the model frame rate, JPEG, with capture timestamps | [inferred] |
| Hop for partial text | 1.0 s | [assumed] |
| Maximum segment | 16 s or the base model's maximum, whichever is smaller | [assumed] |
| Framing | One signer, upper body, face and both hands visible | [assumed] |

Real timestamps are used everywhere; no code assumes a frame rate.

### 6.3 Service (`serve/`)

- `GET /health` → `{"status": "ready" | "not_ready", "detail", "checkpoint_id", "preprocessing_hash", "device"}`. Ready means the model is loaded and a one-frame self-test passed.
- `POST /translate` (multipart video: MP4, WebM, or MOV) → `{"text", "segments": [{"text", "start_ms", "end_ms"}], "timings", "model"}`. The file may hold one sentence or several; it goes through the same extraction, boundary detector, and translator as the live path, using the file's own timestamps. Video-file input is a required feature, used for testing and as the backup at the show.
- `WS /stream`:
  - Client → server, first (JSON): `{"type": "start", "width", "height", "client_fps"}`.
  - Client → server, then binary: 8 bytes little-endian capture timestamp in milliseconds, followed by one JPEG frame.
  - Client → server (JSON): `{"type": "stop"}` flushes the open segment.
  - Server → client (JSON): `{"type": "partial" | "final", "segment_id", "text", "start_ms", "end_ms", "latency_ms"}`, `{"type": "status", "fps_in", "queue_depth", "rtf", "signing": bool}`, `{"type": "error", "detail"}`.
  - If the server falls behind it drops the oldest unprocessed frames and reports the count; it never queues without bound.
- Pose estimation runs on the server, so the live path uses exactly the extraction code that produced the training cache.
- Binds to localhost by default and makes no outbound network calls, so the demo works without internet.

### 6.4 Web app (`web/`)

Components: `CameraView` (permission, preview, framing guide, frame grabbing), `useStream` (socket lifecycle, timestamps, back-pressure), `Captions` (committed text plus a visually distinct partial line), `DomainPanel` (what the demo covers, with example sentences), `VideoInputPanel` (choose a video file, then either "translate file" through `POST /translate`, or "play as live", which plays the video in the page and sends its frames with their media timestamps over `WS /stream` exactly as the webcam path does; also the backup path at the show), `StatusBar`. No model code in the browser.

## 7. Evaluation

- **Domain, offline:** sacreBLEU (signature recorded), ROUGE-L, and exact-match rate after light text normalization (case and punctuation), reported separately for `test_seen` and `test_unseen`. A manual correct-in-meaning judgment by the user is recorded for `test_unseen`.
- **Reproduction check:** BLEU on the How2Sign test split with the published keypoints, to confirm the port of the base model (`PLAN.md`, Step 2.1).
- **Streaming:** on continuous takes, boundary F1 at ±1 s, and the same text metrics on committed sentences aligned to the marked segments.
- **Latency:** per-stage compute time, real-time factor, time between partial updates, and commit delay, on this Mac.
- **Live rehearsal:** a written log of live sentences and outputs (`PLAN.md`, Step 4.2).
- **Discipline:** everything is chosen on validation. `eval/run_eval.py` on a test split appends to `test_runs.log` and refuses a second run for the same checkpoint without `--force`.
- **Scope of claims:** results describe this domain, this signer or these signers, and these recording conditions. They are not comparable to published benchmark scores.

## 8. Adding another sign language or domain later

Nothing is built for this now. A new domain is a new `domain/*.yaml` and new recordings. A new sign language uses the same formats (the manifest carries language fields) and the same base architecture, whose decoder is multilingual [verified]; it needs data for that language and its own evaluation.
