# STATUS.md — where the project stands

Written 2026-11-17 for picking the project up again after a break. Read this first, then `PLAN.md` (section 0 holds the decisions made after revision 2). When work resumes, update this file or delete the parts that are no longer true.

## In one paragraph

Everything that can be built without recordings is built and merged: environment, extraction and normalization (matching the base model), keypoint cache, model wrapper, streaming engine, live service, web page with read-aloud, evaluation harness, and the recording tool. The demo runs end to end today with the **unadapted** pretrained model, so its sentences are rough or wrong, and it cuts sentences at fixed 8-second windows. No fine-tuning has been done. The project is waiting on data: the ASL versions of the domain sentences and the demo signer's recordings.

## Plan steps

| Step | State on `main` (commit `93aa62b`) |
| --- | --- |
| 0.1 Scaffold, pinned environment | Done |
| 0.2 License check, model files | Done |
| 0.3 Feasibility on this Mac | Done: go, with lightweight pose models and translation on MPS |
| 1.1 Extraction and normalization | Done: parity with the Uni-Sign reference pipeline |
| 1.2 Keypoint cache | Done |
| 2.1 Model wrapper | Done, **except** the How2Sign reproduction check, which has not produced a score (see "Unfinished") |
| 2.2 Recording tool, manifests, splits | Tool done and camera-tested by the user. **Recordings not started** |
| 2.3 Zero-shot baseline | Not started; needs recordings |
| 2.4 Fine-tuning | Not started; needs recordings and the user's sign-off on the training design |
| 3.1 Streaming engine | Done; 30-minute memory replay passed |
| 3.2 Sentence-end detector | Not started; needs continuous takes with marked sentence ends |
| 3.3 Live service | Done; 10-minute latency report made |
| 4.1 Web app against the real service | Done in Chromium. **Not checked in Safari** |
| 4.2 Launcher, README, runbook, rehearsal | Not started |

## What the user has to do first (nothing else can move without these)

1. **Ask the demo signer** whether she can record before the show and how much time she has, and get her consent to be recorded and shown. She is not fluent in ASL.
2. **Fill in the ASL version of each of the 103 sentences** in `domain/sentences.yaml` (`asl` and `source` fields, all empty today) from a reliable source: an ASL course or dictionary with video, or a fluent signer. The implementing model must not invent how a sentence is signed. A fluent signer's review is strongly recommended.
3. **Record** with `uv run --frozen s2e record --signer <id> --session <label>` (protocol in `docs/recording_protocol.md`): about 6 takes of each training sentence over at least two sessions, plus continuous takes (`--continuous`). Frame down to the waist; head-and-shoulders framing loses the hands (Step 0.3 finding).

## Then, in order

1. Run the How2Sign reproduction check (below).
2. Step 2.3: score both pretrained checkpoints on her validation recordings; pick one. The OpenASL checkpoint looked better on real signing so far [informal, not measured].
3. Step 2.4: state the training design to the user and wait for confirmation before writing any training code. The design must cover "train as you run" (same frame rate and JPEG step as the live path) and how batches are formed (the base model's padding changes outputs).
4. Step 3.2: the rest-position sentence-end detector, tuned on her continuous takes. It must use its own evidence, not the keypoint mask.
5. Test the CoreML pose model on her recordings (extract both ways, compare on validation) before adopting it for the live path.
6. Step 4.2: launcher, README, show runbook, 30-sentence rehearsal.

## Unfinished and unverified

- **How2Sign reproduction check: not run to completion.** It was started on 2026-11-17 and killed when that session ended, before writing results. The files are downloaded and verified, and the test split is extracted (2,349 files). Command, about 10–14 minutes on a quiet machine (ask the user first; it is a test-set scoring and a job over 10 minutes):
  `uv run --frozen s2e reproduce --pose-dir assets/how2sign/test-pose-rtmpose-192 --labels assets/how2sign/labels.test --checkpoint how2sign --device mps --subset 500 --seed 0 --run-dir runs/repro_how2sign_500`
  Target: BLEU of at least 13.0 computed the Uni-Sign way (the paper reports 14.5). It should also help settle which pose model produced the published keypoints; the archive's folder names end in `rtmpose-192`, a hint (not proof) toward the lightweight size.
- **Safari is untested.** Chrome is not installed on this Mac. The main risk is that Safari refuses speech that is not started directly by a click, which would silence read-aloud. The user has run the page and the upload path but has not reported which browser was used or whether speech was audible.
- **Live path is near its limit on CPU.** At 10 frames per second with the machine busy, 1.9–2.8% of frames were dropped; at 15 it was 28%. Close other applications at the show. On 2026-11-17 another project's test runner on this Mac pushed the load average above 80 and made a 16-second upload take over 2 minutes.
- **Deferred acceptance checks:** one segment per marked sentence on an uploaded continuous take (Step 3.2); browser tests with a recorded domain clip (after Step 2.2); streaming scores on domain validation (after recordings).
- **Open, minor:** ESLint 9 is flagged as no longer supported and `npm audit` shows 7 moderate dev-only advisories; both wait for a decision on a version change.

## How to run the demo as it is

Terminal 1 (wait for `/health` to say ready, 5 to 50 s):
```
cd /Users/naymyokhant/Programming/Personal/Projects/SignToEnglish
uv run --frozen s2e serve
```
Terminal 2:
```
cd /Users/naymyokhant/Programming/Personal/Projects/SignToEnglish/web
npm ci --offline && npm run build && npm start
```
Open `http://127.0.0.1:3000`. "Camera" tab for the live camera; "Video file" tab for "Translate file" and "Play as live". A 16-second clip of real signing for testing: `tests/fixtures/clips/asl_demo_16s.mp4` (local excerpt of the public-domain CDC video).

Checks: `make check` (quick, about 2 minutes) and `make check-full` (everything, with the real model; 6–9 minutes on a quiet machine; required before every merge).

## Local files that are not in git

| Path | What | Size |
| --- | --- | --- |
| `assets/unisign/` | Both Uni-Sign checkpoints (OpenASL, How2Sign) | 2.4 GB |
| `assets/rtmlib/` | Lightweight and performance pose models | 0.7 GB |
| `assets/mt5-base/` | mT5 tokenizer and config only. The 2.33 GB weight file is **not needed**: the checkpoints hold every mT5 weight (proven on load) | 4 MB |
| `assets/how2sign/` | How2Sign keypoints: four parts (8.46 GB), a joined copy of the same archive (8.46 GB, can be deleted), the extracted test split (1.0 GB), and `labels.test` | 17 GB |
| `third_party/Uni-Sign/` | Sparse checkout of the unlicensed Uni-Sign code at commit `eed438b` | 1 MB |
| `tests/fixtures/clips/` | `hands_leave.mov` (the user's), `hands_visible.mp4` and `asl_demo_16s.mp4` (excerpts of the public-domain CDC video in `commons_source/`) | 20 MB |
| `cache/`, `runs/` | Keypoint cache; the 30-minute streaming and 10-minute latency reports | small |

Licences and attributions for all of these are in `docs/DATA_LICENSES.md` and the asset table in `OPEN_QUESTIONS.md`.

## Leftovers to clean up (ask the user first)

- Worktrees `../s2e-a`, `../s2e-c`, `../s2e-e`, `../s2e-g`, `../s2e-i` and branches `track/a`, `track/c`, `track/e`, `track/g`, `track/i` are all merged. `s2e-c/data/` holds the user's throwaway trial recordings (signer `trial`); they must never be used for training and go away with that worktree.
- The joined `assets/how2sign/how2sign_pose_format.zip` duplicates the four parts.

## Standing decisions and rules from the user

- Heads-up, with name, size, source and licence, before any download over 50 MB. On 2026-11-17 the user also held downloads over 100 MB until on Wi-Fi; that hold was for that day.
- No trading away accuracy: exact parity with the base model is kept; lightweight pose models; CPU pose until CoreML is measured.
- The demo has two inputs, a video clip and the live camera, and reads committed sentences aloud with a local browser voice.
- The user does not record; the demo signer does. Other data is downloaded, each dataset approved by name.
- Ask before merging any branch; run `make check-full` first.

## Things that went wrong once and are now guarded by tests

- transformers 5.23.0 silently ties mT5's output layer to its embedding; loading the checkpoints that way gives a model that outputs empty text with no error. The wrapper separates them and checks every tensor against the checkpoint.
- PyAV's default colour conversion differs slightly from OpenCV's and was enough to flip keypoints across the base model's 0.3 score rule. Decoding now matches OpenCV bit for bit.
- A speed assertion inside the test suite failed whenever the Mac was busy. Tests now measure speed and report it; none assert it.
