# Feasibility of the base model on this Mac (Step 0.3)

**Result on 2026-11-17: go, with the lightweight pose models and translation on MPS.** The Uni-Sign pose-only model runs on this Mac in 32-bit floats on CPU and on MPS, both checkpoints produce English text for both fixture clips, CPU and MPS give identical output, extraction runs at 20 to 47 frames per second, and a 5 s clip translates in 0.5 to 1.3 s on MPS. Three things did not go cleanly and are stated in section 1: translation on CPU alone can exceed 3 s, the larger "performance" pose models miss the speed target and do not work with the base model's score rule, and the `hands_visible` clip does not show the hands for most of its length.

Sections 2 to 5 are the first half of the step, measured on synthetic input before the clips existed. Section 6 is the real clips. Where both exist, the real-clip number is the one to use.

**Hardware and software:** Apple M5, 10 cores, 16 GB unified memory, macOS 27.0.1. Python 3.12, `torch` 2.15.0, `transformers` 5.23.0, `onnxruntime` 1.32.0, `rtmlib` 0.0.16, `sentencepiece` 0.2.3. Uni-Sign code at commit `eed438b`, unmodified, in `third_party/`. All model runs are in 32-bit floats. Other applications were open and the machine was using 5.8 GB of swap during the clip runs, which shows in the CPU timings. The spike scripts are throwaway code under the git-ignored `scratch/` (`spike_common.py`, `spike_model.py`, `spike_pose.py`, `spike_e2e.py`, `analyze_clips.py`, `probe_tying.py`).

Labels: **measured** on this machine on 2026-11-17; **[inferred]**; **[assumed]**.

## 1. Verdict

### Acceptance tests

| Test | Result |
| --- | --- |
| The script produces English text for both fixture clips on this machine | **Passed.** Both checkpoints, both clips, CPU and MPS (section 6.3). The user was not asked to sign ASL, so the text means nothing; it is reported as produced. |
| CPU and MPS give the same text for the same input, or the difference is explained | **Passed.** Identical output token ids in 12 of 12 clip comparisons and 8 of 8 synthetic ones. |

### Targets

| Target | Measured on the clips | Result |
| --- | --- | --- |
| Extraction at 10 frames per second or better | Lightweight: 20.1 to 24.0 on CPU; 41.4 to 47.2 with the pose model on CoreML | **Met** |
| | Performance: 5.5 to 6.4 on CPU; 5.6 to 8.5 with the pose model on CoreML | **Missed** |
| A 6 s clip translated in under 3 s | MPS: 0.54 to 1.29 s for the 4.93 s clips (150 frames); 0.28 to 0.98 s for 180 synthetic frames | **Met** |
| | CPU: 0.78 to 4.21 s for the clips, and the same input varied between runs (for example 2.57 s and 1.12 s). Over 3 s in 4 of 12 first-run cases, all with outputs of 31 or more tokens | **Not reliably met** |

The clips are 4.93 s, not 6 s; the 180-frame figure is synthetic. Translation time is driven by output length, and non-signing input makes the How2Sign checkpoint produce long repetitive sentences. [inferred]

### What did not go cleanly

1. **CPU-only translation is borderline.** The demo needs MPS for translation. MPS met the target in every case measured.
2. **The performance pose models are too slow here and break the base model's masking rule.** Their confidence scores are on a different scale: 0.42 to 9.78 on these clips, never at or below 0.3, against 0.06 to 1.10 for the lightweight models. The base model's rule "score ≤ 0.3 means missing" (`docs/base_model_notes.md`, section 2.4) therefore never fires: with the performance models, 0% of keypoints were masked in both clips, including the two seconds where the hands are out of the frame, and on the test pattern with no person at all. Details in section 6.4.
3. **`hands_visible` does not match its instructions.** Judged from the keypoints, the framing is head and shoulders (nose at 46% of frame height, shoulders at 90%), and the hands are at or beyond the frame edge for about the first 3 s. Both hands pass the 0.3 rule in 88 of 150 frames (59%). Step 1.1's target of 90% cannot be met with this clip. `hands_leave` does match: hands in view for the first 3 s, lowered for the last 2 s. Details in section 6.2.
4. **A silent weight-sharing fault in the pinned `transformers`** was found and worked around (section 2, item 4). It needs a permanent test in Step 2.1.

## 2. What had to change to run without CUDA

Nothing in `third_party/` was edited. The spike does these things around the reference code:

1. **Device and precision.** The model is moved to `cpu` or `mps` and kept in float32, in place of `model.cuda()` and `model.to(torch.bfloat16)` (`demo/online_inference.py:41,60,121`). The checkpoints store bfloat16 values (583 of 627 tensors; the other 44 are integer counters); they are widened to float32 on load, which is exact. measured
2. **Unused imports.** Empty placeholder modules stand in for `decord` and `deepspeed`, which `datasets.py` and `utils.py` import and the pose-only path never calls (user decision, 2026-11-16).
3. **mT5 is built from `config.json`, with no pretrained weight file** (user decision, 2026-11-17: `pytorch_model.bin` was not downloaded). `Uni_Sign.__init__` calls `MT5ForConditionalGeneration.from_pretrained`; the spike swaps that call for construction from the local configuration while the model is built.
   **Proof that nothing is lost,** for each checkpoint: before loading, every parameter and buffer in the model was overwritten with a sentinel (NaN, or -7 for integers). Loading with `strict=True` reported no missing and no unexpected keys. Afterwards no sentinel remained, and every one of the model's tensors compared equal to the checkpoint's tensor of the same name, including all 284 mT5 tensors (582,401,280 mT5 parameters, matching the paper's 582.4 M). measured, both checkpoints
4. **A weight-sharing fault in the pinned `transformers`, found and worked around.** `transformers` 5.23.0 forces mT5's output layer (`lm_head`) to share one tensor with the input embedding, ignoring `tie_word_embeddings: false` in `config.json` (`transformers/models/mt5/configuration_mt5.py:78-80`: "Force because official weights have False serialized, but we have to tie always"). These checkpoints were trained with the two separate (the reference pins `transformers` 4.40.0) and hold different values for them (largest difference 113.8). Loaded into the shared layout, the embedding is silently replaced by the output layer's weights: no error, no missing key, no sentinel left, and the model returns an empty string. The first run of this spike did exactly that. What gave it away was the parameter count, 390 M instead of 582 M. The spike now gives `lm_head` its own tensor before loading and checks tensor-by-tensor equality with the checkpoint, which is the check that catches this. measured
   - Consequence for Step 2.1: `models/translator.py` must separate the two and must keep the equality check as a test. The same fault would have occurred with `pytorch_model.bin` present, so skipping that download did not cause it. [inferred]
   - Consequence for Step 2.4: adapter code must not re-join them (for example through `tie_weights()`); test it.
   - No dependency change is needed for this. Pinning `transformers` 4.x would also avoid it; I do not recommend that on current evidence.
5. **The prompt language.** `dataset` is set to `OpenASL` or `How2Sign` so the prompt says "English"; the demo's default would say "Chinese" (`docs/base_model_notes.md`, section 3).
6. **Pose extraction** uses the installed `rtmlib` 0.0.16 with local `.onnx` paths, calls detector and pose model separately, and keeps the largest box itself, as the reference's patched copy does.

Checked and found fine: `T5Tokenizer` loads from `spiece.model` alone and turns the prompt into `[89349, 11329, 17896, 1552, 288, 5413, 267, 1]`; `sentencepiece` run directly on the same file gives the same seven ids before the end marker, which is what the older tokenizer does, so the prompt ids should match what the checkpoints were trained with. measured for 5.23.0 and `sentencepiece`; equality with 4.40.0 [inferred]

## 3. Model: load time, memory, translation time (synthetic keypoints; real clips are in section 6)

Reference inference path as written: one `forward` call (pose encoder plus a full mT5 pass that inference does not need) and then `generate` with 4 beams and at most 100 new tokens (`demo/online_inference.py:123-128`). One checkpoint loaded at a time.

**Load** (median of the four runs): 15 to 16 s from process start to ready, of which about 12 s is constructing mT5 with random initial values that are then overwritten, 0.2 s is reading the checkpoint, and 0.5 to 0.7 s is the move to MPS. measured. Skipping the random initialization should save most of the 12 s; not tried. [inferred]

**Memory** (resident set size of the process):

| | CPU | MPS |
| --- | --- | --- |
| Peak during load and the spike's checks | 5.0 to 6.4 GB | 5.4 to 6.2 GB |
| After load, while translating | 2.3 to 4.5 GB | 1.0 to 1.4 GB, plus 2.4 to 2.6 GB allocated by the MPS driver |

measured. The peak is a load-time transient and includes copies made by the sentinel and equality checks.

**Translation time**, median of 3 runs after one warm-up, random keypoints run through the reference normalization. Time depends mostly on how many tokens come out, which for random input is arbitrary.

| Checkpoint | Frames | Output tokens | CPU | MPS |
| --- | --- | --- | --- | --- |
| OpenASL | 90 | 8 | 0.49 s | 0.26 s |
| OpenASL | 180 | 11 | 0.49 s | 0.28 s |
| OpenASL | 256 | 41 | 1.70 s | 0.97 s |
| How2Sign | 90 | 11 | 0.43 s | 0.23 s |
| How2Sign | 180 | 50 | 1.75 s | 0.98 s |
| How2Sign | 256 | 71 | 2.61 s | 1.49 s |
| OpenASL, synthetic video, 150 frames | 150 | 100 (hit the limit) | 3.53 s | 1.90 s |
| How2Sign, synthetic video, 150 frames | 150 | 40 | 1.88 s | 1.04 s |

measured, synthetic input. The `forward` call alone is 0.07 to 0.16 s on CPU and 0.04 to 0.09 s on MPS; the rest is beam search. A first call in a fresh process took up to 2.4 s on MPS (warm-up).

For the demo's sentences of 2 to 5 signs, outputs should be about 10 tokens, which is the 0.2 to 0.5 s rows. [inferred] The 100-token ceiling is the reference's setting; a lower ceiling for this domain is a decoding choice for Step 2.3.

**CPU against MPS:** the output token ids were identical for all three lengths with both checkpoints and for the synthetic video with both checkpoints (8 of 8). measured

## 4. Pose extraction speed (synthetic video; real clips are in section 6)

Input: `ffmpeg` test pattern, 1280×720, 30 frames per second, 150 frames, H.264. **It contains no person.** The detector found a box in 0 of 150 frames, so the pose model ran on the whole frame every time, as the reference does. Per-frame cost is one detector run and one pose-model run either way, so the speed should carry over to a real signer [inferred], but that has to be re-measured on the fixture clips, and nothing here says anything about keypoint quality. Frames processed one at a time, first three excluded, decoding not included (0.07 s for the 150 frames).

| Pose models | Detector on | Pose model on | Detector, median | Pose model, median | Frames per second |
| --- | --- | --- | --- | --- | --- |
| Lightweight (`yolox_tiny` 416, `rtmw-dw-l-m` 256×192) | CPU | CPU | 13.9 ms | 28.6 ms | **23.5** |
| Lightweight | CPU | CoreML | 14.4 ms | 6.8 ms | **46.3** |
| Performance (`yolox_m` 640, `rtmw-dw-x-l` 384×288) | CPU | CPU | 90.9 ms | 66.9 ms | **6.3** |
| Performance | CPU | CoreML | 91.6 ms | 28.9 ms | **8.4** |

measured.

- **CoreML cannot run either detector.** Both detector files have suppression built in, and ONNX Runtime's CoreML provider fails on its output shape ("CoreML static output shape ({1,1,1,3549,3549}) and inferred shape ({1,3549}) have different…"). So `rtmlib`'s `device="mps"` for the whole pipeline does not work here; the detector stays on CPU. measured
- **CoreML changes the numbers slightly.** On the same frames, the count of frames with mean left-hand score above 0.3 was 41 on CPU and 42 with CoreML (body: 14 and 16). The execution provider is already part of the preprocessing hash, which is right; a cache built with one provider must not be mixed with the other, and the Step 1.1 "byte-identical" and parity tests should run on CPU. [inferred]
- **Performance mode misses the 10 frames-per-second target** on this machine, mainly because of its detector (91 ms). The larger pose model alone on CoreML costs 29 ms; paired with the small detector that would be about 43 ms a frame, around 23 frames per second. [inferred from the two rows; that pairing was not run and is not an `rtmlib` mode]
- CoreML session start takes 1.7 s (lightweight) and 3.2 s (performance); CPU sessions start in under 0.1 s. measured

## 5. End to end on the synthetic video (before the clips existed)

`spike_pose.py` (lightweight, CPU) → reference `load_part_kp` → model → text, for both checkpoints on both devices: ran without error and returned English text, identical between CPU and MPS. With no person in the video, 73 to 90% of the keypoints were masked by the 0.3 score rule, and the text is invented by the decoder ("They're going to have a smaller area, …" from the How2Sign checkpoint; a repeated "Description: [Man stands]" caption from the OpenASL one). That is expected and says nothing about translation quality. It does show that the model will produce fluent text from an empty scene, which matters for the demo: the streaming engine must not send segments without signing to the model. measured; the last sentence [inferred]

## 6. The fixture clips

`tests/fixtures/clips/hands_visible.mov` and `hands_leave.mov`: each 150 frames, 1280×720, 30 frames per second, H.264 High 4:2:2 in MOV (checked with `ffprobe`). Recorded by the user, who was asked to move the hands, not to sign. No frame from them appears here or anywhere in the repository; the keypoint files derived from them stay in the git-ignored `scratch/out/`.

### 6.1 Extraction speed with a person in the frame

One frame at a time, first three frames excluded, decoding excluded (0.07 to 0.08 s per clip with `cv2.VideoCapture`). The detector found a person in 150 of 150 frames in every run. measured

| Pose models | Pose model on | `hands_visible` | `hands_leave` | Detector / pose model, median ms |
| --- | --- | --- | --- | --- |
| Lightweight | CPU | **24.0 fps** | **20.1 fps** | 13.4–15.9 / 28.0–33.0 |
| Lightweight | CoreML | **47.2 fps** | **41.4 fps** | 14.5–16.3 / 6.1–7.4 |
| Performance | CPU | **6.4 fps** | **5.5 fps** | 91.6–104.2 / 65.0–77.9 |
| Performance | CoreML | **8.5 fps** | **5.6 fps** | 89.1–138.0 / 29.5–40.3 |

The detector always runs on CPU (section 4). These agree with the test-pattern numbers. In `hands_leave` the performance detector reported two boxes in some frames; the largest was kept.

### 6.2 What the keypoints say about the clips (lightweight models, CPU)

A hand "passes" in a frame when at least 11 of its 21 points have a score above 0.3. This is a description for this report, not a rule of the base model, which masks point by point.

| | `hands_visible` | `hands_leave` |
| --- | --- | --- |
| Left hand passes | 88 / 150 frames | 91 / 150 |
| Right hand passes | 140 / 150 | 144 / 150 |
| Both hands pass | **88 / 150 (59%)** | 91 / 150 |
| Frames 0–89: left / right pass rate | 0.37 / 0.89 | 1.00 / 1.00 |
| Frames 90–149: left / right pass rate | 0.92 / 1.00 | 0.02 / 0.90 |
| Body points (9) above 0.3 | 98.0% | 98.8% |
| Nose and shoulder height (0 top, 1 bottom) | 0.46; 0.90 and 0.93 | 0.47; 0.89 and 0.92 |

measured.

- **`hands_visible`.** For the first 3 s the hand points sit at the bottom and side edges of the frame (right-hand points inside the frame: 5%; mean right-hand score 0.39, left 0.28). The right hand "passes" there only because its scores hover just above 0.3. From about 3 s on, both hands are in view (inside the frame 88% and 93%; scores 0.55 and 0.68). So the hands are clearly visible for roughly the last 2 s only. [inferred from the numbers; I did not look at the frames] With the shoulders at 90% of the frame height, the chest, where signing happens, is below the frame. **This framing will not work for the project**; the camera needs to be farther back or higher so the frame reaches the waist.
- **`hands_leave`.** Frames 0–89: both hands pass in every frame, mean scores 0.74. Frames 90–149: the left hand is masked almost completely (pass rate 0.02, mean score 0.24). The right hand is **not** masked: its mean score sits at 0.32, just over the threshold, and 90% of frames pass, with the points placed inside the frame near the bottom. So the base model's rule hides one out-of-frame hand and lets the other through at marginal confidence. Step 1.1's test "the presence mask is false for those frames" would fail for the right hand on this clip with the 0.3 rule as it stands. That is the reference behaviour, not a port error; it needs a decision in Step 1.1.

### 6.3 Text produced

150 frames each, reference normalization, 4 beams, up to 100 new tokens. Time is the median of 3 runs after one warm-up. "Masked" is the share of left-hand / right-hand inputs zeroed by the 0.3 rule. measured

Keypoints from the **lightweight** models on CPU:

| Checkpoint | Clip | Masked L / R | Text | Tokens | CPU | MPS |
| --- | --- | --- | --- | --- | --- | --- |
| OpenASL | `hands_visible` | 0.41 / 0.06 | "It's been a while since I was in high school." | 17 | 0.92 s | 0.64 s |
| OpenASL | `hands_leave` | 0.39 / 0.08 | "It's very beautiful." | 9 | 0.78 s | 0.66 s |
| How2Sign | `hands_visible` | 0.41 / 0.06 | "I'm going to take a piece of sandpaper and I'm going to cut it out." | 26 | 2.57 s (1.12 s on a repeat) | 0.63 s |
| How2Sign | `hands_leave` | 0.39 / 0.08 | "I'm going to go ahead and grab my tissue paper and I'm going to go ahead and grab my tissue paper and I'm going to go ahead and grab my tissue paper." | 50 | 3.88 s (2.25 s on a repeat) | 1.21 s |

Keypoints from the **performance** models on CPU (nothing masked):

| Checkpoint | Clip | Text | Tokens | CPU | MPS |
| --- | --- | --- | --- | --- | --- |
| OpenASL | `hands_visible` | "I took a sip of water to absorb the moisture from the moisture." | 23 | 1.20 s | 0.94 s |
| OpenASL | `hands_leave` | "This is the sign for bounce." | 10 | 0.86 s | 0.54 s |
| How2Sign | `hands_visible` | "Then you're going to take your mascara brush, and you're going to use your mascara brush." | 31 | 3.12 s | 0.95 s |
| How2Sign | `hands_leave` | "You're going to hold it in place, hold it in place, … hold it while you're holding it" (one phrase repeated seven times) | 53 | 4.21 s (2.44 s on a repeat) | 1.29 s |

Keypoints from the lightweight models with the **pose model on CoreML**: the OpenASL checkpoint gave the same two sentences as with CPU keypoints. The How2Sign checkpoint gave different ones ("I'm going to show you a little bit of what it looks like here, …", 47 tokens, 3.67 s CPU / 1.13 s MPS; and "You're going to want to hold it between your shoulder and your shoulder.", 22 tokens, 2.23 s / 0.67 s).

- **CPU against MPS:** identical output token ids in all 12 cases (2 checkpoints × 6 keypoint files). The acceptance test passes with no difference to explain.
- **Memory:** peak 4.6 to 5.7 GB during load; while translating 1.7 to 5.0 GB on CPU, or 0.8 to 1.1 GB plus 2.4 GB allocated by the MPS driver on MPS.
- The text changes with the pose models in every case, and for the How2Sign checkpoint it changes even between CPU and CoreML keypoints. With input that is not signing, that shows how sensitive the output is to small keypoint changes; it does not show which estimator is better. [inferred]
- Each checkpoint answers in the style of its training data: OpenASL short and conversational, How2Sign instructional and repetitive. That fits the plan's guess that OpenASL suits small talk better; Step 2.3 decides it on real signing.

### 6.4 Lightweight against performance on the same frames

| | `hands_visible` | `hands_leave` |
| --- | --- | --- |
| Score range, lightweight | 0.06 to 1.09 (median 0.94) | 0.07 to 1.10 (median 0.94) |
| Score range, performance | 0.42 to 9.74 (median 6.65) | 0.48 to 9.78 (median 6.84) |
| Share of the 69 used points above 0.3, lightweight / performance | 0.853 / 1.000 | 0.854 / 1.000 |
| Position difference on the 69 used points where both are above 0.3: median / 95th percentile | 10.1 px / 697 px | 7.1 px / 155 px |
| Position difference, all used points: median / 95th percentile | 18.8 px / 908 px | 9.5 px / 241 px |

measured, at 1280×720.

- Where both are confident the two estimators agree to about 7 to 10 pixels in the median. The large tail is points one of them places somewhere else entirely, mostly hands near or beyond the frame edge.
- The performance models' scores are not comparable to a 0.3 threshold, so the base model would be fed every guessed position as if it were certain. Using them would need a rescaled threshold, which the base model's code gives no value for.
- **CPU against CoreML, same pose model:** median position difference 0.000 px, but single points move by up to about 1,000 px when a low-confidence point's peak flips; 28 and 52 of 19,950 points changed sides of the 0.3 rule (lightweight; 0 for performance). Score differences up to 0.03 (lightweight) and 0.28 (performance). So CoreML is not interchangeable with CPU for a cache or for parity tests.

### 6.5 Which pose models: recommendation

This is my recommendation, not a decision. **Use the lightweight models, with the pose model on CPU until Step 1.1's parity test says CoreML is safe.**

| | Lightweight | Performance |
| --- | --- | --- |
| Speed here | 20–24 fps on CPU, 41–47 with CoreML | 5.5–8.5 fps, under the 10 fps target |
| Works with the base model's 0.3 score rule | Yes, scores run from 0 to about 1.1 | No, scores run from 0.4 to 9.8 and nothing is ever masked |
| Is the Uni-Sign demo's own default | Yes (`demo/online_inference.py:74`) | No |
| Keypoint precision | Lower input resolution (256×192) | Higher (384×288); not measured against ground truth here |

What is known about the checkpoints' training keypoints: the paper says "RTMPose-x from MMPose"; the repository says the published keypoints were extracted with "RTMPose" and points to a script whose default is the lightweight models; neither names the model used for How2Sign and OpenASL. What this step adds: the training code masks at 0.3, which only makes sense if the training keypoints' scores were on a 0-to-1-like scale. The lightweight models' scores are; the performance models' scores, as `rtmlib` returns them, are not. That points toward the lightweight scale, but it is indirect: an MMPose pipeline could have produced 0-to-1 scores from a larger model. [inferred]

What would settle it: the published How2Sign keypoints themselves. Their score range and the share of points at or below 0.3 can be compared directly with the two tables above, and the Step 2.1 reproduction check then shows whether the port reaches the published score with them. If those keypoints turn out to come from a larger model than the lightweight one, the accuracy cost of the lightweight models can be measured at that point by scoring the same test videos both ways, and the hybrid of the small detector with the large pose model (about 23 frames per second by arithmetic from section 4, not run) becomes worth testing with a rescaled threshold. That needs the How2Sign download, which is Step 2.1's stop point and not approved.

Accuracy is not being traded away knowingly: on current evidence the lightweight models are the ones consistent with how the base model was built. What is not known is whether a larger estimator would be more accurate for this project's own fine-tuned model; nothing measured here answers that.

## 7. Measurements to replace the estimates in PLAN.md section 4

Measured on the fixture clips unless marked.

| Job | Estimate in the plan | Measured |
| --- | --- | --- |
| Pose extraction, lightweight mode | 10–25 frames per second | 20.1–24.0 on CPU; 41.4–47.2 with the pose model on CoreML (detector on CPU) |
| Pose extraction, performance mode | — | 5.5–6.4 on CPU; 5.6–8.5 with CoreML. Below target, and incompatible with the 0.3 score rule |
| Translating one sentence (150 frames, 4 beams) | 0.5–2 s | MPS: 0.54–1.29 s. CPU: 0.78–4.21 s, varying run to run |
| Model load | — | 15–16 s; peak memory 4.6–6.4 GB; steady about 3.2–3.5 GB on MPS (process plus driver), 1.7–5.0 GB on CPU |
| Disk for weights and pose files | about 4 GB | 3.1 GB (no mT5 weight file) |

Extraction of 1.5 hours of recordings at 20 frames per second on CPU would take about 2.3 hours, and about 1.1 hours with CoreML. [inferred from the rates above]

## 8. Open after this step

1. **`hands_visible` should be recorded again** with the frame reaching the waist and both hands in view throughout; Step 1.1's 90% target needs it. The recording protocol (Track C) should state the framing in numbers, for example shoulders no lower than the middle of the frame. [my suggestion]
2. Step 1.1 decisions: the out-of-frame right hand that the 0.3 rule lets through (section 6.2); reference points and masking (`docs/base_model_notes.md`, section 2.4); CPU or CoreML for the pose model, given they do not produce identical keypoints.
3. Step 2.1: untie mT5's output layer before loading and keep the tensor-equality check as a permanent test; skip the reference's extra forward pass; cut the 12 s of random initialization at load.
4. Step 2.1: the How2Sign reproduction check is the real test of the port, and the published keypoints' score range is what settles the pose-model question (section 6.5).
5. Streaming (Tracks E, F, G): the model returns fluent sentences for input with no signing. Segments without signing must not reach it.
6. Keypoint parity against the reference's bundled `rtmlib` 0.0.13: Step 1.1.
