# Feasibility of the base model on this Mac (Step 0.3)

**Status on 2026-11-17: the model runs. The step is not finished.** The two fixture clips have not been recorded yet, so every number below comes from synthetic input (random keypoints, or an `ffmpeg` test pattern with no person in it). The acceptance tests on real clips, and the go/no-go decision, are still open. Section 7 lists what is missing.

**Hardware and software:** Apple M5, 10 cores, 16 GB unified memory, macOS 27.0.1. Python 3.12, `torch` 2.15.0, `transformers` 5.23.0, `onnxruntime` 1.32.0, `rtmlib` 0.0.16, `sentencepiece` 0.2.3. Uni-Sign code at commit `eed438b`, unmodified, in `third_party/`. All model runs are in 32-bit floats. The spike scripts are throwaway code under the git-ignored `scratch/` (`spike_common.py`, `spike_model.py`, `spike_pose.py`, `spike_e2e.py`, `probe_tying.py`).

Labels: **measured** on this machine on 2026-11-17; **[inferred]**; **[assumed]**.

## 1. Verdict so far

| Question | Answer | Basis |
| --- | --- | --- |
| Does the Uni-Sign pose-only model construct and load here without CUDA or bfloat16? | Yes, both checkpoints, on CPU and on MPS | measured |
| Does it produce English text end to end (video file → keypoints → model)? | Yes, on a synthetic video. The text is meaningless because the input is. | measured |
| Do CPU and MPS agree? | Yes: identical output token ids in all 8 synthetic comparisons | measured |
| Extraction at 10 frames per second or better? | Lightweight pose models: 23.5 on CPU, 46.3 with the pose model on CoreML. Performance models: 6.3 and 8.4, **below target**. Test pattern only, see section 4. | measured, synthetic |
| A 6 s clip translated in under 3 s? | 0.28 to 0.98 s on MPS and 0.49 to 1.75 s on CPU for 180 frames. Worst case seen, when decoding runs to the 100-token limit: 1.9 s on MPS, **3.5 s on CPU**. | measured, synthetic |
| Text for the two fixture clips, both checkpoints | **Not done: no clips** | — |

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

## 3. Model: load time, memory, translation time (synthetic keypoints)

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

## 4. Pose extraction speed (synthetic video)

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

## 5. End to end on the synthetic video

`spike_pose.py` (lightweight, CPU) → reference `load_part_kp` → model → text, for both checkpoints on both devices: ran without error and returned English text, identical between CPU and MPS. With no person in the video, 73 to 90% of the keypoints were masked by the 0.3 score rule, and the text is invented by the decoder ("They're going to have a smaller area, …" from the How2Sign checkpoint; a repeated "Description: [Man stands]" caption from the OpenASL one). That is expected and says nothing about translation quality. It does show that the model will produce fluent text from an empty scene, which matters for the demo: the streaming engine must not send segments without signing to the model. measured; the last sentence [inferred]

## 6. Measurements to replace the estimates in PLAN.md section 4

All synthetic; to be confirmed on the fixture clips.

| Job | Estimate in the plan | Measured |
| --- | --- | --- |
| Pose extraction, lightweight mode | 10–25 frames per second | 23.5 (CPU); 46.3 (pose model on CoreML, detector on CPU) |
| Pose extraction, performance mode | — | 6.3 (CPU); 8.4 (pose model on CoreML) |
| Translating one sentence | 0.5–2 s | 0.23–0.98 s on MPS and 0.43–1.75 s on CPU for up to 180 frames; up to 1.9 s (MPS) and 3.5 s (CPU) at the 100-token limit |
| Model load | — | 15–16 s; peak memory 5.0–6.4 GB; steady 2.3–4.5 GB (CPU) or about 3.5–4 GB in total (MPS) |
| Disk for weights and pose files | about 4 GB | 3.1 GB on disk (no mT5 weight file) |

## 7. Still open

Blocked on the two fixture clips (`tests/fixtures/clips/`, about 5 s each: hands visible; hands leaving the frame):

1. **Acceptance test "the script produces English text for both fixture clips": open.**
2. **Acceptance test "CPU and MPS give the same text for the same input, or the difference is explained": passed on synthetic input only; open for the clips.**
3. Extraction speed with a person in the frame, lightweight and performance, CPU and CoreML.
4. The two pose modes compared on real frames (how often hands pass the 0.3 score rule in each), which is the evidence for choosing between them.
5. The text from both checkpoints for both clips.
6. The go/no-go decision.

Not blocked, deferred to later steps: keypoint parity against the reference's bundled `rtmlib` 0.0.13 (Step 1.1); the How2Sign reproduction score (Step 2.1), which is the real test that the port, the tokenizer, and the weight handling are right.
