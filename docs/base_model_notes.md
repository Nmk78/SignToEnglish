# Base model notes — Uni-Sign pose-only

Written in Step 0.2 (2026-11-16) by reading the Uni-Sign source online. Nothing was downloaded to disk: no weights, no model files, no checkout. Nothing here has been run; Step 0.3 is the first time the model executes on this machine.

**Source read:** https://github.com/ZechengLi19/Uni-Sign at commit `eed438bcb49e30405cd6ccdfcccca330c134e830` (latest on `main`, committed 2026-02-11, read 2026-11-16). File and line references below are to that commit, in the form `file:line`. Files were read at `https://raw.githubusercontent.com/ZechengLi19/Uni-Sign/eed438bcb49e30405cd6ccdfcccca330c134e830/<path>`.

Claim labels: **[verified]** read in the source named, on 2026-11-16; **[inferred]** reasoned from verified facts; **[assumed]** not checked.

## 1. Answers to every "per base model" item in ARCHITECTURE.md

| ARCHITECTURE.md item | Answer | Where |
| --- | --- | --- |
| §4.4 `K`, the keypoint count | 133, COCO-WholeBody order, one person per frame. Per-frame arrays are `keypoints (1, 133, 2)` and `scores (1, 133)`. [verified] | `datasets.py:579-581`, `datasets.py:28-29` |
| §4.4 which person is kept when several are detected | The detection box with the largest area. This is a one-off patch Uni-Sign made to its bundled copy of `rtmlib`; upstream `rtmlib` (0.0.13 and 0.0.16) returns every box. If nothing is detected, the pose model runs on the whole frame as the box, so there is always exactly one skeleton. [verified] | `demo/rtmlib-main/rtmlib/tools/object_detection/yolox.py:135-138`; `.../pose_estimation/rtmpose.py:25-26` |
| §4.5 which keypoints are used, and grouping | 69 of the 133, in four groups: body 9, left hand 21, right hand 21, face 18. Section 2.2. [verified] | `datasets.py:23-45` |
| §4.5 centering and scaling | Body: one bounding box over the whole clip, mapped to [-1, 1]. Hands: relative to the wrist point of that hand. Face: relative to one face point. Hands and face are divided by the body box size. Section 2.3. [verified] | `datasets.py:56-71`, `datasets.py:80-105` |
| §4.5 low-confidence and missing keypoints | A keypoint with score ≤ 0.3 is replaced by `(0, 0, 0)`: x, y, and the score channel. There is no separate mask input. Section 2.4. [verified] | `datasets.py:19`, `datasets.py:71`, `datasets.py:104` |
| §4.5 whole-clip statistics | Yes. The body box and scale are computed over every frame of the clip. The live path must compute them over the open segment. [verified] | `datasets.py:86-99` |
| §4.5 fixed-length resampling | None. Clips up to `max_length` frames are used as they are. Longer clips are cut down by random frame sampling. Section 2.5. [verified] | `datasets.py:640-643` |
| §4.6 and §6.2 model frame rate | The code has none. Every decoded frame of the video is used, at whatever rate the file has; no timestamps are read. Section 2.5. [verified] | `demo/online_inference.py:81-92`, `demo/pose_extraction.py:25-36` |
| §4.6 maximum input length | 256 frames by default (`--max_length`). One pose token per frame, no temporal downsampling. It is a training-time setting, not a hard architectural limit. Section 2.5. [verified for the default and the code path; inferred that it is not a hard limit] | `utils.py:523`, `stgcn_layers/stgcn_block.py:69-88` |
| §4.6 decoding defaults | Beam search with 4 beams, `max_new_tokens=100`, all other `generate` settings left at the mT5-Base defaults. Section 4. [verified] | `fine_tuning.py:263-266`, `demo/online_inference.py:125-128`, `models.py:317-321` |
| §6.2 maximum segment | 256 frames. At 30 frames per second that is 8.5 s, shorter than the 16 s default in ARCHITECTURE.md. [inferred; the 30 is assumed] | as above |

## 2. Input pipeline

### 2.1 Pose extraction (reference)

- Estimator: `rtmlib.Wholebody(to_openpose=False, mode="lightweight", backend="onnxruntime", device=...)`. [verified] `demo/online_inference.py:72-77`, `demo/pose_extraction.py:60,70-75`.
- `lightweight` mode means detector `yolox_tiny_8xb8-300e_humanart-6f3252f9` at 416×416 and pose model `rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122` at 192×256 (width×height). [verified] `demo/rtmlib-main/rtmlib/tools/solution/wholebody.py:62-69`; identical in the installed `rtmlib` 0.0.16.
- Frames come from `cv2.VideoCapture(...).read()`, so they are BGR and are passed to the estimator unchanged. [verified] `demo/pose_extraction.py:25-36,11-13`. Our PyAV decoding path has to hand over BGR frames too. [inferred]
- Each frame's keypoints are divided by `[W, H]`, so x and y are each in [0, 1] and the aspect ratio is not preserved. Scores are stored as returned. [verified] `demo/pose_extraction.py:44-46`.
- The stored format is a pickle `{"keypoints": [T × (1,133,2)], "scores": [T × (1,133)]}`. [verified] `demo/pose_extraction.py:23,48-49`.
- No smoothing, no tracking, no left-right flip. [verified by absence in the files above]

**Which estimator produced the training keypoints is not settled.** The paper says "We employ the RTMPose-x from MMPose to extract whole-body keypoints" (Appendix A.1, https://arxiv.org/html/2501.15187, read 2026-11-16) [verified]. The repository's dataset notes say the published keypoints "are extracted using RTMPose from MMPose. The pose extraction process is provided in the code" and point to `demo/pose_extraction.py`, whose default is the smaller `lightweight` model (`docs/DATASET.md:52`) [verified]. Neither says which model produced the How2Sign and OpenASL keypoints the two ASL checkpoints were trained on. So the demo's default estimator may be smaller than the one used for training. [inferred] `rtmlib`'s `performance` mode (`rtmw-dw-x-l`, 384×288, with `yolox_m`) is the closest thing it offers to "RTMPose-x". [inferred] Step 0.3 can measure both modes if both file sets are approved.

### 2.2 Keypoints used (indices into the 133)

| Group | Indices | Count | Reference point | Source |
| --- | --- | --- | --- | --- |
| `body` | 0, 3, 4, 5, 6, 7, 8, 9, 10 | 9 | none (box-normalized) | `datasets.py:31-33` |
| `left` | 91–111 | 21 | index 91 (first point of the hand) | `datasets.py:34-37` |
| `right` | 112–132 | 21 | index 112 | `datasets.py:38-41` |
| `face_all` | 23, 25, 27, 29, 31, 33, 35, 37, 39, then 83–90, then 53 | 18 | index 53 (the last of the 18) | `datasets.py:42-45` |

All [verified]. Names: in the COCO-WholeBody layout shipped with `rtmlib` (`rtmlib/visualization/skeleton/coco133.py`), 0 is the nose, 3 and 4 the ears, 5 and 6 the shoulders, 7 and 8 the elbows, 9 and 10 the wrists [verified in the installed 0.0.16]. That the face points are every second jaw-line point, the inner mouth, and the nose tip is [inferred] from the 68-point face convention, not stated in the code. Feet, hips, legs, eyes, and most of the face are not used.

Graph edges per group are in `stgcn_layers/gcn_utils.py:45-99`. In the body graph the last two nodes are the wrists, which matters in section 3.

### 2.3 Normalization, exactly

Input: per frame, `skeleton (133, 2)` already divided by `[W, H]`, and `conf (133,)`. Threshold `thr = 0.3` (`datasets.py:19`). The function is `load_part_kp` (`datasets.py:18-75`) with `crop_scale` (`datasets.py:80-105`). All [verified].

1. **Body** (`crop_scale` on the `[T, 9, 3]` array of x, y, score):
   - Take every body keypoint in every frame with score > 0.3. If there are fewer than 4, or the box below has zero size, the body output is all zeros and `scale = 0` (`datasets.py:86-88,96-97`).
   - `xmin, xmax, ymin, ymax` over those points; `scale = max(xmax - xmin, ymax - ymin)` (`datasets.py:89-95`).
   - `xs = (xmin + xmax - scale) / 2`, `ys = (ymin + ymax - scale) / 2`; `xy = ((xy - [xs, ys]) / scale - 0.5) * 2` (`datasets.py:98-101`). This centers the box and maps its longer side to [-1, 1].
   - Clip the whole array, score channel included, to [-1, 1] (`datasets.py:102`).
   - Set every keypoint whose score is ≤ 0.3 to `(0, 0, 0)` (`datasets.py:104`).
2. **Each hand**: subtract the hand's first point (index 91 or 112) from all 21, frame by frame, using its raw position whatever its score (`datasets.py:36,40`). Divide x and y by the **body** `scale` (`datasets.py:68`). Clip to [-1, 1] (`datasets.py:69`). Zero every keypoint with score ≤ 0.3 (`datasets.py:71`).
3. **Face**: the same as a hand, with point 53 as the reference (`datasets.py:44`).
4. If the body `scale` is 0, hands and face are all zeros too (`datasets.py:65-66`).

Output: four tensors `body [T, 9, 3]`, `left [T, 21, 3]`, `right [T, 21, 3]`, `face_all [T, 18, 3]`, each `(x, y, score)`. They are float64 from NumPy until cast (`datasets.py:73`); the reference then casts to bfloat16 (`demo/online_inference.py:121`). We cast to float32.

Consequences for `preprocess/normalize.py`:

- The cache stores pixel coordinates (ARCHITECTURE.md §4.4), so `normalize_keypoints` divides by `frame_size` first. This is why `frame_size` is in its signature. [inferred]
- The model input is the dict of four `[T, V, 3]` arrays, not one array. The "mask" in the contract has no counterpart inside the model; it is derived (`score > 0.3`) for our tests and for the boundary detector. [inferred]
- The body box depends on the whole clip, so the same frame normalizes differently in a different segment. The live path recomputes over the open segment on every hop. [inferred]
- `force_ok` does nothing: both branches are the same line (`datasets.py:57-61`). [verified]

### 2.4 Missing and low-confidence keypoints: the documented rule

The rule the model saw in training is: **score ≤ 0.3 → the keypoint's three inputs are 0** (`datasets.py:71,104`). [verified] There is no interpolation and no per-part presence flag. A hand out of frame therefore shows up as 21 rows of zeros, provided the estimator scores it at or below 0.3; the estimator always returns 133 points, so "missing" only ever means "low score". [inferred]

Two quirks the port must reproduce, both [verified]:

- A hand's coordinates are taken relative to its wrist point even when that wrist point's own score is ≤ 0.3. The wrist is then zeroed but the other fingers keep offsets from an unreliable origin (`datasets.py:36,71`).
- A masked keypoint's raw value does not affect the output of *that* keypoint, but a masked wrist or face-reference point still shifts its group, and points above the threshold feed the body box. The Step 1.1 test "changing the raw values of a masked keypoint does not change the normalized output" holds as written only for non-reference points. For points 91, 112, and 53 it does not hold in the reference. This needs a decision in Step 1.1: keep exact parity and narrow the test, which I recommend, or deviate from the base model. [inferred]

`person_found` has no counterpart: the reference never knows whether the detector found anyone (section 1, row 2). [verified]

### 2.5 Frame rate and length

- No frame rate anywhere. The demo reads every frame (`demo/online_inference.py:81-92`); training reads every stored pose frame (`datasets.py:469-475`). [verified]
- The native frame rates of the How2Sign and OpenASL videos behind the two checkpoints were **not verified** in this step. A model rate of 30 frames per second is [assumed] until checked; Step 0.3 should record the fixture clips' rate and Step 2.1's reproduction check will show whether the port is right at the dataset's native rate.
- `max_length` defaults to 256 (`utils.py:523`). Which value the two ASL checkpoints were trained with is not recorded in the repository; the stage 3 script does not override it for translation (`script/train_stage3.sh:6-16`), so 256 is [inferred].
- A clip longer than `max_length` is reduced to `max_length` frames by `sorted(random.sample(range(duration), k=max_length))`, in training **and** at inference (`datasets.py:640-643`, `datasets.py:472-475`). [verified] The reference is therefore non-deterministic for long clips unless seeded. Our port needs a deterministic rule for long inputs (for example evenly spaced frames); parity with the reference can only be tested on clips of at most 256 frames. [inferred]
- Batches are padded by repeating each clip's last frame, with attention mask 0 on the padding (`datasets.py:350-373`). [verified] The temporal convolution has kernel 5 (`models.py:93`), so repeated frames influence the last two real frames' features. The Step 2.1 test "padding does not change outputs" may need a tolerance or batch size 1; the reference demo uses batch size 1 (`demo/online_inference.py:32-35`). [inferred]

## 3. Model

All [verified] in `models.py` unless marked.

- **Per group:** `Linear(3 → 64)` on `(x, y, score)` (`models.py:86,222`), then a three-block spatial graph network 64 → 64 → 128 → 256 (`models.py:92`, `stgcn_layers/stgcn_block.py:122-123`), then a three-block temporal graph network at 256 with temporal kernel 5, stride 1 (`models.py:93`, `stgcn_block.py:124-125`). Adjacency matrices are learnable (`stgcn_block.py:36-37`). BatchNorm throughout.
- **The left hand reuses the right hand's weights** (`models.py:95-97`). Coordinates are not mirrored.
- **Body features are added into the other groups** before the temporal network, detached: the left hand gets the body's second-to-last node, the right hand the last node, the face node 0 (`models.py:240,252,255`). In the body group those are the left wrist, right wrist, and nose (section 2.2).
- **Per frame:** mean over the group's keypoints (`models.py:262`), concatenate the four 256-vectors, add a learned 1024-vector `part_para`, `Linear(1024 → 768)` (`models.py:266-267`, `models.py:99-100`). One 768-dimensional token per frame.
- **Into mT5:** the text prompt `"Translate sign language video to English: "` is tokenized, embedded with the mT5 encoder's own embedding table, and placed **before** the pose tokens (`models.py:269-277`). The attention mask is the prompt's mask followed by the frame mask (`models.py:279-280`). The result goes to `MT5ForConditionalGeneration` as `inputs_embeds` (`models.py:291-295`, `models.py:317-321`).
- **The prompt language depends on `args.dataset`:** "Chinese" if the name contains `CSL`, else "English" (`models.py:104-107`). The default for `--dataset` is `CSL_Daily` (`utils.py:526`), so the demo command in `demo/README.md` would prompt for Chinese. For the ASL checkpoints `dataset` must be `How2Sign` or `OpenASL`.
- **Tokenizer:** `T5Tokenizer.from_pretrained(mt5_path, legacy=False)` (`models.py:140`). The exact token ids of the prompt, including the end-of-sequence token the tokenizer appends, are part of what the checkpoint expects. The reference pins `transformers` 4.40.0 (`requirements.txt:24`); this project pins 5.23.0, where `T5Tokenizer` is built on the `tokenizers` backend (seen locally: its base class is `TokenizersBackend`). Whether both produce the same ids for this prompt is **not verified** and is a Step 0.3 check.
- **Sizes:** `hidden_dim` must be 256 (`utils.py:455`, `models.py:100`). The paper gives 4.5 M parameters for the pose encoder and 582.4 M for mT5-Base (Table 12) [verified in the paper].
- **Loss, for Step 2.4:** cross-entropy with label smoothing 0.2, targets truncated to 50 tokens (`models.py:282-301`, `utils.py:532`). Stage 3 recipe: AdamW, learning rate 3e-4, cosine schedule, 20 epochs, batch 8 (`script/train_stage3.sh:6-16`).

### Checkpoint format

- `torch.load(path, map_location='cpu')['model']`, loaded with `strict=True` (`demo/online_inference.py:51-53`). [verified] The files are pickles; we load them with `weights_only=True`. [inferred]
- The saved state covers the whole model, mT5 included: `get_requires_grad_dict` keeps every entry (`models.py:325-335`, `fine_tuning.py:148-150`). [verified] The files are 1.19 GB each, about 2 bytes per parameter for roughly 590 M parameters, so they are probably stored in bfloat16. [inferred from the size; check in Step 0.3]
- So the `google/mt5-base` weight file is overwritten in full when a checkpoint loads. [inferred] It is still needed to run the reference code unmodified, because `Uni_Sign.__init__` calls `MT5ForConditionalGeneration.from_pretrained(mt5_path)` (`models.py:139`). Building mT5 from `config.json` alone would avoid a 2.33 GB download but means changing how the reference constructs the model.

## 4. Decoding

- `mt5_model.generate(inputs_embeds, attention_mask, max_new_tokens=100, num_beams=4)` (`models.py:317-321`; callers `fine_tuning.py:263-266`, `demo/online_inference.py:125-128`). [verified]
- Nothing else is set: no length penalty, no repetition penalty, no n-gram blocking, no sampling. `google/mt5-base`'s `generation_config.json` holds only the start, end, and padding token ids (read 2026-11-16 at https://huggingface.co/google/mt5-base/raw/main/generation_config.json). [verified] Library defaults for the rest may differ between `transformers` 4.40 and 5.23. [assumed]
- Output ids are decoded with `batch_decode(..., skip_special_tokens=True)` (`fine_tuning.py:280`). [verified]
- The reference calls `model(src, tgt)` first and then `generate` on the returned embeddings (`demo/online_inference.py:123-128`). That first call runs a full mT5 forward pass and computes a loss against an empty target, which inference does not need. [verified] Our wrapper can skip it in Step 2.1 by building the embeddings directly. [inferred]

**BLEU in the paper** is not the current sacreBLEU default: the code calls `raw_corpus_bleu` from a vendored sacreBLEU 1.4.2, which applies no tokenization (`SLRT_metrics.py:26-43`, `external_metrics/sacrebleu.py:43`). [verified] This closes the "which BLEU implementation" item in PLAN.md Step 2.1: expect our sacreBLEU 2.6.0 number with the default `13a` tokenizer to differ from the paper's, and report the reference-style number beside it in the reproduction check. [inferred]

## 5. What has to change to run on this Mac

| Reference behaviour | Where | Needed here |
| --- | --- | --- |
| `model.cuda()` and `.cuda()` on inputs and padding | `demo/online_inference.py:41,121,136` | Use the selected device (MPS or CPU) |
| `model.to(torch.bfloat16)` and inputs cast to bfloat16 | `demo/online_inference.py:60,112,121` | 32-bit floats. Outputs can differ slightly in the last digits from a bfloat16 run. [inferred] |
| Pose estimator on `device="cuda"` | `demo/online_inference.py:76` | `cpu`, or `mps`, which `rtmlib` maps to ONNX Runtime's CoreML provider (`rtmlib/tools/base.py`); that provider is present in the installed `onnxruntime` 1.32.0 [verified locally] |
| `torch.cuda.amp.autocast` | `models.py:159` | Only reached in the RGB branch (`models.py:186`). Not executed for pose-only. [verified] |
| `torchvision` EfficientNet with `pretrained=True` | `models.py:110` | Only with `--rgb_support`. It would fetch weights implicitly, so `rgb_support` stays off. [verified] |
| `utils.py` imports `deepspeed`; `datasets.py` imports `decord` | `utils.py:29,37-38`, `datasets.py:11` | Neither is used for pose-only inference, but both imports run at module load. `decord` 0.6.0 has no macOS arm64 wheel and `deepspeed` ships a source distribution only (PyPI, read 2026-11-16) [verified]. See section 6. |
| `mt5_path = "./pretrained_weight/mt5-base"`, relative to the working directory | `config.py:1` | Set to the local `assets/` path before the model is built |
| Top-level module names `models`, `utils`, `datasets`, `config` | repository root | They only import with the checkout on `sys.path`, and the names are generic. The single loader in `models/translator.py` handles this (ARCHITECTURE.md §2). |
| `rtmlib` fetches model files by URL when given a URL | `rtmlib/tools/base.py` (`download_checkpoint`) | Always pass local `.onnx` paths, so nothing is fetched implicitly. [inferred] |

Reference pins that differ from ours (`requirements.txt`): `torch` 2.1.1+cu121 (ours 2.15.0), `transformers` 4.40.0 (5.23.0), `numpy` 1.22.4 (2.5.5), `opencv-python` 4.6.0.66 (5.0.0.93), `einops` 0.6.1 (0.8.2), Python 3.9 (3.12). [verified] No incompatibility was found by reading; none has been tested.

## 6. What this means for the port

**Model code: import it, do not rewrite it.** `models.py` and `stgcn_layers/` import cleanly with packages already pinned (`torch`, `torchvision`, `einops`, `transformers`). The recommendation for Q3's open choice is to keep a checkout in `third_party/` and import `Uni_Sign` from `models/translator.py`, rather than write an independent loader: it keeps the unlicensed code out of `src/`, and the state-dict keys match without a mapping. [inferred]

**Normalization: our own implementation, tested against theirs.** `preprocess/` may not import third-party code (ARCHITECTURE.md §2), so `normalize.py` implements the arithmetic of section 2.3 independently, and the Step 1.1 parity test compares it with the reference `load_part_kp`. That reference function lives in `datasets.py`, which cannot be imported here without `decord` and `deepspeed`. Loading it needs one of: empty placeholder modules for those two unused imports inside the test helper, or two new dependencies of which one does not install on this machine. **Decided (user, 2026-11-16):** the helper that loads the reference code may supply empty placeholder modules for the two unused imports, `decord` and `deepspeed`. Nothing under test is replaced, and the placeholders are used nowhere else.

**Keypoint parity: `rtmlib` 0.0.13 (bundled, patched) against 0.0.16 (ours).** Compared file by file on 2026-11-16, bundled copy against the upstream `0.0.13` tag and against the installed 0.0.16. [verified]

- The bundled copy is upstream 0.0.13 plus one change: keep only the largest detection box (`yolox.py:135-138`). 0.0.16 does not have it. Our extractor must call the detector and the pose model separately and select the largest box itself; calling `Wholebody(...)` directly would return several people in detector order.
- `Wholebody` mode table, `RTMPose` pre-processing and post-processing, and the score formula (mean of the two axis maxima) are the same in both versions.
- Detector post-processing differs in the branch for models exported without built-in suppression: 0.0.13 additionally requires score > 0.3 and class 0, and it raises if nothing passes suppression; 0.0.16 does neither. The `yolox_tiny` file does not take that branch: read after download on 2026-11-16, its outputs are `dets [1, N, 5]` and `labels [1, N]`, so suppression is built in and both versions take the other branch, which is the same in both (keep boxes with score > 0.3). [verified for `yolox_tiny`; the `yolox_m` file is checked when it is used]
- 0.0.16 skips resizing when a frame is already exactly the detector's input size. Not relevant at camera resolutions.
- So with the same ONNX files and our own largest-box selection, 0.0.16 should reproduce the reference keypoints. [inferred] Remaining sources of small differences: OpenCV 4.6 against 5.0 in the affine warp, and PyAV against `cv2.VideoCapture` in decoding. The Step 1.1 parity test states its tolerance against keypoints from the bundled copy run on identical decoded frames.

**Limits the plan should know about** (no file changed; `PLAN.md` and `ARCHITECTURE.md` belong to W0):

1. ARCHITECTURE.md §6.2 "Maximum segment: 16 s or the base model's maximum" resolves to 256 frames, 8.5 s at an assumed 30 frames per second.
2. ARCHITECTURE.md §4.5 gives `normalize_keypoints` one `model_inputs` output; the base model takes four arrays. A dict fits the signature, but `preprocess/schema.py` should say so.
3. PLAN.md Step 2.1 expects "How2Sign test-split keypoints". The hub offers How2Sign keypoints only as one archive for all splits, in four parts totalling 8.46 GB, and the test references it pairs with are a pickled label file inside the code repository (`data/How2Sign/labels.test`, 94 KB), not the How2Sign CSVs. That stop point needs restating when Step 2.1 is reached.
4. A full clone of the code repository would also bring dataset text: label files for How2Sign, OpenASL, CSL-Daily, and WLASL under `data/` (about 7.3 MB) and prediction and reference dumps under `out/`. The checkout proposed for approval is a sparse one that leaves those out.

## 7. Step 1.1: what the port does, and what was measured (2026-11-17)

Implemented in `src/signtoenglish/preprocess/` (`video_io.py`, `extract.py`, `normalize.py`, `schema.py`). Measured on this Mac with the lightweight pose models on CPU, on the two fixture clips (150 frames each).

**Parity with the reference pipeline** (`tests/preprocess/test_parity.py`; the reference runs unmodified in its own process, with its bundled rtmlib 0.0.13):

| Compared | Stated tolerance | Measured, `hands_visible` / `hands_leave` |
| --- | --- | --- |
| Raw keypoints, pixels | 1e-3 | 3.1e-5 / 6.0e-5 |
| Scores | 1e-6 | 0 / 0 |
| Model inputs | 1e-6 | 1.8e-7 / 1.6e-7 |
| Normalization alone, same raw keypoints in | bit-identical as float32 | bit-identical |
| Translated text and token ids (OpenASL checkpoint, CPU) | identical | identical |

The remaining difference is the cache format storing pixel coordinates as float32 where the reference kept float64.

**Decoding has to match OpenCV exactly, and now does.** The reference decodes with `cv2.VideoCapture`. PyAV's default conversion to BGR differs from it by up to 3 levels per pixel. That is invisible, but it moved low-confidence keypoints by up to 1,078 px, flipped 270 of 19,950 points across the 0.3 rule on `hands_leave`, and changed the whole-clip body box (model inputs off by up to 1.04). `video_io` therefore converts with swscale flags `POINT + ACCURATE_RND`, which gives frames bit-identical to OpenCV 5.0.0 on both fixtures (H.264 4:2:0 and 4:2:2), and a test compares every frame. [measured] Two consequences:

- The model inputs are fragile to tiny pixel changes wherever keypoint scores sit near 0.3. The live path gets JPEG frames from the browser, which differ from the camera's pixels by far more than 3 levels, so live and file results for the same signing will not be identical, and the whole-clip body box amplifies single flipped points. This is a property of the base model's rule, not a port error. [inferred]
- Equality with OpenCV 4.6, which the reference pinned, was not tested.

**Frame rate.** `model_rate_fps` defaults to 30. Sources at or below it (30, 29.97, 25) keep every frame, which is what the reference does; faster sources are thinned by timestamp before pose estimation, identically in the file and live paths.

**The two fixtures.** `hands_visible` is now a 5-second excerpt of a public-domain ASL video (`docs/DATA_LICENSES.md`); both hands pass the 0.3 rule in 150 of 150 frames (target 90%). `hands_leave` is the user's recording.

**Where exact parity and PLAN.md's wording disagree** (parity implemented; each needs the user's decision):

1. *Masked reference points.* PLAN.md: "changing the raw values of a masked keypoint does not change the normalized output". True for 66 of the 69 used keypoints, tested on synthetic arrays and on every masked point of `hands_leave`. Not true for points 91, 112, and 53 (each hand's root and the face reference): masked or not, their raw position shifts the rest of their group. A test pins that. Recommendation: keep parity and narrow the plan's wording.
2. *The lowered right hand.* In `hands_leave`, after the hands leave the frame, the left hand is masked (at most 5% of frames pass) but the right hand still scores about 0.32 and passes in more than half of the frames. PLAN.md: "the presence mask is false for those frames". A test pins the actual behaviour. Recommendation: keep parity for the model inputs, and let the Phase 3 sentence-end detector use its own, stricter evidence (for example mean hand score and hand position) rather than this mask. Do not raise the 0.3 threshold for the model's inputs without the Step 2.1 reproduction check to measure the cost.
3. *Blank video.* PLAN.md: "yields an all-false presence mask". It does, and `person_found` is false in every frame; but the reason is the base model's "fewer than 4 valid body points zeroes everything" rule. Some keypoints on black frames score up to 0.35. No decision needed; noted so nobody relies on scores being low on empty frames.
4. *`person_found` is recorded but not used by normalization*, as in the reference. Recommendation: keep.

**One real-signing data point.** Windows of the public-domain video were run through the port as a sanity check (scratch, not a test). The OpenASL checkpoint produced "If you think you have a flu or COVID-19 symptom, contact your doctor." for one 8-second window and sentences mentioning "60%" for another; the How2Sign checkpoint was less coherent on the same input. The signer's actual words in those seconds were not checked against a transcript, so this is topical agreement with a COVID-19 prevention video, not a measured translation. It is the first sign that the port translates real ASL, and it favours the OpenASL checkpoint; Step 2.1 measures it properly. [measured output; interpretation inferred]
