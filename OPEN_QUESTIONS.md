# OPEN_QUESTIONS.md — SignToEnglish

Revision 2, 2026-11-16. Every decision, with a recommended default and what changes if the user picks differently. **All blocking questions were answered by the user on 2026-11-16.** Q5 to Q9 were not answered individually, so their recommended defaults apply and can still be changed.

Claim labels: **[verified]** read from a primary source on 2026-11-16 (see "Verification log"), **[inferred]**, **[assumed]**.

## Decisions already made by the user (2026-11-16)

| Decision | Effect on the plan |
| --- | --- |
| No rented GPU; train and run on this Mac (Apple M5, 16 GB, no NVIDIA GPU) | Revision 1's cloud path and its from-scratch How2Sign training are removed |
| Limit the scope to one conversation domain | Domain data is recorded for this project; evaluation is on that domain |
| Use an existing pretrained model built for sign language translation | Base model is a published pretrained translator, adapted rather than trained from scratch |
| The goal is a project-show demo that works well in a demoable area | Gates measure in-domain quality with the demo signer; general ASL translation is not claimed |
| The user does not know ASL and will learn the domain sentences to record and present them | The domain is small (103 sentences drafted); how each sentence is signed must come from a reliable ASL source; see Q10 |
| Video files must work as input as well as the webcam | Upload-and-translate and play-a-video-through-the-live-pipeline are required features of the service and the web app |
| Partial text while signing plus committed text at each sentence end is an acceptable start | Phase 3 design stands; lower-latency word-by-word output is not in scope |

## Summary of open questions

| # | Question | Blocking | Recommended default |
| --- | --- | --- | --- |
| Q1 | Which domain | Decided | Small talk |
| Q2 | Which pretrained base model | Decided | Uni-Sign pose-only; SHuBERT pipeline as fallback |
| Q3 | Licensing: non-commercial use, and the unlicensed Uni-Sign code | Decided | Personal non-commercial demo; use the code locally without redistributing it |
| Q4 | Compute | Decided | This Mac only |
| Q5 | Demo shell | No | Keep Next.js |
| Q6 | LLM polish | No | None |
| Q7 | Language scope | No | ASL to English only |
| Q8 | Webcam settings and how sentence ends are detected | No | 720p requested; rest-position boundary detector; 16 s maximum segment |
| Q9 | How the model is adapted | No | Adapters on a frozen base |
| Q10 | Who signs, and how much can be recorded | Decided | The user learns and records the 103 drafted sentences, 6 or more takes each |
| Q11 | Test fixture clips | Decided | The user records two short clips |
| Q12 | What counts as success | Decided | Works for the recorded domain and signer; stated limits elsewhere |

---

## Q1. Which domain — blocking

**Facts:** How2Sign is entirely instructional how-to video in ten categories (personal care, games, arts, hobbies, vehicles, sports, education and communication, food, home and garden, pets); none is conversational. [verified] No verified dataset of small-talk, classroom, or workplace ASL sentences was found. [verified that none was found; the search was not exhaustive] So the domain is defined by the sentences recorded for this project, and any domain is possible.

**What makes a domain work for this demo** [inferred]:

- Small sign vocabulary (about 200 signs) that recombines into many sentences.
- Little fingerspelling. About 20% of How2Sign is fingerspelled [verified], and names and technical terms are hard for these models.
- Sentences an audience recognizes and can check at a glance.
- The signer can produce them fluently and consistently.

| Domain | For | Against |
| --- | --- | --- |
| **Introductions and small talk** (greetings, name, where from, family, work or study, likes, weather, time, food, how are you) | Smallest vocabulary; natural for a live audience conversation; easy to rehearse | Names and places need fingerspelling; limit to a fixed short list |
| School or classroom | Clear scenario | Subject names and jargon push toward fingerspelling |
| Office | Clear scenario | Wider vocabulary; more abstract terms |

**Recommendation: introductions and small talk**, about 100 sentences over about 76 signs (sized for a signer who is learning them, see Q10), with a short fixed list of names and places.

**If the user picks differently:** only `domain/sentences.yaml` and the recording effort change. A wider domain needs more sentences and more takes for the same quality.

**Decision (user, 2026-11-16):** small talk (introductions and everyday small talk), at the default size above.

---

## Q2. Which pretrained base model — blocking

Candidates checked for publicly downloadable weights, license, and whether they can run on this Mac:

| Candidate | Weights available | License | Fit for this Mac |
| --- | --- | --- | --- |
| **Uni-Sign, pose-only** | Yes: `how2sign_pose_only_slt.pth` and `openasl_pose_only_slt.pth`, 1.19 GB each, full translation models, on the Hugging Face hub; inference code released 2025-12-20 [verified] | Weights tagged CC BY-NC 4.0 on the hub; code repository has no license [verified] | Lightest pipeline: whole-body keypoints through `rtmlib` on ONNX Runtime, then a small pose encoder and mT5-Base (582 M parameters). The pose script offers `cpu`, `cuda`, `mps`. The video-to-text script hardcodes CUDA and bfloat16 and needs patching. [verified] Whether it runs well here is not verified; Step 0.3 tests it. |
| SHuBERT with ByT5 | Yes: encoder, two DINOv2 crop models, and a ByT5 translation checkpoint, plus a complete video-to-text app (a Hugging Face Space) [verified] | Tagged MIT; no license file; bundles a YOLOv8 model whose license was not checked [verified, except YOLO] | Five model stages and fairseq; near-real-time is unlikely [inferred]. The translation checkpoint's training data and scores are undocumented [verified]. |
| SSVP-SLT | No How2Sign or YouTube-ASL checkpoints. Only weights trained on DailyMoth, one news signer [verified] | CC BY-NC 4.0 | Stated requirements are Linux and CUDA [verified]. Not suitable. |
| Qwen3-VL "sign-language-bridge" adapter | Yes | Apache-2.0 | Self-reported BLEU-4 of 1.64 [verified as self-reported]. Too weak. |
| SignMusketeers, YouTube-ASL baselines, others | None found [verified for SignMusketeers; unverified for the Google baselines] | — | — |

Published How2Sign test scores for context: Uni-Sign pose-only 14.5 BLEU; SSVP-SLT up to 15.5. [verified]

**Recommendation: Uni-Sign pose-only.** It is the only candidate with full translation checkpoints, a documented score, and a pipeline light enough for a laptop. Start from whichever of its two ASL checkpoints does better on the domain validation set; I expect the OpenASL one (news and vlogs) to be closer to conversational signing than the How2Sign one (instructional). [assumed]

**What this changes from the original brief:** keypoints come from RTMPose whole-body instead of MediaPipe Holistic, and the decoder is mT5-Base instead of T5 or mBART, because those are what the pretrained weights expect. There is no separate appearance-feature stream; the released ASL checkpoints are pose-only. [verified] The extraction, caching, and normalization rules from the brief still apply.

**If the user picks differently**

- *SHuBERT:* permissive tags, but a heavier pipeline with an undocumented translation checkpoint. Phase 1 becomes a five-stage extraction, live speed is at risk, and Step 0.3 becomes a larger job. This is the fallback if Uni-Sign fails Step 0.3 or Q3 rules it out.
- *Insist on MediaPipe:* there is no pretrained translator found that takes MediaPipe Holistic input, so this means training from scratch, which scores about 1 BLEU on this hardware and data [verified anchor]. Not compatible with the user's decisions.

**Decision (user, 2026-11-16):** Uni-Sign pose-only. SHuBERT stays as the fallback if Step 0.3 fails.

---

## Q3. Licensing — blocking

**Facts**

- Uni-Sign weights are tagged CC BY-NC 4.0 on the hub: non-commercial use with attribution. [verified]
- The Uni-Sign GitHub repository has no license file. [verified] Without a license, the default is that the authors keep all rights; there is no explicit grant to copy or modify the code. The authors publish it with instructions for running it, so running it privately matches their evident intent, but that is my reading and not a license. I am not giving legal advice.
- mT5-Base, `rtmlib`, and the RTMPose model files: licenses not verified in planning; Step 0.2 checks them.
- The recordings are the signer's own. If the signer is someone other than the user, get their consent for use and for showing them at the demo.
- The old prototype's weights (CC BY-NC-SA 4.0) are not reused.

**Recommendation**

1. Treat the project as a personal, non-commercial demo. A project show fits; selling it, or putting it in a product, does not.
2. Keep the Uni-Sign code out of this repository. Check it out locally at a pinned commit into a git-ignored folder and import from it, or write an independent loader for the weights if the model definition is simple enough (decided in Step 0.2 after reading the code). Do not publish the code, the weights, or the adapted weights.
3. Credit Uni-Sign, its pose estimator, and mT5 in the README and on the demo page.
4. Optional and worth doing: the user opens an issue on the Uni-Sign repository asking the authors to add a license. I will not contact anyone on the user's behalf.

**If the user picks differently**

- *Wants a clean license position:* use SHuBERT (MIT-tagged) instead, accepting the heavier pipeline, or wait for a reply from the Uni-Sign authors.
- *Wants to publish the repository:* possible for this project's own code as long as it contains none of Uni-Sign's code or weights and the README tells readers to obtain those themselves. Treat publication as a new stop point.
- *Commercial use:* not possible with these weights.

**Decision (user, 2026-11-16):** Uni-Sign accepted on the terms above: personal non-commercial demo, its code kept out of this repository and not redistributed. Each download still needs approval by name in Step 0.2.

---

## Q4. Compute — decided

**Decision (user, 2026-11-16):** this Mac only. No rented GPU.

Consequences already built into the plan: no from-scratch training; adapter-based fine-tuning; 32-bit floats on MPS or CPU; estimates in `PLAN.md` section 4 are replaced by measurements in Steps 0.3, 1.2, and 2.4. MediaPipe's and RTMPose's GPU paths are not relied on. Whether ONNX Runtime's CoreML provider speeds up pose estimation here is not verified and is measured in Step 0.3.

---

## Q5. Demo shell

**Recommendation: keep the Next.js app and rewrite the page** (`REUSE.md`, item 7). The camera flow, upload control, layout, and styles carry over and look presentable for a show.

**If the user picks a minimal page:** one static HTML file served by FastAPI; removes Node and one dev server; gives up the existing styling. Nothing before Phase 4 changes.

**Decision:** _pending (default applies if unanswered)_

---

## Q6. LLM polish

**Recommendation: none. The model's output is the system's output.** An LLM can turn a wrong translation into a fluent wrong one, it makes the scores measure the LLM, it needs a network connection at the show, and the old LLM translator is on the do-not-reuse list.

**If the user wants it later:** optional display-only step, off by default, on committed sentences only, with scores still computed on raw output and the raw text visible. It needs its own approval because it sends content to an external service.

**Decision:** _pending (default applies if unanswered)_

---

## Q7. Language scope

**Recommendation: ASL to English only.** Data formats carry language fields. The base model happens to help here: Uni-Sign uses a multilingual decoder (mT5) and was pretrained on Chinese and American sign languages [verified], so adding another sign language later means recording or finding data for it and fine-tuning the same architecture. No dataset for another language was looked for.

**Decision:** _pending (default applies if unanswered)_

---

## Q8. Webcam settings and sentence boundaries

**Recommendation**

- Request 1280×720 at 30 fps from the camera and accept what it gives, down to 640×480. Always use real timestamps.
- Model frame rate and maximum clip length: whatever the base model was trained with, read from its code in Step 0.2. [not verified in planning]
- Segments capped at 16 s; partial text every 1 s. [assumed]
- One signer, upper body with face and both hands in frame, plain background if possible.
- **Sentence ends:** the signer lowers the hands to a rest position between sentences, and a rest-position detector commits the sentence (`PLAN.md`, Step 3.2). It is checked against marked boundaries in recorded continuous takes.

Why a rule here, given that the old pause-based splitter is banned: the old one cut individual signs wherever hand motion dipped, and over-segmented. This one detects a deliberate rest posture between whole sentences, held for a minimum time, and is measured. A learned detector needs continuous training data that does not exist for this domain beyond what gets recorded; it stays an option if the rule misses its target.

**If the user picks differently**

- *No rule of any kind:* use a button or key press to end a sentence (reliable, less "continuous"), or record more continuous takes and train a small tagger (more recording and one more training step).
- *Longer segments:* depends on the base model's maximum length.

**Decision:** _pending (default applies if unanswered)_

---

## Q9. How the model is adapted

**Recommendation:** freeze the pretrained weights and train low-rank adapters on the decoder and the pose encoder, possibly with the pose encoder's small layers unfrozen (it has about 10 M parameters [verified]). Exact design is confirmed with the user before training code is written (`CLAUDE.md`).

Reasons [assumed until Step 2.4 measures]: full fine-tuning of 582 M parameters may not fit in 16 GB; about 500 clips is too little data to move all weights safely; adapters keep the base model intact, so the "before" and "after" can be compared and the adapter can be turned off.

**If the user picks differently:** full fine-tuning is tried only if memory allows and validation shows adapters are not enough.

**Decision:** _pending (default applies if unanswered)_

---

## Q10. Who signs, and how much can be recorded — decided

**Decision (user, 2026-11-16):** the user does not know ASL and will learn the domain sentences, record them, and present.

What this changes:

- **Smaller domain.** The drafted list is 103 sentences over 76 signs, with 20 of the sentences held out as never-trained test sentences, and 6 or more takes of each training sentence over at least 2 sessions. That is about 600 takes, roughly 3 hours of recording after the sentences are learned. [assumed]
- **Where the signing comes from matters.** ASL has its own grammar and word order; signing English words in English order is not ASL. Each sentence needs an ASL version from a reliable source (an ASL course or dictionary with video, or a fluent signer), written into `domain/sentences.yaml` before recording. Finding and checking those sources is the user's task; none was verified in planning.
- **Have a fluent signer review if at all possible**, even once, by video. Otherwise the model learns the user's version of each sign, and nobody can tell whether a fluent signer would be understood.
- **Learn first, record after.** Record only once each sentence is signed the same way every time; early inconsistent takes hurt more than they help. [assumed]
- **The pretrained model helps less.** It was trained on fluent signers, and a learner's signing will look different to it, so more of the demo's accuracy will come from the recordings and less from the base model. Memorizing the sentence list becomes more likely, and the never-trained test sentences will show whether that happened. [inferred]
- **Say it at the show.** The demo is tuned to one learner's signing of a fixed set of sentences. Presenting it as that is accurate and still a good demo.

The original analysis is kept below for reference.

The whole approach depends on this. I need to know:

1. **Does the user, or someone available, sign ASL** well enough to produce about 250 sentences consistently? The translations will only be as correct as the signing.
2. **Will the same person sign at the show?** The adapted model will be tuned to the people in the recordings.
3. **How much recording time is realistic?** The default needs about 1,000 to 1,250 takes of about 8 s each, over at least two sessions: roughly 3 to 5 hours including retakes. [assumed]

**Original recommendation (superseded by the decision above):** one fluent signer who will also present, recording about 250 sentences; a second signer if one is available.

**If the answers differ**

- *Less time:* shrink the domain (for example 100 sentences over 100 signs). Quality inside the smaller set should hold; coverage drops. [assumed]
- *No one who signs ASL is available:* adaptation cannot be done properly. The remaining honest demo is the unadapted pretrained model on pre-recorded clips of fluent signers, with the live camera as a secondary feature and low accuracy stated openly. That is a much weaker show piece, and I would want to discuss it before starting.
- *The signer uses a different sign language:* the same method applies to that language's data, starting from the same base model, but the starting quality is unknown.

---

## Q11. Test fixture clips — blocking (small)

Tests must run real extraction on real video, no dataset may be downloaded without approval, and the old demo clips have no recorded source or license.

**Recommendation:** the user records two webcam clips of about 5 s (hands visible; hands leaving the frame), kept local and git-ignored.

**If not:** tests before the domain recordings exist can only use synthetic video, which covers the no-detection path only, and Steps 0.3 and 1.1 lose most of their gate.

**Decision (user, 2026-11-16):** yes, the user records the two clips.

---

## Q12. What counts as success — blocking

**Proposed definition:** for sentences in the chosen domain, signed by the demo signer under show-like conditions, the live system produces a correct English sentence at least 80% of the time, with partial text appearing while signing and committed text within about 3 s of the sentence end. Sentences outside the recorded list but built from the same signs are translated correctly at least half the time. Outside the domain, or for another signer, no quality is promised, and the demo says so.

All three numbers are estimates with no published anchor. If the measured result is that only the exact recorded sentences work, the demo is still showable, but it must be presented as covering a fixed sentence list.

**Decision (user, 2026-11-16):** accepted as proposed. The user added a requirement: the demo must also accept video files as input, not only the webcam (see `PLAN.md`, Steps 3.3 and 4.1).

---

## Approved assets

Step 0.2 (2026-11-16) filled in the licenses, sizes, sources, and hashes below and put rows A1 to A7 to the user, who approved all seven on 2026-11-16 ("yes to all", relayed by the orchestrator), exactly as worded: these files, sources, and revisions, and this sparse-checkout path list. A8 was not asked and is not approved. No approved row, no download. License details and obligations are in `docs/DATA_LICENSES.md`.

Sizes and hashes were read from the hosts' metadata on 2026-11-16 [verified as reported by the host]. Each hash is checked again after download and the loader refuses a mismatch. The "On disk" column says what is present and checked as of the end of Step 0.2; files live under the git-ignored `assets/` and `third_party/` of the Track A worktree (`/Users/naymyokhant/Programming/Personal/Projects/s2e-a`). Other worktrees reach them by setting `S2E_ASSETS_ROOT` and `S2E_THIRD_PARTY_ROOT`, or the folders can be moved into the main checkout. The outstanding files are fetched by `scratch/download_large_assets.sh` (git-ignored, resumable, checks size and SHA-256), which the user runs because the downloads are expected to exceed 10 minutes.

| # | On disk (end of Step 0.2) |
| --- | --- |
| A1 | **Not yet.** `assets/unisign/openasl_pose_only_slt.pth`, by the script |
| A2 | **Not yet.** `assets/unisign/how2sign_pose_only_slt.pth`, by the script |
| A3 | Yes. `third_party/Uni-Sign/`, HEAD `eed438bcb49e30405cd6ccdfcccca330c134e830`, 59 files, 366,792 bytes, fetched as a blob-filtered sparse clone so excluded files were never transferred. It includes three stale `.pyc` files and a `.DS_Store` under `stgcn_layers/` that the repository tracks; they are not used |
| A4 | Yes. `assets/mt5-base/`. SHA-256: `spiece.model` (4,309,802 bytes) `ef78f86560d809067d12bac6c09f19a462cb3af3f54d2b8acbba26e1433125d6`; `config.json` `7ce360bdccf5b665dbb8c32f6912f1d8fd394aa553c133863283a16503d6d550`; `generation_config.json` `f5a1c7e2be8092018d8835128987edf0111637dd98e90599cc80310fef75d95a`; `tokenizer_config.json` `3be4bbd24c65845b1dc86186f3c7d1d61ac2e29154cb0d71133753562f8d1959`; `special_tokens_map.json` `09059cedc26bc46bc09a52f05b92d4922e11917e87f3b92059bb1a63a59ab2c4`. The host reports no hash for these small files, so these are the values as downloaded |
| A5 | **Not yet.** `assets/mt5-base/pytorch_model.bin`, by the script |
| A6 | **Half.** Detector: yes, `assets/rtmlib/lightweight/yolox_tiny/`; archive SHA-256 matches the mirror-reported value; extracted `yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx` is 20,283,006 bytes, SHA-256 `ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250`. Pose model: **not yet**, 19.9 MB of 120.0 MB is on disk as a partial file that must not be used; the script resumes it. Its `.onnx` is extracted and hashed at the start of Step 0.3 |
| A7 | **Not yet.** Both archives by the script, into `assets/rtmlib/performance/`. Mirror-reported archive SHA-256: `a000224fd8ba283202bc62d4a5fcdfe353adb9f468777dbac1ea2ada2093adde` (`yolox_m`) and `a87e1af41a0a067776dba7d46e1c21c8f6e9f18e247e0e606718dd1f31e96ffd` (`rtmw-dw-x-l`) |

| # | Asset | Purpose | License (URL, date read) | Size | Source and pin | Approved by user (date) |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | Uni-Sign `openasl_pose_only_slt.pth` | Base model candidate | CC BY-NC 4.0 (https://huggingface.co/ZechengLi19/Uni-Sign, 2026-11-16) | 1,186,924,838 bytes (1.19 GB) | https://huggingface.co/ZechengLi19/Uni-Sign/resolve/eab251b7fe7e8521afc0e67be98add670ea40a0d/openasl_pose_only_slt.pth, SHA-256 `f836ea66bc837bbe6ed717a4b9bece87875f03ef96d4bf1092ca3dd767982798` | approved by user 2026-11-16 |
| A2 | Uni-Sign `how2sign_pose_only_slt.pth` | Base model candidate, reproduction check | CC BY-NC 4.0 (same page, 2026-11-16) | 1,186,925,007 bytes (1.19 GB) | same revision, `how2sign_pose_only_slt.pth`, SHA-256 `1bfd5f3312f04e4736f0a52f4ef9535916e6de9676a2a0d00c708748683fb00d` | approved by user 2026-11-16 |
| A3 | Uni-Sign code, sparse checkout at commit `eed438bcb49e30405cd6ccdfcccca330c134e830` into `third_party/Uni-Sign/`: `models.py`, `config.py`, `datasets.py`, `utils.py`, `deformable_attention_2d.py`, `stgcn_layers/`, `demo/` (scripts and the bundled, patched `rtmlib` 0.0.13). Excludes `data/`, `out/`, and everything else. | Model definition and reference pipeline | **None declared** (https://api.github.com/repos/ZechengLi19/Uni-Sign reports `license: null`, 2026-11-16); bundled `rtmlib` copy is Apache-2.0 | about 0.4 MB | https://github.com/ZechengLi19/Uni-Sign | approved by user 2026-11-16 |
| A4 | `google/mt5-base` tokenizer and configuration: `spiece.model`, `config.json`, `generation_config.json`, `tokenizer_config.json`, `special_tokens_map.json` | Tokenizer and model configuration required by the checkpoints | Apache-2.0 (https://huggingface.co/google/mt5-base, 2026-11-16) | 4.3 MB | https://huggingface.co/google/mt5-base at revision `2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f` | approved by user 2026-11-16 |
| A5 | `google/mt5-base` weights `pytorch_model.bin` | Needed to run the reference code unmodified; every value is then overwritten by A1 or A2 (`docs/base_model_notes.md`, section 3) | Apache-2.0 (same page, 2026-11-16) | 2,329,735,129 bytes (2.33 GB) | same revision, SHA-256 `180573b534144580f04af026da62bf71bc976ee1b7eb311b8945e2fefde8d614` | approved by user 2026-11-16 |
| A6 | Pose model files, `rtmlib` "lightweight" mode (the Uni-Sign demo default): `yolox_tiny_8xb8-300e_humanart-6f3252f9.zip` and `rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip`, each holding one `.onnx` | Keypoints | No license stated for the files; published by MMPose, which is Apache-2.0 (https://github.com/open-mmlab/mmpose/blob/main/LICENSE, 2026-11-16); mirror tagged Apache-2.0 (https://huggingface.co/Tau-J/RTMPose, 2026-11-16) | 18,887,737 and 120,002,068 bytes (139 MB together) | https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip and https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip. Archive SHA-256 as reported by the mirror: `36e09ca555916253fa1b5d51bec01e48e98e1f0af966c51ce3e844d8c8fc4cfc` and `a6d08a575c9d8b8da7ce6c13e1c7341047fca35594a6d35de04b739eba51139c`. The SHA-256 of each extracted `.onnx` is recorded after download. | approved by user 2026-11-16 |
| A7 | Pose model files, `rtmlib` "performance" mode: `yolox_m_8xb8-300e_humanart-c2c7a14a.zip` and `rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip` | Optional. Lets Step 0.3 compare against the larger estimator, because which estimator produced the training keypoints is not settled (`docs/base_model_notes.md`, section 2.1) | as A6 | 94,223,081 and 213,433,855 bytes (308 MB together) | https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip and https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip. Hashes recorded after download. | approved by user 2026-11-16 |
| A8 | How2Sign keypoints from the Uni-Sign hub repository (`how2sign_pose_format.zip.00` to `.03`; all splits in one archive) and How2Sign test references | Reproduction check in Step 2.1 | Hub card CC BY-NC 4.0; derived from How2Sign, CC BY-NC 4.0 | 8,465,474,575 bytes (8.46 GB) for the keypoints | https://huggingface.co/ZechengLi19/Uni-Sign | **not requested in Step 0.2**; this is Step 2.1's stop point. Note that it is 8.46 GB, not a test-split-only file. |

Total for A1 to A7: about 5.16 GB, of which about 0.04 GB was on disk at the end of Step 0.2.

## Candidate dependency versions

Seen on PyPI on 2026-11-16 [verified]. Candidates only; Step 0.1 reads the index again.

| Package | Version | Notes |
| --- | --- | --- |
| `torch` | 2.15.0 | Python 3.11 or later; macOS arm64 wheel needs macOS 14 or later. MPS operator coverage not checked. |
| `transformers` | 5.23.0 | Python 3.11 or later. Apache-2.0. |
| `sacrebleu` | 2.6.0 | Apache-2.0. |
| `rouge-score` | 0.1.2 | Apache license classifier; source distribution only. |
| `rtmlib`, `onnxruntime`, an adapter library | not checked | Step 0.1 and 0.2. Uni-Sign's code may need older versions than the latest; pin what works and record why. |

`mediapipe` is no longer a dependency under the recommended base model. Plan default Python: 3.12. [assumed]

## Verification log

Read on 2026-11-16. Paper facts were read from arXiv PDFs. Web pages were read through a fetch tool that returns summaries with quoted fragments. Research was done by delegated sub-sessions whose reports I reviewed; I did not re-open each source myself.

### Pretrained models

| Claim | Source | Status |
| --- | --- | --- |
| Uni-Sign hub files, sizes, release note, CC BY-NC 4.0 tag | https://huggingface.co/ZechengLi19/Uni-Sign/tree/main and https://github.com/ZechengLi19/Uni-Sign | verified |
| Uni-Sign code has no license | https://github.com/ZechengLi19/Uni-Sign | verified |
| Uni-Sign input (RTMPose via `rtmlib`), mT5-Base, device options, CUDA and bfloat16 hardcoded in the demo script | https://github.com/ZechengLi19/Uni-Sign/blob/main/demo/README.md and repo files | verified. Frame rate, resolution, normalization details: not found; Step 0.2 |
| Uni-Sign How2Sign results (14.5 pose-only, 14.9 pose plus RGB), model sizes, pretraining data | https://arxiv.org/pdf/2501.15187 | verified. BLEU implementation: not verified |
| SHuBERT weights, pipeline, MIT tag, undocumented translation checkpoint, Space | https://github.com/ShesterG/SHuBERT, https://huggingface.co/ShesterG/SHuBERT, https://huggingface.co/spaces/ShesterG/TTIC-SHuBERT-ASLVideo-to-EnglishText | verified. YOLOv8 license: not checked |
| SSVP-SLT: no How2Sign or YouTube-ASL checkpoints; DailyMoth SONAR weights only; Linux and CUDA requirements; CC BY-NC 4.0 | https://github.com/facebookresearch/ssvp_slt (README, INSTALL.md, examples/sonar) | verified |
| SignMusketeers repository returns 404 | project page link | verified |
| Google YouTube-ASL baseline checkpoints not released | — | not verified (directory not opened) |
| `sign-language-bridge` adapter: Apache-2.0, self-reported BLEU-4 1.64 | https://huggingface.co/mamounyosef/sign-language-bridge | verified as self-reported |
| Licenses of `google/mt5-base`, `rtmlib`, RTMPose ONNX files | see "Step 0.2" below | checked in Step 0.2 |

### Step 0.2 (2026-11-16, Track A)

Read directly with `curl` from the hosts' raw-file and metadata endpoints; nothing was saved to disk. Line-level findings are in `docs/base_model_notes.md`; license terms are in `docs/DATA_LICENSES.md`.

| Claim | Source | Status |
| --- | --- | --- |
| Uni-Sign weights are CC BY-NC 4.0 (model card front matter `license: cc-by-nc-4.0`; no license file in the hub repository); hub revision `eab251b7fe7e8521afc0e67be98add670ea40a0d`; file sizes and SHA-256 values in "Approved assets" | https://huggingface.co/ZechengLi19/Uni-Sign/raw/main/README.md, https://huggingface.co/api/models/ZechengLi19/Uni-Sign?blobs=true | verified |
| Uni-Sign code still has no license (`license: null`; license endpoint 404; no license file in the tree); latest commit `eed438bcb49e30405cd6ccdfcccca330c134e830`, 2026-02-11 | https://api.github.com/repos/ZechengLi19/Uni-Sign, `/license`, `/commits`, `/git/trees/eed438b…?recursive=1` | verified |
| The Uni-Sign code repository contains dataset label files (`data/How2Sign`, `data/OpenASL`, `data/CSL_Daily`, `data/WLASL`) and prediction and reference dumps (`out/`) | same file tree | verified (listed, not opened) |
| `google/mt5-base` is Apache-2.0; revision `2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f`; `pytorch_model.bin` 2.33 GB | https://huggingface.co/google/mt5-base/raw/main/README.md, https://huggingface.co/api/models/google/mt5-base?blobs=true | verified |
| `rtmlib` is Apache-2.0 | https://github.com/Tau-J/rtmlib/blob/main/LICENSE (GitHub API `spdx_id: Apache-2.0`); license file in the installed 0.0.16 package | verified |
| RTMPose and YOLOX ONNX files: no license stated for the files; publisher MMPose is Apache-2.0; mirror `Tau-J/RTMPose` tagged Apache-2.0; sizes in "Approved assets" | https://github.com/open-mmlab/mmpose/blob/main/LICENSE, https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/README.md, https://huggingface.co/api/models/Tau-J/RTMPose, HTTP headers of the download URLs | verified. That Apache-2.0 covers the files: inferred. Terms of the 14 training datasets: not read |
| The copy of `rtmlib` bundled with Uni-Sign is upstream 0.0.13 plus one change (keep the largest detection box) | in-memory diff of `demo/rtmlib-main/rtmlib/…` against https://raw.githubusercontent.com/Tau-J/rtmlib/0.0.13/rtmlib/… and against the installed 0.0.16 | verified |
| Keypoints used, normalization, 0.3 score threshold, 256-frame default, no frame rate, 4 beams and 100 new tokens, prompt text, checkpoint format | Uni-Sign source at the commit above (`datasets.py`, `models.py`, `utils.py`, `fine_tuning.py`, `demo/`) | verified; see `docs/base_model_notes.md` |
| The paper's keypoints came from "RTMPose-x from MMPose"; BLEU is from SacreBLEU; pose encoder 4.5 M and mT5-Base 582.4 M parameters | https://arxiv.org/html/2501.15187 (Appendix A.1, "Evaluation metrics" paragraph, Table 12) | verified |
| The repository computes BLEU with `raw_corpus_bleu` from a vendored sacreBLEU 1.4.2 (no tokenization) | `SLRT_metrics.py:26-43`, `external_metrics/sacrebleu.py:43` | verified. Closes "BLEU implementation: not verified" above |
| `decord` 0.6.0 has no macOS arm64 wheel; `deepspeed` 0.20.0 is source-only | https://pypi.org/pypi/decord/json, https://pypi.org/pypi/deepspeed/json | verified |
| Which pose estimator produced the How2Sign and OpenASL training keypoints; their native frame rates; the `max_length` used for the two ASL checkpoints; whether `transformers` 5.23.0 tokenizes the prompt like 4.40.0; whether the checkpoints are stored in bfloat16 | — | **not verified**; first three are open, last two are Step 0.3 checks |
| The `yolox_tiny` detector file has suppression built in (outputs `dets [1, N, 5]`, `labels [1, N]`), so `rtmlib` 0.0.13 and 0.0.16 post-process it the same way | the downloaded file, opened with ONNX Runtime on CPU, 2026-11-16 | verified |
| Download rate on this network, 2026-11-16 23:13 to 23:27: 440 to 480 kB/s on three files (4.3 MB, 18.9 MB, and the first minutes of 120 MB), then 27 to 46 kB/s for several minutes with dropped connections | `curl` transfer statistics | measured |

User decisions recorded in Step 0.2 (2026-11-16, relayed by the orchestrator in the user's words "yes to all: A items 1–8"): asset rows A1 to A7 approved; and the helper that loads the Uni-Sign reference code may supply empty placeholder modules for its unused `decord` and `deepspeed` imports.

### Datasets

| Claim | Source | Status |
| --- | --- | --- |
| How2Sign license CC BY-NC 4.0; direct downloads; sizes | https://how2sign.github.io/ | verified |
| How2Sign statistics, ten instructional categories, about 20% fingerspelling, test split shares signers with train except 26 videos | https://arxiv.org/pdf/2008.08143 | verified |
| OpenASL: CC BY-NC-ND 4.0, links to YouTube videos, 288 h | https://github.com/chevalierNoir/OpenASL, https://arxiv.org/pdf/2205.12870 | verified |
| YouTube-ASL: IDs under CC BY 4.0, videos under YouTube's terms, 984 h | https://arxiv.org/pdf/2306.15162 | verified |
| YouTube Terms of Service restrict downloading and automated access | https://www.youtube.com/t/terms | verified |
| No conversational ASL sentence dataset found | hub and web search | not exhaustive |

### Published results (How2Sign test), used as context

| Claim | Source | Status |
| --- | --- | --- |
| Keypoints into T5, How2Sign only: 1.22; with YouTube-ASL pretraining: 12.39 | https://arxiv.org/pdf/2306.15162 (Table 3) | verified |
| SignMusketeers How2Sign-only: 2.4 | https://arxiv.org/pdf/2406.06907 | verified |
| SSVP-SLT: 7.0 to 15.5 depending on pretraining | https://arxiv.org/pdf/2402.09611 (Table 2) | verified |
| Tarrés et al.: 8.03 | https://arxiv.org/pdf/2304.06371 | verified |

### Streaming prior work

| Claim | Source | Status |
| --- | --- | --- |
| Moryossef et al. 2023: pose-based tagger for sign and phrase segmentation, MIT code | https://arxiv.org/pdf/2310.13960 | verified |
| SimulSLT 2021: wait-k decoding with a boundary predictor | https://arxiv.org/pdf/2112.04228 | verified |
| Doan 2026: chunked streaming with a sentence-boundary state machine, mean latency 1.354 to 1.873 s | https://arxiv.org/abs/2607.09611 | abstract only |
| ReSCUE 2026: re-translation with sentence commitment | https://arxiv.org/abs/2610.03022 | abstract only |

### Measured locally in the planning session

- Machine: Apple M5, 16 GB memory, 100 GB free, system Python 3.14.6, Node 26.4.0; `uv` and `ffmpeg` not installed.
- Old repo contents and line references in `REUSE.md`.

### Assumed (no source)

All speed, memory, and time estimates in `PLAN.md` section 4; every metric target in Phases 2 to 4 for the domain data; that the OpenASL checkpoint suits conversation better than the How2Sign one; that adapters are sufficient; recording time; that a smaller domain keeps its quality.
