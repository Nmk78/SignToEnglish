# Licenses of models, code, and data

Started in Step 0.2, 2026-11-16. Every license below was read on that date at the URL given. **[verified]** means read at that primary source; anything else is labelled. I am not giving legal advice.

Whether an asset may be downloaded is recorded in `OPEN_QUESTIONS.md`, "Approved assets". This file records the terms.

**Standing position (user decision, Q3):** this is a personal, non-commercial demo. No hosted public demo, no paid wrapper, no redistribution of base or adapted weights, of third-party code, or of dataset material.

## Pretrained models and model files

| Asset | License | Where it was read | Obligations and notes |
| --- | --- | --- | --- |
| Uni-Sign weights `openasl_pose_only_slt.pth`, `how2sign_pose_only_slt.pth` | CC BY-NC 4.0, declared by the model card front matter (`license: cc-by-nc-4.0`). The repository has no separate license file. [verified] | https://huggingface.co/ZechengLi19/Uni-Sign/raw/main/README.md and https://huggingface.co/api/models/ZechengLi19/Uni-Sign (revision `eab251b7fe7e8521afc0e67be98add670ea40a0d`) | Non-commercial use, attribution. They were trained on OpenASL (CC BY-NC-ND 4.0) and How2Sign (CC BY-NC 4.0); how OpenASL's no-derivatives term bears on a model trained on it is not something I can settle. Not redistributed by this project. |
| Uni-Sign code | **No license.** The GitHub API reports `license: null` and the license endpoint returns 404; there is no license file in the tree at commit `eed438b`. [verified] | https://api.github.com/repos/ZechengLi19/Uni-Sign and its file tree | All rights stay with the authors. Used locally only, in git-ignored `third_party/`, never copied into `src/`, never committed or published (Q3). Parts carry other notices: `utils.py` is marked as modified from Facebook's DeiT with "All rights reserved" (`utils.py:1-7`), `external_metrics/sacrebleu.py` is Apache-2.0 (Amazon), and `demo/rtmlib-main/` is Apache-2.0 with its own `LICENSE`. |
| `google/mt5-base` (weights, tokenizer, configuration) | Apache-2.0, declared by the model card front matter. [verified] | https://huggingface.co/google/mt5-base/raw/main/README.md and https://huggingface.co/api/models/google/mt5-base (revision `2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f`) | Attribution and license notice. |
| `rtmlib` 0.0.16 (code, installed from PyPI in Step 0.1) | Apache-2.0. [verified] | https://github.com/Tau-J/rtmlib/blob/main/LICENSE; the same file ships in the installed package under `rtmlib-0.0.16.dist-info/licenses/LICENSE` | Attribution and license notice. |
| RTMPose and YOLOX ONNX files that `rtmlib` uses (`rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122`, `yolox_tiny_8xb8-300e_humanart-6f3252f9`, and the larger `rtmw-dw-x-l` and `yolox_m` files) | **No license is stated for the model files themselves.** The repository that publishes them, MMPose, is Apache-2.0, and its RTMPose project page has no separate terms for weights (searched for "licen": no match). A mirror kept by the `rtmlib` author is tagged Apache-2.0. [verified for those three facts; that Apache-2.0 covers the files is inferred] | https://github.com/open-mmlab/mmpose/blob/main/LICENSE; https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/README.md; https://huggingface.co/Tau-J/RTMPose (revision `cd4d7095f5cfc9cfc4f46289bee91ea4a1e1d9fd`) | The pose model was trained on 14 public datasets ("Cocktail14": AI Challenger, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18, COCO-WholeBody, UBody, Human-Art, WFLW, 300W, COFW, LaPa, InterHand) and the detector on Human-Art [verified in the RTMPose README]. Those datasets' own terms were **not read**; several are commonly research-only [assumed]. This does not change anything for a non-commercial demo, but it is one more reason the project cannot go commercial. |

## Datasets

No dataset has been downloaded. Terms as verified in planning (`OPEN_QUESTIONS.md`, "Verification log") unless dated here.

| Dataset | License | Use in this project |
| --- | --- | --- |
| How2Sign | CC BY-NC 4.0 (https://how2sign.github.io/) | Only for the optional reproduction check in Step 2.1, which has its own stop point. Never redistributed, never in fixtures or docs. |
| How2Sign keypoints published by the Uni-Sign authors (`how2sign_pose_format.zip.00`–`.03`, 8.46 GB) | Hosted in the Uni-Sign hub repository, whose card says CC BY-NC 4.0; derived from How2Sign, so How2Sign's terms apply as well [inferred] | Same as above. Not approved, not downloaded. |
| OpenASL | CC BY-NC-ND 4.0 | Not used directly. One base checkpoint was trained on it. |
| Dataset text inside the Uni-Sign code repository (`data/*/labels.*`, `out/eval_*`) | The datasets' own licenses | Excluded from the proposed checkout. Not read, not downloaded. |
| Domain recordings made for this project | Belong to the signer | Local, git-ignored, used only for this project unless the signer agrees otherwise. Manifest `license` value: `own-recording`. |

## Terms that models trained here inherit

An adapted model combines the Uni-Sign weights (CC BY-NC 4.0) with adapters trained on the signer's recordings. It is non-commercial, needs attribution to Uni-Sign, and is not redistributed. Each run's `model_card.md` repeats this (`CLAUDE.md`, "License rules").

## Attributions to show in the README and on the demo page

- Uni-Sign: Li, Zhou, Zhao, Wu, Hu, Li. "Uni-Sign: Toward Unified Sign Language Understanding at Scale", ICLR 2025, arXiv:2501.15187. Weights under CC BY-NC 4.0. https://github.com/ZechengLi19/Uni-Sign
- RTMPose and RTMW whole-body pose models from MMPose (OpenMMLab), run through `rtmlib` (author listed as Tau-J), both Apache-2.0. https://github.com/open-mmlab/mmpose, https://github.com/Tau-J/rtmlib
- mT5-Base: Xue et al., "mT5: A massively multilingual pre-trained text-to-text transformer", NAACL 2021. Apache-2.0. https://huggingface.co/google/mt5-base
