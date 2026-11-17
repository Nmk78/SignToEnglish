"""Runs the Uni-Sign REFERENCE code from third_party/ in its own process. Test support only.

Nothing under src/ imports this. The parity tests call it with `subprocess` so that the
reference's top-level module names (`models`, `datasets`, `config`, `utils`) and its bundled,
patched rtmlib 0.0.13 never share a process with the project's own code and rtmlib 0.0.16.

Subcommands (all paths are arguments; nothing is downloaded):

  extract   <uni_sign_dir> <video> <det.onnx> <pose.onnx> <out.npz>
      demo/pose_extraction.py's procedure with the bundled rtmlib: cv2.VideoCapture frames,
      Wholebody in "lightweight" sizes, keypoints divided by [W, H].
  normalize <uni_sign_dir> <in.npz> <out.npz>
      datasets.load_part_kp on keypoints already divided by [W, H].
  translate <uni_sign_dir> <mt5_dir> <checkpoint.pth> <dataset_name> <out.json> <in.npz>...
      Uni_Sign on CPU in float32, 4 beams, 100 new tokens, as demo/online_inference.py.

As approved by the user (2026-11-16), empty placeholder modules stand in for the reference's
unused `decord` and `deepspeed` imports. mT5 is built from config.json (no weight file is
needed: the checkpoint supplies every tensor, which is checked here on every load), and its
output layer is separated from the embedding before loading, because transformers 5.23.0
joins them and these checkpoints do not (docs/feasibility.md, section 2).
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PARTS = ("body", "left", "right", "face_all")


def _placeholders() -> None:
    decord = types.ModuleType("decord")
    decord.VideoReader = None  # type: ignore[attr-defined]
    decord.cpu = None  # type: ignore[attr-defined]
    deepspeed = types.ModuleType("deepspeed")
    comm = types.ModuleType("deepspeed.comm")
    acc = types.ModuleType("deepspeed.accelerator")
    acc.get_accelerator = None  # type: ignore[attr-defined]
    deepspeed.comm = comm  # type: ignore[attr-defined]
    deepspeed.accelerator = acc  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "decord": decord,
            "deepspeed": deepspeed,
            "deepspeed.comm": comm,
            "deepspeed.accelerator": acc,
        }
    )


def cmd_extract(uni_sign: str, video: str, det: str, pose: str, out: str) -> None:
    sys.path.insert(0, str(Path(uni_sign) / "demo" / "rtmlib-main"))
    import cv2
    import numpy as np
    import rtmlib
    from rtmlib import Wholebody

    assert "third_party" in rtmlib.__file__ and "rtmlib-main" in rtmlib.__file__, rtmlib.__file__
    from rtmlib.version import __version__ as rtmlib_version

    wholebody = Wholebody(
        det=det,
        det_input_size=(416, 416),
        pose=pose,
        pose_input_size=(192, 256),
        to_openpose=False,
        backend="onnxruntime",
        device="cpu",
    )
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), video
    kps, scs = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = np.uint8(frame)
        k, s = wholebody(frame)  # demo/pose_extraction.py:11-15
        h, w, _ = frame.shape
        kps.append(k / np.array([w, h])[None, None])  # demo/pose_extraction.py:45
        scs.append(s)
    cap.release()
    np.savez(
        out,
        keypoints=np.stack(kps),
        scores=np.stack(scs),
        width=w,
        height=h,
        rtmlib_version=rtmlib_version,
        n_people=np.array([len(k) for k in kps]),
    )


def cmd_normalize(uni_sign: str, src: str, out: str) -> None:
    import numpy as np

    _placeholders()
    sys.path.insert(0, uni_sign)
    import datasets as us_datasets  # the reference's datasets.py

    assert Path(us_datasets.__file__).resolve().is_relative_to(Path(uni_sign).resolve())
    d = np.load(src)
    kps, scs = d["keypoints"], d["scores"]  # [T, 1, 133, 2] divided by [W, H]; [T, 1, 133]
    parts = us_datasets.load_part_kp(list(kps), list(scs), force_ok=True)
    np.savez(out, **{k: v.numpy() for k, v in parts.items()})


def cmd_translate(
    uni_sign: str, mt5_dir: str, ckpt: str, dataset: str, out: str, *inputs: str
) -> None:
    import numpy as np
    import torch
    from transformers import MT5Config, MT5ForConditionalGeneration

    _placeholders()
    sys.path.insert(0, uni_sign)
    import config as us_config

    us_config.mt5_path = mt5_dir
    import models as us_models

    assert Path(us_models.__file__).resolve().is_relative_to(Path(uni_sign).resolve())
    us_models.mt5_path = mt5_dir
    original = us_models.MT5ForConditionalGeneration.from_pretrained
    us_models.MT5ForConditionalGeneration.from_pretrained = staticmethod(
        lambda path, *a, **k: MT5ForConditionalGeneration(
            MT5Config.from_pretrained(path, local_files_only=True)
        )
    )
    try:
        args = types.SimpleNamespace(
            hidden_dim=256, dataset=dataset, rgb_support=False, label_smoothing=0.2
        )
        model = us_models.Uni_Sign(args=args)
    finally:
        us_models.MT5ForConditionalGeneration.from_pretrained = original
    mt5 = model.mt5_model
    mt5.config.tie_word_embeddings = False
    mt5.lm_head.weight = torch.nn.Parameter(torch.empty_like(mt5.shared.weight))
    state = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(state, strict=True)
    differ = [k for k, v in model.state_dict().items() if not torch.equal(v, state[k].to(v.dtype))]
    if differ:
        raise SystemExit(f"model tensors differ from the checkpoint after load: {differ[:5]}")
    del state
    model.eval().float()

    results = []
    with torch.no_grad():
        for path in inputs:
            d = np.load(path)
            src = {p: torch.from_numpy(d[p]).to(torch.float32)[None] for p in PARTS}
            t = src["body"].shape[1]
            src["attention_mask"] = torch.ones(1, t, dtype=torch.long)
            stack = model(src, {"gt_sentence": [""], "gt_gloss": [""]})
            ids = model.generate(stack, max_new_tokens=100, num_beams=4)
            text = model.mt5_tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
            results.append(
                {"input": Path(path).name, "frames": t, "text": text, "ids": ids[0].tolist()}
            )
    with Path(out).open("w", encoding="utf-8") as fh:
        json.dump(results, fh)


if __name__ == "__main__":
    {"extract": cmd_extract, "normalize": cmd_normalize, "translate": cmd_translate}[sys.argv[1]](
        *sys.argv[2:]
    )
