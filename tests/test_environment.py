"""Step 0.1 gate: a real tensor operation on MPS and a real ONNX Runtime session on CPU.

Neither test is skipped when the capability is missing: this machine is the only target,
so a missing MPS backend is a failure to report, not a condition to tolerate.
"""

import os
from pathlib import Path

import numpy as np


def test_tensor_operation_on_mps() -> None:
    import torch

    assert torch.backends.mps.is_built(), "this torch build has no MPS support"
    assert torch.backends.mps.is_available(), "MPS is not available on this machine"

    generator = torch.Generator().manual_seed(0)
    a = torch.randn(64, 32, generator=generator, dtype=torch.float32)
    b = torch.randn(32, 16, generator=generator, dtype=torch.float32)
    on_mps = a.to("mps") @ b.to("mps")

    assert on_mps.device.type == "mps"
    assert on_mps.dtype == torch.float32
    torch.testing.assert_close(on_mps.cpu(), a @ b, rtol=1e-4, atol=1e-4)


def test_onnxruntime_session_on_cpu(tmp_path: Path) -> None:
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    # y = x @ W + b, built here so that no model file is downloaded or committed.
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 3)).astype(np.float32)
    b = rng.standard_normal((3,)).astype(np.float32)
    graph = helper.make_graph(
        nodes=[helper.make_node("Gemm", ["x", "w", "b"], ["y"])],
        name="affine",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 4])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n", 3])],
        initializer=[
            helper.make_tensor("w", TensorProto.FLOAT, w.shape, w.flatten().tolist()),
            helper.make_tensor("b", TensorProto.FLOAT, b.shape, b.tolist()),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    path = tmp_path / "affine.onnx"
    path.write_bytes(model.SerializeToString())

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert session.get_providers() == ["CPUExecutionProvider"]

    x = rng.standard_normal((5, 4)).astype(np.float32)
    (y,) = session.run(None, {"x": x})
    assert y.dtype == np.float32
    np.testing.assert_allclose(y, x @ w + b, rtol=1e-5, atol=1e-5)


def test_model_hubs_are_offline_by_default() -> None:
    import signtoenglish  # noqa: F401

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    import huggingface_hub.constants as hub_constants

    assert hub_constants.HF_HUB_OFFLINE is True


def test_pinned_libraries_import() -> None:
    """Every pinned runtime library imports in this environment (no model is loaded)."""
    import av
    import cv2
    import fastapi
    import peft
    import rouge_score.rouge_scorer
    import rtmlib
    import sacrebleu
    import sentencepiece
    import transformers
    import uvicorn
    import websockets

    for module in (av, cv2, fastapi, peft, rtmlib, sacrebleu, sentencepiece, transformers):
        assert module.__name__
    assert uvicorn.__version__ and websockets.__name__ and rouge_score.rouge_scorer.RougeScorer
