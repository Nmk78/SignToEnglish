"""Overlay drawing and prompt loading. Opening the camera and the window is not tested
here: that needs a camera and a person, and is covered by the user's recording trial."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from signtoenglish.common.config import ConfigError
from signtoenglish.record.camera import render_overlay
from signtoenglish.record.prompts import load_prompts
from signtoenglish.record.recorder import Overlay


def test_overlay_is_drawn_on_a_mirrored_copy_and_the_frame_is_untouched(tools) -> None:
    image = tools.frame_image(3)
    before = image.copy()
    overlay = Overlay(
        "recording", "t01 take 1", "Hello, how are you?", ("ASL: PLACEHOLDER",), keys="SPACE stop",
        elapsed_ms=1234, notice="note",
    )  # fmt: skip
    drawn = render_overlay(image, overlay)
    assert np.array_equal(image, before)
    assert drawn.shape == image.shape and drawn.dtype == np.uint8
    middle = slice(70, 150)  # rows the text bands do not cover
    assert np.array_equal(drawn[middle], image[middle, ::-1])
    assert not np.array_equal(drawn, image[:, ::-1])  # the bands and text are there

    big = render_overlay(image, Overlay("countdown", "h", "text", big="3"))
    assert not np.array_equal(big[middle], image[middle, ::-1])  # the digit is in the middle


@pytest.mark.parametrize("state", ["prompt", "countdown", "recording", "review"])
def test_overlay_handles_long_and_non_ascii_text_at_camera_size(state: str) -> None:
    image = np.full((720, 1280, 3), 90, dtype=np.uint8)
    overlay = Overlay(
        state,  # type: ignore[arg-type]
        "heading " * 10,
        "I am happy to meet your family — café " * 6,
        tuple(f"{i}. a fairly long listed sentence for a continuous take" for i in range(5)),
        keys="A accept   R redo   P replay   N skip   Q quit (drops this take)",
        notice="Saved.",
    )
    assert render_overlay(image, overlay).shape == image.shape


def test_prompts_from_the_committed_list(repo_root: Path) -> None:
    prompts = load_prompts(repo_root / "domain" / "sentences.yaml")
    assert (prompts.sign_language, prompts.text_language) == ("ase", "en")
    assert len(prompts.prompts) == 103 and prompts.n_without_asl == 103
    first = prompts.prompts[0]
    assert (first.id, first.text, first.role, first.asl) == (
        "t01", "Hello, how are you?", "train", None,
    )  # fmt: skip
    assert sum(p.role == "unseen_test" for p in prompts.prompts) == 20


def test_prompts_match_the_validated_sentence_list(repo_root: Path) -> None:
    """Two readers of one file must agree on ids, text, and roles."""
    from signtoenglish.data.domain import load_domain

    path = repo_root / "domain" / "sentences.yaml"
    assert [(p.id, p.text, p.role) for p in load_prompts(path).prompts] == [
        (s.id, s.text, s.role) for s in load_domain(path).sentences
    ]


def test_prompt_loader_rejects_unusable_lists(tmp_path: Path) -> None:
    head = "sign_language: ase\ntext_language: en\nsentences:\n"
    cases = {
        "duplicate": '  - {id: a, role: train, text: "x"}\n  - {id: a, role: train, text: "y"}\n',
        "no text": "  - {id: a, role: train, text: ''}\n",
        "file name": '  - {id: "a/b", role: train, text: "x"}\n',
        "role": '  - {id: a, role: test, text: "x"}\n',
        "non-empty": "  []\n",
    }
    for expected, body in cases.items():
        path = tmp_path / "s.yaml"
        path.write_text(head + body)
        with pytest.raises(ConfigError, match=expected):
            load_prompts(path)
    path.write_text("sentences: []\n")
    with pytest.raises(ConfigError, match="missing key"):
        load_prompts(path)
