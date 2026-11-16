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


def test_recorder_refuses_a_sentence_list_with_errors(tmp_path: Path) -> None:
    head = (
        "domain: d\nsign_language: ase\ntext_language: en\nvocabulary:\n  - {id: A}\nsentences:\n"
    )
    line = '  - {{id: {id}, role: {role}, text: "{text}", signs: [{signs}]}}\n'
    good = [
        line.format(id=f"a{i}", role="train", text=f"Sentence {i}.", signs="A") for i in range(3)
    ]
    cases = {
        "duplicate_id": [*good, line.format(id="a0", role="train", text="Other.", signs="A")],
        "duplicate_text": [
            *good,
            line.format(id="b", role="unseen_test", text="sentence 1", signs="A"),
        ],
        "unknown_sign": [*good, line.format(id="b", role="train", text="Other.", signs="ZZ")],
        "coverage": [good[0], line.format(id="b", role="unseen_test", text="Other.", signs="A")],
        "file name": [*good, line.format(id='"a/b"', role="train", text="Other.", signs="A")],
        "role": [*good, line.format(id="b", role="test", text="Other.", signs="A")],
    }
    path = tmp_path / "s.yaml"
    for expected, body in cases.items():
        path.write_text(head + "".join(body))
        with pytest.raises(ConfigError, match=expected):
            load_prompts(path)
    path.write_text(head + "".join(good))
    prompts = load_prompts(path)  # missing ASL versions warn, they do not block
    assert [p.id for p in prompts.prompts] == ["a0", "a1", "a2"] and prompts.n_without_asl == 3
