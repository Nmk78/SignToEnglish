"""Stand-ins for the camera and the window, used to drive the real `Recorder`.

`GeneratedSource` produces real image frames with uneven, camera-like capture times.
`ScriptedUI` presses keys according to a plan and remembers what it was shown. Everything
else in these tests is the real thing: the H.264 writer, the files on disk, the manifests,
and the validator.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from signtoenglish.record.prompts import Prompt
from signtoenglish.record.recorder import Frame, Overlay, Recorder, RecorderSettings

HEIGHT, WIDTH = 240, 320
Step = tuple[str, int, str]  # (state, shows in that state since the last key, key)


def frame_image(index: int) -> NDArray[np.uint8]:
    """A frame that is different for every index and is not left-right symmetric."""
    image = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)
    image[:, :24] = (200, 60, 40)  # a blue bar on the left edge only
    x = 40 + (index * 7) % 220
    image[100:140, x : x + 40] = 255  # a white square that moves with the index
    image[200:220, 40 : 40 + (index % 64) * 4] = (0, 180, 0)
    return image


class GeneratedSource:
    def __init__(self, n_frames: int = 20_000, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        gaps_ms = 33.3 + rng.uniform(-6.0, 6.0, n_frames)
        gaps_ms[50::50] += 70.0  # now and then the camera skips two frames
        self.times_ns: list[int] = (
            987_654_321_000 + np.cumsum(gaps_ms * 1_000_000).astype(np.int64)
        ).tolist()
        self.n_frames = n_frames
        self.n_read = 0
        self.closed = False

    def read(self) -> Frame | None:
        if self.n_read >= self.n_frames:
            return None
        index = self.n_read
        self.n_read += 1
        return Frame(frame_image(index), self.times_ns[index])

    def close(self) -> None:
        self.closed = True


@dataclass
class Shown:
    index: int  # index of the source frame on screen
    overlay: Overlay


class ScriptedUI:
    """Presses each key of `plan` in turn, once its state has been on screen long enough."""

    def __init__(self, source: GeneratedSource, plan: Sequence[Step]) -> None:
        self.source = source
        self.plan = list(plan)
        self.shown: list[Shown] = []
        self.keys: list[tuple[int, str]] = []  # (source frame index, key)
        self.replayed: list[Path] = []
        self.closed = False
        self._streak = 0

    def show(self, image: NDArray[np.uint8], overlay: Overlay) -> None:
        index = self.source.n_read - 1
        assert image.shape == (HEIGHT, WIDTH, 3)
        if self.shown and self.shown[-1].overlay.state == overlay.state:
            self._streak += 1
        else:
            self._streak = 1
        self.shown.append(Shown(index, overlay))

    def poll_key(self) -> str | None:
        if not self.plan or not self.shown:
            return None
        state, after, key = self.plan[0]
        if self.shown[-1].overlay.state == state and self._streak >= after:
            self.plan.pop(0)
            self._streak = 0
            self.keys.append((self.shown[-1].index, key))
            return key
        return None

    def replay(self, path: Path) -> None:
        assert path.is_file()
        self.replayed.append(path)

    def close(self) -> None:
        self.closed = True

    def frames_in(self, state: str) -> list[int]:
        return [s.index for s in self.shown if s.overlay.state == state]

    def recording_runs(self) -> list[list[int]]:
        """Source frame indices of each uninterrupted stretch of recording."""
        runs: list[list[int]] = []
        previous_state = ""
        for s in self.shown:
            if s.overlay.state == "recording":
                if previous_state != "recording":
                    runs.append([])
                runs[-1].append(s.index)
            previous_state = s.overlay.state
        return runs


PROMPTS = (
    Prompt("t01", "Hello, how are you?", "train"),
    Prompt("t02", "Where does your mother live?", "train"),
    Prompt("t03", "My sister is hungry.", "train"),
    Prompt("u01", "Where does your friend live?", "unseen_test"),
)
FAST = RecorderSettings(countdown_s=0.2, takes_per_sentence=2, min_take_s=0.3, tail_s=0.3)

MakeRecorder = Callable[..., tuple[Recorder, GeneratedSource, ScriptedUI]]


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def make_recorder(data_root: Path) -> MakeRecorder:
    def make(
        plan: Sequence[Step],
        *,
        prompts: Sequence[Prompt] = PROMPTS,
        settings: RecorderSettings = FAST,
        session_id: str = "2026-11-16_desk",
        signer_id: str = "me",
        tags: Sequence[str] = (),
        n_frames: int = 20_000,
        seed: int = 0,
    ) -> tuple[Recorder, GeneratedSource, ScriptedUI]:
        source = GeneratedSource(n_frames, seed)
        ui = ScriptedUI(source, plan)
        recorder = Recorder(
            prompts=prompts,
            source=source,
            ui=ui,
            data_root=data_root,
            signer_id=signer_id,
            session_id=session_id,
            sign_language="ase",
            text_language="en",
            tags=tags,
            settings=settings,
            log=lambda _: None,
        )
        return recorder, source, ui

    return make


def read_times_ms(path: Path) -> list[int]:
    """Presentation time of every decoded frame, in milliseconds, from the container."""
    import av

    with av.open(str(path)) as container:
        return [
            round(frame.pts * frame.time_base * 1000)
            for frame in container.decode(video=0)
            if frame.pts is not None and frame.time_base is not None
        ]


def read_images(path: Path) -> list[NDArray[np.uint8]]:
    import av

    with av.open(str(path)) as container:
        return [
            np.asarray(frame.to_ndarray(format="bgr24"), dtype=np.uint8)
            for frame in container.decode(video=0)
        ]


@dataclass(frozen=True)
class Tools:
    read_times_ms: Callable[[Path], list[int]]
    read_images: Callable[[Path], list[NDArray[np.uint8]]]
    frame_image: Callable[[int], NDArray[np.uint8]]
    prompts: tuple[Prompt, ...]
    fast: RecorderSettings
    source_cls: type[GeneratedSource]
    ui_cls: type[ScriptedUI]


@pytest.fixture
def tools() -> Tools:
    return Tools(
        read_times_ms, read_images, frame_image, PROMPTS, FAST, GeneratedSource, ScriptedUI
    )
