"""The wire format and the bounded frame queue: pure logic, no model, no files."""

from __future__ import annotations

import threading

import pytest
from pydantic import ValidationError

from signtoenglish.infer.streaming import StreamEvent
from signtoenglish.serve.service import Spread
from signtoenglish.serve.settings import ServeSettings
from signtoenglish.serve.ws import FrameError, FrameQueue, StartMessage, event_message, parse_frame

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9"


def test_frame_is_signed_64_bit_little_endian_milliseconds_then_jpeg() -> None:
    epoch_ms = 1_794_900_000_123  # a capture time in milliseconds since the Unix epoch
    t_ms, jpeg = parse_frame(epoch_ms.to_bytes(8, "little", signed=True) + JPEG)
    assert (t_ms, jpeg) == (epoch_ms, JPEG)
    assert parse_frame(b"\x01\x02\x00\x00\x00\x00\x00\x00" + JPEG)[0] == 0x0201
    assert parse_frame((0).to_bytes(8, "little") + JPEG)[0] == 0


@pytest.mark.parametrize(
    ("message", "says"),
    [
        (b"", "too short"),
        (b"\x00" * 8, "too short"),
        ((-5).to_bytes(8, "little", signed=True) + JPEG, "negative"),
        ((7).to_bytes(8, "little") + b"\x89PNG" + b"\x00" * 20, "not a JPEG"),
        (JPEG + b"\x00" * 8, "not a JPEG"),  # timestamp bytes missing
    ],
)
def test_bad_frames_are_refused_with_a_reason(message: bytes, says: str) -> None:
    with pytest.raises(FrameError, match=says):
        parse_frame(message)


def test_oversized_frame_is_refused() -> None:
    with pytest.raises(FrameError, match="over the 20 limit"):
        parse_frame((1).to_bytes(8, "little") + JPEG, max_bytes=20)


def test_queue_drops_the_oldest_and_counts() -> None:
    queue: FrameQueue[int] = FrameQueue(3)
    assert [queue.put(i) for i in range(3)] == [0, 0, 0]
    assert queue.full
    assert queue.put(3) == 1 and queue.put(4) == 1
    assert (queue.depth, queue.dropped, queue.max_depth) == (3, 2, 3)
    assert [queue.get() for _ in range(3)] == [2, 3, 4]  # the newest survive, in order


def test_queue_never_grows_past_its_limit() -> None:
    queue: FrameQueue[int] = FrameQueue(4)
    for i in range(10_000):
        queue.put(i)
        assert queue.depth <= 4
    assert queue.dropped == 10_000 - 4 and queue.max_depth == 4


def test_queue_close_drains_then_ends_and_discard_empties() -> None:
    queue: FrameQueue[int] = FrameQueue(4)
    queue.put(1)
    queue.close()
    assert queue.put(2) == 0  # refused after close
    assert queue.get() == 1 and queue.get() is None

    other: FrameQueue[int] = FrameQueue(4)
    other.put(1)
    other.close(discard=True)
    assert other.get() is None and other.dropped == 0


def test_queue_get_blocks_until_a_frame_or_close() -> None:
    queue: FrameQueue[int] = FrameQueue(2)
    got: list[int | None] = []
    thread = threading.Thread(target=lambda: got.extend([queue.get(), queue.get()]))
    thread.start()
    queue.put(7)
    queue.close()
    thread.join(timeout=10)
    assert not thread.is_alive() and got == [7, None]


def test_queue_needs_a_positive_limit() -> None:
    with pytest.raises(ValueError):
        FrameQueue(0)


def test_start_message_validation() -> None:
    start = StartMessage.model_validate(
        {"type": "start", "width": 1280, "height": 720, "client_fps": 15, "extra": "ignored"}
    )
    assert (start.width, start.height, start.client_fps, start.overload) == (
        1280, 720, 15.0, "drop_oldest",
    )  # fmt: skip
    assert StartMessage.model_validate(
        {"type": "start", "width": 2, "height": 2, "overload": "wait"}
    ).overload == "wait"  # fmt: skip
    for bad in (
        {"type": "start", "width": 0, "height": 720},
        {"type": "start", "width": 1280},
        {"type": "start", "width": 1280, "height": 720, "client_fps": 0},
        {"type": "start", "width": 1280, "height": 720, "overload": "queue_forever"},
    ):
        with pytest.raises(ValidationError):
            StartMessage.model_validate(bad)


def _event(kind: str, **more: object) -> StreamEvent:
    base = {"segment_id": 3, "text": "", "start_ms": 100, "end_ms": 2100, "n_frames": 60,
            "signing_ms": 1500}  # fmt: skip
    return StreamEvent(type=kind, **{**base, **more})  # type: ignore[arg-type]


def test_engine_events_become_protocol_messages() -> None:
    final = event_message(
        _event("final", text="Hello.", closed_by="flush", decode_ms=512.34, latency_ms=640.6)
    )
    assert final == {
        "type": "final", "segment_id": 3, "text": "Hello.", "start_ms": 100, "end_ms": 2100,
        "latency_ms": 641, "decode_ms": 512.3, "closed_by": "flush",
    }  # fmt: skip
    partial = event_message(_event("partial", text="Hel", decode_ms=1.0))
    assert partial["type"] == "partial" and partial["latency_ms"] is None
    assert "closed_by" not in partial
    assert event_message(_event("dropped", reason="no_signing", closed_by="boundary")) == {
        "type": "dropped", "segment_id": 3, "start_ms": 100, "end_ms": 2100,
        "reason": "no_signing",
    }  # fmt: skip
    error = event_message(_event("error", reason="final decode failed: boom"))
    assert error == {"type": "error", "detail": "final decode failed: boom", "segment_id": 3}


def test_spread_summary() -> None:
    spread = Spread()
    assert spread.summary() is None
    for value in (10.0, 20.0, 30.0, 40.0):
        spread.add(value)
    assert spread.summary() == {"n": 4, "mean": 25.0, "median": 25.0, "p90": 37.0, "max": 40.0}
    assert spread.total == 100.0


def test_settings_default_to_localhost_and_reject_unknown_keys() -> None:
    settings = ServeSettings()
    assert settings.host == "127.0.0.1" and settings.port == 8000
    assert "http://127.0.0.1:3000" in settings.cors_origins
    with pytest.raises(ValidationError):
        ServeSettings.model_validate({"hots": "0.0.0.0"})
    with pytest.raises(ValidationError):
        settings.port = 9  # type: ignore[misc]
