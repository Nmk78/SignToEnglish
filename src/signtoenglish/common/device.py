"""Device selection. Order: MPS, then CPU. There is no CUDA path in this project."""

from __future__ import annotations

from typing import Literal

DevicePreference = Literal["auto", "mps", "cpu"]
DeviceName = Literal["mps", "cpu"]


def mps_available() -> bool:
    import torch

    return bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())


def select_device(prefer: DevicePreference = "auto") -> DeviceName:
    """Return the device name to pass to `torch.device` / `.to()`.

    `auto` picks MPS when it is available and CPU otherwise. Asking for `mps` explicitly
    when it is unavailable is an error rather than a silent fallback.
    """
    if prefer == "cpu":
        return "cpu"
    if prefer == "mps":
        if not mps_available():
            raise RuntimeError("device 'mps' was requested but MPS is not available")
        return "mps"
    if prefer == "auto":
        return "mps" if mps_available() else "cpu"
    raise ValueError(f"unknown device preference {prefer!r}; use 'auto', 'mps', or 'cpu'")
