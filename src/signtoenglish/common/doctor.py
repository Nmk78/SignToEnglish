"""Environment report: Python, pinned package versions, device, free disk, ffmpeg."""

from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

DISTRIBUTION = "signtoenglish"
_PIN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;]+)")


def _pinned_requirements() -> dict[str, str]:
    """Direct dependencies of this project and the versions it pins, from package metadata."""
    pins: dict[str, str] = {}
    for req in md.requires(DISTRIBUTION) or []:
        m = _PIN.match(req)
        if m:
            pins[m.group(1)] = m.group(2)
    return pins


def _tool_version(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    if path is None:
        return {"found": False, "path": None, "version": None}
    try:
        cmd = [path, "-version"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        first = out.stdout.splitlines()[0] if out.stdout else ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        first = f"error: {exc}"
    return {"found": True, "path": path, "version": first}


def _memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return None


def collect(disk_path: str | Path | None = None) -> dict[str, Any]:
    """Gather the report. Imports torch and onnxruntime, so it takes a few seconds."""
    import onnxruntime as ort
    import torch

    from signtoenglish.common.device import mps_available, select_device

    problems: list[str] = []
    warnings: list[str] = []

    packages: dict[str, dict[str, str | None]] = {}
    for name, pinned in sorted(_pinned_requirements().items(), key=lambda kv: kv[0].lower()):
        try:
            installed: str | None = md.version(name)
        except md.PackageNotFoundError:
            installed = None
        packages[name] = {"pinned": pinned, "installed": installed}
        if installed != pinned:
            problems.append(f"{name}: pinned {pinned}, installed {installed}")

    requires_python = md.metadata(DISTRIBUTION).get("Requires-Python") or ""
    py = platform.python_version()
    wanted_minor = requires_python.removeprefix("==").removesuffix(".*")
    if wanted_minor and not py.startswith(wanted_minor + "."):
        problems.append(f"Python is {py}, project requires {requires_python}")

    usage = shutil.disk_usage(Path(disk_path) if disk_path is not None else Path.cwd())
    tools = {name: _tool_version(name) for name in ("ffmpeg", "ffprobe")}
    for name, info in tools.items():
        if not info["found"]:
            warnings.append(f"{name} not found on PATH (used for inspection and timestamp tests)")

    memory = _memory_bytes()
    return {
        "ok": not problems,
        "problems": problems,
        "warnings": warnings,
        "python": {
            "version": py,
            "requires": requires_python,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "macos": platform.mac_ver()[0] or None,
            "machine": platform.machine(),
            "memory_gb": round(memory / 2**30, 1) if memory else None,
        },
        "device": {
            "selected": select_device("auto"),
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": mps_available(),
            "torch_threads": torch.get_num_threads(),
        },
        "onnxruntime": {
            "version": ort.__version__,
            "available_providers": list(ort.get_available_providers()),
        },
        "disk": {
            "free_gb": round(usage.free / 2**30, 1),
            "total_gb": round(usage.total / 2**30, 1),
        },
        "tools": tools,
        "hub_offline": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "packages": packages,
    }


def format_report(report: dict[str, Any]) -> str:
    py, plat, dev = report["python"], report["platform"], report["device"]
    lines = [
        f"status        {'OK' if report['ok'] else 'PROBLEMS'}",
        f"python        {py['version']} (requires {py['requires']})  {py['executable']}",
        f"platform      {plat['system']} {plat['macos'] or plat['release']} {plat['machine']}, "
        f"{plat['memory_gb']} GB memory",
        f"device        {dev['selected']} (mps built={dev['mps_built']}, "
        f"available={dev['mps_available']})",
        f"onnxruntime   {report['onnxruntime']['version']} "
        f"providers={report['onnxruntime']['available_providers']}",
        f"disk          {report['disk']['free_gb']} GB free of {report['disk']['total_gb']} GB",
    ]
    for name, info in report["tools"].items():
        lines.append(f"{name:<13} {info['version'] if info['found'] else 'NOT FOUND'}")
    lines.append(f"hub offline   {report['hub_offline']}")
    lines.append("packages      (pinned -> installed)")
    for name, v in report["packages"].items():
        mark = "" if v["pinned"] == v["installed"] else "   <-- MISMATCH"
        lines.append(f"  {name:<24} {v['pinned']:<18} {v['installed']}{mark}")
    for w in report["warnings"]:
        lines.append(f"warning: {w}")
    for p in report["problems"]:
        lines.append(f"problem: {p}")
    return "\n".join(lines)


def run(as_json: bool = False) -> int:
    """Print the report. Exit code 0 when the environment matches the pins, 1 otherwise."""
    report = collect()
    print(json.dumps(report, indent=2) if as_json else format_report(report))
    return 0 if report["ok"] else 1
