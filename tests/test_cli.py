"""`s2e` discovers subcommands from an optional `commands.py` in each package."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from signtoenglish import cli


def _make_package(root: Path, name: str, subpackages: dict[str, str | None]) -> None:
    pkg = root / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for sub, commands_source in subpackages.items():
        (pkg / sub).mkdir()
        (pkg / sub / "__init__.py").write_text("")
        if commands_source is not None:
            (pkg / sub / "commands.py").write_text(commands_source)


GOOD = """
def _run(args):
    print("hello from", args.who)
    return 7

def register(subparsers):
    p = subparsers.add_parser("greet")
    p.add_argument("--who", default="alpha")
    p.set_defaults(func=_run)
"""


def test_discovers_commands_and_skips_packages_without_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_package(tmp_path, "fakepkg_ok", {"alpha": GOOD, "beta": None})
    monkeypatch.syspath_prepend(str(tmp_path))

    modules = cli.discover_command_modules("fakepkg_ok")
    assert [m.__name__ for m in modules] == ["fakepkg_ok.alpha.commands"]

    parser = cli.build_parser("fakepkg_ok")
    args = parser.parse_args(["greet", "--who", "tests"])
    assert args.func(args) == 7
    assert capsys.readouterr().out.strip() == "hello from tests"


def test_broken_commands_module_is_an_error_naming_the_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_package(tmp_path, "fakepkg_broken", {"alpha": "import does_not_exist_anywhere\n"})
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(cli.CommandDiscoveryError, match=r"fakepkg_broken\.alpha\.commands"):
        cli.discover_command_modules("fakepkg_broken")


def test_commands_module_without_register_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_package(tmp_path, "fakepkg_noreg", {"alpha": "x = 1\n"})
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(cli.CommandDiscoveryError, match="register"):
        cli.discover_command_modules("fakepkg_noreg")


def test_command_without_handler_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = "def register(subparsers):\n    subparsers.add_parser('nohandler')\n"
    _make_package(tmp_path, "fakepkg_nofunc", {"alpha": source})
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(cli.CommandDiscoveryError, match="nohandler"):
        cli.build_parser("fakepkg_nofunc")


def test_two_packages_registering_the_same_command_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_package(tmp_path, "fakepkg_dup", {"alpha": GOOD, "beta": GOOD})
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(cli.CommandDiscoveryError, match="greet"):
        cli.build_parser("fakepkg_dup")


def test_real_cli_registers_doctor() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["doctor", "--json"])
    assert callable(args.func)
    assert args.json is True


def test_no_command_prints_help_and_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 2
    assert "doctor" in capsys.readouterr().out


def test_importing_the_cli_does_not_import_heavy_libraries() -> None:
    code = (
        "import sys; from signtoenglish import cli; cli.build_parser(); "
        "heavy = [m for m in ('torch', 'transformers', 'onnxruntime', 'cv2') if m in sys.modules]; "
        "print(heavy); sys.exit(1 if heavy else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_s2e_doctor_runs_for_real_through_the_installed_entry_point(repo_root: Path) -> None:
    """Runs the installed `s2e` script: real torch, real ONNX Runtime, real disk and tools."""
    s2e = Path(sys.executable).parent / "s2e"
    assert s2e.is_file(), "the s2e entry point is not installed in this environment"
    result = subprocess.run(
        [str(s2e), "doctor", "--json"], capture_output=True, text=True, cwd=repo_root, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)

    assert report["ok"] is True, report["problems"]
    assert report["python"]["version"].startswith("3.12.")
    assert report["device"]["selected"] == "mps"
    assert "CPUExecutionProvider" in report["onnxruntime"]["available_providers"]
    assert report["disk"]["free_gb"] > 0
    assert set(report["tools"]) == {"ffmpeg", "ffprobe"}
    assert report["hub_offline"]["HF_HUB_OFFLINE"] == "1"
    for name in ("torch", "transformers", "onnxruntime", "rtmlib", "numpy", "pydantic"):
        assert report["packages"][name]["installed"] == report["packages"][name]["pinned"]


def test_scripts_doctor_gives_the_same_report(repo_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "doctor.py")],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("status        OK")
