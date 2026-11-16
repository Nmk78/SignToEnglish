"""`s2e` entry point.

Subcommands are discovered, not listed here (WORKERS.md, ground rule 4). To add one, a
track creates `commands.py` inside its own package (`signtoenglish/<package>/commands.py`)
with:

    def register(subparsers) -> None:
        p = subparsers.add_parser("my-command", help="...")
        p.add_argument(...)
        p.set_defaults(func=handler)      # handler(args: argparse.Namespace) -> int

Keep `commands.py` cheap to import: import torch, the model, or OpenCV inside the handler,
not at module top level, because every `s2e` invocation imports every `commands.py`.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import pkgutil
import sys
from collections.abc import Sequence
from types import ModuleType

PACKAGE = "signtoenglish"
COMMANDS_MODULE = "commands"


class CommandDiscoveryError(RuntimeError):
    """A `commands.py` exists but cannot be imported or does not follow the contract."""


def discover_command_modules(package_name: str = PACKAGE) -> list[ModuleType]:
    """Import `<package>.<subpackage>.commands` for every subpackage that has one.

    Subpackages without a `commands.py` are skipped. A `commands.py` that fails to import
    or lacks a callable `register` is an error naming the module, never a silent skip.
    """
    package = importlib.import_module(package_name)
    modules: list[ModuleType] = []
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda i: i.name):
        if not info.ispkg:
            continue
        name = f"{package_name}.{info.name}.{COMMANDS_MODULE}"
        if importlib.util.find_spec(name) is None:
            continue
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            raise CommandDiscoveryError(f"cannot import {name}: {exc!r}") from exc
        if not callable(getattr(module, "register", None)):
            raise CommandDiscoveryError(f"{name} has no callable register(subparsers)")
        modules.append(module)
    return modules


def build_parser(package_name: str = PACKAGE) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="s2e", description="SignToEnglish command line.")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for module in discover_command_modules(package_name):
        before = set(subparsers.choices)
        try:
            module.register(subparsers)
        except argparse.ArgumentError as exc:
            raise CommandDiscoveryError(f"{module.__name__}: {exc}") from exc
        for added in set(subparsers.choices) - before:
            if subparsers.choices[added].get_default("func") is None:
                raise CommandDiscoveryError(
                    f"{module.__name__}: command {added!r} does not set_defaults(func=...)"
                )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
