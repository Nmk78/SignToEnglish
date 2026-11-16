"""Import rules from ARCHITECTURE.md, section 2, enforced by walking the source tree.

The table there is an allow-list: a package may import only what its row permits, at
module granularity where the row names modules (for example `data` may import
`preprocess.schema` but not `preprocess.extract`). Imports inside functions count too.

Third-party imports are not restricted here, with two exceptions that the documents
state: the third-party base-model code and `sys.path` edits are allowed only in
`models/translator.py`.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = "signtoenglish"

# package -> prefixes (relative to `signtoenglish.`) it may import, besides itself.
ALLOWED: dict[str, tuple[str, ...]] = {
    "common": (),
    "preprocess": ("common",),
    "data": ("common", "preprocess.schema", "preprocess.cache", "preprocess.normalize"),
    "record": ("common", "preprocess.video_io", "data.manifest", "data.domain"),
    "models": ("common",),
    "train": ("common", "data", "models", "eval"),
    "eval": ("common", "data", "models", "infer"),
    "infer": ("common", "preprocess", "models"),
    "serve": ("common", "infer"),
    # The entry point finds commands by discovery and needs nothing but `common`.
    "cli": ("common",),
}

# The "must not import" column, kept as a cross-check that ALLOWED never grants these.
FORBIDDEN: dict[str, tuple[str, ...]] = {
    "preprocess": ("data", "models", "train", "eval", "infer", "serve"),
    "data": ("train", "infer", "serve"),
    "record": ("models", "train", "infer", "serve"),
    "models": ("train", "data", "serve"),
    "train": ("infer", "serve"),
    "eval": ("train", "serve"),
    "infer": ("train", "data.dataset", "serve"),
}

THIRD_PARTY_GATEWAY = "models/translator.py"


def _matches(target: str, prefix: str) -> bool:
    return target == prefix or target.startswith(prefix + ".")


def _module_name(path: Path, src_root: Path) -> str:
    parts = list(path.relative_to(src_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _project_imports(tree: ast.AST, module: str, is_package: bool) -> list[tuple[int, str]]:
    """(line, dotted target relative to the project package) for every project import."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _matches(alias.name, PACKAGE):
                    found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = module.split(".")
                if not is_package:
                    base = base[:-1]
                base = base[: len(base) - (node.level - 1)]
                stem = ".".join([*base, node.module] if node.module else base)
            else:
                stem = node.module or ""
            if not _matches(stem, PACKAGE):
                continue
            # `from signtoenglish.data import dataset` targets `data.dataset`.
            for alias in node.names:
                found.append((node.lineno, f"{stem}.{alias.name}"))
    return [(line, t.removeprefix(PACKAGE).lstrip(".")) for line, t in found]


def find_violations(src_root: Path) -> list[str]:
    """All import-rule violations under `src_root/signtoenglish`, as readable strings."""
    violations: list[str] = []
    package_root = src_root / PACKAGE
    for path in sorted(package_root.rglob("*.py")):
        rel = path.relative_to(package_root).as_posix()
        module = _module_name(path, src_root)
        owner = module.removeprefix(PACKAGE).lstrip(".").split(".")[0]
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

        if rel != THIRD_PARTY_GATEWAY:
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "path"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "sys"
                ):
                    violations.append(f"{rel}:{node.lineno}: uses sys.path")
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if name.split(".")[0] == "third_party":
                        violations.append(f"{rel}:{node.lineno}: imports third-party model code")

        if not owner:  # signtoenglish/__init__.py
            allowed: tuple[str, ...] = ()
        elif owner not in ALLOWED:
            violations.append(f"{rel}: package {owner!r} has no row in the import rules")
            continue
        else:
            allowed = (owner, *ALLOWED[owner])

        for line, target in _project_imports(tree, module, path.name == "__init__.py"):
            if not target:
                continue  # `import signtoenglish`
            # `from signtoenglish.data import manifest` yields `data.manifest`; a name
            # imported from a module yields `data.manifest.Row`. Either is fine if any
            # allowed prefix covers it.
            if any(_matches(target, prefix) for prefix in allowed):
                continue
            violations.append(f"{rel}:{line}: {owner or PACKAGE} may not import {target}")
    return violations


def _write(root: Path, rel: str, text: str = "") -> None:
    path = root / PACKAGE / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_repository_follows_the_import_rules(repo_root: Path) -> None:
    assert find_violations(repo_root / "src") == []


def test_every_package_in_the_layout_exists_and_has_a_rule(repo_root: Path) -> None:
    package_root = repo_root / "src" / PACKAGE
    packages = {p.name for p in package_root.iterdir() if (p / "__init__.py").is_file()}
    assert packages == set(ALLOWED) - {"cli"}
    assert (package_root / "cli.py").is_file()


def test_allow_list_never_grants_a_forbidden_import() -> None:
    for owner, banned in FORBIDDEN.items():
        for target in banned:
            granted = [p for p in ALLOWED[owner] if _matches(target, p) or _matches(p, target)]
            # `infer` may import nothing from `data` at all, which covers `data.dataset`.
            assert granted == [], f"{owner} is allowed {granted} but must not import {target}"


def test_checker_accepts_allowed_imports(tmp_path: Path) -> None:
    _write(tmp_path, "common/config.py")
    _write(tmp_path, "preprocess/schema.py", "from signtoenglish.common import config\n")
    _write(tmp_path, "data/manifest.py", "from signtoenglish.preprocess.schema import X\n")
    _write(tmp_path, "data/splits.py", "from . import manifest\nfrom .manifest import Row\n")
    _write(tmp_path, "record/recorder.py", "from signtoenglish.data import manifest\n")
    _write(tmp_path, "serve/app.py", "def f():\n    from signtoenglish.infer import offline\n")
    _write(tmp_path, "models/translator.py", "import sys\nsys.path.insert(0, 'x')\n")
    assert find_violations(tmp_path) == []


def test_checker_catches_violations(tmp_path: Path) -> None:
    cases = {
        "infer/streaming.py": "from signtoenglish.train import loop\n",
        "serve/app.py": "def f():\n    import signtoenglish.train.loop\n",
        "infer/offline.py": "from signtoenglish.data import dataset\n",
        "data/dataset.py": "from signtoenglish.preprocess import extract\n",
        "record/recorder.py": "from signtoenglish.data.splits import build\n",
        "common/config.py": "from signtoenglish.preprocess import schema\n",
        "preprocess/extract.py": "from ..models import translator\n",
        "models/adapters.py": "from ..data.manifest import Row\n",
        "eval/metrics.py": "import sys\nsys.path.append('x')\n",
        "infer/loader.py": "from third_party.unisign import model\n",
        "extras/thing.py": "x = 1\n",
    }
    for rel, text in cases.items():
        root = tmp_path / rel.replace("/", "_")
        _write(root, rel, text)
        assert len(find_violations(root)) == 1, f"{rel} should be exactly one violation"
