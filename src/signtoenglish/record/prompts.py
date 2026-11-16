"""The recorder's view of `domain/sentences.yaml`: what to show and what text to store.

The file is read and checked by `data.domain`, the one loader for the sentence list. A list
with errors (duplicate ids or text, unknown signs, thin sign coverage) is refused, because
takes recorded against it would have to be thrown away. Missing ASL versions are warnings
and do not stop recording; `s2e domain check` lists them.

The recorder shows a sentence's `asl` line if the user has filled it in. It never writes or
invents one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from signtoenglish.common.config import ConfigError
from signtoenglish.data.domain import check_domain, load_domain


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    role: str
    asl: str | None = None


@dataclass(frozen=True)
class PromptList:
    sign_language: str
    text_language: str
    prompts: tuple[Prompt, ...]
    n_without_asl: int


def load_prompts(path: str | Path) -> PromptList:
    """Prompts from a valid sentence list. Raises `ConfigError` if the file is malformed,
    has content errors, or has a sentence id that cannot be part of a file name."""
    domain = load_domain(path)
    report = check_domain(domain)
    if not report.ok:
        shown = "\n".join(f"  {issue}" for issue in report.errors[:10])
        raise ConfigError(
            f"{path}: {len(report.errors)} error(s) in the sentence list; "
            f"run `s2e domain check`:\n{shown}"
        )
    for sentence in domain.sentences:
        if any(c in "/\\" for c in sentence.id) or sentence.id.startswith("."):
            raise ConfigError(f"{path}: sentence id {sentence.id!r} cannot be used in a file name")
    prompts = tuple(
        Prompt(s.id, s.text, s.role, s.asl.strip() if s.asl and s.asl.strip() else None)
        for s in domain.sentences
    )
    return PromptList(
        domain.sign_language,
        domain.text_language,
        prompts,
        n_without_asl=len(report.with_code("asl_missing")),
    )
