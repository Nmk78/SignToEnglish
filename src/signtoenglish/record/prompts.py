"""The recorder's view of `domain/sentences.yaml`: what to show and what text to store.

`record` may not import `data.domain` (ARCHITECTURE.md, section 2), so this reads the few
fields the recorder needs through the shared YAML loader. The full check of the sentence
list, including the warnings for missing ASL versions, is `s2e domain check`.

The recorder shows a sentence's `asl` line if the user has filled it in. It never writes or
invents one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from signtoenglish.common.config import ConfigError, load_yaml

ROLES = ("train", "unseen_test")


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

    @property
    def n_without_asl(self) -> int:
        return sum(1 for p in self.prompts if not p.asl)


def load_prompts(path: str | Path) -> PromptList:
    data = load_yaml(path)
    for key in ("sign_language", "text_language", "sentences"):
        if key not in data:
            raise ConfigError(f"{path}: missing key {key!r}")
    raw = data["sentences"]
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{path}: `sentences` must be a non-empty list")
    prompts: list[Prompt] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ConfigError(f"{path}: sentence {i} is not a mapping")
        sentence_id, text, role = item.get("id"), item.get("text"), item.get("role")
        if not isinstance(sentence_id, str) or not sentence_id.strip():
            raise ConfigError(f"{path}: sentence {i} has no id")
        if any(c.isspace() or c in "/\\" for c in sentence_id):
            raise ConfigError(f"{path}: sentence id {sentence_id!r} cannot be used in a file name")
        if sentence_id in seen:
            raise ConfigError(f"{path}: duplicate sentence id {sentence_id!r}")
        seen.add(sentence_id)
        if not isinstance(text, str) or not text.strip():
            raise ConfigError(f"{path}: sentence {sentence_id} has no text")
        if role not in ROLES:
            raise ConfigError(f"{path}: sentence {sentence_id} has role {role!r}")
        asl = item.get("asl")
        if asl is not None and not isinstance(asl, str):
            raise ConfigError(f"{path}: sentence {sentence_id}: `asl` must be text or null")
        prompts.append(
            Prompt(sentence_id, text, role, asl.strip() if asl and asl.strip() else None)
        )
    sign_language, text_language = data["sign_language"], data["text_language"]
    if not isinstance(sign_language, str) or not isinstance(text_language, str):
        raise ConfigError(f"{path}: sign_language and text_language must be text")
    return PromptList(sign_language, text_language, tuple(prompts))
