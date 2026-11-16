"""Loader and validator for the domain sentence list (`domain/sentences.yaml`).

Format: ARCHITECTURE.md, section 4.1, plus the fields the committed file adds: a top-level
`status`, `train_sentences` on vocabulary entries, and `asl` / `source` on sentences.

`signs`, `asl`, and `source` are planning aids for the signer and for vocabulary-coverage
checks. They are never a model input or target. Nothing in this module writes them: how a
sentence is signed is supplied by the user from a reliable ASL source.
"""

from __future__ import annotations

import string
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ValidationError

from signtoenglish.common.config import ConfigError, FrozenConfig

Role = Literal["train", "unseen_test"]
MIN_TRAIN_SENTENCES_PER_SIGN = 3
EXPECTED_SIGN_LANGUAGE = "ase"
EXPECTED_TEXT_LANGUAGE = "en"

_PUNCTUATION = str.maketrans("", "", string.punctuation + "\u2018\u2019\u201c\u201d")


class VocabularyEntry(FrozenConfig):
    id: str
    train_sentences: int | None = None  # annotation written when the list was drafted


class Sentence(FrozenConfig):
    id: str
    text: str
    signs: tuple[str, ...] = ()
    role: Role
    asl: str | None = None  # the ASL version, supplied by the user
    source: str | None = None  # where that ASL version came from


class DomainSentences(FrozenConfig):
    domain: str
    status: str | None = None
    sign_language: str
    text_language: str
    vocabulary: tuple[VocabularyEntry, ...]
    sentences: tuple[Sentence, ...]

    def by_id(self) -> dict[str, Sentence]:
        return {s.id: s for s in self.sentences}

    def text_by_id(self) -> dict[str, str]:
        return {s.id: s.text for s in self.sentences}

    def with_role(self, role: Role) -> tuple[Sentence, ...]:
        return tuple(s for s in self.sentences if s.role == role)


@dataclass(frozen=True)
class DomainIssue:
    level: Literal["error", "warning"]
    code: str
    message: str
    sentence_id: str | None = None

    def __str__(self) -> str:
        who = f" [{self.sentence_id}]" if self.sentence_id else ""
        return f"{self.level.upper()} {self.code}{who}: {self.message}"


@dataclass
class DomainReport:
    issues: list[DomainIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[DomainIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[DomainIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def with_code(self, code: str) -> list[DomainIssue]:
        return [i for i in self.issues if i.code == code]


def text_key(text: str) -> str:
    """Key for "is this the same sentence text": case, punctuation, and spacing ignored.

    Used only for leakage checks between splits. It is not the evaluation normalization,
    which belongs to `eval/metrics.py`.
    """
    return " ".join(text.translate(_PUNCTUATION).casefold().split())


class _SentenceListLoader(yaml.SafeLoader):
    """YAML loader for the sentence list: bare YES, NO, ON, OFF stay text.

    PyYAML follows YAML 1.1, where an unquoted `NO` or `YES` is a boolean. The sentence
    list uses both as sign ids, and nothing in it is ever a boolean, so this loader drops
    the boolean rule. Numbers and `null` are read as usual. Duplicate keys are rejected,
    as in `common.config.load_yaml`.
    """

    yaml_implicit_resolvers = {  # noqa: RUF012 - PyYAML's own class-level table
        first: [(tag, rx) for tag, rx in rules if tag != "tag:yaml.org,2002:bool"]
        for first, rules in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_domain(path: str | Path) -> DomainSentences:
    """Load and type-check the file. Raises `ConfigError` on a malformed file. Content
    problems (duplicates, coverage, missing ASL versions) come from `check_domain`."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read sentence list {path}: {exc}") from exc
    try:
        data = yaml.load(text, Loader=_SentenceListLoader)  # a SafeLoader subclass
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    try:
        return DomainSentences.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def sign_coverage(
    domain: DomainSentences, train_sentence_ids: Collection[str] | None = None
) -> Counter[str]:
    """Sign id -> number of distinct `role: train` sentences listing it.

    With `train_sentence_ids`, only those sentences count (the split builder passes the
    sentences that really have a take in the train split).
    """
    counts: Counter[str] = Counter()
    for sentence in domain.sentences:
        if sentence.role != "train":
            continue
        if train_sentence_ids is not None and sentence.id not in train_sentence_ids:
            continue
        counts.update(set(sentence.signs))
    return counts


def uncovered_signs(
    domain: DomainSentences,
    sentence_ids: Collection[str],
    train_sentence_ids: Collection[str] | None = None,
    minimum: int = MIN_TRAIN_SENTENCES_PER_SIGN,
) -> dict[str, dict[str, int]]:
    """For each given sentence, the signs seen in fewer than `minimum` training sentences.

    Returns sentence id -> {sign: count}; a sentence with full coverage is absent.
    """
    counts = sign_coverage(domain, train_sentence_ids)
    by_id = domain.by_id()
    result: dict[str, dict[str, int]] = {}
    for sentence_id in sentence_ids:
        short = {s: counts[s] for s in by_id[sentence_id].signs if counts[s] < minimum}
        if short:
            result[sentence_id] = short
    return result


def _blank(value: str | None) -> bool:
    return value is None or not value.strip()


def check_domain(
    domain: DomainSentences, minimum: int = MIN_TRAIN_SENTENCES_PER_SIGN
) -> DomainReport:
    """Every content problem in the sentence list.

    Errors: duplicate ids, empty text, the same text twice, a sign that is not in the
    vocabulary, an unseen-test sentence without signs, and an unseen-test sign that appears
    in fewer than `minimum` training sentences.

    Warnings: one per sentence whose `asl` is empty and one per sentence whose `source` is
    empty, plus vocabulary entries that no sentence uses, stale `train_sentences`
    annotations, training sentences without signs, and unexpected language codes.
    """
    report = DomainReport()

    def error(code: str, message: str, sentence_id: str | None = None) -> None:
        report.issues.append(DomainIssue("error", code, message, sentence_id))

    def warn(code: str, message: str, sentence_id: str | None = None) -> None:
        report.issues.append(DomainIssue("warning", code, message, sentence_id))

    if domain.sign_language != EXPECTED_SIGN_LANGUAGE:
        warn("language", f"sign_language is {domain.sign_language!r}, expected 'ase'")
    if domain.text_language != EXPECTED_TEXT_LANGUAGE:
        warn("language", f"text_language is {domain.text_language!r}, expected 'en'")
    if not domain.sentences:
        error("empty", "the sentence list is empty")

    vocabulary_counts = Counter(v.id for v in domain.vocabulary)
    for sign, n in vocabulary_counts.items():
        if n > 1:
            error("duplicate_vocabulary", f"vocabulary lists {sign!r} {n} times")
        if not sign.strip():
            error("empty_vocabulary", "vocabulary has an entry with an empty id")
    vocabulary = set(vocabulary_counts)

    id_counts = Counter(s.id for s in domain.sentences)
    for sentence_id, n in id_counts.items():
        if n > 1:
            error("duplicate_id", f"sentence id used {n} times", sentence_id)

    first_with_text: dict[str, Sentence] = {}
    used_signs: set[str] = set()
    for sentence in domain.sentences:
        if not sentence.id.strip() or any(c.isspace() for c in sentence.id):
            error("bad_id", f"sentence id {sentence.id!r} is empty or contains whitespace")
        if not sentence.text.strip():
            error("empty_text", "text is empty", sentence.id)
        else:
            key = text_key(sentence.text)
            earlier = first_with_text.get(key)
            if earlier is not None:
                error(
                    "duplicate_text",
                    f"same text as {earlier.id} ({earlier.role}): {sentence.text!r}",
                    sentence.id,
                )
            else:
                first_with_text[key] = sentence

        if not sentence.signs:
            if sentence.role == "unseen_test":
                error("no_signs", "unseen-test sentence lists no signs", sentence.id)
            else:
                warn("no_signs", "sentence lists no signs", sentence.id)
        repeated = [s for s, n in Counter(sentence.signs).items() if n > 1]
        if repeated:
            warn("repeated_sign", f"signs lists {repeated} more than once", sentence.id)
        for sign in sentence.signs:
            used_signs.add(sign)
            if sign not in vocabulary:
                error("unknown_sign", f"sign {sign!r} is not in the vocabulary", sentence.id)

        if _blank(sentence.asl):
            warn("asl_missing", "no ASL version yet (`asl` is empty)", sentence.id)
        if _blank(sentence.source):
            warn("source_missing", "no source for the ASL version (`source` is empty)", sentence.id)

    unseen_ids = [s.id for s in domain.sentences if s.role == "unseen_test"]
    if id_counts and max(id_counts.values()) == 1:
        for sentence_id, short in uncovered_signs(domain, unseen_ids, minimum=minimum).items():
            listing = ", ".join(f"{sign} ({n})" for sign, n in sorted(short.items()))
            error(
                "coverage",
                f"signs in fewer than {minimum} training sentences: {listing}",
                sentence_id,
            )

    counts = sign_coverage(domain)
    for entry in domain.vocabulary:
        if entry.id not in used_signs:
            warn("unused_sign", f"vocabulary entry {entry.id!r} is used by no sentence")
        if entry.train_sentences is not None and entry.train_sentences != counts[entry.id]:
            warn(
                "stale_annotation",
                f"vocabulary entry {entry.id!r} says train_sentences: {entry.train_sentences}, "
                f"counted {counts[entry.id]}",
            )
    return report
