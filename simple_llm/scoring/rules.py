"""Small, conservative ASD-STE100 rule scorers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from . import Scorer

_CODE = re.compile(r"```.*?```|`[^`\n]+`", re.DOTALL)
_SENTENCE = re.compile(r"[.!?](?:['\"”’)\]]*)\s+(?=[A-Z0-9])")
_LIST_ITEM = re.compile(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+")
_LIST_DOT = "<LIST_DOT>"
_MEASUREMENT = re.compile(r"\b\d+(?:\.\d+)?\s*[A-Za-zµ°]+\b")
_IDENTIFIER = re.compile(
    r"\b(?=[A-Za-z0-9][A-Za-z0-9._:/-]*\d)"
    r"[A-Za-z0-9][A-Za-z0-9._:/-]*[A-Za-z0-9](?:['’]s)?\b"
)
_WORD = re.compile(r"\b[\w]+(?:[-'][\w]+)*\b")
_CONTRACTION = re.compile(
    r"\b(?:\w+n['’]t|\w+['’](?:ll|re|ve|d|m)|"
    r"(?:it|that|there|what|who|let|he|she|i|you|we|they)['’]s)\b",
    re.I,
)
_LATIN = re.compile(r"(?:e\.g\.|i\.e\.|etc\.?)(?=\s|$|[,;)])", re.I)
_BANNED_MODAL = re.compile(r"\b(?:should|would|may|might|could)\b", re.I)
_PERFECT = re.compile(r"\b(?:has|have|had)\s+(?:been|\w+ed)\b", re.I)
_PROGRESSIVE = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being)\s+\w+ing\b", re.I
)
_ING_CLAUSE = re.compile(r",\s*\w+ing\b", re.I)
_TRAILING_CONDITION = re.compile(r"\b(?:if|when)\b", re.I)

_IMPERATIVE_START = re.compile(
    r"^(?:install|remove|set|run|make|use|add|configure|create|open|close|"
    r"select|enter|press|save|read|refer|start|stop|restart|update|connect|"
    r"copy|move|turn|push|pull|place|put|insert|attach|detach|adjust)\b",
    re.I,
)

_TERM_GROUPS = (
    frozenset({"check", "verify", "confirm", "ensure"}),
    frozenset({"config", "configuration", "settings"}),
)

_PROCEDURE_SENTENCE_LIMIT = 20
_DESCRIPTIVE_SENTENCE_LIMIT = 25
_PARAGRAPH_SENTENCE_LIMIT = 6
_DEFAULT_APPROVED_WORDS = (
    Path(__file__).resolve().parents[2] / "data/ste_approved_words.txt"
)


def _mask_code(text: str) -> str:
    return _CODE.sub(" CODE ", text).replace("’", "'").replace("‘", "'")


def _sentences(text: str) -> list[str]:
    text = _mask_code(text).strip()
    if not text:
        return []
    def mark_list_item(match: re.Match[str]) -> str:
        marker = match.group(0)
        if re.match(r"\s*\d+\.\s+", marker):
            marker = marker.replace(".", _LIST_DOT, 1)
        return "\0" + marker

    text = _LIST_ITEM.sub(mark_list_item, text)
    return [
        part.strip().replace(_LIST_DOT, ".")
        for segment in text.split("\0")
        for part in _SENTENCE.split(segment)
        if part.strip()
    ]


def _words(text: str) -> list[str]:
    text = _MEASUREMENT.sub(" MEASUREMENT ", _mask_code(text))
    text = _IDENTIFIER.sub(" IDENTIFIER ", text)
    return _WORD.findall(text)


def _compliance_fraction(total: int, violations: int) -> float:
    return 1.0 if total == 0 else max(0.0, 1.0 - violations / total)


def _looks_procedural(answer: str) -> bool:
    # ponytail: imperative detection is intentionally conservative; add a parser
    # only if validation shows this heuristic is a material source of error.
    return bool(
        re.search(r"(?m)^\s*(?:\d+[.)]|[A-Z][.)])\s+", answer)
        or any(_IMPERATIVE_START.search(sentence) for sentence in _sentences(answer))
    )


def _sentence_lengths(answer: str) -> tuple[list[int], int]:
    sentences = _sentences(answer)
    limit = (
        _PROCEDURE_SENTENCE_LIMIT
        if _looks_procedural(answer)
        else _DESCRIPTIVE_SENTENCE_LIMIT
    )
    lengths = [len(_words(sentence)) for sentence in sentences]
    return lengths, limit


def average_sentence_length(prompt: str, answer: str) -> float:
    """Return the mean number of words per sentence."""

    lengths, _ = _sentence_lengths(answer)
    return sum(lengths) / len(lengths) if lengths else 0.0


def long_sentence_fraction(prompt: str, answer: str) -> float:
    """Return the fraction of sentences over the inferred STE word limit."""

    lengths, limit = _sentence_lengths(answer)
    return sum(length > limit for length in lengths) / len(lengths) if lengths else 0.0


def sentence_mechanics(prompt: str, answer: str) -> float:
    """Check high-confidence mechanical restrictions."""

    sentences = _sentences(answer)
    violations = sum(
        bool(
            _CONTRACTION.search(sentence)
            or ";" in sentence
            or _LATIN.search(sentence)
        )
        for sentence in sentences
    )
    return _compliance_fraction(len(sentences), violations)


def verb_forms_and_modals(prompt: str, answer: str) -> float:
    """Check obvious banned modals and complex/progressive verb forms."""

    sentences = _sentences(answer)
    violations = sum(
        bool(
            _BANNED_MODAL.search(sentence)
            or _PERFECT.search(sentence)
            or _PROGRESSIVE.search(sentence)
            or _ING_CLAUSE.search(sentence)
        )
        for sentence in sentences
    )
    return _compliance_fraction(len(sentences), violations)


def document_limits(prompt: str, answer: str) -> float:
    """Check the sentence maximum for descriptive paragraphs."""

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", answer) if part.strip()]
    violations = sum(
        len(_sentences(paragraph)) > _PARAGRAPH_SENTENCE_LIMIT
        and not _looks_like_vertical_list(paragraph)
        for paragraph in paragraphs
    )
    return _compliance_fraction(len(paragraphs), violations)


def _looks_like_vertical_list(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    labeled = sum(
        bool(re.match(r"(?:[-*]|\d+[.)])\s+|[A-Z][\w /-]{1,30}:", line))
        for line in lines
    )
    return len(lines) >= 2 and labeled >= 2


def procedure_syntax(prompt: str, answer: str) -> float:
    """Flag a clear command followed by a trailing condition."""

    sentences = _sentences(answer)
    violations = sum(_has_trailing_condition(sentence) for sentence in sentences)
    return _compliance_fraction(len(sentences), violations)


def _has_trailing_condition(sentence: str) -> bool:
    condition = _TRAILING_CONDITION.search(sentence)
    return bool(
        condition
        and condition.start() > 0
        and _IMPERATIVE_START.search(sentence)
    )


def terminology_consistency(prompt: str, answer: str) -> float:
    """Penalize use of multiple terms from one known synonym group."""

    words = {word.lower() for word in _words(answer)}
    present = [group & words for group in _TERM_GROUPS if group & words]
    violations = sum(len(group) > 1 for group in present)
    return _compliance_fraction(len(present), violations)


def controlled_vocabulary_scorer(
    approved_words: Iterable[str],
    technical_terms: Iterable[str] = (),
) -> Scorer:
    """Build a vocabulary scorer from a locally supplied glossary.

    The official dictionary is deliberately an input, not copied into this repo.
    Terms in the prompt are also treated as candidate domain terms.
    """

    approved = {word.casefold() for word in approved_words}
    configured_terms = {term.casefold() for term in technical_terms}

    def scorer(prompt: str, answer: str) -> float:
        allowed = approved | configured_terms | {
            word.casefold() for word in _words(prompt)
        }
        words = _words(answer)
        violations = sum(word.casefold() not in allowed for word in words)
        return _compliance_fraction(len(words), violations)

    return scorer


def controlled_vocabulary_scorer_from_file(
    path: str | Path = _DEFAULT_APPROVED_WORDS,
    technical_terms: Iterable[str] = (),
) -> Scorer:
    """Build a vocabulary scorer from an extracted STE word-list file."""

    words = Path(path).read_text(encoding="utf-8").splitlines()
    return controlled_vocabulary_scorer(words, technical_terms)


RULE_SCORERS: dict[str, Scorer] = {
    "controlled_vocabulary": controlled_vocabulary_scorer_from_file(),
    "average_sentence_length": average_sentence_length,
    "long_sentence_fraction": long_sentence_fraction,
    "sentence_mechanics": sentence_mechanics,
    "verb_forms_and_modals": verb_forms_and_modals,
    "procedure_syntax": procedure_syntax,
    "terminology_consistency": terminology_consistency,
    "document_limits": document_limits,
}


__all__ = [
    "RULE_SCORERS",
    "controlled_vocabulary_scorer",
    "controlled_vocabulary_scorer_from_file",
    "document_limits",
    "average_sentence_length",
    "long_sentence_fraction",
    "procedure_syntax",
    "sentence_mechanics",
    "terminology_consistency",
    "verb_forms_and_modals",
]
