"""High-confidence checks for unusable generated answers."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerValidity:
    valid: bool
    reasons: tuple[str, ...] = ()


_REFUSAL_ONLY = re.compile(
    r"^(?:sorry[,.]?\s*)?(?:i\s+(?:cannot|can't|can not)|"
    r"i(?:'m| am)\s+unable\s+to|unable\s+to)(?:\s+\w+){1,8}[.!?]?$",
    re.IGNORECASE,
)
_REPEATED_TOKEN = re.compile(r"\b(\w+)(?:\s+\1){7,}\b", re.IGNORECASE)
_ALLOWED_CONTROL = re.compile(r"[\n\r\t]")
_LEXICAL = re.compile(r"[A-Za-z0-9]")


def validate_answer(
    prompt: str,
    answer: str,
    *,
    truncated: bool = False,
) -> AnswerValidity:
    """Reject only high-confidence empty, malformed, or non-responsive output."""

    reasons: list[str] = []
    text = answer.strip()
    normalized_prompt = " ".join(prompt.split()).casefold()
    normalized_answer = " ".join(text.split()).casefold()

    if not text:
        reasons.append("empty")
    if truncated:
        reasons.append("truncated")
    if text and not _LEXICAL.search(text):
        reasons.append("no_lexical_content")
    if normalized_answer and normalized_answer == normalized_prompt:
        reasons.append("prompt_echo")
    if _REFUSAL_ONLY.fullmatch(text):
        reasons.append("refusal_only")
    if text.count("```") % 2:
        reasons.append("malformed")
    if any(
        ord(character) < 32 and not _ALLOWED_CONTROL.fullmatch(character)
        for character in answer
    ):
        reasons.append("malformed")
    if _REPEATED_TOKEN.search(text):
        reasons.append("degenerate_repetition")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return AnswerValidity(valid=not unique_reasons, reasons=unique_reasons)


__all__ = ["AnswerValidity", "validate_answer"]
