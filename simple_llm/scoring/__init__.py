"""Minimal, composable ASD-STE100 scoring."""

from collections.abc import Callable, Mapping
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Score = Annotated[float, Field(ge=0, le=1)]
NonNegativeMetric = Annotated[float, Field(ge=0)]
Fraction = Annotated[float, Field(ge=0, le=1)]
Scorer = Callable[[str, str], float]


class RuleScores(BaseModel):
    """Independent rule-based dimensions; no aggregate."""

    model_config = ConfigDict(extra="forbid")

    # Raw metrics: lower is better for both fields. Other fields are 0–1
    # compliance scores where higher is better.
    average_sentence_length: NonNegativeMetric | None = None
    long_sentence_fraction: Fraction | None = None
    controlled_vocabulary: Score | None = None
    verb_forms_and_modals: Score | None = None
    sentence_mechanics: Score | None = None
    procedure_syntax: Score | None = None
    terminology_consistency: Score | None = None
    document_limits: Score | None = None


def score(
    prompt: str,
    answer: str,
    scorers: Mapping[str, Scorer],
) -> RuleScores:
    """Run each named scorer and return its uncombined score."""

    return RuleScores(
        **{name: scorer(prompt, answer) for name, scorer in scorers.items()}
    )


from .rules import (
    RULE_SCORERS,
    controlled_vocabulary_scorer,
    controlled_vocabulary_scorer_from_file,
    average_sentence_length,
    long_sentence_fraction,
)
from .validity import AnswerValidity, validate_answer

__all__ = [
    "RULE_SCORERS",
    "Scorer",
    "RuleScores",
    "controlled_vocabulary_scorer",
    "controlled_vocabulary_scorer_from_file",
    "average_sentence_length",
    "long_sentence_fraction",
    "AnswerValidity",
    "validate_answer",
    "score",
]
