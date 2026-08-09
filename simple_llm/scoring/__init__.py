"""Minimal, composable ASD-STE100 scoring."""

from collections.abc import Callable, Mapping
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Score = Annotated[float, Field(ge=0, le=1)]
Scorer = Callable[[str, str], float]


class STEScores(BaseModel):
    """Independent scores for one prompt and answer; no aggregate score."""

    model_config = ConfigDict(extra="forbid")

    # Rule-based dimensions
    sentence_length: Score | None = None
    controlled_vocabulary: Score | None = None
    verb_forms_and_modals: Score | None = None
    sentence_mechanics: Score | None = None
    procedure_syntax: Score | None = None
    multi_word_nouns: Score | None = None
    terminology_consistency: Score | None = None
    document_limits: Score | None = None

    # LLM-judge dimensions
    clear_unambiguous_meaning: Score | None = None
    one_instruction_or_fact_unit: Score | None = None
    word_sense_and_part_of_speech: Score | None = None
    voice_and_agent_clarity: Score | None = None
    direct_action_expression: Score | None = None
    reference_and_discourse_coherence: Score | None = None
    safety_communication: Score | None = None
    technical_adequacy: Score | None = None


def score(
    prompt: str,
    answer: str,
    scorers: Mapping[str, Scorer],
) -> STEScores:
    """Run each named scorer and return its uncombined score."""

    return STEScores(
        **{name: scorer(prompt, answer) for name, scorer in scorers.items()}
    )


from .rules import (
    RULE_SCORERS,
    controlled_vocabulary_scorer,
    controlled_vocabulary_scorer_from_file,
)

__all__ = [
    "RULE_SCORERS",
    "Scorer",
    "STEScores",
    "controlled_vocabulary_scorer",
    "controlled_vocabulary_scorer_from_file",
    "score",
]
