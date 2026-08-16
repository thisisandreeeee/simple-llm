"""Schemas for the prompt-to-answer stage of SFT dataset generation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AnswerSpec(BaseModel):
    """Intermediate factual contract generated from one prompt."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^SFT-[A-Z]+-\d{4}$")
    required_facts: list[str] = Field(min_length=1)
    required_sections: list[str] = Field(default_factory=list)
    caveats_and_safety: list[str] = Field(default_factory=list)
    valid_commands_or_code: list[str] = Field(default_factory=list)
    prohibited_claims: list[str] = Field(default_factory=list)
    target_length: Literal["short", "medium", "long"]


class AnswerRecord(BaseModel):
    """Accepted answer persisted to ``data/sft_answers.jsonl``."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^SFT-[A-Z]+-\d{4}$")
    final_response: str = Field(min_length=1)


def build_answer_record(spec: AnswerSpec, final_response: str) -> AnswerRecord:
    """Create the persisted answer record for a completed specification."""

    return AnswerRecord(id=spec.id, final_response=final_response)


__all__ = ["AnswerRecord", "AnswerSpec", "build_answer_record"]
