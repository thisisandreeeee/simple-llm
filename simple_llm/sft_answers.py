"""Schemas for the prompt-to-answer stage of SFT dataset generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .sft_prompts import PromptRecord


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_PROMPTS = DATA_DIR / "sft_prompts.jsonl"

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


def load_prompts(path: Path = DEFAULT_PROMPTS) -> list[PromptRecord]:
    """Load the prompt rows that feed answer-spec generation."""

    if not path.exists():
        raise FileNotFoundError(path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [PromptRecord.model_validate(row) for row in rows]
    if len({record.id for record in records}) != len(records):
        raise ValueError(f"duplicate prompt IDs in {path}")
    return records


def answer_spec_instruction(prompt: PromptRecord) -> str:
    """Build the structured instruction used to generate one answer spec."""

    return f"""Create a factual answer specification for this user request.

Return only JSON matching the AnswerSpec schema. Do not write the answer prose.
List the facts the response must contain, sections the user explicitly requests,
necessary caveats or safety conditions, valid commands or code when applicable,
and claims the response must not make. Choose a proportionate target length.

Prompt ID: {prompt.id}
User request:
{prompt.prompt}
"""


async def generate_answer_spec(model: object, prompt: PromptRecord) -> AnswerSpec:
    """Generate and validate one in-memory answer specification."""

    result, _ = await model.a_generate(
        answer_spec_instruction(prompt),
        schema=AnswerSpec,
    )
    spec = AnswerSpec.model_validate(result)
    if spec.id != prompt.id:
        raise ValueError(f"prompt and answer spec IDs differ: {prompt.id!r}, {spec.id!r}")
    return spec


def build_answer_record(spec: AnswerSpec, final_response: str) -> AnswerRecord:
    """Create the persisted answer record for a completed specification."""

    return AnswerRecord(id=spec.id, final_response=final_response)


__all__ = [
    "AnswerRecord",
    "AnswerSpec",
    "DATA_DIR",
    "DEFAULT_PROMPTS",
    "answer_spec_instruction",
    "build_answer_record",
    "generate_answer_spec",
    "load_prompts",
]
