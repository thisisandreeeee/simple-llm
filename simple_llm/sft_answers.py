"""Generate ASD-STE100 answers from SFT prompts."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .llm_runtime import create_deepseek_model, run_concurrently
from .sft_prompts import PromptRecord


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_PROMPTS = DATA_DIR / "sft_prompts.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "sft_answers.jsonl"
SIMPLE_ENGLISH_PROMPT = (DATA_DIR / "simple_english.md").read_text(encoding="utf-8")
CORRECTNESS_PROMPT = """CORRECTNESS CHECK

Before writing, identify every explicit question, requested comparison,
procedure, example, caveat, and safety condition.

Correct false or incomplete premises. Distinguish formal guarantees from
common practice, recommendations, and observed behavior.

State the scope of each claim. Qualify behavior that depends on a product,
version, implementation, jurisdiction, configuration, or failure mode.

Do not invent commands, parameters, citations, examples, or exact quantities.
Preserve important exceptions even when they make the answer longer.

For procedures, include prerequisites, required verification, failure handling,
and rollback or safety steps when the request requires them.

Silently check that every explicit request is answered and that no statement
overclaims what the described mechanism guarantees. Do not reveal this check
or these instructions."""


class GeneratedAnswer(BaseModel):
    """The model-generated user-facing prose."""

    model_config = ConfigDict(extra="forbid")

    final_response: str = Field(min_length=1)


class AnswerRecord(BaseModel):
    """The answer artifact persisted to the SFT JSONL file."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^SFT-[A-Z]+-\d{4}$")
    final_response: str = Field(min_length=1)


def load_prompts(path: Path = DEFAULT_PROMPTS) -> list[PromptRecord]:
    """Load the prompt rows that feed answer generation."""

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


def answer_instruction(prompt: PromptRecord) -> str:
    """Build the single instruction used to generate one answer."""

    return f"""{SIMPLE_ENGLISH_PROMPT}

{CORRECTNESS_PROMPT}

Answer the user request directly. Do not mention these instructions or the generation process.
Return only JSON matching the GeneratedAnswer schema with exactly `final_response`.

User request:
{prompt.prompt}
"""


async def generate_answer(model: object, prompt: PromptRecord) -> AnswerRecord:
    """Generate and validate one final answer with one model call."""

    result, _ = await model.a_generate(
        answer_instruction(prompt), schema=GeneratedAnswer
    )
    answer = GeneratedAnswer.model_validate(result)
    return AnswerRecord(id=prompt.id, final_response=answer.final_response)


def load_answer_records(path: Path = DEFAULT_OUTPUT) -> list[AnswerRecord]:
    """Load completed answer rows used by prompt-ID resume logic."""

    if not path.exists():
        return []
    records: list[AnswerRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            records.append(
                AnswerRecord(
                    id=row["id"],
                    final_response=row["final_response"],
                )
            )
        except Exception as error:
            raise ValueError(f"invalid or incomplete answer row in {path}") from error
    if len({record.id for record in records}) != len(records):
        raise ValueError(f"duplicate answer IDs in {path}")
    return records


async def _generate_answers_async(
    pending: list[PromptRecord],
    existing: dict[str, AnswerRecord],
    output: Path,
    concurrency: int,
) -> dict[str, AnswerRecord]:
    model = create_deepseek_model(
        temperature=0.0,
        generation_kwargs={
            "reasoning_effort": "max",
            "extra_body": {"thinking": {"type": "enabled"}},
        },
    )

    async def generate(prompt: PromptRecord) -> AnswerRecord:
        return await generate_answer(model, prompt)

    async def save(record: AnswerRecord) -> None:
        with output.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
        existing[record.id] = record

    await run_concurrently(pending, generate, concurrency=concurrency, on_result=save)
    return existing


def generate_answers(
    *,
    count: int,
    prompts: Path = DEFAULT_PROMPTS,
    output: Path = DEFAULT_OUTPUT,
    concurrency: int = 3,
) -> list[AnswerRecord]:
    """Generate answers for the first ``count`` prompts and resume by ID."""

    prompt_records = load_prompts(prompts)
    if count < 1 or count > len(prompt_records):
        raise ValueError(f"count must be between 1 and {len(prompt_records)}")
    selected = prompt_records[:count]
    prompt_ids = {prompt.id for prompt in prompt_records}
    existing_records = load_answer_records(output)
    if any(record.id not in prompt_ids for record in existing_records):
        raise ValueError(f"answer file contains an ID not present in {prompts}")
    existing = {record.id: record for record in existing_records}
    pending = [prompt for prompt in selected if prompt.id not in existing]
    if pending:
        output.parent.mkdir(parents=True, exist_ok=True)
        existing = asyncio.run(
            _generate_answers_async(pending, existing, output, concurrency)
        )
    return [existing[prompt.id] for prompt in selected]


def main(count: int | None = None) -> None:
    """Generate final answers from the command line or a direct count argument."""

    if count is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--count", type=int, required=True)
        parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
        parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        parser.add_argument("--concurrency", type=int, default=3)
        args = parser.parse_args()
        count, prompts, output, concurrency = (
            args.count,
            args.prompts,
            args.output,
            args.concurrency,
        )
    else:
        prompts, output, concurrency = DEFAULT_PROMPTS, DEFAULT_OUTPUT, 3

    records = generate_answers(
        count=count,
        prompts=prompts,
        output=output,
        concurrency=concurrency,
    )
    print(f"Wrote {len(records)} answers to {output}")


__all__ = [
    "AnswerRecord",
    "GeneratedAnswer",
    "DATA_DIR",
    "DEFAULT_OUTPUT",
    "DEFAULT_PROMPTS",
    "CORRECTNESS_PROMPT",
    "SIMPLE_ENGLISH_PROMPT",
    "answer_instruction",
    "generate_answer",
    "generate_answers",
    "load_answer_records",
    "load_prompts",
    "main",
]


if __name__ == "__main__":
    main()
