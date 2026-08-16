"""Generate factual answer specifications for the SFT prompts."""

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


class AnswerSpec(BaseModel):
    """Minimal factual contract generated from one prompt."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^SFT-[A-Z]+-\d{4}$")
    required_facts: list[str] = Field(min_length=1)


class AnswerRecord(BaseModel):
    """Accepted answer persisted after a final response is written."""

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

Return only JSON matching the AnswerSpec schema with exactly `id` and
`required_facts`. Do not write the answer prose. List the factual claims the final
response must contain, without adding unsupported or speculative claims.

Prompt ID: {prompt.id}
User request:
{prompt.prompt}
"""


async def generate_answer_spec(model: object, prompt: PromptRecord) -> AnswerSpec:
    """Generate and validate one in-memory answer specification."""

    result, _ = await model.a_generate(answer_spec_instruction(prompt), schema=AnswerSpec)
    spec = AnswerSpec.model_validate(result)
    if spec.id != prompt.id:
        raise ValueError(f"prompt and answer spec IDs differ: {prompt.id!r}, {spec.id!r}")
    return spec


def build_answer_record(spec: AnswerSpec, final_response: str) -> AnswerRecord:
    """Create the persisted answer record for a completed specification."""

    return AnswerRecord(id=spec.id, final_response=final_response)


async def _generate_answer_specs_async(
    prompts: list[PromptRecord],
    output: Path,
    concurrency: int,
) -> list[AnswerSpec]:
    model = create_deepseek_model(temperature=0.0)

    async def generate(prompt: PromptRecord) -> AnswerSpec:
        return await generate_answer_spec(model, prompt)

    specs = await run_concurrently(prompts, generate, concurrency=concurrency)
    by_id = {spec.id: spec for spec in specs}
    ordered = [by_id[prompt.id] for prompt in prompts]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(spec.model_dump_json() + "\n" for spec in ordered),
        encoding="utf-8",
    )
    return ordered


def generate_answer_specs(
    *,
    count: int,
    prompts: Path = DEFAULT_PROMPTS,
    output: Path = DEFAULT_OUTPUT,
    concurrency: int = 3,
) -> list[AnswerSpec]:
    """Generate exactly ``count`` answer specs and write them as JSONL."""

    records = load_prompts(prompts)
    if count < 1 or count > len(records):
        raise ValueError(f"count must be between 1 and {len(records)}")
    return asyncio.run(_generate_answer_specs_async(records[:count], output, concurrency))


def main(count: int | None = None) -> None:
    """Generate answer specs from the command line or a direct count argument."""

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

    specs = generate_answer_specs(
        count=count,
        prompts=prompts,
        output=output,
        concurrency=concurrency,
    )
    print(f"Wrote {len(specs)} answer specs to {output}")


__all__ = [
    "AnswerRecord",
    "AnswerSpec",
    "DATA_DIR",
    "DEFAULT_OUTPUT",
    "DEFAULT_PROMPTS",
    "answer_spec_instruction",
    "build_answer_record",
    "generate_answer_spec",
    "generate_answer_specs",
    "load_prompts",
    "main",
]


if __name__ == "__main__":
    main()
