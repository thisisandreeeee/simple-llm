"""Generate ASD-STE100 answers from SFT prompts."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .llm_runtime import create_deepseek_model, run_concurrently
from .scoring.rules import (
    document_limits,
    long_sentence_fraction,
    sentence_mechanics,
)
from .sft_prompts import PromptRecord


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_PROMPTS = DATA_DIR / "sft_prompts.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "sft_answers.jsonl"
KeyPoint = Annotated[str, Field(min_length=1)]


class AnswerSpecDraft(BaseModel):
    """Model-generated contract without code-owned metadata."""

    model_config = ConfigDict(extra="forbid")

    key_points: list[KeyPoint] = Field(min_length=1, max_length=8)


class AnswerSpec(AnswerSpecDraft):
    """Content contract generated before answer prose."""

    id: str = Field(pattern=r"^SFT-[A-Z]+-\d{4}$")


class AnswerRecord(BaseModel):
    """Answer artifact persisted to the SFT JSONL file."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^SFT-[A-Z]+-\d{4}$")
    key_points: list[KeyPoint] = Field(min_length=1, max_length=8)
    final_response: str = Field(min_length=1)


class GeneratedAnswer(BaseModel):
    """Only the model-generated user-facing prose."""

    model_config = ConfigDict(extra="forbid")

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

    return f"""Create the smallest sufficient content contract for this user request.

Return only JSON matching the answer content schema with `key_points`.
Do not write answer prose.
Do not include `id`; the caller assigns the prompt ID.

Each key point must be necessary to answer the explicit request correctly,
completely, or safely.
- Put one essential claim and its necessary qualifications in each point.
- Do not combine independent claims only to reduce the number of points.
- Keep a qualification with its claim. Usually make a separate consequence,
  procedure, or trade-off a separate point.
- Include a caveat only when omitting it could make the answer misleading.
- Include safety information only when the task requires it.
- Include commands, code, examples, or sections only when requested or necessary.
- Include a misconception correction as one consolidated key point.
- Do not add background knowledge, adjacent topics, generic best practices,
  repeated claims, or optional examples.
- Usually return 2 to 6 key points. Use up to 8 only when the request explicitly
  requires several independent topics or procedural steps.

Before returning, audit every factual claim:
- Correct a false or incomplete premise in the user request.
- Distinguish a formal guarantee from common practice or a recommendation.
- Distinguish a specification from observed runtime behavior.
- Do not infer behavior from a name, notation, category, or configuration label.
- State important conditions, exceptions, and scope boundaries.
- Qualify behavior that depends on a product, version, jurisdiction, or
  implementation instead of presenting it as universal.
- Do not use exact quantities or absolute terms such as `always` or `never`
  unless the facts require them.
- Ask whether a valid implementation, process, or deployment could contradict
  each claim. If yes, qualify the claim or name the applicable implementation.
- Check whether the relevant standard defines the behavior or whether it is only
  a common operational practice.
- Do not confuse a representation or configuration label with runtime behavior,
  lifecycle policy, statistical behavior, or security strength.
- Use `must` only for a formal requirement or necessary safety condition.
- Do not assign properties such as static, long-lived, secure, atomic, or
  exactly-once unless they follow from the stated mechanism.
- When the request covers several implementations, state the common guarantee
  first. Describe implementation-specific behavior separately and name the
  implementation.

Before returning, remove each point whose absence would not materially reduce
correctness, task completion, or safety.

Prompt ID: {prompt.id}
User request:
{prompt.prompt}
"""


def answer_instruction(prompt: PromptRecord, spec: AnswerSpec) -> str:
    """Build the correctness-first instruction used to write one answer."""

    contract = json.dumps(
        spec.model_dump(exclude={"id"}), ensure_ascii=False, indent=2
    )
    return f"""Write the final answer to the user request below.

Return only JSON matching the GeneratedAnswer schema with exactly
`final_response`. Do not mention the content contract, this instruction, or the
generation process.

Priorities, in order:
1. Be factually correct. Never trade correctness for simplicity.
2. Fulfill every explicit request and preserve necessary detail.
3. Organize the answer so that each sentence has one clear purpose.
4. Use clear, simplified technical English where it does not change meaning.

Cover every key point. A key point can require more than one sentence. Do not
introduce a new factual claim, recommendation, example, or preference unless it
is necessary for accuracy. Do not add background or optional detail.

Writing rules:
- Use one main idea per sentence. It can include one closely related condition,
  qualification, or result.
- Most descriptive sentences should contain 12 to 20 words.
- Avoid consecutive short sentences that repeat the same subject.
- Combine adjacent short sentences when the result remains clear and within the
  applicable word limit.
- Do not exceed 20 words in procedural sentences or 25 words in descriptive
  sentences unless exact technical text makes this unavoidable.
- Do not use semicolons.
- Use `for example` instead of `e.g.` and `that is` instead of `i.e.`.
- Put a condition before the action that depends on it.
- Prefer active voice when the actor is known and important.
- Use one term for one concept. Explain an unavoidable technical term at first use.
- Do not put more than six sentences in one paragraph.
- Use a vertical list for three or more parallel items.
- Use numbered steps only when sequence is important.
- Usually express each key point in one or two sentences. Use more only for a
  requested procedure, example, or necessary qualification.
- Do not add an introduction that repeats the user request.
- Do not add a conclusion that only restates the answer.
- Keep the final response shorter than the combined key points unless expansion
  is necessary for an explanation or procedure.

Keep technical terms, commands, identifiers, units, and code exact. Preserve
necessary caveats, prerequisites, and safety conditions. Use the tense and verb
forms needed for accurate meaning. Do not create fragments, repetitive sentence
patterns, or unsupported claims. Keep the answer proportionate to the request.

Prompt ID: {prompt.id}
User request:
{prompt.prompt}

Content contract:
{contract}
"""


def answer_style_violations(final_response: str) -> list[str]:
    """Return deterministic STE-style violations that warrant one rewrite."""

    violations: list[str] = []
    if long_sentence_fraction("", final_response) > 0:
        violations.append(
            "One or more sentences exceed 20 words for procedures or 25 words "
            "for descriptive text. Edit only those sentences."
        )
    if sentence_mechanics("", final_response) < 1:
        violations.append(
            "The answer contains a semicolon, contraction, or Latin abbreviation."
        )
    if document_limits("", final_response) < 1:
        violations.append(
            "One or more paragraphs exceed six sentences. Split only those "
            "paragraphs."
        )
    return violations


def answer_repair_instruction(
    prompt: PromptRecord,
    spec: AnswerSpec,
    final_response: str,
    violations: list[str],
) -> str:
    """Build one style-only repair instruction for a generated answer."""

    contract = json.dumps(
        spec.model_dump(exclude={"id"}), ensure_ascii=False, indent=2
    )
    problems = "\n".join(f"- {violation}" for violation in violations)
    return f"""Rewrite the answer to correct only the listed style violations.

Return only JSON matching the GeneratedAnswer schema with exactly
`final_response`.

Preserve every factual claim, qualification, condition, technical term, command,
and warning. Do not add or remove information. Do not mention this instruction
or the rewrite.

Edit only the sentences or paragraphs that violate the listed rules. Copy all
compliant text without rewriting it.
- Split only an over-limit sentence. Do not split a compliant sentence.
- Combine consecutive sentences shorter than 8 words when the result stays
  within the applicable limit.
- Avoid repeating the same subject in consecutive sentences.
- Preserve the original paragraph and list structure unless that structure is
  itself a violation.
- Do not increase the total word count.

User request:
{prompt.prompt}

Content contract:
{contract}

Answer to repair:
{final_response}

Violations:
{problems}
"""


async def generate_answer_spec(model: object, prompt: PromptRecord) -> AnswerSpec:
    """Generate and validate one in-memory answer specification."""

    result, _ = await model.a_generate(
        answer_spec_instruction(prompt), schema=AnswerSpecDraft
    )
    draft = AnswerSpecDraft.model_validate(result)
    return AnswerSpec(id=prompt.id, **draft.model_dump())


def build_answer_record(spec: AnswerSpec, final_response: str) -> AnswerRecord:
    """Create the persisted answer record for a completed specification."""

    return AnswerRecord(
        id=spec.id,
        key_points=spec.key_points,
        final_response=final_response,
    )


async def generate_answer(
    model: object,
    prompt: PromptRecord,
    spec: AnswerSpec,
) -> AnswerRecord:
    """Generate and validate one final answer from its factual specification."""

    if prompt.id != spec.id:
        raise ValueError(f"prompt and answer spec IDs differ: {prompt.id!r}, {spec.id!r}")
    result, _ = await model.a_generate(
        answer_instruction(prompt, spec), schema=GeneratedAnswer
    )
    answer = GeneratedAnswer.model_validate(result)
    violations = answer_style_violations(answer.final_response)
    if violations:
        result, _ = await model.a_generate(
            answer_repair_instruction(
                prompt, spec, answer.final_response, violations
            ),
            schema=GeneratedAnswer,
        )
        answer = GeneratedAnswer.model_validate(result)
    return build_answer_record(spec, answer.final_response)


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
            if "key_points" not in row and "required_facts" in row:
                row["key_points"] = row.pop("required_facts")
                for field in (
                    "required_sections",
                    "caveats_and_safety",
                    "valid_commands_or_code",
                    "prohibited_claims",
                    "answer_type",
                    "approximate_length",
                ):
                    row.pop(field, None)
            records.append(AnswerRecord.model_validate(row))
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
    spec_model = create_deepseek_model(
        temperature=0.0,
        generation_kwargs={
            "reasoning_effort": "max",
            "extra_body": {"thinking": {"type": "enabled"}},
        },
    )
    answer_model = create_deepseek_model(temperature=0.0)

    async def generate(prompt: PromptRecord) -> AnswerRecord:
        spec = await generate_answer_spec(spec_model, prompt)
        return await generate_answer(answer_model, prompt, spec)

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
    "AnswerSpec",
    "AnswerSpecDraft",
    "GeneratedAnswer",
    "DATA_DIR",
    "DEFAULT_OUTPUT",
    "DEFAULT_PROMPTS",
    "answer_spec_instruction",
    "answer_instruction",
    "answer_repair_instruction",
    "answer_style_violations",
    "build_answer_record",
    "generate_answer",
    "generate_answers",
    "generate_answer_spec",
    "load_answer_records",
    "load_prompts",
    "main",
]


if __name__ == "__main__":
    main()
