"""Typed scaffolding and deterministic planning for SFT dataset generation."""

from __future__ import annotations

import random
import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal
from difflib import SequenceMatcher

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_COUNT = 3_000
DEFAULT_SEED = 42
TEACHER_MODEL = "deepseek-v4-pro"

SUBJECTS = (
    "networking and internet",
    "machine learning",
    "data formats and APIs",
    "databases and data systems",
    "distributed systems",
    "security and identity",
    "cloud and infrastructure",
    "programming languages and runtimes",
    "developer tools",
    "software architecture",
)
SUBJECT_CODES = {
    "networking and internet": "NET",
    "machine learning": "ML",
    "data formats and APIs": "API",
    "databases and data systems": "DBS",
    "distributed systems": "DST",
    "security and identity": "SEC",
    "cloud and infrastructure": "CLD",
    "programming languages and runtimes": "PLR",
    "developer tools": "DEV",
    "software architecture": "ARC",
}

Dimension = Literal["easy", "medium", "hard"]
Length = Literal["short", "medium", "long"]
Terminology = Literal["minimal", "moderate", "heavy"]
Risk = Literal["low", "medium", "high"]
TaskType = Literal["explanation", "documentation"]
TargetCategory = Literal[
    "concise rewrite",
    "corrected answer",
    "procedure",
    "qualified comparison",
    "direct answer",
]


class PromptSpec(BaseModel):
    """One planned prompt before the teacher writes its user-facing wording."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^SFT-[A-Z]+-\d{4}$")
    subject: str
    difficulty: Dimension
    expected_length: Length
    terminology: Terminology
    oversimplification_risk: Risk
    task_type: TaskType
    target_category: TargetCategory


class PromptRecord(BaseModel):
    """The minimal serialized prompt artifact."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^SFT-[A-Z]+-\d{4}$")
    prompt: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class GeneratedPrompt(BaseModel):
    """The minimal structured response requested from the prompt teacher."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^SFT-[A-Z]+-\d{4}$")
    prompt: str = Field(min_length=1)


class PromptBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompts: list[GeneratedPrompt] = Field(min_length=1)


class AnswerArtifacts(BaseModel):
    """Ordered teacher artifacts; only ``final_response`` is training data."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^SFT-[A-Z]+-\d{4}$")
    facts: list[str] = Field(min_length=1)
    required_sections: list[str] = Field(default_factory=list)
    caveats_and_safety: list[str] = Field(default_factory=list)
    valid_commands_or_code: list[str] = Field(default_factory=list)
    prohibited_claims: list[str] = Field(default_factory=list)
    answer_type: str = Field(min_length=1)
    approximate_length: str = Field(min_length=1)
    prose: str = Field(min_length=1)
    final_response: str = Field(min_length=1)


class SFTMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class SFTExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[SFTMessage] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def require_user_then_assistant(self) -> "SFTExample":
        if [message.role for message in self.messages] != ["user", "assistant"]:
            raise ValueError("messages must contain user then assistant only")
        return self


def allocate_counts(total: int, proportions: dict[str, float]) -> dict[str, int]:
    """Allocate ``total`` items by largest remainder, preserving exact totals."""

    if total < 0 or not proportions or any(value < 0 for value in proportions.values()):
        raise ValueError("total must be non-negative and proportions must be non-empty")
    denominator = sum(proportions.values())
    if denominator <= 0:
        raise ValueError("proportions must have a positive sum")
    exact = {key: total * value / denominator for key, value in proportions.items()}
    result = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(result.values())
    order = sorted(
        proportions,
        key=lambda key: (exact[key] - result[key], key),
        reverse=True,
    )
    for key in order[:remainder]:
        result[key] += 1
    return result


def build_strata(count: int = DEFAULT_COUNT, seed: int = DEFAULT_SEED) -> list[PromptSpec]:
    """Create an independently shuffled, exactly sized stratification schedule."""

    if count < 1:
        raise ValueError("count must be at least 1")
    rng = random.Random(seed)
    dimensions = [
        ("subject", {subject: 1 for subject in SUBJECTS}),
        ("difficulty", {"easy": 0.30, "medium": 0.40, "hard": 0.30}),
        ("expected_length", {"short": 0.30, "medium": 0.40, "long": 0.30}),
        ("terminology", {"minimal": 0.30, "moderate": 0.40, "heavy": 0.30}),
        (
            "oversimplification_risk",
            {"low": 0.30, "medium": 0.35, "high": 0.35},
        ),
        ("task_type", {"explanation": 2, "documentation": 1}),
        (
            "target_category",
            {
                "concise rewrite": 0.40,
                "corrected answer": 0.30,
                "procedure": 0.15,
                "qualified comparison": 0.10,
                "direct answer": 0.05,
            },
        ),
    ]
    values: dict[str, list[str]] = {}
    for name, proportions in dimensions:
        counts = allocate_counts(count, proportions)
        values[name] = [value for value, amount in counts.items() for _ in range(amount)]
        rng.shuffle(values[name])

    specs: list[PromptSpec] = []
    subject_numbers = Counter[str]()
    for index in range(count):
        subject = values["subject"][index]
        subject_numbers[subject] += 1
        specs.append(
            PromptSpec(
                id=f"SFT-{SUBJECT_CODES[subject]}-{subject_numbers[subject]:04d}",
                subject=subject,
                difficulty=values["difficulty"][index],
                expected_length=values["expected_length"][index],
                terminology=values["terminology"][index],
                oversimplification_risk=values["oversimplification_risk"][index],
                task_type=values["task_type"][index],
                target_category=values["target_category"][index],
            )
        )
    return specs


def format_sft_example(prompt: PromptRecord, answer: AnswerArtifacts) -> SFTExample:
    """Convert validated prompt and answer artifacts to the two-role SFT format."""

    if prompt.id != answer.id:
        raise ValueError(f"prompt and answer IDs differ: {prompt.id!r}, {answer.id!r}")
    return SFTExample(
        messages=[
            SFTMessage(role="user", content=prompt.prompt),
            SFTMessage(role="assistant", content=answer.final_response),
        ]
    )


def _normalise_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.casefold()).strip()


def _is_duplicate(prompt: str, existing: list[str]) -> bool:
    """Reject exact and obvious near-duplicates against prior prompts."""

    normalised = _normalise_prompt(prompt)
    words = set(normalised.split())
    for candidate in existing:
        other = _normalise_prompt(candidate)
        if normalised == other:
            return True
        other_words = set(other.split())
        if words and other_words:
            overlap = len(words & other_words) / len(words | other_words)
            if overlap >= 0.90 or SequenceMatcher(None, normalised, other).ratio() >= 0.94:
                return True
    return False


def _load_prompt_records(path: Path) -> list[PromptRecord]:
    if not path.exists():
        return []
    records = [PromptRecord.model_validate(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({record.id for record in records}) != len(records):
        raise ValueError(f"duplicate prompt IDs in {path}")
    return records


def _load_eval_prompts(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(set(row) != {"id", "prompt"} for row in rows):
        raise ValueError(f"evaluation rows must contain only id and prompt: {path}")
    return [row["prompt"] for row in rows]


def _prompt_instruction(specs: list[PromptSpec]) -> str:
    assignments = json.dumps([spec.model_dump() for spec in specs], ensure_ascii=False)
    return f"""Generate one natural technical user request for each assignment below.

Return JSON with exactly one `prompts` array. Each item must contain only `id` and `prompt`,
and each assignment ID must appear exactly once. Do not include answers, facts, labels, or
generation commentary. Phrase requests as genuine user questions or documentation tasks.
Use stable, authoritative technical topics. Do not copy or paraphrase evaluation prompts;
the caller checks this locally.
Keep the primary purpose clear, but vary wording, audience, failure mode, and requested format.

Assignments:
{assignments}
"""


async def _generate_prompt_batch(
    model: object,
    specs: list[PromptSpec],
    eval_prompts: list[str],
    retry_limit: int,
) -> list[GeneratedPrompt]:
    from pydantic import ValidationError

    expected_ids = {spec.id for spec in specs}
    for attempt in range(retry_limit + 1):
        try:
            result, _ = await model.a_generate(
                _prompt_instruction(specs), schema=PromptBatch
            )
            batch = PromptBatch.model_validate(result)
            if {item.id for item in batch.prompts} != expected_ids:
                raise ValueError("teacher returned the wrong prompt IDs")
            if len(batch.prompts) != len(specs):
                raise ValueError("teacher returned the wrong prompt count")
            prompts = [item.prompt for item in batch.prompts]
            if len({_normalise_prompt(prompt) for prompt in prompts}) != len(prompts):
                raise ValueError("teacher returned duplicate prompts")
            if any(_is_duplicate(prompt, eval_prompts) for prompt in prompts):
                raise ValueError("teacher reused an evaluation prompt")
            return batch.prompts
        except (ValidationError, ValueError):
            if attempt == retry_limit:
                raise
    raise AssertionError("unreachable")


async def _generate_prompts_async(
    specs: list[PromptSpec],
    eval_prompts: list[str],
    output: Path,
    batch_size: int,
    concurrency: int,
    retry_limit: int,
    resume: bool,
) -> list[PromptRecord]:
    from deepeval.models import DeepSeekModel

    if batch_size < 1 or concurrency < 1 or retry_limit < 0:
        raise ValueError("batch_size and concurrency must be positive; retry_limit cannot be negative")
    existing = _load_prompt_records(output) if resume else []
    spec_by_id = {spec.id: spec for spec in specs}
    if any(record.id not in spec_by_id for record in existing):
        raise ValueError("existing prompt file contains IDs outside the current schedule")
    known_prompts = eval_prompts + [record.prompt for record in existing]
    pending = [spec for spec in specs if spec.id not in {record.id for record in existing}]
    if not pending:
        return sorted(existing, key=lambda record: record.id)

    model = DeepSeekModel(model=TEACHER_MODEL, temperature=0.7)
    semaphore = asyncio.Semaphore(concurrency)
    batches = [pending[index : index + batch_size] for index in range(0, len(pending), batch_size)]

    async def generate_one(batch: list[PromptSpec]) -> list[GeneratedPrompt]:
        async with semaphore:
            return await _generate_prompt_batch(model, batch, eval_prompts, retry_limit)

    results = await asyncio.gather(*(generate_one(batch) for batch in batches))
    records = list(existing)
    for batch_specs, generated in zip(batches, results):
        generated_by_id = {item.id: item for item in generated}
        for spec in batch_specs:
            prompt = generated_by_id[spec.id].prompt
            if _is_duplicate(prompt, known_prompts):
                raise ValueError(f"generated prompt duplicates an existing prompt: {spec.id}")
            record = PromptRecord(id=spec.id, prompt=prompt)
            with output.open("a", encoding="utf-8") as handle:
                handle.write(record.model_dump_json() + "\n")
            records.append(record)
            known_prompts.append(prompt)
    return sorted(records, key=lambda record: record.id)


def generate_prompts(
    *,
    count: int = DEFAULT_COUNT,
    seed: int = DEFAULT_SEED,
    output: Path = DATA_DIR / "sft_prompts.jsonl",
    evals: Path = DATA_DIR / "evals.jsonl",
    batch_size: int = 10,
    concurrency: int = 10,
    retry_limit: int = 2,
    resume: bool = True,
) -> list[PromptRecord]:
    """Generate and checkpoint stratified prompts using the DeepSeek teacher."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if not resume and output.exists():
        output.write_text("", encoding="utf-8")
    return asyncio.run(
        _generate_prompts_async(
            build_strata(count, seed),
            _load_eval_prompts(evals),
            output,
            batch_size,
            concurrency,
            retry_limit,
            resume,
        )
    )


def main() -> None:
    """CLI orchestration entry point; later stages plug into this pipeline."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "sft_prompts.jsonl")
    parser.add_argument("--evals", type=Path, default=DATA_DIR / "evals.jsonl")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--retry-limit", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    records = generate_prompts(
        count=args.count,
        seed=args.seed,
        output=args.output,
        evals=args.evals,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        retry_limit=args.retry_limit,
        resume=not args.no_resume,
    )
    print(f"Wrote {len(records)} prompts to {args.output}")


__all__ = [
    "AnswerArtifacts",
    "DATA_DIR",
    "DEFAULT_COUNT",
    "DEFAULT_SEED",
    "PromptRecord",
    "PromptSpec",
    "GeneratedPrompt",
    "PromptBatch",
    "SFTExample",
    "SFTMessage",
    "TEACHER_MODEL",
    "allocate_counts",
    "build_strata",
    "format_sft_example",
    "generate_prompts",
    "main",
]


if __name__ == "__main__":
    main()
