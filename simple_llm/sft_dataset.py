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
    "mathematics and statistics",
    "natural sciences",
    "engineering and manufacturing",
    "business and operations",
    "workplace communication",
    "education and learning",
    "household and consumer systems",
    "transport and logistics",
    "history and geography",
    "writing and information design",
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
    "mathematics and statistics": "MTH",
    "natural sciences": "SCI",
    "engineering and manufacturing": "ENG",
    "business and operations": "BUS",
    "workplace communication": "COM",
    "education and learning": "EDU",
    "household and consumer systems": "HOM",
    "transport and logistics": "TRN",
    "history and geography": "HIS",
    "writing and information design": "WRI",
}
TECHNICAL_SUBJECTS = SUBJECTS[:10]
GENERAL_SUBJECTS = SUBJECTS[10:]
SUBJECT_WEIGHTS = {
    **{subject: 0.06 for subject in TECHNICAL_SUBJECTS},
    **{subject: 0.04 for subject in GENERAL_SUBJECTS},
}

TOPIC_CATALOG_PATH = DATA_DIR / "sft_topics.json"


def _load_topic_catalog(path: Path = TOPIC_CATALOG_PATH) -> dict[str, tuple[str, ...]]:
    """Load and validate the subject/topic catalog used by prompt planning."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != set(SUBJECTS):
        raise ValueError("topic catalog subjects must match SUBJECTS exactly")
    catalog: dict[str, tuple[str, ...]] = {}
    for subject in SUBJECTS:
        topics = raw[subject]
        if not isinstance(topics, list) or not topics or not all(isinstance(topic, str) for topic in topics):
            raise ValueError(f"topic catalog entry is invalid: {subject}")
        normalised = [re.sub(r"\s+", " ", topic.casefold()).strip() for topic in topics]
        if len(set(normalised)) != len(normalised):
            raise ValueError(f"topic catalog contains duplicate topics: {subject}")
        catalog[subject] = tuple(topics)
    return catalog


TOPICS = _load_topic_catalog()

Length = Literal["short", "medium", "long"]
Intent = Literal[
    "explanation",
    "troubleshooting",
    "procedure",
    "comparison",
    "documentation",
    "misconception correction",
]
Audience = Literal["beginner", "practitioner", "expert"]
AUDIENCE_GUIDANCE = {
    "beginner": "Assume little prior knowledge and introduce necessary concepts clearly.",
    "practitioner": "Assume practical familiarity and focus on application and trade-offs.",
    "expert": "Assume strong background knowledge and focus on precision and edge cases.",
}


class PromptSpec(BaseModel):
    """One planned prompt before the teacher writes its user-facing wording."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^SFT-[A-Z]+-\d{4}$")
    subject: str
    topic: str
    intent: Intent
    audience: Audience
    expected_length: Length


class PromptRecord(BaseModel):
    """The minimal serialized prompt artifact."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^SFT-[A-Z]+-\d{4}$")
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
        ("subject", SUBJECT_WEIGHTS),
        ("audience", {"beginner": 0.30, "practitioner": 0.40, "expert": 0.30}),
        ("expected_length", {"short": 0.30, "medium": 0.40, "long": 0.30}),
        (
            "intent",
            {
                "explanation": 0.25,
                "documentation": 0.20,
                "troubleshooting": 0.15,
                "procedure": 0.15,
                "comparison": 0.15,
                "misconception correction": 0.10,
            },
        ),
    ]
    values: dict[str, list[str]] = {}
    for name, proportions in dimensions:
        counts = allocate_counts(count, proportions)
        values[name] = [value for value, amount in counts.items() for _ in range(amount)]
        rng.shuffle(values[name])

    topic_orders = {subject: list(topics) for subject, topics in TOPICS.items()}
    for topics in topic_orders.values():
        rng.shuffle(topics)

    specs: list[PromptSpec] = []
    subject_numbers = Counter[str]()
    for index in range(count):
        subject = values["subject"][index]
        subject_numbers[subject] += 1
        audience = values["audience"][index]
        specs.append(
            PromptSpec(
                id=f"SFT-{SUBJECT_CODES[subject]}-{subject_numbers[subject]:04d}",
                subject=subject,
                topic=topic_orders[subject][(subject_numbers[subject] - 1) % len(topic_orders[subject])],
                intent=values["intent"][index],
                audience=audience,
                expected_length=values["expected_length"][index],
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


def _prompt_issues(prompt: str) -> list[str]:
    """Return cheap checks for prompts that cannot be answered standalone."""

    words = prompt.split()
    issues: list[str] = []
    if len(words) < 8:
        issues.append("prompt is too short")
    if len(words) > 100:
        issues.append("prompt is too long")
    if re.search(r"\[\s*insert|\battached\b|\bfollowing excerpt\b|\bthis documentation\b", prompt, re.I):
        issues.append("prompt refers to source material that was not supplied")
    return issues


def _load_prompt_records(path: Path) -> list[PromptRecord]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [PromptRecord(id=row["id"], prompt=row["prompt"]) for row in rows]
    if len({record.id for record in records}) != len(records):
        raise ValueError(f"duplicate prompt IDs in {path}")
    if any(set(row) != {"id", "prompt"} for row in rows):
        path.write_text(
            "".join(record.model_dump_json() + "\n" for record in records),
            encoding="utf-8",
        )
    return records


def _load_eval_prompts(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(set(row) != {"id", "prompt"} for row in rows):
        raise ValueError(f"evaluation rows must contain only id and prompt: {path}")
    return [row["prompt"] for row in rows]


def _prompt_instruction(specs: list[PromptSpec], feedback: str = "") -> str:
    assignments = json.dumps(
        [
            {
                "id": spec.id,
                "subject": spec.subject,
                "topic": spec.topic,
                "intent": spec.intent,
                "audience_guidance": AUDIENCE_GUIDANCE[spec.audience],
                "expected_length": spec.expected_length,
            }
            for spec in specs
        ],
        ensure_ascii=False,
    )
    retry = f"\nPrevious attempt failed: {feedback}\nFix those issues.\n" if feedback else ""
    return f"""Generate one natural user request for each assignment below.

Return JSON with exactly one `prompts` array. Each item must contain only `id` and `prompt`,
and each assignment ID must appear exactly once. Do not include answers, facts, labels, or
generation commentary. Phrase requests as genuine user questions or documentation tasks.
Use the assigned subject, topic, and intent. Use the intent when it fits the topic naturally;
if it does not, use a direct explanation or comparison instead. When correcting a misconception,
state the concrete mistaken claim rather than asking generally about “common misconceptions”.
Express the user's background through context and requested depth; do not mechanically say
“beginner”, “practitioner”, “expert”, or “for an expert audience”. Every prompt must be answerable
from its own text: do not refer to attachments, excerpts, drafts, “this documentation”, or missing
context. Do not ask for exact commands unless the relevant product or environment is named.
Prefer stable facts and concepts. Avoid current prices, product rankings, interface layouts, and
unspecified software versions. A product-specific procedure must name the product and environment;
a jurisdiction-dependent question must name the jurisdiction or stay general. For potentially
hazardous equipment, electrical work, laboratory work, or transport operations, request safety-first
guidance and avoid pretending that a generic answer replaces a qualified operator or manual. Vary
wording and requested format naturally, but avoid repeating stock phrases such as “without oversimplifying”.
Do not copy or paraphrase evaluation prompts; the caller checks this locally.

Assignments:
{assignments}
{retry}
"""


async def _generate_prompt_batch(
    model: object,
    specs: list[PromptSpec],
    eval_prompts: list[str],
    retry_limit: int,
) -> list[GeneratedPrompt]:
    from openai import APITimeoutError
    from pydantic import ValidationError

    expected_ids = {spec.id for spec in specs}
    feedback = ""
    for attempt in range(retry_limit + 1):
        try:
            result, _ = await model.a_generate(
                _prompt_instruction(specs, feedback), schema=PromptBatch
            )
            batch = PromptBatch.model_validate(result)
            if {item.id for item in batch.prompts} != expected_ids:
                raise ValueError("teacher returned the wrong prompt IDs")
            if len(batch.prompts) != len(specs):
                raise ValueError("teacher returned the wrong prompt count")
            prompts = [item.prompt for item in batch.prompts]
            if len({_normalise_prompt(prompt) for prompt in prompts}) != len(prompts):
                raise ValueError("teacher returned duplicate prompts")
            issues: dict[str, list[str]] = {}
            for spec, prompt in zip(specs, prompts):
                prompt_issues = _prompt_issues(prompt)
                if prompt_issues:
                    issues[spec.id] = prompt_issues
            if issues:
                raise ValueError(f"invalid standalone prompts: {issues}")
            if any(_is_duplicate(prompt, eval_prompts) for prompt in prompts):
                raise ValueError("teacher reused an evaluation prompt")
            return batch.prompts
        except (ValidationError, ValueError) as error:
            feedback = str(error)
            if attempt == retry_limit:
                raise
        except (TimeoutError, APITimeoutError):
            if attempt == retry_limit:
                raise
            await asyncio.sleep(2**attempt)
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

    async def generate_one(
        batch_index: int, batch: list[PromptSpec]
    ) -> tuple[int, list[GeneratedPrompt]]:
        async with semaphore:
            return batch_index, await _generate_prompt_batch(
                model, batch, eval_prompts, retry_limit
            )

    records = list(existing)
    tasks = [
        asyncio.create_task(generate_one(index, batch))
        for index, batch in enumerate(batches)
    ]
    try:
        for completed in asyncio.as_completed(tasks):
            batch_index, generated = await completed
            batch_specs = batches[batch_index]
            generated_by_id = {item.id: item for item in generated}
            with output.open("a", encoding="utf-8") as handle:
                for spec in batch_specs:
                    prompt = generated_by_id[spec.id].prompt
                    if _is_duplicate(prompt, known_prompts):
                        raise ValueError(
                            f"generated prompt duplicates an existing prompt: {spec.id}"
                        )
                    record = PromptRecord(id=spec.id, prompt=prompt)
                    handle.write(record.model_dump_json() + "\n")
                    records.append(record)
                    known_prompts.append(prompt)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return sorted(records, key=lambda record: record.id)


def generate_prompts(
    *,
    count: int = DEFAULT_COUNT,
    seed: int = DEFAULT_SEED,
    output: Path = DATA_DIR / "sft_prompts.jsonl",
    evals: Path = DATA_DIR / "evals.jsonl",
    batch_size: int = 10,
    concurrency: int = 3,
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
    parser.add_argument("--concurrency", type=int, default=3)
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
    "TOPICS",
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
