"""Build a two-turn JSONL dataset for supervised fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

from .sft_answers import AnswerRecord, load_answer_records, load_prompts
from .sft_prompts import (
    DATA_DIR,
    PromptRecord,
    SFTExample,
    SFTMessage,
)

DEFAULT_PROMPTS = DATA_DIR / "sft_prompts.jsonl"
DEFAULT_ANSWERS = DATA_DIR / "sft_answers.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "sft_dataset.jsonl"


def build_dataset(
    prompts: list[PromptRecord], answers: list[AnswerRecord]
) -> list[SFTExample]:
    """Join completed answers to prompts by ID, preserving prompt order."""

    prompt_by_id = {prompt.id: prompt for prompt in prompts}
    answer_by_id = {answer.id: answer for answer in answers}
    unknown = answer_by_id.keys() - prompt_by_id.keys()
    if unknown:
        raise ValueError(f"answers contain unknown prompt IDs: {sorted(unknown)}")

    return [
        SFTExample(
            messages=[
                SFTMessage(role="user", content=prompt.prompt),
                SFTMessage(role="assistant", content=answer_by_id[prompt.id].final_response),
            ]
        )
        for prompt in prompts
        if prompt.id in answer_by_id
    ]


def write_dataset(
    output: Path = DEFAULT_OUTPUT,
    prompts_path: Path = DEFAULT_PROMPTS,
    answers_path: Path = DEFAULT_ANSWERS,
) -> int:
    """Write one JSON object containing only ``messages`` per output row."""

    examples = build_dataset(load_prompts(prompts_path), load_answer_records(answers_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(example.model_dump_json() + "\n" for example in examples),
        encoding="utf-8",
    )
    return len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    count = write_dataset(args.output, args.prompts, args.answers)
    print(f"Wrote {count} examples to {args.output}")


if __name__ == "__main__":
    main()
