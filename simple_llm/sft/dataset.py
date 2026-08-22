"""Build a two-turn JSONL dataset for supervised fine-tuning."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from random import Random

from .answers import AnswerRecord, load_answer_records, load_prompts
from .prompts import (
    DATA_DIR,
    PromptRecord,
    SFTExample,
    SFTMessage,
    SFTTextContent,
)

DEFAULT_PROMPTS = DATA_DIR / "sft_prompts.jsonl"
DEFAULT_ANSWERS = DATA_DIR / "sft_answers.jsonl"
DEFAULT_TRAIN_OUTPUT = DATA_DIR / "sft_train.jsonl"
DEFAULT_EVAL_OUTPUT = DATA_DIR / "sft_eval.jsonl"
EVAL_FRACTION = 0.1
SEED = 42


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
                SFTMessage(
                    role="user", content=[SFTTextContent(text=prompt.prompt)]
                ),
                SFTMessage(
                    role="assistant",
                    content=[
                        SFTTextContent(text=answer_by_id[prompt.id].final_response)
                    ],
                ),
            ]
        )
        for prompt in prompts
        if prompt.id in answer_by_id
    ]


def split_dataset(
    prompts: list[PromptRecord],
    answers: list[AnswerRecord],
    eval_fraction: float = EVAL_FRACTION,
    seed: int = SEED,
) -> tuple[list[SFTExample], list[SFTExample]]:
    """Create deterministic subject-stratified train and evaluation sets."""

    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between 0 and 1")
    examples = build_dataset(prompts, answers)
    answer_ids = {answer.id for answer in answers}
    completed_ids = [prompt.id for prompt in prompts if prompt.id in answer_ids]
    example_by_id = dict(zip(completed_ids, examples, strict=True))
    ids_by_subject: dict[str, list[str]] = defaultdict(list)
    for prompt_id in completed_ids:
        ids_by_subject[prompt_id.split("-")[1]].append(prompt_id)

    rng = Random(seed)
    eval_ids: set[str] = set()
    for subject_ids in ids_by_subject.values():
        rng.shuffle(subject_ids)
        eval_count = min(
            max(round(len(subject_ids) * eval_fraction), 1), len(subject_ids) - 1
        )
        eval_ids.update(subject_ids[:eval_count])

    train = [
        example_by_id[prompt_id]
        for prompt_id in completed_ids
        if prompt_id not in eval_ids
    ]
    evaluation = [
        example_by_id[prompt_id]
        for prompt_id in completed_ids
        if prompt_id in eval_ids
    ]
    return train, evaluation


def write_datasets(
    train_output: Path = DEFAULT_TRAIN_OUTPUT,
    eval_output: Path = DEFAULT_EVAL_OUTPUT,
    prompts_path: Path = DEFAULT_PROMPTS,
    answers_path: Path = DEFAULT_ANSWERS,
) -> tuple[int, int]:
    """Write train and evaluation JSONL datasets."""

    train, evaluation = split_dataset(
        load_prompts(prompts_path), load_answer_records(answers_path)
    )
    for output, examples in ((train_output, train), (eval_output, evaluation)):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(example.model_dump_json() + "\n" for example in examples),
            encoding="utf-8",
        )
    return len(train), len(evaluation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    args = parser.parse_args()
    train_count, eval_count = write_datasets(
        args.train_output, args.eval_output, args.prompts, args.answers
    )
    print(f"Wrote {train_count} training examples to {args.train_output}")
    print(f"Wrote {eval_count} evaluation examples to {args.eval_output}")


if __name__ == "__main__":
    main()
