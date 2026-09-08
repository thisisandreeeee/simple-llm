"""Prepare prompt-only data for GRPO training."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from simple_llm.sft.training import load_dataset_rows

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_TRAIN_DATASET_PATH = DATA_DIR / "sft_train.jsonl"


def make_prompt_id(row_number: int, prompt: str) -> str:
    """Create a deterministic, row-distinct identifier for a prompt."""

    normalized = " ".join(prompt.split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{row_number:06d}-{digest}"


def load_grpo_rows(
    path: Path = DEFAULT_TRAIN_DATASET_PATH,
) -> list[dict[str, Any]]:
    """Load, validate, and convert an SFT JSONL file for GRPO."""
    rows = load_dataset_rows(path)
    return [
        {
            "prompt": [row["messages"][0]],
            "prompt_id": make_prompt_id(
                row_number, row["messages"][0]["content"][0]["text"]
            ),
        }
        for row_number, row in enumerate(rows, 1)
    ]
