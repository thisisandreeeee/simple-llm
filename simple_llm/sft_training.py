"""Fine-tune Qwen3.5-4B with bf16 LoRA on a Modal GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

REMOTE_TRAIN_DATASET = "/workspace/sft_train.jsonl"
REMOTE_EVAL_DATASET = "/workspace/sft_eval.jsonl"
HF_CACHE_DIR = "/cache/huggingface"
TRAINING_DIR = "/training"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PACKAGE_DIR = Path(__file__).resolve().parent
TRAIN_DATASET_PATH = DATA_DIR / "sft_train.jsonl"
EVAL_DATASET_PATH = DATA_DIR / "sft_eval.jsonl"
MODEL_NAME = "Qwen/Qwen3.5-4B"
SEED = 42
MAX_LENGTH = 2048


def load_dataset_rows(path: Path) -> list[dict[str, Any]]:
    """Load and validate a two-message SFT dataset."""

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    for index, row in enumerate(rows, 1):
        messages = (
            row.get("messages")
            if isinstance(row, dict) and set(row) == {"messages"}
            else None
        )
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or any(not isinstance(message, dict) for message in messages)
            or [message.get("role") for message in messages] != ["user", "assistant"]
            or any(
                not isinstance(message.get("content"), list)
                or len(message["content"]) != 1
                or not isinstance(message["content"][0], dict)
                or set(message["content"][0]) != {"type", "text"}
                or message["content"][0]["type"] != "text"
                or not isinstance(message["content"][0]["text"], str)
                or not message["content"][0]["text"].strip()
                for message in messages
            )
        ):
            raise ValueError(f"Invalid messages in dataset row {index}")
    return rows


def to_prompt_completion_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert two-message conversations to TRL prompt-completion rows."""

    return [
        {
            "prompt": [row["messages"][0]],
            "completion": [row["messages"][1]],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for row in rows
    ]


def run_training() -> None:
    """Run bf16 LoRA supervised fine-tuning on Modal."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="L4", help="Modal GPU type (default: L4).")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--detach", action="store_true", help="Keep the Modal app running on exit."
    )
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 or a positive integer")
    load_dataset_rows(TRAIN_DATASET_PATH)
    load_dataset_rows(EVAL_DATASET_PATH)
    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "qwen35-4b-sft-%Y%m%d-%H%M%S"
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        parser.error("--run-name may contain only letters, numbers, '.', '_', and '-'")

    hf_cache = modal.Volume.from_name(
        "simple-llm-huggingface-cache", create_if_missing=True
    )
    training_volume = modal.Volume.from_name(
        "simple-llm-training", create_if_missing=True
    )
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(
            "unsloth==2026.7.6",
            "torch==2.11.0",
            "transformers==5.5.0",
            "trl==0.24.0",
            "datasets==4.3.0",
        )
        .env({"HF_HOME": HF_CACHE_DIR, "HF_XET_HIGH_PERFORMANCE": "1"})
        .add_local_file(TRAIN_DATASET_PATH, REMOTE_TRAIN_DATASET, copy=True)
        .add_local_file(EVAL_DATASET_PATH, REMOTE_EVAL_DATASET, copy=True)
        .add_local_dir(PACKAGE_DIR, "/root/simple_llm", ignore=["**/__pycache__/**"])
    )
    app = modal.App("simple-llm-sft")

    @app.function(
        serialized=True,
        image=image,
        gpu="L4",
        volumes={HF_CACHE_DIR: hf_cache, TRAINING_DIR: training_volume},
        secrets=[modal.Secret.from_name("huggingface")],
        retries=1,
        timeout=24 * 60 * 60,
    )
    def train(run_name: str, epochs: float, max_steps: int) -> str:
        import importlib.metadata

        # Unsloth must patch Transformers and TRL before they are imported.
        from unsloth import FastLanguageModel

        import torch
        from datasets import Dataset
        from transformers.trainer_utils import get_last_checkpoint
        from trl import SFTConfig, SFTTrainer

        remote_train_path = Path(REMOTE_TRAIN_DATASET)
        remote_eval_path = Path(REMOTE_EVAL_DATASET)
        train_rows = load_dataset_rows(remote_train_path)
        eval_rows = load_dataset_rows(remote_eval_path)
        train_dataset = Dataset.from_list(to_prompt_completion_rows(train_rows))
        eval_dataset = Dataset.from_list(to_prompt_completion_rows(eval_rows))
        run_dir = Path(TRAINING_DIR) / run_name
        adapter_dir = run_dir / "adapter"
        checkpoint_dir = run_dir / "checkpoints"
        if adapter_dir.exists():
            raise FileExistsError(f"Completed run already exists: {run_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_LENGTH,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
            max_seq_length=MAX_LENGTH,
        )

        checkpoint_steps = min(25, max_steps) if max_steps > 0 else 25
        training_args = SFTConfig(
            output_dir=str(checkpoint_dir),
            max_length=MAX_LENGTH,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=epochs,
            max_steps=max_steps,
            learning_rate=1e-4,
            warmup_ratio=0.05,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            completion_only_loss=True,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=checkpoint_steps,
            save_steps=checkpoint_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=True,
            seed=SEED,
            report_to="none",
        )
        config = {
            "model": MODEL_NAME,
            "train_dataset_sha256": hashlib.sha256(
                remote_train_path.read_bytes()
            ).hexdigest(),
            "eval_dataset_sha256": hashlib.sha256(
                remote_eval_path.read_bytes()
            ).hexdigest(),
            "train_example_count": len(train_rows),
            "eval_example_count": len(eval_rows),
            "thinking": False,
            "lora_rank": 16,
            "lora_alpha": 16,
            "training": training_args.to_dict(),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "versions": {
                package: importlib.metadata.version(package)
                for package in ("unsloth", "torch", "transformers", "trl", "datasets")
            },
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

        sample = trainer.data_collator([trainer.train_dataset[0]])
        labels = sample["labels"]
        if not torch.any(labels == -100) or not torch.any(labels != -100):
            raise RuntimeError("Completion-only loss masking was not applied")

        checkpoint = get_last_checkpoint(str(checkpoint_dir))
        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_metrics("train", result.metrics)
        trainer.save_metrics("eval", trainer.evaluate())
        trainer.save_state()
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        training_volume.commit()
        return str(run_dir)

    print(f"Starting {run_name} on {args.gpu}")
    with modal.enable_output(), app.run(detach=args.detach):
        call = train.with_options(gpu=args.gpu).spawn(
            run_name, args.epochs, args.max_steps
        )
        print(f"Monitor training at {call.get_dashboard_url()}")
        print(f"Saved training artifacts to {call.get()}")


if __name__ == "__main__":
    run_training()
