"""Prepare prompt-only data and run a GRPO reward smoke test."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

from simple_llm.grpo.rewards import (
    multiplicative_rewards,
    relative_rewards,
    score_completions,
)
from simple_llm.modal import (
    HF_CACHE_DIR,
    TRAINING_DIR,
    build_training_image,
    get_training_volumes,
    register_tensorboard_app,
    validate_run_name,
)
from simple_llm.sft.training import load_dataset_rows

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_TRAIN_DATASET_PATH = DATA_DIR / "sft_train.jsonl"
DEFAULT_EVAL_DATASET_PATH = DATA_DIR / "sft_eval.jsonl"
APPROVED_WORDS_PATH = DATA_DIR / "ste_approved_words.txt"
REMOTE_TRAIN_DATASET = "/workspace/sft_train.jsonl"
REMOTE_EVAL_DATASET = "/workspace/sft_eval.jsonl"
MODEL_NAME = "Qwen/Qwen3.5-4B"
SEED = 42
MAX_LENGTH = 2048
DEFAULT_GPU = "L4"
DEFAULT_NUM_GENERATIONS = 4
DEFAULT_TEMPERATURE = 0.9
DEFAULT_REWARD_FUNCTION = "multiplicative"
REWARD_FUNCTIONS = {
    "multiplicative": multiplicative_rewards,
    "relative": relative_rewards,
}


def make_reward_funcs(reward_function: str = DEFAULT_REWARD_FUNCTION):
    """Build one training reward plus zero-weight diagnostics for TRL logging."""

    try:
        combine_scores = REWARD_FUNCTIONS[reward_function]
    except KeyError as error:
        raise ValueError(f"Unknown reward function: {reward_function}") from error

    cached_key = None
    cached_scores = None

    def scores_for(prompts, completions):
        nonlocal cached_key, cached_scores
        key = (tuple(prompts), tuple(completions))
        if key != cached_key:
            cached_key = key
            cached_scores = score_completions(prompts, completions)
        return cached_scores

    def combined(prompts, completions, **kwargs):
        return combine_scores(prompts, scores_for(prompts, completions))

    def correctness(prompts, completions, **kwargs):
        return [score["correctness"] for score in scores_for(prompts, completions)]

    def simplicity(prompts, completions, **kwargs):
        return [score["simplicity"] for score in scores_for(prompts, completions)]

    def asd_ste100(prompts, completions, **kwargs):
        return [score["asd_ste100"] for score in scores_for(prompts, completions)]

    return [combined, correctness, simplicity, asd_ste100]


def make_prompt_id(row_number: int, prompt: str) -> str:
    """Create a deterministic, row-distinct identifier for a prompt."""

    normalized = " ".join(prompt.split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{row_number:06d}-{digest}"


def to_prompt_rows(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    """Convert validated SFT rows to rendered prompt-only GRPO rows."""

    return [
        {
            "prompt": tokenizer.apply_chat_template(
                [row["messages"][0]],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
            "prompt_id": make_prompt_id(
                row_number, row["messages"][0]["content"][0]["text"]
            ),
        }
        for row_number, row in enumerate(rows, 1)
    ]


def load_grpo_rows(
    path: Path = DEFAULT_TRAIN_DATASET_PATH, *, tokenizer: Any
) -> list[dict[str, Any]]:
    """Load, validate, and convert an SFT JSONL file for GRPO."""
    return to_prompt_rows(load_dataset_rows(path), tokenizer)


def validate_trainable_lora_parameters(model: Any) -> None:
    """Require the loaded SFT adapter, and only that adapter, to be trainable."""
    parameters = list(model.named_parameters())
    trainable = [
        (name, parameter) for name, parameter in parameters if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("The loaded SFT adapter has no trainable parameters")
    unexpected = [name for name, _ in trainable if "lora_" not in name]
    if unexpected:
        raise RuntimeError(
            "Unexpected non-LoRA parameters are trainable: " + ", ".join(unexpected[:5])
        )
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    total_count = sum(parameter.numel() for _, parameter in parameters)
    print(
        f"Trainable LoRA parameters: {trainable_count:,} / {total_count:,} "
        f"({100 * trainable_count / total_count:.4f}%)"
    )


def run_training() -> None:
    """Run the GRPO reward-variance smoke test on Modal."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--adapter-run",
        required=True,
        help="Completed SFT run used to initialize GRPO.",
    )
    parser.add_argument("--num-generations", type=int, default=DEFAULT_NUM_GENERATIONS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--reward-function",
        choices=REWARD_FUNCTIONS,
        default=DEFAULT_REWARD_FUNCTION,
    )
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--detach", action="store_true", help="Keep the Modal app running on exit."
    )
    args = parser.parse_args()
    if args.num_generations < 2:
        parser.error("--num-generations must be at least 2")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 or a positive integer")

    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "qwen35-4b-grpo-%Y%m%d-%H%M%S"
    )
    try:
        validate_run_name(run_name)
        validate_run_name(args.adapter_run)
    except ValueError as error:
        parser.error(str(error))

    volumes = get_training_volumes()
    hf_cache = volumes[HF_CACHE_DIR]
    training_volume = volumes[TRAINING_DIR]
    image = build_training_image(
        (
            "unsloth==2026.9.3",
            "torch==2.11.0",
            "transformers==5.5.0",
            "trl==0.24.0",
            "datasets==4.3.0",
            "openai==3.0.0",
            "tensorboard",
        ),
        (
            (DEFAULT_TRAIN_DATASET_PATH, REMOTE_TRAIN_DATASET),
            (DEFAULT_EVAL_DATASET_PATH, REMOTE_EVAL_DATASET),
            (APPROVED_WORDS_PATH, "/root/data/ste_approved_words.txt"),
        ),
    )
    app = modal.App("simple-llm-grpo")
    register_tensorboard_app(app, image, training_volume)

    @app.function(
        serialized=True,
        image=image,
        gpu="L4",
        volumes={HF_CACHE_DIR: hf_cache, TRAINING_DIR: training_volume},
        secrets=[
            modal.Secret.from_name("huggingface"),
            modal.Secret.from_name("deepseek"),
        ],
        timeout=24 * 60 * 60,
    )
    def train(
        run_name: str,
        adapter_run: str,
        num_generations: int,
        temperature: float,
        reward_function: str,
        max_steps: int,
    ) -> str:
        import sys

        from huggingface_hub import snapshot_download

        kernel_snapshot = snapshot_download(
            "kernels-community/causal-conv1d",
            revision="f2651e776f66069cdcf842840db637583def1223",
            allow_patterns="build/torch211-cxx11-cu130-x86_64-linux/*",
        )
        sys.path.insert(0, f"{kernel_snapshot}/build/torch211-cxx11-cu130-x86_64-linux")

        # Unsloth must patch Transformers and TRL before they are imported.
        from unsloth import FastLanguageModel, PatchFastRL

        PatchFastRL("GRPO", FastLanguageModel)
        from datasets import Dataset
        from transformers.trainer_utils import get_last_checkpoint
        from trl import GRPOConfig, GRPOTrainer

        train_source_rows = load_dataset_rows(Path(REMOTE_TRAIN_DATASET))
        eval_source_rows = load_dataset_rows(Path(REMOTE_EVAL_DATASET))
        if not train_source_rows:
            raise ValueError("No GRPO prompts are available for the smoke test")
        run_dir = Path(TRAINING_DIR) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = run_dir / "adapter"
        checkpoint_dir = run_dir / "checkpoints"
        if adapter_dir.exists():
            raise FileExistsError(f"Completed run already exists: {run_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        sft_adapter_dir = Path(TRAINING_DIR) / adapter_run / "adapter"
        if not (sft_adapter_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(f"SFT adapter does not exist: {sft_adapter_dir}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(sft_adapter_dir),
            max_seq_length=MAX_LENGTH,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
        )
        from transformers.models.qwen3_5 import modeling_qwen3_5

        if not modeling_qwen3_5.is_fast_path_available:
            raise RuntimeError("Qwen3.5 optimized kernel path is unavailable")
        print("Qwen3.5 optimized kernel path is active")
        validate_trainable_lora_parameters(model)
        train_rows = to_prompt_rows(train_source_rows, tokenizer)
        eval_rows = to_prompt_rows(eval_source_rows, tokenizer)
        train_dataset = Dataset.from_list(train_rows)
        eval_dataset = Dataset.from_list(eval_rows)
        reward_funcs = make_reward_funcs(reward_function)

        training_args = GRPOConfig(
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            logging_steps=1,
            log_completions=False,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=num_generations,
            gradient_accumulation_steps=4,
            num_generations=num_generations,  # Decrease if out of memory
            max_prompt_length=512,
            max_completion_length=1024,
            max_steps=max_steps,
            num_train_epochs=1,
            mask_truncated_completions=True,
            eval_strategy="epoch",
            bf16=True,
            seed=SEED,
            temperature=temperature,
            reward_weights=[1.0] + [0.0] * (len(reward_funcs) - 1),
            report_to="tensorboard",
            output_dir=str(checkpoint_dir),
            remove_unused_columns=False,
            save_steps=5,
            save_total_limit=2,
        )
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )
        checkpoint = get_last_checkpoint(str(checkpoint_dir))
        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        training_volume.commit()
        return str(run_dir)

    print(f"Starting {run_name} on {args.gpu}")
    with modal.enable_output(), app.run(detach=args.detach):
        call = train.with_options(gpu=args.gpu).spawn(
            run_name,
            args.adapter_run,
            args.num_generations,
            args.temperature,
            args.reward_function,
            args.max_steps,
        )
        print(f"Monitor training at {call.get_dashboard_url()}")
        print(f"Saved reward report to {call.get()}")


if __name__ == "__main__":
    run_training()
