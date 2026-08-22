"""vLLM-independent helpers for the Modal vLLM inference backend."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .local import MAX_NEW_TOKENS


def format_prompt(tokenizer: Any, prompt: str, system_prompt: str | None) -> str:
    """Apply the shared chat template without importing vLLM."""
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_sampling_params(
    SamplingParams: Any,
    *,
    eos_token_ids: list[int],
    seed: int,
    presence_penalty: float | None,
    repetition_penalty: float | None,
) -> Any:
    """Construct vLLM sampling parameters from the shared generation settings."""
    kwargs: dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": MAX_NEW_TOKENS,
        "seed": seed,
        "stop_token_ids": eos_token_ids,
        "skip_special_tokens": True,
    }
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty
    return SamplingParams(**kwargs)


def prepare_scaled_adapter(adapter_path: Path, scale: float, destination: Path) -> Path:
    """Copy an adapter and scale its LoRA alpha without touching its weights."""
    if scale < 0:
        raise ValueError("adapter scale must be non-negative")
    config_path = adapter_path / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        alpha = config["lora_alpha"]
    except (json.JSONDecodeError, KeyError, OSError, TypeError) as exc:
        raise ValueError(f"Invalid adapter config: {config_path}") from exc
    if not isinstance(config, dict) or isinstance(alpha, bool) or not isinstance(
        alpha, (int, float)
    ):
        raise ValueError(f"Invalid adapter config: {config_path}")

    shutil.copytree(adapter_path, destination)
    config["lora_alpha"] = alpha * scale
    (destination / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def normalize_vllm_output(
    output: Any, started: float, max_tokens: int
) -> dict[str, Any]:
    """Translate a completed vLLM request result to the shared result shape."""
    if not output.outputs:
        raise ValueError(f"vLLM request {output.request_id} returned no output choices")
    choice = output.outputs[0]
    output_tokens = len(choice.token_ids)
    return {
        "response": choice.text.strip(),
        "input_tokens": len(output.prompt_token_ids),
        "output_tokens": output_tokens,
        "generation_seconds": time.perf_counter() - started,
        "truncated": output_tokens >= max_tokens,
    }
