"""Local inference and prediction generation."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

MAX_NEW_TOKENS = 2048
GENERATION = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_new_tokens": MAX_NEW_TOKENS,
}

Generator = Callable[[str, str | None], dict[str, Any]]


def generation_eos_token_ids(model: Any, tokenizer: Any) -> list[int]:
    """Stop on both the model EOS and the chat template's end-of-turn token."""
    configured = model.generation_config.eos_token_id
    token_ids = [configured] if isinstance(configured, int) else list(configured or [])
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in token_ids:
        token_ids.append(tokenizer.eos_token_id)
    return token_ids


def device() -> Any:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate(
    model: Any,
    tokenizer: Any,
    target: Any,
    prompt: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Generate one response using an already-loaded Transformers model."""
    import torch

    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {name: value.to(target) for name, value in inputs.items()}
    input_tokens = inputs["input_ids"].shape[-1]

    started = time.perf_counter()
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            **GENERATION,
            eos_token_id=generation_eos_token_ids(model, tokenizer),
            pad_token_id=pad_token_id,
        )
    generated = output[0, input_tokens:]
    output_tokens = len(generated)
    return {
        "response": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "generation_seconds": time.perf_counter() - started,
        "truncated": output_tokens >= MAX_NEW_TOKENS,
    }


@contextmanager
def local_generator(
    model_name: str, seed: int
) -> Iterator[tuple[Generator, dict[str, Any]]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    target = device()
    print(f"Loading {model_name} on {target} ...")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(
        target
    )
    model.eval()
    metadata = {
        "backend": "local",
        "device": str(target),
        "model_revision": getattr(model.config, "_commit_hash", None),
        "torch_dtype": str(model.dtype),
        "model_load_seconds": time.perf_counter() - started,
    }
    yield lambda prompt, system_prompt: generate(
        model, tokenizer, target, prompt, system_prompt
    ), metadata


def generate_predictions(
    evals: list[dict[str, str]],
    generator: Generator,
    predictions_path: Path,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Generate and incrementally persist each evaluation response."""

    results: list[dict[str, Any]] = []
    if predictions_path.exists():
        try:
            results = [
                json.loads(line)
                for line in predictions_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Invalid predictions file: {predictions_path}") from exc
        if len(results) > len(evals) or any(
            result.get("id") != item["id"] or result.get("prompt") != item["prompt"]
            for result, item in zip(results, evals)
        ):
            raise ValueError(
                f"Existing predictions do not match the evaluation set: {predictions_path}"
            )
        print(f"Resuming after {len(results)}/{len(evals)} prompt(s)")

    with predictions_path.open("a" if results else "w", encoding="utf-8") as output:
        for index, item in enumerate(evals[len(results) :], len(results) + 1):
            result: dict[str, Any] = {
                "id": item["id"],
                "domain": item["id"].split("-", 1)[0],
                "prompt": item["prompt"],
            }
            try:
                generated = generator(item["prompt"], system_prompt)
                result.update(generated)
                print(f"[{index}/{len(evals)}] {item['id']}")
            except Exception as exc:  # Keep completed results if one prompt fails.
                result["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{index}/{len(evals)}] {item['id']}: {result['error']}")
            results.append(result)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
    return results
