"""Shared local and remote inference experiment runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "data/evals.jsonl"
SEED = 42
MAX_NEW_TOKENS = 2048
GENERATION = {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS}

Generator = Callable[[str, str | None], dict[str, Any]]


def load_evals(path: Path = EVALS) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or any(set(row) != {"id", "prompt"} for row in rows):
        raise ValueError(f"Expected non-empty JSONL with only id and prompt: {path}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation IDs must be unique")
    return rows


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
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            **GENERATION,
            pad_token_id=tokenizer.eos_token_id,
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


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status)}


def evaluate(
    evals: list[dict[str, str]],
    generator: Generator,
    predictions_path: Path,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Generate, score, and incrementally persist each evaluation response."""
    from simple_llm.scoring import RULE_SCORERS, score, validate_answer

    results = []
    with predictions_path.open("w", encoding="utf-8") as output:
        for index, item in enumerate(evals, 1):
            result: dict[str, Any] = {
                "id": item["id"],
                "domain": item["id"].split("-", 1)[0],
                "prompt": item["prompt"],
            }
            try:
                generated = generator(item["prompt"], system_prompt)
                result.update(generated)
                validity = validate_answer(
                    item["prompt"],
                    generated["response"],
                    truncated=generated["truncated"],
                )
                result["validity"] = {
                    "valid": validity.valid,
                    "reasons": list(validity.reasons),
                }
                result["scores"] = (
                    score(
                        item["prompt"], generated["response"], RULE_SCORERS
                    ).model_dump(exclude_none=True)
                    if validity.valid
                    else None
                )
                print(f"[{index}/{len(evals)}] {item['id']}")
            except Exception as exc:  # Keep completed results if one prompt fails.
                result["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{index}/{len(evals)}] {item['id']}: {result['error']}")
            results.append(result)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
    return results


def summary(
    results: list[dict[str, Any]], wall_seconds: float | None = None
) -> dict[str, Any]:
    successful = [
        result for result in results if isinstance(result.get("scores"), dict)
    ]
    generated = [
        result
        for result in results
        if isinstance(result.get("generation_seconds"), (int, float))
    ]
    failed = [result for result in results if "error" in result]
    invalid = [
        result for result in results if result.get("validity", {}).get("valid") is False
    ]
    all_scores: dict[str, list[float]] = defaultdict(list)
    domain_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for result in successful:
        domain = result["id"].split("-", 1)[0]
        for name, value in result["scores"].items():
            all_scores[name].append(value)
            domain_scores[domain][name].append(value)

    def means(scores: dict[str, list[float]]) -> dict[str, float]:
        return {name: mean(values) for name, values in sorted(scores.items())}

    durations = [result["generation_seconds"] for result in generated]
    tokens = [result["output_tokens"] for result in generated]
    total_seconds = sum(durations)
    total_tokens = sum(tokens)
    result = {
        "prompt_count": len(results),
        "successful_count": len(successful),
        "failed_count": len(failed),
        "invalid_count": len(invalid),
        "invalid_reasons": dict(
            Counter(
                reason for item in invalid for reason in item["validity"]["reasons"]
            )
        ),
        "truncated_count": sum(item.get("truncated", False) for item in results),
        "score_means": means(all_scores),
        "domain_score_means": {
            domain: means(scores) for domain, scores in sorted(domain_scores.items())
        },
        "generation_seconds": {
            "total": total_seconds,
            "mean": mean(durations) if durations else None,
        },
        "output_tokens": {
            "total": total_tokens,
            "mean": mean(tokens) if tokens else None,
        },
        "output_tokens_per_second": (
            total_tokens / total_seconds if total_seconds else None
        ),
    }
    if wall_seconds is not None:
        result["wall_seconds"] = wall_seconds
    return result


def run_experiment(
    *,
    experiment: str,
    model: str,
    condition: str,
    system_prompt: str | None = None,
    system_prompt_source: str | None = None,
    default_backend: str = "local",
    description: str | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--limit", type=int, help="Run only the first N prompts.")
    parser.add_argument(
        "--backend", choices=("local", "modal"), default=default_backend
    )
    parser.add_argument("--gpu", default="L4", help="Modal GPU type (default: L4).")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    evals = load_evals()
    if args.limit:
        evals = evals[: args.limit]
    print(f"Validated {len(evals)} prompt(s)")

    random.seed(SEED)
    if args.backend == "modal":
        from simple_llm.modal_inference import modal_generator

        generator_context = modal_generator(model, args.gpu, SEED)
    else:
        generator_context = local_generator(model, SEED)

    with generator_context as (generator, runtime):
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
        run_dir = ROOT / "runs" / f"{experiment}-{timestamp}"
        run_dir.mkdir(parents=True)
        config = {
            "model": model,
            "condition": condition,
            "system_prompt": system_prompt,
            "system_prompt_source": system_prompt_source,
            "eval_sha256": hashlib.sha256(EVALS.read_bytes()).hexdigest(),
            "prompt_count": len(evals),
            "seed": SEED,
            "enable_thinking": False,
            "generation": GENERATION,
            "git": git_info(),
            **runtime,
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )

        started = time.perf_counter()
        results = evaluate(
            evals, generator, run_dir / "predictions.jsonl", system_prompt
        )
        wall_seconds = time.perf_counter() - started
        (run_dir / "run_summary.json").write_text(
            json.dumps(summary(results, wall_seconds), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {run_dir}")
