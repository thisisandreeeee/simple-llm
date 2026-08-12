"""Run the Qwen3-0.6B base-model benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from simple_llm.scoring import RULE_SCORERS, score, validate_answer

MODEL = "Qwen/Qwen3-0.6B"
EVALS = ROOT / "data/evals.jsonl"
SEED = 42
MAX_NEW_TOKENS = 1024
GENERATION = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_new_tokens": MAX_NEW_TOKENS,
}


def load_evals() -> list[dict[str, str]]:
    rows = [json.loads(line) for line in EVALS.read_text(encoding="utf-8").splitlines()]
    if not rows or any(set(row) != {"id", "prompt"} for row in rows):
        raise ValueError(f"Expected non-empty JSONL with only id and prompt: {EVALS}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation IDs must be unique")
    return rows


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status)}


def generate(
    model: Any,
    tokenizer: Any,
    target: torch.device,
    prompt: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
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
    return {
        "response": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "input_tokens": input_tokens,
        "output_tokens": len(generated),
        "generation_seconds": time.perf_counter() - started,
        "truncated": len(generated) >= MAX_NEW_TOKENS,
    }


def summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        result for result in results if isinstance(result.get("scores"), dict)
    ]
    failed = [result for result in results if "error" in result]
    invalid = [
        result
        for result in results
        if result.get("validity", {}).get("valid") is False
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

    durations = [result["generation_seconds"] for result in successful]
    tokens = [result["output_tokens"] for result in successful]
    return {
        "prompt_count": len(results),
        "successful_count": len(successful),
        "failed_count": len(failed),
        "invalid_count": len(invalid),
        "invalid_reasons": dict(
            Counter(
                reason
                for result in invalid
                for reason in result["validity"]["reasons"]
            )
        ),
        "truncated_count": sum(result.get("truncated", False) for result in results),
        "score_means": means(all_scores),
        "domain_score_means": {
            domain: means(scores) for domain, scores in sorted(domain_scores.items())
        },
        "generation_seconds": {"total": sum(durations), "mean": mean(durations) if durations else None},
        "output_tokens": {"total": sum(tokens), "mean": mean(tokens) if tokens else None},
    }


def main(
    *,
    experiment: str = "01_qwen3_06b_base",
    condition: str = "base_raw",
    system_prompt: str | None = None,
    system_prompt_source: str | None = None,
    description: str = __doc__,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--limit", type=int, help="Run only the first N prompts.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    evals = load_evals()
    if args.limit:
        evals = evals[: args.limit]
    print(f"Validated {len(evals)} prompt(s)")

    random.seed(SEED)
    torch.manual_seed(SEED)
    target = device()
    print(f"Loading {MODEL} on {target} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto").to(target)
    model.eval()

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = ROOT / "runs" / f"{experiment}-{timestamp}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "model_revision": getattr(model.config, "_commit_hash", None),
                "condition": condition,
                "system_prompt": system_prompt,
                "system_prompt_source": system_prompt_source,
                "eval_sha256": hashlib.sha256(EVALS.read_bytes()).hexdigest(),
                "prompt_count": len(evals),
                "seed": SEED,
                "enable_thinking": False,
                "generation": GENERATION,
                "device": str(target),
                "git": git_info(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    results = []
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as output:
        for index, item in enumerate(evals, 1):
            result: dict[str, Any] = {
                "id": item["id"],
                "domain": item["id"].split("-", 1)[0],
                "prompt": item["prompt"],
            }
            try:
                generated = generate(
                    model, tokenizer, target, item["prompt"], system_prompt
                )
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
                    score(item["prompt"], generated["response"], RULE_SCORERS)
                    .model_dump(exclude_none=True)
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

    (run_dir / "summary.json").write_text(
        json.dumps(summary(results), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()
