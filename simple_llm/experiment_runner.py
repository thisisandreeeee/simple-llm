"""Experiment configuration and stage orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from simple_llm.inference import GENERATION, generate_predictions, local_generator
from simple_llm.scoring.rule_scoring import score_predictions

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "data/evals.jsonl"
SEED = 42


def load_evals(path: Path = EVALS) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or any(set(row) != {"id", "prompt"} for row in rows):
        raise ValueError(f"Expected non-empty JSONL with only id and prompt: {path}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation IDs must be unique")
    return rows


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status)}


def summarize_inference(
    results: list[dict[str, Any]], wall_seconds: float | None = None
) -> dict[str, Any]:
    """Summarize run completion and generation diagnostics."""

    generated = [
        result
        for result in results
        if isinstance(result.get("generation_seconds"), (int, float))
    ]
    durations = [result["generation_seconds"] for result in generated]
    tokens = [result["output_tokens"] for result in generated]
    total_seconds = sum(durations)
    total_tokens = sum(tokens)
    summary = {
        "prompt_count": len(results),
        "successful_count": len(generated),
        "failed_count": sum("error" in result for result in results),
        "truncated_count": sum(item.get("truncated", False) for item in results),
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
        summary["wall_seconds"] = wall_seconds
    return summary


def run_experiment(
    *,
    experiment: str,
    model: str,
    condition: str,
    system_prompt: str | None = None,
    system_prompt_source: str | None = None,
    default_backend: str = "local",
    description: str | None = None,
    require_adapter_run: bool = False,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--limit", type=int, help="Run only the first N prompts.")
    parser.add_argument(
        "--backend", choices=("local", "modal"), default=default_backend
    )
    parser.add_argument("--gpu", default="L4", help="Modal GPU type (default: L4).")
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIR",
        help="Resume a run by appending its remaining predictions.",
    )
    if require_adapter_run:
        parser.add_argument(
            "--adapter-run",
            help="Completed SFT training run whose LoRA adapter to merge.",
        )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if require_adapter_run and args.backend != "modal":
        parser.error("--adapter-run is available only with --backend modal")

    run_dir = args.resume.resolve() if args.resume else None
    previous_config: dict[str, Any] | None = None
    if run_dir:
        try:
            previous_config = json.loads((run_dir / "config.json").read_text())
        except (json.JSONDecodeError, OSError) as exc:
            parser.error(f"invalid resume directory: {exc}")
        if args.limit is None:
            args.limit = previous_config.get("prompt_count")
        adapter = previous_config.get("adapter")
        if (
            require_adapter_run
            and args.adapter_run is None
            and isinstance(adapter, dict)
        ):
            args.adapter_run = adapter.get("run")
    if require_adapter_run and not args.adapter_run:
        parser.error("--adapter-run is required for a new run")

    evals = load_evals()
    if args.limit:
        evals = evals[: args.limit]
    print(f"Validated {len(evals)} prompt(s)")

    generation = dict(GENERATION)
    logits_processors = []
    if repetition_penalty is not None:
        logits_processors.append(
            {
                "type": "repetition_penalty",
                "penalty": repetition_penalty,
                "prompt_tokens": "ignored",
            }
        )
    if presence_penalty is not None:
        logits_processors.append(
            {"type": "presence_penalty", "penalty": presence_penalty}
        )
    if logits_processors:
        generation["logits_processor"] = (
            logits_processors[0] if len(logits_processors) == 1 else logits_processors
        )
    expected_config = {
        "model": model,
        "condition": condition,
        "system_prompt": system_prompt,
        "system_prompt_source": system_prompt_source,
        "eval_sha256": hashlib.sha256(EVALS.read_bytes()).hexdigest(),
        "prompt_count": len(evals),
        "seed": SEED,
        "enable_thinking": False,
        "generation": generation,
        "backend": args.backend,
    }
    if previous_config:
        mismatches = [
            key
            for key, value in expected_config.items()
            if previous_config.get(key) != value
        ]
        if require_adapter_run:
            adapter = previous_config.get("adapter")
            if not isinstance(adapter, dict) or adapter.get("run") != args.adapter_run:
                mismatches.append("adapter.run")
        if mismatches:
            parser.error("resume configuration mismatch: " + ", ".join(mismatches))

    random.seed(SEED)
    if args.backend == "modal":
        from simple_llm.inference.modal import modal_generator

        generator_context = modal_generator(
            model,
            args.gpu,
            SEED,
            adapter_run=args.adapter_run if require_adapter_run else None,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
        )
    else:
        generator_context = local_generator(
            model, SEED, presence_penalty, repetition_penalty
        )

    with generator_context as (generator, runtime):
        if run_dir is None:
            timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
            run_dir = ROOT / "runs" / f"{experiment}-{timestamp}"
            run_dir.mkdir(parents=True)
        config = previous_config or {
            **expected_config,
            "git": git_info(),
            **runtime,
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )

        started = time.perf_counter()
        results = generate_predictions(
            evals, generator, run_dir / "predictions.jsonl", system_prompt
        )
        wall_seconds = time.perf_counter() - started
        (run_dir / "summary.json").write_text(
            json.dumps(summarize_inference(results, wall_seconds), indent=2) + "\n",
            encoding="utf-8",
        )
        score_predictions(
            run_dir / "predictions.jsonl", run_dir / "rule_scores.json"
        )
        print(f"Wrote {run_dir}")
