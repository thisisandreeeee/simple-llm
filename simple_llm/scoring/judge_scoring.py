"""DeepEval G-Eval scoring for completed inference predictions."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

DEFAULT_CONCURRENCY = 50
DEFAULT_RETRY_LIMIT = 2
JUDGE_TEMPERATURE = 0.0
_INVALID_JSON_ERROR = "Evaluation LLM outputted an invalid JSON"

_DIMENSIONS = {
    "technical_adequacy": {
        "criteria": (
            "Judge whether the answer is factually correct, technically adequate, "
            "and preserves necessary caveats or safety conditions."
        ),
        "steps": (
            "Identify the main technical claims made in the answer.",
            "Evaluate each claim independently using your technical knowledge; do not assume a claim is correct merely because it is detailed, confident, or well written.",
            "Check for contradictions within the answer and for important counterexamples, exceptions, outdated claims, or missing safety conditions.",
            "Classify each problem as minor or material: a material problem changes the user's understanding, recommended action, or the answer to the main question.",
            "Determine completeness separately from correctness; covering every requested topic does not compensate for factual errors.",
            "Assign a score from 1 to 5 using the rubric. In the reason, name the most important errors or omissions that affected the score.",
        ),
        "rubric": (
            (1, "The answer is incorrect, unsafe, or seriously misleading."),
            (2, "The answer has major factual errors or omits most necessary details."),
            (3, "The answer is partly correct but has material omissions or errors."),
            (4, "The answer is mostly correct and complete, with minor issues."),
            (5, "The answer is correct, complete, and appropriately qualified."),
        ),
    },
    "task_fulfillment": {
        "criteria": (
            "Judge whether the answer directly fulfills the user's requested task, "
            "scope, audience, and requested format."
        ),
        "steps": (
            "List every explicit task, requested topic, constraint, audience requirement, and format requirement in the prompt.",
            "Check whether the answer substantively addresses each requested item; merely mentioning an item or providing a matching heading does not count as fulfilling it.",
            "Check whether instructions, examples, or explanations are usable at the level requested by the user rather than generic placeholders that require missing information.",
            "Penalize missing requested items, irrelevant meta-commentary, invented requirements, unnecessary detours, and violations of explicit scope or format.",
            "Judge fulfillment separately from technical correctness, but do not count content as fulfilling a requested item when it discusses a different concept under the requested label.",
            "Assign a score from 1 to 5 using the rubric. In the reason, identify the most important fulfilled and unfulfilled requirements.",
        ),
        "rubric": (
            (1, "The answer does not address the requested task."),
            (2, "The answer addresses only a small part of the requested task."),
            (3, "The answer addresses the task but misses important requested parts."),
            (4, "The answer fulfills the task with minor omissions or detours."),
            (5, "The answer fully and directly fulfills the requested task."),
        ),
    },
    "clarity_and_coherence": {
        "criteria": (
            "Judge whether the answer has clear meaning, coherent organization, "
            "and understandable references between ideas."
        ),
        "steps": (
            "Check whether each claim has an unambiguous meaning.",
            "Check whether the ideas follow a logical order.",
            "Penalize vague references, contradictions, and confusing organization.",
            "Assign a score from 1 to 5.",
        ),
        "rubric": (
            (1, "The answer is confusing or incoherent."),
            (2, "The answer is difficult to follow and has frequent ambiguity."),
            (
                3,
                "The answer is understandable but has noticeable clarity or organization problems.",
            ),
            (4, "The answer is clear and coherent with minor weaknesses."),
            (5, "The answer is consistently clear, coherent, and unambiguous."),
        ),
    },
    "semantic_simplicity": {
        "criteria": (
            "Judge whether the answer expresses ideas directly and simply without "
            "removing necessary meaning or technical precision."
        ),
        "steps": (
            "Evaluate simplicity at the level of the whole answer, not only individual sentences.",
            "Check whether the answer uses direct wording, clear agents or actions, and necessary technical terms without avoidable abstraction.",
            "Check whether the amount of detail and structure is proportionate to the user's request.",
            "Penalize repetition, fragmented or choppy presentation, unnecessary headings, excessive itemization, meta-commentary, redundant summaries, and repeated sentence patterns.",
            "Do not award the highest score merely because sentences are short; the complete answer must also be concise, cohesive, and free of unnecessary content.",
            "Do not penalize technical terminology, examples, or structure when they materially improve precision or usability.",
            "Assign a score from 1 to 5 using the rubric. In the reason, identify any answer-level excess complexity or redundancy that affected the score.",
        ),
        "rubric": (
            (1, "The answer is needlessly complex, indirect, or verbose."),
            (
                2,
                "The answer often uses complex or indirect wording that harms understanding.",
            ),
            (3, "The answer has a mix of simple and unnecessarily complex expression."),
            (4, "The answer is mostly direct and simple with minor excess complexity."),
            (
                5,
                "The answer is direct and simple while preserving all necessary meaning.",
            ),
        ),
    },
}


def _rubrics(values: tuple[tuple[int, str], ...]) -> list[Any]:
    from deepeval.metrics.g_eval import Rubric

    return [
        Rubric(score_range=(score, score), expected_outcome=outcome)
        for score, outcome in values
    ]


def build_metrics(model: str) -> dict[str, Any]:
    """Build the fixed G-Eval metrics used by this project."""

    from deepeval.metrics import GEval
    from deepeval.models import DeepSeekModel
    from deepeval.test_case import SingleTurnParams

    judge_model = DeepSeekModel(model=model, temperature=JUDGE_TEMPERATURE)
    return {
        name: GEval(
            name=name,
            criteria=spec["criteria"],
            evaluation_steps=list(spec["steps"]),
            rubric=_rubrics(spec["rubric"]),
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            model=judge_model,
            threshold=None,
            async_mode=True,
        )
        for name, spec in _DIMENSIONS.items()
    }


def _provenance(model: str) -> dict[str, Any]:
    rubric = json.dumps(_DIMENSIONS, sort_keys=True, separators=(",", ":"))
    return {
        "framework": "deepeval",
        "framework_version": importlib.metadata.version("deepeval"),
        "model": model,
        "temperature": JUDGE_TEMPERATURE,
        "rubric_sha256": hashlib.sha256(rubric.encode()).hexdigest(),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def judge_predictions(
    predictions_path: Path,
    rule_scores_path: Path,
    scores_path: Path,
    model: str,
    concurrency: int = DEFAULT_CONCURRENCY,
    retry_limit: int = DEFAULT_RETRY_LIMIT,
) -> dict[str, Any]:
    """Judge rule-valid predictions and incrementally write judge scores."""

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if retry_limit < 0:
        raise ValueError("retry_limit must be at least 0")

    predictions = _load_jsonl(predictions_path)
    rule_artifact = json.loads(rule_scores_path.read_text(encoding="utf-8"))
    rule_results = {item["id"]: item for item in rule_artifact["results"]}
    return asyncio.run(
        _judge_predictions(
            predictions, rule_results, scores_path, model, concurrency, retry_limit
        )
    )


async def _judge_predictions(
    predictions: list[dict[str, Any]],
    rule_results: dict[str, dict[str, Any]],
    scores_path: Path,
    model: str,
    concurrency: int,
    retry_limit: int,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)

    async def judge_one(
        index: int, prediction: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        item = rule_results.get(prediction["id"])
        if item is None:
            result = {
                "id": prediction["id"],
                "domain": prediction["domain"],
                "error": "Missing rule-score result",
            }
        elif "error" in prediction or "error" in item:
            result = {
                "id": prediction["id"],
                "domain": prediction["domain"],
                "error": prediction.get("error", item.get("error")),
            }
        else:
            result = {
                "id": prediction["id"],
                "domain": prediction["domain"],
                "validity": item.get("validity"),
            }
            if not item.get("validity", {}).get("valid", False):
                result["scores"] = None
            else:
                # DeepEval metrics store score and reason on the instance.
                metrics = build_metrics(model)
                result.update(
                    await _measure_metrics(metrics, prediction, semaphore, retry_limit)
                )
        return index, result

    results: list[dict[str, Any] | None] = [None] * len(predictions)
    tasks = [
        asyncio.create_task(judge_one(index, prediction))
        for index, prediction in enumerate(predictions)
    ]
    for completed in asyncio.as_completed(tasks):
        index, result = await completed
        results[index] = result
        checkpoint = [item for item in results if item is not None]
        scores_path.write_text(
            json.dumps(_summarize(checkpoint, model), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

    return _summarize([item for item in results if item is not None], model)


async def _measure_metrics(
    metrics: dict[str, Any],
    prediction: dict[str, Any],
    semaphore: asyncio.Semaphore,
    retry_limit: int,
) -> dict[str, Any]:
    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(
        input=prediction["prompt"], actual_output=prediction["response"]
    )

    async def measure_one(
        name: str, metric: Any
    ) -> tuple[str, float | None, str | None, str | None]:
        for attempt in range(retry_limit + 1):
            try:
                async with semaphore:
                    await metric.a_measure(test_case)
                    if metric.score is None:
                        raise ValueError("DeepEval returned no score")
                    return (
                        name,
                        float(metric.score),
                        str(metric.reason) if metric.reason else None,
                        None,
                    )
            except (
                Exception
            ) as exc:  # Preserve successful dimensions if one call fails.
                if attempt == retry_limit or _INVALID_JSON_ERROR not in str(exc):
                    return name, None, None, f"{type(exc).__name__}: {exc}"
        raise AssertionError("unreachable")

    measured = await asyncio.gather(
        *(measure_one(name, metric) for name, metric in metrics.items())
    )
    scores = {name: score for name, score, _, _ in measured if score is not None}
    reasons = {name: reason for name, _, reason, _ in measured if reason is not None}
    errors = {name: error for name, _, _, error in measured if error is not None}
    result: dict[str, Any] = {"scores": scores, "reasons": reasons}
    if errors:
        result["errors"] = errors
    return result


def _summarize(results: list[dict[str, Any]], model: str) -> dict[str, Any]:
    scored = [
        item
        for item in results
        if isinstance(item.get("scores"), dict) and item["scores"]
    ]
    invalid = [
        item for item in results if item.get("validity", {}).get("valid") is False
    ]
    fully_scored = [
        item for item in scored if all(name in item["scores"] for name in _DIMENSIONS)
    ]
    all_scores: dict[str, list[float]] = defaultdict(list)
    domain_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in scored:
        for name, value in item["scores"].items():
            all_scores[name].append(value)
            domain_scores[item["domain"]][name].append(value)

    def means(values: dict[str, list[float]]) -> dict[str, float]:
        return {name: mean(scores) for name, scores in sorted(values.items())}

    return {
        "judge": _provenance(model),
        "prompt_count": len(results),
        "scored_count": len(scored),
        "fully_scored_count": len(fully_scored),
        "partially_scored_count": len(scored) - len(fully_scored),
        "dimension_scored_counts": {
            name: sum(name in item["scores"] for item in scored)
            for name in sorted(_DIMENSIONS)
        },
        "failed_dimension_count": sum(len(item.get("errors", {})) for item in results),
        "failed_count": sum("error" in item or "errors" in item for item in results),
        "invalid_count": len(invalid),
        "invalid_reasons": dict(
            Counter(
                reason for item in invalid for reason in item["validity"]["reasons"]
            )
        ),
        "score_means": means(all_scores),
        "domain_score_means": {
            domain: means(scores) for domain, scores in sorted(domain_scores.items())
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("rule_scores", type=Path)
    parser.add_argument(
        "--model", required=True, help="DeepEval judge model/provider name."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: judge_scores.json beside predictions).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Maximum concurrent judge requests (default: {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--retry-limit",
        type=int,
        default=DEFAULT_RETRY_LIMIT,
        help=(
            "Retries per dimension after invalid judge JSON "
            f"(default: {DEFAULT_RETRY_LIMIT})."
        ),
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.retry_limit < 0:
        parser.error("--retry-limit must be at least 0")
    output = args.output or args.predictions.with_name("judge_scores.json")
    judge_predictions(
        args.predictions,
        args.rule_scores,
        output,
        args.model,
        args.concurrency,
        args.retry_limit,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
