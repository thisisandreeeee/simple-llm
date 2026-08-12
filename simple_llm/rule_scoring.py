"""Rule-based scoring step for completed inference predictions."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from simple_llm.scoring import RULE_SCORERS, score, validate_answer


def score_predictions(predictions_path: Path, scores_path: Path) -> dict[str, Any]:
    """Read predictions, score valid answers, and write one score artifact."""

    predictions = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    results = []
    for prediction in predictions:
        result: dict[str, Any] = {
            "id": prediction["id"],
            "domain": prediction["domain"],
        }
        if "error" in prediction:
            result["error"] = prediction["error"]
            results.append(result)
            continue

        try:
            validity = validate_answer(
                prediction["prompt"],
                prediction["response"],
                truncated=prediction["truncated"],
            )
            result["validity"] = {
                "valid": validity.valid,
                "reasons": list(validity.reasons),
            }
            result["scores"] = (
                score(
                    prediction["prompt"], prediction["response"], RULE_SCORERS
                ).model_dump(exclude_none=True)
                if validity.valid
                else None
            )
        except Exception as exc:  # Keep other scores if one prediction fails.
            result["error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)

    artifact = summarize_scores(results)
    scores_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return artifact


def summarize_scores(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return per-prediction results and aggregate rule-based scores."""

    scored = [item for item in results if isinstance(item.get("scores"), dict)]
    invalid = [
        item for item in results if item.get("validity", {}).get("valid") is False
    ]
    all_scores: dict[str, list[float]] = defaultdict(list)
    domain_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in scored:
        for name, value in item["scores"].items():
            all_scores[name].append(value)
            domain_scores[item["domain"]][name].append(value)

    def means(scores: dict[str, list[float]]) -> dict[str, float]:
        return {name: mean(values) for name, values in sorted(scores.items())}

    return {
        "prompt_count": len(results),
        "scored_count": len(scored),
        "failed_count": sum("error" in item for item in results),
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


__all__ = ["score_predictions", "summarize_scores"]
