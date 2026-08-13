import asyncio
import json

from simple_llm import judge_scoring


class FakeMetric:
    def __init__(self, score: float, reason: str) -> None:
        self.score = score
        self.reason = reason

    async def a_measure(self, test_case) -> None:
        assert test_case.input == "Explain a pump."
        assert test_case.actual_output == "A pump moves liquid."


def test_judge_uses_zero_temperature() -> None:
    metrics = judge_scoring.build_metrics("deepseek-chat")

    assert all(metric.model.temperature == 0.0 for metric in metrics.values())


def test_judge_scores_only_rule_valid_predictions(tmp_path, monkeypatch) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "id": "ONE-01",
                "domain": "ONE",
                "prompt": "Explain a pump.",
                "response": "A pump moves liquid.",
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "TWO-01",
                "domain": "TWO",
                "prompt": "Explain a valve.",
                "response": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rule_scores = tmp_path / "rule_scores.json"
    rule_scores.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "id": "ONE-01",
                        "domain": "ONE",
                        "validity": {"valid": True, "reasons": []},
                    },
                    {
                        "id": "TWO-01",
                        "domain": "TWO",
                        "validity": {"valid": False, "reasons": ["empty"]},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        judge_scoring,
        "build_metrics",
        lambda model: {"technical_adequacy": FakeMetric(0.75, "Mostly correct.")},
    )
    output = tmp_path / "judge_scores.json"

    result = judge_scoring.judge_predictions(predictions, rule_scores, output, "fake")

    assert result["scored_count"] == 1
    assert result["invalid_count"] == 1
    assert result["score_means"] == {"technical_adequacy": 0.75}
    assert result["results"][0]["reasons"] == {"technical_adequacy": "Mostly correct."}
    assert result["results"][1]["scores"] is None
    assert json.loads(output.read_text()) == result


def test_judge_limits_concurrency_and_preserves_order(tmp_path, monkeypatch) -> None:
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {
            "id": f"ONE-{index:02}",
            "domain": "ONE",
            "prompt": f"Prompt {index}",
            "response": f"Response {index}",
        }
        for index in range(3)
    ]
    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    rule_scores = tmp_path / "rule_scores.json"
    rule_scores.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "id": row["id"],
                        "validity": {"valid": True, "reasons": []},
                    }
                    for row in rows
                ]
            }
        ),
        encoding="utf-8",
    )

    active = 0
    max_active = 0

    class SlowMetric:
        score = 0.8
        reason = "Good."

        async def a_measure(self, test_case) -> None:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1

    monkeypatch.setattr(
        judge_scoring,
        "build_metrics",
        lambda model: {"first": SlowMetric(), "second": SlowMetric()},
    )

    result = judge_scoring.judge_predictions(
        predictions, rule_scores, tmp_path / "judge_scores.json", "fake", concurrency=2
    )

    assert max_active == 2
    assert [item["id"] for item in result["results"]] == [row["id"] for row in rows]
    assert all(
        item["scores"] == {"first": 0.8, "second": 0.8}
        for item in result["results"]
    )


def test_judge_preserves_successful_dimensions(tmp_path, monkeypatch) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "id": "ONE-01",
                "domain": "ONE",
                "prompt": "Explain a pump.",
                "response": "A pump moves liquid.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rule_scores = tmp_path / "rule_scores.json"
    rule_scores.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "id": "ONE-01",
                        "validity": {"valid": True, "reasons": []},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FailedMetric:
        score = None
        reason = None

        async def a_measure(self, test_case) -> None:
            raise RuntimeError("rate limited")

    monkeypatch.setattr(
        judge_scoring,
        "build_metrics",
        lambda model: {
            "successful": FakeMetric(0.75, "Good."),
            "failed": FailedMetric(),
        },
    )

    result = judge_scoring.judge_predictions(
        predictions, rule_scores, tmp_path / "judge_scores.json", "fake"
    )

    assert result["results"][0]["scores"] == {"successful": 0.75}
    assert result["results"][0]["errors"] == {
        "failed": "RuntimeError: rate limited"
    }


def test_judge_retries_invalid_json_and_reports_partial_scores(
    tmp_path, monkeypatch
) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "id": "ONE-01",
                "domain": "ONE",
                "prompt": "Explain a pump.",
                "response": "A pump moves liquid.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rule_scores = tmp_path / "rule_scores.json"
    rule_scores.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "id": "ONE-01",
                        "validity": {"valid": True, "reasons": []},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class RetryMetric(FakeMetric):
        attempts = 0

        async def a_measure(self, test_case) -> None:
            self.attempts += 1
            if self.attempts < 3:
                raise RuntimeError(
                    "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
                )

    retry_metric = RetryMetric(0.75, "Good after retry.")
    monkeypatch.setattr(
        judge_scoring,
        "build_metrics",
        lambda model: {
            "technical_adequacy": retry_metric,
            "task_fulfillment": FakeMetric(1.0, "Complete."),
        },
    )

    result = judge_scoring.judge_predictions(
        predictions,
        rule_scores,
        tmp_path / "judge_scores.json",
        "fake",
        retry_limit=2,
    )

    assert retry_metric.attempts == 3
    assert result["scored_count"] == 1
    assert result["fully_scored_count"] == 0
    assert result["partially_scored_count"] == 1
    assert result["dimension_scored_counts"] == {
        "clarity_and_coherence": 0,
        "semantic_simplicity": 0,
        "task_fulfillment": 1,
        "technical_adequacy": 1,
    }
    assert result["failed_dimension_count"] == 0
