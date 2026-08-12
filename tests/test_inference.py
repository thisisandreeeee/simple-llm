import json

from simple_llm.experiment import summarize_inference
from simple_llm.inference import generate_predictions
from simple_llm.rule_scoring import score_predictions


def test_inference_and_rule_scoring_are_separate_steps(tmp_path) -> None:
    evals = [
        {"id": "ONE-01", "prompt": "Explain a pump."},
        {"id": "TWO-01", "prompt": "fail"},
    ]

    def generator(prompt: str, system_prompt: str | None):
        if prompt == "fail":
            raise RuntimeError("expected")
        return {
            "response": "A pump moves liquid.",
            "input_tokens": 3,
            "output_tokens": 5,
            "generation_seconds": 0.5,
            "truncated": False,
        }

    predictions_path = tmp_path / "predictions.jsonl"
    results = generate_predictions(evals, generator, predictions_path)

    saved = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    assert len(saved) == 2
    assert saved[0]["response"] == "A pump moves liquid."
    assert "validity" not in saved[0]
    assert "scores" not in saved[0]
    assert saved[1]["error"] == "RuntimeError: expected"
    diagnostics = summarize_inference(results)
    assert diagnostics["output_tokens_per_second"] == 10
    assert diagnostics["successful_count"] == 1

    scores_path = tmp_path / "rule_scores.json"
    scores = score_predictions(predictions_path, scores_path)
    assert json.loads(scores_path.read_text()) == scores
    assert scores["score_means"]["average_sentence_length"] == 4
    assert scores["domain_score_means"]["ONE"]["average_sentence_length"] == 4
    assert scores["results"][0]["scores"]["average_sentence_length"] == 4
    assert scores["results"][1]["error"] == "RuntimeError: expected"
