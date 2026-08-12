import json

from simple_llm.inference import evaluate, summary


def test_evaluate_keeps_results_when_one_prompt_fails(tmp_path) -> None:
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

    path = tmp_path / "predictions.jsonl"
    results = evaluate(evals, generator, path)

    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(saved) == 2
    assert saved[0]["response"] == "A pump moves liquid."
    assert saved[1]["error"] == "RuntimeError: expected"
    assert summary(results)["output_tokens_per_second"] == 10
