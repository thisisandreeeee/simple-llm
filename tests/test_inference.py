import json
from types import SimpleNamespace

import pytest
import torch

from simple_llm.experiment import summarize_inference
from simple_llm.inference import (
    PresencePenaltyLogitsProcessor,
    generate,
    generate_predictions,
)
from simple_llm.rule_scoring import score_predictions


def test_generate_stops_on_model_and_chat_eos_tokens() -> None:
    class Tokenizer:
        eos_token_id = 248046
        pad_token_id = 248044

        def apply_chat_template(self, *args, **kwargs):
            return "formatted prompt"

        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[10, 11]])}

        def decode(self, *args, **kwargs):
            return "answer"

    class Model:
        generation_config = SimpleNamespace(eos_token_id=248044)

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return torch.tensor([[10, 11, 20, 248046]])

    model = Model()
    result = generate(model, Tokenizer(), torch.device("cpu"), "prompt")

    assert model.kwargs["eos_token_id"] == [248044, 248046]
    assert model.kwargs["pad_token_id"] == 248044
    assert "logits_processor" not in model.kwargs
    assert result["response"] == "answer"


def test_presence_penalty_applies_once_to_generated_tokens_only() -> None:
    processor = PresencePenaltyLogitsProcessor(prompt_length=2, penalty=1.5)
    input_ids = torch.tensor([[10, 11, 3, 3, 4]])
    scores = torch.zeros((1, 12))

    result = processor(input_ids, scores)

    assert result[0, 3].item() == -1.5
    assert result[0, 4].item() == -1.5
    assert result[0, 10].item() == 0
    assert result[0, 11].item() == 0
    assert torch.equal(scores, torch.zeros((1, 12)))


def test_generate_passes_configured_logits_processor() -> None:
    class Tokenizer:
        eos_token_id = 248046
        pad_token_id = 248044

        def apply_chat_template(self, *args, **kwargs):
            return "formatted prompt"

        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[10, 11]])}

        def decode(self, *args, **kwargs):
            return "answer"

    class Model:
        generation_config = SimpleNamespace(eos_token_id=248044)

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return torch.tensor([[10, 11, 20, 248046]])

    model = Model()
    result = generate(
        model,
        Tokenizer(),
        torch.device("cpu"),
        "prompt",
        presence_penalty=1.5,
    )

    assert isinstance(
        model.kwargs["logits_processor"][0], PresencePenaltyLogitsProcessor
    )
    assert model.kwargs["logits_processor"][0].penalty == 1.5
    assert result["response"] == "answer"


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


def test_generate_predictions_resumes_existing_prefix(tmp_path) -> None:
    evals = [
        {"id": "ONE-01", "prompt": "first"},
        {"id": "ONE-02", "prompt": "second"},
    ]
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"id": "ONE-01", "prompt": "first", "response": "done"}) + "\n"
    )
    calls = []

    def generator(prompt: str, system_prompt: str | None):
        calls.append(prompt)
        return {"response": "new"}

    results = generate_predictions(evals, generator, predictions_path)

    assert calls == ["second"]
    assert [result["response"] for result in results] == ["done", "new"]
    assert len(predictions_path.read_text().splitlines()) == 2


def test_generate_predictions_rejects_mismatched_prefix(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(json.dumps({"id": "OTHER", "prompt": "first"}) + "\n")

    with pytest.raises(ValueError, match="do not match"):
        generate_predictions(
            [{"id": "ONE-01", "prompt": "first"}],
            lambda *_: {},
            predictions_path,
        )
