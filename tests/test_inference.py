import asyncio
import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from transformers import RepetitionPenaltyLogitsProcessor

from simple_llm import experiment_runner
from simple_llm.inference import (
    PresencePenaltyLogitsProcessor,
    async_generate_predictions,
    generate,
    generate_predictions,
)
from simple_llm.scoring.rule_scoring import score_predictions


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


def test_generate_combines_repetition_and_presence_penalties() -> None:
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
    generate(
        model,
        Tokenizer(),
        torch.device("cpu"),
        "prompt",
        presence_penalty=0.5,
        repetition_penalty=1.05,
    )

    repetition, presence = model.kwargs["logits_processor"]
    assert isinstance(repetition, RepetitionPenaltyLogitsProcessor)
    assert repetition.penalty == 1.05
    assert repetition.prompt_ignore_length == 2
    assert isinstance(presence, PresencePenaltyLogitsProcessor)
    assert presence.penalty == 0.5


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
    diagnostics = experiment_runner.summarize_inference(results)
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


def test_async_generate_predictions_flushes_in_evaluation_order(tmp_path) -> None:
    async def generator(prompt: str, system_prompt: str | None):
        await asyncio.sleep(0.02 if prompt == "first" else 0.0)
        return {"response": prompt}

    predictions_path = tmp_path / "predictions.jsonl"
    results = asyncio.run(
        async_generate_predictions(
            [{"id": "A", "prompt": "first"}, {"id": "B", "prompt": "second"}],
            generator,
            predictions_path,
            max_in_flight=2,
        )
    )

    assert [item["id"] for item in results] == ["A", "B"]
    rows = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    assert [row["id"] for row in rows] == ["A", "B"]


def test_async_generate_predictions_records_one_failure_and_continues(tmp_path) -> None:
    async def generator(prompt: str, system_prompt: str | None):
        if prompt == "bad":
            raise RuntimeError("expected")
        return {"response": "ok"}

    results = asyncio.run(
        async_generate_predictions(
            [{"id": "A", "prompt": "bad"}, {"id": "B", "prompt": "good"}],
            generator,
            tmp_path / "predictions.jsonl",
        )
    )

    assert results[0]["error"] == "RuntimeError: expected"
    assert results[1]["response"] == "ok"


def _run_backend(monkeypatch, tmp_path, backend: str):
    from simple_llm.inference import modal, modal_vllm

    calls = []

    @contextmanager
    def modal_context(*args, **kwargs):
        calls.append(("modal_context", args, kwargs))
        yield lambda *_: {}, {
            "gpu_actual": "test-gpu",
            "runtime_version": "test-version",
        }

    @contextmanager
    def vllm_context(*args, **kwargs):
        calls.append(("vllm_context", args, kwargs))
        yield lambda *_: {}, {
            "gpu_actual": "test-gpu",
            "runtime_version": "test-version",
        }

    def sync_writer(*args):
        calls.append(("sync", args))
        return []

    async def async_writer(*args):
        calls.append(("async", args))
        return []

    monkeypatch.setattr(experiment_runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        experiment_runner, "load_evals", lambda: [{"id": "A-1", "prompt": "p"}]
    )
    monkeypatch.setattr(experiment_runner, "git_info", lambda: {"commit": "test"})
    monkeypatch.setattr(experiment_runner, "generate_predictions", sync_writer)
    monkeypatch.setattr(
        experiment_runner, "async_generate_predictions", async_writer, raising=False
    )
    monkeypatch.setattr(experiment_runner, "score_predictions", lambda *_: None)
    monkeypatch.setattr(modal, "modal_generator", modal_context)
    monkeypatch.setattr(modal_vllm, "modal_vllm_generator", vllm_context)
    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment", "--backend", backend, "--adapter-run", "training-run"],
    )

    experiment_runner.run_experiment(
        experiment="test",
        model="Qwen/Qwen3.5-4B",
        condition="test",
        require_adapter_run=True,
        presence_penalty=0.5,
        repetition_penalty=1.05,
    )

    config_path = next((tmp_path / "runs").glob("*/config.json"))
    return calls, json.loads(config_path.read_text())


def test_vllm_backend_uses_async_writer_and_preserves_config(
    monkeypatch, tmp_path
) -> None:
    calls, config = _run_backend(monkeypatch, tmp_path, "vllm")

    assert [call[0] for call in calls] == ["vllm_context", "async"]
    assert config["backend"] == "vllm"
    assert config["gpu_actual"] == "test-gpu"
    assert config["runtime_version"] == "test-version"
    assert config["seed"] == 42
    assert config["generation"] == {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_new_tokens": 2048,
        "logits_processor": [
            {
                "type": "repetition_penalty",
                "penalty": 1.05,
                "prompt_tokens": "ignored",
            },
            {"type": "presence_penalty", "penalty": 0.5},
        ],
    }


def test_modal_backend_still_uses_sync_writer(monkeypatch, tmp_path) -> None:
    calls, _ = _run_backend(monkeypatch, tmp_path, "modal")

    assert [call[0] for call in calls] == ["modal_context", "sync"]
