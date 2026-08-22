import json
import time
from types import SimpleNamespace

import pytest

from simple_llm.inference.vllm import (
    build_sampling_params,
    format_prompt,
    normalize_vllm_output,
    prepare_scaled_adapter,
)


def test_format_prompt_matches_chat_template() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
            ]
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return "formatted"

    assert format_prompt(Tokenizer(), "question", "system") == "formatted"


def test_build_sampling_params_maps_native_penalties_and_stop_ids() -> None:
    class Params:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    params = build_sampling_params(
        Params,
        eos_token_ids=[1, 2],
        seed=42,
        presence_penalty=0.5,
        repetition_penalty=1.05,
    )

    assert params.kwargs == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 2048,
        "seed": 42,
        "stop_token_ids": [1, 2],
        "skip_special_tokens": True,
        "presence_penalty": 0.5,
        "repetition_penalty": 1.05,
    }


def test_build_sampling_params_uses_vllm_defaults_for_missing_penalties() -> None:
    class Params:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    params = build_sampling_params(
        Params,
        eos_token_ids=[1],
        seed=42,
        presence_penalty=None,
        repetition_penalty=None,
    )

    assert "presence_penalty" not in params.kwargs
    assert "repetition_penalty" not in params.kwargs


def test_prepare_scaled_adapter_copies_weights_and_scales_only_copy(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = {"lora_alpha": 16, "other": "unchanged"}
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    destination = tmp_path / "scaled"

    assert prepare_scaled_adapter(adapter, 0.25, destination) == destination
    assert json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8")) == config
    assert json.loads((destination / "adapter_config.json").read_text(encoding="utf-8"))["lora_alpha"] == 4
    assert (destination / "adapter_model.safetensors").read_bytes() == b"weights"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ("{not json", "Invalid adapter config"),
        (json.dumps({}), "Invalid adapter config"),
    ],
)
def test_prepare_scaled_adapter_rejects_malformed_config(tmp_path, config, message) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(config, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        prepare_scaled_adapter(adapter, 0.25, tmp_path / "scaled")


def test_prepare_scaled_adapter_rejects_negative_scale(tmp_path) -> None:
    with pytest.raises(ValueError, match="scale"):
        prepare_scaled_adapter(tmp_path / "missing", -0.25, tmp_path / "scaled")


def test_normalize_vllm_output_counts_eos_token() -> None:
    output = SimpleNamespace(
        request_id="request-1",
        prompt_token_ids=[10, 11],
        outputs=[SimpleNamespace(token_ids=[20, 21, 2], text="answer")],
    )
    started = time.perf_counter() - 0.01

    result = normalize_vllm_output(output, started, max_tokens=3)

    assert result["response"] == "answer"
    assert result["input_tokens"] == 2
    assert result["output_tokens"] == 3
    assert result["truncated"] is True
    assert result["generation_seconds"] >= 0


def test_normalize_vllm_output_rejects_missing_choice_with_request_id() -> None:
    output = SimpleNamespace(request_id="request-7", prompt_token_ids=[10], outputs=[])

    with pytest.raises(ValueError, match="request-7"):
        normalize_vllm_output(output, time.perf_counter(), max_tokens=2048)
