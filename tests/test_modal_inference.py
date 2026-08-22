import json
import warnings

import pytest

from simple_llm.inference.modal import (
    ADAPTER_MODEL_CLASS,
    load_peft_adapter,
    scale_peft_adapter,
    training_adapter_path,
    validate_adapter_config,
)


def test_training_adapter_path_accepts_training_run_names() -> None:
    assert (
        training_adapter_path("qwen35-4b-sft-20260818-120000")
        == "/training/qwen35-4b-sft-20260818-120000/adapter"
    )


@pytest.mark.parametrize("run_name", ["", ".hidden", "../other", "has space"])
def test_training_adapter_path_rejects_unsafe_run_names(run_name: str) -> None:
    with pytest.raises(ValueError, match="Adapter run"):
        training_adapter_path(run_name)


def test_validate_adapter_config_accepts_matching_architecture(tmp_path) -> None:
    config = {
        "base_model_name_or_path": "Qwen/Qwen3.5-4B",
        "auto_mapping": {"base_model_class": ADAPTER_MODEL_CLASS},
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))

    assert validate_adapter_config(tmp_path, "Qwen/Qwen3.5-4B") == config


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {
                "base_model_name_or_path": "other/model",
                "auto_mapping": {"base_model_class": ADAPTER_MODEL_CLASS},
            },
            "was not trained for",
        ),
        (
            {
                "base_model_name_or_path": "Qwen/Qwen3.5-4B",
                "auto_mapping": {"base_model_class": "WrongModel"},
            },
            "Unsupported",
        ),
    ],
)
def test_validate_adapter_config_rejects_mismatch(
    tmp_path, config, message
) -> None:
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match=message):
        validate_adapter_config(tmp_path, "Qwen/Qwen3.5-4B")


def test_load_peft_adapter_rejects_missing_keys(monkeypatch, tmp_path) -> None:
    from peft import PeftModel

    def warn_about_missing_keys(*args, **kwargs):
        warnings.warn("Found missing adapter keys while loading the checkpoint")

    monkeypatch.setattr(PeftModel, "from_pretrained", warn_about_missing_keys)

    with pytest.raises(UserWarning, match="missing adapter keys"):
        load_peft_adapter(object(), tmp_path)


def test_scale_peft_adapter_scales_loaded_layers() -> None:
    class Layer:
        def scale_layer(self, scale):
            self.scale = scale

    layers = [Layer(), Layer()]

    class Model:
        def modules(self):
            return [self, *layers]

    assert scale_peft_adapter(Model(), 0.5) == 2
    assert [layer.scale for layer in layers] == [0.5, 0.5]


def test_scale_peft_adapter_rejects_model_without_lora() -> None:
    class Model:
        def modules(self):
            return [self]

    with pytest.raises(RuntimeError, match="No LoRA layers"):
        scale_peft_adapter(Model(), 0.5)
