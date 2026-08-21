"""Modal infrastructure adapter for the shared inference runner."""

import json
import re
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import modal

from simple_llm.inference import Generator, generate

CACHE_DIR = "/cache/huggingface"
TRAINING_DIR = "/training"
RUN_NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]*"
ADAPTER_MODEL_CLASS = "Qwen3_5ForConditionalGeneration"
ADAPTER_SCALE = 0.25

app = modal.App("simple-llm-inference")
cache = modal.Volume.from_name("simple-llm-huggingface-cache", create_if_missing=True)
training = modal.Volume.from_name("simple-llm-training", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("peft==0.20.0", "torch==2.11.0", "transformers==5.5.0")
    .env({"HF_HOME": CACHE_DIR, "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("simple_llm")
)


def training_adapter_path(adapter_run: str) -> str:
    """Return the mounted adapter path for a validated SFT training run."""
    if not re.fullmatch(RUN_NAME_PATTERN, adapter_run):
        raise ValueError(
            "Adapter run may contain only letters, numbers, '.', '_', and '-'"
        )
    return f"{TRAINING_DIR}/{adapter_run}/adapter"


def validate_adapter_config(adapter_path: Path, model_name: str) -> dict[str, Any]:
    """Validate that an adapter was trained for this model architecture."""
    config_path = adapter_path / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Invalid SFT adapter config: {config_path}") from exc
    base_class = config.get("auto_mapping", {}).get("base_model_class")
    if config.get("base_model_name_or_path") != model_name:
        raise ValueError(f"SFT adapter was not trained for {model_name}")
    if base_class != ADAPTER_MODEL_CLASS:
        raise ValueError(f"Unsupported SFT adapter model class: {base_class}")
    return config


def load_peft_adapter(model: Any, adapter_path: Path) -> Any:
    """Load an adapter, failing instead of silently accepting missing weights."""
    from peft import PeftModel

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=r"Found missing adapter keys.*")
        return PeftModel.from_pretrained(model, adapter_path)


def scale_peft_adapter(model: Any, scale: float) -> int:
    """Scale every loaded LoRA layer and return the number changed."""
    scaled = 0
    for module in model.modules():
        scale_layer = getattr(module, "scale_layer", None)
        if callable(scale_layer):
            scale_layer(scale)
            scaled += 1
    if not scaled:
        raise RuntimeError("No LoRA layers were available to scale")
    return scaled


@app.cls(
    image=image,
    volumes={CACHE_DIR: cache, TRAINING_DIR: training},
    secrets=[modal.Secret.from_name("huggingface")],
    max_containers=1,
    timeout=60 * 60,
)
class ModalModel:
    model_name: str = modal.parameter()
    seed: int = modal.parameter()
    adapter_run: str = modal.parameter(default="")
    presence_penalty: str = modal.parameter(default="0")
    repetition_penalty: str = modal.parameter(default="0")

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Qwen3_5ForConditionalGeneration,
        )

        torch.manual_seed(self.seed)
        started = time.perf_counter()
        self.target = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        adapter_path: Path | None = None
        adapter_config: dict[str, Any] | None = None
        model_class = AutoModelForCausalLM
        if self.adapter_run:
            adapter_path = Path(training_adapter_path(self.adapter_run))
            adapter_config = validate_adapter_config(adapter_path, self.model_name)
            model_class = Qwen3_5ForConditionalGeneration

        self.model = model_class.from_pretrained(
            self.model_name, torch_dtype="auto"
        ).to(self.target)
        loaded_model_class = type(self.model).__name__
        model_revision = getattr(self.model.config, "_commit_hash", None)
        adapter: dict[str, Any] | None = None
        if adapter_path and adapter_config:
            if loaded_model_class != ADAPTER_MODEL_CLASS:
                raise RuntimeError(
                    f"Loaded unexpected model class: {loaded_model_class}"
                )
            self.model = load_peft_adapter(self.model, adapter_path)
            scaled_layers = scale_peft_adapter(self.model, ADAPTER_SCALE)
            self.model = self.model.merge_and_unload(safe_merge=True)
            adapter = {
                "run": self.adapter_run,
                "path": str(adapter_path),
                "base_model_class": loaded_model_class,
                "scale": ADAPTER_SCALE,
                "scaled_layer_count": scaled_layers,
                "validated": True,
                "merged": True,
            }
        self.model.eval()
        properties = torch.cuda.get_device_properties(0)
        self.metadata = {
            "backend": "modal",
            "device": "cuda",
            "gpu_actual": properties.name,
            "gpu_memory_gib": properties.total_memory / 1024**3,
            "model_class": loaded_model_class,
            "model_revision": model_revision,
            "torch_dtype": str(self.model.dtype),
            "model_load_seconds": time.perf_counter() - started,
        }
        if adapter:
            self.metadata["adapter"] = adapter

    @modal.method()
    def info(self) -> dict[str, Any]:
        return self.metadata

    @modal.method()
    def generate(self, prompt: str, system_prompt: str | None = None) -> dict[str, Any]:
        presence_penalty = float(self.presence_penalty)
        repetition_penalty = float(self.repetition_penalty)
        return generate(
            self.model,
            self.tokenizer,
            self.target,
            prompt,
            system_prompt,
            presence_penalty or None,
            repetition_penalty or None,
        )


@contextmanager
def modal_generator(
    model_name: str,
    gpu: str,
    seed: int,
    adapter_run: str | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
) -> Iterator[tuple[Generator, dict[str, Any]]]:
    """Run an ephemeral Modal app and expose its remote model as a generator."""
    if adapter_run:
        training_adapter_path(adapter_run)
    with modal.enable_output(), app.run():
        remote = ModalModel.with_options(gpu=gpu)(
            model_name=model_name,
            seed=seed,
            adapter_run=adapter_run or "",
            presence_penalty=str(presence_penalty or 0.0),
            repetition_penalty=str(repetition_penalty or 0.0),
        )
        metadata = {"gpu_requested": gpu, **remote.info.remote()}
        yield remote.generate.remote, metadata
