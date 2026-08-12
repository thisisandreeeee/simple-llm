"""Modal infrastructure adapter for the shared inference runner."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import modal

from simple_llm.inference import Generator, generate

CACHE_DIR = "/cache/huggingface"

app = modal.App("simple-llm-inference")
cache = modal.Volume.from_name("simple-llm-huggingface-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.13.0", "transformers==5.14.1")
    .env({"HF_HOME": CACHE_DIR, "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("simple_llm")
)


@app.cls(
    image=image,
    volumes={CACHE_DIR: cache},
    secrets=[modal.Secret.from_name("huggingface")],
    max_containers=1,
    timeout=60 * 60,
)
class ModalModel:
    model_name: str = modal.parameter()
    seed: int = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(self.seed)
        started = time.perf_counter()
        self.target = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype="auto"
        ).to(self.target)
        self.model.eval()
        properties = torch.cuda.get_device_properties(0)
        self.metadata = {
            "backend": "modal",
            "device": "cuda",
            "gpu_actual": properties.name,
            "gpu_memory_gib": properties.total_memory / 1024**3,
            "model_revision": getattr(self.model.config, "_commit_hash", None),
            "torch_dtype": str(self.model.dtype),
            "model_load_seconds": time.perf_counter() - started,
        }

    @modal.method()
    def info(self) -> dict[str, Any]:
        return self.metadata

    @modal.method()
    def generate(self, prompt: str, system_prompt: str | None = None) -> dict[str, Any]:
        return generate(self.model, self.tokenizer, self.target, prompt, system_prompt)


@contextmanager
def modal_generator(
    model_name: str, gpu: str, seed: int
) -> Iterator[tuple[Generator, dict[str, Any]]]:
    """Run an ephemeral Modal app and expose its remote model as a generator."""
    with modal.enable_output(), app.run():
        remote = ModalModel.with_options(gpu=gpu)(model_name=model_name, seed=seed)
        metadata = {"gpu_requested": gpu, **remote.info.remote()}
        yield remote.generate.remote, metadata
