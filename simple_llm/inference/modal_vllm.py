"""Isolated Modal backend for continuously batched vLLM inference."""

import inspect
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import modal

from .local import AsyncGenerator, MAX_NEW_TOKENS
from .modal import (
    ADAPTER_SCALE,
    CACHE_DIR,
    TRAINING_DIR,
    cache,
    training,
    training_adapter_path,
    validate_adapter_config,
)
from .vllm import (
    build_sampling_params,
    format_prompt,
    normalize_vllm_output,
    prepare_scaled_adapter,
)

VLLM_CACHE_DIR = "/cache/vllm"
CONCURRENCY = 16

app = modal.App("simple-llm-vllm-inference")
vllm_cache = modal.Volume.from_name("simple-llm-vllm-cache", create_if_missing=True)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.13"
    )
    .uv_pip_install("vllm==0.17.0", "huggingface-hub==0.36.0")
    .env(
        {
            "HF_HOME": CACHE_DIR,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_CACHE_ROOT": VLLM_CACHE_DIR,
        }
    )
    .add_local_python_source("simple_llm")
)


def _runtime() -> SimpleNamespace:
    """Import GPU-only dependencies inside the Modal container."""
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.lora.request import LoRARequest

    properties = torch.cuda.get_device_properties(0)
    return SimpleNamespace(
        AsyncEngineArgs=AsyncEngineArgs,
        AsyncLLM=AsyncLLM,
        SamplingParams=SamplingParams,
        LoRARequest=LoRARequest,
        AutoTokenizer=AutoTokenizer,
        vllm_version=vllm.__version__,
        gpu_name=properties.name,
        gpu_memory_gib=properties.total_memory / 1024**3,
    )


@app.cls(
    image=image,
    volumes={
        CACHE_DIR: cache,
        TRAINING_DIR: training,
        VLLM_CACHE_DIR: vllm_cache,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    max_containers=1,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=CONCURRENCY, target_inputs=CONCURRENCY)
class VLLMModel:
    model_name: str = modal.parameter()
    seed: int = modal.parameter()
    adapter_run: str = modal.parameter(default="")
    presence_penalty: str = modal.parameter(default="0")
    repetition_penalty: str = modal.parameter(default="0")

    @modal.enter()
    async def load(self) -> None:
        runtime = _runtime()
        started = time.perf_counter()
        self.tokenizer = runtime.AutoTokenizer.from_pretrained(self.model_name)
        self.sampling_params = build_sampling_params(
            runtime.SamplingParams,
            eos_token_ids=_eos_token_ids(self.tokenizer.eos_token_id),
            seed=self.seed,
            presence_penalty=float(self.presence_penalty) or None,
            repetition_penalty=float(self.repetition_penalty) or None,
        )
        self.lora_request = None
        adapter_metadata = None
        if self.adapter_run:
            adapter_path = Path(training_adapter_path(self.adapter_run))
            validate_adapter_config(adapter_path, self.model_name)
            destination = Path(VLLM_CACHE_DIR) / "adapters" / self.adapter_run
            scaled_path = prepare_scaled_adapter(
                adapter_path, ADAPTER_SCALE, destination
            )
            self.lora_request = runtime.LoRARequest(
                self.adapter_run, 1, str(scaled_path)
            )
            adapter_metadata = {
                "run": self.adapter_run,
                "path": str(scaled_path),
                "scale": ADAPTER_SCALE,
                "validated": True,
                "merged": False,
            }

        engine_args = runtime.AsyncEngineArgs(
            model=self.model_name,
            seed=self.seed,
            enable_lora=self.lora_request is not None,
        )
        self.engine = runtime.AsyncLLM.from_engine_args(engine_args)
        self.metadata = {
            "backend": "vllm",
            "device": "cuda",
            "gpu_actual": runtime.gpu_name,
            "gpu_memory_gib": runtime.gpu_memory_gib,
            "model_revision": self.tokenizer.init_kwargs.get("_commit_hash"),
            "model_load_seconds": time.perf_counter() - started,
            "vllm_version": runtime.vllm_version,
            "concurrency": CONCURRENCY,
        }
        if adapter_metadata:
            self.metadata["adapter"] = adapter_metadata

        warmup_started = time.perf_counter()
        await self._generate("Warm up", runtime.SamplingParams(max_tokens=1))
        self.metadata["warmup_seconds"] = time.perf_counter() - warmup_started

    @modal.method()
    def info(self) -> dict[str, Any]:
        return self.metadata

    @modal.method()
    async def generate(
        self, prompt: str, system_prompt: str | None = None
    ) -> dict[str, Any]:
        try:
            return await self._generate(
                format_prompt(self.tokenizer, prompt, system_prompt),
                self.sampling_params,
            )
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    async def _generate(self, prompt_text: str, sampling_params: Any) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        try:
            input_id = modal.current_input_id()
        except Exception:
            input_id = None
        print(f"vLLM request {request_id} (Modal input {input_id or 'warmup'})")
        kwargs = (
            {"lora_request": self.lora_request} if self.lora_request is not None else {}
        )
        started = time.perf_counter()
        completed = None
        async for output in self.engine.generate(
            prompt_text, sampling_params, request_id, **kwargs
        ):
            if output.finished:
                completed = output
                break
        if completed is None:
            raise RuntimeError(f"vLLM request {request_id} ended without a final output")
        return normalize_vllm_output(completed, started, MAX_NEW_TOKENS)

    @modal.exit()
    async def cleanup(self) -> None:
        engine = getattr(self, "engine", None)
        self.engine = None
        shutdown = getattr(engine, "shutdown", None)
        if not callable(shutdown):
            shutdown = getattr(engine, "shutdown_background_loop", None)
        if callable(shutdown):
            result = shutdown()
            if inspect.isawaitable(result):
                await result


def _eos_token_ids(eos_token_id: int | list[int] | None) -> list[int]:
    if eos_token_id is None:
        return []
    return [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)


@contextmanager
def modal_vllm_generator(
    model_name: str,
    gpu: str,
    seed: int,
    adapter_run: str | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
) -> Iterator[tuple[AsyncGenerator, dict[str, Any]]]:
    """Run one vLLM container and expose its async remote generation method."""
    if adapter_run:
        training_adapter_path(adapter_run)
    with modal.enable_output(), app.run():
        remote = VLLMModel.with_options(gpu=gpu)(
            model_name=model_name,
            seed=seed,
            adapter_run=adapter_run or "",
            presence_penalty=str(presence_penalty or 0.0),
            repetition_penalty=str(repetition_penalty or 0.0),
        )
        metadata = {"gpu_requested": gpu, **remote.info.remote()}
        yield remote.generate.remote.aio, metadata
