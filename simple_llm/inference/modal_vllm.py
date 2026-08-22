"""Isolated Modal backend for continuously batched vLLM inference."""

import inspect
import json
import os
import shutil
import tempfile
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
    _adapter_manifest,
    build_sampling_params,
    format_prompt,
    normalize_vllm_output,
)

VLLM_CACHE_DIR = "/cache/vllm"
CONCURRENCY = 16
MERGED_MODEL_MARKER = ".simple-llm-merged-model.json"

app = modal.App("simple-llm-vllm-inference")
vllm_cache = modal.Volume.from_name("simple-llm-vllm-cache", create_if_missing=True)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.13"
    )
    .uv_pip_install("vllm==0.17.0", "huggingface-hub==0.36.0", "peft==0.20.0")
    .env(
        {
            "HF_HOME": CACHE_DIR,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_CACHE_ROOT": VLLM_CACHE_DIR,
        }
    )
    .add_local_python_source("simple_llm")
)
merge_image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        "peft==0.20.0",
        "torch==2.11.0",
        "transformers==5.5.0",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_HOME": CACHE_DIR, "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("simple_llm")
)


@app.function(
    image=merge_image,
    volumes={
        CACHE_DIR: cache,
        TRAINING_DIR: training,
        VLLM_CACHE_DIR: vllm_cache,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,
)
def build_merged_model(
    adapter_run: str, model_name: str, scale: float, destination: str
) -> str:
    """Build the PEFT-merged artifact in a Transformers-compatible image."""
    adapter_path = Path(training_adapter_path(adapter_run))
    validate_adapter_config(adapter_path, model_name)
    merged_path = prepare_merged_model(
        adapter_path, model_name, scale, Path(destination)
    )
    vllm_cache.commit()
    return str(merged_path)


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
    model_source: str = modal.parameter(default="")
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
            if not self.model_source:
                raise RuntimeError(
                    "Adapter runs require a prebuilt merged model source"
                )
            model_source = self.model_source
            adapter_metadata = {
                "run": self.adapter_run,
                "path": str(model_source),
                "scale": ADAPTER_SCALE,
                "validated": True,
                "merged": True,
            }
        else:
            model_source = self.model_name

        engine_args = runtime.AsyncEngineArgs(
            model=str(model_source),
            seed=self.seed,
            enable_lora=False,
            language_model_only=True,
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


def prepare_merged_model(
    adapter_path: Path, model_name: str, scale: float, destination: Path
) -> Path:
    """Materialize the scaled PEFT adapter into a vLLM-loadable model."""
    import gc

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    from .modal import load_peft_adapter, scale_peft_adapter

    source_manifest = _adapter_manifest(adapter_path)
    if _valid_merged_model(destination, model_name, scale, source_manifest):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    model = None
    try:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto"
        ).to("cuda")
        model = load_peft_adapter(model, adapter_path)
        scale_peft_adapter(model, scale)
        model = model.merge_and_unload(safe_merge=True)
        model.eval()
        model.save_pretrained(stage, safe_serialization=True)
        AutoTokenizer.from_pretrained(model_name).save_pretrained(stage)
        (stage / MERGED_MODEL_MARKER).write_text(
            json.dumps(
                {
                    "version": 1,
                    "model_name": model_name,
                    "scale": scale,
                    "source_manifest": source_manifest,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            backup = destination.with_name(
                f".{destination.name}.stale-{uuid.uuid4().hex}"
            )
            os.replace(destination, backup)
        else:
            backup = None
        try:
            os.replace(stage, destination)
        except Exception:
            if backup is not None and not destination.exists():
                os.replace(backup, destination)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return destination


def _valid_merged_model(
    destination: Path,
    model_name: str,
    scale: float,
    source_manifest: dict[str, str],
) -> bool:
    try:
        marker = json.loads(
            (destination / MERGED_MODEL_MARKER).read_text(encoding="utf-8")
        )
        return (
            marker.get("version") == 1
            and marker.get("model_name") == model_name
            and marker.get("scale") == scale
            and marker.get("source_manifest") == source_manifest
            and (destination / "config.json").is_file()
        )
    except (json.JSONDecodeError, OSError, TypeError):
        return False


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
        model_source = ""
        if adapter_run:
            destination = str(Path(VLLM_CACHE_DIR) / "merged" / adapter_run)
            model_source = build_merged_model.with_options(gpu=gpu).remote(
                adapter_run, model_name, ADAPTER_SCALE, destination
            )
        remote = VLLMModel.with_options(gpu=gpu)(
            model_name=model_name,
            seed=seed,
            adapter_run=adapter_run or "",
            model_source=model_source,
            presence_penalty=str(presence_penalty or 0.0),
            repetition_penalty=str(repetition_penalty or 0.0),
        )
        metadata = {"gpu_requested": gpu, **remote.info.remote()}
        yield remote.generate.remote.aio, metadata
