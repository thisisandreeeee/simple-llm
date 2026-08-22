import asyncio
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_llm.inference import modal_vllm


class FakeParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTokenizer:
    eos_token_id = 2
    init_kwargs = {"_commit_hash": "revision-1"}

    @classmethod
    def from_pretrained(cls, model_name):
        assert model_name == "model/name"
        return cls()

    def apply_chat_template(self, messages, **kwargs):
        return f"formatted:{messages[-1]['content']}"


class FakeEngineArgs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLoRARequest:
    def __init__(self, name, request_id, path):
        self.name = name
        self.request_id = request_id
        self.path = path


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.shutdown_called = False

    async def generate(self, prompt, params, request_id, **kwargs):
        self.calls.append((prompt, params, request_id, kwargs))
        yield SimpleNamespace(
            request_id=request_id,
            prompt_token_ids=[10, 11],
            outputs=[SimpleNamespace(text=" answer ", token_ids=[20, 2])],
            finished=True,
        )

    async def shutdown(self):
        self.shutdown_called = True


def test_runtime_imports_async_engine_from_supported_vllm_modules(monkeypatch) -> None:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        get_device_properties=lambda index: types.SimpleNamespace(
            name="Fake GPU", total_memory=24 * 1024**3
        )
    )
    vllm = types.ModuleType("vllm")
    vllm.__version__ = "0.17.0"
    vllm.SamplingParams = FakeParams
    engine_arg_utils = types.ModuleType("vllm.engine.arg_utils")
    engine_arg_utils.AsyncEngineArgs = FakeEngineArgs
    async_llm = types.ModuleType("vllm.v1.engine.async_llm")
    async_llm.AsyncLLM = object
    lora_request = types.ModuleType("vllm.lora.request")
    lora_request.LoRARequest = FakeLoRARequest
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = FakeTokenizer
    modules = {
        "torch": torch,
        "vllm": vllm,
        "vllm.engine": types.ModuleType("vllm.engine"),
        "vllm.engine.arg_utils": engine_arg_utils,
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.engine": types.ModuleType("vllm.v1.engine"),
        "vllm.v1.engine.async_llm": async_llm,
        "vllm.lora": types.ModuleType("vllm.lora"),
        "vllm.lora.request": lora_request,
        "transformers": transformers,
    }
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    runtime = modal_vllm._runtime()

    assert runtime.AsyncEngineArgs is FakeEngineArgs
    assert runtime.AsyncLLM is async_llm.AsyncLLM
    assert runtime.SamplingParams is FakeParams


def fake_runtime(engine):
    class FakeAsyncLLM:
        engine_args = None

        @classmethod
        def from_engine_args(cls, engine_args):
            cls.engine_args = engine_args
            return engine

    return SimpleNamespace(
        AsyncEngineArgs=FakeEngineArgs,
        AsyncLLM=FakeAsyncLLM,
        SamplingParams=FakeParams,
        LoRARequest=FakeLoRARequest,
        AutoTokenizer=FakeTokenizer,
        vllm_version="0.17.0",
        gpu_name="Fake GPU",
        gpu_memory_gib=24.0,
    )


def new_model(**overrides):
    model = modal_vllm.VLLMModel._get_user_cls()()
    defaults = {
        "model_name": "model/name",
        "seed": 42,
        "adapter_run": "",
        "presence_penalty": "0.5",
        "repetition_penalty": "1.05",
    }
    for name, value in {**defaults, **overrides}.items():
        setattr(model, name, value)
    return model


def test_lifecycle_generates_with_unique_request_ids_and_cleans_up(monkeypatch) -> None:
    engine = FakeEngine()
    runtime = fake_runtime(engine)
    monkeypatch.setattr(modal_vllm, "_runtime", lambda: runtime)
    model = new_model()

    asyncio.run(model.load())
    result = asyncio.run(model.generate("question", "system"))

    assert runtime.AsyncLLM.engine_args.kwargs == {
        "model": "model/name",
        "seed": 42,
        "enable_lora": False,
        "language_model_only": True,
    }
    assert len(engine.calls) == 2
    warmup, request = engine.calls
    assert warmup[2] != request[2]
    assert warmup[1].kwargs["max_tokens"] == 1
    assert request[0] == "formatted:question"
    assert request[1].kwargs == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 2048,
        "seed": 42,
        "stop_token_ids": [2],
        "skip_special_tokens": True,
        "presence_penalty": 0.5,
        "repetition_penalty": 1.05,
    }
    assert request[3] == {}
    assert result["response"] == "answer"
    assert result["input_tokens"] == 2
    assert result["output_tokens"] == 2
    assert result["truncated"] is False
    assert model.metadata["vllm_version"] == "0.17.0"
    assert model.metadata["gpu_actual"] == "Fake GPU"
    assert model.metadata["model_revision"] == "revision-1"
    assert model.metadata["concurrency"] == 16

    asyncio.run(model.cleanup())

    assert engine.shutdown_called is True
    assert model.engine is None


def test_cleanup_falls_back_to_background_loop_shutdown() -> None:
    class LegacyEngine:
        def __init__(self):
            self.shutdown_called = False

        def shutdown_background_loop(self):
            self.shutdown_called = True

    engine = LegacyEngine()
    model = new_model()
    model.engine = engine

    asyncio.run(model.cleanup())

    assert engine.shutdown_called is True
    assert model.engine is None


def test_adapter_is_merged_before_vllm_load(monkeypatch, tmp_path) -> None:
    engine = FakeEngine()
    runtime = fake_runtime(engine)
    merged_path = tmp_path / "merged"
    merge_calls = []

    def merge(source, model_name, scale, destination):
        merge_calls.append((source, model_name, scale, destination))
        return merged_path

    monkeypatch.setattr(modal_vllm, "_runtime", lambda: runtime)
    monkeypatch.setattr(modal_vllm, "VLLM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        modal_vllm, "training_adapter_path", lambda run: "/training/run/adapter"
    )
    monkeypatch.setattr(modal_vllm, "validate_adapter_config", lambda path, model: {})
    monkeypatch.setattr(modal_vllm, "prepare_merged_model", merge, raising=False)
    model = new_model(adapter_run="run")

    asyncio.run(model.load())
    asyncio.run(model.generate("question"))

    assert runtime.AsyncLLM.engine_args.kwargs == {
        "model": str(merged_path),
        "seed": 42,
        "enable_lora": False,
        "language_model_only": True,
    }
    request = engine.calls[-1]
    assert request[3] == {}
    assert merge_calls == [
        (
            Path("/training/run/adapter"),
            "model/name",
            0.25,
            Path(tmp_path / "cache") / "merged" / "run",
        )
    ]
    assert model.metadata["adapter"] == {
        "run": "run",
        "path": str(merged_path),
        "scale": 0.25,
        "validated": True,
        "merged": True,
    }


def test_prepare_merged_model_scales_and_reuses_cached_artifact(monkeypatch, tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"lora_alpha": 16}), encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    destination = tmp_path / "merged" / "run"
    calls = []

    class FakeMergedModel:
        def to(self, device):
            assert device == "cuda"
            return self

        def eval(self):
            return self

        def merge_and_unload(self, safe_merge):
            assert safe_merge is True
            calls.append("merge")
            return self

        def save_pretrained(self, path, safe_serialization):
            assert safe_serialization is True
            Path(path, "config.json").write_text("{}", encoding="utf-8")

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, model_name, torch_dtype):
            assert (model_name, torch_dtype) == ("model/name", "auto")
            calls.append("load")
            return FakeMergedModel()

    class FakeTokenizerSaver:
        @classmethod
        def from_pretrained(cls, model_name):
            assert model_name == "model/name"
            return cls()

        def save_pretrained(self, path):
            Path(path, "tokenizer.json").write_text("{}", encoding="utf-8")

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(empty_cache=lambda: calls.append("empty"))
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = FakeTokenizerSaver
    transformers.Qwen3_5ForConditionalGeneration = FakeModelClass
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    import simple_llm.inference.modal as legacy_modal

    monkeypatch.setattr(legacy_modal, "load_peft_adapter", lambda model, path: calls.append("adapter") or model)
    monkeypatch.setattr(legacy_modal, "scale_peft_adapter", lambda model, scale: calls.append(scale) or 7)

    assert modal_vllm.prepare_merged_model(adapter, "model/name", 0.25, destination) == destination
    assert calls == ["load", "adapter", 0.25, "merge", "empty"]
    assert (destination / modal_vllm.MERGED_MODEL_MARKER).is_file()

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("cached merged model should be reused")

    monkeypatch.setattr(FakeModelClass, "from_pretrained", classmethod(fail_if_loaded))
    assert modal_vllm.prepare_merged_model(adapter, "model/name", 0.25, destination) == destination


def test_engine_exception_is_returned_as_a_request_error(monkeypatch) -> None:
    class FailingAfterWarmupEngine(FakeEngine):
        async def generate(self, prompt, params, request_id, **kwargs):
            if self.calls:
                raise RuntimeError("engine failed")
            async for output in super().generate(prompt, params, request_id, **kwargs):
                yield output

    engine = FailingAfterWarmupEngine()
    monkeypatch.setattr(modal_vllm, "_runtime", lambda: fake_runtime(engine))
    model = new_model()
    asyncio.run(model.load())

    result = asyncio.run(model.generate("question"))

    assert result == {"error": "RuntimeError: engine failed"}


def test_generator_exposes_async_remote_callable_and_metadata(monkeypatch) -> None:
    calls = []

    @contextmanager
    def active_context():
        calls.append("entered")
        yield

    async def generate(prompt, system_prompt=None):
        return {"response": prompt}

    remote = SimpleNamespace(
        generate=SimpleNamespace(remote=SimpleNamespace(aio=generate)),
        info=SimpleNamespace(remote=lambda: {"backend": "vllm"}),
    )

    class FakeModel:
        @staticmethod
        def with_options(**options):
            calls.append(options)

            def construct(**parameters):
                calls.append(parameters)
                return remote

            return construct

    monkeypatch.setattr(modal_vllm.modal, "enable_output", active_context)
    monkeypatch.setattr(modal_vllm.app, "run", active_context)
    monkeypatch.setattr(modal_vllm, "VLLMModel", FakeModel)

    with modal_vllm.modal_vllm_generator(
        "model/name", "L4", 42, presence_penalty=0.5
    ) as (generator, metadata):
        assert generator is generate
        assert metadata == {"gpu_requested": "L4", "backend": "vllm"}

    assert calls == [
        "entered",
        "entered",
        {"gpu": "L4"},
        {
            "model_name": "model/name",
            "seed": 42,
            "adapter_run": "",
            "presence_penalty": "0.5",
            "repetition_penalty": "0.0",
        },
    ]


def test_generator_validates_adapter_before_starting_app(monkeypatch) -> None:
    def unexpected_run():
        raise AssertionError("app.run must not be called")

    monkeypatch.setattr(modal_vllm.app, "run", unexpected_run)

    with pytest.raises(ValueError, match="Adapter run"):
        with modal_vllm.modal_vllm_generator(
            "model/name", "L4", 42, adapter_run="../unsafe"
        ):
            pass
