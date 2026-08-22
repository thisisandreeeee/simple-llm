import asyncio
from contextlib import contextmanager
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
        self.shutdown = False

    async def generate(self, prompt, params, request_id, **kwargs):
        self.calls.append((prompt, params, request_id, kwargs))
        yield SimpleNamespace(
            request_id=request_id,
            prompt_token_ids=[10, 11],
            outputs=[SimpleNamespace(text=" answer ", token_ids=[20, 2])],
            finished=True,
        )

    def shutdown_background_loop(self):
        self.shutdown = True


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
        vllm_version="0.13.0",
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
    }
    assert len(engine.calls) == 2
    warmup, request = engine.calls
    assert warmup[2] != request[2]
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
    assert model.metadata["vllm_version"] == "0.13.0"
    assert model.metadata["gpu_actual"] == "Fake GPU"
    assert model.metadata["model_revision"] == "revision-1"
    assert model.metadata["concurrency"] == 16

    asyncio.run(model.cleanup())

    assert engine.shutdown is True
    assert model.engine is None


def test_adapter_is_scaled_and_sent_as_lora_request(monkeypatch, tmp_path) -> None:
    engine = FakeEngine()
    runtime = fake_runtime(engine)
    scaled_path = tmp_path / "scaled"
    monkeypatch.setattr(modal_vllm, "_runtime", lambda: runtime)
    monkeypatch.setattr(modal_vllm, "VLLM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        modal_vllm, "training_adapter_path", lambda run: "/training/run/adapter"
    )
    monkeypatch.setattr(modal_vllm, "validate_adapter_config", lambda path, model: {})
    monkeypatch.setattr(
        modal_vllm,
        "prepare_scaled_adapter",
        lambda source, scale, destination: scaled_path,
    )
    model = new_model(adapter_run="run")

    asyncio.run(model.load())
    asyncio.run(model.generate("question"))

    assert runtime.AsyncLLM.engine_args.kwargs["enable_lora"] is True
    request = engine.calls[-1]
    lora = request[3]["lora_request"]
    assert (lora.name, lora.request_id, lora.path) == ("run", 1, str(scaled_path))
    assert model.metadata["adapter"] == {
        "run": "run",
        "path": str(scaled_path),
        "scale": 0.25,
        "validated": True,
        "merged": False,
    }


def test_engine_exception_is_surfaced_for_the_request(monkeypatch) -> None:
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

    with pytest.raises(RuntimeError, match="engine failed"):
        asyncio.run(model.generate("question"))


def test_generator_exposes_async_remote_callable_and_metadata(monkeypatch) -> None:
    calls = []

    @contextmanager
    def active_context():
        calls.append("entered")
        yield

    async def generate(prompt, system_prompt=None):
        return {"response": prompt}

    remote = SimpleNamespace(
        generate=SimpleNamespace(aio=generate),
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
