# vLLM Continuous Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an AsyncLLM-backed Modal backend, validate it as Experiment 08 against Experiment 07, and preserve ordered run history and resume behavior.

**Architecture:** Keep the existing synchronous Transformers Modal backend unchanged for historical experiments. Add a separate vLLM Modal module with a class-level `@modal.concurrent` limit; its async method submits each request to one `AsyncLLM` engine, which performs continuous batching. Add an async ordered writer in the shared inference layer so out-of-order completions are buffered but JSONL remains an exact evaluation-order prefix.

**Tech Stack:** Python 3.11+ locally, Python 3.13 in Modal, Modal 1.5.3+, vLLM 0.17.0, CUDA 12.9 image, asyncio, pytest, existing Transformers/PEFT adapter artifacts, Modal Volumes.

**Spec:** `docs/superpowers/specs/2026-08-22-vllm-continuous-batching-design.md`

## Global Constraints

- Experiment 08 is the only validation experiment; Experiments 03–07 and their existing run artifacts remain unchanged.
- Preserve the existing `predictions.jsonl` schema, evaluation-order prefix contract, incremental flushing, and `--resume` validation.
- The initial Modal concurrency is `@modal.concurrent(max_inputs=16, target_inputs=16)` and is tuned only from benchmark evidence.
- Use vLLM-native `SamplingParams` penalties first; do not add a custom repetition processor unless validation shows a meaningful regression.
- Preserve the current effective SFT LoRA scale of `0.25`; never silently serve the raw adapter. Qwen3.5 SFT uses a cached PEFT-merged model because vLLM 0.17 native LoRA activation can fail on fused projections.
- Acceptance requires lower inference wall time than Experiment 07, no new systematic failures, no lower validity rate, and no rule-score mean decrease greater than `0.02` absolute.
- Pin `vllm==0.17.0` in the Modal CUDA image; this ruling remains unverified until Task 5 smoke-tests the exact `Qwen/Qwen3.5-4B` path. Do not add vLLM to the host dependency set because local runs must not install the GPU serving stack.

### Task 1: Add an ordered asynchronous prediction writer

**Files:**
- Modify: `simple_llm/inference/local.py` near `Generator` and `generate_predictions`
- Modify: `simple_llm/inference/__init__.py`
- Test: `tests/test_inference.py`

**Interfaces:**
- Produces `AsyncGenerator = Callable[[str, str | None], Awaitable[dict[str, Any]]]`.
- Produces `async_generate_predictions(evals, generator, predictions_path, system_prompt=None, max_in_flight=16) -> list[dict[str, Any]]`.
- The async writer accepts the same result dictionaries as `generate_predictions` and writes the same JSONL fields and error format.

- [ ] **Step 1: Write the failing ordering and failure-isolation tests**

Add tests using an async fake generator that sleeps for different durations:

```python
def test_async_generate_predictions_flushes_in_evaluation_order(tmp_path):
    async def generator(prompt, system_prompt):
        await asyncio.sleep(0.02 if prompt == "first" else 0.0)
        return {"response": prompt}

    results = asyncio.run(
        async_generate_predictions(
            [{"id": "A", "prompt": "first"}, {"id": "B", "prompt": "second"}],
            generator,
            tmp_path / "predictions.jsonl",
            max_in_flight=2,
        )
    )

    assert [item["id"] for item in results] == ["A", "B"]
    rows = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert [row["id"] for row in rows] == ["A", "B"]

def test_async_generate_predictions_records_one_request_failure_and_continues(tmp_path):
    async def generator(prompt, system_prompt):
        if prompt == "bad":
            return {"error": "RuntimeError: expected"}
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
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/test_inference.py -k async_generate -v`

Expected: FAIL because `async_generate_predictions` is not defined.

- [ ] **Step 3: Implement bounded scheduling and ordered flushing**

Add `asyncio` imports and implement the writer with these rules:

```python
async def async_generate_predictions(evals, generator, predictions_path, system_prompt=None, max_in_flight=16):
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be positive")
    # Reuse generate_predictions' prefix validation, then keep at most
    # max_in_flight asyncio tasks. Store each (index, result) as it completes,
    # flush buffer[next_index] and increment next_index while it is available,
    # and flush the file after every row.
```

Use `asyncio.Semaphore(max_in_flight)` or a bounded worker queue; do not create unbounded tasks for the full evaluation set. Reuse the existing resume-prefix validation instead of inventing a second file format. Export the new type and function from `simple_llm/inference/__init__.py`.

Request-level engine failures arrive as result dictionaries from the remote
method and are persisted in order. An exception raised by the async generator
is a Modal transport/container failure: propagate it, cancel every pending
task, and leave only the already flushed contiguous prefix on disk.

- [ ] **Step 4: Run all inference tests**

Run: `uv run pytest tests/test_inference.py -v`

Expected: PASS, including the existing synchronous resume tests.

- [ ] **Step 5: Commit**

```bash
git add simple_llm/inference/local.py simple_llm/inference/__init__.py tests/test_inference.py
git commit -m "feat: preserve ordered history for async inference"
```

### Task 2: Implement vLLM prompt, sampling, output, and scaled-adapter helpers

**Files:**
- Create: `simple_llm/inference/vllm.py`
- Test: `tests/test_vllm_inference.py`

**Interfaces:**
- Produces `format_prompt(tokenizer: Any, prompt: str, system_prompt: str | None) -> str`.
- Produces `build_sampling_params(SamplingParams: Any, *, eos_token_ids: list[int], seed: int, presence_penalty: float | None, repetition_penalty: float | None) -> Any`.
- Produces `prepare_scaled_adapter(adapter_path: Path, scale: float, destination: Path) -> Path`.
- Produces `normalize_vllm_output(output: Any, started: float, max_tokens: int) -> dict[str, Any]`.

- [ ] **Step 1: Write tests for exact prompt and sampling translation**

Use fake tokenizer and fake `SamplingParams` classes so tests do not import vLLM locally:

```python
def test_format_prompt_matches_chat_template():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "system", "content": "system"}, {"role": "user", "content": "question"}]
            assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
            return "formatted"

    assert format_prompt(Tokenizer(), "question", "system") == "formatted"

def test_build_sampling_params_maps_native_penalties_and_stop_ids():
    class Params:
        def __init__(self, **kwargs): self.kwargs = kwargs

    params = build_sampling_params(
        Params, eos_token_ids=[1, 2], seed=42,
        presence_penalty=0.5, repetition_penalty=1.05,
    )

    assert params.kwargs["presence_penalty"] == 0.5
    assert params.kwargs["repetition_penalty"] == 1.05
    assert params.kwargs["stop_token_ids"] == [1, 2]
    assert params.kwargs["max_tokens"] == 2048
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/test_vllm_inference.py -k 'prompt or sampling' -v`

Expected: FAIL because the helper module does not exist.

- [ ] **Step 3: Implement helpers with lazy vLLM imports**

Build the prompt with the repository’s current message shape and `enable_thinking=False`. Map `temperature=0.7`, `top_p=0.8`, `top_k=20`, `max_tokens=2048`, `seed`, `stop_token_ids`, and `skip_special_tokens=True`. Add penalty keys only when supplied; map `None` to vLLM defaults.

For adapter scaling, parse `adapter_config.json`, multiply each LoRA `lora_alpha` by the supplied `scale` (Experiment 08 passes `0.25`), and materialize the adapter in a temporary sibling without changing weight files. Validate the staged config and a source/destination manifest marker before atomically installing it at the requested destination. Reuse only a destination whose marker, scale, config, and file hashes still validate; replace stale, raw, or incomplete caches. Reject missing or malformed config and reject scales below zero. Return the destination path. This preserves the existing `scale_layer(0.25)` effective scaling without importing PEFT in the vLLM image.

Normalize `RequestOutput` using `prompt_token_ids`, `outputs[0].token_ids`, and `outputs[0].text`; set `truncated` when output token count reaches `MAX_NEW_TOKENS`; measure generation duration from the request start.

- [ ] **Step 4: Add adapter and output tests**

Cover config immutability, scaled `lora_alpha`, validated cache reuse, stale/incomplete destination replacement, interrupted staging cleanup, malformed config rejection, EOS-inclusive output normalization, and a missing output choice. The missing-choice test must assert a `ValueError` with the request ID so the remote layer can record it as a request failure.

- [ ] **Step 5: Run focused and full unit tests**

Run: `uv run pytest tests/test_vllm_inference.py tests/test_modal_inference.py -v`

Expected: PASS without importing the GPU-only vLLM package on the host.

- [ ] **Step 6: Commit**

```bash
git add simple_llm/inference/vllm.py tests/test_vllm_inference.py
git commit -m "feat: add vLLM sampling and adapter helpers"
```

### Task 3: Add the AsyncLLM Modal backend

**Files:**
- Create: `simple_llm/inference/modal_vllm.py`
- Test: `tests/test_modal_vllm.py`

**Interfaces:**
- Produces `modal_vllm_generator(model_name: str, gpu: str, seed: int, adapter_run: str | None = None, presence_penalty: float | None = None, repetition_penalty: float | None = None) -> ContextManager[tuple[AsyncGenerator, dict[str, Any]]]`.
- Produces an async remote method `VLLMModel.generate(prompt: str, system_prompt: str | None = None) -> dict[str, Any]`.
- Consumes all helper interfaces from Task 2 and the existing adapter validation functions from `simple_llm.inference.modal`.

- [ ] **Step 1: Write lifecycle and request tests with fakes**

Test the class logic without starting Modal or CUDA by replacing the lazy `AsyncLLM`, tokenizer, and remote object with fakes. Assert that one request calls `engine.generate` with a unique request ID, the configured `SamplingParams`, and a `LoRARequest` only when an adapter is present. Assert that the final streamed output is normalized and that an engine exception is returned as a request-level error dictionary.

- [ ] **Step 2: Run focused tests to verify they fail**

Run: `uv run pytest tests/test_modal_vllm.py -v`

Expected: FAIL because the vLLM Modal module does not exist.

- [ ] **Step 3: Define the isolated Modal image and class**

Create a vLLM-specific image from `nvidia/cuda:12.9.0-devel-ubuntu22.04` with Python 3.13, install `vllm==0.17.0` and `huggingface-hub==0.36.0`, add the local `simple_llm` source, and set `HF_HOME=/cache/huggingface` plus the vLLM cache environment. Treat the vLLM pin as unverified until Task 5 loads the exact target model. Mount the existing Hugging Face and training Volumes plus a `simple-llm-vllm-cache` Volume at `/cache/vllm`.

Define `VLLMModel` with `@app.cls(...)` and class-level `@modal.concurrent(max_inputs=16, target_inputs=16)`. Keep `max_containers=1` for the benchmark so one GPU measures one continuously batched engine.

- [ ] **Step 4: Initialize and warm the AsyncLLM engine**

 In a separate Transformers 5.5 Modal function, validate the adapter run and materialize the scaled PEFT-merged model under the container cache. Then `@modal.enter` loads the tokenizer, validates the prepared source, constructs `AsyncEngineArgs` with `enable_lora=False` and `language_model_only=True`, creates `AsyncLLM.from_engine_args(...)`, builds metadata including vLLM version, GPU, model revision, adapter identity/scale, and concurrency, and issues one short warm-up request.

- [ ] **Step 5: Implement the async request method and cleanup**

Use `engine.generate(prompt_text, sampling_params, request_id, lora_request=...)` and consume the async iterator until `finished`. Convert per-request engine exceptions to shared result dictionaries with an `error` field; Modal transport/container exceptions occur outside the remote method and must still propagate locally. Use `modal.current_input_id()` in logs when available. Add `@modal.exit` cleanup that drops the engine reference and calls the supported `shutdown()` method when available, falling back to `shutdown_background_loop()` only for compatibility.

- [ ] **Step 6: Add the local context-manager wrapper**

Start `app.run()`, instantiate the parameterized class with `.with_options(gpu=gpu)`, return `remote.generate.remote.aio` as the async generator callable, and include `gpu_requested` plus remote metadata. Validate adapter names before starting the app, matching `modal_generator` behavior.

- [ ] **Step 7: Run unit tests and commit**

Run: `uv run pytest tests/test_modal_vllm.py tests/test_vllm_inference.py -v`

Expected: PASS with all GPU/Modal imports mocked or lazy.

```bash
git add simple_llm/inference/modal_vllm.py tests/test_modal_vllm.py
git commit -m "feat: add AsyncLLM Modal backend"
```

### Task 4: Wire the backend into the runner and create Experiment 08

**Files:**
- Modify: `simple_llm/experiment_runner.py`
- Create: `experiments/08_qwen35_4b_sft_vllm.py`
- Modify: `tests/test_inference.py`
- Modify: `README.md`

**Interfaces:**
- `run_experiment` accepts `--backend vllm` in addition to `local` and `modal`.
- The vLLM branch selects `modal_vllm_generator` and `async_generate_predictions` while preserving all existing config and resume checks.
- Experiment 08 uses the same model, adapter requirement, seed, penalties, and evaluation set as Experiment 07, changing only `experiment`, `condition`, and `backend`.

- [ ] **Step 1: Add runner tests for backend dispatch and config preservation**

Mock both generator context managers and assert that `backend="vllm"` calls the async writer, while `backend="modal"` still calls the synchronous writer. Assert that a vLLM config records `backend`, runtime metadata, and native repetition-penalty scope over prompt plus generated tokens, while legacy Transformers metadata remains generated-only.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/test_inference.py -k backend -v`

Expected: FAIL because `vllm` is not yet an accepted backend.

- [ ] **Step 3: Add the vLLM dispatch branch**

Extend parser choices to `("local", "modal", "vllm")`. Import the vLLM context manager lazily. Keep the existing synchronous `with generator_context` flow for local/legacy Modal and call `asyncio.run(async_generate_predictions(...))` only for vLLM. Preserve the existing incremental `summary.json` and `rule_scores.json` writes after generation.

- [ ] **Step 4: Add Experiment 08**

Create:

```python
run_experiment(
    experiment="08_qwen35_4b_sft_vllm",
    model="Qwen/Qwen3.5-4B",
    condition="sft_lora_vllm_combined_penalties",
    default_backend="vllm",
    description=__doc__,
    require_adapter_run=True,
    presence_penalty=0.5,
    repetition_penalty=1.05,
)
```

Keep the same `--gpu`, `--limit`, `--adapter-run`, and `--resume` flags exposed by `run_experiment`.

- [ ] **Step 5: Document the comparison workflow**

Add README instructions that run Experiment 07 and 08 against the same adapter and `--limit`, then compare `summary.json`, `rule_scores.json`, `config.json`, and prediction samples. State that vLLM is not the default Modal backend until acceptance criteria pass.

- [ ] **Step 6: Run the full unit suite and commit**

Run: `uv run pytest -q`

Expected: PASS with no Modal credentials or GPU required.

```bash
git add simple_llm/experiment_runner.py experiments/08_qwen35_4b_sft_vllm.py tests/test_inference.py README.md
git commit -m "feat: add Experiment 08 vLLM benchmark"
```

### Task 5: Run the Modal benchmark and make the promotion decision

**Files:**
- Modify: `README.md` only if benchmark instructions need correction after the smoke test
- Create: no committed benchmark artifacts; keep generated runs under ignored `runs/`

**Interfaces:**
- Consumes the existing adapter run name and the same evaluation limit for Experiments 07 and 08.
- Produces two run directories with `config.json`, `predictions.jsonl`, `summary.json`, and `rule_scores.json`.

- [ ] **Step 1: Run a bounded legacy baseline**

Run:

```bash
uv run python experiments/07_qwen35_4b_sft_combined_penalties.py \
  --adapter-run RUN \
  --gpu L4 \
  --limit 32
```

Record the output run directory and its wall time, successful/failed/truncated counts, validity rate, and rule-score means.

- [ ] **Step 2: Run the matched vLLM trial**

Run:

```bash
uv run python experiments/08_qwen35_4b_sft_vllm.py \
  --adapter-run RUN \
  --gpu L4 \
  --limit 32
```

First confirm that the pinned vLLM release loads the exact model path `Qwen/Qwen3.5-4B`; the `vllm==0.17.0` pin remains unverified until this smoke test succeeds. Confirm `config.json` records vLLM `0.17.0`, concurrency `16`, adapter scale `0.25`, and native penalty settings.

- [ ] **Step 3: Verify resume and history preservation**

Interrupt a vLLM run after at least 8 rows, rerun it with `--resume RUN_DIR`, and assert that the final `predictions.jsonl` has exactly one row per evaluation item in original order and that already flushed rows were not regenerated.

Run: `uv run pytest tests/test_inference.py -k 'resume or async' -v`

- [ ] **Step 4: Compare quality and throughput**

Compare the two run directories. Verify lower total wall time and higher wall-clock output tokens/second for vLLM; verify no new systematic failures, no lower validity rate, and no rule-score mean decrease greater than `0.02` absolute. If any metric is close to the threshold, repeat both runs on the same fixed 32-item set.

- [ ] **Step 5: Scale to the full Experiment 07 set**

Only after the bounded trial passes, run both backends without `--limit` on the same GPU and adapter. Save the run paths and comparison results in experiment notes or README; do not overwrite either run.

- [ ] **Step 6: Promote only with evidence**

If the full comparison passes, make a separate follow-up change that switches the default Modal experiment path to vLLM while retaining an explicit legacy backend for reproducibility. If it fails, leave the legacy default unchanged and inspect whether concurrency, LoRA scaling, or the native repetition-penalty semantic difference caused the regression before adding a custom processor.

## Self-review checklist

- Spec coverage: Tasks 1–4 cover ordered persistence, AsyncLLM lifecycle, adapter scaling, native penalties, Experiment 08, and documentation; Task 5 covers the acceptance gate and promotion decision.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation step remains; `RUN` is a command-line placeholder intentionally replaced by the user’s adapter run name.
- Type consistency: `AsyncGenerator`, `async_generate_predictions`, `modal_vllm_generator`, and `VLLMModel.generate` signatures are defined before later tasks consume them.
- Scope: no public HTTP service, multi-GPU serving, or changes to historical experiments are included.
