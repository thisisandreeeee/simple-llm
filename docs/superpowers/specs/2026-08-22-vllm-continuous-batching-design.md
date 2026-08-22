# vLLM Continuous Batching for Modal Inference

**Status:** Draft for review  
**Date:** 2026-08-22

## Goal

Add a vLLM-backed Modal inference path that uses `AsyncLLM` continuous
batching to reduce Experiment 07-style evaluation wall time while preserving
the existing run history, prediction JSONL format, resume behavior, adapter
validation, and scoring workflow.

Experiment 08 will be the isolated validation vehicle. Existing Modal
experiments and their artifacts remain unchanged until the comparison passes.

## Context and current behavior

`simple_llm/inference/modal.py` loads a Transformers model once in a Modal
class and exposes a synchronous `generate(prompt, system_prompt)` method.
`simple_llm/inference/local.py:generate_predictions` invokes that method once
per evaluation item and appends each result immediately to
`predictions.jsonl`. The runner resumes by validating that the existing file
is an exact prefix of the evaluation set.

The current SFT path validates and loads a PEFT adapter, scales every LoRA
layer by `0.25`, merges the adapter into the base model, and then generates
with Transformers. Experiment 07 additionally uses `presence_penalty=0.5`
and `repetition_penalty=1.05`.

## Chosen architecture

Use an in-process vLLM `AsyncLLM` engine inside a dedicated vLLM Modal class.
The local runner submits a bounded number of requests concurrently to one
remote container. The engine schedules the active requests with continuous
batching; each request keeps its own prompt, sampling parameters, seed, and
output stream.

The existing synchronous Transformers class remains available as the legacy
Modal backend during validation. Experiment 08 selects the vLLM backend
explicitly. A later change may make vLLM the default only after validation.

The vLLM image will follow Modal's supported throughput pattern and pin
`vllm==0.17.0` in a CUDA-compatible Python 3.13 image. This pin is unverified
until the Task 5 Modal smoke test loads the exact `Qwen/Qwen3.5-4B` model path;
the smoke test must revise the pin if that load fails. Hugging Face and vLLM
cache volumes remain mounted so model and compilation artifacts survive
container replacement. The exact GPU remains a runner option and is held
constant for the 07/08 comparison.

## Data flow

1. The runner loads the same evaluation JSONL, adapter run, seed, system
   prompt, and generation configuration used by Experiment 07.
2. The vLLM generator formats each prompt with the same tokenizer chat
   template and `enable_thinking=False` setting used by the current path.
3. The runner submits up to a bounded in-flight request count to the Modal
   vLLM method. Each request receives a stable evaluation index and request
   ID.
4. `AsyncLLM.generate` is consumed until its final output. The response is
   normalized to the existing result fields: `response`, `input_tokens`,
   `output_tokens`, `generation_seconds`, and `truncated`.
5. Completed responses are buffered by evaluation index. The writer appends
   only the next contiguous result in evaluation order, preserving the
   existing JSONL prefix contract and making an interrupted run resumable.
6. The existing scoring stage reads the same `predictions.jsonl` and writes
   the same `rule_scores.json` format.

Out-of-order completion is allowed inside the engine but never changes the
on-disk ordering. The Modal class is decorated with
`@modal.concurrent(max_inputs=16, target_inputs=16)` for the initial trial.
The limit is configurable and bounded to avoid GPU out-of-memory errors; it
is tuned using measurements.

The runner uses `remote.generate.remote.aio` from an asyncio coordinator and
keeps a bounded set of in-flight calls. The existing synchronous runner
remains the public entry point; it invokes the coordinator with `asyncio.run`
only for the vLLM backend.

## Model and adapter handling

The vLLM class loads the base model directly through vLLM for base-model
experiments. For the Qwen3.5 SFT experiment, the adapter path is validated with
the existing `training_adapter_path` and `validate_adapter_config` checks, then
merged into a cached model artifact before the engine starts. vLLM 0.17's
Qwen3.5 fused-projection LoRA path can fail during engine warm-up, so native
`LoRARequest` loading is intentionally disabled for this validation path.

The current path's effective LoRA scale is `0.25`. PEFT applies that scale,
merges the adapter, and writes the result to a temporary sibling before an
atomic install. A source-manifest marker validates reuse; stale or incomplete
artifacts are rebuilt. The engine loads the merged model with
`language_model_only=True`, preserving the text-only workload while retaining
vLLM continuous batching.

## Sampling and stop behavior

Use vLLM's native `SamplingParams` for the existing generation settings:

- `temperature=0.7`
- `top_p=0.8`
- `top_k=20`
- `max_tokens=2048`
- per-request `seed=42`
- both model EOS and tokenizer EOS as `stop_token_ids`
- `skip_special_tokens=True`
- `presence_penalty` and `repetition_penalty` when configured

Presence penalty is native and matches the current intent: penalize tokens
already generated in the response. vLLM's native repetition penalty includes
tokens from the prompt as well as generated tokens, while the current
Transformers processor ignores prompt tokens. Experiment 08 will use the
native vLLM penalty first and record this semantic difference in its config.
If the score or validity comparison shows a meaningful regression, add a
version-pinned custom vLLM logits processor that reproduces the current
generated-only repetition behavior. No custom processor is part of the first
path.

## Modal lifecycle and failure handling

- `@modal.enter` constructs the async engine, tokenizer, metadata, and a
  warm-up request. Model load and adapter-load failures fail the container
  before any predictions are written.
- The remote generation method is async and must not block the event loop.
  It returns one normalized result per request and includes a stable request
  ID in diagnostic logs.
- `@modal.exit` releases the engine cleanly.
- A request-level generation failure is converted to the existing result
  shape with an `error` field, allowing later prompts to finish and preserving
  the current partial-run behavior.
- A container-level failure leaves only the already flushed contiguous JSONL
  prefix. `--resume` reruns the remaining suffix against the same validated
  configuration.
- The runner rejects a resume when backend, model, adapter, evaluation hash,
  seed, generation settings, or prompt count differ, as it does today.

## Validation plan

Experiment 08 is a vLLM counterpart to Experiment 07. Run both with the same
adapter run, `--limit`/evaluation set, GPU, seed, and generation settings.
Record:

- total wall time and wall-clock output tokens/second;
- model-load and warm-up time separately from generation time;
- successful, failed, and truncated counts;
- validity rate and every rule-score mean;
- response and token-count samples for manual inspection;
- vLLM version, GPU, concurrency, adapter identity/scale, and sampling
  parameters in `config.json`.

The vLLM path is accepted only when it is faster than Experiment 07 and shows
no meaningful quality regression: no new systematic failures, no lower
validity rate, and no rule-score mean decrease greater than `0.02` absolute.
Because sampling is stochastic, repeat the comparison on the same fixed
evaluation set if a metric is close to the threshold.

## Scope and non-goals

In scope:

- one new vLLM Modal backend;
- Experiment 08;
- bounded async submission and ordered persistence;
- tests for request normalization, ordering, resume behavior, sampling
  translation, adapter scaling, and failure isolation;
- README documentation and benchmark comparison instructions.

Out of scope:

- a public HTTP/OpenAI-compatible service;
- changes to existing Experiment 03–07 code or historical run artifacts;
- distributed tensor parallelism or multi-GPU serving;
- streaming partial tokens to the local terminal;
- a custom repetition processor before the native-penalty comparison has
  evidence that it is needed.

## References

- [Modal input concurrency](https://modal.com/docs/guide/concurrent-inputs)
- [Modal vLLM throughput example](https://modal.com/docs/examples/vllm_throughput)
- [vLLM offline inference and continuous batching](https://docs.vllm.ai/en/latest/serving/offline_inference/)
- [vLLM SamplingParams](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/)
- [vLLM LoRA adapters](https://docs.vllm.ai/en/stable/features/lora/)
- [vLLM custom logits processors](https://docs.vllm.ai/en/stable/features/custom_logitsprocs/)
