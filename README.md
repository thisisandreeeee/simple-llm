# simple-llm

Fine-tune a small language model to write using [ASD-STE100 Simplified Technical English](https://www.asd-ste100.org/).

The first version will compare a base Qwen model with a supervised fine-tuned (SFT) model. Preference training comes only after the SFT pipeline and evaluation are working end to end.

The model should:

1. Prefer sentences shorter than about 20 words.
2. Prefer common words when they are accurate.
3. Prefer active voice.
4. Avoid unnecessary qualifiers and hedging.
5. Express one main idea per sentence.
6. Avoid long introductory clauses.
7. Preserve necessary technical terms.
8. Never trade factual correctness for simplicity.

## Setup

Install [uv](https://docs.astral.sh/uv/), then create the virtual environment and install the locked dependencies:

```bash
uv sync
```

`uv sync` creates `.venv` automatically. Activate it if you want to run tools without the `uv run` prefix:

```bash
source .venv/bin/activate
```

The initial training stack uses [TRL](https://huggingface.co/docs/trl/) with its PEFT extra for LoRA-based post-training. uv installs its supporting packages, including Transformers, Accelerate, Datasets, PyTorch, and PEFT, from the checked-in lockfile.

Create a [Weights & Biases](https://wandb.ai/) account and API key. Then copy the environment template and set your API key and W&B user or team:

```bash
cp .env.example .env
```

Load the variables before you run an experiment:

```bash
set -a
source .env
set +a
```

Runs are uploaded to the `simple-llm` W&B project. W&B also creates a local `wandb/` directory for temporary logs and upload staging. Git ignores this directory; you can delete it when no run is active because W&B is the source of truth for completed runs.

## Experiments

Run the raw Qwen3-0.6B baseline:

```bash
uv run python experiments/01_qwen3_06b_base.py
```

Run Qwen3-0.6B with the [SimpleEnglish](https://github.com/AminBlg/SimpleEnglish) system prompt:

```bash
uv run python experiments/02_qwen3_06b_simple_english.py
```

Add `--limit N` to either command for a smaller run.

## Backlog

Done:

- Define rules
- Build evaluation set of 100 prompts stored in data/evals.jsonl; stratified by subject, difficulty, expected response length, need for technical terminology, risk of oversimplification
- Build simple english scorer: avg/max sentence length, response length, Flesch reading ease, percentage of long sentences, passive-voice estimate, complex-word ratio
- Benchmark Qwen3-0.6B (base vs enhanced system prompt) on all 100 prompts and run scorer

Todo:

- Add W&B artifact management for model checkpoints
- Run Qwen3-4B on Modal L4 GPU
- Build LLM judge scorer

Later:

- Generate SFT training data of 1-3k prompts using teacher model: first call for correct answer, second call for simplification
- Implement SFT script
- Run local training smoke test: loss decreases, resume from checkpoint, e2e evaluation
- Run training on colab (or other GPU platform)
- Evaluate SFT vs base model using scorer (rule-based and LLM judge)
- Evaluate using benchmarks: viol/100w, MMLU-Pro
- Upload to huggingface: LoRA adapter, model card, training configuration, evaluation results, base-model attribution
- Add DPO
- Add GRPO
- Serve on vLLM
