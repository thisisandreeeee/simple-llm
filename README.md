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

Source files will be added in later phases.

## Experiments

Run the raw Qwen3-0.6B baseline:

```bash
uv run python experiments/01_qwen3_06b_base.py
```

Run Qwen3-0.6B with the [SimpleEnglish](https://github.com/AminBlg/SimpleEnglish) system prompt:

```bash
uv run python experiments/02_qwen3_06b_sysprompt.py
```

Add `--limit N` to either command for a smaller run.

Run Qwen3-4B on a Modal L4 GPU:

```bash
uv run modal setup
uv run python experiments/03_qwen3_4b_base.py --limit 2
uv run python experiments/03_qwen3_4b_base.py
uv run python experiments/04_qwen3_4b_sysprompt.py
```

Model weights are cached in the `simple-llm-huggingface-cache` Modal Volume.
Predictions, configuration, and summaries are still written locally to `runs/`.
Use `--gpu A10` or `--gpu L40S` to compare throughput and cost with L4.

## Backlog

Done:

- Define rules
- Build evaluation set of 100 prompts stored in data/evals.jsonl; stratified by subject, difficulty, expected response length, need for technical terminology, risk of oversimplification
- Build simple english scorer: avg/max sentence length, response length, Flesch reading ease, percentage of long sentences, passive-voice estimate, complex-word ratio
- Benchmark Qwen3-0.6B (base vs enhanced system prompt) on all 100 prompts and run scorer
- Run Qwen3-4B on Modal L4 GPU

Todo:

- Build LLM judge scorer (GEval)
- Run end to end benchmarks

Later:

- Implement post training with dataset of 1-3k prompts: SFT vs. DPO
- Evaluate post trained model vs base model using rule-based scorer, LLM-as-a-judge, and benchmarks (viol/100w, MMLU-Pro)
- Upload to huggingface: LoRA adapter, model card, training configuration, evaluation results, base-model attribution
- Implement RLAIF with GRPO
- Serve on vLLM
