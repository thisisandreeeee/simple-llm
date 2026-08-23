# simple-llm

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Hugging Face model](https://img.shields.io/badge/%F0%9F%A4%97_Model-simple--llm--lora-yellow)](https://huggingface.co/thisisandreeeee/simple-llm-lora)
[![Hugging Face dataset](https://img.shields.io/badge/%F0%9F%A4%97_Dataset-simple--llm--sft-yellow)](https://huggingface.co/datasets/thisisandreeeee/simple-llm-sft)

**Can we post-train a small language model to produce technical answers that are both correct and simple?**

`simple-llm` is an end-to-end experiment built around that question.

Starting from Qwen3.5-4B, the project compares:

- the base model,
- prompt engineering for simpler answers, and
- supervised fine-tuning (SFT) with LoRA.

The target style is inspired by [ASD-STE100](https://www.asd-ste100.org/) Simplified Technical English: short sentences, common words, direct language, and one main idea at a time — without sacrificing technical correctness.

## How it works

```
Technical prompts
  │
  ▼
Generate simple answers
  │
  ▼
Build SFT dataset
  │
  ▼
LoRA fine-tune Qwen3.5-4B
  │
  ▼
Run controlled experiments
  │
  ├── Base model
  ├── Prompt engineered
  └── SFT
  │
  ▼
Evaluate
  ├── Deterministic language checks
  └── LLM-as-a-judge
```

## Target writing style

The model is trained toward technical writing that:

1. Uses sentences of about 20 words or fewer.
2. Prefers common words and active voice.
3. Expresses one main idea per sentence.
4. Avoids unnecessary qualifiers, hedging, and long introductions.
5. Preserves necessary technical terms and factual correctness.

## Experiment results

Prompt engineering simplified answers most aggressively, but at a clear cost to quality; SFT achieved the strongest balance, improving simplicity while largely preserving correctness.

| Metric                    |       Base |  Prompted |        SFT |
| ------------------------- | ---------: | --------: | ---------: |
| **Semantic simplicity ↑** |     65.82% |    74.75% | **79.21%** |
| **Technical adequacy ↑**  |     68.30% |    59.18% | **71.88%** |
| **Task fulfilment ↑**     | **97.96%** |    87.75% |     95.57% |
| **Clarity & coherence ↑** | **92.09%** |    85.25% |     90.63% |
| Average sentence length ↓ |      17.93 | **11.15** |      15.37 |
| Long-sentence fraction ↓  |     25.44% | **4.22%** |     15.76% |
| Mean output tokens ↓      |    1,245.8 | **279.0** |      679.2 |

## Setup

Install [uv](https://docs.astral.sh/uv/), then install the locked dependencies:

```bash
uv sync
```

Use `uv run` for the commands below, or activate the environment with
`source .venv/bin/activate`.

### DeepSeek

DeepSeek generates SFT answers and judges experiment results. Copy the example
environment file, add your API key, and load it into the current shell:

```bash
cp .env.example .env

# Add your credentials to .env.
set -a
source .env
set +a
```

`.env` is gitignored. Do not commit it.

### Modal and Hugging Face

Training and 4B inference run on [Modal](https://modal.com/). Authenticate the
CLI and create the Hugging Face secret expected by the training job:

```bash
uv run modal setup
uv run modal secret create huggingface HF_TOKEN=hf_your_token
```

## Workflow

### 1. Generate the SFT dataset

Generate a pool of candidate user prompts:

```bash
uv run python -m simple_llm.sft.prompts --count 3000
```

This writes `data/sft_prompts.jsonl`. Generation resumes from existing prompt
IDs. Use `--no-resume` to replace the output.

Generate answers for the first 500 prompts:

```bash
uv run python -m simple_llm.sft.answers --count 500
```

The prompt and answer counts are independent. The first command creates a
larger prompt pool; the second controls how many examples receive answers and
enter the dataset. Increase `--count` when you want a larger training set.
Rerunning answer generation resumes unfinished work by prompt ID.

Build the dataset:

```bash
uv run python -m simple_llm.sft.dataset
```

This creates a deterministic, subject-stratified 90/10 split in
`data/sft_train.jsonl` and `data/sft_eval.jsonl`.

### 2. Fine-tune on Modal

Run a one-step smoke test on an L4 before starting the full job:

```bash
uv run python -m simple_llm.sft.training --run-name sft-smoke --max-steps 1
```

Start the default two-epoch run in detached mode:

```bash
uv run python -m simple_llm.sft.training --detach
```

The job trains a bf16 LoRA adapter for `Qwen/Qwen3.5-4B` with Qwen3.5's
non-thinking chat format. It evaluates every 25 steps and restores the
checkpoint with the lowest evaluation loss. Use `--gpu A10` or `--gpu L40S`
to change the GPU and `--run-name` to name the run.

Artifacts are stored in the `simple-llm-training` Modal Volume. Model downloads
reuse `simple-llm-huggingface-cache`. Modal prints a temporary TensorBoard URL
while training is active.

To inspect TensorBoard after training, replace `RUN` with the run name:

```bash
uv run modal volume get simple-llm-training RUN/checkpoints/runs ./tensorboard-logs
uvx --from tensorboard tensorboard --logdir ./tensorboard-logs
```

Then open <http://localhost:6006>.

### 3. Run experiments

Run the main 4B baselines on Modal:

```bash
uv run python experiments/03_qwen35_4b_base.py
uv run python experiments/04_qwen35_4b_sysprompt.py
```

Evaluate a completed training run's LoRA adapter:

```bash
uv run python experiments/05_qwen35_4b_sft.py --adapter-run RUN
```

Experiments 06 and 07 test penalties that reduce repetitive SFT output:

```bash
uv run python experiments/06_qwen35_4b_sft_presence_penalty.py --adapter-run RUN
uv run python experiments/07_qwen35_4b_sft_combined_penalties.py --adapter-run RUN
```

The 4B experiments use an L4 by default. Pass `--gpu A10` or `--gpu L40S` to
compare hardware, or `--limit N` for a smaller run. Model weights are cached in
the `simple-llm-huggingface-cache` Modal Volume.

Each experiment writes predictions, configuration, summary, and rule-based
scores to a timestamped directory under `runs/`. Resume interrupted inference
without regenerating completed predictions:

```bash
uv run python experiments/05_qwen35_4b_sft.py --resume runs/05_qwen35_4b_sft-YYYYMMDD-HHMMSS-ffffff
```

Optional local 0.8B baselines are also available:

```bash
uv run python experiments/01_qwen35_08b_base.py
uv run python experiments/02_qwen35_08b_sysprompt.py
```

### 4. Judge a run

After an experiment completes, judge its predictions with up to 50 concurrent
requests:

```bash
uv run python -m simple_llm.scoring.judge_scoring \
  runs/RUN/predictions.jsonl runs/RUN/rule_scores.json \
  --model "$DEEPSEEK_MODEL_NAME" --concurrency 50 --retry-limit 2
```

The retry limit applies only when the judge returns invalid JSON. Other
failures are recorded immediately.

## Backlog

- Add DPO data generation and training
- Implement RLAIF with GRPO
- Serve inference with vLLM

## License

The repository code is licensed under [Apache-2.0](LICENSE). The model adapter
and dataset are separate artifacts with licenses declared on their Hugging Face
pages.
