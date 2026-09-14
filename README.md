# simple-llm

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Hugging Face model](https://img.shields.io/badge/%F0%9F%A4%97_Model-simple--llm--lora-yellow)](https://huggingface.co/thisisandreeeee/simple-llm-lora)
[![Hugging Face dataset](https://img.shields.io/badge/%F0%9F%A4%97_Dataset-simple--llm--sft-yellow)](https://huggingface.co/datasets/thisisandreeeee/simple-llm-sft)

**Can we post-train a small language model to produce technical answers that are both correct and simple?**

`simple-llm` is an end-to-end experiment built around that question.

Starting from Qwen3.5-4B, the project compares:

- the base model,
- prompt engineering for simpler answers,
- supervised fine-tuning (SFT) with LoRA, and
- Group Relative Policy Optimization (GRPO) with AI feedback.

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
Continue training with GRPO
  │
  ▼
Run controlled experiments
  │
  ├── Base model
  ├── Prompt engineered
  ├── SFT
  └── SFT + GRPO
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

Prompt engineering produced the shortest answers, but at a clear cost to quality. SFT improved simplicity while largely preserving correctness. GRPO improved the overall balance further: it produced much simpler and shorter answers than SFT, with the highest technical adequacy in this comparison.

| Metric                    |       Base |  Prompted |        SFT |       GRPO |
| ------------------------- | ---------: | --------: | ---------: | ---------: |
| **Semantic simplicity ↑** |     65.82% |    74.75% |     85.46% | **95.83%** |
| **Technical adequacy ↑**  |     68.30% |    59.18% |     70.15% | **72.42%** |
| **Task fulfilment ↑**     | **97.96%** |    87.75% |     95.71% |     94.33% |
| **Clarity & coherence ↑** | **92.09%** |    85.25% | **92.09%** |     89.18% |
| Average sentence length ↓ |      17.93 | **11.15** |      15.32 |      11.80 |
| Long-sentence fraction ↓  |     25.44% |      4.22% |     15.35% |  **1.88%** |
| Mean output tokens ↓      |    1,245.8 |  **279.0** |      622.3 |      331.2 |

The SFT column is Run 07: adapter scale `0.25`, presence penalty `0.5`, and repetition penalty `1.05`. The GRPO column is Run 08: the same penalties with the adapter at full scale (`1.0`). Both use the same 100 evaluation prompts and decoding settings.

## Setup

Install [uv](https://docs.astral.sh/uv/), then install the locked dependencies:

```bash
uv sync
```

Use `uv run` for the commands below, or activate the environment with
`source .venv/bin/activate`.

### DeepSeek

DeepSeek generates SFT answers, scores GRPO training completions, and judges
experiment results. Copy the example environment file, add your API key, and
load it into the current shell:

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

### 3. Train with GRPO from the SFT adapter

Start GRPO from the completed `qwen35-4b-sft-20260821-025306` SFT adapter:

```bash
uv run python -m simple_llm.grpo.training \
  --adapter-run qwen35-4b-sft-20260821-025306 \
  --detach
```

`--adapter-run` is required. The job loads the adapter from the shared
`simple-llm-training` Modal Volume and continues training its LoRA parameters;
it does not merge the adapter into the base model first. Use
`--run-name` to name the GRPO output run or `--max-steps 1` for a smoke test.

### 4. Run experiments

Run the main 4B baselines on Modal:

```bash
uv run python experiments/03_qwen35_4b_base.py
uv run python experiments/04_qwen35_4b_sysprompt.py
```

Evaluate a completed training run's LoRA adapter (scaled to `0.25` by default):

```bash
uv run python experiments/05_qwen35_4b_sft.py --adapter-run RUN
```

Pass `--adapter-scale 1.0` to evaluate the adapter at full strength.

Experiments 06, 07, and 09 test penalties that reduce repetitive SFT output:

```bash
uv run python experiments/06_qwen35_4b_sft_presence_penalty.py --adapter-run RUN
uv run python experiments/07_qwen35_4b_sft_combined_penalties.py --adapter-run RUN

# Experiment 09: the same penalties at full adapter strength
uv run python experiments/09_qwen35_4b_sft_combined_penalties_scale_1.py --adapter-run RUN
```

Evaluate a completed GRPO adapter at full strength with the same combined
penalties:

```bash
uv run python experiments/08_qwen35_4b_grpo.py \
  --adapter-run RUN \
  --adapter-scale 1.0
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

### 5. Judge a run

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

- [ ] Add DPO data generation and training
- [x] Implement RLAIF with GRPO
- [ ] Serve inference with vLLM

## License

The repository code is licensed under [Apache-2.0](LICENSE). The model adapter
and dataset are separate artifacts with licenses declared on their Hugging Face
pages.
