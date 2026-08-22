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

To run the DeepSeek judge, copy the example environment file, add your API key,
and export its variables into the current shell:

```bash
cp .env.example .env

# add your credentials to .env
set -a
source .env
set +a
```

`.env` is gitignored. Keep the API key there and do not commit it.

## Run LLM judge

Judge a completed run with up to 50 concurrent requests:

```bash
uv run python -m simple_llm.scoring.judge_scoring \
  runs/RUN/predictions.jsonl runs/RUN/rule_scores.json \
  --model "$DEEPSEEK_MODEL_NAME" --concurrency 50 --retry-limit 2
```

The retry limit applies only when the judge returns invalid JSON; other failures are preserved immediately.

The initial training stack uses [TRL](https://huggingface.co/docs/trl/) with its PEFT extra for LoRA-based post-training. uv installs its supporting packages, including Transformers, Accelerate, Datasets, PyTorch, and PEFT, from the checked-in lockfile.

## Generate the SFT dataset

Configure the DeepSeek credentials described above before running the generation commands. The pipeline has three stages:

1. Generate user prompts:

   ```bash
   uv run python -m simple_llm.sft_prompts --count 3000
   ```

   This writes prompts to `data/sft_prompts.jsonl`. Generation resumes from existing prompt IDs. Use `--no-resume` to regenerate the output.

2. Generate assistant answers:

   ```bash
   uv run python -m simple_llm.sft_answers --count 500
   ```

   Answers are generated for the first `--count` prompts and appended by ID. Rerunning the command resumes unfinished answers.

3. Build the SFT dataset:

   ```bash
   uv run python -m simple_llm.sft_dataset
   ```

   This writes a deterministic, subject-stratified 90/10 split to `data/sft_train.jsonl` and `data/sft_eval.jsonl`.

## Fine-tune with Unsloth on Modal

Run a one-step smoke test on an L4 before starting the full job:

```bash
uv run python simple_llm/sft_training.py --run-name sft-smoke --max-steps 1
```

Start the two-epoch run in detached mode so it continues after the terminal closes:

```bash
uv run python simple_llm/sft_training.py --detach
```

The script trains a bf16 LoRA adapter for Qwen/Qwen3.5-4B. Use `--gpu A10` or `--gpu L40S` to select a GPU and `--run-name` to name runs. Training uses Qwen3.5's non-thinking chat format, evaluates every 25 steps, and restores the checkpoint with the lowest evaluation loss.

Training artifacts are stored in the simple-llm-training Modal Volume, while model downloads reuse simple-llm-huggingface-cache. Modal prints a TensorBoard URL for monitoring the active run; the URL stops when the run ends.

To inspect TensorBoard afterward, replace `RUN` with the run name and download its persisted event files:

```bash
uv run modal volume get simple-llm-training RUN/checkpoints/runs ./tensorboard-logs
uvx --from tensorboard tensorboard --logdir ./tensorboard-logs
```

Open http://localhost:6006.

## Experiments

Run the raw Qwen3.5-0.8B baseline:

```bash
uv run python experiments/01_qwen35_08b_base.py
```

Run Qwen3.5-0.8B with the [SimpleEnglish](https://github.com/AminBlg/SimpleEnglish) system prompt:

```bash
uv run python experiments/02_qwen35_08b_sysprompt.py
```

Add `--limit N` to either command for a smaller run.

Run Qwen3.5-4B on a Modal L4 GPU:

```bash
uv run modal setup
uv run python experiments/03_qwen35_4b_base.py --limit 2
uv run python experiments/03_qwen35_4b_base.py
uv run python experiments/04_qwen35_4b_sysprompt.py
```

Model weights are cached in the `simple-llm-huggingface-cache` Modal Volume.
Use `--gpu A10` or `--gpu L40S` to compare throughput and cost with L4.

Run the SFT evaluation with a completed training run's LoRA adapter:

```bash
uv run python experiments/05_qwen35_4b_sft.py --adapter-run qwen35-4b-sft-YYYYMMDD-HHMMSS
```

The adapter is loaded from `simple-llm-training`, safely merged into
Qwen3.5-4B on Modal, then evaluated without a system prompt.

If inference is interrupted, resume the same run without regenerating completed
predictions:

```bash
uv run python experiments/05_qwen35_4b_sft.py --resume runs/05_qwen35_4b_sft-YYYYMMDD-HHMMSS-ffffff
```

## Backlog

Done:

- Define rules
- Build evaluation set of 100 prompts stored in data/evals.jsonl; stratified by subject, difficulty, expected response length, need for technical terminology, risk of oversimplification
- Build simple english scorer: avg/max sentence length, response length, Flesch reading ease, percentage of long sentences, passive-voice estimate, complex-word ratio
- Benchmark Qwen3.5-0.8B (base vs enhanced system prompt) on all 100 prompts and run scorer
- Run Qwen3.5-4B on Modal L4 GPU
- Build LLM judge scorer (GEval)
- Analyse LLM judge scores
- Create SFT dataset
- Implement SFT
- Analyse base model vs SFT performance

Todo:

- Serve on vLLM to speed up inference
- Run benchmarks (viol/100w, MMLU-Pro)
- Upload to huggingface: LoRA adapter, model card, training configuration, evaluation results, base-model attribution

Later:

- Create DPO dataset
- Implement DPO
- Implement RLAIF with GRPO
