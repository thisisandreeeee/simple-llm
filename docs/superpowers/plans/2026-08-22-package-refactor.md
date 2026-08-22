# Package Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Group scoring, inference, and SFT code into cohesive packages, rename the orchestration module to `experiment_runner.py`, and preserve the existing public behavior.

**Architecture:** `simple_llm.scoring` owns rule and judge scoring workflows; `simple_llm.inference` owns shared generation plus local and Modal backends; `simple_llm.sft` owns prompt generation, answer generation, dataset construction, training, and its teacher-model runtime. `experiment_runner.py` remains the application-level composition root because it coordinates inference and scoring.

**Tech Stack:** Python 3.11+, pytest, uv, Transformers, Modal, DeepEval, Pydantic.

**Spec:** Approved in conversation.

## Global Constraints

- Preserve existing callable behavior and artifact formats.
- Update all imports, module execution paths, tests, and README examples.
- Keep the package public API minimal and avoid speculative utility packages.
- Verify with `uv run pytest -q`.

---

### Task 1: Add the scoring package workflows

**Files:**
- Create: `simple_llm/scoring/rule_scoring.py` (move existing rule artifact workflow)
- Create: `simple_llm/scoring/judge_scoring.py` (move existing judge workflow)
- Modify: `simple_llm/scoring/__init__.py` (retain core exports only)
- Modify: `simple_llm/experiment_runner.py` (import scoring workflow from package)
- Modify: `tests/test_inference.py` and `tests/test_judge_scoring.py` (new import paths)

**Interfaces:**
- Preserve `score_predictions`, `summarize_scores`, `judge_predictions`, and the judge CLI.
- Preserve `simple_llm.scoring` exports for `RuleScores`, `score`, `RULE_SCORERS`, and `validate_answer`.

- [ ] Move the two scoring workflow modules and update imports.
- [ ] Update tests and README judge command.
- [ ] Run `uv run pytest -q` and commit with `refactor: group scoring workflows under scoring package`.

### Task 2: Add inference and SFT packages

**Files:**
- Create: `simple_llm/inference/__init__.py`, `simple_llm/inference/local.py`, `simple_llm/inference/modal.py`
- Create: `simple_llm/sft/__init__.py`, `simple_llm/sft/prompts.py`, `simple_llm/sft/answers.py`, `simple_llm/sft/dataset.py`, `simple_llm/sft/training.py`, `simple_llm/sft/runtime.py`
- Remove: old top-level `inference.py`, `modal_inference.py`, `sft_*.py`, `llm_runtime.py` after migration
- Modify: all tests and internal imports referencing the old paths

**Interfaces:**
- Preserve inference exports used by tests and experiment orchestration.
- Preserve all SFT functions, Pydantic models, CLI entry points, and data formats.
- Keep runtime helpers private to SFT unless a future consumer demonstrates shared use.

- [ ] Move modules with minimal content changes and fix relative imports.
- [ ] Update tests and all package references.
- [ ] Run `uv run pytest -q` and commit with `refactor: group inference and sft code into packages`.

### Task 3: Rename the orchestration module and documentation

**Files:**
- Rename: `simple_llm/experiment.py` → `simple_llm/experiment_runner.py`
- Modify: `experiments/*.py`, `tests/test_inference.py`, and `README.md`

**Interfaces:**
- Preserve `run_experiment`, `load_evals`, `git_info`, and `summarize_inference`.
- Keep concrete experiment scripts runnable unchanged apart from their import path.

- [ ] Update all imports and README module commands.
- [ ] Run `uv run pytest -q` plus import/CLI smoke checks.
- [ ] Commit with `refactor: rename experiment module to experiment runner`.

### Task 4: Final verification

- [ ] Run `uv run pytest -q`.
- [ ] Run `python -m compileall simple_llm experiments`.
- [ ] Inspect `git diff --check` and `git status`.
- [ ] Report branch and commit hashes.
