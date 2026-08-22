"""Run the Qwen3.5-4B benchmark on Modal without an added system prompt."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_llm.experiment_runner import run_experiment

if __name__ == "__main__":
    run_experiment(
        experiment="03_qwen35_4b_base",
        model="Qwen/Qwen3.5-4B",
        condition="base_raw",
        default_backend="modal",
        description=__doc__,
    )
