"""Run the Qwen3.5-0.8B benchmark without an added system prompt."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_llm.experiment_runner import run_experiment

if __name__ == "__main__":
    run_experiment(
        experiment="01_qwen35_08b_base",
        model="Qwen/Qwen3.5-0.8B",
        condition="base_raw",
        description=__doc__,
    )
