"""Run the Qwen3-0.6B benchmark without an added system prompt."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_llm.inference import run_experiment

if __name__ == "__main__":
    run_experiment(
        experiment="01_qwen3_06b_base",
        model="Qwen/Qwen3-0.6B",
        condition="base_raw",
        description=__doc__,
    )
