"""Run the Qwen3-0.6B base-model benchmark."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simple_llm.evaluation import run_experiment


if __name__ == "__main__":
    run_experiment(
        experiment="01_qwen3_06b_base",
        condition="base_raw",
        description=__doc__,
    )
