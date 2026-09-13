"""Run Experiment 07's SFT penalty setup at full adapter scale."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_llm.experiment_runner import run_experiment

if __name__ == "__main__":
    run_experiment(
        experiment="09_qwen35_4b_sft_combined_penalties_scale_1",
        model="Qwen/Qwen3.5-4B",
        condition="sft_lora_merged_combined_penalties_scale_1",
        default_backend="modal",
        description=__doc__,
        require_adapter_run=True,
        default_adapter_scale=1.0,
        presence_penalty=0.5,
        repetition_penalty=1.05,
    )
