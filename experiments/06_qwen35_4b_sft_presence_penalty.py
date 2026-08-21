"""Run the merged Qwen3.5-4B SFT LoRA adapter with a presence penalty."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_llm.experiment import run_experiment

if __name__ == "__main__":
    run_experiment(
        experiment="06_qwen35_4b_sft_presence_penalty",
        model="Qwen/Qwen3.5-4B",
        condition="sft_lora_merged_presence_penalty",
        default_backend="modal",
        description=__doc__,
        require_adapter_run=True,
        presence_penalty=1.5,
    )
