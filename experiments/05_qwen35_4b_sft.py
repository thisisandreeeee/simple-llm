"""Run the merged Qwen3.5-4B SFT LoRA adapter benchmark on Modal."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_llm.experiment import run_experiment

if __name__ == "__main__":
    run_experiment(
        experiment="05_qwen35_4b_sft",
        model="Qwen/Qwen3.5-4B",
        condition="sft_lora_merged",
        default_backend="modal",
        description=__doc__,
        require_adapter_run=True,
    )
