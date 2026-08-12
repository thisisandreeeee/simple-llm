"""Run Qwen3.5-4B on Modal with the SimpleEnglish system prompt."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_llm.experiment import run_experiment

if __name__ == "__main__":
    prompt = ROOT / "prompts/simple_english.md"
    run_experiment(
        experiment="04_qwen35_4b_simple_english",
        model="Qwen/Qwen3.5-4B",
        condition="simple_english_system_prompt",
        system_prompt=prompt.read_text(encoding="utf-8").strip(),
        system_prompt_source="https://github.com/AminBlg/SimpleEnglish/blob/main/prompts/system-prompt.md",
        default_backend="modal",
        description=__doc__,
    )
