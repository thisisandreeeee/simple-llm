"""Run Qwen3-0.6B with the SimpleEnglish skill system prompt."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simple_llm.evaluation import run_experiment

PROMPT = ROOT / "prompts/simple_english.md"
SOURCE = "https://github.com/AminBlg/SimpleEnglish/blob/main/prompts/system-prompt.md"


if __name__ == "__main__":
    run_experiment(
        experiment="02_qwen3_06b_simple_english",
        condition="simple_english_system_prompt",
        system_prompt=PROMPT.read_text(encoding="utf-8").strip(),
        system_prompt_source=SOURCE,
        description=__doc__,
    )
