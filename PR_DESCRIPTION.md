## Summary

- Add an optional token presence penalty to local and Modal inference.
- Add Experiment 06 with a presence penalty of 1.5.
- Document Experiment 05 performance and the progression of looping mitigations.

## Results

- Experiment 06 completed 100 prompts with no truncations or severe phrase loops.
- Mean output length fell 5.2% and generation time fell 5.9% from Experiment 05.
- Judge scores were broadly unchanged, while several mechanical Simple English scores declined.

## Test plan

- `uv run pytest` — 75 passed, 6 expected failures.
