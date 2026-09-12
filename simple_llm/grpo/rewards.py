"""Minimal, deterministic placeholder rewards for GRPO."""

import json
import os

from openai import OpenAI

MODEL = "deepseek-v4-pro"
SIMPLICITY_COEFF = 0.15
ASD_COEFF = 0.15
STYLE_BONUS = 0.24
MIN_STYLE_SPREAD = 0.25
JUDGE_RETRIES = 1
SCORE_FIELDS = {"correctness", "simplicity", "asd_ste100"}
CORRECTNESS_GRADES = {0.0, 0.25, 0.5, 0.75, 1.0}
JUDGE_PROMPT = """You are an evaluator. Judge whether the assistant response is **correct**, **simple**, and compliant with **ASD-STE100-style technical English** for the user request. You will be given a list of (user, assistant) pairs to be scored while preserving the identifiers.

## 1. Correctness

### Principles

- Be factually correct and technically adequate.
- Fulfill the user's requested task, scope, audience, and format.
- Preserve necessary caveats, constraints, and safety conditions.
- Be grounded: do not accept invented tools, commands, APIs, product names, architectures, or technical facts.
- Preserve literal technical content such as code, identifiers, CLI commands, file paths, quoted errors, and product names.

### Rubric

Score correctness from 0.0 to 1.0:

- 0.0: materially wrong, irrelevant, fabricated, or fails the task.
- 0.25: mostly wrong, but contains a small amount of useful correct content.
- 0.5: partially correct or useful, but has meaningful errors, omissions, or task-fulfillment issues.
- 0.75: mostly correct and useful, with only minor errors or omissions.
- 1.0: fully correct, technically adequate, grounded, and fulfills the requested task, scope, audience, and format.

## 2. Simplicity

### Principles

- Organize ideas clearly and coherently.
- Be concise without omitting necessary information.
- Avoid unnecessary explanation, repetition, and verbosity.
- Match the amount of detail to the task.

### Rubric

Score simplicity from 0.0 to 1.0:

- 0.0: difficult to follow, poorly organized, or excessively verbose.
- 0.5: understandable, but has noticeable verbosity, repetition, or structural complexity.
- 1.0: easy to follow, concise, well organized, and appropriately detailed.

## 3. ASD-STE100 Compliance

### Principles

Judge whether the response follows the main writing principles of ASD-STE100 Simplified Technical English:

- Use simple and commonly understood words where possible.
- Use one word consistently for one meaning.
- Prefer short, direct sentence structures.
- Prefer active voice when it improves clarity.
- Avoid idioms, figurative language, unnecessary jargon, and complex noun phrases.
- Make instructions explicit and unambiguous.
- Keep references between sentences clear.
- Preserve necessary technical terms and literal technical content.

### Rubric

Score ASD-STE100 compliance from 0.0 to 1.0:

- 0.0: frequently violates these controlled-language principles.
- 0.5: generally follows them, but contains several noticeable violations.
- 1.0: consistently uses clear, controlled technical English with no meaningful violations.

## Output

Return only valid JSON in this format:

```json
{
    "0": {
        "correctness": 0.5,
        "simplicity": 0.8,
        "asd_ste100": 0.3
    },
    "1": {
        "correctness": 0.75,
        "simplicity": 0.7,
        "asd_ste100": 0.2
    },
    ...
}
```

Correctness must be exactly 0.0, 0.25, 0.5, 0.75, or 1.0. Simplicity and
ASD-STE100 scores may take any value from 0.0 to 1.0.
"""


def create_deepseek_client():
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        max_retries=2,
    )
    return client


def judge_scores(client, user_message, expected_count):
    """Request and validate one judge result, retrying malformed responses once."""

    for attempt in range(JUDGE_RETRIES + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": user_message},
            ],
            stream=False,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=256,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content
        try:
            scores = json.loads(content)
            expected_ids = {str(i) for i in range(expected_count)}
            if not isinstance(scores, dict) or set(scores) != expected_ids:
                raise ValueError(f"expected score IDs {sorted(expected_ids)}")
            for score_id, score in scores.items():
                if not isinstance(score, dict) or set(score) != SCORE_FIELDS:
                    raise ValueError(f"invalid fields for score ID {score_id}")
                if any(
                    type(value) not in (int, float) or not 0 <= value <= 1
                    for value in score.values()
                ):
                    raise ValueError(f"invalid value for score ID {score_id}")
                if score["correctness"] not in CORRECTNESS_GRADES:
                    raise ValueError(f"invalid correctness grade for score ID {score_id}")
            return scores
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            if attempt == JUDGE_RETRIES:
                raise RuntimeError(
                    f"DeepSeek judge failed after {attempt + 1} attempts: "
                    f"{error}; content={content!r}"
                ) from error
            print(
                f"Retrying DeepSeek judge after invalid response "
                f"({attempt + 1}/{JUDGE_RETRIES}): {error}; content={content!r}"
            )


def reward_func(prompts, completions, **kwargs) -> list[float]:
    """Score completions with the original multiplicative reward."""

    scores = score_completions(prompts, completions)
    return multiplicative_rewards(prompts, scores)


def score_completions(prompts, completions) -> list[dict[str, float]]:
    """Return the judge's component scores for each completion."""

    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have equal lengths")

    client = create_deepseek_client()
    user_message = """Evaluate the assistant response against the user message for each of the following completions."""
    for i, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
        user_message += f"\n\nID {i}:\n<User message>{prompt}</User message>\n<Assistant response>{completion}</Assistant response>"

    scores = judge_scores(client, user_message, len(prompts))
    return [scores[str(i)] for i in range(len(prompts))]


def multiplicative_rewards(
    prompts: list[str], scores: list[dict[str, float]]
) -> list[float]:
    """Return the original correctness-multiplied reward."""

    if len(prompts) != len(scores):
        raise ValueError("prompts and scores must have equal lengths")
    alpha = 1.0 - SIMPLICITY_COEFF - ASD_COEFF
    return [
        score["correctness"]
        * (
            alpha
            + SIMPLICITY_COEFF * score["simplicity"]
            + ASD_COEFF * score["asd_ste100"]
        )
        for score in scores
    ]


def relative_rewards(
    prompts: list[str], scores: list[dict[str, float]]
) -> list[float]:
    """Give style its full range without letting it cross correctness grades."""

    if len(prompts) != len(scores):
        raise ValueError("prompts and scores must have equal lengths")

    groups: dict[str, list[int]] = {}
    for index, prompt in enumerate(prompts):
        groups.setdefault(prompt, []).append(index)

    normalized_styles = [0.0] * len(scores)
    for indices in groups.values():
        useful = [index for index in indices if scores[index]["correctness"] > 0]
        styles = [
            scores[index]["simplicity"] * scores[index]["asd_ste100"]
            for index in useful
        ]
        if not styles or max(styles) == min(styles):
            continue
        low = min(styles)
        spread = max(max(styles) - low, MIN_STYLE_SPREAD)
        for index, style in zip(useful, styles, strict=True):
            normalized_styles[index] = (style - low) / spread

    return [
        (
            (score["correctness"] + STYLE_BONUS * style) / (1.0 + STYLE_BONUS)
            if score["correctness"] > 0
            else 0.0
        )
        for score, style in zip(scores, normalized_styles, strict=True)
    ]
