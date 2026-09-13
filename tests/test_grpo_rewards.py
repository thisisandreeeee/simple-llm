import json
from itertools import combinations, product
from types import SimpleNamespace

import pytest

from simple_llm.grpo import rewards


def test_reward_func_batches_completions_in_one_judge_call(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "0": {
                                        "correctness": 1.0,
                                        "simplicity": 0.8,
                                        "clarity": 0.9,
                                        "asd_ste100": 0.6,
                                    },
                                    "1": {
                                        "correctness": 0.5,
                                        "simplicity": 1.0,
                                        "clarity": 0.8,
                                        "asd_ste100": 1.0,
                                    },
                                }
                            )
                        )
                    )
                ]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(rewards, "create_deepseek_client", lambda: client)

    result = rewards.reward_func(["Question", "Question"], ["Answer 1", "Answer 2"])

    assert result == pytest.approx([0.91, 0.5])
    assert len(calls) == 1
    assert "ID 0:" in calls[0]["messages"][1]["content"]
    assert "ID 1:" in calls[0]["messages"][1]["content"]


def test_reward_func_rejects_mismatched_batch_lengths():
    try:
        rewards.reward_func(["Question"], [])
    except ValueError as error:
        assert str(error) == "prompts and completions must have equal lengths"
    else:
        raise AssertionError("expected ValueError")


def test_relative_reward_is_correctness_first_over_the_judge_grid():
    grades = sorted(rewards.CORRECTNESS_GRADES)
    styles = [0.0, 0.25, 0.5, 0.75, 1.0]

    for lower, higher in combinations(grades, 2):
        for lower_s, lower_a, higher_s, higher_a in product(styles, repeat=4):
            result = rewards.relative_rewards(
                ["prompt", "prompt"],
                [
                    {
                        "correctness": lower,
                        "simplicity": lower_s,
                        "asd_ste100": lower_a,
                    },
                    {
                        "correctness": higher,
                        "simplicity": higher_s,
                        "asd_ste100": higher_a,
                    },
                ],
            )
            assert result[0] < result[1]


def test_combined_reward_only_rewards_style_when_answer_has_value():
    incorrect = {"correctness": 0.0, "simplicity": 1.0, "asd_ste100": 1.0}
    complex_answer = {"correctness": 1.0, "simplicity": 0.0, "asd_ste100": 0.0}
    simple_answer = {"correctness": 1.0, "simplicity": 1.0, "asd_ste100": 1.0}

    result = rewards.relative_rewards(
        ["prompt"] * 3, [incorrect, complex_answer, simple_answer]
    )

    assert result == pytest.approx([0.0, 1 / 1.24, 1.0])


def test_style_signal_is_not_attenuated_at_low_correctness():
    weak = rewards.relative_rewards(
        ["weak", "weak"],
        [
            {"correctness": 0.25, "simplicity": 0.0, "asd_ste100": 0.0},
            {"correctness": 0.25, "simplicity": 1.0, "asd_ste100": 1.0},
        ],
    )
    strong = rewards.relative_rewards(
        ["strong", "strong"],
        [
            {"correctness": 1.0, "simplicity": 0.0, "asd_ste100": 0.0},
            {"correctness": 1.0, "simplicity": 1.0, "asd_ste100": 1.0},
        ],
    )

    assert weak[1] - weak[0] == pytest.approx(strong[1] - strong[0])
    assert weak[1] - weak[0] == pytest.approx(0.24 / 1.24)


def test_style_normalization_does_not_amplify_tiny_differences():
    result = rewards.relative_rewards(
        ["prompt", "prompt"],
        [
            {"correctness": 1.0, "simplicity": 0.8, "asd_ste100": 0.8},
            {"correctness": 1.0, "simplicity": 0.801, "asd_ste100": 0.8},
        ],
    )

    assert result[1] - result[0] < 0.001


def test_judge_retries_non_discrete_correctness_score():
    contents = iter(
        [
            '{"0":{"correctness":0.8,"simplicity":1,"clarity":1,"asd_ste100":1}}',
            '{"0":{"correctness":0.75,"simplicity":1,"clarity":1,"asd_ste100":1}}',
        ]
    )

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=next(contents))
                    )
                ]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    assert rewards.judge_scores(client, "message", 1)["0"]["correctness"] == 0.75
