import json
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
                                        "asd_ste100": 0.6,
                                    },
                                    "1": {
                                        "correctness": 0.5,
                                        "simplicity": 1.0,
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

    result = rewards.reward_func(["Question 1", "Question 2"], ["Answer 1", "Answer 2"])

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
