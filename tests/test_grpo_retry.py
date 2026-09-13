import json
from types import SimpleNamespace

import pytest

from simple_llm.grpo.rewards import judge_scores


VALID_SCORES = {
    "0": {"correctness": 1, "simplicity": 0.8, "asd_ste100": 0.9}
}


@pytest.mark.parametrize(
    "invalid_content",
    [
        "{invalid",
        json.dumps({}),
        json.dumps({"0": {"correctness": 1, "simplicity": 0.8}}),
        json.dumps(
            {"0": {"correctness": 1, "simplicity": 0.8, "asd_ste100": 1.1}}
        ),
        json.dumps(
            {"0": {"correctness": True, "simplicity": 0.8, "asd_ste100": 0.9}}
        ),
    ],
)
def test_judge_scores_retries_invalid_response(invalid_content):
    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=invalid_content),
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=json.dumps(VALID_SCORES)),
                )
            ]
        ),
    ]
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return responses.pop(0)

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    assert judge_scores(client, "batch", 1) == VALID_SCORES
    assert len(calls) == 2
