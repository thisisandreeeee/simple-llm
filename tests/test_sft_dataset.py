from collections import Counter

import pytest
from pydantic import ValidationError

from simple_llm.sft_dataset import (
    AnswerArtifacts,
    PromptRecord,
    SFTExample,
    allocate_counts,
    build_strata,
    format_sft_example,
)
from simple_llm import sft_dataset


def test_strata_are_reproducible_and_match_3000_quotas() -> None:
    first = build_strata(3_000)
    second = build_strata(3_000)

    assert first == second
    assert len(first) == 3_000
    assert sorted(Counter(item.subject for item in first).values()) == [300] * 10
    assert Counter(item.difficulty for item in first) == {
        "easy": 900,
        "medium": 1_200,
        "hard": 900,
    }
    assert Counter(item.target_category for item in first) == {
        "concise rewrite": 1_200,
        "corrected answer": 900,
        "procedure": 450,
        "qualified comparison": 300,
        "direct answer": 150,
    }


def test_allocate_counts_preserves_total_for_small_counts() -> None:
    result = allocate_counts(7, {"a": 0.3, "b": 0.7})
    assert sum(result.values()) == 7
    assert result == {"a": 2, "b": 5}


def test_sft_format_keeps_only_user_and_assistant() -> None:
    prompt = PromptRecord(
        id=build_strata(1)[0].id,
        prompt="Explain a pump.",
    )
    answer = AnswerArtifacts(
        id=prompt.id,
        facts=["A pump moves liquid."],
        answer_type="definition",
        approximate_length="short",
        prose="A pump moves liquid.",
        final_response="A pump moves liquid.",
    )

    result = format_sft_example(prompt, answer)
    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert set(prompt.model_dump()) == {"id", "prompt"}


def test_sft_example_rejects_other_roles() -> None:
    with pytest.raises(ValidationError):
        SFTExample(
            messages=[
                {"role": "system", "content": "hidden"},
                {"role": "assistant", "content": "answer"},
            ]
        )


def test_duplicate_detection_catches_eval_reuse() -> None:
    assert sft_dataset._is_duplicate(" Explain a pump. ", ["Explain a pump."])
    assert not sft_dataset._is_duplicate("How do I troubleshoot a pump?", ["Explain a pump."])
