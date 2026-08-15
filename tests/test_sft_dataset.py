from collections import Counter
import asyncio

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
    assert sorted(Counter(item.subject for item in first).values()) == [120] * 10 + [180] * 10
    assert Counter(item.audience for item in first) == {
        "beginner": 900,
        "practitioner": 1_200,
        "expert": 900,
    }
    assert Counter(item.expected_length for item in first) == {
        "short": 900,
        "medium": 1_200,
        "long": 900,
    }
    assert Counter(item.intent for item in first) == {
        "explanation": 750,
        "documentation": 600,
        "troubleshooting": 450,
        "procedure": 450,
        "comparison": 450,
        "misconception correction": 300,
    }


def test_strata_assign_valid_topics_and_audiences() -> None:
    for item in build_strata(100):
        assert item.topic in sft_dataset.TOPICS[item.subject]
        assert item.audience in sft_dataset.AUDIENCE_GUIDANCE


def test_subject_mix_is_sixty_percent_technical() -> None:
    specs = build_strata(1_000)
    technical = sum(item.subject in sft_dataset.TECHNICAL_SUBJECTS for item in specs)
    assert technical == 600


def test_prompt_instruction_uses_audience_guidance() -> None:
    instruction = sft_dataset._prompt_instruction(build_strata(1))
    assert "audience_guidance" in instruction
    assert '"audience":' not in instruction
    assert "Prefer stable facts and concepts" in instruction


def test_topic_catalog_has_broad_coverage() -> None:
    assert len(sft_dataset.TOPICS) == 20
    assert sum(len(topics) for topics in sft_dataset.TOPICS.values()) == 400
    assert {len(topics) for topics in sft_dataset.TOPICS.values()} == {20}


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


def test_prompt_issues_reject_missing_source_material() -> None:
    assert sft_dataset._prompt_issues(
        "Explain how DNS resolution works for a junior engineer, including caching and recursion."
    ) == []
    assert sft_dataset._prompt_issues("Rewrite the attached excerpt in plain language.")


def test_prompt_generation_retries_timeouts(monkeypatch) -> None:
    spec = build_strata(1)[0]

    class TimeoutThenSuccessModel:
        calls = 0

        async def a_generate(self, prompt, schema):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError
            return {
                "prompts": [
                    {
                        "id": spec.id,
                        "prompt": "Explain this topic clearly with practical details and relevant trade-offs.",
                    }
                ]
            }, 0

    async def no_sleep(_delay):
        pass

    model = TimeoutThenSuccessModel()
    monkeypatch.setattr(sft_dataset.asyncio, "sleep", no_sleep)

    result = asyncio.run(sft_dataset._generate_prompt_batch(model, [spec], [], 1))

    assert result[0].id == spec.id
    assert model.calls == 2


def test_legacy_prompt_rows_are_migrated_to_minimal_shape(tmp_path) -> None:
    path = tmp_path / "sft_prompts.jsonl"
    path.write_text(
        '{"id":"SFT-NET-0001","subject":"networking and internet",'
        '"difficulty":"easy","prompt":"Explain a pump."}\n',
        encoding="utf-8",
    )

    records = sft_dataset._load_prompt_records(path)

    assert records[0].model_dump() == {
        "id": "SFT-NET-0001",
        "prompt": "Explain a pump.",
    }
    assert path.read_text(encoding="utf-8") == (
        '{"id":"SFT-NET-0001","prompt":"Explain a pump."}\n'
    )
