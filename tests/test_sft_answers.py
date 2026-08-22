import asyncio
import json

import pytest
from pydantic import ValidationError

import simple_llm.sft.answers as sft_answers
from simple_llm.sft.answers import (
    AnswerRecord,
    CORRECTNESS_PROMPT,
    GeneratedAnswer,
    QUALITY_PROMPT,
    SIMPLE_ENGLISH_PROMPT,
    answer_instruction,
    generate_answer,
    generate_answers,
    load_answer_records,
    load_prompts,
)
from simple_llm.sft.prompts import PromptRecord


def test_answer_record_contains_only_the_final_target() -> None:
    record = AnswerRecord(
        id="SFT-NET-0001", final_response="Resolvers cache DNS records."
    )

    assert record.model_dump() == {
        "id": "SFT-NET-0001",
        "final_response": "Resolvers cache DNS records.",
    }


def test_answer_models_reject_invalid_shape() -> None:
    with pytest.raises(ValidationError):
        GeneratedAnswer(final_response="", extra="no")

    with pytest.raises(ValidationError):
        AnswerRecord(id="bad", final_response="An answer.")


def test_load_prompts_reads_minimal_prompt_rows(tmp_path) -> None:
    path = tmp_path / "sft_prompts.jsonl"
    path.write_text(
        '{"id":"SFT-NET-0001","prompt":"Explain DNS caching clearly."}\n',
        encoding="utf-8",
    )

    assert load_prompts(path) == [
        PromptRecord(id="SFT-NET-0001", prompt="Explain DNS caching clearly.")
    ]


def test_answer_instruction_contains_simple_english_prompt_and_request() -> None:
    prompt = PromptRecord(id="SFT-NET-0001", prompt="Explain DNS caching clearly.")

    instruction = answer_instruction(prompt)

    assert SIMPLE_ENGLISH_PROMPT in instruction
    assert CORRECTNESS_PROMPT in instruction
    assert QUALITY_PROMPT in instruction
    assert prompt.prompt in instruction
    assert "Answer the user request directly" in instruction
    assert "final_response" in instruction


def test_quality_prompt_disallows_unrequested_headings() -> None:
    assert (
        "Do not use a title, heading, or section label unless the user explicitly "
        "requests one."
    ) in QUALITY_PROMPT
    assert "Begin directly with the answer." in QUALITY_PROMPT


def test_generate_answer_uses_one_call() -> None:
    prompt = PromptRecord(id="SFT-NET-0001", prompt="Explain DNS caching clearly.")

    class Model:
        calls = 0

        async def a_generate(self, instruction, schema):
            self.calls += 1
            assert schema is GeneratedAnswer
            assert prompt.prompt in instruction
            return {"final_response": "A resolver can cache DNS records."}, 0

    model = Model()
    result = asyncio.run(generate_answer(model, prompt))

    assert model.calls == 1
    assert result == AnswerRecord(
        id=prompt.id, final_response="A resolver can cache DNS records."
    )


def test_load_answer_records_ignores_old_intermediate_fields(tmp_path) -> None:
    path = tmp_path / "sft_answers.jsonl"
    path.write_text(
        '{"id":"SFT-NET-0001","key_points":["A fact"],'
        '"final_response":"An answer."}\n',
        encoding="utf-8",
    )

    assert load_answer_records(path) == [
        AnswerRecord(id="SFT-NET-0001", final_response="An answer.")
    ]


def test_generate_answers_writes_two_field_rows_with_max_reasoning(
    tmp_path, monkeypatch
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"id":"SFT-NET-0001","prompt":"Explain DNS caching clearly."}\n',
        encoding="utf-8",
    )
    output = tmp_path / "answers.jsonl"

    class Model:
        async def a_generate(self, instruction, schema):
            assert schema is GeneratedAnswer
            return {"final_response": "An answer."}, 0

    model_configs: list[dict] = []

    def create_model(**kwargs):
        model_configs.append(kwargs)
        return Model()

    monkeypatch.setattr(sft_answers, "create_deepseek_model", create_model)
    generate_answers(count=1, prompts=prompts, output=output)

    assert model_configs == [
        {
            "temperature": 0.0,
            "generation_kwargs": {
                "reasoning_effort": "max",
                "extra_body": {"thinking": {"type": "enabled"}},
            },
        }
    ]
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "id": "SFT-NET-0001",
        "final_response": "An answer.",
    }


def test_generate_answers_resumes_by_prompt_id(tmp_path, monkeypatch) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "".join(
            [
                '{"id":"SFT-NET-0001","prompt":"Explain DNS caching."}\n',
                '{"id":"SFT-NET-0002","prompt":"Explain DNS TTL values."}\n',
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "answers.jsonl"
    output.write_text(
        '{"id":"SFT-NET-0001","key_points":["Existing fact"],'
        '"final_response":"Existing answer."}\n',
        encoding="utf-8",
    )
    calls: list[str] = []

    class Model:
        async def a_generate(self, instruction, schema):
            calls.append(instruction)
            assert schema is GeneratedAnswer
            return {"final_response": "New answer."}, 0

    monkeypatch.setattr(sft_answers, "create_deepseek_model", lambda **kwargs: Model())

    records = generate_answers(count=2, prompts=prompts, output=output)

    assert [record.id for record in records] == ["SFT-NET-0001", "SFT-NET-0002"]
    assert len(calls) == 1
    assert [json.loads(line)["id"] for line in output.read_text().splitlines()] == [
        "SFT-NET-0001",
        "SFT-NET-0002",
    ]
