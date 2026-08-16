import asyncio
import json

import pytest
from pydantic import ValidationError

import simple_llm.sft_answers as sft_answers
from simple_llm.sft_answers import (
    AnswerRecord,
    AnswerSpec,
    answer_spec_instruction,
    build_answer_record,
    generate_answer_spec,
    load_prompts,
)
from simple_llm.sft_prompts import PromptRecord


def make_spec() -> AnswerSpec:
    return AnswerSpec(
        id="SFT-NET-0001",
        required_facts=["A resolver can cache DNS records."],
    )


def test_answer_spec_accepts_intermediate_contract() -> None:
    spec = make_spec()

    assert spec.model_dump() == {
        "id": "SFT-NET-0001",
        "required_facts": ["A resolver can cache DNS records."],
    }


def test_answer_record_contains_only_the_final_target() -> None:
    record = build_answer_record(make_spec(), "Resolvers cache DNS records.")

    assert record.model_dump() == {
        "id": "SFT-NET-0001",
        "final_response": "Resolvers cache DNS records.",
    }


def test_answer_models_reject_invalid_shape() -> None:
    with pytest.raises(ValidationError):
        AnswerSpec(
            id="SFT-NET-0001",
            required_facts=[],
        )

    with pytest.raises(ValidationError):
        AnswerSpec(
            id="SFT-NET-0001",
            required_facts=["A fact"],
            target_length="short",
        )

    with pytest.raises(ValidationError):
        AnswerRecord(id="SFT-NET-0001", final_response="", extra="no")


def test_load_prompts_reads_minimal_prompt_rows(tmp_path) -> None:
    path = tmp_path / "sft_prompts.jsonl"
    path.write_text(
        '{"id":"SFT-NET-0001","prompt":"Explain DNS caching clearly."}\n',
        encoding="utf-8",
    )

    assert load_prompts(path) == [
        PromptRecord(id="SFT-NET-0001", prompt="Explain DNS caching clearly.")
    ]


def test_answer_spec_instruction_contains_prompt_without_answering() -> None:
    prompt = PromptRecord(
        id="SFT-NET-0001",
        prompt="Explain DNS caching clearly.",
    )

    instruction = answer_spec_instruction(prompt)

    assert prompt.id in instruction
    assert prompt.prompt in instruction
    assert "Do not write the answer prose" in instruction


def test_generate_answer_spec_validates_prompt_id() -> None:
    prompt = PromptRecord(id="SFT-NET-0001", prompt="Explain DNS caching clearly.")

    class Model:
        async def a_generate(self, instruction, schema):
            assert schema is AnswerSpec
            return {
                "id": prompt.id,
                "required_facts": ["Resolvers cache DNS records."],
            }, 0

    result = asyncio.run(generate_answer_spec(Model(), prompt))

    assert result.id == prompt.id
    assert result.required_facts == ["Resolvers cache DNS records."]


def test_generate_answer_spec_rejects_mismatched_id() -> None:
    prompt = PromptRecord(id="SFT-NET-0001", prompt="Explain DNS caching clearly.")

    class Model:
        async def a_generate(self, instruction, schema):
            return {
                "id": "SFT-NET-0002",
                "required_facts": ["Resolvers cache DNS records."],
            }, 0

    with pytest.raises(ValueError, match="IDs differ"):
        asyncio.run(generate_answer_spec(Model(), prompt))


def test_generate_answer_specs_writes_requested_rows_in_prompt_order(tmp_path, monkeypatch) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "".join(
            [
                '{"id":"SFT-NET-0001","prompt":"Explain DNS caching clearly."}\n',
                '{"id":"SFT-NET-0002","prompt":"Explain DNS TTL values clearly."}\n',
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "answers.jsonl"

    class Model:
        async def a_generate(self, instruction, schema):
            prompt_id = "SFT-NET-0001" if "0001" in instruction else "SFT-NET-0002"
            await asyncio.sleep(0 if prompt_id.endswith("2") else 0.01)
            return {"id": prompt_id, "required_facts": [prompt_id]}, 0

    monkeypatch.setattr(sft_answers, "create_deepseek_model", lambda temperature: Model())
    specs = sft_answers.generate_answer_specs(count=2, prompts=prompts, output=output, concurrency=2)

    assert [spec.id for spec in specs] == ["SFT-NET-0001", "SFT-NET-0002"]
    assert [json.loads(line)["id"] for line in output.read_text().splitlines()] == [
        "SFT-NET-0001",
        "SFT-NET-0002",
    ]
