import asyncio
import json

import pytest
from pydantic import ValidationError

import simple_llm.sft_answers as sft_answers
from simple_llm.sft_answers import (
    AnswerRecord,
    AnswerSpec,
    AnswerSpecDraft,
    GeneratedAnswer,
    answer_instruction,
    answer_repair_instruction,
    answer_spec_instruction,
    answer_style_violations,
    build_answer_record,
    generate_answer,
    generate_answers,
    generate_answer_spec,
    load_answer_records,
    load_prompts,
)
from simple_llm.sft_prompts import PromptRecord


def make_spec() -> AnswerSpec:
    return AnswerSpec(
        id="SFT-NET-0001",
        key_points=["A resolver can cache DNS records."],
    )


def test_answer_spec_accepts_intermediate_contract() -> None:
    spec = make_spec()

    assert spec.model_dump() == {
        "id": "SFT-NET-0001",
        "key_points": ["A resolver can cache DNS records."],
    }


def test_answer_record_contains_only_the_final_target() -> None:
    record = build_answer_record(make_spec(), "Resolvers cache DNS records.")

    assert record.model_dump() == {
        "id": "SFT-NET-0001",
        "key_points": ["A resolver can cache DNS records."],
        "final_response": "Resolvers cache DNS records.",
    }


def test_answer_models_reject_invalid_shape() -> None:
    with pytest.raises(ValidationError):
        AnswerSpec(
            id="SFT-NET-0001",
            key_points=[],
        )

    with pytest.raises(ValidationError):
        AnswerSpec(
            id="SFT-NET-0001",
            key_points=["A fact"],
            target_length="short",
        )

    with pytest.raises(ValidationError):
        AnswerRecord(
            id="SFT-NET-0001",
            key_points=["A fact"],
            final_response="",
            extra="no",
        )

    with pytest.raises(ValidationError):
        AnswerSpec(
            id="SFT-NET-0001",
            key_points=[f"Fact {index}" for index in range(9)],
        )


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
    assert "Do not write answer prose" in instruction
    assert "Do not include `id`" in instruction
    assert "smallest sufficient content contract" in instruction
    assert "one essential claim" in instruction
    assert "audit every factual claim" in instruction
    assert "valid implementation, process, or deployment" in instruction
    assert "static, long-lived, secure, atomic" in instruction
    assert "Use up to 8" in instruction


def test_generate_answer_spec_validates_prompt_id() -> None:
    prompt = PromptRecord(id="SFT-NET-0001", prompt="Explain DNS caching clearly.")

    class Model:
        async def a_generate(self, instruction, schema):
            assert schema is AnswerSpecDraft
            return {
                "key_points": ["Resolvers cache DNS records."],
            }, 0

    result = asyncio.run(generate_answer_spec(Model(), prompt))

    assert result.id == prompt.id
    assert result.key_points == ["Resolvers cache DNS records."]


def test_generate_answer_spec_uses_prompt_id() -> None:
    prompt = PromptRecord(id="SFT-NET-0001", prompt="Explain DNS caching clearly.")

    class Model:
        async def a_generate(self, instruction, schema):
            assert schema is AnswerSpecDraft
            return {
                "key_points": ["Resolvers cache DNS records."],
            }, 0

    result = asyncio.run(generate_answer_spec(Model(), prompt))

    assert result.id == prompt.id


def test_answer_instruction_contains_key_points_and_ste_guidance() -> None:
    prompt = PromptRecord(id="SFT-NET-0001", prompt="Explain DNS caching clearly.")
    spec = make_spec()

    instruction = answer_instruction(prompt, spec)

    assert "A resolver can cache DNS records." in instruction
    assert "Never trade correctness for simplicity" in instruction
    assert "Do not add background or optional detail" in instruction
    assert "Use one main idea per sentence" in instruction
    assert "12 to 20 words" in instruction
    assert "one or two sentences" in instruction
    assert "Do not use semicolons" in instruction


def test_answer_style_violations_identifies_repairable_problems() -> None:
    answer = (
        "This sentence contains many words because it combines several independent "
        "claims into one dense statement that is difficult for a reader to follow; "
        "it also uses a semicolon."
    )

    violations = answer_style_violations(answer)

    assert len(violations) == 2
    assert "exceed 20 words" in violations[0]
    assert "semicolon" in violations[1]


def test_answer_repair_instruction_preserves_content() -> None:
    prompt = PromptRecord(id="SFT-NET-0001", prompt="Explain DNS caching clearly.")
    spec = make_spec()

    instruction = answer_repair_instruction(
        prompt,
        spec,
        "Resolvers cache records; this reduces queries.",
        ["The answer contains a semicolon."],
    )

    assert prompt.prompt in instruction
    assert spec.key_points[0] in instruction
    assert "Do not add or remove information" in instruction
    assert "compliant text without rewriting it" in instruction
    assert "Do not increase the total word count" in instruction
    assert "The answer contains a semicolon." in instruction


def test_generate_answer_builds_complete_record() -> None:
    prompt = PromptRecord(id="SFT-NET-0001", prompt="Explain DNS caching clearly.")
    spec = make_spec()

    class Model:
        async def a_generate(self, instruction, schema):
            assert schema is GeneratedAnswer
            return {"final_response": "A resolver can cache DNS records."}, 0

    result = asyncio.run(generate_answer(Model(), prompt, spec))

    assert result.model_dump() == {
        "id": prompt.id,
        "key_points": spec.key_points,
        "final_response": "A resolver can cache DNS records.",
    }


def test_generate_answer_repairs_style_once() -> None:
    prompt = PromptRecord(id="SFT-NET-0001", prompt="Explain DNS caching clearly.")
    spec = make_spec()

    class Model:
        calls = 0

        async def a_generate(self, instruction, schema):
            assert schema is GeneratedAnswer
            self.calls += 1
            if self.calls == 1:
                return {
                    "final_response": "Resolvers cache records; this reduces queries."
                }, 0
            return {
                "final_response": "Resolvers cache records. This reduces queries."
            }, 0

    model = Model()
    result = asyncio.run(generate_answer(model, prompt, spec))

    assert model.calls == 2
    assert result.final_response == "Resolvers cache records. This reduces queries."


def test_load_answer_records_migrates_required_facts(tmp_path) -> None:
    path = tmp_path / "sft_answers.jsonl"
    path.write_text(
        '{"id":"SFT-NET-0001","required_facts":["A fact"],'
        '"required_sections":[],"answer_type":"explanation",'
        '"approximate_length":"short","final_response":"An answer."}\n',
        encoding="utf-8",
    )

    assert load_answer_records(path) == [
        AnswerRecord(
            id="SFT-NET-0001",
            key_points=["A fact"],
            final_response="An answer.",
        )
    ]


def test_generate_answers_writes_three_field_rows(tmp_path, monkeypatch) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"id":"SFT-NET-0001","prompt":"Explain DNS caching clearly."}\n',
        encoding="utf-8",
    )
    output = tmp_path / "answers.jsonl"

    class Model:
        async def a_generate(self, instruction, schema):
            if schema is AnswerSpecDraft:
                return {
                    "key_points": ["A fact"],
                }, 0
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
        },
        {"temperature": 0.0},
    ]

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row == {
        "id": "SFT-NET-0001",
        "key_points": ["A fact"],
        "final_response": "An answer.",
    }


def test_generate_answers_resumes_by_prompt_id(tmp_path, monkeypatch) -> None:
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
    output.write_text(
        '{"id":"SFT-NET-0001","required_facts":["Existing fact"],"final_response":"Existing answer."}\n',
        encoding="utf-8",
    )
    calls: list[str] = []

    class Model:
        async def a_generate(self, instruction, schema):
            calls.append(instruction)
            if schema is AnswerSpecDraft:
                return {
                    "key_points": ["New fact"],
                }, 0
            assert schema is GeneratedAnswer
            return {"final_response": "New answer."}, 0

    monkeypatch.setattr(sft_answers, "create_deepseek_model", lambda **kwargs: Model())

    records = generate_answers(count=2, prompts=prompts, output=output)

    assert [record.id for record in records] == ["SFT-NET-0001", "SFT-NET-0002"]
    assert len(calls) == 2
    assert [json.loads(line)["id"] for line in output.read_text().splitlines()] == [
        "SFT-NET-0001",
        "SFT-NET-0002",
    ]
