import pytest
from pydantic import ValidationError

from simple_llm.sft_answers import AnswerRecord, AnswerSpec, build_answer_record


def make_spec() -> AnswerSpec:
    return AnswerSpec(
        id="SFT-NET-0001",
        required_facts=["A resolver can cache DNS records."],
        target_length="short",
    )


def test_answer_spec_accepts_intermediate_contract() -> None:
    spec = make_spec()

    assert spec.required_sections == []
    assert spec.caveats_and_safety == []
    assert spec.valid_commands_or_code == []
    assert spec.prohibited_claims == []


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
            target_length="short",
        )

    with pytest.raises(ValidationError):
        AnswerRecord(id="SFT-NET-0001", final_response="", extra="no")
