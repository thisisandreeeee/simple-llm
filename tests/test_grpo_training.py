import json

import pytest

from simple_llm.grpo.training import (
    load_grpo_rows,
    validate_trainable_lora_parameters,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return messages[0]["content"][0]["text"] + " [assistant]"


class FakeParameter:
    def __init__(self, size: int, *, requires_grad: bool):
        self.size = size
        self.requires_grad = requires_grad

    def numel(self):
        return self.size


class FakeModel:
    def __init__(self, parameters):
        self.parameters = parameters

    def named_parameters(self):
        return iter(self.parameters)


def _row(prompt: str, answer: str = "Reference answer") -> dict:
    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]
    }


def test_load_grpo_rows_keeps_only_the_user_message(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(_row("Question", "Do not expose this")) + "\n",
        encoding="utf-8",
    )

    converted = load_grpo_rows(path, tokenizer=FakeTokenizer())

    assert set(converted[0]) == {"prompt", "prompt_id"}
    assert converted[0]["prompt"] == "Question [assistant]"
    assert "Do not expose this" not in json.dumps(converted)


def test_prompt_ids_are_deterministic_and_distinguish_duplicate_rows(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(
            [json.dumps(_row("Same   prompt")), json.dumps(_row("Same prompt"))]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_grpo_rows(path, tokenizer=FakeTokenizer())

    assert rows[0]["prompt_id"] != rows[1]["prompt_id"]
    assert rows[0]["prompt_id"].split("-", 1)[1] == rows[1]["prompt_id"].split(
        "-", 1
    )[1]
    assert rows[0]["prompt_id"].startswith("000001-")
    assert rows[1]["prompt_id"].startswith("000002-")


def test_load_grpo_rows_reuses_dataset_validation(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(_row("Question")) + "\n", encoding="utf-8")

    assert len(load_grpo_rows(path, tokenizer=FakeTokenizer())) == 1

    path.write_text('{"messages": []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="row 1"):
        load_grpo_rows(path, tokenizer=FakeTokenizer())


def test_validate_trainable_lora_parameters_reports_adapter(capsys):
    model = FakeModel(
        [
            ("base.weight", FakeParameter(90, requires_grad=False)),
            ("q_proj.lora_A.default.weight", FakeParameter(10, requires_grad=True)),
        ]
    )

    validate_trainable_lora_parameters(model)

    assert "10 / 100 (10.0000%)" in capsys.readouterr().out


def test_validate_trainable_lora_parameters_rejects_frozen_adapter():
    model = FakeModel(
        [("q_proj.lora_A.default.weight", FakeParameter(10, requires_grad=False))]
    )

    with pytest.raises(RuntimeError, match="no trainable parameters"):
        validate_trainable_lora_parameters(model)


def test_validate_trainable_lora_parameters_rejects_trainable_base_model():
    model = FakeModel([("base.weight", FakeParameter(10, requires_grad=True))])

    with pytest.raises(RuntimeError, match="base.weight"):
        validate_trainable_lora_parameters(model)
