import json

import pytest

from simple_llm.grpo.training import load_grpo_rows


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return messages[0]["content"][0]["text"] + " [assistant]"


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
