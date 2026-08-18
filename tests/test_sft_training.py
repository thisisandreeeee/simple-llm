import json

import pytest

from simple_llm.sft_training import load_dataset_rows, to_prompt_completion_rows


def test_load_dataset_rows_validates_two_message_conversations(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Question"}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Answer"}],
                    },
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_dataset_rows(path)[0]["messages"][1]["content"][0]["text"] == "Answer"

    path.write_text('{"messages": []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="row 1"):
        load_dataset_rows(path)


def test_to_prompt_completion_rows_separates_the_training_target():
    user = {"role": "user", "content": [{"type": "text", "text": "Question"}]}
    assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Answer"}],
    }

    assert to_prompt_completion_rows([{"messages": [user, assistant]}]) == [
        {
            "prompt": [user],
            "completion": [assistant],
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ]
