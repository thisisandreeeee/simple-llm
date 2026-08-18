from simple_llm.sft_answers import AnswerRecord
from simple_llm.sft_dataset import split_dataset
from simple_llm.sft_prompts import PromptRecord


def test_split_dataset_is_deterministic_and_stratified_by_subject():
    prompts = [
        PromptRecord(id=f"SFT-{subject}-{number:04d}", prompt=f"{subject} {number}")
        for subject in ("API", "NET")
        for number in range(10)
    ]
    answers = [
        AnswerRecord(id=prompt.id, final_response=f"Answer {prompt.id}")
        for prompt in prompts
    ]

    first = split_dataset(prompts, answers)
    second = split_dataset(prompts, answers)

    assert first == second
    assert [len(part) for part in first] == [18, 2]
    assert {
        example.messages[0].content[0].text.split()[0] for example in first[1]
    } == {"API", "NET"}
