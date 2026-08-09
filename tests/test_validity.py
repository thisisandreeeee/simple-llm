import pytest

from simple_llm.scoring.validity import validate_answer


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ("", "empty"),
        ("   \n\t", "empty"),
        ("!!!", "no_lexical_content"),
        ("What does DNS do?", "prompt_echo"),
        ("I cannot answer that.", "refusal_only"),
        ("```python\nprint('x')", "malformed"),
        ("word word word word word word word word", "degenerate_repetition"),
    ],
)
def test_rejects_unusable_answers(answer: str, reason: str) -> None:
    result = validate_answer("What does DNS do?", answer)
    assert not result.valid
    assert reason in result.reasons


def test_rejects_explicit_truncation() -> None:
    result = validate_answer("Explain DNS.", "DNS maps names to addresses.", truncated=True)
    assert result.valid is False
    assert result.reasons == ("truncated",)


def test_accepts_a_normal_answer() -> None:
    result = validate_answer(
        "What does DNS do?", "DNS maps a domain name to an IP address."
    )
    assert result == validate_answer(
        "What does DNS do?", "DNS maps a domain name to an IP address."
    )
    assert result.valid
    assert result.reasons == ()
