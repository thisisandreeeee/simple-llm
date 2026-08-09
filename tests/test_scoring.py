import pytest
from pydantic import ValidationError

from simple_llm.scoring import STEScores, score


def test_runs_independent_scorers() -> None:
    result = score(
        "What does the pump do?",
        "The pump moves oil.",
        {
            "average_sentence_length": lambda prompt, answer: 3.0,
            "long_sentence_fraction": lambda prompt, answer: 0.25,
            "technical_adequacy": lambda prompt, answer: 0.75,
        },
    )

    assert result.average_sentence_length == 3.0
    assert result.long_sentence_fraction == 0.25
    assert result.technical_adequacy == 0.75
    assert result.controlled_vocabulary is None
    assert not hasattr(result, "overall_score")


def test_rejects_unknown_dimensions() -> None:
    with pytest.raises(ValidationError):
        score("prompt", "answer", {"unknown": lambda prompt, answer: 1.0})


def test_scores_must_be_normalized() -> None:
    with pytest.raises(ValidationError):
        STEScores(technical_adequacy=1.1)
