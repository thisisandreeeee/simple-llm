"""Adversarial checks that document known scorer blind spots."""

from pathlib import Path

import pytest

from simple_llm.scoring.rules import (
    _sentences,
    controlled_vocabulary_scorer,
    controlled_vocabulary_scorer_from_file,
    procedure_syntax,
    verb_forms_and_modals,
)
from simple_llm.scoring.validity import validate_answer


def test_empty_answer_does_not_get_perfect_compliance() -> None:
    result = validate_answer("Explain DNS.", "")
    assert result.valid is False
    assert "empty" in result.reasons


def test_each_unpunctuated_list_item_is_a_sentence_unit() -> None:
    assert _sentences("- Start the pump\n- Open the valve\n- Check the gauge") == [
        "- Start the pump",
        "- Open the valve",
        "- Check the gauge",
    ]


@pytest.mark.xfail(strict=True, reason="sentence splitting requires an uppercase next token")
def test_sentence_boundary_survives_markdown_emphasis() -> None:
    assert _sentences("Stop the pump. **Then open the valve.**") == [
        "Stop the pump",
        "**Then open the valve.**",
    ]


@pytest.mark.xfail(strict=True, reason="the imperative allowlist does not include delete")
def test_trailing_condition_on_unlisted_imperative_is_detected() -> None:
    assert procedure_syntax("", "Delete the file if the checksum is wrong.") == 0.0


@pytest.mark.xfail(strict=True, reason="irregular perfect participles are not detected")
def test_irregular_perfect_form_is_detected() -> None:
    assert verb_forms_and_modals("", "The operator has written the file.") == 0.0


@pytest.mark.xfail(strict=True, reason="an -ing adjective is mistaken for progressive tense")
def test_ing_adjective_is_not_treated_as_progressive_tense() -> None:
    assert verb_forms_and_modals("", "The file is missing.") == 1.0


@pytest.mark.xfail(strict=True, reason="May is mistaken for the banned modal may")
def test_proper_noun_may_is_not_treated_as_modal() -> None:
    assert verb_forms_and_modals("", "The release date is 1 May 2027.") == 1.0


@pytest.mark.xfail(strict=True, reason="configured phrases are compared with single tokens")
def test_multiword_configured_technical_term_is_allowed() -> None:
    scorer = controlled_vocabulary_scorer(
        {"the", "moves", "data"}, technical_terms={"message broker"}
    )
    assert scorer("", "The message broker moves data.") == 1.0


def test_default_glossary_loads_outside_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert controlled_vocabulary_scorer_from_file()("", "Start the motor.") <= 1.0
