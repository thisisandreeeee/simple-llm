import pytest

from simple_llm.scoring.rules import (
    _words,
    controlled_vocabulary_scorer,
    controlled_vocabulary_scorer_from_file,
    document_limits,
    procedure_syntax,
    average_sentence_length,
    long_sentence_fraction,
    sentence_mechanics,
    terminology_consistency,
    verb_forms_and_modals,
)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Install the pump.", 0.0),
        ("Install the pump. Then start the motor.", 0.0),
        (
            "Install the pump and start the motor after you confirm that the valve "
            "is open and the pressure is stable and the area is clear.",
            1.0,
        ),
    ],
)
def test_long_sentence_fraction_and_procedure_detection(
    answer: str, expected: float
) -> None:
    assert long_sentence_fraction("", answer) == expected


def test_sentence_length_metrics_are_separate() -> None:
    answer = "Install the pump. Then start the motor."
    assert average_sentence_length("", answer) == 3.5
    assert long_sentence_fraction("", answer) == 0.0

    long_answer = (
        "Install the pump and start the motor after you confirm that the valve is "
        "open and the pressure is stable and the area is clear."
    )
    assert average_sentence_length("", long_answer) > 20
    assert long_sentence_fraction("", long_answer) == 1.0


def test_sentence_mechanics() -> None:
    assert sentence_mechanics("", "Install the pump. Start the motor.") == 1.0
    assert sentence_mechanics("", "Install the pump; start the motor.") == 0.0
    assert sentence_mechanics("", "Use e.g. the red port.") == 0.0
    assert sentence_mechanics("", "IPv4's address is private. It is not public.") == 1.0
    assert sentence_mechanics(
        "", "Run `foo(); bar();`. Then inspect the output."
    ) == 1.0


def test_sentence_ending_after_quote() -> None:
    assert long_sentence_fraction(
        "",
        'IPv6 is not “IPv4 with bigger addresses.” In practice, it uses multicast.',
    ) == 0.0


def test_sentence_boundaries_handle_decimals_and_quotes() -> None:
    answer = 'Set the value to 1.5 V. Then start the pump.'
    assert long_sentence_fraction("", answer) == 0.0
    assert long_sentence_fraction(
        "",
        'IPv6 is “IPv4 with bigger addresses.” In practice, it uses multicast.',
    ) == 0.0


def test_technical_identifiers_count_as_one_word() -> None:
    assert len(_words("Use 192.168.1.10 at 1.5 V.")) == 4


def test_verb_forms_and_modals() -> None:
    assert verb_forms_and_modals("", "The pump moves oil.") == 1.0
    assert verb_forms_and_modals("", "The pump should move oil.") == 0.0
    assert verb_forms_and_modals("", "The pump has been moving oil.") == 0.0


def test_document_limits() -> None:
    paragraph = " ".join(f"Sentence {number}." for number in range(7))
    assert document_limits("", paragraph) == 0.0
    assert document_limits("", "One sentence.\n\nAnother sentence.") == 1.0
    vertical_list = "\n".join(
        [
            "Addressing: IPv4 uses 32-bit addresses.",
            "Broadcast: IPv6 uses multicast.",
            "Routing: IPv6 uses a simpler header.",
            "Discovery: IPv6 uses NDP.",
            "Configuration: IPv6 supports SLAAC.",
            "Subnets: IPv6 commonly uses /64.",
            "Deployment: Many networks use dual stack.",
        ]
    )
    assert document_limits("", vertical_list) == 1.0


def test_procedure_syntax() -> None:
    assert procedure_syntax("", "Install the pump if the valve is open.") == 0.0
    assert procedure_syntax("", "If the valve is open, install the pump.") == 1.0


def test_terminology_consistency() -> None:
    assert terminology_consistency("", "Check the file. Check the path.") == 1.0
    assert terminology_consistency("", "Check the file. Verify the path.") == 0.0


def test_controlled_vocabulary_uses_prompt_and_glossary_terms() -> None:
    scorer = controlled_vocabulary_scorer({"the", "pump", "moves", "oil"})
    assert scorer("", "The pump moves oil.") == 1.0
    assert scorer("", "The pump moves water.") < 1.0
    assert scorer("webhook", "The webhook moves oil.") == 1.0


def test_controlled_vocabulary_uses_extracted_issue9_file() -> None:
    scorer = controlled_vocabulary_scorer_from_file(technical_terms={"motor"})
    assert scorer("", "Start the motor.") == 1.0
    assert scorer("", "Commence the motor.") < 1.0
