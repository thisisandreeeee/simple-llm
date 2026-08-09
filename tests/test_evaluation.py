import json

from simple_llm.evaluation import load_evals, summary


def test_load_evals_validates_rows_and_ids(tmp_path) -> None:
    path = tmp_path / "evals.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "A-01", "prompt": "Explain A."}),
                json.dumps({"id": "B-01", "prompt": "Explain B."}),
            ]
        ),
        encoding="utf-8",
    )

    assert load_evals(path) == [
        {"id": "A-01", "prompt": "Explain A."},
        {"id": "B-01", "prompt": "Explain B."},
    ]


def test_summary_preserves_success_invalid_and_failed_counts() -> None:
    results = [
        {
            "id": "NET-01",
            "domain": "NET",
            "scores": {"technical_adequacy": 0.75},
            "generation_seconds": 2.0,
            "output_tokens": 10,
        },
        {
            "id": "NET-02",
            "domain": "NET",
            "validity": {"valid": False, "reasons": ["truncated"]},
            "truncated": True,
        },
        {"id": "NET-03", "domain": "NET", "error": "generation failed"},
    ]

    result = summary(results)

    assert result["successful_count"] == 1
    assert result["invalid_count"] == 1
    assert result["failed_count"] == 1
    assert result["invalid_reasons"] == {"truncated": 1}
    assert result["score_means"] == {"technical_adequacy": 0.75}
    assert result["domain_score_means"] == {
        "NET": {"technical_adequacy": 0.75}
    }
