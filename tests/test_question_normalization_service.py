"""Unit tests for speaker-question normalization helpers."""

from ghostmic.services.question_normalization_service import (
    _extract_json_payload,
    _build_normalization_context_block,
    _parse_normalization_result,
    _sanitize_follow_up_questions,
)


def test_extract_json_payload_parses_fenced_json_block() -> None:
    payload = _extract_json_payload(
        """
```json
{
  "normalized_question": "How do you test ETL pipelines?",
  "follow_up_questions": [
    "Can you share a production example?",
    "How do you handle schema drift?",
    "What metrics do you track?"
  ]
}
```
"""
    )

    assert payload is not None
    assert payload["normalized_question"] == "How do you test ETL pipelines?"
    assert len(payload["follow_up_questions"]) == 3


def test_sanitize_follow_up_questions_deduplicates_and_fills_three_items() -> None:
    cleaned = _sanitize_follow_up_questions(
        [
            "Can you share a production incident",
            "Can you share a production incident",
            "How do you monitor this in production?",
        ],
        "How do you test ETL pipelines?",
    )

    assert len(cleaned) == 3
    assert all(item.endswith("?") for item in cleaned)
    assert len(set(cleaned)) == 3


def test_parse_normalization_result_supports_plain_text_fallback() -> None:
    result = _parse_normalization_result(
        "How do you validate row counts between source and target tables",
        fallback_question="raw transcript",
    )

    assert result.normalized_question == (
        "How do you validate row counts between source and target tables?"
    )
    assert len(result.follow_up_questions) == 3


def test_build_normalization_context_block_includes_resume_and_sql_context() -> None:
    context = _build_normalization_context_block(
        "Can you tell me about your experience using SQL window functions in ETL pipelines?",
        {
            "resume_profile": {
                "summary": "Data engineer with SQL and Python experience.",
                "skills": ["SQL", "Python", "ETL"],
                "companies": ["Acme"],
            },
            "resume_context_enabled": True,
            "sql_profile_enabled": True,
        },
    )

    assert "Resume context:" in context
    assert "Skills: SQL, Python, ETL" in context
    assert "SQL context:" in context
    assert "ROW_NUMBER()" in context
