"""Unit tests for ghostmic.utils.sql_context."""

from ghostmic.utils.sql_context import (
    apply_sql_corrections,
    build_sql_profile_summary,
    is_sql_related_text,
)


def test_sql_profile_summary_contains_key_function_categories():
    summary = build_sql_profile_summary()
    joined = "\n".join(summary)

    assert "Aggregate Functions:" in joined
    assert "COUNT() - number of rows" in joined
    assert "String Functions:" in joined
    assert "ROW_NUMBER() - assigns row sequence" in joined


def test_sql_corrections_normalize_common_function_terms():
    result = apply_sql_corrections(
        "Use call as with current timestamp and row number in this query."
    )

    assert "COALESCE()" in result["text"]
    assert "CURRENT_TIMESTAMP" in result["text"]
    assert "ROW_NUMBER()" in result["text"]
    assert result["high_confidence"]


def test_sql_related_detection_finds_sql_terms():
    assert is_sql_related_text("How do I use COALESCE in a SQL query?")
    assert not is_sql_related_text("Explain the business rule.")
