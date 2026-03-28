"""Unit tests for ghostmic.utils.resume_context."""

from ghostmic.utils.resume_context import (
    apply_resume_corrections,
    build_resume_context_summary,
    is_resume_related_text,
)


def _profile():
    return {
        "identity": {
            "full_name": "Jane Doe",
            "location": "Seattle, WA",
            "email": "jane.doe@example.com",
        },
        "summary": "Data engineer with ETL and platform experience.",
        "companies": ["Microsoft", "Contoso"],
        "job_titles": ["Senior Data Engineer"],
        "skills": ["Python", "SQL", "Airflow"],
        "projects": ["Customer 360 Platform"],
        "certifications": ["AWS Certified Solutions Architect"],
        "tools": ["Azure Data Factory"],
        "technologies": ["Spark"],
        "aliases": {
            "Microsoft": ["micro soft"],
            "Python": ["py thon"],
        },
    }


def test_resume_related_detection_with_keywords_and_terms():
    profile = _profile()

    assert is_resume_related_text("Can you walk me through your experience?", profile)
    assert is_resume_related_text("What did you do at Microsoft?", profile)
    assert not is_resume_related_text("Explain CAP theorem.", profile)


def test_apply_resume_corrections_applies_high_confidence_company_fix():
    profile = _profile()
    text = "I worked at Micro hard and built ETL pipelines."

    result = apply_resume_corrections(text, profile)

    assert "Microsoft" in result["text"]
    assert result["high_confidence"]
    assert result["high_confidence"][0]["corrected"] == "Microsoft"


def test_apply_resume_corrections_keeps_medium_confidence_as_suggestion():
    profile = _profile()
    text = "I prefer pithon for backend tasks."

    result = apply_resume_corrections(text, profile)

    assert result["text"] == text
    assert result["medium_confidence"]
    assert result["medium_confidence"][0]["corrected"] == "Python"


def test_build_resume_context_summary_contains_key_sections_only():
    profile = _profile()

    summary = build_resume_context_summary(profile)
    joined = "\n".join(summary)

    assert "Name: Jane Doe" in joined
    assert "Companies:" in joined
    assert "Skills:" in joined
    assert "Projects:" in joined
    assert "jane.doe@example.com" not in joined
