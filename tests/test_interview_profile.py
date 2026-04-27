"""Unit tests for generic interview profile helpers."""

from ghostmic.utils.interview_profile import (
    apply_profile_corrections,
    build_profile_summary,
    create_blank_interview_profile,
    get_active_interview_profile,
    is_profile_related_text,
    parse_profile_glossary,
)


def test_active_profile_uses_legacy_sql_toggle() -> None:
    profile = get_active_interview_profile({"sql_profile_enabled": True})

    assert profile is not None
    assert profile["id"] == "sql"


def test_custom_profile_context_and_corrections() -> None:
    profile = {
        "id": "python",
        "name": "Python",
        "context": "Backend Python interview",
        "keywords": ["python", "fastapi"],
        "sections": [
            {
                "name": "Frameworks",
                "items": [
                    {
                        "canonical": "FastAPI",
                        "definition": "Python API framework",
                        "aliases": ["fast api"],
                    }
                ],
            }
        ],
    }

    assert is_profile_related_text("How do you structure a fast api service?", profile)
    assert "Backend Python interview" in "\n".join(build_profile_summary(profile))

    result = apply_profile_corrections("How do you structure a fast api service?", profile)

    assert "FastAPI" in result["text"]
    assert result["high_confidence"][0]["corrected"] == "FastAPI"


def test_parse_profile_glossary_supports_aliases() -> None:
    sections = parse_profile_glossary(
        "Frameworks:\nFastAPI - Python API framework | aliases: fast api, fast a p i"
    )

    assert sections[0]["name"] == "Frameworks"
    assert sections[0]["items"][0]["canonical"] == "FastAPI"
    assert sections[0]["items"][0]["aliases"] == ["fast api", "fast a p i"]


def test_create_blank_profile_uses_unique_slug() -> None:
    profile = create_blank_interview_profile("Python", [{"id": "python"}])

    assert profile["id"] == "python-2"
    assert profile["name"] == "Python"
