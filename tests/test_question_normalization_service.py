"""Unit tests for speaker-question normalization helpers."""

import sys
import types as pytypes

from ghostmic.services.question_normalization_service import (
    _extract_json_payload,
    _build_normalization_context_block,
    _parse_normalization_result,
    _sanitize_follow_up_questions,
    normalize_question_with_followups,
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


def test_normalize_question_with_followups_supports_gemini_backend(monkeypatch) -> None:
    calls = {}

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls["model"] = model
            calls["contents"] = contents
            calls["config"] = config
            return pytypes.SimpleNamespace(
                text=(
                    "{"
                    '"normalized_question":"How do you test ETL pipelines?",'
                    '"follow_up_questions":['
                    '"Can you share a production incident?",'
                    '"How do you handle schema drift?",'
                    '"What metrics do you track?"'
                    "]"
                    "}"
                )
            )

    class FakeClient:
        def __init__(self, *, api_key):
            calls["api_key"] = api_key
            self.models = FakeModels()

    fake_types = pytypes.ModuleType("google.genai.types")
    fake_types.GenerateContentConfig = FakeGenerateContentConfig

    fake_genai = pytypes.ModuleType("google.genai")
    fake_genai.Client = FakeClient
    fake_genai.types = fake_types

    fake_google = pytypes.ModuleType("google")
    fake_google.genai = fake_genai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    result = normalize_question_with_followups(
        "how do you test etl pipelines",
        {
            "main_backend": "gemini",
            "gemini_api_key": "test-gemini-key",
            "gemini_model": "gemini-3-flash-preview",
        },
    )

    assert result.normalized_question == "How do you test ETL pipelines?"
    assert len(result.follow_up_questions) == 3
    assert calls["api_key"] == "test-gemini-key"
    assert calls["model"] == "gemini-3-flash-preview"
    assert calls["config"].kwargs["temperature"] == 0.2


def test_normalize_question_with_followups_corrects_rdbms_expansion(monkeypatch) -> None:
    calls = {}

    class FakeCompletions:
        def create(self, *, model, messages, temperature, max_tokens, timeout, stream):
            calls["model"] = model
            calls["messages"] = messages
            calls["temperature"] = temperature
            calls["max_tokens"] = max_tokens
            calls["timeout"] = timeout
            calls["stream"] = stream
            return pytypes.SimpleNamespace(
                choices=[
                    pytypes.SimpleNamespace(
                        message=pytypes.SimpleNamespace(
                            content=(
                                "{"
                                '"normalized_question":"What is the difference between a Database Management System and a Real-Time Database Management System (RTDBMS)?",'
                                '"follow_up_questions":['
                                '"Can you give a practical example?",'
                                '"How does RDBMS differ from DBMS in architecture?",'
                                '"What trade-offs matter when choosing one?"'
                                "]"
                                "}"
                            )
                        )
                    )
                ]
            )

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self, *, api_key, base_url=None):
            calls["api_key"] = api_key
            calls["base_url"] = base_url
            self.chat = FakeChat()

    fake_openai = pytypes.ModuleType("openai")
    fake_openai.OpenAI = FakeClient

    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    result = normalize_question_with_followups(
        "What is the difference between DBMS and RDBMS",
        {
            "groq_api_key": "test-groq-key",
            "groq_model": "llama-4-maverick-17b-128e-instruct",
        },
    )

    assert result.normalized_question == (
        "What is the difference between a Database Management System and a Relational Database Management System (RDBMS)?"
    )
    assert len(result.follow_up_questions) == 3
    assert calls["api_key"] == "test-groq-key"
    assert calls["base_url"] == "https://api.groq.com/openai/v1"
    assert "Database acronym context:" in "\n".join(
        message["content"] for message in calls["messages"] if message["role"] == "user"
    )
