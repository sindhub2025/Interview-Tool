"""Unit tests for ghostmic.services.screen_analysis_service."""

from __future__ import annotations

import sys
import types as pytypes

import requests

from ghostmic.services.screen_analysis_service import (
    DEFAULT_SCREEN_PROMPT,
    DEEP_ANALYSIS_SCREEN_PROMPT,
    GEMINI_VISION_MODEL,
    GROQ_VISION_MODEL,
    SCREEN_ANALYSIS_MAX_COMPLETION_TOKENS,
    SCREEN_ANALYSIS_TEMPERATURE,
    analyze_screenshot_with_gemini,
    analyze_screenshot_with_groq,
    build_screen_analysis_prompt,
    encode_image_data_url,
    extract_groq_text,
    resolve_screen_analysis_provider,
)


def test_encode_image_data_url_prefix() -> None:
    data_url = encode_image_data_url(b"abc")

    assert data_url.startswith("data:image/png;base64,")


def test_extract_groq_text_from_string_content() -> None:
    response_json = {
        "choices": [
            {
                "message": {
                    "content": "  Screenshot looks fine.  ",
                }
            }
        ]
    }

    assert extract_groq_text(response_json) == "Screenshot looks fine."


def test_extract_groq_text_from_output_parts() -> None:
    response_json = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "output_text", "text": "First line."},
                        {"type": "text", "text": "Second line."},
                    ]
                }
            }
        ]
    }

    assert extract_groq_text(response_json) == "First line.\nSecond line."


def test_default_screen_prompt_is_question_focused() -> None:
    prompt = DEFAULT_SCREEN_PROMPT.lower()

    assert "question" in prompt
    assert "short summary" in prompt
    assert "1-2 sentences" in prompt
    assert "connect related" in prompt


def test_deep_analysis_prompt_extracts_tables_and_er() -> None:
    prompt = DEEP_ANALYSIS_SCREEN_PROMPT.lower()

    assert "tables detected" in prompt
    assert "er relationships" in prompt
    assert "column" in prompt
    assert "primary key" in prompt or "pk" in prompt
    assert "foreign key" in prompt or "fk" in prompt
    assert "code" in prompt
    assert "queries" in prompt
    assert "content summary" in prompt
    assert "sample" in prompt


def test_screen_analysis_prompt_includes_active_profile_context() -> None:
    prompt = build_screen_analysis_prompt(
        {
            "interview_profile_enabled": True,
            "active_interview_profile_id": "sql",
        }
    ).lower()

    assert "active interview profile" in prompt
    assert "profile: sql" in prompt
    assert "row_number()" in prompt


def test_max_completion_tokens_supports_detailed_output() -> None:
    assert SCREEN_ANALYSIS_MAX_COMPLETION_TOKENS >= 4096


def test_resolve_screen_analysis_provider_prefers_gemini_when_selected() -> None:
    assert resolve_screen_analysis_provider({"main_backend": "gemini"}) == "gemini"
    assert resolve_screen_analysis_provider({"main_backend": "groq"}) == "groq"
    # Non-multimodal fallback stays on Groq path.
    assert resolve_screen_analysis_provider({"main_backend": "openai"}) == "groq"


def test_analyze_screenshot_with_gemini_builds_expected_payload(monkeypatch) -> None:
    calls = {}

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakePart:
        @staticmethod
        def from_bytes(*, data, mime_type):
            calls["part_data"] = data
            calls["part_mime"] = mime_type
            return {
                "data": data,
                "mime_type": mime_type,
            }

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls["model"] = model
            calls["contents"] = contents
            calls["config"] = config
            return pytypes.SimpleNamespace(text="Gemini found a SQL error dialog.")

    class FakeClient:
        def __init__(self, *, api_key):
            calls["api_key"] = api_key
            self.models = FakeModels()

    fake_types = pytypes.ModuleType("google.genai.types")
    fake_types.Part = FakePart
    fake_types.GenerateContentConfig = FakeGenerateContentConfig

    fake_genai = pytypes.ModuleType("google.genai")
    fake_genai.Client = FakeClient
    fake_genai.types = fake_types

    fake_google = pytypes.ModuleType("google")
    fake_google.genai = fake_genai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    result = analyze_screenshot_with_gemini(
        b"fake-image-bytes",
        "test-gemini-key",
    )

    assert result == "Gemini found a SQL error dialog."
    assert calls["api_key"] == "test-gemini-key"
    assert calls["model"] == GEMINI_VISION_MODEL
    assert calls["part_data"] == b"fake-image-bytes"
    assert calls["part_mime"] == "image/png"
    assert calls["contents"][1] == DEEP_ANALYSIS_SCREEN_PROMPT
    assert calls["config"].kwargs["temperature"] == SCREEN_ANALYSIS_TEMPERATURE
    assert (
        calls["config"].kwargs["max_output_tokens"]
        == SCREEN_ANALYSIS_MAX_COMPLETION_TOKENS
    )


def test_analyze_screenshot_with_groq_builds_expected_payload(monkeypatch) -> None:
    calls = {}

    class FakeResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Found a login dialog.",
                        }
                    }
                ]
            }

    def fake_post(url, json, headers, timeout):
        calls["url"] = url
        calls["json"] = json
        calls["headers"] = headers
        calls["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("ghostmic.services.screen_analysis_service.requests.post", fake_post)

    result = analyze_screenshot_with_groq(
        b"fake-image-bytes",
        "test-groq-key",
    )

    assert result == "Found a login dialog."
    assert calls["url"].endswith("/chat/completions")
    assert calls["json"]["model"] == GROQ_VISION_MODEL
    assert calls["json"]["messages"][0]["content"][0]["text"] == DEEP_ANALYSIS_SCREEN_PROMPT
    assert calls["json"]["messages"][0]["content"][1]["type"] == "image_url"
    assert calls["json"]["temperature"] == SCREEN_ANALYSIS_TEMPERATURE
    assert calls["json"]["max_completion_tokens"] == SCREEN_ANALYSIS_MAX_COMPLETION_TOKENS
    assert calls["headers"]["Authorization"] == "Bearer test-groq-key"
    assert calls["timeout"] == 90.0


def test_analyze_screenshot_with_groq_wraps_request_exception(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        raise requests.exceptions.ConnectionError("temporary network failure")

    monkeypatch.setattr("ghostmic.services.screen_analysis_service.requests.post", fake_post)

    try:
        analyze_screenshot_with_groq(
            b"fake-image-bytes",
            "test-groq-key",
        )
    except RuntimeError as exc:
        assert "Groq request failed" in str(exc)
        assert "temporary network failure" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError to be raised")
