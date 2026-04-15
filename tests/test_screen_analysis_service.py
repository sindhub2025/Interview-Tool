"""Unit tests for ghostmic.services.screen_analysis_service."""

from __future__ import annotations

import requests

from ghostmic.services.screen_analysis_service import (
    DEFAULT_SCREEN_PROMPT,
    GROQ_VISION_MODEL,
    SCREEN_ANALYSIS_MAX_COMPLETION_TOKENS,
    SCREEN_ANALYSIS_TEMPERATURE,
    analyze_screenshot_with_groq,
    encode_image_data_url,
    extract_groq_text,
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
    assert calls["json"]["messages"][0]["content"][0]["text"] == DEFAULT_SCREEN_PROMPT
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