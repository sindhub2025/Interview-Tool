"""Unit tests for ghostmic.utils.errors — the shared error taxonomy."""

import pytest

from ghostmic.utils.errors import (
    APIConnectionFailed,
    APIKeyMissing,
    ConfigError,
    GhostMicError,
    ModelLoadFailed,
    RateLimited,
    TranscriptionUnavailable,
    is_rate_limited,
)


class TestIsRateLimited:
    """Tests for the shared rate-limit detection function."""

    @pytest.mark.parametrize(
        "message",
        [
            "Error code: 429 Too Many Requests",
            "Status code: 429",
            "too many requests - please slow down",
            "rate limit exceeded for model",
            "rate_limit_error: you are sending too many requests",
            "HTTP 429",
        ],
    )
    def test_detects_rate_limit(self, message: str) -> None:
        exc = RuntimeError(message)
        assert is_rate_limited(exc) is True

    @pytest.mark.parametrize(
        "message",
        [
            "Connection refused",
            "Timeout error",
            "Invalid API key",
            "Internal server error 500",
            "Model not found",
            "",
        ],
    )
    def test_non_rate_limit_errors(self, message: str) -> None:
        exc = RuntimeError(message)
        assert is_rate_limited(exc) is False


class TestErrorHierarchy:
    """Verify the exception hierarchy for catch-all patterns."""

    def test_all_errors_inherit_from_base(self) -> None:
        for cls in (
            ConfigError,
            APIKeyMissing,
            APIConnectionFailed,
            RateLimited,
            ModelLoadFailed,
            TranscriptionUnavailable,
        ):
            assert issubclass(cls, GhostMicError)
            assert issubclass(cls, Exception)

    def test_user_message_exists(self) -> None:
        for cls in (
            GhostMicError,
            ConfigError,
            APIKeyMissing,
            APIConnectionFailed,
            RateLimited,
            ModelLoadFailed,
            TranscriptionUnavailable,
        ):
            assert hasattr(cls, "user_message")
            assert isinstance(cls.user_message, str)
            assert len(cls.user_message) > 0
