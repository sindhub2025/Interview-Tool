"""
GhostMic error taxonomy and shared error-detection utilities.

Provides a hierarchy of domain-specific exceptions with user-facing
messages, plus helpers used by multiple modules.
"""

from __future__ import annotations


class GhostMicError(Exception):
    """Base error for all GhostMic exceptions."""
    user_message: str = "An unexpected error occurred."


class ConfigError(GhostMicError):
    """Invalid or missing configuration."""
    user_message = "Configuration error. Check settings."


class APIKeyMissing(GhostMicError):
    """Required API key is not configured."""
    user_message = "API key not configured. Go to Settings → AI."


class APIConnectionFailed(GhostMicError):
    """Could not reach the AI backend."""
    user_message = "Could not connect to AI service. Check your internet and API key."


class RateLimited(GhostMicError):
    """HTTP 429 or equivalent rate-limit response."""
    user_message = "Rate limited. Please wait a moment and try again."


class ModelLoadFailed(GhostMicError):
    """Whisper model failed to load."""
    user_message = "Speech recognition model failed to load."


class TranscriptionUnavailable(GhostMicError):
    """Neither local model nor remote fallback is available."""
    user_message = "Transcription service is not ready."


def is_rate_limited(exc: Exception) -> bool:
    """Return True if *exc* looks like a rate-limit / HTTP 429 error.

    Used by both ``AIThread`` and ``TranscriptionThread`` to decide
    whether to retry.
    """
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            " 429",
            "status code: 429",
            "too many requests",
            "rate limit",
            "rate_limit",
        )
    )
