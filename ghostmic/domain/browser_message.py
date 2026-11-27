"""
BrowserMessage — shared message contract for the Browser Input feature.

This module defines the data model and validation logic for messages
transmitted from the sender webpage to the GhostMic receiver.

Design constraints:
- No dependency on Supabase, PyQt or any networking library.
- Pure Python dataclass; safe to import in tests without a Qt context.
- All validation is centralised here so both transport and UI can rely on it.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = 1

#: Maximum UTF-8 encoded message size in bytes.
MAX_MESSAGE_BYTES = 102_400  # 100 KB

ALLOWED_MESSAGE_TYPES = frozenset({"question", "code", "answer"})

ALLOWED_LANGUAGES = frozenset({
    "python",
    "javascript",
    "typescript",
    "java",
    "c",
    "cpp",
    "csharp",
    "go",
    "rust",
    "sql",
    "bash",
    "powershell",
    "json",
    "yaml",
    "html",
    "css",
    "php",
    "ruby",
    "plaintext",
})

# Crockford Base32 alphabet (lowercase, no I/L/O/U)
_CROCKFORD_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"

# UUID4 pattern
_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _is_valid_uuid4(value: str) -> bool:
    return bool(_UUID4_RE.match(str(value or "")))


def _is_valid_iso8601(value: str) -> bool:
    """Accept ISO-8601 timestamps with or without 'Z' / offset."""
    try:
        # Try common formats
        for fmt in (
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S+00:00",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
        ):
            try:
                datetime.strptime(str(value), fmt)
                return True
            except ValueError:
                continue
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class BrowserMessage:
    """
    Immutable value object representing one message sent from the browser.

    Attributes:
        message_id: UUID4 string uniquely identifying this message.
        type: One of 'question', 'code', 'answer'.
        content: The raw UTF-8 text content. Treated as untrusted plain text.
        language: Programming language hint for code messages (may be None).
        created_at: ISO-8601 timestamp string (UTC preferred).
        protocol_version: Protocol version number (always 1 for Phase 1).
    """

    message_id: str
    type: str  # 'question' | 'code' | 'answer'
    content: str
    language: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    protocol_version: int = PROTOCOL_VERSION

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "BrowserMessage":
        """
        Construct a BrowserMessage from a raw dict (e.g. from Supabase payload).

        Raises:
            BrowserMessageValidationError: if any field is invalid.
        """
        if not isinstance(data, dict):
            raise BrowserMessageValidationError("Payload must be a JSON object.")

        errors: list[str] = []

        # protocol_version
        pv = data.get("protocol_version")
        if pv != PROTOCOL_VERSION:
            errors.append(
                f"protocol_version must be {PROTOCOL_VERSION}, got {pv!r}."
            )

        # message_id
        message_id = str(data.get("message_id", "")).strip()
        if not _is_valid_uuid4(message_id):
            errors.append(f"message_id must be a valid UUID4, got {message_id!r}.")

        # type
        msg_type = str(data.get("type", "")).strip().lower()
        if msg_type not in ALLOWED_MESSAGE_TYPES:
            errors.append(
                f"type must be one of {sorted(ALLOWED_MESSAGE_TYPES)}, got {msg_type!r}."
            )

        # content
        content = data.get("content")
        if not isinstance(content, str):
            errors.append("content must be a string.")
            content = ""
        content_bytes = content.encode("utf-8", errors="replace")
        if len(content) == 0:
            errors.append("content must not be empty.")
        if len(content_bytes) > MAX_MESSAGE_BYTES:
            errors.append(
                f"content exceeds maximum size of {MAX_MESSAGE_BYTES} bytes "
                f"(got {len(content_bytes)} bytes)."
            )

        # language (optional)
        language = data.get("language")
        if language is not None:
            language = str(language).strip().lower()
            if language == "" or language == "null":
                language = None
            elif language not in ALLOWED_LANGUAGES:
                # Gracefully fall back to plaintext
                language = "plaintext"

        # created_at
        created_at = str(data.get("created_at", "")).strip()
        if not created_at:
            created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        elif not _is_valid_iso8601(created_at):
            errors.append(f"created_at is not a valid ISO-8601 timestamp: {created_at!r}.")

        if errors:
            raise BrowserMessageValidationError(
                "Invalid browser message payload: " + " ".join(errors)
            )

        return cls(
            protocol_version=PROTOCOL_VERSION,
            message_id=message_id,
            type=msg_type,
            content=content,
            language=language if msg_type == "code" else None,
            created_at=created_at,
        )

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON encoding."""
        return {
            "protocol_version": self.protocol_version,
            "message_id": self.message_id,
            "type": self.type,
            "language": self.language,
            "content": self.content,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BrowserMessageValidationError(ValueError):
    """Raised when a received payload fails validation."""


# ---------------------------------------------------------------------------
# Pairing code helpers
# ---------------------------------------------------------------------------


def generate_pairing_code(length: int = 10) -> str:
    """
    Generate a cryptographically secure Crockford Base32 pairing code.

    Uses ``secrets`` for secure randomness.  The result is uppercase for
    display but case-insensitive on entry.
    """
    import secrets

    return "".join(
        secrets.choice(_CROCKFORD_ALPHABET) for _ in range(length)
    ).upper()


def new_message_id() -> str:
    """Return a new UUID4 string for use as a message identifier."""
    return str(uuid.uuid4())
