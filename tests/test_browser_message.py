"""
Tests for BrowserMessage model and validation.

No Qt, no Supabase, no network required.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from ghostmic.domain.browser_message import (
    ALLOWED_LANGUAGES,
    ALLOWED_MESSAGE_TYPES,
    MAX_MESSAGE_BYTES,
    BrowserMessage,
    BrowserMessageValidationError,
    generate_pairing_code,
    new_message_id,
)


def _make_valid_dict(**overrides) -> dict:
    base = {
        "protocol_version": 1,
        "message_id": str(uuid.uuid4()),
        "type": "question",
        "content": "What is polymorphism?",
        "language": None,
        "created_at": "2026-11-26T17:30:00Z",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Valid payloads
# ---------------------------------------------------------------------------


class TestValidPayloads:
    def test_valid_question(self) -> None:
        msg = BrowserMessage.from_dict(_make_valid_dict(type="question"))
        assert msg.type == "question"
        assert msg.language is None

    def test_valid_code(self) -> None:
        msg = BrowserMessage.from_dict(
            _make_valid_dict(
                type="code",
                language="python",
                content="def hello():\n    return 'world'\n",
            )
        )
        assert msg.type == "code"
        assert msg.language == "python"

    def test_valid_answer(self) -> None:
        msg = BrowserMessage.from_dict(_make_valid_dict(type="answer"))
        assert msg.type == "answer"

    def test_multiline_code_preserved(self) -> None:
        code = "def foo():\n    x = 1\n    return x\n"
        msg = BrowserMessage.from_dict(
            _make_valid_dict(type="code", language="python", content=code)
        )
        assert msg.content == code

    def test_whitespace_preserved(self) -> None:
        content = "    indented line\n\ttabbed line\n"
        msg = BrowserMessage.from_dict(
            _make_valid_dict(type="code", language="plaintext", content=content)
        )
        assert msg.content == content

    def test_all_allowed_languages(self) -> None:
        for lang in ALLOWED_LANGUAGES:
            msg = BrowserMessage.from_dict(
                _make_valid_dict(type="code", language=lang, content="x=1")
            )
            assert msg.language == lang

    def test_language_stripped_for_non_code(self) -> None:
        msg = BrowserMessage.from_dict(
            _make_valid_dict(type="question", language="python")
        )
        # language must be ignored for non-code types
        assert msg.language is None

    def test_to_dict_round_trip(self) -> None:
        original = _make_valid_dict(type="code", language="rust", content="fn main() {}")
        msg = BrowserMessage.from_dict(original)
        d = msg.to_dict()
        assert d["type"] == "code"
        assert d["language"] == "rust"
        assert d["content"] == "fn main() {}"
        assert d["protocol_version"] == 1


# ---------------------------------------------------------------------------
# Invalid payloads
# ---------------------------------------------------------------------------


class TestInvalidPayloads:
    def test_not_a_dict(self) -> None:
        with pytest.raises(BrowserMessageValidationError):
            BrowserMessage.from_dict("not a dict")  # type: ignore[arg-type]

    def test_wrong_protocol_version(self) -> None:
        with pytest.raises(BrowserMessageValidationError, match="protocol_version"):
            BrowserMessage.from_dict(_make_valid_dict(protocol_version=99))

    def test_invalid_message_id(self) -> None:
        with pytest.raises(BrowserMessageValidationError, match="message_id"):
            BrowserMessage.from_dict(_make_valid_dict(message_id="not-a-uuid"))

    def test_unknown_message_type(self) -> None:
        with pytest.raises(BrowserMessageValidationError, match="type"):
            BrowserMessage.from_dict(_make_valid_dict(type="summary"))

    def test_empty_content(self) -> None:
        with pytest.raises(BrowserMessageValidationError, match="empty"):
            BrowserMessage.from_dict(_make_valid_dict(content=""))

    def test_oversized_content(self) -> None:
        big = "x" * (MAX_MESSAGE_BYTES + 1)
        with pytest.raises(BrowserMessageValidationError, match="size"):
            BrowserMessage.from_dict(_make_valid_dict(content=big))

    def test_malformed_json_dict_missing_keys(self) -> None:
        # Missing required fields raises validation error
        with pytest.raises(BrowserMessageValidationError):
            BrowserMessage.from_dict({"type": "question"})

    def test_invalid_timestamp(self) -> None:
        with pytest.raises(BrowserMessageValidationError, match="ISO-8601"):
            BrowserMessage.from_dict(_make_valid_dict(created_at="not-a-date"))

    def test_duplicate_message_ids_are_values(self) -> None:
        """Two messages with the same ID must both parse — dedup is a service concern."""
        msg_id = str(uuid.uuid4())
        m1 = BrowserMessage.from_dict(_make_valid_dict(message_id=msg_id))
        m2 = BrowserMessage.from_dict(_make_valid_dict(message_id=msg_id))
        assert m1.message_id == m2.message_id

    def test_unknown_language_falls_back_to_plaintext(self) -> None:
        msg = BrowserMessage.from_dict(
            _make_valid_dict(type="code", language="cobol", content="x")
        )
        assert msg.language == "plaintext"

    def test_untrusted_html_stored_as_literal(self) -> None:
        html = "<script>alert('xss')</script>"
        msg = BrowserMessage.from_dict(_make_valid_dict(content=html))
        assert msg.content == html  # stored verbatim — UI handles escaping

    def test_content_must_be_string(self) -> None:
        with pytest.raises(BrowserMessageValidationError):
            BrowserMessage.from_dict(_make_valid_dict(content=12345))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_generate_pairing_code_length(self) -> None:
        code = generate_pairing_code()
        assert len(code) == 10

    def test_generate_pairing_code_uppercase(self) -> None:
        code = generate_pairing_code()
        assert code == code.upper()

    def test_generate_pairing_code_uniqueness(self) -> None:
        codes = {generate_pairing_code() for _ in range(50)}
        assert len(codes) >= 48  # extremely unlikely collision in 50 samples

    def test_new_message_id_is_uuid4(self) -> None:
        msg_id = new_message_id()
        parsed = uuid.UUID(msg_id)
        assert parsed.version == 4
