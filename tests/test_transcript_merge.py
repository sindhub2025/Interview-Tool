"""Unit tests for speaker transcript coalescing policy."""

from ghostmic.domain import TranscriptSegment
from ghostmic.main import GhostMicApp


def _app_with_defaults():
    app = GhostMicApp.__new__(GhostMicApp)
    app._config = {"transcription": {}}
    return app


def _speaker(text: str, ts: float) -> TranscriptSegment:
    return TranscriptSegment(text=text, source="speaker", timestamp=ts)


def _user(text: str, ts: float) -> TranscriptSegment:
    return TranscriptSegment(text=text, source="user", timestamp=ts)


def test_merge_when_previous_clause_is_incomplete():
    app = _app_with_defaults()
    previous = _speaker("Can you explain your ETL testing approach", 10.0)
    incoming = _speaker("especially around late-arriving data?", 11.1)

    assert app._should_merge_speaker_segments(previous, incoming) is True


def test_no_merge_when_pause_is_too_long():
    app = _app_with_defaults()
    previous = _speaker("Can you explain your ETL testing approach", 10.0)
    incoming = _speaker("especially around late-arriving data?", 13.0)

    assert app._should_merge_speaker_segments(previous, incoming) is False


def test_no_merge_for_new_question_after_complete_sentence():
    app = _app_with_defaults()
    previous = _speaker("How do you optimize SQL joins?", 20.0)
    incoming = _speaker("What is dense rank?", 21.0)

    assert app._should_merge_speaker_segments(previous, incoming) is False


def test_merge_when_continuation_starter_follows_quickly():
    app = _app_with_defaults()
    previous = _speaker("How do you optimize SQL joins?", 30.0)
    incoming = _speaker("and how would you validate performance?", 30.6)

    assert app._should_merge_speaker_segments(previous, incoming) is True


def test_no_merge_for_non_speaker_segments():
    app = _app_with_defaults()
    previous = _user("I would start by checking indexes.", 40.0)
    incoming = _speaker("Can you share an example?", 40.4)

    assert app._should_merge_speaker_segments(previous, incoming) is False
