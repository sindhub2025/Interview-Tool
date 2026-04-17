"""Unit tests for speaker transcript coalescing policy."""

import threading

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


class _NoopLogger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


class _SessionStoreStub:
    def append_transcript(self, segment):
        return segment


def _app_for_append() -> GhostMicApp:
    app = GhostMicApp.__new__(GhostMicApp)
    app._config = {"transcription": {}}
    app._recording_active = True
    app._logger = _NoopLogger()
    app._transcript_lock = threading.Lock()
    app._transcript_history = []
    app._ai_context_history = []
    app._session_context_store = _SessionStoreStub()
    app._window = None
    app._schedule_auto_speaker_analysis = lambda segment: None
    return app


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


def test_append_merge_updates_timestamp_for_next_pause_gap_check():
    app = _app_for_append()

    first = _speaker("Can you explain", 10.0)
    second = _speaker("how you test ETL pipelines", 11.0)
    third = _speaker("when source data arrives late?", 12.6)

    assert app._append_transcript_segment(first, require_recording=True) is True
    assert app._append_transcript_segment(second, require_recording=True) is True
    assert len(app._transcript_history) == 1
    assert app._transcript_history[0].timestamp == 11.0

    # Third chunk should still merge because pause from second->third is short.
    assert app._append_transcript_segment(third, require_recording=True) is True
    assert len(app._transcript_history) == 1
    assert "when source data arrives late?" in app._transcript_history[0].text.lower()
