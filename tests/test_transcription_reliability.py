"""Reliability tests for bounded real-time transcription requests."""

import numpy as np

from ghostmic.core.transcription_engine import (
    TranscriptionThread,
    _TranscriptionRequest,
)
from ghostmic.domain import TranscriptSegment


def _thread(config=None):
    return TranscriptionThread(
        model=object(),
        remote_config=config or {"min_segment_seconds": 0.2, "trim_silence": False},
    )


def _audio():
    return (np.ones(8_000, dtype=np.int16) * 1200)


def test_request_identity_rejects_duplicate_but_allows_same_audio_in_new_session():
    thread = _thread()
    assert thread.push_segment(
        _audio(), "speaker", session_id=1, chunk_ids=["chunk-1"], timestamp_start=1.0, timestamp_end=1.5
    ) is True
    assert thread.push_segment(
        _audio(), "speaker", session_id=1, chunk_ids=["chunk-1"], timestamp_start=1.0, timestamp_end=1.5
    ) is False
    assert thread.push_segment(
        _audio(), "speaker", session_id=2, chunk_ids=["chunk-1"], timestamp_start=1.0, timestamp_end=1.5
    ) is True

    statuses = [item["status"] for item in thread.get_diagnostic_history()]
    assert "duplicate" in statuses


def test_out_of_order_result_is_marked_stale_and_not_accepted():
    thread = _thread()
    newer = _TranscriptionRequest("new", "speaker", 7, (), 2.0, 3.0, 0.0)
    older = _TranscriptionRequest("old", "speaker", 7, (), 1.0, 2.0, 0.0)

    assert thread._accept_result(TranscriptSegment("new", "speaker"), newer) is True
    assert thread._accept_result(TranscriptSegment("old", "speaker"), older) is False
    assert thread.get_diagnostic_history()[-1]["stale"] is True


def test_overlap_is_bounded_and_does_not_use_unlimited_audio_context():
    thread = _thread({"overlap_ms": 100, "max_context_seconds": 0.25})
    first = np.ones(16_000, dtype=np.int16)
    second = np.ones(16_000, dtype=np.int16)
    thread.push_segment(first, "speaker", session_id=1, timestamp_start=0.0, timestamp_end=1.0)
    _first_audio, _source, _queued, _session = thread._queue.get_nowait()
    thread.push_segment(second, "speaker", session_id=1, timestamp_start=1.0, timestamp_end=2.0)
    queued_audio, _source, _queued, _session = thread._queue.get_nowait()

    assert queued_audio.size == 16_000 + 1_600
    assert thread._audio_context[(1, "speaker")].size <= 4_000


def test_empty_whisper_result_is_safe():
    class EmptyModel:
        def transcribe(self, audio, **kwargs):  # noqa: ARG002
            return [], {}

    thread = _thread()
    thread.set_model(EmptyModel())
    request = _TranscriptionRequest("empty", "speaker", 1, (), 1.0, 1.5, 0.0)
    assert thread._transcribe(_audio(), "speaker", 1.5, request=request) is None


def test_whisper_exception_uses_remote_fallback_when_available(monkeypatch):
    class BrokenModel:
        def transcribe(self, audio, **kwargs):  # noqa: ARG002
            raise RuntimeError("decoder failed")

    thread = _thread()
    thread.set_model(BrokenModel())
    monkeypatch.setattr(thread, "_prepare_local_audio", lambda audio, source="speaker": audio)
    thread._remote_transcription_enabled = True
    request = _TranscriptionRequest("fallback", "speaker", 1, (), 1.0, 1.5, 0.0)
    fallback = TranscriptSegment("fallback text", "speaker")
    monkeypatch.setattr(thread, "_transcribe_remote", lambda *args, **kwargs: fallback)

    result = thread._transcribe(_audio(), "speaker", 1.5, request=request)

    assert result is fallback
    assert "retry" in [item["status"] for item in thread.get_diagnostic_history()]


def test_queue_overflow_marks_dropped_request():
    thread = _thread()
    for index in range(25):
        thread.push_segment(
            _audio(),
            "speaker",
            session_id=1,
            chunk_ids=[f"chunk-{index}"],
            timestamp_start=float(index),
            timestamp_end=float(index) + 0.5,
        )

    assert any(item["status"] == "dropped" for item in thread.get_diagnostic_history())
