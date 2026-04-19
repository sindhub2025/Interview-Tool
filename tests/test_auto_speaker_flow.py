"""Unit tests for automatic speaker-question analysis flow."""

import threading
from types import SimpleNamespace

from ghostmic.domain import TranscriptSegment
from ghostmic.services.ai_trigger_service import AITriggerService
from ghostmic.main import (
    GhostMicApp,
    INITIAL_RECORDING_QUESTION_NORMALIZATION_DELAY_MS,
)


class _NoopLogger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


class _EventStoreStub:
    def __init__(self) -> None:
        self.events = []

    def append_event(self, kind, text, source="speaker", metadata=None):
        self.events.append(
            {
                "kind": kind,
                "text": text,
                "source": source,
                "metadata": metadata,
            }
        )

    def append_transcript(self, segment) -> None:
        self.append_event(
            "transcript",
            getattr(segment, "text", ""),
            source=getattr(segment, "source", "speaker"),
            metadata={"confidence": round(float(getattr(segment, "confidence", 1.0)), 4)},
        )


def _app_for_auto(trigger_mode: str = "auto") -> GhostMicApp:
    app = GhostMicApp.__new__(GhostMicApp)
    app._config = {
        "ai": {
            "trigger_mode": trigger_mode,
            "auto_speaker_analysis_enabled": True,
            "auto_speaker_silence_seconds": 1.2,
            "auto_speaker_min_words": 4,
            "auto_speaker_min_chars": 18,
            "auto_speaker_retrigger_cooldown_seconds": 5.0,
        }
    }
    app._recording_active = True
    app._transcript_lock = threading.Lock()
    app._transcript_history = []
    app._ai_context_history = []
    app._session_context_store = _EventStoreStub()
    app._question_normalization_worker = None
    app._initial_recording_question_pending = False
    app._initial_recording_question_consumed = False
    app._initial_recording_question_generation = 0
    app._auto_speaker_silence_generation = 1
    app._auto_speaker_last_signature = ""
    app._auto_speaker_last_trigger_ts = 0.0
    app._auto_primary_question_sent = False
    app._active_primary_question_text = ""
    app._queued_normalized_questions = []
    app._normalized_segment_items = []
    app._normalized_segment_lookup = {}
    app._logger = _NoopLogger()
    app._window = None
    return app


def _app_for_normalization_callbacks() -> GhostMicApp:
    app = GhostMicApp.__new__(GhostMicApp)
    app._logger = _NoopLogger()
    app._window = None
    app._transcript_lock = threading.Lock()
    app._transcript_history = []
    app._ai_context_history = []
    app._session_context_store = _EventStoreStub()
    app._initial_recording_question_pending = False
    app._initial_recording_question_consumed = False
    app._initial_recording_question_generation = 0
    app._auto_speaker_silence_generation = 1
    app._auto_primary_question_sent = False
    app._active_primary_question_text = ""
    app._queued_normalized_questions = []
    app._normalized_segment_items = []
    app._normalized_segment_lookup = {}
    app._generate_ai_response = lambda force_follow_up=False: None
    app._prime_ai_context_with_question = lambda segment, question: None
    return app


def test_auto_speaker_enablement_respects_trigger_mode_and_flag():
    app_auto = _app_for_auto("auto")
    assert app_auto._is_auto_speaker_analysis_enabled() is True

    app_continuous = _app_for_auto("continuous")
    assert app_continuous._is_auto_speaker_analysis_enabled() is True

    app_manual = _app_for_auto("manual")
    assert app_manual._is_auto_speaker_analysis_enabled() is False

    app_disabled = _app_for_auto("auto")
    app_disabled._config["ai"]["auto_speaker_analysis_enabled"] = False
    assert app_disabled._is_auto_speaker_analysis_enabled() is False


def test_auto_speaker_candidate_text_uses_min_word_and_char_thresholds():
    app = _app_for_auto("auto")

    assert app._is_auto_speaker_candidate_text("Thanks") is False
    assert app._is_auto_speaker_candidate_text("Can you explain your ETL testing approach?") is True


def test_auto_speaker_silence_elapsed_invokes_normalization_with_auto_send():
    app = _app_for_auto("auto")
    segment = TranscriptSegment(
        text="Can you explain how you validate source and target tables in ETL?",
        source="speaker",
        timestamp=10.0,
    )

    calls = []

    def _capture(
        seg,
        text: str,
        *,
        auto_send_after: bool = False,
        auto_send_generation=None,
    ):
        calls.append((seg, text, auto_send_after, auto_send_generation))

    app._on_speaker_question_normalize_requested = _capture

    app._on_auto_speaker_silence_elapsed(segment, generation=1)

    assert len(calls) == 1
    assert calls[0][0] is segment
    assert calls[0][2] is True
    assert calls[0][3] == 1
    assert app._auto_speaker_last_signature


def test_auto_speaker_silence_elapsed_supports_user_source_segments():
    app = _app_for_auto("auto")
    segment = TranscriptSegment(
        text="Can you explain how you approach API test automation in CI pipelines?",
        source="user",
        timestamp=11.0,
    )

    calls = []

    def _capture(
        seg,
        text: str,
        *,
        auto_send_after: bool = False,
        auto_send_generation=None,
    ):
        calls.append((seg, text, auto_send_after, auto_send_generation))

    app._on_speaker_question_normalize_requested = _capture

    app._on_auto_speaker_silence_elapsed(segment, generation=1)

    assert len(calls) == 1
    assert calls[0][0] is segment
    assert calls[0][2] is True
    assert calls[0][3] == 1


def test_auto_speaker_silence_elapsed_skips_stale_generation_tokens():
    app = _app_for_auto("auto")
    segment = TranscriptSegment(
        text="Can you explain your SQL optimization strategy in production?",
        source="speaker",
        timestamp=12.0,
    )

    calls = []

    def _capture(
        seg,
        text: str,
        *,
        auto_send_after: bool = False,
        auto_send_generation=None,
    ):
        calls.append((seg, text, auto_send_after, auto_send_generation))

    app._on_speaker_question_normalize_requested = _capture

    app._on_auto_speaker_silence_elapsed(segment, generation=0)

    assert calls == []


def test_auto_speaker_duplicate_signature_is_ignored_within_cooldown():
    app = _app_for_auto("auto")
    segment = TranscriptSegment(
        text="Can you explain your SQL optimization strategy in production?",
        source="speaker",
        timestamp=15.0,
    )

    calls = []

    def _capture(
        seg,
        text: str,
        *,
        auto_send_after: bool = False,
        auto_send_generation=None,
    ):
        calls.append((seg, text, auto_send_after, auto_send_generation))

    app._on_speaker_question_normalize_requested = _capture

    app._on_auto_speaker_silence_elapsed(segment, generation=1)
    app._on_auto_speaker_silence_elapsed(segment, generation=1)

    assert len(calls) == 1
    assert calls[0][3] == 1


def test_initial_recording_question_uses_delayed_normalization_window(monkeypatch):
    app = _app_for_auto("auto")
    segment = TranscriptSegment(
        text="Can you explain how you validate source and target tables in ETL?",
        source="speaker",
        timestamp=20.0,
    )

    timer_calls = []

    class _TimerStub:
        @staticmethod
        def singleShot(delay_ms, callback):
            timer_calls.append((delay_ms, callback))

    monkeypatch.setattr("PyQt6.QtCore.QTimer", _TimerStub)

    auto_calls = []
    app._schedule_auto_speaker_analysis = lambda seg, delay_ms=None: auto_calls.append(
        (seg, delay_ms)
    )

    normalize_calls = []
    app._on_speaker_question_normalize_requested = lambda seg, text, **kwargs: normalize_calls.append(
        (seg, text, kwargs)
    )

    accepted = app._append_transcript_segment(segment, require_recording=True)

    assert accepted is True
    assert auto_calls == []
    assert app._initial_recording_question_pending is True
    assert timer_calls and timer_calls[0][0] == INITIAL_RECORDING_QUESTION_NORMALIZATION_DELAY_MS

    timer_calls[0][1]()

    assert normalize_calls == [
        (
            segment,
            segment.text,
            {
                "auto_send_after": False,
                "initial_recording_question": True,
                "recording_question_generation": 0,
            },
        )
    ]


def test_initial_recording_question_completion_clears_pending_state():
    app = _app_for_normalization_callbacks()
    app._initial_recording_question_pending = True
    app._initial_recording_question_consumed = False
    app._initial_recording_question_generation = 4

    segment = TranscriptSegment(
        text="Explain your retry logic",
        source="speaker",
        timestamp=30.0,
    )

    app._on_speaker_question_normalized(
        segment,
        "Can you explain your retry and circuit breaker strategy",
        follow_up_questions=[],
        initial_recording_question=True,
        recording_question_generation=4,
    )

    assert app._initial_recording_question_pending is False
    assert app._initial_recording_question_consumed is True
    assert segment.text == "Can you explain your retry and circuit breaker strategy?"


def test_register_normalized_segment_keeps_question_format_for_queueing():
    app = _app_for_normalization_callbacks()
    app._recording_active = False
    app._normalized_segment_items = []
    app._normalized_segment_lookup = {}

    normalized_segment = SimpleNamespace(
        segment_id="segment-1",
        normalized_text="How do you validate data quality in production",
        source="speaker",
        source_chunk_ids=[],
    )

    app._register_normalized_segment(normalized_segment)

    assert len(app._normalized_segment_items) == 1
    assert app._normalized_segment_items[0]["text"] == (
        "How do you validate data quality in production?"
    )
    assert app._session_context_store.events[0]["text"] == (
        "How do you validate data quality in production?"
    )


def test_register_normalized_segment_routes_auto_send_through_normalization_worker():
    app = _app_for_normalization_callbacks()
    app._recording_active = True
    app._ai_trigger_service = AITriggerService()

    captured = []

    def _capture(seg, text: str, **kwargs):
        captured.append((seg, text, kwargs))

    app._on_speaker_question_normalize_requested = _capture

    normalized_segment = SimpleNamespace(
        segment_id="segment-1",
        normalized_text="First one, what is the possibility if the target has more records than the source",
        source="speaker",
        source_chunk_ids=[],
    )

    app._register_normalized_segment(normalized_segment)

    assert len(app._normalized_segment_items) == 1
    assert captured and captured[0][0] is normalized_segment
    assert captured[0][1] == app._normalized_segment_items[0]["text"]
    assert captured[0][2]["auto_send_after"] is True
    assert getattr(captured[0][0], "text") == app._normalized_segment_items[0]["text"]


def test_normalized_question_callback_updates_streaming_row_before_send():
    app = _app_for_normalization_callbacks()
    app._ai_trigger_service = AITriggerService()
    app._generate_ai_response = lambda force_follow_up=False: None
    app._normalized_segment_items = [
        {
            "segment_id": "segment-1",
            "text": "First one, what is the possibility if the target has more records than the source",
            "status": "pending",
            "source": "speaker",
            "source_chunk_ids": [],
        }
    ]
    app._normalized_segment_lookup = {"segment-1": app._normalized_segment_items[0]}

    segment = SimpleNamespace(
        segment_id="segment-1",
        text="First one, what is the possibility if the target has more records than the source",
        source="speaker",
        confidence=1.0,
        timestamp=30.0,
    )

    app._on_speaker_question_normalized(
        segment,
        "What is the likelihood that the target has more records than the source?",
        follow_up_questions=["How would you validate row counts between systems?"],
        auto_send=True,
    )

    item = app._normalized_segment_items[0]
    assert item["text"] == "What is the likelihood that the target has more records than the source?"
    assert item["status"] == "sent"
    assert app._auto_primary_question_sent is True
    assert app._active_primary_question_text == (
        "What is the likelihood that the target has more records than the source?"
    )
    assert app._session_context_store.events[-1]["kind"] == "auto_ai_request"


def test_suggested_follow_up_selection_forces_refine_prompt_submission():
    app = GhostMicApp.__new__(GhostMicApp)
    app._window = None
    captured = []

    def _capture(prompt: str, refine: bool) -> None:
        captured.append((prompt, refine))

    app._on_ai_text_prompt_submitted = _capture

    app._on_suggested_follow_up_selected(
        "Can you walk me through a production incident and how you resolved it?"
    )

    assert captured == [
        (
            "Can you walk me through a production incident and how you resolved it?",
            True,
        )
    ]


def test_suggested_follow_up_selection_shows_next_three_follow_ups():
    class _WindowStub:
        def __init__(self) -> None:
            self.current_question = ""
            self.follow_ups = []
            self.sent_confirmation = ""

        def set_current_question_text(self, text: str, *, force: bool = False) -> None:
            self.current_question = text

        def set_question_follow_up_suggestions(self, questions):
            self.follow_ups = list(questions)

        def show_follow_up_sent_confirmation(self, question: str) -> None:
            self.sent_confirmation = question

    app = GhostMicApp.__new__(GhostMicApp)
    app._window = _WindowStub()

    refresh_calls = []
    app._start_follow_up_suggestion_refresh = lambda question: refresh_calls.append(question)

    sent_prompts = []
    app._on_ai_text_prompt_submitted = lambda prompt, refine: sent_prompts.append((prompt, refine))

    selected = "How do you handle schema drift in production ETL pipelines?"
    app._on_suggested_follow_up_selected(selected)

    assert app._window.current_question == selected
    assert app._window.sent_confirmation == selected
    assert len(app._window.follow_ups) == 3
    assert all(question.endswith("?") for question in app._window.follow_ups)
    assert refresh_calls == [selected]
    assert sent_prompts == [(selected, True)]
