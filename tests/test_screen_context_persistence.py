"""Unit tests for sticky screen-summary context retention."""

import threading

from ghostmic.domain import TranscriptSegment
from ghostmic.main import GhostMicApp


class _NoopLogger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


class _SessionStoreStub:
    def __init__(self) -> None:
        self.screen_summaries = []

    def append_screen_summary(self, text: str) -> None:
        self.screen_summaries.append(text)

    def get_latest_organized_context(self, max_chars: int = 1200) -> str:
        return ""

    def build_compaction_source(self, max_events: int = 20, max_chars: int = 1200):
        return {"text": ""}


class _AIThreadStub:
    def __init__(self) -> None:
        self.requests = []

    def request_response(self, transcript, is_new_topic=False, runtime_context_tail=""):
        self.requests.append(
            {
                "transcript": list(transcript),
                "is_new_topic": is_new_topic,
                "runtime_context_tail": runtime_context_tail,
            }
        )
        return True

    def last_reject_reason(self) -> str:
        return "unknown"


class _PanelStub:
    def __init__(self) -> None:
        self.text = ""

    def finish_response(self, text: str) -> None:
        self.text = text


class _ControlsStub:
    def __init__(self) -> None:
        self.busy = None
        self.status = None

    def set_screen_analysis_busy(self, busy: bool) -> None:
        self.busy = busy

    def set_status(self, text: str, color: str) -> None:
        self.status = (text, color)


class _WindowStub:
    def __init__(self) -> None:
        self.ai_panel = _PanelStub()
        self.controls = _ControlsStub()


def _app_for_screen_context() -> GhostMicApp:
    app = GhostMicApp.__new__(GhostMicApp)
    app._logger = _NoopLogger()
    app._config = {"ai": {}}
    app._window = None
    app._transcript_lock = threading.Lock()
    app._session_context_store = _SessionStoreStub()
    app._transcript_history = []
    app._ai_context_history = []
    app._last_question_text = ""
    app._last_ai_response_text = ""
    app._active_screen_summary_text = ""
    app._active_screen_anchor_question = ""
    app._ensure_ai_thread = lambda: None
    app._ai_thread = _AIThreadStub()
    return app


def test_screen_analysis_ready_sets_active_context_and_resets_anchor():
    app = _app_for_screen_context()
    app._window = _WindowStub()
    app._active_screen_anchor_question = "Old anchor"

    app._on_screen_analysis_ready("The stack trace points to failing SQL migration.")

    assert app._active_screen_summary_text == "The stack trace points to failing SQL migration."
    assert app._active_screen_anchor_question == ""
    assert app._session_context_store.screen_summaries == [
        "The stack trace points to failing SQL migration."
    ]


def test_resolve_active_screen_context_keeps_context_when_topic_is_related():
    app = _app_for_screen_context()
    app._active_screen_summary_text = "A failing pytest assertion is visible on screen."

    context = app._resolve_active_screen_summary_context(
        "Can you explain why that assertion fails?",
        topic_shift_classifier=lambda _prev, _new: False,
    )

    assert context == "A failing pytest assertion is visible on screen."
    assert app._active_screen_anchor_question == "Can you explain why that assertion fails?"


def test_resolve_active_screen_context_clears_context_after_topic_shift():
    app = _app_for_screen_context()
    app._active_screen_summary_text = "The current screen shows a failing API contract test."
    app._active_screen_anchor_question = "Why is this API contract test failing?"

    context = app._resolve_active_screen_summary_context(
        "How would you design a Kafka topic partition strategy?",
        topic_shift_classifier=lambda _prev, _new: True,
    )

    assert context == ""
    assert app._active_screen_summary_text == ""
    assert app._active_screen_anchor_question == ""


def test_resolve_active_screen_context_force_follow_up_keeps_context():
    app = _app_for_screen_context()
    app._active_screen_summary_text = "The screenshot contains an SQL deadlock error."
    app._active_screen_anchor_question = "What causes this SQL deadlock?"

    context = app._resolve_active_screen_summary_context(
        "Give me more detail on that.",
        force_follow_up=True,
        topic_shift_classifier=lambda _prev, _new: True,
    )

    assert context == "The screenshot contains an SQL deadlock error."
    assert app._active_screen_anchor_question == "Give me more detail on that."


def test_generate_ai_response_includes_active_screen_context_in_runtime_tail():
    app = _app_for_screen_context()
    app._active_screen_summary_text = "UI shows ValueError in resume parsing module."
    app._transcript_history = [
        TranscriptSegment(
            text="How do I fix the resume parser error?",
            source="speaker",
            confidence=1.0,
        )
    ]

    app._generate_ai_response(force_follow_up=False)

    assert len(app._ai_thread.requests) == 1
    runtime_tail = app._ai_thread.requests[0]["runtime_context_tail"]
    assert "Active Screen Context:" in runtime_tail
    assert "UI shows ValueError in resume parsing module." in runtime_tail
