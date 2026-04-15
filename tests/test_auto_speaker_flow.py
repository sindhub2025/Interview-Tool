"""Unit tests for automatic speaker-question analysis flow."""

from ghostmic.domain import TranscriptSegment
from ghostmic.main import GhostMicApp


class _NoopLogger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


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
    app._question_normalization_worker = None
    app._auto_speaker_silence_generation = 1
    app._auto_speaker_last_signature = ""
    app._auto_speaker_last_trigger_ts = 0.0
    app._logger = _NoopLogger()
    app._window = None
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

    def _capture(seg, text: str, *, auto_send_after: bool = False):
        calls.append((seg, text, auto_send_after))

    app._on_speaker_question_normalize_requested = _capture

    app._on_auto_speaker_silence_elapsed(segment, generation=1)

    assert len(calls) == 1
    assert calls[0][0] is segment
    assert calls[0][2] is True
    assert app._auto_speaker_last_signature


def test_auto_speaker_silence_elapsed_skips_stale_generation_tokens():
    app = _app_for_auto("auto")
    segment = TranscriptSegment(
        text="Can you explain your SQL optimization strategy in production?",
        source="speaker",
        timestamp=12.0,
    )

    calls = []

    def _capture(seg, text: str, *, auto_send_after: bool = False):
        calls.append((seg, text, auto_send_after))

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

    def _capture(seg, text: str, *, auto_send_after: bool = False):
        calls.append((seg, text, auto_send_after))

    app._on_speaker_question_normalize_requested = _capture

    app._on_auto_speaker_silence_elapsed(segment, generation=1)
    app._on_auto_speaker_silence_elapsed(segment, generation=1)

    assert len(calls) == 1
