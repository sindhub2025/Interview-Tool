"""Optional live stress test for two-stage Groq -> Gemini handoff."""

import os
import threading
import time

import pytest

import ghostmic.core.ai_engine as ai_engine_module
from ghostmic.core.ai_engine import AIThread
from ghostmic.core.transcription_engine import TranscriptSegment


RUN_LIVE_STRESS = os.getenv("RUN_LIVE_GROQ_GEMINI_STRESS", "").strip().lower() in {
    "1",
    "true",
    "yes",
}


def _build_interview_question(index: int) -> str:
    return (
        f"Interview question {index}: You have a production ETL API with burst traffic. "
        "How would you design retries, backoff, and idempotency so the pipeline stays reliable?"
    )


@pytest.mark.skipif(
    not RUN_LIVE_STRESS,
    reason="Set RUN_LIVE_GROQ_GEMINI_STRESS=1 to run this live stress test.",
)
def test_live_two_stage_handoff_stress_avoids_groq_fallback(monkeypatch):
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not groq_key or not gemini_key:
        pytest.skip("Requires GROQ_API_KEY and GEMINI_API_KEY for live stress test.")

    monkeypatch.setattr(ai_engine_module, "pyqtSignal", None)

    rounds_raw = os.getenv("LIVE_STRESS_ROUNDS", "12").strip()
    try:
        rounds = int(rounds_raw)
    except ValueError:
        rounds = 12
    rounds = max(1, min(rounds, 40))

    thread = AIThread.__new__(AIThread)
    thread._config = {
        "backend": "groq",
        "main_backend": "groq",
        "fallback_backend": "gemini",
        "enable_fallback": True,
        "two_stage_enabled": True,
        "groq_api_key": groq_key,
        "gemini_api_key": gemini_key,
        "groq_model": os.getenv("GROQ_MODEL", "llama-4-maverick-17b-128e-instruct"),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
        "temperature": 0.2,
        "stream_timeout": 45.0,
    }
    thread._stop_event = threading.Event()

    initial_responses = []
    thread._on_ready = initial_responses.append

    fallback_count = 0
    groq_stage_failures = []

    original_run_groq_initial = AIThread._run_groq_initial
    original_generate_gemini = AIThread._generate_gemini

    def _tracked_run_groq_initial(context, system_prompt, temperature):
        try:
            return original_run_groq_initial(thread, context, system_prompt, temperature)
        except Exception as exc:  # pylint: disable=broad-except
            groq_stage_failures.append(str(exc))
            raise

    def _tracked_fallback_generate_gemini(context, system_prompt, temperature):
        nonlocal fallback_count
        fallback_count += 1
        return original_generate_gemini(thread, context, system_prompt, temperature)

    monkeypatch.setattr(thread, "_run_groq_initial", _tracked_run_groq_initial)
    monkeypatch.setattr(thread, "_generate_gemini", _tracked_fallback_generate_gemini)

    start = time.perf_counter()
    for idx in range(1, rounds + 1):
        question = _build_interview_question(idx)
        transcript = [TranscriptSegment(text=question, source="speaker")]
        thread._generate(transcript, is_new_topic=True)
    elapsed = time.perf_counter() - start

    assert len(initial_responses) == rounds
    assert not groq_stage_failures, (
        "Groq stage failed during two-stage stress run. "
        f"Failures: {groq_stage_failures}"
    )
    assert fallback_count == 0, (
        "Groq stage returned empty/failure at least once, triggering "
        "Gemini-only fallback; this indicates Groq pressure or availability issues."
    )

    max_seconds = float(os.getenv("LIVE_STRESS_MAX_SECONDS", "240"))
    assert elapsed <= max_seconds, (
        f"Live two-stage stress run took {elapsed:.2f}s which exceeded "
        f"LIVE_STRESS_MAX_SECONDS={max_seconds:.2f}s."
    )
