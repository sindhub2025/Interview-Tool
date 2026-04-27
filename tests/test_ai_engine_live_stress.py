"""Optional live stress test for Groq-only response generation."""

import os
import threading
import time

import pytest

import ghostmic.core.ai_engine as ai_engine_module
from ghostmic.core.ai_engine import AIThread
from ghostmic.core.transcription_engine import TranscriptSegment


RUN_LIVE_STRESS = os.getenv("RUN_LIVE_GROQ_STRESS", "").strip().lower() in {
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
    reason="Set RUN_LIVE_GROQ_STRESS=1 to run this live stress test.",
)
def test_live_groq_stress_generates_responses(monkeypatch):
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if not groq_key:
        pytest.skip("Requires GROQ_API_KEY for live stress test.")

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
        "fallback_backend": "groq",
        "enable_fallback": False,
        "groq_api_key": groq_key,
        "groq_model": os.getenv("GROQ_MODEL", "llama-4-maverick-17b-128e-instruct"),
        "temperature": 0.2,
        "stream_timeout": 45.0,
    }
    thread._stop_event = threading.Event()

    initial_responses = []
    thread._on_ready = initial_responses.append

    start = time.perf_counter()
    for idx in range(1, rounds + 1):
        question = _build_interview_question(idx)
        transcript = [TranscriptSegment(text=question, source="speaker")]
        thread._generate(transcript, is_new_topic=True)
    elapsed = time.perf_counter() - start

    assert len(initial_responses) == rounds

    max_seconds = float(os.getenv("LIVE_STRESS_MAX_SECONDS", "240"))
    assert elapsed <= max_seconds, (
        f"Live Groq stress run took {elapsed:.2f}s which exceeded "
        f"LIVE_STRESS_MAX_SECONDS={max_seconds:.2f}s."
    )
