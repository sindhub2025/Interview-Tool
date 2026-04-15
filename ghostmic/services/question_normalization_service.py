"""Speaker-question normalization using the configured AI backend."""

from __future__ import annotations

import random
import time
from typing import Optional

from ghostmic.utils.errors import is_rate_limited as _is_rate_limited_shared
from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 20.0

NORMALIZE_SYSTEM_PROMPT = (
    "You rewrite noisy speech-to-text interview questions into clear written English. "
    "Keep the original meaning. Fix grammar, punctuation, and obvious transcription mistakes. "
    "Do not answer the question. Return only one cleaned question sentence."
)


try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:  # pragma: no cover - fallback for non-Qt environments
    QThread = object  # type: ignore[misc,assignment]
    pyqtSignal = None  # type: ignore[assignment]


def _resolve_backend(ai_config: dict) -> str:
    backend = str(ai_config.get("main_backend") or ai_config.get("backend") or "groq").strip().lower()
    expose_openai = bool(ai_config.get("expose_openai_provider", False))
    if backend == "openai" and expose_openai:
        return "openai"
    return "groq"


def _extract_response_text(response) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                if text:
                    pieces.append(text)
        return " ".join(pieces).strip()

    return ""


def normalize_question(question_text: str, ai_config: dict) -> str:
    """Return a normalized interview question using the configured backend."""
    cleaned_question = str(question_text).strip()
    if not cleaned_question:
        raise ValueError("Question text is empty.")

    try:
        from openai import OpenAI  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("openai package not installed. Run: pip install openai") from exc

    backend = _resolve_backend(ai_config)
    retries = int(ai_config.get("question_normalization_retries", DEFAULT_MAX_RETRIES))
    retries = max(1, retries)
    base_delay = float(
        ai_config.get("question_normalization_retry_delay", DEFAULT_BASE_DELAY_SECONDS)
    )
    timeout = float(
        ai_config.get("question_normalization_timeout", DEFAULT_TIMEOUT_SECONDS)
    )

    if backend == "openai":
        api_key = str(ai_config.get("openai_api_key", "")).strip()
        if not api_key:
            raise ValueError("OpenAI API key not set. Add it in Settings -> AI.")
        model = str(ai_config.get("openai_model", "gpt-5-mini")).strip() or "gpt-5-mini"
        client = OpenAI(api_key=api_key)
    else:
        api_key = str(ai_config.get("groq_api_key", "")).strip()
        if not api_key:
            raise ValueError("Groq API key not set. Add it in Settings -> AI.")
        model = str(ai_config.get("groq_model", "llama-3.3-70b-versatile")).strip() or "llama-3.3-70b-versatile"
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    session_context = str(ai_config.get("session_context", "")).strip()
    context_line = f"\nSession context: {session_context}" if session_context else ""
    user_prompt = (
        "Normalize this speaker question transcript. "
        "Output only the cleaned question text, no labels and no explanation.\n"
        f"Raw transcript: {cleaned_question}{context_line}"
    )

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": NORMALIZE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=120,
                timeout=timeout,
                stream=False,
            )
            normalized = _extract_response_text(response)
            if normalized:
                return normalized
            raise RuntimeError("Normalization returned an empty response.")
        except Exception as exc:  # pylint: disable=broad-except
            can_retry = (
                _is_rate_limited_shared(exc)
                and attempt < retries - 1
            )
            if can_retry:
                delay = min(8.0, base_delay * (2**attempt)) + random.uniform(0.0, 0.25)
                logger.warning(
                    "Question normalization rate-limited; retrying in %.2fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    retries,
                )
                time.sleep(delay)
                continue
            raise

    raise RuntimeError("Question normalization failed after retries.")


class QuestionNormalizationWorker(QThread):  # type: ignore[misc]
    """Background worker that normalizes one speaker question."""

    if pyqtSignal is not None:
        normalized_ready = pyqtSignal(str)
        normalization_error = pyqtSignal(str)

    def __init__(self, ai_config: dict, question_text: str, parent=None) -> None:
        super().__init__(parent)
        self._ai_config = dict(ai_config or {})
        self._question_text = str(question_text or "")

    def run(self) -> None:
        try:
            normalized = normalize_question(self._question_text, self._ai_config)
            self.normalized_ready.emit(normalized)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Question normalization failed: %s", exc)
            self.normalization_error.emit(str(exc))
