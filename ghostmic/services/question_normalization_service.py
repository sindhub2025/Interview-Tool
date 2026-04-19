"""Speaker-question normalization using the configured AI backend."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
import re
import time
from typing import List, Optional

from ghostmic.utils.errors import is_rate_limited as _is_rate_limited_shared
from ghostmic.utils.logger import get_logger
from ghostmic.utils.text_processing import ensure_question_format

logger = get_logger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_FOLLOW_UP_COUNT = 3

NORMALIZE_SYSTEM_PROMPT = (
    "You are an interview assistant that rewrites noisy speech-to-text interview questions "
    "into clear written English and predicts realistic follow-up interview questions. "
    "Keep the original meaning. Fix grammar, punctuation, and obvious transcription mistakes. "
    "Do not answer any question. Return strict JSON only."
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


def _normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _extract_json_payload(raw_text: str) -> Optional[dict]:
    text = str(raw_text or "").strip()
    if not text:
        return None

    candidates = [text]
    fenced_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())

    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        candidates.append(brace_match.group(0).strip())

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    return None


def _fallback_follow_up_questions(normalized_question: str) -> List[str]:
    topic = _normalize_whitespace(normalized_question).rstrip(".?!")
    if topic:
        return [
            f"Can you walk me through a real project example where you applied this: {topic}?",
            "What trade-offs did you evaluate, and why did you choose your final approach?",
            "How would you measure success and troubleshoot issues if this fails in production?",
        ]

    return [
        "Can you walk me through a real project example where you handled this end-to-end?",
        "What trade-offs did you evaluate, and why did you choose your final approach?",
        "How would you measure success and troubleshoot issues if this fails in production?",
    ]


def _sanitize_follow_up_questions(
    raw_questions,
    normalized_question: str,
    *,
    follow_up_count: int = DEFAULT_FOLLOW_UP_COUNT,
) -> List[str]:
    cleaned: List[str] = []
    seen: set[str] = set()
    normalized_anchor = _normalize_whitespace(normalized_question).rstrip(".?!").lower()

    if not isinstance(raw_questions, list):
        raw_questions = []

    for raw in raw_questions:
        candidate = _normalize_whitespace(str(raw or ""))
        if not candidate:
            continue
        if len(candidate.split()) < 4:
            continue
        if not candidate.endswith("?"):
            candidate = f"{candidate.rstrip('.!')}?"

        canonical = candidate.rstrip(".?!").lower()
        if canonical == normalized_anchor:
            continue
        if canonical in seen:
            continue

        seen.add(canonical)
        cleaned.append(candidate)
        if len(cleaned) >= follow_up_count:
            break

    if len(cleaned) >= follow_up_count:
        return cleaned[:follow_up_count]

    for fallback in _fallback_follow_up_questions(normalized_question):
        canonical = fallback.rstrip(".?!").lower()
        if canonical in seen:
            continue
        seen.add(canonical)
        cleaned.append(fallback)
        if len(cleaned) >= follow_up_count:
            break

    return cleaned[:follow_up_count]


@dataclass(frozen=True)
class QuestionNormalizationResult:
    normalized_question: str
    follow_up_questions: List[str]


def _parse_normalization_result(
    model_text: str,
    *,
    fallback_question: str,
) -> QuestionNormalizationResult:
    payload = _extract_json_payload(model_text)
    if payload is not None:
        normalized = ensure_question_format(payload.get("normalized_question", ""))
        if not normalized:
            normalized = ensure_question_format(fallback_question)
        follow_ups = _sanitize_follow_up_questions(
            payload.get("follow_up_questions", []),
            normalized,
        )
        return QuestionNormalizationResult(
            normalized_question=normalized,
            follow_up_questions=follow_ups,
        )

    normalized = ensure_question_format(model_text)
    if not normalized:
        normalized = ensure_question_format(fallback_question)
    follow_ups = _sanitize_follow_up_questions([], normalized)
    return QuestionNormalizationResult(
        normalized_question=normalized,
        follow_up_questions=follow_ups,
    )


def normalize_question_with_followups(
    question_text: str,
    ai_config: dict,
) -> QuestionNormalizationResult:
    """Return normalized question text plus likely follow-up interview questions."""
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
        "Normalize this speaker interview question transcript and generate likely interviewer follow-up questions. "
        "Return strict JSON only with this exact schema: "
        '{"normalized_question":"...","follow_up_questions":["...","...","..."]}. '
        "Rules: "
        "1) Keep the normalized question faithful to the speaker intent. "
        "2) follow_up_questions must contain exactly 3 realistic real-world interview follow-up questions. "
        "3) Do not answer any question. "
        "4) Avoid duplicates and keep each follow-up concise.\n"
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
                temperature=0.2,
                max_tokens=280,
                timeout=timeout,
                stream=False,
            )
            raw_result = _extract_response_text(response)
            if raw_result:
                return _parse_normalization_result(
                    raw_result,
                    fallback_question=cleaned_question,
                )

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


def normalize_question(question_text: str, ai_config: dict) -> str:
    """Return a normalized interview question using the configured backend."""
    result = normalize_question_with_followups(question_text, ai_config)
    return result.normalized_question


class QuestionNormalizationWorker(QThread):  # type: ignore[misc]
    """Background worker that normalizes one speaker question."""

    if pyqtSignal is not None:
        normalized_ready = pyqtSignal(str)
        normalized_with_followups_ready = pyqtSignal(str, list)
        normalization_error = pyqtSignal(str)

    def __init__(self, ai_config: dict, question_text: str, parent=None) -> None:
        super().__init__(parent)
        self._ai_config = dict(ai_config or {})
        self._question_text = str(question_text or "")

    def run(self) -> None:
        try:
            result = normalize_question_with_followups(
                self._question_text,
                self._ai_config,
            )
            if pyqtSignal is not None:
                self.normalized_with_followups_ready.emit(
                    result.normalized_question,
                    list(result.follow_up_questions),
                )
                self.normalized_ready.emit(result.normalized_question)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Question normalization failed: %s", exc)
            self.normalization_error.emit(str(exc))
