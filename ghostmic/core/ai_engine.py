"""
AI response engine.

Groq is the active backend exposed in the app.
OpenAI code paths are intentionally preserved for future re-enable.

Runs in a dedicated QThread and emits streaming response tokens.
"""

from __future__ import annotations

import queue
import random
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from ghostmic.utils.errors import is_rate_limited as _is_rate_limited_shared

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    QThread = object  # type: ignore[misc,assignment]
    pyqtSignal = None  # type: ignore[assignment]

from ghostmic.core.transcription_engine import TranscriptSegment
from ghostmic.utils.resume_context import (
    apply_resume_corrections,
    build_resume_context_summary,
    is_resume_related_text,
)
from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are a real-time interview/meeting assistant. You are helping the user \
respond to questions and participate in a conversation.

Based on the conversation transcript provided, generate:
1. A concise, natural-sounding suggested response to the last question/topic
2. Key talking points the user should mention

Rules:
- Answer ONLY the current question shown in the transcript. Do NOT reference, \
explain, or repeat content from any previous questions or answers unless \
the interviewer explicitly asks you to.
- Keep responses concise (2-4 sentences max for the main response)
- Sound natural and conversational, not robotic
- If it's a technical question, provide accurate technical details
- If context is unclear, provide the most likely helpful response
- Format: Start with the direct response, then bullet points for key points
- Do NOT include phrases like "Based on the transcript" or "Here's a suggestion"
- Write in first person as if the user is speaking
"""

DEBOUNCE_SECONDS: float = 3.0
MAX_CONTEXT_SEGMENTS: int = 10
AI_QUEUE_MAXSIZE: int = 8
RUNTIME_RATE_LIMIT_RETRIES: int = 3
RUNTIME_RATE_LIMIT_BASE_DELAY: float = 1.0
RUNTIME_RATE_LIMIT_MAX_DELAY: float = 8.0
RESUME_CORRECTION_HIGH_THRESHOLD: float = 0.87
RESUME_CORRECTION_MEDIUM_THRESHOLD: float = 0.74

# Jaccard similarity threshold below which two questions are considered
# different topics.  0.25 means less than 25 % word overlap → new topic.
TOPIC_SHIFT_THRESHOLD: float = 0.25

# Common English stop-words excluded from topic-shift comparison.
_STOP_WORDS: frozenset = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "from",
    "up", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "out", "and", "but", "or", "nor",
    "so", "yet", "both", "either", "neither", "not", "only", "own",
    "same", "than", "too", "very", "just", "that", "this", "these",
    "those", "which", "who", "whom", "what", "how", "when", "where",
    "why", "if", "as", "its", "it", "your", "you", "me", "my",
    "we", "our", "they", "their", "them", "he", "she", "his", "her",
    "i", "im", "ive", "dont", "doesnt", "didnt", "wont", "wouldnt",
    "between", "difference", "explain", "describe", "give",
})

ETL_CONTEXT_KEYWORDS = (
    "etl",
    "data warehouse",
    "data warehousing",
    "source table",
    "target table",
    "staging",
    "pipeline",
    "fact table",
    "dimension table",
)
FOLLOW_UP_PATTERNS = (
    r"\bexplain more\b",
    r"\bexplain further\b",
    r"\bexplain further on (?:this|that) topic\b",
    r"\belaborate\b",
    r"\belaborate more\b",
    r"\bcan you elaborate\b",
    r"\bcan you explain (?:that|this|it) more\b",
    r"\btell me more\b",
    r"\bmore detail(?:s)?\b",
    r"\bgo deeper\b",
    r"\bexpand on (?:that|this|it|the topic)\b",
    r"\bcontinue\b",
    r"\bwhat else\b",
    r"\bcan you give (?:an?|another) example\b",
    r"\bgive (?:an?|another) example\b",
    r"\b(?:what|how) about (?:this|that|it)\b",
    r"\bwhy (?:is|was) (?:this|that|it)\b",
    r"\bhow so\b",
    r"\bcan you clarify\b",
    r"\bclarify (?:this|that|it)\b",
    r"\bwhat do you mean\b",
    r"\bwhat does (?:this|that|it) mean\b",
    r"\bcould you expand on (?:this|that|it)\b",
    r"\bwalk me through (?:this|that|it)\b",
)
FOLLOW_UP_CONTEXT_PREFIX = "Previous AI answer context:"
PRIOR_QUESTION_CONTEXT_PREFIX = "Previous speaker question context:"
FOLLOW_UP_REFERENCE_TERMS: frozenset = frozenset(
    {
        "this",
        "that",
        "it",
        "those",
        "these",
        "them",
        "same",
        "earlier",
        "previous",
        "above",
        "example",
        "examples",
        "again",
        "more",
        "detail",
        "details",
        "part",
        "step",
        "steps",
        "approach",
    }
)
SELECTION_FOLLOW_UP_PATTERNS = (
    r"\bwhich one\b",
    r"\bwhich (?:is|one) (?:better|best)\b",
    r"\bwhich (?:should|would) (?:i|we) use\b",
    r"\bshould (?:i|we) (?:use|choose|pick)\b",
    r"\bwhat should (?:i|we) use\b",
    r"\bwhich (?:would you|do you) (?:use|choose|pick|recommend)\b",
    r"\bwhat (?:would you|do you) recommend\b",
)
GENERIC_SELECTION_TERMS: frozenset = frozenset(
    {
        "better",
        "best",
        "choose",
        "choice",
        "option",
        "prefer",
        "recommend",
        "recommended",
        "versus",
        "vs",
    }
)
SHORT_FOLLOW_UP_MAX_TOKENS: int = 12


class AIThread(QThread):  # type: ignore[misc]
    """Generates AI responses in a background thread.

    Emits:
        ``ai_response_chunk(str)`` – individual streaming tokens.
        ``ai_response_ready(str)`` – full assembled response.
        ``ai_error(str)`` – error message.
        ``ai_thinking()`` – emitted when generation starts.

    Args:
        config: The AI section of config.json.
        on_chunk: Optional streaming callback.
        on_ready: Optional completion callback.
    """

    if pyqtSignal is not None:
        ai_response_chunk = pyqtSignal(str)
        ai_response_ready = pyqtSignal(str)
        ai_error = pyqtSignal(str)
        ai_thinking = pyqtSignal()

    def __init__(
        self,
        config: dict,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_ready: Optional[Callable[[str], None]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._on_chunk = on_chunk
        self._on_ready = on_ready
        self._stop_event = threading.Event()
        # Queue carries (transcript, is_new_topic) tuples.
        self._queue: "queue.Queue[tuple]" = queue.Queue(
            maxsize=AI_QUEUE_MAXSIZE
        )
        self._last_request_time: float = 0.0
        self._last_reject_reason: str = ""

    def stop(self) -> None:
        self._stop_event.set()

    def request_response(
        self,
        transcript: List[TranscriptSegment],
        is_new_topic: bool = False,
    ) -> bool:
        """Queue a response-generation request.

        Applies debounce: requests closer than DEBOUNCE_SECONDS apart are
        discarded.

        Args:
            transcript: Recent transcript segments for context.
            is_new_topic: When True the AI will not use any previous-answer
                context so the response focuses strictly on the new question.

        Returns:
            True when accepted into the queue, otherwise False.
        """
        now = time.time()
        if now - self._last_request_time < DEBOUNCE_SECONDS:
            logger.debug("AIThread: debounce – request ignored.")
            self._last_reject_reason = "debounced"
            return False
        self._last_request_time = now
        payload = (transcript, is_new_topic)
        try:
            self._queue.put_nowait(payload)
            self._last_reject_reason = ""
            return True
        except queue.Full:
            # Keep freshest context by dropping oldest queued work.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                logger.debug("AIThread: queue was full but empty on readback.")
                self._last_reject_reason = "queue-full"
                return False

            try:
                self._queue.put_nowait(payload)
                logger.debug("AIThread: queue full – dropped oldest queued request.")
                self._last_reject_reason = ""
                return True
            except queue.Full:
                logger.debug("AIThread: queue remained full – dropping request.")
                self._last_reject_reason = "queue-full"
                return False

    def last_reject_reason(self) -> str:
        """Return the most recent queue-rejection reason."""
        return self._last_reject_reason or "unknown"

    def clear_pending_requests(self) -> int:
        """Drop queued requests that have not started execution yet.

        Returns:
            Number of dropped queued requests.
        """
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        if dropped:
            logger.debug("AIThread: dropped %d pending request(s).", dropped)
        return dropped

    def update_config(self, config: dict) -> None:
        """Replace the AI configuration at runtime."""
        self._config = config

    def run(self) -> None:
        self._stop_event.clear()
        logger.info("AIThread: started.")
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=0.2)
                transcript, is_new_topic = payload if isinstance(payload, tuple) else (payload, False)
                logger.info(
                    "AIThread: got request from queue with %d segments, is_new_topic=%s",
                    len(transcript),
                    is_new_topic,
                )
            except queue.Empty:
                continue

            # Removed ai_thinking signal - show_thinking will be called from main thread
            # when request is accepted instead
            logger.info("AIThread: calling _generate() now...")
            try:
                self._generate(transcript, is_new_topic=is_new_topic)
                logger.info("AIThread: _generate() completed")
            except Exception as e:
                logger.error("AIThread: _generate() raised exception: %s", e, exc_info=True)

        logger.info("AIThread: stopped.")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _resolve_active_backend(self, requested_backend: Optional[str]) -> str:
        """Return the active backend for execution.

        OpenAI is available only when expose_openai_provider is enabled.
        Otherwise, runtime execution is normalized to Groq while OpenAI
        code remains in the codebase.
        """
        expose_openai = bool(self._config.get("expose_openai_provider", False))

        if requested_backend in (None, "", "groq"):
            return "groq"

        if requested_backend == "openai":
            if expose_openai:
                return "openai"
            logger.info("AIThread: OpenAI backend is hidden; using 'groq'")
            return "groq"

        logger.info(
            "AIThread: backend %r is unsupported; using 'groq'",
            requested_backend,
        )
        return "groq"

    def _generate(self, transcript: List[TranscriptSegment], is_new_topic: bool = False) -> None:
        # Keep old/new config keys for compatibility, then normalize to active provider.
        logger.info(
            "AIThread: _generate() starting with %d segments, is_new_topic=%s",
            len(transcript),
            is_new_topic,
        )
        requested_main = self._config.get("main_backend") or self._config.get("backend", "groq")
        requested_fallback = self._config.get("fallback_backend", "groq")
        main_backend = self._resolve_active_backend(requested_main)
        fallback_backend = self._resolve_active_backend(requested_fallback)
        enable_fallback = bool(self._config.get("enable_fallback", False))
        
        logger.info("AIThread: using main_backend=%s, fallback_backend=%s, enable_fallback=%s", 
                   main_backend, fallback_backend, enable_fallback)
        
        session_context = str(self._config.get("session_context", "")).strip()
        resume_context_enabled = bool(self._config.get("resume_context_enabled", True))
        resume_profile_raw = self._config.get("resume_profile")
        resume_profile = (
            resume_profile_raw
            if resume_context_enabled and isinstance(resume_profile_raw, dict)
            else None
        )
        correction_high_raw = self._config.get(
            "resume_correction_threshold_high",
            RESUME_CORRECTION_HIGH_THRESHOLD,
        )
        try:
            correction_high_threshold = float(correction_high_raw)
        except (TypeError, ValueError):
            logger.warning(
                "AIThread: invalid resume_correction_threshold_high=%r; using default %.2f",
                correction_high_raw,
                RESUME_CORRECTION_HIGH_THRESHOLD,
            )
            correction_high_threshold = RESUME_CORRECTION_HIGH_THRESHOLD

        correction_medium_raw = self._config.get(
            "resume_correction_threshold_medium",
            RESUME_CORRECTION_MEDIUM_THRESHOLD,
        )
        try:
            correction_medium_threshold = float(correction_medium_raw)
        except (TypeError, ValueError):
            logger.warning(
                "AIThread: invalid resume_correction_threshold_medium=%r; using default %.2f",
                correction_medium_raw,
                RESUME_CORRECTION_MEDIUM_THRESHOLD,
            )
            correction_medium_threshold = RESUME_CORRECTION_MEDIUM_THRESHOLD
        context = self._build_context(
            transcript,
            session_context=session_context,
            is_new_topic=is_new_topic,
            resume_profile=resume_profile,
            correction_high_threshold=correction_high_threshold,
            correction_medium_threshold=correction_medium_threshold,
        )
        logger.info("AIThread: context built (length=%d, is_new_topic=%s)", len(context), is_new_topic)
        system_prompt = self._build_system_prompt(
            self._config.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            session_context,
            resume_profile=resume_profile,
        )
        temperature = float(self._config.get("temperature", 0.7))
        logger.info("AIThread: temperature=%.1f", temperature)

        # Try main backend first
        logger.info("AIThread: trying main backend: %s", main_backend)
        success = self._try_generate(main_backend, context, system_prompt, temperature)
        logger.info("AIThread: main backend result: success=%s", success)
        
        # If main failed and fallback is enabled, try fallback backend
        if not success and enable_fallback and fallback_backend != main_backend:
            logger.info("AIThread: main backend (%s) failed, trying fallback (%s)", 
                       main_backend, fallback_backend)
            success = self._try_generate(fallback_backend, context, system_prompt, temperature)
        
        if not success:
            logger.error("AIThread: both main and fallback backends failed")

    def _try_generate(self, backend: str, context: str, system_prompt: str, 
                     temperature: float) -> bool:
        """Try to generate response using specified backend.
        
        Returns True if successful, False if failed.
        """
        logger.info("AIThread: _try_generate() starting for backend=%s", backend)
        retries = int(self._config.get("runtime_rate_limit_retries", RUNTIME_RATE_LIMIT_RETRIES))
        base_delay = float(
            self._config.get("runtime_rate_limit_retry_delay", RUNTIME_RATE_LIMIT_BASE_DELAY)
        )
        retries = max(1, retries)
        logger.info("AIThread: _try_generate() retries=%d, base_delay=%.1f", retries, base_delay)

        for attempt in range(retries):
            logger.info("AIThread: _try_generate() attempt %d/%d", attempt + 1, retries)
            try:
                if backend == "openai":
                    logger.info("AIThread: calling _generate_openai()")
                    self._generate_openai(context, system_prompt, temperature)
                    logger.info("AIThread: _generate_openai() completed")
                    return True
                if backend == "groq":
                    logger.info("AIThread: calling _generate_groq()")
                    self._generate_groq(context, system_prompt, temperature)
                    logger.info("AIThread: _generate_groq() completed")
                    return True

                logger.error("AIThread: unknown backend %r", backend)
                if pyqtSignal is not None:
                    self.ai_error.emit(f"Unknown backend: {backend}")  # type: ignore[attr-defined]
                return False
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("AIThread: _try_generate() caught exception: %s", exc, exc_info=True)
                rate_limited = _is_rate_limited_shared(exc)
                can_retry = (
                    rate_limited
                    and attempt < retries - 1
                    and not self._stop_event.is_set()
                )
                if can_retry:
                    delay = min(
                        RUNTIME_RATE_LIMIT_MAX_DELAY,
                        base_delay * (2**attempt),
                    ) + random.uniform(0.0, 0.25)
                    logger.warning(
                        "AIThread: %s rate limited, retrying in %.2fs (attempt %d/%d)",
                        backend,
                        delay,
                        attempt + 1,
                        retries,
                    )
                    self._stop_event.wait(delay)
                    continue

                if rate_limited:
                    error_msg = (
                        "Rate limit reached on AI generation (HTTP 429). "
                        "Please wait a moment and try again."
                    )
                else:
                    error_msg = f"{backend.title()} generation failed: {exc}"

                logger.debug("AIThread: backend %r failed: %s", backend, exc)
                if pyqtSignal is not None:
                    self.ai_error.emit(error_msg)  # type: ignore[attr-defined]
                return False
        return False

    @staticmethod
    def _is_rate_limited_exception(exc: Exception) -> bool:
        """Delegate to shared utility. Kept for backward compatibility."""
        return _is_rate_limited_shared(exc)

    def _generate_openai(
        self, context: str, system_prompt: str, temperature: float
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as e:
            error_msg = "openai package not installed. Run: pip install openai"
            if pyqtSignal is not None:
                self.ai_error.emit(error_msg)  # type: ignore[attr-defined]
            raise RuntimeError(error_msg) from e

        api_key = self._config.get("openai_api_key", "")
        if not api_key:
            error_msg = "OpenAI API key not set. Add it in Settings -> AI."
            if pyqtSignal is not None:
                self.ai_error.emit(error_msg)  # type: ignore[attr-defined]
            raise ValueError(error_msg)

        client = OpenAI(api_key=api_key)
        self._generate_stream(client, system_prompt, context, temperature, "openai")

    def _generate_groq(
        self, context: str, system_prompt: str, temperature: float
    ) -> None:
        logger.info("AIThread: _generate_groq() called")
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as e:
            error_msg = "openai package not installed. Run: pip install openai"
            if pyqtSignal is not None:
                self.ai_error.emit(error_msg)  # type: ignore[attr-defined]
            raise RuntimeError(error_msg) from e

        api_key = self._config.get("groq_api_key", "")
        if not api_key:
            error_msg = "Groq API key not set. Add it in Settings → AI."
            if pyqtSignal is not None:
                self.ai_error.emit(error_msg)  # type: ignore[attr-defined]
            raise ValueError(error_msg)

        logger.info("AIThread: creating Groq client...")
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        logger.info("AIThread: Groq client created, calling _generate_stream()...")
        self._generate_stream(client, system_prompt, context, temperature, "groq")

    # ------------------------------------------------------------------
    # Stream Processing (shared by OpenAI and Groq)
    # ------------------------------------------------------------------

    def _generate_stream(
        self, client, system_prompt: str, context: str, temperature: float, backend_name: str
    ) -> None:
        """Generate response using standard Chat Completions streaming API.
        
        More stable than the responses API and widely supported by both OpenAI and Groq.
        Collects all chunks and emits complete response at the end to avoid
        threading issues with UI updates.
        
        Args:
            client: OpenAI client instance (supports both OpenAI and Groq via base_url)
            system_prompt: System prompt for the model
            context: User context/input
            temperature: Temperature setting
            backend_name: "openai" or "groq" for logging
        """
        logger.info("AIThread (%s): _generate_stream() starting", backend_name)
        full_response: List[str] = []
        stream = None

        try:
            default_model = "llama-3.3-70b-versatile" if backend_name == "groq" else "gpt-4o-mini"
            model = self._config.get(f"{backend_name}_model", default_model)
            logger.info("AIThread (%s): using model=%s, context_length=%d", 
                       backend_name, model, len(context))
            
            # Use standard Chat Completions API with streaming
            logger.info("AIThread (%s): calling client.chat.completions.create()", backend_name)
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                ],
                temperature=temperature,
                max_tokens=512,
                stream=True,
            )
            logger.info("AIThread (%s): stream object created, starting iteration", backend_name)

            stream_timeout = self._config.get("stream_timeout", 30.0)
            stream_start = time.time()
            chunk_count = 0
            logger.info("AIThread (%s): stream_timeout=%s", backend_name, stream_timeout)

            for chunk in stream:
                if self._stop_event.is_set():
                    logger.info("AIThread (%s): stream cancelled by stop event", backend_name)
                    break

                # Hard timeout check on each chunk
                elapsed = time.time() - stream_start
                if elapsed > stream_timeout:
                    error_msg = f"Stream timeout after {stream_timeout}s"
                    logger.error("AIThread (%s): %s", backend_name, error_msg)
                    raise RuntimeError(error_msg)

                # Extract delta from chunk
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    
                    # Check for finish_reason to detect completion
                    if choice.finish_reason in ("stop", "length", "tool_calls"):
                        logger.debug(
                            "AIThread (%s): stream finished (reason: %s, chunks: %d)",
                            backend_name,
                            choice.finish_reason,
                            chunk_count,
                        )
                        break
                    
                    # Extract and accumulate delta content
                    if choice.delta and choice.delta.content:
                        delta = choice.delta.content
                        full_response.append(delta)
                        chunk_count += 1

        except Exception as exc:  # pylint: disable=broad-except
            error_msg = f"Stream error: {exc}"
            logger.error("AIThread (%s): %s", backend_name, error_msg)
            raise RuntimeError(error_msg) from exc

        finally:
            # Explicitly close the stream to prevent resource leak
            if stream is not None and hasattr(stream, 'close'):
                try:
                    stream.close()
                except Exception as e:  # pylint: disable=broad-except
                    logger.debug("AIThread (%s): error closing stream: %s", backend_name, e)

        # Emit the complete response only once (thread-safe)
        complete = "".join(full_response)
        if complete:
            if self._on_ready:
                self._on_ready(complete)
            if pyqtSignal is not None:
                self.ai_response_ready.emit(complete)  # type: ignore[attr-defined]
            logger.info("AIThread (%s): response ready (%d chars)", backend_name, len(complete))
        else:
            error_msg = "No response generated from AI"
            logger.warning("AIThread (%s): %s", backend_name, error_msg)
            raise RuntimeError(error_msg)

    # ------------------------------------------------------------------
    # Connectivity Test (for startup and manual testing)
    # ------------------------------------------------------------------

    def test_api_connectivity(self, backend: Optional[str] = None) -> tuple:
        """Test API connectivity by sending a simple message.
        
        Args:
            backend: Specific backend to test, or None to test main backend.
            
        Returns:
            Tuple of (success: bool, backend_used: str, message_or_error: str)
        """
        requested_backend = (
            backend
            or self._config.get("main_backend")
            or self._config.get("backend", "groq")
        )
        test_backend = self._resolve_active_backend(requested_backend)
        test_msg = self._config.get("api_test_message", "Hi, testing API connectivity")
        
        try:
            if test_backend == "openai":
                response = self._test_openai(test_msg)
                return (True, "openai", response)
            elif test_backend == "groq":
                response = self._test_groq(test_msg)
                return (True, "groq", response)
            else:
                return (False, test_backend, f"Unknown backend: {test_backend}")
        except Exception as exc:
            return (False, test_backend, f"Connection failed: {exc}")

    def _test_openai(self, message: str) -> str:
        """Send test message to OpenAI and return response.
        
        Uses exponential backoff for transient errors (max 3 retries).
        """
        api_key = self._config.get("openai_api_key", "")
        if not api_key:
            raise ValueError("OpenAI API key not set")

        model = self._config.get("openai_model", "gpt-4o-mini")
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                try:
                    from openai import OpenAI  # type: ignore[import]
                except ImportError as e:
                    raise RuntimeError("openai package not installed") from e

                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": message}],
                    temperature=0.7,
                    max_tokens=100,
                    timeout=10.0,
                )
                
                # Extract response text from standard Chat Completions response
                if response.choices and len(response.choices) > 0:
                    choice = response.choices[0]
                    if choice.message and choice.message.content:
                        return choice.message.content
                
                logger.warning("AIThread: OpenAI response had no content")
                return ""

            except TimeoutError as e:
                if attempt < max_retries - 1:
                    logger.debug("AIThread: OpenAI test timeout, retrying in %.1fs... (attempt %d/%d)", 
                               retry_delay, attempt + 1, max_retries)
                    time.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                else:
                    raise RuntimeError(f"OpenAI test timeout after {max_retries} retries") from e
            except Exception as e:
                if attempt < max_retries - 1 and "rate" in str(e).lower():
                    logger.debug("AIThread: OpenAI rate limit, retrying in %.1fs... (attempt %d/%d)", 
                               retry_delay, attempt + 1, max_retries)
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

    def _test_groq(self, message: str) -> str:
        """Send test message to Groq and return response.
        
        Uses exponential backoff for transient errors (max 3 retries).
        """
        api_key = self._config.get("groq_api_key", "")
        if not api_key:
            raise ValueError("Groq API key not set")

        model = self._config.get("groq_model", "llama-3.3-70b-versatile")
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                try:
                    from openai import OpenAI  # type: ignore[import]
                except ImportError as e:
                    raise RuntimeError("openai package not installed") from e

                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1",
                )

                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": message}],
                    temperature=0.7,
                    max_tokens=100,
                    timeout=10.0,
                )
                
                # Extract response text from standard Chat Completions response
                if response.choices and len(response.choices) > 0:
                    choice = response.choices[0]
                    if choice.message and choice.message.content:
                        return choice.message.content
                
                logger.warning("AIThread: Groq response had no content")
                return ""

            except TimeoutError as e:
                if attempt < max_retries - 1:
                    logger.debug("AIThread: Groq test timeout, retrying in %.1fs... (attempt %d/%d)", 
                               retry_delay, attempt + 1, max_retries)
                    time.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                else:
                    raise RuntimeError(f"Groq test timeout after {max_retries} retries") from e
            except Exception as e:
                if attempt < max_retries - 1 and "rate" in str(e).lower():
                    logger.debug("AIThread: Groq rate limit, retrying in %.1fs... (attempt %d/%d)", 
                               retry_delay, attempt + 1, max_retries)
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    @staticmethod
    def is_explicit_follow_up_request(text: str) -> bool:
        """Return True when *text* explicitly asks to continue prior topic."""
        cleaned = text.strip().lower()
        if not cleaned:
            return False
        return any(re.search(pattern, cleaned) for pattern in FOLLOW_UP_PATTERNS)

    @staticmethod
    def is_context_dependent_follow_up(prev_question: str, new_question: str) -> bool:
        """Return True when *new_question* likely depends on prior context.

        Detects explicit continuation phrases plus short referential prompts like
        "Can you give an example?" or "Why is that?" which often have low lexical
        overlap with the previous question but are still follow-ups.
        """
        cleaned_new = new_question.strip().lower()
        if not cleaned_new:
            return False
        if AIThread.is_explicit_follow_up_request(cleaned_new):
            return True
        if not prev_question.strip():
            return False

        tokens = re.findall(r"[a-z0-9']+", cleaned_new)
        if not tokens:
            return False

        short_follow_up = len(tokens) <= SHORT_FOLLOW_UP_MAX_TOKENS
        has_reference_term = any(token in FOLLOW_UP_REFERENCE_TERMS for token in tokens)
        has_selection_phrase = any(
            re.search(pattern, cleaned_new) for pattern in SELECTION_FOLLOW_UP_PATTERNS
        )
        starts_with_bridge = cleaned_new.startswith(
            ("and ", "also ", "so ", "then ", "right ", "okay ", "ok ")
        )

        content_words = [
            token
            for token in re.sub(r"[^a-z0-9\s]", "", cleaned_new).split()
            if len(token) >= 4 and token not in _STOP_WORDS
        ]
        non_generic_content = [
            token
            for token in content_words
            if token not in FOLLOW_UP_REFERENCE_TERMS and token not in GENERIC_SELECTION_TERMS
        ]
        likely_standalone_question = len(non_generic_content) >= 2

        return bool(
            short_follow_up
            and not likely_standalone_question
            and (has_reference_term or starts_with_bridge or has_selection_phrase)
        )

    @staticmethod
    def classify_topic_shift(prev_question: str, new_question: str) -> bool:
        """Return True when *new_question* is a semantically different topic.

        Uses Jaccard similarity on content words (non-stop-words ≥ 4 chars).
        A similarity below ``TOPIC_SHIFT_THRESHOLD`` is treated as a new topic.

        Special cases:
        - If *prev_question* is empty it is the first question → new topic.
        - If *new_question* is empty, return False (not enough info to judge).
        - Explicit follow-up phrases always override (return False).
        """
        if not new_question.strip():
            return False
        if not prev_question.strip():
            return True  # No prior question means this is always the first one.

        # Context-dependent follow-ups are never a topic shift.
        if AIThread.is_context_dependent_follow_up(prev_question, new_question):
            return False

        def _content_words(text: str) -> frozenset:
            # Lowercase, strip punctuation, keep words ≥ 4 chars that are not stop-words.
            words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
            return frozenset(
                w for w in words if len(w) >= 4 and w not in _STOP_WORDS
            )

        prev_words = _content_words(prev_question)
        new_words = _content_words(new_question)

        # If both sets are empty (e.g. very short questions) fall back to
        # character-level comparison: treat as different if texts differ.
        if not prev_words and not new_words:
            return prev_question.strip().lower() != new_question.strip().lower()

        # Jaccard similarity = |intersection| / |union|
        intersection = len(prev_words & new_words)
        union = len(prev_words | new_words)
        similarity = intersection / union if union else 0.0

        logger.debug(
            "classify_topic_shift: similarity=%.2f (threshold=%.2f) "
            "prev=%r new=%r",
            similarity,
            TOPIC_SHIFT_THRESHOLD,
            prev_question[:60],
            new_question[:60],
        )
        return similarity < TOPIC_SHIFT_THRESHOLD

    @staticmethod
    def _build_context(
        transcript: List[TranscriptSegment],
        session_context: str = "",
        is_new_topic: bool = False,
        resume_profile: Optional[Dict[str, Any]] = None,
        correction_high_threshold: float = RESUME_CORRECTION_HIGH_THRESHOLD,
        correction_medium_threshold: float = RESUME_CORRECTION_MEDIUM_THRESHOLD,
    ) -> str:
        """Format the most recent question-focused transcript block.

        Args:
            transcript: Full transcript history including any injected segments.
            session_context: Optional domain/role context set by the user.
            is_new_topic: When True, suppress the Follow-up Context block so
                the AI focuses solely on the current question.
        """
        if not transcript:
            return ""

        last_speaker_index = -1
        speaker_block_start = -1
        for index in range(len(transcript) - 1, -1, -1):
            if transcript[index].source == "speaker":
                last_speaker_index = index
                break

        if last_speaker_index < 0:
            recent = transcript[-MAX_CONTEXT_SEGMENTS:]
        else:
            speaker_block_start = last_speaker_index
            while (
                speaker_block_start > 0
                and transcript[speaker_block_start - 1].source == "speaker"
            ):
                speaker_block_start -= 1
            recent = [
                seg
                for seg in transcript[speaker_block_start:]
                if seg.source == "speaker"
            ]

        previous_answer_context = ""
        previous_question_context = ""
        if not is_new_topic:
            # Only look for a follow-up context block when we are genuinely
            # continuing on the same topic.
            for seg in reversed(transcript):
                text = str(getattr(seg, "text", "")).strip()
                if not previous_answer_context and text.startswith(FOLLOW_UP_CONTEXT_PREFIX):
                    previous_answer_context = text[len(FOLLOW_UP_CONTEXT_PREFIX):].strip()
                if not previous_question_context and text.startswith(PRIOR_QUESTION_CONTEXT_PREFIX):
                    previous_question_context = text[len(PRIOR_QUESTION_CONTEXT_PREFIX):].strip()
                if previous_answer_context and previous_question_context:
                    break

        lower_session_context = session_context.lower()
        joined_recent_text = " ".join(seg.text.lower() for seg in recent)
        etl_context_active = any(
            keyword in lower_session_context or keyword in joined_recent_text
            for keyword in ETL_CONTEXT_KEYWORDS
        )

        segment_payloads: List[Dict[str, Any]] = []
        resume_related = False
        latest_speaker_text = " ".join(
            seg.text.strip() for seg in recent if seg.source == "speaker" and seg.text.strip()
        ).strip()

        if (
            not is_new_topic
            and not previous_question_context
            and latest_speaker_text
            and speaker_block_start > 0
        ):
            prior_speaker_end = -1
            for index in range(speaker_block_start - 1, -1, -1):
                if transcript[index].source == "speaker":
                    prior_speaker_end = index
                    break

            if prior_speaker_end >= 0:
                prior_speaker_start = prior_speaker_end
                while (
                    prior_speaker_start > 0
                    and transcript[prior_speaker_start - 1].source == "speaker"
                ):
                    prior_speaker_start -= 1

                prior_speaker_text = " ".join(
                    str(transcript[idx].text).strip()
                    for idx in range(prior_speaker_start, prior_speaker_end + 1)
                    if str(transcript[idx].text).strip()
                )
                if AIThread.is_context_dependent_follow_up(
                    prior_speaker_text,
                    latest_speaker_text,
                ):
                    previous_question_context = prior_speaker_text

        if resume_profile and latest_speaker_text:
            resume_related = is_resume_related_text(latest_speaker_text, resume_profile)

        for seg in recent:
            label = "Speaker" if seg.source == "speaker" else "You"
            text = seg.text
            if etl_context_active:
                text = AIThread._normalize_etl_transcript_terms(text)

            high_matches: List[Dict[str, Any]] = []
            medium_matches: List[Dict[str, Any]] = []

            if resume_profile and seg.source == "speaker":
                try:
                    correction_result = apply_resume_corrections(
                        text,
                        resume_profile,
                        high_threshold=correction_high_threshold,
                        medium_threshold=correction_medium_threshold,
                    )
                except Exception as exc:  # Defensive: ensure failures here don't break context building
                    logger.exception("AIThread: apply_resume_corrections failed: %s", exc)
                    correction_result = {"high_confidence": [], "medium_confidence": [], "text": text}

                # Ensure the result is a mapping/dict before using .get()
                if not isinstance(correction_result, dict):
                    logger.debug("AIThread: apply_resume_corrections returned non-dict; falling back to defaults")
                    correction_result = {"high_confidence": [], "medium_confidence": [], "text": text}

                high_matches = list(correction_result.get("high_confidence", []))
                medium_matches = list(correction_result.get("medium_confidence", []))
                corrected_text = str(correction_result.get("text", text))
                if high_matches:
                    text = corrected_text
                if high_matches or medium_matches:
                    resume_related = True

            segment_payloads.append(
                {
                    "label": label,
                    "text": text,
                    "high_matches": high_matches,
                    "medium_matches": medium_matches,
                }
            )

        resume_summary_lines: List[str] = []
        if resume_profile and resume_related:
            resume_summary_lines = build_resume_context_summary(resume_profile)

        lines: List[str] = []
        if session_context:
            lines.append(f"[Session Context]: {session_context}")
        if previous_question_context:
            lines.append(f"[Prior Question Context]: {previous_question_context}")
        if previous_answer_context:
            lines.append(f"[Follow-up Context]: {previous_answer_context}")

        if resume_summary_lines:
            lines.append("[Resume Context]:")
            for item in resume_summary_lines:
                lines.append(f"- {item}")

        for payload in segment_payloads:
            lines.append(f"[{payload['label']}]: {payload['text']}")

            high_matches = payload.get("high_matches", [])
            medium_matches = payload.get("medium_matches", [])
            for match in high_matches[:2]:
                original = str(match.get("original", "")).strip()
                corrected = str(match.get("corrected", "")).strip()
                category = str(match.get("category", "term")).strip()
                if original and corrected:
                    lines.append(
                        f"[Resume Correction Applied]: \"{original}\" -> \"{corrected}\" ({category}, high confidence)"
                    )

            if not high_matches:
                for match in medium_matches[:2]:
                    original = str(match.get("original", "")).strip()
                    corrected = str(match.get("corrected", "")).strip()
                    category = str(match.get("category", "term")).strip()
                    if original and corrected:
                        lines.append(
                            f"[Resume Correction Candidate]: \"{original}\" -> \"{corrected}\" ({category}, medium confidence)"
                        )

        return "\n".join(lines)

    @staticmethod
    def _normalize_etl_transcript_terms(text: str) -> str:
        """Repair common ETL-specific speech-to-text homophone mistakes."""
        normalized = text
        replacements = [
            (r"\bsort and target table\b", "source and target table"),
            (r"\bsort table\b", "source table"),
            (r"\bsort to target\b", "source to target"),
            (r"\bsort and target tables\b", "source and target tables"),
            (r"\bsort tables\b", "source tables"),
            (r"\bsort-to-target\b", "source-to-target"),
        ]
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        return normalized

    @staticmethod
    def _build_system_prompt(
        system_prompt: str,
        session_context: str,
        resume_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        prompt = system_prompt
        if session_context:
            addendum = (
                "\n\nSession context (role/background): "
                f"{session_context}\n"
                "Tailor your response to this context if relevant, but do not force it "
                "into answers for general concepts (e.g., general SQL or programming questions)."
            )
            prompt = f"{prompt}{addendum}"

        if resume_profile:
            resume_policy = (
                "\n\nResume usage policy:\n"
                "- The structured resume context is the source of truth for the user's professional background.\n"
                "- Use resume facts for resume/background questions (companies, roles, dates, projects, skills, education, certifications).\n"
                "- Apply transcript corrections only when confidence is high and the corrected term is grounded in the resume context.\n"
                "- When confidence is medium, mention the likely correction cautiously or ask for confirmation if needed.\n"
                "- If a detail is not present in the resume context, explicitly say it is not available rather than inventing details.\n"
                "- For non-resume questions, do not let resume context dominate the answer."
            )
            prompt = f"{prompt}{resume_policy}"

        follow_up_policy = (
            "\n\nFollow-up handling policy:\n"
            "- If [Prior Question Context] and/or [Follow-up Context] is provided, treat it as authoritative for resolving references like 'this', 'that', 'it', 'why', and 'example'.\n"
            "- Keep your answer focused on the current speaker question, while using prior context only to disambiguate what the follow-up refers to.\n"
            "- Do not ignore follow-up context unless it clearly conflicts with the current question."
        )
        prompt = f"{prompt}{follow_up_policy}"

        return prompt
