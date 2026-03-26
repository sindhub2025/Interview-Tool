"""
AI response engine.

Groq is the active backend exposed in the app.
OpenAI code paths are intentionally preserved for future re-enable.

Runs in a dedicated QThread and emits streaming response tokens.
"""

from __future__ import annotations

import queue
import random
import threading
import time
from typing import Callable, List, Optional

from ghostmic.utils.errors import is_rate_limited as _is_rate_limited_shared

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    QThread = object  # type: ignore[misc,assignment]
    pyqtSignal = None  # type: ignore[assignment]

from ghostmic.core.transcription_engine import TranscriptSegment
from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are a real-time interview/meeting assistant. You are helping the user \
respond to questions and participate in a conversation.

Based on the conversation transcript provided, generate:
1. A concise, natural-sounding suggested response to the last question/topic
2. Key talking points the user should mention

Rules:
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
        self._queue: "queue.Queue[List[TranscriptSegment]]" = queue.Queue(
            maxsize=AI_QUEUE_MAXSIZE
        )
        self._last_request_time: float = 0.0
        self._last_reject_reason: str = ""

    def stop(self) -> None:
        self._stop_event.set()

    def request_response(self, transcript: List[TranscriptSegment]) -> bool:
        """Queue a response-generation request.

        Applies debounce: requests closer than DEBOUNCE_SECONDS apart are
        discarded.

        Args:
            transcript: Recent transcript segments for context.

        Returns:
            True when accepted into the queue, otherwise False.
        """
        now = time.time()
        if now - self._last_request_time < DEBOUNCE_SECONDS:
            logger.debug("AIThread: debounce – request ignored.")
            self._last_reject_reason = "debounced"
            return False
        self._last_request_time = now
        try:
            self._queue.put_nowait(transcript)
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
                self._queue.put_nowait(transcript)
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
                transcript = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if pyqtSignal is not None:
                self.ai_thinking.emit()  # type: ignore[attr-defined]
            self._generate(transcript)

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

    def _generate(self, transcript: List[TranscriptSegment]) -> None:
        # Keep old/new config keys for compatibility, then normalize to active provider.
        requested_main = self._config.get("main_backend") or self._config.get("backend", "groq")
        requested_fallback = self._config.get("fallback_backend", "groq")
        main_backend = self._resolve_active_backend(requested_main)
        fallback_backend = self._resolve_active_backend(requested_fallback)
        enable_fallback = bool(self._config.get("enable_fallback", False))
        
        context = self._build_context(transcript)
        system_prompt = self._config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        temperature = float(self._config.get("temperature", 0.7))

        # Try main backend first
        success = self._try_generate(main_backend, context, system_prompt, temperature)
        
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
        retries = int(self._config.get("runtime_rate_limit_retries", RUNTIME_RATE_LIMIT_RETRIES))
        base_delay = float(
            self._config.get("runtime_rate_limit_retry_delay", RUNTIME_RATE_LIMIT_BASE_DELAY)
        )
        retries = max(1, retries)

        for attempt in range(retries):
            try:
                if backend == "openai":
                    self._generate_openai(context, system_prompt, temperature)
                    return True
                if backend == "groq":
                    self._generate_groq(context, system_prompt, temperature)
                    return True

                logger.error("AIThread: unknown backend %r", backend)
                if pyqtSignal is not None:
                    self.ai_error.emit(f"Unknown backend: {backend}")  # type: ignore[attr-defined]
                return False
            except Exception as exc:  # pylint: disable=broad-except
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

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self._generate_stream(client, system_prompt, context, temperature, "groq")

    # ------------------------------------------------------------------
    # Stream Processing (shared by OpenAI and Groq)
    # ------------------------------------------------------------------

    def _generate_stream(
        self, client, system_prompt: str, context: str, temperature: float, backend_name: str
    ) -> None:
        """Generate response from client using Responses API streaming.
        
        Handles event-driven state machine for both OpenAI and Groq backends.
        
        Args:
            client: OpenAI client instance (supports both OpenAI and Groq via base_url)
            system_prompt: System prompt for the model
            context: User context/input
            temperature: Temperature setting
            backend_name: "openai" or "groq" for logging
        """
        # Response state machine: pending → streaming → completed/refusal
        full_response: List[str] = []
        response_state = "pending"
        error_to_emit = None
        stream = None
        unknown_event_types: set[str] = set()

        try:
            default_model = "llama-3.3-70b-versatile" if backend_name == "groq" else "gpt-5-mini"
            model = self._config.get(f"{backend_name}_model", default_model)
            stream = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                ],
                temperature=temperature,
                max_output_tokens=512,
                stream=True,
            )

            stream_timeout = self._config.get("stream_timeout", 30.0)
            stream_start = time.time()

            for event in stream:
                if self._stop_event.is_set():
                    response_state = "cancelled"
                    logger.info("AIThread (%s): stream cancelled by stop event", backend_name)
                    break

                # Hard timeout check on each event
                elapsed = time.time() - stream_start
                if elapsed > stream_timeout:
                    response_state = "error"
                    error_to_emit = f"Stream timeout after {stream_timeout}s"
                    logger.error("AIThread (%s): %s", backend_name, error_to_emit)
                    break

                # State transition: pending → streaming on first delta
                if response_state == "pending" and event.type == "response.output_text.delta":
                    response_state = "streaming"

                # Event-driven state machine
                if event.type == "response.created":
                    logger.debug("AIThread (%s): response created", backend_name)

                elif event.type == "response.output_text.delta":
                    # Events guarantee ordering - no state guard needed
                    delta = event.delta or ""
                    if delta:
                        full_response.append(delta)
                        if self._on_chunk:
                            self._on_chunk(delta)
                        if pyqtSignal is not None:
                            self.ai_response_chunk.emit(delta)  # type: ignore[attr-defined]

                elif event.type == "response.output_text.done":
                    logger.debug("AIThread (%s): output text done", backend_name)

                elif event.type == "response.refusal.delta":
                    # Accumulate refusal text
                    refusal_delta = event.delta or ""
                    if refusal_delta:
                        full_response.append(refusal_delta)

                elif event.type == "response.refusal.done":
                    # Refusal completed - state transition and stop processing
                    response_state = "refusal"
                    refusal_msg = "".join(full_response) or "Model refused to respond"
                    error_to_emit = f"Model declined: {refusal_msg}"
                    logger.warning("AIThread (%s): model refusal: %s", backend_name, refusal_msg)
                    break

                elif event.type == "response.completed":
                    response_state = "completed"
                    logger.debug("AIThread (%s): response completed", backend_name)

                elif event.type == "response.tool_calls":
                    self._route_tool_calls(event, backend_name)

                else:
                    if event.type not in unknown_event_types:
                        unknown_event_types.add(event.type)
                        logger.debug(
                            "AIThread (%s): unknown event type observed: %s",
                            backend_name,
                            event.type,
                        )

        except Exception as exc:  # pylint: disable=broad-except
            response_state = "error"
            error_to_emit = f"Stream error: {exc}"
            logger.error("AIThread (%s): %s", backend_name, error_to_emit)

        finally:
            # Hard cancel: explicitly close the stream to prevent resource leak
            if stream is not None:
                try:
                    if hasattr(stream, 'close'):
                        stream.close()
                except Exception as e:  # pylint: disable=broad-except
                    logger.debug("AIThread (%s): error closing stream: %s", backend_name, e)

        if unknown_event_types:
            logger.debug(
                "AIThread (%s): suppressed repeated unknown event logs for types: %s",
                backend_name,
                sorted(unknown_event_types),
            )

        # Final state handling
        complete = "".join(full_response)

        if response_state == "cancelled":
            if complete:
                logger.warning(
                    "AIThread (%s): cancelled stream emitted partial response (%d chars)",
                    backend_name,
                    len(complete),
                )
                if self._on_ready:
                    self._on_ready(complete)
                if pyqtSignal is not None:
                    self.ai_response_ready.emit(complete)  # type: ignore[attr-defined]
            return

        if error_to_emit:
            if complete:
                logger.warning(
                    "AIThread (%s): stream ended with error, emitting partial response (%d chars)",
                    backend_name,
                    len(complete),
                )
                if self._on_ready:
                    self._on_ready(complete)
                if pyqtSignal is not None:
                    self.ai_response_ready.emit(complete)  # type: ignore[attr-defined]
                return
            raise RuntimeError(error_to_emit)

        if response_state == "completed":
            if self._on_ready:
                self._on_ready(complete)
            if pyqtSignal is not None:
                self.ai_response_ready.emit(complete)  # type: ignore[attr-defined]
        elif response_state != "completed":
            if complete:
                logger.warning(
                    "AIThread (%s): stream ended in state %s, emitting partial response (%d chars)",
                    backend_name,
                    response_state,
                    len(complete),
                )
                if self._on_ready:
                    self._on_ready(complete)
                if pyqtSignal is not None:
                    self.ai_response_ready.emit(complete)  # type: ignore[attr-defined]
            else:
                error_msg = f"Response incomplete: {response_state}"
                logger.warning(
                    "AIThread (%s): stream ended in unexpected state: %s (response incomplete)",
                    backend_name,
                    response_state,
                )
                raise RuntimeError(error_msg)

    def _route_tool_calls(self, event, backend_name: str) -> None:
        """Route tool call events to a dedicated handler path.

        Agents/tool execution is not wired yet, but this keeps event handling centralized.
        """
        tool_calls = getattr(event, "tool_calls", None)
        if tool_calls:
            logger.info(
                "AIThread (%s): received %d tool call(s); execution path not enabled yet",
                backend_name,
                len(tool_calls),
            )
        else:
            logger.info(
                "AIThread (%s): received tool call event without payload",
                backend_name,
            )

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

        model = self._config.get("openai_model", "gpt-5-mini")
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                try:
                    from openai import OpenAI  # type: ignore[import]
                except ImportError as e:
                    raise RuntimeError("openai package not installed") from e

                client = OpenAI(api_key=api_key)
                response = client.responses.create(
                    model=model,
                    input=[{"role": "user", "content": message}],
                    temperature=0.7,
                    max_output_tokens=100,
                    timeout=10.0,
                )
                
                # Safely extract response text from output structure
                if hasattr(response, 'output') and response.output:
                    text_parts = []
                    for item in response.output:
                        if getattr(item, 'type', None) == 'text':
                            text_parts.append(getattr(item, 'text', ''))
                    if text_parts:
                        return "".join(text_parts)
                
                # Fallback to output_text property if available
                if hasattr(response, 'output_text') and response.output_text:
                    return response.output_text
                
                # Last resort: empty string
                logger.warning("AIThread: OpenAI response had no output_text or output items")
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

                response = client.responses.create(
                    model=model,
                    input=[{"role": "user", "content": message}],
                    temperature=0.7,
                    max_output_tokens=100,
                    timeout=10.0,
                )
                
                # Safely extract response text from output structure
                if hasattr(response, 'output') and response.output:
                    text_parts = []
                    for item in response.output:
                        if getattr(item, 'type', None) == 'text':
                            text_parts.append(getattr(item, 'text', ''))
                    if text_parts:
                        return "".join(text_parts)
                
                # Fallback to output_text property if available
                if hasattr(response, 'output_text') and response.output_text:
                    return response.output_text
                
                # Last resort: empty string
                logger.warning("AIThread: Groq response had no output_text or output items")
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
    def _build_context(transcript: List[TranscriptSegment]) -> str:
        """Format the last N transcript segments as a conversation string."""
        recent = transcript[-MAX_CONTEXT_SEGMENTS:]
        lines: List[str] = []
        for seg in recent:
            label = "Speaker" if seg.source == "speaker" else "You"
            lines.append(f"[{label}]: {seg.text}")
        return "\n".join(lines)
