"""
AI response engine.

Supports two backends (user configurable via config.json):
    - OpenAI API (default)
    - Groq API (manual backup)

Runs in a dedicated QThread and emits streaming response tokens.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, List, Optional

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
        self._queue: "queue.Queue[List[TranscriptSegment]]" = queue.Queue()
        self._last_request_time: float = 0.0

    def stop(self) -> None:
        self._stop_event.set()

    def request_response(self, transcript: List[TranscriptSegment]) -> None:
        """Queue a response-generation request.

        Applies debounce: requests closer than DEBOUNCE_SECONDS apart are
        discarded.

        Args:
            transcript: Recent transcript segments for context.
        """
        now = time.time()
        if now - self._last_request_time < DEBOUNCE_SECONDS:
            logger.debug("AIThread: debounce – request ignored.")
            return
        self._last_request_time = now
        try:
            self._queue.put_nowait(transcript)
        except queue.Full:
            logger.debug("AIThread: queue full – dropping request.")

    def update_config(self, config: dict) -> None:
        """Replace the AI configuration at runtime."""
        self._config = config

    def run(self) -> None:
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

    def _generate(self, transcript: List[TranscriptSegment]) -> None:
        # Support both old 'backend' and new 'main_backend' config for backward compatibility
        main_backend = self._config.get("main_backend") or self._config.get("backend", "openai")
        fallback_backend = self._config.get("fallback_backend", "groq")
        enable_fallback = self._config.get("enable_fallback", True)
        
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
        try:
            if backend == "openai":
                self._generate_openai(context, system_prompt, temperature)
                return True
            elif backend == "groq":
                self._generate_groq(context, system_prompt, temperature)
                return True
            else:
                logger.error("AIThread: unknown backend %r", backend)
                return False
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("AIThread: backend %r failed: %s", backend, exc)
            return False

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

        model = self._config.get("openai_model", "gpt-4o-mini")
        client = OpenAI(api_key=api_key)

        full_response: List[str] = []
        # OpenAI API Parameters (current as of 2026):
        # - model: gpt-4o-mini is the configured low-latency default; override in config.json if needed
        # - messages: standard chat completion format (system + user roles)
        # - temperature: 0.7 balances creativity and consistency (0.0=deterministic, 1.0=maximum randomness)
        # - max_tokens: 512 is sufficient for interview/meeting suggestions
        # - stream: enables incremental token delivery for real-time response display
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

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_response.append(delta)
                if self._on_chunk:
                    self._on_chunk(delta)
                if pyqtSignal is not None:
                    self.ai_response_chunk.emit(delta)  # type: ignore[attr-defined]

        complete = "".join(full_response)
        if self._on_ready:
            self._on_ready(complete)
        if pyqtSignal is not None:
            self.ai_response_ready.emit(complete)  # type: ignore[attr-defined]

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

        model = self._config.get("groq_model", "openai/gpt-oss-20b")
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        full_response: List[str] = []
        # Groq API Parameters (current as of 2026):
        # - base_url: https://api.groq.com/openai/v1 uses Groq's OpenAI-compatible endpoint
        # - model: openai/gpt-oss-20b matches the documented Groq example and is configurable
        # - responses.create: current OpenAI-compatible Responses API
        # - max_output_tokens: 512 is sufficient for interview/meeting suggestions
        # - stream: enables incremental token delivery for real-time response display
        stream = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=context,
            temperature=temperature,
            max_output_tokens=512,
            stream=True,
        )

        for chunk in stream:
            if getattr(chunk, "type", None) != "response.output_text.delta":
                continue
            delta = getattr(chunk, "delta", "") or ""
            if delta:
                full_response.append(delta)
                if self._on_chunk:
                    self._on_chunk(delta)
                if pyqtSignal is not None:
                    self.ai_response_chunk.emit(delta)  # type: ignore[attr-defined]

        complete = "".join(full_response)
        if self._on_ready:
            self._on_ready(complete)
        if pyqtSignal is not None:
            self.ai_response_ready.emit(complete)  # type: ignore[attr-defined]

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
        test_backend = backend or self._config.get("main_backend") or self._config.get("backend", "openai")
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
        """Send test message to OpenAI and return response."""
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as e:
            raise RuntimeError("openai package not installed") from e

        api_key = self._config.get("openai_api_key", "")
        if not api_key:
            raise ValueError("OpenAI API key not set")

        model = self._config.get("openai_model", "gpt-4o-mini")
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
            temperature=0.7,
            max_tokens=100,
        )
        return response.choices[0].message.content or ""

    def _test_groq(self, message: str) -> str:
        """Send test message to Groq and return response."""
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as e:
            raise RuntimeError("openai package not installed") from e

        api_key = self._config.get("groq_api_key", "")
        if not api_key:
            raise ValueError("Groq API key not set")

        model = self._config.get("groq_model", "openai/gpt-oss-20b")
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        response = client.responses.create(
            model=model,
            input=message,
            temperature=0.7,
            max_output_tokens=100,
        )
        return getattr(response, "output_text", "") or ""

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
