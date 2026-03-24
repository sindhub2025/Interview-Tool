"""
AI response engine.

Supports two backends (user configurable via config.json):
  - Groq API (default, free tier)
  - Ollama (local, fully offline)

Runs in a dedicated QThread and emits streaming response tokens.
"""

from __future__ import annotations

import json
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
        backend = self._config.get("backend", "groq")
        context = self._build_context(transcript)
        system_prompt = self._config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        temperature = float(self._config.get("temperature", 0.7))

        try:
            if backend == "groq":
                self._generate_groq(context, system_prompt, temperature)
            elif backend == "ollama":
                self._generate_ollama(context, system_prompt, temperature)
            else:
                logger.error("AIThread: unknown backend %r", backend)
        except Exception as exc:  # pylint: disable=broad-except
            msg = f"AI generation error: {exc}"
            logger.error(msg, exc_info=True)
            if pyqtSignal is not None:
                self.ai_error.emit(str(exc))  # type: ignore[attr-defined]

    def _generate_groq(
        self, context: str, system_prompt: str, temperature: float
    ) -> None:
        try:
            from groq import Groq  # type: ignore[import]
        except ImportError:
            if pyqtSignal is not None:
                self.ai_error.emit(  # type: ignore[attr-defined]
                    "groq package not installed. Run: pip install groq"
                )
            return

        api_key = self._config.get("groq_api_key", "")
        if not api_key:
            if pyqtSignal is not None:
                self.ai_error.emit(  # type: ignore[attr-defined]
                    "Groq API key not set. Add it in Settings → AI."
                )
            return

        model = self._config.get("groq_model", "llama-3.1-70b-versatile")
        client = Groq(api_key=api_key)

        full_response: List[str] = []
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

    def _generate_ollama(
        self, context: str, system_prompt: str, temperature: float
    ) -> None:
        import requests  # type: ignore[import]

        url = self._config.get("ollama_url", "http://localhost:11434")
        model = self._config.get("ollama_model", "llama3.1:8b")

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            "stream": True,
            "options": {"temperature": temperature},
        }

        full_response: List[str] = []
        try:
            response = requests.post(
                f"{url}/api/chat",
                json=payload,
                stream=True,
                timeout=60,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta = obj.get("message", {}).get("content", "")
                if delta:
                    full_response.append(delta)
                    if self._on_chunk:
                        self._on_chunk(delta)
                    if pyqtSignal is not None:
                        self.ai_response_chunk.emit(delta)  # type: ignore[attr-defined]
                if obj.get("done"):
                    break
        except requests.ConnectionError:
            msg = (
                "Cannot connect to Ollama at "
                f"{url}. Is Ollama running?"
            )
            if pyqtSignal is not None:
                self.ai_error.emit(msg)  # type: ignore[attr-defined]
            return

        complete = "".join(full_response)
        if self._on_ready:
            self._on_ready(complete)
        if pyqtSignal is not None:
            self.ai_response_ready.emit(complete)  # type: ignore[attr-defined]

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
