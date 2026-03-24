"""
Real-time speech-to-text transcription using faster-whisper.

Runs in a dedicated QThread, receives speech segments from the VAD
pipeline, and emits TranscriptSegment results.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    QThread = object  # type: ignore[misc,assignment]
    pyqtSignal = None  # type: ignore[assignment]

from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptSegment:
    """A single transcribed speech segment."""

    text: str
    source: str                 # "speaker" or "user"
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0


class ModelLoader(QThread):  # type: ignore[misc]
    """Loads the faster-whisper model in a background thread.

    Emits ``model_ready()`` when loading is complete and
    ``model_error(str)`` on failure.
    """

    if pyqtSignal is not None:
        model_ready = pyqtSignal()
        model_error = pyqtSignal(str)
        progress = pyqtSignal(str)

    def __init__(
        self,
        model_size: str = "base.en",
        compute_type: str = "int8",
        device: str = "auto",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self.model = None

    def run(self) -> None:
        try:
            self._emit_progress(f"Loading Whisper model '{self.model_size}' …")
            from faster_whisper import WhisperModel  # type: ignore[import]

            resolved_device = self._resolve_device()
            self._emit_progress(
                f"Initialising model on {resolved_device} ({self.compute_type}) …"
            )
            self.model = WhisperModel(
                self.model_size,
                device=resolved_device,
                compute_type=self.compute_type,
            )
            self._emit_progress("Whisper model ready.")
            if pyqtSignal is not None:
                self.model_ready.emit()  # type: ignore[attr-defined]
            logger.info(
                "ModelLoader: model '%s' loaded on %s.", self.model_size, resolved_device
            )
        except Exception as exc:  # pylint: disable=broad-except
            msg = f"Failed to load Whisper model: {exc}"
            logger.error(msg, exc_info=True)
            if pyqtSignal is not None:
                self.model_error.emit(str(exc))  # type: ignore[attr-defined]

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch  # type: ignore[import]
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _emit_progress(self, msg: str) -> None:
        logger.info("ModelLoader: %s", msg)
        if pyqtSignal is not None:
            self.progress.emit(msg)  # type: ignore[attr-defined]


class TranscriptionThread(QThread):  # type: ignore[misc]
    """Transcribes speech segments using faster-whisper.

    Receives audio segments via :meth:`push_segment` and emits
    ``transcription_ready(TranscriptSegment)`` for each result.
    Shows ``transcribing(source)`` while processing is in progress.

    Args:
        model: A loaded ``faster_whisper.WhisperModel`` instance.
        language: BCP-47 language code (default "en").
        beam_size: Beam size for decoding (default 3).
        on_result: Optional callback.
    """

    if pyqtSignal is not None:
        transcription_ready = pyqtSignal(object)
        transcribing = pyqtSignal(str)

    def __init__(
        self,
        model=None,
        language: str = "en",
        beam_size: int = 3,
        on_result: Optional[Callable[[TranscriptSegment], None]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._language = language
        self._beam_size = beam_size
        self._on_result = on_result
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[Tuple[np.ndarray, str]]" = queue.Queue()

    def set_model(self, model) -> None:
        """Attach a (newly loaded) model."""
        self._model = model

    def stop(self) -> None:
        self._stop_event.set()

    def push_segment(self, audio: np.ndarray, source: str) -> None:
        """Enqueue an audio segment for transcription."""
        self._queue.put_nowait((audio, source))

    def run(self) -> None:
        logger.info("TranscriptionThread: started.")
        while not self._stop_event.is_set():
            try:
                audio, source = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if self._model is None:
                logger.warning(
                    "TranscriptionThread: no model loaded – dropping segment."
                )
                continue

            if pyqtSignal is not None:
                self.transcribing.emit(source)  # type: ignore[attr-defined]

            segment = self._transcribe(audio, source)
            if segment:
                if self._on_result:
                    self._on_result(segment)
                if pyqtSignal is not None:
                    self.transcription_ready.emit(segment)  # type: ignore[attr-defined]

        logger.info("TranscriptionThread: stopped.")

    def _transcribe(
        self, audio: np.ndarray, source: str
    ) -> Optional[TranscriptSegment]:
        """Run whisper inference on *audio* and return a TranscriptSegment."""
        try:
            audio_float = audio.astype(np.float32) / 32768.0
            segments, info = self._model.transcribe(
                audio_float,
                language=self._language,
                beam_size=self._beam_size,
                best_of=self._beam_size,
                word_timestamps=True,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
            )

            texts: List[str] = []
            avg_prob: float = 0.0
            seg_count: int = 0

            for seg in segments:
                text = seg.text.strip()
                if text:
                    texts.append(text)
                    avg_prob += getattr(seg, "avg_logprob", 0.0)
                    seg_count += 1

            if not texts:
                return None

            full_text = " ".join(texts)
            confidence = float(
                min(1.0, max(0.0, (avg_prob / seg_count + 1.0)))
                if seg_count
                else 1.0
            )

            return TranscriptSegment(
                text=full_text,
                source=source,
                confidence=confidence,
            )

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("TranscriptionThread: error: %s", exc, exc_info=True)
            return None
