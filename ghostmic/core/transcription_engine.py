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
            errors: List[str] = []
            for device, compute_type in self._candidate_load_configs(resolved_device):
                self._emit_progress(
                    f"Initialising model on {device} ({compute_type}) …"
                )
                try:
                    self.model = WhisperModel(
                        self.model_size,
                        device=device,
                        compute_type=compute_type,
                    )
                    self._emit_progress("Whisper model ready.")
                    if pyqtSignal is not None:
                        self.model_ready.emit()  # type: ignore[attr-defined]
                    logger.info(
                        "ModelLoader: model '%s' loaded on %s (%s).",
                        self.model_size,
                        device,
                        compute_type,
                    )
                    return
                except Exception as exc:  # pylint: disable=broad-except
                    diagnostic = self._diagnose_load_error(exc)
                    errors.append(
                        f"{device}/{compute_type}: {exc}"
                    )
                    logger.error(
                        "ModelLoader: load attempt failed on %s (%s): %s",
                        device,
                        compute_type,
                        diagnostic,
                        exc_info=True,
                    )

            detail = "; ".join(errors) if errors else "unknown error"
            msg = (
                "Failed to load Whisper model after fallback attempts. "
                f"Attempts: {detail}"
            )
            logger.error("%s", msg)
            if pyqtSignal is not None:
                self.model_error.emit(msg)  # type: ignore[attr-defined]
        except Exception as exc:  # pylint: disable=broad-except
            diagnostic = self._diagnose_load_error(exc)
            msg = f"Failed to load Whisper model: {exc}. {diagnostic}"
            logger.error(msg, exc_info=True)
            if pyqtSignal is not None:
                self.model_error.emit(msg)  # type: ignore[attr-defined]

    def _candidate_load_configs(self, resolved_device: str) -> List[Tuple[str, str]]:
        candidates: List[Tuple[str, str]] = []
        preferred = (resolved_device, self.compute_type)
        candidates.append(preferred)

        if resolved_device == "cuda":
            fallback_chain = ["float16", "int8_float16", "int8", "float32"]
        else:
            fallback_chain = ["int8", "float32"]

        for compute_type in fallback_chain:
            item = (resolved_device, compute_type)
            if item not in candidates:
                candidates.append(item)

        if resolved_device != "cpu":
            cpu_fallbacks = ["int8", "float32"]
            for compute_type in cpu_fallbacks:
                item = ("cpu", compute_type)
                if item not in candidates:
                    candidates.append(item)

        return candidates

    def _diagnose_load_error(self, exc: Exception) -> str:
        text = str(exc).lower()
        if "winerror 1114" in text or "dynamic link library" in text:
            return (
                "Windows DLL initialisation failed. Install/repair the Microsoft "
                "Visual C++ 2015-2022 x64 Redistributable and ensure PyInstaller "
                "includes torch/ctranslate2/onnxruntime runtime DLLs."
            )
        if "dll load failed" in text:
            return "A required native dependency DLL is missing or incompatible."
        if "no module named" in text:
            return "A Python dependency is missing from the runtime environment."
        return "See logs for the full traceback and failing dependency."

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch  # type: ignore[import]
            if torch.cuda.is_available():
                return "cuda"
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "ModelLoader: failed to probe CUDA via torch (%s). Falling back to CPU.",
                exc,
            )
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

    def has_model(self) -> bool:
        return self._model is not None

    def is_ready(self) -> bool:
        return self.isRunning() and self.has_model() and not self._stop_event.is_set()

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
            # avg_logprob is in [-inf, 0]; adding 1.0 maps the typical
            # range [-1, 0] to [0, 1] as an approximate confidence score.
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
