"""
Real-time speech-to-text transcription using faster-whisper.

Runs in a dedicated QThread, receives speech segments from the VAD
pipeline, and emits TranscriptSegment results.
"""

from __future__ import annotations

import io
import queue
import random
import re
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ghostmic.utils.errors import is_rate_limited as _is_rate_limited_shared


import numpy as np

# NOTE: torch is imported lazily (inside methods) to avoid a Windows DLL
# conflict when PyQt6's QApplication is created before torch is loaded.
# See _preload_torch_runtime() in main.py which ensures torch is loaded
# before QApplication on startup.

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    QThread = object  # type: ignore[misc,assignment]
    pyqtSignal = None  # type: ignore[assignment]

from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

TRANSCRIPTION_QUEUE_MAXSIZE: int = 24
MAX_PENDING_SEGMENT_AGE_SECONDS: float = 8.0
REMOTE_RATE_LIMIT_RETRIES: int = 3
REMOTE_RATE_LIMIT_BASE_DELAY: float = 1.0
REMOTE_RATE_LIMIT_MAX_DELAY: float = 8.0
REMOTE_REPEAT_WINDOW_SECONDS: float = 3.0
REMOTE_REPEAT_MAX_CHARS: int = 42
LOCAL_PROMPT_MAX_CHARS: int = 240
SOURCE_STATE_MAX_ENTRIES: int = 16
SOURCE_STATE_TTL_SECONDS: float = 900.0
LOCAL_REPEAT_WINDOW_SECONDS: float = 2.5
LOCAL_REPEAT_MAX_CHARS: int = 48
DEFAULT_MIN_SEGMENT_SECONDS: float = 0.45
DEFAULT_MIN_SEGMENT_RMS: float = 140.0
DEFAULT_TARGET_RMS: float = 2200.0
DEFAULT_MAX_GAIN: float = 8.0
DEFAULT_SILENCE_TRIM_THRESHOLD: int = 220
DEFAULT_SILENCE_TRIM_PAD_SECONDS: float = 0.08


from ghostmic.domain import TranscriptSegment  # re-exported for backward compat

__all__ = ["TranscriptSegment", "ModelLoader", "TranscriptionThread"]


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
        ai_config: Optional[dict] = None,
        remote_config: Optional[dict] = None,
        on_result: Optional[Callable[[TranscriptSegment], None]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._language = language
        self._beam_size = beam_size
        self._ai_config = ai_config or {}
        self._remote_config = remote_config or {}
        self._on_result = on_result
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[Tuple[np.ndarray, str, float]]" = queue.Queue(
            maxsize=TRANSCRIPTION_QUEUE_MAXSIZE
        )
        self._remote_transcription_enabled = False
        self._remote_backend: Optional[str] = None
        self._remote_model: Optional[str] = None
        self._remote_api_key: Optional[str] = None
        self._remote_base_url: Optional[str] = None
        self._remote_client = None
        self._max_pending_age_seconds = float(
            self._remote_config.get(
                "max_pending_segment_age_seconds",
                MAX_PENDING_SEGMENT_AGE_SECONDS,
            )
        )
        self._use_context_prompt = bool(
            self._remote_config.get("use_context_prompt", False)
        )
        self._min_segment_seconds = float(
            self._remote_config.get("min_segment_seconds", DEFAULT_MIN_SEGMENT_SECONDS)
        )
        self._min_segment_rms = float(
            self._remote_config.get("min_segment_rms", DEFAULT_MIN_SEGMENT_RMS)
        )
        self._trim_silence = bool(self._remote_config.get("trim_silence", True))
        self._silence_trim_threshold = int(
            self._remote_config.get(
                "silence_trim_threshold",
                DEFAULT_SILENCE_TRIM_THRESHOLD,
            )
        )
        self._silence_trim_pad_seconds = float(
            self._remote_config.get(
                "silence_trim_pad_seconds",
                DEFAULT_SILENCE_TRIM_PAD_SECONDS,
            )
        )
        self._target_rms = float(self._remote_config.get("target_rms", DEFAULT_TARGET_RMS))
        self._max_gain = float(self._remote_config.get("max_gain", DEFAULT_MAX_GAIN))
        self._last_local_emit_text_by_source: Dict[str, str] = {}
        self._last_local_emit_ts_by_source: Dict[str, float] = {}
        self._last_queue_drop_log = 0.0
        self._last_remote_text_by_source: Dict[str, str] = {}
        self._last_remote_text_ts_by_source: Dict[str, float] = {}
        self._last_local_text_by_source: Dict[str, str] = {}
        self._last_local_text_ts_by_source: Dict[str, float] = {}

    def set_model(self, model) -> None:
        """Attach a (newly loaded) model."""
        self._model = model

    def stop(self) -> None:
        self._stop_event.set()

    def has_model(self) -> bool:
        return self._model is not None

    def has_remote_transcriber(self) -> bool:
        return self._remote_transcription_enabled

    def enable_remote_fallback(self) -> Tuple[bool, str]:
        """Enable cloud transcription fallback using configured AI provider."""
        if self._remote_transcription_enabled:
            detail = f"{self._remote_backend}/{self._remote_model}"
            return True, detail

        enabled, detail = self._configure_remote_transcriber()
        if enabled:
            self._remote_transcription_enabled = True
            logger.info("TranscriptionThread: remote fallback enabled (%s)", detail)
            return True, detail
        logger.warning("TranscriptionThread: remote fallback unavailable (%s)", detail)
        return False, detail

    def is_ready(self) -> bool:
        return bool(
            self.isRunning()
            and (self.has_model() or self.has_remote_transcriber())
            and not self._stop_event.is_set()
        )

    def push_segment(self, audio: np.ndarray, source: str) -> None:
        """Enqueue an audio segment for transcription."""
        item = (audio, source, time.time())
        try:
            self._queue.put_nowait(item)
            return
        except queue.Full:
            # Keep near-real-time behavior: discard oldest and keep latest.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

        try:
            self._queue.put_nowait(item)
        except queue.Full:
            now = time.time()
            if now - self._last_queue_drop_log >= 5.0:
                logger.warning("TranscriptionThread: queue full; dropping incoming segment.")
                self._last_queue_drop_log = now

    def clear_pending_segments(self) -> int:
        """Drop queued segments that have not started transcription yet."""
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        if dropped:
            logger.debug("TranscriptionThread: dropped %d pending queued segment(s).", dropped)
        return dropped

    def run(self) -> None:
        self._stop_event.clear()
        logger.info("TranscriptionThread: started.")
        while not self._stop_event.is_set():
            try:
                audio, source, enqueued_at = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            age = time.time() - enqueued_at
            if age > self._max_pending_age_seconds:
                logger.debug(
                    "TranscriptionThread: dropping stale segment (%s, age=%.2fs)",
                    source,
                    age,
                )
                continue

            if self._model is None and not self._remote_transcription_enabled:
                logger.warning(
                    "TranscriptionThread: no model loaded – dropping segment."
                )
                continue

            if pyqtSignal is not None:
                self.transcribing.emit(source)  # type: ignore[attr-defined]

            segment = self._transcribe(audio, source, segment_timestamp=enqueued_at)
            if segment:
                if self._on_result:
                    self._on_result(segment)
                if pyqtSignal is not None:
                    self.transcription_ready.emit(segment)  # type: ignore[attr-defined]

        logger.info("TranscriptionThread: stopped.")

    def _transcribe(
        self,
        audio: np.ndarray,
        source: str,
        segment_timestamp: Optional[float] = None,
    ) -> Optional[TranscriptSegment]:
        """Run whisper inference on *audio* and return a TranscriptSegment."""
        try:
            transcript_ts = float(segment_timestamp) if segment_timestamp else time.time()
            if self._model is None:
                return self._transcribe_remote(
                    audio,
                    source,
                    segment_timestamp=transcript_ts,
                )

            processed = self._prepare_local_audio(audio)
            if processed is None:
                return None

            audio_float = processed.astype(np.float32) / 32768.0
            transcribe_kwargs: Dict[str, object] = {
                "language": self._language,
                "beam_size": self._beam_size,
                "best_of": self._beam_size,
                "word_timestamps": False,
                "no_speech_threshold": float(
                    self._remote_config.get("no_speech_threshold", 0.7)
                ),
                "log_prob_threshold": float(
                    self._remote_config.get("log_prob_threshold", -1.0)
                ),
                "compression_ratio_threshold": float(
                    self._remote_config.get("compression_ratio_threshold", 2.0)
                ),
                "temperature": float(self._remote_config.get("temperature", 0.0)),
                "patience": float(self._remote_config.get("patience", 1.2)),
                "repetition_penalty": float(
                    self._remote_config.get("repetition_penalty", 1.05)
                ),
                "vad_filter": True,
                "condition_on_previous_text": False,
            }
            if self._use_context_prompt:
                initial_prompt = self._build_initial_prompt(source)
                if initial_prompt:
                    transcribe_kwargs["initial_prompt"] = initial_prompt

            segments, _ = self._model.transcribe(audio_float, **transcribe_kwargs)

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

            if self._should_drop_local_artifact(full_text, confidence, source):
                return None

            self._remember_local_text(source, full_text)

            return TranscriptSegment(
                text=full_text,
                source=source,
                timestamp=transcript_ts,
                confidence=confidence,
            )

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("TranscriptionThread: error: %s", exc, exc_info=True)
            return None

    def _configure_remote_transcriber(self) -> Tuple[bool, str]:
        if not self._remote_config.get("remote_fallback", True):
            return False, "disabled in transcription settings"

        preferred = str(self._remote_config.get("remote_backend", "auto")).lower()
        ai_backend = str(
            self._ai_config.get("main_backend")
            or self._ai_config.get("backend", "groq")
        ).lower()

        candidates: List[str] = []
        if preferred in ("groq", "openai"):
            candidates.append(preferred)
        elif ai_backend in ("groq", "openai"):
            candidates.append(ai_backend)

        for provider in ("groq", "openai"):
            if provider not in candidates:
                candidates.append(provider)

        init_errors: List[str] = []
        for provider in candidates:
            if provider == "groq":
                api_key = str(self._ai_config.get("groq_api_key", "")).strip()
                if not api_key:
                    continue
                self._remote_backend = "groq"
                self._remote_api_key = api_key
                self._remote_base_url = "https://api.groq.com/openai/v1"
                self._remote_model = str(
                    self._remote_config.get(
                        "remote_model_groq", "whisper-large-v3-turbo"
                    )
                )
                ok, detail = self._init_remote_client()
                if ok:
                    return True, f"groq/{self._remote_model}"
                init_errors.append(detail)
                self._remote_client = None
                continue

            if provider == "openai":
                api_key = str(self._ai_config.get("openai_api_key", "")).strip()
                if not api_key:
                    continue
                self._remote_backend = "openai"
                self._remote_api_key = api_key
                self._remote_base_url = None
                self._remote_model = str(
                    self._remote_config.get(
                        "remote_model_openai", "gpt-4o-mini-transcribe"
                    )
                )
                ok, detail = self._init_remote_client()
                if ok:
                    return True, f"openai/{self._remote_model}"
                init_errors.append(detail)
                self._remote_client = None
                continue

        if init_errors:
            return False, "; ".join(init_errors)

        return False, "no configured API key for remote transcription"

    def _init_remote_client(self) -> Tuple[bool, str]:
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError:
            return False, "openai package missing for remote STT"

        if not self._remote_api_key:
            return False, "remote API key unavailable"

        try:
            if self._remote_base_url:
                self._remote_client = OpenAI(
                    api_key=self._remote_api_key,
                    base_url=self._remote_base_url,
                )
            else:
                self._remote_client = OpenAI(api_key=self._remote_api_key)
        except Exception as exc:  # pylint: disable=broad-except
            return False, f"failed to initialize remote client: {exc}"

        return True, "ready"

    def _transcribe_remote(
        self,
        audio: np.ndarray,
        source: str,
        segment_timestamp: Optional[float] = None,
    ) -> Optional[TranscriptSegment]:
        if not self._remote_transcription_enabled:
            return None

        if not self._remote_api_key or not self._remote_model:
            logger.error("TranscriptionThread: remote STT not configured correctly.")
            return None

        if self._remote_client is None:
            logger.error("TranscriptionThread: remote STT client unavailable.")
            return None

        try:
            client = self._remote_client

            wav_bytes = self._to_wav_bytes(audio)
            wav_file = io.BytesIO(wav_bytes)
            wav_file.name = "segment.wav"

            request_kwargs: Dict[str, object] = {
                "model": self._remote_model,
                "file": wav_file,
            }
            if self._language and self._language.lower() not in ("auto", ""):
                request_kwargs["language"] = self._language

            retries = int(
                self._remote_config.get("remote_rate_limit_retries", REMOTE_RATE_LIMIT_RETRIES)
            )
            base_delay = float(
                self._remote_config.get(
                    "remote_rate_limit_retry_delay",
                    REMOTE_RATE_LIMIT_BASE_DELAY,
                )
            )
            retries = max(1, retries)

            for attempt in range(retries):
                if self._stop_event.is_set():
                    return None

                try:
                    wav_file.seek(0)
                    response = client.audio.transcriptions.create(**request_kwargs)
                    text = str(getattr(response, "text", "")).strip()
                    if not text and hasattr(response, "model_dump"):
                        dumped = response.model_dump()
                        text = str(dumped.get("text", "")).strip()

                    if not text:
                        return None

                    if self._should_drop_remote_repeat(text, source):
                        return None

                    return TranscriptSegment(
                        text=text,
                        source=source,
                        timestamp=float(segment_timestamp) if segment_timestamp else time.time(),
                        confidence=0.75,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    is_rate_limited = self._is_rate_limited_exception(exc)
                    can_retry = (
                        is_rate_limited
                        and attempt < retries - 1
                        and not self._stop_event.is_set()
                    )
                    if can_retry:
                        delay = min(
                            REMOTE_RATE_LIMIT_MAX_DELAY,
                            base_delay * (2**attempt),
                        ) + random.uniform(0.0, 0.25)
                        logger.warning(
                            "TranscriptionThread: remote STT rate limited, retrying in %.2fs "
                            "(attempt %d/%d)",
                            delay,
                            attempt + 1,
                            retries,
                        )
                        self._stop_event.wait(delay)
                        continue

                    logger.error(
                        "TranscriptionThread: remote transcription failed (%s/%s): %s",
                        self._remote_backend,
                        self._remote_model,
                        exc,
                        exc_info=True,
                    )
                    return None
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "TranscriptionThread: remote transcription failed (%s/%s): %s",
                self._remote_backend,
                self._remote_model,
                exc,
                exc_info=True,
            )
            return None

    @staticmethod
    def _is_rate_limited_exception(exc: Exception) -> bool:
        """Delegate to shared utility. Kept for backward compatibility."""
        return _is_rate_limited_shared(exc)

    def _should_drop_remote_repeat(self, text: str, source: str) -> bool:
        now = time.time()
        normalized = self._normalize_remote_text(text)
        if not normalized:
            return True

        self._prune_source_state(
            self._last_remote_text_by_source,
            self._last_remote_text_ts_by_source,
            now,
        )

        prev = self._last_remote_text_by_source.get(source)
        prev_ts = self._last_remote_text_ts_by_source.get(source, 0.0)
        self._last_remote_text_by_source[source] = normalized
        self._last_remote_text_ts_by_source[source] = now

        if (
            prev == normalized
            and len(normalized) <= REMOTE_REPEAT_MAX_CHARS
            and (now - prev_ts) <= REMOTE_REPEAT_WINDOW_SECONDS
        ):
            logger.debug(
                "TranscriptionThread: dropped repeated remote text for %s: %r",
                source,
                normalized,
            )
            return True

        return False

    def _build_initial_prompt(self, source: str) -> Optional[str]:
        now = time.time()
        self._prune_source_state(
            self._last_local_text_by_source,
            self._last_local_text_ts_by_source,
            now,
        )
        prompt = self._last_local_text_by_source.get(source, "")
        if not prompt:
            return None
        return prompt[-LOCAL_PROMPT_MAX_CHARS:]

    def _remember_local_text(self, source: str, text: str) -> None:
        now = time.time()
        normalized = self._normalize_local_text(text)
        if not normalized:
            return
        self._last_local_text_by_source[source] = normalized[-LOCAL_PROMPT_MAX_CHARS:]
        self._last_local_text_ts_by_source[source] = now
        self._prune_source_state(
            self._last_local_text_by_source,
            self._last_local_text_ts_by_source,
            now,
        )

    def _prune_source_state(
        self,
        text_map: Dict[str, str],
        ts_map: Dict[str, float],
        now: Optional[float] = None,
    ) -> None:
        current = time.time() if now is None else now
        stale_sources = [
            source
            for source, ts in ts_map.items()
            if (current - ts) > SOURCE_STATE_TTL_SECONDS
        ]
        for source in stale_sources:
            text_map.pop(source, None)
            ts_map.pop(source, None)

        if len(text_map) <= SOURCE_STATE_MAX_ENTRIES:
            return

        overflow = len(text_map) - SOURCE_STATE_MAX_ENTRIES
        oldest = sorted(ts_map.items(), key=lambda item: item[1])
        for source, _ in oldest[:overflow]:
            text_map.pop(source, None)
            ts_map.pop(source, None)

    @staticmethod
    def _normalize_local_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    @staticmethod
    def _normalize_remote_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _prepare_local_audio(self, audio: np.ndarray) -> Optional[np.ndarray]:
        mono = np.asarray(audio, dtype=np.int16)
        if mono.size == 0:
            return None

        min_samples = int(16_000 * max(0.2, self._min_segment_seconds))
        if mono.size < min_samples:
            return None

        if self._trim_silence:
            mono = self._trim_edges(mono)
            if mono.size < min_samples:
                return None

        centered = mono.astype(np.float32)
        centered -= float(np.mean(centered))

        rms = float(np.sqrt(np.mean(np.square(centered))))
        if rms < self._min_segment_rms:
            return None

        gain = self._target_rms / max(rms, 1e-6)
        gain = min(self._max_gain, max(0.5, gain))
        boosted = centered * gain
        boosted = np.clip(boosted, -32768.0, 32767.0)
        return boosted.astype(np.int16)

    def _trim_edges(self, audio: np.ndarray) -> np.ndarray:
        threshold = max(1, self._silence_trim_threshold)
        active = np.flatnonzero(np.abs(audio.astype(np.int32)) >= threshold)
        if active.size == 0:
            return audio
        pad = int(16_000 * max(0.0, self._silence_trim_pad_seconds))
        start = max(0, int(active[0]) - pad)
        end = min(audio.size, int(active[-1]) + pad + 1)
        return audio[start:end]

    def _should_drop_local_artifact(
        self,
        text: str,
        confidence: float,
        source: str,
    ) -> bool:
        now = time.time()
        normalized = self._normalize_remote_text(text)
        if not normalized:
            return True

        words = normalized.split()
        if len(words) >= 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.35 and confidence < 0.55:
                logger.debug(
                    "TranscriptionThread: dropped low-diversity local transcript (%s): %r",
                    source,
                    normalized,
                )
                return True

        self._prune_source_state(
            self._last_local_emit_text_by_source,
            self._last_local_emit_ts_by_source,
            now,
        )
        prev = self._last_local_emit_text_by_source.get(source)
        prev_ts = self._last_local_emit_ts_by_source.get(source, 0.0)

        self._last_local_emit_text_by_source[source] = normalized
        self._last_local_emit_ts_by_source[source] = now

        if (
            prev == normalized
            and len(normalized) <= LOCAL_REPEAT_MAX_CHARS
            and (now - prev_ts) <= LOCAL_REPEAT_WINDOW_SECONDS
        ):
            logger.debug(
                "TranscriptionThread: dropped repeated local text for %s: %r",
                source,
                normalized,
            )
            return True

        return False

    @staticmethod
    def _to_wav_bytes(audio: np.ndarray) -> bytes:
        mono = np.asarray(audio, dtype=np.int16)
        with io.BytesIO() as buff:
            with wave.open(buff, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16_000)
                wf.writeframes(mono.tobytes())
            return buff.getvalue()
