"""
Real-time speech-to-text transcription using faster-whisper.

Runs in a dedicated QThread, receives speech segments from the VAD
pipeline, and emits TranscriptSegment results.
"""

from __future__ import annotations

import io
import hashlib
import os
import queue
import random
import re
import threading
import time
import wave
from collections import deque
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


def resolve_faster_whisper_vad_asset() -> tuple[str, bool]:
    """Resolve the bundled faster-whisper VAD asset without assuming _MEIPASS."""
    try:
        import faster_whisper  # type: ignore[import]

        package_dir = os.path.dirname(os.path.abspath(faster_whisper.__file__))
        asset_path = os.path.join(package_dir, "assets", "silero_vad_v6.onnx")
        return asset_path, bool(os.path.isfile(asset_path) and os.access(asset_path, os.R_OK))
    except Exception:  # pylint: disable=broad-except
        return "", False


def log_faster_whisper_vad_asset() -> bool:
    """Log a concise packaged-runtime diagnostic for the faster-whisper asset."""
    path, available = resolve_faster_whisper_vad_asset()
    logger.info("faster-whisper VAD asset: exists=%s path=%s", str(available).lower(), path)
    return available

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
DEFAULT_MIN_SEGMENT_RMS_MIC: float = 70.0
DEFAULT_TARGET_RMS: float = 2200.0
DEFAULT_MAX_GAIN: float = 8.0
DEFAULT_SILENCE_TRIM_THRESHOLD: int = 220
DEFAULT_SILENCE_TRIM_PAD_SECONDS: float = 0.08
DEFAULT_OVERLAP_MS: float = 0.0
DEFAULT_MAX_CONTEXT_SECONDS: float = 2.0
MAX_DIAGNOSTIC_HISTORY: int = 128


@dataclass
class _TranscriptionRequest:
    request_id: str
    source: str
    session_id: int | None
    chunk_ids: tuple[str, ...]
    timestamp_start: float
    timestamp_end: float
    enqueued_at: float
    status: str = "partial"


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
            logger.info("local_transcription_vad_filter=false")
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
                    log_faster_whisper_vad_asset()
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
        transcription_diagnostic = pyqtSignal(dict)

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
        self._accepting_segments = True
        self._queue: "queue.Queue[Tuple[np.ndarray, str, float, int | None]]" = queue.Queue(
            maxsize=TRANSCRIPTION_QUEUE_MAXSIZE
        )
        self._request_lock = threading.Lock()
        self._request_metadata: Dict[Tuple[str, str, float], _TranscriptionRequest] = {}
        self._seen_request_ids: deque[str] = deque(maxlen=TRANSCRIPTION_QUEUE_MAXSIZE * 8)
        self._seen_request_id_set: set[str] = set()
        self._request_counter = 0
        self._last_finalized_end: Dict[Tuple[int | None, str], float] = {}
        self._diagnostic_history: deque[dict] = deque(maxlen=MAX_DIAGNOSTIC_HISTORY)
        self._overlap_ms = max(0.0, min(1000.0, float(self._remote_config.get("overlap_ms", DEFAULT_OVERLAP_MS))))
        self._max_context_seconds = max(
            0.0,
            min(10.0, float(self._remote_config.get("max_context_seconds", DEFAULT_MAX_CONTEXT_SECONDS))),
        )
        self._audio_context: Dict[Tuple[int | None, str], np.ndarray] = {}
        self._stage_metrics: Dict[str, dict[str, float]] = {}
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
        self._min_segment_rms_mic = float(
            self._remote_config.get("min_segment_rms_mic", DEFAULT_MIN_SEGMENT_RMS_MIC)
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

    def stop_accepting_segments(self) -> None:
        """Prevent new audio from entering while queued work drains."""
        self._accepting_segments = False

    def resume_accepting_segments(self) -> None:
        """Allow audio segments for a newly started recording session."""
        self._accepting_segments = True

    def drain(self, timeout_seconds: float = 3.0) -> bool:
        """Wait briefly for queued and in-flight transcription to finish."""
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while time.monotonic() < deadline:
            if self._queue.empty() and not getattr(self, "_transcribing", False):
                return True
            time.sleep(0.01)
        return self._queue.empty() and not getattr(self, "_transcribing", False)

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

    def push_segment(
        self,
        audio: np.ndarray,
        source: str,
        session_id: int | None = None,
        *,
        chunk_ids: Optional[List[str]] = None,
        timestamp_start: Optional[float] = None,
        timestamp_end: Optional[float] = None,
        partial: bool = True,
    ) -> bool:
        """Enqueue bounded audio context for transcription.

        The queue tuple remains backward-compatible; request identity and timing
        live in a side map so callers can opt into chunk-level ordering metadata.
        """
        if not self._accepting_segments:
            logger.debug(
                "TranscriptionThread: rejecting new segment while stopping (%s).",
                source,
            )
            return False
        now = time.time()
        source_name = str(source or "speaker").strip().lower() or "speaker"
        audio_array = np.asarray(audio, dtype=np.int16)
        end = float(timestamp_end if timestamp_end is not None else now)
        start = float(
            timestamp_start
            if timestamp_start is not None
            else end - (audio_array.size / 16_000.0)
        )
        ids = tuple(str(item).strip() for item in (chunk_ids or []) if str(item).strip())
        request_id = self._request_identity(session_id, source_name, start, end, ids)
        with self._request_lock:
            if request_id in self._seen_request_id_set:
                self._emit_diagnostic(
                    segment_id=request_id,
                    source=source_name,
                    dropped=True,
                    stale=False,
                    status="duplicate",
                    text_chars=0,
                )
                return False
            self._remember_request_id(request_id)

        context_key = (session_id, source_name)
        queued_audio = audio_array
        if self._overlap_ms > 0 and self._max_context_seconds > 0:
            prior = self._audio_context.get(context_key)
            overlap_samples = min(audio_array.size, int(16_000 * self._overlap_ms / 1000.0))
            if prior is not None and overlap_samples:
                queued_audio = np.concatenate((prior[-overlap_samples:], audio_array))
            max_samples = int(16_000 * self._max_context_seconds)
            self._audio_context[context_key] = np.concatenate((
                (prior if prior is not None else np.empty(0, dtype=np.int16)),
                audio_array,
            ))[-max_samples:]
            if len(self._audio_context) > SOURCE_STATE_MAX_ENTRIES:
                self._audio_context.pop(next(iter(self._audio_context)))

        item = (queued_audio, source_name, now, session_id)
        request = _TranscriptionRequest(
            request_id=request_id,
            source=source_name,
            session_id=session_id,
            chunk_ids=ids,
            timestamp_start=start,
            timestamp_end=end,
            enqueued_at=now,
            status="partial" if partial else "finalized",
        )
        with self._request_lock:
            self._request_metadata[(source_name, str(session_id), now)] = request
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            # Keep near-real-time behavior: discard oldest and keep latest.
            try:
                dropped_item = self._queue.get_nowait()
                self._mark_queued_item(dropped_item, status="dropped")
            except queue.Empty:
                self._mark_request(request, status="dropped")
                self._forget_request(request)
                return False

        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            self._mark_request(request, status="dropped")
            self._forget_request(request)
            now = time.time()
            if now - self._last_queue_drop_log >= 5.0:
                logger.warning("TranscriptionThread: queue full; dropping incoming segment.")
                self._last_queue_drop_log = now
            return False

    @staticmethod
    def _request_identity(
        session_id: int | None,
        source: str,
        timestamp_start: float,
        timestamp_end: float,
        chunk_ids: tuple[str, ...],
    ) -> str:
        material = "|".join(
            [
                str(session_id),
                source,
                f"{timestamp_start:.6f}",
                f"{timestamp_end:.6f}",
                ",".join(chunk_ids),
            ]
        )
        return hashlib.sha1(material.encode("utf-8")).hexdigest()[:20]

    def _remember_request_id(self, request_id: str) -> None:
        if len(self._seen_request_ids) >= self._seen_request_ids.maxlen:
            expired = self._seen_request_ids.popleft()
            self._seen_request_id_set.discard(expired)
        self._seen_request_ids.append(request_id)
        self._seen_request_id_set.add(request_id)

    def _request_for_item(
        self,
        source: str,
        session_id: int | None,
        enqueued_at: float,
    ) -> Optional[_TranscriptionRequest]:
        with self._request_lock:
            return self._request_metadata.get((source, str(session_id), enqueued_at))

    def _mark_queued_item(self, item, *, status: str) -> None:
        _audio, source, enqueued_at, session_id = item
        request = self._request_for_item(source, session_id, enqueued_at)
        if request is not None:
            self._mark_request(request, status=status)
            self._forget_request(request)

    def _mark_request(self, request: _TranscriptionRequest, *, status: str) -> None:
        request.status = status
        self._emit_diagnostic(
            segment_id=request.request_id,
            source=request.source,
            queue_wait_ms=max(0.0, (time.time() - request.enqueued_at) * 1000.0),
            text_chars=0,
            dropped=status in {"dropped", "empty", "error", "duplicate"},
            stale=status == "stale",
            status=status,
        )

    def _forget_request(self, request: _TranscriptionRequest) -> None:
        with self._request_lock:
            keys = [
                key for key, value in self._request_metadata.items()
                if value is request
            ]
            for key in keys:
                self._request_metadata.pop(key, None)

    def _emit_diagnostic(
        self,
        *,
        segment_id: str,
        source: str,
        queue_wait_ms: float = 0.0,
        preprocessing_ms: float = 0.0,
        inference_ms: float = 0.0,
        postprocessing_ms: float = 0.0,
        context_normalization_ms: float = 0.0,
        text_chars: int = 0,
        dropped: bool = False,
        stale: bool = False,
        status: str = "finalized",
    ) -> dict:
        diagnostic = {
            "segment_id": segment_id,
            "queue_wait_ms": round(queue_wait_ms, 2),
            "preprocessing_ms": round(preprocessing_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "postprocessing_ms": round(postprocessing_ms, 2),
            "context_normalization_ms": round(context_normalization_ms, 2),
            "text_chars": int(max(0, text_chars)),
            "source": source,
            "dropped": bool(dropped),
            "stale": bool(stale),
            "status": status,
        }
        self._diagnostic_history.append(diagnostic)
        if pyqtSignal is not None:
            self.transcription_diagnostic.emit(diagnostic)  # type: ignore[attr-defined]
        return diagnostic

    def get_diagnostic_history(self) -> list[dict]:
        return list(self._diagnostic_history)

    def clear_pending_segments(self) -> int:
        """Drop queued segments that have not started transcription yet."""
        dropped = 0
        while True:
            try:
                item = self._queue.get_nowait()
                self._mark_queued_item(item, status="dropped")
                dropped += 1
            except queue.Empty:
                break
        if dropped:
            logger.debug("TranscriptionThread: dropped %d pending queued segment(s).", dropped)
        return dropped

    def run(self) -> None:
        self._stop_event.clear()
        self._accepting_segments = True
        logger.info("TranscriptionThread: started.")
        while not self._stop_event.is_set():
            try:
                audio, source, enqueued_at, session_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            request = self._request_for_item(source, session_id, enqueued_at)
            age = time.time() - enqueued_at
            if age > self._max_pending_age_seconds:
                logger.debug(
                    "TranscriptionThread: dropping stale segment (%s, age=%.2fs)",
                    source,
                    age,
                )
                if request is not None:
                    self._mark_request(request, status="stale")
                    self._forget_request(request)
                continue

            if self._model is None and not self._remote_transcription_enabled:
                logger.warning(
                    "TranscriptionThread: no model loaded – dropping segment."
                )
                if request is not None:
                    self._mark_request(request, status="dropped")
                    self._forget_request(request)
                continue

            if pyqtSignal is not None:
                self.transcribing.emit(source)  # type: ignore[attr-defined]

            self._transcribing = True
            try:
                started = time.perf_counter()
                segment = self._transcribe(
                    audio,
                    source,
                    segment_timestamp=(request.timestamp_end if request else enqueued_at),
                    request=request,
                )
                if segment and self._accept_result(segment, request):
                    if request is not None:
                        request.status = "finalized"
                    segment.session_id = session_id
                    if self._on_result:
                        self._on_result(segment)
                    if pyqtSignal is not None:
                        self.transcription_ready.emit(segment)  # type: ignore[attr-defined]
                elif segment is None and request is not None and request.status == "partial":
                    self._mark_request(request, status="empty")
                if request is not None and request.status == "partial":
                    request.status = "finalized"
                if request is not None:
                    stage_metrics = self._stage_metrics.pop(request.request_id, {})
                    self._emit_diagnostic(
                        segment_id=request.request_id,
                        source=source,
                        queue_wait_ms=max(0.0, (time.time() - request.enqueued_at) * 1000.0),
                        preprocessing_ms=stage_metrics.get("preprocessing_ms", 0.0),
                        inference_ms=stage_metrics.get(
                            "inference_ms", (time.perf_counter() - started) * 1000.0
                        ),
                        postprocessing_ms=stage_metrics.get("postprocessing_ms", 0.0),
                        text_chars=len(getattr(segment, "text", "") or "") if segment else 0,
                        dropped=segment is None,
                        stale=request.status == "stale",
                        status=request.status,
                    )
                    self._forget_request(request)
            finally:
                self._transcribing = False

        logger.info("TranscriptionThread: stopped.")

    def _accept_result(
        self,
        segment: TranscriptSegment,
        request: Optional[_TranscriptionRequest],
    ) -> bool:
        """Accept only the newest finalized interval for a session/source."""
        if request is None:
            return bool(str(getattr(segment, "text", "") or "").strip())
        ordering_key = (request.session_id, request.source)
        with self._request_lock:
            last_end = self._last_finalized_end.get(ordering_key, float("-inf"))
            if request.timestamp_end <= last_end:
                request.status = "stale"
                self._emit_diagnostic(
                    segment_id=request.request_id,
                    source=request.source,
                    dropped=False,
                    stale=True,
                    status="stale",
                    text_chars=len(getattr(segment, "text", "") or ""),
                )
                return False
            self._last_finalized_end[ordering_key] = request.timestamp_end
            segment.status = "finalized"
            return True

    def _transcribe(
        self,
        audio: np.ndarray,
        source: str,
        segment_timestamp: Optional[float] = None,
        request: Optional[_TranscriptionRequest] = None,
    ) -> Optional[TranscriptSegment]:
        """Run whisper inference on *audio* and return a TranscriptSegment."""
        try:
            transcript_ts = float(segment_timestamp) if segment_timestamp else time.time()
            if self._model is None:
                return self._transcribe_remote(
                    audio,
                    source,
                    segment_timestamp=transcript_ts,
                    request=request,
                )

            preprocessing_started = time.perf_counter()
            processed = self._prepare_local_audio(audio, source=source)
            if processed is None:
                return None
            preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000.0

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
                # VADThread already segments audio; disabled by default to keep
                # latency low and avoid a second silence boundary pass.
                "vad_filter": bool(self._remote_config.get("vad_filter", False)),
                # Independent chunks avoid Whisper carrying stale text across
                # VAD boundaries; callers can opt in when using larger chunks.
                "condition_on_previous_text": bool(
                    self._remote_config.get("condition_on_previous_text", False)
                ),
            }
            if self._use_context_prompt:
                initial_prompt = self._build_initial_prompt(source)
                if initial_prompt:
                    transcribe_kwargs["initial_prompt"] = initial_prompt

            inference_started = time.perf_counter()
            segments, _ = self._model.transcribe(audio_float, **transcribe_kwargs)
            inference_ms = (time.perf_counter() - inference_started) * 1000.0

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

            postprocessing_started = time.perf_counter()
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
            result = TranscriptSegment(
                text=full_text,
                source=source,
                timestamp=transcript_ts,
                confidence=confidence,
                raw_stt_text=full_text,
                normalized_text=full_text,
                segment_id=request.request_id if request else self._request_identity(
                    None, source, transcript_ts, transcript_ts, ()
                ),
                chunk_ids=list(request.chunk_ids) if request else [],
                timestamp_start=request.timestamp_start if request else transcript_ts,
                timestamp_end=request.timestamp_end if request else transcript_ts,
                status="finalized",
            )
            if request is not None:
                self._stage_metrics[request.request_id] = {
                    "preprocessing_ms": preprocessing_ms,
                    "inference_ms": inference_ms,
                    "postprocessing_ms": (time.perf_counter() - postprocessing_started) * 1000.0,
                }
            return result

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("TranscriptionThread: error: %s", exc, exc_info=True)
            if request is not None:
                self._mark_request(request, status="retry")
            if self._model is not None and self._remote_transcription_enabled:
                try:
                    return self._transcribe_remote(
                        audio,
                        source,
                        segment_timestamp=segment_timestamp,
                        request=request,
                    )
                except Exception:  # pragma: no cover - defensive fallback boundary
                    logger.debug("TranscriptionThread: remote fallback after local failure failed.", exc_info=True)
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
        request: Optional[_TranscriptionRequest] = None,
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
                        raw_stt_text=text,
                        normalized_text=text,
                        segment_id=request.request_id if request else "",
                        chunk_ids=list(request.chunk_ids) if request else [],
                        timestamp_start=request.timestamp_start if request else float(segment_timestamp or time.time()),
                        timestamp_end=request.timestamp_end if request else float(segment_timestamp or time.time()),
                        status="finalized",
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
        terms: list[str] = []
        profile = self._ai_config.get("resume_profile")
        if isinstance(profile, dict):
            normalization = profile.get("normalization", {})
            if isinstance(normalization, dict):
                terms.extend(str(item) for item in normalization.get("canonical_terms", []) or [])
                terms.extend(str(item) for item in normalization.get("role_keywords", []) or [])
        terms.extend(str(item) for item in self._ai_config.get("recent_technical_entities", []) or [])
        terms.extend(str(item) for item in self._ai_config.get("current_topic_terms", []) or [])
        now = time.time()
        self._prune_source_state(
            self._last_local_text_by_source,
            self._last_local_text_ts_by_source,
            now,
        )
        prompt_parts = [item.strip() for item in terms if item.strip()]
        previous = self._last_local_text_by_source.get(source, "")
        if previous:
            prompt_parts.append(previous)
        if not prompt_parts:
            return None
        # Only bounded, relevant vocabulary and one recent phrase enter Whisper;
        # the complete resume/session history never becomes an initial prompt.
        return " ".join(dict.fromkeys(prompt_parts))[-LOCAL_PROMPT_MAX_CHARS:]

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

    def _prepare_local_audio(self, audio: np.ndarray, source: str = "speaker") -> Optional[np.ndarray]:
        mono = np.asarray(audio, dtype=np.int16)
        if mono.size == 0:
            return None

        source_name = str(source or "").strip().lower()
        min_rms = self._min_segment_rms_mic if source_name == "user" else self._min_segment_rms

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
        if rms < min_rms:
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
