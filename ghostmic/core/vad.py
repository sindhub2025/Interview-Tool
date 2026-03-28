"""
Voice Activity Detection using Silero VAD.

State machine: SILENCE → SPEAKING → SILENCE
When a complete speech segment is detected the audio is emitted for
transcription, including local pre-roll padding from the VAD timeline.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np
import torch  # type: ignore[import]

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    QThread = object  # type: ignore[misc,assignment]
    pyqtSignal = None  # type: ignore[assignment]

from ghostmic.core.audio_buffer import AudioBuffer, Source
from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

# Silero VAD parameters
VAD_SAMPLE_RATE: int = 16_000
VAD_WINDOW_SIZE: int = 512          # samples per window (~32 ms)
VAD_SPEECH_THRESHOLD: float = 0.5
VAD_SILENCE_THRESHOLD: float = 0.3
SPEECH_START_WINDOWS: int = 3       # consecutive speech windows to start
SPEECH_START_MISS_TOLERANCE: int = 1  # tolerated low windows before candidate reset
SPEECH_END_WINDOWS: int = 8         # consecutive silence windows to end (~256 ms)
MIN_SPEECH_SAMPLES: int = 8_000     # 0.5 s minimum
MAX_SPEECH_SAMPLES: int = 480_000   # 30 s maximum
PADDING_SAMPLES: int = 4_800        # 300 ms padding
BYPASS_SEGMENT_SECONDS: float = 2.0
BYPASS_MAX_BUFFER_SECONDS: float = 8.0
BYPASS_MIN_RMS: float = 120.0


class VADState:
    SILENCE = "silence"
    SPEAKING = "speaking"


class VADThread(QThread):  # type: ignore[misc]
    """Runs Silero VAD and emits complete speech segments.

    Consumes chunks from an internal queue fed by the capture threads.
    Emits ``speech_segment_ready(np.ndarray, source)`` when a segment ends.

    Args:
        buffer: Shared AudioBuffer instance (retained for compatibility).
        on_segment: Optional callback ``(audio, source)`` in addition to
            the Qt signal.
        bypass_vad: If True, pass all audio through without VAD filtering.
    """

    if pyqtSignal is not None:
        speech_segment_ready = pyqtSignal(object, str)

    _shared_model = None
    _shared_torch = None
    _shared_get_speech_ts = None
    _shared_model_failed: bool = False
    _shared_model_error: str = ""
    _shared_model_lock = threading.Lock()

    def __init__(
        self,
        buffer: AudioBuffer,
        on_segment: Optional[Callable[[np.ndarray, str], None]] = None,
        bypass_vad: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._buffer = buffer
        self._on_segment = on_segment
        self._bypass_vad = bypass_vad
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[Tuple[np.ndarray, str]]" = queue.Queue(maxsize=200)
        self._model = None
        self._torch = None
        self._get_speech_ts = None

    @classmethod
    def preload_model(cls) -> bool:
        """Preload Silero model once so first recording starts faster."""
        return cls._load_shared_model()

    def stop(self) -> None:
        self._stop_event.set()

    def push_chunk(self, data: np.ndarray, source: Source) -> None:
        """Feed a new audio chunk into the VAD queue (non-blocking)."""
        try:
            self._queue.put_nowait((data, source))
        except queue.Full:
            logger.debug("VAD queue full – dropping chunk from %s.", source)

    def run(self) -> None:
        self._stop_event.clear()
        if not self._bypass_vad:
            self._load_model()

        # Separate state machine per source
        states = {
            "speaker": self._make_state(),
            "user": self._make_state(),
        }
        bypass_states = {
            "speaker": self._make_bypass_state(),
            "user": self._make_bypass_state(),
        }

        while not self._stop_event.is_set():
            try:
                data, source = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._bypass_vad:
                self._process_bypass_chunk(data, source, bypass_states.get(source))
                continue

            if source not in states:
                continue

            self._process_chunk(data, source, states[source])

        logger.info("VADThread: stopped.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _make_state() -> dict:
        return {
            "state": VADState.SILENCE,
            "speech_windows": 0,
            "speech_miss_windows": 0,
            "silence_windows": 0,
            "speech_buffer": [],   # list of np.ndarray windows
            "pre_roll_windows": [],
            "pre_roll_samples": 0,
            "prefix_audio": np.array([], dtype=np.int16),
        }

    @staticmethod
    def _make_bypass_state() -> dict:
        return {
            "buffer": [],
            "samples": 0,
        }

    def _load_model(self) -> None:
        """Load Silero VAD from shared cache, or bypass on failure."""
        loaded = self._load_shared_model()
        if not loaded:
            self._bypass_vad = True
            return

        self._model = self.__class__._shared_model
        self._torch = self.__class__._shared_torch
        self._get_speech_ts = self.__class__._shared_get_speech_ts

    @classmethod
    def _load_shared_model(cls) -> bool:
        """Load Silero model once for all VADThread instances."""
        with cls._shared_model_lock:
            if cls._shared_model is not None and cls._shared_torch is not None:
                return True

            if cls._shared_model_failed:
                if cls._shared_model_error:
                    logger.warning(
                        "VADThread: using bypass mode (previous Silero load failed: %s)",
                        cls._shared_model_error,
                    )
                return False

            try:
                logger.info("VADThread: loading Silero VAD model …")
                model, utils = torch.hub.load(
                    "snakers4/silero-vad",
                    "silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )
                cls._shared_model = model
                cls._shared_torch = torch
                (
                    cls._shared_get_speech_ts,
                    _,
                    _,
                    _,
                    _,
                ) = utils
                logger.info("VADThread: Silero VAD model loaded.")
                return True
            except Exception as exc:  # pylint: disable=broad-except
                cls._shared_model_failed = True
                cls._shared_model_error = str(exc)
                logger.error(
                    "VADThread: failed to load Silero VAD: %s - bypassing VAD.",
                    exc,
                )
                return False

    def _vad_probability(self, window: np.ndarray) -> float:
        """Return the speech probability for a 512-sample window."""
        if self._model is None:
            return 0.0
        try:
            torch = self._torch
            if torch is None:
                return 0.0
            audio_float = torch.from_numpy(
                window.astype(np.float32) / 32768.0
            ).unsqueeze(0)
            prob = self._vad_probability_tensor(audio_float)
            return float(prob)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("VAD probability error: %s", exc)
            return 0.0

    def _vad_probability_tensor(self, window_tensor) -> float:
        """Return speech probability for a pre-normalized tensor window."""
        if self._model is None:
            return 0.0
        torch = self._torch
        if torch is None:
            return 0.0
        with torch.inference_mode():
            return float(self._model(window_tensor, VAD_SAMPLE_RATE).item())

    def _process_chunk(
        self, chunk: np.ndarray, source: str, state: dict
    ) -> None:
        """Run the VAD state machine on *chunk* for *source*."""
        chunk_tensor = None
        torch = self._torch
        if self._model is not None and torch is not None:
            # Normalize once per chunk to reduce per-window conversion overhead.
            chunk_tensor = torch.from_numpy(chunk.astype(np.float32) / 32768.0)

        pos = 0
        while pos + VAD_WINDOW_SIZE <= len(chunk):
            window = chunk[pos : pos + VAD_WINDOW_SIZE]
            if chunk_tensor is not None:
                window_tensor = chunk_tensor[pos : pos + VAD_WINDOW_SIZE].unsqueeze(0)
                prob = self._vad_probability_tensor(window_tensor)
            else:
                prob = self._vad_probability(window)

            if state["state"] == VADState.SILENCE:
                if prob > VAD_SPEECH_THRESHOLD:
                    if state["speech_windows"] == 0:
                        state["prefix_audio"] = self._concat_windows(
                            state["pre_roll_windows"]
                        )
                    state["speech_windows"] += 1
                    state["speech_miss_windows"] = 0
                    state["speech_buffer"].append(window)
                    if state["speech_windows"] >= SPEECH_START_WINDOWS:
                        state["state"] = VADState.SPEAKING
                        state["silence_windows"] = 0
                        logger.debug("VAD[%s]: SPEECH START", source)
                else:
                    self._update_pre_roll(state, window)
                    if state["speech_windows"] > 0:
                        state["speech_miss_windows"] += 1
                        if state["speech_miss_windows"] <= SPEECH_START_MISS_TOLERANCE:
                            # Keep near-threshold onset continuity during brief dips.
                            state["speech_buffer"].append(window)
                        else:
                            self._reset_speech_candidate(state)
                    else:
                        self._reset_speech_candidate(state)

            elif state["state"] == VADState.SPEAKING:
                state["speech_buffer"].append(window)
                total = sum(len(w) for w in state["speech_buffer"])

                if prob < VAD_SILENCE_THRESHOLD:
                    state["silence_windows"] += 1
                    if state["silence_windows"] >= SPEECH_END_WINDOWS:
                        # End of speech segment
                        logger.debug("VAD[%s]: SPEECH END", source)
                        self._flush_segment(state, source)
                else:
                    state["silence_windows"] = 0

                # Force-send if maximum duration reached
                if total >= MAX_SPEECH_SAMPLES:
                    logger.debug("VAD[%s]: MAX DURATION reached", source)
                    self._flush_segment(state, source)

            pos += VAD_WINDOW_SIZE

    def _flush_segment(self, state: dict, source: str) -> None:
        """Concatenate and emit the buffered speech segment."""
        if state["speech_buffer"]:
            audio = np.concatenate(state["speech_buffer"]).astype(np.int16)
            prefix_audio = np.asarray(
                state.get("prefix_audio", np.array([], dtype=np.int16)),
                dtype=np.int16,
            )
            segment = np.concatenate([prefix_audio, audio])
            total = len(segment)
            if total >= MIN_SPEECH_SAMPLES:
                self._emit(segment, source)
            else:
                logger.debug(
                    "VAD[%s]: segment too short (%d samples) – discarded.",
                    source,
                    total,
                )
        # Reset state
        state["state"] = VADState.SILENCE
        state["silence_windows"] = 0
        self._reset_speech_candidate(state)

    @staticmethod
    def _reset_speech_candidate(state: dict) -> None:
        state["speech_windows"] = 0
        state["speech_miss_windows"] = 0
        state["speech_buffer"] = []
        state["prefix_audio"] = np.array([], dtype=np.int16)

    @staticmethod
    def _concat_windows(windows: list[np.ndarray]) -> np.ndarray:
        if not windows:
            return np.array([], dtype=np.int16)
        return np.concatenate(windows).astype(np.int16)

    @staticmethod
    def _update_pre_roll(state: dict, window: np.ndarray) -> None:
        windows: list[np.ndarray] = state["pre_roll_windows"]
        windows.append(window)
        state["pre_roll_samples"] += len(window)
        while state["pre_roll_samples"] > PADDING_SAMPLES and windows:
            removed = windows.pop(0)
            state["pre_roll_samples"] -= len(removed)

    def _emit(self, audio: np.ndarray, source: str) -> None:
        if self._on_segment:
            self._on_segment(audio, source)
        if pyqtSignal is not None:
            self.speech_segment_ready.emit(audio, source)  # type: ignore[attr-defined]

    def _process_bypass_chunk(self, chunk: np.ndarray, source: str, state: Optional[dict]) -> None:
        """Coarse fallback segmentation when Silero VAD is unavailable.

        Prevents per-chunk pass-through from flooding cloud STT endpoints.
        """
        if state is None:
            return

        state["buffer"].append(chunk)
        state["samples"] += len(chunk)

        min_samples = int(BYPASS_SEGMENT_SECONDS * VAD_SAMPLE_RATE)
        max_samples = int(BYPASS_MAX_BUFFER_SECONDS * VAD_SAMPLE_RATE)

        if state["samples"] < min_samples:
            return

        audio = np.concatenate(state["buffer"]).astype(np.int16)
        if len(audio) > max_samples:
            audio = audio[-max_samples:]

        state["buffer"] = []
        state["samples"] = 0

        if self._is_low_energy(audio):
            logger.debug("VAD[%s]: bypass segment dropped due to low energy", source)
            return

        self._emit(audio, source)

    @staticmethod
    def _is_low_energy(audio: np.ndarray) -> bool:
        if audio.size == 0:
            return True
        rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))
        return rms < BYPASS_MIN_RMS
