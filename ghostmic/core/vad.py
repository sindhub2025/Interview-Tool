"""
Voice Activity Detection using Silero VAD.

State machine: SILENCE → SPEAKING → SILENCE
When a complete speech segment is detected the audio is extracted from the
AudioBuffer and emitted for transcription.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np

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
SPEECH_END_WINDOWS: int = 8         # consecutive silence windows to end (~256 ms)
MIN_SPEECH_SAMPLES: int = 8_000     # 0.5 s minimum
MAX_SPEECH_SAMPLES: int = 480_000   # 30 s maximum
PADDING_SAMPLES: int = 4_800        # 300 ms padding


class VADState:
    SILENCE = "silence"
    SPEAKING = "speaking"


class VADThread(QThread):  # type: ignore[misc]
    """Runs Silero VAD and emits complete speech segments.

    Consumes chunks from an internal queue fed by the capture threads.
    Emits ``speech_segment_ready(np.ndarray, source)`` when a segment ends.

    Args:
        buffer: Shared AudioBuffer (used to retrieve padding context).
        on_segment: Optional callback ``(audio, source)`` in addition to
            the Qt signal.
        bypass_vad: If True, pass all audio through without VAD filtering.
    """

    if pyqtSignal is not None:
        speech_segment_ready = pyqtSignal(object, str)

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
        self._get_speech_ts = None

    def stop(self) -> None:
        self._stop_event.set()

    def push_chunk(self, data: np.ndarray, source: Source) -> None:
        """Feed a new audio chunk into the VAD queue (non-blocking)."""
        try:
            self._queue.put_nowait((data, source))
        except queue.Full:
            logger.debug("VAD queue full – dropping chunk from %s.", source)

    def run(self) -> None:
        if not self._bypass_vad:
            self._load_model()

        # Separate state machine per source
        states = {
            "speaker": self._make_state(),
            "user": self._make_state(),
        }

        while not self._stop_event.is_set():
            try:
                data, source = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._bypass_vad:
                self._emit(data, source)
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
            "silence_windows": 0,
            "speech_buffer": [],   # list of np.ndarray windows
        }

    def _load_model(self) -> None:
        """Load Silero VAD from torch.hub."""
        try:
            import torch  # type: ignore[import]
            logger.info("VADThread: loading Silero VAD model …")
            model, utils = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            (
                self._get_speech_ts,
                _,
                _,
                _,
                _,
            ) = utils
            logger.info("VADThread: Silero VAD model loaded.")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "VADThread: failed to load Silero VAD: %s - bypassing VAD.",
                exc,
            )
            self._bypass_vad = True

    def _vad_probability(self, window: np.ndarray) -> float:
        """Return the speech probability for a 512-sample window."""
        if self._model is None:
            return 0.0
        try:
            import torch  # type: ignore[import]
            audio_float = torch.from_numpy(
                window.astype(np.float32) / 32768.0
            ).unsqueeze(0)
            prob = self._model(audio_float, VAD_SAMPLE_RATE).item()
            return float(prob)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("VAD probability error: %s", exc)
            return 0.0

    def _process_chunk(
        self, chunk: np.ndarray, source: str, state: dict
    ) -> None:
        """Run the VAD state machine on *chunk* for *source*."""
        pos = 0
        while pos + VAD_WINDOW_SIZE <= len(chunk):
            window = chunk[pos : pos + VAD_WINDOW_SIZE]
            prob = self._vad_probability(window)

            if state["state"] == VADState.SILENCE:
                if prob > VAD_SPEECH_THRESHOLD:
                    state["speech_windows"] += 1
                    state["speech_buffer"].append(window)
                    if state["speech_windows"] >= SPEECH_START_WINDOWS:
                        state["state"] = VADState.SPEAKING
                        state["silence_windows"] = 0
                        logger.debug("VAD[%s]: SPEECH START", source)
                else:
                    # Reset accumulator
                    state["speech_windows"] = 0
                    state["speech_buffer"] = []

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
            # Add padding from the buffer
            before = self._buffer.get_last_n_seconds(0.3, source)  # type: ignore[arg-type]
            segment = np.concatenate([before, audio])
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
        state["speech_windows"] = 0
        state["silence_windows"] = 0
        state["speech_buffer"] = []

    def _emit(self, audio: np.ndarray, source: str) -> None:
        if self._on_segment:
            self._on_segment(audio, source)
        if pyqtSignal is not None:
            self.speech_segment_ready.emit(audio, source)  # type: ignore[attr-defined]
