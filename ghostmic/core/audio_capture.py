"""
Dual audio stream capture: WASAPI loopback (system audio) and microphone.

Each capture runs in a dedicated QThread and emits audio_chunk_ready
signals that feed into the AudioBuffer / VAD pipeline.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:  # allow headless import for unit tests
    QThread = object  # type: ignore[misc,assignment]
    pyqtSignal = None  # type: ignore[assignment]

from ghostmic.core.audio_buffer import AudioBuffer, Source
from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

SAMPLE_RATE: int = 16_000
CHANNELS: int = 1
DTYPE: str = "int16"
CHUNK_FRAMES: int = 1024  # ~64 ms at 16 kHz


# ---------------------------------------------------------------------------
# WASAPI loopback capture thread
# ---------------------------------------------------------------------------


class SystemAudioCaptureThread(QThread):  # type: ignore[misc]
    """Captures system audio via WASAPI loopback using pyaudiowpatch.

    Emits ``audio_chunk_ready(np.ndarray, "speaker")`` for each chunk.

    Args:
        buffer: Shared AudioBuffer instance.
        device_index: pyaudio device index for the loopback device.
            If None the default WASAPI loopback device is used.
        on_chunk: Optional callback ``(data, source)`` in addition to
            the Qt signal (useful when running without an event loop).
    """

    if pyqtSignal is not None:
        audio_chunk_ready = pyqtSignal(object, str)

    def __init__(
        self,
        buffer: AudioBuffer,
        device_index: Optional[int] = None,
        on_chunk: Optional[Callable[[np.ndarray, str], None]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._buffer = buffer
        self._device_index = device_index
        self._on_chunk = on_chunk
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Signal the thread to stop capturing."""
        self._stop_event.set()

    def run(self) -> None:  # noqa: C901
        """Main capture loop – executed in the worker thread."""
        self._stop_event.clear()
        try:
            import pyaudiowpatch as pyaudio  # type: ignore[import]
        except ImportError:
            logger.error(
                "pyaudiowpatch not installed – system audio capture unavailable."
            )
            return

        p = pyaudio.PyAudio()
        stream = None
        try:
            device_index = self._device_index
            if device_index is None:
                device_index = self._find_loopback_device(p)

            if device_index is None:
                logger.error("No WASAPI loopback device found.")
                return

            info = p.get_device_info_by_index(device_index)
            native_rate = int(info.get("defaultSampleRate", SAMPLE_RATE))
            max_input_channels = int(info.get("maxInputChannels", CHANNELS) or CHANNELS)
            capture_channels = 2 if max_input_channels >= 2 else 1
            logger.info(
                "SystemAudioCapture: opening loopback device %d (%s) "
                "at %d Hz, channels=%d",
                device_index,
                info.get("name"),
                native_rate,
                capture_channels,
            )

            stream = p.open(
                format=pyaudio.paInt16,
                channels=capture_channels,
                rate=native_rate,
                frames_per_buffer=CHUNK_FRAMES,
                input=True,
                input_device_index=device_index,
            )

            self._capture_loop(stream, native_rate, capture_channels)

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("SystemAudioCapture error: %s", exc, exc_info=True)
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:  # pylint: disable=broad-except
                    pass
            p.terminate()
            logger.info("SystemAudioCapture: stopped.")

    def _find_loopback_device(self, p) -> Optional[int]:
        """Return the index of the default WASAPI loopback device."""
        # pyaudiowpatch may expose a direct helper for this.
        if hasattr(p, "get_default_wasapi_loopback"):
            try:
                info = p.get_default_wasapi_loopback()
                if isinstance(info, dict):
                    idx = info.get("index")
                    if idx is not None:
                        return int(idx)
            except Exception:  # pylint: disable=broad-except
                pass

        try:
            wasapi_info = p.get_host_api_info_by_type(p.paWASAPI)  # type: ignore[attr-defined]
        except Exception:  # pylint: disable=broad-except
            logger.warning("WASAPI not available on this system.")
            return None

        default_speakers = wasapi_info.get("defaultOutputDevice")
        if default_speakers is None:
            return None

        default_output_name = ""
        try:
            output_info = p.get_device_info_by_index(int(default_speakers))
            default_output_name = str(output_info.get("name", "")).strip().lower()
        except Exception:  # pylint: disable=broad-except
            default_output_name = ""

        # First try strict index match for wrappers that keep paired indices.
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get("isLoopbackDevice") and info.get("index") == default_speakers:
                return i

        # Then try name matching against the default output device.
        if default_output_name:
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if not info.get("isLoopbackDevice"):
                    continue
                loopback_name = str(info.get("name", "")).strip().lower()
                if default_output_name in loopback_name:
                    return i

        # Fallback: first loopback device
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get("isLoopbackDevice"):
                return i

        return None

    def _capture_loop(self, stream, native_rate: int, channels: int) -> None:
        """Read from *stream* until stop is requested."""
        need_resample = native_rate != SAMPLE_RATE

        while not self._stop_event.is_set():
            try:
                raw = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
                data = np.frombuffer(raw, dtype=np.int16)

                if channels > 1 and len(data) >= channels:
                    # Loopback is commonly stereo. Mix to mono for the shared pipeline.
                    frames = len(data) // channels
                    data = data[: frames * channels].reshape(frames, channels)
                    data = np.mean(data, axis=1).astype(np.int16)

                if need_resample:
                    data = self._resample(data, native_rate, SAMPLE_RATE)

                self._buffer.push_chunk(data, "speaker")

                if self._on_chunk:
                    self._on_chunk(data, "speaker")

                if pyqtSignal is not None:
                    self.audio_chunk_ready.emit(data, "speaker")  # type: ignore[attr-defined]

            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("SystemAudioCapture read error: %s", exc)
                time.sleep(0.1)

    @staticmethod
    def _resample(data: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Simple linear-interpolation resampler."""
        if from_rate == to_rate:
            return data
        try:
            import resampy  # type: ignore[import]
            return resampy.resample(
                data.astype(np.float32), from_rate, to_rate
            ).astype(np.int16)
        except ImportError:
            pass
        # Fallback: numpy interpolation
        n_out = int(len(data) * to_rate / from_rate)
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, n_out)
        return np.interp(x_new, x_old, data.astype(np.float32)).astype(np.int16)


# ---------------------------------------------------------------------------
# Microphone capture thread
# ---------------------------------------------------------------------------


class MicCaptureThread(QThread):  # type: ignore[misc]
    """Captures microphone audio using sounddevice.

    Emits ``audio_chunk_ready(np.ndarray, "user")`` for each chunk.

    Args:
        buffer: Shared AudioBuffer instance.
        device_index: sounddevice device index.  None = system default.
        on_chunk: Optional callback.
    """

    if pyqtSignal is not None:
        audio_chunk_ready = pyqtSignal(object, str)

    def __init__(
        self,
        buffer: AudioBuffer,
        device_index: Optional[int] = None,
        on_chunk: Optional[Callable[[np.ndarray, str], None]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._buffer = buffer
        self._device_index = device_index
        self._on_chunk = on_chunk
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Signal the thread to stop capturing."""
        self._stop_event.set()

    def run(self) -> None:
        """Main capture loop."""
        self._stop_event.clear()
        try:
            import sounddevice as sd  # type: ignore[import]
        except ImportError:
            logger.error("sounddevice not installed – mic capture unavailable.")
            return

        logger.info(
            "MicCapture: opening device %s at %d Hz",
            self._device_index,
            SAMPLE_RATE,
        )

        def _callback(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                logger.debug("MicCapture status: %s", status)
            chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
            chunk = (chunk * 32767).astype(np.int16)
            self._buffer.push_chunk(chunk, "user")
            if self._on_chunk:
                self._on_chunk(chunk, "user")
            if pyqtSignal is not None:
                self.audio_chunk_ready.emit(chunk, "user")  # type: ignore[attr-defined]

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                device=self._device_index,
                blocksize=CHUNK_FRAMES,
                callback=_callback,
            ):
                while not self._stop_event.is_set():
                    time.sleep(0.05)

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("MicCapture error: %s", exc, exc_info=True)
        finally:
            logger.info("MicCapture: stopped.")


# ---------------------------------------------------------------------------
# Device enumeration helpers
# ---------------------------------------------------------------------------


def list_input_devices() -> list[dict]:
    """Return all available input (microphone) devices."""
    try:
        import sounddevice as sd  # type: ignore[import]
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
    except ImportError:
        return []


def prime_input_device(
    device_index: Optional[int] = None,
    sample_rate: int = SAMPLE_RATE,
    warmup_reads: int = 20,
) -> tuple[bool, str]:
    """Open the microphone briefly to initialize driver/permission state.

    Returns:
        (success, detail)
    """
    try:
        import sounddevice as sd  # type: ignore[import]
    except ImportError:
        return False, "sounddevice not installed"

    reads = max(1, int(warmup_reads))
    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=CHANNELS,
            dtype="float32",
            device=device_index,
            blocksize=CHUNK_FRAMES,
        ) as stream:
            for _ in range(reads):
                stream.read(CHUNK_FRAMES)
        logger.info("Mic prime succeeded on device %s.", device_index)
        return True, "ok"
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Mic prime failed on device %s: %s", device_index, exc)
        return False, str(exc)


def list_loopback_devices() -> list[dict]:
    """Return all WASAPI loopback devices."""
    try:
        import pyaudiowpatch as pyaudio  # type: ignore[import]
        p = pyaudio.PyAudio()
        devices = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get("isLoopbackDevice"):
                devices.append({"index": i, "name": info.get("name", "")})
        p.terminate()
        return devices
    except ImportError:
        return []
