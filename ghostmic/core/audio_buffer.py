"""
Thread-safe circular audio buffer.

Maintains separate buffers for system-audio ("speaker") and
microphone ("user") streams.  Each buffer holds up to *max_seconds*
of audio at the configured sample rate.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Dict, List, Literal

import numpy as np

Source = Literal["speaker", "user"]


class AudioBuffer:
    """Thread-safe circular buffer for raw audio chunks.

    Args:
        sample_rate: Audio sample rate in Hz (default 16 000).
        max_seconds: Maximum seconds of audio to keep per source (default 30).
    """

    def __init__(self, sample_rate: int = 16_000, max_seconds: int = 30) -> None:
        self._sample_rate = sample_rate
        self._max_samples = sample_rate * max_seconds
        self._buffers: Dict[Source, Deque[np.ndarray]] = {
            "speaker": deque(),
            "user": deque(),
        }
        self._locks: Dict[Source, threading.Lock] = {
            "speaker": threading.Lock(),
            "user": threading.Lock(),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_chunk(self, data: np.ndarray, source: Source) -> None:
        """Append *data* to the buffer for *source*.

        Old samples are automatically discarded when the buffer exceeds
        *max_seconds*.

        Args:
            data: 1-D int16 numpy array of audio samples.
            source: "speaker" or "user".
        """
        if source not in self._buffers:
            raise ValueError(f"Unknown source: {source!r}")

        chunk = np.asarray(data, dtype=np.int16)
        with self._locks[source]:
            buf = self._buffers[source]
            buf.append(chunk)
            # Evict oldest chunks until total <= max_samples
            total = sum(len(c) for c in buf)
            while total > self._max_samples and buf:
                evicted = buf.popleft()
                total -= len(evicted)

    def get_last_n_seconds(self, n: float, source: Source) -> np.ndarray:
        """Return the last *n* seconds of audio for *source*.

        Args:
            n: Number of seconds.
            source: "speaker" or "user".

        Returns:
            1-D int16 numpy array.  May be shorter than *n* seconds if
            there is not enough data in the buffer.
        """
        if source not in self._buffers:
            raise ValueError(f"Unknown source: {source!r}")

        want = int(n * self._sample_rate)
        with self._locks[source]:
            all_audio = self._concat(source)

        if len(all_audio) == 0:
            return np.array([], dtype=np.int16)
        return all_audio[-want:]

    def get_all(self, source: Source) -> np.ndarray:
        """Return the entire buffered audio for *source*.

        Args:
            source: "speaker" or "user".

        Returns:
            1-D int16 numpy array.
        """
        if source not in self._buffers:
            raise ValueError(f"Unknown source: {source!r}")

        with self._locks[source]:
            return self._concat(source)

    def clear(self, source: Source) -> None:
        """Discard all buffered audio for *source*.

        Args:
            source: "speaker" or "user".
        """
        if source not in self._buffers:
            raise ValueError(f"Unknown source: {source!r}")

        with self._locks[source]:
            self._buffers[source].clear()

    def total_samples(self, source: Source) -> int:
        """Return the number of samples currently in *source* buffer."""
        with self._locks[source]:
            return sum(len(c) for c in self._buffers[source])

    def total_seconds(self, source: Source) -> float:
        """Return the number of seconds currently in *source* buffer."""
        return self.total_samples(source) / self._sample_rate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _concat(self, source: Source) -> np.ndarray:
        """Concatenate all chunks for *source* (caller holds the lock)."""
        chunks: List[np.ndarray] = list(self._buffers[source])
        if not chunks:
            return np.array([], dtype=np.int16)
        return np.concatenate(chunks).astype(np.int16)
