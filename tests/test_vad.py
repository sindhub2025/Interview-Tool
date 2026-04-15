"""Regression tests for VAD chunk accumulation behavior."""

import numpy as np

from ghostmic.core.audio_buffer import AudioBuffer
from ghostmic.core import vad as vad_module


def test_vad_accumulates_subwindow_chunks(monkeypatch):
    """Chunks smaller than VAD window should be carried over and processed."""
    monkeypatch.setattr(vad_module, "SPEECH_START_WINDOWS", 1)

    thread = vad_module.VADThread(AudioBuffer())
    state = thread._make_state()
    thread._vad_probability = lambda _window: 0.95  # type: ignore[method-assign]

    # Typical loopback chunk after 48k -> 16k resample for 1024 native frames.
    small_chunk = np.ones(341, dtype=np.int16) * 1400

    thread._process_chunk(small_chunk, "speaker", state)
    assert state["state"] == vad_module.VADState.SILENCE
    assert state["pending_audio"].size == 341

    thread._process_chunk(small_chunk, "speaker", state)
    assert state["state"] == vad_module.VADState.SPEAKING
    # 170 == (341 + 341) - VAD_WINDOW_SIZE; two 341-sample chunks minus one processed window
    assert state["pending_audio"].size == 170
