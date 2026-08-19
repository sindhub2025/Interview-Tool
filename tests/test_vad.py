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


def test_vad_emits_segment_after_silence_boundary(monkeypatch):
    monkeypatch.setattr(vad_module, "SPEECH_START_WINDOWS", 1)

    emitted = []
    thread = vad_module.VADThread(AudioBuffer())
    thread._on_segment = lambda audio, source: emitted.append((audio, source))
    speech_windows = 16
    silence_windows = vad_module.SPEECH_END_WINDOWS
    probabilities = iter([0.95] * speech_windows + [0.1] * silence_windows)
    thread._vad_probability = lambda _window: next(probabilities)  # type: ignore[method-assign]

    thread._process_chunk(
        np.ones(512 * (speech_windows + silence_windows), dtype=np.int16) * 1400,
        "speaker",
        thread._make_state(),
    )

    assert len(emitted) == 1
    assert emitted[0][0].size > 0
    assert emitted[0][1] == "speaker"