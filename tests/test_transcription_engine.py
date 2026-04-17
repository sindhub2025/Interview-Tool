"""Unit tests for local transcription preprocessing and anti-artifact filters."""

import numpy as np

from ghostmic.core.transcription_engine import TranscriptionThread


def _thread(remote_cfg=None):
    return TranscriptionThread(
        model=object(),
        language="en",
        beam_size=5,
        ai_config={},
        remote_config=remote_cfg or {},
    )


def test_prepare_local_audio_drops_low_energy_segment():
    thread = _thread()
    low_energy = np.zeros(16_000, dtype=np.int16)
    assert thread._prepare_local_audio(low_energy) is None


def test_prepare_local_audio_trims_and_boosts_valid_segment():
    thread = _thread(
        {
            "min_segment_seconds": 0.3,
            "min_segment_rms": 80.0,
            "trim_silence": True,
            "silence_trim_threshold": 120,
            "target_rms": 1800.0,
        }
    )

    signal = np.sin(np.linspace(0, 40.0, 8_000)) * 350.0
    segment = np.concatenate(
        [
            np.zeros(5_000, dtype=np.int16),
            signal.astype(np.int16),
            np.zeros(5_000, dtype=np.int16),
        ]
    )

    processed = thread._prepare_local_audio(segment)
    assert processed is not None
    assert len(processed) < len(segment)

    original_rms = float(
        np.sqrt(np.mean(np.square(segment.astype(np.float32))))
    )
    processed_rms = float(np.sqrt(np.mean(np.square(processed.astype(np.float32)))))
    assert processed_rms > original_rms


def test_should_drop_local_artifact_for_repeated_short_text():
    thread = _thread()
    first = thread._should_drop_local_artifact("Thanks", 0.95, "speaker")
    second = thread._should_drop_local_artifact("Thanks", 0.95, "speaker")
    assert first is False
    assert second is True


def test_should_drop_low_diversity_low_confidence_text():
    thread = _thread()
    artifact = "uh uh uh uh uh uh uh uh"
    assert thread._should_drop_local_artifact(artifact, 0.2, "speaker") is True


class _FakeWhisperSegment:
    def __init__(self, text: str, avg_logprob: float = -0.2) -> None:
        self.text = text
        self.avg_logprob = avg_logprob


class _FakeWhisperModel:
    def transcribe(self, audio_float, **kwargs):  # noqa: ARG002
        return [_FakeWhisperSegment("Can you explain your ETL test strategy?")], {}


def test_transcribe_uses_segment_enqueue_timestamp():
    thread = _thread()
    thread.set_model(_FakeWhisperModel())
    audio = (np.sin(np.linspace(0, 60.0, 16_000)) * 4200).astype(np.int16)
    queued_at = 1234.5

    result = thread._transcribe(audio, "speaker", segment_timestamp=queued_at)

    assert result is not None
    assert result.timestamp == queued_at
