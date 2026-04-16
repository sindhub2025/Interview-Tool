"""Unit tests for microphone capture toggle behavior in GhostMicApp."""

from ghostmic.main import GhostMicApp


class _NoopLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


class _Signal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)


class _FakeCoordinator:
    def __init__(self) -> None:
        self.registered = []
        self.stopped = []

    def register(self, name, thread) -> None:
        self.registered.append(name)

    def stop_one(self, name, timeout_ms=3000):  # noqa: ARG002
        self.stopped.append(name)
        return True


class _FakeAudioBuffer:
    def __init__(self, sample_rate=16000) -> None:
        self.sample_rate = sample_rate


class _FakeVADThread:
    def __init__(self, buffer) -> None:
        self.buffer = buffer
        self.speech_segment_ready = _Signal()

    def push_chunk(self, data, source) -> None:  # noqa: ARG002
        return None


class _FakeCaptureThread:
    def __init__(self, buffer, device_index=None) -> None:
        self.buffer = buffer
        self.device_index = device_index
        self.audio_chunk_ready = _Signal()
        self._running = False

    def start(self) -> None:
        self._running = True

    def isRunning(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._running = False

    def wait(self, msecs=0):  # noqa: ARG002
        return True


class _FakeControls:
    def __init__(self) -> None:
        self.status_calls = []

    def set_status(self, text, color):
        self.status_calls.append((text, color))

    def set_recording(self, recording):  # noqa: ARG002
        return None


class _FakeWindow:
    def __init__(self) -> None:
        self.controls = _FakeControls()


def _make_app(capture_mic: bool) -> GhostMicApp:
    app = GhostMicApp.__new__(GhostMicApp)
    app._config = {
        "audio": {
            "sample_rate": 16000,
            "loopback_device": 7,
            "input_device": 3,
            "capture_mic": capture_mic,
        }
    }
    app._buffer = None
    app._thread_coordinator = _FakeCoordinator()
    app._logger = _NoopLogger()
    app._on_speech_segment = lambda audio, source: None
    app._vad_thread = None
    app._sys_audio_thread = None
    app._mic_thread = None
    app._window = _FakeWindow()
    app._tray = None
    app._config_path = "ignored.json"
    app._recording_active = False
    app._mic_prime_in_progress = False
    return app


def _patch_audio_dependencies(monkeypatch) -> None:
    monkeypatch.setattr("ghostmic.core.audio_buffer.AudioBuffer", _FakeAudioBuffer)
    monkeypatch.setattr("ghostmic.core.vad.VADThread", _FakeVADThread)
    monkeypatch.setattr(
        "ghostmic.core.audio_capture.SystemAudioCaptureThread",
        _FakeCaptureThread,
    )
    monkeypatch.setattr("ghostmic.core.audio_capture.MicCaptureThread", _FakeCaptureThread)


def test_create_audio_threads_defaults_to_speaker_only(monkeypatch):
    _patch_audio_dependencies(monkeypatch)
    app = _make_app(capture_mic=False)

    assert app._create_audio_threads() is True
    assert app._mic_thread is None
    assert app._thread_coordinator.registered == ["vad", "sys_audio"]


def test_create_audio_threads_includes_mic_when_enabled(monkeypatch):
    _patch_audio_dependencies(monkeypatch)
    app = _make_app(capture_mic=True)

    assert app._create_audio_threads() is True
    assert isinstance(app._mic_thread, _FakeCaptureThread)
    assert app._thread_coordinator.registered == ["vad", "sys_audio", "mic"]


def test_mic_capture_toggle_default_is_false_when_missing():
    app = GhostMicApp.__new__(GhostMicApp)
    app._config = {"audio": {}}

    assert app._is_mic_capture_enabled() is False


def test_idle_toggle_on_primes_microphone(monkeypatch):
    app = _make_app(capture_mic=False)
    called = []

    monkeypatch.setattr("ghostmic.main._save_config", lambda cfg, path: None)
    app._prime_mic_capture_async = lambda: called.append("prime")
    app._start_audio_capture = lambda: called.append("start")
    app._stop_audio_capture = lambda: called.append("stop")

    app._on_mic_toggled(True)

    assert app._config["audio"]["capture_mic"] is True
    assert called == ["prime"]


def test_recording_toggle_on_uses_live_enable_without_restart(monkeypatch):
    app = _make_app(capture_mic=False)
    app._recording_active = True
    called = []

    monkeypatch.setattr("ghostmic.main._save_config", lambda cfg, path: None)
    app._enable_mic_capture_live = lambda: called.append("live_on") or True
    app._stop_audio_capture = lambda: called.append("stop")
    app._start_audio_capture = lambda: called.append("start") or True

    app._on_mic_toggled(True)

    assert app._config["audio"]["capture_mic"] is True
    assert called == ["live_on"]


def test_recording_toggle_off_uses_live_disable_without_restart(monkeypatch):
    app = _make_app(capture_mic=True)
    app._recording_active = True
    called = []

    monkeypatch.setattr("ghostmic.main._save_config", lambda cfg, path: None)
    app._disable_mic_capture_live = lambda: called.append("live_off")
    app._stop_audio_capture = lambda: called.append("stop")
    app._start_audio_capture = lambda: called.append("start") or True

    app._on_mic_toggled(False)

    assert app._config["audio"]["capture_mic"] is False
    assert called == ["live_off"]
