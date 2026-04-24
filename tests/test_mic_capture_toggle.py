"""Unit tests for microphone capture toggle behavior in GhostMicApp."""

from ghostmic.main import GhostMicApp


class _NoopLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
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
        self.mic_enabled_calls = []

    def set_status(self, text, color):
        self.status_calls.append((text, color))

    def set_recording(self, recording):  # noqa: ARG002
        return None

    def set_mic_enabled(self, enabled):
        self.mic_enabled_calls.append(bool(enabled))


class _FakeAIPanel:
    def __init__(self) -> None:
        self.errors = []

    def show_error(self, message):
        self.errors.append(str(message))


class _FakeWindow:
    def __init__(self) -> None:
        self.controls = _FakeControls()
        self.ai_panel = _FakeAIPanel()


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
    app._mic_device_fallback_attempted = False
    app._mic_recovery_pending = False
    app._mic_recovery_generation = 0
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


def test_mic_runtime_failure_retries_default_input_device(monkeypatch):
    app = _make_app(capture_mic=True)
    app._recording_active = True
    app._config["audio"]["input_device"] = 9
    called = []

    monkeypatch.setattr("ghostmic.main._save_config", lambda cfg, path: None)
    app._disable_mic_capture_live = lambda: called.append("disable")

    def _enable_with_current_device() -> bool:
        called.append(("enable", app._config["audio"].get("input_device")))
        return True

    app._enable_mic_capture_live = _enable_with_current_device

    app._on_mic_capture_failed("permission denied")

    assert app._mic_device_fallback_attempted is True
    assert app._config["audio"]["capture_mic"] is True
    assert app._config["audio"]["input_device"] is None
    assert called == ["disable", ("enable", None)]
    assert app._window.controls.mic_enabled_calls == []


def test_transient_mic_runtime_failure_schedules_delayed_retry(monkeypatch):
    app = _make_app(capture_mic=True)
    app._recording_active = True
    called = []

    monkeypatch.setattr("ghostmic.main._save_config", lambda cfg, path: None)
    app._schedule_mic_live_restart = (
        lambda detail, *, delay_ms=700: called.append((detail, delay_ms)) or True
    )
    app._disable_mic_capture_live = lambda: called.append("disable")

    app._on_mic_capture_failed("device busy")

    assert called == [("device busy", 700)]
    assert app._config["audio"]["capture_mic"] is True
    assert app._window.controls.mic_enabled_calls == []


def test_mic_default_device_failure_retries_once_before_switching_off(monkeypatch):
    app = _make_app(capture_mic=True)
    app._recording_active = True
    app._config["audio"]["input_device"] = None
    called = []

    monkeypatch.setattr("ghostmic.main._save_config", lambda cfg, path: None)
    app._disable_mic_capture_live = lambda: called.append("disable")
    app._enable_mic_capture_live = lambda: called.append("enable") or False

    app._on_mic_capture_failed("permission denied")

    assert called == ["disable", "enable", "disable"]
    assert app._config["audio"]["capture_mic"] is False
    assert app._mic_device_fallback_attempted is False
    assert app._window.controls.mic_enabled_calls == [False]


def test_mic_runtime_failure_after_fallback_switches_to_speaker_only(monkeypatch):
    app = _make_app(capture_mic=True)
    app._recording_active = True
    app._config["audio"]["input_device"] = None
    app._mic_device_fallback_attempted = True
    called = []

    monkeypatch.setattr("ghostmic.main._save_config", lambda cfg, path: None)
    app._disable_mic_capture_live = lambda: called.append("disable")

    app._on_mic_capture_failed("permission denied")

    assert called == ["disable"]
    assert app._config["audio"]["capture_mic"] is False
    assert app._mic_device_fallback_attempted is False
    assert app._window.controls.mic_enabled_calls == [False]


def test_startup_mic_state_primes_enabled_microphone(monkeypatch):
    app = _make_app(capture_mic=True)
    called = []

    app._prime_mic_capture_async = lambda: called.append("prime")

    app._sync_startup_mic_state()

    assert called == ["prime"]


def test_startup_mic_state_skips_when_disabled(monkeypatch):
    app = _make_app(capture_mic=False)
    called = []

    app._prime_mic_capture_async = lambda: called.append("prime")

    app._sync_startup_mic_state()

    assert called == []


def test_mic_only_toggle_uses_cloud_fast_start_while_model_loading(monkeypatch):
    app = _make_app(capture_mic=False)
    app._mic_recording_active = False
    app._is_model_loader_running = lambda: True
    app._try_enable_cloud_fast_start = lambda: (True, "groq/whisper-large-v3-turbo")
    app._start_mic_only_capture = lambda: True
    app._reset_auto_question_session_state = lambda *args, **kwargs: None
    app._is_streaming_normalization_enabled = lambda: False
    app._ensure_ai_thread = lambda: None

    monkeypatch.setattr("ghostmic.main._save_config", lambda cfg, path: None)

    app._on_mic_toggled(True)

    assert app._config["audio"]["capture_mic"] is True
    assert app._mic_recording_active is True
    assert any(
        "cloud transcription active" in status
        for status, _color in app._window.controls.status_calls
    )


def test_mic_only_toggle_sets_active_before_capture_start(monkeypatch):
    app = _make_app(capture_mic=False)
    app._mic_recording_active = False
    app._is_model_loader_running = lambda: False
    app._reset_auto_question_session_state = lambda *args, **kwargs: None
    app._is_streaming_normalization_enabled = lambda: False
    app._ensure_ai_thread = lambda: None

    observed_active_state = []

    def _start_capture() -> bool:
        observed_active_state.append(bool(app._mic_recording_active))
        return True

    app._start_mic_only_capture = _start_capture

    monkeypatch.setattr("ghostmic.main._save_config", lambda cfg, path: None)

    app._on_mic_toggled(True)

    assert observed_active_state == [True]
    assert app._mic_recording_active is True


def test_mic_only_toggle_blocks_when_loader_running_and_cloud_unavailable(monkeypatch):
    app = _make_app(capture_mic=False)
    app._mic_recording_active = False
    app._is_model_loader_running = lambda: True
    app._try_enable_cloud_fast_start = lambda: (False, "no configured API key")
    app._reset_auto_question_session_state = lambda *args, **kwargs: None
    app._is_streaming_normalization_enabled = lambda: False

    started = []
    app._start_mic_only_capture = lambda: started.append("start") or True

    monkeypatch.setattr("ghostmic.main._save_config", lambda cfg, path: None)

    app._on_mic_toggled(True)

    assert started == []
    assert app._config["audio"]["capture_mic"] is False
    assert app._mic_recording_active is False
    assert app._window.controls.mic_enabled_calls == [False]
    assert app._window.ai_panel.errors
    assert "no configured API key" in app._window.ai_panel.errors[-1]
