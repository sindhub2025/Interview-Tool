"""Regression tests for transcription startup and local Whisper loading."""

from __future__ import annotations

import ghostmic.core.transcription_engine as transcription_engine_module
from ghostmic.main import GhostMicApp


class _NoopLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


class _Signal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs) -> None:
        for callback in list(self._callbacks):
            callback(*args, **kwargs)


class _FakeCoordinator:
    def __init__(self) -> None:
        self.registered = []

    def register(self, name, thread) -> None:
        self.registered.append((name, thread))


class _FakeTranscriptionThread:
    def __init__(self, language="en", beam_size=3, ai_config=None, remote_config=None, parent=None):
        self.language = language
        self.beam_size = beam_size
        self.ai_config = dict(ai_config or {})
        self.remote_config = dict(remote_config or {})
        self.transcription_ready = _Signal()
        self.transcribing = _Signal()
        self._running = False
        self._model = None

    def start(self) -> None:
        self._running = True

    def isRunning(self) -> bool:
        return self._running

    def set_model(self, model) -> None:
        self._model = model

    def is_ready(self) -> bool:
        return bool(self._running and self._model is not None)


class _FakeModelLoader:
    def __init__(self, model_size="base.en", compute_type="int8", device="auto", parent=None):
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self.model_ready = _Signal()
        self.model_error = _Signal()
        self.progress = _Signal()
        self.model = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self.progress.emit(f"Loading Whisper model '{self.model_size}' …")
        self.model = {
            "model_size": self.model_size,
            "compute_type": self.compute_type,
            "device": self.device,
        }
        self.model_ready.emit()

    def isRunning(self) -> bool:
        return self._running


def test_startup_loads_local_whisper_model_without_remote_key(monkeypatch):
    monkeypatch.setattr(
        transcription_engine_module,
        "TranscriptionThread",
        _FakeTranscriptionThread,
    )
    monkeypatch.setattr(
        transcription_engine_module,
        "ModelLoader",
        _FakeModelLoader,
    )

    app = GhostMicApp.__new__(GhostMicApp)
    app._config = {
        "ai": {
            "groq_api_key": "",
            "openai_api_key": "",
            "main_backend": "groq",
            "backend": "groq",
        },
        "transcription": {
            "model_size": "base.en",
            "compute_type": "int8",
            "language": "en",
            "beam_size": 5,
            "remote_fallback": True,
            "remote_backend": "auto",
        },
    }
    app._logger = _NoopLogger()
    app._thread_coordinator = _FakeCoordinator()
    app._window = None
    app._model_loader = None
    app._model_loader_thread = None
    app._transcription_thread = None
    app._model_ready = False
    app._model_error_message = "stale error"

    app._start_model_loader()

    assert app._model_error_message is None
    assert app._model_ready is True
    assert app._model_loader == {
        "model_size": "base.en",
        "compute_type": "int8",
        "device": "auto",
    }
    assert app._transcription_thread is not None
    assert app._transcription_thread.is_ready() is True
    assert app._transcription_thread._model == app._model_loader
    assert [name for name, _thread in app._thread_coordinator.registered] == [
        "transcription"
    ]