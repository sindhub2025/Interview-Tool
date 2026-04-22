"""Unit tests for periodic session context compaction."""

import sys
import types as pytypes

from ghostmic.services.session_context_compactor import SessionContextCompactor
from ghostmic.services.session_context_store import SessionContextStore


def test_compactor_writes_organized_snapshot(tmp_path, monkeypatch):
    store = SessionContextStore(base_dir=str(tmp_path))
    store.append_event("transcript", "How do I tune join performance?", source="speaker")
    store.append_event("screen_summary", "Execution plan shows hash join spill.")

    compactor = SessionContextCompactor(store, {"context_compaction_enabled": True})

    calls = []

    def fake_request(source_text, _config):
        calls.append(source_text)
        return (
            "Current Focus: SQL join tuning\n"
            "Recent Questions: join performance improvements\n"
            "Open Follow-ups: pick index strategy"
        )

    monkeypatch.setattr(compactor, "_request_compaction", fake_request)

    assert compactor.compact_once() is True
    assert len(calls) == 1

    tail = store.build_context_tail(max_events=20, max_chars=5000)
    assert "Organized Context" in tail
    assert "Current Focus: SQL join tuning" in tail


def test_compactor_skips_when_source_unchanged(tmp_path, monkeypatch):
    store = SessionContextStore(base_dir=str(tmp_path))
    store.append_event("transcript", "What is dense_rank?", source="speaker")

    compactor = SessionContextCompactor(store, {"context_compaction_enabled": True})

    call_count = {"value": 0}

    def fake_request(_source_text, _config):
        call_count["value"] += 1
        return "Current Focus: ranking functions"

    monkeypatch.setattr(compactor, "_request_compaction", fake_request)

    assert compactor.compact_once() is True
    assert compactor.compact_once() is False
    assert call_count["value"] == 1


def test_compactor_reacts_after_new_source_event(tmp_path, monkeypatch):
    store = SessionContextStore(base_dir=str(tmp_path))
    store.append_event("transcript", "Explain partitions in SQL windows.", source="speaker")

    compactor = SessionContextCompactor(store, {"context_compaction_enabled": True})

    call_count = {"value": 0}

    def fake_request(_source_text, _config):
        call_count["value"] += 1
        return f"Current Focus: pass {call_count['value']}"

    monkeypatch.setattr(compactor, "_request_compaction", fake_request)

    assert compactor.compact_once() is True
    store.append_event("screen_summary", "Window frame docs opened in browser.")
    assert compactor.compact_once() is True

    assert call_count["value"] == 2


def test_compactor_incremental_send_includes_only_new_events(tmp_path, monkeypatch):
    store = SessionContextStore(base_dir=str(tmp_path))
    store.append_event("transcript", "Initial interview question", source="speaker")
    store.append_event("ai_response", "Initial answer")

    compactor = SessionContextCompactor(
        store,
        {
            "context_compaction_enabled": True,
            "context_compaction_full_refresh_every": 5,
        },
    )

    calls = []

    def fake_request(source_text, _config):
        calls.append(source_text)
        return "Current Focus: captured"

    monkeypatch.setattr(compactor, "_request_compaction", fake_request)

    assert compactor.compact_once() is True  # full bootstrap

    store.append_event("transcript", "New follow-up question", source="speaker")
    assert compactor.compact_once() is True

    assert len(calls) == 2
    assert "Initial interview question" in calls[0]
    assert "Initial interview question" not in calls[1]
    assert "New follow-up question" in calls[1]


def test_compactor_full_refresh_after_five_incremental_sends(tmp_path, monkeypatch):
    store = SessionContextStore(base_dir=str(tmp_path))
    store.append_event("transcript", "Base question", source="speaker")

    compactor = SessionContextCompactor(
        store,
        {
            "context_compaction_enabled": True,
            "context_compaction_full_refresh_every": 5,
            "context_compaction_source_max_chars": 10000,
        },
    )

    calls = []

    def fake_request(source_text, _config):
        calls.append(source_text)
        return "Current Focus: refreshed"

    monkeypatch.setattr(compactor, "_request_compaction", fake_request)

    assert compactor.compact_once() is True  # initial full

    for i in range(1, 6):
        store.append_event("transcript", f"Delta event {i}", source="speaker")
        assert compactor.compact_once() is True

    store.append_event("transcript", "Delta event 6", source="speaker")
    assert compactor.compact_once() is True

    # Last call should be forced full refresh and include base + latest event.
    assert "Base question" in calls[-1]
    assert "Delta event 6" in calls[-1]


def test_resolve_active_backend_supports_gemini() -> None:
    assert SessionContextCompactor._resolve_active_backend(
        {"main_backend": "gemini"}
    ) == "gemini"


def test_request_compaction_uses_gemini_backend(tmp_path, monkeypatch):
    store = SessionContextStore(base_dir=str(tmp_path))
    compactor = SessionContextCompactor(
        store,
        {
            "main_backend": "gemini",
            "gemini_api_key": "test-gemini-key",
            "gemini_model": "gemini-3-flash-preview",
            "context_compaction_temperature": 0.25,
        },
    )

    calls = {}

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls["model"] = model
            calls["contents"] = contents
            calls["config"] = config
            return pytypes.SimpleNamespace(text="Current Focus: Gemini compaction")

    class FakeClient:
        def __init__(self, *, api_key):
            calls["api_key"] = api_key
            self.models = FakeModels()

    fake_types = pytypes.ModuleType("google.genai.types")
    fake_types.GenerateContentConfig = FakeGenerateContentConfig

    fake_genai = pytypes.ModuleType("google.genai")
    fake_genai.Client = FakeClient
    fake_genai.types = fake_types

    fake_google = pytypes.ModuleType("google")
    fake_google.genai = fake_genai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    result = compactor._request_compaction(
        "[12:00:00] Transcript: Explain ETL retries",
        compactor._get_config_snapshot(),
    )

    assert result == "Current Focus: Gemini compaction"
    assert calls["api_key"] == "test-gemini-key"
    assert calls["model"] == "gemini-3-flash-preview"
    assert calls["config"].kwargs["temperature"] == 0.25
