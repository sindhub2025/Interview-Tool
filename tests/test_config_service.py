"""Unit tests for ghostmic.services.config_service — ConfigService."""

import json
import os
import uuid

import pytest

from ghostmic.services.config_service import ConfigService


@pytest.fixture
def tmp_config():
    """Return a path to a temporary config file."""
    path = os.path.abspath(f"config-test-{uuid.uuid4().hex}.json")
    try:
        yield path
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


@pytest.fixture
def defaults():
    """Minimal default config dict."""
    return {
        "ai": {
            "backend": "groq",
            "main_backend": "groq",
            "groq_api_key": "",
            "temperature": 0.7,
            "trigger_mode": "auto",
            "resume_context_enabled": True,
            "sql_profile_enabled": False,
            "interview_profile_enabled": False,
            "active_interview_profile_id": "sql",
            "interview_profiles": [],
        },
        "audio": {
            "sample_rate": 16000,
        },
        "ui": {
            "opacity": 0.95,
            "font_size": 11,
            "window_width": 420,
            "window_height": 650,
            "stealth_enabled": True,
            "compact_mode": False,
            "docked": False,
            "dock_height": 56,
            "pre_dock_x": 0,
            "pre_dock_y": 0,
            "pre_dock_width": 420,
            "pre_dock_height": 650,
        },
        "transcription": {
            "language": "en",
            "model_size": "base.en",
            "compute_type": "int8",
        },
        "dictation": {
            "enabled": True,
            "commit_idle_ms": 1200,
        },
    }


class TestConfigServiceLoadSave:
    def test_loads_defaults_when_no_file(self, tmp_config, defaults):
        svc = ConfigService(tmp_config, defaults)
        cfg = svc.load()
        assert cfg["ai"]["backend"] == "groq"
        assert cfg["ui"]["opacity"] == 0.95

    def test_loads_and_merges_user_overrides(self, tmp_config, defaults):
        with open(tmp_config, "w") as f:
            json.dump({"ai": {"temperature": 0.3}}, f)
        svc = ConfigService(tmp_config, defaults)
        cfg = svc.load()
        assert cfg["ai"]["temperature"] == 0.3
        assert cfg["ai"]["backend"] == "groq"  # default preserved

    def test_save_writes_file(self, tmp_config, defaults):
        svc = ConfigService(tmp_config, defaults)
        svc.load()
        svc.save()
        assert os.path.exists(tmp_config)
        with open(tmp_config) as f:
            data = json.load(f)
        assert data["ai"]["backend"] == "groq"

    def test_save_persists_api_keys(self, tmp_config, defaults):
        svc = ConfigService(tmp_config, defaults)
        svc.load()
        svc.update_section(
            "ai",
            {
                "groq_api_key": "live-groq-key",
                "gemini_api_key": "live-gemini-key",
                "openai_api_key": "live-openai-key",
            },
        )

        svc.save()

        with open(tmp_config) as f:
            data = json.load(f)

        assert data["ai"]["groq_api_key"] == "live-groq-key"
        assert data["ai"]["gemini_api_key"] == "live-gemini-key"
        assert data["ai"]["openai_api_key"] == "live-openai-key"
        assert svc.get("ai.groq_api_key") == "live-groq-key"

    def test_roundtrip(self, tmp_config, defaults):
        svc = ConfigService(tmp_config, defaults)
        svc.load()
        svc.update_section("ai", {"temperature": 0.9})
        svc.save()

        svc2 = ConfigService(tmp_config, defaults)
        cfg = svc2.load()
        assert cfg["ai"]["temperature"] == 0.9


class TestConfigServiceValidation:
    def test_invalid_type_resets_to_default(self, tmp_config, defaults):
        with open(tmp_config, "w") as f:
            json.dump({"ai": {"temperature": "not_a_float"}}, f)
        svc = ConfigService(tmp_config, defaults)
        cfg = svc.load()
        assert cfg["ai"]["temperature"] == 0.7  # default

    def test_out_of_range_resets_to_default(self, tmp_config, defaults):
        with open(tmp_config, "w") as f:
            json.dump({"ui": {"opacity": 5.0}}, f)
        svc = ConfigService(tmp_config, defaults)
        cfg = svc.load()
        assert cfg["ui"]["opacity"] == 0.95  # default

    def test_invalid_stealth_toggle_type_resets_to_default(self, tmp_config, defaults):
        with open(tmp_config, "w") as f:
            json.dump({"ui": {"stealth_enabled": "yes"}}, f)
        svc = ConfigService(tmp_config, defaults)
        cfg = svc.load()
        assert cfg["ui"]["stealth_enabled"] is True

    def test_invalid_enum_resets_to_default(self, tmp_config, defaults):
        with open(tmp_config, "w") as f:
            json.dump({"ai": {"trigger_mode": "invalid_mode"}}, f)
        svc = ConfigService(tmp_config, defaults)
        cfg = svc.load()
        assert cfg["ai"]["trigger_mode"] == "auto"

    def test_backend_accepts_gemini_value(self, tmp_config, defaults):
        with open(tmp_config, "w") as f:
            json.dump({"ai": {"backend": "gemini", "main_backend": "gemini"}}, f)
        svc = ConfigService(tmp_config, defaults)
        cfg = svc.load()
        assert cfg["ai"]["backend"] == "gemini"
        assert cfg["ai"]["main_backend"] == "gemini"

    def test_invalid_dock_height_resets_to_default(self, tmp_config, defaults):
        with open(tmp_config, "w") as f:
            json.dump({"ui": {"dock_height": 5}}, f)
        svc = ConfigService(tmp_config, defaults)
        cfg = svc.load()
        assert cfg["ui"]["dock_height"] == 56

    def test_invalid_resume_context_toggle_resets_to_default(self, tmp_config, defaults):
        with open(tmp_config, "w") as f:
            json.dump({"ai": {"resume_context_enabled": "yes"}}, f)
        svc = ConfigService(tmp_config, defaults)
        cfg = svc.load()
        assert cfg["ai"]["resume_context_enabled"] is True

    def test_invalid_sql_profile_toggle_resets_to_default(self, tmp_config, defaults):
        with open(tmp_config, "w") as f:
            json.dump({"ai": {"sql_profile_enabled": "yes"}}, f)
        svc = ConfigService(tmp_config, defaults)
        cfg = svc.load()
        assert cfg["ai"]["sql_profile_enabled"] is False

    def test_corrupt_json_uses_defaults(self, tmp_config, defaults):
        with open(tmp_config, "w") as f:
            f.write("{invalid json")
        svc = ConfigService(tmp_config, defaults)
        cfg = svc.load()
        assert cfg["ai"]["backend"] == "groq"


class TestConfigServiceAccess:
    def test_dotted_get(self, tmp_config, defaults):
        svc = ConfigService(tmp_config, defaults)
        svc.load()
        assert svc.get("ai.temperature") == 0.7
        assert svc.get("ui.font_size") == 11
        assert svc.get("nonexistent.key", "fallback") == "fallback"

    def test_snapshot_is_deep_copy(self, tmp_config, defaults):
        svc = ConfigService(tmp_config, defaults)
        svc.load()
        snap = svc.snapshot()
        snap["ai"]["temperature"] = 999
        assert svc.get("ai.temperature") == 0.7

    def test_update_replaces_and_validates(self, tmp_config, defaults):
        svc = ConfigService(tmp_config, defaults)
        svc.load()
        new_cfg = svc.update({"ai": {"temperature": 1.5, "trigger_mode": "manual"}})
        assert new_cfg["ai"]["temperature"] == 1.5
        assert new_cfg["ai"]["trigger_mode"] == "manual"
        assert svc.dirty is True

    def test_get_section_returns_copy(self, tmp_config, defaults):
        svc = ConfigService(tmp_config, defaults)
        svc.load()
        section = svc.get_section("ai")
        section["temperature"] = 999
        assert svc.get("ai.temperature") == 0.7
