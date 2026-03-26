"""Unit tests for ghostmic.services.config_service — ConfigService."""

import json
import os
import tempfile

import pytest

from ghostmic.services.config_service import ConfigService


@pytest.fixture
def tmp_config(tmp_path):
    """Return a path to a temporary config file."""
    return str(tmp_path / "config.json")


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
        },
        "audio": {
            "sample_rate": 16000,
        },
        "ui": {
            "opacity": 0.95,
            "font_size": 11,
            "window_width": 420,
            "window_height": 650,
            "compact_mode": False,
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

    def test_invalid_enum_resets_to_default(self, tmp_config, defaults):
        with open(tmp_config, "w") as f:
            json.dump({"ai": {"trigger_mode": "invalid_mode"}}, f)
        svc = ConfigService(tmp_config, defaults)
        cfg = svc.load()
        assert cfg["ai"]["trigger_mode"] == "auto"

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
