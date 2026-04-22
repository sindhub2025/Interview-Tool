from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ghostmic import main as ghostmic_main


def test_save_config_redacts_api_keys_only_for_bundled_dev_config(tmp_path, monkeypatch):
    target = tmp_path / "config.json"
    monkeypatch.setattr(ghostmic_main, "BUNDLED_CONFIG_PATH", str(target))
    monkeypatch.setattr(ghostmic_main.sys, "frozen", False, raising=False)

    ghostmic_main._save_config(
        {
            "ai": {
                "groq_api_key": "live-groq-key",
                "gemini_api_key": "live-gemini-key",
                "openai_api_key": "live-openai-key",
            }
        },
        str(target),
    )

    data = json.loads(target.read_text(encoding="utf-8"))

    assert data["ai"]["groq_api_key"] == ""
    assert data["ai"]["gemini_api_key"] == ""
    assert data["ai"]["openai_api_key"] == ""


def test_save_config_keeps_api_keys_for_frozen_exe(tmp_path, monkeypatch):
    target = tmp_path / "user-config.json"
    monkeypatch.setattr(ghostmic_main.sys, "frozen", True, raising=False)

    ghostmic_main._save_config(
        {
            "ai": {
                "groq_api_key": "live-groq-key",
                "gemini_api_key": "live-gemini-key",
                "openai_api_key": "live-openai-key",
            }
        },
        str(target),
    )

    data = json.loads(target.read_text(encoding="utf-8"))

    assert data["ai"]["groq_api_key"] == "live-groq-key"
    assert data["ai"]["gemini_api_key"] == "live-gemini-key"
    assert data["ai"]["openai_api_key"] == "live-openai-key"