"""
Centralized configuration service with schema validation.

Replaces raw dict access throughout the application with a single
service that validates values, provides defaults, and persists config
through a single write point.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

# Type alias for a validator predicate
Validator = Callable[[Any], bool]

# Schema entries: (expected_type, default_value, optional_validator)
SchemaEntry = Tuple[Type, Any, Optional[Validator]]

# Full schema — dotted key path → (type, default, validator | None)
CONFIG_SCHEMA: Dict[str, SchemaEntry] = {
    # AI
    "ai.backend":              (str,   "groq",   lambda v: v in ("groq", "openai")),
    "ai.main_backend":         (str,   "groq",   lambda v: v in ("groq", "openai")),
    "ai.groq_api_key":         (str,   "",       None),
    "ai.groq_model":           (str,   "llama-3.3-70b-versatile", None),
    "ai.openai_api_key":       (str,   "",       None),
    "ai.openai_model":         (str,   "gpt-5-mini", None),
    "ai.temperature":          (float, 0.7,      lambda v: 0.0 <= v <= 2.0),
    "ai.trigger_mode":         (str,   "auto",   lambda v: v in ("auto", "manual", "continuous")),
    "ai.auto_speaker_analysis_enabled": (bool, True, None),
    "ai.auto_speaker_silence_seconds": (float, 1.2, lambda v: 0.3 <= v <= 6.0),
    "ai.auto_speaker_min_words": (int, 4, lambda v: 1 <= v <= 24),
    "ai.auto_speaker_min_chars": (int, 18, lambda v: 1 <= v <= 500),
    "ai.auto_speaker_retrigger_cooldown_seconds": (float, 5.0, lambda v: 0.5 <= v <= 30.0),
    "ai.max_tokens":           (int,   2048,     lambda v: 1 <= v <= 32768),
    "ai.resume_context_enabled": (bool, True, None),
    "ai.sql_profile_enabled":  (bool,  False,    None),
    "ai.resume_correction_threshold_high": (float, 0.87, lambda v: 0.5 <= v <= 1.0),
    "ai.resume_correction_threshold_medium": (float, 0.74, lambda v: 0.4 <= v <= 1.0),
    # Audio
    "audio.sample_rate":       (int,   16000,    lambda v: v in (8000, 16000, 44100, 48000)),
    "audio.channels":          (int,   1,        lambda v: v in (1, 2)),
    # Transcription
    "transcription.language":  (str,   "en",     None),
    "transcription.model_size": (str,  "base.en", None),
    "transcription.compute_type": (str, "int8",  lambda v: v in ("int8", "float16", "float32")),
    "transcription.beam_size": (int,   5,         lambda v: 1 <= v <= 10),
    "transcription.use_context_prompt": (bool, False, None),
    "transcription.min_segment_seconds": (float, 0.45, lambda v: 0.2 <= v <= 3.0),
    "transcription.min_segment_rms": (float, 140.0, lambda v: 20.0 <= v <= 5000.0),
    "transcription.trim_silence": (bool, True, None),
    "transcription.silence_trim_threshold": (int, 220, lambda v: 1 <= v <= 8000),
    "transcription.silence_trim_pad_seconds": (float, 0.08, lambda v: 0.0 <= v <= 0.4),
    "transcription.target_rms": (float, 2200.0, lambda v: 200.0 <= v <= 12000.0),
    "transcription.max_gain": (float, 8.0, lambda v: 1.0 <= v <= 20.0),
    "transcription.no_speech_threshold": (float, 0.7, lambda v: 0.0 <= v <= 1.0),
    "transcription.log_prob_threshold": (float, -1.0, lambda v: -5.0 <= v <= 0.0),
    "transcription.compression_ratio_threshold": (float, 2.0, lambda v: 1.0 <= v <= 4.0),
    "transcription.temperature": (float, 0.0, lambda v: 0.0 <= v <= 1.0),
    "transcription.patience": (float, 1.2, lambda v: 0.5 <= v <= 3.0),
    "transcription.repetition_penalty": (float, 1.05, lambda v: 1.0 <= v <= 2.0),
    # UI
    "ui.opacity":              (float, 0.95,     lambda v: 0.3 <= v <= 1.0),
    "ui.font_size":            (int,   11,       lambda v: 8 <= v <= 20),
    "ui.window_width":         (int,   420,      lambda v: 200 <= v <= 3000),
    "ui.window_height":        (int,   650,      lambda v: 200 <= v <= 3000),
    "ui.compact_mode":         (bool,  False,    None),
    "ui.docked":               (bool,  False,    None),
    "ui.dock_height":          (int,   56,       lambda v: 38 <= v <= 120),
    "ui.pre_dock_x":           (int,   0,        None),
    "ui.pre_dock_y":           (int,   0,        None),
    "ui.pre_dock_width":       (int,   420,      lambda v: 200 <= v <= 3000),
    "ui.pre_dock_height":      (int,   650,      lambda v: 200 <= v <= 3000),
    # Dictation
    "dictation.enabled":       (bool,  True,     None),
    "dictation.commit_idle_ms": (int,  1200,     lambda v: 300 <= v <= 10000),
}


class ConfigService:
    """Centralized config management with validation and thread-safe access.

    Args:
        path: Path to the JSON config file.
        defaults: A full default config dict (nested).
    """

    def __init__(self, path: str, defaults: Dict[str, Any]) -> None:
        self._path = path
        self._defaults = copy.deepcopy(defaults)
        self._config: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._dirty = False

    def load(self) -> Dict[str, Any]:
        """Load config from file, merge with defaults, validate, and return a copy."""
        raw = self._read_file()
        merged = self._deep_merge(self._defaults, raw)
        validated = self._validate(merged)
        with self._lock:
            self._config = validated
            self._dirty = False
        return self.snapshot()

    def save(self) -> None:
        """Persist current config to disk (single write point)."""
        with self._lock:
            data = copy.deepcopy(self._config)
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            with self._lock:
                self._dirty = False
            logger.debug("ConfigService: saved config to %s", self._path)
        except OSError as exc:
            logger.error("ConfigService: could not save config: %s", exc)

    def snapshot(self) -> Dict[str, Any]:
        """Return a deep copy of the current config."""
        with self._lock:
            return copy.deepcopy(self._config)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Get a config value by dotted path (e.g. ``'ai.temperature'``)."""
        keys = dotted_key.split(".")
        with self._lock:
            node: Any = self._config
        for k in keys:
            if isinstance(node, dict):
                node = node.get(k, default)
            else:
                return default
        return node

    def get_section(self, section: str) -> Dict[str, Any]:
        """Return a deep copy of a top-level config section."""
        with self._lock:
            return copy.deepcopy(self._config.get(section, {}))

    def update(self, full_config: Dict[str, Any]) -> Dict[str, Any]:
        """Replace the entire config, validate, and mark dirty.

        Returns the validated config snapshot.
        """
        validated = self._validate(self._deep_merge(self._defaults, full_config))
        with self._lock:
            self._config = validated
            self._dirty = True
        return self.snapshot()

    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        """Update a specific section of the config."""
        with self._lock:
            self._config.setdefault(section, {}).update(values)
            self._dirty = True

    @property
    def dirty(self) -> bool:
        """Return True if the config has unsaved changes."""
        with self._lock:
            return self._dirty

    @property
    def path(self) -> str:
        return self._path

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Validate config against the schema. Out-of-range values are reset to defaults."""
        for dotted_key, (expected_type, default_val, validator) in CONFIG_SCHEMA.items():
            keys = dotted_key.split(".")
            # Navigate to parent
            node = cfg
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            leaf = keys[-1]
            val = node.get(leaf, default_val)

            # Type check
            if not isinstance(val, expected_type):
                # Allow int where float is expected
                if expected_type is float and isinstance(val, int):
                    val = float(val)
                else:
                    logger.warning(
                        "Config %s: expected %s, got %s — using default %r",
                        dotted_key, expected_type.__name__, type(val).__name__, default_val,
                    )
                    node[leaf] = default_val
                    continue

            # Value validation
            if validator is not None:
                try:
                    if not validator(val):
                        logger.warning(
                            "Config %s: value %r failed validation — using default %r",
                            dotted_key, val, default_val,
                        )
                        node[leaf] = default_val
                        continue
                except Exception:
                    node[leaf] = default_val
                    continue

            node[leaf] = val
        # Cross-field validation: ensure medium < high for resume correction thresholds.
        try:
            ai_node = cfg.setdefault("ai", {})
            high_key = "resume_correction_threshold_high"
            med_key = "resume_correction_threshold_medium"
            high_val = ai_node.get(high_key, CONFIG_SCHEMA["ai.resume_correction_threshold_high"][1])
            med_val = ai_node.get(med_key, CONFIG_SCHEMA["ai.resume_correction_threshold_medium"][1])
            # Only enforce when both are numeric
            if isinstance(high_val, (int, float)) and isinstance(med_val, (int, float)):
                if med_val >= high_val:
                    logger.warning(
                        "Config ai: %s (%.3f) >= %s (%.3f); resetting both to defaults",
                        f"ai.{med_key}",
                        med_val,
                        f"ai.{high_key}",
                        high_val,
                    )
                    ai_node[high_key] = CONFIG_SCHEMA["ai.resume_correction_threshold_high"][1]
                    ai_node[med_key] = CONFIG_SCHEMA["ai.resume_correction_threshold_medium"][1]
        except Exception:
            # Non-fatal: continue with validated cfg
            logger.exception("ConfigService: error during cross-field validation")

        return cfg

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _read_file(self) -> Dict[str, Any]:
        """Read the JSON config file. Returns empty dict on failure."""
        if not os.path.exists(self._path):
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
            logger.warning("ConfigService: config file is not a JSON object, using defaults")
            return {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("ConfigService: could not read %s: %s — using defaults", self._path, exc)
            return {}

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge *override* into a copy of *base*."""
        result = copy.deepcopy(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ConfigService._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result
