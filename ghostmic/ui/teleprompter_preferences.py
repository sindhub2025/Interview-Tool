"""Persistent Teleprompter reading preferences."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

_FONT_SIZE_MIN = 16
_FONT_SIZE_MAX = 40
_SPEED_MIN = 0.5
_SPEED_MAX = 5.0


@dataclass
class TeleprompterPreferences:
    """User reading preferences shared across all response cards.
    
    These are persisted in the UI section of config.json.
    """
    font_size: int = 24
    scroll_speed: float = 2.0
    line_spacing: float = 1.45
    reading_width: str = "comfortable"   # "comfortable" | "wide" | "full"

    def clamp(self) -> "TeleprompterPreferences":
        """Return a new instance with all values clamped to valid ranges."""
        return TeleprompterPreferences(
            font_size=max(_FONT_SIZE_MIN, min(_FONT_SIZE_MAX, self.font_size)),
            scroll_speed=max(_SPEED_MIN, min(_SPEED_MAX, self.scroll_speed)),
            line_spacing=max(1.0, min(2.5, self.line_spacing)),
            reading_width=self.reading_width if self.reading_width in {"comfortable", "wide", "full"} else "comfortable",
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "TeleprompterPreferences":
        prefs = TeleprompterPreferences(
            font_size=int(data.get("font_size", 24)),
            scroll_speed=float(data.get("scroll_speed", 2.0)),
            line_spacing=float(data.get("line_spacing", 1.45)),
            reading_width=str(data.get("reading_width", "comfortable")),
        )
        return prefs.clamp()

    @staticmethod
    def load_from_config(config: dict) -> "TeleprompterPreferences":
        """Load preferences from application config dict (ui.teleprompter section)."""
        ui_cfg = config.get("ui", {}) if isinstance(config, dict) else {}
        tp_cfg = ui_cfg.get("teleprompter", {}) if isinstance(ui_cfg, dict) else {}
        return TeleprompterPreferences.from_dict(tp_cfg)

    def save_to_config(self, config: dict) -> None:
        """Write preferences into config dict in-place (does not persist to disk)."""
        config.setdefault("ui", {})
        config["ui"]["teleprompter"] = self.to_dict()
