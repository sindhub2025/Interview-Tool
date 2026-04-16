"""Unit tests for AI response panel reuse behavior."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ghostmic.ui.ai_response_panel import AIResponsePanel


def _qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_start_response_reuses_existing_card() -> None:
    app = _qt_app()
    assert app is not None
    panel = AIResponsePanel()

    panel.start_response()
    first_card = panel._active_card
    first_layout_count = panel._responses_layout.count()

    panel.start_response()

    assert panel._active_card is first_card
    assert panel._responses_layout.count() == first_layout_count
    assert len(panel._cards) == 1
