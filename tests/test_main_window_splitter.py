from __future__ import annotations

import copy
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMainWindow

from ghostmic.ui.main_window import MainWindow


def _qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _NoResizeRatioMainWindow(MainWindow):
    def resizeEvent(self, event) -> None:
        QMainWindow.resizeEvent(self, event)


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "ghostmic" / "config.json"
    return copy.deepcopy(json.loads(config_path.read_text(encoding="utf-8")))


def test_splitter_proportions_are_applied_after_show() -> None:
    app = _qt_app()
    assert app is not None

    window = _NoResizeRatioMainWindow(_load_config())
    window.show()

    for _ in range(10):
        app.processEvents()

    transcript_height, ai_height = window._splitter.sizes()

    assert ai_height > transcript_height * 2



def test_splitter_proportions_are_reapplied_on_config_update() -> None:
    app = _qt_app()
    assert app is not None

    window = _NoResizeRatioMainWindow(_load_config())
    window.show()

    for _ in range(10):
        app.processEvents()

    window._splitter.setSizes([400, 400])

    updated_config = _load_config()
    updated_config["ui"]["window_width"] = 1100
    updated_config["ui"]["window_height"] = 900

    window.update_config(updated_config)

    for _ in range(10):
        app.processEvents()

    transcript_height, ai_height = window._splitter.sizes()

    assert ai_height > transcript_height * 2