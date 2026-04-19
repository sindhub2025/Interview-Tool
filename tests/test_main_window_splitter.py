from __future__ import annotations

import copy
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtTest import QTest
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


def test_speaker_question_layout_preserves_majority_height_for_ai_panel() -> None:
    app = _qt_app()
    assert app is not None

    window = MainWindow(_load_config())
    window.show()
    for _ in range(10):
        app.processEvents()

    window._set_question_text(
        "Can you explain your ETL validation strategy in production and how you "
        "handle late-arriving source records?"
    )
    window._reveal_question_answer_area()
    QTest.qWait(500)
    for _ in range(10):
        app.processEvents()

    question_height, ai_height = window._splitter.sizes()
    total = max(1, question_height + ai_height)

    assert ai_height > question_height
    assert ai_height / total >= 0.70
    assert question_height <= 220


def test_follow_up_suggestions_are_limited_and_clickable() -> None:
    app = _qt_app()
    assert app is not None

    window = MainWindow(_load_config())
    window.show()
    for _ in range(10):
        app.processEvents()

    window._set_question_text(
        "How do you validate data quality when ETL source schema changes in production?"
    )
    window._reveal_question_answer_area()
    window.set_question_follow_up_suggestions(
        [
            "Can you share a real production incident where this happened",
            "How would you prevent the same issue from recurring?",
            "What monitoring and alerts would you add?",
            "What monitoring and alerts would you add?",
            "How do you align this with release timelines?",
        ]
    )
    for _ in range(10):
        app.processEvents()

    emitted: list[str] = []
    window.suggested_follow_up_selected.connect(emitted.append)

    visible_buttons = [btn for btn in window._follow_up_buttons if btn.isVisible()]
    assert len(visible_buttons) == 3

    second_text = visible_buttons[1].text()
    assert second_text.endswith("?")

    visible_buttons[1].click()
    for _ in range(5):
        app.processEvents()

    assert emitted == [second_text]
    assert window._follow_up_status_label is not None
    assert window._follow_up_status_label.isVisible() is True
    assert "Sent to AI" in window._follow_up_status_label.text()


def test_queued_normalized_questions_show_send_actions_and_respect_lock() -> None:
    app = _qt_app()
    assert app is not None

    window = MainWindow(_load_config())
    window.show()
    for _ in range(10):
        app.processEvents()

    window.set_current_question_text("What is your ETL validation strategy?")
    window.set_question_lock_enabled(True)
    window.set_current_question_text("This should not override")

    assert window._question_text.toPlainText().strip() == "What is your ETL validation strategy?"

    window.set_queued_normalized_questions(
        [
            "How do you handle schema drift in production pipelines",
            "How do you tune retry backoff across dependent services?",
            "How do you validate data completeness after late arrivals?",
            "How do you validate data completeness after late arrivals?",
            "How do you monitor SLA breaches?",
        ]
    )
    window._reveal_question_answer_area()
    QTest.qWait(300)
    for _ in range(10):
        app.processEvents()

    assert window._queued_questions_container is not None
    assert window._queued_questions_container.isVisible() is True
    assert len(window._queued_normalized_questions) == 4

    emitted: list[str] = []
    window.queued_question_send_requested.connect(emitted.append)

    visible_buttons = [
        btn
        for btn in window._queued_question_send_buttons
        if btn.parentWidget() is not None and btn.parentWidget().isVisible()
    ]
    assert len(visible_buttons) == 4

    visible_buttons[1].click()
    for _ in range(5):
        app.processEvents()

    assert emitted == [window._queued_normalized_questions[1]]


def test_pending_normalized_segments_render_as_questions() -> None:
    app = _qt_app()
    assert app is not None

    window = MainWindow(_load_config())
    window.show()
    for _ in range(10):
        app.processEvents()

    window.set_normalized_segments(
        [
            {
                "segment_id": "segment-1",
                "normalized_text": "How do you validate data quality in production",
                "status": "pending",
            }
        ]
    )
    for _ in range(5):
        app.processEvents()

    assert window._queued_question_labels[0].text() == (
        "How do you validate data quality in production?"
    )
    assert window._queued_normalized_questions[0] == (
        "How do you validate data quality in production?"
    )
