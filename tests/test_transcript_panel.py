"""Unit tests for transcript panel question sync behavior."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ghostmic.domain import TranscriptSegment
from ghostmic.ui.transcript_panel import TranscriptPanel


def _qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_add_segment_emits_latest_question_changed_for_question_sources() -> None:
    app = _qt_app()
    assert app is not None

    panel = TranscriptPanel()
    seen: list[str] = []
    panel.latest_question_changed.connect(seen.append)

    panel.add_segment(
        TranscriptSegment(
            text="Can you explain your ETL validation strategy?",
            source="speaker",
        )
    )

    assert seen == ["Can you explain your ETL validation strategy?"]


def test_add_segment_does_not_emit_latest_question_for_non_question_sources() -> None:
    app = _qt_app()
    assert app is not None

    panel = TranscriptPanel()
    seen: list[str] = []
    panel.latest_question_changed.connect(seen.append)

    panel.add_segment(
        TranscriptSegment(
            text="Background processing update",
            source="system",
        )
    )

    assert seen == []
