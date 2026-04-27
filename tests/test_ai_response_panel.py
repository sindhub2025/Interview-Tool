"""Unit tests for AI response panel reuse behavior."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from ghostmic.ui.ai_response_panel import AIResponsePanel, render_response_html


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


def test_finish_response_scrolls_to_top() -> None:
    app = _qt_app()
    assert app is not None
    panel = AIResponsePanel()
    panel.resize(420, 240)
    panel.show()

    for _ in range(10):
        app.processEvents()

    panel.start_response()
    long_text = "\n\n".join(
        f"Paragraph {index}: " + ("lorem ipsum dolor sit amet " * 12)
        for index in range(12)
    )
    panel.finish_response(long_text)

    QTest.qWait(50)
    for _ in range(10):
        app.processEvents()

    card = panel._cards[-1]
    scroll_bar = card._text_edit.verticalScrollBar()

    assert scroll_bar.maximum() > 0
    assert scroll_bar.value() == scroll_bar.minimum()


def test_show_thinking_keeps_answer_card_full_height() -> None:
    app = _qt_app()
    assert app is not None
    panel = AIResponsePanel()
    panel.resize(520, 360)
    panel.show()

    for _ in range(10):
        app.processEvents()

    panel.start_response()
    long_text = "\n\n".join(
        f"Paragraph {index}: " + ("lorem ipsum dolor sit amet " * 10)
        for index in range(10)
    )
    panel.finish_response(long_text)

    QTest.qWait(50)
    for _ in range(10):
        app.processEvents()

    card = panel._cards[-1]
    height_before = card.height()

    panel.show_thinking("Finalizing response…")

    QTest.qWait(50)
    for _ in range(10):
        app.processEvents()

    height_after = card.height()

    assert height_after >= height_before - 20


def test_render_response_html_highlights_sql_and_python_code_blocks() -> None:
    html = render_response_html(
        "Direct answer.\n\n"
        "- First point\n"
        "- Second point\n"
        "- Third point\n"
        "- Fourth point\n"
        "- Fifth point\n\n"
        "```sql\n"
        "SELECT id, name\n"
        "FROM users;\n"
        "```\n\n"
        "```python\n"
        "print('hello')\n"
        "```"
    )

    lowered = html.lower()

    assert html.count("<pre") == 2
    assert "<ul" in lowered
    assert "sql" in lowered
    assert "python" in lowered
    assert "#10241b" in lowered
    assert "#101826" in lowered
    assert "select id, name" in lowered
    assert html.count("font-size:11pt") >= 3
    assert html.count("font-size:10pt") >= 2
    assert html.count("line-height:1.5") >= 2
    assert "padding:12px 14px" in html
