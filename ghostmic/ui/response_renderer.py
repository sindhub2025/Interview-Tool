"""ResponseRenderer: converts a ResponseDocument into styled HTML for Qt display."""
from __future__ import annotations

import re
from html import escape
from typing import List, Optional

from ghostmic.domain.response_document import (
    ResponseDocument, ResponseBlock,
    BLOCK_HEADING, BLOCK_PARAGRAPH, BLOCK_BULLET_LIST, BLOCK_NUMBERED_LIST,
    BLOCK_CODE, BLOCK_TABLE, BLOCK_KEY_VALUE, BLOCK_WARNING, BLOCK_UNCERTAINTY,
    BLOCK_RELATIONSHIP,
)
from ghostmic.ui.styles import (
    ACCENT_BLUE, ACCENT_GREEN, BG_CARD, BG_DEEP, BG_MID,
    BORDER, TEXT_PRIMARY, TEXT_SECONDARY,
)

_CODE_BLOCK_BG = "#11161d"
_CODE_BLOCK_SQL_BG = "#10241b"
_CODE_BLOCK_PY_BG = "#101826"
_WARNING_BG = "#2a1800"
_UNCERTAINTY_BG = "#1a1200"


class ResponseRenderer:
    """Renders a ResponseDocument to styled HTML.
    
    Supports two modes:
    - standard: Normal response panel presentation (font_size=11pt)
    - teleprompter: Large-text reading view (configurable font_size)
    
    Both modes consume the same ResponseDocument.
    Clipboard text is derived from the plain_text field (not HTML).
    """

    def __init__(
        self,
        font_size: int = 11,
        line_spacing: float = 1.45,
        reading_width: str = "comfortable",
        teleprompter: bool = False,
    ) -> None:
        self._font_size = font_size
        self._line_spacing = line_spacing
        self._reading_width = reading_width if reading_width in {"comfortable", "wide", "full"} else "comfortable"
        self._teleprompter = teleprompter

    def render(self, document: ResponseDocument) -> str:
        """Render a ResponseDocument to HTML.
        
        If the document has blocks, renders from blocks.
        Otherwise falls back to parsing plain_text as Markdown.
        """
        if document.blocks:
            html_parts = [self._render_block(b) for b in document.blocks]
            body = "".join(html_parts)
        elif document.plain_text:
            # Fallback: parse plain text the legacy way
            body = self._render_plain_text_fallback(document.plain_text)
        else:
            return ""

        fs = self._font_size
        ls = self._line_spacing

        # Respect reading_width in teleprompter mode
        container_style = "padding:2px 4px 4px 4px;"
        if self._teleprompter:
            if self._reading_width == "comfortable":
                container_style = "padding:2px 10px 10px 10px; max-width:640px; margin:0 auto;"
            elif self._reading_width == "wide":
                container_style = "padding:2px 10px 10px 10px; max-width:920px; margin:0 auto;"
            else:  # "full"
                container_style = "padding:2px 8px 8px 8px; width:100%; margin:0;"

        return (
            f"<html><body style='margin:0; padding:0; background:transparent; "
            f"color:{TEXT_PRIMARY}; font-family:\"Segoe UI\"; font-size:{fs}pt; "
            f"line-height:{ls};'>"
            f"<div style='{container_style}'>"
            f"{body}"
            f"</div></body></html>"
        )

    def render_plain_text(self, document: ResponseDocument) -> str:
        """Return clean, readable plain text suitable for clipboard copy."""
        if not document.blocks:
            return document.plain_text or ""

        lines = []
        if document.title and document.title != "Answer":
            lines.append(f"# {document.title}")
            lines.append("")

        for block in document.blocks:
            t = block.block_type
            c = block.content
            if t == BLOCK_HEADING:
                level = c.get("level", 2)
                prefix = "#" * level
                lines.append(f"{prefix} {c.get('text', '')}")
                lines.append("")
            elif t == BLOCK_PARAGRAPH:
                lines.append(c.get("text", ""))
                lines.append("")
            elif t == BLOCK_BULLET_LIST:
                for item in c.get("items", []):
                    lines.append(f"- {item}")
                lines.append("")
            elif t == BLOCK_NUMBERED_LIST:
                for i, item in enumerate(c.get("items", []), 1):
                    lines.append(f"{i}. {item}")
                lines.append("")
            elif t == BLOCK_CODE:
                lang = c.get("language", "")
                code = c.get("code", "")
                lines.append(f"```{lang}\n{code}\n```")
                lines.append("")
            elif t == BLOCK_TABLE:
                name = c.get("name", "")
                if name:
                    lines.append(f"### Table: {name}")
                cols = c.get("columns", [])
                types = c.get("column_types", [])
                keys = c.get("column_key_hints", [])
                if cols and (any(types) or any(keys)):
                    lines.append("| Column | Type | Key |")
                    lines.append("| --- | --- | --- |")
                    for i, col in enumerate(cols):
                        t_val = types[i] if i < len(types) else ""
                        k_val = keys[i] if i < len(keys) else ""
                        lines.append(f"| {col} | {t_val} | {k_val} |")
                    lines.append("")
                rows = c.get("rows", [])
                if cols and rows:
                    lines.append("Visible rows:")
                    lines.append("| " + " | ".join(str(x) for x in cols) + " |")
                    lines.append("| " + " | ".join("---" for _ in cols) + " |")
                    for row in rows:
                        lines.append("| " + " | ".join(str(x) for x in row) + " |")
                    lines.append("")
                for note in c.get("uncertainty_notes", []):
                    lines.append(f"- [Uncertainty] {note}")
                lines.append("")
            elif t == BLOCK_KEY_VALUE:
                for k, v in c.get("items", {}).items():
                    lines.append(f"- {k}: {v}")
                lines.append("")
            elif t == BLOCK_UNCERTAINTY:
                lines.append(f"- [Uncertainty] {c.get('text', '')}")
                lines.append("")
            elif t == BLOCK_RELATIONSHIP:
                for item in c.get("items", []):
                    lines.append(f"- {item}")
                lines.append("")

        result = "\n".join(lines).strip()
        return result or document.plain_text or ""

    def _render_block(self, block: ResponseBlock) -> str:
        t = block.block_type
        c = block.content

        if t == BLOCK_HEADING:
            return self._render_heading(c.get("level", 2), c.get("text", ""))
        if t == BLOCK_PARAGRAPH:
            return self._render_paragraph(c.get("text", ""))
        if t == BLOCK_BULLET_LIST:
            return self._render_bullet_list(c.get("items", []))
        if t == BLOCK_NUMBERED_LIST:
            return self._render_numbered_list(c.get("items", []))
        if t == BLOCK_CODE:
            return self._render_code_block(c.get("language", ""), c.get("code", ""))
        if t == BLOCK_TABLE:
            return self._render_table(c)
        if t == BLOCK_KEY_VALUE:
            return self._render_key_value(c.get("items", {}))
        if t == BLOCK_WARNING:
            return self._render_warning(c.get("text", ""))
        if t == BLOCK_UNCERTAINTY:
            return self._render_uncertainty(c.get("text", ""))
        if t == BLOCK_RELATIONSHIP:
            return self._render_relationship(c.get("items", []))
        return ""

    def _render_heading(self, level: int, text: str) -> str:
        if self._teleprompter:
            scale_offset = {1: 5, 2: 3, 3: 1, 4: 0}.get(level, 0)
            size = max(14, self._font_size + scale_offset)
        else:
            sizes = {1: 14, 2: 13, 3: 12, 4: 11}
            size = sizes.get(level, 11)
        margin_top = "16px" if level <= 2 else "10px"
        return (
            f"<div style='margin:{margin_top} 0 6px 0; color:{ACCENT_BLUE}; "
            f"font-size:{size}pt; font-weight:700;'>{escape(text)}</div>"
        )

    def _render_paragraph(self, text: str) -> str:
        # Inline formatting: **bold**, *italic*, `code`
        html = escape(text)
        html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html)
        html = re.sub(r"\*(.+?)\*", r"<i>\1</i>", html)
        html = re.sub(r"`(.+?)`",
                      f"<code style='font-family:monospace; background:{BG_MID}; "
                      f"padding:0 3px; border-radius:3px;'>\\1</code>", html)
        return (
            f"<p style='margin:0 0 10px 0; line-height:{self._line_spacing}; "
            f"color:{TEXT_PRIMARY};'>{html}</p>"
        )

    def _render_bullet_list(self, items: list) -> str:
        items_html = "".join(
            f"<li style='margin:0 0 6px 0; padding:0;'>{escape(str(item))}</li>"
            for item in items
        )
        return (
            f"<ul style='margin:0 0 12px 20px; padding:0 0 0 16px; line-height:{self._line_spacing};'>"
            f"{items_html}</ul>"
        )

    def _render_numbered_list(self, items: list) -> str:
        items_html = "".join(
            f"<li style='margin:0 0 6px 0; padding:0;'>{escape(str(item))}</li>"
            for item in items
        )
        return (
            f"<ol style='margin:0 0 12px 20px; padding:0 0 0 16px; line-height:{self._line_spacing};'>"
            f"{items_html}</ol>"
        )

    def _render_code_block(self, language: str, code: str) -> str:
        lang_norm = language.strip().lower()
        if lang_norm in {"sql", "postgres", "postgresql", "mysql", "sqlite", "tsql"}:
            label, accent, bg = "SQL", ACCENT_GREEN, _CODE_BLOCK_SQL_BG
        elif lang_norm in {"python", "py"}:
            label, accent, bg = "Python", ACCENT_BLUE, _CODE_BLOCK_PY_BG
        else:
            label = language.strip().upper() or "Code"
            accent, bg = TEXT_SECONDARY, _CODE_BLOCK_BG

        fs_code = max(11, min(22, int(self._font_size * 0.78))) if self._teleprompter else 11

        return (
            f"<div style='margin:12px 0 16px 0;'>"
            f"<div style='margin:0 0 6px 0; color:{accent}; font-size:10pt; "
            f"font-weight:700; letter-spacing:0.1px;'>{escape(label)}</div>"
            f"<pre style='margin:0; padding:12px 14px; line-height:1.45; "
            f"border:1px solid {accent}; border-radius:8px; background-color:{bg}; "
            f"color:{TEXT_PRIMARY}; "
            f"font-family:\"Consolas\",\"Cascadia Mono\",\"Courier New\",monospace; "
            f"font-size:{fs_code}pt; white-space:pre-wrap;'>"
            f"{escape(code)}</pre></div>"
        )

    def _render_table(self, content: dict) -> str:
        name = content.get("name", "")
        columns = content.get("columns", [])
        rows = content.get("rows", [])
        column_types = content.get("column_types", [])
        column_key_hints = content.get("column_key_hints", [])
        uncertainty_notes = content.get("uncertainty_notes", [])

        if not columns:
            return ""

        title_html = (
            f"<div style='margin:4px 0 6px 0; color:{ACCENT_BLUE}; "
            f"font-size:10pt; font-weight:700;'>{escape(name)}</div>"
            if name else ""
        )

        columns_meta = content.get("columns_meta") or []
        has_evidence = any(bool(c.get("evidence")) for c in columns_meta if isinstance(c, dict))

        # -- Schema sub-table (columns / types / key hints / evidence) --
        schema_html = ""
        has_types = any(t for t in column_types if str(t).strip())
        has_key_hints = any(h for h in column_key_hints if str(h).strip())
        if has_types or has_key_hints or has_evidence:
            schema_header = "<th style='padding:3px 7px; text-align:left; color:{c}; font-weight:600; border-bottom:1px solid {b};'>Column</th>".format(
                c=ACCENT_BLUE, b=BORDER
            )
            if has_types:
                schema_header += "<th style='padding:3px 7px; text-align:left; color:{c}; font-weight:600; border-bottom:1px solid {b};'>Type</th>".format(
                    c=ACCENT_BLUE, b=BORDER
                )
            if has_key_hints:
                schema_header += "<th style='padding:3px 7px; text-align:left; color:{c}; font-weight:600; border-bottom:1px solid {b};'>Key</th>".format(
                    c=ACCENT_BLUE, b=BORDER
                )
            if has_evidence:
                schema_header += "<th style='padding:3px 7px; text-align:left; color:{c}; font-weight:600; border-bottom:1px solid {b};'>Evidence</th>".format(
                    c=ACCENT_BLUE, b=BORDER
                )

            schema_body = ""
            for i, col in enumerate(columns):
                col_type = column_types[i] if i < len(column_types) else ""
                col_key = column_key_hints[i] if i < len(column_key_hints) else ""
                row_cells = (
                    f"<td style='padding:3px 7px; border-bottom:1px solid rgba(100,116,139,0.12); "
                    f"color:{TEXT_PRIMARY}; font-weight:600;'>{escape(str(col))}</td>"
                )
                if has_types:
                    # Mark inferred types with a subtle italic style
                    type_style = "font-style:italic; color:#8795a5;" if "[inferred]" in str(col_type).lower() else f"color:{TEXT_SECONDARY};"
                    row_cells += (
                        f"<td style='padding:3px 7px; border-bottom:1px solid rgba(100,116,139,0.12); "
                        f"{type_style}'>{escape(str(col_type).replace('[inferred]', '').replace('[observed]', '').strip())}</td>"
                    )
                if has_key_hints:
                    key_color = "#e3b341" if str(col_key).strip().upper() in {"PK", "FK"} else TEXT_SECONDARY
                    row_cells += (
                        f"<td style='padding:3px 7px; border-bottom:1px solid rgba(100,116,139,0.12); "
                        f"color:{key_color}; font-weight:600;'>{escape(str(col_key))}</td>"
                    )
                if has_evidence:
                    meta = columns_meta[i] if i < len(columns_meta) else {}
                    ev_text = meta.get("evidence", "") if isinstance(meta, dict) else ""
                    row_cells += (
                        f"<td style='padding:3px 7px; border-bottom:1px solid rgba(100,116,139,0.12); "
                        f"color:#8795a5; font-size:{max(8, self._font_size - 3)}pt;'>{escape(str(ev_text))}</td>"
                    )
                schema_body += f"<tr>{row_cells}</tr>"

            schema_html = (
                f"<div style='margin:4px 0 2px 0; color:{TEXT_SECONDARY}; "
                f"font-size:8pt; font-weight:600; letter-spacing:0.05em;'>SCHEMA</div>"
                f"<table style='border-collapse:collapse; width:100%; font-size:{max(8, self._font_size - 2)}pt; margin-bottom:8px;'>"
                f"<thead><tr>{schema_header}</tr></thead>"
                f"<tbody>{schema_body}</tbody>"
                f"</table>"
            )

        # -- Sample data table --
        data_html = ""
        if rows:
            header_cells = "".join(
                f"<th style='padding:4px 8px; text-align:left; border-bottom:1px solid {BORDER}; "
                f"color:{ACCENT_BLUE}; font-weight:600;'>{escape(str(c))}</th>"
                for c in columns
            )
            body_rows = ""
            for row in rows:
                cells = "".join(
                    f"<td style='padding:4px 8px; border-bottom:1px solid rgba(100,116,139,0.15); "
                    f"color:{TEXT_PRIMARY};'>{escape(str(c))}</td>"
                    for c in row
                )
                body_rows += f"<tr>{cells}</tr>"
            data_label = (
                f"<div style='margin:4px 0 2px 0; color:{TEXT_SECONDARY}; "
                f"font-size:8pt; font-weight:600; letter-spacing:0.05em;'>SAMPLE DATA</div>"
                if schema_html else ""
            )
            data_html = (
                f"{data_label}"
                f"<table style='border-collapse:collapse; width:100%; font-size:{self._font_size - 1}pt;'>"
                f"<thead><tr>{header_cells}</tr></thead>"
                f"<tbody>{body_rows}</tbody>"
                f"</table>"
            )
        elif not schema_html and columns:
            header_cells = "".join(
                f"<th style='padding:4px 8px; text-align:left; border-bottom:1px solid {BORDER}; "
                f"color:{ACCENT_BLUE}; font-weight:600;'>{escape(str(c))}</th>"
                for c in columns
            )
            data_html = (
                f"<table style='border-collapse:collapse; width:100%; font-size:{self._font_size - 1}pt;'>"
                f"<thead><tr>{header_cells}</tr></thead>"
                f"</table>"
            )

        # -- Uncertainty notes --
        uncertainty_html = ""
        for note in (uncertainty_notes or []):
            if note:
                uncertainty_html += self._render_uncertainty(str(note))

        return (
            f"<div style='margin:10px 0 16px 0; overflow-x:auto;'>"
            f"{title_html}{schema_html}{data_html}{uncertainty_html}"
            f"</div>"
        )

    def _render_key_value(self, items: dict) -> str:
        if not items:
            return ""
        rows_html = "".join(
            f"<div style='display:flex; margin:0 0 4px 0;'>"
            f"<span style='color:{TEXT_SECONDARY}; font-weight:600; min-width:120px; "
            f"flex-shrink:0;'>{escape(str(k))}:</span>"
            f"<span style='color:{TEXT_PRIMARY}; margin-left:8px;'>{escape(str(v))}</span>"
            f"</div>"
            for k, v in items.items()
        )
        return f"<div style='margin:0 0 12px 0;'>{rows_html}</div>"

    def _render_warning(self, text: str) -> str:
        return (
            f"<div style='margin:8px 0; padding:8px 12px; background:{_WARNING_BG}; "
            f"border-left:3px solid #f85149; border-radius:4px; color:#f85149;'>"
            f"\u26a0 {escape(text)}</div>"
        )

    def _render_uncertainty(self, text: str) -> str:
        return (
            f"<div style='margin:4px 0; padding:6px 10px; background:{_UNCERTAINTY_BG}; "
            f"border-left:3px solid #e3b341; border-radius:4px; color:#e3b341; "
            f"font-size:{self._font_size - 1}pt;'>"
            f"? {escape(text)}</div>"
        )

    def _render_relationship(self, items: list) -> str:
        if not items:
            return ""
        return self._render_bullet_list(items)

    def _render_plain_text_fallback(self, text: str) -> str:
        """Minimal fallback: render plain text as paragraphs with basic block detection."""
        from ghostmic.services.response_composer import ResponseComposer
        composer = ResponseComposer()
        blocks = composer._parse_markdown_blocks(text)
        return "".join(self._render_block(b) for b in blocks)
