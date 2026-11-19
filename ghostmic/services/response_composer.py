"""ResponseComposer: converts AI output or ScreenUnderstanding into ResponseDocument."""
from __future__ import annotations

import re
from typing import Optional
from ghostmic.utils.logger import get_logger
from ghostmic.domain.response_document import (
    ResponseDocument, ResponseBlock,
    BLOCK_HEADING, BLOCK_PARAGRAPH, BLOCK_BULLET_LIST, BLOCK_NUMBERED_LIST,
    BLOCK_CODE, BLOCK_TABLE, BLOCK_KEY_VALUE, BLOCK_WARNING, BLOCK_UNCERTAINTY,
    BLOCK_RELATIONSHIP,
    RESPONSE_TYPE_SCREEN_ANALYSIS, RESPONSE_TYPE_INTERVIEW_ANSWER,
    RESPONSE_TYPE_FOLLOW_UP, RESPONSE_TYPE_REFINED, RESPONSE_TYPE_GENERAL,
)
from ghostmic.domain.screen_understanding import ScreenUnderstanding

logger = get_logger(__name__)


class ResponseComposer:
    """Converts AI generation results or ScreenUnderstanding into ResponseDocument.
    
    The ResponseDocument is independent of its presentation (Standard vs Teleprompter).
    """

    def from_screen_understanding(
        self,
        understanding: ScreenUnderstanding,
        title: str = "Screen Analysis",
    ) -> ResponseDocument:
        """Compose a ResponseDocument from a ScreenUnderstanding."""
        doc = ResponseDocument(
            title=title,
            response_type=RESPONSE_TYPE_SCREEN_ANALYSIS,
            plain_text=understanding.raw_text,
            metadata={
                "capture_id": understanding.capture_id,
                "detected_application": understanding.detected_application,
                "detected_task": understanding.detected_task,
            },
        )

        # Summary heading
        if understanding.summary:
            doc.blocks.append(ResponseBlock(
                block_type=BLOCK_HEADING,
                content={"level": 2, "text": "Summary"},
            ))
            doc.blocks.append(ResponseBlock(
                block_type=BLOCK_PARAGRAPH,
                content={"text": understanding.summary},
            ))

        # Application / task info
        if understanding.detected_application or understanding.detected_task:
            items = {}
            if understanding.detected_application:
                items["Application"] = understanding.detected_application
            if understanding.detected_task:
                items["Task"] = understanding.detected_task
            doc.blocks.append(ResponseBlock(
                block_type=BLOCK_KEY_VALUE,
                content={"items": items},
            ))

        # Tables
        if understanding.tables:
            doc.blocks.append(ResponseBlock(
                block_type=BLOCK_HEADING,
                content={"level": 2, "text": "Tables Detected"},
            ))
            for tbl in understanding.tables:
                doc.blocks.append(ResponseBlock(
                    block_type=BLOCK_TABLE,
                    content={
                        "name": tbl.name,
                        "columns": tbl.columns,
                        "column_types": tbl.column_types,
                        "column_key_hints": tbl.column_key_hints,
                        "rows": tbl.visible_rows,
                        "uncertainty_notes": tbl.uncertainty_notes,
                    },
                ))
                if tbl.uncertainty_notes:
                    for note in tbl.uncertainty_notes:
                        doc.blocks.append(ResponseBlock(
                            block_type=BLOCK_UNCERTAINTY,
                            content={"text": note},
                        ))

        # Code blocks
        if understanding.code_blocks:
            doc.blocks.append(ResponseBlock(
                block_type=BLOCK_HEADING,
                content={"level": 2, "text": "Code & Errors"},
            ))
            for cb in understanding.code_blocks:
                doc.blocks.append(ResponseBlock(
                    block_type=BLOCK_CODE,
                    content={"language": cb.language, "code": cb.code, "description": cb.description},
                ))

        # Key values
        if understanding.key_values:
            doc.blocks.append(ResponseBlock(
                block_type=BLOCK_HEADING,
                content={"level": 2, "text": "Key Data Points"},
            ))
            doc.blocks.append(ResponseBlock(
                block_type=BLOCK_KEY_VALUE,
                content={"items": {k: v for k, v in understanding.key_values}},
            ))

        # Relationships
        if understanding.inferred_relationships:
            doc.blocks.append(ResponseBlock(
                block_type=BLOCK_HEADING,
                content={"level": 2, "text": "Relationships"},
            ))
            doc.blocks.append(ResponseBlock(
                block_type=BLOCK_RELATIONSHIP,
                content={"items": understanding.inferred_relationships},
            ))

        # Uncertainties
        if understanding.uncertainties:
            doc.blocks.append(ResponseBlock(
                block_type=BLOCK_HEADING,
                content={"level": 2, "text": "Uncertainties"},
            ))
            for u in understanding.uncertainties:
                doc.blocks.append(ResponseBlock(
                    block_type=BLOCK_UNCERTAINTY,
                    content={"text": u},
                ))

        return doc

    def from_plain_text(
        self,
        text: str,
        title: str = "Answer",
        response_type: str = RESPONSE_TYPE_GENERAL,
    ) -> ResponseDocument:
        """Compose a ResponseDocument from a plain-text AI response.
        
        Parses Markdown structure into blocks.
        """
        doc = ResponseDocument(
            title=title,
            response_type=response_type,
            plain_text=text,
        )
        doc.blocks = self._parse_markdown_blocks(text)
        return doc

    def _parse_markdown_blocks(self, text: str) -> list:
        """Parse a Markdown text string into ResponseBlock list."""
        blocks = []
        lines = text.splitlines()
        i = 0
        bullet_items = []
        numbered_items = []
        paragraph_lines = []

        def flush_bullets():
            if bullet_items:
                blocks.append(ResponseBlock(
                    block_type=BLOCK_BULLET_LIST,
                    content={"items": list(bullet_items)},
                ))
                bullet_items.clear()

        def flush_numbered():
            if numbered_items:
                blocks.append(ResponseBlock(
                    block_type=BLOCK_NUMBERED_LIST,
                    content={"items": list(numbered_items)},
                ))
                numbered_items.clear()

        def flush_paragraph():
            if paragraph_lines:
                text_joined = " ".join(paragraph_lines).strip()
                if text_joined:
                    blocks.append(ResponseBlock(
                        block_type=BLOCK_PARAGRAPH,
                        content={"text": text_joined},
                    ))
                paragraph_lines.clear()

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Code fence
            if stripped.startswith("```"):
                flush_bullets(); flush_numbered(); flush_paragraph()
                lang = stripped[3:].strip()
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                code = "\n".join(code_lines)
                blocks.append(ResponseBlock(
                    block_type=BLOCK_CODE,
                    content={"language": lang, "code": code, "description": ""},
                ))
                i += 1
                continue

            # Heading
            heading_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
            if heading_match:
                flush_bullets(); flush_numbered(); flush_paragraph()
                level = len(heading_match.group(1))
                text_h = heading_match.group(2).strip()
                blocks.append(ResponseBlock(
                    block_type=BLOCK_HEADING,
                    content={"level": level, "text": text_h},
                ))
                i += 1
                continue

            # Bullet
            bullet_match = re.match(r"^[-*\u2022]\s+(.+)$", stripped)
            if bullet_match:
                flush_numbered(); flush_paragraph()
                bullet_items.append(bullet_match.group(1).strip())
                i += 1
                continue

            # Numbered list
            num_match = re.match(r"^\d+[.)]\s+(.+)$", stripped)
            if num_match:
                flush_bullets(); flush_paragraph()
                numbered_items.append(num_match.group(1).strip())
                i += 1
                continue

            # Blank line
            if not stripped:
                flush_bullets(); flush_numbered(); flush_paragraph()
                i += 1
                continue

            # Markdown table row
            if stripped.startswith("|"):
                flush_bullets(); flush_numbered(); flush_paragraph()
                # Accumulate table lines
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    table_lines.append(lines[i].strip())
                    i += 1
                # Parse table
                tbl_rows = []
                for tl in table_lines:
                    cells = [c.strip() for c in tl.strip("|").split("|")]
                    tbl_rows.append(cells)
                if len(tbl_rows) >= 2:
                    headers = tbl_rows[0]
                    data = [r for r in tbl_rows[2:] if any(c.strip("-") for c in r)]
                    # Pad / trim data rows to align with headers without shifting values
                    for r in data:
                        while len(r) < len(headers):
                            r.append("")
                        if len(r) > len(headers):
                            r[:] = r[:len(headers)]
                    blocks.append(ResponseBlock(
                        block_type=BLOCK_TABLE,
                        content={"name": "", "columns": headers, "rows": data,
                                 "column_types": [], "column_key_hints": [], "uncertainty_notes": []},
                    ))
                elif len(tbl_rows) == 1:
                    headers = tbl_rows[0]
                    blocks.append(ResponseBlock(
                        block_type=BLOCK_TABLE,
                        content={"name": "", "columns": headers, "rows": [],
                                 "column_types": [], "column_key_hints": [], "uncertainty_notes": []},
                    ))
                continue

            # Paragraph line
            flush_bullets(); flush_numbered()
            paragraph_lines.append(stripped)
            i += 1

        flush_bullets(); flush_numbered(); flush_paragraph()
        return blocks
