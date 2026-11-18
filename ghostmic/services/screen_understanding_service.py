"""ScreenUnderstandingService: analyzes a PreparedCapture using a vision model."""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

import requests

from ghostmic.utils.logger import get_logger
from ghostmic.domain.captured_screen import PreparedCapture
from ghostmic.domain.screen_understanding import (
    ScreenUnderstanding, ScreenObservation, ScreenTable, ScreenCodeBlock,
)
from ghostmic.services.vision_provider_router import VisionProviderConfig
from ghostmic.services.image_preprocessor import encode_image_data_url

logger = get_logger(__name__)

SCREEN_ANALYSIS_TEMPERATURE = 0.1
SCREEN_ANALYSIS_MAX_COMPLETION_TOKENS = 4096

# ---------------------------------------------------------------------------
# Prompt templates — task-specific, versioned
# ---------------------------------------------------------------------------

_ACCURACY_POLICY = """\
ACCURACY POLICY:
- Transcribe visible text exactly as shown. Do not paraphrase or invent values.
- Mark inferences explicitly: use "[inferred]" after any value you deduced rather than directly read.
- Mark uncertain items: use "[uncertain]" after anything you cannot read clearly.
- Do NOT invent rows, column values, or relationships that are not visible.
- Do NOT classify a column as PK/FK merely because its name ends in "id" — only label it if there is a visible key indicator.
- Separate observed facts from inferences in every section.
"""

_GENERAL_UNDERSTANDING_PROMPT = """\
Analyze this screenshot and provide a structured understanding of its content.

{accuracy_policy}

Respond in these sections:

## Application & Task
Identify: application name, page/view type, visible task or objective, main question or problem if any.

## Content Summary
2-4 sentence overview of what is on screen.

## Tables Detected
If tables, grids, or tabular data are visible, extract EACH table:
### Table: <name>
| Column | Type [observed/inferred] | Key hint |
| --- | --- | --- |
Sample rows (transcribe exactly):
| val1 | val2 |

If no tables: write "No tables detected."

## ER Relationships
If multiple tables suggest relationships, describe them:
- TableA.column → TableB.column (one-to-many) [inferred] OR [observed from FK indicator]
If only one table or no tables, skip or note "N/A".

## Code & Errors
Transcribe verbatim: code, SQL queries, error messages, stack traces, log entries, file paths.
If none: "None detected."

## Key Data Points
List important values, settings, URLs, form field values, status indicators visible.

## Uncertainties
List anything you cannot clearly read or confidently identify.
"""

_TASK_PROMPTS = {
    "general": _GENERAL_UNDERSTANDING_PROMPT,
}


class ScreenUnderstandingService:
    """Analyzes a PreparedCapture using the configured vision model.

    Responsibilities:
    - Constructs vision request with appropriate prompt
    - Submits to provider
    - Parses response into ScreenUnderstanding
    - Handles provider errors

    Does NOT manipulate Qt widgets.
    """

    def __init__(
        self,
        provider_config: VisionProviderConfig,
        *,
        timeout: float = 90.0,
    ) -> None:
        self._provider = provider_config
        self._timeout = timeout

    def analyze(
        self,
        prepared: PreparedCapture,
        request_context: Optional[str] = None,
    ) -> ScreenUnderstanding:
        """Analyze a PreparedCapture and return structured ScreenUnderstanding.

        Args:
            prepared: Image data ready for the vision provider.
            request_context: Optional hint about what the user is trying to do.

        Returns:
            ScreenUnderstanding with parsed structured content.
        """
        prompt = _TASK_PROMPTS["general"].format(
            accuracy_policy=_ACCURACY_POLICY
        )
        raw_text = self._call_provider(prepared, prompt)
        understanding = self._parse_response(raw_text, prepared.capture_id)
        return understanding

    def _call_provider(self, prepared: PreparedCapture, prompt: str) -> str:
        """Call the vision provider and return the raw response text."""
        data_url = encode_image_data_url(
            prepared.primary_image_bytes, prepared.primary_mime_type
        )
        payload = {
            "model": self._provider.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": SCREEN_ANALYSIS_TEMPERATURE,
            "max_completion_tokens": SCREEN_ANALYSIS_MAX_COMPLETION_TOKENS,
            "top_p": 1,
            "stream": False,
        }
        base_url = self._provider.endpoint_url or "https://api.openai.com/v1"
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._provider.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Vision provider request failed: {exc}") from exc

        if resp.status_code >= 400:
            raise RuntimeError(
                f"Vision provider HTTP {resp.status_code}: {resp.text.strip()[:400]}"
            )

        return self._extract_text(resp.json())

    def _extract_text(self, response_json: Dict[str, Any]) -> str:
        """Extract text from an OpenAI-compatible chat-completions response."""
        choices = response_json.get("choices") or []
        if not choices:
            raise RuntimeError("Vision provider returned no choices.")
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                    parts.append(str(item.get("text", "")).strip())
            return "\n".join(parts).strip()
        raise RuntimeError("Vision provider response contained no readable text.")

    def _parse_response(self, raw_text: str, capture_id: str) -> ScreenUnderstanding:
        """Parse raw vision model text into a ScreenUnderstanding object."""
        understanding = ScreenUnderstanding(
            capture_id=capture_id,
            summary="",
            raw_text=raw_text,
        )

        # Extract summary from Content Summary section
        summary_match = re.search(
            r"##\s+Content Summary\s*\n(.*?)(?=\n##|\Z)", raw_text, re.DOTALL | re.IGNORECASE
        )
        if summary_match:
            understanding.summary = summary_match.group(1).strip()
        else:
            # Fallback: first paragraph
            first_para = raw_text.split("\n\n")[0].strip() if raw_text else ""
            understanding.summary = first_para[:500]

        # Extract application/task
        app_match = re.search(
            r"##\s+Application[^#\n]*\n(.*?)(?=\n##|\Z)", raw_text, re.DOTALL | re.IGNORECASE
        )
        if app_match:
            app_text = app_match.group(1).strip()
            # Look for "Application:" line
            app_name_match = re.search(r"(?:application|app)[:\s]+(.+)", app_text, re.IGNORECASE)
            if app_name_match:
                understanding.detected_application = app_name_match.group(1).strip()[:100]
            task_match = re.search(r"(?:task|objective|problem)[:\s]+(.+)", app_text, re.IGNORECASE)
            if task_match:
                understanding.detected_task = task_match.group(1).strip()[:200]

        # Extract tables
        table_blocks = re.finditer(
            r"###\s+Table:\s*(.+?)\n(.*?)(?=\n###\s+Table:|\n##|\Z)",
            raw_text, re.DOTALL | re.IGNORECASE
        )
        for match in table_blocks:
            table_name = match.group(1).strip()
            table_text = match.group(2).strip()
            table = self._parse_table_block(table_name, table_text)
            understanding.tables.append(table)

        # Extract code blocks
        code_blocks = re.finditer(
            r"```(\w*)\n(.*?)```",
            raw_text, re.DOTALL
        )
        for match in code_blocks:
            lang = match.group(1).strip() or "text"
            code = match.group(2).strip()
            if code:
                understanding.code_blocks.append(ScreenCodeBlock(language=lang, code=code))

        # Extract uncertainties section
        uncertain_match = re.search(
            r"##\s+Uncertainties\s*\n(.*?)(?=\n##|\Z)", raw_text, re.DOTALL | re.IGNORECASE
        )
        if uncertain_match:
            for line in uncertain_match.group(1).split("\n"):
                line = line.strip().lstrip("- •*").strip()
                if line and line.lower() not in {"none", "n/a", "none detected"}:
                    understanding.uncertainties.append(line)

        # Extract key data points
        kv_match = re.search(
            r"##\s+Key Data[^#\n]*\n(.*?)(?=\n##|\Z)", raw_text, re.DOTALL | re.IGNORECASE
        )
        if kv_match:
            for line in kv_match.group(1).split("\n"):
                line = line.strip().lstrip("- •*").strip()
                if ":" in line:
                    k, _, v = line.partition(":")
                    understanding.key_values.append((k.strip(), v.strip()))
                elif line and line.lower() not in {"none", "n/a"}:
                    understanding.key_values.append(("info", line))

        # Extract ER relationships
        er_match = re.search(
            r"##\s+ER[^#\n]*\n(.*?)(?=\n##|\Z)", raw_text, re.DOTALL | re.IGNORECASE
        )
        if er_match:
            for line in er_match.group(1).split("\n"):
                line = line.strip().lstrip("- •*").strip()
                if line and line.lower() not in {"none", "n/a", "n/a."}:
                    is_inferred = "[inferred]" in line.lower()
                    understanding.inferred_relationships.append(line)
                    if is_inferred:
                        understanding.observations.append(ScreenObservation(
                            category="relationship",
                            content=line,
                            is_inferred=True,
                        ))

        # Build suggested_answer from summary + first detected elements
        summary_parts = [understanding.summary]
        if understanding.tables:
            tbl_names = ", ".join(t.name for t in understanding.tables[:3])
            summary_parts.append(f"Tables detected: {tbl_names}")
        understanding.suggested_answer = "\n".join(p for p in summary_parts if p)

        return understanding

    def _parse_table_block(self, name: str, text: str) -> ScreenTable:
        """Parse a markdown table block into a ScreenTable."""
        table = ScreenTable(name=name)
        rows = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("Sample") or line.startswith("If "):
                continue
            if line.startswith("|"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                rows.append(cells)

        if rows:
            # First row = headers
            header_row = rows[0]
            table.columns = header_row
            # Second row = separator (---)
            # Remaining = data rows
            data_rows = [r for r in rows[2:] if any(c.strip("-") for c in r)]
            table.visible_rows = data_rows

            # Try to infer column types / key hints from second column if present
            if len(rows) > 1 and len(rows[1]) > 0 and not all(c.strip().startswith("-") for c in rows[1]):
                # rows[1] might be type/key info
                type_row = rows[1]
                table.column_types = type_row[:len(header_row)]

        # Look for uncertainty notes
        for line in text.split("\n"):
            if "[uncertain]" in line.lower() or "[inferred]" in line.lower():
                table.uncertainty_notes.append(line.strip())

        return table
