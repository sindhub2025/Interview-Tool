"""ScreenUnderstandingService: analyzes a PreparedCapture using a vision model."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import requests

from ghostmic.utils.logger import get_logger
from ghostmic.domain.captured_screen import PreparedCapture
from ghostmic.domain.screen_understanding import (
    ScreenUnderstanding, ScreenObservation, ScreenTable, ScreenTableColumn, ScreenCodeBlock,
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

_STRUCTURED_JSON_SCHEMA = """\
```json
{
  "detected_application": "string or null (e.g. DBeaver, VS Code, Chrome)",
  "detected_task": "string or null (e.g. Writing SQL query, Debugging error)",
  "summary": "2-4 sentence overview of what is on screen",
  "tables": [
    {
      "name": "table name or description",
      "columns": [
        {
          "name": "exact column name",
          "type": "VARCHAR / INTEGER / DATE / etc.",
          "type_is_inferred": true,
          "key_hint": "PK / FK or empty string",
          "key_hint_is_inferred": false,
          "evidence": "visible header label or indicator"
        }
      ],
      "visible_rows": [
        ["sample cell value 1", "sample cell value 2"]
      ],
      "uncertainty_notes": []
    }
  ],
  "code_blocks": [
    {
      "language": "sql/python/etc.",
      "code": "verbatim code or query",
      "description": "purpose or context"
    }
  ],
  "errors": ["exact error messages or stack traces"],
  "key_values": [
    {"key": "label", "value": "value"}
  ],
  "relationships": [
    {
      "description": "TableA.column -> TableB.column (1:N)",
      "is_inferred": true
    }
  ],
  "uncertainties": ["anything not clearly legible"]
}
```\
"""

_GENERAL_UNDERSTANDING_PROMPT = """\
Analyze this screenshot thoroughly and extract all meaningful content in structured JSON format.

{accuracy_policy}

Please provide your output as a single valid JSON object inside a ```json ``` block adhering to this schema:
{schema}

CRITICAL RULES FOR EXTRACTION:
- Separate table schema from visible sample data rows. Never combine them.
- In "columns", extract exact column names visible in headers.
- If data types are not explicitly written in the UI, set "type_is_inferred": true.
- Only mark key_hint as "PK" or "FK" if there is explicit visual evidence (key icon, "(PK)", etc.). Do NOT classify a column as a key merely because its name ends in "_id"; if inferring, set "key_hint_is_inferred": true.
- In "visible_rows", include ONLY rows physically visible in the screenshot. Do not invent or extrapolate rows. Preserve column ordering.
- If no tables are visible, set "tables": [].
- Transcribe code, queries, and errors verbatim.
- If you cannot output JSON, you may provide your analysis in standard markdown sections (## Content Summary, ### Table: <name>, ## Code & Errors, etc.).
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
            accuracy_policy=_ACCURACY_POLICY,
            schema=_STRUCTURED_JSON_SCHEMA,
        )
        raw_text = self._call_provider(prepared, prompt)
        understanding = self._parse_response(raw_text, prepared.capture_id)
        understanding.metadata = {
            "capture_metadata": getattr(prepared, "metadata", {}) or {},
            "model": self._provider.model,
            "provider": self._provider.provider,
        }
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
        """Parse raw vision model text into a ScreenUnderstanding object.

        Attempts structured JSON parsing first. If valid JSON cannot be found
        or parsed, gracefully falls back to Markdown section parsing.
        """
        try:
            json_understanding = self._parse_json_response(raw_text, capture_id)
            if json_understanding is not None:
                return json_understanding
        except Exception as exc:
            logger.debug(
                "Structured JSON parsing failed for capture %s: %s; falling back to Markdown",
                capture_id,
                exc,
            )

        return self._parse_markdown_response(raw_text, capture_id)

    def _parse_json_response(self, raw_text: str, capture_id: str) -> Optional[ScreenUnderstanding]:
        """Extract and validate structured JSON from raw vision response text."""
        # Search for ```json ... ``` or outermost { ... }
        json_str = ""
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            first_brace = raw_text.find("{")
            last_brace = raw_text.rfind("}")
            if first_brace != -1 and last_brace > first_brace:
                json_str = raw_text[first_brace : last_brace + 1].strip()

        if not json_str:
            return None

        try:
            data = json.loads(json_str)
        except Exception as exc:
            logger.debug("Failed to decode JSON from screen understanding: %s", exc)
            return None

        if not isinstance(data, dict):
            return None

        summary = str(data.get("summary", "") or "").strip()
        app_name = data.get("detected_application")
        task_name = data.get("detected_task")

        understanding = ScreenUnderstanding(
            capture_id=capture_id,
            summary=summary,
            detected_application=str(app_name).strip() if app_name else None,
            detected_task=str(task_name).strip() if task_name else None,
            raw_text=raw_text,
        )

        # ── Parse and validate tables ──
        raw_tables = data.get("tables")
        if isinstance(raw_tables, list):
            for t in raw_tables:
                if not isinstance(t, dict):
                    continue
                tbl_name = str(t.get("name", "") or "Table").strip()
                table = ScreenTable(name=tbl_name)

                # Schema columns: exact names, types, key hints with observed/inferred tags
                raw_cols = t.get("columns")
                if isinstance(raw_cols, list):
                    for col in raw_cols:
                        if isinstance(col, dict):
                            col_name = str(col.get("name", "") or "").strip()
                            if not col_name:
                                continue
                            raw_type = str(col.get("type", "") or "").strip()
                            type_inferred = bool(col.get("type_is_inferred", False))
                            if type_inferred and raw_type and "[inferred]" not in raw_type.lower():
                                col_type = f"{raw_type} [inferred]"
                            else:
                                col_type = raw_type

                            raw_key = str(col.get("key_hint", "") or "").strip()
                            key_inferred = bool(col.get("key_hint_is_inferred", False))
                            if key_inferred and raw_key and "[inferred]" not in raw_key.lower():
                                key_hint = f"{raw_key} [inferred]"
                            else:
                                key_hint = raw_key

                            col_meta = ScreenTableColumn(
                                name=col_name,
                                type=raw_type,
                                type_is_inferred=type_inferred,
                                key_hint=raw_key,
                                key_hint_is_inferred=key_inferred,
                                evidence=str(col.get("evidence", "") or "").strip(),
                            )
                            table.columns_meta.append(col_meta)
                            table.columns.append(col_name)
                            table.column_types.append(col_type)
                            table.column_key_hints.append(key_hint)
                        elif isinstance(col, str) and col.strip():
                            col_name = col.strip()
                            col_meta = ScreenTableColumn(name=col_name)
                            table.columns_meta.append(col_meta)
                            table.columns.append(col_name)
                            table.column_types.append("")
                            table.column_key_hints.append("")

                num_cols = len(table.columns)

                # Sample data rows: ensure alignment without shifting column values
                raw_rows = t.get("visible_rows")
                if isinstance(raw_rows, list):
                    for row in raw_rows:
                        if isinstance(row, list):
                            cells = [str(c) if c is not None else "" for c in row]
                            if num_cols > 0:
                                while len(cells) < num_cols:
                                    cells.append("")
                                cells = cells[:num_cols]
                            table.visible_rows.append(cells)

                # Uncertainty notes
                notes = t.get("uncertainty_notes")
                if isinstance(notes, list):
                    table.uncertainty_notes = [str(n).strip() for n in notes if str(n).strip()]

                if table.columns or table.visible_rows:
                    understanding.tables.append(table)

        # ── Parse code blocks ──
        raw_code = data.get("code_blocks")
        if isinstance(raw_code, list):
            for cb in raw_code:
                if isinstance(cb, dict):
                    code = str(cb.get("code", "") or "").strip()
                    if code:
                        lang = str(cb.get("language", "") or "text").strip()
                        desc = str(cb.get("description", "") or "").strip()
                        understanding.code_blocks.append(
                            ScreenCodeBlock(language=lang, code=code, description=desc)
                        )

        # ── Parse errors ──
        raw_err = data.get("errors")
        if isinstance(raw_err, list):
            understanding.errors = [str(e).strip() for e in raw_err if str(e).strip()]

        # ── Parse key values ──
        raw_kvs = data.get("key_values")
        if isinstance(raw_kvs, list):
            for kv in raw_kvs:
                if isinstance(kv, dict):
                    k = str(kv.get("key", "") or "").strip()
                    v = str(kv.get("value", "") or "").strip()
                    if k:
                        understanding.key_values.append((k, v))
                elif isinstance(kv, (list, tuple)) and len(kv) >= 2:
                    understanding.key_values.append((str(kv[0]).strip(), str(kv[1]).strip()))

        # ── Parse relationships ──
        raw_rels = data.get("relationships")
        if isinstance(raw_rels, list):
            for rel in raw_rels:
                if isinstance(rel, dict):
                    desc = str(rel.get("description", "") or "").strip()
                    is_inferred = bool(rel.get("is_inferred", False))
                    if desc:
                        rel_str = f"{desc} [inferred]" if (is_inferred and "[inferred]" not in desc.lower()) else desc
                        understanding.inferred_relationships.append(rel_str)
                        if is_inferred:
                            understanding.observations.append(
                                ScreenObservation(category="relationship", content=desc, is_inferred=True)
                            )
                elif isinstance(rel, str) and rel.strip():
                    desc = rel.strip()
                    is_inferred = "[inferred]" in desc.lower()
                    understanding.inferred_relationships.append(desc)
                    if is_inferred:
                        understanding.observations.append(
                            ScreenObservation(category="relationship", content=desc, is_inferred=True)
                        )

        # ── Parse uncertainties ──
        raw_unc = data.get("uncertainties")
        if isinstance(raw_unc, list):
            understanding.uncertainties = [str(u).strip() for u in raw_unc if str(u).strip()]

        # Fallback summary if empty in JSON
        if not understanding.summary:
            first_para = raw_text.split("\n\n")[0].strip() if raw_text else ""
            understanding.summary = first_para[:500]

        summary_parts = [understanding.summary]
        if understanding.tables:
            tbl_names = ", ".join(t.name for t in understanding.tables[:3])
            summary_parts.append(f"Tables detected: {tbl_names}")
        understanding.suggested_answer = "\n".join(p for p in summary_parts if p)

        return understanding

    def _parse_markdown_response(self, raw_text: str, capture_id: str) -> ScreenUnderstanding:
        """Parse raw markdown text into a ScreenUnderstanding object (fallback parser)."""
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

        # If extraction was largely incomplete, record an uncertainty note
        if not understanding.tables and not understanding.code_blocks and not understanding.key_values:
            understanding.uncertainties.append("Structured extraction was incomplete; displaying raw analysis.")

        # Build suggested_answer from summary + first detected elements
        summary_parts = [understanding.summary]
        if understanding.tables:
            tbl_names = ", ".join(t.name for t in understanding.tables[:3])
            summary_parts.append(f"Tables detected: {tbl_names}")
        understanding.suggested_answer = "\n".join(p for p in summary_parts if p)

        return understanding

    def _parse_table_block(self, name: str, text: str) -> ScreenTable:
        """Parse a markdown table block into a ScreenTable.

        The prompt requests tables in this format::

            | Column | Type [observed/inferred] | Key hint |
            | ---    | ---                      | ---      |
            | col_a  | VARCHAR                  | PK       |

            Sample rows (transcribe exactly):
            | val1 | val2 |

        We separate the *schema* table (column-name / type / key-hint header
        section) from the *sample data* rows so they are never accidentally
        merged.
        """
        table = ScreenTable(name=name)

        # Split the block at an explicit "Sample rows" / "Sample data" marker.
        # Everything before that marker is the schema table.
        # Everything after is sample data.
        sample_marker_re = re.compile(
            r"(?im)^(?:sample\s+rows?|sample\s+data|data\s+rows?)[^\n]*:?\s*$"
        )
        schema_text = text
        data_text = ""
        marker_match = sample_marker_re.search(text)
        if marker_match:
            schema_text = text[: marker_match.start()]
            data_text = text[marker_match.end():]

        # -- Parse schema table --
        schema_rows = []
        for line in schema_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("|"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                schema_rows.append(cells)

        if schema_rows:
            # First non-separator row = header (column names)
            header_row = schema_rows[0]
            table.columns = [c for c in header_row]

            # Identify the separator row (all dashes)
            # Rows after separator = schema data rows (type / key-hint per column)
            schema_data_rows = []
            past_separator = False
            for row in schema_rows[1:]:
                if not past_separator:
                    if all(re.match(r"^-+$", c.replace(" ", "")) for c in row if c):
                        past_separator = True
                        continue
                    # No separator found yet — treat row as data row
                    schema_data_rows.append(row)
                else:
                    schema_data_rows.append(row)

            # schema_data_rows contains one row per column in format:
            #   [col_name, type_string, key_hint]  (3 columns from the schema table)
            # OR the prompt may produce a single row per column if the schema
            # table uses the first row as headers and subsequent rows as per-column data.
            if schema_data_rows:
                # Header row already captured column names; check if schema_data_rows
                # represent per-column metadata (3-col format: col | type | key).
                first_data_row = schema_data_rows[0]
                if len(first_data_row) >= 3 and len(schema_data_rows) == len(table.columns):
                    # Each data row corresponds to one column in the schema table
                    types = []
                    key_hints = []
                    table.columns_meta = []
                    for i, row in enumerate(schema_data_rows):
                        col_type = row[1].strip() if len(row) > 1 else ""
                        key_hint = row[2].strip() if len(row) > 2 else ""
                        # Strip [observed]/[inferred] tags from type for cleanliness
                        col_type_clean = re.sub(r"\[(observed|inferred)\]", "", col_type, flags=re.IGNORECASE).strip()
                        types.append(col_type_clean)
                        key_hints.append(key_hint)
                        col_name = table.columns[i] if i < len(table.columns) else ""
                        table.columns_meta.append(
                            ScreenTableColumn(
                                name=col_name,
                                type=col_type_clean,
                                type_is_inferred="[inferred]" in col_type.lower(),
                                key_hint=key_hint,
                                key_hint_is_inferred="[inferred]" in key_hint.lower(),
                            )
                        )
                    table.column_types = types
                    table.column_key_hints = key_hints
                else:
                    if not table.columns_meta and table.columns:
                        table.columns_meta = [ScreenTableColumn(name=c) for c in table.columns]
                    # Data rows are actual sample rows, not per-column metadata.
                    # Visible rows from schema section (no explicit sample marker).
                    if not data_text:
                        for row in schema_data_rows:
                            if any(c.strip("-") for c in row):
                                table.visible_rows.append(row)

        # -- Parse sample data rows --
        if data_text:
            for line in data_text.split("\n"):
                line = line.strip()
                if not line or not line.startswith("|"):
                    continue
                cells = [c.strip() for c in line.strip("|").split("|")]
                # Skip separator lines
                if all(re.match(r"^-+$", c.replace(" ", "")) for c in cells if c):
                    continue
                # Align cells to column count if possible
                if table.columns:
                    ncols = len(table.columns)
                    while len(cells) < ncols:
                        cells.append("")
                    cells = cells[:ncols]
                table.visible_rows.append(cells)

        # -- Collect uncertainty notes --
        for line in text.split("\n"):
            if "[uncertain]" in line.lower() or "[inferred]" in line.lower():
                cleaned = line.strip()
                if cleaned and cleaned not in table.uncertainty_notes:
                    table.uncertainty_notes.append(cleaned)

        return table
