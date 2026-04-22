"""Per-launch temporary context file for fast interview context recall."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from ghostmic.domain import TranscriptSegment
from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TAIL_EVENTS = 24
DEFAULT_TAIL_CHARS = 2200
DEFAULT_COMPACTION_SOURCE_EVENTS = 80
DEFAULT_COMPACTION_SOURCE_CHARS = 7000
MAX_EVENT_TEXT_CHARS = 8000
MAX_FORMATTED_LINE_CHARS = 320
MAX_ORGANIZED_CONTEXT_CHARS = 2800


class SessionContextStore:
    """Append-only per-session JSONL context file.

    Each app launch gets a unique file path. The file is intended to be
    temporary and removed at shutdown.
    """

    def __init__(self, base_dir: Optional[str] = None) -> None:
        root_dir = base_dir or os.path.join(tempfile.gettempdir(), "ghostmic", "sessions")
        os.makedirs(root_dir, exist_ok=True)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        token = uuid.uuid4().hex[:8]
        filename = f"context_{stamp}_{os.getpid()}_{token}.jsonl"

        self._path = os.path.join(root_dir, filename)
        self._lock = threading.Lock()
        self._closed = False

        self.append_event(
            "session_start",
            "Session context initialized.",
            metadata={"pid": os.getpid()},
        )

    @property
    def path(self) -> str:
        """Return the full path of the temporary context file."""
        return self._path

    def append_transcript(self, segment: TranscriptSegment) -> None:
        """Append a transcript segment event."""
        text = str(getattr(segment, "text", "")).strip()
        if not text:
            return
        source = str(getattr(segment, "source", ""))
        confidence = float(getattr(segment, "confidence", 1.0))
        self.append_event(
            "transcript",
            text,
            source=source,
            metadata={"confidence": round(confidence, 4)},
        )

    def append_typed_prompt(self, prompt: str, refine: bool) -> None:
        """Append a typed prompt/refine prompt event."""
        kind = "typed_refine_prompt" if refine else "typed_prompt"
        self.append_event(kind, prompt)

    def append_ai_response(self, response_text: str) -> None:
        """Append a generated AI response event."""
        self.append_event("ai_response", response_text)

    def append_screen_summary(self, summary_text: str) -> None:
        """Append a screen-analysis summary event."""
        self.append_event("screen_summary", summary_text)

    def append_organized_context(
        self,
        organized_text: str,
        *,
        source_signature: str,
        source_event_count: int,
    ) -> None:
        """Append an AI-cleaned context snapshot to the session file."""
        clipped = str(organized_text).strip()
        if len(clipped) > MAX_ORGANIZED_CONTEXT_CHARS:
            clipped = clipped[:MAX_ORGANIZED_CONTEXT_CHARS] + "..."

        self.append_event(
            "organized_context",
            clipped,
            metadata={
                "source_signature": str(source_signature),
                "source_event_count": int(max(0, source_event_count)),
            },
            compact_whitespace=False,
        )

    def append_event(
        self,
        kind: str,
        text: str,
        *,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        compact_whitespace: bool = True,
    ) -> None:
        """Append an arbitrary event as one JSON object per line."""
        raw_text = str(text)
        if compact_whitespace:
            normalized = " ".join(raw_text.split()).strip()
        else:
            normalized_lines: List[str] = []
            for line in raw_text.splitlines():
                cleaned = " ".join(line.split()).strip()
                if cleaned:
                    normalized_lines.append(cleaned)
            normalized = "\n".join(normalized_lines).strip()

        if not normalized:
            return
        if len(normalized) > MAX_EVENT_TEXT_CHARS:
            normalized = normalized[:MAX_EVENT_TEXT_CHARS] + "..."

        entry: Dict[str, Any] = {
            "ts": time.time(),
            "kind": str(kind).strip() or "event",
            "text": normalized,
        }
        if source:
            entry["source"] = source
        if metadata:
            entry["metadata"] = metadata

        with self._lock:
            if self._closed:
                return
            try:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False))
                    fh.write("\n")
            except OSError as exc:
                logger.warning(
                    "SessionContextStore: failed to append event to %s: %s",
                    self._path,
                    exc,
                )

    def build_compaction_source(
        self,
        max_events: int = DEFAULT_COMPACTION_SOURCE_EVENTS,
        max_chars: int = DEFAULT_COMPACTION_SOURCE_CHARS,
        *,
        start_index: int = 0,
        consume_from_start: bool = False,
    ) -> Dict[str, Any]:
        """Return compaction input and cursor metadata.

        Args:
            max_events: Maximum number of source events to include. ``0`` means no
                event-count limit.
            max_chars: Maximum character budget for formatted source text.
            start_index: Index in the source-event stream to start from.
            consume_from_start: When True, consume earliest unsent events first.
                When False, favor the most recent events.
        """
        with self._lock:
            events = self._read_events_unlocked()

        source_events = [
            event
            for event in events
            if str(event.get("kind", "")) not in {"session_start", "organized_context"}
        ]
        total_event_count = len(source_events)

        start_index = int(max(0, start_index))
        if start_index > total_event_count:
            start_index = total_event_count

        scoped_events = source_events[start_index:]
        if max_events > 0:
            if consume_from_start:
                scoped_events = scoped_events[:max_events]
            else:
                scoped_events = scoped_events[-max_events:]

        if not scoped_events:
            return {
                "text": "",
                "signature": "",
                "event_count": 0,
                "total_event_count": total_event_count,
                "start_index": start_index,
                "end_index": start_index,
            }

        formatted_pairs = [
            (event, self._format_event_for_compaction(event))
            for event in scoped_events
        ]
        formatted_pairs = [pair for pair in formatted_pairs if pair[1]]
        if not formatted_pairs:
            end_index = start_index + len(scoped_events)
            return {
                "text": "",
                "signature": "",
                "event_count": 0,
                "total_event_count": total_event_count,
                "start_index": start_index,
                "end_index": end_index,
            }

        selected_pairs = list(formatted_pairs)
        if max_chars > 0:
            total = 0
            selected_pairs = []
            iterable = formatted_pairs if consume_from_start else list(reversed(formatted_pairs))
            for pair in iterable:
                line = pair[1]
                projected = total + len(line) + (1 if selected_pairs else 0)
                if projected > max_chars and selected_pairs:
                    break
                selected_pairs.append(pair)
                total = projected
            if not consume_from_start:
                selected_pairs.reverse()

        selected_events = [pair[0] for pair in selected_pairs]
        selected_lines = [pair[1] for pair in selected_pairs]
        if not selected_events:
            return {
                "text": "",
                "signature": "",
                "event_count": 0,
                "total_event_count": total_event_count,
                "start_index": start_index,
                "end_index": start_index,
            }

        if consume_from_start:
            output_start_index = start_index
            output_end_index = start_index + len(selected_events)
        else:
            output_start_index = start_index + max(0, len(scoped_events) - len(selected_events))
            output_end_index = output_start_index + len(selected_events)

        signature_material = "\n".join(
            f"{event.get('ts', '')}|{event.get('kind', '')}|"
            f"{event.get('source', '')}|{event.get('text', '')}"
            for event in selected_events
        )
        signature = hashlib.sha1(signature_material.encode("utf-8")).hexdigest()
        return {
            "text": "\n".join(selected_lines),
            "signature": signature,
            "event_count": len(selected_events),
            "total_event_count": total_event_count,
            "start_index": output_start_index,
            "end_index": output_end_index,
        }

    def get_latest_organized_context(self, max_chars: int = 1200) -> str:
        """Return latest organized snapshot text for direct AI injection."""
        with self._lock:
            events = self._read_events_unlocked()

        for event in reversed(events):
            if str(event.get("kind", "")) != "organized_context":
                continue
            lines: List[str] = []
            for raw_line in str(event.get("text", "")).splitlines():
                cleaned = " ".join(raw_line.split()).strip()
                if cleaned:
                    lines.append(cleaned)

            text = "\n".join(lines).strip()
            if not text:
                return ""

            if max_chars > 0 and len(text) > max_chars:
                return text[:max_chars] + "..."
            return text

        return ""

    def build_context_tail(
        self,
        max_events: int = DEFAULT_TAIL_EVENTS,
        max_chars: int = DEFAULT_TAIL_CHARS,
    ) -> str:
        """Return a compact latest-first context tail for prompt injection."""
        with self._lock:
            events = self._read_events_unlocked()

        events = [
            event
            for event in events
            if str(event.get("kind", "")) != "session_start"
        ]

        if max_events > 0:
            events = events[-max_events:]

        formatted = [self._format_event(event) for event in events]
        formatted = [line for line in formatted if line]
        if not formatted:
            return ""

        if max_chars <= 0:
            return "\n".join(formatted)

        total = 0
        selected: List[str] = []
        for line in reversed(formatted):
            projected = total + len(line) + (1 if selected else 0)
            if projected > max_chars and selected:
                break
            selected.append(line)
            total = projected

        selected.reverse()
        return "\n".join(selected)

    def cleanup(self, delete_file: bool = True) -> None:
        """Mark the store as closed and optionally delete the backing file."""
        with self._lock:
            self._closed = True
            if not delete_file:
                return
            try:
                if os.path.exists(self._path):
                    os.remove(self._path)
            except OSError as exc:
                logger.warning(
                    "SessionContextStore: failed to delete context file %s: %s",
                    self._path,
                    exc,
                )

    def _read_events_unlocked(self) -> List[Dict[str, Any]]:
        """Read JSONL events from disk. Caller must hold ``self._lock``."""
        if not os.path.exists(self._path):
            return []

        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                raw_lines = fh.readlines()
        except OSError as exc:
            logger.warning(
                "SessionContextStore: failed to read context file %s: %s",
                self._path,
                exc,
            )
            return []

        events: List[Dict[str, Any]] = []
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    @staticmethod
    def _event_label(kind: str, source: str) -> str:
        if kind == "transcript":
            return "Speaker" if source == "speaker" else "You"
        if kind in {"typed_prompt", "typed_refine_prompt"}:
            return "Typed Prompt"
        if kind == "ai_response":
            return "AI Response"
        if kind == "screen_summary":
            return "Screen Summary"
        if kind == "normalized_question":
            return "Normalized Question"
        if kind == "manual_ai_request":
            return "Manual AI Request"
        if kind == "auto_ai_request":
            return "Auto AI Request"
        if kind == "organized_context":
            return "Organized Context"
        return kind.replace("_", " ").title() or "Event"

    @staticmethod
    def _format_event_for_compaction(event: Dict[str, Any]) -> str:
        ts_raw = event.get("ts", 0)
        try:
            ts = time.strftime("%H:%M:%S", time.localtime(float(ts_raw)))
        except (TypeError, ValueError):
            ts = "--:--:--"

        kind = str(event.get("kind", "event"))
        source = str(event.get("source", "")).strip().lower()
        text = " ".join(str(event.get("text", "")).split()).strip()
        if not text:
            return ""

        if len(text) > MAX_FORMATTED_LINE_CHARS:
            text = text[:MAX_FORMATTED_LINE_CHARS] + "..."

        label = SessionContextStore._event_label(kind, source)
        return f"[{ts}] {label}: {text}"

    @staticmethod
    def _format_event(event: Dict[str, Any]) -> str:
        ts_raw = event.get("ts", 0)
        try:
            ts = time.strftime("%H:%M:%S", time.localtime(float(ts_raw)))
        except (TypeError, ValueError):
            ts = "--:--:--"

        kind = str(event.get("kind", "event"))
        source = str(event.get("source", "")).strip().lower()
        if kind == "organized_context":
            lines: List[str] = []
            for raw_line in str(event.get("text", "")).splitlines():
                cleaned = " ".join(raw_line.split()).strip()
                if not cleaned:
                    continue
                if len(cleaned) > MAX_FORMATTED_LINE_CHARS:
                    cleaned = cleaned[:MAX_FORMATTED_LINE_CHARS] + "..."
                lines.append(cleaned)

            if not lines:
                return ""

            bullet_block = "\n".join(f"  - {line}" for line in lines[:10])
            return f"[{ts}] Organized Context:\n{bullet_block}"

        text = " ".join(str(event.get("text", "")).split()).strip()
        if not text:
            return ""

        if len(text) > MAX_FORMATTED_LINE_CHARS:
            text = text[:MAX_FORMATTED_LINE_CHARS] + "..."

        label = SessionContextStore._event_label(kind, source)

        return f"[{ts}] {label}: {text}"
