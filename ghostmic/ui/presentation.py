"""Shared presentation state values used by the desktop UI."""

from __future__ import annotations

from enum import Enum


class Workspace(str, Enum):
    """Content presentation workspace."""

    STANDARD = "standard"
    TELEPROMPTER = "teleprompter"


def coerce_workspace(value: Workspace | str) -> Workspace:
    """Normalize a workspace value and reject invalid transitions."""
    if isinstance(value, Workspace):
        return value
    try:
        return Workspace(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(f"Unknown workspace: {value!r}") from exc

