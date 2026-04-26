"""Pure helpers for Streamlit response trace visibility density."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

TraceSectionName = Literal[
    "memory",
    "retrieval",
    "context",
    "prompt_input",
    "rendered_prompt",
    "generation_input",
]


class TraceVisibilityLevel(StrEnum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    FULL = "full"


def default_trace_visibility() -> TraceVisibilityLevel:
    """Return the default Streamlit trace visibility level."""

    return TraceVisibilityLevel.STANDARD


def resolve_session_trace_visibility(
    value: object | None,
) -> TraceVisibilityLevel:
    """Resolve a stored trace visibility selection with safe fallback behavior."""

    if isinstance(value, TraceVisibilityLevel):
        return value
    if value is None:
        return default_trace_visibility()
    try:
        return TraceVisibilityLevel(str(value).strip())
    except ValueError:
        return default_trace_visibility()


def trace_visibility_summary(
    level: TraceVisibilityLevel,
) -> str:
    """Return a short sidebar summary for the active trace density setting."""

    if level is TraceVisibilityLevel.MINIMAL:
        return "Retrieval only"
    if level is TraceVisibilityLevel.FULL:
        return "Full internal chain"
    return "Context chain"


def trace_sections_for_level(
    level: TraceVisibilityLevel,
) -> tuple[TraceSectionName, ...]:
    """Return the ordered visible trace sections for one density level."""

    if level is TraceVisibilityLevel.MINIMAL:
        return ("retrieval",)
    if level is TraceVisibilityLevel.FULL:
        return (
            "memory",
            "retrieval",
            "context",
            "prompt_input",
            "rendered_prompt",
            "generation_input",
        )
    return ("memory", "retrieval", "context")
