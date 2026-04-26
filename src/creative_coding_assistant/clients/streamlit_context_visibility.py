"""Pure helpers for memory and assembled-context visibility in Streamlit."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import CreativeCodingDomain, StreamEvent

VisibilityState = Literal["unknown", "empty", "available"]
VisibilityKind = Literal["memory", "context"]


class ContextDisplayItem(BaseModel):
    """Small context payload shape safe to render in the Streamlit UI."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(min_length=1)
    source_id: str | None = None
    domain: CreativeCodingDomain | None = None
    snippet: str = Field(min_length=1)


def memory_updates_from_event(event: StreamEvent) -> dict[str, object]:
    if event.payload.get("code") != "memory_completed":
        return {}

    raw_context = event.payload.get("context")
    if not isinstance(raw_context, dict):
        return {"memory_items": (), "memory_state": "empty"}

    items = _memory_items(raw_context)
    if not items:
        return {"memory_items": (), "memory_state": "empty"}
    return {"memory_items": items, "memory_state": "available"}


def context_updates_from_event(event: StreamEvent) -> dict[str, object]:
    if event.payload.get("code") != "context_assembled":
        return {}

    raw_context = event.payload.get("context")
    if not isinstance(raw_context, dict):
        return {"context_items": (), "context_state": "empty"}

    items = _assembled_context_items(raw_context)
    if not items:
        return {"context_items": (), "context_state": "empty"}
    return {"context_items": items, "context_state": "available"}


def context_expander_label(
    *,
    kind: VisibilityKind,
    items: Sequence[ContextDisplayItem],
    visibility_state: VisibilityState,
) -> str:
    if visibility_state == "available":
        count = len(items)
        suffix = "s" if count != 1 else ""
        return f"{_title(kind)} ({count} item{suffix})"
    return _title(kind)


def context_empty_message(
    *,
    kind: VisibilityKind,
    visibility_state: VisibilityState,
) -> str | None:
    if visibility_state == "unknown":
        return None
    if visibility_state == "empty":
        if kind == "memory":
            return "No memory context was available for this response."
        return "No assembled context was produced for this response."
    return None


def _memory_items(raw_context: dict[str, object]) -> tuple[ContextDisplayItem, ...]:
    items: list[ContextDisplayItem] = []

    running_summary = raw_context.get("running_summary")
    if isinstance(running_summary, dict):
        content = _clean_text(running_summary.get("content"))
        if content is not None:
            items.append(
                ContextDisplayItem(
                    label="Running summary",
                    snippet=_truncate_preview(content),
                )
            )

    recent_turns = raw_context.get("recent_turns")
    if isinstance(recent_turns, list):
        for raw_turn in recent_turns[-2:]:
            if not isinstance(raw_turn, dict):
                continue
            content = _clean_text(raw_turn.get("content"))
            role = _clean_text(raw_turn.get("role"))
            turn_index = raw_turn.get("turn_index")
            if content is None or role is None:
                continue
            label = f"{role.title()} turn"
            if isinstance(turn_index, int):
                label = f"{label} {turn_index}"
            items.append(
                ContextDisplayItem(
                    label=label,
                    snippet=_truncate_preview(content),
                )
            )

    project_memories = raw_context.get("project_memories")
    if isinstance(project_memories, list):
        for raw_memory in project_memories[:2]:
            if not isinstance(raw_memory, dict):
                continue
            content = _clean_text(raw_memory.get("content"))
            memory_kind = _clean_text(raw_memory.get("memory_kind"))
            source = _clean_text(raw_memory.get("source"))
            if content is None:
                continue
            label = "Project memory"
            if memory_kind is not None:
                label = f"{label} ({memory_kind})"
            items.append(
                ContextDisplayItem(
                    label=label,
                    source_id=source,
                    snippet=_truncate_preview(content),
                )
            )

    return tuple(items)


def _assembled_context_items(
    raw_context: dict[str, object],
) -> tuple[ContextDisplayItem, ...]:
    items: list[ContextDisplayItem] = []

    raw_memory_context = raw_context.get("memory_context")
    if isinstance(raw_memory_context, dict):
        items.extend(_memory_items(raw_memory_context)[:3])

    raw_retrieval_context = raw_context.get("retrieval_context")
    if isinstance(raw_retrieval_context, dict):
        raw_chunks = raw_retrieval_context.get("chunks")
        if isinstance(raw_chunks, list):
            for raw_chunk in raw_chunks[:2]:
                if not isinstance(raw_chunk, dict):
                    continue
                source_id = _clean_text(raw_chunk.get("source_id"))
                domain = _coerce_domain(raw_chunk.get("domain"))
                title = (
                    _clean_text(raw_chunk.get("document_title"))
                    or _clean_text(raw_chunk.get("registry_title"))
                    or source_id
                )
                excerpt = _clean_text(raw_chunk.get("excerpt"))
                if title is None or excerpt is None:
                    continue
                items.append(
                    ContextDisplayItem(
                        label=title,
                        source_id=source_id,
                        domain=domain,
                        snippet=_truncate_preview(excerpt),
                    )
                )

    return tuple(items[:5])


def _title(kind: VisibilityKind) -> str:
    return "Memory context" if kind == "memory" else "Assembled context"


def _coerce_domain(value: object) -> CreativeCodingDomain | None:
    text = _clean_text(value)
    if text is None:
        return None
    try:
        return CreativeCodingDomain(text)
    except ValueError:
        return None


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _truncate_preview(text: str, *, limit: int = 160) -> str:
    compact_text = " ".join(text.split())
    if len(compact_text) <= limit:
        return compact_text
    truncated = compact_text[: limit - 3].rstrip()
    return f"{truncated}..."
