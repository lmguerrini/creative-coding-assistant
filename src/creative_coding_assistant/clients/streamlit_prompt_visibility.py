"""Pure helpers for prompt-input and rendered-prompt visibility in Streamlit."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import CreativeCodingDomain, StreamEvent

VisibilityState = Literal["unknown", "empty", "available"]
PromptVisibilityKind = Literal["prompt_input", "rendered_prompt"]


class PromptDisplayItem(BaseModel):
    """Small prompt payload shape safe to render in the Streamlit UI."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(min_length=1)
    source_id: str | None = None
    domain: CreativeCodingDomain | None = None
    snippet: str = Field(min_length=1)


class PromptVisibilitySummary(BaseModel):
    """Small summary metadata for one prompt visibility expander."""

    model_config = ConfigDict(frozen=True)

    route: str | None = None
    mode: str | None = None
    items: tuple[PromptDisplayItem, ...] = Field(default_factory=tuple)


def prompt_input_updates_from_event(event: StreamEvent) -> dict[str, object]:
    if event.payload.get("code") != "prompt_inputs_prepared":
        return {}

    raw_prompt_input = event.payload.get("prompt_input")
    if not isinstance(raw_prompt_input, dict):
        return {"prompt_input_summary": None, "prompt_input_state": "empty"}

    summary = _prompt_input_summary(raw_prompt_input)
    if summary is None or not summary.items:
        return {"prompt_input_summary": None, "prompt_input_state": "empty"}
    return {
        "prompt_input_summary": summary,
        "prompt_input_state": "available",
    }


def rendered_prompt_updates_from_event(event: StreamEvent) -> dict[str, object]:
    if event.payload.get("code") != "prompt_rendered":
        return {}

    raw_rendered_prompt = event.payload.get("rendered_prompt")
    if not isinstance(raw_rendered_prompt, dict):
        return {"rendered_prompt_summary": None, "rendered_prompt_state": "empty"}

    summary = _rendered_prompt_summary(raw_rendered_prompt)
    if summary is None or not summary.items:
        return {"rendered_prompt_summary": None, "rendered_prompt_state": "empty"}
    return {
        "rendered_prompt_summary": summary,
        "rendered_prompt_state": "available",
    }


def prompt_visibility_expander_label(
    *,
    kind: PromptVisibilityKind,
    visibility_state: VisibilityState,
    summary: PromptVisibilitySummary | None,
) -> str:
    title = (
        "Prompt input summary"
        if kind == "prompt_input"
        else "Rendered prompt summary"
    )
    if visibility_state == "available" and summary is not None:
        count = len(summary.items)
        suffix = "s" if count != 1 else ""
        return f"{title} ({count} item{suffix})"
    return title


def prompt_visibility_empty_message(
    *,
    kind: PromptVisibilityKind,
    visibility_state: VisibilityState,
) -> str | None:
    if visibility_state == "unknown":
        return None
    if visibility_state == "empty":
        if kind == "prompt_input":
            return "No prompt input summary was available for this response."
        return "No rendered prompt summary was available for this response."
    return None


def prompt_visibility_meta(
    summary: PromptVisibilitySummary | None,
) -> str | None:
    if summary is None:
        return None

    parts: list[str] = []
    if summary.route is not None:
        parts.append(summary.route)
    if summary.mode is not None:
        parts.append(summary.mode)
    if not parts:
        return None
    return " | ".join(parts)


def _prompt_input_summary(
    raw_prompt_input: dict[str, object],
) -> PromptVisibilitySummary | None:
    raw_request = raw_prompt_input.get("request")
    route = (
        _clean_text(raw_request.get("route"))
        if isinstance(raw_request, dict)
        else None
    )

    raw_user_input = raw_prompt_input.get("user_input")
    if not isinstance(raw_user_input, dict):
        return None

    items: list[PromptDisplayItem] = []
    mode = _clean_text(raw_user_input.get("mode"))
    query = _clean_text(raw_user_input.get("query"))
    domain = _coerce_domain(raw_user_input.get("domain"))
    if query is not None:
        items.append(
            PromptDisplayItem(
                label="User request",
                domain=domain,
                snippet=_truncate_preview(query, limit=120),
            )
        )

    raw_memory_input = raw_prompt_input.get("memory_input")
    if isinstance(raw_memory_input, dict):
        items.extend(_memory_input_items(raw_memory_input))

    raw_retrieval_input = raw_prompt_input.get("retrieval_input")
    if isinstance(raw_retrieval_input, dict):
        items.extend(_retrieval_input_items(raw_retrieval_input))

    return PromptVisibilitySummary(
        route=route,
        mode=mode,
        items=tuple(items[:6]),
    )


def _rendered_prompt_summary(
    raw_rendered_prompt: dict[str, object],
) -> PromptVisibilitySummary | None:
    raw_request = raw_rendered_prompt.get("request")
    route = (
        _clean_text(raw_request.get("route"))
        if isinstance(raw_request, dict)
        else None
    )
    mode = None
    if isinstance(raw_request, dict):
        raw_prompt_input = raw_request.get("prompt_input")
        if isinstance(raw_prompt_input, dict):
            raw_user_input = raw_prompt_input.get("user_input")
            if isinstance(raw_user_input, dict):
                mode = _clean_text(raw_user_input.get("mode"))

    raw_sections = raw_rendered_prompt.get("sections")
    if not isinstance(raw_sections, list):
        return None

    items = tuple(
        item
        for item in (
            _rendered_prompt_item(raw_section)
            for raw_section in raw_sections
            if isinstance(raw_section, dict)
        )
        if item is not None
    )
    return PromptVisibilitySummary(route=route, mode=mode, items=items)


def _memory_input_items(
    raw_memory_input: dict[str, object],
) -> tuple[PromptDisplayItem, ...]:
    items: list[PromptDisplayItem] = []

    running_summary = raw_memory_input.get("running_summary")
    if isinstance(running_summary, dict):
        content = _clean_text(running_summary.get("content"))
        if content is not None:
            items.append(
                PromptDisplayItem(
                    label="Running summary",
                    snippet=_truncate_preview(content, limit=110),
                )
            )

    recent_turns = raw_memory_input.get("recent_turns")
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
                PromptDisplayItem(
                    label=label,
                    snippet=_truncate_preview(content, limit=100),
                )
            )

    project_memories = raw_memory_input.get("project_memories")
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
                PromptDisplayItem(
                    label=label,
                    source_id=source,
                    snippet=_truncate_preview(content, limit=100),
                )
            )

    return tuple(items)


def _retrieval_input_items(
    raw_retrieval_input: dict[str, object],
) -> tuple[PromptDisplayItem, ...]:
    raw_chunks = raw_retrieval_input.get("chunks")
    if not isinstance(raw_chunks, list):
        return ()

    items: list[PromptDisplayItem] = []
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
            PromptDisplayItem(
                label=title,
                source_id=source_id,
                domain=domain,
                snippet=_truncate_preview(excerpt, limit=100),
            )
        )

    return tuple(items)


def _rendered_prompt_item(raw_section: dict[str, object]) -> PromptDisplayItem | None:
    name = _clean_text(raw_section.get("name"))
    role = _clean_text(raw_section.get("role"))
    content = _clean_text(raw_section.get("content"))
    if name is None or content is None:
        return None

    label = name.replace("_", " ").title()
    if role is not None:
        label = f"{label} ({role})"
    return PromptDisplayItem(
        label=label,
        snippet=_truncate_preview(content, limit=120),
    )


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


def _truncate_preview(text: str, *, limit: int) -> str:
    compact_text = " ".join(text.split())
    if len(compact_text) <= limit:
        return compact_text
    truncated = compact_text[: limit - 3].rstrip()
    return f"{truncated}..."
