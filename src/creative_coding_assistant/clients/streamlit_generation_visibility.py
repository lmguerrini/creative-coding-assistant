"""Pure helpers for provider-ready generation-input visibility in Streamlit."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import StreamEvent

VisibilityState = Literal["unknown", "empty", "available"]


class GenerationInputDisplayItem(BaseModel):
    """Small generation-input payload shape safe to render in Streamlit."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(min_length=1)
    role: str | None = None
    snippet: str = Field(min_length=1)


class GenerationInputVisibilitySummary(BaseModel):
    """Small summary metadata for one generation-input visibility expander."""

    model_config = ConfigDict(frozen=True)

    route: str | None = None
    stream: bool | None = None
    message_count: int = Field(ge=0)
    items: tuple[GenerationInputDisplayItem, ...] = Field(default_factory=tuple)


def generation_input_updates_from_event(event: StreamEvent) -> dict[str, object]:
    if event.payload.get("code") != "generation_input_prepared":
        return {}

    raw_generation_input = event.payload.get("generation_input")
    if not isinstance(raw_generation_input, dict):
        return {"generation_input_summary": None, "generation_input_state": "empty"}

    summary = _generation_input_summary(raw_generation_input)
    if summary is None or not summary.items:
        return {"generation_input_summary": None, "generation_input_state": "empty"}
    return {
        "generation_input_summary": summary,
        "generation_input_state": "available",
    }


def generation_input_expander_label(
    *,
    visibility_state: VisibilityState,
    summary: GenerationInputVisibilitySummary | None,
) -> str:
    title = "Generation input"
    if visibility_state == "available" and summary is not None:
        count = summary.message_count
        suffix = "s" if count != 1 else ""
        return f"{title} ({count} message{suffix})"
    return title


def generation_input_empty_message(
    *,
    visibility_state: VisibilityState,
) -> str | None:
    if visibility_state == "unknown":
        return None
    if visibility_state == "empty":
        return "No generation input was available for this response."
    return None


def generation_input_meta(
    summary: GenerationInputVisibilitySummary | None,
) -> str | None:
    if summary is None:
        return None

    parts: list[str] = []
    if summary.route is not None:
        parts.append(summary.route)
    if summary.stream is True:
        parts.append("stream")
    elif summary.stream is False:
        parts.append("single response")
    if not parts:
        return None
    return " | ".join(parts)


def _generation_input_summary(
    raw_generation_input: dict[str, object],
) -> GenerationInputVisibilitySummary | None:
    raw_request = raw_generation_input.get("request")
    route = None
    stream = None
    if isinstance(raw_request, dict):
        route = _clean_text(raw_request.get("route"))
        raw_stream = raw_request.get("stream")
        if isinstance(raw_stream, bool):
            stream = raw_stream

    raw_messages = raw_generation_input.get("messages")
    if not isinstance(raw_messages, list):
        return None

    items = tuple(
        item
        for item in (
            _generation_input_item(raw_message)
            for raw_message in raw_messages
            if isinstance(raw_message, dict)
        )
        if item is not None
    )
    return GenerationInputVisibilitySummary(
        route=route,
        stream=stream,
        message_count=len(items),
        items=items[:6],
    )


def _generation_input_item(
    raw_message: dict[str, object],
) -> GenerationInputDisplayItem | None:
    content = _clean_text(raw_message.get("content"))
    if content is None:
        return None

    role = _clean_text(raw_message.get("role"))
    name = _clean_text(raw_message.get("name"))
    label = _title(name or role or "message")
    return GenerationInputDisplayItem(
        label=label,
        role=role,
        snippet=_truncate_preview(content, limit=120),
    )


def _title(value: str) -> str:
    return value.replace("_", " ").strip().title()


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
