"""Pure helpers for the Streamlit chat client."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.clients.streamlit_context_visibility import (
    ContextDisplayItem,
    context_updates_from_event,
    memory_updates_from_event,
)
from creative_coding_assistant.clients.streamlit_generation_visibility import (
    GenerationInputVisibilitySummary,
    generation_input_updates_from_event,
)
from creative_coding_assistant.clients.streamlit_prompt_visibility import (
    PromptVisibilitySummary,
    prompt_input_updates_from_event,
    rendered_prompt_updates_from_event,
)
from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEvent,
    StreamEventType,
)
from creative_coding_assistant.core import GenerationProviderName, Settings

_STATUS_EVENT_TYPES = frozenset(
    {
        StreamEventType.STATUS,
        StreamEventType.MEMORY,
        StreamEventType.RETRIEVAL,
        StreamEventType.CONTEXT,
        StreamEventType.PROMPT_INPUT,
        StreamEventType.PROMPT_RENDERED,
        StreamEventType.GENERATION_INPUT,
    }
)


class ChatHistoryEntry(BaseModel):
    """Minimal chat history shape stored in Streamlit session state."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)
    memory_items: tuple[ContextDisplayItem, ...] = Field(default_factory=tuple)
    memory_state: Literal["unknown", "empty", "available"] = "unknown"
    retrieval_items: tuple[RetrievalDisplayItem, ...] = Field(default_factory=tuple)
    retrieval_state: Literal["unknown", "empty", "available"] = "unknown"
    context_items: tuple[ContextDisplayItem, ...] = Field(default_factory=tuple)
    context_state: Literal["unknown", "empty", "available"] = "unknown"
    prompt_input_summary: PromptVisibilitySummary | None = None
    prompt_input_state: Literal["unknown", "empty", "available"] = "unknown"
    rendered_prompt_summary: PromptVisibilitySummary | None = None
    rendered_prompt_state: Literal["unknown", "empty", "available"] = "unknown"
    generation_input_summary: GenerationInputVisibilitySummary | None = None
    generation_input_state: Literal["unknown", "empty", "available"] = "unknown"


class RetrievalDisplayItem(BaseModel):
    """Small retrieval payload shape safe to render in the Streamlit UI."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    domain: CreativeCodingDomain
    score: float | None = Field(default=None, ge=0, le=1)
    distance: float | None = Field(default=None, ge=0)
    snippet: str = Field(min_length=1)


class StreamRenderState(BaseModel):
    """Small reducer state for rendering one streamed assistant turn."""

    model_config = ConfigDict(frozen=True)

    status_message: str | None = None
    streamed_text: str = ""
    final_answer: str | None = None
    error_message: str | None = None
    memory_items: tuple[ContextDisplayItem, ...] = Field(default_factory=tuple)
    memory_state: Literal["unknown", "empty", "available"] = "unknown"
    retrieval_items: tuple[RetrievalDisplayItem, ...] = Field(default_factory=tuple)
    retrieval_state: Literal["unknown", "empty", "available"] = "unknown"
    context_items: tuple[ContextDisplayItem, ...] = Field(default_factory=tuple)
    context_state: Literal["unknown", "empty", "available"] = "unknown"
    prompt_input_summary: PromptVisibilitySummary | None = None
    prompt_input_state: Literal["unknown", "empty", "available"] = "unknown"
    rendered_prompt_summary: PromptVisibilitySummary | None = None
    rendered_prompt_state: Literal["unknown", "empty", "available"] = "unknown"
    generation_input_summary: GenerationInputVisibilitySummary | None = None
    generation_input_state: Literal["unknown", "empty", "available"] = "unknown"

    @property
    def answer_text(self) -> str:
        return self.final_answer or self.streamed_text


def build_chat_request(
    *,
    query: str,
    conversation_id: str,
    settings: Settings,
    domains: Sequence[CreativeCodingDomain] | None = None,
    mode: AssistantMode | None = None,
) -> AssistantRequest:
    """Build one assistant request from UI input and runtime defaults."""

    resolved_domains = resolve_request_domains(domains, settings=settings)
    return AssistantRequest(
        query=query,
        conversation_id=conversation_id,
        domain=resolve_request_domain(resolved_domains),
        domains=resolved_domains,
        mode=mode or default_mode(settings),
    )


def default_domain(settings: Settings) -> CreativeCodingDomain:
    """Resolve the configured default domain safely for the client."""

    return _coerce_enum(
        enum_cls=CreativeCodingDomain,
        raw_value=settings.default_domain,
        fallback=CreativeCodingDomain.THREE_JS,
    )


def default_domain_selection() -> tuple[CreativeCodingDomain, ...]:
    """Return the default UI domain selection for Streamlit."""

    return tuple(CreativeCodingDomain)


def default_mode(settings: Settings) -> AssistantMode:
    """Resolve the configured default mode safely for the client."""

    return _coerce_enum(
        enum_cls=AssistantMode,
        raw_value=settings.default_mode,
        fallback=AssistantMode.GENERATE,
    )


def resolve_request_domain(
    domains: Sequence[CreativeCodingDomain],
) -> CreativeCodingDomain | None:
    """Resolve domain selections into the current legacy single-domain field."""

    if len(domains) == 1:
        return domains[0]
    return None


def resolve_request_domains(
    domains: Sequence[CreativeCodingDomain] | None,
    *,
    settings: Settings,
) -> tuple[CreativeCodingDomain, ...]:
    """Resolve UI selections into the first-class multi-domain request field."""

    if domains is None:
        return (default_domain(settings),)
    return tuple(dict.fromkeys(domains))


def domain_selection_summary(
    domains: Sequence[CreativeCodingDomain],
) -> str:
    """Return a short, readable sidebar summary for selected domains."""

    if not domains:
        return "Selected: none (unconstrained)"
    if len(domains) == len(CreativeCodingDomain):
        return f"Selected: all {len(domains)} domains"

    return f"Selected: {', '.join(_domain_display_name(domain) for domain in domains)}"


def mode_selection_summary(mode: AssistantMode) -> str:
    """Return a short sidebar summary for the active primary mode."""

    return f"Primary mode: {mode.value.replace('_', ' ')}"


def build_provider_warning(settings: Settings) -> str | None:
    """Return a user-safe generation readiness warning when needed."""

    if (
        settings.default_generation_provider is GenerationProviderName.OPENAI
        and not settings.has_openai_api_key
    ):
        return "Set OPENAI_API_KEY or CCA_OPENAI_API_KEY to enable live generation."
    return None


def reduce_stream_event(
    state: StreamRenderState,
    event: StreamEvent,
) -> StreamRenderState:
    """Apply one backend event to the current UI render state."""

    if event.event_type in _STATUS_EVENT_TYPES:
        message = _payload_text(event, key="message")
        updates: dict[str, object] = {}
        if message is not None:
            updates["status_message"] = message
        if event.event_type is StreamEventType.MEMORY:
            updates.update(memory_updates_from_event(event))
        if event.event_type is StreamEventType.RETRIEVAL:
            updates.update(_retrieval_updates(event))
        if event.event_type is StreamEventType.CONTEXT:
            updates.update(context_updates_from_event(event))
        if event.event_type is StreamEventType.PROMPT_INPUT:
            updates.update(prompt_input_updates_from_event(event))
        if event.event_type is StreamEventType.PROMPT_RENDERED:
            updates.update(rendered_prompt_updates_from_event(event))
        if event.event_type is StreamEventType.GENERATION_INPUT:
            updates.update(generation_input_updates_from_event(event))
        if not updates:
            return state
        return state.model_copy(update=updates)

    if event.event_type is StreamEventType.TOKEN_DELTA:
        delta = _payload_text(event, key="text") or ""
        return state.model_copy(
            update={"streamed_text": f"{state.streamed_text}{delta}"}
        )

    if event.event_type is StreamEventType.ERROR:
        message = _payload_text(event, key="message") or "Assistant request failed."
        return state.model_copy(update={"error_message": message})

    if event.event_type is StreamEventType.FINAL:
        answer = _payload_text(event, key="answer") or state.answer_text
        return state.model_copy(
            update={"final_answer": answer, "status_message": None}
        )

    return state


def assistant_history_entry(state: StreamRenderState) -> ChatHistoryEntry:
    """Build the assistant history message after one streamed turn completes."""

    content = state.final_answer or state.streamed_text or state.error_message
    if not content:
        content = "No response was generated."
    return ChatHistoryEntry(
        role="assistant",
        content=content,
        memory_items=state.memory_items,
        memory_state=state.memory_state,
        retrieval_items=state.retrieval_items,
        retrieval_state=state.retrieval_state,
        context_items=state.context_items,
        context_state=state.context_state,
        prompt_input_summary=state.prompt_input_summary,
        prompt_input_state=state.prompt_input_state,
        rendered_prompt_summary=state.rendered_prompt_summary,
        rendered_prompt_state=state.rendered_prompt_state,
        generation_input_summary=state.generation_input_summary,
        generation_input_state=state.generation_input_state,
    )


def retrieval_expander_label(
    retrieval_items: Sequence[RetrievalDisplayItem],
    *,
    retrieval_state: Literal["unknown", "empty", "available"],
) -> str:
    if retrieval_state == "available":
        count = len(retrieval_items)
        suffix = "s" if count != 1 else ""
        return f"Retrieval context ({count} chunk{suffix})"
    return "Retrieval context"


def retrieval_empty_message(
    retrieval_state: Literal["unknown", "empty", "available"],
) -> str | None:
    if retrieval_state == "empty":
        return "No retrieval context was found for this response."
    if retrieval_state == "unknown":
        return None
    return None


def _coerce_enum(*, enum_cls: type, raw_value: str, fallback: object) -> object:
    try:
        return enum_cls(str(raw_value).strip())
    except ValueError:
        return fallback


def _domain_display_name(domain: CreativeCodingDomain) -> str:
    if domain is CreativeCodingDomain.THREE_JS:
        return "Three.js"
    if domain is CreativeCodingDomain.REACT_THREE_FIBER:
        return "React Three Fiber"
    if domain is CreativeCodingDomain.P5_JS:
        return "p5.js"
    return "GLSL"


def _payload_text(event: StreamEvent, *, key: str) -> str | None:
    value = event.payload.get(key)
    if value is None:
        return None
    text = str(value)
    return text or None


def _retrieval_updates(event: StreamEvent) -> dict[str, object]:
    if event.payload.get("code") != "retrieval_completed":
        return {}

    raw_context = event.payload.get("context")
    if not isinstance(raw_context, dict):
        return {"retrieval_items": (), "retrieval_state": "empty"}

    raw_chunks = raw_context.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        return {"retrieval_items": (), "retrieval_state": "empty"}

    retrieval_items = tuple(
        item
        for item in (
            _build_retrieval_item(raw_chunk)
            for raw_chunk in raw_chunks
            if isinstance(raw_chunk, dict)
        )
        if item is not None
    )
    if not retrieval_items:
        return {"retrieval_items": (), "retrieval_state": "empty"}
    return {
        "retrieval_items": retrieval_items,
        "retrieval_state": "available",
    }


def _build_retrieval_item(raw_chunk: dict[str, object]) -> RetrievalDisplayItem | None:
    source_id = _clean_text(raw_chunk.get("source_id"))
    title = (
        _clean_text(raw_chunk.get("document_title"))
        or _clean_text(raw_chunk.get("registry_title"))
        or source_id
    )
    domain_value = _clean_text(raw_chunk.get("domain"))
    excerpt = _clean_text(raw_chunk.get("excerpt"))
    if source_id is None or title is None or domain_value is None or excerpt is None:
        return None

    try:
        domain = CreativeCodingDomain(domain_value)
    except ValueError:
        return None

    return RetrievalDisplayItem(
        source_id=source_id,
        title=title,
        domain=domain,
        score=_coerce_float(raw_chunk.get("score")),
        distance=_coerce_float(raw_chunk.get("distance")),
        snippet=_truncate_preview(excerpt),
    )


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _truncate_preview(text: str, *, limit: int = 180) -> str:
    compact_text = " ".join(text.split())
    if len(compact_text) <= limit:
        return compact_text
    truncated = compact_text[: limit - 3].rstrip()
    return f"{truncated}..."
