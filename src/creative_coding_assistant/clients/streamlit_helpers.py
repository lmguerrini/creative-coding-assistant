"""Pure helpers for the Streamlit chat client."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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


class StreamRenderState(BaseModel):
    """Small reducer state for rendering one streamed assistant turn."""

    model_config = ConfigDict(frozen=True)

    status_message: str | None = None
    streamed_text: str = ""
    final_answer: str | None = None
    error_message: str | None = None

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
        if message is None:
            return state
        return state.model_copy(update={"status_message": message})

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
    return ChatHistoryEntry(role="assistant", content=content)


def _coerce_enum(*, enum_cls: type, raw_value: str, fallback: object) -> object:
    try:
        return enum_cls(str(raw_value).strip())
    except ValueError:
        return fallback


def _payload_text(event: StreamEvent, *, key: str) -> str | None:
    value = event.payload.get(key)
    if value is None:
        return None
    text = str(value)
    return text or None
