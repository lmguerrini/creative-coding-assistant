"""Client-facing helper modules."""

from creative_coding_assistant.clients.streamlit_context_visibility import (
    ContextDisplayItem,
    context_empty_message,
    context_expander_label,
)
from creative_coding_assistant.clients.streamlit_helpers import (
    ChatHistoryEntry,
    RetrievalDisplayItem,
    StreamRenderState,
    assistant_history_entry,
    build_chat_request,
    build_provider_warning,
    default_domain,
    default_domain_selection,
    default_mode,
    reduce_stream_event,
    resolve_request_domain,
    resolve_request_domains,
    retrieval_empty_message,
    retrieval_expander_label,
)

__all__ = [
    "ChatHistoryEntry",
    "ContextDisplayItem",
    "RetrievalDisplayItem",
    "StreamRenderState",
    "assistant_history_entry",
    "build_chat_request",
    "build_provider_warning",
    "context_empty_message",
    "context_expander_label",
    "default_domain",
    "default_domain_selection",
    "default_mode",
    "reduce_stream_event",
    "retrieval_empty_message",
    "retrieval_expander_label",
    "resolve_request_domain",
    "resolve_request_domains",
]
