"""HTTP API helpers for browser-facing assistant clients."""

from creative_coding_assistant.api.streaming import (
    AssistantStreamingApplication,
    AssistantStreamRequest,
    create_assistant_streaming_app,
    iter_assistant_stream_ndjson,
    serialize_stream_event,
)
from creative_coding_assistant.api.workspace_sessions import (
    WorkspaceSessionApplication,
    create_workspace_session_app,
)

__all__ = [
    "AssistantStreamRequest",
    "AssistantStreamingApplication",
    "WorkspaceSessionApplication",
    "create_assistant_streaming_app",
    "create_workspace_session_app",
    "iter_assistant_stream_ndjson",
    "serialize_stream_event",
]
