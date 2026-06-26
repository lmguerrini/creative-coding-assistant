"""HTTP API helpers for browser-facing assistant clients."""

from creative_coding_assistant.api.dev_server import (
    BackendDevApplication,
    MountedWsgiApp,
    create_backend_dev_app,
    run_backend_dev_server,
)
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
    "BackendDevApplication",
    "MountedWsgiApp",
    "WorkspaceSessionApplication",
    "create_backend_dev_app",
    "create_assistant_streaming_app",
    "create_workspace_session_app",
    "iter_assistant_stream_ndjson",
    "run_backend_dev_server",
    "serialize_stream_event",
]
