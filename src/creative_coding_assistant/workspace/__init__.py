"""Local workspace session persistence helpers."""

from creative_coding_assistant.workspace.contracts import (
    DEFAULT_LOCAL_PROJECT_ID,
    DEFAULT_LOCAL_SESSION_ID,
    DEFAULT_LOCAL_USER_ID,
    WorkspaceSessionArtifact,
    WorkspaceSessionMessage,
    WorkspaceSessionPreferences,
    WorkspaceSessionPreview,
    WorkspaceSessionProductOutcome,
    WorkspaceSessionRecord,
    WorkspaceSessionWorkflow,
    WorkspaceSessionWorkflowStep,
    WorkspaceSessionWorkspace,
)
from creative_coding_assistant.workspace.repository import (
    SQLiteWorkspaceSessionRepository,
)
from creative_coding_assistant.workspace.service import (
    WorkspaceSessionPersistenceService,
    build_workspace_session_persistence_service,
)

__all__ = [
    "DEFAULT_LOCAL_PROJECT_ID",
    "DEFAULT_LOCAL_SESSION_ID",
    "DEFAULT_LOCAL_USER_ID",
    "SQLiteWorkspaceSessionRepository",
    "WorkspaceSessionArtifact",
    "WorkspaceSessionMessage",
    "WorkspaceSessionPersistenceService",
    "WorkspaceSessionPreferences",
    "WorkspaceSessionPreview",
    "WorkspaceSessionProductOutcome",
    "WorkspaceSessionRecord",
    "WorkspaceSessionWorkflow",
    "WorkspaceSessionWorkflowStep",
    "WorkspaceSessionWorkspace",
    "build_workspace_session_persistence_service",
]
