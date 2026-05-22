"""Typed contracts for durable local workspace sessions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_LOCAL_USER_ID = "local-user"
DEFAULT_LOCAL_SESSION_ID = "local-nextjs-session"
DEFAULT_LOCAL_PROJECT_ID = "local-nextjs-workspace"
WORKSPACE_SESSION_SCHEMA_VERSION = 3

InspectorTabName = Literal["Overview", "Code", "Workflow", "Artifacts", "Retrieval"]
MessageRole = Literal["user", "assistant"]
WorkspaceDensity = Literal["cozy", "compact"]
WorkspaceThemePreset = Literal["aqua", "codex", "matrix"]


class WorkspaceSessionMessage(BaseModel):
    """One chat message stored with the local workspace session."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: MessageRole
    time: str = Field(min_length=1)
    content: str


class WorkspaceSessionWorkspace(BaseModel):
    """Compact workspace identity restored by the frontend."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    name: str = Field(default="Session workspace", min_length=1)
    focus: str = Field(default="Creative coding session", min_length=1)


class WorkspaceSessionWorkflowStep(BaseModel):
    """Persisted workflow node state from the current LangGraph-style UI."""

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    node_id: str = Field(alias="nodeId", min_length=1)
    display_label: str = Field(alias="displayLabel", min_length=1)
    state: str = Field(min_length=1)
    detail: str


class WorkspaceSessionWorkflow(BaseModel):
    """Workflow snapshot stored for quiet restore after refresh."""

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    status: str = Field(default="Ready", min_length=1)
    current_node: str = Field(default="intake", alias="currentNode", min_length=1)
    current_step: str = Field(default="Intake", alias="currentStep", min_length=1)
    steps: tuple[WorkspaceSessionWorkflowStep, ...] = Field(default_factory=tuple)


class WorkspaceSessionArtifact(BaseModel):
    """Artifact summary linked to a local workspace session."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    type: str = Field(min_length=1)
    language: str = Field(min_length=1)
    status: str = Field(min_length=1)
    summary: str
    actions: tuple[str, ...] = Field(default_factory=tuple)


class WorkspaceSessionPreview(BaseModel):
    """Preview shelf state restored by the frontend."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    available: bool = False
    active: bool = False
    collapsed: bool = True
    title: str = "Preview"
    target: str = ""
    status: str = ""
    artifact_name: str = Field(default="", alias="artifactName")
    summary: str = ""
    renderer: str = ""
    trigger: str = ""
    version: str = ""


class WorkspaceSessionLayout(BaseModel):
    """Persisted IDE-like layout preferences for the Next.js workspace."""

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    density: WorkspaceDensity = "cozy"
    inspector_collapsed: bool = Field(default=False, alias="inspectorCollapsed")
    inspector_width: int = Field(default=420, alias="inspectorWidth", ge=320, le=560)
    preview_height: int = Field(default=220, alias="previewHeight", ge=160, le=360)


class WorkspaceSessionPreferences(BaseModel):
    """Persisted app-level presentation preferences for the Next.js workspace."""

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    theme: WorkspaceThemePreset = "aqua"
    auto_open_preview: bool = Field(default=True, alias="autoOpenPreview")
    show_debug_panels: bool = Field(default=True, alias="showDebugPanels")


class WorkspaceSessionRecord(BaseModel):
    """Single-user workspace persistence envelope shared with Next.js."""

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1, 2, 3] = Field(
        default=WORKSPACE_SESSION_SCHEMA_VERSION,
        alias="schemaVersion",
    )
    user_id: str = Field(default=DEFAULT_LOCAL_USER_ID, alias="userId", min_length=1)
    session_id: str = Field(
        default=DEFAULT_LOCAL_SESSION_ID,
        alias="sessionId",
        min_length=1,
    )
    project_id: str = Field(
        default=DEFAULT_LOCAL_PROJECT_ID,
        alias="projectId",
        min_length=1,
    )
    title: str = Field(default="Session workspace", min_length=1)
    active_artifact_id: str = Field(default="", alias="activeArtifactId")
    active_inspector_tab: InspectorTabName = Field(
        default="Overview",
        alias="activeInspectorTab",
    )
    preview_open: bool = Field(default=False, alias="previewOpen")
    preview_artifact_id: str = Field(default="", alias="previewArtifactId")
    workspace: WorkspaceSessionWorkspace = Field(
        default_factory=WorkspaceSessionWorkspace
    )
    messages: tuple[WorkspaceSessionMessage, ...] = Field(default_factory=tuple)
    workflow: WorkspaceSessionWorkflow | None = None
    artifacts: tuple[WorkspaceSessionArtifact, ...] = Field(default_factory=tuple)
    preview: WorkspaceSessionPreview | None = None
    layout: WorkspaceSessionLayout = Field(default_factory=WorkspaceSessionLayout)
    preferences: WorkspaceSessionPreferences = Field(
        default_factory=WorkspaceSessionPreferences
    )
    snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = Field(default=None, alias="createdAt")
    updated_at: datetime | None = Field(default=None, alias="updatedAt")

    @field_validator("user_id", "session_id", "project_id")
    @classmethod
    def reject_blank_ids(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Workspace session identifiers must not be blank.")
        return value

    def with_timestamps(
        self,
        *,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> WorkspaceSessionRecord:
        now = datetime.now(UTC)
        return self.model_copy(
            update={
                "created_at": created_at or self.created_at or now,
                "updated_at": updated_at or now,
            }
        )
