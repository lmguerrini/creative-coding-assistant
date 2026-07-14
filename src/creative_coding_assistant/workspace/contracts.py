"""Typed contracts for durable local workspace sessions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

DEFAULT_LOCAL_USER_ID = "local-user"
DEFAULT_LOCAL_SESSION_ID = "local-nextjs-session"
DEFAULT_LOCAL_PROJECT_ID = "local-nextjs-workspace"
WORKSPACE_SESSION_SCHEMA_VERSION = 5

InspectorTabName = Literal[
    "Overview",
    "Preview",
    "Runtime",
    "Code",
    "Workflow",
    "Telemetry",
    "Artifacts",
    "Retrieval",
]
MessageRole = Literal["user", "assistant"]
WorkspaceDensity = Literal["cozy", "compact"]
WorkspaceThemePreset = Literal[
    "aqua",
    "codex",
    "codex_white",
    "light",
    "matrix",
    "terminal",
    "horizon",
    "zen",
    "blueprint",
]
WorkspaceCreativityProfile = Literal["controlled", "balanced", "exploratory"]
WorkspaceFontScale = Literal["small", "medium", "large"]
WorkspacePreviewState = Literal["generating", "ready", "unavailable", "error"]


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


class WorkspaceSessionProductOutcome(BaseModel):
    """Canonical terminal product outcome retained across workspace reloads."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    orchestration_status: str = Field(min_length=1)
    provider_status: str = Field(min_length=1)
    generation_status: str = Field(min_length=1)
    deliverable_status: str = Field(min_length=1)
    artifact_extraction_status: str = Field(min_length=1)
    artifact_runnability: str = Field(min_length=1)
    preview_status: str = Field(min_length=1)
    runtime_health: str = Field(min_length=1)
    product_outcome: Literal[
        "IN_PROGRESS", "SUCCESS", "PARTIAL", "FAILURE"
    ] = Field(alias="product_outcome")
    summary: str = Field(min_length=1)
    recovery_action: str


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
    product_outcome: WorkspaceSessionProductOutcome | None = Field(
        default=None,
        alias="productOutcome",
    )


class WorkspaceSessionArtifact(BaseModel):
    """Artifact summary linked to a local workspace session."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    type: str = Field(min_length=1)
    language: str = Field(min_length=1)
    status: str = Field(min_length=1)
    summary: str
    content: str | None = None
    actions: tuple[str, ...] = Field(default_factory=tuple)


class WorkspaceSessionPreview(BaseModel):
    """Preview shelf state restored by the frontend."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    available: bool = False
    active: bool = False
    collapsed: bool = True
    state: WorkspacePreviewState = "unavailable"
    title: str = "Preview"
    target_id: str = Field(default="", alias="targetId")
    target: str = ""
    status: str = ""
    artifact_name: str = Field(default="", alias="artifactName")
    source_artifact_id: str = Field(default="", alias="sourceArtifactId")
    source_artifact_name: str = Field(default="", alias="sourceArtifactName")
    output_artifact_name: str = Field(default="", alias="outputArtifactName")
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
    sidebar_collapsed: bool = Field(default=False, alias="sidebarCollapsed")
    inspector_collapsed: bool = Field(default=False, alias="inspectorCollapsed")
    inspector_width: int = Field(default=420, alias="inspectorWidth", ge=320, le=560)
    preview_height: int = Field(default=320, alias="previewHeight", ge=160, le=520)


class WorkspaceFeedbackSignal(BaseModel):
    """One explicit local preference signal; it never represents model training."""

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    id: str = Field(min_length=1)
    sentiment: Literal["positive", "negative"]
    comment: str | None = Field(default=None, max_length=500)
    session_id: str = Field(alias="sessionId", min_length=1)
    artifact_id: str | None = Field(default=None, alias="artifactId")
    artifact_title: str | None = Field(default=None, alias="artifactTitle")
    domain: str | None = None
    workflow_mode: str = Field(alias="workflowMode", min_length=1)
    creativity: WorkspaceCreativityProfile
    categories: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    created_at: str = Field(alias="createdAt", min_length=1)
    prompt_excerpt: str | None = Field(default=None, alias="promptExcerpt", max_length=240)
    provider_name: str | None = Field(default=None, alias="providerName")
    provider_model: str | None = Field(default=None, alias="providerModel")
    requested_temperature: float | None = Field(
        default=None, alias="requestedTemperature", ge=0, le=2
    )
    effective_temperature: float | None = Field(
        default=None, alias="effectiveTemperature", ge=0, le=2
    )
    parameter_application: Literal[
        "requested_not_confirmed", "provider_reported"
    ] = Field(default="requested_not_confirmed", alias="parameterApplication")
    product_outcome: str | None = Field(default=None, alias="productOutcome")


class WorkspaceEvaluationHistoryEntry(BaseModel):
    """A local record of an explicitly requested evaluation attempt."""

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    id: str = Field(min_length=1)
    run_id: str | None = Field(default=None, alias="runId")
    dataset_id: str | None = Field(default=None, alias="datasetId")
    metrics: tuple[str, ...] = Field(default_factory=tuple, max_length=20)
    status: str = Field(min_length=1, max_length=80)
    detail: str = Field(min_length=1, max_length=800)
    evaluated_at: str = Field(alias="evaluatedAt", min_length=1)
    result_rows: int | None = Field(default=None, alias="resultRows", ge=0)
    metric_failures: int | None = Field(default=None, alias="metricFailures", ge=0)
    dry_run: bool | None = Field(default=None, alias="dryRun")
    provider_calls_allowed: bool | None = Field(
        default=None, alias="providerCallsAllowed"
    )
    benchmark: dict[str, JsonValue] | None = Field(
        default=None,
        alias="benchmark",
        max_length=64,
    )


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
    creativity: WorkspaceCreativityProfile = "balanced"
    personalization_enabled: bool = Field(default=True, alias="personalizationEnabled")
    heading_font_size: WorkspaceFontScale = Field(
        default="medium", alias="headingFontSize"
    )
    ui_font_size: WorkspaceFontScale = Field(default="medium", alias="uiFontSize")
    label_font_size: WorkspaceFontScale = Field(
        default="medium", alias="labelFontSize"
    )
    code_font_size: WorkspaceFontScale = Field(default="medium", alias="codeFontSize")
    feedback_signals: tuple[WorkspaceFeedbackSignal, ...] = Field(
        default_factory=tuple,
        alias="feedbackSignals",
        max_length=120,
    )
    evaluation_history: tuple[WorkspaceEvaluationHistoryEntry, ...] = Field(
        default_factory=tuple,
        alias="evaluationHistory",
        max_length=24,
    )


class WorkspaceSessionRecord(BaseModel):
    """Single-user workspace persistence envelope shared with Next.js."""

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1, 2, 3, 4, 5] = Field(
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
