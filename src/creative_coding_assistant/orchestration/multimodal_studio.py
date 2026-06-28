"""Passive V4.5 Multimodal Studio metadata surfaces."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.preview import PreviewTarget

LivePreviewProfileKind = Literal[
    "browser_sandbox_preview",
    "media_asset_preview",
    "structured_panel_preview",
    "runtime_status_preview",
]
MultiPreviewLayoutKind = Literal["empty", "single", "split", "grid"]
MultiPreviewOutputKind = Literal["visual", "audio", "audiovisual", "code"]
MultiPreviewProfileKind = Literal[
    "candidate_grid_preview",
    "split_comparison_preview",
    "recommended_candidate_preview",
    "comparison_fallback_preview",
]
InteractiveCanvasProfileKind = Literal[
    "canvas_surface_inspection",
    "webgl_canvas_inspection",
    "input_boundary_inspection",
    "canvas_status_inspection",
]
InteractiveCanvasSurfaceKind = Literal[
    "canvas_2d",
    "webgl_canvas",
    "input_boundary",
    "runtime_status",
]
VisualWorkspaceProfileKind = Literal[
    "workspace_shell",
    "artifact_workspace",
    "preview_workspace",
    "inspector_workspace",
]
VisualWorkspaceSurfaceKind = Literal[
    "shell",
    "artifact_selection",
    "preview",
    "inspector",
]
RuntimeCollaborationProfileKind = Literal[
    "runtime_trace_collaboration",
    "runtime_console_collaboration",
    "stream_event_collaboration",
    "operator_context_collaboration",
]
RuntimeCollaborationSurfaceKind = Literal[
    "trace",
    "console",
    "stream",
    "operator_context",
]
ArtifactCollaborationProfileKind = Literal[
    "artifact_selection_collaboration",
    "artifact_comparison_collaboration",
    "artifact_inspection_collaboration",
    "artifact_refinement_collaboration",
]
ArtifactCollaborationSurfaceKind = Literal[
    "selection",
    "comparison",
    "inspection",
    "refinement",
]
ArtifactProvenanceProfileKind = Literal[
    "source_evidence_provenance",
    "artifact_payload_provenance",
    "evaluation_provenance",
    "missing_source_provenance",
]
ArtifactProvenanceSurfaceKind = Literal[
    "evidence",
    "artifact_payload",
    "evaluation",
    "missing_source",
]
ArtifactLineageProfileKind = Literal[
    "dependency_graph_lineage",
    "timeline_stage_lineage",
    "source_transition_lineage",
    "missing_lineage",
]
ArtifactLineageSurfaceKind = Literal[
    "dependency_graph",
    "timeline_stage",
    "source_transition",
    "missing_lineage",
]

LIVE_PREVIEW_PROFILE_SERIALIZATION_VERSION = "multimodal_live_preview_profile.v1"
LIVE_PREVIEW_REGISTRY_SERIALIZATION_VERSION = "multimodal_live_preview_registry.v1"
MULTI_PREVIEW_PROFILE_SERIALIZATION_VERSION = "multimodal_multi_preview_profile.v1"
MULTI_PREVIEW_REGISTRY_SERIALIZATION_VERSION = "multimodal_multi_preview_registry.v1"
INTERACTIVE_CANVAS_PROFILE_SERIALIZATION_VERSION = (
    "multimodal_interactive_canvas_profile.v1"
)
INTERACTIVE_CANVAS_REGISTRY_SERIALIZATION_VERSION = (
    "multimodal_interactive_canvas_registry.v1"
)
VISUAL_WORKSPACE_PROFILE_SERIALIZATION_VERSION = (
    "multimodal_visual_workspace_profile.v1"
)
VISUAL_WORKSPACE_REGISTRY_SERIALIZATION_VERSION = (
    "multimodal_visual_workspace_registry.v1"
)
RUNTIME_COLLABORATION_PROFILE_SERIALIZATION_VERSION = (
    "multimodal_runtime_collaboration_profile.v1"
)
RUNTIME_COLLABORATION_REGISTRY_SERIALIZATION_VERSION = (
    "multimodal_runtime_collaboration_registry.v1"
)
ARTIFACT_COLLABORATION_PROFILE_SERIALIZATION_VERSION = (
    "multimodal_artifact_collaboration_profile.v1"
)
ARTIFACT_COLLABORATION_REGISTRY_SERIALIZATION_VERSION = (
    "multimodal_artifact_collaboration_registry.v1"
)
ARTIFACT_PROVENANCE_PROFILE_SERIALIZATION_VERSION = (
    "multimodal_artifact_provenance_profile.v1"
)
ARTIFACT_PROVENANCE_REGISTRY_SERIALIZATION_VERSION = (
    "multimodal_artifact_provenance_registry.v1"
)
ARTIFACT_LINEAGE_PROFILE_SERIALIZATION_VERSION = (
    "multimodal_artifact_lineage_profile.v1"
)
ARTIFACT_LINEAGE_REGISTRY_SERIALIZATION_VERSION = (
    "multimodal_artifact_lineage_registry.v1"
)
LIVE_PREVIEW_AUTHORITY_BOUNDARY = (
    "Live Preview metadata describes passive V4.5 Multimodal Studio surfaces "
    "for inspection only; it does not execute rendering, change browser canvas "
    "runtime behavior, route providers or models, call external providers, "
    "trigger retries, mutate generated outputs, open networking, persist "
    "collaboration state, or activate Studio runtime behavior."
)
MULTI_PREVIEW_AUTHORITY_BOUNDARY = (
    "Multi Preview metadata describes passive V4.5 Multimodal Studio "
    "comparison surfaces for inspection only; it does not execute rendering, "
    "select artifacts, mutate generated outputs, change browser canvas runtime "
    "behavior, route providers or models, call external providers, trigger "
    "retries, open networking, persist collaboration state, or activate Studio "
    "runtime behavior."
)
INTERACTIVE_CANVAS_AUTHORITY_BOUNDARY = (
    "Interactive Canvas metadata describes passive V4.5 Multimodal Studio "
    "canvas inspection surfaces only; it does not execute rendering, bind "
    "interactive input handlers, mutate canvas contexts, change browser canvas "
    "runtime behavior, route providers or models, call external providers, "
    "trigger retries, mutate generated outputs, open networking, persist "
    "collaboration state, or activate Studio runtime behavior."
)
VISUAL_WORKSPACE_AUTHORITY_BOUNDARY = (
    "Visual Workspace metadata describes passive V4.5 Multimodal Studio "
    "workspace surfaces for inspection only; it does not mutate workspace "
    "state, create persistent storage behavior, execute rendering, route "
    "providers or models, call external providers, trigger retries, mutate "
    "generated outputs, open networking, or activate Studio runtime behavior."
)
RUNTIME_COLLABORATION_AUTHORITY_BOUNDARY = (
    "Runtime Collaboration metadata describes passive V4.5 Multimodal Studio "
    "runtime collaboration surfaces for inspection only; it does not open "
    "real-time networking, synchronize external peers, persist collaboration "
    "state, execute runtime behavior, control workflows, request human input, "
    "route providers or models, trigger retries, or mutate generated outputs."
)
ARTIFACT_COLLABORATION_AUTHORITY_BOUNDARY = (
    "Artifact Collaboration metadata describes passive V4.5 Multimodal Studio "
    "artifact collaboration surfaces for inspection only; it does not mutate "
    "artifacts, modify generated outputs, create persistent collaboration "
    "storage, execute rendering, control workflows, request human input, route "
    "providers or models, trigger retries, or open networking."
)
ARTIFACT_PROVENANCE_AUTHORITY_BOUNDARY = (
    "Artifact Provenance metadata describes passive V4.5 Multimodal Studio "
    "provenance surfaces for inspection only; it does not record provenance, "
    "persist provenance storage, mutate artifacts, modify generated outputs, "
    "execute rendering, control workflows, request human input, route providers "
    "or models, trigger retries, or open networking."
)
ARTIFACT_LINEAGE_AUTHORITY_BOUNDARY = (
    "Artifact Lineage metadata describes passive V4.5 Multimodal Studio "
    "lineage surfaces for inspection only; it does not infer lineage "
    "dynamically, reconstruct timelines, record provenance, persist lineage "
    "storage, mutate artifacts, modify generated outputs, execute rendering, "
    "control workflows, request human input, route providers or models, "
    "trigger retries, or open networking."
)

_LIVE_PREVIEW_SOURCE_REGISTRIES = (
    "preview_contracts",
    "workflow_artifact_preview_preparation",
    "nextjs_preview_targets",
    "nextjs_preview_renderers",
    "nextjs_preview_runtime_adapters",
    "nextjs_preview_sandbox_runtime",
)

_LIVE_PREVIEW_SOURCE_REFERENCES = (
    "preview.contracts.PreviewTarget",
    "preview.contracts.PreviewRequest",
    "preview.contracts.PreviewResult",
    "preview.contracts.PreviewStatus",
    "orchestration.artifacts.prepare_workflow_preview_results",
    "clients.nextjs.preview_targets.derivePreviewTargetIdFromArtifact",
    "clients.nextjs.preview_renderers.creativePreviewRendererRegistry",
    "clients.nextjs.preview_runtime_adapters.PreviewRuntimeStatus",
    "clients.nextjs.preview_sandbox_runtime.mountPreviewSandboxRuntime",
)

_LIVE_PREVIEW_SURFACES = (
    "live_preview_shelf",
    "live_preview_target_panel",
    "preview_renderer_match_panel",
    "preview_source_metadata_panel",
    "preview_status_panel",
    "preview_boundary_panel",
)

_LIVE_PREVIEW_OBSERVABILITY_SURFACES = (
    "profile_id",
    "surface_kind",
    "preview_targets",
    "renderer_contract_refs",
    "source_reference_ids",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_LIVE_PREVIEW_BLOCKED_RUNTIME_BEHAVIORS = (
    "rendering_execution",
    "browser_canvas_runtime_change",
    "provider_or_model_routing",
    "external_provider_calling",
    "retry_triggering",
    "generated_output_mutation",
    "networking",
    "persistent_collaboration_storage",
    "active_studio_runtime_behavior",
)

_MULTI_PREVIEW_SOURCE_REGISTRIES = (
    "multimodal_live_preview_registry",
    "nextjs_multi_preview_comparison",
    "nextjs_multi_preview_workspace",
    "nextjs_artifact_comparison",
    "nextjs_preview_renderers",
    "nextjs_preview_runtime_adapters",
)

_MULTI_PREVIEW_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_LIVE_PREVIEW_REGISTRY",
    "clients.nextjs.multi_preview_comparison.buildMultiPreviewComparisonModel",
    "clients.nextjs.multi_preview_comparison.resolveMultiPreviewLayout",
    "clients.nextjs.multi_preview_comparison.MultiPreviewCandidate",
    "clients.nextjs.components.MultiPreviewComparisonWorkspace",
    "clients.nextjs.artifact_comparison.buildArtifactComparisonModel",
    "clients.nextjs.preview_renderers.buildPreviewRendererRoute",
    "clients.nextjs.preview_runtime_adapters.buildPreviewRuntimeSource",
)

_MULTI_PREVIEW_SURFACES = (
    "multi_preview_workspace",
    "multi_preview_candidate_grid",
    "multi_preview_split_layout",
    "candidate_preview_card",
    "comparison_fallback_panel",
    "recommendation_summary_panel",
    "multi_preview_boundary_panel",
)

_MULTI_PREVIEW_OBSERVABILITY_SURFACES = (
    "profile_id",
    "preview_kind",
    "comparison_layouts",
    "source_live_preview_profile_ids",
    "candidate_state_fields",
    "source_reference_ids",
    "authority_boundary",
)

_MULTI_PREVIEW_BLOCKED_RUNTIME_BEHAVIORS = (
    "rendering_execution",
    "candidate_selection_execution",
    "artifact_selection_mutation",
    "browser_canvas_runtime_change",
    "provider_or_model_routing",
    "external_provider_calling",
    "retry_triggering",
    "generated_output_mutation",
    "networking",
    "persistent_collaboration_storage",
    "active_studio_runtime_behavior",
)

_INTERACTIVE_CANVAS_SOURCE_REGISTRIES = (
    "multimodal_live_preview_registry",
    "multimodal_multi_preview_registry",
    "nextjs_svg_canvas_runtime",
    "nextjs_preview_runtime_adapters",
    "nextjs_preview_sandbox_runtime",
    "nextjs_preview_renderers",
)

_INTERACTIVE_CANVAS_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_LIVE_PREVIEW_REGISTRY",
    "multimodal_studio.MULTIMODAL_MULTI_PREVIEW_REGISTRY",
    "clients.nextjs.svg_canvas_runtime.hasCanvasPreviewSignal",
    "clients.nextjs.svg_canvas_runtime.getCanvasRuntimeSupportIssue",
    "clients.nextjs.preview_runtime_adapters.buildPreviewRuntimeSource",
    "clients.nextjs.preview_runtime_adapters.mountPreviewRuntime",
    "clients.nextjs.preview_sandbox_runtime.buildPreviewSandboxDocument",
    "clients.nextjs.preview_renderers.surface.canvas",
)

_INTERACTIVE_CANVAS_SURFACES = (
    "interactive_canvas_panel",
    "canvas_surface_contract_panel",
    "canvas_input_boundary_panel",
    "canvas_runtime_status_panel",
    "canvas_source_guardrail_panel",
    "canvas_fallback_panel",
    "interactive_canvas_boundary_panel",
)

_INTERACTIVE_CANVAS_OBSERVABILITY_SURFACES = (
    "profile_id",
    "canvas_profile_kind",
    "canvas_surface_kind",
    "source_live_preview_profile_ids",
    "source_multi_preview_profile_ids",
    "source_reference_ids",
    "authority_boundary",
)

_INTERACTIVE_CANVAS_BLOCKED_RUNTIME_BEHAVIORS = (
    "rendering_execution",
    "interactive_input_binding",
    "browser_canvas_runtime_change",
    "canvas_context_mutation",
    "provider_or_model_routing",
    "external_provider_calling",
    "retry_triggering",
    "generated_output_mutation",
    "networking",
    "persistent_collaboration_storage",
    "active_studio_runtime_behavior",
)

_VISUAL_WORKSPACE_SOURCE_REGISTRIES = (
    "multimodal_live_preview_registry",
    "multimodal_multi_preview_registry",
    "multimodal_interactive_canvas_registry",
    "nextjs_workstation_state",
    "nextjs_workstation_dashboard",
    "nextjs_workstation_shell",
    "nextjs_workspace_persistence",
)

_VISUAL_WORKSPACE_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_LIVE_PREVIEW_REGISTRY",
    "multimodal_studio.MULTIMODAL_MULTI_PREVIEW_REGISTRY",
    "multimodal_studio.MULTIMODAL_INTERACTIVE_CANVAS_REGISTRY",
    "clients.nextjs.workstation_state.buildWorkstationState",
    "clients.nextjs.workstation_dashboard.buildWorkstationDashboardModel",
    "clients.nextjs.workstation_shell.WorkstationShell",
    "clients.nextjs.workspace_persistence.createWorkspaceSessionRecord",
    "clients.nextjs.assistant_client.AssistantWorkspaceSnapshot",
)

_VISUAL_WORKSPACE_SURFACES = (
    "visual_workspace_shell",
    "artifact_selection_surface",
    "preview_workspace_surface",
    "inspector_workspace_surface",
    "workspace_dashboard_surface",
    "visual_context_surface",
    "workspace_boundary_panel",
)

_VISUAL_WORKSPACE_OBSERVABILITY_SURFACES = (
    "profile_id",
    "workspace_profile_kind",
    "workspace_surface_kind",
    "source_preview_profile_ids",
    "workspace_state_fields",
    "source_reference_ids",
    "authority_boundary",
)

_VISUAL_WORKSPACE_BLOCKED_RUNTIME_BEHAVIORS = (
    "workspace_state_mutation",
    "persistent_storage_mutation",
    "rendering_execution",
    "provider_or_model_routing",
    "external_provider_calling",
    "retry_triggering",
    "generated_output_mutation",
    "networking",
    "active_studio_runtime_behavior",
)

_RUNTIME_COLLABORATION_SOURCE_REGISTRIES = (
    "multimodal_visual_workspace_registry",
    "nextjs_workflow_runtime",
    "nextjs_runtime_console",
    "nextjs_assistant_stream",
    "nextjs_workstation_shell",
    "nextjs_provider_telemetry",
    "nextjs_session_intelligence",
)

_RUNTIME_COLLABORATION_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_VISUAL_WORKSPACE_REGISTRY",
    "clients.nextjs.workflow_runtime.buildWorkflowRuntimeModel",
    "clients.nextjs.runtime_console.buildRuntimeConsoleModel",
    "clients.nextjs.assistant_stream.streamAssistantEvents",
    "clients.nextjs.workstation_shell.applyStreamEventToWorkspace",
    "clients.nextjs.provider_telemetry.buildProviderTelemetryModel",
    "clients.nextjs.session_intelligence.buildSessionIntelligenceModel",
    "clients.nextjs.session_intelligence.readSessionIntelligenceMetadata",
)

_RUNTIME_COLLABORATION_SURFACES = (
    "runtime_collaboration_panel",
    "runtime_trace_surface",
    "runtime_console_surface",
    "stream_event_surface",
    "operator_context_surface",
    "runtime_health_surface",
    "runtime_collaboration_boundary_panel",
)

_RUNTIME_COLLABORATION_OBSERVABILITY_SURFACES = (
    "profile_id",
    "collaboration_profile_kind",
    "collaboration_surface_kind",
    "source_visual_workspace_profile_ids",
    "runtime_context_fields",
    "source_reference_ids",
    "authority_boundary",
)

_RUNTIME_COLLABORATION_BLOCKED_RUNTIME_BEHAVIORS = (
    "real_time_networking",
    "external_peer_synchronization",
    "persistent_collaboration_storage",
    "runtime_execution",
    "workflow_control",
    "human_input_request",
    "provider_or_model_routing",
    "retry_triggering",
    "generated_output_mutation",
)

_ARTIFACT_COLLABORATION_SOURCE_REGISTRIES = (
    "multimodal_visual_workspace_registry",
    "multimodal_runtime_collaboration_registry",
    "nextjs_artifact_comparison",
    "nextjs_artifact_inspector",
    "nextjs_artifact_refinement",
    "nextjs_multi_preview_comparison",
    "nextjs_workstation_shell",
)

_ARTIFACT_COLLABORATION_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_VISUAL_WORKSPACE_REGISTRY",
    "multimodal_studio.MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY",
    "clients.nextjs.artifact_comparison.buildArtifactComparisonModel",
    "clients.nextjs.artifact_inspector.buildArtifactDocument",
    "clients.nextjs.artifact_inspector.highlightArtifactDocument",
    "clients.nextjs.artifact_refinement.enrichArtifactRefinementRequest",
    "clients.nextjs.multi_preview_comparison.buildMultiPreviewComparisonModel",
    "clients.nextjs.workstation_shell.handleArtifactRefine",
)

_ARTIFACT_COLLABORATION_SURFACES = (
    "artifact_collaboration_panel",
    "artifact_selection_surface",
    "artifact_comparison_surface",
    "artifact_inspection_surface",
    "artifact_refinement_surface",
    "artifact_action_feedback_surface",
    "artifact_collaboration_boundary_panel",
)

_ARTIFACT_COLLABORATION_OBSERVABILITY_SURFACES = (
    "profile_id",
    "artifact_profile_kind",
    "artifact_surface_kind",
    "source_workspace_profile_ids",
    "source_runtime_collaboration_profile_ids",
    "source_reference_ids",
    "authority_boundary",
)

_ARTIFACT_COLLABORATION_BLOCKED_RUNTIME_BEHAVIORS = (
    "artifact_mutation",
    "generated_output_mutation",
    "persistent_collaboration_storage",
    "rendering_execution",
    "workflow_control",
    "human_input_request",
    "provider_or_model_routing",
    "retry_triggering",
    "networking",
)

_ARTIFACT_PROVENANCE_SOURCE_REGISTRIES = (
    "multimodal_artifact_collaboration_registry",
    "multimodal_runtime_collaboration_registry",
    "nextjs_provenance_engine",
    "nextjs_v3_inspector_panels",
    "nextjs_workstation_state",
    "preview_contracts",
    "workflow_trace_events",
)

_ARTIFACT_PROVENANCE_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY",
    "multimodal_studio.MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY",
    "clients.nextjs.provenance_engine.buildProvenanceEngineModel",
    "clients.nextjs.provenance_engine.ProvenanceSource",
    "clients.nextjs.v3_inspector_panels.buildProvenancePanel",
    "clients.nextjs.workstation_state.buildWorkstationState",
    "preview.contracts.PreviewProvenance",
    "clients.nextjs.workflow_runtime.WorkflowRuntimeTraceEvent",
)

_ARTIFACT_PROVENANCE_SURFACES = (
    "artifact_provenance_panel",
    "evidence_source_surface",
    "artifact_payload_source_surface",
    "evaluation_source_surface",
    "missing_source_surface",
    "provenance_summary_surface",
    "artifact_provenance_boundary_panel",
)

_ARTIFACT_PROVENANCE_OBSERVABILITY_SURFACES = (
    "profile_id",
    "provenance_profile_kind",
    "provenance_surface_kind",
    "source_artifact_collaboration_profile_ids",
    "source_runtime_collaboration_profile_ids",
    "source_reference_ids",
    "authority_boundary",
)

_ARTIFACT_PROVENANCE_BLOCKED_RUNTIME_BEHAVIORS = (
    "provenance_recording",
    "persistent_provenance_storage",
    "artifact_mutation",
    "generated_output_mutation",
    "rendering_execution",
    "workflow_control",
    "human_input_request",
    "provider_or_model_routing",
    "retry_triggering",
    "networking",
)

_ARTIFACT_LINEAGE_SOURCE_REGISTRIES = (
    "multimodal_artifact_provenance_registry",
    "orchestration_artifact_dependency_graph",
    "nextjs_provenance_engine",
    "nextjs_creative_timeline",
    "nextjs_workflow_explorer",
    "nextjs_workflow_runtime",
    "nextjs_workstation_shell",
)

_ARTIFACT_LINEAGE_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY",
    "orchestration.artifact_dependency_graph.ArtifactDependencyGraph",
    "orchestration.artifact_dependency_graph.ArtifactDependencyEdge",
    "clients.nextjs.provenance_engine.buildProvenanceEngineModel",
    "clients.nextjs.provenance_engine.ProvenanceSource",
    "clients.nextjs.creative_timeline.buildCreativeTimelineModel",
    "clients.nextjs.creative_timeline.provenanceSourceCount",
    "clients.nextjs.workflow_explorer.WorkflowExplorerStage",
    "clients.nextjs.workflow_runtime.WorkflowRuntimeTraceEvent",
    "clients.nextjs.workstation_shell.WorkstationShell",
)

_ARTIFACT_LINEAGE_SURFACES = (
    "artifact_lineage_panel",
    "dependency_graph_lineage_surface",
    "timeline_stage_lineage_surface",
    "source_transition_lineage_surface",
    "missing_lineage_surface",
    "lineage_summary_surface",
    "artifact_lineage_boundary_panel",
)

_ARTIFACT_LINEAGE_OBSERVABILITY_SURFACES = (
    "profile_id",
    "lineage_profile_kind",
    "lineage_surface_kind",
    "source_artifact_provenance_profile_ids",
    "lineage_context_fields",
    "source_reference_ids",
    "authority_boundary",
)

_ARTIFACT_LINEAGE_BLOCKED_RUNTIME_BEHAVIORS = (
    "lineage_inference",
    "timeline_reconstruction",
    "provenance_recording",
    "persistent_lineage_storage",
    "artifact_mutation",
    "generated_output_mutation",
    "rendering_execution",
    "workflow_control",
    "human_input_request",
    "provider_or_model_routing",
    "retry_triggering",
    "networking",
)


class LivePreviewProfile(BaseModel):
    """Inspectable metadata for one passive Live Preview surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1, max_length=120)
    profile_name: str = Field(min_length=1, max_length=140)
    surface_kind: LivePreviewProfileKind
    preview_targets: tuple[PreviewTarget, ...] = Field(min_length=1, max_length=6)
    renderer_contract_refs: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_reference_ids: tuple[str, ...] = Field(min_length=1, max_length=9)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    live_preview_surfaces: tuple[str, ...] = Field(min_length=1, max_length=6)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=LIVE_PREVIEW_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_LIVE_PREVIEW_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    rendering_execution_implemented: Literal[False] = False
    browser_canvas_runtime_change_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    external_provider_calls_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_storage_implemented: Literal[False] = False
    active_studio_behavior_implemented: Literal[False] = False
    serialization_version: Literal["multimodal_live_preview_profile.v1"] = (
        LIVE_PREVIEW_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class MultimodalLivePreviewRegistry(BaseModel):
    """Stable passive registry for V4.5 Multimodal Studio Live Preview."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["multimodal_live_preview_registry"] = (
        "multimodal_live_preview_registry"
    )
    serialization_version: Literal["multimodal_live_preview_registry.v1"] = (
        LIVE_PREVIEW_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=LIVE_PREVIEW_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    live_preview_profiles: tuple[LivePreviewProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    surface_kinds: tuple[LivePreviewProfileKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    preview_targets: tuple[PreviewTarget, ...] = Field(min_length=6, max_length=6)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_reference_ids: tuple[str, ...] = Field(min_length=9, max_length=9)
    live_preview_surface_refs: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_LIVE_PREVIEW_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    rendering_execution_implemented: Literal[False] = False
    browser_canvas_runtime_change_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    external_provider_calls_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_storage_implemented: Literal[False] = False
    active_studio_behavior_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.profile_id for profile in self.live_preview_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("profile_ids must be unique")
        if self.profile_ids != derived_profile_ids:
            raise ValueError("profile_ids must match live_preview_profiles")
        if self.profile_count != len(self.live_preview_profiles):
            raise ValueError("profile_count must match live_preview_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.preview_targets != tuple(PreviewTarget):
            raise ValueError("preview_targets must match preview target enum order")

        derived_surface_kinds = _ordered_unique(
            profile.surface_kind for profile in self.live_preview_profiles
        )
        if self.surface_kinds != derived_surface_kinds:
            raise ValueError("surface_kinds must match live preview profiles")

        profile_source_references = {
            source_reference
            for profile in self.live_preview_profiles
            for source_reference in profile.source_reference_ids
        }
        if set(self.source_reference_ids) != profile_source_references:
            raise ValueError(
                "source_reference_ids must match profile source references"
            )

        known_routes = set(self.route_names)
        known_targets = set(self.preview_targets)
        known_surfaces = set(self.live_preview_surface_refs)
        known_source_references = set(self.source_reference_ids)
        for profile in self.live_preview_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must use known routes")
            if not set(profile.preview_targets).issubset(known_targets):
                raise ValueError("preview_targets must use known targets")
            if not set(profile.live_preview_surfaces).issubset(known_surfaces):
                raise ValueError("live_preview_surfaces must be known surfaces")
            if not set(profile.source_reference_ids).issubset(
                known_source_references
            ):
                raise ValueError(
                    "source_reference_ids must be known registry references"
                )
        return self


def multimodal_live_preview_registry() -> MultimodalLivePreviewRegistry:
    """Return passive V4.5 Multimodal Studio Live Preview metadata."""

    return MULTIMODAL_LIVE_PREVIEW_REGISTRY


def multimodal_live_preview_profile_by_id(
    profile_id: str,
    registry: MultimodalLivePreviewRegistry | None = None,
) -> LivePreviewProfile | None:
    """Return one Live Preview profile without executing preview behavior."""

    source_registry = registry or MULTIMODAL_LIVE_PREVIEW_REGISTRY
    normalized_profile_id = str(profile_id).strip()
    for profile in source_registry.live_preview_profiles:
        if profile.profile_id == normalized_profile_id:
            return profile
    return None


def multimodal_live_preview_profiles_for_route(
    route: RouteName | str,
    registry: MultimodalLivePreviewRegistry | None = None,
) -> tuple[LivePreviewProfile, ...]:
    """Return passive Live Preview profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or MULTIMODAL_LIVE_PREVIEW_REGISTRY
    return tuple(
        profile
        for profile in source_registry.live_preview_profiles
        if route_name in profile.route_applicability
    )


def multimodal_live_preview_profiles_for_target(
    target: PreviewTarget | str,
    registry: MultimodalLivePreviewRegistry | None = None,
) -> tuple[LivePreviewProfile, ...]:
    """Return passive Live Preview profiles covering a preview target."""

    preview_target = (
        target if isinstance(target, PreviewTarget) else PreviewTarget(str(target))
    )
    source_registry = registry or MULTIMODAL_LIVE_PREVIEW_REGISTRY
    return tuple(
        profile
        for profile in source_registry.live_preview_profiles
        if preview_target in profile.preview_targets
    )


def multimodal_live_preview_profiles_for_source_reference(
    source_reference_id: str,
    registry: MultimodalLivePreviewRegistry | None = None,
) -> tuple[LivePreviewProfile, ...]:
    """Return passive Live Preview profiles referencing one source surface."""

    source_registry = registry or MULTIMODAL_LIVE_PREVIEW_REGISTRY
    source_reference = str(source_reference_id).strip()
    return tuple(
        profile
        for profile in source_registry.live_preview_profiles
        if source_reference in profile.source_reference_ids
    )


def _live_preview_profile(
    *,
    profile_id: str,
    profile_name: str,
    surface_kind: LivePreviewProfileKind,
    preview_targets: tuple[PreviewTarget, ...],
    renderer_contract_refs: tuple[str, ...],
    source_reference_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    live_preview_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> LivePreviewProfile:
    return LivePreviewProfile(
        profile_id=profile_id,
        profile_name=profile_name,
        surface_kind=surface_kind,
        preview_targets=preview_targets,
        renderer_contract_refs=renderer_contract_refs,
        source_reference_ids=source_reference_ids,
        route_applicability=route_applicability,
        live_preview_surfaces=live_preview_surfaces,
        advisory_outputs=advisory_outputs,
        source_registries=_LIVE_PREVIEW_SOURCE_REGISTRIES,
        observability_surfaces=_LIVE_PREVIEW_OBSERVABILITY_SURFACES,
    )


_T = TypeVar("_T")


def _ordered_unique(values: Iterable[_T]) -> tuple[_T, ...]:
    resolved: list[_T] = []
    for value in values:
        if value not in resolved:
            resolved.append(value)
    return tuple(resolved)


MULTIMODAL_LIVE_PREVIEW_PROFILES = (
    _live_preview_profile(
        profile_id="browser_sandbox_live_preview",
        profile_name="Browser Sandbox Live Preview",
        surface_kind="browser_sandbox_preview",
        preview_targets=(PreviewTarget.BROWSER_SANDBOX,),
        renderer_contract_refs=(
            "surface.p5",
            "surface.three",
            "surface.glsl",
            "surface.hydra",
            "surface.tone",
            "surface.gsap",
            "surface.svg",
            "surface.canvas",
        ),
        source_reference_ids=(
            "preview.contracts.PreviewTarget",
            "preview.contracts.PreviewRequest",
            "orchestration.artifacts.prepare_workflow_preview_results",
            "clients.nextjs.preview_renderers.creativePreviewRendererRegistry",
            "clients.nextjs.preview_sandbox_runtime.mountPreviewSandboxRuntime",
        ),
        route_applicability=tuple(RouteName),
        live_preview_surfaces=(
            "live_preview_shelf",
            "live_preview_target_panel",
            "preview_renderer_match_panel",
            "preview_source_metadata_panel",
            "preview_boundary_panel",
        ),
        advisory_outputs=(
            "browser_sandbox_preview_inventory",
            "manual_preview_target_review_hint",
            "no_rendering_execution_notice",
        ),
    ),
    _live_preview_profile(
        profile_id="media_asset_live_preview",
        profile_name="Media Asset Live Preview",
        surface_kind="media_asset_preview",
        preview_targets=(
            PreviewTarget.IMAGE_ASSET,
            PreviewTarget.AUDIO_ASSET,
            PreviewTarget.VIDEO_ASSET,
        ),
        renderer_contract_refs=(
            "image_asset_surface",
            "audio_asset_surface",
            "video_asset_surface",
        ),
        source_reference_ids=(
            "preview.contracts.PreviewTarget",
            "preview.contracts.PreviewResult",
            "clients.nextjs.preview_targets.derivePreviewTargetIdFromArtifact",
        ),
        route_applicability=tuple(RouteName),
        live_preview_surfaces=(
            "live_preview_shelf",
            "live_preview_target_panel",
            "preview_source_metadata_panel",
            "preview_boundary_panel",
        ),
        advisory_outputs=(
            "media_asset_preview_inventory",
            "manual_media_target_review_hint",
            "no_asset_mutation_notice",
        ),
    ),
    _live_preview_profile(
        profile_id="structured_panel_live_preview",
        profile_name="Structured Panel Live Preview",
        surface_kind="structured_panel_preview",
        preview_targets=(
            PreviewTarget.TEXT_PANEL,
            PreviewTarget.JSON_PANEL,
        ),
        renderer_contract_refs=(
            "text_panel_surface",
            "json_panel_surface",
        ),
        source_reference_ids=(
            "preview.contracts.PreviewTarget",
            "preview.contracts.PreviewResult",
            "clients.nextjs.preview_targets.derivePreviewTargetIdFromArtifact",
            "clients.nextjs.preview_renderers.creativePreviewRendererRegistry",
        ),
        route_applicability=tuple(RouteName),
        live_preview_surfaces=(
            "live_preview_shelf",
            "live_preview_target_panel",
            "preview_renderer_match_panel",
            "preview_source_metadata_panel",
            "preview_boundary_panel",
        ),
        advisory_outputs=(
            "structured_panel_preview_inventory",
            "manual_panel_review_hint",
            "no_generated_output_mutation_notice",
        ),
    ),
    _live_preview_profile(
        profile_id="runtime_status_live_preview",
        profile_name="Runtime Status Live Preview",
        surface_kind="runtime_status_preview",
        preview_targets=(
            PreviewTarget.BROWSER_SANDBOX,
            PreviewTarget.TEXT_PANEL,
            PreviewTarget.JSON_PANEL,
        ),
        renderer_contract_refs=(
            "preview_runtime_status",
            "preview_runtime_source",
        ),
        source_reference_ids=(
            "preview.contracts.PreviewResult",
            "preview.contracts.PreviewStatus",
            "clients.nextjs.preview_runtime_adapters.PreviewRuntimeStatus",
            "clients.nextjs.preview_sandbox_runtime.mountPreviewSandboxRuntime",
        ),
        route_applicability=tuple(RouteName),
        live_preview_surfaces=(
            "live_preview_shelf",
            "preview_status_panel",
            "preview_source_metadata_panel",
            "preview_boundary_panel",
        ),
        advisory_outputs=(
            "runtime_status_preview_inventory",
            "manual_runtime_status_review_hint",
            "no_runtime_control_notice",
        ),
    ),
)

MULTIMODAL_LIVE_PREVIEW_REGISTRY = MultimodalLivePreviewRegistry(
    live_preview_profiles=MULTIMODAL_LIVE_PREVIEW_PROFILES,
    profile_ids=tuple(
        profile.profile_id for profile in MULTIMODAL_LIVE_PREVIEW_PROFILES
    ),
    surface_kinds=tuple(
        profile.surface_kind for profile in MULTIMODAL_LIVE_PREVIEW_PROFILES
    ),
    preview_targets=tuple(PreviewTarget),
    route_names=tuple(RouteName),
    profile_count=len(MULTIMODAL_LIVE_PREVIEW_PROFILES),
    source_registries=_LIVE_PREVIEW_SOURCE_REGISTRIES,
    source_reference_ids=_LIVE_PREVIEW_SOURCE_REFERENCES,
    live_preview_surface_refs=_LIVE_PREVIEW_SURFACES,
    observability_surfaces=_LIVE_PREVIEW_OBSERVABILITY_SURFACES,
)


class MultiPreviewProfile(BaseModel):
    """Inspectable metadata for one passive Multi Preview surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1, max_length=120)
    profile_name: str = Field(min_length=1, max_length=140)
    preview_kind: MultiPreviewProfileKind
    comparison_layouts: tuple[MultiPreviewLayoutKind, ...] = Field(
        min_length=1,
        max_length=4,
    )
    output_kinds: tuple[MultiPreviewOutputKind, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_live_preview_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    candidate_state_fields: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_reference_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    multi_preview_surfaces: tuple[str, ...] = Field(min_length=1, max_length=7)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=MULTI_PREVIEW_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_MULTI_PREVIEW_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    rendering_execution_implemented: Literal[False] = False
    candidate_selection_execution_implemented: Literal[False] = False
    artifact_selection_mutation_implemented: Literal[False] = False
    browser_canvas_runtime_change_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    external_provider_calls_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_storage_implemented: Literal[False] = False
    active_studio_behavior_implemented: Literal[False] = False
    serialization_version: Literal["multimodal_multi_preview_profile.v1"] = (
        MULTI_PREVIEW_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class MultimodalMultiPreviewRegistry(BaseModel):
    """Stable passive registry for V4.5 Multimodal Studio Multi Preview."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["multimodal_multi_preview_registry"] = (
        "multimodal_multi_preview_registry"
    )
    serialization_version: Literal["multimodal_multi_preview_registry.v1"] = (
        MULTI_PREVIEW_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=MULTI_PREVIEW_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    multi_preview_profiles: tuple[MultiPreviewProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    preview_kinds: tuple[MultiPreviewProfileKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    comparison_layouts: tuple[MultiPreviewLayoutKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    output_kinds: tuple[MultiPreviewOutputKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    live_preview_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_reference_ids: tuple[str, ...] = Field(min_length=8, max_length=8)
    multi_preview_surface_refs: tuple[str, ...] = Field(min_length=7, max_length=7)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_MULTI_PREVIEW_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    rendering_execution_implemented: Literal[False] = False
    candidate_selection_execution_implemented: Literal[False] = False
    artifact_selection_mutation_implemented: Literal[False] = False
    browser_canvas_runtime_change_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    external_provider_calls_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_storage_implemented: Literal[False] = False
    active_studio_behavior_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.profile_id for profile in self.multi_preview_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("profile_ids must be unique")
        if self.profile_ids != derived_profile_ids:
            raise ValueError("profile_ids must match multi_preview_profiles")
        if self.profile_count != len(self.multi_preview_profiles):
            raise ValueError("profile_count must match multi_preview_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if (
            self.live_preview_profile_ids
            != MULTIMODAL_LIVE_PREVIEW_REGISTRY.profile_ids
        ):
            raise ValueError(
                "live_preview_profile_ids must match Live Preview registry"
            )

        derived_preview_kinds = _ordered_unique(
            profile.preview_kind for profile in self.multi_preview_profiles
        )
        if self.preview_kinds != derived_preview_kinds:
            raise ValueError("preview_kinds must match multi preview profiles")

        profile_source_references = {
            source_reference
            for profile in self.multi_preview_profiles
            for source_reference in profile.source_reference_ids
        }
        if set(self.source_reference_ids) != profile_source_references:
            raise ValueError(
                "source_reference_ids must match profile source references"
            )

        profile_layouts = {
            layout
            for profile in self.multi_preview_profiles
            for layout in profile.comparison_layouts
        }
        if set(self.comparison_layouts) != profile_layouts:
            raise ValueError("comparison_layouts must match profile layouts")

        profile_output_kinds = {
            output_kind
            for profile in self.multi_preview_profiles
            for output_kind in profile.output_kinds
        }
        if set(self.output_kinds) != profile_output_kinds:
            raise ValueError("output_kinds must match profile output kinds")

        known_routes = set(self.route_names)
        known_layouts = set(self.comparison_layouts)
        known_output_kinds = set(self.output_kinds)
        known_live_profiles = set(self.live_preview_profile_ids)
        known_surfaces = set(self.multi_preview_surface_refs)
        known_source_references = set(self.source_reference_ids)
        for profile in self.multi_preview_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must use known routes")
            if not set(profile.comparison_layouts).issubset(known_layouts):
                raise ValueError("comparison_layouts must use known layouts")
            if not set(profile.output_kinds).issubset(known_output_kinds):
                raise ValueError("output_kinds must use known output kinds")
            if not set(profile.source_live_preview_profile_ids).issubset(
                known_live_profiles
            ):
                raise ValueError(
                    "source_live_preview_profile_ids must be known profiles"
                )
            if not set(profile.multi_preview_surfaces).issubset(known_surfaces):
                raise ValueError("multi_preview_surfaces must be known surfaces")
            if not set(profile.source_reference_ids).issubset(
                known_source_references
            ):
                raise ValueError(
                    "source_reference_ids must be known registry references"
                )
        return self


def multimodal_multi_preview_registry() -> MultimodalMultiPreviewRegistry:
    """Return passive V4.5 Multimodal Studio Multi Preview metadata."""

    return MULTIMODAL_MULTI_PREVIEW_REGISTRY


def multimodal_multi_preview_profile_by_id(
    profile_id: str,
    registry: MultimodalMultiPreviewRegistry | None = None,
) -> MultiPreviewProfile | None:
    """Return one Multi Preview profile without executing preview behavior."""

    source_registry = registry or MULTIMODAL_MULTI_PREVIEW_REGISTRY
    normalized_profile_id = str(profile_id).strip()
    for profile in source_registry.multi_preview_profiles:
        if profile.profile_id == normalized_profile_id:
            return profile
    return None


def multimodal_multi_preview_profiles_for_route(
    route: RouteName | str,
    registry: MultimodalMultiPreviewRegistry | None = None,
) -> tuple[MultiPreviewProfile, ...]:
    """Return passive Multi Preview profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or MULTIMODAL_MULTI_PREVIEW_REGISTRY
    return tuple(
        profile
        for profile in source_registry.multi_preview_profiles
        if route_name in profile.route_applicability
    )


def multimodal_multi_preview_profiles_for_layout(
    layout: MultiPreviewLayoutKind | str,
    registry: MultimodalMultiPreviewRegistry | None = None,
) -> tuple[MultiPreviewProfile, ...]:
    """Return passive Multi Preview profiles covering a comparison layout."""

    layout_name = str(layout).strip()
    source_registry = registry or MULTIMODAL_MULTI_PREVIEW_REGISTRY
    return tuple(
        profile
        for profile in source_registry.multi_preview_profiles
        if layout_name in profile.comparison_layouts
    )


def multimodal_multi_preview_profiles_for_live_preview_profile(
    live_preview_profile_id: str,
    registry: MultimodalMultiPreviewRegistry | None = None,
) -> tuple[MultiPreviewProfile, ...]:
    """Return Multi Preview profiles referencing one Live Preview profile."""

    source_registry = registry or MULTIMODAL_MULTI_PREVIEW_REGISTRY
    source_profile_id = str(live_preview_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.multi_preview_profiles
        if source_profile_id in profile.source_live_preview_profile_ids
    )


def _multi_preview_profile(
    *,
    profile_id: str,
    profile_name: str,
    preview_kind: MultiPreviewProfileKind,
    comparison_layouts: tuple[MultiPreviewLayoutKind, ...],
    output_kinds: tuple[MultiPreviewOutputKind, ...],
    source_live_preview_profile_ids: tuple[str, ...],
    candidate_state_fields: tuple[str, ...],
    source_reference_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    multi_preview_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> MultiPreviewProfile:
    return MultiPreviewProfile(
        profile_id=profile_id,
        profile_name=profile_name,
        preview_kind=preview_kind,
        comparison_layouts=comparison_layouts,
        output_kinds=output_kinds,
        source_live_preview_profile_ids=source_live_preview_profile_ids,
        candidate_state_fields=candidate_state_fields,
        source_reference_ids=source_reference_ids,
        route_applicability=route_applicability,
        multi_preview_surfaces=multi_preview_surfaces,
        advisory_outputs=advisory_outputs,
        source_registries=_MULTI_PREVIEW_SOURCE_REGISTRIES,
        observability_surfaces=_MULTI_PREVIEW_OBSERVABILITY_SURFACES,
    )


MULTIMODAL_MULTI_PREVIEW_PROFILES = (
    _multi_preview_profile(
        profile_id="candidate_grid_multi_preview",
        profile_name="Candidate Grid Multi Preview",
        preview_kind="candidate_grid_preview",
        comparison_layouts=("grid",),
        output_kinds=("visual", "audio", "audiovisual", "code"),
        source_live_preview_profile_ids=MULTIMODAL_LIVE_PREVIEW_REGISTRY.profile_ids,
        candidate_state_fields=(
            "candidate_count",
            "layout",
            "can_render",
            "runtime_session_key",
            "runtime_source",
            "output_kind",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_LIVE_PREVIEW_REGISTRY",
            "clients.nextjs.multi_preview_comparison.buildMultiPreviewComparisonModel",
            "clients.nextjs.multi_preview_comparison.MultiPreviewCandidate",
            "clients.nextjs.components.MultiPreviewComparisonWorkspace",
            "clients.nextjs.preview_renderers.buildPreviewRendererRoute",
            "clients.nextjs.preview_runtime_adapters.buildPreviewRuntimeSource",
        ),
        route_applicability=tuple(RouteName),
        multi_preview_surfaces=(
            "multi_preview_workspace",
            "multi_preview_candidate_grid",
            "candidate_preview_card",
            "multi_preview_boundary_panel",
        ),
        advisory_outputs=(
            "candidate_grid_preview_inventory",
            "manual_candidate_review_hint",
            "no_rendering_execution_notice",
        ),
    ),
    _multi_preview_profile(
        profile_id="split_comparison_multi_preview",
        profile_name="Split Comparison Multi Preview",
        preview_kind="split_comparison_preview",
        comparison_layouts=("split",),
        output_kinds=("visual", "audio", "audiovisual", "code"),
        source_live_preview_profile_ids=(
            "browser_sandbox_live_preview",
            "media_asset_live_preview",
            "structured_panel_live_preview",
        ),
        candidate_state_fields=(
            "active_artifact_id",
            "comparison_rows",
            "preview_route",
            "preview_target",
            "score_label",
        ),
        source_reference_ids=(
            "clients.nextjs.multi_preview_comparison.buildMultiPreviewComparisonModel",
            "clients.nextjs.multi_preview_comparison.resolveMultiPreviewLayout",
            "clients.nextjs.artifact_comparison.buildArtifactComparisonModel",
            "clients.nextjs.preview_renderers.buildPreviewRendererRoute",
        ),
        route_applicability=tuple(RouteName),
        multi_preview_surfaces=(
            "multi_preview_workspace",
            "multi_preview_split_layout",
            "candidate_preview_card",
            "recommendation_summary_panel",
            "multi_preview_boundary_panel",
        ),
        advisory_outputs=(
            "split_comparison_preview_inventory",
            "manual_split_review_hint",
            "no_artifact_selection_mutation_notice",
        ),
    ),
    _multi_preview_profile(
        profile_id="recommended_candidate_multi_preview",
        profile_name="Recommended Candidate Multi Preview",
        preview_kind="recommended_candidate_preview",
        comparison_layouts=("single", "split", "grid"),
        output_kinds=("visual", "audio", "audiovisual", "code"),
        source_live_preview_profile_ids=MULTIMODAL_LIVE_PREVIEW_REGISTRY.profile_ids,
        candidate_state_fields=(
            "recommended_title",
            "recommended_reason",
            "is_recommended",
            "rank_label",
            "score_label",
        ),
        source_reference_ids=(
            "clients.nextjs.multi_preview_comparison.buildMultiPreviewComparisonModel",
            "clients.nextjs.components.MultiPreviewComparisonWorkspace",
            "clients.nextjs.artifact_comparison.buildArtifactComparisonModel",
        ),
        route_applicability=tuple(RouteName),
        multi_preview_surfaces=(
            "multi_preview_workspace",
            "candidate_preview_card",
            "recommendation_summary_panel",
            "multi_preview_boundary_panel",
        ),
        advisory_outputs=(
            "recommended_candidate_preview_inventory",
            "manual_recommendation_review_hint",
            "no_candidate_selection_execution_notice",
        ),
    ),
    _multi_preview_profile(
        profile_id="fallback_multi_preview",
        profile_name="Fallback Multi Preview",
        preview_kind="comparison_fallback_preview",
        comparison_layouts=("empty", "single", "split", "grid"),
        output_kinds=("code", "audio"),
        source_live_preview_profile_ids=(
            "structured_panel_live_preview",
            "runtime_status_live_preview",
        ),
        candidate_state_fields=(
            "can_render",
            "runtime_support",
            "preview_state",
            "preview_label",
            "audio_safety_label",
        ),
        source_reference_ids=(
            "clients.nextjs.multi_preview_comparison.resolveMultiPreviewLayout",
            "clients.nextjs.multi_preview_comparison.MultiPreviewCandidate",
            "clients.nextjs.components.MultiPreviewComparisonWorkspace",
        ),
        route_applicability=tuple(RouteName),
        multi_preview_surfaces=(
            "multi_preview_workspace",
            "comparison_fallback_panel",
            "candidate_preview_card",
            "multi_preview_boundary_panel",
        ),
        advisory_outputs=(
            "fallback_multi_preview_inventory",
            "manual_fallback_review_hint",
            "no_runtime_control_notice",
        ),
    ),
)

MULTIMODAL_MULTI_PREVIEW_REGISTRY = MultimodalMultiPreviewRegistry(
    multi_preview_profiles=MULTIMODAL_MULTI_PREVIEW_PROFILES,
    profile_ids=tuple(
        profile.profile_id for profile in MULTIMODAL_MULTI_PREVIEW_PROFILES
    ),
    preview_kinds=tuple(
        profile.preview_kind for profile in MULTIMODAL_MULTI_PREVIEW_PROFILES
    ),
    comparison_layouts=("empty", "single", "split", "grid"),
    output_kinds=("visual", "audio", "audiovisual", "code"),
    live_preview_profile_ids=MULTIMODAL_LIVE_PREVIEW_REGISTRY.profile_ids,
    route_names=tuple(RouteName),
    profile_count=len(MULTIMODAL_MULTI_PREVIEW_PROFILES),
    source_registries=_MULTI_PREVIEW_SOURCE_REGISTRIES,
    source_reference_ids=_MULTI_PREVIEW_SOURCE_REFERENCES,
    multi_preview_surface_refs=_MULTI_PREVIEW_SURFACES,
    observability_surfaces=_MULTI_PREVIEW_OBSERVABILITY_SURFACES,
)


class InteractiveCanvasProfile(BaseModel):
    """Inspectable metadata for one passive Interactive Canvas surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1, max_length=120)
    profile_name: str = Field(min_length=1, max_length=140)
    canvas_profile_kind: InteractiveCanvasProfileKind
    canvas_surface_kind: InteractiveCanvasSurfaceKind
    preview_targets: tuple[PreviewTarget, ...] = Field(min_length=1, max_length=2)
    source_live_preview_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_multi_preview_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    canvas_signal_refs: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_reference_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    interactive_canvas_surfaces: tuple[str, ...] = Field(
        min_length=1,
        max_length=7,
    )
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=INTERACTIVE_CANVAS_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_INTERACTIVE_CANVAS_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    rendering_execution_implemented: Literal[False] = False
    interactive_input_binding_implemented: Literal[False] = False
    browser_canvas_runtime_change_implemented: Literal[False] = False
    canvas_context_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    external_provider_calls_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_storage_implemented: Literal[False] = False
    active_studio_behavior_implemented: Literal[False] = False
    serialization_version: Literal["multimodal_interactive_canvas_profile.v1"] = (
        INTERACTIVE_CANVAS_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class MultimodalInteractiveCanvasRegistry(BaseModel):
    """Stable passive registry for V4.5 Multimodal Studio Interactive Canvas."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["multimodal_interactive_canvas_registry"] = (
        "multimodal_interactive_canvas_registry"
    )
    serialization_version: Literal["multimodal_interactive_canvas_registry.v1"] = (
        INTERACTIVE_CANVAS_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=INTERACTIVE_CANVAS_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    interactive_canvas_profiles: tuple[InteractiveCanvasProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    canvas_profile_kinds: tuple[InteractiveCanvasProfileKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    canvas_surface_kinds: tuple[InteractiveCanvasSurfaceKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    preview_targets: tuple[PreviewTarget, ...] = Field(min_length=1, max_length=1)
    live_preview_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    multi_preview_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_reference_ids: tuple[str, ...] = Field(min_length=8, max_length=8)
    interactive_canvas_surface_refs: tuple[str, ...] = Field(
        min_length=7,
        max_length=7,
    )
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_INTERACTIVE_CANVAS_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    rendering_execution_implemented: Literal[False] = False
    interactive_input_binding_implemented: Literal[False] = False
    browser_canvas_runtime_change_implemented: Literal[False] = False
    canvas_context_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    external_provider_calls_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_storage_implemented: Literal[False] = False
    active_studio_behavior_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.profile_id for profile in self.interactive_canvas_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("profile_ids must be unique")
        if self.profile_ids != derived_profile_ids:
            raise ValueError("profile_ids must match interactive_canvas_profiles")
        if self.profile_count != len(self.interactive_canvas_profiles):
            raise ValueError(
                "profile_count must match interactive_canvas_profiles"
            )
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.preview_targets != (PreviewTarget.BROWSER_SANDBOX,):
            raise ValueError(
                "preview_targets must describe the browser sandbox canvas"
            )
        if (
            self.live_preview_profile_ids
            != MULTIMODAL_LIVE_PREVIEW_REGISTRY.profile_ids
        ):
            raise ValueError(
                "live_preview_profile_ids must match Live Preview registry"
            )
        if (
            self.multi_preview_profile_ids
            != MULTIMODAL_MULTI_PREVIEW_REGISTRY.profile_ids
        ):
            raise ValueError(
                "multi_preview_profile_ids must match Multi Preview registry"
            )

        derived_profile_kinds = _ordered_unique(
            profile.canvas_profile_kind
            for profile in self.interactive_canvas_profiles
        )
        if self.canvas_profile_kinds != derived_profile_kinds:
            raise ValueError(
                "canvas_profile_kinds must match interactive canvas profiles"
            )

        derived_surface_kinds = _ordered_unique(
            profile.canvas_surface_kind
            for profile in self.interactive_canvas_profiles
        )
        if self.canvas_surface_kinds != derived_surface_kinds:
            raise ValueError(
                "canvas_surface_kinds must match interactive canvas profiles"
            )

        profile_source_references = {
            source_reference
            for profile in self.interactive_canvas_profiles
            for source_reference in profile.source_reference_ids
        }
        if set(self.source_reference_ids) != profile_source_references:
            raise ValueError(
                "source_reference_ids must match profile source references"
            )

        known_routes = set(self.route_names)
        known_targets = set(self.preview_targets)
        known_live_profiles = set(self.live_preview_profile_ids)
        known_multi_profiles = set(self.multi_preview_profile_ids)
        known_surfaces = set(self.interactive_canvas_surface_refs)
        known_source_references = set(self.source_reference_ids)
        for profile in self.interactive_canvas_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must use known routes")
            if not set(profile.preview_targets).issubset(known_targets):
                raise ValueError("preview_targets must use known targets")
            if not set(profile.source_live_preview_profile_ids).issubset(
                known_live_profiles
            ):
                raise ValueError(
                    "source_live_preview_profile_ids must be known profiles"
                )
            if not set(profile.source_multi_preview_profile_ids).issubset(
                known_multi_profiles
            ):
                raise ValueError(
                    "source_multi_preview_profile_ids must be known profiles"
                )
            if not set(profile.interactive_canvas_surfaces).issubset(
                known_surfaces
            ):
                raise ValueError(
                    "interactive_canvas_surfaces must be known surfaces"
                )
            if not set(profile.source_reference_ids).issubset(
                known_source_references
            ):
                raise ValueError(
                    "source_reference_ids must be known registry references"
                )
        return self


def multimodal_interactive_canvas_registry() -> (
    MultimodalInteractiveCanvasRegistry
):
    """Return passive V4.5 Multimodal Studio Interactive Canvas metadata."""

    return MULTIMODAL_INTERACTIVE_CANVAS_REGISTRY


def multimodal_interactive_canvas_profile_by_id(
    profile_id: str,
    registry: MultimodalInteractiveCanvasRegistry | None = None,
) -> InteractiveCanvasProfile | None:
    """Return one Interactive Canvas profile without executing canvas behavior."""

    source_registry = registry or MULTIMODAL_INTERACTIVE_CANVAS_REGISTRY
    normalized_profile_id = str(profile_id).strip()
    for profile in source_registry.interactive_canvas_profiles:
        if profile.profile_id == normalized_profile_id:
            return profile
    return None


def multimodal_interactive_canvas_profiles_for_route(
    route: RouteName | str,
    registry: MultimodalInteractiveCanvasRegistry | None = None,
) -> tuple[InteractiveCanvasProfile, ...]:
    """Return passive Interactive Canvas profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or MULTIMODAL_INTERACTIVE_CANVAS_REGISTRY
    return tuple(
        profile
        for profile in source_registry.interactive_canvas_profiles
        if route_name in profile.route_applicability
    )


def multimodal_interactive_canvas_profiles_for_surface_kind(
    surface_kind: InteractiveCanvasSurfaceKind | str,
    registry: MultimodalInteractiveCanvasRegistry | None = None,
) -> tuple[InteractiveCanvasProfile, ...]:
    """Return passive Interactive Canvas profiles for one surface kind."""

    surface_value = str(surface_kind).strip()
    source_registry = registry or MULTIMODAL_INTERACTIVE_CANVAS_REGISTRY
    return tuple(
        profile
        for profile in source_registry.interactive_canvas_profiles
        if profile.canvas_surface_kind == surface_value
    )


def multimodal_interactive_canvas_profiles_for_live_preview_profile(
    live_preview_profile_id: str,
    registry: MultimodalInteractiveCanvasRegistry | None = None,
) -> tuple[InteractiveCanvasProfile, ...]:
    """Return Interactive Canvas profiles referencing one Live Preview profile."""

    source_registry = registry or MULTIMODAL_INTERACTIVE_CANVAS_REGISTRY
    source_profile_id = str(live_preview_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.interactive_canvas_profiles
        if source_profile_id in profile.source_live_preview_profile_ids
    )


def multimodal_interactive_canvas_profiles_for_multi_preview_profile(
    multi_preview_profile_id: str,
    registry: MultimodalInteractiveCanvasRegistry | None = None,
) -> tuple[InteractiveCanvasProfile, ...]:
    """Return Interactive Canvas profiles referencing one Multi Preview profile."""

    source_registry = registry or MULTIMODAL_INTERACTIVE_CANVAS_REGISTRY
    source_profile_id = str(multi_preview_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.interactive_canvas_profiles
        if source_profile_id in profile.source_multi_preview_profile_ids
    )


def _interactive_canvas_profile(
    *,
    profile_id: str,
    profile_name: str,
    canvas_profile_kind: InteractiveCanvasProfileKind,
    canvas_surface_kind: InteractiveCanvasSurfaceKind,
    source_live_preview_profile_ids: tuple[str, ...],
    source_multi_preview_profile_ids: tuple[str, ...],
    canvas_signal_refs: tuple[str, ...],
    source_reference_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    interactive_canvas_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> InteractiveCanvasProfile:
    return InteractiveCanvasProfile(
        profile_id=profile_id,
        profile_name=profile_name,
        canvas_profile_kind=canvas_profile_kind,
        canvas_surface_kind=canvas_surface_kind,
        preview_targets=(PreviewTarget.BROWSER_SANDBOX,),
        source_live_preview_profile_ids=source_live_preview_profile_ids,
        source_multi_preview_profile_ids=source_multi_preview_profile_ids,
        canvas_signal_refs=canvas_signal_refs,
        source_reference_ids=source_reference_ids,
        route_applicability=route_applicability,
        interactive_canvas_surfaces=interactive_canvas_surfaces,
        advisory_outputs=advisory_outputs,
        source_registries=_INTERACTIVE_CANVAS_SOURCE_REGISTRIES,
        observability_surfaces=_INTERACTIVE_CANVAS_OBSERVABILITY_SURFACES,
    )


MULTIMODAL_INTERACTIVE_CANVAS_PROFILES = (
    _interactive_canvas_profile(
        profile_id="canvas_2d_interactive_canvas",
        profile_name="Canvas 2D Interactive Canvas",
        canvas_profile_kind="canvas_surface_inspection",
        canvas_surface_kind="canvas_2d",
        source_live_preview_profile_ids=(
            "browser_sandbox_live_preview",
            "runtime_status_live_preview",
        ),
        source_multi_preview_profile_ids=(
            "candidate_grid_multi_preview",
            "fallback_multi_preview",
        ),
        canvas_signal_refs=(
            "hasCanvasPreviewSignal",
            "getCanvasRuntimeSupportIssue",
            "surface.canvas",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_LIVE_PREVIEW_REGISTRY",
            "multimodal_studio.MULTIMODAL_MULTI_PREVIEW_REGISTRY",
            "clients.nextjs.svg_canvas_runtime.hasCanvasPreviewSignal",
            "clients.nextjs.svg_canvas_runtime.getCanvasRuntimeSupportIssue",
            "clients.nextjs.preview_renderers.surface.canvas",
        ),
        route_applicability=tuple(RouteName),
        interactive_canvas_surfaces=(
            "interactive_canvas_panel",
            "canvas_surface_contract_panel",
            "canvas_source_guardrail_panel",
            "interactive_canvas_boundary_panel",
        ),
        advisory_outputs=(
            "canvas_2d_surface_inventory",
            "manual_canvas_surface_review_hint",
            "no_canvas_context_mutation_notice",
        ),
    ),
    _interactive_canvas_profile(
        profile_id="webgl_interactive_canvas",
        profile_name="WebGL Interactive Canvas",
        canvas_profile_kind="webgl_canvas_inspection",
        canvas_surface_kind="webgl_canvas",
        source_live_preview_profile_ids=(
            "browser_sandbox_live_preview",
            "runtime_status_live_preview",
        ),
        source_multi_preview_profile_ids=(
            "candidate_grid_multi_preview",
            "recommended_candidate_multi_preview",
        ),
        canvas_signal_refs=(
            "surface.three",
            "surface.glsl",
            "buildPreviewRuntimeSource",
            "mountPreviewRuntime",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_LIVE_PREVIEW_REGISTRY",
            "multimodal_studio.MULTIMODAL_MULTI_PREVIEW_REGISTRY",
            "clients.nextjs.preview_runtime_adapters.buildPreviewRuntimeSource",
            "clients.nextjs.preview_runtime_adapters.mountPreviewRuntime",
            "clients.nextjs.preview_sandbox_runtime.buildPreviewSandboxDocument",
        ),
        route_applicability=tuple(RouteName),
        interactive_canvas_surfaces=(
            "interactive_canvas_panel",
            "canvas_surface_contract_panel",
            "canvas_runtime_status_panel",
            "interactive_canvas_boundary_panel",
        ),
        advisory_outputs=(
            "webgl_canvas_surface_inventory",
            "manual_webgl_canvas_review_hint",
            "no_rendering_execution_notice",
        ),
    ),
    _interactive_canvas_profile(
        profile_id="input_boundary_interactive_canvas",
        profile_name="Input Boundary Interactive Canvas",
        canvas_profile_kind="input_boundary_inspection",
        canvas_surface_kind="input_boundary",
        source_live_preview_profile_ids=(
            "browser_sandbox_live_preview",
            "runtime_status_live_preview",
        ),
        source_multi_preview_profile_ids=(
            "candidate_grid_multi_preview",
            "split_comparison_multi_preview",
            "fallback_multi_preview",
        ),
        canvas_signal_refs=(
            "interactive_input_handler_patterns",
            "getCanvasRuntimeSupportIssue",
            "sandbox_boundary_panel",
        ),
        source_reference_ids=(
            "clients.nextjs.svg_canvas_runtime.getCanvasRuntimeSupportIssue",
            "clients.nextjs.preview_sandbox_runtime.buildPreviewSandboxDocument",
            "clients.nextjs.preview_runtime_adapters.mountPreviewRuntime",
        ),
        route_applicability=tuple(RouteName),
        interactive_canvas_surfaces=(
            "interactive_canvas_panel",
            "canvas_input_boundary_panel",
            "canvas_source_guardrail_panel",
            "interactive_canvas_boundary_panel",
        ),
        advisory_outputs=(
            "input_boundary_inventory",
            "manual_input_boundary_review_hint",
            "no_interactive_input_binding_notice",
        ),
    ),
    _interactive_canvas_profile(
        profile_id="runtime_status_interactive_canvas",
        profile_name="Runtime Status Interactive Canvas",
        canvas_profile_kind="canvas_status_inspection",
        canvas_surface_kind="runtime_status",
        source_live_preview_profile_ids=("runtime_status_live_preview",),
        source_multi_preview_profile_ids=(
            "candidate_grid_multi_preview",
            "fallback_multi_preview",
        ),
        canvas_signal_refs=(
            "preview_runtime_status",
            "preview_runtime_source",
            "canvas_fallback_panel",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_LIVE_PREVIEW_REGISTRY",
            "multimodal_studio.MULTIMODAL_MULTI_PREVIEW_REGISTRY",
            "clients.nextjs.preview_runtime_adapters.buildPreviewRuntimeSource",
            "clients.nextjs.preview_sandbox_runtime.buildPreviewSandboxDocument",
        ),
        route_applicability=tuple(RouteName),
        interactive_canvas_surfaces=(
            "interactive_canvas_panel",
            "canvas_runtime_status_panel",
            "canvas_fallback_panel",
            "interactive_canvas_boundary_panel",
        ),
        advisory_outputs=(
            "canvas_runtime_status_inventory",
            "manual_canvas_status_review_hint",
            "no_runtime_control_notice",
        ),
    ),
)

MULTIMODAL_INTERACTIVE_CANVAS_REGISTRY = MultimodalInteractiveCanvasRegistry(
    interactive_canvas_profiles=MULTIMODAL_INTERACTIVE_CANVAS_PROFILES,
    profile_ids=tuple(
        profile.profile_id for profile in MULTIMODAL_INTERACTIVE_CANVAS_PROFILES
    ),
    canvas_profile_kinds=tuple(
        profile.canvas_profile_kind
        for profile in MULTIMODAL_INTERACTIVE_CANVAS_PROFILES
    ),
    canvas_surface_kinds=tuple(
        profile.canvas_surface_kind
        for profile in MULTIMODAL_INTERACTIVE_CANVAS_PROFILES
    ),
    preview_targets=(PreviewTarget.BROWSER_SANDBOX,),
    live_preview_profile_ids=MULTIMODAL_LIVE_PREVIEW_REGISTRY.profile_ids,
    multi_preview_profile_ids=MULTIMODAL_MULTI_PREVIEW_REGISTRY.profile_ids,
    route_names=tuple(RouteName),
    profile_count=len(MULTIMODAL_INTERACTIVE_CANVAS_PROFILES),
    source_registries=_INTERACTIVE_CANVAS_SOURCE_REGISTRIES,
    source_reference_ids=_INTERACTIVE_CANVAS_SOURCE_REFERENCES,
    interactive_canvas_surface_refs=_INTERACTIVE_CANVAS_SURFACES,
    observability_surfaces=_INTERACTIVE_CANVAS_OBSERVABILITY_SURFACES,
)


class VisualWorkspaceProfile(BaseModel):
    """Inspectable metadata for one passive Visual Workspace surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1, max_length=120)
    profile_name: str = Field(min_length=1, max_length=140)
    workspace_profile_kind: VisualWorkspaceProfileKind
    workspace_surface_kind: VisualWorkspaceSurfaceKind
    source_live_preview_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_multi_preview_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_interactive_canvas_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    workspace_state_fields: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_reference_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    visual_workspace_surfaces: tuple[str, ...] = Field(
        min_length=1,
        max_length=7,
    )
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=VISUAL_WORKSPACE_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_VISUAL_WORKSPACE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    workspace_state_mutation_implemented: Literal[False] = False
    persistent_storage_mutation_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    external_provider_calls_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    active_studio_behavior_implemented: Literal[False] = False
    serialization_version: Literal["multimodal_visual_workspace_profile.v1"] = (
        VISUAL_WORKSPACE_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class MultimodalVisualWorkspaceRegistry(BaseModel):
    """Stable passive registry for V4.5 Multimodal Studio Visual Workspace."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["multimodal_visual_workspace_registry"] = (
        "multimodal_visual_workspace_registry"
    )
    serialization_version: Literal["multimodal_visual_workspace_registry.v1"] = (
        VISUAL_WORKSPACE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=VISUAL_WORKSPACE_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    visual_workspace_profiles: tuple[VisualWorkspaceProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    workspace_profile_kinds: tuple[VisualWorkspaceProfileKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    workspace_surface_kinds: tuple[VisualWorkspaceSurfaceKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    live_preview_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    multi_preview_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    interactive_canvas_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    source_reference_ids: tuple[str, ...] = Field(min_length=8, max_length=8)
    visual_workspace_surface_refs: tuple[str, ...] = Field(
        min_length=7,
        max_length=7,
    )
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_VISUAL_WORKSPACE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    workspace_state_mutation_implemented: Literal[False] = False
    persistent_storage_mutation_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    external_provider_calls_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    active_studio_behavior_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.profile_id for profile in self.visual_workspace_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("profile_ids must be unique")
        if self.profile_ids != derived_profile_ids:
            raise ValueError("profile_ids must match visual_workspace_profiles")
        if self.profile_count != len(self.visual_workspace_profiles):
            raise ValueError("profile_count must match visual_workspace_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if (
            self.live_preview_profile_ids
            != MULTIMODAL_LIVE_PREVIEW_REGISTRY.profile_ids
        ):
            raise ValueError(
                "live_preview_profile_ids must match Live Preview registry"
            )
        if (
            self.multi_preview_profile_ids
            != MULTIMODAL_MULTI_PREVIEW_REGISTRY.profile_ids
        ):
            raise ValueError(
                "multi_preview_profile_ids must match Multi Preview registry"
            )
        if (
            self.interactive_canvas_profile_ids
            != MULTIMODAL_INTERACTIVE_CANVAS_REGISTRY.profile_ids
        ):
            raise ValueError(
                "interactive_canvas_profile_ids must match Interactive Canvas registry"
            )

        derived_profile_kinds = _ordered_unique(
            profile.workspace_profile_kind
            for profile in self.visual_workspace_profiles
        )
        if self.workspace_profile_kinds != derived_profile_kinds:
            raise ValueError(
                "workspace_profile_kinds must match visual workspace profiles"
            )

        derived_surface_kinds = _ordered_unique(
            profile.workspace_surface_kind
            for profile in self.visual_workspace_profiles
        )
        if self.workspace_surface_kinds != derived_surface_kinds:
            raise ValueError(
                "workspace_surface_kinds must match visual workspace profiles"
            )

        profile_source_references = {
            source_reference
            for profile in self.visual_workspace_profiles
            for source_reference in profile.source_reference_ids
        }
        if set(self.source_reference_ids) != profile_source_references:
            raise ValueError(
                "source_reference_ids must match profile source references"
            )

        known_routes = set(self.route_names)
        known_live_profiles = set(self.live_preview_profile_ids)
        known_multi_profiles = set(self.multi_preview_profile_ids)
        known_canvas_profiles = set(self.interactive_canvas_profile_ids)
        known_surfaces = set(self.visual_workspace_surface_refs)
        known_source_references = set(self.source_reference_ids)
        for profile in self.visual_workspace_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must use known routes")
            if not set(profile.source_live_preview_profile_ids).issubset(
                known_live_profiles
            ):
                raise ValueError(
                    "source_live_preview_profile_ids must be known profiles"
                )
            if not set(profile.source_multi_preview_profile_ids).issubset(
                known_multi_profiles
            ):
                raise ValueError(
                    "source_multi_preview_profile_ids must be known profiles"
                )
            if not set(profile.source_interactive_canvas_profile_ids).issubset(
                known_canvas_profiles
            ):
                raise ValueError(
                    "source_interactive_canvas_profile_ids must be known profiles"
                )
            if not set(profile.visual_workspace_surfaces).issubset(known_surfaces):
                raise ValueError("visual_workspace_surfaces must be known surfaces")
            if not set(profile.source_reference_ids).issubset(
                known_source_references
            ):
                raise ValueError(
                    "source_reference_ids must be known registry references"
                )
        return self


def multimodal_visual_workspace_registry() -> MultimodalVisualWorkspaceRegistry:
    """Return passive V4.5 Multimodal Studio Visual Workspace metadata."""

    return MULTIMODAL_VISUAL_WORKSPACE_REGISTRY


def multimodal_visual_workspace_profile_by_id(
    profile_id: str,
    registry: MultimodalVisualWorkspaceRegistry | None = None,
) -> VisualWorkspaceProfile | None:
    """Return one Visual Workspace profile without mutating workspace state."""

    source_registry = registry or MULTIMODAL_VISUAL_WORKSPACE_REGISTRY
    normalized_profile_id = str(profile_id).strip()
    for profile in source_registry.visual_workspace_profiles:
        if profile.profile_id == normalized_profile_id:
            return profile
    return None


def multimodal_visual_workspace_profiles_for_route(
    route: RouteName | str,
    registry: MultimodalVisualWorkspaceRegistry | None = None,
) -> tuple[VisualWorkspaceProfile, ...]:
    """Return passive Visual Workspace profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or MULTIMODAL_VISUAL_WORKSPACE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.visual_workspace_profiles
        if route_name in profile.route_applicability
    )


def multimodal_visual_workspace_profiles_for_surface_kind(
    surface_kind: VisualWorkspaceSurfaceKind | str,
    registry: MultimodalVisualWorkspaceRegistry | None = None,
) -> tuple[VisualWorkspaceProfile, ...]:
    """Return passive Visual Workspace profiles for one surface kind."""

    surface_value = str(surface_kind).strip()
    source_registry = registry or MULTIMODAL_VISUAL_WORKSPACE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.visual_workspace_profiles
        if profile.workspace_surface_kind == surface_value
    )


def multimodal_visual_workspace_profiles_for_interactive_canvas_profile(
    interactive_canvas_profile_id: str,
    registry: MultimodalVisualWorkspaceRegistry | None = None,
) -> tuple[VisualWorkspaceProfile, ...]:
    """Return Visual Workspace profiles referencing one canvas profile."""

    source_registry = registry or MULTIMODAL_VISUAL_WORKSPACE_REGISTRY
    source_profile_id = str(interactive_canvas_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.visual_workspace_profiles
        if source_profile_id in profile.source_interactive_canvas_profile_ids
    )


def _visual_workspace_profile(
    *,
    profile_id: str,
    profile_name: str,
    workspace_profile_kind: VisualWorkspaceProfileKind,
    workspace_surface_kind: VisualWorkspaceSurfaceKind,
    source_live_preview_profile_ids: tuple[str, ...],
    source_multi_preview_profile_ids: tuple[str, ...],
    source_interactive_canvas_profile_ids: tuple[str, ...],
    workspace_state_fields: tuple[str, ...],
    source_reference_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    visual_workspace_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> VisualWorkspaceProfile:
    return VisualWorkspaceProfile(
        profile_id=profile_id,
        profile_name=profile_name,
        workspace_profile_kind=workspace_profile_kind,
        workspace_surface_kind=workspace_surface_kind,
        source_live_preview_profile_ids=source_live_preview_profile_ids,
        source_multi_preview_profile_ids=source_multi_preview_profile_ids,
        source_interactive_canvas_profile_ids=source_interactive_canvas_profile_ids,
        workspace_state_fields=workspace_state_fields,
        source_reference_ids=source_reference_ids,
        route_applicability=route_applicability,
        visual_workspace_surfaces=visual_workspace_surfaces,
        advisory_outputs=advisory_outputs,
        source_registries=_VISUAL_WORKSPACE_SOURCE_REGISTRIES,
        observability_surfaces=_VISUAL_WORKSPACE_OBSERVABILITY_SURFACES,
    )


MULTIMODAL_VISUAL_WORKSPACE_PROFILES = (
    _visual_workspace_profile(
        profile_id="shell_visual_workspace",
        profile_name="Shell Visual Workspace",
        workspace_profile_kind="workspace_shell",
        workspace_surface_kind="shell",
        source_live_preview_profile_ids=MULTIMODAL_LIVE_PREVIEW_REGISTRY.profile_ids,
        source_multi_preview_profile_ids=MULTIMODAL_MULTI_PREVIEW_REGISTRY.profile_ids,
        source_interactive_canvas_profile_ids=(
            MULTIMODAL_INTERACTIVE_CANVAS_REGISTRY.profile_ids
        ),
        workspace_state_fields=(
            "session",
            "currentRun",
            "selection",
            "panels",
            "readiness",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_LIVE_PREVIEW_REGISTRY",
            "multimodal_studio.MULTIMODAL_MULTI_PREVIEW_REGISTRY",
            "multimodal_studio.MULTIMODAL_INTERACTIVE_CANVAS_REGISTRY",
            "clients.nextjs.workstation_state.buildWorkstationState",
            "clients.nextjs.workstation_shell.WorkstationShell",
            "clients.nextjs.workspace_persistence.createWorkspaceSessionRecord",
        ),
        route_applicability=tuple(RouteName),
        visual_workspace_surfaces=(
            "visual_workspace_shell",
            "workspace_dashboard_surface",
            "visual_context_surface",
            "workspace_boundary_panel",
        ),
        advisory_outputs=(
            "workspace_shell_inventory",
            "manual_workspace_state_review_hint",
            "no_workspace_state_mutation_notice",
        ),
    ),
    _visual_workspace_profile(
        profile_id="artifact_selection_visual_workspace",
        profile_name="Artifact Selection Visual Workspace",
        workspace_profile_kind="artifact_workspace",
        workspace_surface_kind="artifact_selection",
        source_live_preview_profile_ids=(
            "browser_sandbox_live_preview",
            "media_asset_live_preview",
            "structured_panel_live_preview",
        ),
        source_multi_preview_profile_ids=MULTIMODAL_MULTI_PREVIEW_REGISTRY.profile_ids,
        source_interactive_canvas_profile_ids=(
            "canvas_2d_interactive_canvas",
            "webgl_interactive_canvas",
        ),
        workspace_state_fields=(
            "activeArtifactId",
            "activeArtifact",
            "previewArtifactId",
            "previewArtifact",
        ),
        source_reference_ids=(
            "clients.nextjs.assistant_client.AssistantWorkspaceSnapshot",
            "clients.nextjs.workstation_state.buildWorkstationState",
            "clients.nextjs.workstation_shell.WorkstationShell",
        ),
        route_applicability=tuple(RouteName),
        visual_workspace_surfaces=(
            "visual_workspace_shell",
            "artifact_selection_surface",
            "workspace_boundary_panel",
        ),
        advisory_outputs=(
            "artifact_selection_inventory",
            "manual_artifact_selection_review_hint",
            "no_artifact_mutation_notice",
        ),
    ),
    _visual_workspace_profile(
        profile_id="preview_visual_workspace",
        profile_name="Preview Visual Workspace",
        workspace_profile_kind="preview_workspace",
        workspace_surface_kind="preview",
        source_live_preview_profile_ids=MULTIMODAL_LIVE_PREVIEW_REGISTRY.profile_ids,
        source_multi_preview_profile_ids=MULTIMODAL_MULTI_PREVIEW_REGISTRY.profile_ids,
        source_interactive_canvas_profile_ids=(
            MULTIMODAL_INTERACTIVE_CANVAS_REGISTRY.profile_ids
        ),
        workspace_state_fields=(
            "previewOpen",
            "previewFullscreen",
            "previewArtifact",
            "activeInspectorTab",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_LIVE_PREVIEW_REGISTRY",
            "multimodal_studio.MULTIMODAL_MULTI_PREVIEW_REGISTRY",
            "multimodal_studio.MULTIMODAL_INTERACTIVE_CANVAS_REGISTRY",
            "clients.nextjs.assistant_client.AssistantWorkspaceSnapshot",
            "clients.nextjs.workstation_shell.WorkstationShell",
        ),
        route_applicability=tuple(RouteName),
        visual_workspace_surfaces=(
            "preview_workspace_surface",
            "visual_workspace_shell",
            "workspace_boundary_panel",
        ),
        advisory_outputs=(
            "preview_workspace_inventory",
            "manual_preview_workspace_review_hint",
            "no_rendering_execution_notice",
        ),
    ),
    _visual_workspace_profile(
        profile_id="inspector_visual_workspace",
        profile_name="Inspector Visual Workspace",
        workspace_profile_kind="inspector_workspace",
        workspace_surface_kind="inspector",
        source_live_preview_profile_ids=("runtime_status_live_preview",),
        source_multi_preview_profile_ids=("recommended_candidate_multi_preview",),
        source_interactive_canvas_profile_ids=(
            "input_boundary_interactive_canvas",
            "runtime_status_interactive_canvas",
        ),
        workspace_state_fields=(
            "activeInspectorTab",
            "metadata",
            "readiness",
            "status",
            "dashboard",
        ),
        source_reference_ids=(
            "clients.nextjs.workstation_state.buildWorkstationState",
            "clients.nextjs.workstation_dashboard.buildWorkstationDashboardModel",
            "clients.nextjs.workstation_shell.WorkstationShell",
            "clients.nextjs.assistant_client.AssistantWorkspaceSnapshot",
        ),
        route_applicability=tuple(RouteName),
        visual_workspace_surfaces=(
            "inspector_workspace_surface",
            "workspace_dashboard_surface",
            "visual_context_surface",
            "workspace_boundary_panel",
        ),
        advisory_outputs=(
            "inspector_workspace_inventory",
            "manual_inspector_review_hint",
            "no_provider_or_model_routing_notice",
        ),
    ),
)

MULTIMODAL_VISUAL_WORKSPACE_REGISTRY = MultimodalVisualWorkspaceRegistry(
    visual_workspace_profiles=MULTIMODAL_VISUAL_WORKSPACE_PROFILES,
    profile_ids=tuple(
        profile.profile_id for profile in MULTIMODAL_VISUAL_WORKSPACE_PROFILES
    ),
    workspace_profile_kinds=tuple(
        profile.workspace_profile_kind
        for profile in MULTIMODAL_VISUAL_WORKSPACE_PROFILES
    ),
    workspace_surface_kinds=tuple(
        profile.workspace_surface_kind
        for profile in MULTIMODAL_VISUAL_WORKSPACE_PROFILES
    ),
    live_preview_profile_ids=MULTIMODAL_LIVE_PREVIEW_REGISTRY.profile_ids,
    multi_preview_profile_ids=MULTIMODAL_MULTI_PREVIEW_REGISTRY.profile_ids,
    interactive_canvas_profile_ids=MULTIMODAL_INTERACTIVE_CANVAS_REGISTRY.profile_ids,
    route_names=tuple(RouteName),
    profile_count=len(MULTIMODAL_VISUAL_WORKSPACE_PROFILES),
    source_registries=_VISUAL_WORKSPACE_SOURCE_REGISTRIES,
    source_reference_ids=_VISUAL_WORKSPACE_SOURCE_REFERENCES,
    visual_workspace_surface_refs=_VISUAL_WORKSPACE_SURFACES,
    observability_surfaces=_VISUAL_WORKSPACE_OBSERVABILITY_SURFACES,
)


class RuntimeCollaborationProfile(BaseModel):
    """Inspectable metadata for one passive Runtime Collaboration surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1, max_length=120)
    profile_name: str = Field(min_length=1, max_length=140)
    collaboration_profile_kind: RuntimeCollaborationProfileKind
    collaboration_surface_kind: RuntimeCollaborationSurfaceKind
    source_visual_workspace_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    runtime_context_fields: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_reference_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    runtime_collaboration_surfaces: tuple[str, ...] = Field(
        min_length=1,
        max_length=7,
    )
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=RUNTIME_COLLABORATION_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_RUNTIME_COLLABORATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    real_time_networking_implemented: Literal[False] = False
    external_peer_synchronization_implemented: Literal[False] = False
    persistent_collaboration_storage_implemented: Literal[False] = False
    runtime_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["multimodal_runtime_collaboration_profile.v1"] = (
        RUNTIME_COLLABORATION_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class MultimodalRuntimeCollaborationRegistry(BaseModel):
    """Stable passive registry for V4.5 Runtime Collaboration metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["multimodal_runtime_collaboration_registry"] = (
        "multimodal_runtime_collaboration_registry"
    )
    serialization_version: Literal["multimodal_runtime_collaboration_registry.v1"] = (
        RUNTIME_COLLABORATION_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RUNTIME_COLLABORATION_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    runtime_collaboration_profiles: tuple[RuntimeCollaborationProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    collaboration_profile_kinds: tuple[RuntimeCollaborationProfileKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    collaboration_surface_kinds: tuple[RuntimeCollaborationSurfaceKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    visual_workspace_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    source_reference_ids: tuple[str, ...] = Field(min_length=8, max_length=8)
    runtime_collaboration_surface_refs: tuple[str, ...] = Field(
        min_length=7,
        max_length=7,
    )
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_RUNTIME_COLLABORATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    real_time_networking_implemented: Literal[False] = False
    external_peer_synchronization_implemented: Literal[False] = False
    persistent_collaboration_storage_implemented: Literal[False] = False
    runtime_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.profile_id
            for profile in self.runtime_collaboration_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("profile_ids must be unique")
        if self.profile_ids != derived_profile_ids:
            raise ValueError(
                "profile_ids must match runtime_collaboration_profiles"
            )
        if self.profile_count != len(self.runtime_collaboration_profiles):
            raise ValueError(
                "profile_count must match runtime_collaboration_profiles"
            )
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if (
            self.visual_workspace_profile_ids
            != MULTIMODAL_VISUAL_WORKSPACE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "visual_workspace_profile_ids must match Visual Workspace registry"
            )

        derived_profile_kinds = _ordered_unique(
            profile.collaboration_profile_kind
            for profile in self.runtime_collaboration_profiles
        )
        if self.collaboration_profile_kinds != derived_profile_kinds:
            raise ValueError(
                "collaboration_profile_kinds must match runtime profiles"
            )

        derived_surface_kinds = _ordered_unique(
            profile.collaboration_surface_kind
            for profile in self.runtime_collaboration_profiles
        )
        if self.collaboration_surface_kinds != derived_surface_kinds:
            raise ValueError(
                "collaboration_surface_kinds must match runtime profiles"
            )

        profile_source_references = {
            source_reference
            for profile in self.runtime_collaboration_profiles
            for source_reference in profile.source_reference_ids
        }
        if set(self.source_reference_ids) != profile_source_references:
            raise ValueError(
                "source_reference_ids must match profile source references"
            )

        known_routes = set(self.route_names)
        known_workspace_profiles = set(self.visual_workspace_profile_ids)
        known_surfaces = set(self.runtime_collaboration_surface_refs)
        known_source_references = set(self.source_reference_ids)
        for profile in self.runtime_collaboration_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must use known routes")
            if not set(profile.source_visual_workspace_profile_ids).issubset(
                known_workspace_profiles
            ):
                raise ValueError(
                    "source_visual_workspace_profile_ids must be known profiles"
                )
            if not set(profile.runtime_collaboration_surfaces).issubset(
                known_surfaces
            ):
                raise ValueError(
                    "runtime_collaboration_surfaces must be known surfaces"
                )
            if not set(profile.source_reference_ids).issubset(
                known_source_references
            ):
                raise ValueError(
                    "source_reference_ids must be known registry references"
                )
        return self


def multimodal_runtime_collaboration_registry() -> (
    MultimodalRuntimeCollaborationRegistry
):
    """Return passive V4.5 Runtime Collaboration metadata."""

    return MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY


def multimodal_runtime_collaboration_profile_by_id(
    profile_id: str,
    registry: MultimodalRuntimeCollaborationRegistry | None = None,
) -> RuntimeCollaborationProfile | None:
    """Return one Runtime Collaboration profile without synchronization."""

    source_registry = registry or MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY
    normalized_profile_id = str(profile_id).strip()
    for profile in source_registry.runtime_collaboration_profiles:
        if profile.profile_id == normalized_profile_id:
            return profile
    return None


def multimodal_runtime_collaboration_profiles_for_route(
    route: RouteName | str,
    registry: MultimodalRuntimeCollaborationRegistry | None = None,
) -> tuple[RuntimeCollaborationProfile, ...]:
    """Return passive Runtime Collaboration profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY
    return tuple(
        profile
        for profile in source_registry.runtime_collaboration_profiles
        if route_name in profile.route_applicability
    )


def multimodal_runtime_collaboration_profiles_for_surface_kind(
    surface_kind: RuntimeCollaborationSurfaceKind | str,
    registry: MultimodalRuntimeCollaborationRegistry | None = None,
) -> tuple[RuntimeCollaborationProfile, ...]:
    """Return Runtime Collaboration profiles for one passive surface kind."""

    surface_value = str(surface_kind).strip()
    source_registry = registry or MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY
    return tuple(
        profile
        for profile in source_registry.runtime_collaboration_profiles
        if profile.collaboration_surface_kind == surface_value
    )


def multimodal_runtime_collaboration_profiles_for_visual_workspace_profile(
    visual_workspace_profile_id: str,
    registry: MultimodalRuntimeCollaborationRegistry | None = None,
) -> tuple[RuntimeCollaborationProfile, ...]:
    """Return Runtime Collaboration profiles referencing one workspace profile."""

    source_registry = registry or MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY
    source_profile_id = str(visual_workspace_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.runtime_collaboration_profiles
        if source_profile_id in profile.source_visual_workspace_profile_ids
    )


def _runtime_collaboration_profile(
    *,
    profile_id: str,
    profile_name: str,
    collaboration_profile_kind: RuntimeCollaborationProfileKind,
    collaboration_surface_kind: RuntimeCollaborationSurfaceKind,
    source_visual_workspace_profile_ids: tuple[str, ...],
    runtime_context_fields: tuple[str, ...],
    source_reference_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    runtime_collaboration_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> RuntimeCollaborationProfile:
    return RuntimeCollaborationProfile(
        profile_id=profile_id,
        profile_name=profile_name,
        collaboration_profile_kind=collaboration_profile_kind,
        collaboration_surface_kind=collaboration_surface_kind,
        source_visual_workspace_profile_ids=source_visual_workspace_profile_ids,
        runtime_context_fields=runtime_context_fields,
        source_reference_ids=source_reference_ids,
        route_applicability=route_applicability,
        runtime_collaboration_surfaces=runtime_collaboration_surfaces,
        advisory_outputs=advisory_outputs,
        source_registries=_RUNTIME_COLLABORATION_SOURCE_REGISTRIES,
        observability_surfaces=_RUNTIME_COLLABORATION_OBSERVABILITY_SURFACES,
    )


MULTIMODAL_RUNTIME_COLLABORATION_PROFILES = (
    _runtime_collaboration_profile(
        profile_id="trace_runtime_collaboration",
        profile_name="Trace Runtime Collaboration",
        collaboration_profile_kind="runtime_trace_collaboration",
        collaboration_surface_kind="trace",
        source_visual_workspace_profile_ids=(
            "shell_visual_workspace",
            "inspector_visual_workspace",
        ),
        runtime_context_fields=(
            "traceEventCount",
            "transitionCount",
            "retryCount",
            "currentNode",
            "currentStep",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_VISUAL_WORKSPACE_REGISTRY",
            "clients.nextjs.workflow_runtime.buildWorkflowRuntimeModel",
            "clients.nextjs.workstation_shell.applyStreamEventToWorkspace",
        ),
        route_applicability=tuple(RouteName),
        runtime_collaboration_surfaces=(
            "runtime_collaboration_panel",
            "runtime_trace_surface",
            "runtime_health_surface",
            "runtime_collaboration_boundary_panel",
        ),
        advisory_outputs=(
            "runtime_trace_collaboration_inventory",
            "manual_trace_review_hint",
            "no_workflow_control_notice",
        ),
    ),
    _runtime_collaboration_profile(
        profile_id="console_runtime_collaboration",
        profile_name="Console Runtime Collaboration",
        collaboration_profile_kind="runtime_console_collaboration",
        collaboration_surface_kind="console",
        source_visual_workspace_profile_ids=(
            "preview_visual_workspace",
            "inspector_visual_workspace",
        ),
        runtime_context_fields=(
            "runtimeId",
            "source",
            "status",
            "metrics",
            "diagnostics",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_VISUAL_WORKSPACE_REGISTRY",
            "clients.nextjs.runtime_console.buildRuntimeConsoleModel",
            "clients.nextjs.workflow_runtime.buildWorkflowRuntimeModel",
        ),
        route_applicability=tuple(RouteName),
        runtime_collaboration_surfaces=(
            "runtime_collaboration_panel",
            "runtime_console_surface",
            "runtime_health_surface",
            "runtime_collaboration_boundary_panel",
        ),
        advisory_outputs=(
            "runtime_console_collaboration_inventory",
            "manual_console_review_hint",
            "no_runtime_execution_notice",
        ),
    ),
    _runtime_collaboration_profile(
        profile_id="stream_event_runtime_collaboration",
        profile_name="Stream Event Runtime Collaboration",
        collaboration_profile_kind="stream_event_collaboration",
        collaboration_surface_kind="stream",
        source_visual_workspace_profile_ids=(
            "shell_visual_workspace",
            "artifact_selection_visual_workspace",
            "preview_visual_workspace",
        ),
        runtime_context_fields=(
            "event_type",
            "sequence",
            "payload",
            "workflow",
            "providerTelemetry",
        ),
        source_reference_ids=(
            "clients.nextjs.assistant_stream.streamAssistantEvents",
            "clients.nextjs.workstation_shell.applyStreamEventToWorkspace",
            "clients.nextjs.provider_telemetry.buildProviderTelemetryModel",
        ),
        route_applicability=tuple(RouteName),
        runtime_collaboration_surfaces=(
            "runtime_collaboration_panel",
            "stream_event_surface",
            "runtime_trace_surface",
            "runtime_collaboration_boundary_panel",
        ),
        advisory_outputs=(
            "stream_event_collaboration_inventory",
            "manual_stream_event_review_hint",
            "no_real_time_networking_notice",
        ),
    ),
    _runtime_collaboration_profile(
        profile_id="operator_context_runtime_collaboration",
        profile_name="Operator Context Runtime Collaboration",
        collaboration_profile_kind="operator_context_collaboration",
        collaboration_surface_kind="operator_context",
        source_visual_workspace_profile_ids=(
            "shell_visual_workspace",
            "inspector_visual_workspace",
        ),
        runtime_context_fields=(
            "sessionIntelligence",
            "streamedMetadata",
            "operatorReview",
            "readiness",
        ),
        source_reference_ids=(
            "clients.nextjs.session_intelligence.buildSessionIntelligenceModel",
            "clients.nextjs.session_intelligence.readSessionIntelligenceMetadata",
            "clients.nextjs.provider_telemetry.buildProviderTelemetryModel",
        ),
        route_applicability=tuple(RouteName),
        runtime_collaboration_surfaces=(
            "runtime_collaboration_panel",
            "operator_context_surface",
            "runtime_health_surface",
            "runtime_collaboration_boundary_panel",
        ),
        advisory_outputs=(
            "operator_context_collaboration_inventory",
            "manual_operator_context_review_hint",
            "no_human_input_request_notice",
        ),
    ),
)

MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY = (
    MultimodalRuntimeCollaborationRegistry(
        runtime_collaboration_profiles=MULTIMODAL_RUNTIME_COLLABORATION_PROFILES,
        profile_ids=tuple(
            profile.profile_id
            for profile in MULTIMODAL_RUNTIME_COLLABORATION_PROFILES
        ),
        collaboration_profile_kinds=tuple(
            profile.collaboration_profile_kind
            for profile in MULTIMODAL_RUNTIME_COLLABORATION_PROFILES
        ),
        collaboration_surface_kinds=tuple(
            profile.collaboration_surface_kind
            for profile in MULTIMODAL_RUNTIME_COLLABORATION_PROFILES
        ),
        visual_workspace_profile_ids=MULTIMODAL_VISUAL_WORKSPACE_REGISTRY.profile_ids,
        route_names=tuple(RouteName),
        profile_count=len(MULTIMODAL_RUNTIME_COLLABORATION_PROFILES),
        source_registries=_RUNTIME_COLLABORATION_SOURCE_REGISTRIES,
        source_reference_ids=_RUNTIME_COLLABORATION_SOURCE_REFERENCES,
        runtime_collaboration_surface_refs=_RUNTIME_COLLABORATION_SURFACES,
        observability_surfaces=_RUNTIME_COLLABORATION_OBSERVABILITY_SURFACES,
    )
)


class ArtifactCollaborationProfile(BaseModel):
    """Inspectable metadata for one passive Artifact Collaboration surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1, max_length=120)
    profile_name: str = Field(min_length=1, max_length=140)
    artifact_profile_kind: ArtifactCollaborationProfileKind
    artifact_surface_kind: ArtifactCollaborationSurfaceKind
    source_workspace_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_runtime_collaboration_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    artifact_context_fields: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_reference_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    artifact_collaboration_surfaces: tuple[str, ...] = Field(
        min_length=1,
        max_length=7,
    )
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=ARTIFACT_COLLABORATION_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_ARTIFACT_COLLABORATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_collaboration_storage_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    serialization_version: Literal["multimodal_artifact_collaboration_profile.v1"] = (
        ARTIFACT_COLLABORATION_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class MultimodalArtifactCollaborationRegistry(BaseModel):
    """Stable passive registry for V4.5 Artifact Collaboration metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["multimodal_artifact_collaboration_registry"] = (
        "multimodal_artifact_collaboration_registry"
    )
    serialization_version: Literal["multimodal_artifact_collaboration_registry.v1"] = (
        ARTIFACT_COLLABORATION_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ARTIFACT_COLLABORATION_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    artifact_collaboration_profiles: tuple[ArtifactCollaborationProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    artifact_profile_kinds: tuple[ArtifactCollaborationProfileKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    artifact_surface_kinds: tuple[ArtifactCollaborationSurfaceKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    visual_workspace_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    runtime_collaboration_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    source_reference_ids: tuple[str, ...] = Field(min_length=8, max_length=8)
    artifact_collaboration_surface_refs: tuple[str, ...] = Field(
        min_length=7,
        max_length=7,
    )
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_ARTIFACT_COLLABORATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_collaboration_storage_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.profile_id
            for profile in self.artifact_collaboration_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("profile_ids must be unique")
        if self.profile_ids != derived_profile_ids:
            raise ValueError(
                "profile_ids must match artifact_collaboration_profiles"
            )
        if self.profile_count != len(self.artifact_collaboration_profiles):
            raise ValueError(
                "profile_count must match artifact_collaboration_profiles"
            )
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if (
            self.visual_workspace_profile_ids
            != MULTIMODAL_VISUAL_WORKSPACE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "visual_workspace_profile_ids must match Visual Workspace registry"
            )
        if (
            self.runtime_collaboration_profile_ids
            != MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY.profile_ids
        ):
            raise ValueError(
                "runtime_collaboration_profile_ids must match Runtime Collaboration registry"
            )

        if self.artifact_profile_kinds != _ordered_unique(
            profile.artifact_profile_kind
            for profile in self.artifact_collaboration_profiles
        ):
            raise ValueError(
                "artifact_profile_kinds must match collaboration profiles"
            )
        if self.artifact_surface_kinds != _ordered_unique(
            profile.artifact_surface_kind
            for profile in self.artifact_collaboration_profiles
        ):
            raise ValueError(
                "artifact_surface_kinds must match collaboration profiles"
            )

        profile_source_references = {
            source_reference
            for profile in self.artifact_collaboration_profiles
            for source_reference in profile.source_reference_ids
        }
        if set(self.source_reference_ids) != profile_source_references:
            raise ValueError(
                "source_reference_ids must match profile source references"
            )

        known_routes = set(self.route_names)
        known_workspace_profiles = set(self.visual_workspace_profile_ids)
        known_runtime_profiles = set(self.runtime_collaboration_profile_ids)
        known_surfaces = set(self.artifact_collaboration_surface_refs)
        known_source_references = set(self.source_reference_ids)
        for profile in self.artifact_collaboration_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must use known routes")
            if not set(profile.source_workspace_profile_ids).issubset(
                known_workspace_profiles
            ):
                raise ValueError("source_workspace_profile_ids must be known profiles")
            if not set(profile.source_runtime_collaboration_profile_ids).issubset(
                known_runtime_profiles
            ):
                raise ValueError(
                    "source_runtime_collaboration_profile_ids must be known profiles"
                )
            if not set(profile.artifact_collaboration_surfaces).issubset(
                known_surfaces
            ):
                raise ValueError(
                    "artifact_collaboration_surfaces must be known surfaces"
                )
            if not set(profile.source_reference_ids).issubset(
                known_source_references
            ):
                raise ValueError(
                    "source_reference_ids must be known registry references"
                )
        return self


def multimodal_artifact_collaboration_registry() -> (
    MultimodalArtifactCollaborationRegistry
):
    """Return passive V4.5 Artifact Collaboration metadata."""

    return MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY


def multimodal_artifact_collaboration_profile_by_id(
    profile_id: str,
    registry: MultimodalArtifactCollaborationRegistry | None = None,
) -> ArtifactCollaborationProfile | None:
    """Return one Artifact Collaboration profile without mutating artifacts."""

    source_registry = registry or MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY
    normalized_profile_id = str(profile_id).strip()
    for profile in source_registry.artifact_collaboration_profiles:
        if profile.profile_id == normalized_profile_id:
            return profile
    return None


def multimodal_artifact_collaboration_profiles_for_route(
    route: RouteName | str,
    registry: MultimodalArtifactCollaborationRegistry | None = None,
) -> tuple[ArtifactCollaborationProfile, ...]:
    """Return passive Artifact Collaboration profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY
    return tuple(
        profile
        for profile in source_registry.artifact_collaboration_profiles
        if route_name in profile.route_applicability
    )


def multimodal_artifact_collaboration_profiles_for_surface_kind(
    surface_kind: ArtifactCollaborationSurfaceKind | str,
    registry: MultimodalArtifactCollaborationRegistry | None = None,
) -> tuple[ArtifactCollaborationProfile, ...]:
    """Return Artifact Collaboration profiles for one passive surface kind."""

    surface_value = str(surface_kind).strip()
    source_registry = registry or MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY
    return tuple(
        profile
        for profile in source_registry.artifact_collaboration_profiles
        if profile.artifact_surface_kind == surface_value
    )


def multimodal_artifact_collaboration_profiles_for_workspace_profile(
    workspace_profile_id: str,
    registry: MultimodalArtifactCollaborationRegistry | None = None,
) -> tuple[ArtifactCollaborationProfile, ...]:
    """Return Artifact Collaboration profiles referencing one workspace profile."""

    source_registry = registry or MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY
    source_profile_id = str(workspace_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.artifact_collaboration_profiles
        if source_profile_id in profile.source_workspace_profile_ids
    )


def _artifact_collaboration_profile(
    *,
    profile_id: str,
    profile_name: str,
    artifact_profile_kind: ArtifactCollaborationProfileKind,
    artifact_surface_kind: ArtifactCollaborationSurfaceKind,
    source_workspace_profile_ids: tuple[str, ...],
    source_runtime_collaboration_profile_ids: tuple[str, ...],
    artifact_context_fields: tuple[str, ...],
    source_reference_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    artifact_collaboration_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> ArtifactCollaborationProfile:
    return ArtifactCollaborationProfile(
        profile_id=profile_id,
        profile_name=profile_name,
        artifact_profile_kind=artifact_profile_kind,
        artifact_surface_kind=artifact_surface_kind,
        source_workspace_profile_ids=source_workspace_profile_ids,
        source_runtime_collaboration_profile_ids=(
            source_runtime_collaboration_profile_ids
        ),
        artifact_context_fields=artifact_context_fields,
        source_reference_ids=source_reference_ids,
        route_applicability=route_applicability,
        artifact_collaboration_surfaces=artifact_collaboration_surfaces,
        advisory_outputs=advisory_outputs,
        source_registries=_ARTIFACT_COLLABORATION_SOURCE_REGISTRIES,
        observability_surfaces=_ARTIFACT_COLLABORATION_OBSERVABILITY_SURFACES,
    )


MULTIMODAL_ARTIFACT_COLLABORATION_PROFILES = (
    _artifact_collaboration_profile(
        profile_id="selection_artifact_collaboration",
        profile_name="Selection Artifact Collaboration",
        artifact_profile_kind="artifact_selection_collaboration",
        artifact_surface_kind="selection",
        source_workspace_profile_ids=(
            "shell_visual_workspace",
            "artifact_selection_visual_workspace",
        ),
        source_runtime_collaboration_profile_ids=(
            "stream_event_runtime_collaboration",
            "operator_context_runtime_collaboration",
        ),
        artifact_context_fields=(
            "activeArtifactId",
            "previewArtifactId",
            "artifactActions",
            "selectionState",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_VISUAL_WORKSPACE_REGISTRY",
            "multimodal_studio.MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY",
            "clients.nextjs.workstation_shell.handleArtifactRefine",
        ),
        route_applicability=tuple(RouteName),
        artifact_collaboration_surfaces=(
            "artifact_collaboration_panel",
            "artifact_selection_surface",
            "artifact_action_feedback_surface",
            "artifact_collaboration_boundary_panel",
        ),
        advisory_outputs=(
            "artifact_selection_collaboration_inventory",
            "manual_artifact_selection_review_hint",
            "no_artifact_mutation_notice",
        ),
    ),
    _artifact_collaboration_profile(
        profile_id="comparison_artifact_collaboration",
        profile_name="Comparison Artifact Collaboration",
        artifact_profile_kind="artifact_comparison_collaboration",
        artifact_surface_kind="comparison",
        source_workspace_profile_ids=(
            "artifact_selection_visual_workspace",
            "preview_visual_workspace",
        ),
        source_runtime_collaboration_profile_ids=(
            "trace_runtime_collaboration",
            "stream_event_runtime_collaboration",
        ),
        artifact_context_fields=(
            "comparisonRows",
            "recommendedRow",
            "runtimeSupport",
            "scoreLabel",
        ),
        source_reference_ids=(
            "clients.nextjs.artifact_comparison.buildArtifactComparisonModel",
            "clients.nextjs.multi_preview_comparison.buildMultiPreviewComparisonModel",
            "multimodal_studio.MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY",
        ),
        route_applicability=tuple(RouteName),
        artifact_collaboration_surfaces=(
            "artifact_collaboration_panel",
            "artifact_comparison_surface",
            "artifact_collaboration_boundary_panel",
        ),
        advisory_outputs=(
            "artifact_comparison_collaboration_inventory",
            "manual_artifact_comparison_review_hint",
            "no_generated_output_mutation_notice",
        ),
    ),
    _artifact_collaboration_profile(
        profile_id="inspection_artifact_collaboration",
        profile_name="Inspection Artifact Collaboration",
        artifact_profile_kind="artifact_inspection_collaboration",
        artifact_surface_kind="inspection",
        source_workspace_profile_ids=(
            "artifact_selection_visual_workspace",
            "inspector_visual_workspace",
        ),
        source_runtime_collaboration_profile_ids=(
            "console_runtime_collaboration",
            "operator_context_runtime_collaboration",
        ),
        artifact_context_fields=(
            "artifactDocument",
            "highlightedLines",
            "mimeType",
            "lineCount",
        ),
        source_reference_ids=(
            "clients.nextjs.artifact_inspector.buildArtifactDocument",
            "clients.nextjs.artifact_inspector.highlightArtifactDocument",
            "clients.nextjs.workstation_shell.handleArtifactRefine",
        ),
        route_applicability=tuple(RouteName),
        artifact_collaboration_surfaces=(
            "artifact_collaboration_panel",
            "artifact_inspection_surface",
            "artifact_action_feedback_surface",
            "artifact_collaboration_boundary_panel",
        ),
        advisory_outputs=(
            "artifact_inspection_collaboration_inventory",
            "manual_artifact_inspection_review_hint",
            "no_persistent_collaboration_storage_notice",
        ),
    ),
    _artifact_collaboration_profile(
        profile_id="refinement_artifact_collaboration",
        profile_name="Refinement Artifact Collaboration",
        artifact_profile_kind="artifact_refinement_collaboration",
        artifact_surface_kind="refinement",
        source_workspace_profile_ids=(
            "artifact_selection_visual_workspace",
            "inspector_visual_workspace",
        ),
        source_runtime_collaboration_profile_ids=(
            "operator_context_runtime_collaboration",
        ),
        artifact_context_fields=(
            "artifactRefinement",
            "refinementInstruction",
            "refinementPreview",
            "pendingRefinement",
        ),
        source_reference_ids=(
            "clients.nextjs.artifact_refinement.enrichArtifactRefinementRequest",
            "clients.nextjs.workstation_shell.handleArtifactRefine",
            "multimodal_studio.MULTIMODAL_VISUAL_WORKSPACE_REGISTRY",
        ),
        route_applicability=tuple(RouteName),
        artifact_collaboration_surfaces=(
            "artifact_collaboration_panel",
            "artifact_refinement_surface",
            "artifact_action_feedback_surface",
            "artifact_collaboration_boundary_panel",
        ),
        advisory_outputs=(
            "artifact_refinement_collaboration_inventory",
            "manual_refinement_review_hint",
            "no_retry_triggering_notice",
        ),
    ),
)

MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY = (
    MultimodalArtifactCollaborationRegistry(
        artifact_collaboration_profiles=MULTIMODAL_ARTIFACT_COLLABORATION_PROFILES,
        profile_ids=tuple(
            profile.profile_id
            for profile in MULTIMODAL_ARTIFACT_COLLABORATION_PROFILES
        ),
        artifact_profile_kinds=tuple(
            profile.artifact_profile_kind
            for profile in MULTIMODAL_ARTIFACT_COLLABORATION_PROFILES
        ),
        artifact_surface_kinds=tuple(
            profile.artifact_surface_kind
            for profile in MULTIMODAL_ARTIFACT_COLLABORATION_PROFILES
        ),
        visual_workspace_profile_ids=MULTIMODAL_VISUAL_WORKSPACE_REGISTRY.profile_ids,
        runtime_collaboration_profile_ids=(
            MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY.profile_ids
        ),
        route_names=tuple(RouteName),
        profile_count=len(MULTIMODAL_ARTIFACT_COLLABORATION_PROFILES),
        source_registries=_ARTIFACT_COLLABORATION_SOURCE_REGISTRIES,
        source_reference_ids=_ARTIFACT_COLLABORATION_SOURCE_REFERENCES,
        artifact_collaboration_surface_refs=_ARTIFACT_COLLABORATION_SURFACES,
        observability_surfaces=_ARTIFACT_COLLABORATION_OBSERVABILITY_SURFACES,
    )
)


class ArtifactProvenanceProfile(BaseModel):
    """Inspectable metadata for one passive Artifact Provenance surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1, max_length=120)
    profile_name: str = Field(min_length=1, max_length=140)
    provenance_profile_kind: ArtifactProvenanceProfileKind
    provenance_surface_kind: ArtifactProvenanceSurfaceKind
    source_artifact_collaboration_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_runtime_collaboration_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    provenance_context_fields: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_reference_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    artifact_provenance_surfaces: tuple[str, ...] = Field(
        min_length=1,
        max_length=7,
    )
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=ARTIFACT_PROVENANCE_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_ARTIFACT_PROVENANCE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    provenance_recording_implemented: Literal[False] = False
    persistent_provenance_storage_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    serialization_version: Literal["multimodal_artifact_provenance_profile.v1"] = (
        ARTIFACT_PROVENANCE_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class MultimodalArtifactProvenanceRegistry(BaseModel):
    """Stable passive registry for V4.5 Artifact Provenance metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["multimodal_artifact_provenance_registry"] = (
        "multimodal_artifact_provenance_registry"
    )
    serialization_version: Literal["multimodal_artifact_provenance_registry.v1"] = (
        ARTIFACT_PROVENANCE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ARTIFACT_PROVENANCE_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    artifact_provenance_profiles: tuple[ArtifactProvenanceProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    provenance_profile_kinds: tuple[ArtifactProvenanceProfileKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    provenance_surface_kinds: tuple[ArtifactProvenanceSurfaceKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    artifact_collaboration_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    runtime_collaboration_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    source_reference_ids: tuple[str, ...] = Field(min_length=8, max_length=8)
    artifact_provenance_surface_refs: tuple[str, ...] = Field(
        min_length=7,
        max_length=7,
    )
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_ARTIFACT_PROVENANCE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    provenance_recording_implemented: Literal[False] = False
    persistent_provenance_storage_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.profile_id for profile in self.artifact_provenance_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("profile_ids must be unique")
        if self.profile_ids != derived_profile_ids:
            raise ValueError("profile_ids must match artifact_provenance_profiles")
        if self.profile_count != len(self.artifact_provenance_profiles):
            raise ValueError("profile_count must match artifact_provenance_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if (
            self.artifact_collaboration_profile_ids
            != MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY.profile_ids
        ):
            raise ValueError(
                "artifact_collaboration_profile_ids must match Artifact Collaboration registry"
            )
        if (
            self.runtime_collaboration_profile_ids
            != MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY.profile_ids
        ):
            raise ValueError(
                "runtime_collaboration_profile_ids must match Runtime Collaboration registry"
            )
        if self.provenance_profile_kinds != _ordered_unique(
            profile.provenance_profile_kind
            for profile in self.artifact_provenance_profiles
        ):
            raise ValueError("provenance_profile_kinds must match profiles")
        if self.provenance_surface_kinds != _ordered_unique(
            profile.provenance_surface_kind
            for profile in self.artifact_provenance_profiles
        ):
            raise ValueError("provenance_surface_kinds must match profiles")

        profile_source_references = {
            source_reference
            for profile in self.artifact_provenance_profiles
            for source_reference in profile.source_reference_ids
        }
        if set(self.source_reference_ids) != profile_source_references:
            raise ValueError(
                "source_reference_ids must match profile source references"
            )

        known_routes = set(self.route_names)
        known_artifact_profiles = set(self.artifact_collaboration_profile_ids)
        known_runtime_profiles = set(self.runtime_collaboration_profile_ids)
        known_surfaces = set(self.artifact_provenance_surface_refs)
        known_source_references = set(self.source_reference_ids)
        for profile in self.artifact_provenance_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must use known routes")
            if not set(profile.source_artifact_collaboration_profile_ids).issubset(
                known_artifact_profiles
            ):
                raise ValueError(
                    "source_artifact_collaboration_profile_ids must be known profiles"
                )
            if not set(profile.source_runtime_collaboration_profile_ids).issubset(
                known_runtime_profiles
            ):
                raise ValueError(
                    "source_runtime_collaboration_profile_ids must be known profiles"
                )
            if not set(profile.artifact_provenance_surfaces).issubset(
                known_surfaces
            ):
                raise ValueError(
                    "artifact_provenance_surfaces must be known surfaces"
                )
            if not set(profile.source_reference_ids).issubset(
                known_source_references
            ):
                raise ValueError(
                    "source_reference_ids must be known registry references"
                )
        return self


def multimodal_artifact_provenance_registry() -> (
    MultimodalArtifactProvenanceRegistry
):
    """Return passive V4.5 Artifact Provenance metadata."""

    return MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY


def multimodal_artifact_provenance_profile_by_id(
    profile_id: str,
    registry: MultimodalArtifactProvenanceRegistry | None = None,
) -> ArtifactProvenanceProfile | None:
    """Return one Artifact Provenance profile without recording provenance."""

    source_registry = registry or MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY
    normalized_profile_id = str(profile_id).strip()
    for profile in source_registry.artifact_provenance_profiles:
        if profile.profile_id == normalized_profile_id:
            return profile
    return None


def multimodal_artifact_provenance_profiles_for_route(
    route: RouteName | str,
    registry: MultimodalArtifactProvenanceRegistry | None = None,
) -> tuple[ArtifactProvenanceProfile, ...]:
    """Return passive Artifact Provenance profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.artifact_provenance_profiles
        if route_name in profile.route_applicability
    )


def multimodal_artifact_provenance_profiles_for_surface_kind(
    surface_kind: ArtifactProvenanceSurfaceKind | str,
    registry: MultimodalArtifactProvenanceRegistry | None = None,
) -> tuple[ArtifactProvenanceProfile, ...]:
    """Return Artifact Provenance profiles for one passive surface kind."""

    surface_value = str(surface_kind).strip()
    source_registry = registry or MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.artifact_provenance_profiles
        if profile.provenance_surface_kind == surface_value
    )


def multimodal_artifact_provenance_profiles_for_artifact_collaboration_profile(
    artifact_collaboration_profile_id: str,
    registry: MultimodalArtifactProvenanceRegistry | None = None,
) -> tuple[ArtifactProvenanceProfile, ...]:
    """Return provenance profiles referencing one artifact collaboration profile."""

    source_registry = registry or MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY
    source_profile_id = str(artifact_collaboration_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.artifact_provenance_profiles
        if source_profile_id in profile.source_artifact_collaboration_profile_ids
    )


def _artifact_provenance_profile(
    *,
    profile_id: str,
    profile_name: str,
    provenance_profile_kind: ArtifactProvenanceProfileKind,
    provenance_surface_kind: ArtifactProvenanceSurfaceKind,
    source_artifact_collaboration_profile_ids: tuple[str, ...],
    source_runtime_collaboration_profile_ids: tuple[str, ...],
    provenance_context_fields: tuple[str, ...],
    source_reference_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    artifact_provenance_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> ArtifactProvenanceProfile:
    return ArtifactProvenanceProfile(
        profile_id=profile_id,
        profile_name=profile_name,
        provenance_profile_kind=provenance_profile_kind,
        provenance_surface_kind=provenance_surface_kind,
        source_artifact_collaboration_profile_ids=(
            source_artifact_collaboration_profile_ids
        ),
        source_runtime_collaboration_profile_ids=(
            source_runtime_collaboration_profile_ids
        ),
        provenance_context_fields=provenance_context_fields,
        source_reference_ids=source_reference_ids,
        route_applicability=route_applicability,
        artifact_provenance_surfaces=artifact_provenance_surfaces,
        advisory_outputs=advisory_outputs,
        source_registries=_ARTIFACT_PROVENANCE_SOURCE_REGISTRIES,
        observability_surfaces=_ARTIFACT_PROVENANCE_OBSERVABILITY_SURFACES,
    )


MULTIMODAL_ARTIFACT_PROVENANCE_PROFILES = (
    _artifact_provenance_profile(
        profile_id="evidence_artifact_provenance",
        profile_name="Evidence Artifact Provenance",
        provenance_profile_kind="source_evidence_provenance",
        provenance_surface_kind="evidence",
        source_artifact_collaboration_profile_ids=(
            "inspection_artifact_collaboration",
            "comparison_artifact_collaboration",
        ),
        source_runtime_collaboration_profile_ids=(
            "trace_runtime_collaboration",
            "stream_event_runtime_collaboration",
        ),
        provenance_context_fields=(
            "evidence_sources",
            "retrieval.sources",
            "reasoning_evidence",
            "sourceKeys",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY",
            "multimodal_studio.MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY",
            "clients.nextjs.provenance_engine.buildProvenanceEngineModel",
            "clients.nextjs.provenance_engine.ProvenanceSource",
        ),
        route_applicability=tuple(RouteName),
        artifact_provenance_surfaces=(
            "artifact_provenance_panel",
            "evidence_source_surface",
            "provenance_summary_surface",
            "artifact_provenance_boundary_panel",
        ),
        advisory_outputs=(
            "evidence_provenance_inventory",
            "manual_evidence_review_hint",
            "no_provenance_recording_notice",
        ),
    ),
    _artifact_provenance_profile(
        profile_id="payload_artifact_provenance",
        profile_name="Payload Artifact Provenance",
        provenance_profile_kind="artifact_payload_provenance",
        provenance_surface_kind="artifact_payload",
        source_artifact_collaboration_profile_ids=(
            "selection_artifact_collaboration",
            "inspection_artifact_collaboration",
        ),
        source_runtime_collaboration_profile_ids=("stream_event_runtime_collaboration",),
        provenance_context_fields=(
            "artifact_sources",
            "artifact.id",
            "artifact.title",
            "eventSequence",
        ),
        source_reference_ids=(
            "clients.nextjs.provenance_engine.buildProvenanceEngineModel",
            "clients.nextjs.provenance_engine.ProvenanceSource",
            "preview.contracts.PreviewProvenance",
        ),
        route_applicability=tuple(RouteName),
        artifact_provenance_surfaces=(
            "artifact_provenance_panel",
            "artifact_payload_source_surface",
            "provenance_summary_surface",
            "artifact_provenance_boundary_panel",
        ),
        advisory_outputs=(
            "artifact_payload_provenance_inventory",
            "manual_payload_review_hint",
            "no_artifact_mutation_notice",
        ),
    ),
    _artifact_provenance_profile(
        profile_id="evaluation_artifact_provenance",
        profile_name="Evaluation Artifact Provenance",
        provenance_profile_kind="evaluation_provenance",
        provenance_surface_kind="evaluation",
        source_artifact_collaboration_profile_ids=(
            "comparison_artifact_collaboration",
            "refinement_artifact_collaboration",
        ),
        source_runtime_collaboration_profile_ids=(
            "trace_runtime_collaboration",
            "operator_context_runtime_collaboration",
        ),
        provenance_context_fields=(
            "evaluation_sources",
            "final_payload",
            "eval_update",
            "totals",
        ),
        source_reference_ids=(
            "clients.nextjs.provenance_engine.buildProvenanceEngineModel",
            "clients.nextjs.v3_inspector_panels.buildProvenancePanel",
            "clients.nextjs.workflow_runtime.WorkflowRuntimeTraceEvent",
        ),
        route_applicability=tuple(RouteName),
        artifact_provenance_surfaces=(
            "artifact_provenance_panel",
            "evaluation_source_surface",
            "provenance_summary_surface",
            "artifact_provenance_boundary_panel",
        ),
        advisory_outputs=(
            "evaluation_provenance_inventory",
            "manual_evaluation_provenance_review_hint",
            "no_workflow_control_notice",
        ),
    ),
    _artifact_provenance_profile(
        profile_id="missing_source_artifact_provenance",
        profile_name="Missing Source Artifact Provenance",
        provenance_profile_kind="missing_source_provenance",
        provenance_surface_kind="missing_source",
        source_artifact_collaboration_profile_ids=(
            "inspection_artifact_collaboration",
            "refinement_artifact_collaboration",
        ),
        source_runtime_collaboration_profile_ids=(
            "operator_context_runtime_collaboration",
        ),
        provenance_context_fields=(
            "unsupported_or_missing_sources",
            "missingSourceCount",
            "unsupportedSourceCount",
            "readiness",
        ),
        source_reference_ids=(
            "clients.nextjs.provenance_engine.buildProvenanceEngineModel",
            "clients.nextjs.v3_inspector_panels.buildProvenancePanel",
            "clients.nextjs.workstation_state.buildWorkstationState",
        ),
        route_applicability=tuple(RouteName),
        artifact_provenance_surfaces=(
            "artifact_provenance_panel",
            "missing_source_surface",
            "provenance_summary_surface",
            "artifact_provenance_boundary_panel",
        ),
        advisory_outputs=(
            "missing_source_provenance_inventory",
            "manual_missing_source_review_hint",
            "no_persistent_provenance_storage_notice",
        ),
    ),
)

MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY = MultimodalArtifactProvenanceRegistry(
    artifact_provenance_profiles=MULTIMODAL_ARTIFACT_PROVENANCE_PROFILES,
    profile_ids=tuple(
        profile.profile_id for profile in MULTIMODAL_ARTIFACT_PROVENANCE_PROFILES
    ),
    provenance_profile_kinds=tuple(
        profile.provenance_profile_kind
        for profile in MULTIMODAL_ARTIFACT_PROVENANCE_PROFILES
    ),
    provenance_surface_kinds=tuple(
        profile.provenance_surface_kind
        for profile in MULTIMODAL_ARTIFACT_PROVENANCE_PROFILES
    ),
    artifact_collaboration_profile_ids=(
        MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY.profile_ids
    ),
    runtime_collaboration_profile_ids=(
        MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY.profile_ids
    ),
    route_names=tuple(RouteName),
    profile_count=len(MULTIMODAL_ARTIFACT_PROVENANCE_PROFILES),
    source_registries=_ARTIFACT_PROVENANCE_SOURCE_REGISTRIES,
    source_reference_ids=_ARTIFACT_PROVENANCE_SOURCE_REFERENCES,
    artifact_provenance_surface_refs=_ARTIFACT_PROVENANCE_SURFACES,
    observability_surfaces=_ARTIFACT_PROVENANCE_OBSERVABILITY_SURFACES,
)


class ArtifactLineageProfile(BaseModel):
    """Inspectable metadata for one passive Artifact Lineage surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1, max_length=120)
    profile_name: str = Field(min_length=1, max_length=140)
    lineage_profile_kind: ArtifactLineageProfileKind
    lineage_surface_kind: ArtifactLineageSurfaceKind
    source_artifact_provenance_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    lineage_context_fields: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_reference_ids: tuple[str, ...] = Field(min_length=1, max_length=10)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    artifact_lineage_surfaces: tuple[str, ...] = Field(
        min_length=1,
        max_length=7,
    )
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=ARTIFACT_LINEAGE_AUTHORITY_BOUNDARY,
        max_length=960,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_ARTIFACT_LINEAGE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    lineage_inference_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    provenance_recording_implemented: Literal[False] = False
    persistent_lineage_storage_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    serialization_version: Literal["multimodal_artifact_lineage_profile.v1"] = (
        ARTIFACT_LINEAGE_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class MultimodalArtifactLineageRegistry(BaseModel):
    """Stable passive registry for V4.5 Artifact Lineage metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["multimodal_artifact_lineage_registry"] = (
        "multimodal_artifact_lineage_registry"
    )
    serialization_version: Literal["multimodal_artifact_lineage_registry.v1"] = (
        ARTIFACT_LINEAGE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ARTIFACT_LINEAGE_AUTHORITY_BOUNDARY,
        max_length=960,
    )
    artifact_lineage_profiles: tuple[ArtifactLineageProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    lineage_profile_kinds: tuple[ArtifactLineageProfileKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    lineage_surface_kinds: tuple[ArtifactLineageSurfaceKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    artifact_provenance_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    source_reference_ids: tuple[str, ...] = Field(min_length=10, max_length=10)
    artifact_lineage_surface_refs: tuple[str, ...] = Field(
        min_length=7,
        max_length=7,
    )
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_ARTIFACT_LINEAGE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    lineage_inference_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    provenance_recording_implemented: Literal[False] = False
    persistent_lineage_storage_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.profile_id for profile in self.artifact_lineage_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("profile_ids must be unique")
        if self.profile_ids != derived_profile_ids:
            raise ValueError("profile_ids must match artifact_lineage_profiles")
        if self.profile_count != len(self.artifact_lineage_profiles):
            raise ValueError("profile_count must match artifact_lineage_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if (
            self.artifact_provenance_profile_ids
            != MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "artifact_provenance_profile_ids must match Artifact Provenance registry"
            )
        if self.lineage_profile_kinds != _ordered_unique(
            profile.lineage_profile_kind
            for profile in self.artifact_lineage_profiles
        ):
            raise ValueError("lineage_profile_kinds must match profiles")
        if self.lineage_surface_kinds != _ordered_unique(
            profile.lineage_surface_kind
            for profile in self.artifact_lineage_profiles
        ):
            raise ValueError("lineage_surface_kinds must match profiles")

        profile_source_references = {
            source_reference
            for profile in self.artifact_lineage_profiles
            for source_reference in profile.source_reference_ids
        }
        if set(self.source_reference_ids) != profile_source_references:
            raise ValueError(
                "source_reference_ids must match profile source references"
            )

        known_routes = set(self.route_names)
        known_provenance_profiles = set(self.artifact_provenance_profile_ids)
        known_surfaces = set(self.artifact_lineage_surface_refs)
        known_source_references = set(self.source_reference_ids)
        for profile in self.artifact_lineage_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must use known routes")
            if not set(profile.source_artifact_provenance_profile_ids).issubset(
                known_provenance_profiles
            ):
                raise ValueError(
                    "source_artifact_provenance_profile_ids must be known profiles"
                )
            if not set(profile.artifact_lineage_surfaces).issubset(known_surfaces):
                raise ValueError("artifact_lineage_surfaces must be known surfaces")
            if not set(profile.source_reference_ids).issubset(
                known_source_references
            ):
                raise ValueError(
                    "source_reference_ids must be known registry references"
                )
        return self


def multimodal_artifact_lineage_registry() -> MultimodalArtifactLineageRegistry:
    """Return passive V4.5 Artifact Lineage metadata."""

    return MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY


def multimodal_artifact_lineage_profile_by_id(
    profile_id: str,
    registry: MultimodalArtifactLineageRegistry | None = None,
) -> ArtifactLineageProfile | None:
    """Return one Artifact Lineage profile without inferring lineage."""

    source_registry = registry or MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY
    normalized_profile_id = str(profile_id).strip()
    for profile in source_registry.artifact_lineage_profiles:
        if profile.profile_id == normalized_profile_id:
            return profile
    return None


def multimodal_artifact_lineage_profiles_for_route(
    route: RouteName | str,
    registry: MultimodalArtifactLineageRegistry | None = None,
) -> tuple[ArtifactLineageProfile, ...]:
    """Return passive Artifact Lineage profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.artifact_lineage_profiles
        if route_name in profile.route_applicability
    )


def multimodal_artifact_lineage_profiles_for_surface_kind(
    surface_kind: ArtifactLineageSurfaceKind | str,
    registry: MultimodalArtifactLineageRegistry | None = None,
) -> tuple[ArtifactLineageProfile, ...]:
    """Return Artifact Lineage profiles for one passive surface kind."""

    surface_value = str(surface_kind).strip()
    source_registry = registry or MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.artifact_lineage_profiles
        if profile.lineage_surface_kind == surface_value
    )


def multimodal_artifact_lineage_profiles_for_artifact_provenance_profile(
    artifact_provenance_profile_id: str,
    registry: MultimodalArtifactLineageRegistry | None = None,
) -> tuple[ArtifactLineageProfile, ...]:
    """Return lineage profiles referencing one artifact provenance profile."""

    source_registry = registry or MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY
    source_profile_id = str(artifact_provenance_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.artifact_lineage_profiles
        if source_profile_id in profile.source_artifact_provenance_profile_ids
    )


def _artifact_lineage_profile(
    *,
    profile_id: str,
    profile_name: str,
    lineage_profile_kind: ArtifactLineageProfileKind,
    lineage_surface_kind: ArtifactLineageSurfaceKind,
    source_artifact_provenance_profile_ids: tuple[str, ...],
    lineage_context_fields: tuple[str, ...],
    source_reference_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    artifact_lineage_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> ArtifactLineageProfile:
    return ArtifactLineageProfile(
        profile_id=profile_id,
        profile_name=profile_name,
        lineage_profile_kind=lineage_profile_kind,
        lineage_surface_kind=lineage_surface_kind,
        source_artifact_provenance_profile_ids=(
            source_artifact_provenance_profile_ids
        ),
        lineage_context_fields=lineage_context_fields,
        source_reference_ids=source_reference_ids,
        route_applicability=route_applicability,
        artifact_lineage_surfaces=artifact_lineage_surfaces,
        advisory_outputs=advisory_outputs,
        source_registries=_ARTIFACT_LINEAGE_SOURCE_REGISTRIES,
        observability_surfaces=_ARTIFACT_LINEAGE_OBSERVABILITY_SURFACES,
    )


MULTIMODAL_ARTIFACT_LINEAGE_PROFILES = (
    _artifact_lineage_profile(
        profile_id="dependency_graph_artifact_lineage",
        profile_name="Dependency Graph Artifact Lineage",
        lineage_profile_kind="dependency_graph_lineage",
        lineage_surface_kind="dependency_graph",
        source_artifact_provenance_profile_ids=(
            "evidence_artifact_provenance",
            "payload_artifact_provenance",
        ),
        lineage_context_fields=(
            "dependency_sources",
            "artifact_dependency_graph.artifact_nodes",
            "artifact_dependency_graph.dependency_edges",
            "downstream_consumers",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY",
            "orchestration.artifact_dependency_graph.ArtifactDependencyGraph",
            "orchestration.artifact_dependency_graph.ArtifactDependencyEdge",
            "clients.nextjs.provenance_engine.buildProvenanceEngineModel",
        ),
        route_applicability=tuple(RouteName),
        artifact_lineage_surfaces=(
            "artifact_lineage_panel",
            "dependency_graph_lineage_surface",
            "lineage_summary_surface",
            "artifact_lineage_boundary_panel",
        ),
        advisory_outputs=(
            "dependency_graph_lineage_inventory",
            "manual_dependency_review_hint",
            "no_lineage_inference_notice",
        ),
    ),
    _artifact_lineage_profile(
        profile_id="timeline_stage_artifact_lineage",
        profile_name="Timeline Stage Artifact Lineage",
        lineage_profile_kind="timeline_stage_lineage",
        lineage_surface_kind="timeline_stage",
        source_artifact_provenance_profile_ids=(
            "evaluation_artifact_provenance",
            "missing_source_artifact_provenance",
        ),
        lineage_context_fields=(
            "creative_timeline.events",
            "metadataAvailability",
            "sourceCount",
            "workflow_explorer.stages",
        ),
        source_reference_ids=(
            "clients.nextjs.creative_timeline.buildCreativeTimelineModel",
            "clients.nextjs.creative_timeline.provenanceSourceCount",
            "clients.nextjs.workflow_explorer.WorkflowExplorerStage",
            "clients.nextjs.workflow_runtime.WorkflowRuntimeTraceEvent",
            "clients.nextjs.workstation_shell.WorkstationShell",
        ),
        route_applicability=tuple(RouteName),
        artifact_lineage_surfaces=(
            "artifact_lineage_panel",
            "timeline_stage_lineage_surface",
            "lineage_summary_surface",
            "artifact_lineage_boundary_panel",
        ),
        advisory_outputs=(
            "timeline_stage_lineage_inventory",
            "manual_timeline_review_hint",
            "no_timeline_reconstruction_notice",
        ),
    ),
    _artifact_lineage_profile(
        profile_id="source_transition_artifact_lineage",
        profile_name="Source Transition Artifact Lineage",
        lineage_profile_kind="source_transition_lineage",
        lineage_surface_kind="source_transition",
        source_artifact_provenance_profile_ids=(
            "evidence_artifact_provenance",
            "payload_artifact_provenance",
            "evaluation_artifact_provenance",
        ),
        lineage_context_fields=(
            "evidence_sources",
            "dependency_sources",
            "artifact_sources",
            "evaluation_sources",
            "eventSequence",
            "sourceKeys",
        ),
        source_reference_ids=(
            "clients.nextjs.provenance_engine.buildProvenanceEngineModel",
            "clients.nextjs.provenance_engine.ProvenanceSource",
            "clients.nextjs.creative_timeline.provenanceSourceCount",
            "clients.nextjs.workflow_runtime.WorkflowRuntimeTraceEvent",
        ),
        route_applicability=tuple(RouteName),
        artifact_lineage_surfaces=(
            "artifact_lineage_panel",
            "source_transition_lineage_surface",
            "lineage_summary_surface",
            "artifact_lineage_boundary_panel",
        ),
        advisory_outputs=(
            "source_transition_lineage_inventory",
            "manual_transition_review_hint",
            "no_provenance_recording_notice",
        ),
    ),
    _artifact_lineage_profile(
        profile_id="missing_artifact_lineage",
        profile_name="Missing Artifact Lineage",
        lineage_profile_kind="missing_lineage",
        lineage_surface_kind="missing_lineage",
        source_artifact_provenance_profile_ids=("missing_source_artifact_provenance",),
        lineage_context_fields=(
            "unsupported_or_missing_sources",
            "missingSourceCount",
            "metadataGroups",
            "readiness",
        ),
        source_reference_ids=(
            "clients.nextjs.provenance_engine.buildProvenanceEngineModel",
            "clients.nextjs.provenance_engine.ProvenanceSource",
            "clients.nextjs.creative_timeline.buildCreativeTimelineModel",
            "clients.nextjs.workflow_explorer.WorkflowExplorerStage",
        ),
        route_applicability=tuple(RouteName),
        artifact_lineage_surfaces=(
            "artifact_lineage_panel",
            "missing_lineage_surface",
            "lineage_summary_surface",
            "artifact_lineage_boundary_panel",
        ),
        advisory_outputs=(
            "missing_lineage_inventory",
            "manual_missing_lineage_review_hint",
            "no_persistent_lineage_storage_notice",
        ),
    ),
)

MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY = MultimodalArtifactLineageRegistry(
    artifact_lineage_profiles=MULTIMODAL_ARTIFACT_LINEAGE_PROFILES,
    profile_ids=tuple(
        profile.profile_id for profile in MULTIMODAL_ARTIFACT_LINEAGE_PROFILES
    ),
    lineage_profile_kinds=tuple(
        profile.lineage_profile_kind
        for profile in MULTIMODAL_ARTIFACT_LINEAGE_PROFILES
    ),
    lineage_surface_kinds=tuple(
        profile.lineage_surface_kind
        for profile in MULTIMODAL_ARTIFACT_LINEAGE_PROFILES
    ),
    artifact_provenance_profile_ids=(
        MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY.profile_ids
    ),
    route_names=tuple(RouteName),
    profile_count=len(MULTIMODAL_ARTIFACT_LINEAGE_PROFILES),
    source_registries=_ARTIFACT_LINEAGE_SOURCE_REGISTRIES,
    source_reference_ids=_ARTIFACT_LINEAGE_SOURCE_REFERENCES,
    artifact_lineage_surface_refs=_ARTIFACT_LINEAGE_SURFACES,
    observability_surfaces=_ARTIFACT_LINEAGE_OBSERVABILITY_SURFACES,
)
