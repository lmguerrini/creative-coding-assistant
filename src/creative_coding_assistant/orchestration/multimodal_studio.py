"""Passive V4.5 Multimodal Studio metadata surfaces."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.blackboard_memory import (
    BLACKBOARD_MEMORY_REGISTRY,
)
from creative_coding_assistant.orchestration.hybrid_studio import (
    AGENT_WORKSPACE_REGISTRY,
    SESSION_REPLAY_REGISTRY,
    WORKSPACE_SNAPSHOT_REGISTRY,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.shared_context_views import (
    SHARED_CONTEXT_VIEW_REGISTRY,
)
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
CrossAgentWorkspaceProfileKind = Literal[
    "planning_cross_agent_workspace",
    "artifact_runtime_cross_agent_workspace",
    "critique_curation_cross_agent_workspace",
    "refinement_synthesis_cross_agent_workspace",
]
CrossAgentWorkspaceSurfaceKind = Literal[
    "planning_context",
    "artifact_runtime",
    "critique_curation",
    "refinement_synthesis",
]
SharedArtifactBoardProfileKind = Literal[
    "artifact_selection_board",
    "comparison_review_board",
    "provenance_lineage_board",
    "handoff_refinement_board",
]
SharedArtifactBoardSurfaceKind = Literal[
    "selection",
    "comparison",
    "provenance_lineage",
    "handoff_refinement",
]
WorkspaceHistoryProfileKind = Literal[
    "session_record_history",
    "snapshot_history",
    "artifact_board_history",
    "runtime_event_history",
]
WorkspaceHistorySurfaceKind = Literal[
    "session_record",
    "snapshot",
    "artifact_board",
    "runtime_event",
]
BranchingTimelineProfileKind = Literal[
    "workflow_branch_timeline",
    "artifact_variant_branch_timeline",
    "review_retry_branch_timeline",
    "fallback_failure_branch_timeline",
]
BranchingTimelineSurfaceKind = Literal[
    "workflow_branch",
    "artifact_variant",
    "review_retry",
    "fallback_failure",
]
CreativeEvolutionTimelineProfileKind = Literal[
    "intent_evolution_timeline",
    "artifact_iteration_evolution_timeline",
    "quality_refinement_evolution_timeline",
    "final_synthesis_evolution_timeline",
]
CreativeEvolutionTimelineSurfaceKind = Literal[
    "intent_evolution",
    "artifact_iteration",
    "quality_refinement",
    "final_synthesis",
]
RealTimeWorkflowVisualizationProfileKind = Literal[
    "runtime_state_visualization",
    "timeline_event_visualization",
    "metadata_stage_visualization",
    "console_health_visualization",
]
RealTimeWorkflowVisualizationSurfaceKind = Literal[
    "runtime_state",
    "timeline_event",
    "metadata_stage",
    "console_health",
]
MultimodalStudioIntegrationKind = Literal[
    "preview_workspace_integration",
    "collaboration_artifact_integration",
    "history_lineage_integration",
    "timeline_visualization_integration",
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
CROSS_AGENT_WORKSPACE_PROFILE_SERIALIZATION_VERSION = (
    "multimodal_cross_agent_workspace_profile.v1"
)
CROSS_AGENT_WORKSPACE_REGISTRY_SERIALIZATION_VERSION = (
    "multimodal_cross_agent_workspace_registry.v1"
)
SHARED_ARTIFACT_BOARD_PROFILE_SERIALIZATION_VERSION = (
    "multimodal_shared_artifact_board_profile.v1"
)
SHARED_ARTIFACT_BOARD_REGISTRY_SERIALIZATION_VERSION = (
    "multimodal_shared_artifact_board_registry.v1"
)
WORKSPACE_HISTORY_PROFILE_SERIALIZATION_VERSION = (
    "multimodal_workspace_history_profile.v1"
)
WORKSPACE_HISTORY_REGISTRY_SERIALIZATION_VERSION = (
    "multimodal_workspace_history_registry.v1"
)
BRANCHING_TIMELINE_PROFILE_SERIALIZATION_VERSION = (
    "multimodal_branching_timeline_profile.v1"
)
BRANCHING_TIMELINE_REGISTRY_SERIALIZATION_VERSION = (
    "multimodal_branching_timeline_registry.v1"
)
CREATIVE_EVOLUTION_TIMELINE_PROFILE_SERIALIZATION_VERSION = (
    "multimodal_creative_evolution_timeline_profile.v1"
)
CREATIVE_EVOLUTION_TIMELINE_REGISTRY_SERIALIZATION_VERSION = (
    "multimodal_creative_evolution_timeline_registry.v1"
)
REAL_TIME_WORKFLOW_VISUALIZATION_PROFILE_SERIALIZATION_VERSION = (
    "multimodal_real_time_workflow_visualization_profile.v1"
)
REAL_TIME_WORKFLOW_VISUALIZATION_REGISTRY_SERIALIZATION_VERSION = (
    "multimodal_real_time_workflow_visualization_registry.v1"
)
MULTIMODAL_STUDIO_INTEGRATION_PROFILE_SERIALIZATION_VERSION = (
    "multimodal_studio_integration_profile.v1"
)
MULTIMODAL_STUDIO_INTEGRATION_REGISTRY_SERIALIZATION_VERSION = (
    "multimodal_studio_integration_registry.v1"
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
CROSS_AGENT_WORKSPACE_AUTHORITY_BOUNDARY = (
    "Cross-Agent Workspace metadata describes passive V4.5 Multimodal Studio "
    "cross-agent workspace surfaces for inspection only; it does not "
    "instantiate agents, invoke agents, orchestrate multiple agents, "
    "materialize shared context, read or write blackboard state, mutate "
    "workspace state, persist collaboration storage, execute rendering, "
    "control workflows, request human input, route providers or models, "
    "trigger retries, or modify generated output."
)
SHARED_ARTIFACT_BOARD_AUTHORITY_BOUNDARY = (
    "Shared Artifact Board metadata describes passive V4.5 Multimodal Studio "
    "artifact board surfaces for inspection only; it does not create "
    "collaborative board state, mutate artifacts, change artifact selection, "
    "persist board storage, execute rendering, invoke agents, materialize "
    "shared context, control workflows, request human input, route providers "
    "or models, trigger retries, open networking, or modify generated output."
)
WORKSPACE_HISTORY_AUTHORITY_BOUNDARY = (
    "Workspace History metadata describes passive V4.5 Multimodal Studio "
    "workspace history surfaces for inspection only; it does not record "
    "workspace history, capture snapshots, reconstruct timelines, persist "
    "history storage, replay runtime events, mutate workspace state, mutate "
    "artifacts, execute rendering, control workflows, request human input, "
    "route providers or models, trigger retries, open networking, or modify "
    "generated output."
)
BRANCHING_TIMELINE_AUTHORITY_BOUNDARY = (
    "Branching Timeline metadata describes passive V4.5 Multimodal Studio "
    "branching timeline surfaces for inspection only; it does not create "
    "branches, execute branch routing, reconstruct timelines, replay runtime "
    "events, trigger retries or refinements, mutate workflow state, mutate "
    "workspace state, mutate artifacts, persist branch storage, execute "
    "rendering, request human input, route providers or models, open "
    "networking, or modify generated output."
)
CREATIVE_EVOLUTION_TIMELINE_AUTHORITY_BOUNDARY = (
    "Creative Evolution Timeline metadata describes passive V4.5 Multimodal "
    "Studio evolution timeline surfaces for inspection only; it does not "
    "generate creative evolution, reconstruct timelines, create branches, "
    "mutate artifacts, modify generated output, change quality scores, record "
    "provenance, replay runtime events, control workflows, request human "
    "input, route providers or models, trigger retries, open networking, or "
    "execute rendering."
)
REAL_TIME_WORKFLOW_VISUALIZATION_AUTHORITY_BOUNDARY = (
    "Real-Time Workflow Visualization metadata describes passive V4.5 "
    "Multimodal Studio workflow visualization surfaces for inspection only; "
    "it does not subscribe to live streams, mutate workflow state, reconstruct "
    "timelines, replay events, control runtime consoles, control preview "
    "runtimes, mutate artifacts, modify generated output, persist "
    "collaboration storage, execute rendering, control workflows, request "
    "human input, route providers or models, trigger retries, or open "
    "networking."
)
MULTIMODAL_STUDIO_INTEGRATION_AUTHORITY_BOUNDARY = (
    "Multimodal Studio Integration metadata describes a passive V4.5 inventory "
    "of Multimodal Studio preview, canvas, workspace, collaboration, artifact, "
    "provenance, lineage, history, branching, creative evolution, and workflow "
    "visualization registries only; it does not activate Studio runtime, "
    "execute rendering, route providers or models, execute providers, mutate "
    "artifacts, modify generated output, control workflows, request human "
    "input, trigger retries, mutate storage, persist collaboration storage, "
    "reconstruct timelines, subscribe to live streams, or open networking."
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

_CROSS_AGENT_WORKSPACE_SOURCE_REGISTRIES = (
    "multimodal_visual_workspace_registry",
    "multimodal_artifact_lineage_registry",
    "agent_workspace_registry",
    "shared_context_view_registry",
    "blackboard_memory_registry",
    "nextjs_workstation_shell",
    "nextjs_workstation_state",
)

_CROSS_AGENT_WORKSPACE_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_VISUAL_WORKSPACE_REGISTRY",
    "multimodal_studio.MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY",
    "hybrid_studio.AGENT_WORKSPACE_REGISTRY",
    "shared_context_views.SHARED_CONTEXT_VIEW_REGISTRY",
    "blackboard_memory.BLACKBOARD_MEMORY_REGISTRY",
    "hybrid_studio.AgentWorkspaceProfile",
    "shared_context_views.SharedContextViewContract",
    "blackboard_memory.BlackboardMemoryChannelContract",
    "clients.nextjs.workstation_shell.WorkstationShell",
    "clients.nextjs.workstation_state.buildWorkstationState",
)

_CROSS_AGENT_WORKSPACE_SURFACES = (
    "cross_agent_workspace_panel",
    "cross_agent_roster_surface",
    "shared_context_scope_surface",
    "blackboard_channel_surface",
    "lineage_context_surface",
    "workspace_handoff_surface",
    "cross_agent_workspace_boundary_panel",
)

_CROSS_AGENT_WORKSPACE_OBSERVABILITY_SURFACES = (
    "profile_id",
    "cross_agent_workspace_kind",
    "cross_agent_surface_kind",
    "source_agent_workspace_profile_ids",
    "source_shared_context_view_ids",
    "source_reference_ids",
    "authority_boundary",
)

_CROSS_AGENT_WORKSPACE_BLOCKED_RUNTIME_BEHAVIORS = (
    "workspace_execution",
    "agent_instantiation",
    "agent_invocation",
    "multi_agent_orchestration",
    "shared_context_materialization",
    "blackboard_state_reads",
    "blackboard_state_writes",
    "workspace_state_mutation",
    "collaboration_storage_persistence",
    "rendering_execution",
    "workflow_control",
    "human_input_request",
    "provider_or_model_routing",
    "retry_triggering",
    "generated_output_mutation",
)

_SHARED_ARTIFACT_BOARD_SOURCE_REGISTRIES = (
    "multimodal_cross_agent_workspace_registry",
    "multimodal_artifact_collaboration_registry",
    "multimodal_multi_preview_registry",
    "multimodal_artifact_provenance_registry",
    "multimodal_artifact_lineage_registry",
    "nextjs_artifact_comparison",
    "nextjs_artifact_inspector",
    "nextjs_workstation_shell",
)

_SHARED_ARTIFACT_BOARD_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_CROSS_AGENT_WORKSPACE_REGISTRY",
    "multimodal_studio.MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY",
    "multimodal_studio.MULTIMODAL_MULTI_PREVIEW_REGISTRY",
    "multimodal_studio.MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY",
    "multimodal_studio.MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY",
    "clients.nextjs.artifact_comparison.buildArtifactComparisonModel",
    "clients.nextjs.artifact_comparison.ArtifactComparisonRow",
    "clients.nextjs.artifact_inspector.buildArtifactDocument",
    "clients.nextjs.artifact_inspector.highlightArtifactDocument",
    "clients.nextjs.workstation_shell.WorkstationShell",
)

_SHARED_ARTIFACT_BOARD_SURFACES = (
    "shared_artifact_board_panel",
    "artifact_selection_board_surface",
    "artifact_comparison_board_surface",
    "artifact_provenance_board_surface",
    "artifact_lineage_board_surface",
    "artifact_handoff_board_surface",
    "shared_artifact_board_boundary_panel",
)

_SHARED_ARTIFACT_BOARD_OBSERVABILITY_SURFACES = (
    "profile_id",
    "board_profile_kind",
    "board_surface_kind",
    "source_cross_agent_workspace_profile_ids",
    "source_artifact_collaboration_profile_ids",
    "source_reference_ids",
    "authority_boundary",
)

_SHARED_ARTIFACT_BOARD_BLOCKED_RUNTIME_BEHAVIORS = (
    "board_state_creation",
    "collaborative_board_persistence",
    "artifact_selection_mutation",
    "artifact_mutation",
    "generated_output_mutation",
    "rendering_execution",
    "agent_invocation",
    "shared_context_materialization",
    "workflow_control",
    "human_input_request",
    "provider_or_model_routing",
    "retry_triggering",
    "networking",
)

_WORKSPACE_HISTORY_SOURCE_REGISTRIES = (
    "multimodal_shared_artifact_board_registry",
    "multimodal_runtime_collaboration_registry",
    "workspace_snapshot_registry",
    "session_replay_registry",
    "nextjs_workspace_persistence",
    "nextjs_creative_timeline",
    "nextjs_workflow_runtime",
    "nextjs_workstation_shell",
)

_WORKSPACE_HISTORY_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY",
    "multimodal_studio.MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY",
    "hybrid_studio.WORKSPACE_SNAPSHOT_REGISTRY",
    "hybrid_studio.SESSION_REPLAY_REGISTRY",
    "clients.nextjs.workspace_persistence.WorkspaceSessionRecord",
    "clients.nextjs.workspace_persistence.createWorkspaceSessionRecord",
    "clients.nextjs.workspace_persistence.snapshotFromWorkspaceSessionRecord",
    "clients.nextjs.creative_timeline.buildCreativeTimelineModel",
    "clients.nextjs.workflow_runtime.WorkflowRuntimeTraceEvent",
    "clients.nextjs.workstation_shell.WorkstationShell",
)

_WORKSPACE_HISTORY_SURFACES = (
    "workspace_history_panel",
    "session_record_history_surface",
    "snapshot_history_surface",
    "artifact_board_history_surface",
    "runtime_event_history_surface",
    "history_summary_surface",
    "workspace_history_boundary_panel",
)

_WORKSPACE_HISTORY_OBSERVABILITY_SURFACES = (
    "profile_id",
    "history_profile_kind",
    "history_surface_kind",
    "source_workspace_snapshot_profile_ids",
    "source_session_replay_profile_ids",
    "source_reference_ids",
    "authority_boundary",
)

_WORKSPACE_HISTORY_BLOCKED_RUNTIME_BEHAVIORS = (
    "history_recording",
    "snapshot_capture",
    "timeline_reconstruction",
    "history_persistence",
    "session_replay_execution",
    "runtime_event_replay",
    "workspace_state_mutation",
    "artifact_mutation",
    "generated_output_mutation",
    "rendering_execution",
    "workflow_control",
    "human_input_request",
    "provider_or_model_routing",
    "retry_triggering",
    "networking",
)

_BRANCHING_TIMELINE_SOURCE_REGISTRIES = (
    "multimodal_workspace_history_registry",
    "multimodal_artifact_lineage_registry",
    "multimodal_shared_artifact_board_registry",
    "multimodal_runtime_collaboration_registry",
    "session_replay_registry",
    "nextjs_workflow_runtime",
    "nextjs_workflow_timeline",
    "nextjs_workstation_shell",
)

_BRANCHING_TIMELINE_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_WORKSPACE_HISTORY_REGISTRY",
    "multimodal_studio.MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY",
    "multimodal_studio.MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY",
    "multimodal_studio.MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY",
    "hybrid_studio.SESSION_REPLAY_REGISTRY",
    "clients.nextjs.workflow_runtime.WorkflowRuntimeVisualState",
    "clients.nextjs.workflow_runtime.deriveWorkflowVisualState",
    "clients.nextjs.workflow_timeline.buildWorkflowTimelineModel",
    "clients.nextjs.workflow_timeline.WorkflowTimelineEvent",
    "clients.nextjs.workstation_shell.WorkstationShell",
)

_BRANCHING_TIMELINE_SURFACES = (
    "branching_timeline_panel",
    "workflow_branch_surface",
    "artifact_variant_branch_surface",
    "review_retry_branch_surface",
    "fallback_failure_branch_surface",
    "branch_summary_surface",
    "branching_timeline_boundary_panel",
)

_BRANCHING_TIMELINE_OBSERVABILITY_SURFACES = (
    "profile_id",
    "branching_timeline_kind",
    "branch_surface_kind",
    "source_workspace_history_profile_ids",
    "source_runtime_collaboration_profile_ids",
    "source_reference_ids",
    "authority_boundary",
)

_BRANCHING_TIMELINE_BLOCKED_RUNTIME_BEHAVIORS = (
    "branch_creation",
    "branch_routing_execution",
    "timeline_reconstruction",
    "runtime_event_replay",
    "retry_triggering",
    "refinement_triggering",
    "workflow_state_mutation",
    "workspace_state_mutation",
    "artifact_mutation",
    "generated_output_mutation",
    "branch_storage_persistence",
    "rendering_execution",
    "human_input_request",
    "provider_or_model_routing",
    "networking",
)

_CREATIVE_EVOLUTION_TIMELINE_SOURCE_REGISTRIES = (
    "multimodal_branching_timeline_registry",
    "multimodal_workspace_history_registry",
    "multimodal_shared_artifact_board_registry",
    "multimodal_artifact_lineage_registry",
    "multimodal_artifact_provenance_registry",
    "nextjs_creative_timeline",
    "nextjs_workflow_explorer",
    "nextjs_workstation_shell",
)

_CREATIVE_EVOLUTION_TIMELINE_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_BRANCHING_TIMELINE_REGISTRY",
    "multimodal_studio.MULTIMODAL_WORKSPACE_HISTORY_REGISTRY",
    "multimodal_studio.MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY",
    "multimodal_studio.MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY",
    "multimodal_studio.MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY",
    "clients.nextjs.creative_timeline.buildCreativeTimelineModel",
    "clients.nextjs.creative_timeline.CreativeTimelineEvent",
    "clients.nextjs.creative_timeline.provenanceSourceCount",
    "clients.nextjs.workflow_explorer.WorkflowExplorerStage",
    "clients.nextjs.workstation_shell.WorkstationShell",
)

_CREATIVE_EVOLUTION_TIMELINE_SURFACES = (
    "creative_evolution_timeline_panel",
    "intent_evolution_surface",
    "artifact_iteration_evolution_surface",
    "quality_refinement_evolution_surface",
    "final_synthesis_evolution_surface",
    "evolution_summary_surface",
    "creative_evolution_boundary_panel",
)

_CREATIVE_EVOLUTION_TIMELINE_OBSERVABILITY_SURFACES = (
    "profile_id",
    "evolution_profile_kind",
    "evolution_surface_kind",
    "source_branching_timeline_profile_ids",
    "source_workspace_history_profile_ids",
    "source_reference_ids",
    "authority_boundary",
)

_CREATIVE_EVOLUTION_TIMELINE_BLOCKED_RUNTIME_BEHAVIORS = (
    "evolution_generation",
    "timeline_reconstruction",
    "branch_creation",
    "artifact_mutation",
    "generated_output_mutation",
    "quality_score_mutation",
    "provenance_recording",
    "runtime_event_replay",
    "workflow_control",
    "human_input_request",
    "provider_or_model_routing",
    "retry_triggering",
    "networking",
    "rendering_execution",
)

_REAL_TIME_WORKFLOW_VISUALIZATION_SOURCE_REGISTRIES = (
    "multimodal_creative_evolution_timeline_registry",
    "multimodal_branching_timeline_registry",
    "multimodal_runtime_collaboration_registry",
    "multimodal_workspace_history_registry",
    "nextjs_workflow_runtime",
    "nextjs_workflow_timeline",
    "nextjs_workflow_explorer",
    "nextjs_runtime_console",
)

_REAL_TIME_WORKFLOW_VISUALIZATION_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_REGISTRY",
    "multimodal_studio.MULTIMODAL_BRANCHING_TIMELINE_REGISTRY",
    "multimodal_studio.MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY",
    "multimodal_studio.MULTIMODAL_WORKSPACE_HISTORY_REGISTRY",
    "clients.nextjs.workflow_runtime.WorkflowRuntimeModel",
    "clients.nextjs.workflow_runtime.WorkflowRuntimeVisualState",
    "clients.nextjs.workflow_timeline.WorkflowTimelineModel",
    "clients.nextjs.workflow_timeline.WorkflowTimelineEvent",
    "clients.nextjs.workflow_explorer.WorkflowExplorerStage",
    "clients.nextjs.runtime_console.RuntimeConsoleModel",
)

_REAL_TIME_WORKFLOW_VISUALIZATION_SURFACES = (
    "real_time_workflow_visualization_panel",
    "runtime_state_visual_surface",
    "timeline_event_visual_surface",
    "metadata_stage_visual_surface",
    "console_health_visual_surface",
    "workflow_visualization_summary_surface",
    "real_time_workflow_visualization_boundary_panel",
)

_REAL_TIME_WORKFLOW_VISUALIZATION_OBSERVABILITY_SURFACES = (
    "profile_id",
    "visualization_profile_kind",
    "visualization_surface_kind",
    "source_creative_evolution_timeline_profile_ids",
    "source_runtime_collaboration_profile_ids",
    "source_reference_ids",
    "authority_boundary",
)

_REAL_TIME_WORKFLOW_VISUALIZATION_BLOCKED_RUNTIME_BEHAVIORS = (
    "real_time_stream_subscription",
    "workflow_state_mutation",
    "timeline_reconstruction",
    "event_replay",
    "runtime_console_control",
    "preview_runtime_control",
    "artifact_mutation",
    "generated_output_mutation",
    "collaboration_storage_persistence",
    "rendering_execution",
    "workflow_control",
    "human_input_request",
    "provider_or_model_routing",
    "retry_triggering",
    "networking",
)

_MULTIMODAL_STUDIO_INTEGRATION_SOURCE_REGISTRIES = (
    "multimodal_live_preview_registry",
    "multimodal_multi_preview_registry",
    "multimodal_interactive_canvas_registry",
    "multimodal_visual_workspace_registry",
    "multimodal_runtime_collaboration_registry",
    "multimodal_artifact_collaboration_registry",
    "multimodal_artifact_provenance_registry",
    "multimodal_artifact_lineage_registry",
    "multimodal_cross_agent_workspace_registry",
    "multimodal_shared_artifact_board_registry",
    "multimodal_workspace_history_registry",
    "multimodal_branching_timeline_registry",
    "multimodal_creative_evolution_timeline_registry",
    "multimodal_real_time_workflow_visualization_registry",
)

_MULTIMODAL_STUDIO_INTEGRATION_PROFILE_GROUPS = (
    "live_preview_profiles",
    "multi_preview_profiles",
    "interactive_canvas_profiles",
    "visual_workspace_profiles",
    "runtime_collaboration_profiles",
    "artifact_collaboration_profiles",
    "artifact_provenance_profiles",
    "artifact_lineage_profiles",
    "cross_agent_workspace_profiles",
    "shared_artifact_board_profiles",
    "workspace_history_profiles",
    "branching_timeline_profiles",
    "creative_evolution_timeline_profiles",
    "real_time_workflow_visualization_profiles",
)

_MULTIMODAL_STUDIO_INTEGRATION_SURFACES = (
    "multimodal_studio_shell",
    "preview_workspace_integration_surface",
    "collaboration_artifact_integration_surface",
    "history_lineage_integration_surface",
    "timeline_visualization_integration_surface",
    "integration_summary_surface",
    "multimodal_studio_integration_boundary_panel",
)

_MULTIMODAL_STUDIO_INTEGRATION_OBSERVABILITY_SURFACES = (
    "integration_profile_id",
    "integration_kind",
    "source_registry_names",
    "linked_profile_group_refs",
    "route_applicability",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_MULTIMODAL_STUDIO_INTEGRATION_BLOCKED_RUNTIME_BEHAVIORS = (
    "studio_runtime_activation",
    "rendering_execution",
    "provider_or_model_routing",
    "provider_execution",
    "artifact_mutation",
    "generated_output_mutation",
    "workflow_control",
    "human_input_request",
    "retry_triggering",
    "storage_mutation",
    "collaboration_storage_persistence",
    "timeline_reconstruction",
    "real_time_stream_subscription",
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


class CrossAgentWorkspaceProfile(BaseModel):
    """Inspectable metadata for one passive Cross-Agent Workspace surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1, max_length=140)
    profile_name: str = Field(min_length=1, max_length=160)
    cross_agent_workspace_kind: CrossAgentWorkspaceProfileKind
    cross_agent_surface_kind: CrossAgentWorkspaceSurfaceKind
    source_agent_workspace_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=2,
    )
    source_visual_workspace_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_artifact_lineage_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_shared_context_view_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_blackboard_channel_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    workspace_context_fields: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_reference_ids: tuple[str, ...] = Field(min_length=1, max_length=10)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    cross_agent_workspace_surfaces: tuple[str, ...] = Field(
        min_length=1,
        max_length=7,
    )
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=CROSS_AGENT_WORKSPACE_AUTHORITY_BOUNDARY,
        max_length=1300,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CROSS_AGENT_WORKSPACE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    workspace_execution_implemented: Literal[False] = False
    agent_instantiation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    multi_agent_orchestration_implemented: Literal[False] = False
    shared_context_materialization_implemented: Literal[False] = False
    blackboard_state_read_implemented: Literal[False] = False
    blackboard_state_write_implemented: Literal[False] = False
    workspace_state_mutation_implemented: Literal[False] = False
    collaboration_storage_persistence_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["multimodal_cross_agent_workspace_profile.v1"] = (
        CROSS_AGENT_WORKSPACE_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class MultimodalCrossAgentWorkspaceRegistry(BaseModel):
    """Stable passive registry for V4.5 Cross-Agent Workspace metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["multimodal_cross_agent_workspace_registry"] = (
        "multimodal_cross_agent_workspace_registry"
    )
    serialization_version: Literal["multimodal_cross_agent_workspace_registry.v1"] = (
        CROSS_AGENT_WORKSPACE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CROSS_AGENT_WORKSPACE_AUTHORITY_BOUNDARY,
        max_length=1300,
    )
    cross_agent_workspace_profiles: tuple[CrossAgentWorkspaceProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    cross_agent_workspace_kinds: tuple[CrossAgentWorkspaceProfileKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    cross_agent_surface_kinds: tuple[CrossAgentWorkspaceSurfaceKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    agent_workspace_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    visual_workspace_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    artifact_lineage_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    shared_context_view_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    blackboard_channel_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    source_reference_ids: tuple[str, ...] = Field(min_length=10, max_length=10)
    cross_agent_workspace_surface_refs: tuple[str, ...] = Field(
        min_length=7,
        max_length=7,
    )
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CROSS_AGENT_WORKSPACE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    workspace_execution_implemented: Literal[False] = False
    agent_instantiation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    multi_agent_orchestration_implemented: Literal[False] = False
    shared_context_materialization_implemented: Literal[False] = False
    blackboard_state_read_implemented: Literal[False] = False
    blackboard_state_write_implemented: Literal[False] = False
    workspace_state_mutation_implemented: Literal[False] = False
    collaboration_storage_persistence_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.profile_id for profile in self.cross_agent_workspace_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("profile_ids must be unique")
        if self.profile_ids != derived_profile_ids:
            raise ValueError("profile_ids must match cross_agent_workspace_profiles")
        if self.profile_count != len(self.cross_agent_workspace_profiles):
            raise ValueError("profile_count must match cross_agent_workspace_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.agent_workspace_profile_ids != AGENT_WORKSPACE_REGISTRY.workspace_profile_ids:
            raise ValueError(
                "agent_workspace_profile_ids must match Agent Workspace registry"
            )
        if (
            self.visual_workspace_profile_ids
            != MULTIMODAL_VISUAL_WORKSPACE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "visual_workspace_profile_ids must match Visual Workspace registry"
            )
        if (
            self.artifact_lineage_profile_ids
            != MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "artifact_lineage_profile_ids must match Artifact Lineage registry"
            )
        if self.shared_context_view_ids != SHARED_CONTEXT_VIEW_REGISTRY.view_ids:
            raise ValueError(
                "shared_context_view_ids must match Shared Context View registry"
            )
        if self.blackboard_channel_ids != BLACKBOARD_MEMORY_REGISTRY.channel_ids:
            raise ValueError(
                "blackboard_channel_ids must match Blackboard Memory registry"
            )
        if self.cross_agent_workspace_kinds != _ordered_unique(
            profile.cross_agent_workspace_kind
            for profile in self.cross_agent_workspace_profiles
        ):
            raise ValueError("cross_agent_workspace_kinds must match profiles")
        if self.cross_agent_surface_kinds != _ordered_unique(
            profile.cross_agent_surface_kind
            for profile in self.cross_agent_workspace_profiles
        ):
            raise ValueError("cross_agent_surface_kinds must match profiles")

        profile_source_references = {
            source_reference
            for profile in self.cross_agent_workspace_profiles
            for source_reference in profile.source_reference_ids
        }
        if set(self.source_reference_ids) != profile_source_references:
            raise ValueError(
                "source_reference_ids must match profile source references"
            )

        known_routes = set(self.route_names)
        known_agent_workspaces = set(self.agent_workspace_profile_ids)
        known_visual_workspaces = set(self.visual_workspace_profile_ids)
        known_lineage_profiles = set(self.artifact_lineage_profile_ids)
        known_shared_context_views = set(self.shared_context_view_ids)
        known_blackboard_channels = set(self.blackboard_channel_ids)
        known_surfaces = set(self.cross_agent_workspace_surface_refs)
        known_source_references = set(self.source_reference_ids)
        for profile in self.cross_agent_workspace_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must use known routes")
            if not set(profile.source_agent_workspace_profile_ids).issubset(
                known_agent_workspaces
            ):
                raise ValueError(
                    "source_agent_workspace_profile_ids must be known profiles"
                )
            if not set(profile.source_visual_workspace_profile_ids).issubset(
                known_visual_workspaces
            ):
                raise ValueError(
                    "source_visual_workspace_profile_ids must be known profiles"
                )
            if not set(profile.source_artifact_lineage_profile_ids).issubset(
                known_lineage_profiles
            ):
                raise ValueError(
                    "source_artifact_lineage_profile_ids must be known profiles"
                )
            if not set(profile.source_shared_context_view_ids).issubset(
                known_shared_context_views
            ):
                raise ValueError(
                    "source_shared_context_view_ids must be known views"
                )
            if not set(profile.source_blackboard_channel_ids).issubset(
                known_blackboard_channels
            ):
                raise ValueError(
                    "source_blackboard_channel_ids must be known channels"
                )
            if not set(profile.cross_agent_workspace_surfaces).issubset(
                known_surfaces
            ):
                raise ValueError(
                    "cross_agent_workspace_surfaces must be known surfaces"
                )
            if not set(profile.source_reference_ids).issubset(
                known_source_references
            ):
                raise ValueError(
                    "source_reference_ids must be known registry references"
                )
        return self


def multimodal_cross_agent_workspace_registry() -> (
    MultimodalCrossAgentWorkspaceRegistry
):
    """Return passive V4.5 Cross-Agent Workspace metadata."""

    return MULTIMODAL_CROSS_AGENT_WORKSPACE_REGISTRY


def multimodal_cross_agent_workspace_profile_by_id(
    profile_id: str,
    registry: MultimodalCrossAgentWorkspaceRegistry | None = None,
) -> CrossAgentWorkspaceProfile | None:
    """Return one Cross-Agent Workspace profile without invoking agents."""

    source_registry = registry or MULTIMODAL_CROSS_AGENT_WORKSPACE_REGISTRY
    normalized_profile_id = str(profile_id).strip()
    for profile in source_registry.cross_agent_workspace_profiles:
        if profile.profile_id == normalized_profile_id:
            return profile
    return None


def multimodal_cross_agent_workspace_profiles_for_route(
    route: RouteName | str,
    registry: MultimodalCrossAgentWorkspaceRegistry | None = None,
) -> tuple[CrossAgentWorkspaceProfile, ...]:
    """Return passive Cross-Agent Workspace profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or MULTIMODAL_CROSS_AGENT_WORKSPACE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.cross_agent_workspace_profiles
        if route_name in profile.route_applicability
    )


def multimodal_cross_agent_workspace_profiles_for_surface_kind(
    surface_kind: CrossAgentWorkspaceSurfaceKind | str,
    registry: MultimodalCrossAgentWorkspaceRegistry | None = None,
) -> tuple[CrossAgentWorkspaceProfile, ...]:
    """Return Cross-Agent Workspace profiles for one passive surface kind."""

    surface_value = str(surface_kind).strip()
    source_registry = registry or MULTIMODAL_CROSS_AGENT_WORKSPACE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.cross_agent_workspace_profiles
        if profile.cross_agent_surface_kind == surface_value
    )


def multimodal_cross_agent_workspace_profiles_for_agent_workspace_profile(
    agent_workspace_profile_id: str,
    registry: MultimodalCrossAgentWorkspaceRegistry | None = None,
) -> tuple[CrossAgentWorkspaceProfile, ...]:
    """Return cross-agent profiles referencing one agent workspace profile."""

    source_registry = registry or MULTIMODAL_CROSS_AGENT_WORKSPACE_REGISTRY
    source_profile_id = str(agent_workspace_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.cross_agent_workspace_profiles
        if source_profile_id in profile.source_agent_workspace_profile_ids
    )


def _cross_agent_workspace_profile(
    *,
    profile_id: str,
    profile_name: str,
    cross_agent_workspace_kind: CrossAgentWorkspaceProfileKind,
    cross_agent_surface_kind: CrossAgentWorkspaceSurfaceKind,
    source_agent_workspace_profile_ids: tuple[str, ...],
    source_visual_workspace_profile_ids: tuple[str, ...],
    source_artifact_lineage_profile_ids: tuple[str, ...],
    source_shared_context_view_ids: tuple[str, ...],
    source_blackboard_channel_ids: tuple[str, ...],
    workspace_context_fields: tuple[str, ...],
    source_reference_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    cross_agent_workspace_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> CrossAgentWorkspaceProfile:
    return CrossAgentWorkspaceProfile(
        profile_id=profile_id,
        profile_name=profile_name,
        cross_agent_workspace_kind=cross_agent_workspace_kind,
        cross_agent_surface_kind=cross_agent_surface_kind,
        source_agent_workspace_profile_ids=source_agent_workspace_profile_ids,
        source_visual_workspace_profile_ids=source_visual_workspace_profile_ids,
        source_artifact_lineage_profile_ids=source_artifact_lineage_profile_ids,
        source_shared_context_view_ids=source_shared_context_view_ids,
        source_blackboard_channel_ids=source_blackboard_channel_ids,
        workspace_context_fields=workspace_context_fields,
        source_reference_ids=source_reference_ids,
        route_applicability=route_applicability,
        cross_agent_workspace_surfaces=cross_agent_workspace_surfaces,
        advisory_outputs=advisory_outputs,
        source_registries=_CROSS_AGENT_WORKSPACE_SOURCE_REGISTRIES,
        observability_surfaces=_CROSS_AGENT_WORKSPACE_OBSERVABILITY_SURFACES,
    )


MULTIMODAL_CROSS_AGENT_WORKSPACE_PROFILES = (
    _cross_agent_workspace_profile(
        profile_id="planning_cross_agent_workspace",
        profile_name="Planning Cross-Agent Workspace",
        cross_agent_workspace_kind="planning_cross_agent_workspace",
        cross_agent_surface_kind="planning_context",
        source_agent_workspace_profile_ids=("planning_context_agent_workspace",),
        source_visual_workspace_profile_ids=(
            "shell_visual_workspace",
            "inspector_visual_workspace",
        ),
        source_artifact_lineage_profile_ids=(
            "dependency_graph_artifact_lineage",
            "source_transition_artifact_lineage",
        ),
        source_shared_context_view_ids=(
            "planner_agent_shared_context_view",
            "research_agent_shared_context_view",
            "style_agent_shared_context_view",
        ),
        source_blackboard_channel_ids=(
            "planner_agent_blackboard_channel",
            "research_agent_blackboard_channel",
            "style_agent_blackboard_channel",
        ),
        workspace_context_fields=(
            "agent_workspace_profiles",
            "visible_blackboard_channel_ids",
            "planning_lineage_context",
            "workspace.focus",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_VISUAL_WORKSPACE_REGISTRY",
            "multimodal_studio.MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY",
            "hybrid_studio.AGENT_WORKSPACE_REGISTRY",
            "shared_context_views.SHARED_CONTEXT_VIEW_REGISTRY",
            "blackboard_memory.BLACKBOARD_MEMORY_REGISTRY",
            "hybrid_studio.AgentWorkspaceProfile",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.PREVIEW,
        ),
        cross_agent_workspace_surfaces=(
            "cross_agent_workspace_panel",
            "cross_agent_roster_surface",
            "shared_context_scope_surface",
            "lineage_context_surface",
            "cross_agent_workspace_boundary_panel",
        ),
        advisory_outputs=(
            "planning_cross_agent_workspace_inventory",
            "manual_planning_context_review_hint",
            "no_agent_invocation_notice",
        ),
    ),
    _cross_agent_workspace_profile(
        profile_id="artifact_runtime_cross_agent_workspace",
        profile_name="Artifact Runtime Cross-Agent Workspace",
        cross_agent_workspace_kind="artifact_runtime_cross_agent_workspace",
        cross_agent_surface_kind="artifact_runtime",
        source_agent_workspace_profile_ids=("artifact_runtime_agent_workspace",),
        source_visual_workspace_profile_ids=(
            "artifact_selection_visual_workspace",
            "preview_visual_workspace",
        ),
        source_artifact_lineage_profile_ids=(
            "dependency_graph_artifact_lineage",
            "source_transition_artifact_lineage",
        ),
        source_shared_context_view_ids=(
            "runtime_agent_shared_context_view",
            "artifact_agent_shared_context_view",
            "art_direction_agent_shared_context_view",
        ),
        source_blackboard_channel_ids=(
            "runtime_agent_blackboard_channel",
            "artifact_agent_blackboard_channel",
            "art_direction_agent_blackboard_channel",
        ),
        workspace_context_fields=(
            "artifact_runtime_agent_workspace",
            "artifact_lineage_context",
            "preview_workspace_surface",
            "blackboard_metadata_keys",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_VISUAL_WORKSPACE_REGISTRY",
            "multimodal_studio.MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY",
            "hybrid_studio.AGENT_WORKSPACE_REGISTRY",
            "shared_context_views.SharedContextViewContract",
            "blackboard_memory.BlackboardMemoryChannelContract",
            "clients.nextjs.workstation_shell.WorkstationShell",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.PREVIEW,
        ),
        cross_agent_workspace_surfaces=(
            "cross_agent_workspace_panel",
            "cross_agent_roster_surface",
            "shared_context_scope_surface",
            "blackboard_channel_surface",
            "lineage_context_surface",
            "cross_agent_workspace_boundary_panel",
        ),
        advisory_outputs=(
            "artifact_runtime_cross_agent_workspace_inventory",
            "manual_runtime_context_review_hint",
            "no_rendering_execution_notice",
        ),
    ),
    _cross_agent_workspace_profile(
        profile_id="critique_curation_cross_agent_workspace",
        profile_name="Critique Curation Cross-Agent Workspace",
        cross_agent_workspace_kind="critique_curation_cross_agent_workspace",
        cross_agent_surface_kind="critique_curation",
        source_agent_workspace_profile_ids=("critique_curation_agent_workspace",),
        source_visual_workspace_profile_ids=(
            "shell_visual_workspace",
            "inspector_visual_workspace",
        ),
        source_artifact_lineage_profile_ids=(
            "timeline_stage_artifact_lineage",
            "source_transition_artifact_lineage",
            "missing_artifact_lineage",
        ),
        source_shared_context_view_ids=(
            "aesthetic_critic_agent_shared_context_view",
            "narrative_symbolic_agent_shared_context_view",
            "creative_curator_agent_shared_context_view",
            "critic_agent_shared_context_view",
        ),
        source_blackboard_channel_ids=(
            "aesthetic_critic_agent_blackboard_channel",
            "narrative_symbolic_agent_blackboard_channel",
            "creative_curator_agent_blackboard_channel",
            "critic_agent_blackboard_channel",
        ),
        workspace_context_fields=(
            "critique_curation_agent_workspace",
            "timeline_stage_lineage",
            "missing_lineage_context",
            "visible_metadata_keys",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_VISUAL_WORKSPACE_REGISTRY",
            "multimodal_studio.MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY",
            "hybrid_studio.AGENT_WORKSPACE_REGISTRY",
            "shared_context_views.SHARED_CONTEXT_VIEW_REGISTRY",
            "blackboard_memory.BLACKBOARD_MEMORY_REGISTRY",
            "clients.nextjs.workstation_state.buildWorkstationState",
        ),
        route_applicability=(
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        cross_agent_workspace_surfaces=(
            "cross_agent_workspace_panel",
            "shared_context_scope_surface",
            "blackboard_channel_surface",
            "lineage_context_surface",
            "workspace_handoff_surface",
            "cross_agent_workspace_boundary_panel",
        ),
        advisory_outputs=(
            "critique_curation_cross_agent_workspace_inventory",
            "manual_critique_context_review_hint",
            "no_multi_agent_orchestration_notice",
        ),
    ),
    _cross_agent_workspace_profile(
        profile_id="refinement_synthesis_cross_agent_workspace",
        profile_name="Refinement Synthesis Cross-Agent Workspace",
        cross_agent_workspace_kind="refinement_synthesis_cross_agent_workspace",
        cross_agent_surface_kind="refinement_synthesis",
        source_agent_workspace_profile_ids=("refinement_synthesis_agent_workspace",),
        source_visual_workspace_profile_ids=(
            "artifact_selection_visual_workspace",
            "inspector_visual_workspace",
        ),
        source_artifact_lineage_profile_ids=(
            "timeline_stage_artifact_lineage",
            "missing_artifact_lineage",
        ),
        source_shared_context_view_ids=(
            "refiner_agent_shared_context_view",
            "final_synthesizer_agent_shared_context_view",
        ),
        source_blackboard_channel_ids=(
            "refiner_agent_blackboard_channel",
            "final_synthesizer_agent_blackboard_channel",
        ),
        workspace_context_fields=(
            "refinement_synthesis_agent_workspace",
            "final_payload_lineage_context",
            "shared_context_scope",
            "workspace.readiness",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_VISUAL_WORKSPACE_REGISTRY",
            "multimodal_studio.MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY",
            "hybrid_studio.AGENT_WORKSPACE_REGISTRY",
            "shared_context_views.SharedContextViewContract",
            "blackboard_memory.BlackboardMemoryChannelContract",
            "clients.nextjs.workstation_shell.WorkstationShell",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        cross_agent_workspace_surfaces=(
            "cross_agent_workspace_panel",
            "cross_agent_roster_surface",
            "shared_context_scope_surface",
            "blackboard_channel_surface",
            "workspace_handoff_surface",
            "cross_agent_workspace_boundary_panel",
        ),
        advisory_outputs=(
            "refinement_synthesis_cross_agent_workspace_inventory",
            "manual_synthesis_context_review_hint",
            "no_workspace_state_mutation_notice",
        ),
    ),
)

MULTIMODAL_CROSS_AGENT_WORKSPACE_REGISTRY = MultimodalCrossAgentWorkspaceRegistry(
    cross_agent_workspace_profiles=MULTIMODAL_CROSS_AGENT_WORKSPACE_PROFILES,
    profile_ids=tuple(
        profile.profile_id
        for profile in MULTIMODAL_CROSS_AGENT_WORKSPACE_PROFILES
    ),
    cross_agent_workspace_kinds=tuple(
        profile.cross_agent_workspace_kind
        for profile in MULTIMODAL_CROSS_AGENT_WORKSPACE_PROFILES
    ),
    cross_agent_surface_kinds=tuple(
        profile.cross_agent_surface_kind
        for profile in MULTIMODAL_CROSS_AGENT_WORKSPACE_PROFILES
    ),
    agent_workspace_profile_ids=AGENT_WORKSPACE_REGISTRY.workspace_profile_ids,
    visual_workspace_profile_ids=MULTIMODAL_VISUAL_WORKSPACE_REGISTRY.profile_ids,
    artifact_lineage_profile_ids=MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY.profile_ids,
    shared_context_view_ids=SHARED_CONTEXT_VIEW_REGISTRY.view_ids,
    blackboard_channel_ids=BLACKBOARD_MEMORY_REGISTRY.channel_ids,
    route_names=tuple(RouteName),
    profile_count=len(MULTIMODAL_CROSS_AGENT_WORKSPACE_PROFILES),
    source_registries=_CROSS_AGENT_WORKSPACE_SOURCE_REGISTRIES,
    source_reference_ids=_CROSS_AGENT_WORKSPACE_SOURCE_REFERENCES,
    cross_agent_workspace_surface_refs=_CROSS_AGENT_WORKSPACE_SURFACES,
    observability_surfaces=_CROSS_AGENT_WORKSPACE_OBSERVABILITY_SURFACES,
)


class SharedArtifactBoardProfile(BaseModel):
    """Inspectable metadata for one passive Shared Artifact Board surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1, max_length=140)
    profile_name: str = Field(min_length=1, max_length=160)
    board_profile_kind: SharedArtifactBoardProfileKind
    board_surface_kind: SharedArtifactBoardSurfaceKind
    source_cross_agent_workspace_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=3,
    )
    source_artifact_collaboration_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_multi_preview_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_artifact_provenance_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_artifact_lineage_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    board_context_fields: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_reference_ids: tuple[str, ...] = Field(min_length=1, max_length=10)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    shared_artifact_board_surfaces: tuple[str, ...] = Field(
        min_length=1,
        max_length=7,
    )
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_registries: tuple[str, ...] = Field(min_length=8, max_length=8)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=SHARED_ARTIFACT_BOARD_AUTHORITY_BOUNDARY,
        max_length=1300,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_SHARED_ARTIFACT_BOARD_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    board_state_creation_implemented: Literal[False] = False
    collaborative_board_persistence_implemented: Literal[False] = False
    artifact_selection_mutation_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    shared_context_materialization_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    serialization_version: Literal["multimodal_shared_artifact_board_profile.v1"] = (
        SHARED_ARTIFACT_BOARD_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class MultimodalSharedArtifactBoardRegistry(BaseModel):
    """Stable passive registry for V4.5 Shared Artifact Board metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["multimodal_shared_artifact_board_registry"] = (
        "multimodal_shared_artifact_board_registry"
    )
    serialization_version: Literal["multimodal_shared_artifact_board_registry.v1"] = (
        SHARED_ARTIFACT_BOARD_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=SHARED_ARTIFACT_BOARD_AUTHORITY_BOUNDARY,
        max_length=1300,
    )
    shared_artifact_board_profiles: tuple[SharedArtifactBoardProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    board_profile_kinds: tuple[SharedArtifactBoardProfileKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    board_surface_kinds: tuple[SharedArtifactBoardSurfaceKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    cross_agent_workspace_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    artifact_collaboration_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    multi_preview_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    artifact_provenance_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    artifact_lineage_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=8, max_length=8)
    source_reference_ids: tuple[str, ...] = Field(min_length=10, max_length=10)
    shared_artifact_board_surface_refs: tuple[str, ...] = Field(
        min_length=7,
        max_length=7,
    )
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_SHARED_ARTIFACT_BOARD_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    board_state_creation_implemented: Literal[False] = False
    collaborative_board_persistence_implemented: Literal[False] = False
    artifact_selection_mutation_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    shared_context_materialization_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.profile_id for profile in self.shared_artifact_board_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("profile_ids must be unique")
        if self.profile_ids != derived_profile_ids:
            raise ValueError("profile_ids must match shared_artifact_board_profiles")
        if self.profile_count != len(self.shared_artifact_board_profiles):
            raise ValueError("profile_count must match shared_artifact_board_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if (
            self.cross_agent_workspace_profile_ids
            != MULTIMODAL_CROSS_AGENT_WORKSPACE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "cross_agent_workspace_profile_ids must match Cross-Agent Workspace registry"
            )
        if (
            self.artifact_collaboration_profile_ids
            != MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY.profile_ids
        ):
            raise ValueError(
                "artifact_collaboration_profile_ids must match Artifact Collaboration registry"
            )
        if self.multi_preview_profile_ids != MULTIMODAL_MULTI_PREVIEW_REGISTRY.profile_ids:
            raise ValueError(
                "multi_preview_profile_ids must match Multi Preview registry"
            )
        if (
            self.artifact_provenance_profile_ids
            != MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "artifact_provenance_profile_ids must match Artifact Provenance registry"
            )
        if (
            self.artifact_lineage_profile_ids
            != MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "artifact_lineage_profile_ids must match Artifact Lineage registry"
            )
        if self.board_profile_kinds != _ordered_unique(
            profile.board_profile_kind
            for profile in self.shared_artifact_board_profiles
        ):
            raise ValueError("board_profile_kinds must match profiles")
        if self.board_surface_kinds != _ordered_unique(
            profile.board_surface_kind
            for profile in self.shared_artifact_board_profiles
        ):
            raise ValueError("board_surface_kinds must match profiles")

        profile_source_references = {
            source_reference
            for profile in self.shared_artifact_board_profiles
            for source_reference in profile.source_reference_ids
        }
        if set(self.source_reference_ids) != profile_source_references:
            raise ValueError(
                "source_reference_ids must match profile source references"
            )

        known_routes = set(self.route_names)
        known_cross_agent_workspaces = set(self.cross_agent_workspace_profile_ids)
        known_artifact_collaboration_profiles = set(
            self.artifact_collaboration_profile_ids
        )
        known_multi_preview_profiles = set(self.multi_preview_profile_ids)
        known_provenance_profiles = set(self.artifact_provenance_profile_ids)
        known_lineage_profiles = set(self.artifact_lineage_profile_ids)
        known_surfaces = set(self.shared_artifact_board_surface_refs)
        known_source_references = set(self.source_reference_ids)
        for profile in self.shared_artifact_board_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must use known routes")
            if not set(profile.source_cross_agent_workspace_profile_ids).issubset(
                known_cross_agent_workspaces
            ):
                raise ValueError(
                    "source_cross_agent_workspace_profile_ids must be known profiles"
                )
            if not set(profile.source_artifact_collaboration_profile_ids).issubset(
                known_artifact_collaboration_profiles
            ):
                raise ValueError(
                    "source_artifact_collaboration_profile_ids must be known profiles"
                )
            if not set(profile.source_multi_preview_profile_ids).issubset(
                known_multi_preview_profiles
            ):
                raise ValueError(
                    "source_multi_preview_profile_ids must be known profiles"
                )
            if not set(profile.source_artifact_provenance_profile_ids).issubset(
                known_provenance_profiles
            ):
                raise ValueError(
                    "source_artifact_provenance_profile_ids must be known profiles"
                )
            if not set(profile.source_artifact_lineage_profile_ids).issubset(
                known_lineage_profiles
            ):
                raise ValueError(
                    "source_artifact_lineage_profile_ids must be known profiles"
                )
            if not set(profile.shared_artifact_board_surfaces).issubset(
                known_surfaces
            ):
                raise ValueError(
                    "shared_artifact_board_surfaces must be known surfaces"
                )
            if not set(profile.source_reference_ids).issubset(
                known_source_references
            ):
                raise ValueError(
                    "source_reference_ids must be known registry references"
                )
        return self


def multimodal_shared_artifact_board_registry() -> (
    MultimodalSharedArtifactBoardRegistry
):
    """Return passive V4.5 Shared Artifact Board metadata."""

    return MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY


def multimodal_shared_artifact_board_profile_by_id(
    profile_id: str,
    registry: MultimodalSharedArtifactBoardRegistry | None = None,
) -> SharedArtifactBoardProfile | None:
    """Return one Shared Artifact Board profile without mutating artifacts."""

    source_registry = registry or MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY
    normalized_profile_id = str(profile_id).strip()
    for profile in source_registry.shared_artifact_board_profiles:
        if profile.profile_id == normalized_profile_id:
            return profile
    return None


def multimodal_shared_artifact_board_profiles_for_route(
    route: RouteName | str,
    registry: MultimodalSharedArtifactBoardRegistry | None = None,
) -> tuple[SharedArtifactBoardProfile, ...]:
    """Return passive Shared Artifact Board profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY
    return tuple(
        profile
        for profile in source_registry.shared_artifact_board_profiles
        if route_name in profile.route_applicability
    )


def multimodal_shared_artifact_board_profiles_for_surface_kind(
    surface_kind: SharedArtifactBoardSurfaceKind | str,
    registry: MultimodalSharedArtifactBoardRegistry | None = None,
) -> tuple[SharedArtifactBoardProfile, ...]:
    """Return Shared Artifact Board profiles for one passive surface kind."""

    surface_value = str(surface_kind).strip()
    source_registry = registry or MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY
    return tuple(
        profile
        for profile in source_registry.shared_artifact_board_profiles
        if profile.board_surface_kind == surface_value
    )


def multimodal_shared_artifact_board_profiles_for_cross_agent_workspace_profile(
    cross_agent_workspace_profile_id: str,
    registry: MultimodalSharedArtifactBoardRegistry | None = None,
) -> tuple[SharedArtifactBoardProfile, ...]:
    """Return board profiles referencing one cross-agent workspace profile."""

    source_registry = registry or MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY
    source_profile_id = str(cross_agent_workspace_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.shared_artifact_board_profiles
        if source_profile_id in profile.source_cross_agent_workspace_profile_ids
    )


def _shared_artifact_board_profile(
    *,
    profile_id: str,
    profile_name: str,
    board_profile_kind: SharedArtifactBoardProfileKind,
    board_surface_kind: SharedArtifactBoardSurfaceKind,
    source_cross_agent_workspace_profile_ids: tuple[str, ...],
    source_artifact_collaboration_profile_ids: tuple[str, ...],
    source_multi_preview_profile_ids: tuple[str, ...],
    source_artifact_provenance_profile_ids: tuple[str, ...],
    source_artifact_lineage_profile_ids: tuple[str, ...],
    board_context_fields: tuple[str, ...],
    source_reference_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    shared_artifact_board_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> SharedArtifactBoardProfile:
    return SharedArtifactBoardProfile(
        profile_id=profile_id,
        profile_name=profile_name,
        board_profile_kind=board_profile_kind,
        board_surface_kind=board_surface_kind,
        source_cross_agent_workspace_profile_ids=(
            source_cross_agent_workspace_profile_ids
        ),
        source_artifact_collaboration_profile_ids=(
            source_artifact_collaboration_profile_ids
        ),
        source_multi_preview_profile_ids=source_multi_preview_profile_ids,
        source_artifact_provenance_profile_ids=(
            source_artifact_provenance_profile_ids
        ),
        source_artifact_lineage_profile_ids=source_artifact_lineage_profile_ids,
        board_context_fields=board_context_fields,
        source_reference_ids=source_reference_ids,
        route_applicability=route_applicability,
        shared_artifact_board_surfaces=shared_artifact_board_surfaces,
        advisory_outputs=advisory_outputs,
        source_registries=_SHARED_ARTIFACT_BOARD_SOURCE_REGISTRIES,
        observability_surfaces=_SHARED_ARTIFACT_BOARD_OBSERVABILITY_SURFACES,
    )


MULTIMODAL_SHARED_ARTIFACT_BOARD_PROFILES = (
    _shared_artifact_board_profile(
        profile_id="selection_shared_artifact_board",
        profile_name="Selection Shared Artifact Board",
        board_profile_kind="artifact_selection_board",
        board_surface_kind="selection",
        source_cross_agent_workspace_profile_ids=(
            "planning_cross_agent_workspace",
            "artifact_runtime_cross_agent_workspace",
        ),
        source_artifact_collaboration_profile_ids=(
            "selection_artifact_collaboration",
            "inspection_artifact_collaboration",
        ),
        source_multi_preview_profile_ids=(
            "candidate_grid_multi_preview",
            "recommended_candidate_multi_preview",
        ),
        source_artifact_provenance_profile_ids=(
            "payload_artifact_provenance",
            "evidence_artifact_provenance",
        ),
        source_artifact_lineage_profile_ids=(
            "dependency_graph_artifact_lineage",
            "source_transition_artifact_lineage",
        ),
        board_context_fields=(
            "activeArtifactId",
            "artifacts",
            "artifact_selection_surface",
            "sourceKeys",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_CROSS_AGENT_WORKSPACE_REGISTRY",
            "multimodal_studio.MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY",
            "multimodal_studio.MULTIMODAL_MULTI_PREVIEW_REGISTRY",
            "clients.nextjs.artifact_comparison.buildArtifactComparisonModel",
            "clients.nextjs.workstation_shell.WorkstationShell",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.PREVIEW,
        ),
        shared_artifact_board_surfaces=(
            "shared_artifact_board_panel",
            "artifact_selection_board_surface",
            "artifact_comparison_board_surface",
            "shared_artifact_board_boundary_panel",
        ),
        advisory_outputs=(
            "artifact_selection_board_inventory",
            "manual_selection_review_hint",
            "no_artifact_selection_mutation_notice",
        ),
    ),
    _shared_artifact_board_profile(
        profile_id="comparison_shared_artifact_board",
        profile_name="Comparison Shared Artifact Board",
        board_profile_kind="comparison_review_board",
        board_surface_kind="comparison",
        source_cross_agent_workspace_profile_ids=(
            "artifact_runtime_cross_agent_workspace",
            "critique_curation_cross_agent_workspace",
        ),
        source_artifact_collaboration_profile_ids=(
            "comparison_artifact_collaboration",
            "selection_artifact_collaboration",
        ),
        source_multi_preview_profile_ids=(
            "split_comparison_multi_preview",
            "candidate_grid_multi_preview",
        ),
        source_artifact_provenance_profile_ids=(
            "evaluation_artifact_provenance",
            "payload_artifact_provenance",
        ),
        source_artifact_lineage_profile_ids=(
            "dependency_graph_artifact_lineage",
            "source_transition_artifact_lineage",
        ),
        board_context_fields=(
            "comparison.rows",
            "recommendedRow",
            "artifact.runtimeSupport",
            "qualityRank",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_CROSS_AGENT_WORKSPACE_REGISTRY",
            "multimodal_studio.MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY",
            "multimodal_studio.MULTIMODAL_MULTI_PREVIEW_REGISTRY",
            "clients.nextjs.artifact_comparison.buildArtifactComparisonModel",
            "clients.nextjs.artifact_comparison.ArtifactComparisonRow",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DESIGN,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        shared_artifact_board_surfaces=(
            "shared_artifact_board_panel",
            "artifact_comparison_board_surface",
            "artifact_lineage_board_surface",
            "shared_artifact_board_boundary_panel",
        ),
        advisory_outputs=(
            "artifact_comparison_board_inventory",
            "manual_comparison_review_hint",
            "no_rendering_execution_notice",
        ),
    ),
    _shared_artifact_board_profile(
        profile_id="provenance_lineage_shared_artifact_board",
        profile_name="Provenance Lineage Shared Artifact Board",
        board_profile_kind="provenance_lineage_board",
        board_surface_kind="provenance_lineage",
        source_cross_agent_workspace_profile_ids=(
            "critique_curation_cross_agent_workspace",
            "refinement_synthesis_cross_agent_workspace",
        ),
        source_artifact_collaboration_profile_ids=(
            "inspection_artifact_collaboration",
            "comparison_artifact_collaboration",
        ),
        source_multi_preview_profile_ids=(
            "fallback_multi_preview",
            "recommended_candidate_multi_preview",
        ),
        source_artifact_provenance_profile_ids=(
            "evidence_artifact_provenance",
            "evaluation_artifact_provenance",
            "missing_source_artifact_provenance",
        ),
        source_artifact_lineage_profile_ids=(
            "timeline_stage_artifact_lineage",
            "source_transition_artifact_lineage",
            "missing_artifact_lineage",
        ),
        board_context_fields=(
            "artifact_provenance_surfaces",
            "artifact_lineage_surfaces",
            "unsupported_or_missing_sources",
            "eventSequence",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY",
            "multimodal_studio.MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY",
            "clients.nextjs.artifact_inspector.buildArtifactDocument",
            "clients.nextjs.artifact_inspector.highlightArtifactDocument",
            "clients.nextjs.workstation_shell.WorkstationShell",
        ),
        route_applicability=(
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        shared_artifact_board_surfaces=(
            "shared_artifact_board_panel",
            "artifact_provenance_board_surface",
            "artifact_lineage_board_surface",
            "shared_artifact_board_boundary_panel",
        ),
        advisory_outputs=(
            "provenance_lineage_board_inventory",
            "manual_provenance_review_hint",
            "no_collaborative_board_persistence_notice",
        ),
    ),
    _shared_artifact_board_profile(
        profile_id="handoff_refinement_shared_artifact_board",
        profile_name="Handoff Refinement Shared Artifact Board",
        board_profile_kind="handoff_refinement_board",
        board_surface_kind="handoff_refinement",
        source_cross_agent_workspace_profile_ids=(
            "artifact_runtime_cross_agent_workspace",
            "refinement_synthesis_cross_agent_workspace",
        ),
        source_artifact_collaboration_profile_ids=(
            "refinement_artifact_collaboration",
            "inspection_artifact_collaboration",
        ),
        source_multi_preview_profile_ids=(
            "recommended_candidate_multi_preview",
            "fallback_multi_preview",
        ),
        source_artifact_provenance_profile_ids=(
            "evaluation_artifact_provenance",
            "payload_artifact_provenance",
            "missing_source_artifact_provenance",
        ),
        source_artifact_lineage_profile_ids=(
            "timeline_stage_artifact_lineage",
            "missing_artifact_lineage",
        ),
        board_context_fields=(
            "artifact_refinement_surface",
            "artifact_document",
            "handoff_context",
            "final_payload_lineage_context",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_CROSS_AGENT_WORKSPACE_REGISTRY",
            "multimodal_studio.MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY",
            "clients.nextjs.artifact_inspector.buildArtifactDocument",
            "clients.nextjs.artifact_inspector.highlightArtifactDocument",
            "clients.nextjs.workstation_shell.WorkstationShell",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        shared_artifact_board_surfaces=(
            "shared_artifact_board_panel",
            "artifact_handoff_board_surface",
            "artifact_provenance_board_surface",
            "shared_artifact_board_boundary_panel",
        ),
        advisory_outputs=(
            "handoff_refinement_board_inventory",
            "manual_refinement_handoff_review_hint",
            "no_artifact_mutation_notice",
        ),
    ),
)

MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY = MultimodalSharedArtifactBoardRegistry(
    shared_artifact_board_profiles=MULTIMODAL_SHARED_ARTIFACT_BOARD_PROFILES,
    profile_ids=tuple(
        profile.profile_id
        for profile in MULTIMODAL_SHARED_ARTIFACT_BOARD_PROFILES
    ),
    board_profile_kinds=tuple(
        profile.board_profile_kind
        for profile in MULTIMODAL_SHARED_ARTIFACT_BOARD_PROFILES
    ),
    board_surface_kinds=tuple(
        profile.board_surface_kind
        for profile in MULTIMODAL_SHARED_ARTIFACT_BOARD_PROFILES
    ),
    cross_agent_workspace_profile_ids=(
        MULTIMODAL_CROSS_AGENT_WORKSPACE_REGISTRY.profile_ids
    ),
    artifact_collaboration_profile_ids=(
        MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY.profile_ids
    ),
    multi_preview_profile_ids=MULTIMODAL_MULTI_PREVIEW_REGISTRY.profile_ids,
    artifact_provenance_profile_ids=(
        MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY.profile_ids
    ),
    artifact_lineage_profile_ids=MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY.profile_ids,
    route_names=tuple(RouteName),
    profile_count=len(MULTIMODAL_SHARED_ARTIFACT_BOARD_PROFILES),
    source_registries=_SHARED_ARTIFACT_BOARD_SOURCE_REGISTRIES,
    source_reference_ids=_SHARED_ARTIFACT_BOARD_SOURCE_REFERENCES,
    shared_artifact_board_surface_refs=_SHARED_ARTIFACT_BOARD_SURFACES,
    observability_surfaces=_SHARED_ARTIFACT_BOARD_OBSERVABILITY_SURFACES,
)


class WorkspaceHistoryProfile(BaseModel):
    """Inspectable metadata for one passive Workspace History surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1, max_length=140)
    profile_name: str = Field(min_length=1, max_length=160)
    history_profile_kind: WorkspaceHistoryProfileKind
    history_surface_kind: WorkspaceHistorySurfaceKind
    source_shared_artifact_board_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_runtime_collaboration_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_workspace_snapshot_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_session_replay_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    history_context_fields: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_reference_ids: tuple[str, ...] = Field(min_length=1, max_length=10)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    workspace_history_surfaces: tuple[str, ...] = Field(
        min_length=1,
        max_length=7,
    )
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_registries: tuple[str, ...] = Field(min_length=8, max_length=8)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=WORKSPACE_HISTORY_AUTHORITY_BOUNDARY,
        max_length=1300,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_WORKSPACE_HISTORY_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    history_recording_implemented: Literal[False] = False
    snapshot_capture_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    history_persistence_implemented: Literal[False] = False
    session_replay_execution_implemented: Literal[False] = False
    runtime_event_replay_implemented: Literal[False] = False
    workspace_state_mutation_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    serialization_version: Literal["multimodal_workspace_history_profile.v1"] = (
        WORKSPACE_HISTORY_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class MultimodalWorkspaceHistoryRegistry(BaseModel):
    """Stable passive registry for V4.5 Workspace History metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["multimodal_workspace_history_registry"] = (
        "multimodal_workspace_history_registry"
    )
    serialization_version: Literal["multimodal_workspace_history_registry.v1"] = (
        WORKSPACE_HISTORY_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=WORKSPACE_HISTORY_AUTHORITY_BOUNDARY,
        max_length=1300,
    )
    workspace_history_profiles: tuple[WorkspaceHistoryProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    history_profile_kinds: tuple[WorkspaceHistoryProfileKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    history_surface_kinds: tuple[WorkspaceHistorySurfaceKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    shared_artifact_board_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    runtime_collaboration_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    workspace_snapshot_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    session_replay_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=8, max_length=8)
    source_reference_ids: tuple[str, ...] = Field(min_length=10, max_length=10)
    workspace_history_surface_refs: tuple[str, ...] = Field(
        min_length=7,
        max_length=7,
    )
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_WORKSPACE_HISTORY_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    history_recording_implemented: Literal[False] = False
    snapshot_capture_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    history_persistence_implemented: Literal[False] = False
    session_replay_execution_implemented: Literal[False] = False
    runtime_event_replay_implemented: Literal[False] = False
    workspace_state_mutation_implemented: Literal[False] = False
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
            profile.profile_id for profile in self.workspace_history_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("profile_ids must be unique")
        if self.profile_ids != derived_profile_ids:
            raise ValueError("profile_ids must match workspace_history_profiles")
        if self.profile_count != len(self.workspace_history_profiles):
            raise ValueError("profile_count must match workspace_history_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if (
            self.shared_artifact_board_profile_ids
            != MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY.profile_ids
        ):
            raise ValueError(
                "shared_artifact_board_profile_ids must match Shared Artifact Board registry"
            )
        if (
            self.runtime_collaboration_profile_ids
            != MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY.profile_ids
        ):
            raise ValueError(
                "runtime_collaboration_profile_ids must match Runtime Collaboration registry"
            )
        if (
            self.workspace_snapshot_profile_ids
            != WORKSPACE_SNAPSHOT_REGISTRY.workspace_snapshot_profile_ids
        ):
            raise ValueError(
                "workspace_snapshot_profile_ids must match Workspace Snapshot registry"
            )
        if (
            self.session_replay_profile_ids
            != SESSION_REPLAY_REGISTRY.session_replay_profile_ids
        ):
            raise ValueError(
                "session_replay_profile_ids must match Session Replay registry"
            )
        if self.history_profile_kinds != _ordered_unique(
            profile.history_profile_kind
            for profile in self.workspace_history_profiles
        ):
            raise ValueError("history_profile_kinds must match profiles")
        if self.history_surface_kinds != _ordered_unique(
            profile.history_surface_kind
            for profile in self.workspace_history_profiles
        ):
            raise ValueError("history_surface_kinds must match profiles")

        profile_source_references = {
            source_reference
            for profile in self.workspace_history_profiles
            for source_reference in profile.source_reference_ids
        }
        if set(self.source_reference_ids) != profile_source_references:
            raise ValueError(
                "source_reference_ids must match profile source references"
            )

        known_routes = set(self.route_names)
        known_boards = set(self.shared_artifact_board_profile_ids)
        known_runtime_profiles = set(self.runtime_collaboration_profile_ids)
        known_snapshots = set(self.workspace_snapshot_profile_ids)
        known_replays = set(self.session_replay_profile_ids)
        known_surfaces = set(self.workspace_history_surface_refs)
        known_source_references = set(self.source_reference_ids)
        for profile in self.workspace_history_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must use known routes")
            if not set(profile.source_shared_artifact_board_profile_ids).issubset(
                known_boards
            ):
                raise ValueError(
                    "source_shared_artifact_board_profile_ids must be known profiles"
                )
            if not set(profile.source_runtime_collaboration_profile_ids).issubset(
                known_runtime_profiles
            ):
                raise ValueError(
                    "source_runtime_collaboration_profile_ids must be known profiles"
                )
            if not set(profile.source_workspace_snapshot_profile_ids).issubset(
                known_snapshots
            ):
                raise ValueError(
                    "source_workspace_snapshot_profile_ids must be known profiles"
                )
            if not set(profile.source_session_replay_profile_ids).issubset(
                known_replays
            ):
                raise ValueError(
                    "source_session_replay_profile_ids must be known profiles"
                )
            if not set(profile.workspace_history_surfaces).issubset(known_surfaces):
                raise ValueError("workspace_history_surfaces must be known surfaces")
            if not set(profile.source_reference_ids).issubset(
                known_source_references
            ):
                raise ValueError(
                    "source_reference_ids must be known registry references"
                )
        return self


def multimodal_workspace_history_registry() -> MultimodalWorkspaceHistoryRegistry:
    """Return passive V4.5 Workspace History metadata."""

    return MULTIMODAL_WORKSPACE_HISTORY_REGISTRY


def multimodal_workspace_history_profile_by_id(
    profile_id: str,
    registry: MultimodalWorkspaceHistoryRegistry | None = None,
) -> WorkspaceHistoryProfile | None:
    """Return one Workspace History profile without recording history."""

    source_registry = registry or MULTIMODAL_WORKSPACE_HISTORY_REGISTRY
    normalized_profile_id = str(profile_id).strip()
    for profile in source_registry.workspace_history_profiles:
        if profile.profile_id == normalized_profile_id:
            return profile
    return None


def multimodal_workspace_history_profiles_for_route(
    route: RouteName | str,
    registry: MultimodalWorkspaceHistoryRegistry | None = None,
) -> tuple[WorkspaceHistoryProfile, ...]:
    """Return passive Workspace History profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or MULTIMODAL_WORKSPACE_HISTORY_REGISTRY
    return tuple(
        profile
        for profile in source_registry.workspace_history_profiles
        if route_name in profile.route_applicability
    )


def multimodal_workspace_history_profiles_for_surface_kind(
    surface_kind: WorkspaceHistorySurfaceKind | str,
    registry: MultimodalWorkspaceHistoryRegistry | None = None,
) -> tuple[WorkspaceHistoryProfile, ...]:
    """Return Workspace History profiles for one passive surface kind."""

    surface_value = str(surface_kind).strip()
    source_registry = registry or MULTIMODAL_WORKSPACE_HISTORY_REGISTRY
    return tuple(
        profile
        for profile in source_registry.workspace_history_profiles
        if profile.history_surface_kind == surface_value
    )


def multimodal_workspace_history_profiles_for_workspace_snapshot_profile(
    workspace_snapshot_profile_id: str,
    registry: MultimodalWorkspaceHistoryRegistry | None = None,
) -> tuple[WorkspaceHistoryProfile, ...]:
    """Return history profiles referencing one workspace snapshot profile."""

    source_registry = registry or MULTIMODAL_WORKSPACE_HISTORY_REGISTRY
    source_profile_id = str(workspace_snapshot_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.workspace_history_profiles
        if source_profile_id in profile.source_workspace_snapshot_profile_ids
    )


def _workspace_history_profile(
    *,
    profile_id: str,
    profile_name: str,
    history_profile_kind: WorkspaceHistoryProfileKind,
    history_surface_kind: WorkspaceHistorySurfaceKind,
    source_shared_artifact_board_profile_ids: tuple[str, ...],
    source_runtime_collaboration_profile_ids: tuple[str, ...],
    source_workspace_snapshot_profile_ids: tuple[str, ...],
    source_session_replay_profile_ids: tuple[str, ...],
    history_context_fields: tuple[str, ...],
    source_reference_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    workspace_history_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> WorkspaceHistoryProfile:
    return WorkspaceHistoryProfile(
        profile_id=profile_id,
        profile_name=profile_name,
        history_profile_kind=history_profile_kind,
        history_surface_kind=history_surface_kind,
        source_shared_artifact_board_profile_ids=(
            source_shared_artifact_board_profile_ids
        ),
        source_runtime_collaboration_profile_ids=(
            source_runtime_collaboration_profile_ids
        ),
        source_workspace_snapshot_profile_ids=source_workspace_snapshot_profile_ids,
        source_session_replay_profile_ids=source_session_replay_profile_ids,
        history_context_fields=history_context_fields,
        source_reference_ids=source_reference_ids,
        route_applicability=route_applicability,
        workspace_history_surfaces=workspace_history_surfaces,
        advisory_outputs=advisory_outputs,
        source_registries=_WORKSPACE_HISTORY_SOURCE_REGISTRIES,
        observability_surfaces=_WORKSPACE_HISTORY_OBSERVABILITY_SURFACES,
    )


MULTIMODAL_WORKSPACE_HISTORY_PROFILES = (
    _workspace_history_profile(
        profile_id="session_record_workspace_history",
        profile_name="Session Record Workspace History",
        history_profile_kind="session_record_history",
        history_surface_kind="session_record",
        source_shared_artifact_board_profile_ids=(
            "selection_shared_artifact_board",
            "comparison_shared_artifact_board",
        ),
        source_runtime_collaboration_profile_ids=(
            "stream_event_runtime_collaboration",
            "operator_context_runtime_collaboration",
        ),
        source_workspace_snapshot_profile_ids=(
            "studio_overview_workspace_snapshot",
            "agent_context_workspace_snapshot",
        ),
        source_session_replay_profile_ids=(
            "session_overview_replay_profile",
            "conversation_timeline_replay_profile",
        ),
        history_context_fields=(
            "WorkspaceSessionRecord",
            "createdAt",
            "updatedAt",
            "snapshot.session",
            "messages",
        ),
        source_reference_ids=(
            "hybrid_studio.SESSION_REPLAY_REGISTRY",
            "clients.nextjs.workspace_persistence.WorkspaceSessionRecord",
            "clients.nextjs.workspace_persistence.createWorkspaceSessionRecord",
            "clients.nextjs.workspace_persistence.snapshotFromWorkspaceSessionRecord",
            "clients.nextjs.workstation_shell.WorkstationShell",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.PREVIEW,
        ),
        workspace_history_surfaces=(
            "workspace_history_panel",
            "session_record_history_surface",
            "history_summary_surface",
            "workspace_history_boundary_panel",
        ),
        advisory_outputs=(
            "session_record_history_inventory",
            "manual_session_history_review_hint",
            "no_history_recording_notice",
        ),
    ),
    _workspace_history_profile(
        profile_id="snapshot_workspace_history",
        profile_name="Snapshot Workspace History",
        history_profile_kind="snapshot_history",
        history_surface_kind="snapshot",
        source_shared_artifact_board_profile_ids=(
            "provenance_lineage_shared_artifact_board",
            "handoff_refinement_shared_artifact_board",
        ),
        source_runtime_collaboration_profile_ids=(
            "trace_runtime_collaboration",
            "operator_context_runtime_collaboration",
        ),
        source_workspace_snapshot_profile_ids=tuple(
            WORKSPACE_SNAPSHOT_REGISTRY.workspace_snapshot_profile_ids
        ),
        source_session_replay_profile_ids=(
            "snapshot_transition_replay_profile",
            "review_decision_replay_profile",
        ),
        history_context_fields=(
            "workspace_snapshot_profile_ids",
            "snapshot_context_fields",
            "snapshot_surfaces",
            "snapshot_transition_refs",
        ),
        source_reference_ids=(
            "hybrid_studio.WORKSPACE_SNAPSHOT_REGISTRY",
            "hybrid_studio.SESSION_REPLAY_REGISTRY",
            "clients.nextjs.workspace_persistence.createWorkspaceSessionRecord",
            "clients.nextjs.workspace_persistence.snapshotFromWorkspaceSessionRecord",
            "clients.nextjs.creative_timeline.buildCreativeTimelineModel",
        ),
        route_applicability=(
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        workspace_history_surfaces=(
            "workspace_history_panel",
            "snapshot_history_surface",
            "history_summary_surface",
            "workspace_history_boundary_panel",
        ),
        advisory_outputs=(
            "snapshot_history_inventory",
            "manual_snapshot_history_review_hint",
            "no_snapshot_capture_notice",
        ),
    ),
    _workspace_history_profile(
        profile_id="artifact_board_workspace_history",
        profile_name="Artifact Board Workspace History",
        history_profile_kind="artifact_board_history",
        history_surface_kind="artifact_board",
        source_shared_artifact_board_profile_ids=tuple(
            MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY.profile_ids
        ),
        source_runtime_collaboration_profile_ids=(
            "trace_runtime_collaboration",
            "stream_event_runtime_collaboration",
        ),
        source_workspace_snapshot_profile_ids=(
            "execution_context_workspace_snapshot",
            "review_audit_workspace_snapshot",
        ),
        source_session_replay_profile_ids=(
            "snapshot_transition_replay_profile",
            "review_decision_replay_profile",
        ),
        history_context_fields=(
            "shared_artifact_board_surfaces",
            "activeArtifactId",
            "artifact_lineage_refs",
            "artifact_provenance_refs",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY",
            "multimodal_studio.MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY",
            "clients.nextjs.workspace_persistence.WorkspaceSessionRecord",
            "clients.nextjs.workstation_shell.WorkstationShell",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        workspace_history_surfaces=(
            "workspace_history_panel",
            "artifact_board_history_surface",
            "history_summary_surface",
            "workspace_history_boundary_panel",
        ),
        advisory_outputs=(
            "artifact_board_history_inventory",
            "manual_artifact_history_review_hint",
            "no_artifact_mutation_notice",
        ),
    ),
    _workspace_history_profile(
        profile_id="runtime_event_workspace_history",
        profile_name="Runtime Event Workspace History",
        history_profile_kind="runtime_event_history",
        history_surface_kind="runtime_event",
        source_shared_artifact_board_profile_ids=(
            "comparison_shared_artifact_board",
            "provenance_lineage_shared_artifact_board",
        ),
        source_runtime_collaboration_profile_ids=tuple(
            MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY.profile_ids
        ),
        source_workspace_snapshot_profile_ids=(
            "execution_context_workspace_snapshot",
            "review_audit_workspace_snapshot",
        ),
        source_session_replay_profile_ids=(
            "conversation_timeline_replay_profile",
            "snapshot_transition_replay_profile",
        ),
        history_context_fields=(
            "WorkflowRuntimeTraceEvent",
            "workflow.timeline",
            "creativeTimeline.events",
            "eventSequence",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY",
            "hybrid_studio.SESSION_REPLAY_REGISTRY",
            "clients.nextjs.creative_timeline.buildCreativeTimelineModel",
            "clients.nextjs.workflow_runtime.WorkflowRuntimeTraceEvent",
            "clients.nextjs.workstation_shell.WorkstationShell",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        workspace_history_surfaces=(
            "workspace_history_panel",
            "runtime_event_history_surface",
            "history_summary_surface",
            "workspace_history_boundary_panel",
        ),
        advisory_outputs=(
            "runtime_event_history_inventory",
            "manual_runtime_history_review_hint",
            "no_runtime_event_replay_notice",
        ),
    ),
)

MULTIMODAL_WORKSPACE_HISTORY_REGISTRY = MultimodalWorkspaceHistoryRegistry(
    workspace_history_profiles=MULTIMODAL_WORKSPACE_HISTORY_PROFILES,
    profile_ids=tuple(
        profile.profile_id for profile in MULTIMODAL_WORKSPACE_HISTORY_PROFILES
    ),
    history_profile_kinds=tuple(
        profile.history_profile_kind
        for profile in MULTIMODAL_WORKSPACE_HISTORY_PROFILES
    ),
    history_surface_kinds=tuple(
        profile.history_surface_kind
        for profile in MULTIMODAL_WORKSPACE_HISTORY_PROFILES
    ),
    shared_artifact_board_profile_ids=(
        MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY.profile_ids
    ),
    runtime_collaboration_profile_ids=(
        MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY.profile_ids
    ),
    workspace_snapshot_profile_ids=(
        WORKSPACE_SNAPSHOT_REGISTRY.workspace_snapshot_profile_ids
    ),
    session_replay_profile_ids=SESSION_REPLAY_REGISTRY.session_replay_profile_ids,
    route_names=tuple(RouteName),
    profile_count=len(MULTIMODAL_WORKSPACE_HISTORY_PROFILES),
    source_registries=_WORKSPACE_HISTORY_SOURCE_REGISTRIES,
    source_reference_ids=_WORKSPACE_HISTORY_SOURCE_REFERENCES,
    workspace_history_surface_refs=_WORKSPACE_HISTORY_SURFACES,
    observability_surfaces=_WORKSPACE_HISTORY_OBSERVABILITY_SURFACES,
)


class BranchingTimelineProfile(BaseModel):
    """Inspectable metadata for one passive Branching Timeline surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1, max_length=140)
    profile_name: str = Field(min_length=1, max_length=160)
    branching_timeline_kind: BranchingTimelineProfileKind
    branch_surface_kind: BranchingTimelineSurfaceKind
    source_workspace_history_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_artifact_lineage_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_shared_artifact_board_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_runtime_collaboration_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_session_replay_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    branch_context_fields: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_reference_ids: tuple[str, ...] = Field(min_length=1, max_length=10)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    branching_timeline_surfaces: tuple[str, ...] = Field(
        min_length=1,
        max_length=7,
    )
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_registries: tuple[str, ...] = Field(min_length=8, max_length=8)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=BRANCHING_TIMELINE_AUTHORITY_BOUNDARY,
        max_length=1300,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BRANCHING_TIMELINE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    branch_creation_implemented: Literal[False] = False
    branch_routing_execution_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    runtime_event_replay_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    workflow_state_mutation_implemented: Literal[False] = False
    workspace_state_mutation_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    branch_storage_persistence_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    serialization_version: Literal["multimodal_branching_timeline_profile.v1"] = (
        BRANCHING_TIMELINE_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class MultimodalBranchingTimelineRegistry(BaseModel):
    """Stable passive registry for V4.5 Branching Timeline metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["multimodal_branching_timeline_registry"] = (
        "multimodal_branching_timeline_registry"
    )
    serialization_version: Literal["multimodal_branching_timeline_registry.v1"] = (
        BRANCHING_TIMELINE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=BRANCHING_TIMELINE_AUTHORITY_BOUNDARY,
        max_length=1300,
    )
    branching_timeline_profiles: tuple[BranchingTimelineProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    branching_timeline_kinds: tuple[BranchingTimelineProfileKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    branch_surface_kinds: tuple[BranchingTimelineSurfaceKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    workspace_history_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    artifact_lineage_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    shared_artifact_board_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    runtime_collaboration_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    session_replay_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=8, max_length=8)
    source_reference_ids: tuple[str, ...] = Field(min_length=10, max_length=10)
    branching_timeline_surface_refs: tuple[str, ...] = Field(
        min_length=7,
        max_length=7,
    )
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BRANCHING_TIMELINE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    branch_creation_implemented: Literal[False] = False
    branch_routing_execution_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    runtime_event_replay_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    workflow_state_mutation_implemented: Literal[False] = False
    workspace_state_mutation_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    branch_storage_persistence_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.profile_id for profile in self.branching_timeline_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("profile_ids must be unique")
        if self.profile_ids != derived_profile_ids:
            raise ValueError("profile_ids must match branching_timeline_profiles")
        if self.profile_count != len(self.branching_timeline_profiles):
            raise ValueError("profile_count must match branching_timeline_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if (
            self.workspace_history_profile_ids
            != MULTIMODAL_WORKSPACE_HISTORY_REGISTRY.profile_ids
        ):
            raise ValueError(
                "workspace_history_profile_ids must match Workspace History registry"
            )
        if (
            self.artifact_lineage_profile_ids
            != MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "artifact_lineage_profile_ids must match Artifact Lineage registry"
            )
        if (
            self.shared_artifact_board_profile_ids
            != MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY.profile_ids
        ):
            raise ValueError(
                "shared_artifact_board_profile_ids must match Shared Artifact Board registry"
            )
        if (
            self.runtime_collaboration_profile_ids
            != MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY.profile_ids
        ):
            raise ValueError(
                "runtime_collaboration_profile_ids must match Runtime Collaboration registry"
            )
        if (
            self.session_replay_profile_ids
            != SESSION_REPLAY_REGISTRY.session_replay_profile_ids
        ):
            raise ValueError(
                "session_replay_profile_ids must match Session Replay registry"
            )
        if self.branching_timeline_kinds != _ordered_unique(
            profile.branching_timeline_kind
            for profile in self.branching_timeline_profiles
        ):
            raise ValueError("branching_timeline_kinds must match profiles")
        if self.branch_surface_kinds != _ordered_unique(
            profile.branch_surface_kind
            for profile in self.branching_timeline_profiles
        ):
            raise ValueError("branch_surface_kinds must match profiles")

        profile_source_references = {
            source_reference
            for profile in self.branching_timeline_profiles
            for source_reference in profile.source_reference_ids
        }
        if set(self.source_reference_ids) != profile_source_references:
            raise ValueError(
                "source_reference_ids must match profile source references"
            )

        known_routes = set(self.route_names)
        known_history = set(self.workspace_history_profile_ids)
        known_lineage = set(self.artifact_lineage_profile_ids)
        known_boards = set(self.shared_artifact_board_profile_ids)
        known_runtime = set(self.runtime_collaboration_profile_ids)
        known_replays = set(self.session_replay_profile_ids)
        known_surfaces = set(self.branching_timeline_surface_refs)
        known_source_references = set(self.source_reference_ids)
        for profile in self.branching_timeline_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must use known routes")
            if not set(profile.source_workspace_history_profile_ids).issubset(
                known_history
            ):
                raise ValueError(
                    "source_workspace_history_profile_ids must be known profiles"
                )
            if not set(profile.source_artifact_lineage_profile_ids).issubset(
                known_lineage
            ):
                raise ValueError(
                    "source_artifact_lineage_profile_ids must be known profiles"
                )
            if not set(profile.source_shared_artifact_board_profile_ids).issubset(
                known_boards
            ):
                raise ValueError(
                    "source_shared_artifact_board_profile_ids must be known profiles"
                )
            if not set(profile.source_runtime_collaboration_profile_ids).issubset(
                known_runtime
            ):
                raise ValueError(
                    "source_runtime_collaboration_profile_ids must be known profiles"
                )
            if not set(profile.source_session_replay_profile_ids).issubset(
                known_replays
            ):
                raise ValueError(
                    "source_session_replay_profile_ids must be known profiles"
                )
            if not set(profile.branching_timeline_surfaces).issubset(
                known_surfaces
            ):
                raise ValueError("branching_timeline_surfaces must be known surfaces")
            if not set(profile.source_reference_ids).issubset(
                known_source_references
            ):
                raise ValueError(
                    "source_reference_ids must be known registry references"
                )
        return self


def multimodal_branching_timeline_registry() -> MultimodalBranchingTimelineRegistry:
    """Return passive V4.5 Branching Timeline metadata."""

    return MULTIMODAL_BRANCHING_TIMELINE_REGISTRY


def multimodal_branching_timeline_profile_by_id(
    profile_id: str,
    registry: MultimodalBranchingTimelineRegistry | None = None,
) -> BranchingTimelineProfile | None:
    """Return one Branching Timeline profile without creating branches."""

    source_registry = registry or MULTIMODAL_BRANCHING_TIMELINE_REGISTRY
    normalized_profile_id = str(profile_id).strip()
    for profile in source_registry.branching_timeline_profiles:
        if profile.profile_id == normalized_profile_id:
            return profile
    return None


def multimodal_branching_timeline_profiles_for_route(
    route: RouteName | str,
    registry: MultimodalBranchingTimelineRegistry | None = None,
) -> tuple[BranchingTimelineProfile, ...]:
    """Return passive Branching Timeline profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or MULTIMODAL_BRANCHING_TIMELINE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.branching_timeline_profiles
        if route_name in profile.route_applicability
    )


def multimodal_branching_timeline_profiles_for_surface_kind(
    surface_kind: BranchingTimelineSurfaceKind | str,
    registry: MultimodalBranchingTimelineRegistry | None = None,
) -> tuple[BranchingTimelineProfile, ...]:
    """Return Branching Timeline profiles for one passive surface kind."""

    surface_value = str(surface_kind).strip()
    source_registry = registry or MULTIMODAL_BRANCHING_TIMELINE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.branching_timeline_profiles
        if profile.branch_surface_kind == surface_value
    )


def multimodal_branching_timeline_profiles_for_workspace_history_profile(
    workspace_history_profile_id: str,
    registry: MultimodalBranchingTimelineRegistry | None = None,
) -> tuple[BranchingTimelineProfile, ...]:
    """Return branching timeline profiles for one workspace history profile."""

    source_registry = registry or MULTIMODAL_BRANCHING_TIMELINE_REGISTRY
    source_profile_id = str(workspace_history_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.branching_timeline_profiles
        if source_profile_id in profile.source_workspace_history_profile_ids
    )


def _branching_timeline_profile(
    *,
    profile_id: str,
    profile_name: str,
    branching_timeline_kind: BranchingTimelineProfileKind,
    branch_surface_kind: BranchingTimelineSurfaceKind,
    source_workspace_history_profile_ids: tuple[str, ...],
    source_artifact_lineage_profile_ids: tuple[str, ...],
    source_shared_artifact_board_profile_ids: tuple[str, ...],
    source_runtime_collaboration_profile_ids: tuple[str, ...],
    source_session_replay_profile_ids: tuple[str, ...],
    branch_context_fields: tuple[str, ...],
    source_reference_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    branching_timeline_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> BranchingTimelineProfile:
    return BranchingTimelineProfile(
        profile_id=profile_id,
        profile_name=profile_name,
        branching_timeline_kind=branching_timeline_kind,
        branch_surface_kind=branch_surface_kind,
        source_workspace_history_profile_ids=source_workspace_history_profile_ids,
        source_artifact_lineage_profile_ids=source_artifact_lineage_profile_ids,
        source_shared_artifact_board_profile_ids=(
            source_shared_artifact_board_profile_ids
        ),
        source_runtime_collaboration_profile_ids=(
            source_runtime_collaboration_profile_ids
        ),
        source_session_replay_profile_ids=source_session_replay_profile_ids,
        branch_context_fields=branch_context_fields,
        source_reference_ids=source_reference_ids,
        route_applicability=route_applicability,
        branching_timeline_surfaces=branching_timeline_surfaces,
        advisory_outputs=advisory_outputs,
        source_registries=_BRANCHING_TIMELINE_SOURCE_REGISTRIES,
        observability_surfaces=_BRANCHING_TIMELINE_OBSERVABILITY_SURFACES,
    )


MULTIMODAL_BRANCHING_TIMELINE_PROFILES = (
    _branching_timeline_profile(
        profile_id="workflow_branching_timeline",
        profile_name="Workflow Branching Timeline",
        branching_timeline_kind="workflow_branch_timeline",
        branch_surface_kind="workflow_branch",
        source_workspace_history_profile_ids=(
            "runtime_event_workspace_history",
            "snapshot_workspace_history",
        ),
        source_artifact_lineage_profile_ids=(
            "timeline_stage_artifact_lineage",
            "source_transition_artifact_lineage",
        ),
        source_shared_artifact_board_profile_ids=(
            "comparison_shared_artifact_board",
            "provenance_lineage_shared_artifact_board",
        ),
        source_runtime_collaboration_profile_ids=(
            "trace_runtime_collaboration",
            "stream_event_runtime_collaboration",
        ),
        source_session_replay_profile_ids=(
            "conversation_timeline_replay_profile",
            "snapshot_transition_replay_profile",
        ),
        branch_context_fields=(
            "WorkflowRuntimeVisualState",
            "workflow.steps.state",
            "branch_node_refs",
            "transitionReason",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_WORKSPACE_HISTORY_REGISTRY",
            "multimodal_studio.MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY",
            "clients.nextjs.workflow_runtime.WorkflowRuntimeVisualState",
            "clients.nextjs.workflow_runtime.deriveWorkflowVisualState",
            "clients.nextjs.workflow_timeline.buildWorkflowTimelineModel",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        branching_timeline_surfaces=(
            "branching_timeline_panel",
            "workflow_branch_surface",
            "branch_summary_surface",
            "branching_timeline_boundary_panel",
        ),
        advisory_outputs=(
            "workflow_branch_timeline_inventory",
            "manual_workflow_branch_review_hint",
            "no_branch_routing_execution_notice",
        ),
    ),
    _branching_timeline_profile(
        profile_id="artifact_variant_branching_timeline",
        profile_name="Artifact Variant Branching Timeline",
        branching_timeline_kind="artifact_variant_branch_timeline",
        branch_surface_kind="artifact_variant",
        source_workspace_history_profile_ids=(
            "artifact_board_workspace_history",
            "session_record_workspace_history",
        ),
        source_artifact_lineage_profile_ids=(
            "dependency_graph_artifact_lineage",
            "source_transition_artifact_lineage",
        ),
        source_shared_artifact_board_profile_ids=(
            "selection_shared_artifact_board",
            "comparison_shared_artifact_board",
            "handoff_refinement_shared_artifact_board",
        ),
        source_runtime_collaboration_profile_ids=(
            "stream_event_runtime_collaboration",
            "operator_context_runtime_collaboration",
        ),
        source_session_replay_profile_ids=(
            "session_overview_replay_profile",
            "snapshot_transition_replay_profile",
        ),
        branch_context_fields=(
            "activeArtifactId",
            "comparison.rows",
            "recommendedRow",
            "artifact_lineage_refs",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY",
            "multimodal_studio.MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY",
            "clients.nextjs.workflow_timeline.buildWorkflowTimelineModel",
            "clients.nextjs.workstation_shell.WorkstationShell",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        branching_timeline_surfaces=(
            "branching_timeline_panel",
            "artifact_variant_branch_surface",
            "branch_summary_surface",
            "branching_timeline_boundary_panel",
        ),
        advisory_outputs=(
            "artifact_variant_branch_timeline_inventory",
            "manual_variant_branch_review_hint",
            "no_artifact_mutation_notice",
        ),
    ),
    _branching_timeline_profile(
        profile_id="review_retry_branching_timeline",
        profile_name="Review Retry Branching Timeline",
        branching_timeline_kind="review_retry_branch_timeline",
        branch_surface_kind="review_retry",
        source_workspace_history_profile_ids=(
            "runtime_event_workspace_history",
            "snapshot_workspace_history",
        ),
        source_artifact_lineage_profile_ids=(
            "timeline_stage_artifact_lineage",
            "missing_artifact_lineage",
        ),
        source_shared_artifact_board_profile_ids=(
            "provenance_lineage_shared_artifact_board",
            "handoff_refinement_shared_artifact_board",
        ),
        source_runtime_collaboration_profile_ids=(
            "trace_runtime_collaboration",
            "console_runtime_collaboration",
        ),
        source_session_replay_profile_ids=(
            "review_decision_replay_profile",
            "snapshot_transition_replay_profile",
        ),
        branch_context_fields=(
            "review_failed",
            "retry_started",
            "refinement_requested",
            "transitionReason",
        ),
        source_reference_ids=(
            "hybrid_studio.SESSION_REPLAY_REGISTRY",
            "clients.nextjs.workflow_runtime.WorkflowRuntimeVisualState",
            "clients.nextjs.workflow_timeline.WorkflowTimelineEvent",
            "clients.nextjs.workstation_shell.WorkstationShell",
        ),
        route_applicability=(
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        branching_timeline_surfaces=(
            "branching_timeline_panel",
            "review_retry_branch_surface",
            "branch_summary_surface",
            "branching_timeline_boundary_panel",
        ),
        advisory_outputs=(
            "review_retry_branch_timeline_inventory",
            "manual_retry_branch_review_hint",
            "no_retry_triggering_notice",
        ),
    ),
    _branching_timeline_profile(
        profile_id="fallback_failure_branching_timeline",
        profile_name="Fallback Failure Branching Timeline",
        branching_timeline_kind="fallback_failure_branch_timeline",
        branch_surface_kind="fallback_failure",
        source_workspace_history_profile_ids=("runtime_event_workspace_history",),
        source_artifact_lineage_profile_ids=(
            "missing_artifact_lineage",
            "timeline_stage_artifact_lineage",
        ),
        source_shared_artifact_board_profile_ids=(
            "provenance_lineage_shared_artifact_board",
        ),
        source_runtime_collaboration_profile_ids=(
            "console_runtime_collaboration",
            "operator_context_runtime_collaboration",
        ),
        source_session_replay_profile_ids=("review_decision_replay_profile",),
        branch_context_fields=(
            "failure_node",
            "failed_state",
            "fallback_panel",
            "workflow_error",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_WORKSPACE_HISTORY_REGISTRY",
            "hybrid_studio.SESSION_REPLAY_REGISTRY",
            "clients.nextjs.workflow_runtime.deriveWorkflowVisualState",
            "clients.nextjs.workflow_timeline.WorkflowTimelineEvent",
            "clients.nextjs.workstation_shell.WorkstationShell",
        ),
        route_applicability=(
            RouteName.DEBUG,
            RouteName.REVIEW,
        ),
        branching_timeline_surfaces=(
            "branching_timeline_panel",
            "fallback_failure_branch_surface",
            "branch_summary_surface",
            "branching_timeline_boundary_panel",
        ),
        advisory_outputs=(
            "fallback_failure_branch_timeline_inventory",
            "manual_failure_branch_review_hint",
            "no_branch_creation_notice",
        ),
    ),
)

MULTIMODAL_BRANCHING_TIMELINE_REGISTRY = MultimodalBranchingTimelineRegistry(
    branching_timeline_profiles=MULTIMODAL_BRANCHING_TIMELINE_PROFILES,
    profile_ids=tuple(
        profile.profile_id for profile in MULTIMODAL_BRANCHING_TIMELINE_PROFILES
    ),
    branching_timeline_kinds=tuple(
        profile.branching_timeline_kind
        for profile in MULTIMODAL_BRANCHING_TIMELINE_PROFILES
    ),
    branch_surface_kinds=tuple(
        profile.branch_surface_kind
        for profile in MULTIMODAL_BRANCHING_TIMELINE_PROFILES
    ),
    workspace_history_profile_ids=MULTIMODAL_WORKSPACE_HISTORY_REGISTRY.profile_ids,
    artifact_lineage_profile_ids=MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY.profile_ids,
    shared_artifact_board_profile_ids=(
        MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY.profile_ids
    ),
    runtime_collaboration_profile_ids=(
        MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY.profile_ids
    ),
    session_replay_profile_ids=SESSION_REPLAY_REGISTRY.session_replay_profile_ids,
    route_names=tuple(RouteName),
    profile_count=len(MULTIMODAL_BRANCHING_TIMELINE_PROFILES),
    source_registries=_BRANCHING_TIMELINE_SOURCE_REGISTRIES,
    source_reference_ids=_BRANCHING_TIMELINE_SOURCE_REFERENCES,
    branching_timeline_surface_refs=_BRANCHING_TIMELINE_SURFACES,
    observability_surfaces=_BRANCHING_TIMELINE_OBSERVABILITY_SURFACES,
)


class CreativeEvolutionTimelineProfile(BaseModel):
    """Inspectable metadata for one passive Creative Evolution Timeline surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1, max_length=150)
    profile_name: str = Field(min_length=1, max_length=170)
    evolution_profile_kind: CreativeEvolutionTimelineProfileKind
    evolution_surface_kind: CreativeEvolutionTimelineSurfaceKind
    source_branching_timeline_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_workspace_history_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_shared_artifact_board_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_artifact_lineage_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_artifact_provenance_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    evolution_context_fields: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_reference_ids: tuple[str, ...] = Field(min_length=1, max_length=10)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    creative_evolution_surfaces: tuple[str, ...] = Field(
        min_length=1,
        max_length=7,
    )
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_registries: tuple[str, ...] = Field(min_length=8, max_length=8)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=CREATIVE_EVOLUTION_TIMELINE_AUTHORITY_BOUNDARY,
        max_length=1300,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CREATIVE_EVOLUTION_TIMELINE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    evolution_generation_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    branch_creation_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    quality_score_mutation_implemented: Literal[False] = False
    provenance_recording_implemented: Literal[False] = False
    runtime_event_replay_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    serialization_version: Literal[
        "multimodal_creative_evolution_timeline_profile.v1"
    ] = CREATIVE_EVOLUTION_TIMELINE_PROFILE_SERIALIZATION_VERSION
    metadata_only: Literal[True] = True


class MultimodalCreativeEvolutionTimelineRegistry(BaseModel):
    """Stable passive registry for V4.5 Creative Evolution Timeline metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["multimodal_creative_evolution_timeline_registry"] = (
        "multimodal_creative_evolution_timeline_registry"
    )
    serialization_version: Literal[
        "multimodal_creative_evolution_timeline_registry.v1"
    ] = CREATIVE_EVOLUTION_TIMELINE_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=CREATIVE_EVOLUTION_TIMELINE_AUTHORITY_BOUNDARY,
        max_length=1300,
    )
    creative_evolution_timeline_profiles: tuple[
        CreativeEvolutionTimelineProfile,
        ...,
    ] = Field(
        min_length=4,
        max_length=4,
    )
    profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    evolution_profile_kinds: tuple[
        CreativeEvolutionTimelineProfileKind,
        ...,
    ] = Field(min_length=4, max_length=4)
    evolution_surface_kinds: tuple[
        CreativeEvolutionTimelineSurfaceKind,
        ...,
    ] = Field(min_length=4, max_length=4)
    branching_timeline_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    workspace_history_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    shared_artifact_board_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    artifact_lineage_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    artifact_provenance_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=8, max_length=8)
    source_reference_ids: tuple[str, ...] = Field(min_length=10, max_length=10)
    creative_evolution_surface_refs: tuple[str, ...] = Field(
        min_length=7,
        max_length=7,
    )
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CREATIVE_EVOLUTION_TIMELINE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    evolution_generation_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    branch_creation_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    quality_score_mutation_implemented: Literal[False] = False
    provenance_recording_implemented: Literal[False] = False
    runtime_event_replay_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.profile_id
            for profile in self.creative_evolution_timeline_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("profile_ids must be unique")
        if self.profile_ids != derived_profile_ids:
            raise ValueError(
                "profile_ids must match creative_evolution_timeline_profiles"
            )
        if self.profile_count != len(self.creative_evolution_timeline_profiles):
            raise ValueError(
                "profile_count must match creative_evolution_timeline_profiles"
            )
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if (
            self.branching_timeline_profile_ids
            != MULTIMODAL_BRANCHING_TIMELINE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "branching_timeline_profile_ids must match Branching Timeline registry"
            )
        if (
            self.workspace_history_profile_ids
            != MULTIMODAL_WORKSPACE_HISTORY_REGISTRY.profile_ids
        ):
            raise ValueError(
                "workspace_history_profile_ids must match Workspace History registry"
            )
        if (
            self.shared_artifact_board_profile_ids
            != MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY.profile_ids
        ):
            raise ValueError(
                "shared_artifact_board_profile_ids must match Shared Artifact Board registry"
            )
        if (
            self.artifact_lineage_profile_ids
            != MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "artifact_lineage_profile_ids must match Artifact Lineage registry"
            )
        if (
            self.artifact_provenance_profile_ids
            != MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "artifact_provenance_profile_ids must match Artifact Provenance registry"
            )
        if self.evolution_profile_kinds != _ordered_unique(
            profile.evolution_profile_kind
            for profile in self.creative_evolution_timeline_profiles
        ):
            raise ValueError("evolution_profile_kinds must match profiles")
        if self.evolution_surface_kinds != _ordered_unique(
            profile.evolution_surface_kind
            for profile in self.creative_evolution_timeline_profiles
        ):
            raise ValueError("evolution_surface_kinds must match profiles")

        profile_source_references = {
            source_reference
            for profile in self.creative_evolution_timeline_profiles
            for source_reference in profile.source_reference_ids
        }
        if set(self.source_reference_ids) != profile_source_references:
            raise ValueError(
                "source_reference_ids must match profile source references"
            )

        known_routes = set(self.route_names)
        known_branching = set(self.branching_timeline_profile_ids)
        known_history = set(self.workspace_history_profile_ids)
        known_boards = set(self.shared_artifact_board_profile_ids)
        known_lineage = set(self.artifact_lineage_profile_ids)
        known_provenance = set(self.artifact_provenance_profile_ids)
        known_surfaces = set(self.creative_evolution_surface_refs)
        known_source_references = set(self.source_reference_ids)
        for profile in self.creative_evolution_timeline_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must use known routes")
            if not set(profile.source_branching_timeline_profile_ids).issubset(
                known_branching
            ):
                raise ValueError(
                    "source_branching_timeline_profile_ids must be known profiles"
                )
            if not set(profile.source_workspace_history_profile_ids).issubset(
                known_history
            ):
                raise ValueError(
                    "source_workspace_history_profile_ids must be known profiles"
                )
            if not set(profile.source_shared_artifact_board_profile_ids).issubset(
                known_boards
            ):
                raise ValueError(
                    "source_shared_artifact_board_profile_ids must be known profiles"
                )
            if not set(profile.source_artifact_lineage_profile_ids).issubset(
                known_lineage
            ):
                raise ValueError(
                    "source_artifact_lineage_profile_ids must be known profiles"
                )
            if not set(profile.source_artifact_provenance_profile_ids).issubset(
                known_provenance
            ):
                raise ValueError(
                    "source_artifact_provenance_profile_ids must be known profiles"
                )
            if not set(profile.creative_evolution_surfaces).issubset(known_surfaces):
                raise ValueError("creative_evolution_surfaces must be known surfaces")
            if not set(profile.source_reference_ids).issubset(
                known_source_references
            ):
                raise ValueError(
                    "source_reference_ids must be known registry references"
                )
        return self


def multimodal_creative_evolution_timeline_registry() -> (
    MultimodalCreativeEvolutionTimelineRegistry
):
    """Return passive V4.5 Creative Evolution Timeline metadata."""

    return MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_REGISTRY


def multimodal_creative_evolution_timeline_profile_by_id(
    profile_id: str,
    registry: MultimodalCreativeEvolutionTimelineRegistry | None = None,
) -> CreativeEvolutionTimelineProfile | None:
    """Return one Creative Evolution Timeline profile without generating evolution."""

    source_registry = registry or MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_REGISTRY
    normalized_profile_id = str(profile_id).strip()
    for profile in source_registry.creative_evolution_timeline_profiles:
        if profile.profile_id == normalized_profile_id:
            return profile
    return None


def multimodal_creative_evolution_timeline_profiles_for_route(
    route: RouteName | str,
    registry: MultimodalCreativeEvolutionTimelineRegistry | None = None,
) -> tuple[CreativeEvolutionTimelineProfile, ...]:
    """Return passive Creative Evolution Timeline profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.creative_evolution_timeline_profiles
        if route_name in profile.route_applicability
    )


def multimodal_creative_evolution_timeline_profiles_for_surface_kind(
    surface_kind: CreativeEvolutionTimelineSurfaceKind | str,
    registry: MultimodalCreativeEvolutionTimelineRegistry | None = None,
) -> tuple[CreativeEvolutionTimelineProfile, ...]:
    """Return Creative Evolution Timeline profiles for one passive surface kind."""

    surface_value = str(surface_kind).strip()
    source_registry = registry or MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.creative_evolution_timeline_profiles
        if profile.evolution_surface_kind == surface_value
    )


def multimodal_creative_evolution_timeline_profiles_for_branching_timeline_profile(
    branching_timeline_profile_id: str,
    registry: MultimodalCreativeEvolutionTimelineRegistry | None = None,
) -> tuple[CreativeEvolutionTimelineProfile, ...]:
    """Return creative evolution profiles for one branching timeline profile."""

    source_registry = registry or MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_REGISTRY
    source_profile_id = str(branching_timeline_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.creative_evolution_timeline_profiles
        if source_profile_id in profile.source_branching_timeline_profile_ids
    )


def _creative_evolution_timeline_profile(
    *,
    profile_id: str,
    profile_name: str,
    evolution_profile_kind: CreativeEvolutionTimelineProfileKind,
    evolution_surface_kind: CreativeEvolutionTimelineSurfaceKind,
    source_branching_timeline_profile_ids: tuple[str, ...],
    source_workspace_history_profile_ids: tuple[str, ...],
    source_shared_artifact_board_profile_ids: tuple[str, ...],
    source_artifact_lineage_profile_ids: tuple[str, ...],
    source_artifact_provenance_profile_ids: tuple[str, ...],
    evolution_context_fields: tuple[str, ...],
    source_reference_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    creative_evolution_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> CreativeEvolutionTimelineProfile:
    return CreativeEvolutionTimelineProfile(
        profile_id=profile_id,
        profile_name=profile_name,
        evolution_profile_kind=evolution_profile_kind,
        evolution_surface_kind=evolution_surface_kind,
        source_branching_timeline_profile_ids=(
            source_branching_timeline_profile_ids
        ),
        source_workspace_history_profile_ids=source_workspace_history_profile_ids,
        source_shared_artifact_board_profile_ids=(
            source_shared_artifact_board_profile_ids
        ),
        source_artifact_lineage_profile_ids=source_artifact_lineage_profile_ids,
        source_artifact_provenance_profile_ids=(
            source_artifact_provenance_profile_ids
        ),
        evolution_context_fields=evolution_context_fields,
        source_reference_ids=source_reference_ids,
        route_applicability=route_applicability,
        creative_evolution_surfaces=creative_evolution_surfaces,
        advisory_outputs=advisory_outputs,
        source_registries=_CREATIVE_EVOLUTION_TIMELINE_SOURCE_REGISTRIES,
        observability_surfaces=(
            _CREATIVE_EVOLUTION_TIMELINE_OBSERVABILITY_SURFACES
        ),
    )


MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_PROFILES = (
    _creative_evolution_timeline_profile(
        profile_id="intent_creative_evolution_timeline",
        profile_name="Intent Creative Evolution Timeline",
        evolution_profile_kind="intent_evolution_timeline",
        evolution_surface_kind="intent_evolution",
        source_branching_timeline_profile_ids=("workflow_branching_timeline",),
        source_workspace_history_profile_ids=(
            "session_record_workspace_history",
            "snapshot_workspace_history",
        ),
        source_shared_artifact_board_profile_ids=(
            "selection_shared_artifact_board",
        ),
        source_artifact_lineage_profile_ids=(
            "dependency_graph_artifact_lineage",
            "source_transition_artifact_lineage",
        ),
        source_artifact_provenance_profile_ids=("evidence_artifact_provenance",),
        evolution_context_fields=(
            "request_intake",
            "planning",
            "retrieval",
            "creative_intelligence",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_BRANCHING_TIMELINE_REGISTRY",
            "multimodal_studio.MULTIMODAL_WORKSPACE_HISTORY_REGISTRY",
            "clients.nextjs.creative_timeline.buildCreativeTimelineModel",
            "clients.nextjs.creative_timeline.CreativeTimelineEvent",
            "clients.nextjs.workflow_explorer.WorkflowExplorerStage",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.PREVIEW,
        ),
        creative_evolution_surfaces=(
            "creative_evolution_timeline_panel",
            "intent_evolution_surface",
            "evolution_summary_surface",
            "creative_evolution_boundary_panel",
        ),
        advisory_outputs=(
            "intent_evolution_timeline_inventory",
            "manual_intent_evolution_review_hint",
            "no_evolution_generation_notice",
        ),
    ),
    _creative_evolution_timeline_profile(
        profile_id="artifact_iteration_creative_evolution_timeline",
        profile_name="Artifact Iteration Creative Evolution Timeline",
        evolution_profile_kind="artifact_iteration_evolution_timeline",
        evolution_surface_kind="artifact_iteration",
        source_branching_timeline_profile_ids=(
            "artifact_variant_branching_timeline",
            "workflow_branching_timeline",
        ),
        source_workspace_history_profile_ids=(
            "artifact_board_workspace_history",
        ),
        source_shared_artifact_board_profile_ids=(
            "selection_shared_artifact_board",
            "comparison_shared_artifact_board",
        ),
        source_artifact_lineage_profile_ids=(
            "dependency_graph_artifact_lineage",
            "source_transition_artifact_lineage",
        ),
        source_artifact_provenance_profile_ids=(
            "payload_artifact_provenance",
            "evidence_artifact_provenance",
        ),
        evolution_context_fields=(
            "artifact_intelligence",
            "generative_design",
            "activeArtifactId",
            "comparison.rows",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY",
            "multimodal_studio.MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY",
            "clients.nextjs.creative_timeline.buildCreativeTimelineModel",
            "clients.nextjs.creative_timeline.provenanceSourceCount",
            "clients.nextjs.workstation_shell.WorkstationShell",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        creative_evolution_surfaces=(
            "creative_evolution_timeline_panel",
            "artifact_iteration_evolution_surface",
            "evolution_summary_surface",
            "creative_evolution_boundary_panel",
        ),
        advisory_outputs=(
            "artifact_iteration_evolution_inventory",
            "manual_artifact_iteration_review_hint",
            "no_artifact_mutation_notice",
        ),
    ),
    _creative_evolution_timeline_profile(
        profile_id="quality_refinement_creative_evolution_timeline",
        profile_name="Quality Refinement Creative Evolution Timeline",
        evolution_profile_kind="quality_refinement_evolution_timeline",
        evolution_surface_kind="quality_refinement",
        source_branching_timeline_profile_ids=(
            "review_retry_branching_timeline",
            "fallback_failure_branching_timeline",
        ),
        source_workspace_history_profile_ids=(
            "runtime_event_workspace_history",
            "snapshot_workspace_history",
        ),
        source_shared_artifact_board_profile_ids=(
            "provenance_lineage_shared_artifact_board",
            "handoff_refinement_shared_artifact_board",
        ),
        source_artifact_lineage_profile_ids=(
            "timeline_stage_artifact_lineage",
            "missing_artifact_lineage",
        ),
        source_artifact_provenance_profile_ids=(
            "evaluation_artifact_provenance",
            "missing_source_artifact_provenance",
        ),
        evolution_context_fields=(
            "creative_evaluation",
            "review_failed",
            "refinement_requested",
            "warningCount",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY",
            "multimodal_studio.MULTIMODAL_BRANCHING_TIMELINE_REGISTRY",
            "clients.nextjs.creative_timeline.CreativeTimelineEvent",
            "clients.nextjs.workflow_explorer.WorkflowExplorerStage",
            "clients.nextjs.workstation_shell.WorkstationShell",
        ),
        route_applicability=(
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        creative_evolution_surfaces=(
            "creative_evolution_timeline_panel",
            "quality_refinement_evolution_surface",
            "evolution_summary_surface",
            "creative_evolution_boundary_panel",
        ),
        advisory_outputs=(
            "quality_refinement_evolution_inventory",
            "manual_quality_refinement_review_hint",
            "no_quality_score_mutation_notice",
        ),
    ),
    _creative_evolution_timeline_profile(
        profile_id="final_synthesis_creative_evolution_timeline",
        profile_name="Final Synthesis Creative Evolution Timeline",
        evolution_profile_kind="final_synthesis_evolution_timeline",
        evolution_surface_kind="final_synthesis",
        source_branching_timeline_profile_ids=(
            "workflow_branching_timeline",
            "review_retry_branching_timeline",
        ),
        source_workspace_history_profile_ids=(
            "session_record_workspace_history",
            "runtime_event_workspace_history",
        ),
        source_shared_artifact_board_profile_ids=(
            "handoff_refinement_shared_artifact_board",
            "provenance_lineage_shared_artifact_board",
        ),
        source_artifact_lineage_profile_ids=(
            "timeline_stage_artifact_lineage",
            "source_transition_artifact_lineage",
            "missing_artifact_lineage",
        ),
        source_artifact_provenance_profile_ids=(
            "evaluation_artifact_provenance",
            "payload_artifact_provenance",
        ),
        evolution_context_fields=(
            "final_synthesis",
            "final_payload",
            "sourceCount",
            "evolution_summary",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_WORKSPACE_HISTORY_REGISTRY",
            "multimodal_studio.MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY",
            "clients.nextjs.creative_timeline.buildCreativeTimelineModel",
            "clients.nextjs.creative_timeline.provenanceSourceCount",
            "clients.nextjs.workstation_shell.WorkstationShell",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        creative_evolution_surfaces=(
            "creative_evolution_timeline_panel",
            "final_synthesis_evolution_surface",
            "evolution_summary_surface",
            "creative_evolution_boundary_panel",
        ),
        advisory_outputs=(
            "final_synthesis_evolution_inventory",
            "manual_final_synthesis_review_hint",
            "no_timeline_reconstruction_notice",
        ),
    ),
)

MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_REGISTRY = (
    MultimodalCreativeEvolutionTimelineRegistry(
        creative_evolution_timeline_profiles=(
            MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_PROFILES
        ),
        profile_ids=tuple(
            profile.profile_id
            for profile in MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_PROFILES
        ),
        evolution_profile_kinds=tuple(
            profile.evolution_profile_kind
            for profile in MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_PROFILES
        ),
        evolution_surface_kinds=tuple(
            profile.evolution_surface_kind
            for profile in MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_PROFILES
        ),
        branching_timeline_profile_ids=(
            MULTIMODAL_BRANCHING_TIMELINE_REGISTRY.profile_ids
        ),
        workspace_history_profile_ids=(
            MULTIMODAL_WORKSPACE_HISTORY_REGISTRY.profile_ids
        ),
        shared_artifact_board_profile_ids=(
            MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY.profile_ids
        ),
        artifact_lineage_profile_ids=(
            MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY.profile_ids
        ),
        artifact_provenance_profile_ids=(
            MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY.profile_ids
        ),
        route_names=tuple(RouteName),
        profile_count=len(MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_PROFILES),
        source_registries=_CREATIVE_EVOLUTION_TIMELINE_SOURCE_REGISTRIES,
        source_reference_ids=_CREATIVE_EVOLUTION_TIMELINE_SOURCE_REFERENCES,
        creative_evolution_surface_refs=_CREATIVE_EVOLUTION_TIMELINE_SURFACES,
        observability_surfaces=(
            _CREATIVE_EVOLUTION_TIMELINE_OBSERVABILITY_SURFACES
        ),
    )
)


class RealTimeWorkflowVisualizationProfile(BaseModel):
    """Inspectable metadata for one passive Real-Time Workflow Visualization surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1, max_length=160)
    profile_name: str = Field(min_length=1, max_length=180)
    visualization_profile_kind: RealTimeWorkflowVisualizationProfileKind
    visualization_surface_kind: RealTimeWorkflowVisualizationSurfaceKind
    source_creative_evolution_timeline_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_branching_timeline_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_runtime_collaboration_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_workspace_history_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    visualization_context_fields: tuple[str, ...] = Field(
        min_length=1,
        max_length=10,
    )
    source_reference_ids: tuple[str, ...] = Field(min_length=1, max_length=10)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    workflow_visualization_surfaces: tuple[str, ...] = Field(
        min_length=1,
        max_length=7,
    )
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_registries: tuple[str, ...] = Field(min_length=8, max_length=8)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=REAL_TIME_WORKFLOW_VISUALIZATION_AUTHORITY_BOUNDARY,
        max_length=1400,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_REAL_TIME_WORKFLOW_VISUALIZATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=15,
    )
    real_time_stream_subscription_implemented: Literal[False] = False
    workflow_state_mutation_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    event_replay_implemented: Literal[False] = False
    runtime_console_control_implemented: Literal[False] = False
    preview_runtime_control_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    collaboration_storage_persistence_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    serialization_version: Literal[
        "multimodal_real_time_workflow_visualization_profile.v1"
    ] = REAL_TIME_WORKFLOW_VISUALIZATION_PROFILE_SERIALIZATION_VERSION
    metadata_only: Literal[True] = True


class MultimodalRealTimeWorkflowVisualizationRegistry(BaseModel):
    """Stable passive registry for V4.5 Real-Time Workflow Visualization metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["multimodal_real_time_workflow_visualization_registry"] = (
        "multimodal_real_time_workflow_visualization_registry"
    )
    serialization_version: Literal[
        "multimodal_real_time_workflow_visualization_registry.v1"
    ] = REAL_TIME_WORKFLOW_VISUALIZATION_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=REAL_TIME_WORKFLOW_VISUALIZATION_AUTHORITY_BOUNDARY,
        max_length=1400,
    )
    real_time_workflow_visualization_profiles: tuple[
        RealTimeWorkflowVisualizationProfile,
        ...,
    ] = Field(
        min_length=4,
        max_length=4,
    )
    profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    visualization_profile_kinds: tuple[
        RealTimeWorkflowVisualizationProfileKind,
        ...,
    ] = Field(min_length=4, max_length=4)
    visualization_surface_kinds: tuple[
        RealTimeWorkflowVisualizationSurfaceKind,
        ...,
    ] = Field(min_length=4, max_length=4)
    creative_evolution_timeline_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    branching_timeline_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    runtime_collaboration_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    workspace_history_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=8, max_length=8)
    source_reference_ids: tuple[str, ...] = Field(min_length=10, max_length=10)
    workflow_visualization_surface_refs: tuple[str, ...] = Field(
        min_length=7,
        max_length=7,
    )
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_REAL_TIME_WORKFLOW_VISUALIZATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=15,
    )
    real_time_stream_subscription_implemented: Literal[False] = False
    workflow_state_mutation_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    event_replay_implemented: Literal[False] = False
    runtime_console_control_implemented: Literal[False] = False
    preview_runtime_control_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    collaboration_storage_persistence_implemented: Literal[False] = False
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
            for profile in self.real_time_workflow_visualization_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("profile_ids must be unique")
        if self.profile_ids != derived_profile_ids:
            raise ValueError(
                "profile_ids must match real_time_workflow_visualization_profiles"
            )
        if self.profile_count != len(
            self.real_time_workflow_visualization_profiles
        ):
            raise ValueError(
                "profile_count must match real_time_workflow_visualization_profiles"
            )
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if (
            self.creative_evolution_timeline_profile_ids
            != MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "creative_evolution_timeline_profile_ids must match Creative Evolution Timeline registry"
            )
        if (
            self.branching_timeline_profile_ids
            != MULTIMODAL_BRANCHING_TIMELINE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "branching_timeline_profile_ids must match Branching Timeline registry"
            )
        if (
            self.runtime_collaboration_profile_ids
            != MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY.profile_ids
        ):
            raise ValueError(
                "runtime_collaboration_profile_ids must match Runtime Collaboration registry"
            )
        if (
            self.workspace_history_profile_ids
            != MULTIMODAL_WORKSPACE_HISTORY_REGISTRY.profile_ids
        ):
            raise ValueError(
                "workspace_history_profile_ids must match Workspace History registry"
            )
        if self.visualization_profile_kinds != _ordered_unique(
            profile.visualization_profile_kind
            for profile in self.real_time_workflow_visualization_profiles
        ):
            raise ValueError("visualization_profile_kinds must match profiles")
        if self.visualization_surface_kinds != _ordered_unique(
            profile.visualization_surface_kind
            for profile in self.real_time_workflow_visualization_profiles
        ):
            raise ValueError("visualization_surface_kinds must match profiles")

        profile_source_references = {
            source_reference
            for profile in self.real_time_workflow_visualization_profiles
            for source_reference in profile.source_reference_ids
        }
        if set(self.source_reference_ids) != profile_source_references:
            raise ValueError(
                "source_reference_ids must match profile source references"
            )

        known_routes = set(self.route_names)
        known_evolution = set(self.creative_evolution_timeline_profile_ids)
        known_branching = set(self.branching_timeline_profile_ids)
        known_runtime = set(self.runtime_collaboration_profile_ids)
        known_history = set(self.workspace_history_profile_ids)
        known_surfaces = set(self.workflow_visualization_surface_refs)
        known_source_references = set(self.source_reference_ids)
        for profile in self.real_time_workflow_visualization_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must use known routes")
            if not set(
                profile.source_creative_evolution_timeline_profile_ids
            ).issubset(known_evolution):
                raise ValueError(
                    "source_creative_evolution_timeline_profile_ids must be known profiles"
                )
            if not set(profile.source_branching_timeline_profile_ids).issubset(
                known_branching
            ):
                raise ValueError(
                    "source_branching_timeline_profile_ids must be known profiles"
                )
            if not set(profile.source_runtime_collaboration_profile_ids).issubset(
                known_runtime
            ):
                raise ValueError(
                    "source_runtime_collaboration_profile_ids must be known profiles"
                )
            if not set(profile.source_workspace_history_profile_ids).issubset(
                known_history
            ):
                raise ValueError(
                    "source_workspace_history_profile_ids must be known profiles"
                )
            if not set(profile.workflow_visualization_surfaces).issubset(
                known_surfaces
            ):
                raise ValueError(
                    "workflow_visualization_surfaces must be known surfaces"
                )
            if not set(profile.source_reference_ids).issubset(
                known_source_references
            ):
                raise ValueError(
                    "source_reference_ids must be known registry references"
                )
        return self


def multimodal_real_time_workflow_visualization_registry() -> (
    MultimodalRealTimeWorkflowVisualizationRegistry
):
    """Return passive V4.5 Real-Time Workflow Visualization metadata."""

    return MULTIMODAL_REAL_TIME_WORKFLOW_VISUALIZATION_REGISTRY


def multimodal_real_time_workflow_visualization_profile_by_id(
    profile_id: str,
    registry: MultimodalRealTimeWorkflowVisualizationRegistry | None = None,
) -> RealTimeWorkflowVisualizationProfile | None:
    """Return one Real-Time Workflow Visualization profile without execution."""

    source_registry = registry or MULTIMODAL_REAL_TIME_WORKFLOW_VISUALIZATION_REGISTRY
    normalized_profile_id = str(profile_id).strip()
    for profile in source_registry.real_time_workflow_visualization_profiles:
        if profile.profile_id == normalized_profile_id:
            return profile
    return None


def multimodal_real_time_workflow_visualization_profiles_for_route(
    route: RouteName | str,
    registry: MultimodalRealTimeWorkflowVisualizationRegistry | None = None,
) -> tuple[RealTimeWorkflowVisualizationProfile, ...]:
    """Return passive Real-Time Workflow Visualization profiles for a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or MULTIMODAL_REAL_TIME_WORKFLOW_VISUALIZATION_REGISTRY
    return tuple(
        profile
        for profile in source_registry.real_time_workflow_visualization_profiles
        if route_name in profile.route_applicability
    )


def multimodal_real_time_workflow_visualization_profiles_for_surface_kind(
    surface_kind: RealTimeWorkflowVisualizationSurfaceKind | str,
    registry: MultimodalRealTimeWorkflowVisualizationRegistry | None = None,
) -> tuple[RealTimeWorkflowVisualizationProfile, ...]:
    """Return Real-Time Workflow Visualization profiles for one passive surface."""

    surface_value = str(surface_kind).strip()
    source_registry = registry or MULTIMODAL_REAL_TIME_WORKFLOW_VISUALIZATION_REGISTRY
    return tuple(
        profile
        for profile in source_registry.real_time_workflow_visualization_profiles
        if profile.visualization_surface_kind == surface_value
    )


def multimodal_real_time_workflow_visualization_profiles_for_creative_evolution_timeline_profile(
    creative_evolution_timeline_profile_id: str,
    registry: MultimodalRealTimeWorkflowVisualizationRegistry | None = None,
) -> tuple[RealTimeWorkflowVisualizationProfile, ...]:
    """Return workflow visualization profiles for one creative evolution profile."""

    source_registry = registry or MULTIMODAL_REAL_TIME_WORKFLOW_VISUALIZATION_REGISTRY
    source_profile_id = str(creative_evolution_timeline_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.real_time_workflow_visualization_profiles
        if source_profile_id
        in profile.source_creative_evolution_timeline_profile_ids
    )


def _real_time_workflow_visualization_profile(
    *,
    profile_id: str,
    profile_name: str,
    visualization_profile_kind: RealTimeWorkflowVisualizationProfileKind,
    visualization_surface_kind: RealTimeWorkflowVisualizationSurfaceKind,
    source_creative_evolution_timeline_profile_ids: tuple[str, ...],
    source_branching_timeline_profile_ids: tuple[str, ...],
    source_runtime_collaboration_profile_ids: tuple[str, ...],
    source_workspace_history_profile_ids: tuple[str, ...],
    visualization_context_fields: tuple[str, ...],
    source_reference_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    workflow_visualization_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> RealTimeWorkflowVisualizationProfile:
    return RealTimeWorkflowVisualizationProfile(
        profile_id=profile_id,
        profile_name=profile_name,
        visualization_profile_kind=visualization_profile_kind,
        visualization_surface_kind=visualization_surface_kind,
        source_creative_evolution_timeline_profile_ids=(
            source_creative_evolution_timeline_profile_ids
        ),
        source_branching_timeline_profile_ids=(
            source_branching_timeline_profile_ids
        ),
        source_runtime_collaboration_profile_ids=(
            source_runtime_collaboration_profile_ids
        ),
        source_workspace_history_profile_ids=source_workspace_history_profile_ids,
        visualization_context_fields=visualization_context_fields,
        source_reference_ids=source_reference_ids,
        route_applicability=route_applicability,
        workflow_visualization_surfaces=workflow_visualization_surfaces,
        advisory_outputs=advisory_outputs,
        source_registries=_REAL_TIME_WORKFLOW_VISUALIZATION_SOURCE_REGISTRIES,
        observability_surfaces=(
            _REAL_TIME_WORKFLOW_VISUALIZATION_OBSERVABILITY_SURFACES
        ),
    )


MULTIMODAL_REAL_TIME_WORKFLOW_VISUALIZATION_PROFILES = (
    _real_time_workflow_visualization_profile(
        profile_id="runtime_state_real_time_workflow_visualization",
        profile_name="Runtime State Real-Time Workflow Visualization",
        visualization_profile_kind="runtime_state_visualization",
        visualization_surface_kind="runtime_state",
        source_creative_evolution_timeline_profile_ids=(
            "intent_creative_evolution_timeline",
            "artifact_iteration_creative_evolution_timeline",
        ),
        source_branching_timeline_profile_ids=(
            "workflow_branching_timeline",
            "artifact_variant_branching_timeline",
        ),
        source_runtime_collaboration_profile_ids=(
            "trace_runtime_collaboration",
            "stream_event_runtime_collaboration",
        ),
        source_workspace_history_profile_ids=(
            "runtime_event_workspace_history",
            "snapshot_workspace_history",
        ),
        visualization_context_fields=(
            "WorkflowRuntimeModel",
            "WorkflowRuntimeVisualState",
            "workflow.steps.state",
            "activeWorkflowNodeId",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY",
            "multimodal_studio.MULTIMODAL_WORKSPACE_HISTORY_REGISTRY",
            "clients.nextjs.workflow_runtime.WorkflowRuntimeModel",
            "clients.nextjs.workflow_runtime.WorkflowRuntimeVisualState",
            "clients.nextjs.workflow_timeline.WorkflowTimelineModel",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        workflow_visualization_surfaces=(
            "real_time_workflow_visualization_panel",
            "runtime_state_visual_surface",
            "workflow_visualization_summary_surface",
            "real_time_workflow_visualization_boundary_panel",
        ),
        advisory_outputs=(
            "runtime_state_visualization_inventory",
            "manual_runtime_state_review_hint",
            "no_real_time_stream_subscription_notice",
        ),
    ),
    _real_time_workflow_visualization_profile(
        profile_id="timeline_event_real_time_workflow_visualization",
        profile_name="Timeline Event Real-Time Workflow Visualization",
        visualization_profile_kind="timeline_event_visualization",
        visualization_surface_kind="timeline_event",
        source_creative_evolution_timeline_profile_ids=(
            "intent_creative_evolution_timeline",
            "quality_refinement_creative_evolution_timeline",
        ),
        source_branching_timeline_profile_ids=(
            "workflow_branching_timeline",
            "review_retry_branching_timeline",
            "fallback_failure_branching_timeline",
        ),
        source_runtime_collaboration_profile_ids=(
            "trace_runtime_collaboration",
            "console_runtime_collaboration",
        ),
        source_workspace_history_profile_ids=(
            "runtime_event_workspace_history",
            "snapshot_workspace_history",
        ),
        visualization_context_fields=(
            "WorkflowTimelineEvent",
            "event.sequence",
            "transitionReason",
            "warningCount",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_BRANCHING_TIMELINE_REGISTRY",
            "clients.nextjs.workflow_runtime.WorkflowRuntimeModel",
            "clients.nextjs.workflow_timeline.WorkflowTimelineModel",
            "clients.nextjs.workflow_timeline.WorkflowTimelineEvent",
            "clients.nextjs.workflow_explorer.WorkflowExplorerStage",
        ),
        route_applicability=(
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        workflow_visualization_surfaces=(
            "real_time_workflow_visualization_panel",
            "timeline_event_visual_surface",
            "workflow_visualization_summary_surface",
            "real_time_workflow_visualization_boundary_panel",
        ),
        advisory_outputs=(
            "timeline_event_visualization_inventory",
            "manual_timeline_event_review_hint",
            "no_timeline_reconstruction_notice",
        ),
    ),
    _real_time_workflow_visualization_profile(
        profile_id="metadata_stage_real_time_workflow_visualization",
        profile_name="Metadata Stage Real-Time Workflow Visualization",
        visualization_profile_kind="metadata_stage_visualization",
        visualization_surface_kind="metadata_stage",
        source_creative_evolution_timeline_profile_ids=(
            "artifact_iteration_creative_evolution_timeline",
            "final_synthesis_creative_evolution_timeline",
        ),
        source_branching_timeline_profile_ids=(
            "artifact_variant_branching_timeline",
            "workflow_branching_timeline",
        ),
        source_runtime_collaboration_profile_ids=(
            "operator_context_runtime_collaboration",
            "stream_event_runtime_collaboration",
        ),
        source_workspace_history_profile_ids=(
            "session_record_workspace_history",
            "artifact_board_workspace_history",
        ),
        visualization_context_fields=(
            "WorkflowExplorerStage",
            "metadataGroups",
            "workstationState.selection.activeWorkflowNodeId",
            "availableMetadataGroupCount",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_REGISTRY",
            "clients.nextjs.workflow_runtime.WorkflowRuntimeModel",
            "clients.nextjs.workflow_runtime.WorkflowRuntimeVisualState",
            "clients.nextjs.workflow_timeline.WorkflowTimelineEvent",
            "clients.nextjs.workflow_explorer.WorkflowExplorerStage",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.PREVIEW,
        ),
        workflow_visualization_surfaces=(
            "real_time_workflow_visualization_panel",
            "metadata_stage_visual_surface",
            "workflow_visualization_summary_surface",
            "real_time_workflow_visualization_boundary_panel",
        ),
        advisory_outputs=(
            "metadata_stage_visualization_inventory",
            "manual_metadata_stage_review_hint",
            "no_workflow_control_notice",
        ),
    ),
    _real_time_workflow_visualization_profile(
        profile_id="console_health_real_time_workflow_visualization",
        profile_name="Console Health Real-Time Workflow Visualization",
        visualization_profile_kind="console_health_visualization",
        visualization_surface_kind="console_health",
        source_creative_evolution_timeline_profile_ids=(
            "quality_refinement_creative_evolution_timeline",
            "final_synthesis_creative_evolution_timeline",
        ),
        source_branching_timeline_profile_ids=(
            "review_retry_branching_timeline",
            "fallback_failure_branching_timeline",
        ),
        source_runtime_collaboration_profile_ids=(
            "console_runtime_collaboration",
            "operator_context_runtime_collaboration",
        ),
        source_workspace_history_profile_ids=(
            "runtime_event_workspace_history",
            "snapshot_workspace_history",
        ),
        visualization_context_fields=(
            "RuntimeConsoleModel",
            "RuntimeConsoleHealthSignal",
            "diagnostics",
            "reloadHistory",
        ),
        source_reference_ids=(
            "multimodal_studio.MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY",
            "multimodal_studio.MULTIMODAL_WORKSPACE_HISTORY_REGISTRY",
            "clients.nextjs.workflow_runtime.WorkflowRuntimeModel",
            "clients.nextjs.workflow_timeline.WorkflowTimelineEvent",
            "clients.nextjs.runtime_console.RuntimeConsoleModel",
        ),
        route_applicability=(
            RouteName.DEBUG,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        workflow_visualization_surfaces=(
            "real_time_workflow_visualization_panel",
            "console_health_visual_surface",
            "workflow_visualization_summary_surface",
            "real_time_workflow_visualization_boundary_panel",
        ),
        advisory_outputs=(
            "console_health_visualization_inventory",
            "manual_console_health_review_hint",
            "no_runtime_console_control_notice",
        ),
    ),
)

MULTIMODAL_REAL_TIME_WORKFLOW_VISUALIZATION_REGISTRY = (
    MultimodalRealTimeWorkflowVisualizationRegistry(
        real_time_workflow_visualization_profiles=(
            MULTIMODAL_REAL_TIME_WORKFLOW_VISUALIZATION_PROFILES
        ),
        profile_ids=tuple(
            profile.profile_id
            for profile in MULTIMODAL_REAL_TIME_WORKFLOW_VISUALIZATION_PROFILES
        ),
        visualization_profile_kinds=tuple(
            profile.visualization_profile_kind
            for profile in MULTIMODAL_REAL_TIME_WORKFLOW_VISUALIZATION_PROFILES
        ),
        visualization_surface_kinds=tuple(
            profile.visualization_surface_kind
            for profile in MULTIMODAL_REAL_TIME_WORKFLOW_VISUALIZATION_PROFILES
        ),
        creative_evolution_timeline_profile_ids=(
            MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_REGISTRY.profile_ids
        ),
        branching_timeline_profile_ids=(
            MULTIMODAL_BRANCHING_TIMELINE_REGISTRY.profile_ids
        ),
        runtime_collaboration_profile_ids=(
            MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY.profile_ids
        ),
        workspace_history_profile_ids=(
            MULTIMODAL_WORKSPACE_HISTORY_REGISTRY.profile_ids
        ),
        route_names=tuple(RouteName),
        profile_count=len(MULTIMODAL_REAL_TIME_WORKFLOW_VISUALIZATION_PROFILES),
        source_registries=_REAL_TIME_WORKFLOW_VISUALIZATION_SOURCE_REGISTRIES,
        source_reference_ids=_REAL_TIME_WORKFLOW_VISUALIZATION_SOURCE_REFERENCES,
        workflow_visualization_surface_refs=(
            _REAL_TIME_WORKFLOW_VISUALIZATION_SURFACES
        ),
        observability_surfaces=(
            _REAL_TIME_WORKFLOW_VISUALIZATION_OBSERVABILITY_SURFACES
        ),
    )
)


class MultimodalStudioIntegrationProfile(BaseModel):
    """Inspectable passive integration profile for V4.5 Multimodal Studio metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    integration_profile_id: str = Field(min_length=1, max_length=160)
    profile_name: str = Field(min_length=1, max_length=180)
    integration_kind: MultimodalStudioIntegrationKind
    source_registry_names: tuple[str, ...] = Field(min_length=1, max_length=14)
    linked_profile_group_refs: tuple[str, ...] = Field(min_length=1, max_length=14)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    integration_surfaces: tuple[str, ...] = Field(min_length=1, max_length=7)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_registries: tuple[str, ...] = Field(min_length=14, max_length=14)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=MULTIMODAL_STUDIO_INTEGRATION_AUTHORITY_BOUNDARY,
        max_length=1600,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_MULTIMODAL_STUDIO_INTEGRATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    studio_runtime_activation_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    collaboration_storage_persistence_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    real_time_stream_subscription_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    serialization_version: Literal["multimodal_studio_integration_profile.v1"] = (
        MULTIMODAL_STUDIO_INTEGRATION_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class MultimodalStudioIntegrationRegistry(BaseModel):
    """Stable passive registry integrating V4.5 Multimodal Studio metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["multimodal_studio_integration_registry"] = (
        "multimodal_studio_integration_registry"
    )
    serialization_version: Literal["multimodal_studio_integration_registry.v1"] = (
        MULTIMODAL_STUDIO_INTEGRATION_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=MULTIMODAL_STUDIO_INTEGRATION_AUTHORITY_BOUNDARY,
        max_length=1600,
    )
    integration_profiles: tuple[MultimodalStudioIntegrationProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    integration_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    integration_kinds: tuple[MultimodalStudioIntegrationKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    live_preview_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    multi_preview_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    interactive_canvas_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    visual_workspace_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    runtime_collaboration_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    artifact_collaboration_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    artifact_provenance_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    artifact_lineage_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    cross_agent_workspace_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    shared_artifact_board_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    workspace_history_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    branching_timeline_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    creative_evolution_timeline_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    real_time_workflow_visualization_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    profile_group_refs: tuple[str, ...] = Field(min_length=14, max_length=14)
    integration_surface_refs: tuple[str, ...] = Field(min_length=7, max_length=7)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=14, max_length=14)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_MULTIMODAL_STUDIO_INTEGRATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    studio_runtime_activation_implemented: Literal[False] = False
    rendering_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    collaboration_storage_persistence_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    real_time_stream_subscription_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.integration_profile_id for profile in self.integration_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("integration_profile_ids must be unique")
        if self.integration_profile_ids != derived_profile_ids:
            raise ValueError("integration_profile_ids must match integration_profiles")
        if self.profile_count != len(self.integration_profiles):
            raise ValueError("profile_count must match integration_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.integration_kinds != _ordered_unique(
            profile.integration_kind for profile in self.integration_profiles
        ):
            raise ValueError("integration_kinds must match integration_profiles")
        if self.live_preview_profile_ids != MULTIMODAL_LIVE_PREVIEW_REGISTRY.profile_ids:
            raise ValueError(
                "live_preview_profile_ids must match Live Preview registry"
            )
        if self.multi_preview_profile_ids != MULTIMODAL_MULTI_PREVIEW_REGISTRY.profile_ids:
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
        if (
            self.artifact_collaboration_profile_ids
            != MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY.profile_ids
        ):
            raise ValueError(
                "artifact_collaboration_profile_ids must match Artifact Collaboration registry"
            )
        if (
            self.artifact_provenance_profile_ids
            != MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "artifact_provenance_profile_ids must match Artifact Provenance registry"
            )
        if (
            self.artifact_lineage_profile_ids
            != MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "artifact_lineage_profile_ids must match Artifact Lineage registry"
            )
        if (
            self.cross_agent_workspace_profile_ids
            != MULTIMODAL_CROSS_AGENT_WORKSPACE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "cross_agent_workspace_profile_ids must match Cross-Agent Workspace registry"
            )
        if (
            self.shared_artifact_board_profile_ids
            != MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY.profile_ids
        ):
            raise ValueError(
                "shared_artifact_board_profile_ids must match Shared Artifact Board registry"
            )
        if (
            self.workspace_history_profile_ids
            != MULTIMODAL_WORKSPACE_HISTORY_REGISTRY.profile_ids
        ):
            raise ValueError(
                "workspace_history_profile_ids must match Workspace History registry"
            )
        if (
            self.branching_timeline_profile_ids
            != MULTIMODAL_BRANCHING_TIMELINE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "branching_timeline_profile_ids must match Branching Timeline registry"
            )
        if (
            self.creative_evolution_timeline_profile_ids
            != MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_REGISTRY.profile_ids
        ):
            raise ValueError(
                "creative_evolution_timeline_profile_ids must match Creative Evolution Timeline registry"
            )
        if (
            self.real_time_workflow_visualization_profile_ids
            != MULTIMODAL_REAL_TIME_WORKFLOW_VISUALIZATION_REGISTRY.profile_ids
        ):
            raise ValueError(
                "real_time_workflow_visualization_profile_ids must match Real-Time Workflow Visualization registry"
            )

        known_routes = set(self.route_names)
        known_sources = set(self.source_registries)
        known_profile_groups = set(self.profile_group_refs)
        known_surfaces = set(self.integration_surface_refs)
        for profile in self.integration_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("profile source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.source_registry_names).issubset(known_sources):
                raise ValueError("source_registry_names must be known registries")
            if not set(profile.linked_profile_group_refs).issubset(
                known_profile_groups
            ):
                raise ValueError("linked_profile_group_refs must be known groups")
            if not set(profile.integration_surfaces).issubset(known_surfaces):
                raise ValueError("integration_surfaces must be known registry surfaces")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
        return self


def multimodal_studio_integration_registry() -> MultimodalStudioIntegrationRegistry:
    """Return passive V4.5 Multimodal Studio integration metadata."""

    return MULTIMODAL_STUDIO_INTEGRATION_REGISTRY


def multimodal_studio_integration_profile_by_id(
    integration_profile_id: str,
    registry: MultimodalStudioIntegrationRegistry | None = None,
) -> MultimodalStudioIntegrationProfile | None:
    """Return one integration profile without activating Studio behavior."""

    source_registry = registry or MULTIMODAL_STUDIO_INTEGRATION_REGISTRY
    source_profile_id = str(integration_profile_id).strip()
    for profile in source_registry.integration_profiles:
        if profile.integration_profile_id == source_profile_id:
            return profile
    return None


def multimodal_studio_integration_profiles_for_route(
    route: RouteName | str,
    registry: MultimodalStudioIntegrationRegistry | None = None,
) -> tuple[MultimodalStudioIntegrationProfile, ...]:
    """Return passive Multimodal Studio integration profiles for a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or MULTIMODAL_STUDIO_INTEGRATION_REGISTRY
    return tuple(
        profile
        for profile in source_registry.integration_profiles
        if route_name in profile.route_applicability
    )


def multimodal_studio_integration_profiles_for_source_registry(
    source_registry_name: str,
    registry: MultimodalStudioIntegrationRegistry | None = None,
) -> tuple[MultimodalStudioIntegrationProfile, ...]:
    """Return integration profiles referencing one source registry."""

    source_registry = registry or MULTIMODAL_STUDIO_INTEGRATION_REGISTRY
    source_name = str(source_registry_name).strip()
    return tuple(
        profile
        for profile in source_registry.integration_profiles
        if source_name in profile.source_registry_names
    )


def _multimodal_studio_integration_profile(
    *,
    integration_profile_id: str,
    profile_name: str,
    integration_kind: MultimodalStudioIntegrationKind,
    source_registry_names: tuple[str, ...],
    linked_profile_group_refs: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    integration_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> MultimodalStudioIntegrationProfile:
    return MultimodalStudioIntegrationProfile(
        integration_profile_id=integration_profile_id,
        profile_name=profile_name,
        integration_kind=integration_kind,
        source_registry_names=source_registry_names,
        linked_profile_group_refs=linked_profile_group_refs,
        route_applicability=route_applicability,
        integration_surfaces=integration_surfaces,
        advisory_outputs=advisory_outputs,
        source_registries=_MULTIMODAL_STUDIO_INTEGRATION_SOURCE_REGISTRIES,
        observability_surfaces=(
            _MULTIMODAL_STUDIO_INTEGRATION_OBSERVABILITY_SURFACES
        ),
    )


MULTIMODAL_STUDIO_INTEGRATION_PROFILES = (
    _multimodal_studio_integration_profile(
        integration_profile_id="preview_workspace_multimodal_studio_integration",
        profile_name="Preview Workspace Multimodal Studio Integration",
        integration_kind="preview_workspace_integration",
        source_registry_names=(
            "multimodal_live_preview_registry",
            "multimodal_multi_preview_registry",
            "multimodal_interactive_canvas_registry",
            "multimodal_visual_workspace_registry",
            "multimodal_runtime_collaboration_registry",
            "multimodal_real_time_workflow_visualization_registry",
        ),
        linked_profile_group_refs=(
            "live_preview_profiles",
            "multi_preview_profiles",
            "interactive_canvas_profiles",
            "visual_workspace_profiles",
            "runtime_collaboration_profiles",
            "real_time_workflow_visualization_profiles",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.PREVIEW,
        ),
        integration_surfaces=(
            "multimodal_studio_shell",
            "preview_workspace_integration_surface",
            "integration_summary_surface",
            "multimodal_studio_integration_boundary_panel",
        ),
        advisory_outputs=(
            "preview_workspace_integration_inventory",
            "manual_preview_workspace_review_hint",
            "no_rendering_execution_notice",
        ),
    ),
    _multimodal_studio_integration_profile(
        integration_profile_id="collaboration_artifact_multimodal_studio_integration",
        profile_name="Collaboration Artifact Multimodal Studio Integration",
        integration_kind="collaboration_artifact_integration",
        source_registry_names=(
            "multimodal_runtime_collaboration_registry",
            "multimodal_artifact_collaboration_registry",
            "multimodal_artifact_provenance_registry",
            "multimodal_artifact_lineage_registry",
            "multimodal_cross_agent_workspace_registry",
            "multimodal_shared_artifact_board_registry",
        ),
        linked_profile_group_refs=(
            "runtime_collaboration_profiles",
            "artifact_collaboration_profiles",
            "artifact_provenance_profiles",
            "artifact_lineage_profiles",
            "cross_agent_workspace_profiles",
            "shared_artifact_board_profiles",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        integration_surfaces=(
            "multimodal_studio_shell",
            "collaboration_artifact_integration_surface",
            "integration_summary_surface",
            "multimodal_studio_integration_boundary_panel",
        ),
        advisory_outputs=(
            "collaboration_artifact_integration_inventory",
            "manual_collaboration_artifact_review_hint",
            "no_artifact_mutation_notice",
        ),
    ),
    _multimodal_studio_integration_profile(
        integration_profile_id="history_lineage_multimodal_studio_integration",
        profile_name="History Lineage Multimodal Studio Integration",
        integration_kind="history_lineage_integration",
        source_registry_names=(
            "multimodal_workspace_history_registry",
            "multimodal_branching_timeline_registry",
            "multimodal_creative_evolution_timeline_registry",
            "multimodal_artifact_lineage_registry",
            "multimodal_artifact_provenance_registry",
            "multimodal_shared_artifact_board_registry",
        ),
        linked_profile_group_refs=(
            "workspace_history_profiles",
            "branching_timeline_profiles",
            "creative_evolution_timeline_profiles",
            "artifact_lineage_profiles",
            "artifact_provenance_profiles",
            "shared_artifact_board_profiles",
        ),
        route_applicability=(
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        integration_surfaces=(
            "multimodal_studio_shell",
            "history_lineage_integration_surface",
            "integration_summary_surface",
            "multimodal_studio_integration_boundary_panel",
        ),
        advisory_outputs=(
            "history_lineage_integration_inventory",
            "manual_history_lineage_review_hint",
            "no_timeline_reconstruction_notice",
        ),
    ),
    _multimodal_studio_integration_profile(
        integration_profile_id="timeline_visualization_multimodal_studio_integration",
        profile_name="Timeline Visualization Multimodal Studio Integration",
        integration_kind="timeline_visualization_integration",
        source_registry_names=(
            "multimodal_real_time_workflow_visualization_registry",
            "multimodal_creative_evolution_timeline_registry",
            "multimodal_branching_timeline_registry",
            "multimodal_workspace_history_registry",
            "multimodal_runtime_collaboration_registry",
        ),
        linked_profile_group_refs=(
            "real_time_workflow_visualization_profiles",
            "creative_evolution_timeline_profiles",
            "branching_timeline_profiles",
            "workspace_history_profiles",
            "runtime_collaboration_profiles",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        integration_surfaces=(
            "multimodal_studio_shell",
            "timeline_visualization_integration_surface",
            "integration_summary_surface",
            "multimodal_studio_integration_boundary_panel",
        ),
        advisory_outputs=(
            "timeline_visualization_integration_inventory",
            "manual_timeline_visualization_review_hint",
            "no_real_time_stream_subscription_notice",
        ),
    ),
)

MULTIMODAL_STUDIO_INTEGRATION_REGISTRY = MultimodalStudioIntegrationRegistry(
    integration_profiles=MULTIMODAL_STUDIO_INTEGRATION_PROFILES,
    integration_profile_ids=tuple(
        profile.integration_profile_id
        for profile in MULTIMODAL_STUDIO_INTEGRATION_PROFILES
    ),
    integration_kinds=tuple(
        profile.integration_kind for profile in MULTIMODAL_STUDIO_INTEGRATION_PROFILES
    ),
    live_preview_profile_ids=MULTIMODAL_LIVE_PREVIEW_REGISTRY.profile_ids,
    multi_preview_profile_ids=MULTIMODAL_MULTI_PREVIEW_REGISTRY.profile_ids,
    interactive_canvas_profile_ids=(
        MULTIMODAL_INTERACTIVE_CANVAS_REGISTRY.profile_ids
    ),
    visual_workspace_profile_ids=MULTIMODAL_VISUAL_WORKSPACE_REGISTRY.profile_ids,
    runtime_collaboration_profile_ids=(
        MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY.profile_ids
    ),
    artifact_collaboration_profile_ids=(
        MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY.profile_ids
    ),
    artifact_provenance_profile_ids=(
        MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY.profile_ids
    ),
    artifact_lineage_profile_ids=MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY.profile_ids,
    cross_agent_workspace_profile_ids=(
        MULTIMODAL_CROSS_AGENT_WORKSPACE_REGISTRY.profile_ids
    ),
    shared_artifact_board_profile_ids=(
        MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY.profile_ids
    ),
    workspace_history_profile_ids=MULTIMODAL_WORKSPACE_HISTORY_REGISTRY.profile_ids,
    branching_timeline_profile_ids=MULTIMODAL_BRANCHING_TIMELINE_REGISTRY.profile_ids,
    creative_evolution_timeline_profile_ids=(
        MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_REGISTRY.profile_ids
    ),
    real_time_workflow_visualization_profile_ids=(
        MULTIMODAL_REAL_TIME_WORKFLOW_VISUALIZATION_REGISTRY.profile_ids
    ),
    profile_group_refs=_MULTIMODAL_STUDIO_INTEGRATION_PROFILE_GROUPS,
    integration_surface_refs=_MULTIMODAL_STUDIO_INTEGRATION_SURFACES,
    route_names=tuple(RouteName),
    profile_count=len(MULTIMODAL_STUDIO_INTEGRATION_PROFILES),
    source_registries=_MULTIMODAL_STUDIO_INTEGRATION_SOURCE_REGISTRIES,
    observability_surfaces=_MULTIMODAL_STUDIO_INTEGRATION_OBSERVABILITY_SURFACES,
)
