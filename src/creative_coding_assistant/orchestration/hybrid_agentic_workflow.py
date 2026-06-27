"""Passive hybrid agentic workflow metadata for V4.3 preparation."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

BackboneModePhase = Literal[
    "context_intake",
    "planning_reasoning",
    "generation_artifact",
    "review_refinement",
    "terminal_guardrail",
]

V3_BACKBONE_MODE_ID = "v3_backbone_mode"
V3_BACKBONE_MODE_NODE_SERIALIZATION_VERSION = "v3_backbone_mode_node.v1"
V3_BACKBONE_MODE_REGISTRY_SERIALIZATION_VERSION = "v3_backbone_mode_registry.v1"
HYBRID_WORKFLOW_STAGE_SERIALIZATION_VERSION = "hybrid_workflow_stage.v1"
HYBRID_WORKFLOW_REGISTRY_SERIALIZATION_VERSION = "hybrid_workflow_registry.v1"
V3_BACKBONE_MODE_AUTHORITY_BOUNDARY = (
    "V3 Backbone Mode metadata declares the current deterministic V3 workflow "
    "graph as the active backbone for V4.3 hybrid workflow readiness only; it "
    "does not change workflow graph order, perform multi-agent escalation, "
    "invoke agents, route providers or models, select runtimes, trigger "
    "retries, mutate prompts, write memory, execute artifacts, or modify "
    "generated output."
)
HYBRID_WORKFLOW_REGISTRY_AUTHORITY_BOUNDARY = (
    "Hybrid agentic workflow metadata maps current V3 workflow nodes to future "
    "agent capability and escalation policy readiness only; it does not change "
    "workflow graph order, create agents, route providers or models, select "
    "runtimes, trigger retries, execute artifacts, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "workflow_graph_mutation",
    "provider_or_model_routing",
    "runtime_selection",
    "retry_or_refinement_triggering",
    "agent_invocation",
    "artifact_execution",
    "generated_output_modification",
)
_V3_BACKBONE_MODE_BLOCKED_RUNTIME_BEHAVIORS = (
    "workflow_graph_mutation",
    "provider_or_model_routing",
    "runtime_selection",
    "retry_or_refinement_triggering",
    "multi_agent_escalation_execution",
    "agent_invocation",
    "prompt_mutation",
    "memory_write",
    "generated_output_modification",
)
_V3_BACKBONE_MODE_SOURCE_REGISTRIES = (
    "assistant_workflow_node_order",
    "workflow_step_order",
    "artifact_engine_contract_registry",
    "evaluation_engine_contract_registry",
    "workstation_engine_contract_registry",
    "hybrid_agentic_workflow_registry",
)
_V3_BACKBONE_MODE_PHASE_IDS: tuple[BackboneModePhase, ...] = (
    "context_intake",
    "planning_reasoning",
    "generation_artifact",
    "review_refinement",
    "terminal_guardrail",
)


class V3BackboneModeNodeProfile(BaseModel):
    """Metadata-only profile for one preserved V3 workflow backbone node."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    mode_id: Literal["v3_backbone_mode"] = V3_BACKBONE_MODE_ID
    node_id: str = Field(min_length=1, max_length=80)
    phase: BackboneModePhase
    active_runtime_owner: Literal["v3_workflow_graph"] = "v3_workflow_graph"
    preserved_surfaces: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_V3_BACKBONE_MODE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    backbone_runtime_active: Literal[True] = True
    workflow_order_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    multi_agent_escalation_executed: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["v3_backbone_mode_node.v1"] = (
        V3_BACKBONE_MODE_NODE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class V3BackboneModeRegistry(BaseModel):
    """Stable passive registry declaring the preserved V3 workflow backbone."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["v3_backbone_mode_registry"] = "v3_backbone_mode_registry"
    mode_id: Literal["v3_backbone_mode"] = V3_BACKBONE_MODE_ID
    serialization_version: Literal["v3_backbone_mode_registry.v1"] = (
        V3_BACKBONE_MODE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=V3_BACKBONE_MODE_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    node_profiles: tuple[V3BackboneModeNodeProfile, ...] = Field(
        min_length=18,
        max_length=18,
    )
    node_ids: tuple[str, ...] = Field(min_length=18, max_length=18)
    preserved_workflow_order: tuple[str, ...] = Field(min_length=18, max_length=18)
    phase_ids: tuple[BackboneModePhase, ...] = Field(min_length=5, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    node_count: int = Field(ge=18, le=18)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_V3_BACKBONE_MODE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    backbone_runtime_active: Literal[True] = True
    workflow_order_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    multi_agent_escalation_executed: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_backbone_profiles(self) -> Self:
        derived_node_ids = tuple(profile.node_id for profile in self.node_profiles)
        derived_phase_ids = tuple(
            dict.fromkeys(profile.phase for profile in self.node_profiles)
        )
        if self.node_ids != derived_node_ids:
            raise ValueError("node_ids must match node_profiles")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("node_ids must be unique")
        if self.preserved_workflow_order != self.node_ids:
            raise ValueError("preserved_workflow_order must match node_ids")
        if self.phase_ids != derived_phase_ids:
            raise ValueError("phase_ids must match node profile phases")
        if self.node_count != len(self.node_profiles):
            raise ValueError("node_count must match node_profiles")

        source_registries = set(self.source_registries)
        profile_source_registries = {
            source_registry
            for profile in self.node_profiles
            for source_registry in profile.source_registries
        }
        if source_registries != profile_source_registries:
            raise ValueError("source_registries must match node profile sources")
        for profile in self.node_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("node profile sources must match registry sources")
            if profile.workflow_order_mutation_implemented:
                raise ValueError("V3 backbone profiles must not mutate workflow order")
            if profile.multi_agent_escalation_executed:
                raise ValueError("V3 backbone profiles must not execute escalation")
        return self


def v3_backbone_mode_registry() -> V3BackboneModeRegistry:
    """Return passive V3 backbone mode metadata without changing workflow behavior."""

    return V3_BACKBONE_MODE_REGISTRY


def v3_backbone_mode_profile_by_node_id(
    node_id: str,
    registry: V3BackboneModeRegistry | None = None,
) -> V3BackboneModeNodeProfile | None:
    """Return one V3 backbone node profile without executing workflow changes."""

    source_registry = registry or V3_BACKBONE_MODE_REGISTRY
    for profile in source_registry.node_profiles:
        if profile.node_id == node_id:
            return profile
    return None


class HybridAgenticWorkflowStage(BaseModel):
    """Metadata-only future hybrid workflow readiness stage."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    stage_id: str = Field(min_length=1, max_length=80)
    stage_name: str = Field(min_length=1, max_length=140)
    authority_boundary: str = Field(min_length=1, max_length=900)
    v3_workflow_nodes: tuple[str, ...] = Field(min_length=1, max_length=8)
    future_capability_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    escalation_rule_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_metadata_registries: tuple[str, ...] = Field(min_length=1, max_length=6)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=1, max_length=12)
    serialization_version: Literal["hybrid_workflow_stage.v1"] = (
        HYBRID_WORKFLOW_STAGE_SERIALIZATION_VERSION
    )


class HybridAgenticWorkflowRegistry(BaseModel):
    """Stable metadata registry for future hybrid agentic workflow readiness."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["hybrid_agentic_workflow_registry"] = (
        "hybrid_agentic_workflow_registry"
    )
    serialization_version: Literal["hybrid_workflow_registry.v1"] = (
        HYBRID_WORKFLOW_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=HYBRID_WORKFLOW_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    stages: tuple[HybridAgenticWorkflowStage, ...] = Field(
        min_length=5,
        max_length=5,
    )
    stage_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    stage_count: int = Field(ge=5, le=5)
    source_metadata_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    metadata_only: Literal[True] = True


def hybrid_agentic_workflow_registry() -> HybridAgenticWorkflowRegistry:
    """Return the static future hybrid workflow readiness registry."""

    return HYBRID_AGENTIC_WORKFLOW_REGISTRY


def hybrid_agentic_workflow_stage_by_id(
    stage_id: str,
) -> HybridAgenticWorkflowStage | None:
    """Return one hybrid workflow readiness stage without changing behavior."""

    for stage in HYBRID_AGENTIC_WORKFLOW_STAGES:
        if stage.stage_id == stage_id:
            return stage
    return None


def _backbone_profile(
    *,
    node_id: str,
    phase: BackboneModePhase,
    preserved_surfaces: tuple[str, ...],
) -> V3BackboneModeNodeProfile:
    return V3BackboneModeNodeProfile(
        node_id=node_id,
        phase=phase,
        preserved_surfaces=preserved_surfaces,
        source_registries=_V3_BACKBONE_MODE_SOURCE_REGISTRIES,
        authority_boundary=(
            "This node remains part of the deterministic V3 workflow backbone; "
            "its metadata does not change workflow order, perform multi-agent "
            "escalation, invoke agents, route providers or models, mutate "
            "prompts, write memory, or modify generated output."
        ),
    )


def _stage(
    *,
    stage_id: str,
    stage_name: str,
    v3_workflow_nodes: tuple[str, ...],
    future_capability_ids: tuple[str, ...],
    escalation_rule_ids: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> HybridAgenticWorkflowStage:
    return HybridAgenticWorkflowStage(
        stage_id=stage_id,
        stage_name=stage_name,
        authority_boundary=(
            "This stage is future hybrid workflow readiness metadata only; it "
            "does not change V3 workflow graph order, create agents, route "
            "providers or models, select runtimes, trigger retries, execute "
            "artifacts, or modify generated output."
        ),
        v3_workflow_nodes=v3_workflow_nodes,
        future_capability_ids=future_capability_ids,
        escalation_rule_ids=escalation_rule_ids,
        source_metadata_registries=(
            "agent_capability_registry",
            "escalation_policy_registry",
            "artifact_engine_contract_registry",
            "evaluation_engine_contract_registry",
            "workstation_engine_contract_registry",
        ),
        advisory_outputs=advisory_outputs,
        blocked_runtime_behaviors=_BLOCKED_RUNTIME_BEHAVIORS,
    )


V3_BACKBONE_MODE_NODE_PROFILES = (
    _backbone_profile(
        node_id="intake",
        phase="context_intake",
        preserved_surfaces=("request_received_event", "workflow_state"),
    ),
    _backbone_profile(
        node_id="routing",
        phase="context_intake",
        preserved_surfaces=("route_decision", "workflow_transition"),
    ),
    _backbone_profile(
        node_id="memory",
        phase="context_intake",
        preserved_surfaces=("memory_context", "project_memory_context"),
    ),
    _backbone_profile(
        node_id="retrieval",
        phase="context_intake",
        preserved_surfaces=("retrieval_context", "knowledge_chunks"),
    ),
    _backbone_profile(
        node_id="context_assembly",
        phase="context_intake",
        preserved_surfaces=("assembled_context", "context_summary"),
    ),
    _backbone_profile(
        node_id="prompt_input",
        phase="planning_reasoning",
        preserved_surfaces=("prompt_input", "operator_request_shape"),
    ),
    _backbone_profile(
        node_id="planning",
        phase="planning_reasoning",
        preserved_surfaces=("creative_plan", "planning_metadata"),
    ),
    _backbone_profile(
        node_id="director",
        phase="planning_reasoning",
        preserved_surfaces=("director_brief", "creative_direction"),
    ),
    _backbone_profile(
        node_id="reasoning",
        phase="planning_reasoning",
        preserved_surfaces=("creative_reasoning", "reasoning_evidence"),
    ),
    _backbone_profile(
        node_id="prompt_rendering",
        phase="planning_reasoning",
        preserved_surfaces=("rendered_prompt", "prompt_sections"),
    ),
    _backbone_profile(
        node_id="generation",
        phase="generation_artifact",
        preserved_surfaces=("generation_stream", "provider_generation_request"),
    ),
    _backbone_profile(
        node_id="artifact_extraction",
        phase="generation_artifact",
        preserved_surfaces=("workflow_artifacts", "artifact_metadata"),
    ),
    _backbone_profile(
        node_id="preview_preparation",
        phase="generation_artifact",
        preserved_surfaces=("preview_results", "preview_runtime_metadata"),
    ),
    _backbone_profile(
        node_id="artifact_critique",
        phase="generation_artifact",
        preserved_surfaces=("artifact_critique", "quality_observations"),
    ),
    _backbone_profile(
        node_id="review",
        phase="review_refinement",
        preserved_surfaces=("workflow_review", "hitl_recommendation"),
    ),
    _backbone_profile(
        node_id="refinement",
        phase="review_refinement",
        preserved_surfaces=("refinement_history", "refinement_decision"),
    ),
    _backbone_profile(
        node_id="finalization",
        phase="terminal_guardrail",
        preserved_surfaces=("final_workflow_event", "final_answer"),
    ),
    _backbone_profile(
        node_id="failure",
        phase="terminal_guardrail",
        preserved_surfaces=("failure_info", "failure_answer"),
    ),
)
V3_BACKBONE_MODE_REGISTRY = V3BackboneModeRegistry(
    node_profiles=V3_BACKBONE_MODE_NODE_PROFILES,
    node_ids=tuple(profile.node_id for profile in V3_BACKBONE_MODE_NODE_PROFILES),
    preserved_workflow_order=tuple(
        profile.node_id for profile in V3_BACKBONE_MODE_NODE_PROFILES
    ),
    phase_ids=_V3_BACKBONE_MODE_PHASE_IDS,
    source_registries=_V3_BACKBONE_MODE_SOURCE_REGISTRIES,
    node_count=len(V3_BACKBONE_MODE_NODE_PROFILES),
)

HYBRID_AGENTIC_WORKFLOW_STAGES = (
    _stage(
        stage_id="intake_routing_context_readiness",
        stage_name="Intake Routing Context Readiness",
        v3_workflow_nodes=(
            "intake",
            "routing",
            "memory",
            "retrieval",
            "context_assembly",
        ),
        future_capability_ids=("v4_agent_router",),
        escalation_rule_ids=("missing_information_review",),
        advisory_outputs=(
            "routing_context_packet",
            "retrieval_gap_summary",
            "context_handoff_notes",
        ),
    ),
    _stage(
        stage_id="planning_reasoning_readiness",
        stage_name="Planning Reasoning Readiness",
        v3_workflow_nodes=(
            "prompt_input",
            "planning",
            "director",
            "reasoning",
            "prompt_rendering",
        ),
        future_capability_ids=("v4_planner_agent", "v4_agentic_studio"),
        escalation_rule_ids=(
            "missing_information_review",
            "evaluation_confidence_review",
        ),
        advisory_outputs=(
            "planning_context_packet",
            "reasoning_review_notes",
            "prompt_handoff_summary",
        ),
    ),
    _stage(
        stage_id="generation_artifact_readiness",
        stage_name="Generation Artifact Readiness",
        v3_workflow_nodes=(
            "generation",
            "artifact_extraction",
            "preview_preparation",
            "artifact_critique",
        ),
        future_capability_ids=("v4_artifact_agent", "v4_runtime_agent"),
        escalation_rule_ids=(
            "artifact_risk_review",
            "runtime_incompatibility_review",
        ),
        advisory_outputs=(
            "artifact_context_packet",
            "runtime_fit_notes",
            "preview_readiness_summary",
        ),
    ),
    _stage(
        stage_id="review_refinement_readiness",
        stage_name="Review Refinement Readiness",
        v3_workflow_nodes=("review", "refinement"),
        future_capability_ids=(
            "v4_agentic_studio",
            "adaptive_multi_agent_escalation",
        ),
        escalation_rule_ids=(
            "evaluation_confidence_review",
            "future_agent_escalation_readiness",
        ),
        advisory_outputs=(
            "review_context_packet",
            "refinement_candidate_summary",
            "human_review_posture",
        ),
    ),
    _stage(
        stage_id="completion_guardrail_readiness",
        stage_name="Completion Guardrail Readiness",
        v3_workflow_nodes=("finalization", "failure"),
        future_capability_ids=(
            "v4_agent_router",
            "adaptive_multi_agent_escalation",
        ),
        escalation_rule_ids=("future_agent_escalation_readiness",),
        advisory_outputs=(
            "completion_context_packet",
            "failure_guardrail_notes",
            "final_handoff_summary",
        ),
    ),
)

HYBRID_AGENTIC_WORKFLOW_REGISTRY = HybridAgenticWorkflowRegistry(
    stages=HYBRID_AGENTIC_WORKFLOW_STAGES,
    stage_ids=tuple(stage.stage_id for stage in HYBRID_AGENTIC_WORKFLOW_STAGES),
    stage_count=len(HYBRID_AGENTIC_WORKFLOW_STAGES),
    source_metadata_registries=(
        "agent_capability_registry",
        "escalation_policy_registry",
        "artifact_engine_contract_registry",
        "evaluation_engine_contract_registry",
        "workstation_engine_contract_registry",
    ),
)
