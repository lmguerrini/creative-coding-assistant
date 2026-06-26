"""Passive hybrid agentic workflow metadata for V3.6 preparation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

HYBRID_WORKFLOW_STAGE_SERIALIZATION_VERSION = "hybrid_workflow_stage.v1"
HYBRID_WORKFLOW_REGISTRY_SERIALIZATION_VERSION = "hybrid_workflow_registry.v1"
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
