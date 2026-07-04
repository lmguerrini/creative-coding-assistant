"""Passive future-agent capability metadata registry for V3.6 preparation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AgentCapabilityStage = Literal[
    "v4_agent_readiness",
    "adaptive_escalation_readiness",
]

AGENT_CAPABILITY_SERIALIZATION_VERSION = "agent_capability.v1"
AGENT_CAPABILITY_REGISTRY_SERIALIZATION_VERSION = "agent_capability_registry.v1"
AGENT_CAPABILITY_REGISTRY_AUTHORITY_BOUNDARY = (
    "Agent capability registry metadata describes future agent readiness, "
    "source contract registries, advisory outputs, and blocked runtime "
    "behaviors only; it does not create agents, route providers or models, "
    "select runtimes, trigger retries, change workflow control, execute "
    "artifacts, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "provider_or_model_routing",
    "runtime_selection",
    "workflow_control",
    "retry_or_refinement_triggering",
    "artifact_execution",
    "generated_output_modification",
)


class AgentCapabilityProfile(BaseModel):
    """Metadata-only future agent capability profile."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    capability_id: str = Field(min_length=1, max_length=80)
    capability_name: str = Field(min_length=1, max_length=140)
    capability_stage: AgentCapabilityStage
    authority_boundary: str = Field(min_length=1, max_length=900)
    source_contract_registries: tuple[str, ...] = Field(min_length=1, max_length=6)
    required_metadata_sources: tuple[str, ...] = Field(min_length=1, max_length=12)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=12)
    readiness_signals: tuple[str, ...] = Field(min_length=1, max_length=12)
    future_agent_hooks: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=1, max_length=12)
    serialization_version: Literal["agent_capability.v1"] = (
        AGENT_CAPABILITY_SERIALIZATION_VERSION
    )


class AgentCapabilityRegistry(BaseModel):
    """Stable metadata registry for future agent capability preparation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_capability_registry"] = "agent_capability_registry"
    serialization_version: Literal["agent_capability_registry.v1"] = (
        AGENT_CAPABILITY_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_CAPABILITY_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    capabilities: tuple[AgentCapabilityProfile, ...] = Field(
        min_length=6,
        max_length=6,
    )
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    source_contract_registries: tuple[str, ...] = Field(min_length=3, max_length=3)
    metadata_only: Literal[True] = True


def agent_capability_registry() -> AgentCapabilityRegistry:
    """Return the static future agent capability registry."""

    return AGENT_CAPABILITY_REGISTRY


def agent_capability_by_id(capability_id: str) -> AgentCapabilityProfile | None:
    """Return one future capability profile without changing runtime behavior."""

    for capability in AGENT_CAPABILITIES:
        if capability.capability_id == capability_id:
            return capability
    return None


def _capability(
    *,
    capability_id: str,
    capability_name: str,
    capability_stage: AgentCapabilityStage,
    source_contract_registries: tuple[str, ...],
    required_metadata_sources: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
    readiness_signals: tuple[str, ...],
    future_agent_hooks: tuple[str, ...],
) -> AgentCapabilityProfile:
    return AgentCapabilityProfile(
        capability_id=capability_id,
        capability_name=capability_name,
        capability_stage=capability_stage,
        authority_boundary=(
            "This capability is a future-readiness metadata contract only; it "
            "does not create agents, alter routing, select runtimes, trigger "
            "workflow control, execute artifacts, or modify generated output."
        ),
        source_contract_registries=source_contract_registries,
        required_metadata_sources=required_metadata_sources,
        advisory_outputs=advisory_outputs,
        readiness_signals=readiness_signals,
        future_agent_hooks=future_agent_hooks,
        blocked_runtime_behaviors=_BLOCKED_RUNTIME_BEHAVIORS,
    )


AGENT_CAPABILITIES = (
    _capability(
        capability_id="v4_planner_agent",
        capability_name="V4 Planner Agent Readiness",
        capability_stage="v4_agent_readiness",
        source_contract_registries=(
            "artifact_engine_contract_registry",
            "evaluation_engine_contract_registry",
        ),
        required_metadata_sources=(
            "assistant_request",
            "route_decision",
            "creative_intelligence_metadata",
            "artifact_engine_contracts",
            "evaluation_engine_contracts",
        ),
        advisory_outputs=(
            "planner_context_packet",
            "planning_gap_summary",
            "planning_hitl_candidates",
        ),
        readiness_signals=(
            "confidence_signals",
            "ambiguity_signals",
            "risk_signals",
            "hitl_questions",
        ),
        future_agent_hooks=("v4_planner_agent_contract",),
    ),
    _capability(
        capability_id="v4_artifact_agent",
        capability_name="V4 Artifact Agent Readiness",
        capability_stage="v4_agent_readiness",
        source_contract_registries=(
            "artifact_engine_contract_registry",
            "workstation_engine_contract_registry",
        ),
        required_metadata_sources=(
            "artifact_intelligence_metadata",
            "artifact_engine_contracts",
            "workstation_engine_contracts",
        ),
        advisory_outputs=(
            "artifact_context_packet",
            "artifact_risk_summary",
            "artifact_readiness_notes",
        ),
        readiness_signals=(
            "implementation_readiness",
            "implementation_risk",
            "export_readiness",
            "escalation_candidates",
        ),
        future_agent_hooks=("v4_artifact_agent_contract",),
    ),
    _capability(
        capability_id="v4_runtime_agent",
        capability_name="V4 Runtime Agent Readiness",
        capability_stage="v4_agent_readiness",
        source_contract_registries=(
            "artifact_engine_contract_registry",
            "workstation_engine_contract_registry",
        ),
        required_metadata_sources=(
            "runtime_capabilities",
            "runtime_compatibility",
            "artifact_capability_matrix",
            "workstation_engine_contracts",
        ),
        advisory_outputs=(
            "runtime_context_packet",
            "runtime_fit_summary",
            "compatibility_caveats",
        ),
        readiness_signals=(
            "runtime_confidence",
            "compatible_runtimes",
            "unsupported_runtimes",
            "runtime_fit_status",
        ),
        future_agent_hooks=("v4_runtime_agent_contract",),
    ),
    _capability(
        capability_id="v4_agent_router",
        capability_name="V4 Agent Router Readiness",
        capability_stage="v4_agent_readiness",
        source_contract_registries=(
            "artifact_engine_contract_registry",
            "evaluation_engine_contract_registry",
            "workstation_engine_contract_registry",
        ),
        required_metadata_sources=(
            "route_decision",
            "artifact_engine_contracts",
            "evaluation_engine_contracts",
            "workstation_engine_contracts",
        ),
        advisory_outputs=(
            "agent_route_context",
            "handoff_candidate_summary",
            "routing_guardrail_notes",
        ),
        readiness_signals=(
            "future_agent_hooks",
            "cacheability",
            "parallelization_support",
            "estimated_cost_metadata",
            "estimated_latency_metadata",
        ),
        future_agent_hooks=("v4_agent_router_contract",),
    ),
    _capability(
        capability_id="v4_agentic_studio",
        capability_name="V4 Agentic Studio Readiness",
        capability_stage="v4_agent_readiness",
        source_contract_registries=(
            "workstation_engine_contract_registry",
            "evaluation_engine_contract_registry",
        ),
        required_metadata_sources=(
            "workstation_state",
            "v3_inspector_panels",
            "workstation_dashboard",
            "creative_evaluation_metadata",
        ),
        advisory_outputs=(
            "studio_context_packet",
            "operator_review_surface",
            "dashboard_handoff_summary",
        ),
        readiness_signals=(
            "metadata_group_status",
            "hitl_recommendation",
            "evaluation_integrity",
            "workflow_health_card",
        ),
        future_agent_hooks=(
            "v4_agentic_studio_context_packet",
            "v4_agentic_review_surface",
        ),
    ),
    _capability(
        capability_id="adaptive_multi_agent_escalation",
        capability_name="Adaptive Multi-Agent Escalation Readiness",
        capability_stage="adaptive_escalation_readiness",
        source_contract_registries=(
            "artifact_engine_contract_registry",
            "evaluation_engine_contract_registry",
        ),
        required_metadata_sources=(
            "artifact_intelligence_metadata",
            "creative_evaluation_metadata",
            "hitl_signals",
            "escalation_candidates",
        ),
        advisory_outputs=(
            "escalation_context_packet",
            "human_review_posture",
            "agent_escalation_candidates",
        ),
        readiness_signals=(
            "escalation_candidates",
            "hitl_questions",
            "hitl_recommendation",
            "risk_signals",
            "ambiguity_signals",
        ),
        future_agent_hooks=("adaptive_multi_agent_escalation_contract",),
    ),
)

AGENT_CAPABILITY_REGISTRY = AgentCapabilityRegistry(
    capabilities=AGENT_CAPABILITIES,
    capability_ids=tuple(capability.capability_id for capability in AGENT_CAPABILITIES),
    capability_count=len(AGENT_CAPABILITIES),
    source_contract_registries=(
        "artifact_engine_contract_registry",
        "evaluation_engine_contract_registry",
        "workstation_engine_contract_registry",
    ),
)
