"""V5.5 advisory agent activation optimization intelligence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_escalation_optimizer import (
    EscalationOptimizationPlan,
    optimize_escalation_policy,
)
from creative_coding_assistant.orchestration.agent_capabilities import (
    AgentCapabilityRegistry,
    agent_capability_registry,
)
from creative_coding_assistant.orchestration.agent_lifecycle import (
    AgentLifecycleRegistry,
    agent_lifecycle_registry,
)
from creative_coding_assistant.orchestration.agent_metadata import (
    AgentMetadataRegistry,
    agent_metadata_registry,
)
from creative_coding_assistant.orchestration.agent_routing import (
    AgentRoutingPriorityBand,
    AgentRoutingProfile,
    AgentRoutingRegistry,
    agent_routing_profiles_for_route,
    agent_routing_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

AgentActivationStatus = Literal["recommended", "standby", "requires_hitl"]

AGENT_ACTIVATION_CANDIDATE_SERIALIZATION_VERSION = (
    "agent_activation_optimization_candidate.v1"
)
AGENT_ACTIVATION_PLAN_SERIALIZATION_VERSION = (
    "agent_activation_optimization_plan.v1"
)
AGENT_ACTIVATION_OPTIMIZER_AUTHORITY_BOUNDARY = (
    "V5.5 agent activation optimization ranks existing passive agent routing, "
    "metadata, capability, lifecycle, and escalation posture into inspectable "
    "activation recommendations only; it does not instantiate agents, invoke "
    "agents, run lifecycle transitions, change workflow routing, select "
    "runtimes, route providers or models, call providers, trigger retries, "
    "mutate memory, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "agent_instantiation",
    "agent_invocation",
    "agent_activation",
    "runtime_lifecycle_engine",
    "state_transition_execution",
    "workflow_routing_change",
    "workflow_control",
    "runtime_selection",
    "provider_or_model_routing",
    "provider_execution",
    "retry_or_refinement_triggering",
    "memory_storage_or_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class AgentActivationOptimizationCandidate(BaseModel):
    """One advisory agent activation candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=160)
    agent_id: str = Field(min_length=1, max_length=80)
    role_id: str = Field(min_length=1, max_length=80)
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    priority_band: AgentRoutingPriorityBand
    route_candidate_count: int = Field(ge=1, le=6)
    required_metadata_input_count: int = Field(ge=1, le=40)
    produced_metadata_output_count: int = Field(ge=1, le=40)
    source_lifecycle_profile_id: str = Field(min_length=1, max_length=160)
    source_capability_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    estimated_cost_class: str = Field(min_length=1, max_length=40)
    estimated_latency_class: str = Field(min_length=1, max_length=40)
    activation_order: int = Field(ge=1, le=20)
    activation_score: int = Field(ge=0, le=240)
    status: AgentActivationStatus
    hitl_required: bool
    escalation_posture: str = Field(min_length=1, max_length=80)
    activation_reason: str = Field(min_length=1, max_length=320)
    fallback_summary: str = Field(min_length=1, max_length=320)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    agent_activation_optimizer_implemented: Literal[True] = True
    agent_activation_recommendation_implemented: Literal[True] = True
    agent_activation_implemented: Literal[False] = False
    agent_instantiation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    lifecycle_transition_execution_implemented: Literal[False] = False
    workflow_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_activation_optimization_candidate.v1"] = (
        AGENT_ACTIVATION_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_identity(self) -> Self:
        if self.candidate_id != f"agent_activation::{self.agent_id}":
            raise ValueError("candidate_id must match agent_id")
        if self.source_lifecycle_profile_id != f"{self.agent_id}_lifecycle_profile":
            raise ValueError("source_lifecycle_profile_id must match agent_id")
        if self.status == "requires_hitl" and not self.hitl_required:
            raise ValueError("requires_hitl status must set hitl_required")
        if self.activation_order == 1 and self.status not in {
            "recommended",
            "requires_hitl",
        }:
            raise ValueError("first activation candidate must be recommended or gated")
        return self


class AgentActivationOptimizationPlan(BaseModel):
    """Bounded V5.5 advisory agent activation optimization plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_activation_optimizer"] = "agent_activation_optimizer"
    serialization_version: Literal["agent_activation_optimization_plan.v1"] = (
        AGENT_ACTIVATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_ACTIVATION_OPTIMIZER_AUTHORITY_BOUNDARY,
        max_length=1600,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_agent_routing_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_agent_metadata_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_agent_capability_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_agent_lifecycle_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_escalation_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    candidates: tuple[AgentActivationOptimizationCandidate, ...] = Field(
        min_length=1,
        max_length=12,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    recommended_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    standby_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    hitl_required_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    candidate_count: int = Field(ge=1, le=12)
    highest_activation_score: int = Field(ge=0, le=240)
    activation_recommendation_count: int = Field(ge=0, le=12)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    agent_activation_optimizer_implemented: Literal[True] = True
    agent_activation_recommendation_implemented: Literal[True] = True
    agent_activation_implemented: Literal[False] = False
    agent_instantiation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    lifecycle_transition_execution_implemented: Literal[False] = False
    workflow_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_candidates(self) -> Self:
        derived_candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(set(derived_candidate_ids)) != len(derived_candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if self.candidate_ids != derived_candidate_ids:
            raise ValueError("candidate_ids must match candidates")
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count must match candidates")
        if self.recommended_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "recommended",
        ):
            raise ValueError("recommended_candidate_ids must match candidates")
        if self.standby_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "standby",
        ):
            raise ValueError("standby_candidate_ids must match candidates")
        if self.hitl_required_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "requires_hitl",
        ):
            raise ValueError("hitl_required_candidate_ids must match candidates")
        if self.highest_activation_score != max(
            candidate.activation_score for candidate in self.candidates
        ):
            raise ValueError("highest_activation_score must match candidates")
        if self.activation_recommendation_count != (
            len(self.recommended_candidate_ids) + len(self.hitl_required_candidate_ids)
        ):
            raise ValueError("activation_recommendation_count must match candidates")
        return self


def optimize_agent_activation(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    agent_routing: AgentRoutingRegistry | None = None,
    agent_metadata: AgentMetadataRegistry | None = None,
    agent_capabilities: AgentCapabilityRegistry | None = None,
    agent_lifecycle: AgentLifecycleRegistry | None = None,
    escalation: EscalationOptimizationPlan | None = None,
) -> AgentActivationOptimizationPlan:
    """Recommend agent activation candidates without activating agents."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    escalation_plan = escalation or optimize_escalation_policy(
        task_type=normalized_task_type,
        route=route_name,
        execution_mode_id=execution_mode_id,
    )
    normalized_mode = str(execution_mode_id or escalation_plan.decisions[0].execution_mode_id)
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")

    routing_registry = agent_routing or agent_routing_registry()
    metadata_registry = agent_metadata or agent_metadata_registry()
    capability_registry = agent_capabilities or agent_capability_registry()
    lifecycle_registry = agent_lifecycle or agent_lifecycle_registry()
    profiles = agent_routing_profiles_for_route(route_name, routing_registry)
    candidates = _ranked_candidates(
        profiles=profiles,
        metadata_registry=metadata_registry,
        capability_registry=capability_registry,
        lifecycle_registry=lifecycle_registry,
        route_name=route_name,
        task_type=escalation_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        escalation_plan=escalation_plan,
    )
    return AgentActivationOptimizationPlan(
        route_name=route_name,
        task_type=escalation_plan.task_type,
        source_agent_routing_serialization_version=routing_registry.serialization_version,
        source_agent_metadata_serialization_version=metadata_registry.serialization_version,
        source_agent_capability_serialization_version=capability_registry.serialization_version,
        source_agent_lifecycle_serialization_version=lifecycle_registry.serialization_version,
        source_escalation_optimization_serialization_version=(
            escalation_plan.serialization_version
        ),
        execution_mode_ids=execution_modes.execution_mode_ids,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        recommended_candidate_ids=_candidate_ids_for_status(candidates, "recommended"),
        standby_candidate_ids=_candidate_ids_for_status(candidates, "standby"),
        hitl_required_candidate_ids=_candidate_ids_for_status(candidates, "requires_hitl"),
        candidate_count=len(candidates),
        highest_activation_score=max(candidate.activation_score for candidate in candidates),
        activation_recommendation_count=sum(
            1 for candidate in candidates if candidate.status != "standby"
        ),
        advisory_actions=_plan_actions(escalation_plan),
    )


def agent_activation_candidate_by_agent_id(
    agent_id: str,
    plan: AgentActivationOptimizationPlan | None = None,
) -> AgentActivationOptimizationCandidate | None:
    """Return one agent activation candidate without activating it."""

    source_plan = plan or optimize_agent_activation()
    for candidate in source_plan.candidates:
        if candidate.agent_id == agent_id:
            return candidate
    return None


def agent_activation_candidates_for_status(
    status: AgentActivationStatus,
    plan: AgentActivationOptimizationPlan | None = None,
) -> tuple[AgentActivationOptimizationCandidate, ...]:
    """Return candidates for one advisory activation status."""

    source_plan = plan or optimize_agent_activation()
    return tuple(candidate for candidate in source_plan.candidates if candidate.status == status)


def _ranked_candidates(
    *,
    profiles: tuple[AgentRoutingProfile, ...],
    metadata_registry: AgentMetadataRegistry,
    capability_registry: AgentCapabilityRegistry,
    lifecycle_registry: AgentLifecycleRegistry,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    escalation_plan: EscalationOptimizationPlan,
) -> tuple[AgentActivationOptimizationCandidate, ...]:
    source_candidates = tuple(
        _candidate(
            profile=profile,
            metadata_registry=metadata_registry,
            capability_registry=capability_registry,
            lifecycle_registry=lifecycle_registry,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            escalation_plan=escalation_plan,
        )
        for profile in profiles
    )
    ranked = sorted(
        source_candidates,
        key=lambda candidate: (-candidate.activation_score, candidate.agent_id),
    )
    return tuple(
        candidate.model_copy(
            update={
                "activation_order": index,
                "status": _status_for_candidate(index, escalation_plan),
                "hitl_required": _hitl_required(index, escalation_plan),
            }
        )
        for index, candidate in enumerate(ranked, start=1)
    )


def _candidate(
    *,
    profile: AgentRoutingProfile,
    metadata_registry: AgentMetadataRegistry,
    capability_registry: AgentCapabilityRegistry,
    lifecycle_registry: AgentLifecycleRegistry,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    escalation_plan: EscalationOptimizationPlan,
) -> AgentActivationOptimizationCandidate:
    metadata = next(item for item in metadata_registry.metadata if item.agent_id == profile.agent_id)
    lifecycle = next(item for item in lifecycle_registry.profiles if item.agent_id == profile.agent_id)
    capability_ids = _capability_ids_for_profile(profile, capability_registry)
    score = _activation_score(profile, metadata.estimated_cost_class, metadata.estimated_latency_class)
    return AgentActivationOptimizationCandidate(
        candidate_id=f"agent_activation::{profile.agent_id}",
        agent_id=profile.agent_id,
        role_id=profile.role_id,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        priority_band=profile.priority_band,
        route_candidate_count=len(profile.route_candidates),
        required_metadata_input_count=len(profile.required_metadata_inputs),
        produced_metadata_output_count=len(profile.produced_metadata_outputs),
        source_lifecycle_profile_id=lifecycle.lifecycle_profile_id,
        source_capability_ids=capability_ids,
        estimated_cost_class=metadata.estimated_cost_class,
        estimated_latency_class=metadata.estimated_latency_class,
        activation_order=1,
        activation_score=score,
        status="recommended",
        hitl_required=escalation_plan.optimized_escalation_posture == "requires_hitl",
        escalation_posture=escalation_plan.optimized_escalation_posture,
        activation_reason=(
            "Agent route applicability and passive metadata readiness support "
            "activation recommendation."
        ),
        fallback_summary=(
            "Keep the agent as standby metadata if HITL, lifecycle, routing, or "
            "workflow boundaries are not approved."
        ),
        evidence=(
            f"priority_band:{profile.priority_band}",
            f"route_candidates:{len(profile.route_candidates)}",
            f"required_inputs:{len(profile.required_metadata_inputs)}",
            f"produced_outputs:{len(profile.produced_metadata_outputs)}",
            f"escalation_posture:{escalation_plan.optimized_escalation_posture}",
        ),
    )


def _activation_score(
    profile: AgentRoutingProfile,
    cost_class: str,
    latency_class: str,
) -> int:
    priority_points = {
        "foundational_context": 80,
        "final_synthesis": 74,
        "domain_context": 66,
        "execution_context": 62,
        "quality_review": 58,
        "refinement_context": 54,
    }[profile.priority_band]
    cost_points = {"low": 24, "medium": 16, "high": 8}.get(cost_class, 12)
    latency_points = {"low": 24, "medium": 16, "high": 8}.get(latency_class, 12)
    route_points = min(24, len(profile.route_candidates) * 4)
    return min(240, priority_points + cost_points + latency_points + route_points)


def _status_for_candidate(
    activation_order: int,
    escalation_plan: EscalationOptimizationPlan,
) -> AgentActivationStatus:
    if activation_order <= 3:
        if escalation_plan.optimized_escalation_posture == "requires_hitl":
            return "requires_hitl"
        return "recommended"
    return "standby"


def _hitl_required(
    activation_order: int,
    escalation_plan: EscalationOptimizationPlan,
) -> bool:
    return (
        activation_order <= 3
        and escalation_plan.optimized_escalation_posture == "requires_hitl"
    )


def _capability_ids_for_profile(
    profile: AgentRoutingProfile,
    registry: AgentCapabilityRegistry,
) -> tuple[str, ...]:
    if profile.priority_band == "execution_context":
        wanted = ("v4_runtime_agent", "v4_artifact_agent")
    elif profile.priority_band == "quality_review":
        wanted = ("adaptive_multi_agent_escalation", "v4_agentic_studio")
    elif profile.priority_band == "final_synthesis":
        wanted = ("v4_agent_router", "adaptive_multi_agent_escalation")
    elif profile.priority_band == "refinement_context":
        wanted = ("adaptive_multi_agent_escalation", "v4_agent_router")
    elif profile.agent_id == "planner_agent":
        wanted = ("v4_planner_agent", "v4_agent_router")
    else:
        wanted = ("v4_planner_agent", "v4_agentic_studio")
    known = set(registry.capability_ids)
    return tuple(capability_id for capability_id in wanted if capability_id in known)


def _candidate_ids_for_status(
    candidates: tuple[AgentActivationOptimizationCandidate, ...],
    status: AgentActivationStatus,
) -> tuple[str, ...]:
    return tuple(candidate.candidate_id for candidate in candidates if candidate.status == status)


def _plan_actions(
    escalation_plan: EscalationOptimizationPlan,
) -> tuple[str, ...]:
    actions = [
        "Expose agent activation recommendations as advisory metadata only.",
        "Preserve agent instantiation, invocation, lifecycle, routing, workflow, provider, memory, storage, and output boundaries.",
    ]
    if escalation_plan.optimized_escalation_posture == "requires_hitl":
        actions.append("Require HITL before any future agent activation behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
