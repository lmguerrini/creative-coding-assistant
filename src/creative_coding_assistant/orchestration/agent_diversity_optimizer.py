"""V5.5 advisory agent diversity optimization intelligence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_capability_alignment import (
    AgentCapabilityAlignmentProfile,
    AgentCapabilityAlignmentRegistry,
    agent_capability_alignment_by_agent_id,
    agent_capability_alignment_registry,
)
from creative_coding_assistant.orchestration.agent_identities import (
    AgentCapabilityClass,
    AgentRoleFamily,
)
from creative_coding_assistant.orchestration.agent_roles import (
    AgentRoleRegistry,
    agent_role_by_id,
    agent_role_registry,
)
from creative_coding_assistant.orchestration.dynamic_agent_allocation import (
    DynamicAgentAllocationCandidate,
    DynamicAgentAllocationLane,
    DynamicAgentAllocationPlan,
    DynamicAgentAllocationStatus,
    allocate_dynamic_agents,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

AgentDiversityStatus = Literal["recommended", "standby", "guardrail"]

AGENT_DIVERSITY_CANDIDATE_SERIALIZATION_VERSION = (
    "agent_diversity_optimization_candidate.v1"
)
AGENT_DIVERSITY_PLAN_SERIALIZATION_VERSION = "agent_diversity_optimization_plan.v1"
AGENT_DIVERSITY_OPTIMIZER_AUTHORITY_BOUNDARY = (
    "V5.5 agent diversity optimization combines advisory dynamic agent "
    "allocation, passive agent role, and passive capability alignment metadata "
    "into inspectable agent diversity recommendations only; it does not apply "
    "agent diversity behavior, select or rebalance runtime agent pools, "
    "activate capabilities, route runtime work, activate, instantiate, or "
    "invoke agents, run schedulers, execute parallel tasks, change workflow "
    "routing, control or execute workflows, route providers or models, select "
    "runtimes, emit HITL requests, enforce budgets, trigger retries, mutate "
    "memory, write storage, modify generated output, or apply Runtime "
    "Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "agent_diversity_behavior_application",
    "runtime_agent_selection",
    "agent_pool_rebalancing",
    "capability_activation",
    "runtime_work_routing",
    "dynamic_task_routing",
    "agent_activation",
    "agent_instantiation",
    "agent_invocation",
    "scheduler_runtime_hook",
    "parallel_task_execution",
    "workflow_routing_change",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "provider_or_model_routing",
    "runtime_selection",
    "human_review_request",
    "hitl_request_emission",
    "budget_enforcement",
    "retry_or_refinement_triggering",
    "memory_storage_or_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class AgentDiversityOptimizationCandidate(BaseModel):
    """One advisory agent diversity optimization candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    agent_id: str = Field(min_length=1, max_length=80)
    role_id: str = Field(min_length=1, max_length=80)
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_dynamic_allocation_id: str = Field(min_length=1, max_length=180)
    source_allocation_lane: DynamicAgentAllocationLane
    source_allocation_status: DynamicAgentAllocationStatus
    source_allocation_score: int = Field(ge=0, le=320)
    source_capability_alignment_agent_id: str = Field(min_length=1, max_length=80)
    source_role_registry_role_id: str = Field(min_length=1, max_length=80)
    role_family: AgentRoleFamily
    capability_family: AgentCapabilityClass
    priority_band: str = Field(min_length=1, max_length=80)
    scheduling_hint: str = Field(min_length=1, max_length=80)
    parallelizable: bool
    source_capability_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    aligned_capability_ids: tuple[str, ...] = Field(min_length=1, max_length=13)
    required_metadata_input_count: int = Field(ge=1, le=40)
    produced_metadata_output_count: int = Field(ge=1, le=40)
    upstream_dependency_count: int = Field(ge=0, le=32)
    downstream_dependency_count: int = Field(ge=0, le=32)
    role_family_weight: int = Field(ge=0, le=90)
    capability_family_weight: int = Field(ge=0, le=90)
    alignment_breadth_score: int = Field(ge=0, le=100)
    allocation_signal_score: int = Field(ge=0, le=120)
    parallelism_bonus: int = Field(ge=0, le=30)
    hitl_penalty: int = Field(ge=0, le=80)
    agent_diversity_score: int = Field(ge=0, le=360)
    status: AgentDiversityStatus
    recommended_diversity_path_count: int = Field(ge=0, le=4)
    applied_diversity_path_count: int = Field(ge=0, le=0)
    hitl_required: bool
    diversity_summary: str = Field(min_length=1, max_length=360)
    fallback_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    agent_diversity_optimizer_implemented: Literal[True] = True
    agent_diversity_metadata_implemented: Literal[True] = True
    dynamic_agent_allocation_metadata_used: Literal[True] = True
    agent_role_metadata_used: Literal[True] = True
    capability_alignment_metadata_used: Literal[True] = True
    scheduling_metadata_used: Literal[True] = True
    dependency_metadata_used: Literal[True] = True
    hitl_posture_metadata_used: Literal[True] = True
    agent_diversity_behavior_application_implemented: Literal[False] = False
    runtime_agent_selection_implemented: Literal[False] = False
    agent_pool_rebalancing_implemented: Literal[False] = False
    capability_activation_implemented: Literal[False] = False
    runtime_work_routing_implemented: Literal[False] = False
    dynamic_task_routing_implemented: Literal[False] = False
    agent_activation_implemented: Literal[False] = False
    agent_instantiation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    scheduler_runtime_hook_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    workflow_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["agent_diversity_optimization_candidate.v1"] = (
        AGENT_DIVERSITY_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_contract(self) -> Self:
        if self.candidate_id != f"agent_diversity_optimizer::{self.agent_id}":
            raise ValueError("candidate_id must match agent_id")
        if self.source_dynamic_allocation_id != (
            f"dynamic_agent_allocation::{self.agent_id}"
        ):
            raise ValueError("source_dynamic_allocation_id must match agent_id")
        if self.source_capability_alignment_agent_id != self.agent_id:
            raise ValueError("source_capability_alignment_agent_id must match agent_id")
        if self.source_role_registry_role_id != self.role_id:
            raise ValueError("source_role_registry_role_id must match role_id")
        if self.role_family_weight != _family_weight(self.role_family):
            raise ValueError("role_family_weight must match role family")
        if self.capability_family_weight != _family_weight(self.capability_family):
            raise ValueError("capability_family_weight must match capability family")
        if self.alignment_breadth_score != _alignment_breadth_score(
            self.aligned_capability_ids
        ):
            raise ValueError("alignment_breadth_score must match aligned capabilities")
        if self.allocation_signal_score != _allocation_signal_score(
            self.source_allocation_score
        ):
            raise ValueError("allocation_signal_score must match allocation source")
        if self.parallelism_bonus != _parallelism_bonus(self.parallelizable):
            raise ValueError("parallelism_bonus must match parallelizability")
        if self.hitl_penalty != _hitl_penalty(self.hitl_required):
            raise ValueError("hitl_penalty must match HITL posture")
        if self.agent_diversity_score != _agent_diversity_score(
            role_family_weight=self.role_family_weight,
            capability_family_weight=self.capability_family_weight,
            alignment_breadth_score=self.alignment_breadth_score,
            allocation_signal_score=self.allocation_signal_score,
            parallelism_bonus=self.parallelism_bonus,
            hitl_penalty=self.hitl_penalty,
        ):
            raise ValueError("agent_diversity_score must combine source scores")
        if self.status == "guardrail" and self.recommended_diversity_path_count:
            raise ValueError("guardrail diversity candidates must recommend no paths")
        if self.applied_diversity_path_count:
            raise ValueError("applied_diversity_path_count must remain zero")
        if self.source_allocation_lane == "strategy_primary" and (
            self.status != "recommended"
        ):
            raise ValueError("primary allocation diversity must remain recommended")
        if self.source_allocation_status == "requires_hitl" and not self.hitl_required:
            raise ValueError("requires_hitl allocation must carry HITL posture")
        return self


class AgentDiversityOptimizationPlan(BaseModel):
    """Bounded V5.5 advisory agent diversity optimization plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_diversity_optimizer"] = "agent_diversity_optimizer"
    serialization_version: Literal["agent_diversity_optimization_plan.v1"] = (
        AGENT_DIVERSITY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_DIVERSITY_OPTIMIZER_AUTHORITY_BOUNDARY,
        max_length=2100,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_dynamic_agent_allocation_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_agent_capability_alignment_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_agent_role_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    role_family_count: int = Field(ge=1, le=32)
    capability_family_count: int = Field(ge=1, le=32)
    aligned_capability_count: int = Field(ge=1, le=32)
    candidates: tuple[AgentDiversityOptimizationCandidate, ...] = Field(
        min_length=1,
        max_length=12,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    recommended_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=12
    )
    standby_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=12
    )
    hitl_required_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    applied_diversity_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    candidate_count: int = Field(ge=1, le=12)
    recommended_candidate_count: int = Field(ge=0, le=12)
    guardrail_candidate_count: int = Field(ge=0, le=12)
    hitl_required_candidate_count: int = Field(ge=0, le=12)
    total_recommended_diversity_path_count: int = Field(ge=0, le=48)
    total_applied_diversity_path_count: int = Field(ge=0, le=0)
    highest_agent_diversity_score: int = Field(ge=0, le=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    agent_diversity_optimizer_implemented: Literal[True] = True
    agent_diversity_metadata_implemented: Literal[True] = True
    dynamic_agent_allocation_metadata_used: Literal[True] = True
    agent_role_metadata_used: Literal[True] = True
    capability_alignment_metadata_used: Literal[True] = True
    scheduling_metadata_used: Literal[True] = True
    dependency_metadata_used: Literal[True] = True
    hitl_posture_metadata_used: Literal[True] = True
    agent_diversity_behavior_application_implemented: Literal[False] = False
    runtime_agent_selection_implemented: Literal[False] = False
    agent_pool_rebalancing_implemented: Literal[False] = False
    capability_activation_implemented: Literal[False] = False
    runtime_work_routing_implemented: Literal[False] = False
    dynamic_task_routing_implemented: Literal[False] = False
    agent_activation_implemented: Literal[False] = False
    agent_instantiation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    scheduler_runtime_hook_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    workflow_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_candidates(self) -> Self:
        derived_candidate_ids = tuple(
            candidate.candidate_id for candidate in self.candidates
        )
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
        if self.guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "guardrail",
        ):
            raise ValueError("guardrail_candidate_ids must match candidates")
        if self.hitl_required_candidate_ids != tuple(
            candidate.candidate_id
            for candidate in self.candidates
            if candidate.hitl_required
        ):
            raise ValueError("hitl_required_candidate_ids must match candidates")
        if self.applied_diversity_candidate_ids:
            raise ValueError("applied_diversity_candidate_ids must remain empty")
        if self.recommended_candidate_count != len(self.recommended_candidate_ids):
            raise ValueError("recommended_candidate_count must match candidates")
        if self.guardrail_candidate_count != len(self.guardrail_candidate_ids):
            raise ValueError("guardrail_candidate_count must match candidates")
        if self.hitl_required_candidate_count != len(self.hitl_required_candidate_ids):
            raise ValueError("hitl_required_candidate_count must match candidates")
        if self.total_recommended_diversity_path_count != sum(
            candidate.recommended_diversity_path_count for candidate in self.candidates
        ):
            raise ValueError(
                "total_recommended_diversity_path_count must match candidates"
            )
        if self.total_applied_diversity_path_count != 0:
            raise ValueError("total_applied_diversity_path_count must remain zero")
        if self.highest_agent_diversity_score != max(
            candidate.agent_diversity_score for candidate in self.candidates
        ):
            raise ValueError("highest_agent_diversity_score must match candidates")
        for candidate in self.candidates:
            if candidate.route_name != self.route_name:
                raise ValueError("candidate route_name must match plan")
        return self


def optimize_agent_diversity(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    dynamic_agent_allocation: DynamicAgentAllocationPlan | None = None,
    capability_alignment: AgentCapabilityAlignmentRegistry | None = None,
    agent_roles: AgentRoleRegistry | None = None,
) -> AgentDiversityOptimizationPlan:
    """Optimize agent diversity metadata without selecting runtime agents."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    allocation_plan = dynamic_agent_allocation or allocate_dynamic_agents(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    normalized_mode = str(
        execution_mode_id or allocation_plan.allocations[0].execution_mode_id
    )
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")

    alignment_registry = capability_alignment or agent_capability_alignment_registry()
    role_registry = agent_roles or agent_role_registry()
    candidates = tuple(
        _candidate(
            allocation=allocation,
            route_name=route_name,
            task_type=allocation_plan.task_type,
            execution_mode_id=normalized_mode,  # type: ignore[arg-type]
            capability_alignment=alignment_registry,
        )
        for allocation in allocation_plan.allocations
    )
    return AgentDiversityOptimizationPlan(
        route_name=route_name,
        task_type=allocation_plan.task_type,
        source_dynamic_agent_allocation_serialization_version=(
            allocation_plan.serialization_version
        ),
        source_agent_capability_alignment_serialization_version=(
            alignment_registry.serialization_version
        ),
        source_agent_role_serialization_version=role_registry.serialization_version,
        execution_mode_ids=execution_modes.execution_mode_ids,
        role_family_count=len(role_registry.role_families),
        capability_family_count=len(role_registry.capability_families),
        aligned_capability_count=len(alignment_registry.capability_ids),
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        recommended_candidate_ids=_candidate_ids_for_status(
            candidates,
            "recommended",
        ),
        standby_candidate_ids=_candidate_ids_for_status(candidates, "standby"),
        guardrail_candidate_ids=_candidate_ids_for_status(candidates, "guardrail"),
        hitl_required_candidate_ids=tuple(
            candidate.candidate_id
            for candidate in candidates
            if candidate.hitl_required
        ),
        applied_diversity_candidate_ids=(),
        candidate_count=len(candidates),
        recommended_candidate_count=len(
            _candidate_ids_for_status(candidates, "recommended")
        ),
        guardrail_candidate_count=len(
            _candidate_ids_for_status(candidates, "guardrail")
        ),
        hitl_required_candidate_count=sum(
            1 for candidate in candidates if candidate.hitl_required
        ),
        total_recommended_diversity_path_count=sum(
            candidate.recommended_diversity_path_count for candidate in candidates
        ),
        total_applied_diversity_path_count=0,
        highest_agent_diversity_score=max(
            candidate.agent_diversity_score for candidate in candidates
        ),
        advisory_actions=_plan_actions(candidates),
    )


def agent_diversity_candidate_by_agent_id(
    agent_id: str,
    plan: AgentDiversityOptimizationPlan | None = None,
) -> AgentDiversityOptimizationCandidate | None:
    """Return one agent diversity candidate without applying selection."""

    source_plan = plan or optimize_agent_diversity()
    for candidate in source_plan.candidates:
        if candidate.agent_id == agent_id:
            return candidate
    return None


def agent_diversity_candidates_for_status(
    status: AgentDiversityStatus,
    plan: AgentDiversityOptimizationPlan | None = None,
) -> tuple[AgentDiversityOptimizationCandidate, ...]:
    """Return agent diversity candidates for one advisory status."""

    source_plan = plan or optimize_agent_diversity()
    return tuple(
        candidate for candidate in source_plan.candidates if candidate.status == status
    )


def _candidate(
    *,
    allocation: DynamicAgentAllocationCandidate,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    capability_alignment: AgentCapabilityAlignmentRegistry,
) -> AgentDiversityOptimizationCandidate:
    alignment = _required_alignment(allocation.agent_id, capability_alignment)
    role = agent_role_by_id(allocation.role_id)
    if role is None:
        raise ValueError("required agent role metadata is missing")
    status = _status_for_allocation(allocation)
    role_weight = _family_weight(role.role_family)
    capability_weight = _family_weight(role.capability_family)
    breadth_score = _alignment_breadth_score(alignment.capability_ids)
    allocation_score = _allocation_signal_score(allocation.allocation_score)
    parallel_bonus = _parallelism_bonus(allocation.parallelizable)
    hitl_penalty = _hitl_penalty(allocation.hitl_required)
    diversity_score = _agent_diversity_score(
        role_family_weight=role_weight,
        capability_family_weight=capability_weight,
        alignment_breadth_score=breadth_score,
        allocation_signal_score=allocation_score,
        parallelism_bonus=parallel_bonus,
        hitl_penalty=hitl_penalty,
    )
    return AgentDiversityOptimizationCandidate(
        candidate_id=f"agent_diversity_optimizer::{allocation.agent_id}",
        agent_id=allocation.agent_id,
        role_id=allocation.role_id,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_dynamic_allocation_id=allocation.allocation_id,
        source_allocation_lane=allocation.allocation_lane,
        source_allocation_status=allocation.allocation_status,
        source_allocation_score=allocation.allocation_score,
        source_capability_alignment_agent_id=alignment.agent_id,
        source_role_registry_role_id=role.role_id,
        role_family=role.role_family,
        capability_family=role.capability_family,
        priority_band=allocation.priority_band,
        scheduling_hint=allocation.scheduling_hint,
        parallelizable=allocation.parallelizable,
        source_capability_ids=allocation.source_capability_ids,
        aligned_capability_ids=alignment.capability_ids,
        required_metadata_input_count=allocation.required_metadata_input_count,
        produced_metadata_output_count=allocation.produced_metadata_output_count,
        upstream_dependency_count=allocation.upstream_dependency_count,
        downstream_dependency_count=allocation.downstream_dependency_count,
        role_family_weight=role_weight,
        capability_family_weight=capability_weight,
        alignment_breadth_score=breadth_score,
        allocation_signal_score=allocation_score,
        parallelism_bonus=parallel_bonus,
        hitl_penalty=hitl_penalty,
        agent_diversity_score=diversity_score,
        status=status,
        recommended_diversity_path_count=0 if status == "guardrail" else 1,
        applied_diversity_path_count=0,
        hitl_required=allocation.hitl_required,
        diversity_summary=_diversity_summary(status, role.role_family),
        fallback_summary=_fallback_summary(status),
        advisory_actions=_candidate_actions(status),
        evidence=(
            f"dynamic_allocation:{allocation.allocation_id}",
            f"capability_alignment:{alignment.agent_id}",
            f"role_family:{role.role_family}",
            f"capability_family:{role.capability_family}",
            f"aligned_capabilities:{len(alignment.capability_ids)}",
            f"allocation_lane:{allocation.allocation_lane}",
        ),
    )


def _required_alignment(
    agent_id: str,
    registry: AgentCapabilityAlignmentRegistry,
) -> AgentCapabilityAlignmentProfile:
    alignment = agent_capability_alignment_by_agent_id(agent_id, registry)
    if alignment is None:
        raise ValueError("required agent capability alignment metadata is missing")
    return alignment


def _status_for_allocation(
    allocation: DynamicAgentAllocationCandidate,
) -> AgentDiversityStatus:
    if allocation.allocation_lane == "strategy_primary":
        return "recommended"
    if not allocation.parallelizable:
        return "guardrail"
    return "standby"


def _family_weight(_: str) -> int:
    return 70


def _alignment_breadth_score(capability_ids: tuple[str, ...]) -> int:
    return min(100, len(capability_ids) * 7)


def _allocation_signal_score(allocation_score: int) -> int:
    return min(120, allocation_score // 3)


def _parallelism_bonus(parallelizable: bool) -> int:
    return 20 if parallelizable else 6


def _hitl_penalty(hitl_required: bool) -> int:
    return 28 if hitl_required else 0


def _agent_diversity_score(
    *,
    role_family_weight: int,
    capability_family_weight: int,
    alignment_breadth_score: int,
    allocation_signal_score: int,
    parallelism_bonus: int,
    hitl_penalty: int,
) -> int:
    return min(
        360,
        max(
            0,
            role_family_weight
            + capability_family_weight
            + alignment_breadth_score
            + allocation_signal_score
            + parallelism_bonus
            - hitl_penalty,
        ),
    )


def _candidate_ids_for_status(
    candidates: tuple[AgentDiversityOptimizationCandidate, ...],
    status: AgentDiversityStatus,
) -> tuple[str, ...]:
    return tuple(
        candidate.candidate_id for candidate in candidates if candidate.status == status
    )


def _diversity_summary(status: AgentDiversityStatus, role_family: str) -> str:
    if status == "recommended":
        return f"Keep {role_family} agent diversity visible in the primary set."
    if status == "standby":
        return f"Keep {role_family} agent diversity available as standby metadata."
    return f"Keep {role_family} agent diversity guarded until scheduling changes exist."


def _fallback_summary(status: AgentDiversityStatus) -> str:
    if status == "recommended":
        return "Fallback to standby diversity metadata if HITL or routing risk rises."
    if status == "standby":
        return (
            "Fallback to guardrail posture before any runtime agent selection exists."
        )
    return "Preserve guardrail posture without applying agent diversity behavior."


def _candidate_actions(status: AgentDiversityStatus) -> tuple[str, ...]:
    return (
        f"Surface {status} agent diversity posture as advisory metadata.",
        "Keep agent selection, rebalancing, capability activation, routing, scheduling, workflow, provider, memory, storage, and output behavior disabled.",  # noqa: E501
    )


def _plan_actions(
    candidates: tuple[AgentDiversityOptimizationCandidate, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose agent diversity optimization as advisory metadata only.",
        "Keep applied diversity candidate ids empty and applied path count at zero.",
        "Preserve agent selection, rebalancing, capability, routing, scheduling, workflow, provider, memory, storage, and output boundaries.",  # noqa: E501
    ]
    if _candidate_ids_for_status(candidates, "guardrail"):
        actions.append("Require review before any future agent diversity behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
