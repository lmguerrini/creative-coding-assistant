"""V5.3 advisory async execution planning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .latency_optimizer import (
    LatencyOptimizationCandidate,
    LatencyOptimizationPlan,
    optimize_latency,
)
from .parallel_scheduler import (
    ParallelScheduleCandidate,
    ParallelSchedulerPlan,
    plan_parallel_scheduler,
)

AsyncExecutionStatus = Literal["async_ready_candidate", "serial_guardrail"]
AsyncExecutionMode = Literal["bounded_parallel_group", "ordered_serial_stage"]
AsyncExecutionPressure = Literal["low", "medium", "high"]

ASYNC_EXECUTION_CANDIDATE_SERIALIZATION_VERSION = (
    "async_execution_candidate.v1"
)
ASYNC_EXECUTION_PLAN_SERIALIZATION_VERSION = "async_execution_plan.v1"
ASYNC_EXECUTION_AUTHORITY_BOUNDARY = (
    "Async execution planning derives advisory async-ready candidates from "
    "parallel scheduler and latency optimization metadata only; it does not "
    "create event-loop tasks, run tasks in parallel, alter workflow timing, "
    "mutate graph order, compile or execute workflow graphs, invoke agents or "
    "node handlers, route providers or models, enforce cancellation or timeout "
    "policies, trigger retries, mutate prompts, write storage, or modify "
    "generated output."
)

_ASYNC_GUARDS = (
    "requires_explicit_runtime_authority",
    "requires_upstream_dependency_completion",
    "requires_failure_normalization_boundary",
    "requires_cancellation_policy_before_activation",
    "requires_timeout_policy_before_activation",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "event_loop_task_creation",
    "parallel_task_execution",
    "async_runtime_execution",
    "workflow_timing_change",
    "workflow_graph_mutation",
    "langgraph_compilation",
    "workflow_execution",
    "agent_invocation",
    "node_handler_invocation",
    "provider_or_model_routing",
    "cancellation_policy_enforcement",
    "timeout_policy_enforcement",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class AsyncExecutionCandidate(BaseModel):
    """One advisory V5.3 async execution candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    source_schedule_candidate_id: str = Field(min_length=1, max_length=180)
    source_latency_candidate_id: str = Field(min_length=1, max_length=180)
    stage_id: str = Field(min_length=1, max_length=80)
    agent_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    status: AsyncExecutionStatus
    async_execution_mode: AsyncExecutionMode
    advisory_rank: int = Field(ge=1, le=20)
    dependency_depth: int = Field(ge=0, le=20)
    max_parallel_agents: int = Field(ge=1, le=6)
    blocking_input_count: int = Field(ge=0, le=96)
    advisory_async_slot_count: int = Field(ge=1, le=6)
    advisory_async_readiness_score: int = Field(ge=0, le=600)
    required_guards: tuple[str, ...] = Field(min_length=5, max_length=5)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    async_execution_planning_implemented: Literal[True] = True
    event_loop_task_creation_implemented: Literal[False] = False
    async_runtime_execution_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    cancellation_policy_enforcement_implemented: Literal[False] = False
    timeout_policy_enforcement_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["async_execution_candidate.v1"] = (
        ASYNC_EXECUTION_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_sources(self) -> Self:
        if self.candidate_id != f"async_execution::{self.stage_id}":
            raise ValueError("candidate_id must match stage_id")
        if self.max_parallel_agents != len(self.agent_ids):
            raise ValueError("max_parallel_agents must match agent count")
        if self.advisory_async_slot_count != self.max_parallel_agents:
            raise ValueError(
                "advisory_async_slot_count must match max_parallel_agents"
            )
        expected_score = _async_readiness_score(self.max_parallel_agents)
        if self.advisory_async_readiness_score != expected_score:
            raise ValueError(
                "advisory_async_readiness_score must match max_parallel_agents"
            )
        if self.required_guards != _ASYNC_GUARDS:
            raise ValueError("required_guards must match async guard contract")
        if self.status == "async_ready_candidate" and self.max_parallel_agents <= 1:
            raise ValueError("async-ready candidates require multiple agents")
        if self.status == "serial_guardrail" and self.max_parallel_agents != 1:
            raise ValueError("serial guardrails require one agent")
        if self.async_execution_mode != _async_mode(self.status):
            raise ValueError("async_execution_mode must match status")
        return self


class AsyncExecutionPlan(BaseModel):
    """Bounded V5.3 advisory async execution plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["async_execution_planner"] = "async_execution_planner"
    serialization_version: Literal["async_execution_plan.v1"] = (
        ASYNC_EXECUTION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ASYNC_EXECUTION_AUTHORITY_BOUNDARY,
        max_length=1400,
    )
    source_parallel_scheduler_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_latency_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    failure_normalization_required: Literal[True] = True
    candidates: tuple[AsyncExecutionCandidate, ...] = Field(
        min_length=1,
        max_length=20,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    async_ready_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=20,
    )
    serial_guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=20,
    )
    candidate_count: int = Field(ge=1, le=20)
    async_ready_candidate_count: int = Field(ge=0, le=20)
    serial_guardrail_count: int = Field(ge=0, le=20)
    max_async_width: int = Field(ge=1, le=6)
    total_advisory_async_slots: int = Field(ge=1, le=80)
    highest_advisory_async_readiness_score: int = Field(ge=0, le=600)
    total_advisory_async_readiness_score: int = Field(ge=0, le=6000)
    async_execution_pressure: AsyncExecutionPressure
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    async_execution_planning_implemented: Literal[True] = True
    event_loop_task_creation_implemented: Literal[False] = False
    async_runtime_execution_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    cancellation_policy_enforcement_implemented: Literal[False] = False
    timeout_policy_enforcement_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
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
        if self.async_ready_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "async_ready_candidate",
        ):
            raise ValueError("async_ready_candidate_ids must match candidates")
        if self.serial_guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "serial_guardrail",
        ):
            raise ValueError("serial_guardrail_candidate_ids must match candidates")
        if self.async_ready_candidate_count != len(self.async_ready_candidate_ids):
            raise ValueError("async_ready_candidate_count must match candidates")
        if self.serial_guardrail_count != len(self.serial_guardrail_candidate_ids):
            raise ValueError("serial_guardrail_count must match candidates")

        expected_width = max(
            candidate.max_parallel_agents for candidate in self.candidates
        )
        if self.max_async_width != expected_width:
            raise ValueError("max_async_width must match candidates")
        expected_slots = sum(
            candidate.advisory_async_slot_count for candidate in self.candidates
        )
        if self.total_advisory_async_slots != expected_slots:
            raise ValueError("total_advisory_async_slots must match candidates")
        expected_highest_score = max(
            candidate.advisory_async_readiness_score
            for candidate in self.candidates
        )
        if self.highest_advisory_async_readiness_score != expected_highest_score:
            raise ValueError(
                "highest_advisory_async_readiness_score must match candidates"
            )
        expected_total_score = sum(
            candidate.advisory_async_readiness_score
            for candidate in self.candidates
        )
        if self.total_advisory_async_readiness_score != expected_total_score:
            raise ValueError(
                "total_advisory_async_readiness_score must match candidates"
            )
        if self.async_execution_pressure != _async_pressure(
            max_width=self.max_async_width,
            total_score=self.total_advisory_async_readiness_score,
        ):
            raise ValueError("async_execution_pressure must match candidates")
        return self


def plan_async_execution(
    *,
    parallel_scheduler: ParallelSchedulerPlan | None = None,
    latency_optimization: LatencyOptimizationPlan | None = None,
) -> AsyncExecutionPlan:
    """Plan advisory async execution without creating async runtime work."""

    scheduler = parallel_scheduler or plan_parallel_scheduler()
    latency = latency_optimization or optimize_latency(parallel_scheduler=scheduler)
    latency_by_stage = {
        candidate.stage_id: candidate for candidate in latency.candidates
    }
    candidates = tuple(
        _candidate(
            schedule_candidate=schedule_candidate,
            latency_candidate=latency_by_stage[schedule_candidate.stage_id],
        )
        for schedule_candidate in scheduler.candidates
    )
    max_width = max(candidate.max_parallel_agents for candidate in candidates)
    total_score = sum(
        candidate.advisory_async_readiness_score for candidate in candidates
    )

    return AsyncExecutionPlan(
        source_parallel_scheduler_serialization_version=scheduler.serialization_version,
        source_latency_optimization_serialization_version=latency.serialization_version,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        async_ready_candidate_ids=_candidate_ids_for_status(
            candidates,
            "async_ready_candidate",
        ),
        serial_guardrail_candidate_ids=_candidate_ids_for_status(
            candidates,
            "serial_guardrail",
        ),
        candidate_count=len(candidates),
        async_ready_candidate_count=len(
            _candidate_ids_for_status(candidates, "async_ready_candidate")
        ),
        serial_guardrail_count=len(
            _candidate_ids_for_status(candidates, "serial_guardrail")
        ),
        max_async_width=max_width,
        total_advisory_async_slots=sum(
            candidate.advisory_async_slot_count for candidate in candidates
        ),
        highest_advisory_async_readiness_score=max(
            candidate.advisory_async_readiness_score for candidate in candidates
        ),
        total_advisory_async_readiness_score=total_score,
        async_execution_pressure=_async_pressure(
            max_width=max_width,
            total_score=total_score,
        ),
        advisory_actions=_plan_actions(max_width=max_width, total_score=total_score),
    )


def async_execution_candidate_by_id(
    candidate_id: str,
    plan: AsyncExecutionPlan | None = None,
) -> AsyncExecutionCandidate | None:
    """Return one advisory async candidate without executing it."""

    source_plan = plan or plan_async_execution()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def async_execution_candidates_for_status(
    status: AsyncExecutionStatus,
    plan: AsyncExecutionPlan | None = None,
) -> tuple[AsyncExecutionCandidate, ...]:
    """Return advisory async candidates by status without workflow control."""

    source_plan = plan or plan_async_execution()
    return tuple(
        candidate for candidate in source_plan.candidates if candidate.status == status
    )


def _candidate(
    *,
    schedule_candidate: ParallelScheduleCandidate,
    latency_candidate: LatencyOptimizationCandidate,
) -> AsyncExecutionCandidate:
    status: AsyncExecutionStatus = (
        "async_ready_candidate"
        if schedule_candidate.max_parallel_agents > 1
        else "serial_guardrail"
    )
    return AsyncExecutionCandidate(
        candidate_id=f"async_execution::{schedule_candidate.stage_id}",
        source_schedule_candidate_id=schedule_candidate.candidate_id,
        source_latency_candidate_id=latency_candidate.candidate_id,
        stage_id=schedule_candidate.stage_id,
        agent_ids=schedule_candidate.agent_ids,
        status=status,
        async_execution_mode=_async_mode(status),
        advisory_rank=schedule_candidate.advisory_rank,
        dependency_depth=schedule_candidate.dependency_depth,
        max_parallel_agents=schedule_candidate.max_parallel_agents,
        blocking_input_count=latency_candidate.blocking_input_count,
        advisory_async_slot_count=schedule_candidate.max_parallel_agents,
        advisory_async_readiness_score=_async_readiness_score(
            schedule_candidate.max_parallel_agents
        ),
        required_guards=_ASYNC_GUARDS,
        evidence=_candidate_evidence(
            schedule_candidate=schedule_candidate,
            latency_candidate=latency_candidate,
        ),
        advisory_actions=_candidate_actions(status),
    )


def _candidate_ids_for_status(
    candidates: tuple[AsyncExecutionCandidate, ...],
    status: AsyncExecutionStatus,
) -> tuple[str, ...]:
    return tuple(
        candidate.candidate_id
        for candidate in candidates
        if candidate.status == status
    )


def _async_mode(status: AsyncExecutionStatus) -> AsyncExecutionMode:
    if status == "async_ready_candidate":
        return "bounded_parallel_group"
    return "ordered_serial_stage"


def _async_readiness_score(max_parallel_agents: int) -> int:
    return max(0, max_parallel_agents - 1) * 100


def _async_pressure(
    *,
    max_width: int,
    total_score: int,
) -> AsyncExecutionPressure:
    if max_width >= 3 or total_score >= 500:
        return "high"
    if max_width == 2 or total_score >= 200:
        return "medium"
    return "low"


def _candidate_evidence(
    *,
    schedule_candidate: ParallelScheduleCandidate,
    latency_candidate: LatencyOptimizationCandidate,
) -> tuple[str, ...]:
    return (
        f"schedule_candidate:{schedule_candidate.candidate_id}",
        f"latency_candidate:{latency_candidate.candidate_id}",
        f"stage:{schedule_candidate.stage_id}",
        f"agents:{len(schedule_candidate.agent_ids)}",
        f"blocking_inputs:{latency_candidate.blocking_input_count}",
    )


def _candidate_actions(status: AsyncExecutionStatus) -> tuple[str, ...]:
    if status == "async_ready_candidate":
        return (
            "Expose async-ready candidate as advisory metadata only.",
            "Require explicit runtime authority before async activation.",
        )
    return (
        "Retain serial stage as async guardrail metadata only.",
        "Preserve ordered execution until runtime authority changes.",
    )


def _plan_actions(
    *,
    max_width: int,
    total_score: int,
) -> tuple[str, ...]:
    actions = [
        "Expose async execution readiness as advisory metadata only.",
        "Preserve event-loop, timing, graph, execution, routing, and output "
        "boundaries.",
    ]
    if max_width > 1:
        actions.append("Report async width without creating event-loop tasks.")
    if total_score:
        actions.append("Use async readiness score only for later performance review.")
    return tuple(actions)
