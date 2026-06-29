"""V5.3 advisory parallel scheduler planning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_parallel_scheduling import (
    ParallelSchedulingGroup,
    ParallelSchedulingRegistry,
    parallel_scheduling_registry,
)
from creative_coding_assistant.orchestration.execution_graph_analyzer import (
    ExecutionGraphAnalysis,
    analyze_assistant_execution_graph,
)

ParallelScheduleStatus = Literal["parallel_candidate", "serial_guardrail"]
ParallelSchedulerPressure = Literal["low", "medium", "high"]

PARALLEL_SCHEDULE_CANDIDATE_SERIALIZATION_VERSION = (
    "parallel_schedule_candidate.v1"
)
PARALLEL_SCHEDULER_PLAN_SERIALIZATION_VERSION = "parallel_scheduler_plan.v1"
PARALLEL_SCHEDULER_AUTHORITY_BOUNDARY = (
    "Parallel scheduler planning derives advisory concurrency candidates from "
    "passive agent scheduling metadata and static workflow topology only; it "
    "does not run tasks in parallel, create async tasks, alter workflow timing, "
    "mutate graph order, compile or execute the workflow graph, invoke node "
    "handlers, route providers or models, trigger retries, mutate prompts, "
    "write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "parallel_task_execution",
    "async_task_creation",
    "workflow_timing_change",
    "workflow_graph_mutation",
    "langgraph_compilation",
    "workflow_execution",
    "node_handler_invocation",
    "provider_or_model_routing",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class ParallelScheduleCandidate(BaseModel):
    """One advisory V5.3 parallel scheduling candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    source_group_id: str = Field(min_length=1, max_length=120)
    stage_id: str = Field(min_length=1, max_length=80)
    agent_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    status: ParallelScheduleStatus
    advisory_rank: int = Field(ge=1, le=20)
    dependency_depth: int = Field(ge=0, le=20)
    blocking_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    downstream_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    max_parallel_agents: int = Field(ge=1, le=6)
    advisory_parallel_slot_count: int = Field(ge=1, le=6)
    advisory_parallelism_score: int = Field(ge=0, le=600)
    source_scheduling_hint: str = Field(min_length=1, max_length=80)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    parallel_scheduler_planning_implemented: Literal[True] = True
    parallel_execution_implemented: Literal[False] = False
    async_execution_implemented: Literal[False] = False
    workflow_timing_mutation_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["parallel_schedule_candidate.v1"] = (
        PARALLEL_SCHEDULE_CANDIDATE_SERIALIZATION_VERSION
    )
    planning_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_status(self) -> Self:
        if self.candidate_id != f"parallel_scheduler::{self.stage_id}":
            raise ValueError("candidate_id must match stage_id")
        if self.max_parallel_agents != len(self.agent_ids):
            raise ValueError("max_parallel_agents must match agent count")
        if self.advisory_parallel_slot_count != self.max_parallel_agents:
            raise ValueError(
                "advisory_parallel_slot_count must match max_parallel_agents"
            )
        expected_score = _parallelism_score(self.max_parallel_agents)
        if self.advisory_parallelism_score != expected_score:
            raise ValueError(
                "advisory_parallelism_score must match max_parallel_agents"
            )
        if self.status == "parallel_candidate" and self.max_parallel_agents <= 1:
            raise ValueError("parallel candidates require multiple agents")
        if self.status == "serial_guardrail" and self.max_parallel_agents != 1:
            raise ValueError("serial guardrails require one agent")
        return self


class ParallelSchedulerPlan(BaseModel):
    """Bounded V5.3 advisory parallel scheduler plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["parallel_scheduler"] = "parallel_scheduler"
    serialization_version: Literal["parallel_scheduler_plan.v1"] = (
        PARALLEL_SCHEDULER_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PARALLEL_SCHEDULER_AUTHORITY_BOUNDARY,
        max_length=1400,
    )
    source_parallel_scheduling_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    source_execution_graph_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    source_graph_node_count: int = Field(ge=1, le=40)
    source_graph_failure_entry_node_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=40,
    )
    candidates: tuple[ParallelScheduleCandidate, ...] = Field(
        min_length=1,
        max_length=20,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    parallel_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=20,
    )
    serial_guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=20,
    )
    candidate_count: int = Field(ge=1, le=20)
    parallel_candidate_count: int = Field(ge=0, le=20)
    serial_guardrail_count: int = Field(ge=0, le=20)
    max_concurrency_width: int = Field(ge=1, le=6)
    total_advisory_parallel_slots: int = Field(ge=1, le=80)
    total_advisory_parallelism_score: int = Field(ge=0, le=6_000)
    scheduler_pressure: ParallelSchedulerPressure
    failure_normalization_preserved: bool
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    parallel_scheduler_planning_implemented: Literal[True] = True
    parallel_execution_implemented: Literal[False] = False
    async_execution_implemented: Literal[False] = False
    workflow_timing_mutation_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    planning_only: Literal[True] = True

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
        if self.parallel_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "parallel_candidate",
        ):
            raise ValueError("parallel_candidate_ids must match candidates")
        if self.serial_guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "serial_guardrail",
        ):
            raise ValueError("serial_guardrail_candidate_ids must match candidates")
        if self.parallel_candidate_count != len(self.parallel_candidate_ids):
            raise ValueError("parallel_candidate_count must match candidates")
        if self.serial_guardrail_count != len(self.serial_guardrail_candidate_ids):
            raise ValueError("serial_guardrail_count must match candidates")

        expected_width = max(
            candidate.max_parallel_agents for candidate in self.candidates
        )
        if self.max_concurrency_width != expected_width:
            raise ValueError("max_concurrency_width must match candidates")
        expected_slots = sum(
            candidate.advisory_parallel_slot_count for candidate in self.candidates
        )
        if self.total_advisory_parallel_slots != expected_slots:
            raise ValueError("total_advisory_parallel_slots must match candidates")
        expected_score = sum(
            candidate.advisory_parallelism_score for candidate in self.candidates
        )
        if self.total_advisory_parallelism_score != expected_score:
            raise ValueError("total_advisory_parallelism_score must match candidates")
        if self.scheduler_pressure != _scheduler_pressure(
            max_width=self.max_concurrency_width,
            total_score=self.total_advisory_parallelism_score,
        ):
            raise ValueError("scheduler_pressure must match candidates")

        candidate_index = {
            candidate.candidate_id: index
            for index, candidate in enumerate(self.candidates)
        }
        for candidate in self.candidates:
            for blocking_id in candidate.blocking_candidate_ids:
                if blocking_id not in candidate_index:
                    raise ValueError("blocking_candidate_ids must be known candidates")
                if (
                    candidate_index[blocking_id]
                    >= candidate_index[candidate.candidate_id]
                ):
                    raise ValueError("blocking relationships must be acyclic")
            for downstream_id in candidate.downstream_candidate_ids:
                if downstream_id not in candidate_index:
                    raise ValueError(
                        "downstream_candidate_ids must be known candidates"
                    )
                if (
                    candidate_index[downstream_id]
                    <= candidate_index[candidate.candidate_id]
                ):
                    raise ValueError("downstream relationships must be acyclic")
        return self


def plan_parallel_scheduler(
    *,
    scheduling_registry: ParallelSchedulingRegistry | None = None,
    execution_graph: ExecutionGraphAnalysis | None = None,
) -> ParallelSchedulerPlan:
    """Plan advisory parallel scheduling without running work in parallel."""

    scheduling = scheduling_registry or parallel_scheduling_registry()
    graph = execution_graph or analyze_assistant_execution_graph()
    candidates = _candidates(scheduling.groups)
    max_width = max(candidate.max_parallel_agents for candidate in candidates)
    total_slots = sum(
        candidate.advisory_parallel_slot_count for candidate in candidates
    )
    total_score = sum(candidate.advisory_parallelism_score for candidate in candidates)

    return ParallelSchedulerPlan(
        source_parallel_scheduling_serialization_version=scheduling.serialization_version,
        source_execution_graph_serialization_version=graph.serialization_version,
        source_graph_node_count=graph.node_count,
        source_graph_failure_entry_node_ids=graph.failure_entry_node_ids,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        parallel_candidate_ids=_candidate_ids_for_status(
            candidates,
            "parallel_candidate",
        ),
        serial_guardrail_candidate_ids=_candidate_ids_for_status(
            candidates,
            "serial_guardrail",
        ),
        candidate_count=len(candidates),
        parallel_candidate_count=len(
            _candidate_ids_for_status(candidates, "parallel_candidate")
        ),
        serial_guardrail_count=len(
            _candidate_ids_for_status(candidates, "serial_guardrail")
        ),
        max_concurrency_width=max_width,
        total_advisory_parallel_slots=total_slots,
        total_advisory_parallelism_score=total_score,
        scheduler_pressure=_scheduler_pressure(
            max_width=max_width,
            total_score=total_score,
        ),
        failure_normalization_preserved=graph.failure_path_reachable,
        advisory_actions=_plan_actions(max_width=max_width, total_score=total_score),
    )


def parallel_schedule_candidate_by_id(
    candidate_id: str,
    plan: ParallelSchedulerPlan | None = None,
) -> ParallelScheduleCandidate | None:
    """Return one advisory schedule candidate without scheduling it."""

    source_plan = plan or plan_parallel_scheduler()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def parallel_schedule_candidates_for_status(
    status: ParallelScheduleStatus,
    plan: ParallelSchedulerPlan | None = None,
) -> tuple[ParallelScheduleCandidate, ...]:
    """Return advisory schedule candidates by status without workflow control."""

    source_plan = plan or plan_parallel_scheduler()
    return tuple(
        candidate for candidate in source_plan.candidates if candidate.status == status
    )


def _candidates(
    groups: tuple[ParallelSchedulingGroup, ...],
) -> tuple[ParallelScheduleCandidate, ...]:
    candidates: list[ParallelScheduleCandidate] = []
    group_to_candidate_id = {
        group.group_id: f"parallel_scheduler::{group.stage_id}"
        for group in groups
    }
    for index, group in enumerate(groups):
        candidates.append(
            ParallelScheduleCandidate(
                candidate_id=group_to_candidate_id[group.group_id],
                source_group_id=group.group_id,
                stage_id=group.stage_id,
                agent_ids=group.agent_ids,
                status=(
                    "parallel_candidate"
                    if group.max_parallel_agents > 1
                    else "serial_guardrail"
                ),
                advisory_rank=index + 1,
                dependency_depth=index,
                blocking_candidate_ids=tuple(
                    group_to_candidate_id[group_id]
                    for group_id in group.blocking_group_ids
                ),
                downstream_candidate_ids=tuple(
                    group_to_candidate_id[group_id]
                    for group_id in group.downstream_group_ids
                ),
                max_parallel_agents=group.max_parallel_agents,
                advisory_parallel_slot_count=group.max_parallel_agents,
                advisory_parallelism_score=_parallelism_score(
                    group.max_parallel_agents
                ),
                source_scheduling_hint=group.scheduling_hint,
                evidence=_candidate_evidence(group, index),
                advisory_actions=_candidate_actions(group.max_parallel_agents),
            )
        )
    return tuple(candidates)


def _candidate_ids_for_status(
    candidates: tuple[ParallelScheduleCandidate, ...],
    status: ParallelScheduleStatus,
) -> tuple[str, ...]:
    return tuple(
        candidate.candidate_id
        for candidate in candidates
        if candidate.status == status
    )


def _parallelism_score(max_parallel_agents: int) -> int:
    return max(0, max_parallel_agents - 1) * 100


def _scheduler_pressure(
    *,
    max_width: int,
    total_score: int,
) -> ParallelSchedulerPressure:
    if max_width >= 3 or total_score >= 500:
        return "high"
    if max_width == 2 or total_score >= 200:
        return "medium"
    return "low"


def _candidate_evidence(
    group: ParallelSchedulingGroup,
    dependency_depth: int,
) -> tuple[str, ...]:
    return (
        f"source_group:{group.group_id}",
        f"stage:{group.stage_id}",
        f"agents:{len(group.agent_ids)}",
        f"dependency_depth:{dependency_depth}",
        f"scheduling_hint:{group.scheduling_hint}",
    )


def _candidate_actions(max_parallel_agents: int) -> tuple[str, ...]:
    if max_parallel_agents > 1:
        return (
            "Expose candidate as advisory parallel schedule metadata only.",
            "Require upstream dependency completion before future scheduler use.",
        )
    return (
        "Retain stage as serial scheduling guardrail metadata only.",
        "Preserve explicit downstream ordering for future scheduler review.",
    )


def _plan_actions(
    *,
    max_width: int,
    total_score: int,
) -> tuple[str, ...]:
    actions = [
        "Expose parallel scheduler candidates as advisory metadata only.",
        "Preserve graph order, async, routing, retry, storage, and output boundaries.",
    ]
    if max_width > 1:
        actions.append("Report maximum candidate width without running parallel work.")
    if total_score:
        actions.append(
            "Use advisory parallelism score only for later performance review."
        )
    return tuple(actions)
