"""V5.3 advisory latency optimization planning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_performance_tracking_foundation import (
    AgentPerformanceTrackingFoundationProfile,
    AgentPerformanceTrackingFoundationRegistry,
    agent_performance_tracking_foundation_registry,
)
from creative_coding_assistant.orchestration.hybrid_agentic_workflow import (
    LatencyThresholdRoutingRegistry,
    latency_threshold_routing_registry,
)
from creative_coding_assistant.orchestration.parallel_scheduler import (
    ParallelScheduleCandidate,
    ParallelSchedulerPlan,
    plan_parallel_scheduler,
)

LatencyOptimizationStatus = Literal["optimization_candidate", "serial_guardrail"]
LatencyOptimizationBand = Literal["low", "medium", "high", "guarded"]
LatencyOptimizationPressure = Literal["low", "medium", "high"]

LATENCY_OPTIMIZATION_CANDIDATE_SERIALIZATION_VERSION = (
    "latency_optimization_candidate.v1"
)
LATENCY_OPTIMIZATION_PLAN_SERIALIZATION_VERSION = "latency_optimization_plan.v1"
LATENCY_OPTIMIZER_AUTHORITY_BOUNDARY = (
    "Latency optimization derives advisory latency-saving candidates from "
    "passive performance tracking metadata, passive latency threshold metadata, "
    "and advisory parallel scheduler plans only; it does not measure latency, "
    "evaluate thresholds, route by latency, select runtimes, run tasks in "
    "parallel, create async tasks, alter workflow timing, mutate graph order, "
    "compile or execute workflow graphs, invoke agents or node handlers, route "
    "providers or models, trigger retries, mutate prompts, write storage, or "
    "modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "latency_measurement",
    "latency_threshold_evaluation",
    "latency_based_routing",
    "runtime_selection",
    "parallel_task_execution",
    "async_task_creation",
    "workflow_timing_change",
    "workflow_graph_mutation",
    "langgraph_compilation",
    "workflow_execution",
    "agent_invocation",
    "node_handler_invocation",
    "provider_or_model_routing",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class LatencyOptimizationCandidate(BaseModel):
    """One advisory V5.3 latency optimization candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    source_schedule_candidate_id: str = Field(min_length=1, max_length=180)
    source_scheduling_group_id: str = Field(min_length=1, max_length=120)
    stage_id: str = Field(min_length=1, max_length=80)
    agent_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    status: LatencyOptimizationStatus
    latency_band: LatencyOptimizationBand
    advisory_rank: int = Field(ge=1, le=20)
    dependency_depth: int = Field(ge=0, le=20)
    max_parallel_agents: int = Field(ge=1, le=6)
    source_latency_classes: tuple[str, ...] = Field(min_length=1, max_length=6)
    source_latency_threshold_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    source_latency_bands: tuple[str, ...] = Field(min_length=4, max_length=4)
    blocking_input_count: int = Field(ge=0, le=96)
    advisory_latency_savings_score: int = Field(ge=0, le=600)
    advisory_latency_pressure_score: int = Field(ge=0, le=1000)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    latency_optimizer_implemented: Literal[True] = True
    latency_measurement_implemented: Literal[False] = False
    latency_threshold_evaluation_implemented: Literal[False] = False
    latency_based_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    async_execution_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["latency_optimization_candidate.v1"] = (
        LATENCY_OPTIMIZATION_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_sources(self) -> Self:
        if self.candidate_id != f"latency_optimizer::{self.stage_id}":
            raise ValueError("candidate_id must match stage_id")
        if self.max_parallel_agents != len(self.agent_ids):
            raise ValueError("max_parallel_agents must match agent count")
        expected_savings = _latency_savings_score(self.max_parallel_agents)
        if self.advisory_latency_savings_score != expected_savings:
            raise ValueError(
                "advisory_latency_savings_score must match max_parallel_agents"
            )
        expected_pressure = _latency_pressure_score(
            savings_score=self.advisory_latency_savings_score,
            blocking_input_count=self.blocking_input_count,
        )
        if self.advisory_latency_pressure_score != expected_pressure:
            raise ValueError(
                "advisory_latency_pressure_score must match candidate inputs"
            )
        if self.latency_band != _latency_band(
            status=self.status,
            max_parallel_agents=self.max_parallel_agents,
            blocking_input_count=self.blocking_input_count,
        ):
            raise ValueError("latency_band must match candidate inputs")
        if self.status == "optimization_candidate" and self.max_parallel_agents <= 1:
            raise ValueError("optimization candidates require multiple agents")
        if self.status == "serial_guardrail" and self.max_parallel_agents != 1:
            raise ValueError("serial guardrails require one agent")
        return self


class LatencyOptimizationPlan(BaseModel):
    """Bounded V5.3 advisory latency optimization plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["latency_optimizer"] = "latency_optimizer"
    serialization_version: Literal["latency_optimization_plan.v1"] = (
        LATENCY_OPTIMIZATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=LATENCY_OPTIMIZER_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    source_performance_tracking_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_parallel_scheduler_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_latency_threshold_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_latency_threshold_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    source_latency_bands: tuple[str, ...] = Field(min_length=4, max_length=4)
    latency_metadata_sources: tuple[str, ...] = Field(min_length=4, max_length=4)
    candidates: tuple[LatencyOptimizationCandidate, ...] = Field(
        min_length=1,
        max_length=20,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    optimization_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=20,
    )
    serial_guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=20,
    )
    candidate_count: int = Field(ge=1, le=20)
    optimization_candidate_count: int = Field(ge=0, le=20)
    serial_guardrail_count: int = Field(ge=0, le=20)
    highest_advisory_latency_savings_score: int = Field(ge=0, le=600)
    total_advisory_latency_savings_score: int = Field(ge=0, le=6000)
    highest_advisory_latency_pressure_score: int = Field(ge=0, le=1000)
    total_blocking_input_count: int = Field(ge=0, le=400)
    latency_optimization_pressure: LatencyOptimizationPressure
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    latency_optimizer_implemented: Literal[True] = True
    latency_measurement_implemented: Literal[False] = False
    latency_threshold_evaluation_implemented: Literal[False] = False
    latency_based_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    async_execution_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
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
        if self.optimization_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "optimization_candidate",
        ):
            raise ValueError("optimization_candidate_ids must match candidates")
        if self.serial_guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "serial_guardrail",
        ):
            raise ValueError("serial_guardrail_candidate_ids must match candidates")
        if self.optimization_candidate_count != len(self.optimization_candidate_ids):
            raise ValueError("optimization_candidate_count must match candidates")
        if self.serial_guardrail_count != len(self.serial_guardrail_candidate_ids):
            raise ValueError("serial_guardrail_count must match candidates")

        expected_highest_savings = max(
            candidate.advisory_latency_savings_score for candidate in self.candidates
        )
        if self.highest_advisory_latency_savings_score != expected_highest_savings:
            raise ValueError(
                "highest_advisory_latency_savings_score must match candidates"
            )
        expected_total_savings = sum(
            candidate.advisory_latency_savings_score for candidate in self.candidates
        )
        if self.total_advisory_latency_savings_score != expected_total_savings:
            raise ValueError(
                "total_advisory_latency_savings_score must match candidates"
            )
        expected_highest_pressure = max(
            candidate.advisory_latency_pressure_score for candidate in self.candidates
        )
        if self.highest_advisory_latency_pressure_score != expected_highest_pressure:
            raise ValueError(
                "highest_advisory_latency_pressure_score must match candidates"
            )
        expected_blocking_inputs = sum(
            candidate.blocking_input_count for candidate in self.candidates
        )
        if self.total_blocking_input_count != expected_blocking_inputs:
            raise ValueError("total_blocking_input_count must match candidates")
        if self.latency_optimization_pressure != _optimization_pressure(
            highest_pressure=self.highest_advisory_latency_pressure_score,
            total_savings=self.total_advisory_latency_savings_score,
        ):
            raise ValueError("latency_optimization_pressure must match candidates")
        return self


def optimize_latency(
    *,
    performance_registry: AgentPerformanceTrackingFoundationRegistry | None = None,
    parallel_scheduler: ParallelSchedulerPlan | None = None,
    latency_thresholds: LatencyThresholdRoutingRegistry | None = None,
) -> LatencyOptimizationPlan:
    """Plan advisory latency optimization without measuring or routing."""

    performance = (
        performance_registry or agent_performance_tracking_foundation_registry()
    )
    scheduler = parallel_scheduler or plan_parallel_scheduler()
    thresholds = latency_thresholds or latency_threshold_routing_registry()
    profiles_by_agent_id = {
        profile.agent_id: profile for profile in performance.profiles
    }
    candidates = tuple(
        _candidate(
            schedule_candidate=schedule_candidate,
            profiles=tuple(
                profiles_by_agent_id[agent_id]
                for agent_id in schedule_candidate.agent_ids
            ),
            latency_thresholds=thresholds,
        )
        for schedule_candidate in scheduler.candidates
    )
    highest_savings = max(
        candidate.advisory_latency_savings_score for candidate in candidates
    )
    total_savings = sum(
        candidate.advisory_latency_savings_score for candidate in candidates
    )
    highest_pressure = max(
        candidate.advisory_latency_pressure_score for candidate in candidates
    )

    return LatencyOptimizationPlan(
        source_performance_tracking_serialization_version=(
            performance.serialization_version
        ),
        source_parallel_scheduler_serialization_version=scheduler.serialization_version,
        source_latency_threshold_serialization_version=thresholds.serialization_version,
        source_latency_threshold_profile_ids=thresholds.latency_threshold_profile_ids,
        source_latency_bands=thresholds.latency_bands,
        latency_metadata_sources=thresholds.latency_metadata_sources,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        optimization_candidate_ids=_candidate_ids_for_status(
            candidates,
            "optimization_candidate",
        ),
        serial_guardrail_candidate_ids=_candidate_ids_for_status(
            candidates,
            "serial_guardrail",
        ),
        candidate_count=len(candidates),
        optimization_candidate_count=len(
            _candidate_ids_for_status(candidates, "optimization_candidate")
        ),
        serial_guardrail_count=len(
            _candidate_ids_for_status(candidates, "serial_guardrail")
        ),
        highest_advisory_latency_savings_score=highest_savings,
        total_advisory_latency_savings_score=total_savings,
        highest_advisory_latency_pressure_score=highest_pressure,
        total_blocking_input_count=sum(
            candidate.blocking_input_count for candidate in candidates
        ),
        latency_optimization_pressure=_optimization_pressure(
            highest_pressure=highest_pressure,
            total_savings=total_savings,
        ),
        advisory_actions=_plan_actions(
            highest_pressure=highest_pressure,
            total_savings=total_savings,
        ),
    )


def latency_optimization_candidate_by_id(
    candidate_id: str,
    plan: LatencyOptimizationPlan | None = None,
) -> LatencyOptimizationCandidate | None:
    """Return one advisory latency candidate without applying optimization."""

    source_plan = plan or optimize_latency()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def latency_optimization_candidates_for_status(
    status: LatencyOptimizationStatus,
    plan: LatencyOptimizationPlan | None = None,
) -> tuple[LatencyOptimizationCandidate, ...]:
    """Return advisory latency candidates by status without workflow control."""

    source_plan = plan or optimize_latency()
    return tuple(
        candidate for candidate in source_plan.candidates if candidate.status == status
    )


def _candidate(
    *,
    schedule_candidate: ParallelScheduleCandidate,
    profiles: tuple[AgentPerformanceTrackingFoundationProfile, ...],
    latency_thresholds: LatencyThresholdRoutingRegistry,
) -> LatencyOptimizationCandidate:
    blocking_input_count = sum(
        len(profile.contract_blocking_inputs) for profile in profiles
    )
    status: LatencyOptimizationStatus = (
        "optimization_candidate"
        if schedule_candidate.max_parallel_agents > 1
        else "serial_guardrail"
    )
    savings_score = _latency_savings_score(schedule_candidate.max_parallel_agents)
    pressure_score = _latency_pressure_score(
        savings_score=savings_score,
        blocking_input_count=blocking_input_count,
    )
    return LatencyOptimizationCandidate(
        candidate_id=f"latency_optimizer::{schedule_candidate.stage_id}",
        source_schedule_candidate_id=schedule_candidate.candidate_id,
        source_scheduling_group_id=schedule_candidate.source_group_id,
        stage_id=schedule_candidate.stage_id,
        agent_ids=schedule_candidate.agent_ids,
        status=status,
        latency_band=_latency_band(
            status=status,
            max_parallel_agents=schedule_candidate.max_parallel_agents,
            blocking_input_count=blocking_input_count,
        ),
        advisory_rank=schedule_candidate.advisory_rank,
        dependency_depth=schedule_candidate.dependency_depth,
        max_parallel_agents=schedule_candidate.max_parallel_agents,
        source_latency_classes=tuple(
            profile.contract_latency_class for profile in profiles
        ),
        source_latency_threshold_profile_ids=(
            latency_thresholds.latency_threshold_profile_ids
        ),
        source_latency_bands=latency_thresholds.latency_bands,
        blocking_input_count=blocking_input_count,
        advisory_latency_savings_score=savings_score,
        advisory_latency_pressure_score=pressure_score,
        evidence=_candidate_evidence(
            schedule_candidate=schedule_candidate,
            profiles=profiles,
            pressure_score=pressure_score,
        ),
        advisory_actions=_candidate_actions(status),
    )


def _candidate_ids_for_status(
    candidates: tuple[LatencyOptimizationCandidate, ...],
    status: LatencyOptimizationStatus,
) -> tuple[str, ...]:
    return tuple(
        candidate.candidate_id for candidate in candidates if candidate.status == status
    )


def _latency_savings_score(max_parallel_agents: int) -> int:
    return max(0, max_parallel_agents - 1) * 100


def _latency_pressure_score(
    *,
    savings_score: int,
    blocking_input_count: int,
) -> int:
    return min(1000, savings_score + blocking_input_count * 5)


def _latency_band(
    *,
    status: LatencyOptimizationStatus,
    max_parallel_agents: int,
    blocking_input_count: int,
) -> LatencyOptimizationBand:
    if status == "serial_guardrail":
        return "guarded"
    if max_parallel_agents >= 3 or blocking_input_count >= 8:
        return "medium"
    return "low"


def _optimization_pressure(
    *,
    highest_pressure: int,
    total_savings: int,
) -> LatencyOptimizationPressure:
    if highest_pressure >= 180 or total_savings >= 500:
        return "high"
    if highest_pressure >= 100 or total_savings >= 200:
        return "medium"
    return "low"


def _candidate_evidence(
    *,
    schedule_candidate: ParallelScheduleCandidate,
    profiles: tuple[AgentPerformanceTrackingFoundationProfile, ...],
    pressure_score: int,
) -> tuple[str, ...]:
    return (
        f"schedule_candidate:{schedule_candidate.candidate_id}",
        f"stage:{schedule_candidate.stage_id}",
        f"agents:{len(schedule_candidate.agent_ids)}",
        f"blocking_inputs:{sum(len(p.contract_blocking_inputs) for p in profiles)}",
        f"pressure_score:{pressure_score}",
    )


def _candidate_actions(
    status: LatencyOptimizationStatus,
) -> tuple[str, ...]:
    if status == "optimization_candidate":
        return (
            "Expose latency-saving candidate as advisory metadata only.",
            "Require explicit future scheduler authority before runtime use.",
        )
    return (
        "Retain serial latency guardrail as advisory metadata only.",
        "Keep ordering visible for later latency review.",
    )


def _plan_actions(
    *,
    highest_pressure: int,
    total_savings: int,
) -> tuple[str, ...]:
    actions = [
        "Expose latency optimization candidates as advisory metadata only.",
        "Preserve measurement, threshold, routing, timing, execution, and output "
        "boundaries.",
    ]
    if highest_pressure:
        actions.append("Report latency pressure without measuring runtime duration.")
    if total_savings:
        actions.append("Use savings score only for later performance review.")
    return tuple(actions)
