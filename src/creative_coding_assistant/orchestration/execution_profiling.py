"""V5.3 advisory execution profiling planning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agent_performance_tracking_foundation import (
    AgentPerformanceTrackingFoundationRegistry,
    agent_performance_tracking_foundation_registry,
)
from .execution_graph_analyzer import (
    ExecutionGraphAnalysis,
    analyze_assistant_execution_graph,
)
from .latency_optimizer import LatencyOptimizationPlan, optimize_latency
from .load_balancer import LoadBalancerPlan, plan_load_balancer

ExecutionProfileKind = Literal[
    "critical_path_profile",
    "agent_latency_profile",
    "load_pressure_profile",
    "failure_path_profile",
    "measurement_boundary_profile",
]
ExecutionProfileStatus = Literal[
    "profile_candidate",
    "measurement_guardrail",
    "failure_guardrail",
]
ExecutionProfilePressure = Literal["low", "medium", "high", "guarded"]

EXECUTION_PROFILE_CANDIDATE_SERIALIZATION_VERSION = (
    "execution_profile_candidate.v1"
)
EXECUTION_PROFILING_PLAN_SERIALIZATION_VERSION = "execution_profiling_plan.v1"
EXECUTION_PROFILING_AUTHORITY_BOUNDARY = (
    "Execution profiling planning derives advisory profile candidates from "
    "static execution graph topology, passive performance tracking metadata, "
    "latency optimization metadata, and load balancer planning metadata only; "
    "it does not measure timing, install profiler hooks, collect runtime "
    "traces, execute workflow graphs, invoke agents or node handlers, route "
    "providers or models, select runtimes, trigger retries, mutate prompts, "
    "write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_profiling",
    "timing_measurement",
    "profiler_hook_installation",
    "runtime_trace_collection",
    "latency_measurement",
    "latency_threshold_evaluation",
    "load_balancing_runtime",
    "workflow_timing_change",
    "workflow_graph_mutation",
    "langgraph_compilation",
    "workflow_execution",
    "agent_invocation",
    "node_handler_invocation",
    "provider_or_model_routing",
    "runtime_selection",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class ExecutionProfileCandidate(BaseModel):
    """One advisory V5.3 execution profiling candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    profile_id: str = Field(min_length=1, max_length=120)
    profile_kind: ExecutionProfileKind
    status: ExecutionProfileStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=100)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    profiled_node_count: int = Field(ge=0, le=40)
    profiled_agent_count: int = Field(ge=0, le=40)
    blocking_input_count: int = Field(ge=0, le=400)
    advisory_profile_score: int = Field(ge=0, le=2_500)
    profile_pressure: ExecutionProfilePressure
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    execution_profiling_planning_implemented: Literal[True] = True
    runtime_profiling_implemented: Literal[False] = False
    timing_measurement_implemented: Literal[False] = False
    profiler_hook_installation_implemented: Literal[False] = False
    runtime_trace_collection_implemented: Literal[False] = False
    latency_measurement_implemented: Literal[False] = False
    latency_threshold_evaluation_implemented: Literal[False] = False
    load_balancing_runtime_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["execution_profile_candidate.v1"] = (
        EXECUTION_PROFILE_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_profile(self) -> Self:
        if self.candidate_id != f"execution_profiling::{self.profile_id}":
            raise ValueError("candidate_id must match profile_id")
        expected_score = _profile_score(
            status=self.status,
            profiled_node_count=self.profiled_node_count,
            profiled_agent_count=self.profiled_agent_count,
            blocking_input_count=self.blocking_input_count,
        )
        if self.advisory_profile_score != expected_score:
            raise ValueError("advisory_profile_score must match profile inputs")
        if self.profile_pressure != _profile_pressure(
            status=self.status,
            profile_score=self.advisory_profile_score,
        ):
            raise ValueError("profile_pressure must match profile inputs")
        if self.status == "profile_candidate" and self.advisory_profile_score <= 0:
            raise ValueError("profile candidates require advisory score")
        if self.status == "failure_guardrail" and self.profiled_node_count <= 0:
            raise ValueError("failure guardrails require profiled nodes")
        return self


class ExecutionProfilingPlan(BaseModel):
    """Bounded V5.3 advisory execution profiling plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["execution_profiler"] = "execution_profiler"
    serialization_version: Literal["execution_profiling_plan.v1"] = (
        EXECUTION_PROFILING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=EXECUTION_PROFILING_AUTHORITY_BOUNDARY,
        max_length=1600,
    )
    source_graph_serialization_version: str = Field(min_length=1, max_length=100)
    source_performance_tracking_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_latency_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_load_balancer_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_graph_node_count: int = Field(ge=1, le=40)
    source_agent_profile_count: int = Field(ge=1, le=40)
    failure_path_reachable: bool
    candidates: tuple[ExecutionProfileCandidate, ...] = Field(
        min_length=1,
        max_length=12,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    profile_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    measurement_guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    failure_guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    candidate_count: int = Field(ge=1, le=12)
    profile_candidate_count: int = Field(ge=0, le=12)
    measurement_guardrail_count: int = Field(ge=0, le=12)
    failure_guardrail_count: int = Field(ge=0, le=12)
    total_profiled_node_count: int = Field(ge=0, le=120)
    total_profiled_agent_count: int = Field(ge=0, le=80)
    total_blocking_input_count: int = Field(ge=0, le=800)
    highest_advisory_profile_score: int = Field(ge=0, le=2_500)
    total_advisory_profile_score: int = Field(ge=0, le=20_000)
    execution_profile_pressure: ExecutionProfilePressure
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    execution_profiling_planning_implemented: Literal[True] = True
    runtime_profiling_implemented: Literal[False] = False
    timing_measurement_implemented: Literal[False] = False
    profiler_hook_installation_implemented: Literal[False] = False
    runtime_trace_collection_implemented: Literal[False] = False
    latency_measurement_implemented: Literal[False] = False
    latency_threshold_evaluation_implemented: Literal[False] = False
    load_balancing_runtime_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
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
        if self.profile_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "profile_candidate",
        ):
            raise ValueError("profile_candidate_ids must match candidates")
        if self.measurement_guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "measurement_guardrail",
        ):
            raise ValueError(
                "measurement_guardrail_candidate_ids must match candidates"
            )
        if self.failure_guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "failure_guardrail",
        ):
            raise ValueError("failure_guardrail_candidate_ids must match candidates")
        if self.profile_candidate_count != len(self.profile_candidate_ids):
            raise ValueError("profile_candidate_count must match candidates")
        if self.measurement_guardrail_count != len(
            self.measurement_guardrail_candidate_ids
        ):
            raise ValueError("measurement_guardrail_count must match candidates")
        if self.failure_guardrail_count != len(self.failure_guardrail_candidate_ids):
            raise ValueError("failure_guardrail_count must match candidates")

        expected_node_count = sum(
            candidate.profiled_node_count for candidate in self.candidates
        )
        if self.total_profiled_node_count != expected_node_count:
            raise ValueError("total_profiled_node_count must match candidates")
        expected_agent_count = sum(
            candidate.profiled_agent_count for candidate in self.candidates
        )
        if self.total_profiled_agent_count != expected_agent_count:
            raise ValueError("total_profiled_agent_count must match candidates")
        expected_blocking_inputs = sum(
            candidate.blocking_input_count for candidate in self.candidates
        )
        if self.total_blocking_input_count != expected_blocking_inputs:
            raise ValueError("total_blocking_input_count must match candidates")
        expected_highest_score = max(
            candidate.advisory_profile_score for candidate in self.candidates
        )
        if self.highest_advisory_profile_score != expected_highest_score:
            raise ValueError("highest_advisory_profile_score must match candidates")
        expected_total_score = sum(
            candidate.advisory_profile_score for candidate in self.candidates
        )
        if self.total_advisory_profile_score != expected_total_score:
            raise ValueError("total_advisory_profile_score must match candidates")
        if self.execution_profile_pressure != _plan_pressure(
            candidates=self.candidates,
            highest_score=self.highest_advisory_profile_score,
        ):
            raise ValueError("execution_profile_pressure must match candidates")
        return self


def plan_execution_profiling(
    *,
    execution_graph: ExecutionGraphAnalysis | None = None,
    performance_registry: AgentPerformanceTrackingFoundationRegistry | None = None,
    latency_optimization: LatencyOptimizationPlan | None = None,
    load_balancer: LoadBalancerPlan | None = None,
) -> ExecutionProfilingPlan:
    """Plan advisory execution profiling without measuring runtime work."""

    graph = execution_graph or analyze_assistant_execution_graph()
    performance = (
        performance_registry or agent_performance_tracking_foundation_registry()
    )
    latency = latency_optimization or optimize_latency()
    load = load_balancer or plan_load_balancer(latency_optimization=latency)
    candidates = _candidates(
        graph=graph,
        performance=performance,
        latency=latency,
        load=load,
    )
    highest_score = max(candidate.advisory_profile_score for candidate in candidates)

    return ExecutionProfilingPlan(
        source_graph_serialization_version=graph.serialization_version,
        source_performance_tracking_serialization_version=(
            performance.serialization_version
        ),
        source_latency_optimization_serialization_version=(
            latency.serialization_version
        ),
        source_load_balancer_serialization_version=load.serialization_version,
        source_graph_node_count=graph.node_count,
        source_agent_profile_count=performance.profile_count,
        failure_path_reachable=graph.failure_path_reachable,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        profile_candidate_ids=_candidate_ids_for_status(
            candidates,
            "profile_candidate",
        ),
        measurement_guardrail_candidate_ids=_candidate_ids_for_status(
            candidates,
            "measurement_guardrail",
        ),
        failure_guardrail_candidate_ids=_candidate_ids_for_status(
            candidates,
            "failure_guardrail",
        ),
        candidate_count=len(candidates),
        profile_candidate_count=len(
            _candidate_ids_for_status(candidates, "profile_candidate")
        ),
        measurement_guardrail_count=len(
            _candidate_ids_for_status(candidates, "measurement_guardrail")
        ),
        failure_guardrail_count=len(
            _candidate_ids_for_status(candidates, "failure_guardrail")
        ),
        total_profiled_node_count=sum(
            candidate.profiled_node_count for candidate in candidates
        ),
        total_profiled_agent_count=sum(
            candidate.profiled_agent_count for candidate in candidates
        ),
        total_blocking_input_count=sum(
            candidate.blocking_input_count for candidate in candidates
        ),
        highest_advisory_profile_score=highest_score,
        total_advisory_profile_score=sum(
            candidate.advisory_profile_score for candidate in candidates
        ),
        execution_profile_pressure=_plan_pressure(
            candidates=candidates,
            highest_score=highest_score,
        ),
        advisory_actions=_plan_actions(candidates),
    )


def execution_profile_candidate_by_id(
    candidate_id: str,
    plan: ExecutionProfilingPlan | None = None,
) -> ExecutionProfileCandidate | None:
    """Return one advisory execution profile candidate without profiling."""

    source_plan = plan or plan_execution_profiling()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def execution_profile_candidates_for_status(
    status: ExecutionProfileStatus,
    plan: ExecutionProfilingPlan | None = None,
) -> tuple[ExecutionProfileCandidate, ...]:
    """Return execution profile candidates by status without measurement."""

    source_plan = plan or plan_execution_profiling()
    return tuple(
        candidate
        for candidate in source_plan.candidates
        if candidate.status == status
    )


def _candidates(
    *,
    graph: ExecutionGraphAnalysis,
    performance: AgentPerformanceTrackingFoundationRegistry,
    latency: LatencyOptimizationPlan,
    load: LoadBalancerPlan,
) -> tuple[ExecutionProfileCandidate, ...]:
    blocking_input_count = sum(
        len(profile.contract_blocking_inputs) for profile in performance.profiles
    )
    return (
        _candidate(
            profile_id="critical_path_profile",
            kind="critical_path_profile",
            status="profile_candidate",
            source_id="execution_graph_analysis",
            source_serialization_version=graph.serialization_version,
            source_item_ids=graph.critical_path_node_ids,
            profiled_node_count=len(graph.critical_path_node_ids),
            profiled_agent_count=0,
            blocking_input_count=0,
            evidence=(
                f"critical_path_nodes:{len(graph.critical_path_node_ids)}",
                f"branch_count:{graph.branch_count}",
            ),
        ),
        _candidate(
            profile_id="agent_latency_profile",
            kind="agent_latency_profile",
            status="profile_candidate",
            source_id="agent_performance_tracking_foundation_registry",
            source_serialization_version=performance.serialization_version,
            source_item_ids=performance.agent_ids,
            profiled_node_count=0,
            profiled_agent_count=performance.profile_count,
            blocking_input_count=blocking_input_count,
            evidence=(
                f"agent_profiles:{performance.profile_count}",
                f"latency_classes:{len(performance.latency_classes)}",
            ),
        ),
        _candidate(
            profile_id="load_pressure_profile",
            kind="load_pressure_profile",
            status="profile_candidate",
            source_id="load_balancer_plan",
            source_serialization_version=load.serialization_version,
            source_item_ids=load.candidate_ids,
            profiled_node_count=0,
            profiled_agent_count=0,
            blocking_input_count=latency.total_blocking_input_count,
            evidence=(
                f"load_pressure:{load.load_balancing_pressure}",
                f"latency_pressure:{latency.latency_optimization_pressure}",
            ),
        ),
        _candidate(
            profile_id="failure_path_profile",
            kind="failure_path_profile",
            status="failure_guardrail",
            source_id="execution_graph_analysis",
            source_serialization_version=graph.serialization_version,
            source_item_ids=graph.failure_entry_node_ids,
            profiled_node_count=len(graph.failure_entry_node_ids),
            profiled_agent_count=0,
            blocking_input_count=0,
            evidence=(
                f"failure_path_reachable:{graph.failure_path_reachable}",
                f"failure_edges:{graph.failure_edge_count}",
            ),
        ),
        _candidate(
            profile_id="measurement_boundary_profile",
            kind="measurement_boundary_profile",
            status="measurement_guardrail",
            source_id="agent_performance_tracking_foundation_registry",
            source_serialization_version=performance.serialization_version,
            source_item_ids=performance.performance_source_registries,
            profiled_node_count=0,
            profiled_agent_count=0,
            blocking_input_count=0,
            evidence=(
                "runtime_measurement:blocked",
                "trace_collection:blocked",
            ),
        ),
    )


def _candidate(
    *,
    profile_id: str,
    kind: ExecutionProfileKind,
    status: ExecutionProfileStatus,
    source_id: str,
    source_serialization_version: str,
    source_item_ids: tuple[str, ...],
    profiled_node_count: int,
    profiled_agent_count: int,
    blocking_input_count: int,
    evidence: tuple[str, ...],
) -> ExecutionProfileCandidate:
    score = _profile_score(
        status=status,
        profiled_node_count=profiled_node_count,
        profiled_agent_count=profiled_agent_count,
        blocking_input_count=blocking_input_count,
    )
    return ExecutionProfileCandidate(
        candidate_id=f"execution_profiling::{profile_id}",
        profile_id=profile_id,
        profile_kind=kind,
        status=status,
        source_id=source_id,
        source_serialization_version=source_serialization_version,
        source_item_ids=source_item_ids,
        profiled_node_count=profiled_node_count,
        profiled_agent_count=profiled_agent_count,
        blocking_input_count=blocking_input_count,
        advisory_profile_score=score,
        profile_pressure=_profile_pressure(status=status, profile_score=score),
        evidence=evidence,
        advisory_actions=_candidate_actions(status),
    )


def _candidate_ids_for_status(
    candidates: tuple[ExecutionProfileCandidate, ...],
    status: ExecutionProfileStatus,
) -> tuple[str, ...]:
    return tuple(
        candidate.candidate_id
        for candidate in candidates
        if candidate.status == status
    )


def _profile_score(
    *,
    status: ExecutionProfileStatus,
    profiled_node_count: int,
    profiled_agent_count: int,
    blocking_input_count: int,
) -> int:
    if status == "measurement_guardrail":
        return 0
    score = (
        profiled_node_count * 35
        + profiled_agent_count * 40
        + blocking_input_count * 60
    )
    if status == "failure_guardrail":
        score += 100
    return min(2_500, score)


def _profile_pressure(
    *,
    status: ExecutionProfileStatus,
    profile_score: int,
) -> ExecutionProfilePressure:
    if status in {"measurement_guardrail", "failure_guardrail"}:
        return "guarded"
    if profile_score >= 900:
        return "high"
    if profile_score >= 350:
        return "medium"
    return "low"


def _plan_pressure(
    *,
    candidates: tuple[ExecutionProfileCandidate, ...],
    highest_score: int,
) -> ExecutionProfilePressure:
    if any(candidate.status != "profile_candidate" for candidate in candidates):
        return "guarded"
    if highest_score >= 900:
        return "high"
    if highest_score >= 350:
        return "medium"
    return "low"


def _candidate_actions(status: ExecutionProfileStatus) -> tuple[str, ...]:
    if status == "profile_candidate":
        return (
            "Expose execution profile metadata for inspection only.",
            "Require explicit runtime authority before profiling execution.",
        )
    if status == "failure_guardrail":
        return (
            "Preserve failure path visibility without executing failure paths.",
        )
    return (
        "Keep timing measurement and runtime trace collection disabled.",
    )


def _plan_actions(
    candidates: tuple[ExecutionProfileCandidate, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose execution profiling posture as advisory metadata only.",
        "Preserve timing, tracing, workflow, routing, and output boundaries.",
    ]
    if _candidate_ids_for_status(candidates, "measurement_guardrail"):
        actions.append("Keep profiling detached from runtime measurement hooks.")
    return tuple(actions)
