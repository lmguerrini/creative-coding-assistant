"""V5.3 advisory bottleneck detection planning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .execution_profiling import ExecutionProfilingPlan, plan_execution_profiling
from .execution_replay_engine import ExecutionReplayPlan, plan_execution_replay
from .latency_optimizer import LatencyOptimizationPlan, optimize_latency
from .load_balancer import LoadBalancerPlan, plan_load_balancer

BottleneckKind = Literal[
    "latency_pressure",
    "load_pressure",
    "profiling_pressure",
    "replay_boundary",
    "routing_boundary",
]
BottleneckStatus = Literal[
    "bottleneck_candidate",
    "boundary_guardrail",
    "review_only",
]
BottleneckSeverity = Literal["low", "medium", "high", "guarded"]

BOTTLENECK_CANDIDATE_SERIALIZATION_VERSION = "bottleneck_candidate.v1"
BOTTLENECK_DETECTION_PLAN_SERIALIZATION_VERSION = "bottleneck_detection_plan.v1"
BOTTLENECK_DETECTION_AUTHORITY_BOUNDARY = (
    "Bottleneck detection planning derives advisory bottleneck candidates "
    "from latency optimization, load balancer, execution profiling, and "
    "execution replay metadata only; it does not measure runtime latency, "
    "install profilers, collect traces, rebalance load, route providers or "
    "models, select runtimes, execute workflows, invoke agents or node "
    "handlers, trigger retries, mutate prompts, write storage, or modify "
    "generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_bottleneck_detection",
    "runtime_latency_measurement",
    "profiler_hook_installation",
    "runtime_trace_collection",
    "load_balancing_runtime",
    "latency_based_routing",
    "provider_or_model_routing",
    "runtime_selection",
    "workflow_timing_change",
    "workflow_graph_mutation",
    "langgraph_compilation",
    "workflow_execution",
    "agent_invocation",
    "node_handler_invocation",
    "execution_replay_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class BottleneckCandidate(BaseModel):
    """One advisory V5.3 bottleneck detection candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    bottleneck_id: str = Field(min_length=1, max_length=120)
    bottleneck_kind: BottleneckKind
    status: BottleneckStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=100)
    source_candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    signal_count: int = Field(ge=0, le=400)
    blocking_input_count: int = Field(ge=0, le=800)
    advisory_severity_score: int = Field(ge=0, le=3_000)
    severity: BottleneckSeverity
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    bottleneck_detection_planning_implemented: Literal[True] = True
    runtime_bottleneck_detection_implemented: Literal[False] = False
    runtime_latency_measurement_implemented: Literal[False] = False
    profiler_hook_installation_implemented: Literal[False] = False
    runtime_trace_collection_implemented: Literal[False] = False
    load_balancing_runtime_implemented: Literal[False] = False
    latency_based_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    execution_replay_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["bottleneck_candidate.v1"] = (
        BOTTLENECK_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_bottleneck(self) -> Self:
        if self.candidate_id != f"bottleneck_detection::{self.bottleneck_id}":
            raise ValueError("candidate_id must match bottleneck_id")
        expected_score = _severity_score(
            status=self.status,
            signal_count=self.signal_count,
            blocking_input_count=self.blocking_input_count,
        )
        if self.advisory_severity_score != expected_score:
            raise ValueError("advisory_severity_score must match candidate inputs")
        if self.severity != _severity(
            status=self.status,
            severity_score=self.advisory_severity_score,
        ):
            raise ValueError("severity must match candidate inputs")
        if self.status == "bottleneck_candidate" and self.signal_count <= 0:
            raise ValueError("bottleneck candidates require signals")
        return self


class BottleneckDetectionPlan(BaseModel):
    """Bounded V5.3 advisory bottleneck detection plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["bottleneck_detector"] = "bottleneck_detector"
    serialization_version: Literal["bottleneck_detection_plan.v1"] = (
        BOTTLENECK_DETECTION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=BOTTLENECK_DETECTION_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    source_latency_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_load_balancer_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_execution_profiling_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_execution_replay_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    candidates: tuple[BottleneckCandidate, ...] = Field(
        min_length=1,
        max_length=12,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    bottleneck_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    boundary_guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    review_only_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    candidate_count: int = Field(ge=1, le=12)
    bottleneck_candidate_count: int = Field(ge=0, le=12)
    boundary_guardrail_count: int = Field(ge=0, le=12)
    review_only_count: int = Field(ge=0, le=12)
    total_signal_count: int = Field(ge=0, le=1_000)
    total_blocking_input_count: int = Field(ge=0, le=2_000)
    highest_advisory_severity_score: int = Field(ge=0, le=3_000)
    total_advisory_severity_score: int = Field(ge=0, le=20_000)
    bottleneck_detection_severity: BottleneckSeverity
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    bottleneck_detection_planning_implemented: Literal[True] = True
    runtime_bottleneck_detection_implemented: Literal[False] = False
    runtime_latency_measurement_implemented: Literal[False] = False
    profiler_hook_installation_implemented: Literal[False] = False
    runtime_trace_collection_implemented: Literal[False] = False
    load_balancing_runtime_implemented: Literal[False] = False
    latency_based_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    execution_replay_execution_implemented: Literal[False] = False
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
        if self.bottleneck_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "bottleneck_candidate",
        ):
            raise ValueError("bottleneck_candidate_ids must match candidates")
        if self.boundary_guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "boundary_guardrail",
        ):
            raise ValueError("boundary_guardrail_candidate_ids must match candidates")
        if self.review_only_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "review_only",
        ):
            raise ValueError("review_only_candidate_ids must match candidates")
        if self.bottleneck_candidate_count != len(self.bottleneck_candidate_ids):
            raise ValueError("bottleneck_candidate_count must match candidates")
        if self.boundary_guardrail_count != len(self.boundary_guardrail_candidate_ids):
            raise ValueError("boundary_guardrail_count must match candidates")
        if self.review_only_count != len(self.review_only_candidate_ids):
            raise ValueError("review_only_count must match candidates")

        expected_signals = sum(candidate.signal_count for candidate in self.candidates)
        if self.total_signal_count != expected_signals:
            raise ValueError("total_signal_count must match candidates")
        expected_blocking_inputs = sum(
            candidate.blocking_input_count for candidate in self.candidates
        )
        if self.total_blocking_input_count != expected_blocking_inputs:
            raise ValueError("total_blocking_input_count must match candidates")
        expected_highest_score = max(
            candidate.advisory_severity_score for candidate in self.candidates
        )
        if self.highest_advisory_severity_score != expected_highest_score:
            raise ValueError("highest_advisory_severity_score must match candidates")
        expected_total_score = sum(
            candidate.advisory_severity_score for candidate in self.candidates
        )
        if self.total_advisory_severity_score != expected_total_score:
            raise ValueError("total_advisory_severity_score must match candidates")
        if self.bottleneck_detection_severity != _plan_severity(
            candidates=self.candidates,
            highest_score=self.highest_advisory_severity_score,
        ):
            raise ValueError("bottleneck_detection_severity must match candidates")
        return self


def detect_bottlenecks(
    *,
    latency_optimization: LatencyOptimizationPlan | None = None,
    load_balancer: LoadBalancerPlan | None = None,
    execution_profiling: ExecutionProfilingPlan | None = None,
    execution_replay: ExecutionReplayPlan | None = None,
) -> BottleneckDetectionPlan:
    """Plan advisory bottleneck detection without measuring runtime work."""

    latency = latency_optimization or optimize_latency()
    load = load_balancer or plan_load_balancer(latency_optimization=latency)
    profiling = execution_profiling or plan_execution_profiling(
        latency_optimization=latency,
        load_balancer=load,
    )
    replay = execution_replay or plan_execution_replay(
        execution_profiling=profiling
    )
    candidates = _candidates(
        latency=latency,
        load=load,
        profiling=profiling,
        replay=replay,
    )
    highest_score = max(candidate.advisory_severity_score for candidate in candidates)

    return BottleneckDetectionPlan(
        source_latency_optimization_serialization_version=(
            latency.serialization_version
        ),
        source_load_balancer_serialization_version=load.serialization_version,
        source_execution_profiling_serialization_version=(
            profiling.serialization_version
        ),
        source_execution_replay_serialization_version=replay.serialization_version,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        bottleneck_candidate_ids=_candidate_ids_for_status(
            candidates,
            "bottleneck_candidate",
        ),
        boundary_guardrail_candidate_ids=_candidate_ids_for_status(
            candidates,
            "boundary_guardrail",
        ),
        review_only_candidate_ids=_candidate_ids_for_status(candidates, "review_only"),
        candidate_count=len(candidates),
        bottleneck_candidate_count=len(
            _candidate_ids_for_status(candidates, "bottleneck_candidate")
        ),
        boundary_guardrail_count=len(
            _candidate_ids_for_status(candidates, "boundary_guardrail")
        ),
        review_only_count=len(_candidate_ids_for_status(candidates, "review_only")),
        total_signal_count=sum(candidate.signal_count for candidate in candidates),
        total_blocking_input_count=sum(
            candidate.blocking_input_count for candidate in candidates
        ),
        highest_advisory_severity_score=highest_score,
        total_advisory_severity_score=sum(
            candidate.advisory_severity_score for candidate in candidates
        ),
        bottleneck_detection_severity=_plan_severity(
            candidates=candidates,
            highest_score=highest_score,
        ),
        advisory_actions=_plan_actions(candidates),
    )


def bottleneck_candidate_by_id(
    candidate_id: str,
    plan: BottleneckDetectionPlan | None = None,
) -> BottleneckCandidate | None:
    """Return one advisory bottleneck candidate without measuring runtime."""

    source_plan = plan or detect_bottlenecks()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def bottleneck_candidates_for_status(
    status: BottleneckStatus,
    plan: BottleneckDetectionPlan | None = None,
) -> tuple[BottleneckCandidate, ...]:
    """Return bottleneck candidates by status without runtime detection."""

    source_plan = plan or detect_bottlenecks()
    return tuple(
        candidate
        for candidate in source_plan.candidates
        if candidate.status == status
    )


def _candidates(
    *,
    latency: LatencyOptimizationPlan,
    load: LoadBalancerPlan,
    profiling: ExecutionProfilingPlan,
    replay: ExecutionReplayPlan,
) -> tuple[BottleneckCandidate, ...]:
    return (
        _candidate(
            bottleneck_id="latency_pressure",
            kind="latency_pressure",
            status="bottleneck_candidate",
            source_id="latency_optimization_plan",
            source_serialization_version=latency.serialization_version,
            source_candidate_ids=latency.candidate_ids,
            signal_count=latency.candidate_count,
            blocking_input_count=latency.total_blocking_input_count,
            evidence=(
                f"latency_pressure:{latency.latency_optimization_pressure}",
                f"blocking_inputs:{latency.total_blocking_input_count}",
            ),
        ),
        _candidate(
            bottleneck_id="load_pressure",
            kind="load_pressure",
            status="bottleneck_candidate",
            source_id="load_balancer_plan",
            source_serialization_version=load.serialization_version,
            source_candidate_ids=load.candidate_ids,
            signal_count=load.candidate_count,
            blocking_input_count=load.total_advisory_load_units,
            evidence=(
                f"load_pressure:{load.load_balancing_pressure}",
                f"load_units:{load.total_advisory_load_units}",
            ),
        ),
        _candidate(
            bottleneck_id="profiling_pressure",
            kind="profiling_pressure",
            status="bottleneck_candidate",
            source_id="execution_profiling_plan",
            source_serialization_version=profiling.serialization_version,
            source_candidate_ids=profiling.candidate_ids,
            signal_count=profiling.candidate_count,
            blocking_input_count=profiling.total_blocking_input_count,
            evidence=(
                f"profile_pressure:{profiling.execution_profile_pressure}",
                f"profile_signals:{profiling.candidate_count}",
            ),
        ),
        _candidate(
            bottleneck_id="replay_boundary",
            kind="replay_boundary",
            status="boundary_guardrail",
            source_id="execution_replay_plan",
            source_serialization_version=replay.serialization_version,
            source_candidate_ids=replay.candidate_ids,
            signal_count=replay.candidate_count,
            blocking_input_count=0,
            evidence=(
                "execution_replay_execution:blocked",
                "runtime_trace_collection:blocked",
            ),
        ),
        _candidate(
            bottleneck_id="routing_boundary",
            kind="routing_boundary",
            status="review_only",
            source_id="load_balancer_plan",
            source_serialization_version=load.serialization_version,
            source_candidate_ids=load.routing_guardrail_candidate_ids,
            signal_count=load.routing_guardrail_count,
            blocking_input_count=0,
            evidence=(
                "provider_model_routing:blocked",
                "latency_based_routing:blocked",
            ),
        ),
    )


def _candidate(
    *,
    bottleneck_id: str,
    kind: BottleneckKind,
    status: BottleneckStatus,
    source_id: str,
    source_serialization_version: str,
    source_candidate_ids: tuple[str, ...],
    signal_count: int,
    blocking_input_count: int,
    evidence: tuple[str, ...],
) -> BottleneckCandidate:
    score = _severity_score(
        status=status,
        signal_count=signal_count,
        blocking_input_count=blocking_input_count,
    )
    return BottleneckCandidate(
        candidate_id=f"bottleneck_detection::{bottleneck_id}",
        bottleneck_id=bottleneck_id,
        bottleneck_kind=kind,
        status=status,
        source_id=source_id,
        source_serialization_version=source_serialization_version,
        source_candidate_ids=source_candidate_ids,
        signal_count=signal_count,
        blocking_input_count=blocking_input_count,
        advisory_severity_score=score,
        severity=_severity(status=status, severity_score=score),
        evidence=evidence,
        advisory_actions=_candidate_actions(status),
    )


def _candidate_ids_for_status(
    candidates: tuple[BottleneckCandidate, ...],
    status: BottleneckStatus,
) -> tuple[str, ...]:
    return tuple(
        candidate.candidate_id
        for candidate in candidates
        if candidate.status == status
    )


def _severity_score(
    *,
    status: BottleneckStatus,
    signal_count: int,
    blocking_input_count: int,
) -> int:
    if status == "review_only":
        return signal_count * 50
    score = signal_count * 120 + blocking_input_count * 35
    if status == "boundary_guardrail":
        score += 100
    return min(3_000, score)


def _severity(
    *,
    status: BottleneckStatus,
    severity_score: int,
) -> BottleneckSeverity:
    if status == "boundary_guardrail":
        return "guarded"
    if severity_score >= 900:
        return "high"
    if severity_score >= 300:
        return "medium"
    return "low"


def _plan_severity(
    *,
    candidates: tuple[BottleneckCandidate, ...],
    highest_score: int,
) -> BottleneckSeverity:
    if any(candidate.status == "boundary_guardrail" for candidate in candidates):
        return "guarded"
    if highest_score >= 900:
        return "high"
    if highest_score >= 300:
        return "medium"
    return "low"


def _candidate_actions(status: BottleneckStatus) -> tuple[str, ...]:
    if status == "bottleneck_candidate":
        return (
            "Expose bottleneck signal metadata for inspection only.",
            "Require explicit runtime authority before measuring bottlenecks.",
        )
    if status == "boundary_guardrail":
        return (
            "Preserve replay, profiling, workflow, and trace boundaries.",
        )
    return (
        "Keep routing-related bottleneck signals review-only.",
    )


def _plan_actions(
    candidates: tuple[BottleneckCandidate, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose bottleneck detection posture as advisory metadata only.",
        "Preserve measurement, profiling, routing, workflow, and output boundaries.",
    ]
    if _candidate_ids_for_status(candidates, "boundary_guardrail"):
        actions.append("Keep bottleneck detection detached from runtime probes.")
    return tuple(actions)
