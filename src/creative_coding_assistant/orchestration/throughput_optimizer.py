"""V5.3 advisory throughput optimization planning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .async_execution import AsyncExecutionPlan, plan_async_execution
from .bottleneck_detection import BottleneckDetectionPlan, detect_bottlenecks
from .load_balancer import LoadBalancerPlan, plan_load_balancer
from .streaming_optimizer import StreamingOptimizationPlan, optimize_streaming

ThroughputOptimizationKind = Literal[
    "async_slot_throughput",
    "stream_batch_throughput",
    "load_capacity_throughput",
    "bottleneck_backpressure",
    "routing_boundary",
]
ThroughputOptimizationStatus = Literal[
    "throughput_candidate",
    "capacity_guardrail",
    "boundary_guardrail",
]
ThroughputOptimizationPressure = Literal["low", "medium", "high", "guarded"]

THROUGHPUT_OPTIMIZATION_CANDIDATE_SERIALIZATION_VERSION = (
    "throughput_optimization_candidate.v1"
)
THROUGHPUT_OPTIMIZATION_PLAN_SERIALIZATION_VERSION = (
    "throughput_optimization_plan.v1"
)
THROUGHPUT_OPTIMIZER_AUTHORITY_BOUNDARY = (
    "Throughput optimization planning derives advisory throughput candidates "
    "from async execution, streaming optimization, load balancer, and "
    "bottleneck detection metadata only; it does not measure throughput, "
    "change concurrency limits, manage queues, batch stream chunks, distribute "
    "requests, rebalance load, enforce capacity, route providers or models, "
    "alter workflow timing, mutate graph order, compile or execute workflow "
    "graphs, invoke agents or node handlers, trigger retries, mutate prompts, "
    "write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "throughput_runtime_optimization",
    "runtime_throughput_measurement",
    "concurrency_limit_change",
    "queue_management_runtime",
    "stream_chunk_batching_runtime",
    "request_distribution",
    "load_balancing_runtime",
    "capacity_enforcement",
    "provider_or_model_routing",
    "workflow_timing_change",
    "workflow_graph_mutation",
    "langgraph_compilation",
    "workflow_execution",
    "agent_invocation",
    "node_handler_invocation",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class ThroughputOptimizationCandidate(BaseModel):
    """One advisory V5.3 throughput optimization candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    throughput_id: str = Field(min_length=1, max_length=120)
    throughput_kind: ThroughputOptimizationKind
    status: ThroughputOptimizationStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=100)
    source_candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    advisory_throughput_units: int = Field(ge=0, le=20_000)
    advisory_capacity_units: int = Field(ge=0, le=20_000)
    advisory_backpressure_units: int = Field(ge=0, le=2_000)
    advisory_throughput_score: int = Field(ge=0, le=3_000)
    throughput_pressure: ThroughputOptimizationPressure
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    throughput_optimizer_planning_implemented: Literal[True] = True
    throughput_runtime_optimization_implemented: Literal[False] = False
    throughput_measurement_implemented: Literal[False] = False
    concurrency_limit_change_implemented: Literal[False] = False
    queue_management_runtime_implemented: Literal[False] = False
    stream_chunk_batching_runtime_implemented: Literal[False] = False
    request_distribution_implemented: Literal[False] = False
    load_balancing_runtime_implemented: Literal[False] = False
    capacity_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["throughput_optimization_candidate.v1"] = (
        THROUGHPUT_OPTIMIZATION_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_throughput_inputs(self) -> Self:
        if self.candidate_id != f"throughput_optimizer::{self.throughput_id}":
            raise ValueError("candidate_id must match throughput_id")
        expected_score = _throughput_score(
            status=self.status,
            throughput_units=self.advisory_throughput_units,
            capacity_units=self.advisory_capacity_units,
            backpressure_units=self.advisory_backpressure_units,
        )
        if self.advisory_throughput_score != expected_score:
            raise ValueError("advisory_throughput_score must match candidate inputs")
        if self.throughput_pressure != _throughput_pressure(
            status=self.status,
            throughput_score=self.advisory_throughput_score,
        ):
            raise ValueError("throughput_pressure must match candidate inputs")
        if (
            self.status == "throughput_candidate"
            and self.advisory_throughput_units <= 0
        ):
            raise ValueError("throughput candidates require throughput units")
        if (
            self.status == "capacity_guardrail"
            and self.advisory_backpressure_units <= 0
        ):
            raise ValueError("capacity guardrails require backpressure units")
        if self.status == "boundary_guardrail" and (
            self.advisory_throughput_units
            or self.advisory_capacity_units
            or self.advisory_backpressure_units
        ):
            raise ValueError("boundary guardrails must not declare throughput units")
        return self


class ThroughputOptimizationPlan(BaseModel):
    """Bounded V5.3 advisory throughput optimization plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["throughput_optimizer"] = "throughput_optimizer"
    serialization_version: Literal["throughput_optimization_plan.v1"] = (
        THROUGHPUT_OPTIMIZATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=THROUGHPUT_OPTIMIZER_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    source_async_execution_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_streaming_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_load_balancer_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_bottleneck_detection_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    candidates: tuple[ThroughputOptimizationCandidate, ...] = Field(
        min_length=1,
        max_length=12,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    throughput_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    capacity_guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    boundary_guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    candidate_count: int = Field(ge=1, le=12)
    throughput_candidate_count: int = Field(ge=0, le=12)
    capacity_guardrail_count: int = Field(ge=0, le=12)
    boundary_guardrail_count: int = Field(ge=0, le=12)
    total_advisory_throughput_units: int = Field(ge=0, le=50_000)
    total_advisory_capacity_units: int = Field(ge=0, le=50_000)
    total_advisory_backpressure_units: int = Field(ge=0, le=5_000)
    highest_advisory_throughput_score: int = Field(ge=0, le=3_000)
    total_advisory_throughput_score: int = Field(ge=0, le=20_000)
    throughput_optimization_pressure: ThroughputOptimizationPressure
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    throughput_optimizer_planning_implemented: Literal[True] = True
    throughput_runtime_optimization_implemented: Literal[False] = False
    throughput_measurement_implemented: Literal[False] = False
    concurrency_limit_change_implemented: Literal[False] = False
    queue_management_runtime_implemented: Literal[False] = False
    stream_chunk_batching_runtime_implemented: Literal[False] = False
    request_distribution_implemented: Literal[False] = False
    load_balancing_runtime_implemented: Literal[False] = False
    capacity_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
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
        if self.throughput_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "throughput_candidate",
        ):
            raise ValueError("throughput_candidate_ids must match candidates")
        if self.capacity_guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "capacity_guardrail",
        ):
            raise ValueError("capacity_guardrail_candidate_ids must match candidates")
        if self.boundary_guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "boundary_guardrail",
        ):
            raise ValueError("boundary_guardrail_candidate_ids must match candidates")
        if self.throughput_candidate_count != len(self.throughput_candidate_ids):
            raise ValueError("throughput_candidate_count must match candidates")
        if self.capacity_guardrail_count != len(self.capacity_guardrail_candidate_ids):
            raise ValueError("capacity_guardrail_count must match candidates")
        if self.boundary_guardrail_count != len(self.boundary_guardrail_candidate_ids):
            raise ValueError("boundary_guardrail_count must match candidates")

        expected_throughput = sum(
            candidate.advisory_throughput_units for candidate in self.candidates
        )
        if self.total_advisory_throughput_units != expected_throughput:
            raise ValueError("total_advisory_throughput_units must match candidates")
        expected_capacity = sum(
            candidate.advisory_capacity_units for candidate in self.candidates
        )
        if self.total_advisory_capacity_units != expected_capacity:
            raise ValueError("total_advisory_capacity_units must match candidates")
        expected_backpressure = sum(
            candidate.advisory_backpressure_units for candidate in self.candidates
        )
        if self.total_advisory_backpressure_units != expected_backpressure:
            raise ValueError("total_advisory_backpressure_units must match candidates")
        expected_highest_score = max(
            candidate.advisory_throughput_score for candidate in self.candidates
        )
        if self.highest_advisory_throughput_score != expected_highest_score:
            raise ValueError("highest_advisory_throughput_score must match candidates")
        expected_total_score = sum(
            candidate.advisory_throughput_score for candidate in self.candidates
        )
        if self.total_advisory_throughput_score != expected_total_score:
            raise ValueError("total_advisory_throughput_score must match candidates")
        if self.throughput_optimization_pressure != _plan_pressure(
            candidates=self.candidates,
            highest_score=self.highest_advisory_throughput_score,
        ):
            raise ValueError("throughput_optimization_pressure must match candidates")
        return self


def optimize_throughput(
    *,
    async_execution: AsyncExecutionPlan | None = None,
    streaming_optimization: StreamingOptimizationPlan | None = None,
    load_balancer: LoadBalancerPlan | None = None,
    bottleneck_detection: BottleneckDetectionPlan | None = None,
) -> ThroughputOptimizationPlan:
    """Plan advisory throughput optimization without runtime throughput changes."""

    async_plan = async_execution or plan_async_execution()
    streaming = streaming_optimization or optimize_streaming(
        async_execution=async_plan
    )
    load = load_balancer or plan_load_balancer(async_execution=async_plan)
    bottlenecks = bottleneck_detection or detect_bottlenecks(load_balancer=load)
    candidates = _candidates(
        async_plan=async_plan,
        streaming=streaming,
        load=load,
        bottlenecks=bottlenecks,
    )
    highest_score = max(
        candidate.advisory_throughput_score for candidate in candidates
    )

    return ThroughputOptimizationPlan(
        source_async_execution_serialization_version=async_plan.serialization_version,
        source_streaming_optimization_serialization_version=(
            streaming.serialization_version
        ),
        source_load_balancer_serialization_version=load.serialization_version,
        source_bottleneck_detection_serialization_version=(
            bottlenecks.serialization_version
        ),
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        throughput_candidate_ids=_candidate_ids_for_status(
            candidates,
            "throughput_candidate",
        ),
        capacity_guardrail_candidate_ids=_candidate_ids_for_status(
            candidates,
            "capacity_guardrail",
        ),
        boundary_guardrail_candidate_ids=_candidate_ids_for_status(
            candidates,
            "boundary_guardrail",
        ),
        candidate_count=len(candidates),
        throughput_candidate_count=len(
            _candidate_ids_for_status(candidates, "throughput_candidate")
        ),
        capacity_guardrail_count=len(
            _candidate_ids_for_status(candidates, "capacity_guardrail")
        ),
        boundary_guardrail_count=len(
            _candidate_ids_for_status(candidates, "boundary_guardrail")
        ),
        total_advisory_throughput_units=sum(
            candidate.advisory_throughput_units for candidate in candidates
        ),
        total_advisory_capacity_units=sum(
            candidate.advisory_capacity_units for candidate in candidates
        ),
        total_advisory_backpressure_units=sum(
            candidate.advisory_backpressure_units for candidate in candidates
        ),
        highest_advisory_throughput_score=highest_score,
        total_advisory_throughput_score=sum(
            candidate.advisory_throughput_score for candidate in candidates
        ),
        throughput_optimization_pressure=_plan_pressure(
            candidates=candidates,
            highest_score=highest_score,
        ),
        advisory_actions=_plan_actions(candidates),
    )


def throughput_optimization_candidate_by_id(
    candidate_id: str,
    plan: ThroughputOptimizationPlan | None = None,
) -> ThroughputOptimizationCandidate | None:
    """Return one advisory throughput candidate without runtime optimization."""

    source_plan = plan or optimize_throughput()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def throughput_optimization_candidates_for_status(
    status: ThroughputOptimizationStatus,
    plan: ThroughputOptimizationPlan | None = None,
) -> tuple[ThroughputOptimizationCandidate, ...]:
    """Return throughput candidates by status without runtime changes."""

    source_plan = plan or optimize_throughput()
    return tuple(
        candidate
        for candidate in source_plan.candidates
        if candidate.status == status
    )


def _candidates(
    *,
    async_plan: AsyncExecutionPlan,
    streaming: StreamingOptimizationPlan,
    load: LoadBalancerPlan,
    bottlenecks: BottleneckDetectionPlan,
) -> tuple[ThroughputOptimizationCandidate, ...]:
    return (
        _candidate(
            throughput_id="async_slot_throughput",
            kind="async_slot_throughput",
            status="throughput_candidate",
            source_id="async_execution_plan",
            source_serialization_version=async_plan.serialization_version,
            source_candidate_ids=async_plan.async_ready_candidate_ids,
            throughput_units=async_plan.total_advisory_async_slots,
            capacity_units=async_plan.max_async_width,
            backpressure_units=0,
            evidence=(
                f"async_ready_candidates:{async_plan.async_ready_candidate_count}",
                f"max_async_width:{async_plan.max_async_width}",
            ),
        ),
        _candidate(
            throughput_id="stream_batch_throughput",
            kind="stream_batch_throughput",
            status="throughput_candidate",
            source_id="streaming_optimization_plan",
            source_serialization_version=streaming.serialization_version,
            source_candidate_ids=streaming.optimization_candidate_ids,
            throughput_units=max(1, streaming.total_advisory_stream_readiness_score),
            capacity_units=streaming.stream_event_type_count,
            backpressure_units=0,
            evidence=(
                f"stream_events:{streaming.stream_event_type_count}",
                f"stream_pressure:{streaming.streaming_optimization_pressure}",
            ),
        ),
        _candidate(
            throughput_id="load_capacity_throughput",
            kind="load_capacity_throughput",
            status="throughput_candidate",
            source_id="load_balancer_plan",
            source_serialization_version=load.serialization_version,
            source_candidate_ids=load.balancing_candidate_ids,
            throughput_units=load.total_advisory_load_units,
            capacity_units=load.max_advisory_capacity_slots,
            backpressure_units=0,
            evidence=(
                f"load_pressure:{load.load_balancing_pressure}",
                f"capacity_slots:{load.max_advisory_capacity_slots}",
            ),
        ),
        _candidate(
            throughput_id="bottleneck_backpressure",
            kind="bottleneck_backpressure",
            status="capacity_guardrail",
            source_id="bottleneck_detection_plan",
            source_serialization_version=bottlenecks.serialization_version,
            source_candidate_ids=bottlenecks.bottleneck_candidate_ids,
            throughput_units=0,
            capacity_units=0,
            backpressure_units=bottlenecks.total_blocking_input_count,
            evidence=(
                f"bottleneck_severity:{bottlenecks.bottleneck_detection_severity}",
                f"blocking_inputs:{bottlenecks.total_blocking_input_count}",
            ),
        ),
        _candidate(
            throughput_id="routing_boundary",
            kind="routing_boundary",
            status="boundary_guardrail",
            source_id="load_balancer_plan",
            source_serialization_version=load.serialization_version,
            source_candidate_ids=load.routing_guardrail_candidate_ids,
            throughput_units=0,
            capacity_units=0,
            backpressure_units=0,
            evidence=(
                "provider_model_routing:blocked",
                "request_distribution:blocked",
            ),
        ),
    )


def _candidate(
    *,
    throughput_id: str,
    kind: ThroughputOptimizationKind,
    status: ThroughputOptimizationStatus,
    source_id: str,
    source_serialization_version: str,
    source_candidate_ids: tuple[str, ...],
    throughput_units: int,
    capacity_units: int,
    backpressure_units: int,
    evidence: tuple[str, ...],
) -> ThroughputOptimizationCandidate:
    score = _throughput_score(
        status=status,
        throughput_units=throughput_units,
        capacity_units=capacity_units,
        backpressure_units=backpressure_units,
    )
    return ThroughputOptimizationCandidate(
        candidate_id=f"throughput_optimizer::{throughput_id}",
        throughput_id=throughput_id,
        throughput_kind=kind,
        status=status,
        source_id=source_id,
        source_serialization_version=source_serialization_version,
        source_candidate_ids=source_candidate_ids,
        advisory_throughput_units=throughput_units,
        advisory_capacity_units=capacity_units,
        advisory_backpressure_units=backpressure_units,
        advisory_throughput_score=score,
        throughput_pressure=_throughput_pressure(
            status=status,
            throughput_score=score,
        ),
        evidence=evidence,
        advisory_actions=_candidate_actions(status),
    )


def _candidate_ids_for_status(
    candidates: tuple[ThroughputOptimizationCandidate, ...],
    status: ThroughputOptimizationStatus,
) -> tuple[str, ...]:
    return tuple(
        candidate.candidate_id
        for candidate in candidates
        if candidate.status == status
    )


def _throughput_score(
    *,
    status: ThroughputOptimizationStatus,
    throughput_units: int,
    capacity_units: int,
    backpressure_units: int,
) -> int:
    if status == "boundary_guardrail":
        return 0
    score = throughput_units * 60 + capacity_units * 75
    if status == "capacity_guardrail":
        score += backpressure_units * 25 + 100
    return min(3_000, score)


def _throughput_pressure(
    *,
    status: ThroughputOptimizationStatus,
    throughput_score: int,
) -> ThroughputOptimizationPressure:
    if status in {"capacity_guardrail", "boundary_guardrail"}:
        return "guarded"
    if throughput_score >= 1_200:
        return "high"
    if throughput_score >= 400:
        return "medium"
    return "low"


def _plan_pressure(
    *,
    candidates: tuple[ThroughputOptimizationCandidate, ...],
    highest_score: int,
) -> ThroughputOptimizationPressure:
    if any(
        candidate.status in {"capacity_guardrail", "boundary_guardrail"}
        for candidate in candidates
    ):
        return "guarded"
    if highest_score >= 1_200:
        return "high"
    if highest_score >= 400:
        return "medium"
    return "low"


def _candidate_actions(
    status: ThroughputOptimizationStatus,
) -> tuple[str, ...]:
    if status == "throughput_candidate":
        return (
            "Expose throughput optimization candidate as advisory metadata only.",
            "Require explicit runtime authority before throughput changes.",
        )
    if status == "capacity_guardrail":
        return (
            "Keep backpressure metadata advisory until capacity policy exists.",
            "Preserve load, routing, workflow, and output boundaries.",
        )
    return (
        "Preserve provider, model, request distribution, and routing boundaries.",
    )


def _plan_actions(
    candidates: tuple[ThroughputOptimizationCandidate, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose throughput optimization posture as advisory metadata only.",
        "Preserve throughput measurement, concurrency, queue, routing, workflow, "
        "and output boundaries.",
    ]
    if _candidate_ids_for_status(candidates, "capacity_guardrail"):
        actions.append("Treat bottleneck backpressure as review metadata only.")
    return tuple(actions)
