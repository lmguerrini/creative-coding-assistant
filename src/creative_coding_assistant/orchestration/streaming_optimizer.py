"""V5.3 advisory streaming optimization planning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.contracts import StreamEventType

from .async_execution import AsyncExecutionPlan, plan_async_execution

StreamingOptimizationStatus = Literal[
    "optimization_candidate",
    "contract_guardrail",
]
StreamingOptimizationFocus = Literal[
    "lifecycle_ordering",
    "generation_token_flow",
    "review_retry_visibility",
    "artifact_preview_visibility",
    "terminal_integrity",
]
StreamingOptimizationPressure = Literal["low", "medium", "high"]

STREAMING_OPTIMIZATION_CANDIDATE_SERIALIZATION_VERSION = (
    "streaming_optimization_candidate.v1"
)
STREAMING_OPTIMIZATION_PLAN_SERIALIZATION_VERSION = (
    "streaming_optimization_plan.v1"
)
STREAMING_OPTIMIZER_AUTHORITY_BOUNDARY = (
    "Streaming optimization derives advisory stream phase candidates from the "
    "static stream event contract and async execution readiness metadata only; "
    "it does not emit events, buffer token deltas, batch chunks, reorder stream "
    "events, mutate event payloads, alter workflow timing, compile or execute "
    "workflow graphs, invoke agents or node handlers, route providers or "
    "models, trigger retries, mutate prompts, write storage, or modify "
    "generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "stream_event_emission_change",
    "stream_sequence_mutation",
    "token_delta_buffering",
    "chunk_batching_runtime",
    "stream_payload_mutation",
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


class StreamingOptimizationCandidate(BaseModel):
    """One advisory V5.3 streaming optimization candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    phase_id: str = Field(min_length=1, max_length=80)
    status: StreamingOptimizationStatus
    optimization_focus: StreamingOptimizationFocus
    stream_event_types: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_async_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=20,
    )
    event_order_required: Literal[True] = True
    sequence_monotonic_required: Literal[True] = True
    advisory_batch_window_ms: int = Field(ge=0, le=500)
    advisory_stream_readiness_score: int = Field(ge=0, le=500)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    streaming_optimizer_implemented: Literal[True] = True
    stream_event_emission_change_implemented: Literal[False] = False
    stream_sequence_mutation_implemented: Literal[False] = False
    token_delta_buffering_implemented: Literal[False] = False
    chunk_batching_runtime_implemented: Literal[False] = False
    stream_payload_mutation_implemented: Literal[False] = False
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
    serialization_version: Literal["streaming_optimization_candidate.v1"] = (
        STREAMING_OPTIMIZATION_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_phase(self) -> Self:
        if self.candidate_id != f"streaming_optimizer::{self.phase_id}":
            raise ValueError("candidate_id must match phase_id")
        expected_score = _stream_readiness_score(
            status=self.status,
            event_count=len(self.stream_event_types),
        )
        if self.advisory_stream_readiness_score != expected_score:
            raise ValueError(
                "advisory_stream_readiness_score must match candidate status"
            )
        if self.status == "contract_guardrail" and self.advisory_batch_window_ms:
            raise ValueError("contract guardrails must not declare a batch window")
        if self.status == "optimization_candidate" and (
            self.advisory_batch_window_ms <= 0
        ):
            raise ValueError("optimization candidates require advisory batch window")
        return self


class StreamingOptimizationPlan(BaseModel):
    """Bounded V5.3 advisory streaming optimization plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["streaming_optimizer"] = "streaming_optimizer"
    serialization_version: Literal["streaming_optimization_plan.v1"] = (
        STREAMING_OPTIMIZATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=STREAMING_OPTIMIZER_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    source_async_execution_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_stream_event_contract: Literal["StreamEventType"] = "StreamEventType"
    candidates: tuple[StreamingOptimizationCandidate, ...] = Field(
        min_length=1,
        max_length=12,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    optimization_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    contract_guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    stream_event_types: tuple[str, ...] = Field(min_length=1, max_length=32)
    candidate_count: int = Field(ge=1, le=12)
    optimization_candidate_count: int = Field(ge=0, le=12)
    contract_guardrail_count: int = Field(ge=0, le=12)
    stream_event_type_count: int = Field(ge=1, le=32)
    highest_advisory_stream_readiness_score: int = Field(ge=0, le=500)
    total_advisory_stream_readiness_score: int = Field(ge=0, le=5000)
    streaming_optimization_pressure: StreamingOptimizationPressure
    stream_contract_preserved: Literal[True] = True
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    streaming_optimizer_implemented: Literal[True] = True
    stream_event_emission_change_implemented: Literal[False] = False
    stream_sequence_mutation_implemented: Literal[False] = False
    token_delta_buffering_implemented: Literal[False] = False
    chunk_batching_runtime_implemented: Literal[False] = False
    stream_payload_mutation_implemented: Literal[False] = False
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
        if self.contract_guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "contract_guardrail",
        ):
            raise ValueError("contract_guardrail_candidate_ids must match candidates")
        if self.optimization_candidate_count != len(self.optimization_candidate_ids):
            raise ValueError("optimization_candidate_count must match candidates")
        if self.contract_guardrail_count != len(self.contract_guardrail_candidate_ids):
            raise ValueError("contract_guardrail_count must match candidates")

        event_types = _unique_event_types(self.candidates)
        if self.stream_event_types != event_types:
            raise ValueError("stream_event_types must match candidates")
        if self.stream_event_type_count != len(event_types):
            raise ValueError("stream_event_type_count must match stream_event_types")
        expected_highest = max(
            candidate.advisory_stream_readiness_score
            for candidate in self.candidates
        )
        if self.highest_advisory_stream_readiness_score != expected_highest:
            raise ValueError(
                "highest_advisory_stream_readiness_score must match candidates"
            )
        expected_total = sum(
            candidate.advisory_stream_readiness_score
            for candidate in self.candidates
        )
        if self.total_advisory_stream_readiness_score != expected_total:
            raise ValueError(
                "total_advisory_stream_readiness_score must match candidates"
            )
        if self.streaming_optimization_pressure != _streaming_pressure(
            highest_score=self.highest_advisory_stream_readiness_score,
            total_score=self.total_advisory_stream_readiness_score,
        ):
            raise ValueError("streaming_optimization_pressure must match candidates")
        return self


def optimize_streaming(
    *,
    async_execution: AsyncExecutionPlan | None = None,
) -> StreamingOptimizationPlan:
    """Plan advisory streaming optimization without changing stream runtime."""

    async_plan = async_execution or plan_async_execution()
    candidates = _candidates(async_plan)
    highest_score = max(
        candidate.advisory_stream_readiness_score for candidate in candidates
    )
    total_score = sum(
        candidate.advisory_stream_readiness_score for candidate in candidates
    )

    return StreamingOptimizationPlan(
        source_async_execution_serialization_version=async_plan.serialization_version,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        optimization_candidate_ids=_candidate_ids_for_status(
            candidates,
            "optimization_candidate",
        ),
        contract_guardrail_candidate_ids=_candidate_ids_for_status(
            candidates,
            "contract_guardrail",
        ),
        stream_event_types=_unique_event_types(candidates),
        candidate_count=len(candidates),
        optimization_candidate_count=len(
            _candidate_ids_for_status(candidates, "optimization_candidate")
        ),
        contract_guardrail_count=len(
            _candidate_ids_for_status(candidates, "contract_guardrail")
        ),
        stream_event_type_count=len(_unique_event_types(candidates)),
        highest_advisory_stream_readiness_score=highest_score,
        total_advisory_stream_readiness_score=total_score,
        streaming_optimization_pressure=_streaming_pressure(
            highest_score=highest_score,
            total_score=total_score,
        ),
        advisory_actions=_plan_actions(total_score),
    )


def streaming_optimization_candidate_by_id(
    candidate_id: str,
    plan: StreamingOptimizationPlan | None = None,
) -> StreamingOptimizationCandidate | None:
    """Return one advisory streaming candidate without changing streaming."""

    source_plan = plan or optimize_streaming()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def streaming_optimization_candidates_for_status(
    status: StreamingOptimizationStatus,
    plan: StreamingOptimizationPlan | None = None,
) -> tuple[StreamingOptimizationCandidate, ...]:
    """Return streaming candidates by status without workflow control."""

    source_plan = plan or optimize_streaming()
    return tuple(
        candidate for candidate in source_plan.candidates if candidate.status == status
    )


def _candidates(
    async_plan: AsyncExecutionPlan,
) -> tuple[StreamingOptimizationCandidate, ...]:
    async_ready = async_plan.async_ready_candidate_ids
    return (
        _candidate(
            phase_id="workflow_lifecycle",
            status="contract_guardrail",
            focus="lifecycle_ordering",
            stream_event_types=(
                StreamEventType.STATUS.value,
                StreamEventType.NODE_STARTED.value,
                StreamEventType.NODE_COMPLETED.value,
            ),
            source_async_candidate_ids=(),
            advisory_batch_window_ms=0,
        ),
        _candidate(
            phase_id="generation_token_flow",
            status="optimization_candidate",
            focus="generation_token_flow",
            stream_event_types=(
                StreamEventType.GENERATION_INPUT.value,
                StreamEventType.TOKEN_DELTA.value,
            ),
            source_async_candidate_ids=async_ready,
            advisory_batch_window_ms=50,
        ),
        _candidate(
            phase_id="review_retry_visibility",
            status="contract_guardrail",
            focus="review_retry_visibility",
            stream_event_types=(
                StreamEventType.REVIEW_FAILED.value,
                StreamEventType.REFINEMENT_REQUESTED.value,
                StreamEventType.RETRY_STARTED.value,
                StreamEventType.RETRY_COMPLETED.value,
            ),
            source_async_candidate_ids=(),
            advisory_batch_window_ms=0,
        ),
        _candidate(
            phase_id="artifact_preview_visibility",
            status="optimization_candidate",
            focus="artifact_preview_visibility",
            stream_event_types=(
                StreamEventType.ARTIFACT_EXTRACTED.value,
                StreamEventType.PREVIEW_ARTIFACT.value,
                StreamEventType.ARTIFACT_CRITIQUE.value,
            ),
            source_async_candidate_ids=async_ready,
            advisory_batch_window_ms=75,
        ),
        _candidate(
            phase_id="terminal_integrity",
            status="contract_guardrail",
            focus="terminal_integrity",
            stream_event_types=(
                StreamEventType.NODE_FAILED.value,
                StreamEventType.ERROR.value,
                StreamEventType.FINAL.value,
            ),
            source_async_candidate_ids=(),
            advisory_batch_window_ms=0,
        ),
    )


def _candidate(
    *,
    phase_id: str,
    status: StreamingOptimizationStatus,
    focus: StreamingOptimizationFocus,
    stream_event_types: tuple[str, ...],
    source_async_candidate_ids: tuple[str, ...],
    advisory_batch_window_ms: int,
) -> StreamingOptimizationCandidate:
    return StreamingOptimizationCandidate(
        candidate_id=f"streaming_optimizer::{phase_id}",
        phase_id=phase_id,
        status=status,
        optimization_focus=focus,
        stream_event_types=stream_event_types,
        source_async_candidate_ids=source_async_candidate_ids,
        advisory_batch_window_ms=advisory_batch_window_ms,
        advisory_stream_readiness_score=_stream_readiness_score(
            status=status,
            event_count=len(stream_event_types),
        ),
        evidence=(
            f"phase:{phase_id}",
            f"events:{len(stream_event_types)}",
            f"async_candidates:{len(source_async_candidate_ids)}",
        ),
        advisory_actions=_candidate_actions(status),
    )


def _candidate_ids_for_status(
    candidates: tuple[StreamingOptimizationCandidate, ...],
    status: StreamingOptimizationStatus,
) -> tuple[str, ...]:
    return tuple(
        candidate.candidate_id
        for candidate in candidates
        if candidate.status == status
    )


def _unique_event_types(
    candidates: tuple[StreamingOptimizationCandidate, ...],
) -> tuple[str, ...]:
    seen: list[str] = []
    for candidate in candidates:
        for event_type in candidate.stream_event_types:
            if event_type not in seen:
                seen.append(event_type)
    return tuple(seen)


def _stream_readiness_score(
    *,
    status: StreamingOptimizationStatus,
    event_count: int,
) -> int:
    if status == "optimization_candidate":
        return event_count * 100
    return 0


def _streaming_pressure(
    *,
    highest_score: int,
    total_score: int,
) -> StreamingOptimizationPressure:
    if highest_score >= 300 or total_score >= 500:
        return "high"
    if highest_score >= 200 or total_score >= 200:
        return "medium"
    return "low"


def _candidate_actions(
    status: StreamingOptimizationStatus,
) -> tuple[str, ...]:
    if status == "optimization_candidate":
        return (
            "Expose stream optimization candidate as advisory metadata only.",
            "Require explicit runtime authority before stream behavior changes.",
        )
    return (
        "Retain stream contract guardrail as advisory metadata only.",
        "Preserve event ordering and payload integrity.",
    )


def _plan_actions(total_score: int) -> tuple[str, ...]:
    actions = [
        "Expose streaming optimization posture as advisory metadata only.",
        "Preserve stream event emission, ordering, payload, timing, and output "
        "boundaries.",
    ]
    if total_score:
        actions.append("Use stream readiness score only for later performance review.")
    return tuple(actions)
