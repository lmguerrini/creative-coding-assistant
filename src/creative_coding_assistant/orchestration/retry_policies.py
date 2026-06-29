"""V5.3 advisory retry policy planning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .execution_cost_forecasting import ExecutionCostForecast, forecast_execution_cost
from .execution_graph_analyzer import (
    ExecutionGraphAnalysis,
    analyze_assistant_execution_graph,
)
from .streaming_optimizer import StreamingOptimizationPlan, optimize_streaming
from .workflow_pruning import WorkflowPruningPlan, plan_workflow_pruning
from .workflow_review import MAX_WORKFLOW_REFINEMENT_COUNT

RetryPolicyKind = Literal[
    "review_refinement",
    "generation_failure",
    "stream_failure_visibility",
    "cost_retry_reserve",
]
RetryPolicyStatus = Literal["bounded_retry_candidate", "guardrail", "review_only"]
RetryPolicyPressure = Literal["low", "medium", "high"]

RETRY_POLICY_CANDIDATE_SERIALIZATION_VERSION = "retry_policy_candidate.v1"
RETRY_POLICY_PLAN_SERIALIZATION_VERSION = "retry_policy_plan.v1"
RETRY_POLICY_AUTHORITY_BOUNDARY = (
    "Retry policy planning derives advisory retry candidates from static "
    "workflow graph, pruning, cost forecast, review limit, and streaming "
    "visibility metadata only; it does not trigger retries, trigger "
    "refinement, control workflow execution, mutate graph order, compile or "
    "execute workflow graphs, invoke agents or node handlers, route providers "
    "or models, mutate prompts, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "retry_triggering",
    "refinement_triggering",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_order_change",
    "langgraph_compilation",
    "workflow_execution",
    "agent_invocation",
    "node_handler_invocation",
    "provider_or_model_routing",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class RetryPolicyCandidate(BaseModel):
    """One advisory V5.3 retry policy candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    policy_id: str = Field(min_length=1, max_length=120)
    policy_kind: RetryPolicyKind
    status: RetryPolicyStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=100)
    max_retry_attempts: int = Field(ge=0, le=5)
    retry_budget_tokens: int = Field(ge=0, le=240_000)
    advisory_retry_score: int = Field(ge=0, le=800)
    failure_path_required: Literal[True] = True
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    retry_policy_planning_implemented: Literal[True] = True
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_order_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["retry_policy_candidate.v1"] = (
        RETRY_POLICY_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_policy(self) -> Self:
        if self.candidate_id != f"retry_policy::{self.policy_id}":
            raise ValueError("candidate_id must match policy_id")
        expected_score = _retry_score(
            status=self.status,
            max_attempts=self.max_retry_attempts,
        )
        if self.advisory_retry_score != expected_score:
            raise ValueError("advisory_retry_score must match retry policy")
        if self.status == "bounded_retry_candidate" and self.max_retry_attempts <= 0:
            raise ValueError("bounded retry candidates require retry attempts")
        if self.status == "guardrail" and self.max_retry_attempts != 0:
            raise ValueError("guardrails must not declare retry attempts")
        return self


class RetryPolicyPlan(BaseModel):
    """Bounded V5.3 advisory retry policy plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["retry_policy_planner"] = "retry_policy_planner"
    serialization_version: Literal["retry_policy_plan.v1"] = (
        RETRY_POLICY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RETRY_POLICY_AUTHORITY_BOUNDARY,
        max_length=1400,
    )
    source_graph_serialization_version: str = Field(min_length=1, max_length=100)
    source_pruning_serialization_version: str = Field(min_length=1, max_length=100)
    source_cost_forecast_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_streaming_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    workflow_refinement_limit: int = Field(ge=0, le=5)
    bounded_retry_cycle_detected: bool
    failure_path_reachable: bool
    candidates: tuple[RetryPolicyCandidate, ...] = Field(
        min_length=1,
        max_length=12,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    bounded_retry_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    review_only_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    candidate_count: int = Field(ge=1, le=12)
    bounded_retry_candidate_count: int = Field(ge=0, le=12)
    guardrail_candidate_count: int = Field(ge=0, le=12)
    review_only_candidate_count: int = Field(ge=0, le=12)
    max_retry_attempts: int = Field(ge=0, le=5)
    total_retry_budget_tokens: int = Field(ge=0, le=240_000)
    highest_advisory_retry_score: int = Field(ge=0, le=800)
    retry_policy_pressure: RetryPolicyPressure
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    retry_policy_planning_implemented: Literal[True] = True
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_order_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
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
        if self.bounded_retry_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "bounded_retry_candidate",
        ):
            raise ValueError("bounded_retry_candidate_ids must match candidates")
        if self.guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "guardrail",
        ):
            raise ValueError("guardrail_candidate_ids must match candidates")
        if self.review_only_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "review_only",
        ):
            raise ValueError("review_only_candidate_ids must match candidates")
        if self.bounded_retry_candidate_count != len(
            self.bounded_retry_candidate_ids
        ):
            raise ValueError("bounded_retry_candidate_count must match candidates")
        if self.guardrail_candidate_count != len(self.guardrail_candidate_ids):
            raise ValueError("guardrail_candidate_count must match candidates")
        if self.review_only_candidate_count != len(self.review_only_candidate_ids):
            raise ValueError("review_only_candidate_count must match candidates")

        expected_attempts = max(
            candidate.max_retry_attempts for candidate in self.candidates
        )
        if self.max_retry_attempts != expected_attempts:
            raise ValueError("max_retry_attempts must match candidates")
        expected_budget = sum(
            candidate.retry_budget_tokens for candidate in self.candidates
        )
        if self.total_retry_budget_tokens != expected_budget:
            raise ValueError("total_retry_budget_tokens must match candidates")
        expected_score = max(
            candidate.advisory_retry_score for candidate in self.candidates
        )
        if self.highest_advisory_retry_score != expected_score:
            raise ValueError("highest_advisory_retry_score must match candidates")
        if self.retry_policy_pressure != _retry_pressure(
            max_attempts=self.max_retry_attempts,
            highest_score=self.highest_advisory_retry_score,
        ):
            raise ValueError("retry_policy_pressure must match candidates")
        return self


def plan_retry_policies(
    *,
    execution_graph: ExecutionGraphAnalysis | None = None,
    pruning_plan: WorkflowPruningPlan | None = None,
    cost_forecast: ExecutionCostForecast | None = None,
    streaming_optimization: StreamingOptimizationPlan | None = None,
) -> RetryPolicyPlan:
    """Plan advisory retry policies without triggering retries."""

    graph = execution_graph or analyze_assistant_execution_graph()
    pruning = pruning_plan or plan_workflow_pruning(execution_graph=graph)
    forecast = cost_forecast or forecast_execution_cost(pruning_plan=pruning)
    streaming = streaming_optimization or optimize_streaming()
    retry_budget_tokens = max(
        0,
        forecast.single_retry_token_forecast - forecast.minimum_token_forecast,
    )
    candidates = _candidates(
        graph=graph,
        pruning=pruning,
        forecast=forecast,
        streaming=streaming,
        retry_budget_tokens=retry_budget_tokens,
    )
    max_attempts = max(candidate.max_retry_attempts for candidate in candidates)
    highest_score = max(candidate.advisory_retry_score for candidate in candidates)

    return RetryPolicyPlan(
        source_graph_serialization_version=graph.serialization_version,
        source_pruning_serialization_version=pruning.serialization_version,
        source_cost_forecast_serialization_version=forecast.serialization_version,
        source_streaming_optimization_serialization_version=(
            streaming.serialization_version
        ),
        workflow_refinement_limit=MAX_WORKFLOW_REFINEMENT_COUNT,
        bounded_retry_cycle_detected=graph.bounded_retry_cycle_detected,
        failure_path_reachable=graph.failure_path_reachable,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        bounded_retry_candidate_ids=_candidate_ids_for_status(
            candidates,
            "bounded_retry_candidate",
        ),
        guardrail_candidate_ids=_candidate_ids_for_status(candidates, "guardrail"),
        review_only_candidate_ids=_candidate_ids_for_status(candidates, "review_only"),
        candidate_count=len(candidates),
        bounded_retry_candidate_count=len(
            _candidate_ids_for_status(candidates, "bounded_retry_candidate")
        ),
        guardrail_candidate_count=len(
            _candidate_ids_for_status(candidates, "guardrail")
        ),
        review_only_candidate_count=len(
            _candidate_ids_for_status(candidates, "review_only")
        ),
        max_retry_attempts=max_attempts,
        total_retry_budget_tokens=sum(
            candidate.retry_budget_tokens for candidate in candidates
        ),
        highest_advisory_retry_score=highest_score,
        retry_policy_pressure=_retry_pressure(
            max_attempts=max_attempts,
            highest_score=highest_score,
        ),
        advisory_actions=_plan_actions(max_attempts),
    )


def retry_policy_candidate_by_id(
    candidate_id: str,
    plan: RetryPolicyPlan | None = None,
) -> RetryPolicyCandidate | None:
    """Return one advisory retry policy without triggering retries."""

    source_plan = plan or plan_retry_policies()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def retry_policy_candidates_for_status(
    status: RetryPolicyStatus,
    plan: RetryPolicyPlan | None = None,
) -> tuple[RetryPolicyCandidate, ...]:
    """Return retry policy candidates by status without workflow control."""

    source_plan = plan or plan_retry_policies()
    return tuple(
        candidate
        for candidate in source_plan.candidates
        if candidate.status == status
    )


def _candidates(
    *,
    graph: ExecutionGraphAnalysis,
    pruning: WorkflowPruningPlan,
    forecast: ExecutionCostForecast,
    streaming: StreamingOptimizationPlan,
    retry_budget_tokens: int,
) -> tuple[RetryPolicyCandidate, ...]:
    return (
        _candidate(
            policy_id="review_refinement",
            kind="review_refinement",
            status="bounded_retry_candidate",
            source_id="workflow_review",
            source_serialization_version="workflow_review.v1",
            max_retry_attempts=MAX_WORKFLOW_REFINEMENT_COUNT,
            retry_budget_tokens=retry_budget_tokens,
            evidence=(
                f"refinement_limit:{MAX_WORKFLOW_REFINEMENT_COUNT}",
                f"retry_cycle:{graph.bounded_retry_cycle_detected}",
            ),
        ),
        _candidate(
            policy_id="generation_failure",
            kind="generation_failure",
            status="guardrail",
            source_id="execution_graph_analysis",
            source_serialization_version=graph.serialization_version,
            max_retry_attempts=0,
            retry_budget_tokens=0,
            evidence=(
                f"failure_path_reachable:{graph.failure_path_reachable}",
                f"failure_entries:{len(graph.failure_entry_node_ids)}",
            ),
        ),
        _candidate(
            policy_id="stream_failure_visibility",
            kind="stream_failure_visibility",
            status="guardrail",
            source_id="streaming_optimization_plan",
            source_serialization_version=streaming.serialization_version,
            max_retry_attempts=0,
            retry_budget_tokens=0,
            evidence=(
                "stream_phase:review_retry_visibility",
                "stream_phase:terminal_integrity",
            ),
        ),
        _candidate(
            policy_id="cost_retry_reserve",
            kind="cost_retry_reserve",
            status="review_only",
            source_id="execution_cost_forecast",
            source_serialization_version=forecast.serialization_version,
            max_retry_attempts=MAX_WORKFLOW_REFINEMENT_COUNT,
            retry_budget_tokens=retry_budget_tokens,
            evidence=(
                f"single_retry_tokens:{forecast.single_retry_token_forecast}",
                f"pruning_candidates:{pruning.candidate_count}",
            ),
        ),
    )


def _candidate(
    *,
    policy_id: str,
    kind: RetryPolicyKind,
    status: RetryPolicyStatus,
    source_id: str,
    source_serialization_version: str,
    max_retry_attempts: int,
    retry_budget_tokens: int,
    evidence: tuple[str, ...],
) -> RetryPolicyCandidate:
    return RetryPolicyCandidate(
        candidate_id=f"retry_policy::{policy_id}",
        policy_id=policy_id,
        policy_kind=kind,
        status=status,
        source_id=source_id,
        source_serialization_version=source_serialization_version,
        max_retry_attempts=max_retry_attempts,
        retry_budget_tokens=retry_budget_tokens,
        advisory_retry_score=_retry_score(
            status=status,
            max_attempts=max_retry_attempts,
        ),
        evidence=evidence,
        advisory_actions=_candidate_actions(status),
    )


def _candidate_ids_for_status(
    candidates: tuple[RetryPolicyCandidate, ...],
    status: RetryPolicyStatus,
) -> tuple[str, ...]:
    return tuple(
        candidate.candidate_id
        for candidate in candidates
        if candidate.status == status
    )


def _retry_score(
    *,
    status: RetryPolicyStatus,
    max_attempts: int,
) -> int:
    if status == "bounded_retry_candidate":
        return max_attempts * 200
    if status == "review_only":
        return max_attempts * 100
    return 0


def _retry_pressure(
    *,
    max_attempts: int,
    highest_score: int,
) -> RetryPolicyPressure:
    if max_attempts >= 2 or highest_score >= 400:
        return "high"
    if max_attempts == 1 or highest_score >= 200:
        return "medium"
    return "low"


def _candidate_actions(status: RetryPolicyStatus) -> tuple[str, ...]:
    if status == "bounded_retry_candidate":
        return (
            "Expose bounded retry policy as advisory metadata only.",
            "Require explicit workflow authority before retry triggering.",
        )
    if status == "review_only":
        return (
            "Expose retry reserve for review without applying retry behavior.",
        )
    return (
        "Retain retry guardrail metadata without retry activation.",
    )


def _plan_actions(max_attempts: int) -> tuple[str, ...]:
    actions = [
        "Expose retry policies as advisory metadata only.",
        "Preserve retry, refinement, workflow control, routing, and output boundaries.",
    ]
    if max_attempts:
        actions.append("Report bounded retry attempts without triggering retries.")
    return tuple(actions)
