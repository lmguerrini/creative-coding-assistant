"""V5.3 advisory reasoning budget optimization planning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .budget_policies import BudgetPolicyPlan, evaluate_budget_policies
from .context_budget_planner import ContextBudgetPlan, plan_context_budget
from .performance_benchmarking import (
    PerformanceBenchmarkingPlan,
    plan_performance_benchmarking,
)
from .performance_prediction import PerformancePredictionPlan, predict_performance

ReasoningBudgetKind = Literal[
    "context_reasoning_allocation",
    "performance_reasoning_reserve",
    "benchmark_reasoning_reserve",
    "budget_policy_review",
]
ReasoningBudgetStatus = Literal[
    "optimization_candidate",
    "reserve_guardrail",
    "review_guardrail",
]
ReasoningBudgetPressure = Literal["low", "medium", "high", "guarded"]

REASONING_BUDGET_RECOMMENDATION_SERIALIZATION_VERSION = (
    "reasoning_budget_recommendation.v1"
)
REASONING_BUDGET_OPTIMIZATION_PLAN_SERIALIZATION_VERSION = (
    "reasoning_budget_optimization_plan.v1"
)
REASONING_BUDGET_OPTIMIZER_AUTHORITY_BOUNDARY = (
    "Reasoning budget optimization planning derives advisory reasoning budget "
    "recommendations from context budget, performance prediction, performance "
    "benchmarking, and budget policy metadata only; it does not enforce "
    "budgets, allocate reasoning tokens at runtime, trim context, compress "
    "prompts, summarize memory, emit HITL requests, select or route providers "
    "or models, control workflows, trigger retries, mutate prompts, write "
    "storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "reasoning_budget_enforcement",
    "runtime_reasoning_token_allocation",
    "budget_enforcement",
    "context_trimming",
    "prompt_compression",
    "memory_summarization",
    "human_input_request_emission",
    "automatic_provider_selection",
    "automatic_model_selection",
    "provider_or_model_routing",
    "workflow_control",
    "workflow_timing_change",
    "workflow_graph_mutation",
    "workflow_execution",
    "agent_invocation",
    "node_handler_invocation",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class ReasoningBudgetRecommendation(BaseModel):
    """One advisory V5.3 reasoning budget recommendation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    recommendation_id: str = Field(min_length=1, max_length=180)
    budget_id: str = Field(min_length=1, max_length=120)
    budget_kind: ReasoningBudgetKind
    status: ReasoningBudgetStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=100)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    advisory_reasoning_tokens: int = Field(ge=0, le=240_000)
    advisory_reserve_tokens: int = Field(ge=0, le=120_000)
    advisory_pressure_score: int = Field(ge=0, le=3_000)
    reasoning_budget_pressure: ReasoningBudgetPressure
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    reasoning_budget_optimizer_planning_implemented: Literal[True] = True
    reasoning_budget_enforcement_implemented: Literal[False] = False
    runtime_reasoning_token_allocation_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    context_trimming_implemented: Literal[False] = False
    prompt_compression_implemented: Literal[False] = False
    memory_summarization_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["reasoning_budget_recommendation.v1"] = (
        REASONING_BUDGET_RECOMMENDATION_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _recommendation_matches_inputs(self) -> Self:
        if self.recommendation_id != f"reasoning_budget::{self.budget_id}":
            raise ValueError("recommendation_id must match budget_id")
        expected_score = _pressure_score(
            status=self.status,
            reasoning_tokens=self.advisory_reasoning_tokens,
            reserve_tokens=self.advisory_reserve_tokens,
        )
        if self.advisory_pressure_score != expected_score:
            raise ValueError("advisory_pressure_score must match recommendation inputs")
        if self.reasoning_budget_pressure != _pressure(
            status=self.status,
            pressure_score=self.advisory_pressure_score,
        ):
            raise ValueError("reasoning_budget_pressure must match recommendation")
        if (
            self.status == "optimization_candidate"
            and self.advisory_reasoning_tokens <= 0
        ):
            raise ValueError("optimization candidates require reasoning tokens")
        if self.status == "reserve_guardrail" and self.advisory_reserve_tokens <= 0:
            raise ValueError("reserve guardrails require reserve tokens")
        if self.status == "review_guardrail" and (
            self.advisory_reasoning_tokens or self.advisory_reserve_tokens
        ):
            raise ValueError("review guardrails must not declare token changes")
        return self


class ReasoningBudgetOptimizationPlan(BaseModel):
    """Bounded V5.3 advisory reasoning budget optimization plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["reasoning_budget_optimizer"] = "reasoning_budget_optimizer"
    serialization_version: Literal["reasoning_budget_optimization_plan.v1"] = (
        REASONING_BUDGET_OPTIMIZATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=REASONING_BUDGET_OPTIMIZER_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    source_context_budget_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_performance_prediction_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_performance_benchmarking_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_budget_policy_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    recommendations: tuple[ReasoningBudgetRecommendation, ...] = Field(
        min_length=1,
        max_length=12,
    )
    recommendation_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    optimization_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    reserve_guardrail_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    review_guardrail_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    recommendation_count: int = Field(ge=1, le=12)
    optimization_candidate_count: int = Field(ge=0, le=12)
    reserve_guardrail_count: int = Field(ge=0, le=12)
    review_guardrail_count: int = Field(ge=0, le=12)
    total_advisory_reasoning_tokens: int = Field(ge=0, le=240_000)
    total_advisory_reserve_tokens: int = Field(ge=0, le=120_000)
    highest_advisory_pressure_score: int = Field(ge=0, le=3_000)
    total_advisory_pressure_score: int = Field(ge=0, le=20_000)
    reasoning_budget_pressure: ReasoningBudgetPressure
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    reasoning_budget_optimizer_planning_implemented: Literal[True] = True
    reasoning_budget_enforcement_implemented: Literal[False] = False
    runtime_reasoning_token_allocation_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    context_trimming_implemented: Literal[False] = False
    prompt_compression_implemented: Literal[False] = False
    memory_summarization_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_recommendations(self) -> Self:
        derived_recommendation_ids = tuple(
            recommendation.recommendation_id for recommendation in self.recommendations
        )
        if len(set(derived_recommendation_ids)) != len(derived_recommendation_ids):
            raise ValueError("recommendation_ids must be unique")
        if self.recommendation_ids != derived_recommendation_ids:
            raise ValueError("recommendation_ids must match recommendations")
        if self.recommendation_count != len(self.recommendations):
            raise ValueError("recommendation_count must match recommendations")
        if self.optimization_candidate_ids != _recommendation_ids_for_status(
            self.recommendations,
            "optimization_candidate",
        ):
            raise ValueError("optimization_candidate_ids must match recommendations")
        if self.reserve_guardrail_ids != _recommendation_ids_for_status(
            self.recommendations,
            "reserve_guardrail",
        ):
            raise ValueError("reserve_guardrail_ids must match recommendations")
        if self.review_guardrail_ids != _recommendation_ids_for_status(
            self.recommendations,
            "review_guardrail",
        ):
            raise ValueError("review_guardrail_ids must match recommendations")
        if self.optimization_candidate_count != len(self.optimization_candidate_ids):
            raise ValueError("optimization_candidate_count must match recommendations")
        if self.reserve_guardrail_count != len(self.reserve_guardrail_ids):
            raise ValueError("reserve_guardrail_count must match recommendations")
        if self.review_guardrail_count != len(self.review_guardrail_ids):
            raise ValueError("review_guardrail_count must match recommendations")

        expected_reasoning = sum(
            recommendation.advisory_reasoning_tokens
            for recommendation in self.recommendations
        )
        if self.total_advisory_reasoning_tokens != expected_reasoning:
            raise ValueError("total_advisory_reasoning_tokens must match")
        expected_reserve = sum(
            recommendation.advisory_reserve_tokens
            for recommendation in self.recommendations
        )
        if self.total_advisory_reserve_tokens != expected_reserve:
            raise ValueError("total_advisory_reserve_tokens must match")
        expected_highest = max(
            recommendation.advisory_pressure_score
            for recommendation in self.recommendations
        )
        if self.highest_advisory_pressure_score != expected_highest:
            raise ValueError("highest_advisory_pressure_score must match")
        expected_total = sum(
            recommendation.advisory_pressure_score
            for recommendation in self.recommendations
        )
        if self.total_advisory_pressure_score != expected_total:
            raise ValueError("total_advisory_pressure_score must match")
        if self.reasoning_budget_pressure != _plan_pressure(
            recommendations=self.recommendations,
            highest_score=self.highest_advisory_pressure_score,
        ):
            raise ValueError("reasoning_budget_pressure must match recommendations")
        return self


def optimize_reasoning_budget(
    *,
    context_budget: ContextBudgetPlan | None = None,
    performance_prediction: PerformancePredictionPlan | None = None,
    performance_benchmarking: PerformanceBenchmarkingPlan | None = None,
    budget_policies: BudgetPolicyPlan | None = None,
) -> ReasoningBudgetOptimizationPlan:
    """Plan advisory reasoning budget optimization without enforcing budgets."""

    context = context_budget or plan_context_budget()
    prediction = performance_prediction or predict_performance()
    benchmarking = performance_benchmarking or plan_performance_benchmarking(
        performance_prediction=prediction
    )
    policies = budget_policies or evaluate_budget_policies()
    recommendations = _recommendations(
        context=context,
        prediction=prediction,
        benchmarking=benchmarking,
        policies=policies,
    )
    highest_score = max(
        recommendation.advisory_pressure_score for recommendation in recommendations
    )

    return ReasoningBudgetOptimizationPlan(
        source_context_budget_serialization_version=context.serialization_version,
        source_performance_prediction_serialization_version=(
            prediction.serialization_version
        ),
        source_performance_benchmarking_serialization_version=(
            benchmarking.serialization_version
        ),
        source_budget_policy_serialization_version=policies.serialization_version,
        recommendations=recommendations,
        recommendation_ids=tuple(
            recommendation.recommendation_id for recommendation in recommendations
        ),
        optimization_candidate_ids=_recommendation_ids_for_status(
            recommendations,
            "optimization_candidate",
        ),
        reserve_guardrail_ids=_recommendation_ids_for_status(
            recommendations,
            "reserve_guardrail",
        ),
        review_guardrail_ids=_recommendation_ids_for_status(
            recommendations,
            "review_guardrail",
        ),
        recommendation_count=len(recommendations),
        optimization_candidate_count=len(
            _recommendation_ids_for_status(recommendations, "optimization_candidate")
        ),
        reserve_guardrail_count=len(
            _recommendation_ids_for_status(recommendations, "reserve_guardrail")
        ),
        review_guardrail_count=len(
            _recommendation_ids_for_status(recommendations, "review_guardrail")
        ),
        total_advisory_reasoning_tokens=sum(
            recommendation.advisory_reasoning_tokens
            for recommendation in recommendations
        ),
        total_advisory_reserve_tokens=sum(
            recommendation.advisory_reserve_tokens for recommendation in recommendations
        ),
        highest_advisory_pressure_score=highest_score,
        total_advisory_pressure_score=sum(
            recommendation.advisory_pressure_score for recommendation in recommendations
        ),
        reasoning_budget_pressure=_plan_pressure(
            recommendations=recommendations,
            highest_score=highest_score,
        ),
        advisory_actions=_plan_actions(recommendations),
    )


def reasoning_budget_recommendation_by_id(
    recommendation_id: str,
    plan: ReasoningBudgetOptimizationPlan | None = None,
) -> ReasoningBudgetRecommendation | None:
    """Return one advisory reasoning budget recommendation without enforcement."""

    source_plan = plan or optimize_reasoning_budget()
    for recommendation in source_plan.recommendations:
        if recommendation.recommendation_id == recommendation_id:
            return recommendation
    return None


def reasoning_budget_recommendations_for_status(
    status: ReasoningBudgetStatus,
    plan: ReasoningBudgetOptimizationPlan | None = None,
) -> tuple[ReasoningBudgetRecommendation, ...]:
    """Return advisory reasoning budget recommendations by status."""

    source_plan = plan or optimize_reasoning_budget()
    return tuple(
        recommendation
        for recommendation in source_plan.recommendations
        if recommendation.status == status
    )


def _recommendations(
    *,
    context: ContextBudgetPlan,
    prediction: PerformancePredictionPlan,
    benchmarking: PerformanceBenchmarkingPlan,
    policies: BudgetPolicyPlan,
) -> tuple[ReasoningBudgetRecommendation, ...]:
    return (
        _recommendation(
            budget_id="context_reasoning_allocation",
            kind="context_reasoning_allocation",
            status="optimization_candidate",
            source_id="context_budget_plan",
            source_serialization_version=context.serialization_version,
            source_item_ids=context.allocation_ids,
            reasoning_tokens=context.allocated_context_tokens,
            reserve_tokens=context.response_reserve_tokens,
            evidence=(
                f"context_pressure:{context.budget_pressure}",
                f"allocated_context:{context.allocated_context_tokens}",
            ),
        ),
        _recommendation(
            budget_id="performance_reasoning_reserve",
            kind="performance_reasoning_reserve",
            status="optimization_candidate",
            source_id="performance_prediction_plan",
            source_serialization_version=prediction.serialization_version,
            source_item_ids=prediction.prediction_ids,
            reasoning_tokens=_performance_reasoning_tokens(prediction),
            reserve_tokens=prediction.guarded_prediction_count * 400,
            evidence=(
                f"recommended_performance:{prediction.recommended_performance_band}",
                f"guarded_predictions:{prediction.guarded_prediction_count}",
            ),
        ),
        _recommendation(
            budget_id="benchmark_reasoning_reserve",
            kind="benchmark_reasoning_reserve",
            status="reserve_guardrail",
            source_id="performance_benchmarking_plan",
            source_serialization_version=benchmarking.serialization_version,
            source_item_ids=benchmarking.scenario_ids,
            reasoning_tokens=0,
            reserve_tokens=max(1, benchmarking.benchmark_candidate_count) * 600,
            evidence=(
                f"benchmarking_readiness:{benchmarking.benchmarking_readiness}",
                f"benchmark_candidates:{benchmarking.benchmark_candidate_count}",
            ),
        ),
        _recommendation(
            budget_id="budget_policy_review",
            kind="budget_policy_review",
            status="review_guardrail",
            source_id="budget_policy_plan",
            source_serialization_version=policies.serialization_version,
            source_item_ids=policies.policy_ids,
            reasoning_tokens=0,
            reserve_tokens=0,
            evidence=(
                f"budget_posture:{policies.recommended_budget_posture}",
                f"review_recommended:{policies.review_recommended_count}",
            ),
        ),
    )


def _recommendation(
    *,
    budget_id: str,
    kind: ReasoningBudgetKind,
    status: ReasoningBudgetStatus,
    source_id: str,
    source_serialization_version: str,
    source_item_ids: tuple[str, ...],
    reasoning_tokens: int,
    reserve_tokens: int,
    evidence: tuple[str, ...],
) -> ReasoningBudgetRecommendation:
    score = _pressure_score(
        status=status,
        reasoning_tokens=reasoning_tokens,
        reserve_tokens=reserve_tokens,
    )
    return ReasoningBudgetRecommendation(
        recommendation_id=f"reasoning_budget::{budget_id}",
        budget_id=budget_id,
        budget_kind=kind,
        status=status,
        source_id=source_id,
        source_serialization_version=source_serialization_version,
        source_item_ids=source_item_ids,
        advisory_reasoning_tokens=reasoning_tokens,
        advisory_reserve_tokens=reserve_tokens,
        advisory_pressure_score=score,
        reasoning_budget_pressure=_pressure(status=status, pressure_score=score),
        evidence=evidence,
        advisory_actions=_recommendation_actions(status),
    )


def _recommendation_ids_for_status(
    recommendations: tuple[ReasoningBudgetRecommendation, ...],
    status: ReasoningBudgetStatus,
) -> tuple[str, ...]:
    return tuple(
        recommendation.recommendation_id
        for recommendation in recommendations
        if recommendation.status == status
    )


def _performance_reasoning_tokens(
    prediction: PerformancePredictionPlan,
) -> int:
    return (
        prediction.guarded_prediction_count * 800
        + prediction.high_prediction_count * 400
        + prediction.prediction_count * 100
    )


def _pressure_score(
    *,
    status: ReasoningBudgetStatus,
    reasoning_tokens: int,
    reserve_tokens: int,
) -> int:
    if status == "review_guardrail":
        return 0
    score = reasoning_tokens // 20 + reserve_tokens // 10
    if status == "reserve_guardrail":
        score += 100
    return min(3_000, score)


def _pressure(
    *,
    status: ReasoningBudgetStatus,
    pressure_score: int,
) -> ReasoningBudgetPressure:
    if status in {"reserve_guardrail", "review_guardrail"}:
        return "guarded"
    if pressure_score >= 900:
        return "high"
    if pressure_score >= 300:
        return "medium"
    return "low"


def _plan_pressure(
    *,
    recommendations: tuple[ReasoningBudgetRecommendation, ...],
    highest_score: int,
) -> ReasoningBudgetPressure:
    if any(
        recommendation.status in {"reserve_guardrail", "review_guardrail"}
        for recommendation in recommendations
    ):
        return "guarded"
    if highest_score >= 900:
        return "high"
    if highest_score >= 300:
        return "medium"
    return "low"


def _recommendation_actions(status: ReasoningBudgetStatus) -> tuple[str, ...]:
    if status == "optimization_candidate":
        return (
            "Expose reasoning budget recommendation as advisory metadata only.",
            "Require explicit runtime authority before applying token allocation.",
        )
    if status == "reserve_guardrail":
        return (
            "Keep benchmark reasoning reserve advisory until execution is approved.",
        )
    return (
        "Preserve budget policy review without enforcement or HITL emission.",
    )


def _plan_actions(
    recommendations: tuple[ReasoningBudgetRecommendation, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose reasoning budget optimization posture as advisory metadata only.",
        "Preserve budget enforcement, token allocation, context, prompt, HITL, "
        "routing, workflow, and output boundaries.",
    ]
    if _recommendation_ids_for_status(recommendations, "review_guardrail"):
        actions.append("Keep budget-policy review guardrails non-blocking.")
    return tuple(actions)
