"""V5.5 advisory reflection budget optimization intelligence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.reasoning_budget_optimizer import (
    ReasoningBudgetOptimizationPlan,
    ReasoningBudgetRecommendation,
    ReasoningBudgetStatus,
    optimize_reasoning_budget,
    reasoning_budget_recommendation_by_id,
)
from creative_coding_assistant.orchestration.reflection_loop_engine import (
    HitlRecommendation,
    ReflectionDepth,
    ReflectionEstimate,
    ReflectionLoopProfile,
    ReflectionPriority,
)
from creative_coding_assistant.orchestration.workflow_risk_engine import (
    WorkflowRiskFactor,
    WorkflowRiskPlan,
    evaluate_workflow_risk,
    workflow_risk_factor_by_id,
)

ReflectionBudgetKind = Literal[
    "loop_depth_budget",
    "quality_gain_budget",
    "risk_reduction_budget",
    "policy_review_budget",
]
ReflectionBudgetStatus = Literal[
    "recommended",
    "reserve_guardrail",
    "review_guardrail",
]

REFLECTION_BUDGET_CANDIDATE_SERIALIZATION_VERSION = (
    "reflection_budget_optimization_candidate.v1"
)
REFLECTION_BUDGET_PLAN_SERIALIZATION_VERSION = (
    "reflection_budget_optimization_plan.v1"
)
REFLECTION_BUDGET_OPTIMIZER_AUTHORITY_BOUNDARY = (
    "V5.5 reflection budget optimization combines advisory reasoning budget, "
    "reflection loop, and workflow risk metadata into inspectable reflection "
    "budget recommendations only; it does not enforce reflection budgets, "
    "allocate reflection or reasoning tokens at runtime, execute reflection "
    "loops, trigger refinement or retries, trim context, compress prompts, "
    "summarize memory, emit HITL requests, route providers or models, invoke "
    "agents, control or execute workflows, mutate workflow graphs, mutate "
    "prompts, write storage, modify generated output, or apply Runtime "
    "Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "reflection_budget_enforcement",
    "runtime_reflection_token_allocation",
    "runtime_reasoning_token_allocation",
    "reflection_loop_execution",
    "refinement_triggering",
    "retry_or_refinement_triggering",
    "budget_enforcement",
    "context_trimming",
    "prompt_compression",
    "memory_summarization",
    "human_review_request",
    "hitl_request_emission",
    "provider_or_model_routing",
    "automatic_provider_selection",
    "automatic_model_selection",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "agent_invocation",
    "node_handler_invocation",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class ReflectionBudgetOptimizationCandidate(BaseModel):
    """One advisory reflection budget optimization candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    budget_kind: ReflectionBudgetKind
    status: ReflectionBudgetStatus
    source_reasoning_recommendation_id: str = Field(min_length=1, max_length=180)
    source_reasoning_budget_kind: str = Field(min_length=1, max_length=80)
    source_reasoning_budget_status: ReasoningBudgetStatus
    source_workflow_risk_factor_id: str = Field(min_length=1, max_length=180)
    source_workflow_risk_severity: str = Field(min_length=1, max_length=80)
    reflection_priority: ReflectionPriority
    reflection_depth: ReflectionDepth
    reflection_hitl_recommendation: HitlRecommendation
    expected_quality_gain: ReflectionEstimate
    expected_risk_reduction: ReflectionEstimate
    expected_cost: ReflectionEstimate
    expected_latency: ReflectionEstimate
    source_reasoning_tokens: int = Field(ge=0, le=240_000)
    source_reserve_tokens: int = Field(ge=0, le=120_000)
    source_reasoning_pressure_score: int = Field(ge=0, le=3_000)
    source_workflow_risk_score: int = Field(ge=0, le=2_000)
    advisory_reflection_tokens: int = Field(ge=0, le=120_000)
    advisory_reflection_reserve_tokens: int = Field(ge=0, le=120_000)
    depth_weight: int = Field(ge=0, le=140)
    quality_gain_weight: int = Field(ge=0, le=100)
    risk_reduction_weight: int = Field(ge=0, le=100)
    workflow_risk_weight: int = Field(ge=0, le=250)
    guardrail_penalty: int = Field(ge=0, le=260)
    reflection_budget_score: int = Field(ge=0, le=600)
    recommended_reflection_pass_count: int = Field(ge=0, le=4)
    applied_reflection_pass_count: int = Field(ge=0, le=0)
    hitl_required: bool
    budget_summary: str = Field(min_length=1, max_length=360)
    fallback_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    reflection_budget_optimizer_implemented: Literal[True] = True
    reflection_budget_metadata_implemented: Literal[True] = True
    reasoning_budget_metadata_used: Literal[True] = True
    reflection_loop_metadata_used: Literal[True] = True
    workflow_risk_metadata_used: Literal[True] = True
    hitl_posture_metadata_used: Literal[True] = True
    reflection_budget_enforcement_implemented: Literal[False] = False
    runtime_reflection_token_allocation_implemented: Literal[False] = False
    runtime_reasoning_token_allocation_implemented: Literal[False] = False
    reflection_loop_execution_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    context_trimming_implemented: Literal[False] = False
    prompt_compression_implemented: Literal[False] = False
    memory_summarization_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["reflection_budget_optimization_candidate.v1"] = (
        REFLECTION_BUDGET_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_contract(self) -> Self:
        if self.candidate_id != f"reflection_budget_optimizer::{self.budget_kind}":
            raise ValueError("candidate_id must match budget_kind")
        if self.status != _status_from_reasoning_status(
            self.source_reasoning_budget_status
        ):
            raise ValueError("status must match source reasoning status")
        if self.advisory_reflection_tokens != _advisory_reflection_tokens(
            status=self.status,
            source_reasoning_tokens=self.source_reasoning_tokens,
            depth_weight=self.depth_weight,
        ):
            raise ValueError("advisory_reflection_tokens must match source budget")
        if self.advisory_reflection_reserve_tokens != (
            _advisory_reflection_reserve_tokens(
                status=self.status,
                source_reserve_tokens=self.source_reserve_tokens,
                depth_weight=self.depth_weight,
            )
        ):
            raise ValueError(
                "advisory_reflection_reserve_tokens must match source budget"
            )
        if self.depth_weight != _depth_weight(self.reflection_depth):
            raise ValueError("depth_weight must match reflection depth")
        if self.quality_gain_weight != _estimate_weight(self.expected_quality_gain):
            raise ValueError("quality_gain_weight must match expected quality gain")
        if self.risk_reduction_weight != _estimate_weight(
            self.expected_risk_reduction
        ):
            raise ValueError(
                "risk_reduction_weight must match expected risk reduction"
            )
        if self.workflow_risk_weight != _workflow_risk_weight(
            self.source_workflow_risk_score
        ):
            raise ValueError("workflow_risk_weight must match workflow risk score")
        if self.guardrail_penalty != _guardrail_penalty(
            status=self.status,
            source_reasoning_pressure=self.source_reasoning_budget_status,
            workflow_risk_severity=self.source_workflow_risk_severity,
            hitl_recommendation=self.reflection_hitl_recommendation,
        ):
            raise ValueError("guardrail_penalty must match source guardrails")
        if self.reflection_budget_score != _reflection_budget_score(
            reasoning_pressure_score=self.source_reasoning_pressure_score,
            workflow_risk_weight=self.workflow_risk_weight,
            depth_weight=self.depth_weight,
            quality_gain_weight=self.quality_gain_weight,
            risk_reduction_weight=self.risk_reduction_weight,
            guardrail_penalty=self.guardrail_penalty,
        ):
            raise ValueError("reflection_budget_score must combine source scores")
        if self.status != "recommended" and self.recommended_reflection_pass_count:
            raise ValueError("guardrail reflection budget must recommend no passes")
        if self.applied_reflection_pass_count:
            raise ValueError("applied_reflection_pass_count must remain zero")
        return self


class ReflectionBudgetOptimizationPlan(BaseModel):
    """Bounded V5.5 advisory reflection budget optimization plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["reflection_budget_optimizer"] = "reflection_budget_optimizer"
    serialization_version: Literal["reflection_budget_optimization_plan.v1"] = (
        REFLECTION_BUDGET_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=REFLECTION_BUDGET_OPTIMIZER_AUTHORITY_BOUNDARY,
        max_length=2100,
    )
    source_reasoning_budget_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_reflection_loop_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_workflow_risk_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    reflection_priority: ReflectionPriority
    reflection_depth: ReflectionDepth
    reflection_hitl_recommendation: HitlRecommendation
    candidates: tuple[ReflectionBudgetOptimizationCandidate, ...] = Field(
        min_length=1,
        max_length=8,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    recommended_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    reserve_guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    review_guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    hitl_required_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    applied_reflection_budget_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    candidate_count: int = Field(ge=1, le=8)
    recommended_candidate_count: int = Field(ge=0, le=8)
    reserve_guardrail_candidate_count: int = Field(ge=0, le=8)
    review_guardrail_candidate_count: int = Field(ge=0, le=8)
    hitl_required_candidate_count: int = Field(ge=0, le=8)
    total_advisory_reflection_tokens: int = Field(ge=0, le=240_000)
    total_advisory_reflection_reserve_tokens: int = Field(ge=0, le=240_000)
    total_recommended_reflection_pass_count: int = Field(ge=0, le=16)
    total_applied_reflection_pass_count: int = Field(ge=0, le=0)
    highest_reflection_budget_score: int = Field(ge=0, le=600)
    reflection_budget_pressure: Literal["low", "medium", "high", "guarded"]
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    reflection_budget_optimizer_implemented: Literal[True] = True
    reflection_budget_metadata_implemented: Literal[True] = True
    reasoning_budget_metadata_used: Literal[True] = True
    reflection_loop_metadata_used: Literal[True] = True
    workflow_risk_metadata_used: Literal[True] = True
    hitl_posture_metadata_used: Literal[True] = True
    reflection_budget_enforcement_implemented: Literal[False] = False
    runtime_reflection_token_allocation_implemented: Literal[False] = False
    runtime_reasoning_token_allocation_implemented: Literal[False] = False
    reflection_loop_execution_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    context_trimming_implemented: Literal[False] = False
    prompt_compression_implemented: Literal[False] = False
    memory_summarization_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
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
        if self.reserve_guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "reserve_guardrail",
        ):
            raise ValueError("reserve_guardrail_candidate_ids must match candidates")
        if self.review_guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "review_guardrail",
        ):
            raise ValueError("review_guardrail_candidate_ids must match candidates")
        if self.hitl_required_candidate_ids != tuple(
            candidate.candidate_id
            for candidate in self.candidates
            if candidate.hitl_required
        ):
            raise ValueError("hitl_required_candidate_ids must match candidates")
        if self.applied_reflection_budget_candidate_ids:
            raise ValueError(
                "applied_reflection_budget_candidate_ids must remain empty"
            )
        if self.recommended_candidate_count != len(self.recommended_candidate_ids):
            raise ValueError("recommended_candidate_count must match candidates")
        if self.reserve_guardrail_candidate_count != len(
            self.reserve_guardrail_candidate_ids
        ):
            raise ValueError("reserve_guardrail_candidate_count must match candidates")
        if self.review_guardrail_candidate_count != len(
            self.review_guardrail_candidate_ids
        ):
            raise ValueError("review_guardrail_candidate_count must match candidates")
        if self.hitl_required_candidate_count != len(self.hitl_required_candidate_ids):
            raise ValueError("hitl_required_candidate_count must match candidates")
        if self.total_advisory_reflection_tokens != sum(
            candidate.advisory_reflection_tokens for candidate in self.candidates
        ):
            raise ValueError("total_advisory_reflection_tokens must match candidates")
        if self.total_advisory_reflection_reserve_tokens != sum(
            candidate.advisory_reflection_reserve_tokens
            for candidate in self.candidates
        ):
            raise ValueError(
                "total_advisory_reflection_reserve_tokens must match candidates"
            )
        if self.total_recommended_reflection_pass_count != sum(
            candidate.recommended_reflection_pass_count
            for candidate in self.candidates
        ):
            raise ValueError(
                "total_recommended_reflection_pass_count must match candidates"
            )
        if self.total_applied_reflection_pass_count != 0:
            raise ValueError("total_applied_reflection_pass_count must remain zero")
        if self.highest_reflection_budget_score != max(
            candidate.reflection_budget_score for candidate in self.candidates
        ):
            raise ValueError("highest_reflection_budget_score must match candidates")
        if self.reflection_budget_pressure != _plan_pressure(self.candidates):
            raise ValueError("reflection_budget_pressure must match candidates")
        return self


def optimize_reflection_budget(
    *,
    reasoning_budget: ReasoningBudgetOptimizationPlan | None = None,
    reflection_loop: ReflectionLoopProfile | None = None,
    workflow_risk: WorkflowRiskPlan | None = None,
) -> ReflectionBudgetOptimizationPlan:
    """Optimize reflection budget metadata without executing reflection loops."""

    reasoning = reasoning_budget or optimize_reasoning_budget()
    reflection = reflection_loop or _default_reflection_loop()
    risk = workflow_risk or evaluate_workflow_risk()
    candidates = _candidates(
        reasoning_budget=reasoning,
        reflection_loop=reflection,
        workflow_risk=risk,
    )
    return ReflectionBudgetOptimizationPlan(
        source_reasoning_budget_serialization_version=reasoning.serialization_version,
        source_reflection_loop_serialization_version=reflection.serialization_version,
        source_workflow_risk_serialization_version=risk.serialization_version,
        reflection_priority=reflection.reflection_priority,
        reflection_depth=reflection.reflection_depth,
        reflection_hitl_recommendation=reflection.hitl_recommendation,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        recommended_candidate_ids=_candidate_ids_for_status(candidates, "recommended"),
        reserve_guardrail_candidate_ids=_candidate_ids_for_status(
            candidates,
            "reserve_guardrail",
        ),
        review_guardrail_candidate_ids=_candidate_ids_for_status(
            candidates,
            "review_guardrail",
        ),
        hitl_required_candidate_ids=tuple(
            candidate.candidate_id for candidate in candidates if candidate.hitl_required
        ),
        applied_reflection_budget_candidate_ids=(),
        candidate_count=len(candidates),
        recommended_candidate_count=len(
            _candidate_ids_for_status(candidates, "recommended")
        ),
        reserve_guardrail_candidate_count=len(
            _candidate_ids_for_status(candidates, "reserve_guardrail")
        ),
        review_guardrail_candidate_count=len(
            _candidate_ids_for_status(candidates, "review_guardrail")
        ),
        hitl_required_candidate_count=sum(
            1 for candidate in candidates if candidate.hitl_required
        ),
        total_advisory_reflection_tokens=sum(
            candidate.advisory_reflection_tokens for candidate in candidates
        ),
        total_advisory_reflection_reserve_tokens=sum(
            candidate.advisory_reflection_reserve_tokens
            for candidate in candidates
        ),
        total_recommended_reflection_pass_count=sum(
            candidate.recommended_reflection_pass_count for candidate in candidates
        ),
        total_applied_reflection_pass_count=0,
        highest_reflection_budget_score=max(
            candidate.reflection_budget_score for candidate in candidates
        ),
        reflection_budget_pressure=_plan_pressure(candidates),
        advisory_actions=_plan_actions(candidates),
    )


def reflection_budget_candidate_by_id(
    candidate_id: str,
    plan: ReflectionBudgetOptimizationPlan | None = None,
) -> ReflectionBudgetOptimizationCandidate | None:
    """Return one reflection budget candidate without applying it."""

    source_plan = plan or optimize_reflection_budget()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def reflection_budget_candidates_for_status(
    status: ReflectionBudgetStatus,
    plan: ReflectionBudgetOptimizationPlan | None = None,
) -> tuple[ReflectionBudgetOptimizationCandidate, ...]:
    """Return reflection budget candidates by advisory status."""

    source_plan = plan or optimize_reflection_budget()
    return tuple(candidate for candidate in source_plan.candidates if candidate.status == status)


def _candidates(
    *,
    reasoning_budget: ReasoningBudgetOptimizationPlan,
    reflection_loop: ReflectionLoopProfile,
    workflow_risk: WorkflowRiskPlan,
) -> tuple[ReflectionBudgetOptimizationCandidate, ...]:
    mappings: tuple[tuple[ReflectionBudgetKind, str, str], ...] = (
        (
            "loop_depth_budget",
            "reasoning_budget::context_reasoning_allocation",
            "workflow_risk::execution_confidence_risk",
        ),
        (
            "quality_gain_budget",
            "reasoning_budget::performance_reasoning_reserve",
            "workflow_risk::self_tuning_policy_risk",
        ),
        (
            "risk_reduction_budget",
            "reasoning_budget::benchmark_reasoning_reserve",
            "workflow_risk::performance_regression_risk",
        ),
        (
            "policy_review_budget",
            "reasoning_budget::budget_policy_review",
            "workflow_risk::escalation_posture_risk",
        ),
    )
    return tuple(
        _candidate(
            kind=kind,
            reasoning=_required_reasoning_recommendation(
                recommendation_id,
                reasoning_budget,
            ),
            reflection_loop=reflection_loop,
            workflow_risk=_required_workflow_risk_factor(
                risk_factor_id,
                workflow_risk,
            ),
        )
        for kind, recommendation_id, risk_factor_id in mappings
    )


def _candidate(
    *,
    kind: ReflectionBudgetKind,
    reasoning: ReasoningBudgetRecommendation,
    reflection_loop: ReflectionLoopProfile,
    workflow_risk: WorkflowRiskFactor,
) -> ReflectionBudgetOptimizationCandidate:
    status = _status_from_reasoning_status(reasoning.status)
    depth_weight = _depth_weight(reflection_loop.reflection_depth)
    quality_weight = _estimate_weight(reflection_loop.expected_quality_gain)
    risk_weight = _estimate_weight(reflection_loop.expected_risk_reduction)
    workflow_weight = _workflow_risk_weight(workflow_risk.workflow_risk_score)
    penalty = _guardrail_penalty(
        status=status,
        source_reasoning_pressure=reasoning.status,
        workflow_risk_severity=workflow_risk.severity,
        hitl_recommendation=reflection_loop.hitl_recommendation,
    )
    return ReflectionBudgetOptimizationCandidate(
        candidate_id=f"reflection_budget_optimizer::{kind}",
        budget_kind=kind,
        status=status,
        source_reasoning_recommendation_id=reasoning.recommendation_id,
        source_reasoning_budget_kind=reasoning.budget_kind,
        source_reasoning_budget_status=reasoning.status,
        source_workflow_risk_factor_id=workflow_risk.factor_id,
        source_workflow_risk_severity=workflow_risk.severity,
        reflection_priority=reflection_loop.reflection_priority,
        reflection_depth=reflection_loop.reflection_depth,
        reflection_hitl_recommendation=reflection_loop.hitl_recommendation,
        expected_quality_gain=reflection_loop.expected_quality_gain,
        expected_risk_reduction=reflection_loop.expected_risk_reduction,
        expected_cost=reflection_loop.expected_cost,
        expected_latency=reflection_loop.expected_latency,
        source_reasoning_tokens=reasoning.advisory_reasoning_tokens,
        source_reserve_tokens=reasoning.advisory_reserve_tokens,
        source_reasoning_pressure_score=reasoning.advisory_pressure_score,
        source_workflow_risk_score=workflow_risk.workflow_risk_score,
        advisory_reflection_tokens=_advisory_reflection_tokens(
            status=status,
            source_reasoning_tokens=reasoning.advisory_reasoning_tokens,
            depth_weight=depth_weight,
        ),
        advisory_reflection_reserve_tokens=_advisory_reflection_reserve_tokens(
            status=status,
            source_reserve_tokens=reasoning.advisory_reserve_tokens,
            depth_weight=depth_weight,
        ),
        depth_weight=depth_weight,
        quality_gain_weight=quality_weight,
        risk_reduction_weight=risk_weight,
        workflow_risk_weight=workflow_weight,
        guardrail_penalty=penalty,
        reflection_budget_score=_reflection_budget_score(
            reasoning_pressure_score=reasoning.advisory_pressure_score,
            workflow_risk_weight=workflow_weight,
            depth_weight=depth_weight,
            quality_gain_weight=quality_weight,
            risk_reduction_weight=risk_weight,
            guardrail_penalty=penalty,
        ),
        recommended_reflection_pass_count=1 if status == "recommended" else 0,
        applied_reflection_pass_count=0,
        hitl_required=(
            workflow_risk.hitl_required
            or reflection_loop.hitl_recommendation in {"recommended", "required"}
            or status != "recommended"
        ),
        budget_summary=_budget_summary(kind, status),
        fallback_summary=_fallback_summary(status),
        advisory_actions=_candidate_actions(status),
        evidence=(
            f"reasoning_budget:{reasoning.recommendation_id}",
            f"workflow_risk:{workflow_risk.factor_id}",
            f"reflection_priority:{reflection_loop.reflection_priority}",
            f"reflection_depth:{reflection_loop.reflection_depth}",
            f"expected_cost:{reflection_loop.expected_cost}",
            f"expected_latency:{reflection_loop.expected_latency}",
        ),
    )


def _required_reasoning_recommendation(
    recommendation_id: str,
    plan: ReasoningBudgetOptimizationPlan,
) -> ReasoningBudgetRecommendation:
    recommendation = reasoning_budget_recommendation_by_id(recommendation_id, plan)
    if recommendation is None:
        raise ValueError("required reflection reasoning budget metadata is missing")
    return recommendation


def _required_workflow_risk_factor(
    factor_id: str,
    plan: WorkflowRiskPlan,
) -> WorkflowRiskFactor:
    factor = workflow_risk_factor_by_id(factor_id, plan)
    if factor is None:
        raise ValueError("required reflection workflow risk metadata is missing")
    return factor


def _status_from_reasoning_status(
    status: ReasoningBudgetStatus,
) -> ReflectionBudgetStatus:
    if status == "optimization_candidate":
        return "recommended"
    if status == "reserve_guardrail":
        return "reserve_guardrail"
    return "review_guardrail"


def _advisory_reflection_tokens(
    *,
    status: ReflectionBudgetStatus,
    source_reasoning_tokens: int,
    depth_weight: int,
) -> int:
    if status != "recommended":
        return 0
    return source_reasoning_tokens // 2 + depth_weight * 5


def _advisory_reflection_reserve_tokens(
    *,
    status: ReflectionBudgetStatus,
    source_reserve_tokens: int,
    depth_weight: int,
) -> int:
    if status == "review_guardrail":
        return 0
    if status == "reserve_guardrail":
        return source_reserve_tokens // 2 + depth_weight * 5
    return source_reserve_tokens // 3


def _depth_weight(depth: ReflectionDepth) -> int:
    return {
        "none": 0,
        "light": 40,
        "moderate": 80,
        "deep": 120,
    }[depth]


def _estimate_weight(estimate: ReflectionEstimate) -> int:
    return {
        "none": 0,
        "low": 20,
        "medium": 50,
        "high": 80,
    }[estimate]


def _workflow_risk_weight(workflow_risk_score: int) -> int:
    return min(250, workflow_risk_score // 8)


def _guardrail_penalty(
    *,
    status: ReflectionBudgetStatus,
    source_reasoning_pressure: ReasoningBudgetStatus,
    workflow_risk_severity: str,
    hitl_recommendation: HitlRecommendation,
) -> int:
    penalty = 0
    if status == "reserve_guardrail":
        penalty += 80
    elif status == "review_guardrail":
        penalty += 120
    if source_reasoning_pressure in {"reserve_guardrail", "review_guardrail"}:
        penalty += 40
    if workflow_risk_severity == "guarded":
        penalty += 60
    elif workflow_risk_severity == "high":
        penalty += 30
    if hitl_recommendation in {"recommended", "required"}:
        penalty += 30
    return min(260, penalty)


def _reflection_budget_score(
    *,
    reasoning_pressure_score: int,
    workflow_risk_weight: int,
    depth_weight: int,
    quality_gain_weight: int,
    risk_reduction_weight: int,
    guardrail_penalty: int,
) -> int:
    return min(
        600,
        max(
            0,
            reasoning_pressure_score
            + workflow_risk_weight
            + depth_weight
            + quality_gain_weight
            + risk_reduction_weight
            - guardrail_penalty,
        ),
    )


def _candidate_ids_for_status(
    candidates: tuple[ReflectionBudgetOptimizationCandidate, ...],
    status: ReflectionBudgetStatus,
) -> tuple[str, ...]:
    return tuple(candidate.candidate_id for candidate in candidates if candidate.status == status)


def _plan_pressure(
    candidates: tuple[ReflectionBudgetOptimizationCandidate, ...],
) -> Literal["low", "medium", "high", "guarded"]:
    if _candidate_ids_for_status(candidates, "review_guardrail"):
        return "guarded"
    highest = max(candidate.reflection_budget_score for candidate in candidates)
    if highest >= 520:
        return "high"
    if highest >= 260:
        return "medium"
    return "low"


def _budget_summary(
    kind: ReflectionBudgetKind,
    status: ReflectionBudgetStatus,
) -> str:
    if status == "recommended":
        return f"Surface {kind} as advisory reflection budget capacity."
    if status == "reserve_guardrail":
        return f"Keep {kind} as reserve-only reflection budget metadata."
    return f"Keep {kind} in review guardrail posture without token changes."


def _fallback_summary(status: ReflectionBudgetStatus) -> str:
    if status == "recommended":
        return "Fallback to reserve guardrail posture before runtime token allocation."
    if status == "reserve_guardrail":
        return "Fallback to review guardrail posture if risk or pressure rises."
    return "Preserve review guardrail posture without applying reflection budget."


def _candidate_actions(status: ReflectionBudgetStatus) -> tuple[str, ...]:
    return (
        f"Surface {status} reflection budget posture as advisory metadata.",
        "Keep reflection execution, token allocation, budget enforcement, refinement, routing, workflow, storage, and output behavior disabled.",
    )


def _plan_actions(
    candidates: tuple[ReflectionBudgetOptimizationCandidate, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose reflection budget optimization as advisory metadata only.",
        "Keep applied reflection budget candidate ids empty and applied pass count at zero.",
        "Preserve reflection execution, token allocation, budget enforcement, refinement, routing, workflow, storage, and output boundaries.",
    ]
    if _candidate_ids_for_status(candidates, "review_guardrail"):
        actions.append("Keep reflection budget review guardrails non-blocking.")
    return tuple(actions)


def _default_reflection_loop() -> ReflectionLoopProfile:
    return ReflectionLoopProfile(
        reflection_confidence=0.68,
        reflection_summary=(
            "Default V5.5 reflection budget source profile recommends "
            "moderate advisory reflection planning without workflow execution."
        ),
        reflection_required=True,
        reflection_priority="high",
        reflection_rationale=(
            "Reflection budget optimization needs an inspectable source profile.",
        ),
        reflection_depth="moderate",
        expected_quality_gain="medium",
        expected_risk_reduction="medium",
        expected_cost="medium",
        expected_latency="medium",
        confidence_after_reflection=0.78,
        unresolved_questions=(
            "Confirm reflection budget before any future runtime behavior.",
        ),
        refinement_candidates=("Review reflection budget metadata.",),
        stop_conditions=("Do not trigger reflection from this metadata.",),
        hitl_recommendation="recommended",
        prompt_guidance=("Use reflection budget metadata only.",),
        evidence=("default_reflection_budget_source_profile",),
    )
