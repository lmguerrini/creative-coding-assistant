"""V5.5 adaptive advisory cost/quality optimization intelligence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_activation_optimizer import (
    AgentActivationOptimizationPlan,
    optimize_agent_activation,
)
from creative_coding_assistant.orchestration.adaptive_hybrid_workflow_optimizer import (
    HybridWorkflowOptimizationPlan,
    optimize_hybrid_workflow,
)
from creative_coding_assistant.orchestration.cost_prediction_engine import (
    CostPredictionPlan,
    predict_cost_for_route,
)
from creative_coding_assistant.orchestration.hybrid_studio import (
    CostProfileBand,
    QualityProfileLevel,
)
from creative_coding_assistant.orchestration.quality_cost_optimizer import (
    QualityCostOptimizationPlan,
    QualityCostTradeoffPosture,
    optimize_quality_cost,
)
from creative_coding_assistant.orchestration.quality_prediction_engine import (
    QualityPredictionPlan,
    predict_quality_for_route,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

AdaptiveCostQualityStatus = Literal["recommended", "fallback"]
AdaptiveCostQualityPosture = Literal[
    "quality_priority",
    "cost_priority",
    "balanced_adaptive_tradeoff",
]

ADAPTIVE_COST_QUALITY_CANDIDATE_SERIALIZATION_VERSION = (
    "adaptive_cost_quality_candidate.v1"
)
ADAPTIVE_COST_QUALITY_PLAN_SERIALIZATION_VERSION = (
    "adaptive_cost_quality_plan.v1"
)
ADAPTIVE_COST_QUALITY_AUTHORITY_BOUNDARY = (
    "V5.5 adaptive cost/quality optimization combines existing advisory "
    "quality/cost tradeoff, quality prediction, cost prediction, hybrid "
    "workflow, and agent activation metadata into relative recommendations "
    "only; it does not evaluate generated output, look up provider pricing, "
    "meter live usage, enforce budgets, select or switch providers or models, "
    "execute providers, control workflows, trigger retries, mutate prompts, "
    "write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "generated_output_quality_evaluation",
    "provider_pricing_lookup",
    "live_usage_metering",
    "budget_enforcement",
    "cost_based_routing",
    "quality_based_routing",
    "automatic_provider_selection",
    "automatic_model_selection",
    "provider_or_model_routing",
    "provider_execution",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class AdaptiveCostQualityCandidate(BaseModel):
    """One advisory adaptive cost/quality candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    source_quality_cost_candidate_id: str = Field(min_length=1, max_length=180)
    source_model_profile_id: str = Field(min_length=1, max_length=120)
    source_hybrid_workflow_candidate_id: str = Field(min_length=1, max_length=180)
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    quality_level: QualityProfileLevel
    cost_band: CostProfileBand
    quality_cost_tradeoff_posture: QualityCostTradeoffPosture
    predicted_quality_midpoint: int = Field(ge=0, le=100)
    predicted_cost_midpoint: int = Field(ge=0, le=100)
    quality_weight: int = Field(ge=0, le=120)
    cost_weight: int = Field(ge=0, le=120)
    hybrid_context_bonus: int = Field(ge=0, le=40)
    agent_context_bonus: int = Field(ge=0, le=40)
    adaptive_score: int = Field(ge=0, le=240)
    adaptive_posture: AdaptiveCostQualityPosture
    status: AdaptiveCostQualityStatus
    hitl_required: bool
    evidence: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    adaptive_cost_quality_optimizer_implemented: Literal[True] = True
    adaptive_cost_quality_scoring_implemented: Literal[True] = True
    quality_prediction_metadata_used: Literal[True] = True
    cost_prediction_metadata_used: Literal[True] = True
    generated_output_quality_evaluation_implemented: Literal[False] = False
    provider_pricing_lookup_implemented: Literal[False] = False
    live_usage_metering_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    cost_based_routing_implemented: Literal[False] = False
    quality_based_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["adaptive_cost_quality_candidate.v1"] = (
        ADAPTIVE_COST_QUALITY_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_scores(self) -> Self:
        if self.candidate_id != f"adaptive_cost_quality::{self.source_model_profile_id}":
            raise ValueError("candidate_id must match source_model_profile_id")
        expected_score = min(
            240,
            self.quality_weight
            + self.cost_weight
            + self.hybrid_context_bonus
            + self.agent_context_bonus,
        )
        if self.adaptive_score != expected_score:
            raise ValueError("adaptive_score must combine weights")
        if self.adaptive_posture != _adaptive_posture(
            self.quality_weight,
            self.cost_weight,
        ):
            raise ValueError("adaptive_posture must match weights")
        return self


class AdaptiveCostQualityPlan(BaseModel):
    """Bounded V5.5 advisory adaptive cost/quality optimization plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["adaptive_cost_quality_optimizer"] = "adaptive_cost_quality_optimizer"
    serialization_version: Literal["adaptive_cost_quality_plan.v1"] = (
        ADAPTIVE_COST_QUALITY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ADAPTIVE_COST_QUALITY_AUTHORITY_BOUNDARY,
        max_length=1600,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_quality_cost_serialization_version: str = Field(min_length=1, max_length=120)
    source_quality_prediction_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_cost_prediction_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_hybrid_workflow_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_agent_activation_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    candidates: tuple[AdaptiveCostQualityCandidate, ...] = Field(
        min_length=1,
        max_length=12,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    recommended_candidate_id: str = Field(min_length=1, max_length=180)
    fallback_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    recommended_adaptive_posture: AdaptiveCostQualityPosture
    candidate_count: int = Field(ge=1, le=12)
    recommended_adaptive_score: int = Field(ge=0, le=240)
    hitl_required_candidate_count: int = Field(ge=0, le=12)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    adaptive_cost_quality_optimizer_implemented: Literal[True] = True
    adaptive_cost_quality_scoring_implemented: Literal[True] = True
    generated_output_quality_evaluation_implemented: Literal[False] = False
    provider_pricing_lookup_implemented: Literal[False] = False
    live_usage_metering_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    cost_based_routing_implemented: Literal[False] = False
    quality_based_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_candidates(self) -> Self:
        derived_candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(set(derived_candidate_ids)) != len(derived_candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if self.candidate_ids != derived_candidate_ids:
            raise ValueError("candidate_ids must match candidates")
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count must match candidates")
        recommended = tuple(
            candidate for candidate in self.candidates if candidate.status == "recommended"
        )
        if len(recommended) != 1:
            raise ValueError("exactly one recommended candidate is required")
        recommended_candidate = recommended[0]
        if self.recommended_candidate_id != recommended_candidate.candidate_id:
            raise ValueError("recommended_candidate_id must match candidate")
        if self.recommended_adaptive_posture != recommended_candidate.adaptive_posture:
            raise ValueError("recommended_adaptive_posture must match candidate")
        if self.recommended_adaptive_score != recommended_candidate.adaptive_score:
            raise ValueError("recommended_adaptive_score must match candidate")
        if self.fallback_candidate_ids != tuple(
            candidate.candidate_id
            for candidate in self.candidates
            if candidate.status == "fallback"
        ):
            raise ValueError("fallback_candidate_ids must match candidates")
        if self.hitl_required_candidate_count != sum(
            1 for candidate in self.candidates if candidate.hitl_required
        ):
            raise ValueError("hitl_required_candidate_count must match candidates")
        for candidate in self.candidates:
            if candidate.route_name != self.route_name:
                raise ValueError("candidate route_name must match plan route_name")
        return self


def optimize_adaptive_cost_quality(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    quality_cost: QualityCostOptimizationPlan | None = None,
    quality_prediction: QualityPredictionPlan | None = None,
    cost_prediction: CostPredictionPlan | None = None,
    hybrid_workflow: HybridWorkflowOptimizationPlan | None = None,
    agent_activation: AgentActivationOptimizationPlan | None = None,
) -> AdaptiveCostQualityPlan:
    """Recommend adaptive cost/quality posture without applying routing."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    qc_plan = quality_cost or optimize_quality_cost(route=route_name)
    quality_plan = quality_prediction or predict_quality_for_route(route=route_name)
    cost_plan = cost_prediction or predict_cost_for_route(route=route_name)
    hybrid_plan = hybrid_workflow or optimize_hybrid_workflow(
        task_type=normalized_task_type,
        route=route_name,
        execution_mode_id=execution_mode_id,
    )
    agent_plan = agent_activation or optimize_agent_activation(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    execution_modes = routing_execution_mode_registry()
    candidates = tuple(
        _candidate_from_quality_cost(
            source=candidate,
            task_type=hybrid_plan.task_type,
            execution_mode_id=agent_plan.candidates[0].execution_mode_id,
            quality_prediction=quality_plan,
            cost_prediction=cost_plan,
            hybrid_workflow=hybrid_plan,
            agent_activation=agent_plan,
        )
        for candidate in qc_plan.candidates
    )
    recommended = max(candidates, key=lambda candidate: candidate.adaptive_score)
    candidates = tuple(
        candidate.model_copy(
            update={
                "status": (
                    "recommended"
                    if candidate.candidate_id == recommended.candidate_id
                    else "fallback"
                )
            }
        )
        for candidate in candidates
    )
    recommended = next(candidate for candidate in candidates if candidate.status == "recommended")
    return AdaptiveCostQualityPlan(
        route_name=route_name,
        task_type=hybrid_plan.task_type,
        source_quality_cost_serialization_version=qc_plan.serialization_version,
        source_quality_prediction_serialization_version=quality_plan.serialization_version,
        source_cost_prediction_serialization_version=cost_plan.serialization_version,
        source_hybrid_workflow_serialization_version=hybrid_plan.serialization_version,
        source_agent_activation_serialization_version=agent_plan.serialization_version,
        execution_mode_ids=execution_modes.execution_mode_ids,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        recommended_candidate_id=recommended.candidate_id,
        fallback_candidate_ids=tuple(
            candidate.candidate_id for candidate in candidates if candidate.status == "fallback"
        ),
        recommended_adaptive_posture=recommended.adaptive_posture,
        candidate_count=len(candidates),
        recommended_adaptive_score=recommended.adaptive_score,
        hitl_required_candidate_count=sum(
            1 for candidate in candidates if candidate.hitl_required
        ),
        advisory_actions=_plan_actions(recommended),
    )


def adaptive_cost_quality_candidate_by_id(
    candidate_id: str,
    plan: AdaptiveCostQualityPlan | None = None,
) -> AdaptiveCostQualityCandidate | None:
    """Return one adaptive cost/quality candidate without applying it."""

    source_plan = plan or optimize_adaptive_cost_quality()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def adaptive_cost_quality_candidates_for_posture(
    posture: AdaptiveCostQualityPosture,
    plan: AdaptiveCostQualityPlan | None = None,
) -> tuple[AdaptiveCostQualityCandidate, ...]:
    """Return adaptive cost/quality candidates for one posture."""

    source_plan = plan or optimize_adaptive_cost_quality()
    return tuple(
        candidate for candidate in source_plan.candidates if candidate.adaptive_posture == posture
    )


def _candidate_from_quality_cost(
    *,
    source: object,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    quality_prediction: QualityPredictionPlan,
    cost_prediction: CostPredictionPlan,
    hybrid_workflow: HybridWorkflowOptimizationPlan,
    agent_activation: AgentActivationOptimizationPlan,
) -> AdaptiveCostQualityCandidate:
    quality = _matching_quality_midpoint(source, quality_prediction)
    cost = _matching_cost_midpoint(source, cost_prediction)
    quality_weight = min(120, getattr(source, "quality_score") + quality // 4)
    cost_weight = min(120, getattr(source, "cost_score") + max(0, 100 - cost) // 5)
    hybrid_bonus = 20 if hybrid_workflow.recommended_candidate_id else 0
    agent_bonus = min(40, agent_activation.highest_activation_score // 8)
    hitl_required = (
        hybrid_workflow.hitl_required_candidate_count > 0
        or bool(agent_activation.hitl_required_candidate_ids)
    )
    return AdaptiveCostQualityCandidate(
        candidate_id=f"adaptive_cost_quality::{getattr(source, 'source_model_profile_id')}",
        source_quality_cost_candidate_id=getattr(source, "candidate_id"),
        source_model_profile_id=getattr(source, "source_model_profile_id"),
        source_hybrid_workflow_candidate_id=hybrid_workflow.recommended_candidate_id,
        route_name=getattr(source, "route_name"),
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        quality_level=getattr(source, "quality_level"),
        cost_band=getattr(source, "cost_band"),
        quality_cost_tradeoff_posture=getattr(source, "tradeoff_posture"),
        predicted_quality_midpoint=quality,
        predicted_cost_midpoint=cost,
        quality_weight=quality_weight,
        cost_weight=cost_weight,
        hybrid_context_bonus=hybrid_bonus,
        agent_context_bonus=agent_bonus,
        adaptive_score=min(240, quality_weight + cost_weight + hybrid_bonus + agent_bonus),
        adaptive_posture=_adaptive_posture(quality_weight, cost_weight),
        status="fallback",
        hitl_required=hitl_required,
        evidence=(
            f"source_quality_cost:{getattr(source, 'candidate_id')}",
            f"quality_midpoint:{quality}",
            f"cost_midpoint:{cost}",
            f"hybrid_context:{hybrid_workflow.recommended_candidate_id}",
            f"agent_score:{agent_activation.highest_activation_score}",
        ),
        advisory_actions=(
            "Surface adaptive cost/quality tradeoff for review.",
            "Keep pricing, metering, evaluation, budget, routing, and execution disabled.",
        ),
    )


def _matching_quality_midpoint(
    source: object,
    quality_prediction: QualityPredictionPlan,
) -> int:
    model_profile_id = getattr(source, "source_model_profile_id")
    for prediction in quality_prediction.predictions:
        if model_profile_id in prediction.source_model_profile_ids:
            return prediction.predicted_quality_midpoint
    return quality_prediction.recommended_quality_midpoint


def _matching_cost_midpoint(
    source: object,
    cost_prediction: CostPredictionPlan,
) -> int:
    model_profile_id = getattr(source, "source_model_profile_id")
    for prediction in cost_prediction.predictions:
        if model_profile_id in prediction.source_model_profile_ids:
            return prediction.predicted_cost_midpoint
    return cost_prediction.recommended_cost_midpoint


def _adaptive_posture(
    quality_weight: int,
    cost_weight: int,
) -> AdaptiveCostQualityPosture:
    if quality_weight - cost_weight >= 18:
        return "quality_priority"
    if cost_weight - quality_weight >= 18:
        return "cost_priority"
    return "balanced_adaptive_tradeoff"


def _plan_actions(
    recommended: AdaptiveCostQualityCandidate,
) -> tuple[str, ...]:
    return (
        f"Present {recommended.adaptive_posture} as advisory adaptive cost/quality posture.",
        "Use relative quality and cost metadata only.",
        "Preserve provider routing, pricing lookup, live metering, budget enforcement, execution, storage, and output boundaries.",
    )


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
