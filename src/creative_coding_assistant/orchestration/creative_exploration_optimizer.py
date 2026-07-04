"""V5.5 advisory creative exploration optimization intelligence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.creative_diversity_predictor import (
    CreativeDiversityPrediction,
    CreativeDiversityPredictionPlan,
    creative_diversity_prediction_by_id,
    predict_creative_diversity,
)
from creative_coding_assistant.orchestration.exploration_budget_planner import (
    ExplorationBudgetAllocation,
    ExplorationBudgetPlan,
    ExplorationBudgetPressure,
    ExplorationBudgetPriority,
    ExplorationBudgetTopic,
    exploration_budget_allocation_by_id,
    plan_exploration_budget,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    HybridRoutingPolicyDirection,
    ProviderId,
    TaskRoutingType,
    UnavailableReasonCode,
    routing_execution_mode_registry,
)
from creative_coding_assistant.orchestration.workflow_risk_engine import (
    WorkflowRiskFactor,
    WorkflowRiskPlan,
    evaluate_workflow_risk,
    workflow_risk_factor_by_id,
)

CreativeExplorationOptimizationStatus = Literal[
    "recommended",
    "bounded",
    "guardrail",
]
CreativeExplorationStrategy = Literal[
    "balanced_exploration",
    "diversity_priority",
    "risk_bounded_exploration",
    "synthesis_guardrail",
]

CREATIVE_EXPLORATION_OPTIMIZATION_CANDIDATE_SERIALIZATION_VERSION = (
    "creative_exploration_optimization_candidate.v1"
)
CREATIVE_EXPLORATION_OPTIMIZATION_PLAN_SERIALIZATION_VERSION = (
    "creative_exploration_optimization_plan.v1"
)
CREATIVE_EXPLORATION_OPTIMIZER_AUTHORITY_BOUNDARY = (
    "V5.5 creative exploration optimization combines advisory exploration "
    "budget, creative diversity prediction, and workflow risk metadata into "
    "inspectable creative exploration recommendations only; it does not "
    "generate variants, select variants or artifacts, trigger refinement, "
    "enforce budgets, route by cost, change provider or model routing, "
    "execute providers, probe local runtimes, scan or download local models, "
    "invoke agents, allocate resources, apply risk mitigation, emit HITL "
    "requests, control workflows, mutate workflow graphs, execute workflows, "
    "trigger retries, mutate prompts, write storage, modify generated output, "
    "or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "variant_generation",
    "variant_selection",
    "artifact_selection",
    "refinement_triggering",
    "budget_enforcement",
    "cost_based_routing",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
    "provider_execution",
    "local_runtime_probe",
    "local_model_inventory_scan",
    "automatic_model_download",
    "agent_invocation",
    "resource_allocation",
    "risk_mitigation_execution",
    "human_review_request",
    "hitl_request_emission",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class CreativeExplorationOptimizationCandidate(BaseModel):
    """One advisory creative exploration optimization candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    topic_id: ExplorationBudgetTopic
    strategy: CreativeExplorationStrategy
    status: CreativeExplorationOptimizationStatus
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_exploration_allocation_id: str = Field(min_length=1, max_length=180)
    source_diversity_prediction_id: str = Field(min_length=1, max_length=220)
    source_workflow_risk_factor_id: str = Field(min_length=1, max_length=180)
    source_execution_confidence_signal_id: str = Field(min_length=1, max_length=180)
    source_budget_profile_id: str = Field(min_length=1, max_length=180)
    provider_sequence: tuple[ProviderId, ...] = Field(min_length=1, max_length=4)
    model_profile_sequence: tuple[str, ...] = Field(min_length=1, max_length=4)
    hybrid_policy_direction: HybridRoutingPolicyDirection
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=9,
    )
    budget_posture: str = Field(min_length=1, max_length=80)
    diversity_band: str = Field(min_length=1, max_length=80)
    exploration_pressure: ExplorationBudgetPressure
    workflow_risk_severity: str = Field(min_length=1, max_length=80)
    priority: ExplorationBudgetPriority
    requested_variants: int = Field(ge=0, le=20)
    planned_variants: int = Field(ge=0, le=20)
    recommended_advisory_variants: int = Field(ge=0, le=20)
    applied_variant_count: int = Field(ge=0, le=0)
    requested_refinement_passes: int = Field(ge=0, le=20)
    planned_refinement_passes: int = Field(ge=0, le=20)
    recommended_advisory_refinement_passes: int = Field(ge=0, le=20)
    applied_refinement_pass_count: int = Field(ge=0, le=0)
    diversity_readiness_score: int = Field(ge=0, le=100)
    workflow_risk_score: int = Field(ge=0, le=1_000)
    execution_confidence_score: int = Field(ge=0, le=100)
    exploration_budget_score: int = Field(ge=0, le=260)
    priority_weight: int = Field(ge=0, le=80)
    risk_penalty: int = Field(ge=0, le=360)
    creative_exploration_score: int = Field(ge=0, le=500)
    hitl_required: bool
    optimization_summary: str = Field(min_length=1, max_length=360)
    fallback_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    creative_exploration_optimizer_implemented: Literal[True] = True
    exploration_optimization_metadata_implemented: Literal[True] = True
    exploration_budget_metadata_used: Literal[True] = True
    creative_diversity_prediction_metadata_used: Literal[True] = True
    workflow_risk_metadata_used: Literal[True] = True
    provider_intelligence_metadata_used: Literal[True] = True
    availability_awareness_metadata_used: Literal[True] = True
    manual_assisted_auto_mode_metadata_used: Literal[True] = True
    hybrid_transition_metadata_used: Literal[True] = True
    task_aware_category_metadata_used: Literal[True] = True
    execution_simulation_metadata_used: Literal[True] = True
    fallback_safety_metadata_used: Literal[True] = True
    variant_generation_implemented: Literal[False] = False
    variant_selection_implemented: Literal[False] = False
    artifact_selection_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    cost_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    local_model_inventory_scan_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    risk_mitigation_execution_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["creative_exploration_optimization_candidate.v1"] = (
        CREATIVE_EXPLORATION_OPTIMIZATION_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_contract(self) -> Self:
        if self.candidate_id != f"creative_exploration_optimizer::{self.topic_id}":
            raise ValueError("candidate_id must match topic_id")
        if len(self.model_profile_sequence) != len(self.provider_sequence):
            raise ValueError("model_profile_sequence must match provider_sequence")
        if self.recommended_advisory_variants > self.planned_variants:
            raise ValueError("recommended variants must fit planned variants")
        if self.recommended_advisory_refinement_passes > self.planned_refinement_passes:
            raise ValueError("recommended refinement must fit planned refinement")
        if self.applied_variant_count or self.applied_refinement_pass_count:
            raise ValueError("applied exploration counts must remain zero")
        if self.exploration_budget_score != _exploration_budget_score(
            planned_variants=self.planned_variants,
            planned_refinement_passes=self.planned_refinement_passes,
            priority_weight=self.priority_weight,
        ):
            raise ValueError("exploration_budget_score must match source counts")
        if self.risk_penalty != _risk_penalty(
            workflow_risk_score=self.workflow_risk_score,
            workflow_risk_severity=self.workflow_risk_severity,
            status=self.status,
        ):
            raise ValueError("risk_penalty must match workflow risk posture")
        if self.creative_exploration_score != _creative_exploration_score(
            diversity_readiness_score=self.diversity_readiness_score,
            exploration_budget_score=self.exploration_budget_score,
            execution_confidence_score=self.execution_confidence_score,
            risk_penalty=self.risk_penalty,
        ):
            raise ValueError("creative_exploration_score must combine source scores")
        if self.status == "guardrail" and (
            self.recommended_advisory_variants
            or self.recommended_advisory_refinement_passes
        ):
            raise ValueError("guardrail candidates must recommend no exploration")
        if self.status == "recommended" and self.strategy != "diversity_priority":
            raise ValueError("recommended exploration must use diversity priority")
        return self


class CreativeExplorationOptimizationPlan(BaseModel):
    """Bounded V5.5 advisory creative exploration optimization plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_exploration_optimizer"] = "creative_exploration_optimizer"
    serialization_version: Literal["creative_exploration_optimization_plan.v1"] = (
        CREATIVE_EXPLORATION_OPTIMIZATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_EXPLORATION_OPTIMIZER_AUTHORITY_BOUNDARY,
        max_length=2100,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_exploration_budget_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_creative_diversity_prediction_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_workflow_risk_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    provider_ids: tuple[ProviderId, ...] = Field(min_length=4, max_length=4)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    hybrid_policy_directions: tuple[HybridRoutingPolicyDirection, ...] = Field(
        min_length=4,
        max_length=4,
    )
    candidates: tuple[CreativeExplorationOptimizationCandidate, ...] = Field(
        min_length=4,
        max_length=4,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    recommended_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=4
    )
    bounded_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=4
    )
    hitl_required_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    applied_exploration_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    candidate_count: int = Field(ge=4, le=4)
    recommended_candidate_count: int = Field(ge=0, le=4)
    guardrail_candidate_count: int = Field(ge=0, le=4)
    hitl_required_candidate_count: int = Field(ge=0, le=4)
    total_recommended_advisory_variants: int = Field(ge=0, le=40)
    total_recommended_advisory_refinement_passes: int = Field(ge=0, le=40)
    total_applied_variant_count: int = Field(ge=0, le=0)
    total_applied_refinement_pass_count: int = Field(ge=0, le=0)
    highest_creative_exploration_score: int = Field(ge=0, le=500)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    creative_exploration_optimizer_implemented: Literal[True] = True
    exploration_optimization_metadata_implemented: Literal[True] = True
    exploration_budget_metadata_used: Literal[True] = True
    creative_diversity_prediction_metadata_used: Literal[True] = True
    workflow_risk_metadata_used: Literal[True] = True
    provider_intelligence_metadata_used: Literal[True] = True
    availability_awareness_metadata_used: Literal[True] = True
    manual_assisted_auto_mode_metadata_used: Literal[True] = True
    hybrid_transition_metadata_used: Literal[True] = True
    task_aware_category_metadata_used: Literal[True] = True
    execution_simulation_metadata_used: Literal[True] = True
    fallback_safety_metadata_used: Literal[True] = True
    variant_generation_implemented: Literal[False] = False
    variant_selection_implemented: Literal[False] = False
    artifact_selection_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    cost_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    local_model_inventory_scan_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    risk_mitigation_execution_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
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
        if self.bounded_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "bounded",
        ):
            raise ValueError("bounded_candidate_ids must match candidates")
        if self.guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "guardrail",
        ):
            raise ValueError("guardrail_candidate_ids must match candidates")
        if self.hitl_required_candidate_ids != tuple(
            candidate.candidate_id
            for candidate in self.candidates
            if candidate.hitl_required
        ):
            raise ValueError("hitl_required_candidate_ids must match candidates")
        if self.applied_exploration_candidate_ids:
            raise ValueError("applied_exploration_candidate_ids must remain empty")
        if self.recommended_candidate_count != len(self.recommended_candidate_ids):
            raise ValueError("recommended_candidate_count must match candidates")
        if self.guardrail_candidate_count != len(self.guardrail_candidate_ids):
            raise ValueError("guardrail_candidate_count must match candidates")
        if self.hitl_required_candidate_count != len(self.hitl_required_candidate_ids):
            raise ValueError("hitl_required_candidate_count must match candidates")
        if self.total_recommended_advisory_variants != sum(
            candidate.recommended_advisory_variants for candidate in self.candidates
        ):
            raise ValueError(
                "total_recommended_advisory_variants must match candidates"
            )
        if self.total_recommended_advisory_refinement_passes != sum(
            candidate.recommended_advisory_refinement_passes
            for candidate in self.candidates
        ):
            raise ValueError(
                "total_recommended_advisory_refinement_passes must match candidates"
            )
        if self.total_applied_variant_count != 0:
            raise ValueError("total_applied_variant_count must remain zero")
        if self.total_applied_refinement_pass_count != 0:
            raise ValueError("total_applied_refinement_pass_count must remain zero")
        if self.highest_creative_exploration_score != max(
            candidate.creative_exploration_score for candidate in self.candidates
        ):
            raise ValueError("highest_creative_exploration_score must match candidates")
        for candidate in self.candidates:
            if candidate.route_name != self.route_name:
                raise ValueError("candidate route_name must match plan")
        return self


def optimize_creative_exploration(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    exploration_budget: ExplorationBudgetPlan | None = None,
    creative_diversity: CreativeDiversityPredictionPlan | None = None,
    workflow_risk: WorkflowRiskPlan | None = None,
) -> CreativeExplorationOptimizationPlan:
    """Optimize creative exploration metadata without generating variants."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    risk_plan = workflow_risk or evaluate_workflow_risk(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    normalized_mode = str(execution_mode_id or risk_plan.factors[0].execution_mode_id)
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")

    budget_plan = exploration_budget or plan_exploration_budget()
    diversity_plan = creative_diversity or predict_creative_diversity()
    candidates = _candidates(
        route_name=route_name,
        task_type=risk_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        exploration_budget=budget_plan,
        creative_diversity=diversity_plan,
        workflow_risk=risk_plan,
    )
    return CreativeExplorationOptimizationPlan(
        route_name=route_name,
        task_type=risk_plan.task_type,
        source_exploration_budget_serialization_version=(
            budget_plan.serialization_version
        ),
        source_creative_diversity_prediction_serialization_version=(
            diversity_plan.serialization_version
        ),
        source_workflow_risk_serialization_version=risk_plan.serialization_version,
        provider_ids=risk_plan.provider_ids,
        execution_mode_ids=execution_modes.execution_mode_ids,
        hybrid_policy_directions=risk_plan.hybrid_policy_directions,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        recommended_candidate_ids=_candidate_ids_for_status(candidates, "recommended"),
        bounded_candidate_ids=_candidate_ids_for_status(candidates, "bounded"),
        guardrail_candidate_ids=_candidate_ids_for_status(candidates, "guardrail"),
        hitl_required_candidate_ids=tuple(
            candidate.candidate_id
            for candidate in candidates
            if candidate.hitl_required
        ),
        applied_exploration_candidate_ids=(),
        candidate_count=len(candidates),
        recommended_candidate_count=len(
            _candidate_ids_for_status(candidates, "recommended")
        ),
        guardrail_candidate_count=len(
            _candidate_ids_for_status(candidates, "guardrail")
        ),
        hitl_required_candidate_count=sum(
            1 for candidate in candidates if candidate.hitl_required
        ),
        total_recommended_advisory_variants=sum(
            candidate.recommended_advisory_variants for candidate in candidates
        ),
        total_recommended_advisory_refinement_passes=sum(
            candidate.recommended_advisory_refinement_passes for candidate in candidates
        ),
        total_applied_variant_count=0,
        total_applied_refinement_pass_count=0,
        highest_creative_exploration_score=max(
            candidate.creative_exploration_score for candidate in candidates
        ),
        advisory_actions=_plan_actions(candidates),
    )


def creative_exploration_candidate_by_id(
    candidate_id: str,
    plan: CreativeExplorationOptimizationPlan | None = None,
) -> CreativeExplorationOptimizationCandidate | None:
    """Return one creative exploration candidate without applying it."""

    source_plan = plan or optimize_creative_exploration()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def creative_exploration_candidates_for_status(
    status: CreativeExplorationOptimizationStatus,
    plan: CreativeExplorationOptimizationPlan | None = None,
) -> tuple[CreativeExplorationOptimizationCandidate, ...]:
    """Return creative exploration candidates by advisory status."""

    source_plan = plan or optimize_creative_exploration()
    return tuple(
        candidate for candidate in source_plan.candidates if candidate.status == status
    )


def _candidates(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    exploration_budget: ExplorationBudgetPlan,
    creative_diversity: CreativeDiversityPredictionPlan,
    workflow_risk: WorkflowRiskPlan,
) -> tuple[CreativeExplorationOptimizationCandidate, ...]:
    return (
        _candidate(
            topic_id="planning_execution_fit",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            exploration_budget=exploration_budget,
            creative_diversity=creative_diversity,
            workflow_risk=workflow_risk,
            risk_factor_id="workflow_risk::execution_confidence_risk",
        ),
        _candidate(
            topic_id="style_aesthetic_alignment",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            exploration_budget=exploration_budget,
            creative_diversity=creative_diversity,
            workflow_risk=workflow_risk,
            risk_factor_id="workflow_risk::self_tuning_policy_risk",
        ),
        _candidate(
            topic_id="curation_refinement_need",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            exploration_budget=exploration_budget,
            creative_diversity=creative_diversity,
            workflow_risk=workflow_risk,
            risk_factor_id="workflow_risk::resource_capacity_risk",
        ),
        _candidate(
            topic_id="final_synthesis_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            exploration_budget=exploration_budget,
            creative_diversity=creative_diversity,
            workflow_risk=workflow_risk,
            risk_factor_id="workflow_risk::provider_fallback_risk",
        ),
    )


def _candidate(
    *,
    topic_id: ExplorationBudgetTopic,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    exploration_budget: ExplorationBudgetPlan,
    creative_diversity: CreativeDiversityPredictionPlan,
    workflow_risk: WorkflowRiskPlan,
    risk_factor_id: str,
) -> CreativeExplorationOptimizationCandidate:
    allocation = _required_allocation(f"exploration::{topic_id}", exploration_budget)
    prediction = _required_diversity_prediction(
        f"creative_diversity_prediction::{allocation.source_budget_profile_id}",
        creative_diversity,
    )
    risk = _required_workflow_risk_factor(risk_factor_id, workflow_risk)
    status = _status(allocation, prediction, risk)
    priority_weight = _priority_weight(allocation.priority)
    budget_score = _exploration_budget_score(
        planned_variants=allocation.planned_variants,
        planned_refinement_passes=allocation.planned_refinement_passes,
        priority_weight=priority_weight,
    )
    penalty = _risk_penalty(
        workflow_risk_score=risk.workflow_risk_score,
        workflow_risk_severity=risk.severity,
        status=status,
    )
    score = _creative_exploration_score(
        diversity_readiness_score=prediction.diversity_readiness_score,
        exploration_budget_score=budget_score,
        execution_confidence_score=risk.execution_confidence_score,
        risk_penalty=penalty,
    )
    recommended_variants = 0 if status == "guardrail" else allocation.planned_variants
    recommended_refinement = (
        0 if status == "guardrail" else allocation.planned_refinement_passes
    )
    return CreativeExplorationOptimizationCandidate(
        candidate_id=f"creative_exploration_optimizer::{topic_id}",
        topic_id=topic_id,
        strategy=_strategy(status, prediction.predicted_diversity_band),
        status=status,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_exploration_allocation_id=allocation.allocation_id,
        source_diversity_prediction_id=prediction.prediction_id,
        source_workflow_risk_factor_id=risk.factor_id,
        source_execution_confidence_signal_id=(
            risk.source_execution_confidence_signal_id
        ),
        source_budget_profile_id=allocation.source_budget_profile_id,
        provider_sequence=risk.provider_sequence,
        model_profile_sequence=risk.model_profile_sequence,
        hybrid_policy_direction=risk.hybrid_policy_direction,
        unavailable_reason_codes=risk.unavailable_reason_codes,
        budget_posture=allocation.budget_posture,
        diversity_band=prediction.predicted_diversity_band,
        exploration_pressure=allocation.pressure,
        workflow_risk_severity=risk.severity,
        priority=allocation.priority,
        requested_variants=allocation.requested_variants,
        planned_variants=allocation.planned_variants,
        recommended_advisory_variants=recommended_variants,
        applied_variant_count=0,
        requested_refinement_passes=allocation.requested_refinement_passes,
        planned_refinement_passes=allocation.planned_refinement_passes,
        recommended_advisory_refinement_passes=recommended_refinement,
        applied_refinement_pass_count=0,
        diversity_readiness_score=prediction.diversity_readiness_score,
        workflow_risk_score=risk.workflow_risk_score,
        execution_confidence_score=risk.execution_confidence_score,
        exploration_budget_score=budget_score,
        priority_weight=priority_weight,
        risk_penalty=penalty,
        creative_exploration_score=score,
        hitl_required=risk.hitl_required or status == "guardrail",
        optimization_summary=_optimization_summary(status, topic_id),
        fallback_summary=_fallback_summary(status),
        advisory_actions=_candidate_actions(status),
        evidence=(
            f"exploration_allocation:{allocation.allocation_id}",
            f"diversity_prediction:{prediction.prediction_id}",
            f"workflow_risk:{risk.factor_id}",
            f"budget_posture:{allocation.budget_posture}",
            f"diversity_band:{prediction.predicted_diversity_band}",
            f"workflow_risk_severity:{risk.severity}",
        ),
    )


def _creative_exploration_score(
    *,
    diversity_readiness_score: int,
    exploration_budget_score: int,
    execution_confidence_score: int,
    risk_penalty: int,
) -> int:
    return min(
        500,
        max(
            0,
            diversity_readiness_score * 2
            + exploration_budget_score
            + execution_confidence_score
            - risk_penalty,
        ),
    )


def _exploration_budget_score(
    *,
    planned_variants: int,
    planned_refinement_passes: int,
    priority_weight: int,
) -> int:
    return min(
        260,
        planned_variants * 35 + planned_refinement_passes * 30 + priority_weight,
    )


def _risk_penalty(
    *,
    workflow_risk_score: int,
    workflow_risk_severity: str,
    status: CreativeExplorationOptimizationStatus,
) -> int:
    penalty = workflow_risk_score // 4
    if workflow_risk_severity == "guarded" or status == "guardrail":
        penalty += 80
    elif workflow_risk_severity == "high":
        penalty += 40
    return min(360, penalty)


def _status(
    allocation: ExplorationBudgetAllocation,
    prediction: CreativeDiversityPrediction,
    risk: WorkflowRiskFactor,
) -> CreativeExplorationOptimizationStatus:
    if (
        allocation.budget_posture == "guarded"
        or prediction.predicted_diversity_band == "guarded"
        or risk.status == "guardrail"
    ):
        return "guardrail"
    if prediction.status == "recommended" and prediction.predicted_diversity_band == (
        "broad"
    ):
        return "recommended"
    return "bounded"


def _strategy(
    status: CreativeExplorationOptimizationStatus,
    diversity_band: str,
) -> CreativeExplorationStrategy:
    if status == "guardrail":
        return "synthesis_guardrail"
    if status == "recommended" and diversity_band == "broad":
        return "diversity_priority"
    if diversity_band in {"moderate", "broad"}:
        return "balanced_exploration"
    return "risk_bounded_exploration"


def _candidate_ids_for_status(
    candidates: tuple[CreativeExplorationOptimizationCandidate, ...],
    status: CreativeExplorationOptimizationStatus,
) -> tuple[str, ...]:
    return tuple(
        candidate.candidate_id for candidate in candidates if candidate.status == status
    )


def _priority_weight(priority: ExplorationBudgetPriority) -> int:
    return {
        "critical": 70,
        "high": 56,
        "medium": 38,
        "low": 20,
    }[priority]


def _required_allocation(
    allocation_id: str,
    plan: ExplorationBudgetPlan,
) -> ExplorationBudgetAllocation:
    allocation = exploration_budget_allocation_by_id(allocation_id, plan)
    if allocation is None:
        raise ValueError("required creative exploration allocation is missing")
    return allocation


def _required_diversity_prediction(
    prediction_id: str,
    plan: CreativeDiversityPredictionPlan,
) -> CreativeDiversityPrediction:
    prediction = creative_diversity_prediction_by_id(prediction_id, plan)
    if prediction is None:
        raise ValueError("required creative diversity prediction is missing")
    return prediction


def _required_workflow_risk_factor(
    factor_id: str,
    plan: WorkflowRiskPlan,
) -> WorkflowRiskFactor:
    factor = workflow_risk_factor_by_id(factor_id, plan)
    if factor is None:
        raise ValueError("required workflow risk factor is missing")
    return factor


def _optimization_summary(
    status: CreativeExplorationOptimizationStatus,
    topic_id: ExplorationBudgetTopic,
) -> str:
    if status == "recommended":
        return f"Surface {topic_id} as the primary advisory exploration path."
    if status == "bounded":
        return f"Keep {topic_id} exploration bounded by risk and budget metadata."
    return f"Keep {topic_id} exploration in guardrail posture without variants."


def _fallback_summary(status: CreativeExplorationOptimizationStatus) -> str:
    if status == "recommended":
        return "Fallback to bounded exploration if risk or budget posture changes."
    if status == "bounded":
        return "Fallback to guardrail posture before any runtime exploration behavior."
    return "Preserve guardrail posture without variant generation or refinement."


def _candidate_actions(
    status: CreativeExplorationOptimizationStatus,
) -> tuple[str, ...]:
    return (
        f"Surface {status} creative exploration recommendation as metadata.",
        "Keep variant generation, variant selection, refinement, routing, agents, workflow, storage, and output behavior disabled.",  # noqa: E501
    )


def _plan_actions(
    candidates: tuple[CreativeExplorationOptimizationCandidate, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose creative exploration optimization as advisory metadata only.",
        "Keep applied exploration candidate ids empty and applied counts at zero.",
        "Preserve variant generation, refinement, provider, agent, workflow, storage, and output boundaries.",
    ]
    if _candidate_ids_for_status(candidates, "guardrail"):
        actions.append("Require review before any future exploration behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
