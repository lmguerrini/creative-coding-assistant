"""V5.5 advisory adaptive policy explainability intelligence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_escalation_optimizer import (
    EscalationOptimizationPlan,
    optimize_escalation_policy,
)
from creative_coding_assistant.orchestration.adaptive_execution_strategy_selection import (
    AdaptiveExecutionStrategySelectionPlan,
    select_dynamic_execution_strategy,
)
from creative_coding_assistant.orchestration.execution_policy_engine import (
    ExecutionPolicyPlan,
    evaluate_execution_policies,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_explainability import (
    RoutingExplainabilityPlan,
    explain_routing_decision,
)
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    HybridRoutingPolicyDirection,
    ProviderId,
    TaskRoutingType,
)

AdaptivePolicyExplanationSurface = Literal[
    "execution_policy",
    "dynamic_execution_strategy",
    "adaptive_escalation",
    "routing_explainability",
]
AdaptivePolicyExplanationStatus = Literal["primary", "supporting", "guardrail"]

ADAPTIVE_POLICY_EXPLANATION_RECORD_SERIALIZATION_VERSION = (
    "adaptive_policy_explanation_record.v1"
)
ADAPTIVE_POLICY_EXPLAINABILITY_PLAN_SERIALIZATION_VERSION = (
    "adaptive_policy_explainability_plan.v1"
)
ADAPTIVE_POLICY_EXPLAINABILITY_AUTHORITY_BOUNDARY = (
    "V5.5 adaptive policy explainability combines advisory execution policy, "
    "dynamic execution strategy, adaptive escalation, and routing "
    "explainability metadata into inspectable policy explanations only; it "
    "does not apply policies, apply strategies, apply routing decisions, "
    "trigger escalation, emit HITL requests, enforce budgets, select or route "
    "providers or models, execute providers, invoke agents, control or "
    "execute workflows, mutate workflow graphs, trigger retries, mutate "
    "prompts, write storage, modify generated output, or apply Runtime "
    "Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "policy_application",
    "execution_policy_application",
    "strategy_application",
    "routing_application",
    "escalation_triggering",
    "human_review_request",
    "hitl_request_emission",
    "execution_blocking",
    "budget_enforcement",
    "automatic_provider_selection",
    "automatic_model_selection",
    "provider_or_model_routing",
    "provider_execution",
    "agent_invocation",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class AdaptivePolicyExplanationRecord(BaseModel):
    """One advisory adaptive policy explanation record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    explanation_id: str = Field(min_length=1, max_length=180)
    explanation_rank: int = Field(ge=1, le=8)
    source_surface: AdaptivePolicyExplanationSurface
    status: AdaptivePolicyExplanationStatus
    route_name: RouteName
    task_type: TaskRoutingType
    source_serialization_version: str = Field(min_length=1, max_length=120)
    source_primary_record_id: str = Field(min_length=1, max_length=180)
    source_record_count: int = Field(ge=1, le=24)
    source_policy_posture: str = Field(min_length=1, max_length=120)
    source_hitl_required: bool
    source_guardrail_count: int = Field(ge=0, le=24)
    source_signal_score: int = Field(ge=0, le=600)
    evidence_weight: int = Field(ge=0, le=240)
    hitl_explanation_weight: int = Field(ge=0, le=120)
    guardrail_penalty: int = Field(ge=0, le=180)
    policy_explainability_score: int = Field(ge=0, le=600)
    explanation_summary: str = Field(min_length=1, max_length=420)
    referenced_advisory_actions: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    adaptive_policy_explainability_implemented: Literal[True] = True
    adaptive_policy_explanation_metadata_implemented: Literal[True] = True
    execution_policy_metadata_used: Literal[True] = True
    dynamic_strategy_metadata_used: Literal[True] = True
    escalation_metadata_used: Literal[True] = True
    routing_explainability_metadata_used: Literal[True] = True
    policy_application_implemented: Literal[False] = False
    execution_policy_application_implemented: Literal[False] = False
    strategy_application_implemented: Literal[False] = False
    routing_application_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    execution_blocking_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["adaptive_policy_explanation_record.v1"] = (
        ADAPTIVE_POLICY_EXPLANATION_RECORD_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_matches_contract(self) -> Self:
        if self.explanation_id != f"adaptive_policy::{self.source_surface}":
            raise ValueError("explanation_id must match source_surface")
        if self.status == "primary" and self.explanation_rank != 1:
            raise ValueError("primary explanation must be rank 1")
        if self.status != "primary" and self.explanation_rank == 1:
            raise ValueError("rank 1 explanation must be primary")
        if self.evidence_weight != _evidence_weight(self.source_record_count):
            raise ValueError("evidence_weight must match source_record_count")
        if self.hitl_explanation_weight != _hitl_weight(self.source_hitl_required):
            raise ValueError("hitl_explanation_weight must match HITL posture")
        if self.guardrail_penalty != _guardrail_penalty(
            status=self.status,
            source_hitl_required=self.source_hitl_required,
            source_guardrail_count=self.source_guardrail_count,
        ):
            raise ValueError("guardrail_penalty must match source guardrails")
        if self.policy_explainability_score != _policy_explainability_score(
            source_signal_score=self.source_signal_score,
            evidence_weight=self.evidence_weight,
            hitl_explanation_weight=self.hitl_explanation_weight,
            guardrail_penalty=self.guardrail_penalty,
        ):
            raise ValueError("policy_explainability_score must combine source scores")
        if self.status == "guardrail" and not self.source_hitl_required:
            raise ValueError("guardrail explanations must carry HITL posture")
        return self


class AdaptivePolicyExplainabilityPlan(BaseModel):
    """Bounded V5.5 adaptive policy explainability plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["adaptive_policy_explainability"] = "adaptive_policy_explainability"
    serialization_version: Literal["adaptive_policy_explainability_plan.v1"] = (
        ADAPTIVE_POLICY_EXPLAINABILITY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ADAPTIVE_POLICY_EXPLAINABILITY_AUTHORITY_BOUNDARY,
        max_length=2100,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_execution_policy_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_dynamic_strategy_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_escalation_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_routing_explainability_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    provider_ids: tuple[ProviderId, ...] = Field(min_length=4, max_length=4)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    hybrid_policy_directions: tuple[HybridRoutingPolicyDirection, ...] = Field(
        min_length=4,
        max_length=4,
    )
    explanations: tuple[AdaptivePolicyExplanationRecord, ...] = Field(
        min_length=4,
        max_length=4,
    )
    explanation_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    source_surfaces: tuple[AdaptivePolicyExplanationSurface, ...] = Field(
        min_length=4,
        max_length=4,
    )
    primary_explanation_id: str = Field(min_length=1, max_length=180)
    supporting_explanation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    guardrail_explanation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    hitl_required_explanation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    applied_policy_explanation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    explanation_count: int = Field(ge=4, le=4)
    source_surface_count: int = Field(ge=4, le=4)
    supporting_explanation_count: int = Field(ge=0, le=4)
    guardrail_explanation_count: int = Field(ge=0, le=4)
    hitl_required_explanation_count: int = Field(ge=0, le=4)
    highest_policy_explainability_score: int = Field(ge=0, le=600)
    policy_explainability_pressure: Literal["low", "medium", "high", "guarded"]
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    adaptive_policy_explainability_implemented: Literal[True] = True
    adaptive_policy_explanation_metadata_implemented: Literal[True] = True
    execution_policy_metadata_used: Literal[True] = True
    dynamic_strategy_metadata_used: Literal[True] = True
    escalation_metadata_used: Literal[True] = True
    routing_explainability_metadata_used: Literal[True] = True
    policy_application_implemented: Literal[False] = False
    execution_policy_application_implemented: Literal[False] = False
    strategy_application_implemented: Literal[False] = False
    routing_application_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    execution_blocking_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
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
    def _plan_matches_explanations(self) -> Self:
        derived_ids = tuple(
            explanation.explanation_id for explanation in self.explanations
        )
        if len(set(derived_ids)) != len(derived_ids):
            raise ValueError("explanation_ids must be unique")
        if self.explanation_ids != derived_ids:
            raise ValueError("explanation_ids must match explanations")
        if self.source_surfaces != tuple(
            explanation.source_surface for explanation in self.explanations
        ):
            raise ValueError("source_surfaces must match explanations")
        if self.explanation_count != len(self.explanations):
            raise ValueError("explanation_count must match explanations")
        if self.source_surface_count != len(set(self.source_surfaces)):
            raise ValueError("source_surface_count must match explanations")
        primary = tuple(
            explanation
            for explanation in self.explanations
            if explanation.status == "primary"
        )
        if len(primary) != 1:
            raise ValueError("exactly one primary policy explanation is required")
        if self.primary_explanation_id != primary[0].explanation_id:
            raise ValueError("primary_explanation_id must match explanation")
        if self.supporting_explanation_ids != _explanation_ids_for_status(
            self.explanations,
            "supporting",
        ):
            raise ValueError("supporting_explanation_ids must match explanations")
        if self.guardrail_explanation_ids != _explanation_ids_for_status(
            self.explanations,
            "guardrail",
        ):
            raise ValueError("guardrail_explanation_ids must match explanations")
        if self.hitl_required_explanation_ids != tuple(
            explanation.explanation_id
            for explanation in self.explanations
            if explanation.source_hitl_required
        ):
            raise ValueError("hitl_required_explanation_ids must match explanations")
        if self.applied_policy_explanation_ids:
            raise ValueError("applied_policy_explanation_ids must remain empty")
        if self.supporting_explanation_count != len(self.supporting_explanation_ids):
            raise ValueError("supporting_explanation_count must match explanations")
        if self.guardrail_explanation_count != len(self.guardrail_explanation_ids):
            raise ValueError("guardrail_explanation_count must match explanations")
        if self.hitl_required_explanation_count != len(
            self.hitl_required_explanation_ids
        ):
            raise ValueError("hitl_required_explanation_count must match explanations")
        if self.highest_policy_explainability_score != max(
            explanation.policy_explainability_score for explanation in self.explanations
        ):
            raise ValueError("highest_policy_explainability_score must match")
        if self.policy_explainability_pressure != _plan_pressure(self.explanations):
            raise ValueError("policy_explainability_pressure must match explanations")
        for explanation in self.explanations:
            if explanation.route_name != self.route_name:
                raise ValueError("explanation route_name must match plan")
        return self


def explain_adaptive_policy(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_policy: ExecutionPolicyPlan | None = None,
    dynamic_strategy: AdaptiveExecutionStrategySelectionPlan | None = None,
    escalation: EscalationOptimizationPlan | None = None,
    routing_explainability: RoutingExplainabilityPlan | None = None,
) -> AdaptivePolicyExplainabilityPlan:
    """Explain adaptive policy metadata without applying policies."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    strategy_plan = dynamic_strategy or select_dynamic_execution_strategy(
        route=route_name,
        task_type=normalized_task_type,
    )
    execution_plan = execution_policy or evaluate_execution_policies(route=route_name)
    escalation_plan = escalation or optimize_escalation_policy(
        route=route_name,
        task_type=strategy_plan.task_type,
    )
    routing_plan = routing_explainability or explain_routing_decision(route=route_name)
    explanations = _explanations(
        route_name=route_name,
        task_type=strategy_plan.task_type,
        execution_policy=execution_plan,
        dynamic_strategy=strategy_plan,
        escalation=escalation_plan,
        routing_explainability=routing_plan,
    )
    return AdaptivePolicyExplainabilityPlan(
        route_name=route_name,
        task_type=strategy_plan.task_type,
        source_execution_policy_serialization_version=(
            execution_plan.serialization_version
        ),
        source_dynamic_strategy_serialization_version=strategy_plan.serialization_version,
        source_escalation_optimization_serialization_version=(
            escalation_plan.serialization_version
        ),
        source_routing_explainability_serialization_version=(
            routing_plan.serialization_version
        ),
        provider_ids=strategy_plan.provider_ids,
        execution_mode_ids=strategy_plan.execution_mode_ids,
        hybrid_policy_directions=strategy_plan.hybrid_policy_directions,
        explanations=explanations,
        explanation_ids=tuple(
            explanation.explanation_id for explanation in explanations
        ),
        source_surfaces=tuple(
            explanation.source_surface for explanation in explanations
        ),
        primary_explanation_id=_explanation_ids_for_status(
            explanations,
            "primary",
        )[0],
        supporting_explanation_ids=_explanation_ids_for_status(
            explanations,
            "supporting",
        ),
        guardrail_explanation_ids=_explanation_ids_for_status(
            explanations, "guardrail"
        ),
        hitl_required_explanation_ids=tuple(
            explanation.explanation_id
            for explanation in explanations
            if explanation.source_hitl_required
        ),
        applied_policy_explanation_ids=(),
        explanation_count=len(explanations),
        source_surface_count=len(
            {explanation.source_surface for explanation in explanations}
        ),
        supporting_explanation_count=len(
            _explanation_ids_for_status(explanations, "supporting")
        ),
        guardrail_explanation_count=len(
            _explanation_ids_for_status(explanations, "guardrail")
        ),
        hitl_required_explanation_count=sum(
            1 for explanation in explanations if explanation.source_hitl_required
        ),
        highest_policy_explainability_score=max(
            explanation.policy_explainability_score for explanation in explanations
        ),
        policy_explainability_pressure=_plan_pressure(explanations),
        advisory_actions=_plan_actions(explanations),
    )


def adaptive_policy_explanation_by_id(
    explanation_id: str,
    plan: AdaptivePolicyExplainabilityPlan | None = None,
) -> AdaptivePolicyExplanationRecord | None:
    """Return one adaptive policy explanation without applying policy."""

    source_plan = plan or explain_adaptive_policy()
    for explanation in source_plan.explanations:
        if explanation.explanation_id == explanation_id:
            return explanation
    return None


def adaptive_policy_explanations_for_status(
    status: AdaptivePolicyExplanationStatus,
    plan: AdaptivePolicyExplainabilityPlan | None = None,
) -> tuple[AdaptivePolicyExplanationRecord, ...]:
    """Return adaptive policy explanations by advisory status."""

    source_plan = plan or explain_adaptive_policy()
    return tuple(
        explanation
        for explanation in source_plan.explanations
        if explanation.status == status
    )


def _explanations(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_policy: ExecutionPolicyPlan,
    dynamic_strategy: AdaptiveExecutionStrategySelectionPlan,
    escalation: EscalationOptimizationPlan,
    routing_explainability: RoutingExplainabilityPlan,
) -> tuple[AdaptivePolicyExplanationRecord, ...]:
    return (
        _dynamic_strategy_explanation(route_name, task_type, dynamic_strategy),
        _execution_policy_explanation(route_name, task_type, execution_policy),
        _escalation_explanation(route_name, task_type, escalation),
        _routing_explanation(route_name, task_type, routing_explainability),
    )


def _dynamic_strategy_explanation(
    route_name: RouteName,
    task_type: TaskRoutingType,
    plan: AdaptiveExecutionStrategySelectionPlan,
) -> AdaptivePolicyExplanationRecord:
    signal = (
        plan.selected_strategy_score
        + len(plan.guardrail_strategy_ids) * 40
        + len(plan.fallback_strategy_ids) * 20
    )
    return _record(
        rank=1,
        surface="dynamic_execution_strategy",
        status="primary",
        route_name=route_name,
        task_type=task_type,
        source_serialization_version=plan.serialization_version,
        source_primary_record_id=plan.selected_strategy_id,
        source_record_count=plan.strategy_count,
        source_policy_posture=plan.selected_strategy_kind,
        source_hitl_required=plan.selected_strategy_hitl_required,
        source_guardrail_count=len(plan.guardrail_strategy_ids),
        source_signal_score=signal,
        explanation_summary=(
            "Dynamic execution strategy is the primary adaptive policy "
            "explanation because it combines strategy, hybrid, escalation, "
            "agent, cost, and latency posture."
        ),
        referenced_advisory_actions=plan.advisory_actions,
        evidence=(
            f"selected_strategy:{plan.selected_strategy_id}",
            f"selected_score:{plan.selected_strategy_score}",
            f"guardrail_strategies:{len(plan.guardrail_strategy_ids)}",
            f"fallback_strategies:{len(plan.fallback_strategy_ids)}",
        ),
    )


def _execution_policy_explanation(
    route_name: RouteName,
    task_type: TaskRoutingType,
    plan: ExecutionPolicyPlan,
) -> AdaptivePolicyExplanationRecord:
    signal = (
        plan.policy_count * 30
        + plan.guarded_policy_count * 60
        + plan.manual_review_policy_count * 80
    )
    return _record(
        rank=2,
        surface="execution_policy",
        status="supporting",
        route_name=route_name,
        task_type=task_type,
        source_serialization_version=plan.serialization_version,
        source_primary_record_id=plan.recommended_policy_id,
        source_record_count=plan.policy_count,
        source_policy_posture=plan.recommended_execution_policy_posture,
        source_hitl_required=plan.recommended_gate_status != "not_required",
        source_guardrail_count=(
            plan.guarded_policy_count + plan.manual_review_policy_count
        ),
        source_signal_score=signal,
        explanation_summary=(
            "Execution policy explains the runtime recommendation posture "
            "supporting adaptive policy review."
        ),
        referenced_advisory_actions=plan.advisory_actions,
        evidence=(
            f"recommended_policy:{plan.recommended_policy_id}",
            f"policy_posture:{plan.recommended_execution_policy_posture}",
            f"gate_status:{plan.recommended_gate_status}",
            f"policy_count:{plan.policy_count}",
        ),
    )


def _escalation_explanation(
    route_name: RouteName,
    task_type: TaskRoutingType,
    plan: EscalationOptimizationPlan,
) -> AdaptivePolicyExplanationRecord:
    signal = plan.highest_escalation_score + plan.hitl_required_decision_count * 40
    return _record(
        rank=3,
        surface="adaptive_escalation",
        status="guardrail",
        route_name=route_name,
        task_type=task_type,
        source_serialization_version=plan.serialization_version,
        source_primary_record_id=plan.highest_priority_decision_id,
        source_record_count=plan.decision_count,
        source_policy_posture=plan.optimized_escalation_posture,
        source_hitl_required=plan.hitl_required_decision_count > 0,
        source_guardrail_count=plan.hitl_required_decision_count,
        source_signal_score=signal,
        explanation_summary=(
            "Adaptive escalation explains the guarded HITL posture behind "
            "adaptive policy recommendations."
        ),
        referenced_advisory_actions=plan.advisory_actions,
        evidence=(
            f"highest_priority:{plan.highest_priority_decision_id}",
            f"escalation_posture:{plan.optimized_escalation_posture}",
            f"hitl_decisions:{plan.hitl_required_decision_count}",
            f"highest_score:{plan.highest_escalation_score}",
        ),
    )


def _routing_explanation(
    route_name: RouteName,
    task_type: TaskRoutingType,
    plan: RoutingExplainabilityPlan,
) -> AdaptivePolicyExplanationRecord:
    signal = plan.explanation_count * 30 + 40
    return _record(
        rank=4,
        surface="routing_explainability",
        status="supporting",
        route_name=route_name,
        task_type=task_type,
        source_serialization_version=plan.serialization_version,
        source_primary_record_id=plan.primary_explanation_id,
        source_record_count=plan.explanation_count,
        source_policy_posture=plan.recommended_model_profile_id,
        source_hitl_required=False,
        source_guardrail_count=0,
        source_signal_score=signal,
        explanation_summary=(
            "Routing explainability provides supporting policy evidence from "
            "model, local/cloud, hybrid, quality, and cost explanation records."
        ),
        referenced_advisory_actions=plan.advisory_actions,
        evidence=(
            f"primary_explanation:{plan.primary_explanation_id}",
            f"recommended_model:{plan.recommended_model_profile_id}",
            f"source_surfaces:{plan.source_surface_count}",
        ),
    )


def _record(
    *,
    rank: int,
    surface: AdaptivePolicyExplanationSurface,
    status: AdaptivePolicyExplanationStatus,
    route_name: RouteName,
    task_type: TaskRoutingType,
    source_serialization_version: str,
    source_primary_record_id: str,
    source_record_count: int,
    source_policy_posture: str,
    source_hitl_required: bool,
    source_guardrail_count: int,
    source_signal_score: int,
    explanation_summary: str,
    referenced_advisory_actions: tuple[str, ...],
    evidence: tuple[str, ...],
) -> AdaptivePolicyExplanationRecord:
    evidence_weight = _evidence_weight(source_record_count)
    hitl_weight = _hitl_weight(source_hitl_required)
    penalty = _guardrail_penalty(
        status=status,
        source_hitl_required=source_hitl_required,
        source_guardrail_count=source_guardrail_count,
    )
    return AdaptivePolicyExplanationRecord(
        explanation_id=f"adaptive_policy::{surface}",
        explanation_rank=rank,
        source_surface=surface,
        status=status,
        route_name=route_name,
        task_type=task_type,
        source_serialization_version=source_serialization_version,
        source_primary_record_id=source_primary_record_id,
        source_record_count=source_record_count,
        source_policy_posture=source_policy_posture,
        source_hitl_required=source_hitl_required,
        source_guardrail_count=source_guardrail_count,
        source_signal_score=source_signal_score,
        evidence_weight=evidence_weight,
        hitl_explanation_weight=hitl_weight,
        guardrail_penalty=penalty,
        policy_explainability_score=_policy_explainability_score(
            source_signal_score=source_signal_score,
            evidence_weight=evidence_weight,
            hitl_explanation_weight=hitl_weight,
            guardrail_penalty=penalty,
        ),
        explanation_summary=explanation_summary,
        referenced_advisory_actions=referenced_advisory_actions,
        evidence=evidence,
    )


def _evidence_weight(source_record_count: int) -> int:
    return min(240, source_record_count * 10)


def _hitl_weight(source_hitl_required: bool) -> int:
    return 70 if source_hitl_required else 0


def _guardrail_penalty(
    *,
    status: AdaptivePolicyExplanationStatus,
    source_hitl_required: bool,
    source_guardrail_count: int,
) -> int:
    penalty = min(80, source_guardrail_count * 10)
    if status == "guardrail":
        penalty += 90
    elif source_hitl_required:
        penalty += 20
    return min(180, penalty)


def _policy_explainability_score(
    *,
    source_signal_score: int,
    evidence_weight: int,
    hitl_explanation_weight: int,
    guardrail_penalty: int,
) -> int:
    return min(
        600,
        max(
            0,
            source_signal_score
            + evidence_weight
            + hitl_explanation_weight
            - guardrail_penalty,
        ),
    )


def _explanation_ids_for_status(
    explanations: tuple[AdaptivePolicyExplanationRecord, ...],
    status: AdaptivePolicyExplanationStatus,
) -> tuple[str, ...]:
    return tuple(
        explanation.explanation_id
        for explanation in explanations
        if explanation.status == status
    )


def _plan_pressure(
    explanations: tuple[AdaptivePolicyExplanationRecord, ...],
) -> Literal["low", "medium", "high", "guarded"]:
    if _explanation_ids_for_status(explanations, "guardrail"):
        return "guarded"
    highest = max(
        explanation.policy_explainability_score for explanation in explanations
    )
    if highest >= 500:
        return "high"
    if highest >= 250:
        return "medium"
    return "low"


def _plan_actions(
    explanations: tuple[AdaptivePolicyExplanationRecord, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose adaptive policy explanations as advisory metadata only.",
        "Keep applied policy explanation ids empty until explicit runtime authority exists.",
        "Preserve policy, strategy, routing, escalation, provider, workflow, storage, and output boundaries.",
    ]
    if _explanation_ids_for_status(explanations, "guardrail"):
        actions.append("Keep guarded policy explanations non-blocking.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
