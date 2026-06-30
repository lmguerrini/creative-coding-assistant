"""V5.5 advisory escalation optimization intelligence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_hybrid_workflow_optimizer import (
    HybridWorkflowOptimizationPlan,
    optimize_hybrid_workflow,
)
from creative_coding_assistant.orchestration.escalation_diagnostics import (
    EscalationDiagnostics,
    build_escalation_diagnostics,
    escalation_diagnostic_panel_by_id,
)
from creative_coding_assistant.orchestration.hitl_budget_gate import (
    HitlBudgetGatePlan,
    HitlBudgetGateStatus,
    evaluate_hitl_budget_gate,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    RoutingRiskBand,
    TaskRoutingType,
    UnavailableReasonCode,
    routing_execution_mode_registry,
)

EscalationOptimizationDecisionKind = Literal[
    "policy_diagnostics",
    "signal_thresholds",
    "budget_review",
    "hybrid_availability",
    "execution_mode_review",
    "adaptive_boundary",
]
EscalationOptimizationPosture = Literal[
    "requires_hitl",
    "review_recommended",
    "no_review_required",
]

ESCALATION_OPTIMIZATION_DECISION_SERIALIZATION_VERSION = (
    "adaptive_escalation_optimization_decision.v1"
)
ESCALATION_OPTIMIZATION_PLAN_SERIALIZATION_VERSION = (
    "adaptive_escalation_optimization_plan.v1"
)
ESCALATION_OPTIMIZER_AUTHORITY_BOUNDARY = (
    "V5.5 escalation optimization ranks advisory escalation, HITL, budget, "
    "hybrid availability, execution mode, and adaptive boundary metadata only; "
    "it does not evaluate policy, trigger escalation, emit human review "
    "requests, block execution, enforce budgets, invoke agents, select "
    "runtimes, route providers or models, call providers, control workflows, "
    "capture traces, write memory or storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "policy_evaluation",
    "escalation_triggering",
    "escalation_execution",
    "human_review_request",
    "hitl_request_emission",
    "execution_blocking",
    "budget_enforcement",
    "agent_invocation",
    "multi_agent_orchestration",
    "runtime_selection",
    "provider_or_model_routing",
    "provider_execution",
    "workflow_control",
    "retry_or_refinement_triggering",
    "trace_capture",
    "trace_emission",
    "memory_write",
    "persistent_storage_write",
    "generated_output_modification",
)


class EscalationOptimizationDecision(BaseModel):
    """One advisory escalation optimization decision."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    decision_id: str = Field(min_length=1, max_length=180)
    decision_kind: EscalationOptimizationDecisionKind
    source_surface_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=120)
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    posture: EscalationOptimizationPosture
    priority_rank: int = Field(ge=1, le=12)
    escalation_score: int = Field(ge=0, le=240)
    guardrail_signal_count: int = Field(ge=0, le=1000)
    hitl_required: bool
    budget_gate_status: HitlBudgetGateStatus | None = None
    risk_band: RoutingRiskBand
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=9,
    )
    reason_summary: str = Field(min_length=1, max_length=360)
    suggested_action: str = Field(min_length=1, max_length=360)
    fallback_summary: str = Field(min_length=1, max_length=360)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    escalation_optimizer_implemented: Literal[True] = True
    escalation_recommendation_implemented: Literal[True] = True
    policy_evaluation_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    escalation_execution_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    execution_blocking_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    multi_agent_orchestration_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    trace_emission_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["adaptive_escalation_optimization_decision.v1"] = (
        ESCALATION_OPTIMIZATION_DECISION_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _decision_matches_contract(self) -> Self:
        if self.decision_id != f"escalation_optimizer::{self.decision_kind}":
            raise ValueError("decision_id must match decision_kind")
        if self.posture == "requires_hitl" and not self.hitl_required:
            raise ValueError("requires_hitl decisions must set hitl_required")
        if self.unavailable_reason_codes and not self.hitl_required:
            raise ValueError("unavailable reasons require HITL")
        if self.budget_gate_status == "required" and self.posture != "requires_hitl":
            raise ValueError("required budget gates require HITL posture")
        return self


class EscalationOptimizationPlan(BaseModel):
    """Bounded V5.5 advisory escalation optimization plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["escalation_optimizer"] = "escalation_optimizer"
    serialization_version: Literal["adaptive_escalation_optimization_plan.v1"] = (
        ESCALATION_OPTIMIZATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ESCALATION_OPTIMIZER_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    task_type: TaskRoutingType
    source_escalation_diagnostics_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_hitl_budget_gate_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_hybrid_workflow_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    decisions: tuple[EscalationOptimizationDecision, ...] = Field(
        min_length=6,
        max_length=6,
    )
    decision_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    required_decision_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    review_decision_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    not_required_decision_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    highest_priority_decision_id: str = Field(min_length=1, max_length=180)
    optimized_escalation_posture: EscalationOptimizationPosture
    decision_count: int = Field(ge=6, le=6)
    hitl_required_decision_count: int = Field(ge=0, le=6)
    highest_escalation_score: int = Field(ge=0, le=240)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    escalation_optimizer_implemented: Literal[True] = True
    adaptive_escalation_policy_metadata_implemented: Literal[True] = True
    policy_evaluation_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    escalation_execution_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    execution_blocking_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    multi_agent_orchestration_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    trace_emission_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_decisions(self) -> Self:
        derived_decision_ids = tuple(decision.decision_id for decision in self.decisions)
        if len(set(derived_decision_ids)) != len(derived_decision_ids):
            raise ValueError("decision_ids must be unique")
        if self.decision_ids != derived_decision_ids:
            raise ValueError("decision_ids must match decisions")
        if self.decision_count != len(self.decisions):
            raise ValueError("decision_count must match decisions")
        if self.required_decision_ids != _decision_ids_for_posture(
            self.decisions,
            "requires_hitl",
        ):
            raise ValueError("required_decision_ids must match decisions")
        if self.review_decision_ids != _decision_ids_for_posture(
            self.decisions,
            "review_recommended",
        ):
            raise ValueError("review_decision_ids must match decisions")
        if self.not_required_decision_ids != _decision_ids_for_posture(
            self.decisions,
            "no_review_required",
        ):
            raise ValueError("not_required_decision_ids must match decisions")
        if self.hitl_required_decision_count != sum(
            1 for decision in self.decisions if decision.hitl_required
        ):
            raise ValueError("hitl_required_decision_count must match decisions")
        if self.highest_escalation_score != max(
            decision.escalation_score for decision in self.decisions
        ):
            raise ValueError("highest_escalation_score must match decisions")
        if self.highest_priority_decision_id != min(
            self.decisions,
            key=lambda decision: decision.priority_rank,
        ).decision_id:
            raise ValueError("highest_priority_decision_id must match decisions")
        if self.optimized_escalation_posture != _plan_posture(self.decisions):
            raise ValueError("optimized_escalation_posture must match decisions")
        return self


def optimize_escalation_policy(
    *,
    task_type: TaskRoutingType | str = "creative_coding",
    route: RouteName | str = RouteName.GENERATE,
    execution_mode_id: ExecutionModeId | str | None = None,
    diagnostics: EscalationDiagnostics | None = None,
    hitl_budget: HitlBudgetGatePlan | None = None,
    hybrid_workflow: HybridWorkflowOptimizationPlan | None = None,
) -> EscalationOptimizationPlan:
    """Recommend escalation posture without triggering escalation."""

    normalized_task_type = str(task_type).strip()
    hybrid_plan = hybrid_workflow or optimize_hybrid_workflow(
        task_type=normalized_task_type,
        route=route,
        execution_mode_id=execution_mode_id,
    )
    normalized_mode = str(execution_mode_id or hybrid_plan.candidates[0].execution_mode_id)
    diagnostics_source = diagnostics or build_escalation_diagnostics()
    budget_plan = hitl_budget or evaluate_hitl_budget_gate(route=route)
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")

    decisions = _decisions(
        task_type=hybrid_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        diagnostics=diagnostics_source,
        hitl_budget=budget_plan,
        hybrid_workflow=hybrid_plan,
    )
    return EscalationOptimizationPlan(
        task_type=hybrid_plan.task_type,
        source_escalation_diagnostics_serialization_version=(
            diagnostics_source.serialization_version
        ),
        source_hitl_budget_gate_serialization_version=budget_plan.serialization_version,
        source_hybrid_workflow_serialization_version=hybrid_plan.serialization_version,
        execution_mode_ids=execution_modes.execution_mode_ids,
        decisions=decisions,
        decision_ids=tuple(decision.decision_id for decision in decisions),
        required_decision_ids=_decision_ids_for_posture(decisions, "requires_hitl"),
        review_decision_ids=_decision_ids_for_posture(decisions, "review_recommended"),
        not_required_decision_ids=_decision_ids_for_posture(decisions, "no_review_required"),
        highest_priority_decision_id=min(
            decisions,
            key=lambda decision: decision.priority_rank,
        ).decision_id,
        optimized_escalation_posture=_plan_posture(decisions),
        decision_count=len(decisions),
        hitl_required_decision_count=sum(
            1 for decision in decisions if decision.hitl_required
        ),
        highest_escalation_score=max(decision.escalation_score for decision in decisions),
        advisory_actions=_plan_actions(decisions),
    )


def escalation_optimization_decision_by_id(
    decision_id: str,
    plan: EscalationOptimizationPlan | None = None,
) -> EscalationOptimizationDecision | None:
    """Return one escalation optimization decision without applying it."""

    source_plan = plan or optimize_escalation_policy()
    for decision in source_plan.decisions:
        if decision.decision_id == decision_id:
            return decision
    return None


def escalation_optimization_decisions_for_posture(
    posture: EscalationOptimizationPosture,
    plan: EscalationOptimizationPlan | None = None,
) -> tuple[EscalationOptimizationDecision, ...]:
    """Return escalation optimization decisions for one advisory posture."""

    source_plan = plan or optimize_escalation_policy()
    return tuple(decision for decision in source_plan.decisions if decision.posture == posture)


def _decisions(
    *,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    diagnostics: EscalationDiagnostics,
    hitl_budget: HitlBudgetGatePlan,
    hybrid_workflow: HybridWorkflowOptimizationPlan,
) -> tuple[EscalationOptimizationDecision, ...]:
    policy_panel = escalation_diagnostic_panel_by_id(
        "escalation_diagnostics::policy_rules",
        diagnostics,
    )
    signal_panel = escalation_diagnostic_panel_by_id(
        "escalation_diagnostics::signal_thresholds",
        diagnostics,
    )
    adaptive_panel = escalation_diagnostic_panel_by_id(
        "escalation_diagnostics::adaptive_boundary",
        diagnostics,
    )
    if policy_panel is None or signal_panel is None or adaptive_panel is None:
        raise ValueError("required escalation diagnostic panels are missing")
    budget_decision = next(
        decision
        for decision in hitl_budget.decisions
        if decision.gate_id == hitl_budget.recommended_gate_id
    )
    hybrid_recommended = next(
        candidate
        for candidate in hybrid_workflow.candidates
        if candidate.candidate_id == hybrid_workflow.recommended_candidate_id
    )
    return (
        _panel_decision(
            kind="policy_diagnostics",
            panel_id=policy_panel.panel_id,
            source_serialization_version=policy_panel.source_serialization_version,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            guardrail_signal_count=policy_panel.guardrail_signal_count,
            priority_rank=3,
        ),
        _panel_decision(
            kind="signal_thresholds",
            panel_id=signal_panel.panel_id,
            source_serialization_version=signal_panel.source_serialization_version,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            guardrail_signal_count=signal_panel.guardrail_signal_count,
            priority_rank=4,
        ),
        _budget_decision(
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            budget_decision=budget_decision,
            priority_rank=2,
        ),
        _hybrid_decision(
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            hybrid_workflow=hybrid_workflow,
            unavailable_reason_codes=hybrid_recommended.unavailable_reason_codes,
            risk_band=hybrid_recommended.risk_band,
            priority_rank=1,
        ),
        _execution_mode_decision(
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            hybrid_workflow=hybrid_workflow,
            priority_rank=5,
        ),
        _panel_decision(
            kind="adaptive_boundary",
            panel_id=adaptive_panel.panel_id,
            source_serialization_version=adaptive_panel.source_serialization_version,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            guardrail_signal_count=adaptive_panel.guardrail_signal_count,
            priority_rank=6,
        ),
    )


def _panel_decision(
    *,
    kind: EscalationOptimizationDecisionKind,
    panel_id: str,
    source_serialization_version: str,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    guardrail_signal_count: int,
    priority_rank: int,
) -> EscalationOptimizationDecision:
    posture: EscalationOptimizationPosture = (
        "review_recommended" if guardrail_signal_count else "no_review_required"
    )
    return EscalationOptimizationDecision(
        decision_id=f"escalation_optimizer::{kind}",
        decision_kind=kind,
        source_surface_id=panel_id,
        source_serialization_version=source_serialization_version,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        posture=posture,
        priority_rank=priority_rank,
        escalation_score=min(240, 40 + guardrail_signal_count),
        guardrail_signal_count=guardrail_signal_count,
        hitl_required=False,
        risk_band="medium" if guardrail_signal_count else "low",
        reason_summary="Escalation diagnostics contain guarded metadata.",
        suggested_action="Keep escalation posture visible for human review.",
        fallback_summary="Do not apply escalation; retain diagnostics as advisory metadata.",
        evidence=(
            f"source_panel:{panel_id}",
            f"guardrail_signals:{guardrail_signal_count}",
            "diagnostics are read-only",
        ),
    )


def _budget_decision(
    *,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    budget_decision: object,
    priority_rank: int,
) -> EscalationOptimizationDecision:
    gate_status = getattr(budget_decision, "gate_status")
    posture = _posture_for_gate(gate_status)
    hitl_required = posture == "requires_hitl" or gate_status == "review_recommended"
    return EscalationOptimizationDecision(
        decision_id="escalation_optimizer::budget_review",
        decision_kind="budget_review",
        source_surface_id=getattr(budget_decision, "gate_id"),
        source_serialization_version=getattr(budget_decision, "serialization_version"),
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        posture=posture,
        priority_rank=priority_rank,
        escalation_score=_score_for_posture(posture, 12),
        guardrail_signal_count=12 if gate_status != "not_required" else 0,
        hitl_required=hitl_required,
        budget_gate_status=gate_status,
        risk_band="medium" if gate_status != "not_required" else "low",
        reason_summary=getattr(budget_decision, "operator_review_reason"),
        suggested_action="Surface budget posture for review without blocking execution.",
        fallback_summary="Use non-budget escalation metadata if budget review is not required.",
        evidence=(
            f"budget_gate:{getattr(budget_decision, 'gate_id')}",
            f"gate_status:{gate_status}",
            "budget gate does not emit HITL",
        ),
    )


def _hybrid_decision(
    *,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    hybrid_workflow: HybridWorkflowOptimizationPlan,
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...],
    risk_band: RoutingRiskBand,
    priority_rank: int,
) -> EscalationOptimizationDecision:
    posture: EscalationOptimizationPosture = (
        "requires_hitl" if unavailable_reason_codes or risk_band == "high" else "review_recommended"
    )
    return EscalationOptimizationDecision(
        decision_id="escalation_optimizer::hybrid_availability",
        decision_kind="hybrid_availability",
        source_surface_id=hybrid_workflow.recommended_candidate_id,
        source_serialization_version=hybrid_workflow.serialization_version,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        posture=posture,
        priority_rank=priority_rank,
        escalation_score=_score_for_posture(posture, len(unavailable_reason_codes) * 8),
        guardrail_signal_count=len(unavailable_reason_codes),
        hitl_required=posture == "requires_hitl",
        risk_band=risk_band,
        unavailable_reason_codes=unavailable_reason_codes,
        reason_summary="Hybrid workflow availability metadata requires review before use.",
        suggested_action=hybrid_workflow.fallback.suggested_action,
        fallback_summary=hybrid_workflow.fallback.reason_summary,
        evidence=(
            f"recommended_candidate:{hybrid_workflow.recommended_candidate_id}",
            f"fallback_candidate:{hybrid_workflow.fallback.fallback_candidate_id}",
            f"unavailable_reasons:{len(unavailable_reason_codes)}",
        ),
    )


def _execution_mode_decision(
    *,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    hybrid_workflow: HybridWorkflowOptimizationPlan,
    priority_rank: int,
) -> EscalationOptimizationDecision:
    hitl_count = hybrid_workflow.hitl_required_candidate_count
    auto_with_hitl = execution_mode_id == "auto_mode" and bool(hitl_count)
    reason_codes = hybrid_workflow.fallback.reason_codes
    posture: EscalationOptimizationPosture
    if execution_mode_id == "manual_mode":
        posture = "review_recommended"
    elif execution_mode_id == "auto_mode" and hitl_count:
        posture = "requires_hitl"
    else:
        posture = "review_recommended" if hitl_count else "no_review_required"
    return EscalationOptimizationDecision(
        decision_id="escalation_optimizer::execution_mode_review",
        decision_kind="execution_mode_review",
        source_surface_id=hybrid_workflow.recommended_candidate_id,
        source_serialization_version=hybrid_workflow.serialization_version,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        posture=posture,
        priority_rank=priority_rank,
        escalation_score=_score_for_posture(posture, hitl_count * 6),
        guardrail_signal_count=hitl_count,
        hitl_required=posture == "requires_hitl" or bool(reason_codes),
        risk_band="high" if auto_with_hitl else "medium",
        unavailable_reason_codes=reason_codes,
        reason_summary="Execution mode constraints require review before application.",
        suggested_action="Keep Manual, Assisted, and Auto mode behavior advisory until explicitly integrated.",
        fallback_summary="Use assisted review posture when automatic boundaries are not safe.",
        evidence=(
            f"execution_mode:{execution_mode_id}",
            f"hybrid_hitl_candidates:{hitl_count}",
            "execution mode is not applied",
        ),
    )


def _posture_for_gate(gate_status: HitlBudgetGateStatus) -> EscalationOptimizationPosture:
    if gate_status == "required":
        return "requires_hitl"
    if gate_status == "review_recommended":
        return "review_recommended"
    return "no_review_required"


def _score_for_posture(
    posture: EscalationOptimizationPosture,
    extra: int,
) -> int:
    base = {
        "requires_hitl": 120,
        "review_recommended": 80,
        "no_review_required": 20,
    }[posture]
    return min(240, base + extra)


def _decision_ids_for_posture(
    decisions: tuple[EscalationOptimizationDecision, ...],
    posture: EscalationOptimizationPosture,
) -> tuple[str, ...]:
    return tuple(decision.decision_id for decision in decisions if decision.posture == posture)


def _plan_posture(
    decisions: tuple[EscalationOptimizationDecision, ...],
) -> EscalationOptimizationPosture:
    if _decision_ids_for_posture(decisions, "requires_hitl"):
        return "requires_hitl"
    if _decision_ids_for_posture(decisions, "review_recommended"):
        return "review_recommended"
    return "no_review_required"


def _plan_actions(
    decisions: tuple[EscalationOptimizationDecision, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose escalation optimization posture as advisory metadata only.",
        "Preserve HITL request emission, budget enforcement, execution blocking, routing, workflow, agent, trace, storage, and output boundaries.",
    ]
    if _decision_ids_for_posture(decisions, "requires_hitl"):
        actions.append("Require human approval before any future application of required escalation posture.")
    return tuple(actions)
