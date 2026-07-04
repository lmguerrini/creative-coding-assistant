"""V5.5 advisory workflow risk intelligence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_escalation_optimizer import (
    EscalationOptimizationDecision,
    EscalationOptimizationPlan,
    escalation_optimization_decision_by_id,
    optimize_escalation_policy,
)
from creative_coding_assistant.orchestration.dynamic_resource_allocation import (
    DynamicResourceAllocationCandidate,
    DynamicResourceAllocationPlan,
    allocate_dynamic_resources,
    dynamic_resource_allocation_by_id,
)
from creative_coding_assistant.orchestration.execution_confidence_engine import (
    ExecutionConfidencePlan,
    ExecutionConfidenceSignal,
    evaluate_execution_confidence,
    execution_confidence_signal_by_id,
)
from creative_coding_assistant.orchestration.performance_regression_detection import (
    PerformanceRegressionDetectionPlan,
    PerformanceRegressionSignal,
    detect_performance_regressions,
    performance_regression_signal_by_id,
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
from creative_coding_assistant.orchestration.workflow_self_tuning_policies import (
    WorkflowSelfTuningPolicy,
    WorkflowSelfTuningPolicyPlan,
    plan_workflow_self_tuning_policies,
    workflow_self_tuning_policy_by_id,
)

WorkflowRiskFactorKind = Literal[
    "execution_confidence_risk",
    "escalation_posture_risk",
    "resource_capacity_risk",
    "self_tuning_policy_risk",
    "performance_regression_risk",
    "provider_fallback_risk",
]
WorkflowRiskStatus = Literal["watch", "review_required", "guardrail"]
WorkflowRiskSeverity = Literal["low", "medium", "high", "guarded"]

WORKFLOW_RISK_FACTOR_SERIALIZATION_VERSION = "workflow_risk_factor.v1"
WORKFLOW_RISK_PLAN_SERIALIZATION_VERSION = "workflow_risk_plan.v1"
WORKFLOW_RISK_ENGINE_AUTHORITY_BOUNDARY = (
    "V5.5 workflow risk intelligence combines advisory execution confidence, "
    "adaptive escalation, dynamic resource allocation, workflow self-tuning, "
    "and performance regression metadata into inspectable workflow risk "
    "posture only; it does not apply risk decisions, execute mitigation, "
    "block workflows, enforce thresholds, emit alerts, trigger escalation or "
    "HITL requests, change provider or model routing, execute providers, "
    "probe local runtimes, scan or download local models, invoke agents, "
    "allocate resources, apply self-tuning, enforce budgets, control "
    "workflows, mutate workflow graphs, execute workflows, trigger retries, "
    "mutate prompts, write storage, modify generated output, or apply Runtime "
    "Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "risk_decision_application",
    "risk_mitigation_execution",
    "workflow_blocking",
    "threshold_enforcement",
    "alert_emission",
    "escalation_triggering",
    "human_review_request",
    "hitl_request_emission",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
    "provider_execution",
    "local_runtime_probe",
    "local_model_inventory_scan",
    "automatic_model_download",
    "agent_invocation",
    "resource_allocation",
    "workflow_self_tuning_application",
    "budget_enforcement",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "runtime_regression_detection",
    "benchmark_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class WorkflowRiskFactor(BaseModel):
    """One advisory workflow risk factor."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    factor_id: str = Field(min_length=1, max_length=180)
    factor_kind: WorkflowRiskFactorKind
    status: WorkflowRiskStatus
    severity: WorkflowRiskSeverity
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_execution_confidence_signal_id: str = Field(min_length=1, max_length=180)
    source_escalation_decision_id: str = Field(min_length=1, max_length=180)
    source_dynamic_resource_allocation_id: str = Field(min_length=1, max_length=180)
    source_self_tuning_policy_id: str = Field(min_length=1, max_length=180)
    source_performance_regression_signal_id: str = Field(min_length=1, max_length=180)
    provider_sequence: tuple[ProviderId, ...] = Field(min_length=1, max_length=4)
    model_profile_sequence: tuple[str, ...] = Field(min_length=1, max_length=4)
    hybrid_policy_direction: HybridRoutingPolicyDirection
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=9,
    )
    execution_confidence_score: int = Field(ge=0, le=100)
    escalation_score: int = Field(ge=0, le=240)
    dynamic_resource_score: int = Field(ge=0, le=500)
    self_tuning_score: int = Field(ge=0, le=600)
    performance_regression_score: int = Field(ge=0, le=3_000)
    unavailable_reason_count: int = Field(ge=0, le=9)
    guardrail_signal_count: int = Field(ge=0, le=200)
    risk_weight: int = Field(ge=0, le=160)
    workflow_risk_score: int = Field(ge=0, le=1_000)
    hitl_required: bool
    mitigation_summary: str = Field(min_length=1, max_length=360)
    fallback_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=30,
    )
    workflow_risk_engine_implemented: Literal[True] = True
    advisory_workflow_risk_metadata_implemented: Literal[True] = True
    execution_confidence_metadata_used: Literal[True] = True
    escalation_metadata_used: Literal[True] = True
    dynamic_resource_allocation_metadata_used: Literal[True] = True
    workflow_self_tuning_metadata_used: Literal[True] = True
    performance_regression_metadata_used: Literal[True] = True
    provider_intelligence_metadata_used: Literal[True] = True
    availability_awareness_metadata_used: Literal[True] = True
    manual_assisted_auto_mode_metadata_used: Literal[True] = True
    hybrid_transition_metadata_used: Literal[True] = True
    task_aware_category_metadata_used: Literal[True] = True
    execution_simulation_metadata_used: Literal[True] = True
    fallback_safety_metadata_used: Literal[True] = True
    risk_decision_application_implemented: Literal[False] = False
    risk_mitigation_execution_implemented: Literal[False] = False
    workflow_blocking_implemented: Literal[False] = False
    threshold_enforcement_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    local_model_inventory_scan_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    self_tuning_application_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    runtime_regression_detection_implemented: Literal[False] = False
    benchmark_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["workflow_risk_factor.v1"] = (
        WORKFLOW_RISK_FACTOR_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _factor_matches_contract(self) -> Self:
        if self.factor_id != f"workflow_risk::{self.factor_kind}":
            raise ValueError("factor_id must match factor_kind")
        if len(self.model_profile_sequence) != len(self.provider_sequence):
            raise ValueError("model_profile_sequence must match provider_sequence")
        if self.unavailable_reason_count != len(self.unavailable_reason_codes):
            raise ValueError("unavailable_reason_count must match reason codes")
        if self.workflow_risk_score != _workflow_risk_score(
            execution_confidence_score=self.execution_confidence_score,
            escalation_score=self.escalation_score,
            dynamic_resource_score=self.dynamic_resource_score,
            self_tuning_score=self.self_tuning_score,
            performance_regression_score=self.performance_regression_score,
            unavailable_reason_count=self.unavailable_reason_count,
            guardrail_signal_count=self.guardrail_signal_count,
            risk_weight=self.risk_weight,
        ):
            raise ValueError("workflow_risk_score must combine source scores")
        if self.severity != _risk_severity(self.workflow_risk_score, self.status):
            raise ValueError("severity must match score and status")
        if self.status == "guardrail" and not self.hitl_required:
            raise ValueError("guardrail risk factors require HITL posture")
        if self.unavailable_reason_codes and not self.hitl_required:
            raise ValueError("unavailable reasons require HITL posture")
        return self


class WorkflowRiskPlan(BaseModel):
    """Bounded V5.5 advisory workflow risk plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["workflow_risk_engine"] = "workflow_risk_engine"
    serialization_version: Literal["workflow_risk_plan.v1"] = (
        WORKFLOW_RISK_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=WORKFLOW_RISK_ENGINE_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_execution_confidence_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_escalation_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_dynamic_resource_allocation_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_workflow_self_tuning_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_performance_regression_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    provider_ids: tuple[ProviderId, ...] = Field(min_length=4, max_length=4)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    hybrid_policy_directions: tuple[HybridRoutingPolicyDirection, ...] = Field(
        min_length=4,
        max_length=4,
    )
    factors: tuple[WorkflowRiskFactor, ...] = Field(min_length=6, max_length=6)
    factor_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    watch_factor_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    review_required_factor_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    guardrail_factor_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    low_risk_factor_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    medium_risk_factor_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    high_risk_factor_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    guarded_risk_factor_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=6
    )
    hitl_required_factor_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    applied_mitigation_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    factor_count: int = Field(ge=6, le=6)
    review_required_factor_count: int = Field(ge=0, le=6)
    guardrail_factor_count: int = Field(ge=0, le=6)
    hitl_required_factor_count: int = Field(ge=0, le=6)
    highest_workflow_risk_score: int = Field(ge=0, le=1_000)
    highest_risk_factor_id: str = Field(min_length=1, max_length=180)
    overall_workflow_risk_score: int = Field(ge=0, le=1_000)
    overall_workflow_risk_severity: WorkflowRiskSeverity
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=30,
    )
    workflow_risk_engine_implemented: Literal[True] = True
    advisory_workflow_risk_metadata_implemented: Literal[True] = True
    execution_confidence_metadata_used: Literal[True] = True
    escalation_metadata_used: Literal[True] = True
    dynamic_resource_allocation_metadata_used: Literal[True] = True
    workflow_self_tuning_metadata_used: Literal[True] = True
    performance_regression_metadata_used: Literal[True] = True
    provider_intelligence_metadata_used: Literal[True] = True
    availability_awareness_metadata_used: Literal[True] = True
    manual_assisted_auto_mode_metadata_used: Literal[True] = True
    hybrid_transition_metadata_used: Literal[True] = True
    task_aware_category_metadata_used: Literal[True] = True
    execution_simulation_metadata_used: Literal[True] = True
    fallback_safety_metadata_used: Literal[True] = True
    risk_decision_application_implemented: Literal[False] = False
    risk_mitigation_execution_implemented: Literal[False] = False
    workflow_blocking_implemented: Literal[False] = False
    threshold_enforcement_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    local_model_inventory_scan_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    self_tuning_application_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    runtime_regression_detection_implemented: Literal[False] = False
    benchmark_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_factors(self) -> Self:
        derived_factor_ids = tuple(factor.factor_id for factor in self.factors)
        if len(set(derived_factor_ids)) != len(derived_factor_ids):
            raise ValueError("factor_ids must be unique")
        if self.factor_ids != derived_factor_ids:
            raise ValueError("factor_ids must match factors")
        if self.factor_count != len(self.factors):
            raise ValueError("factor_count must match factors")
        if self.watch_factor_ids != _factor_ids_for_status(self.factors, "watch"):
            raise ValueError("watch_factor_ids must match factors")
        if self.review_required_factor_ids != _factor_ids_for_status(
            self.factors,
            "review_required",
        ):
            raise ValueError("review_required_factor_ids must match factors")
        if self.guardrail_factor_ids != _factor_ids_for_status(
            self.factors,
            "guardrail",
        ):
            raise ValueError("guardrail_factor_ids must match factors")
        if self.low_risk_factor_ids != _factor_ids_for_severity(self.factors, "low"):
            raise ValueError("low_risk_factor_ids must match factors")
        if self.medium_risk_factor_ids != _factor_ids_for_severity(
            self.factors,
            "medium",
        ):
            raise ValueError("medium_risk_factor_ids must match factors")
        if self.high_risk_factor_ids != _factor_ids_for_severity(self.factors, "high"):
            raise ValueError("high_risk_factor_ids must match factors")
        if self.guarded_risk_factor_ids != _factor_ids_for_severity(
            self.factors,
            "guarded",
        ):
            raise ValueError("guarded_risk_factor_ids must match factors")
        if self.hitl_required_factor_ids != tuple(
            factor.factor_id for factor in self.factors if factor.hitl_required
        ):
            raise ValueError("hitl_required_factor_ids must match factors")
        if self.applied_mitigation_ids:
            raise ValueError("applied_mitigation_ids must remain empty")
        if self.review_required_factor_count != len(self.review_required_factor_ids):
            raise ValueError("review_required_factor_count must match factors")
        if self.guardrail_factor_count != len(self.guardrail_factor_ids):
            raise ValueError("guardrail_factor_count must match factors")
        if self.hitl_required_factor_count != len(self.hitl_required_factor_ids):
            raise ValueError("hitl_required_factor_count must match factors")
        if self.highest_workflow_risk_score != max(
            factor.workflow_risk_score for factor in self.factors
        ):
            raise ValueError("highest_workflow_risk_score must match factors")
        if (
            self.highest_risk_factor_id
            != max(
                self.factors,
                key=lambda factor: factor.workflow_risk_score,
            ).factor_id
        ):
            raise ValueError("highest_risk_factor_id must match factors")
        if self.overall_workflow_risk_score != _overall_workflow_risk_score(
            self.factors,
        ):
            raise ValueError("overall_workflow_risk_score must match factors")
        if self.overall_workflow_risk_severity != _overall_workflow_risk_severity(
            self.factors,
        ):
            raise ValueError("overall_workflow_risk_severity must match factors")
        for factor in self.factors:
            if factor.route_name != self.route_name:
                raise ValueError("factor route_name must match plan")
        return self


def evaluate_workflow_risk(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    execution_confidence: ExecutionConfidencePlan | None = None,
    escalation: EscalationOptimizationPlan | None = None,
    dynamic_resource_allocation: DynamicResourceAllocationPlan | None = None,
    self_tuning: WorkflowSelfTuningPolicyPlan | None = None,
    performance_regression: PerformanceRegressionDetectionPlan | None = None,
) -> WorkflowRiskPlan:
    """Evaluate advisory workflow risk without applying mitigation."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    confidence_plan = execution_confidence or evaluate_execution_confidence(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    normalized_mode = str(
        execution_mode_id or confidence_plan.signals[0].execution_mode_id
    )
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")

    escalation_plan = escalation or optimize_escalation_policy(
        route=route_name,
        task_type=confidence_plan.task_type,
        execution_mode_id=normalized_mode,
    )
    resource_plan = dynamic_resource_allocation or allocate_dynamic_resources(
        route=route_name,
        task_type=confidence_plan.task_type,
        execution_mode_id=normalized_mode,
    )
    tuning_plan = self_tuning or plan_workflow_self_tuning_policies(
        route=route_name,
        task_type=confidence_plan.task_type,
        execution_mode_id=normalized_mode,
        dynamic_resource_allocation=resource_plan,
    )
    regression_plan = performance_regression or detect_performance_regressions()
    factors = _factors(
        route_name=route_name,
        task_type=confidence_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        execution_confidence=confidence_plan,
        escalation=escalation_plan,
        dynamic_resource_allocation=resource_plan,
        self_tuning=tuning_plan,
        performance_regression=regression_plan,
    )
    return WorkflowRiskPlan(
        route_name=route_name,
        task_type=confidence_plan.task_type,
        source_execution_confidence_serialization_version=(
            confidence_plan.serialization_version
        ),
        source_escalation_optimization_serialization_version=(
            escalation_plan.serialization_version
        ),
        source_dynamic_resource_allocation_serialization_version=(
            resource_plan.serialization_version
        ),
        source_workflow_self_tuning_serialization_version=(
            tuning_plan.serialization_version
        ),
        source_performance_regression_serialization_version=(
            regression_plan.serialization_version
        ),
        provider_ids=confidence_plan.provider_ids,
        execution_mode_ids=execution_modes.execution_mode_ids,
        hybrid_policy_directions=confidence_plan.hybrid_policy_directions,
        factors=factors,
        factor_ids=tuple(factor.factor_id for factor in factors),
        watch_factor_ids=_factor_ids_for_status(factors, "watch"),
        review_required_factor_ids=_factor_ids_for_status(factors, "review_required"),
        guardrail_factor_ids=_factor_ids_for_status(factors, "guardrail"),
        low_risk_factor_ids=_factor_ids_for_severity(factors, "low"),
        medium_risk_factor_ids=_factor_ids_for_severity(factors, "medium"),
        high_risk_factor_ids=_factor_ids_for_severity(factors, "high"),
        guarded_risk_factor_ids=_factor_ids_for_severity(factors, "guarded"),
        hitl_required_factor_ids=tuple(
            factor.factor_id for factor in factors if factor.hitl_required
        ),
        applied_mitigation_ids=(),
        factor_count=len(factors),
        review_required_factor_count=len(
            _factor_ids_for_status(factors, "review_required")
        ),
        guardrail_factor_count=len(_factor_ids_for_status(factors, "guardrail")),
        hitl_required_factor_count=sum(1 for factor in factors if factor.hitl_required),
        highest_workflow_risk_score=max(
            factor.workflow_risk_score for factor in factors
        ),
        highest_risk_factor_id=max(
            factors,
            key=lambda factor: factor.workflow_risk_score,
        ).factor_id,
        overall_workflow_risk_score=_overall_workflow_risk_score(factors),
        overall_workflow_risk_severity=_overall_workflow_risk_severity(factors),
        advisory_actions=_plan_actions(factors),
    )


def workflow_risk_factor_by_id(
    factor_id: str,
    plan: WorkflowRiskPlan | None = None,
) -> WorkflowRiskFactor | None:
    """Return one workflow risk factor without applying mitigation."""

    source_plan = plan or evaluate_workflow_risk()
    for factor in source_plan.factors:
        if factor.factor_id == factor_id:
            return factor
    return None


def workflow_risk_factors_for_severity(
    severity: WorkflowRiskSeverity,
    plan: WorkflowRiskPlan | None = None,
) -> tuple[WorkflowRiskFactor, ...]:
    """Return workflow risk factors by advisory severity."""

    source_plan = plan or evaluate_workflow_risk()
    return tuple(
        factor for factor in source_plan.factors if factor.severity == severity
    )


def _factors(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    execution_confidence: ExecutionConfidencePlan,
    escalation: EscalationOptimizationPlan,
    dynamic_resource_allocation: DynamicResourceAllocationPlan,
    self_tuning: WorkflowSelfTuningPolicyPlan,
    performance_regression: PerformanceRegressionDetectionPlan,
) -> tuple[WorkflowRiskFactor, ...]:
    return (
        _factor(
            kind="execution_confidence_risk",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            execution_confidence=execution_confidence,
            confidence_signal_id="execution_confidence::strategy_execution_confidence",
            escalation=escalation,
            escalation_decision_id="escalation_optimizer::execution_mode_review",
            dynamic_resource_allocation=dynamic_resource_allocation,
            resource_allocation_id=(
                "dynamic_resource_allocation::throughput_capacity_utilization"
            ),
            self_tuning=self_tuning,
            self_tuning_policy_id="workflow_self_tuning::load_self_tuning_policy",
            performance_regression=performance_regression,
            performance_signal_id="performance_regression::prediction_regression_risk",
            risk_weight=40,
        ),
        _factor(
            kind="escalation_posture_risk",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            execution_confidence=execution_confidence,
            confidence_signal_id="execution_confidence::agent_allocation_confidence",
            escalation=escalation,
            escalation_decision_id="escalation_optimizer::hybrid_availability",
            dynamic_resource_allocation=dynamic_resource_allocation,
            resource_allocation_id=(
                "dynamic_resource_allocation::reasoning_budget_utilization"
            ),
            self_tuning=self_tuning,
            self_tuning_policy_id="workflow_self_tuning::retry_self_tuning_policy",
            performance_regression=performance_regression,
            performance_signal_id="performance_regression::reasoning_budget_pressure",
            risk_weight=80,
        ),
        _factor(
            kind="resource_capacity_risk",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            execution_confidence=execution_confidence,
            confidence_signal_id="execution_confidence::resource_capacity_confidence",
            escalation=escalation,
            escalation_decision_id="escalation_optimizer::budget_review",
            dynamic_resource_allocation=dynamic_resource_allocation,
            resource_allocation_id=(
                "dynamic_resource_allocation::benchmark_workload_utilization"
            ),
            self_tuning=self_tuning,
            self_tuning_policy_id="workflow_self_tuning::resource_self_tuning_policy",
            performance_regression=performance_regression,
            performance_signal_id="performance_regression::benchmark_regression_risk",
            risk_weight=65,
        ),
        _factor(
            kind="self_tuning_policy_risk",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            execution_confidence=execution_confidence,
            confidence_signal_id="execution_confidence::workflow_self_tuning_confidence",
            escalation=escalation,
            escalation_decision_id="escalation_optimizer::adaptive_boundary",
            dynamic_resource_allocation=dynamic_resource_allocation,
            resource_allocation_id=(
                "dynamic_resource_allocation::profiling_scope_utilization"
            ),
            self_tuning=self_tuning,
            self_tuning_policy_id="workflow_self_tuning::resource_self_tuning_policy",
            performance_regression=performance_regression,
            performance_signal_id="performance_regression::reasoning_budget_pressure",
            risk_weight=50,
        ),
        _factor(
            kind="performance_regression_risk",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            execution_confidence=execution_confidence,
            confidence_signal_id="execution_confidence::fallback_safety_confidence",
            escalation=escalation,
            escalation_decision_id="escalation_optimizer::budget_review",
            dynamic_resource_allocation=dynamic_resource_allocation,
            resource_allocation_id=(
                "dynamic_resource_allocation::regression_baseline_utilization"
            ),
            self_tuning=self_tuning,
            self_tuning_policy_id="workflow_self_tuning::strategy_guardrail_policy",
            performance_regression=performance_regression,
            performance_signal_id="performance_regression::measurement_boundary",
            risk_weight=100,
        ),
        _factor(
            kind="provider_fallback_risk",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            execution_confidence=execution_confidence,
            confidence_signal_id="execution_confidence::provider_availability_confidence",
            escalation=escalation,
            escalation_decision_id="escalation_optimizer::hybrid_availability",
            dynamic_resource_allocation=dynamic_resource_allocation,
            resource_allocation_id="dynamic_resource_allocation::runtime_resource_boundary",
            self_tuning=self_tuning,
            self_tuning_policy_id="workflow_self_tuning::strategy_guardrail_policy",
            performance_regression=performance_regression,
            performance_signal_id="performance_regression::measurement_boundary",
            risk_weight=120,
        ),
    )


def _factor(
    *,
    kind: WorkflowRiskFactorKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    execution_confidence: ExecutionConfidencePlan,
    confidence_signal_id: str,
    escalation: EscalationOptimizationPlan,
    escalation_decision_id: str,
    dynamic_resource_allocation: DynamicResourceAllocationPlan,
    resource_allocation_id: str,
    self_tuning: WorkflowSelfTuningPolicyPlan,
    self_tuning_policy_id: str,
    performance_regression: PerformanceRegressionDetectionPlan,
    performance_signal_id: str,
    risk_weight: int,
) -> WorkflowRiskFactor:
    confidence = _required_confidence_signal(confidence_signal_id, execution_confidence)
    escalation_decision = _required_escalation_decision(
        escalation_decision_id, escalation
    )
    resource = _required_resource_allocation(
        resource_allocation_id,
        dynamic_resource_allocation,
    )
    policy = _required_self_tuning_policy(self_tuning_policy_id, self_tuning)
    regression = _required_performance_signal(
        performance_signal_id,
        performance_regression,
    )
    guardrail_count = _guardrail_count(
        confidence=confidence,
        escalation=escalation_decision,
        resource=resource,
        policy=policy,
        regression=regression,
    )
    status = _status(kind, confidence, resource, policy, regression)
    score = _workflow_risk_score(
        execution_confidence_score=confidence.execution_confidence_score,
        escalation_score=escalation_decision.escalation_score,
        dynamic_resource_score=resource.dynamic_resource_score,
        self_tuning_score=policy.self_tuning_score,
        performance_regression_score=regression.advisory_regression_score,
        unavailable_reason_count=len(confidence.unavailable_reason_codes),
        guardrail_signal_count=guardrail_count,
        risk_weight=risk_weight,
    )
    hitl_required = (
        confidence.hitl_required
        or escalation_decision.hitl_required
        or resource.hitl_required
        or policy.hitl_required
        or status == "guardrail"
    )
    return WorkflowRiskFactor(
        factor_id=f"workflow_risk::{kind}",
        factor_kind=kind,
        status=status,
        severity=_risk_severity(score, status),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_execution_confidence_signal_id=confidence.signal_id,
        source_escalation_decision_id=escalation_decision.decision_id,
        source_dynamic_resource_allocation_id=resource.allocation_id,
        source_self_tuning_policy_id=policy.policy_id,
        source_performance_regression_signal_id=regression.signal_id,
        provider_sequence=confidence.provider_sequence,
        model_profile_sequence=confidence.model_profile_sequence,
        hybrid_policy_direction=confidence.hybrid_policy_direction,
        unavailable_reason_codes=confidence.unavailable_reason_codes,
        execution_confidence_score=confidence.execution_confidence_score,
        escalation_score=escalation_decision.escalation_score,
        dynamic_resource_score=resource.dynamic_resource_score,
        self_tuning_score=policy.self_tuning_score,
        performance_regression_score=regression.advisory_regression_score,
        unavailable_reason_count=len(confidence.unavailable_reason_codes),
        guardrail_signal_count=guardrail_count,
        risk_weight=risk_weight,
        workflow_risk_score=score,
        hitl_required=hitl_required,
        mitigation_summary=_mitigation_summary(kind, status),
        fallback_summary=_fallback_summary(kind),
        advisory_actions=_factor_actions(kind),
        evidence=(
            f"confidence:{confidence.signal_id}",
            f"escalation:{escalation_decision.decision_id}",
            f"resource:{resource.allocation_id}",
            f"self_tuning:{policy.policy_id}",
            f"performance_regression:{regression.signal_id}",
            f"guardrail_signals:{guardrail_count}",
            f"hitl_required:{hitl_required}",
        ),
    )


def _workflow_risk_score(
    *,
    execution_confidence_score: int,
    escalation_score: int,
    dynamic_resource_score: int,
    self_tuning_score: int,
    performance_regression_score: int,
    unavailable_reason_count: int,
    guardrail_signal_count: int,
    risk_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            (100 - execution_confidence_score) * 4
            + escalation_score * 2
            + max(0, 500 - dynamic_resource_score) // 2
            + self_tuning_score // 3
            + performance_regression_score // 10
            + unavailable_reason_count * 35
            + guardrail_signal_count * 20
            + risk_weight,
        ),
    )


def _status(
    kind: WorkflowRiskFactorKind,
    confidence: ExecutionConfidenceSignal,
    resource: DynamicResourceAllocationCandidate,
    policy: WorkflowSelfTuningPolicy,
    regression: PerformanceRegressionSignal,
) -> WorkflowRiskStatus:
    if kind in {"performance_regression_risk", "provider_fallback_risk"}:
        return "guardrail"
    if (
        confidence.status == "guardrail"
        or resource.allocation_status == "boundary_guardrail"
        or policy.status == "guardrail"
        or regression.status == "baseline_guardrail"
    ):
        return "guardrail"
    if confidence.hitl_required or resource.hitl_required or policy.hitl_required:
        return "review_required"
    return "watch"


def _risk_severity(
    score: int,
    status: WorkflowRiskStatus,
) -> WorkflowRiskSeverity:
    if status == "guardrail":
        return "guarded"
    if score >= 650:
        return "high"
    if score >= 350:
        return "medium"
    return "low"


def _overall_workflow_risk_score(
    factors: tuple[WorkflowRiskFactor, ...],
) -> int:
    return sum(factor.workflow_risk_score for factor in factors) // len(factors)


def _overall_workflow_risk_severity(
    factors: tuple[WorkflowRiskFactor, ...],
) -> WorkflowRiskSeverity:
    if _factor_ids_for_status(factors, "guardrail"):
        return "guarded"
    return _risk_severity(_overall_workflow_risk_score(factors), "watch")


def _factor_ids_for_status(
    factors: tuple[WorkflowRiskFactor, ...],
    status: WorkflowRiskStatus,
) -> tuple[str, ...]:
    return tuple(factor.factor_id for factor in factors if factor.status == status)


def _factor_ids_for_severity(
    factors: tuple[WorkflowRiskFactor, ...],
    severity: WorkflowRiskSeverity,
) -> tuple[str, ...]:
    return tuple(factor.factor_id for factor in factors if factor.severity == severity)


def _guardrail_count(
    *,
    confidence: ExecutionConfidenceSignal,
    escalation: EscalationOptimizationDecision,
    resource: DynamicResourceAllocationCandidate,
    policy: WorkflowSelfTuningPolicy,
    regression: PerformanceRegressionSignal,
) -> int:
    return (
        (1 if confidence.status == "guardrail" else 0)
        + escalation.guardrail_signal_count
        + (
            1
            if resource.allocation_status
            in {"capacity_guardrail", "boundary_guardrail"}
            else 0
        )
        + (1 if policy.status == "guardrail" else 0)
        + (1 if regression.status == "baseline_guardrail" else 0)
    )


def _required_confidence_signal(
    signal_id: str,
    plan: ExecutionConfidencePlan,
) -> ExecutionConfidenceSignal:
    signal = execution_confidence_signal_by_id(signal_id, plan)
    if signal is None:
        raise ValueError("required workflow risk confidence metadata is missing")
    return signal


def _required_escalation_decision(
    decision_id: str,
    plan: EscalationOptimizationPlan,
) -> EscalationOptimizationDecision:
    decision = escalation_optimization_decision_by_id(decision_id, plan)
    if decision is None:
        raise ValueError("required workflow risk escalation metadata is missing")
    return decision


def _required_resource_allocation(
    allocation_id: str,
    plan: DynamicResourceAllocationPlan,
) -> DynamicResourceAllocationCandidate:
    allocation = dynamic_resource_allocation_by_id(allocation_id, plan)
    if allocation is None:
        raise ValueError("required workflow risk resource metadata is missing")
    return allocation


def _required_self_tuning_policy(
    policy_id: str,
    plan: WorkflowSelfTuningPolicyPlan,
) -> WorkflowSelfTuningPolicy:
    policy = workflow_self_tuning_policy_by_id(policy_id, plan)
    if policy is None:
        raise ValueError("required workflow risk self-tuning metadata is missing")
    return policy


def _required_performance_signal(
    signal_id: str,
    plan: PerformanceRegressionDetectionPlan,
) -> PerformanceRegressionSignal:
    signal = performance_regression_signal_by_id(signal_id, plan)
    if signal is None:
        raise ValueError("required workflow risk performance metadata is missing")
    return signal


def _mitigation_summary(
    kind: WorkflowRiskFactorKind,
    status: WorkflowRiskStatus,
) -> str:
    if status == "guardrail":
        return "Keep workflow risk as guardrail metadata without mitigation execution."
    if kind == "execution_confidence_risk":
        return "Review confidence risk before any future adaptive execution behavior."
    if kind == "escalation_posture_risk":
        return "Review escalation posture without emitting HITL or blocking execution."
    if kind == "resource_capacity_risk":
        return (
            "Review resource capacity risk without allocation or capacity enforcement."
        )
    if kind == "self_tuning_policy_risk":
        return "Review self-tuning policy risk without workflow control."
    return "Review performance regression risk without live detection or alerts."


def _fallback_summary(kind: WorkflowRiskFactorKind) -> str:
    return {
        "execution_confidence_risk": (
            "Fallback to advisory confidence review if execution confidence is guarded."
        ),
        "escalation_posture_risk": (
            "Fallback to human-readable escalation metadata without triggering escalation."
        ),
        "resource_capacity_risk": (
            "Fallback to capacity guardrail visibility without resource changes."
        ),
        "self_tuning_policy_risk": (
            "Fallback to self-tuning recommendations without applying policies."
        ),
        "performance_regression_risk": (
            "Fallback to baseline guardrail visibility without regression detection."
        ),
        "provider_fallback_risk": (
            "Fallback to provider availability metadata without provider switching."
        ),
    }[kind]


def _factor_actions(kind: WorkflowRiskFactorKind) -> tuple[str, ...]:
    return (
        f"Surface {kind} as advisory workflow risk metadata.",
        "Keep risk mitigation, workflow blocking, routing, execution, agents, resources, self-tuning, HITL emission, storage, and output behavior disabled.",  # noqa: E501
    )


def _plan_actions(
    factors: tuple[WorkflowRiskFactor, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose workflow risk posture as advisory metadata only.",
        "Keep applied mitigation ids empty until explicit runtime authority exists.",
        "Preserve risk mitigation, workflow blocking, thresholds, alerts, escalation, routing, execution, agents, resources, storage, and output boundaries.",  # noqa: E501
    ]
    if any(factor.hitl_required for factor in factors):
        actions.append("Require review before any future workflow risk behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
