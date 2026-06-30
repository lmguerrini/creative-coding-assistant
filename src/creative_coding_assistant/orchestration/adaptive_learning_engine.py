"""V6.1 advisory adaptive learning engine."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.execution_confidence_engine import (
    ExecutionConfidencePlan,
    ExecutionConfidenceSignal,
    evaluate_execution_confidence,
    execution_confidence_signal_by_id,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)
from creative_coding_assistant.orchestration.workflow_risk_engine import (
    WorkflowRiskFactor,
    WorkflowRiskPlan,
    evaluate_workflow_risk,
    workflow_risk_factor_by_id,
)
from creative_coding_assistant.orchestration.workflow_self_tuning_policies import (
    WorkflowSelfTuningPolicy,
    WorkflowSelfTuningPolicyPlan,
    plan_workflow_self_tuning_policies,
    workflow_self_tuning_policy_by_id,
)

AdaptiveLearningSignalKind = Literal[
    "workflow_pattern_learning",
    "strategy_pattern_learning",
    "routing_boundary_learning",
    "governance_feedback_learning",
    "runtime_guardrail_learning",
]
AdaptiveLearningSignalStatus = Literal["candidate", "review_required", "guardrail"]
AdaptiveLearningPriorityBand = Literal["standard", "elevated", "critical", "guarded"]
AdaptiveLearningPosture = Literal["candidate", "review_required", "guarded"]

ADAPTIVE_LEARNING_SIGNAL_SERIALIZATION_VERSION = "adaptive_learning_signal.v1"
ADAPTIVE_LEARNING_PLAN_SERIALIZATION_VERSION = "adaptive_learning_plan.v1"
ADAPTIVE_LEARNING_ENGINE_AUTHORITY_BOUNDARY = (
    "V6.1 adaptive learning engine combines existing advisory execution "
    "confidence, workflow risk, and workflow self-tuning metadata into "
    "inspectable learning candidates only; it does not persist learning "
    "memory, apply feedback, update policies, mutate strategies, change "
    "provider or model routing, execute providers, probe local runtimes, "
    "download models, invoke agents, allocate resources, emit HITL requests, "
    "control workflows, mutate workflow graphs, trigger retries or "
    "refinements, mutate prompts, write storage, modify generated output, or "
    "apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "learning_memory_persistence",
    "learning_feedback_application",
    "learning_policy_mutation",
    "strategy_mutation",
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
    "human_review_request",
    "hitl_request_emission",
    "budget_enforcement",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class AdaptiveLearningSignal(BaseModel):
    """One advisory learning signal derived from controlled V5 metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: AdaptiveLearningSignalKind
    status: AdaptiveLearningSignalStatus
    priority_band: AdaptiveLearningPriorityBand
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_workflow_risk_factor_id: str = Field(min_length=1, max_length=180)
    source_execution_confidence_signal_id: str = Field(min_length=1, max_length=180)
    source_self_tuning_policy_id: str = Field(min_length=1, max_length=180)
    workflow_risk_score: int = Field(ge=0, le=1_000)
    execution_confidence_score: int = Field(ge=0, le=100)
    self_tuning_score: int = Field(ge=0, le=600)
    unavailable_reason_count: int = Field(ge=0, le=9)
    guardrail_signal_count: int = Field(ge=0, le=200)
    learning_weight: int = Field(ge=0, le=240)
    learning_priority_score: int = Field(ge=0, le=1_000)
    hitl_required: bool
    pattern_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    learning_summary: str = Field(min_length=1, max_length=360)
    proposed_learning_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    adaptive_learning_engine_implemented: Literal[True] = True
    learning_signal_metadata_implemented: Literal[True] = True
    execution_confidence_metadata_used: Literal[True] = True
    workflow_risk_metadata_used: Literal[True] = True
    workflow_self_tuning_metadata_used: Literal[True] = True
    learning_memory_persistence_implemented: Literal[False] = False
    learning_feedback_application_implemented: Literal[False] = False
    learning_policy_mutation_implemented: Literal[False] = False
    strategy_mutation_implemented: Literal[False] = False
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
    hitl_request_emitted: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["adaptive_learning_signal.v1"] = (
        ADAPTIVE_LEARNING_SIGNAL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"adaptive_learning::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.learning_priority_score != _learning_priority_score(
            workflow_risk_score=self.workflow_risk_score,
            execution_confidence_score=self.execution_confidence_score,
            self_tuning_score=self.self_tuning_score,
            unavailable_reason_count=self.unavailable_reason_count,
            guardrail_signal_count=self.guardrail_signal_count,
            learning_weight=self.learning_weight,
        ):
            raise ValueError("learning_priority_score must combine source scores")
        if self.priority_band != _priority_band(
            self.learning_priority_score,
            self.status,
        ):
            raise ValueError("priority_band must match score and status")
        if self.status == "guardrail" and not self.hitl_required:
            raise ValueError("guardrail learning signals require HITL posture")
        return self


class AdaptiveLearningPlan(BaseModel):
    """Bounded V6.1 advisory adaptive learning plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["adaptive_learning_engine"] = "adaptive_learning_engine"
    serialization_version: Literal["adaptive_learning_plan.v1"] = (
        ADAPTIVE_LEARNING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ADAPTIVE_LEARNING_ENGINE_AUTHORITY_BOUNDARY,
        max_length=1900,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_execution_confidence_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_workflow_risk_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_workflow_self_tuning_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[AdaptiveLearningSignal, ...] = Field(min_length=5, max_length=5)
    signal_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    candidate_signal_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    review_required_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guardrail_signal_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    standard_priority_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    elevated_priority_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    critical_priority_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_priority_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    hitl_required_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    persisted_learning_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    applied_learning_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    signal_count: int = Field(ge=5, le=5)
    review_required_signal_count: int = Field(ge=0, le=5)
    guardrail_signal_count: int = Field(ge=0, le=5)
    hitl_required_signal_count: int = Field(ge=0, le=5)
    highest_learning_priority_score: int = Field(ge=0, le=1_000)
    overall_learning_priority_score: int = Field(ge=0, le=1_000)
    overall_learning_posture: AdaptiveLearningPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    adaptive_learning_engine_implemented: Literal[True] = True
    learning_signal_metadata_implemented: Literal[True] = True
    execution_confidence_metadata_used: Literal[True] = True
    workflow_risk_metadata_used: Literal[True] = True
    workflow_self_tuning_metadata_used: Literal[True] = True
    learning_memory_persistence_implemented: Literal[False] = False
    learning_feedback_application_implemented: Literal[False] = False
    learning_policy_mutation_implemented: Literal[False] = False
    strategy_mutation_implemented: Literal[False] = False
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
    hitl_request_emitted: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
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
    def _plan_matches_signals(self) -> Self:
        derived_signal_ids = tuple(signal.signal_id for signal in self.signals)
        if len(set(derived_signal_ids)) != len(derived_signal_ids):
            raise ValueError("signal_ids must be unique")
        if self.signal_ids != derived_signal_ids:
            raise ValueError("signal_ids must match signals")
        if self.signal_count != len(self.signals):
            raise ValueError("signal_count must match signals")
        if self.candidate_signal_ids != _signal_ids_for_status(
            self.signals,
            "candidate",
        ):
            raise ValueError("candidate_signal_ids must match signals")
        if self.review_required_signal_ids != _signal_ids_for_status(
            self.signals,
            "review_required",
        ):
            raise ValueError("review_required_signal_ids must match signals")
        if self.guardrail_signal_ids != _signal_ids_for_status(
            self.signals,
            "guardrail",
        ):
            raise ValueError("guardrail_signal_ids must match signals")
        if self.standard_priority_signal_ids != _signal_ids_for_priority(
            self.signals,
            "standard",
        ):
            raise ValueError("standard_priority_signal_ids must match signals")
        if self.elevated_priority_signal_ids != _signal_ids_for_priority(
            self.signals,
            "elevated",
        ):
            raise ValueError("elevated_priority_signal_ids must match signals")
        if self.critical_priority_signal_ids != _signal_ids_for_priority(
            self.signals,
            "critical",
        ):
            raise ValueError("critical_priority_signal_ids must match signals")
        if self.guarded_priority_signal_ids != _signal_ids_for_priority(
            self.signals,
            "guarded",
        ):
            raise ValueError("guarded_priority_signal_ids must match signals")
        if self.hitl_required_signal_ids != tuple(
            signal.signal_id for signal in self.signals if signal.hitl_required
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.persisted_learning_signal_ids:
            raise ValueError("persisted_learning_signal_ids must remain empty")
        if self.applied_learning_signal_ids:
            raise ValueError("applied_learning_signal_ids must remain empty")
        if self.review_required_signal_count != len(self.review_required_signal_ids):
            raise ValueError("review_required_signal_count must match signals")
        if self.guardrail_signal_count != len(self.guardrail_signal_ids):
            raise ValueError("guardrail_signal_count must match signals")
        if self.hitl_required_signal_count != len(self.hitl_required_signal_ids):
            raise ValueError("hitl_required_signal_count must match signals")
        if self.highest_learning_priority_score != max(
            signal.learning_priority_score for signal in self.signals
        ):
            raise ValueError("highest_learning_priority_score must match signals")
        if self.overall_learning_priority_score != _overall_learning_priority_score(
            self.signals,
        ):
            raise ValueError("overall_learning_priority_score must match signals")
        if self.overall_learning_posture != _overall_learning_posture(self.signals):
            raise ValueError("overall_learning_posture must match signals")
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
        return self


def evaluate_adaptive_learning_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    execution_confidence: ExecutionConfidencePlan | None = None,
    workflow_risk: WorkflowRiskPlan | None = None,
    self_tuning: WorkflowSelfTuningPolicyPlan | None = None,
) -> AdaptiveLearningPlan:
    """Evaluate adaptive learning candidates without applying learning."""

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

    tuning_plan = self_tuning or plan_workflow_self_tuning_policies(
        route=route_name,
        task_type=confidence_plan.task_type,
        execution_mode_id=normalized_mode,
    )
    risk_plan = workflow_risk or evaluate_workflow_risk(
        route=route_name,
        task_type=confidence_plan.task_type,
        execution_mode_id=normalized_mode,
        execution_confidence=confidence_plan,
        self_tuning=tuning_plan,
    )
    signals = _signals(
        route_name=route_name,
        task_type=confidence_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        execution_confidence=confidence_plan,
        workflow_risk=risk_plan,
        self_tuning=tuning_plan,
    )
    return AdaptiveLearningPlan(
        route_name=route_name,
        task_type=confidence_plan.task_type,
        source_execution_confidence_serialization_version=(
            confidence_plan.serialization_version
        ),
        source_workflow_risk_serialization_version=risk_plan.serialization_version,
        source_workflow_self_tuning_serialization_version=(
            tuning_plan.serialization_version
        ),
        execution_mode_ids=execution_modes.execution_mode_ids,
        signals=signals,
        signal_ids=tuple(signal.signal_id for signal in signals),
        candidate_signal_ids=_signal_ids_for_status(signals, "candidate"),
        review_required_signal_ids=_signal_ids_for_status(signals, "review_required"),
        guardrail_signal_ids=_signal_ids_for_status(signals, "guardrail"),
        standard_priority_signal_ids=_signal_ids_for_priority(signals, "standard"),
        elevated_priority_signal_ids=_signal_ids_for_priority(signals, "elevated"),
        critical_priority_signal_ids=_signal_ids_for_priority(signals, "critical"),
        guarded_priority_signal_ids=_signal_ids_for_priority(signals, "guarded"),
        hitl_required_signal_ids=tuple(
            signal.signal_id for signal in signals if signal.hitl_required
        ),
        persisted_learning_signal_ids=(),
        applied_learning_signal_ids=(),
        signal_count=len(signals),
        review_required_signal_count=len(
            _signal_ids_for_status(signals, "review_required")
        ),
        guardrail_signal_count=len(_signal_ids_for_status(signals, "guardrail")),
        hitl_required_signal_count=sum(1 for signal in signals if signal.hitl_required),
        highest_learning_priority_score=max(
            signal.learning_priority_score for signal in signals
        ),
        overall_learning_priority_score=_overall_learning_priority_score(signals),
        overall_learning_posture=_overall_learning_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def adaptive_learning_signal_by_id(
    signal_id: str,
    plan: AdaptiveLearningPlan | None = None,
) -> AdaptiveLearningSignal | None:
    """Return one adaptive learning signal without applying learning."""

    source_plan = plan or evaluate_adaptive_learning_engine()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def adaptive_learning_signals_for_status(
    status: AdaptiveLearningSignalStatus,
    plan: AdaptiveLearningPlan | None = None,
) -> tuple[AdaptiveLearningSignal, ...]:
    """Return adaptive learning signals by advisory status."""

    source_plan = plan or evaluate_adaptive_learning_engine()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def adaptive_learning_signals_for_priority(
    priority_band: AdaptiveLearningPriorityBand,
    plan: AdaptiveLearningPlan | None = None,
) -> tuple[AdaptiveLearningSignal, ...]:
    """Return adaptive learning signals by advisory priority band."""

    source_plan = plan or evaluate_adaptive_learning_engine()
    return tuple(
        signal
        for signal in source_plan.signals
        if signal.priority_band == priority_band
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    execution_confidence: ExecutionConfidencePlan,
    workflow_risk: WorkflowRiskPlan,
    self_tuning: WorkflowSelfTuningPolicyPlan,
) -> tuple[AdaptiveLearningSignal, ...]:
    return (
        _signal(
            kind="workflow_pattern_learning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            risk_factor_id="workflow_risk::execution_confidence_risk",
            execution_confidence=execution_confidence,
            workflow_risk=workflow_risk,
            self_tuning=self_tuning,
            learning_weight=180,
        ),
        _signal(
            kind="strategy_pattern_learning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            risk_factor_id="workflow_risk::escalation_posture_risk",
            execution_confidence=execution_confidence,
            workflow_risk=workflow_risk,
            self_tuning=self_tuning,
            learning_weight=150,
        ),
        _signal(
            kind="routing_boundary_learning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            risk_factor_id="workflow_risk::provider_fallback_risk",
            execution_confidence=execution_confidence,
            workflow_risk=workflow_risk,
            self_tuning=self_tuning,
            learning_weight=220,
        ),
        _signal(
            kind="governance_feedback_learning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            risk_factor_id="workflow_risk::self_tuning_policy_risk",
            execution_confidence=execution_confidence,
            workflow_risk=workflow_risk,
            self_tuning=self_tuning,
            learning_weight=170,
        ),
        _signal(
            kind="runtime_guardrail_learning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            risk_factor_id="workflow_risk::performance_regression_risk",
            execution_confidence=execution_confidence,
            workflow_risk=workflow_risk,
            self_tuning=self_tuning,
            learning_weight=210,
        ),
    )


def _signal(
    *,
    kind: AdaptiveLearningSignalKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    risk_factor_id: str,
    execution_confidence: ExecutionConfidencePlan,
    workflow_risk: WorkflowRiskPlan,
    self_tuning: WorkflowSelfTuningPolicyPlan,
    learning_weight: int,
) -> AdaptiveLearningSignal:
    risk_factor = _required_risk_factor(risk_factor_id, workflow_risk)
    confidence_signal = _required_confidence_signal(
        risk_factor.source_execution_confidence_signal_id,
        execution_confidence,
    )
    self_tuning_policy = _required_self_tuning_policy(
        risk_factor.source_self_tuning_policy_id,
        self_tuning,
    )
    status = _signal_status(risk_factor, confidence_signal, self_tuning_policy)
    priority_score = _learning_priority_score(
        workflow_risk_score=risk_factor.workflow_risk_score,
        execution_confidence_score=confidence_signal.execution_confidence_score,
        self_tuning_score=self_tuning_policy.self_tuning_score,
        unavailable_reason_count=risk_factor.unavailable_reason_count,
        guardrail_signal_count=risk_factor.guardrail_signal_count,
        learning_weight=learning_weight,
    )
    hitl_required = (
        risk_factor.hitl_required
        or confidence_signal.hitl_required
        or self_tuning_policy.hitl_required
        or status != "candidate"
    )
    return AdaptiveLearningSignal(
        signal_id=f"adaptive_learning::{kind}",
        signal_kind=kind,
        status=status,
        priority_band=_priority_band(priority_score, status),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_workflow_risk_factor_id=risk_factor.factor_id,
        source_execution_confidence_signal_id=confidence_signal.signal_id,
        source_self_tuning_policy_id=self_tuning_policy.policy_id,
        workflow_risk_score=risk_factor.workflow_risk_score,
        execution_confidence_score=confidence_signal.execution_confidence_score,
        self_tuning_score=self_tuning_policy.self_tuning_score,
        unavailable_reason_count=risk_factor.unavailable_reason_count,
        guardrail_signal_count=risk_factor.guardrail_signal_count,
        learning_weight=learning_weight,
        learning_priority_score=priority_score,
        hitl_required=hitl_required,
        pattern_tags=_pattern_tags(kind),
        learning_summary=_learning_summary(kind, status),
        proposed_learning_actions=_learning_actions(kind),
        evidence=(
            f"workflow_risk:{risk_factor.factor_id}",
            f"execution_confidence:{confidence_signal.signal_id}",
            f"self_tuning_policy:{self_tuning_policy.policy_id}",
            f"workflow_risk_score:{risk_factor.workflow_risk_score}",
            f"execution_confidence_score:{confidence_signal.execution_confidence_score}",
            f"self_tuning_score:{self_tuning_policy.self_tuning_score}",
            f"hitl_required:{hitl_required}",
        ),
    )


def _learning_priority_score(
    *,
    workflow_risk_score: int,
    execution_confidence_score: int,
    self_tuning_score: int,
    unavailable_reason_count: int,
    guardrail_signal_count: int,
    learning_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            workflow_risk_score // 2
            + max(0, 100 - execution_confidence_score) * 4
            + self_tuning_score // 2
            + unavailable_reason_count * 35
            + guardrail_signal_count * 75
            + learning_weight,
        ),
    )


def _signal_status(
    risk_factor: WorkflowRiskFactor,
    confidence_signal: ExecutionConfidenceSignal,
    self_tuning_policy: WorkflowSelfTuningPolicy,
) -> AdaptiveLearningSignalStatus:
    if risk_factor.status == "guardrail" or confidence_signal.status == "guardrail":
        return "guardrail"
    if (
        risk_factor.hitl_required
        or confidence_signal.hitl_required
        or self_tuning_policy.hitl_required
    ):
        return "review_required"
    return "candidate"


def _priority_band(
    score: int,
    status: AdaptiveLearningSignalStatus,
) -> AdaptiveLearningPriorityBand:
    if status == "guardrail":
        return "guarded"
    if score >= 850:
        return "critical"
    if score >= 650:
        return "elevated"
    return "standard"


def _signal_ids_for_status(
    signals: tuple[AdaptiveLearningSignal, ...],
    status: AdaptiveLearningSignalStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_priority(
    signals: tuple[AdaptiveLearningSignal, ...],
    priority_band: AdaptiveLearningPriorityBand,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.priority_band == priority_band
    )


def _overall_learning_priority_score(
    signals: tuple[AdaptiveLearningSignal, ...],
) -> int:
    return sum(signal.learning_priority_score for signal in signals) // len(signals)


def _overall_learning_posture(
    signals: tuple[AdaptiveLearningSignal, ...],
) -> AdaptiveLearningPosture:
    if any(signal.status == "guardrail" for signal in signals):
        return "guarded"
    if any(signal.hitl_required for signal in signals):
        return "review_required"
    return "candidate"


def _required_risk_factor(
    factor_id: str,
    plan: WorkflowRiskPlan,
) -> WorkflowRiskFactor:
    factor = workflow_risk_factor_by_id(factor_id, plan)
    if factor is None:
        raise ValueError("required adaptive learning workflow risk metadata is missing")
    return factor


def _required_confidence_signal(
    signal_id: str,
    plan: ExecutionConfidencePlan,
) -> ExecutionConfidenceSignal:
    signal = execution_confidence_signal_by_id(signal_id, plan)
    if signal is None:
        raise ValueError(
            "required adaptive learning execution confidence metadata is missing"
        )
    return signal


def _required_self_tuning_policy(
    policy_id: str,
    plan: WorkflowSelfTuningPolicyPlan,
) -> WorkflowSelfTuningPolicy:
    policy = workflow_self_tuning_policy_by_id(policy_id, plan)
    if policy is None:
        raise ValueError(
            "required adaptive learning workflow self-tuning metadata is missing"
        )
    return policy


def _pattern_tags(kind: AdaptiveLearningSignalKind) -> tuple[str, ...]:
    return {
        "workflow_pattern_learning": (
            "workflow_outcome_pattern",
            "execution_confidence",
            "review_loop",
        ),
        "strategy_pattern_learning": (
            "strategy_selection_pattern",
            "escalation_posture",
            "decision_trace",
        ),
        "routing_boundary_learning": (
            "provider_boundary_pattern",
            "availability_guardrail",
            "routing_safety",
        ),
        "governance_feedback_learning": (
            "policy_review_pattern",
            "self_tuning_boundary",
            "governance",
        ),
        "runtime_guardrail_learning": (
            "runtime_guardrail_pattern",
            "regression_watch",
            "safety_review",
        ),
    }[kind]


def _learning_summary(
    kind: AdaptiveLearningSignalKind,
    status: AdaptiveLearningSignalStatus,
) -> str:
    if status == "guardrail":
        return f"Surface {kind} as guarded learning metadata without application."
    if status == "review_required":
        return f"Surface {kind} for review before any future learning behavior."
    return f"Surface {kind} as candidate learning metadata only."


def _learning_actions(kind: AdaptiveLearningSignalKind) -> tuple[str, ...]:
    return (
        f"Expose {kind} as an adaptive learning candidate.",
        "Keep learning memory writes, feedback application, policy mutation, "
        "routing, workflow control, storage, Runtime Evolution, and output "
        "mutation disabled.",
    )


def _plan_actions(
    signals: tuple[AdaptiveLearningSignal, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose adaptive learning signals as advisory metadata only.",
        "Keep persisted and applied learning signal ids empty.",
        "Preserve learning memory, feedback, policy, routing, provider, "
        "workflow, storage, output, and Runtime Evolution boundaries.",
    ]
    if any(signal.hitl_required for signal in signals):
        actions.append("Require review before any future learning behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
