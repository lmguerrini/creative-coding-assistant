"""V6.1 advisory workflow success tracking."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_learning_engine import (
    AdaptiveLearningPlan,
    AdaptiveLearningSignal,
    evaluate_adaptive_learning_engine,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

WorkflowSuccessIndicatorKind = Literal[
    "execution_readiness_success",
    "strategy_alignment_success",
    "routing_safety_success",
    "governance_review_success",
    "runtime_guardrail_success",
]
WorkflowSuccessStatus = Literal[
    "success_candidate",
    "review_required",
    "guarded",
]
WorkflowSuccessConfidenceBand = Literal["strong", "moderate", "weak", "guarded"]
WorkflowSuccessPosture = Literal["candidate", "review_required", "guarded"]

WORKFLOW_SUCCESS_INDICATOR_SERIALIZATION_VERSION = "workflow_success_indicator.v1"
WORKFLOW_SUCCESS_TRACKING_PLAN_SERIALIZATION_VERSION = (
    "workflow_success_tracking_plan.v1"
)
WORKFLOW_SUCCESS_TRACKING_AUTHORITY_BOUNDARY = (
    "V6.1 workflow success tracking derives success indicators from advisory "
    "adaptive learning metadata only; it does not observe live workflow "
    "outcomes, collect telemetry, evaluate generated output, persist success "
    "metrics, apply learning, update policies, change provider or model "
    "routing, execute providers, invoke agents, control workflows, mutate "
    "workflow graphs, trigger retries or refinements, write storage, modify "
    "generated output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_success_observation",
    "live_telemetry_collection",
    "generated_output_evaluation",
    "success_metric_persistence",
    "learning_feedback_application",
    "learning_policy_mutation",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
    "provider_execution",
    "agent_invocation",
    "resource_allocation",
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


class WorkflowSuccessIndicator(BaseModel):
    """One derived workflow success indicator without runtime observation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    indicator_id: str = Field(min_length=1, max_length=180)
    indicator_kind: WorkflowSuccessIndicatorKind
    status: WorkflowSuccessStatus
    confidence_band: WorkflowSuccessConfidenceBand
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    source_workflow_risk_factor_id: str = Field(min_length=1, max_length=180)
    source_execution_confidence_signal_id: str = Field(min_length=1, max_length=180)
    source_self_tuning_policy_id: str = Field(min_length=1, max_length=180)
    workflow_risk_score: int = Field(ge=0, le=1_000)
    execution_confidence_score: int = Field(ge=0, le=100)
    self_tuning_score: int = Field(ge=0, le=600)
    learning_priority_score: int = Field(ge=0, le=1_000)
    unavailable_reason_count: int = Field(ge=0, le=9)
    guardrail_signal_count: int = Field(ge=0, le=200)
    success_weight: int = Field(ge=0, le=240)
    workflow_success_score: int = Field(ge=0, le=1_000)
    hitl_required: bool
    success_pattern_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    success_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    workflow_success_tracking_implemented: Literal[True] = True
    success_indicator_metadata_implemented: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    runtime_success_observation_implemented: Literal[False] = False
    live_telemetry_collection_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    success_metric_persistence_implemented: Literal[False] = False
    learning_feedback_application_implemented: Literal[False] = False
    learning_policy_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
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
    serialization_version: Literal["workflow_success_indicator.v1"] = (
        WORKFLOW_SUCCESS_INDICATOR_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _indicator_matches_contract(self) -> Self:
        if self.indicator_id != f"workflow_success::{self.indicator_kind}":
            raise ValueError("indicator_id must match indicator_kind")
        if self.workflow_success_score != _workflow_success_score(
            workflow_risk_score=self.workflow_risk_score,
            execution_confidence_score=self.execution_confidence_score,
            self_tuning_score=self.self_tuning_score,
            learning_priority_score=self.learning_priority_score,
            unavailable_reason_count=self.unavailable_reason_count,
            guardrail_signal_count=self.guardrail_signal_count,
            success_weight=self.success_weight,
        ):
            raise ValueError("workflow_success_score must combine source scores")
        if self.confidence_band != _confidence_band(
            self.workflow_success_score,
            self.status,
        ):
            raise ValueError("confidence_band must match score and status")
        if self.status == "guarded" and not self.hitl_required:
            raise ValueError("guarded success indicators require HITL posture")
        return self


class WorkflowSuccessTrackingPlan(BaseModel):
    """Bounded V6.1 advisory workflow success tracking plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["workflow_success_tracking"] = "workflow_success_tracking"
    serialization_version: Literal["workflow_success_tracking_plan.v1"] = (
        WORKFLOW_SUCCESS_TRACKING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=WORKFLOW_SUCCESS_TRACKING_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_adaptive_learning_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    indicators: tuple[WorkflowSuccessIndicator, ...] = Field(
        min_length=5,
        max_length=5,
    )
    indicator_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    success_candidate_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    review_required_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_indicator_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    strong_confidence_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    moderate_confidence_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    weak_confidence_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_confidence_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    hitl_required_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    persisted_success_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    applied_success_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    indicator_count: int = Field(ge=5, le=5)
    review_required_indicator_count: int = Field(ge=0, le=5)
    guarded_indicator_count: int = Field(ge=0, le=5)
    hitl_required_indicator_count: int = Field(ge=0, le=5)
    highest_workflow_success_score: int = Field(ge=0, le=1_000)
    overall_workflow_success_score: int = Field(ge=0, le=1_000)
    overall_success_posture: WorkflowSuccessPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    workflow_success_tracking_implemented: Literal[True] = True
    success_indicator_metadata_implemented: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    runtime_success_observation_implemented: Literal[False] = False
    live_telemetry_collection_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    success_metric_persistence_implemented: Literal[False] = False
    learning_feedback_application_implemented: Literal[False] = False
    learning_policy_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
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
    def _plan_matches_indicators(self) -> Self:
        derived_indicator_ids = tuple(
            indicator.indicator_id for indicator in self.indicators
        )
        if len(set(derived_indicator_ids)) != len(derived_indicator_ids):
            raise ValueError("indicator_ids must be unique")
        if self.indicator_ids != derived_indicator_ids:
            raise ValueError("indicator_ids must match indicators")
        if self.indicator_count != len(self.indicators):
            raise ValueError("indicator_count must match indicators")
        if self.success_candidate_indicator_ids != _indicator_ids_for_status(
            self.indicators,
            "success_candidate",
        ):
            raise ValueError("success_candidate_indicator_ids must match indicators")
        if self.review_required_indicator_ids != _indicator_ids_for_status(
            self.indicators,
            "review_required",
        ):
            raise ValueError("review_required_indicator_ids must match indicators")
        if self.guarded_indicator_ids != _indicator_ids_for_status(
            self.indicators,
            "guarded",
        ):
            raise ValueError("guarded_indicator_ids must match indicators")
        if self.strong_confidence_indicator_ids != _indicator_ids_for_confidence(
            self.indicators,
            "strong",
        ):
            raise ValueError("strong_confidence_indicator_ids must match indicators")
        if self.moderate_confidence_indicator_ids != _indicator_ids_for_confidence(
            self.indicators,
            "moderate",
        ):
            raise ValueError("moderate_confidence_indicator_ids must match indicators")
        if self.weak_confidence_indicator_ids != _indicator_ids_for_confidence(
            self.indicators,
            "weak",
        ):
            raise ValueError("weak_confidence_indicator_ids must match indicators")
        if self.guarded_confidence_indicator_ids != _indicator_ids_for_confidence(
            self.indicators,
            "guarded",
        ):
            raise ValueError("guarded_confidence_indicator_ids must match indicators")
        if self.hitl_required_indicator_ids != tuple(
            indicator.indicator_id
            for indicator in self.indicators
            if indicator.hitl_required
        ):
            raise ValueError("hitl_required_indicator_ids must match indicators")
        if self.persisted_success_indicator_ids:
            raise ValueError("persisted_success_indicator_ids must remain empty")
        if self.applied_success_indicator_ids:
            raise ValueError("applied_success_indicator_ids must remain empty")
        if self.review_required_indicator_count != len(
            self.review_required_indicator_ids
        ):
            raise ValueError("review_required_indicator_count must match indicators")
        if self.guarded_indicator_count != len(self.guarded_indicator_ids):
            raise ValueError("guarded_indicator_count must match indicators")
        if self.hitl_required_indicator_count != len(self.hitl_required_indicator_ids):
            raise ValueError("hitl_required_indicator_count must match indicators")
        if self.highest_workflow_success_score != max(
            indicator.workflow_success_score for indicator in self.indicators
        ):
            raise ValueError("highest_workflow_success_score must match indicators")
        if self.overall_workflow_success_score != _overall_success_score(
            self.indicators,
        ):
            raise ValueError("overall_workflow_success_score must match indicators")
        if self.overall_success_posture != _overall_success_posture(self.indicators):
            raise ValueError("overall_success_posture must match indicators")
        for indicator in self.indicators:
            if indicator.route_name != self.route_name:
                raise ValueError("indicator route_name must match plan")
        return self


def track_workflow_success(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    adaptive_learning: AdaptiveLearningPlan | None = None,
) -> WorkflowSuccessTrackingPlan:
    """Derive workflow success indicators without observing runtime outcomes."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    learning_plan = adaptive_learning or evaluate_adaptive_learning_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    normalized_mode = str(
        execution_mode_id or learning_plan.signals[0].execution_mode_id
    )
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")

    indicators = _indicators(
        route_name=route_name,
        task_type=learning_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        adaptive_learning=learning_plan,
    )
    return WorkflowSuccessTrackingPlan(
        route_name=route_name,
        task_type=learning_plan.task_type,
        source_adaptive_learning_serialization_version=(
            learning_plan.serialization_version
        ),
        execution_mode_ids=execution_modes.execution_mode_ids,
        indicators=indicators,
        indicator_ids=tuple(indicator.indicator_id for indicator in indicators),
        success_candidate_indicator_ids=_indicator_ids_for_status(
            indicators,
            "success_candidate",
        ),
        review_required_indicator_ids=_indicator_ids_for_status(
            indicators,
            "review_required",
        ),
        guarded_indicator_ids=_indicator_ids_for_status(indicators, "guarded"),
        strong_confidence_indicator_ids=_indicator_ids_for_confidence(
            indicators,
            "strong",
        ),
        moderate_confidence_indicator_ids=_indicator_ids_for_confidence(
            indicators,
            "moderate",
        ),
        weak_confidence_indicator_ids=_indicator_ids_for_confidence(
            indicators,
            "weak",
        ),
        guarded_confidence_indicator_ids=_indicator_ids_for_confidence(
            indicators,
            "guarded",
        ),
        hitl_required_indicator_ids=tuple(
            indicator.indicator_id
            for indicator in indicators
            if indicator.hitl_required
        ),
        persisted_success_indicator_ids=(),
        applied_success_indicator_ids=(),
        indicator_count=len(indicators),
        review_required_indicator_count=len(
            _indicator_ids_for_status(indicators, "review_required")
        ),
        guarded_indicator_count=len(_indicator_ids_for_status(indicators, "guarded")),
        hitl_required_indicator_count=sum(
            1 for indicator in indicators if indicator.hitl_required
        ),
        highest_workflow_success_score=max(
            indicator.workflow_success_score for indicator in indicators
        ),
        overall_workflow_success_score=_overall_success_score(indicators),
        overall_success_posture=_overall_success_posture(indicators),
        advisory_actions=_plan_actions(indicators),
    )


def workflow_success_indicator_by_id(
    indicator_id: str,
    plan: WorkflowSuccessTrackingPlan | None = None,
) -> WorkflowSuccessIndicator | None:
    """Return one workflow success indicator without applying it."""

    source_plan = plan or track_workflow_success()
    for indicator in source_plan.indicators:
        if indicator.indicator_id == indicator_id:
            return indicator
    return None


def workflow_success_indicators_for_status(
    status: WorkflowSuccessStatus,
    plan: WorkflowSuccessTrackingPlan | None = None,
) -> tuple[WorkflowSuccessIndicator, ...]:
    """Return workflow success indicators by advisory status."""

    source_plan = plan or track_workflow_success()
    return tuple(
        indicator for indicator in source_plan.indicators if indicator.status == status
    )


def workflow_success_indicators_for_confidence(
    confidence_band: WorkflowSuccessConfidenceBand,
    plan: WorkflowSuccessTrackingPlan | None = None,
) -> tuple[WorkflowSuccessIndicator, ...]:
    """Return workflow success indicators by derived confidence band."""

    source_plan = plan or track_workflow_success()
    return tuple(
        indicator
        for indicator in source_plan.indicators
        if indicator.confidence_band == confidence_band
    )


def _indicators(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
) -> tuple[WorkflowSuccessIndicator, ...]:
    return (
        _indicator(
            kind="execution_readiness_success",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            learning_signal_id="adaptive_learning::workflow_pattern_learning",
            adaptive_learning=adaptive_learning,
            success_weight=180,
        ),
        _indicator(
            kind="strategy_alignment_success",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            learning_signal_id="adaptive_learning::strategy_pattern_learning",
            adaptive_learning=adaptive_learning,
            success_weight=160,
        ),
        _indicator(
            kind="routing_safety_success",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            learning_signal_id="adaptive_learning::routing_boundary_learning",
            adaptive_learning=adaptive_learning,
            success_weight=220,
        ),
        _indicator(
            kind="governance_review_success",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            learning_signal_id="adaptive_learning::governance_feedback_learning",
            adaptive_learning=adaptive_learning,
            success_weight=170,
        ),
        _indicator(
            kind="runtime_guardrail_success",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            learning_signal_id="adaptive_learning::runtime_guardrail_learning",
            adaptive_learning=adaptive_learning,
            success_weight=210,
        ),
    )


def _indicator(
    *,
    kind: WorkflowSuccessIndicatorKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    learning_signal_id: str,
    adaptive_learning: AdaptiveLearningPlan,
    success_weight: int,
) -> WorkflowSuccessIndicator:
    learning_signal = _required_learning_signal(learning_signal_id, adaptive_learning)
    status = _indicator_status(learning_signal)
    success_score = _workflow_success_score(
        workflow_risk_score=learning_signal.workflow_risk_score,
        execution_confidence_score=learning_signal.execution_confidence_score,
        self_tuning_score=learning_signal.self_tuning_score,
        learning_priority_score=learning_signal.learning_priority_score,
        unavailable_reason_count=learning_signal.unavailable_reason_count,
        guardrail_signal_count=learning_signal.guardrail_signal_count,
        success_weight=success_weight,
    )
    return WorkflowSuccessIndicator(
        indicator_id=f"workflow_success::{kind}",
        indicator_kind=kind,
        status=status,
        confidence_band=_confidence_band(success_score, status),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_learning_signal_id=learning_signal.signal_id,
        source_workflow_risk_factor_id=learning_signal.source_workflow_risk_factor_id,
        source_execution_confidence_signal_id=(
            learning_signal.source_execution_confidence_signal_id
        ),
        source_self_tuning_policy_id=learning_signal.source_self_tuning_policy_id,
        workflow_risk_score=learning_signal.workflow_risk_score,
        execution_confidence_score=learning_signal.execution_confidence_score,
        self_tuning_score=learning_signal.self_tuning_score,
        learning_priority_score=learning_signal.learning_priority_score,
        unavailable_reason_count=learning_signal.unavailable_reason_count,
        guardrail_signal_count=learning_signal.guardrail_signal_count,
        success_weight=success_weight,
        workflow_success_score=success_score,
        hitl_required=learning_signal.hitl_required or status != "success_candidate",
        success_pattern_tags=_success_pattern_tags(kind),
        success_summary=_success_summary(kind, status),
        advisory_actions=_indicator_actions(kind),
        evidence=(
            f"learning_signal:{learning_signal.signal_id}",
            f"workflow_risk_score:{learning_signal.workflow_risk_score}",
            f"execution_confidence_score:{learning_signal.execution_confidence_score}",
            f"self_tuning_score:{learning_signal.self_tuning_score}",
            f"learning_priority_score:{learning_signal.learning_priority_score}",
        ),
    )


def _workflow_success_score(
    *,
    workflow_risk_score: int,
    execution_confidence_score: int,
    self_tuning_score: int,
    learning_priority_score: int,
    unavailable_reason_count: int,
    guardrail_signal_count: int,
    success_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            execution_confidence_score * 5
            + self_tuning_score // 2
            + max(0, 1_000 - workflow_risk_score) // 2
            + success_weight
            - learning_priority_score // 5
            - unavailable_reason_count * 30
            - guardrail_signal_count * 50,
        ),
    )


def _indicator_status(
    learning_signal: AdaptiveLearningSignal,
) -> WorkflowSuccessStatus:
    if learning_signal.status == "guardrail":
        return "guarded"
    if learning_signal.hitl_required:
        return "review_required"
    return "success_candidate"


def _confidence_band(
    score: int,
    status: WorkflowSuccessStatus,
) -> WorkflowSuccessConfidenceBand:
    if status == "guarded":
        return "guarded"
    if score >= 720:
        return "strong"
    if score >= 420:
        return "moderate"
    return "weak"


def _indicator_ids_for_status(
    indicators: tuple[WorkflowSuccessIndicator, ...],
    status: WorkflowSuccessStatus,
) -> tuple[str, ...]:
    return tuple(
        indicator.indicator_id for indicator in indicators if indicator.status == status
    )


def _indicator_ids_for_confidence(
    indicators: tuple[WorkflowSuccessIndicator, ...],
    confidence_band: WorkflowSuccessConfidenceBand,
) -> tuple[str, ...]:
    return tuple(
        indicator.indicator_id
        for indicator in indicators
        if indicator.confidence_band == confidence_band
    )


def _overall_success_score(
    indicators: tuple[WorkflowSuccessIndicator, ...],
) -> int:
    return sum(indicator.workflow_success_score for indicator in indicators) // len(
        indicators
    )


def _overall_success_posture(
    indicators: tuple[WorkflowSuccessIndicator, ...],
) -> WorkflowSuccessPosture:
    if any(indicator.status == "guarded" for indicator in indicators):
        return "guarded"
    if any(indicator.hitl_required for indicator in indicators):
        return "review_required"
    return "candidate"


def _required_learning_signal(
    signal_id: str,
    plan: AdaptiveLearningPlan,
) -> AdaptiveLearningSignal:
    for signal in plan.signals:
        if signal.signal_id == signal_id:
            return signal
    raise ValueError("required workflow success adaptive learning metadata is missing")


def _success_pattern_tags(
    kind: WorkflowSuccessIndicatorKind,
) -> tuple[str, ...]:
    return {
        "execution_readiness_success": (
            "execution_readiness",
            "confidence_success",
            "workflow_pattern",
        ),
        "strategy_alignment_success": (
            "strategy_alignment",
            "decision_success",
            "review_readiness",
        ),
        "routing_safety_success": (
            "routing_safety",
            "provider_boundary",
            "guardrail_success",
        ),
        "governance_review_success": (
            "governance_review",
            "policy_visibility",
            "feedback_readiness",
        ),
        "runtime_guardrail_success": (
            "runtime_guardrail",
            "regression_visibility",
            "safety_success",
        ),
    }[kind]


def _success_summary(
    kind: WorkflowSuccessIndicatorKind,
    status: WorkflowSuccessStatus,
) -> str:
    if status == "guarded":
        return f"Surface {kind} as guarded success metadata without observation."
    if status == "review_required":
        return f"Surface {kind} for review before future success tracking behavior."
    return f"Surface {kind} as candidate success metadata only."


def _indicator_actions(kind: WorkflowSuccessIndicatorKind) -> tuple[str, ...]:
    return (
        f"Expose {kind} as derived workflow success tracking metadata.",
        "Keep runtime observation, telemetry, metric persistence, learning "
        "application, routing, workflow control, storage, Runtime Evolution, "
        "and output mutation disabled.",
    )


def _plan_actions(
    indicators: tuple[WorkflowSuccessIndicator, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose workflow success indicators as advisory metadata only.",
        "Keep persisted and applied success indicator ids empty.",
        "Preserve runtime observation, telemetry, metrics, learning, routing, "
        "provider, workflow, storage, output, and Runtime Evolution boundaries.",
    ]
    if any(indicator.hitl_required for indicator in indicators):
        actions.append("Require review before any future workflow success behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
