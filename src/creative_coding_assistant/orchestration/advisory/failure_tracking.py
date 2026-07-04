"""V6.1 advisory failure tracking."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_learning_engine import (
    AdaptiveLearningPlan,
    AdaptiveLearningSignal,
    evaluate_adaptive_learning_engine,
)
from creative_coding_assistant.orchestration.failure_analysis import (
    FailureAnalysis,
    FailureAnalysisPanel,
    build_failure_analysis,
    failure_analysis_panel_by_id,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

FailureTrackingIndicatorKind = Literal[
    "langgraph_failure_tracking",
    "execution_failure_tracking",
    "routing_failure_tracking",
    "performance_failure_tracking",
    "retry_failure_tracking",
    "observability_failure_tracking",
]
FailureTrackingStatus = Literal["tracked", "review_required", "guarded"]
FailureTrackingSeverity = Literal["low", "medium", "high", "guarded"]
FailureTrackingPosture = Literal["tracked", "review_required", "guarded"]

FAILURE_TRACKING_INDICATOR_SERIALIZATION_VERSION = "failure_tracking_indicator.v1"
FAILURE_TRACKING_PLAN_SERIALIZATION_VERSION = "failure_tracking_plan.v1"
FAILURE_TRACKING_AUTHORITY_BOUNDARY = (
    "V6.1 failure tracking derives failure indicators from read-only failure "
    "analysis and adaptive learning metadata only; it does not observe runtime "
    "failures, classify live errors, route terminal failures, handle or repair "
    "failures, trigger retries or refinements, emit alerts or HITL requests, "
    "change provider or model routing, execute providers, invoke agents, "
    "control workflows, mutate workflow graphs, write storage, modify "
    "generated output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_failure_observation",
    "live_error_classification",
    "terminal_failure_routing",
    "failure_handling_or_repair",
    "alert_emission",
    "hitl_request_emission",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
    "provider_execution",
    "agent_invocation",
    "resource_allocation",
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


class FailureTrackingIndicator(BaseModel):
    """One advisory failure tracking indicator without live failure handling."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    indicator_id: str = Field(min_length=1, max_length=180)
    indicator_kind: FailureTrackingIndicatorKind
    status: FailureTrackingStatus
    severity: FailureTrackingSeverity
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_failure_panel_id: str = Field(min_length=1, max_length=180)
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    source_workflow_risk_factor_id: str = Field(min_length=1, max_length=180)
    failure_signal_count: int = Field(ge=0, le=6_000)
    guardrail_signal_count: int = Field(ge=0, le=6_000)
    workflow_risk_score: int = Field(ge=0, le=1_000)
    learning_priority_score: int = Field(ge=0, le=1_000)
    failure_tracking_weight: int = Field(ge=0, le=240)
    failure_tracking_score: int = Field(ge=0, le=1_000)
    hitl_required: bool
    failure_pattern_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    failure_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    failure_tracking_implemented: Literal[True] = True
    failure_indicator_metadata_implemented: Literal[True] = True
    failure_analysis_metadata_used: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    runtime_failure_observation_implemented: Literal[False] = False
    live_error_classification_implemented: Literal[False] = False
    terminal_failure_routing_implemented: Literal[False] = False
    failure_handling_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["failure_tracking_indicator.v1"] = (
        FAILURE_TRACKING_INDICATOR_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _indicator_matches_contract(self) -> Self:
        if self.indicator_id != f"failure_tracking::{self.indicator_kind}":
            raise ValueError("indicator_id must match indicator_kind")
        if self.failure_tracking_score != _failure_tracking_score(
            failure_signal_count=self.failure_signal_count,
            guardrail_signal_count=self.guardrail_signal_count,
            workflow_risk_score=self.workflow_risk_score,
            learning_priority_score=self.learning_priority_score,
            failure_tracking_weight=self.failure_tracking_weight,
        ):
            raise ValueError("failure_tracking_score must combine source scores")
        if self.severity != _failure_severity(self.failure_tracking_score, self.status):
            raise ValueError("severity must match score and status")
        if self.status == "guarded" and not self.hitl_required:
            raise ValueError("guarded failure indicators require HITL posture")
        return self


class FailureTrackingPlan(BaseModel):
    """Bounded V6.1 advisory failure tracking plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["failure_tracking"] = "failure_tracking"
    serialization_version: Literal["failure_tracking_plan.v1"] = (
        FAILURE_TRACKING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=FAILURE_TRACKING_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_failure_analysis_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_adaptive_learning_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    indicators: tuple[FailureTrackingIndicator, ...] = Field(
        min_length=6,
        max_length=6,
    )
    indicator_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    tracked_indicator_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    review_required_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    guarded_indicator_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    low_severity_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    medium_severity_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    high_severity_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    guarded_severity_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    hitl_required_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    observed_failure_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    handled_failure_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    indicator_count: int = Field(ge=6, le=6)
    guarded_indicator_count: int = Field(ge=0, le=6)
    hitl_required_indicator_count: int = Field(ge=0, le=6)
    total_failure_signal_count: int = Field(ge=0, le=6_000)
    total_guardrail_signal_count: int = Field(ge=0, le=6_000)
    highest_failure_tracking_score: int = Field(ge=0, le=1_000)
    overall_failure_tracking_score: int = Field(ge=0, le=1_000)
    overall_failure_tracking_posture: FailureTrackingPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    failure_tracking_implemented: Literal[True] = True
    failure_indicator_metadata_implemented: Literal[True] = True
    failure_analysis_metadata_used: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    runtime_failure_observation_implemented: Literal[False] = False
    live_error_classification_implemented: Literal[False] = False
    terminal_failure_routing_implemented: Literal[False] = False
    failure_handling_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
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
        if self.tracked_indicator_ids != _indicator_ids_for_status(
            self.indicators,
            "tracked",
        ):
            raise ValueError("tracked_indicator_ids must match indicators")
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
        if self.low_severity_indicator_ids != _indicator_ids_for_severity(
            self.indicators,
            "low",
        ):
            raise ValueError("low_severity_indicator_ids must match indicators")
        if self.medium_severity_indicator_ids != _indicator_ids_for_severity(
            self.indicators,
            "medium",
        ):
            raise ValueError("medium_severity_indicator_ids must match indicators")
        if self.high_severity_indicator_ids != _indicator_ids_for_severity(
            self.indicators,
            "high",
        ):
            raise ValueError("high_severity_indicator_ids must match indicators")
        if self.guarded_severity_indicator_ids != _indicator_ids_for_severity(
            self.indicators,
            "guarded",
        ):
            raise ValueError("guarded_severity_indicator_ids must match indicators")
        if self.hitl_required_indicator_ids != tuple(
            indicator.indicator_id
            for indicator in self.indicators
            if indicator.hitl_required
        ):
            raise ValueError("hitl_required_indicator_ids must match indicators")
        if self.observed_failure_indicator_ids:
            raise ValueError("observed_failure_indicator_ids must remain empty")
        if self.handled_failure_indicator_ids:
            raise ValueError("handled_failure_indicator_ids must remain empty")
        if self.guarded_indicator_count != len(self.guarded_indicator_ids):
            raise ValueError("guarded_indicator_count must match indicators")
        if self.hitl_required_indicator_count != len(self.hitl_required_indicator_ids):
            raise ValueError("hitl_required_indicator_count must match indicators")
        if self.total_failure_signal_count != sum(
            indicator.failure_signal_count for indicator in self.indicators
        ):
            raise ValueError("total_failure_signal_count must match indicators")
        if self.total_guardrail_signal_count != sum(
            indicator.guardrail_signal_count for indicator in self.indicators
        ):
            raise ValueError("total_guardrail_signal_count must match indicators")
        if self.highest_failure_tracking_score != max(
            indicator.failure_tracking_score for indicator in self.indicators
        ):
            raise ValueError("highest_failure_tracking_score must match indicators")
        if self.overall_failure_tracking_score != _overall_failure_tracking_score(
            self.indicators,
        ):
            raise ValueError("overall_failure_tracking_score must match indicators")
        if self.overall_failure_tracking_posture != _overall_failure_posture(
            self.indicators,
        ):
            raise ValueError("overall_failure_tracking_posture must match indicators")
        for indicator in self.indicators:
            if indicator.route_name != self.route_name:
                raise ValueError("indicator route_name must match plan")
        return self


def track_failures(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    adaptive_learning: AdaptiveLearningPlan | None = None,
    failure_analysis: FailureAnalysis | None = None,
) -> FailureTrackingPlan:
    """Derive failure tracking indicators without observing live failures."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    learning_plan = adaptive_learning or evaluate_adaptive_learning_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    failure_source = failure_analysis or build_failure_analysis()
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
        failure_analysis=failure_source,
    )
    return FailureTrackingPlan(
        route_name=route_name,
        task_type=learning_plan.task_type,
        source_failure_analysis_serialization_version=(
            failure_source.serialization_version
        ),
        source_adaptive_learning_serialization_version=(
            learning_plan.serialization_version
        ),
        execution_mode_ids=execution_modes.execution_mode_ids,
        indicators=indicators,
        indicator_ids=tuple(indicator.indicator_id for indicator in indicators),
        tracked_indicator_ids=_indicator_ids_for_status(indicators, "tracked"),
        review_required_indicator_ids=_indicator_ids_for_status(
            indicators,
            "review_required",
        ),
        guarded_indicator_ids=_indicator_ids_for_status(indicators, "guarded"),
        low_severity_indicator_ids=_indicator_ids_for_severity(indicators, "low"),
        medium_severity_indicator_ids=_indicator_ids_for_severity(
            indicators,
            "medium",
        ),
        high_severity_indicator_ids=_indicator_ids_for_severity(indicators, "high"),
        guarded_severity_indicator_ids=_indicator_ids_for_severity(
            indicators,
            "guarded",
        ),
        hitl_required_indicator_ids=tuple(
            indicator.indicator_id
            for indicator in indicators
            if indicator.hitl_required
        ),
        observed_failure_indicator_ids=(),
        handled_failure_indicator_ids=(),
        indicator_count=len(indicators),
        guarded_indicator_count=len(_indicator_ids_for_status(indicators, "guarded")),
        hitl_required_indicator_count=sum(
            1 for indicator in indicators if indicator.hitl_required
        ),
        total_failure_signal_count=sum(
            indicator.failure_signal_count for indicator in indicators
        ),
        total_guardrail_signal_count=sum(
            indicator.guardrail_signal_count for indicator in indicators
        ),
        highest_failure_tracking_score=max(
            indicator.failure_tracking_score for indicator in indicators
        ),
        overall_failure_tracking_score=_overall_failure_tracking_score(indicators),
        overall_failure_tracking_posture=_overall_failure_posture(indicators),
        advisory_actions=_plan_actions(indicators),
    )


def failure_tracking_indicator_by_id(
    indicator_id: str,
    plan: FailureTrackingPlan | None = None,
) -> FailureTrackingIndicator | None:
    """Return one failure tracking indicator without applying it."""

    source_plan = plan or track_failures()
    for indicator in source_plan.indicators:
        if indicator.indicator_id == indicator_id:
            return indicator
    return None


def failure_tracking_indicators_for_status(
    status: FailureTrackingStatus,
    plan: FailureTrackingPlan | None = None,
) -> tuple[FailureTrackingIndicator, ...]:
    """Return failure tracking indicators by advisory status."""

    source_plan = plan or track_failures()
    return tuple(
        indicator for indicator in source_plan.indicators if indicator.status == status
    )


def failure_tracking_indicators_for_severity(
    severity: FailureTrackingSeverity,
    plan: FailureTrackingPlan | None = None,
) -> tuple[FailureTrackingIndicator, ...]:
    """Return failure tracking indicators by derived severity."""

    source_plan = plan or track_failures()
    return tuple(
        indicator
        for indicator in source_plan.indicators
        if indicator.severity == severity
    )


def _indicators(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    failure_analysis: FailureAnalysis,
) -> tuple[FailureTrackingIndicator, ...]:
    return (
        _indicator(
            kind="langgraph_failure_tracking",
            panel_id="failure_analysis::langgraph_error_paths",
            learning_signal_id="adaptive_learning::workflow_pattern_learning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            failure_analysis=failure_analysis,
            failure_tracking_weight=180,
        ),
        _indicator(
            kind="execution_failure_tracking",
            panel_id="failure_analysis::execution_failure_audit",
            learning_signal_id="adaptive_learning::strategy_pattern_learning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            failure_analysis=failure_analysis,
            failure_tracking_weight=170,
        ),
        _indicator(
            kind="routing_failure_tracking",
            panel_id="failure_analysis::routing_failure_audit",
            learning_signal_id="adaptive_learning::routing_boundary_learning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            failure_analysis=failure_analysis,
            failure_tracking_weight=220,
        ),
        _indicator(
            kind="performance_failure_tracking",
            panel_id="failure_analysis::performance_failure_audit",
            learning_signal_id="adaptive_learning::runtime_guardrail_learning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            failure_analysis=failure_analysis,
            failure_tracking_weight=210,
        ),
        _indicator(
            kind="retry_failure_tracking",
            panel_id="failure_analysis::retry_failure_boundaries",
            learning_signal_id="adaptive_learning::governance_feedback_learning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            failure_analysis=failure_analysis,
            failure_tracking_weight=160,
        ),
        _indicator(
            kind="observability_failure_tracking",
            panel_id="failure_analysis::observability_failure_boundary",
            learning_signal_id="adaptive_learning::runtime_guardrail_learning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            failure_analysis=failure_analysis,
            failure_tracking_weight=190,
        ),
    )


def _indicator(
    *,
    kind: FailureTrackingIndicatorKind,
    panel_id: str,
    learning_signal_id: str,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    failure_analysis: FailureAnalysis,
    failure_tracking_weight: int,
) -> FailureTrackingIndicator:
    panel = _required_failure_panel(panel_id, failure_analysis)
    learning_signal = _required_learning_signal(learning_signal_id, adaptive_learning)
    status = _indicator_status(panel, learning_signal)
    score = _failure_tracking_score(
        failure_signal_count=panel.failure_signal_count,
        guardrail_signal_count=panel.guardrail_signal_count,
        workflow_risk_score=learning_signal.workflow_risk_score,
        learning_priority_score=learning_signal.learning_priority_score,
        failure_tracking_weight=failure_tracking_weight,
    )
    return FailureTrackingIndicator(
        indicator_id=f"failure_tracking::{kind}",
        indicator_kind=kind,
        status=status,
        severity=_failure_severity(score, status),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_failure_panel_id=panel.panel_id,
        source_learning_signal_id=learning_signal.signal_id,
        source_workflow_risk_factor_id=learning_signal.source_workflow_risk_factor_id,
        failure_signal_count=panel.failure_signal_count,
        guardrail_signal_count=panel.guardrail_signal_count,
        workflow_risk_score=learning_signal.workflow_risk_score,
        learning_priority_score=learning_signal.learning_priority_score,
        failure_tracking_weight=failure_tracking_weight,
        failure_tracking_score=score,
        hitl_required=learning_signal.hitl_required or status != "tracked",
        failure_pattern_tags=_failure_pattern_tags(kind),
        failure_summary=_failure_summary(kind, status),
        advisory_actions=_indicator_actions(kind),
        evidence=(
            f"failure_panel:{panel.panel_id}",
            f"learning_signal:{learning_signal.signal_id}",
            f"failure_signal_count:{panel.failure_signal_count}",
            f"guardrail_signal_count:{panel.guardrail_signal_count}",
            f"workflow_risk_score:{learning_signal.workflow_risk_score}",
            f"learning_priority_score:{learning_signal.learning_priority_score}",
        ),
    )


def _failure_tracking_score(
    *,
    failure_signal_count: int,
    guardrail_signal_count: int,
    workflow_risk_score: int,
    learning_priority_score: int,
    failure_tracking_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            failure_signal_count * 5
            + guardrail_signal_count * 18
            + workflow_risk_score // 3
            + learning_priority_score // 4
            + failure_tracking_weight,
        ),
    )


def _indicator_status(
    panel: FailureAnalysisPanel,
    learning_signal: AdaptiveLearningSignal,
) -> FailureTrackingStatus:
    if panel.status == "guarded" or learning_signal.status == "guardrail":
        return "guarded"
    if learning_signal.hitl_required:
        return "review_required"
    return "tracked"


def _failure_severity(
    score: int,
    status: FailureTrackingStatus,
) -> FailureTrackingSeverity:
    if status == "guarded":
        return "guarded"
    if score >= 820:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _indicator_ids_for_status(
    indicators: tuple[FailureTrackingIndicator, ...],
    status: FailureTrackingStatus,
) -> tuple[str, ...]:
    return tuple(
        indicator.indicator_id for indicator in indicators if indicator.status == status
    )


def _indicator_ids_for_severity(
    indicators: tuple[FailureTrackingIndicator, ...],
    severity: FailureTrackingSeverity,
) -> tuple[str, ...]:
    return tuple(
        indicator.indicator_id
        for indicator in indicators
        if indicator.severity == severity
    )


def _overall_failure_tracking_score(
    indicators: tuple[FailureTrackingIndicator, ...],
) -> int:
    return sum(indicator.failure_tracking_score for indicator in indicators) // len(
        indicators
    )


def _overall_failure_posture(
    indicators: tuple[FailureTrackingIndicator, ...],
) -> FailureTrackingPosture:
    if any(indicator.status == "guarded" for indicator in indicators):
        return "guarded"
    if any(indicator.hitl_required for indicator in indicators):
        return "review_required"
    return "tracked"


def _required_failure_panel(
    panel_id: str,
    analysis: FailureAnalysis,
) -> FailureAnalysisPanel:
    panel = failure_analysis_panel_by_id(panel_id, analysis)
    if panel is None:
        raise ValueError("required failure tracking analysis metadata is missing")
    return panel


def _required_learning_signal(
    signal_id: str,
    plan: AdaptiveLearningPlan,
) -> AdaptiveLearningSignal:
    for signal in plan.signals:
        if signal.signal_id == signal_id:
            return signal
    raise ValueError("required failure tracking adaptive learning metadata is missing")


def _failure_pattern_tags(
    kind: FailureTrackingIndicatorKind,
) -> tuple[str, ...]:
    return {
        "langgraph_failure_tracking": (
            "langgraph_error_path",
            "terminal_failure_boundary",
            "failure_visibility",
        ),
        "execution_failure_tracking": (
            "execution_failure_audit",
            "strategy_failure_pattern",
            "review_readiness",
        ),
        "routing_failure_tracking": (
            "routing_failure_audit",
            "provider_boundary",
            "availability_failure",
        ),
        "performance_failure_tracking": (
            "performance_failure_audit",
            "runtime_guardrail",
            "regression_visibility",
        ),
        "retry_failure_tracking": (
            "retry_failure_boundary",
            "refinement_guardrail",
            "governance_review",
        ),
        "observability_failure_tracking": (
            "observability_failure_boundary",
            "failure_panel",
            "read_only_analysis",
        ),
    }[kind]


def _failure_summary(
    kind: FailureTrackingIndicatorKind,
    status: FailureTrackingStatus,
) -> str:
    if status == "guarded":
        return f"Surface {kind} as guarded failure metadata without handling."
    if status == "review_required":
        return f"Surface {kind} for review before future failure tracking behavior."
    return f"Surface {kind} as tracked failure metadata only."


def _indicator_actions(kind: FailureTrackingIndicatorKind) -> tuple[str, ...]:
    return (
        f"Expose {kind} as derived failure tracking metadata.",
        "Keep failure observation, live classification, handling, terminal "
        "routing, retries, alerts, HITL emission, workflow control, storage, "
        "Runtime Evolution, and output mutation disabled.",
    )


def _plan_actions(
    indicators: tuple[FailureTrackingIndicator, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose failure tracking indicators as advisory metadata only.",
        "Keep observed and handled failure indicator ids empty.",
        "Preserve failure observation, classification, handling, routing, "
        "provider, workflow, storage, output, and Runtime Evolution boundaries.",
    ]
    if any(indicator.hitl_required for indicator in indicators):
        actions.append("Require review before any future failure tracking behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
