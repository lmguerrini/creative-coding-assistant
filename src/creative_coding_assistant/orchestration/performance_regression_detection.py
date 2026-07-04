"""V5.3 advisory performance regression detection planning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .performance_benchmarking import (
    PerformanceBenchmarkingPlan,
    plan_performance_benchmarking,
)
from .performance_prediction import PerformancePredictionPlan, predict_performance
from .reasoning_budget_optimizer import (
    ReasoningBudgetOptimizationPlan,
    optimize_reasoning_budget,
)

PerformanceRegressionKind = Literal[
    "prediction_regression_risk",
    "benchmark_regression_risk",
    "reasoning_budget_pressure",
    "measurement_boundary",
]
PerformanceRegressionStatus = Literal[
    "regression_candidate",
    "baseline_guardrail",
    "review_only",
]
PerformanceRegressionSeverity = Literal["low", "medium", "high", "guarded"]

PERFORMANCE_REGRESSION_SIGNAL_SERIALIZATION_VERSION = "performance_regression_signal.v1"
PERFORMANCE_REGRESSION_DETECTION_PLAN_SERIALIZATION_VERSION = (
    "performance_regression_detection_plan.v1"
)
PERFORMANCE_REGRESSION_DETECTION_AUTHORITY_BOUNDARY = (
    "Performance regression detection planning derives advisory regression "
    "signals from performance prediction, performance benchmarking, and "
    "reasoning budget metadata only; it does not detect live regressions, run "
    "benchmarks, measure runtime performance, collect timers, enforce "
    "thresholds, emit alerts, block workflows, route providers or models, "
    "control workflows, trigger retries, mutate prompts, write storage, or "
    "modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_regression_detection",
    "benchmark_execution",
    "runtime_performance_measurement",
    "timer_collection",
    "threshold_enforcement",
    "alert_emission",
    "workflow_blocking",
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


class PerformanceRegressionSignal(BaseModel):
    """One advisory V5.3 performance regression signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    regression_id: str = Field(min_length=1, max_length=120)
    regression_kind: PerformanceRegressionKind
    status: PerformanceRegressionStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=100)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    baseline_reference_count: int = Field(ge=0, le=200)
    source_pressure_score: int = Field(ge=0, le=20_000)
    advisory_regression_score: int = Field(ge=0, le=3_000)
    regression_severity: PerformanceRegressionSeverity
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    performance_regression_detection_planning_implemented: Literal[True] = True
    runtime_regression_detection_implemented: Literal[False] = False
    benchmark_execution_implemented: Literal[False] = False
    runtime_performance_measurement_implemented: Literal[False] = False
    timer_collection_implemented: Literal[False] = False
    threshold_enforcement_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    workflow_blocking_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
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
    serialization_version: Literal["performance_regression_signal.v1"] = (
        PERFORMANCE_REGRESSION_SIGNAL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_inputs(self) -> Self:
        if self.signal_id != f"performance_regression::{self.regression_id}":
            raise ValueError("signal_id must match regression_id")
        expected_score = _regression_score(
            status=self.status,
            baseline_reference_count=self.baseline_reference_count,
            source_pressure_score=self.source_pressure_score,
        )
        if self.advisory_regression_score != expected_score:
            raise ValueError("advisory_regression_score must match signal inputs")
        if self.regression_severity != _severity(
            status=self.status,
            regression_score=self.advisory_regression_score,
        ):
            raise ValueError("regression_severity must match signal inputs")
        if self.status == "regression_candidate" and self.baseline_reference_count <= 0:
            raise ValueError("regression candidates require baseline references")
        if self.status == "baseline_guardrail" and self.source_pressure_score != 0:
            raise ValueError("baseline guardrails must not declare pressure score")
        return self


class PerformanceRegressionDetectionPlan(BaseModel):
    """Bounded V5.3 advisory performance regression detection plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["performance_regression_detector"] = "performance_regression_detector"
    serialization_version: Literal["performance_regression_detection_plan.v1"] = (
        PERFORMANCE_REGRESSION_DETECTION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PERFORMANCE_REGRESSION_DETECTION_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    source_performance_prediction_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_performance_benchmarking_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_reasoning_budget_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    signals: tuple[PerformanceRegressionSignal, ...] = Field(
        min_length=1,
        max_length=12,
    )
    signal_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    regression_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    baseline_guardrail_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    review_only_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    signal_count: int = Field(ge=1, le=12)
    regression_candidate_count: int = Field(ge=0, le=12)
    baseline_guardrail_count: int = Field(ge=0, le=12)
    review_only_count: int = Field(ge=0, le=12)
    total_baseline_reference_count: int = Field(ge=0, le=1_000)
    highest_advisory_regression_score: int = Field(ge=0, le=3_000)
    total_advisory_regression_score: int = Field(ge=0, le=20_000)
    regression_detection_severity: PerformanceRegressionSeverity
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    performance_regression_detection_planning_implemented: Literal[True] = True
    runtime_regression_detection_implemented: Literal[False] = False
    benchmark_execution_implemented: Literal[False] = False
    runtime_performance_measurement_implemented: Literal[False] = False
    timer_collection_implemented: Literal[False] = False
    threshold_enforcement_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    workflow_blocking_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
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
    def _plan_matches_signals(self) -> Self:
        derived_signal_ids = tuple(signal.signal_id for signal in self.signals)
        if len(set(derived_signal_ids)) != len(derived_signal_ids):
            raise ValueError("signal_ids must be unique")
        if self.signal_ids != derived_signal_ids:
            raise ValueError("signal_ids must match signals")
        if self.signal_count != len(self.signals):
            raise ValueError("signal_count must match signals")
        if self.regression_candidate_ids != _signal_ids_for_status(
            self.signals,
            "regression_candidate",
        ):
            raise ValueError("regression_candidate_ids must match signals")
        if self.baseline_guardrail_ids != _signal_ids_for_status(
            self.signals,
            "baseline_guardrail",
        ):
            raise ValueError("baseline_guardrail_ids must match signals")
        if self.review_only_ids != _signal_ids_for_status(
            self.signals,
            "review_only",
        ):
            raise ValueError("review_only_ids must match signals")
        if self.regression_candidate_count != len(self.regression_candidate_ids):
            raise ValueError("regression_candidate_count must match signals")
        if self.baseline_guardrail_count != len(self.baseline_guardrail_ids):
            raise ValueError("baseline_guardrail_count must match signals")
        if self.review_only_count != len(self.review_only_ids):
            raise ValueError("review_only_count must match signals")

        expected_baselines = sum(
            signal.baseline_reference_count for signal in self.signals
        )
        if self.total_baseline_reference_count != expected_baselines:
            raise ValueError("total_baseline_reference_count must match signals")
        expected_highest = max(
            signal.advisory_regression_score for signal in self.signals
        )
        if self.highest_advisory_regression_score != expected_highest:
            raise ValueError("highest_advisory_regression_score must match signals")
        expected_total = sum(
            signal.advisory_regression_score for signal in self.signals
        )
        if self.total_advisory_regression_score != expected_total:
            raise ValueError("total_advisory_regression_score must match signals")
        if self.regression_detection_severity != _plan_severity(
            signals=self.signals,
            highest_score=self.highest_advisory_regression_score,
        ):
            raise ValueError("regression_detection_severity must match signals")
        return self


def detect_performance_regressions(
    *,
    performance_prediction: PerformancePredictionPlan | None = None,
    performance_benchmarking: PerformanceBenchmarkingPlan | None = None,
    reasoning_budget: ReasoningBudgetOptimizationPlan | None = None,
) -> PerformanceRegressionDetectionPlan:
    """Plan advisory performance regression detection without live detection."""

    prediction = performance_prediction or predict_performance()
    benchmarking = performance_benchmarking or plan_performance_benchmarking(
        performance_prediction=prediction
    )
    reasoning = reasoning_budget or optimize_reasoning_budget(
        performance_prediction=prediction,
        performance_benchmarking=benchmarking,
    )
    signals = _signals(
        prediction=prediction,
        benchmarking=benchmarking,
        reasoning=reasoning,
    )
    highest_score = max(signal.advisory_regression_score for signal in signals)

    return PerformanceRegressionDetectionPlan(
        source_performance_prediction_serialization_version=(
            prediction.serialization_version
        ),
        source_performance_benchmarking_serialization_version=(
            benchmarking.serialization_version
        ),
        source_reasoning_budget_serialization_version=reasoning.serialization_version,
        signals=signals,
        signal_ids=tuple(signal.signal_id for signal in signals),
        regression_candidate_ids=_signal_ids_for_status(
            signals,
            "regression_candidate",
        ),
        baseline_guardrail_ids=_signal_ids_for_status(
            signals,
            "baseline_guardrail",
        ),
        review_only_ids=_signal_ids_for_status(signals, "review_only"),
        signal_count=len(signals),
        regression_candidate_count=len(
            _signal_ids_for_status(signals, "regression_candidate")
        ),
        baseline_guardrail_count=len(
            _signal_ids_for_status(signals, "baseline_guardrail")
        ),
        review_only_count=len(_signal_ids_for_status(signals, "review_only")),
        total_baseline_reference_count=sum(
            signal.baseline_reference_count for signal in signals
        ),
        highest_advisory_regression_score=highest_score,
        total_advisory_regression_score=sum(
            signal.advisory_regression_score for signal in signals
        ),
        regression_detection_severity=_plan_severity(
            signals=signals,
            highest_score=highest_score,
        ),
        advisory_actions=_plan_actions(signals),
    )


def performance_regression_signal_by_id(
    signal_id: str,
    plan: PerformanceRegressionDetectionPlan | None = None,
) -> PerformanceRegressionSignal | None:
    """Return one advisory regression signal without live detection."""

    source_plan = plan or detect_performance_regressions()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def performance_regression_signals_for_status(
    status: PerformanceRegressionStatus,
    plan: PerformanceRegressionDetectionPlan | None = None,
) -> tuple[PerformanceRegressionSignal, ...]:
    """Return advisory regression signals by status."""

    source_plan = plan or detect_performance_regressions()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def _signals(
    *,
    prediction: PerformancePredictionPlan,
    benchmarking: PerformanceBenchmarkingPlan,
    reasoning: ReasoningBudgetOptimizationPlan,
) -> tuple[PerformanceRegressionSignal, ...]:
    return (
        _signal(
            regression_id="prediction_regression_risk",
            kind="prediction_regression_risk",
            status="regression_candidate",
            source_id="performance_prediction_plan",
            source_serialization_version=prediction.serialization_version,
            source_item_ids=prediction.prediction_ids,
            baseline_reference_count=prediction.prediction_count,
            source_pressure_score=prediction.guarded_prediction_count * 600,
            evidence=(
                f"recommended_band:{prediction.recommended_performance_band}",
                f"guarded_predictions:{prediction.guarded_prediction_count}",
            ),
        ),
        _signal(
            regression_id="benchmark_regression_risk",
            kind="benchmark_regression_risk",
            status="regression_candidate",
            source_id="performance_benchmarking_plan",
            source_serialization_version=benchmarking.serialization_version,
            source_item_ids=benchmarking.scenario_ids,
            baseline_reference_count=benchmarking.scenario_count,
            source_pressure_score=benchmarking.total_benchmark_priority_score,
            evidence=(
                f"benchmarking_readiness:{benchmarking.benchmarking_readiness}",
                f"benchmark_score:{benchmarking.total_benchmark_priority_score}",
            ),
        ),
        _signal(
            regression_id="reasoning_budget_pressure",
            kind="reasoning_budget_pressure",
            status="review_only",
            source_id="reasoning_budget_optimization_plan",
            source_serialization_version=reasoning.serialization_version,
            source_item_ids=reasoning.recommendation_ids,
            baseline_reference_count=reasoning.recommendation_count,
            source_pressure_score=reasoning.total_advisory_pressure_score,
            evidence=(
                f"reasoning_budget_pressure:{reasoning.reasoning_budget_pressure}",
                f"reasoning_score:{reasoning.total_advisory_pressure_score}",
            ),
        ),
        _signal(
            regression_id="measurement_boundary",
            kind="measurement_boundary",
            status="baseline_guardrail",
            source_id="performance_benchmarking_plan",
            source_serialization_version=benchmarking.serialization_version,
            source_item_ids=benchmarking.guardrail_scenario_ids,
            baseline_reference_count=benchmarking.guardrail_count,
            source_pressure_score=0,
            evidence=(
                "runtime_performance_measurement:blocked",
                "threshold_enforcement:blocked",
            ),
        ),
    )


def _signal(
    *,
    regression_id: str,
    kind: PerformanceRegressionKind,
    status: PerformanceRegressionStatus,
    source_id: str,
    source_serialization_version: str,
    source_item_ids: tuple[str, ...],
    baseline_reference_count: int,
    source_pressure_score: int,
    evidence: tuple[str, ...],
) -> PerformanceRegressionSignal:
    score = _regression_score(
        status=status,
        baseline_reference_count=baseline_reference_count,
        source_pressure_score=source_pressure_score,
    )
    return PerformanceRegressionSignal(
        signal_id=f"performance_regression::{regression_id}",
        regression_id=regression_id,
        regression_kind=kind,
        status=status,
        source_id=source_id,
        source_serialization_version=source_serialization_version,
        source_item_ids=source_item_ids,
        baseline_reference_count=baseline_reference_count,
        source_pressure_score=source_pressure_score,
        advisory_regression_score=score,
        regression_severity=_severity(status=status, regression_score=score),
        evidence=evidence,
        advisory_actions=_signal_actions(status),
    )


def _signal_ids_for_status(
    signals: tuple[PerformanceRegressionSignal, ...],
    status: PerformanceRegressionStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _regression_score(
    *,
    status: PerformanceRegressionStatus,
    baseline_reference_count: int,
    source_pressure_score: int,
) -> int:
    if status == "baseline_guardrail":
        return 0
    score = baseline_reference_count * 150 + source_pressure_score
    if status == "review_only":
        score = baseline_reference_count * 75 + source_pressure_score // 2
    return min(3_000, score)


def _severity(
    *,
    status: PerformanceRegressionStatus,
    regression_score: int,
) -> PerformanceRegressionSeverity:
    if status == "baseline_guardrail":
        return "guarded"
    if regression_score >= 1_200:
        return "high"
    if regression_score >= 400:
        return "medium"
    return "low"


def _plan_severity(
    *,
    signals: tuple[PerformanceRegressionSignal, ...],
    highest_score: int,
) -> PerformanceRegressionSeverity:
    if any(signal.status == "baseline_guardrail" for signal in signals):
        return "guarded"
    if highest_score >= 1_200:
        return "high"
    if highest_score >= 400:
        return "medium"
    return "low"


def _signal_actions(status: PerformanceRegressionStatus) -> tuple[str, ...]:
    if status == "regression_candidate":
        return (
            "Expose regression risk signal as advisory metadata only.",
            "Require explicit runtime authority before measuring regressions.",
        )
    if status == "review_only":
        return ("Keep reasoning-budget regression pressure review-only.",)
    return ("Preserve measurement, threshold, alert, workflow, and output boundaries.",)


def _plan_actions(
    signals: tuple[PerformanceRegressionSignal, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose performance regression signals as advisory metadata only.",
        "Preserve live detection, benchmark execution, measurement, thresholds, "
        "alerts, workflow, routing, and output boundaries.",
    ]
    if _signal_ids_for_status(signals, "baseline_guardrail"):
        actions.append("Keep regression baselines detached from runtime enforcement.")
    return tuple(actions)
