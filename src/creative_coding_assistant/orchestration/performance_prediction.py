"""V5.3 advisory performance prediction planning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .bottleneck_detection import BottleneckDetectionPlan, detect_bottlenecks
from .execution_profiling import ExecutionProfilingPlan, plan_execution_profiling
from .latency_optimizer import LatencyOptimizationPlan, optimize_latency
from .throughput_optimizer import (
    ThroughputOptimizationPlan,
    optimize_throughput,
)

PerformancePredictionFocus = Literal[
    "overall_throughput_posture",
    "latency_pressure_posture",
    "profile_readiness_posture",
    "bottleneck_risk_posture",
]
PerformancePredictionStatus = Literal["recommended", "fallback", "guardrail"]
PerformancePredictionBand = Literal["low", "medium", "high", "guarded"]
PerformancePredictionConfidence = Literal["low", "medium", "high"]

PERFORMANCE_PREDICTION_SERIALIZATION_VERSION = "performance_prediction.v1"
PERFORMANCE_PREDICTION_PLAN_SERIALIZATION_VERSION = "performance_prediction_plan.v1"
PERFORMANCE_PREDICTION_AUTHORITY_BOUNDARY = (
    "Performance prediction planning converts advisory throughput, latency, "
    "bottleneck, and execution profiling metadata into bounded relative "
    "performance predictions only; it does not measure performance, run "
    "benchmarks, install profilers, collect traces, detect regressions, change "
    "throughput or concurrency, manage queues, route providers or models, "
    "control workflows, trigger retries, mutate prompts, write storage, or "
    "modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_performance_measurement",
    "benchmark_execution",
    "profiler_hook_installation",
    "runtime_trace_collection",
    "performance_regression_detection",
    "throughput_runtime_optimization",
    "concurrency_limit_change",
    "queue_management_runtime",
    "provider_or_model_routing",
    "workflow_control",
    "workflow_timing_change",
    "workflow_graph_mutation",
    "langgraph_compilation",
    "workflow_execution",
    "agent_invocation",
    "node_handler_invocation",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)
_PERFORMANCE_RANGE_BY_BAND: dict[PerformancePredictionBand, tuple[int, int]] = {
    "low": (20, 45),
    "medium": (45, 70),
    "high": (70, 90),
    "guarded": (0, 35),
}


class PerformancePrediction(BaseModel):
    """One bounded advisory V5.3 performance prediction."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    prediction_id: str = Field(min_length=1, max_length=180)
    prediction_focus: PerformancePredictionFocus
    status: PerformancePredictionStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=100)
    source_candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    source_signal_score: int = Field(ge=0, le=20_000)
    source_guardrail_count: int = Field(ge=0, le=20)
    predicted_performance_band: PerformancePredictionBand
    predicted_performance_range: tuple[int, int]
    predicted_performance_midpoint: int = Field(ge=0, le=100)
    prediction_confidence: PerformancePredictionConfidence
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    performance_prediction_implemented: Literal[True] = True
    advisory_performance_prediction_implemented: Literal[True] = True
    relative_performance_units_only: Literal[True] = True
    runtime_performance_measurement_implemented: Literal[False] = False
    benchmark_execution_implemented: Literal[False] = False
    profiler_hook_installation_implemented: Literal[False] = False
    runtime_trace_collection_implemented: Literal[False] = False
    regression_detection_implemented: Literal[False] = False
    throughput_runtime_optimization_implemented: Literal[False] = False
    concurrency_limit_change_implemented: Literal[False] = False
    queue_management_runtime_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["performance_prediction.v1"] = (
        PERFORMANCE_PREDICTION_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _prediction_matches_band(self) -> Self:
        if self.prediction_id != f"performance_prediction::{self.prediction_focus}":
            raise ValueError("prediction_id must match prediction_focus")
        expected_band = _performance_band(
            status=self.status,
            source_signal_score=self.source_signal_score,
            source_guardrail_count=self.source_guardrail_count,
        )
        if self.predicted_performance_band != expected_band:
            raise ValueError("predicted_performance_band must match prediction inputs")
        expected_range = _PERFORMANCE_RANGE_BY_BAND[self.predicted_performance_band]
        if self.predicted_performance_range != expected_range:
            raise ValueError("predicted_performance_range must match band")
        low, high = self.predicted_performance_range
        if self.predicted_performance_midpoint != (low + high) // 2:
            raise ValueError("predicted_performance_midpoint must match range")
        if self.prediction_confidence != _confidence(
            band=self.predicted_performance_band,
            source_signal_score=self.source_signal_score,
        ):
            raise ValueError("prediction_confidence must match prediction inputs")
        if self.status == "guardrail" and self.source_guardrail_count <= 0:
            raise ValueError("guardrail predictions require guardrail sources")
        return self


class PerformancePredictionPlan(BaseModel):
    """Bounded V5.3 advisory performance prediction plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["performance_predictor"] = "performance_predictor"
    serialization_version: Literal["performance_prediction_plan.v1"] = (
        PERFORMANCE_PREDICTION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PERFORMANCE_PREDICTION_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    source_throughput_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_latency_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_bottleneck_detection_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_execution_profiling_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    predictions: tuple[PerformancePrediction, ...] = Field(
        min_length=1,
        max_length=12,
    )
    prediction_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    recommended_prediction_id: str = Field(min_length=1, max_length=180)
    recommended_performance_band: PerformancePredictionBand
    recommended_performance_midpoint: int = Field(ge=0, le=100)
    fallback_prediction_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    guardrail_prediction_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    prediction_count: int = Field(ge=1, le=12)
    high_prediction_count: int = Field(ge=0, le=12)
    guarded_prediction_count: int = Field(ge=0, le=12)
    lowest_predicted_performance_midpoint: int = Field(ge=0, le=100)
    highest_predicted_performance_midpoint: int = Field(ge=0, le=100)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    performance_prediction_implemented: Literal[True] = True
    advisory_performance_prediction_implemented: Literal[True] = True
    relative_performance_units_only: Literal[True] = True
    runtime_performance_measurement_implemented: Literal[False] = False
    benchmark_execution_implemented: Literal[False] = False
    profiler_hook_installation_implemented: Literal[False] = False
    runtime_trace_collection_implemented: Literal[False] = False
    regression_detection_implemented: Literal[False] = False
    throughput_runtime_optimization_implemented: Literal[False] = False
    concurrency_limit_change_implemented: Literal[False] = False
    queue_management_runtime_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_predictions(self) -> Self:
        derived_prediction_ids = tuple(
            prediction.prediction_id for prediction in self.predictions
        )
        if len(set(derived_prediction_ids)) != len(derived_prediction_ids):
            raise ValueError("prediction_ids must be unique")
        if self.prediction_ids != derived_prediction_ids:
            raise ValueError("prediction_ids must match predictions")
        if self.prediction_count != len(self.predictions):
            raise ValueError("prediction_count must match predictions")

        recommended = tuple(
            prediction
            for prediction in self.predictions
            if prediction.status == "recommended"
        )
        if len(recommended) != 1:
            raise ValueError(
                "exactly one recommended performance prediction is required"
            )
        recommended_prediction = recommended[0]
        if self.recommended_prediction_id != recommended_prediction.prediction_id:
            raise ValueError("recommended_prediction_id must match prediction")
        if self.recommended_performance_band != (
            recommended_prediction.predicted_performance_band
        ):
            raise ValueError("recommended_performance_band must match prediction")
        if self.recommended_performance_midpoint != (
            recommended_prediction.predicted_performance_midpoint
        ):
            raise ValueError("recommended_performance_midpoint must match prediction")
        if self.fallback_prediction_ids != _prediction_ids_for_status(
            self.predictions,
            "fallback",
        ):
            raise ValueError("fallback_prediction_ids must match predictions")
        if self.guardrail_prediction_ids != _prediction_ids_for_status(
            self.predictions,
            "guardrail",
        ):
            raise ValueError("guardrail_prediction_ids must match predictions")
        if self.high_prediction_count != sum(
            1
            for prediction in self.predictions
            if prediction.predicted_performance_band == "high"
        ):
            raise ValueError("high_prediction_count must match predictions")
        if self.guarded_prediction_count != sum(
            1
            for prediction in self.predictions
            if prediction.predicted_performance_band == "guarded"
        ):
            raise ValueError("guarded_prediction_count must match predictions")
        lowest_midpoint = min(
            prediction.predicted_performance_midpoint for prediction in self.predictions
        )
        if self.lowest_predicted_performance_midpoint != lowest_midpoint:
            raise ValueError(
                "lowest_predicted_performance_midpoint must match predictions"
            )
        highest_midpoint = max(
            prediction.predicted_performance_midpoint for prediction in self.predictions
        )
        if self.highest_predicted_performance_midpoint != highest_midpoint:
            raise ValueError(
                "highest_predicted_performance_midpoint must match predictions"
            )
        return self


def predict_performance(
    *,
    throughput_optimization: ThroughputOptimizationPlan | None = None,
    latency_optimization: LatencyOptimizationPlan | None = None,
    bottleneck_detection: BottleneckDetectionPlan | None = None,
    execution_profiling: ExecutionProfilingPlan | None = None,
) -> PerformancePredictionPlan:
    """Return advisory relative performance predictions without measurement."""

    latency = latency_optimization or optimize_latency()
    profiling = execution_profiling or plan_execution_profiling(
        latency_optimization=latency
    )
    bottlenecks = bottleneck_detection or detect_bottlenecks(
        latency_optimization=latency,
        execution_profiling=profiling,
    )
    throughput = throughput_optimization or optimize_throughput(
        bottleneck_detection=bottlenecks
    )
    predictions = _predictions(
        throughput=throughput,
        latency=latency,
        bottlenecks=bottlenecks,
        profiling=profiling,
    )
    recommended = _recommended_prediction(predictions)

    return PerformancePredictionPlan(
        source_throughput_optimization_serialization_version=(
            throughput.serialization_version
        ),
        source_latency_optimization_serialization_version=latency.serialization_version,
        source_bottleneck_detection_serialization_version=(
            bottlenecks.serialization_version
        ),
        source_execution_profiling_serialization_version=(
            profiling.serialization_version
        ),
        predictions=predictions,
        prediction_ids=tuple(prediction.prediction_id for prediction in predictions),
        recommended_prediction_id=recommended.prediction_id,
        recommended_performance_band=recommended.predicted_performance_band,
        recommended_performance_midpoint=recommended.predicted_performance_midpoint,
        fallback_prediction_ids=_prediction_ids_for_status(predictions, "fallback"),
        guardrail_prediction_ids=_prediction_ids_for_status(predictions, "guardrail"),
        prediction_count=len(predictions),
        high_prediction_count=sum(
            1
            for prediction in predictions
            if prediction.predicted_performance_band == "high"
        ),
        guarded_prediction_count=sum(
            1
            for prediction in predictions
            if prediction.predicted_performance_band == "guarded"
        ),
        lowest_predicted_performance_midpoint=min(
            prediction.predicted_performance_midpoint for prediction in predictions
        ),
        highest_predicted_performance_midpoint=max(
            prediction.predicted_performance_midpoint for prediction in predictions
        ),
        advisory_actions=_plan_actions(predictions),
    )


def performance_prediction_by_id(
    prediction_id: str,
    plan: PerformancePredictionPlan | None = None,
) -> PerformancePrediction | None:
    """Return one advisory performance prediction without measurement."""

    source_plan = plan or predict_performance()
    for prediction in source_plan.predictions:
        if prediction.prediction_id == prediction_id:
            return prediction
    return None


def performance_predictions_for_band(
    performance_band: PerformancePredictionBand,
    plan: PerformancePredictionPlan | None = None,
) -> tuple[PerformancePrediction, ...]:
    """Return advisory performance predictions for one relative band."""

    source_plan = plan or predict_performance()
    return tuple(
        prediction
        for prediction in source_plan.predictions
        if prediction.predicted_performance_band == performance_band
    )


def _predictions(
    *,
    throughput: ThroughputOptimizationPlan,
    latency: LatencyOptimizationPlan,
    bottlenecks: BottleneckDetectionPlan,
    profiling: ExecutionProfilingPlan,
) -> tuple[PerformancePrediction, ...]:
    return (
        _prediction(
            focus="overall_throughput_posture",
            status="recommended",
            source_id="throughput_optimization_plan",
            source_serialization_version=throughput.serialization_version,
            source_candidate_ids=throughput.candidate_ids,
            source_signal_score=throughput.total_advisory_throughput_score,
            source_guardrail_count=(
                throughput.capacity_guardrail_count
                + throughput.boundary_guardrail_count
            ),
            evidence=(
                f"throughput_pressure:{throughput.throughput_optimization_pressure}",
                f"throughput_score:{throughput.total_advisory_throughput_score}",
            ),
        ),
        _prediction(
            focus="latency_pressure_posture",
            status="fallback",
            source_id="latency_optimization_plan",
            source_serialization_version=latency.serialization_version,
            source_candidate_ids=latency.candidate_ids,
            source_signal_score=latency.total_advisory_latency_savings_score,
            source_guardrail_count=latency.serial_guardrail_count,
            evidence=(
                f"latency_pressure:{latency.latency_optimization_pressure}",
                f"latency_savings:{latency.total_advisory_latency_savings_score}",
            ),
        ),
        _prediction(
            focus="profile_readiness_posture",
            status="fallback",
            source_id="execution_profiling_plan",
            source_serialization_version=profiling.serialization_version,
            source_candidate_ids=profiling.candidate_ids,
            source_signal_score=profiling.total_advisory_profile_score,
            source_guardrail_count=(
                profiling.measurement_guardrail_count
                + profiling.failure_guardrail_count
            ),
            evidence=(
                f"profile_pressure:{profiling.execution_profile_pressure}",
                f"profile_score:{profiling.total_advisory_profile_score}",
            ),
        ),
        _prediction(
            focus="bottleneck_risk_posture",
            status="guardrail",
            source_id="bottleneck_detection_plan",
            source_serialization_version=bottlenecks.serialization_version,
            source_candidate_ids=bottlenecks.candidate_ids,
            source_signal_score=bottlenecks.total_advisory_severity_score,
            source_guardrail_count=(
                bottlenecks.boundary_guardrail_count + bottlenecks.review_only_count
            ),
            evidence=(
                f"bottleneck_severity:{bottlenecks.bottleneck_detection_severity}",
                f"bottleneck_score:{bottlenecks.total_advisory_severity_score}",
            ),
        ),
    )


def _prediction(
    *,
    focus: PerformancePredictionFocus,
    status: PerformancePredictionStatus,
    source_id: str,
    source_serialization_version: str,
    source_candidate_ids: tuple[str, ...],
    source_signal_score: int,
    source_guardrail_count: int,
    evidence: tuple[str, ...],
) -> PerformancePrediction:
    band = _performance_band(
        status=status,
        source_signal_score=source_signal_score,
        source_guardrail_count=source_guardrail_count,
    )
    performance_range = _PERFORMANCE_RANGE_BY_BAND[band]
    return PerformancePrediction(
        prediction_id=f"performance_prediction::{focus}",
        prediction_focus=focus,
        status=status,
        source_id=source_id,
        source_serialization_version=source_serialization_version,
        source_candidate_ids=source_candidate_ids,
        source_signal_score=source_signal_score,
        source_guardrail_count=source_guardrail_count,
        predicted_performance_band=band,
        predicted_performance_range=performance_range,
        predicted_performance_midpoint=(performance_range[0] + performance_range[1])
        // 2,
        prediction_confidence=_confidence(
            band=band,
            source_signal_score=source_signal_score,
        ),
        evidence=evidence,
        advisory_actions=_prediction_actions(status),
    )


def _prediction_ids_for_status(
    predictions: tuple[PerformancePrediction, ...],
    status: PerformancePredictionStatus,
) -> tuple[str, ...]:
    return tuple(
        prediction.prediction_id
        for prediction in predictions
        if prediction.status == status
    )


def _performance_band(
    *,
    status: PerformancePredictionStatus,
    source_signal_score: int,
    source_guardrail_count: int,
) -> PerformancePredictionBand:
    if status == "guardrail" or source_guardrail_count:
        return "guarded"
    if source_signal_score >= 1_200:
        return "high"
    if source_signal_score >= 400:
        return "medium"
    return "low"


def _confidence(
    *,
    band: PerformancePredictionBand,
    source_signal_score: int,
) -> PerformancePredictionConfidence:
    if band == "guarded":
        return "medium"
    if source_signal_score >= 1_200:
        return "high"
    if source_signal_score >= 400:
        return "medium"
    return "low"


def _recommended_prediction(
    predictions: tuple[PerformancePrediction, ...],
) -> PerformancePrediction:
    for prediction in predictions:
        if prediction.status == "recommended":
            return prediction
    raise ValueError("performance prediction requires a recommended prediction")


def _prediction_actions(
    status: PerformancePredictionStatus,
) -> tuple[str, ...]:
    if status == "recommended":
        return (
            "Expose relative performance prediction as advisory metadata only.",
            "Require explicit runtime authority before measurement or tuning.",
        )
    if status == "guardrail":
        return (
            "Keep bottleneck risk prediction guarded and review-only.",
            "Preserve measurement, routing, workflow, and output boundaries.",
        )
    return ("Present performance prediction as fallback advisory evidence.",)


def _plan_actions(
    predictions: tuple[PerformancePrediction, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose performance predictions as relative advisory metadata only.",
        "Preserve measurement, benchmarking, profiling, routing, workflow, and "
        "output boundaries.",
    ]
    if any(
        prediction.predicted_performance_band == "guarded" for prediction in predictions
    ):
        actions.append("Require human review before any runtime performance change.")
    return tuple(actions)
