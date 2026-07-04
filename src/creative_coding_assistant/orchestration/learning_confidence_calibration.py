"""V6.1 advisory learning confidence calibration metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_learning_engine import (
    AdaptiveLearningPlan,
    AdaptiveLearningSignal,
    evaluate_adaptive_learning_engine,
)
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

LearningConfidenceBand = Literal["strong", "moderate", "weak", "guarded"]
LearningCalibrationStatus = Literal["calibrated", "review_required", "guarded"]
LearningCalibrationPosture = Literal["calibrated", "review_required", "guarded"]

LEARNING_CONFIDENCE_CALIBRATION_RECORD_SERIALIZATION_VERSION = (
    "learning_confidence_calibration_record.v1"
)
LEARNING_CONFIDENCE_CALIBRATION_PLAN_SERIALIZATION_VERSION = (
    "learning_confidence_calibration_plan.v1"
)
LEARNING_CONFIDENCE_CALIBRATION_AUTHORITY_BOUNDARY = (
    "V6.1 learning confidence calibration classifies confidence from existing "
    "adaptive learning signals and execution confidence metadata only; it does "
    "not train models, apply feedback, write storage, mutate runtime behavior, "
    "change provider/model routing, mutate generated output, or apply Runtime "
    "Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "model_training",
    "learning_feedback_application",
    "persistent_storage_write",
    "runtime_mutation",
    "provider_or_model_routing",
    "generated_output_modification",
    "runtime_evolution_application",
)


class LearningConfidenceCalibrationRecord(BaseModel):
    """One advisory confidence calibration record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    calibration_id: str = Field(min_length=1, max_length=200)
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    source_execution_confidence_signal_id: str = Field(min_length=1, max_length=180)
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    confidence_before: int = Field(ge=0, le=100)
    confidence_after: int = Field(ge=0, le=100)
    confidence_band_before: LearningConfidenceBand
    confidence_band_after: LearningConfidenceBand
    calibration_status: LearningCalibrationStatus
    calibration_rationale: str = Field(min_length=1, max_length=420)
    uncertainty_factors: tuple[str, ...] = Field(min_length=1, max_length=8)
    hitl_required: bool
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=10,
    )
    learning_confidence_calibration_implemented: Literal[True] = True
    calibration_metadata_implemented: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    execution_confidence_metadata_used: Literal[True] = True
    model_training_implemented: Literal[False] = False
    learning_feedback_application_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    runtime_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["learning_confidence_calibration_record.v1"] = (
        LEARNING_CONFIDENCE_CALIBRATION_RECORD_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_matches_contract(self) -> Self:
        if self.calibration_id != (
            f"learning_confidence::{self.source_learning_signal_id}"
        ):
            raise ValueError("calibration_id must match source learning signal")
        if self.confidence_band_before != _confidence_band(self.confidence_before):
            raise ValueError("confidence_band_before must match confidence_before")
        if self.confidence_band_after != _confidence_band(self.confidence_after):
            raise ValueError("confidence_band_after must match confidence_after")
        if self.calibration_status != _calibration_status(
            self.confidence_band_after,
            self.hitl_required,
        ):
            raise ValueError("calibration_status must match calibrated confidence")
        if self.confidence_after > self.confidence_before:
            raise ValueError("confidence_after cannot exceed confidence_before")
        return self


class LearningConfidenceCalibrationPlan(BaseModel):
    """Bounded V6.1 advisory learning confidence calibration plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["learning_confidence_calibration"] = "learning_confidence_calibration"
    serialization_version: Literal["learning_confidence_calibration_plan.v1"] = (
        LEARNING_CONFIDENCE_CALIBRATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=LEARNING_CONFIDENCE_CALIBRATION_AUTHORITY_BOUNDARY,
        max_length=1400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_adaptive_learning_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_execution_confidence_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    records: tuple[LearningConfidenceCalibrationRecord, ...] = Field(
        min_length=5,
        max_length=5,
    )
    calibration_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    strong_confidence_calibration_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    moderate_confidence_calibration_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    weak_confidence_calibration_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_confidence_calibration_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    hitl_required_calibration_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    record_count: int = Field(ge=5, le=5)
    average_confidence_before: int = Field(ge=0, le=100)
    average_confidence_after: int = Field(ge=0, le=100)
    overall_calibration_posture: LearningCalibrationPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=10,
    )
    learning_confidence_calibration_implemented: Literal[True] = True
    calibration_metadata_implemented: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    execution_confidence_metadata_used: Literal[True] = True
    model_training_implemented: Literal[False] = False
    learning_feedback_application_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    runtime_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_records(self) -> Self:
        derived_ids = tuple(record.calibration_id for record in self.records)
        if len(set(derived_ids)) != len(derived_ids):
            raise ValueError("calibration_ids must be unique")
        if self.calibration_ids != derived_ids:
            raise ValueError("calibration_ids must match records")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.strong_confidence_calibration_ids != _record_ids_for_band(
            self.records,
            "strong",
        ):
            raise ValueError("strong_confidence_calibration_ids must match records")
        if self.moderate_confidence_calibration_ids != _record_ids_for_band(
            self.records,
            "moderate",
        ):
            raise ValueError("moderate_confidence_calibration_ids must match records")
        if self.weak_confidence_calibration_ids != _record_ids_for_band(
            self.records,
            "weak",
        ):
            raise ValueError("weak_confidence_calibration_ids must match records")
        if self.guarded_confidence_calibration_ids != _record_ids_for_band(
            self.records,
            "guarded",
        ):
            raise ValueError("guarded_confidence_calibration_ids must match records")
        if self.hitl_required_calibration_ids != tuple(
            record.calibration_id for record in self.records if record.hitl_required
        ):
            raise ValueError("hitl_required_calibration_ids must match records")
        if self.average_confidence_before != _average_confidence_before(self.records):
            raise ValueError("average_confidence_before must match records")
        if self.average_confidence_after != _average_confidence_after(self.records):
            raise ValueError("average_confidence_after must match records")
        if self.overall_calibration_posture != _overall_calibration_posture(
            self.records,
        ):
            raise ValueError("overall_calibration_posture must match records")
        for record in self.records:
            if record.route_name != self.route_name:
                raise ValueError("record route_name must match plan")
            if record.task_type != self.task_type:
                raise ValueError("record task_type must match plan")
        return self


def calibrate_learning_confidence(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    adaptive_learning: AdaptiveLearningPlan | None = None,
    execution_confidence: ExecutionConfidencePlan | None = None,
) -> LearningConfidenceCalibrationPlan:
    """Calibrate learning confidence without training or mutating runtime."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    confidence_plan = execution_confidence or evaluate_execution_confidence(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    learning_plan = adaptive_learning or evaluate_adaptive_learning_engine(
        route=route_name,
        task_type=confidence_plan.task_type,
        execution_mode_id=execution_mode_id,
        execution_confidence=confidence_plan,
    )
    normalized_mode = str(
        execution_mode_id or learning_plan.signals[0].execution_mode_id
    )
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")
    records = _records(
        route_name=route_name,
        task_type=learning_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        adaptive_learning=learning_plan,
        execution_confidence=confidence_plan,
    )
    return LearningConfidenceCalibrationPlan(
        route_name=route_name,
        task_type=learning_plan.task_type,
        source_adaptive_learning_serialization_version=(
            learning_plan.serialization_version
        ),
        source_execution_confidence_serialization_version=(
            confidence_plan.serialization_version
        ),
        execution_mode_ids=execution_modes.execution_mode_ids,
        records=records,
        calibration_ids=tuple(record.calibration_id for record in records),
        strong_confidence_calibration_ids=_record_ids_for_band(records, "strong"),
        moderate_confidence_calibration_ids=_record_ids_for_band(records, "moderate"),
        weak_confidence_calibration_ids=_record_ids_for_band(records, "weak"),
        guarded_confidence_calibration_ids=_record_ids_for_band(records, "guarded"),
        hitl_required_calibration_ids=tuple(
            record.calibration_id for record in records if record.hitl_required
        ),
        record_count=len(records),
        average_confidence_before=_average_confidence_before(records),
        average_confidence_after=_average_confidence_after(records),
        overall_calibration_posture=_overall_calibration_posture(records),
        advisory_actions=_plan_actions(records),
    )


def learning_confidence_calibration_by_id(
    calibration_id: str,
    plan: LearningConfidenceCalibrationPlan | None = None,
) -> LearningConfidenceCalibrationRecord | None:
    """Return one calibration record without applying it."""

    source_plan = plan or calibrate_learning_confidence()
    normalized_id = str(calibration_id).strip()
    for record in source_plan.records:
        if record.calibration_id == normalized_id:
            return record
    return None


def learning_confidence_calibrations_for_band(
    confidence_band: LearningConfidenceBand,
    plan: LearningConfidenceCalibrationPlan | None = None,
) -> tuple[LearningConfidenceCalibrationRecord, ...]:
    """Return calibration records by calibrated confidence band."""

    source_plan = plan or calibrate_learning_confidence()
    return tuple(
        record
        for record in source_plan.records
        if record.confidence_band_after == confidence_band
    )


def _records(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    execution_confidence: ExecutionConfidencePlan,
) -> tuple[LearningConfidenceCalibrationRecord, ...]:
    return tuple(
        _record(
            signal=signal,
            confidence_signal=_required_confidence_signal(
                signal.source_execution_confidence_signal_id,
                execution_confidence,
            ),
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
        )
        for signal in adaptive_learning.signals
    )


def _record(
    *,
    signal: AdaptiveLearningSignal,
    confidence_signal: ExecutionConfidenceSignal,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
) -> LearningConfidenceCalibrationRecord:
    confidence_before = confidence_signal.execution_confidence_score
    confidence_after = _confidence_after(signal, confidence_before)
    band_after = _confidence_band(confidence_after)
    hitl_required = (
        confidence_after < 60
        or signal.status != "candidate"
        or confidence_signal.hitl_required
    )
    return LearningConfidenceCalibrationRecord(
        calibration_id=f"learning_confidence::{signal.signal_id}",
        source_learning_signal_id=signal.signal_id,
        source_execution_confidence_signal_id=confidence_signal.signal_id,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        confidence_before=confidence_before,
        confidence_after=confidence_after,
        confidence_band_before=_confidence_band(confidence_before),
        confidence_band_after=band_after,
        calibration_status=_calibration_status(band_after, hitl_required),
        calibration_rationale=_calibration_rationale(signal, confidence_after),
        uncertainty_factors=_uncertainty_factors(signal, confidence_signal),
        hitl_required=hitl_required,
        advisory_actions=_record_actions(signal),
        evidence=(
            f"learning_signal:{signal.signal_id}",
            f"execution_confidence:{confidence_signal.signal_id}",
            f"confidence_before:{confidence_before}",
            f"confidence_after:{confidence_after}",
            f"learning_priority_score:{signal.learning_priority_score}",
            f"learning_status:{signal.status}",
        ),
    )


def _required_confidence_signal(
    signal_id: str,
    plan: ExecutionConfidencePlan,
) -> ExecutionConfidenceSignal:
    signal = execution_confidence_signal_by_id(signal_id, plan)
    if signal is None:
        raise ValueError("required execution confidence signal metadata is missing")
    return signal


def _confidence_after(
    signal: AdaptiveLearningSignal,
    confidence_before: int,
) -> int:
    status_penalty = {
        "candidate": 0,
        "review_required": 10,
        "guardrail": 25,
    }[signal.status]
    uncertainty_penalty = min(
        35,
        signal.unavailable_reason_count * 4
        + signal.guardrail_signal_count // 10
        + signal.learning_priority_score // 80,
    )
    return max(0, min(100, confidence_before - status_penalty - uncertainty_penalty))


def _confidence_band(score: int) -> LearningConfidenceBand:
    if score >= 80:
        return "strong"
    if score >= 60:
        return "moderate"
    if score >= 40:
        return "weak"
    return "guarded"


def _calibration_status(
    band_after: LearningConfidenceBand,
    hitl_required: bool,
) -> LearningCalibrationStatus:
    if band_after == "guarded":
        return "guarded"
    if hitl_required:
        return "review_required"
    return "calibrated"


def _record_ids_for_band(
    records: tuple[LearningConfidenceCalibrationRecord, ...],
    confidence_band: LearningConfidenceBand,
) -> tuple[str, ...]:
    return tuple(
        record.calibration_id
        for record in records
        if record.confidence_band_after == confidence_band
    )


def _average_confidence_before(
    records: tuple[LearningConfidenceCalibrationRecord, ...],
) -> int:
    return sum(record.confidence_before for record in records) // len(records)


def _average_confidence_after(
    records: tuple[LearningConfidenceCalibrationRecord, ...],
) -> int:
    return sum(record.confidence_after for record in records) // len(records)


def _overall_calibration_posture(
    records: tuple[LearningConfidenceCalibrationRecord, ...],
) -> LearningCalibrationPosture:
    if any(record.calibration_status == "guarded" for record in records):
        return "guarded"
    if any(record.hitl_required for record in records):
        return "review_required"
    return "calibrated"


def _uncertainty_factors(
    signal: AdaptiveLearningSignal,
    confidence_signal: ExecutionConfidenceSignal,
) -> tuple[str, ...]:
    factors = [
        f"learning_status:{signal.status}",
        f"learning_priority_score:{signal.learning_priority_score}",
        f"execution_confidence_status:{confidence_signal.status}",
    ]
    if signal.unavailable_reason_count:
        factors.append(f"unavailable_reasons:{signal.unavailable_reason_count}")
    if signal.guardrail_signal_count:
        factors.append(f"guardrail_signals:{signal.guardrail_signal_count}")
    if signal.hitl_required or confidence_signal.hitl_required:
        factors.append("hitl_required_source")
    return tuple(factors)


def _calibration_rationale(
    signal: AdaptiveLearningSignal,
    confidence_after: int,
) -> str:
    band = _confidence_band(confidence_after)
    return (
        f"Classify {signal.signal_kind} confidence as {band} "
        "after accounting for learning priority, guardrails, unavailable reasons, "
        "and source HITL posture."
    )


def _record_actions(signal: AdaptiveLearningSignal) -> tuple[str, ...]:
    return (
        f"Expose calibrated confidence for {signal.signal_kind}.",
        "Require HITL for low, guarded, or risky learning confidence.",
        "Keep training, feedback, storage, runtime mutation, and Runtime "
        "Evolution disabled.",
    )


def _plan_actions(
    records: tuple[LearningConfidenceCalibrationRecord, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose learning confidence calibration as advisory metadata only.",
        "Keep model training, feedback application, storage writes, runtime "
        "mutation, and Runtime Evolution disabled.",
    ]
    if any(record.hitl_required for record in records):
        actions.append("Require HITL before using low or risky calibrated confidence.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
