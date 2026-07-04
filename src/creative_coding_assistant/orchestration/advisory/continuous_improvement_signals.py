"""V6.1 advisory continuous improvement signals."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_learning_engine import (
    AdaptiveLearningPlan,
    AdaptiveLearningSignal,
    evaluate_adaptive_learning_engine,
)
from creative_coding_assistant.orchestration.artifact_learning import (
    ArtifactLearningPlan,
    learn_artifacts,
)
from creative_coding_assistant.orchestration.evaluation_learning import (
    EvaluationLearningPlan,
    learn_evaluations,
)
from creative_coding_assistant.orchestration.failure_tracking import (
    FailureTrackingPlan,
    track_failures,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)
from creative_coding_assistant.orchestration.workflow_success_tracking import (
    WorkflowSuccessTrackingPlan,
    track_workflow_success,
)

ContinuousImprovementSignalKind = Literal[
    "success_improvement_signal",
    "failure_prevention_signal",
    "artifact_improvement_signal",
    "evaluation_improvement_signal",
]
ContinuousImprovementStatus = Literal["candidate", "review_required", "guarded"]
ContinuousImprovementPriority = Literal["standard", "elevated", "critical", "guarded"]
ContinuousImprovementPosture = Literal["candidate", "review_required", "guarded"]

CONTINUOUS_IMPROVEMENT_SIGNAL_SERIALIZATION_VERSION = "continuous_improvement_signal.v1"
CONTINUOUS_IMPROVEMENT_PLAN_SERIALIZATION_VERSION = (
    "continuous_improvement_signal_plan.v1"
)
CONTINUOUS_IMPROVEMENT_AUTHORITY_BOUNDARY = (
    "V6.1 continuous improvement signals synthesize read-only V6.1 learning "
    "metadata into advisory improvement candidates only; they do not apply "
    "feedback, persist learning memory, update policies, mutate strategies, "
    "change provider or model routing, execute providers, invoke agents, "
    "allocate resources, emit HITL requests, observe runtime outcomes, "
    "evaluate generated output, execute or control workflows, mutate workflow "
    "graphs, trigger retries or refinements, mutate prompts, write storage, "
    "modify generated output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "learning_feedback_application",
    "learning_memory_persistence",
    "learning_policy_update",
    "strategy_mutation",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
    "provider_execution",
    "agent_invocation",
    "resource_allocation",
    "hitl_request_emission",
    "runtime_outcome_observation",
    "generated_output_evaluation",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class ContinuousImprovementSignal(BaseModel):
    """One advisory continuous improvement signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: ContinuousImprovementSignalKind
    status: ContinuousImprovementStatus
    priority: ContinuousImprovementPriority
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_plan_role: str = Field(min_length=1, max_length=120)
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    source_workflow_risk_factor_id: str = Field(min_length=1, max_length=180)
    source_score: int = Field(ge=0, le=1_000)
    source_item_count: int = Field(ge=0, le=20)
    source_review_required_count: int = Field(ge=0, le=20)
    source_guarded_count: int = Field(ge=0, le=20)
    source_hitl_required_count: int = Field(ge=0, le=20)
    learning_priority_score: int = Field(ge=0, le=1_000)
    improvement_weight: int = Field(ge=0, le=240)
    improvement_score: int = Field(ge=0, le=1_000)
    hitl_required: bool
    improvement_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    improvement_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    continuous_improvement_signals_implemented: Literal[True] = True
    improvement_signal_metadata_implemented: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    source_learning_metadata_used: Literal[True] = True
    learning_feedback_application_implemented: Literal[False] = False
    learning_memory_persistence_implemented: Literal[False] = False
    learning_policy_update_implemented: Literal[False] = False
    strategy_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    runtime_outcome_observation_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["continuous_improvement_signal.v1"] = (
        CONTINUOUS_IMPROVEMENT_SIGNAL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"continuous_improvement::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.improvement_score != _improvement_score(
            source_score=self.source_score,
            source_item_count=self.source_item_count,
            source_review_required_count=self.source_review_required_count,
            source_guarded_count=self.source_guarded_count,
            source_hitl_required_count=self.source_hitl_required_count,
            learning_priority_score=self.learning_priority_score,
            improvement_weight=self.improvement_weight,
        ):
            raise ValueError("improvement_score must combine source scores")
        if self.priority != _improvement_priority(self.improvement_score, self.status):
            raise ValueError("priority must match score and status")
        if self.status == "guarded" and not self.hitl_required:
            raise ValueError("guarded improvement signals require HITL posture")
        return self


class ContinuousImprovementSignalPlan(BaseModel):
    """Bounded V6.1 advisory continuous improvement signal plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["continuous_improvement_signals"] = "continuous_improvement_signals"
    serialization_version: Literal["continuous_improvement_signal_plan.v1"] = (
        CONTINUOUS_IMPROVEMENT_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CONTINUOUS_IMPROVEMENT_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_adaptive_learning_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_plan_roles: tuple[str, ...] = Field(min_length=4, max_length=4)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[ContinuousImprovementSignal, ...] = Field(
        min_length=4,
        max_length=4,
    )
    signal_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    candidate_signal_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    review_required_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    guarded_signal_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    standard_priority_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    elevated_priority_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    critical_priority_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    guarded_priority_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    hitl_required_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    applied_improvement_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    signal_count: int = Field(ge=4, le=4)
    review_required_signal_count: int = Field(ge=0, le=4)
    guarded_signal_count: int = Field(ge=0, le=4)
    hitl_required_signal_count: int = Field(ge=0, le=4)
    highest_improvement_score: int = Field(ge=0, le=1_000)
    overall_improvement_score: int = Field(ge=0, le=1_000)
    overall_improvement_posture: ContinuousImprovementPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    continuous_improvement_signals_implemented: Literal[True] = True
    improvement_signal_metadata_implemented: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    source_learning_metadata_used: Literal[True] = True
    learning_feedback_application_implemented: Literal[False] = False
    learning_memory_persistence_implemented: Literal[False] = False
    learning_policy_update_implemented: Literal[False] = False
    strategy_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    runtime_outcome_observation_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
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
        if self.signal_ids != derived_signal_ids:
            raise ValueError("signal_ids must match signals")
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
        if self.guarded_signal_ids != _signal_ids_for_status(
            self.signals,
            "guarded",
        ):
            raise ValueError("guarded_signal_ids must match signals")
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
        if self.applied_improvement_signal_ids:
            raise ValueError("applied_improvement_signal_ids must remain empty")
        if self.signal_count != len(self.signals):
            raise ValueError("signal_count must match signals")
        if self.review_required_signal_count != len(self.review_required_signal_ids):
            raise ValueError("review_required_signal_count must match signals")
        if self.guarded_signal_count != len(self.guarded_signal_ids):
            raise ValueError("guarded_signal_count must match signals")
        if self.hitl_required_signal_count != len(self.hitl_required_signal_ids):
            raise ValueError("hitl_required_signal_count must match signals")
        if self.highest_improvement_score != max(
            signal.improvement_score for signal in self.signals
        ):
            raise ValueError("highest_improvement_score must match signals")
        if self.overall_improvement_score != _overall_improvement_score(self.signals):
            raise ValueError("overall_improvement_score must match signals")
        if self.overall_improvement_posture != _overall_improvement_posture(
            self.signals,
        ):
            raise ValueError("overall_improvement_posture must match signals")
        if self.source_plan_roles != tuple(
            signal.source_plan_role for signal in self.signals
        ):
            raise ValueError("source_plan_roles must match signals")
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.task_type != self.task_type:
                raise ValueError("signal task_type must match plan")
        return self


def derive_continuous_improvement_signals(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    adaptive_learning: AdaptiveLearningPlan | None = None,
    workflow_success: WorkflowSuccessTrackingPlan | None = None,
    failure_tracking: FailureTrackingPlan | None = None,
    artifact_learning: ArtifactLearningPlan | None = None,
    evaluation_learning: EvaluationLearningPlan | None = None,
) -> ContinuousImprovementSignalPlan:
    """Derive continuous improvement signals without applying feedback."""

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
    signals = _signals(
        route_name=route_name,
        task_type=learning_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        adaptive_learning=learning_plan,
        workflow_success=workflow_success
        or track_workflow_success(
            route=route_name,
            task_type=learning_plan.task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=learning_plan,
        ),
        failure_tracking=failure_tracking
        or track_failures(
            route=route_name,
            task_type=learning_plan.task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=learning_plan,
        ),
        artifact_learning=artifact_learning
        or learn_artifacts(
            route=route_name,
            task_type=learning_plan.task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=learning_plan,
        ),
        evaluation_learning=evaluation_learning
        or learn_evaluations(
            route=route_name,
            task_type=learning_plan.task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=learning_plan,
        ),
    )
    return ContinuousImprovementSignalPlan(
        route_name=route_name,
        task_type=learning_plan.task_type,
        source_adaptive_learning_serialization_version=(
            learning_plan.serialization_version
        ),
        source_plan_roles=tuple(signal.source_plan_role for signal in signals),
        execution_mode_ids=execution_modes.execution_mode_ids,
        signals=signals,
        signal_ids=tuple(signal.signal_id for signal in signals),
        candidate_signal_ids=_signal_ids_for_status(signals, "candidate"),
        review_required_signal_ids=_signal_ids_for_status(
            signals,
            "review_required",
        ),
        guarded_signal_ids=_signal_ids_for_status(signals, "guarded"),
        standard_priority_signal_ids=_signal_ids_for_priority(signals, "standard"),
        elevated_priority_signal_ids=_signal_ids_for_priority(signals, "elevated"),
        critical_priority_signal_ids=_signal_ids_for_priority(signals, "critical"),
        guarded_priority_signal_ids=_signal_ids_for_priority(signals, "guarded"),
        hitl_required_signal_ids=tuple(
            signal.signal_id for signal in signals if signal.hitl_required
        ),
        applied_improvement_signal_ids=(),
        signal_count=len(signals),
        review_required_signal_count=len(
            _signal_ids_for_status(signals, "review_required")
        ),
        guarded_signal_count=len(_signal_ids_for_status(signals, "guarded")),
        hitl_required_signal_count=sum(1 for signal in signals if signal.hitl_required),
        highest_improvement_score=max(signal.improvement_score for signal in signals),
        overall_improvement_score=_overall_improvement_score(signals),
        overall_improvement_posture=_overall_improvement_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def continuous_improvement_signal_by_id(
    signal_id: str,
    plan: ContinuousImprovementSignalPlan | None = None,
) -> ContinuousImprovementSignal | None:
    source_plan = plan or derive_continuous_improvement_signals()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def continuous_improvement_signals_for_status(
    status: ContinuousImprovementStatus,
    plan: ContinuousImprovementSignalPlan | None = None,
) -> tuple[ContinuousImprovementSignal, ...]:
    source_plan = plan or derive_continuous_improvement_signals()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def continuous_improvement_signals_for_priority(
    priority: ContinuousImprovementPriority,
    plan: ContinuousImprovementSignalPlan | None = None,
) -> tuple[ContinuousImprovementSignal, ...]:
    source_plan = plan or derive_continuous_improvement_signals()
    return tuple(
        signal for signal in source_plan.signals if signal.priority == priority
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    workflow_success: WorkflowSuccessTrackingPlan,
    failure_tracking: FailureTrackingPlan,
    artifact_learning: ArtifactLearningPlan,
    evaluation_learning: EvaluationLearningPlan,
) -> tuple[ContinuousImprovementSignal, ...]:
    return (
        _signal(
            kind="success_improvement_signal",
            source_plan_role=workflow_success.role,
            source_score=workflow_success.overall_workflow_success_score,
            source_item_count=workflow_success.indicator_count,
            source_review_required_count=(
                workflow_success.review_required_indicator_count
            ),
            source_guarded_count=workflow_success.guarded_indicator_count,
            source_hitl_required_count=workflow_success.hitl_required_indicator_count,
            learning_signal_id="adaptive_learning::workflow_pattern_learning",
            status="review_required",
            weight=220,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
        ),
        _signal(
            kind="failure_prevention_signal",
            source_plan_role=failure_tracking.role,
            source_score=failure_tracking.overall_failure_tracking_score,
            source_item_count=failure_tracking.indicator_count,
            source_review_required_count=0,
            source_guarded_count=failure_tracking.guarded_indicator_count,
            source_hitl_required_count=failure_tracking.hitl_required_indicator_count,
            learning_signal_id="adaptive_learning::routing_boundary_learning",
            status="guarded",
            weight=230,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
        ),
        _signal(
            kind="artifact_improvement_signal",
            source_plan_role=artifact_learning.role,
            source_score=artifact_learning.overall_artifact_learning_score,
            source_item_count=artifact_learning.pattern_count,
            source_review_required_count=artifact_learning.review_required_pattern_count,
            source_guarded_count=artifact_learning.guarded_pattern_count,
            source_hitl_required_count=artifact_learning.hitl_required_pattern_count,
            learning_signal_id="adaptive_learning::governance_feedback_learning",
            status="review_required",
            weight=190,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
        ),
        _signal(
            kind="evaluation_improvement_signal",
            source_plan_role=evaluation_learning.role,
            source_score=evaluation_learning.overall_evaluation_learning_score,
            source_item_count=evaluation_learning.pattern_count,
            source_review_required_count=(
                evaluation_learning.review_required_pattern_count
            ),
            source_guarded_count=evaluation_learning.guarded_pattern_count,
            source_hitl_required_count=evaluation_learning.hitl_required_pattern_count,
            learning_signal_id="adaptive_learning::runtime_guardrail_learning",
            status="guarded",
            weight=210,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
        ),
    )


def _signal(
    *,
    kind: ContinuousImprovementSignalKind,
    source_plan_role: str,
    source_score: int,
    source_item_count: int,
    source_review_required_count: int,
    source_guarded_count: int,
    source_hitl_required_count: int,
    learning_signal_id: str,
    status: ContinuousImprovementStatus,
    weight: int,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
) -> ContinuousImprovementSignal:
    learning_signal = _required_learning_signal(learning_signal_id, adaptive_learning)
    score = _improvement_score(
        source_score=source_score,
        source_item_count=source_item_count,
        source_review_required_count=source_review_required_count,
        source_guarded_count=source_guarded_count,
        source_hitl_required_count=source_hitl_required_count,
        learning_priority_score=learning_signal.learning_priority_score,
        improvement_weight=weight,
    )
    return ContinuousImprovementSignal(
        signal_id=f"continuous_improvement::{kind}",
        signal_kind=kind,
        status=status,
        priority=_improvement_priority(score, status),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_plan_role=source_plan_role,
        source_learning_signal_id=learning_signal.signal_id,
        source_workflow_risk_factor_id=learning_signal.source_workflow_risk_factor_id,
        source_score=source_score,
        source_item_count=source_item_count,
        source_review_required_count=source_review_required_count,
        source_guarded_count=source_guarded_count,
        source_hitl_required_count=source_hitl_required_count,
        learning_priority_score=learning_signal.learning_priority_score,
        improvement_weight=weight,
        improvement_score=score,
        hitl_required=learning_signal.hitl_required or status == "guarded",
        improvement_tags=(source_plan_role, kind.removesuffix("_signal")),
        improvement_summary=_improvement_summary(kind, status),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"source_plan:{source_plan_role}",
            f"source_score:{source_score}",
            f"source_items:{source_item_count}",
            f"review_required:{source_review_required_count}",
            f"guarded:{source_guarded_count}",
            f"hitl_required:{source_hitl_required_count}",
            f"learning_signal:{learning_signal.signal_id}",
        ),
    )


def _improvement_score(
    *,
    source_score: int,
    source_item_count: int,
    source_review_required_count: int,
    source_guarded_count: int,
    source_hitl_required_count: int,
    learning_priority_score: int,
    improvement_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            source_score
            + learning_priority_score // 3
            + improvement_weight
            + source_review_required_count * 40
            + source_guarded_count * 50
            + source_hitl_required_count * 20
            - source_item_count * 5,
        ),
    )


def _improvement_priority(
    score: int,
    status: ContinuousImprovementStatus,
) -> ContinuousImprovementPriority:
    if status == "guarded":
        return "guarded"
    if score >= 840:
        return "critical"
    if score >= 620:
        return "elevated"
    return "standard"


def _signal_ids_for_status(
    signals: tuple[ContinuousImprovementSignal, ...],
    status: ContinuousImprovementStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_priority(
    signals: tuple[ContinuousImprovementSignal, ...],
    priority: ContinuousImprovementPriority,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.priority == priority)


def _overall_improvement_score(
    signals: tuple[ContinuousImprovementSignal, ...],
) -> int:
    return sum(signal.improvement_score for signal in signals) // len(signals)


def _overall_improvement_posture(
    signals: tuple[ContinuousImprovementSignal, ...],
) -> ContinuousImprovementPosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.hitl_required for signal in signals):
        return "review_required"
    return "candidate"


def _required_learning_signal(
    signal_id: str,
    plan: AdaptiveLearningPlan,
) -> AdaptiveLearningSignal:
    for signal in plan.signals:
        if signal.signal_id == signal_id:
            return signal
    raise ValueError("required continuous improvement metadata is missing")


def _improvement_summary(
    kind: ContinuousImprovementSignalKind,
    status: ContinuousImprovementStatus,
) -> str:
    if status == "guarded":
        return f"Surface {kind} as guarded improvement metadata without feedback."
    if status == "review_required":
        return f"Surface {kind} for review before future improvement behavior."
    return f"Surface {kind} as candidate improvement metadata only."


def _signal_actions(kind: ContinuousImprovementSignalKind) -> tuple[str, ...]:
    return (
        f"Expose {kind} as derived continuous improvement metadata.",
        "Keep feedback application, memory persistence, policy updates, "
        "routing, workflow control, retries, storage, Runtime Evolution, "
        "and output mutation disabled.",
    )


def _plan_actions(
    signals: tuple[ContinuousImprovementSignal, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose continuous improvement signals as advisory metadata only.",
        "Keep applied improvement signal ids empty.",
        "Preserve feedback, memory, policy, routing, provider, workflow, "
        "retry, storage, output, and Runtime Evolution boundaries.",
    ]
    if any(signal.hitl_required for signal in signals):
        actions.append("Require review before any future improvement behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
