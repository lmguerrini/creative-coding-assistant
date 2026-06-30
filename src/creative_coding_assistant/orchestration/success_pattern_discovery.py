"""V6.1 advisory success pattern discovery."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_learning_engine import (
    AdaptiveLearningPlan,
    AdaptiveLearningSignal,
    evaluate_adaptive_learning_engine,
)
from creative_coding_assistant.orchestration.continuous_improvement_signals import (
    ContinuousImprovementSignal,
    ContinuousImprovementSignalPlan,
    derive_continuous_improvement_signals,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)
from creative_coding_assistant.orchestration.workflow_success_tracking import (
    WorkflowSuccessIndicator,
    WorkflowSuccessTrackingPlan,
    track_workflow_success,
)

SuccessPatternKind = Literal[
    "execution_success_pattern",
    "routing_success_pattern",
    "artifact_success_pattern",
    "evaluation_success_pattern",
]
SuccessPatternStatus = Literal["discovered", "review_required", "guarded"]
SuccessPatternPriority = Literal["standard", "elevated", "critical", "guarded"]
SuccessPatternPosture = Literal["discovered", "review_required", "guarded"]

SUCCESS_PATTERN_SERIALIZATION_VERSION = "success_pattern.v1"
SUCCESS_PATTERN_DISCOVERY_PLAN_SERIALIZATION_VERSION = (
    "success_pattern_discovery_plan.v1"
)
SUCCESS_PATTERN_DISCOVERY_AUTHORITY_BOUNDARY = (
    "V6.1 success pattern discovery derives candidate success patterns from "
    "read-only workflow success, continuous improvement, and adaptive learning "
    "metadata only; it does not observe runtime success, collect telemetry, "
    "persist success metrics, apply feedback, persist learning memory, update "
    "policies, change provider or model routing, execute providers, invoke "
    "agents, emit HITL requests, evaluate generated output, execute or control "
    "workflows, mutate workflow graphs, trigger retries or refinements, mutate "
    "prompts, write storage, modify generated output, or apply Runtime "
    "Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_success_observation",
    "live_telemetry_collection",
    "success_metric_persistence",
    "success_pattern_application",
    "learning_feedback_application",
    "learning_memory_persistence",
    "learning_policy_update",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
    "provider_execution",
    "agent_invocation",
    "hitl_request_emission",
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


class SuccessPattern(BaseModel):
    """One advisory discovered success pattern."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    pattern_id: str = Field(min_length=1, max_length=180)
    pattern_kind: SuccessPatternKind
    status: SuccessPatternStatus
    priority: SuccessPatternPriority
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_success_indicator_id: str = Field(min_length=1, max_length=180)
    source_improvement_signal_id: str = Field(min_length=1, max_length=180)
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    source_workflow_risk_factor_id: str = Field(min_length=1, max_length=180)
    source_success_status: str = Field(min_length=1, max_length=80)
    source_improvement_status: str = Field(min_length=1, max_length=80)
    success_confidence_band: str = Field(min_length=1, max_length=80)
    workflow_success_score: int = Field(ge=0, le=1_000)
    improvement_score: int = Field(ge=0, le=1_000)
    learning_priority_score: int = Field(ge=0, le=1_000)
    success_pattern_weight: int = Field(ge=0, le=240)
    success_pattern_score: int = Field(ge=0, le=1_000)
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
    success_pattern_discovery_implemented: Literal[True] = True
    success_pattern_metadata_implemented: Literal[True] = True
    workflow_success_metadata_used: Literal[True] = True
    continuous_improvement_metadata_used: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    runtime_success_observation_implemented: Literal[False] = False
    live_telemetry_collection_implemented: Literal[False] = False
    success_metric_persistence_implemented: Literal[False] = False
    success_pattern_application_implemented: Literal[False] = False
    learning_feedback_application_implemented: Literal[False] = False
    learning_memory_persistence_implemented: Literal[False] = False
    learning_policy_update_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["success_pattern.v1"] = (
        SUCCESS_PATTERN_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _pattern_matches_contract(self) -> Self:
        if self.pattern_id != f"success_pattern::{self.pattern_kind}":
            raise ValueError("pattern_id must match pattern_kind")
        if self.success_pattern_score != _success_pattern_score(
            workflow_success_score=self.workflow_success_score,
            improvement_score=self.improvement_score,
            learning_priority_score=self.learning_priority_score,
            success_pattern_weight=self.success_pattern_weight,
        ):
            raise ValueError("success_pattern_score must combine source scores")
        if self.priority != _success_priority(self.success_pattern_score, self.status):
            raise ValueError("priority must match score and status")
        if self.status == "guarded" and not self.hitl_required:
            raise ValueError("guarded success patterns require HITL posture")
        return self


class SuccessPatternDiscoveryPlan(BaseModel):
    """Bounded V6.1 advisory success pattern discovery plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["success_pattern_discovery"] = "success_pattern_discovery"
    serialization_version: Literal["success_pattern_discovery_plan.v1"] = (
        SUCCESS_PATTERN_DISCOVERY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=SUCCESS_PATTERN_DISCOVERY_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_workflow_success_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_continuous_improvement_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_adaptive_learning_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    patterns: tuple[SuccessPattern, ...] = Field(min_length=4, max_length=4)
    pattern_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    discovered_pattern_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    review_required_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    guarded_pattern_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    critical_priority_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    guarded_priority_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    hitl_required_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    applied_success_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    pattern_count: int = Field(ge=4, le=4)
    review_required_pattern_count: int = Field(ge=0, le=4)
    guarded_pattern_count: int = Field(ge=0, le=4)
    hitl_required_pattern_count: int = Field(ge=0, le=4)
    highest_success_pattern_score: int = Field(ge=0, le=1_000)
    overall_success_pattern_score: int = Field(ge=0, le=1_000)
    overall_success_pattern_posture: SuccessPatternPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    success_pattern_discovery_implemented: Literal[True] = True
    success_pattern_metadata_implemented: Literal[True] = True
    workflow_success_metadata_used: Literal[True] = True
    continuous_improvement_metadata_used: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    runtime_success_observation_implemented: Literal[False] = False
    live_telemetry_collection_implemented: Literal[False] = False
    success_metric_persistence_implemented: Literal[False] = False
    success_pattern_application_implemented: Literal[False] = False
    learning_feedback_application_implemented: Literal[False] = False
    learning_memory_persistence_implemented: Literal[False] = False
    learning_policy_update_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
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
    def _plan_matches_patterns(self) -> Self:
        derived_pattern_ids = tuple(pattern.pattern_id for pattern in self.patterns)
        if self.pattern_ids != derived_pattern_ids:
            raise ValueError("pattern_ids must match patterns")
        if self.review_required_pattern_ids != _pattern_ids_for_status(
            self.patterns,
            "review_required",
        ):
            raise ValueError("review_required_pattern_ids must match patterns")
        if self.guarded_pattern_ids != _pattern_ids_for_status(
            self.patterns,
            "guarded",
        ):
            raise ValueError("guarded_pattern_ids must match patterns")
        if self.critical_priority_pattern_ids != _pattern_ids_for_priority(
            self.patterns,
            "critical",
        ):
            raise ValueError("critical_priority_pattern_ids must match patterns")
        if self.guarded_priority_pattern_ids != _pattern_ids_for_priority(
            self.patterns,
            "guarded",
        ):
            raise ValueError("guarded_priority_pattern_ids must match patterns")
        if self.hitl_required_pattern_ids != tuple(
            pattern.pattern_id for pattern in self.patterns if pattern.hitl_required
        ):
            raise ValueError("hitl_required_pattern_ids must match patterns")
        if self.applied_success_pattern_ids:
            raise ValueError("applied_success_pattern_ids must remain empty")
        if self.pattern_count != len(self.patterns):
            raise ValueError("pattern_count must match patterns")
        if self.review_required_pattern_count != len(self.review_required_pattern_ids):
            raise ValueError("review_required_pattern_count must match patterns")
        if self.guarded_pattern_count != len(self.guarded_pattern_ids):
            raise ValueError("guarded_pattern_count must match patterns")
        if self.hitl_required_pattern_count != len(self.hitl_required_pattern_ids):
            raise ValueError("hitl_required_pattern_count must match patterns")
        if self.highest_success_pattern_score != max(
            pattern.success_pattern_score for pattern in self.patterns
        ):
            raise ValueError("highest_success_pattern_score must match patterns")
        if self.overall_success_pattern_score != _overall_success_pattern_score(
            self.patterns,
        ):
            raise ValueError("overall_success_pattern_score must match patterns")
        if self.overall_success_pattern_posture != _overall_success_posture(
            self.patterns,
        ):
            raise ValueError("overall_success_pattern_posture must match patterns")
        for pattern in self.patterns:
            if pattern.route_name != self.route_name:
                raise ValueError("pattern route_name must match plan")
            if pattern.task_type != self.task_type:
                raise ValueError("pattern task_type must match plan")
        return self


def discover_success_patterns(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    adaptive_learning: AdaptiveLearningPlan | None = None,
    workflow_success: WorkflowSuccessTrackingPlan | None = None,
    continuous_improvement: ContinuousImprovementSignalPlan | None = None,
) -> SuccessPatternDiscoveryPlan:
    """Discover success patterns without observing runtime success."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    learning_plan = adaptive_learning or evaluate_adaptive_learning_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    success_plan = workflow_success or track_workflow_success(
        route=route_name,
        task_type=learning_plan.task_type,
        execution_mode_id=execution_mode_id,
        adaptive_learning=learning_plan,
    )
    improvement_plan = continuous_improvement or derive_continuous_improvement_signals(
        route=route_name,
        task_type=learning_plan.task_type,
        execution_mode_id=execution_mode_id,
        adaptive_learning=learning_plan,
        workflow_success=success_plan,
    )
    normalized_mode = str(
        execution_mode_id or learning_plan.signals[0].execution_mode_id
    )
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")
    patterns = _patterns(
        route_name=route_name,
        task_type=learning_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        adaptive_learning=learning_plan,
        workflow_success=success_plan,
        continuous_improvement=improvement_plan,
    )
    return SuccessPatternDiscoveryPlan(
        route_name=route_name,
        task_type=learning_plan.task_type,
        source_workflow_success_serialization_version=(
            success_plan.serialization_version
        ),
        source_continuous_improvement_serialization_version=(
            improvement_plan.serialization_version
        ),
        source_adaptive_learning_serialization_version=(
            learning_plan.serialization_version
        ),
        execution_mode_ids=execution_modes.execution_mode_ids,
        patterns=patterns,
        pattern_ids=tuple(pattern.pattern_id for pattern in patterns),
        discovered_pattern_ids=_pattern_ids_for_status(patterns, "discovered"),
        review_required_pattern_ids=_pattern_ids_for_status(
            patterns,
            "review_required",
        ),
        guarded_pattern_ids=_pattern_ids_for_status(patterns, "guarded"),
        critical_priority_pattern_ids=_pattern_ids_for_priority(patterns, "critical"),
        guarded_priority_pattern_ids=_pattern_ids_for_priority(patterns, "guarded"),
        hitl_required_pattern_ids=tuple(
            pattern.pattern_id for pattern in patterns if pattern.hitl_required
        ),
        applied_success_pattern_ids=(),
        pattern_count=len(patterns),
        review_required_pattern_count=len(
            _pattern_ids_for_status(patterns, "review_required")
        ),
        guarded_pattern_count=len(_pattern_ids_for_status(patterns, "guarded")),
        hitl_required_pattern_count=sum(
            1 for pattern in patterns if pattern.hitl_required
        ),
        highest_success_pattern_score=max(
            pattern.success_pattern_score for pattern in patterns
        ),
        overall_success_pattern_score=_overall_success_pattern_score(patterns),
        overall_success_pattern_posture=_overall_success_posture(patterns),
        advisory_actions=_plan_actions(patterns),
    )


def success_pattern_by_id(
    pattern_id: str,
    plan: SuccessPatternDiscoveryPlan | None = None,
) -> SuccessPattern | None:
    source_plan = plan or discover_success_patterns()
    for pattern in source_plan.patterns:
        if pattern.pattern_id == pattern_id:
            return pattern
    return None


def success_patterns_for_status(
    status: SuccessPatternStatus,
    plan: SuccessPatternDiscoveryPlan | None = None,
) -> tuple[SuccessPattern, ...]:
    source_plan = plan or discover_success_patterns()
    return tuple(
        pattern for pattern in source_plan.patterns if pattern.status == status
    )


def success_patterns_for_priority(
    priority: SuccessPatternPriority,
    plan: SuccessPatternDiscoveryPlan | None = None,
) -> tuple[SuccessPattern, ...]:
    source_plan = plan or discover_success_patterns()
    return tuple(
        pattern for pattern in source_plan.patterns if pattern.priority == priority
    )


def _patterns(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    workflow_success: WorkflowSuccessTrackingPlan,
    continuous_improvement: ContinuousImprovementSignalPlan,
) -> tuple[SuccessPattern, ...]:
    return (
        _pattern(
            kind="execution_success_pattern",
            indicator_id="workflow_success::execution_readiness_success",
            improvement_signal_id=(
                "continuous_improvement::success_improvement_signal"
            ),
            learning_signal_id="adaptive_learning::workflow_pattern_learning",
            status="review_required",
            weight=190,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            workflow_success=workflow_success,
            continuous_improvement=continuous_improvement,
        ),
        _pattern(
            kind="routing_success_pattern",
            indicator_id="workflow_success::routing_safety_success",
            improvement_signal_id=(
                "continuous_improvement::failure_prevention_signal"
            ),
            learning_signal_id="adaptive_learning::routing_boundary_learning",
            status="guarded",
            weight=220,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            workflow_success=workflow_success,
            continuous_improvement=continuous_improvement,
        ),
        _pattern(
            kind="artifact_success_pattern",
            indicator_id="workflow_success::governance_review_success",
            improvement_signal_id=(
                "continuous_improvement::artifact_improvement_signal"
            ),
            learning_signal_id="adaptive_learning::governance_feedback_learning",
            status="review_required",
            weight=180,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            workflow_success=workflow_success,
            continuous_improvement=continuous_improvement,
        ),
        _pattern(
            kind="evaluation_success_pattern",
            indicator_id="workflow_success::runtime_guardrail_success",
            improvement_signal_id=(
                "continuous_improvement::evaluation_improvement_signal"
            ),
            learning_signal_id="adaptive_learning::runtime_guardrail_learning",
            status="guarded",
            weight=210,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            workflow_success=workflow_success,
            continuous_improvement=continuous_improvement,
        ),
    )


def _pattern(
    *,
    kind: SuccessPatternKind,
    indicator_id: str,
    improvement_signal_id: str,
    learning_signal_id: str,
    status: SuccessPatternStatus,
    weight: int,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    workflow_success: WorkflowSuccessTrackingPlan,
    continuous_improvement: ContinuousImprovementSignalPlan,
) -> SuccessPattern:
    indicator = _required_success_indicator(indicator_id, workflow_success)
    improvement = _required_improvement_signal(
        improvement_signal_id,
        continuous_improvement,
    )
    learning_signal = _required_learning_signal(learning_signal_id, adaptive_learning)
    score = _success_pattern_score(
        workflow_success_score=indicator.workflow_success_score,
        improvement_score=improvement.improvement_score,
        learning_priority_score=learning_signal.learning_priority_score,
        success_pattern_weight=weight,
    )
    return SuccessPattern(
        pattern_id=f"success_pattern::{kind}",
        pattern_kind=kind,
        status=status,
        priority=_success_priority(score, status),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_success_indicator_id=indicator.indicator_id,
        source_improvement_signal_id=improvement.signal_id,
        source_learning_signal_id=learning_signal.signal_id,
        source_workflow_risk_factor_id=learning_signal.source_workflow_risk_factor_id,
        source_success_status=indicator.status,
        source_improvement_status=improvement.status,
        success_confidence_band=indicator.confidence_band,
        workflow_success_score=indicator.workflow_success_score,
        improvement_score=improvement.improvement_score,
        learning_priority_score=learning_signal.learning_priority_score,
        success_pattern_weight=weight,
        success_pattern_score=score,
        hitl_required=(
            indicator.hitl_required
            or improvement.hitl_required
            or learning_signal.hitl_required
            or status == "guarded"
        ),
        success_pattern_tags=(indicator.indicator_kind, kind.removesuffix("_pattern")),
        success_summary=_success_summary(kind, status),
        advisory_actions=_pattern_actions(kind),
        evidence=(
            f"success_indicator:{indicator.indicator_id}",
            f"workflow_success_score:{indicator.workflow_success_score}",
            f"improvement_signal:{improvement.signal_id}",
            f"improvement_score:{improvement.improvement_score}",
            f"learning_signal:{learning_signal.signal_id}",
            f"learning_priority_score:{learning_signal.learning_priority_score}",
        ),
    )


def _success_pattern_score(
    *,
    workflow_success_score: int,
    improvement_score: int,
    learning_priority_score: int,
    success_pattern_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            workflow_success_score
            + improvement_score // 2
            + learning_priority_score // 4
            + success_pattern_weight,
        ),
    )


def _success_priority(
    score: int,
    status: SuccessPatternStatus,
) -> SuccessPatternPriority:
    if status == "guarded":
        return "guarded"
    if score >= 840:
        return "critical"
    if score >= 620:
        return "elevated"
    return "standard"


def _pattern_ids_for_status(
    patterns: tuple[SuccessPattern, ...],
    status: SuccessPatternStatus,
) -> tuple[str, ...]:
    return tuple(pattern.pattern_id for pattern in patterns if pattern.status == status)


def _pattern_ids_for_priority(
    patterns: tuple[SuccessPattern, ...],
    priority: SuccessPatternPriority,
) -> tuple[str, ...]:
    return tuple(
        pattern.pattern_id for pattern in patterns if pattern.priority == priority
    )


def _overall_success_pattern_score(
    patterns: tuple[SuccessPattern, ...],
) -> int:
    return sum(pattern.success_pattern_score for pattern in patterns) // len(patterns)


def _overall_success_posture(
    patterns: tuple[SuccessPattern, ...],
) -> SuccessPatternPosture:
    if any(pattern.status == "guarded" for pattern in patterns):
        return "guarded"
    if any(pattern.hitl_required for pattern in patterns):
        return "review_required"
    return "discovered"


def _required_learning_signal(
    signal_id: str,
    plan: AdaptiveLearningPlan,
) -> AdaptiveLearningSignal:
    for signal in plan.signals:
        if signal.signal_id == signal_id:
            return signal
    raise ValueError("required success pattern learning metadata is missing")


def _required_success_indicator(
    indicator_id: str,
    plan: WorkflowSuccessTrackingPlan,
) -> WorkflowSuccessIndicator:
    for indicator in plan.indicators:
        if indicator.indicator_id == indicator_id:
            return indicator
    raise ValueError("required workflow success metadata is missing")


def _required_improvement_signal(
    signal_id: str,
    plan: ContinuousImprovementSignalPlan,
) -> ContinuousImprovementSignal:
    for signal in plan.signals:
        if signal.signal_id == signal_id:
            return signal
    raise ValueError("required continuous improvement metadata is missing")


def _success_summary(
    kind: SuccessPatternKind,
    status: SuccessPatternStatus,
) -> str:
    if status == "guarded":
        return f"Surface {kind} as guarded success metadata without application."
    if status == "review_required":
        return f"Surface {kind} for review before future success learning behavior."
    return f"Surface {kind} as discovered success metadata only."


def _pattern_actions(kind: SuccessPatternKind) -> tuple[str, ...]:
    return (
        f"Expose {kind} as derived success pattern metadata.",
        "Keep success observation, telemetry, metric persistence, feedback, "
        "memory, policy updates, routing, workflow, storage, Runtime "
        "Evolution, and output mutation disabled.",
    )


def _plan_actions(patterns: tuple[SuccessPattern, ...]) -> tuple[str, ...]:
    actions = [
        "Expose success patterns as advisory metadata only.",
        "Keep applied success pattern ids empty.",
        "Preserve success observation, telemetry, feedback, memory, policy, "
        "routing, workflow, storage, output, and Runtime Evolution boundaries.",
    ]
    if any(pattern.hitl_required for pattern in patterns):
        actions.append("Require review before any future success pattern behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
