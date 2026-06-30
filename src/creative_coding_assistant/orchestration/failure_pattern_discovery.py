"""V6.1 advisory failure pattern discovery."""

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
from creative_coding_assistant.orchestration.failure_tracking import (
    FailureTrackingIndicator,
    FailureTrackingPlan,
    track_failures,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

FailurePatternKind = Literal[
    "langgraph_failure_pattern",
    "routing_failure_pattern",
    "performance_failure_pattern",
    "retry_failure_pattern",
]
FailurePatternStatus = Literal["discovered", "review_required", "guarded"]
FailurePatternPriority = Literal["standard", "elevated", "critical", "guarded"]
FailurePatternPosture = Literal["discovered", "review_required", "guarded"]

FAILURE_PATTERN_SERIALIZATION_VERSION = "failure_pattern.v1"
FAILURE_PATTERN_DISCOVERY_PLAN_SERIALIZATION_VERSION = (
    "failure_pattern_discovery_plan.v1"
)
FAILURE_PATTERN_DISCOVERY_AUTHORITY_BOUNDARY = (
    "V6.1 failure pattern discovery derives guarded failure patterns from "
    "read-only failure tracking, continuous improvement, and adaptive learning "
    "metadata only; it does not observe runtime failures, classify live errors, "
    "route terminal failures, handle or repair failures, mutate terminal "
    "routing, apply feedback, persist learning memory, update policies, change "
    "provider or model routing, execute providers, invoke agents, emit HITL "
    "requests, execute or control workflows, mutate workflow graphs, trigger "
    "retries or refinements, mutate prompts, write storage, modify generated "
    "output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_failure_observation",
    "live_error_classification",
    "terminal_failure_routing",
    "failure_handling",
    "failure_repair",
    "terminal_routing_mutation",
    "learning_feedback_application",
    "learning_memory_persistence",
    "learning_policy_update",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
    "provider_execution",
    "agent_invocation",
    "hitl_request_emission",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class FailurePattern(BaseModel):
    """One advisory discovered failure pattern."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    pattern_id: str = Field(min_length=1, max_length=180)
    pattern_kind: FailurePatternKind
    status: FailurePatternStatus
    priority: FailurePatternPriority
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_failure_indicator_id: str = Field(min_length=1, max_length=180)
    source_failure_panel_id: str = Field(min_length=1, max_length=180)
    source_improvement_signal_id: str = Field(min_length=1, max_length=180)
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    source_workflow_risk_factor_id: str = Field(min_length=1, max_length=180)
    failure_signal_count: int = Field(ge=0, le=300)
    guardrail_signal_count: int = Field(ge=0, le=300)
    failure_tracking_score: int = Field(ge=0, le=1_000)
    improvement_score: int = Field(ge=0, le=1_000)
    learning_priority_score: int = Field(ge=0, le=1_000)
    failure_pattern_weight: int = Field(ge=0, le=240)
    failure_pattern_score: int = Field(ge=0, le=1_000)
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
    failure_pattern_discovery_implemented: Literal[True] = True
    failure_pattern_metadata_implemented: Literal[True] = True
    failure_tracking_metadata_used: Literal[True] = True
    continuous_improvement_metadata_used: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    runtime_failure_observation_implemented: Literal[False] = False
    live_error_classification_implemented: Literal[False] = False
    terminal_failure_routing_implemented: Literal[False] = False
    failure_handling_implemented: Literal[False] = False
    failure_repair_implemented: Literal[False] = False
    terminal_routing_mutation_implemented: Literal[False] = False
    learning_feedback_application_implemented: Literal[False] = False
    learning_memory_persistence_implemented: Literal[False] = False
    learning_policy_update_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["failure_pattern.v1"] = (
        FAILURE_PATTERN_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _pattern_matches_contract(self) -> Self:
        if self.pattern_id != f"failure_pattern::{self.pattern_kind}":
            raise ValueError("pattern_id must match pattern_kind")
        if self.failure_pattern_score != _failure_pattern_score(
            failure_tracking_score=self.failure_tracking_score,
            improvement_score=self.improvement_score,
            learning_priority_score=self.learning_priority_score,
            failure_signal_count=self.failure_signal_count,
            guardrail_signal_count=self.guardrail_signal_count,
            failure_pattern_weight=self.failure_pattern_weight,
        ):
            raise ValueError("failure_pattern_score must combine source scores")
        if self.priority != _failure_priority(self.failure_pattern_score, self.status):
            raise ValueError("priority must match score and status")
        if self.status == "guarded" and not self.hitl_required:
            raise ValueError("guarded failure patterns require HITL posture")
        return self


class FailurePatternDiscoveryPlan(BaseModel):
    """Bounded V6.1 advisory failure pattern discovery plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["failure_pattern_discovery"] = "failure_pattern_discovery"
    serialization_version: Literal["failure_pattern_discovery_plan.v1"] = (
        FAILURE_PATTERN_DISCOVERY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=FAILURE_PATTERN_DISCOVERY_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_failure_tracking_serialization_version: str = Field(
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
    patterns: tuple[FailurePattern, ...] = Field(min_length=4, max_length=4)
    pattern_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    guarded_pattern_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    guarded_priority_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    hitl_required_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    applied_failure_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    pattern_count: int = Field(ge=4, le=4)
    guarded_pattern_count: int = Field(ge=0, le=4)
    hitl_required_pattern_count: int = Field(ge=0, le=4)
    highest_failure_pattern_score: int = Field(ge=0, le=1_000)
    overall_failure_pattern_score: int = Field(ge=0, le=1_000)
    overall_failure_pattern_posture: FailurePatternPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    failure_pattern_discovery_implemented: Literal[True] = True
    failure_pattern_metadata_implemented: Literal[True] = True
    failure_tracking_metadata_used: Literal[True] = True
    continuous_improvement_metadata_used: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    runtime_failure_observation_implemented: Literal[False] = False
    live_error_classification_implemented: Literal[False] = False
    terminal_failure_routing_implemented: Literal[False] = False
    failure_handling_implemented: Literal[False] = False
    failure_repair_implemented: Literal[False] = False
    terminal_routing_mutation_implemented: Literal[False] = False
    learning_feedback_application_implemented: Literal[False] = False
    learning_memory_persistence_implemented: Literal[False] = False
    learning_policy_update_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
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
        if self.guarded_pattern_ids != _pattern_ids_for_status(
            self.patterns,
            "guarded",
        ):
            raise ValueError("guarded_pattern_ids must match patterns")
        if self.guarded_priority_pattern_ids != _pattern_ids_for_priority(
            self.patterns,
            "guarded",
        ):
            raise ValueError("guarded_priority_pattern_ids must match patterns")
        if self.hitl_required_pattern_ids != tuple(
            pattern.pattern_id for pattern in self.patterns if pattern.hitl_required
        ):
            raise ValueError("hitl_required_pattern_ids must match patterns")
        if self.applied_failure_pattern_ids:
            raise ValueError("applied_failure_pattern_ids must remain empty")
        if self.pattern_count != len(self.patterns):
            raise ValueError("pattern_count must match patterns")
        if self.guarded_pattern_count != len(self.guarded_pattern_ids):
            raise ValueError("guarded_pattern_count must match patterns")
        if self.hitl_required_pattern_count != len(self.hitl_required_pattern_ids):
            raise ValueError("hitl_required_pattern_count must match patterns")
        if self.highest_failure_pattern_score != max(
            pattern.failure_pattern_score for pattern in self.patterns
        ):
            raise ValueError("highest_failure_pattern_score must match patterns")
        if self.overall_failure_pattern_score != _overall_failure_pattern_score(
            self.patterns,
        ):
            raise ValueError("overall_failure_pattern_score must match patterns")
        if self.overall_failure_pattern_posture != _overall_failure_posture(
            self.patterns,
        ):
            raise ValueError("overall_failure_pattern_posture must match patterns")
        for pattern in self.patterns:
            if pattern.route_name != self.route_name:
                raise ValueError("pattern route_name must match plan")
            if pattern.task_type != self.task_type:
                raise ValueError("pattern task_type must match plan")
        return self


def discover_failure_patterns(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    adaptive_learning: AdaptiveLearningPlan | None = None,
    failure_tracking: FailureTrackingPlan | None = None,
    continuous_improvement: ContinuousImprovementSignalPlan | None = None,
) -> FailurePatternDiscoveryPlan:
    """Discover failure patterns without observing or handling failures."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    learning_plan = adaptive_learning or evaluate_adaptive_learning_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    failure_plan = failure_tracking or track_failures(
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
        failure_tracking=failure_plan,
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
        failure_tracking=failure_plan,
        continuous_improvement=improvement_plan,
    )
    return FailurePatternDiscoveryPlan(
        route_name=route_name,
        task_type=learning_plan.task_type,
        source_failure_tracking_serialization_version=(
            failure_plan.serialization_version
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
        guarded_pattern_ids=_pattern_ids_for_status(patterns, "guarded"),
        guarded_priority_pattern_ids=_pattern_ids_for_priority(patterns, "guarded"),
        hitl_required_pattern_ids=tuple(
            pattern.pattern_id for pattern in patterns if pattern.hitl_required
        ),
        applied_failure_pattern_ids=(),
        pattern_count=len(patterns),
        guarded_pattern_count=len(_pattern_ids_for_status(patterns, "guarded")),
        hitl_required_pattern_count=sum(
            1 for pattern in patterns if pattern.hitl_required
        ),
        highest_failure_pattern_score=max(
            pattern.failure_pattern_score for pattern in patterns
        ),
        overall_failure_pattern_score=_overall_failure_pattern_score(patterns),
        overall_failure_pattern_posture=_overall_failure_posture(patterns),
        advisory_actions=_plan_actions(patterns),
    )


def failure_pattern_by_id(
    pattern_id: str,
    plan: FailurePatternDiscoveryPlan | None = None,
) -> FailurePattern | None:
    source_plan = plan or discover_failure_patterns()
    for pattern in source_plan.patterns:
        if pattern.pattern_id == pattern_id:
            return pattern
    return None


def failure_patterns_for_status(
    status: FailurePatternStatus,
    plan: FailurePatternDiscoveryPlan | None = None,
) -> tuple[FailurePattern, ...]:
    source_plan = plan or discover_failure_patterns()
    return tuple(
        pattern for pattern in source_plan.patterns if pattern.status == status
    )


def failure_patterns_for_priority(
    priority: FailurePatternPriority,
    plan: FailurePatternDiscoveryPlan | None = None,
) -> tuple[FailurePattern, ...]:
    source_plan = plan or discover_failure_patterns()
    return tuple(
        pattern for pattern in source_plan.patterns if pattern.priority == priority
    )


def _patterns(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    failure_tracking: FailureTrackingPlan,
    continuous_improvement: ContinuousImprovementSignalPlan,
) -> tuple[FailurePattern, ...]:
    return (
        _pattern(
            kind="langgraph_failure_pattern",
            indicator_id="failure_tracking::langgraph_failure_tracking",
            improvement_signal_id=(
                "continuous_improvement::failure_prevention_signal"
            ),
            learning_signal_id="adaptive_learning::workflow_pattern_learning",
            weight=200,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            failure_tracking=failure_tracking,
            continuous_improvement=continuous_improvement,
        ),
        _pattern(
            kind="routing_failure_pattern",
            indicator_id="failure_tracking::routing_failure_tracking",
            improvement_signal_id=(
                "continuous_improvement::failure_prevention_signal"
            ),
            learning_signal_id="adaptive_learning::routing_boundary_learning",
            weight=230,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            failure_tracking=failure_tracking,
            continuous_improvement=continuous_improvement,
        ),
        _pattern(
            kind="performance_failure_pattern",
            indicator_id="failure_tracking::performance_failure_tracking",
            improvement_signal_id=(
                "continuous_improvement::evaluation_improvement_signal"
            ),
            learning_signal_id="adaptive_learning::runtime_guardrail_learning",
            weight=220,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            failure_tracking=failure_tracking,
            continuous_improvement=continuous_improvement,
        ),
        _pattern(
            kind="retry_failure_pattern",
            indicator_id="failure_tracking::retry_failure_tracking",
            improvement_signal_id=(
                "continuous_improvement::artifact_improvement_signal"
            ),
            learning_signal_id="adaptive_learning::governance_feedback_learning",
            weight=190,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            failure_tracking=failure_tracking,
            continuous_improvement=continuous_improvement,
        ),
    )


def _pattern(
    *,
    kind: FailurePatternKind,
    indicator_id: str,
    improvement_signal_id: str,
    learning_signal_id: str,
    weight: int,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    failure_tracking: FailureTrackingPlan,
    continuous_improvement: ContinuousImprovementSignalPlan,
) -> FailurePattern:
    indicator = _required_failure_indicator(indicator_id, failure_tracking)
    improvement = _required_improvement_signal(
        improvement_signal_id,
        continuous_improvement,
    )
    learning_signal = _required_learning_signal(learning_signal_id, adaptive_learning)
    score = _failure_pattern_score(
        failure_tracking_score=indicator.failure_tracking_score,
        improvement_score=improvement.improvement_score,
        learning_priority_score=learning_signal.learning_priority_score,
        failure_signal_count=indicator.failure_signal_count,
        guardrail_signal_count=indicator.guardrail_signal_count,
        failure_pattern_weight=weight,
    )
    return FailurePattern(
        pattern_id=f"failure_pattern::{kind}",
        pattern_kind=kind,
        status="guarded",
        priority=_failure_priority(score, "guarded"),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_failure_indicator_id=indicator.indicator_id,
        source_failure_panel_id=indicator.source_failure_panel_id,
        source_improvement_signal_id=improvement.signal_id,
        source_learning_signal_id=learning_signal.signal_id,
        source_workflow_risk_factor_id=learning_signal.source_workflow_risk_factor_id,
        failure_signal_count=indicator.failure_signal_count,
        guardrail_signal_count=indicator.guardrail_signal_count,
        failure_tracking_score=indicator.failure_tracking_score,
        improvement_score=improvement.improvement_score,
        learning_priority_score=learning_signal.learning_priority_score,
        failure_pattern_weight=weight,
        failure_pattern_score=score,
        hitl_required=True,
        failure_pattern_tags=(indicator.source_failure_panel_id, kind),
        failure_summary=_failure_summary(kind),
        advisory_actions=_pattern_actions(kind),
        evidence=(
            f"failure_indicator:{indicator.indicator_id}",
            f"failure_panel:{indicator.source_failure_panel_id}",
            f"failure_signals:{indicator.failure_signal_count}",
            f"guardrails:{indicator.guardrail_signal_count}",
            f"improvement_signal:{improvement.signal_id}",
            f"learning_signal:{learning_signal.signal_id}",
        ),
    )


def _failure_pattern_score(
    *,
    failure_tracking_score: int,
    improvement_score: int,
    learning_priority_score: int,
    failure_signal_count: int,
    guardrail_signal_count: int,
    failure_pattern_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            failure_tracking_score
            + improvement_score // 3
            + learning_priority_score // 4
            + failure_signal_count * 3
            + guardrail_signal_count * 8
            + failure_pattern_weight,
        ),
    )


def _failure_priority(
    score: int,
    status: FailurePatternStatus,
) -> FailurePatternPriority:
    if status == "guarded":
        return "guarded"
    if score >= 840:
        return "critical"
    if score >= 620:
        return "elevated"
    return "standard"


def _pattern_ids_for_status(
    patterns: tuple[FailurePattern, ...],
    status: FailurePatternStatus,
) -> tuple[str, ...]:
    return tuple(pattern.pattern_id for pattern in patterns if pattern.status == status)


def _pattern_ids_for_priority(
    patterns: tuple[FailurePattern, ...],
    priority: FailurePatternPriority,
) -> tuple[str, ...]:
    return tuple(
        pattern.pattern_id for pattern in patterns if pattern.priority == priority
    )


def _overall_failure_pattern_score(
    patterns: tuple[FailurePattern, ...],
) -> int:
    return sum(pattern.failure_pattern_score for pattern in patterns) // len(patterns)


def _overall_failure_posture(
    patterns: tuple[FailurePattern, ...],
) -> FailurePatternPosture:
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
    raise ValueError("required failure pattern learning metadata is missing")


def _required_failure_indicator(
    indicator_id: str,
    plan: FailureTrackingPlan,
) -> FailureTrackingIndicator:
    for indicator in plan.indicators:
        if indicator.indicator_id == indicator_id:
            return indicator
    raise ValueError("required failure tracking metadata is missing")


def _required_improvement_signal(
    signal_id: str,
    plan: ContinuousImprovementSignalPlan,
) -> ContinuousImprovementSignal:
    for signal in plan.signals:
        if signal.signal_id == signal_id:
            return signal
    raise ValueError("required continuous improvement metadata is missing")


def _failure_summary(kind: FailurePatternKind) -> str:
    return f"Surface {kind} as guarded failure metadata without handling."


def _pattern_actions(kind: FailurePatternKind) -> tuple[str, ...]:
    return (
        f"Expose {kind} as derived failure pattern metadata.",
        "Keep failure observation, classification, terminal routing, handling, "
        "repair, feedback, memory, policy updates, workflow, storage, Runtime "
        "Evolution, and output mutation disabled.",
    )


def _plan_actions(patterns: tuple[FailurePattern, ...]) -> tuple[str, ...]:
    actions = [
        "Expose failure patterns as guarded advisory metadata only.",
        "Keep applied failure pattern ids empty.",
        "Preserve failure observation, classification, routing, handling, "
        "repair, feedback, memory, policy, workflow, storage, output, and "
        "Runtime Evolution boundaries.",
    ]
    if any(pattern.hitl_required for pattern in patterns):
        actions.append("Require review before any future failure pattern behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
