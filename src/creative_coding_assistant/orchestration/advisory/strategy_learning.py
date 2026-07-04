"""V6.1 advisory strategy learning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_execution_strategy_selection import (
    AdaptiveExecutionStrategyCandidate,
    AdaptiveExecutionStrategySelectionPlan,
    adaptive_execution_strategy_by_id,
    select_dynamic_execution_strategy,
)
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

StrategyLearningPatternKind = Literal[
    "balanced_strategy_learning",
    "quality_strategy_learning",
    "latency_strategy_learning",
    "human_guarded_strategy_learning",
]
StrategyLearningStatus = Literal["learnable", "review_required", "guarded"]
StrategyLearningPriority = Literal["standard", "elevated", "critical", "guarded"]
StrategyLearningPosture = Literal["learnable", "review_required", "guarded"]

STRATEGY_LEARNING_PATTERN_SERIALIZATION_VERSION = "strategy_learning_pattern.v1"
STRATEGY_LEARNING_PLAN_SERIALIZATION_VERSION = "strategy_learning_plan.v1"
STRATEGY_LEARNING_AUTHORITY_BOUNDARY = (
    "V6.1 strategy learning derives strategy patterns from advisory adaptive "
    "execution strategy selection and adaptive learning metadata only; it does "
    "not apply strategies, mutate strategy selection, change provider or model "
    "routing, execute providers, invoke agents, emit HITL requests, enforce "
    "budgets, control workflows, mutate workflow graphs, compile graphs, "
    "trigger retries or refinements, mutate prompts, write storage, modify "
    "generated output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "strategy_application",
    "strategy_selection_mutation",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
    "automatic_model_download",
    "provider_execution",
    "local_runtime_probe",
    "local_model_inventory_scan",
    "agent_invocation",
    "hitl_request_emission",
    "budget_enforcement",
    "workflow_control",
    "workflow_graph_mutation",
    "graph_compilation",
    "workflow_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class StrategyLearningPattern(BaseModel):
    """One advisory strategy learning pattern."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    pattern_id: str = Field(min_length=1, max_length=180)
    pattern_kind: StrategyLearningPatternKind
    status: StrategyLearningStatus
    priority: StrategyLearningPriority
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_strategy_id: str = Field(min_length=1, max_length=180)
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    source_workflow_risk_factor_id: str = Field(min_length=1, max_length=180)
    source_strategy_status: str = Field(min_length=1, max_length=80)
    source_strategy_kind: str = Field(min_length=1, max_length=120)
    dynamic_strategy_score: int = Field(ge=0, le=400)
    learning_priority_score: int = Field(ge=0, le=1_000)
    unavailable_reason_count: int = Field(ge=0, le=9)
    strategy_learning_weight: int = Field(ge=0, le=240)
    strategy_learning_score: int = Field(ge=0, le=1_000)
    hitl_required: bool
    provider_sequence: tuple[str, ...] = Field(min_length=1, max_length=4)
    model_profile_sequence: tuple[str, ...] = Field(min_length=1, max_length=4)
    strategy_pattern_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    strategy_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    strategy_learning_implemented: Literal[True] = True
    strategy_pattern_metadata_implemented: Literal[True] = True
    adaptive_strategy_metadata_used: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    strategy_application_implemented: Literal[False] = False
    strategy_selection_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    local_model_inventory_scan_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["strategy_learning_pattern.v1"] = (
        STRATEGY_LEARNING_PATTERN_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _pattern_matches_contract(self) -> Self:
        if self.pattern_id != f"strategy_learning::{self.pattern_kind}":
            raise ValueError("pattern_id must match pattern_kind")
        if len(self.model_profile_sequence) != len(self.provider_sequence):
            raise ValueError("model_profile_sequence must match provider_sequence")
        if self.strategy_learning_score != _strategy_learning_score(
            dynamic_strategy_score=self.dynamic_strategy_score,
            learning_priority_score=self.learning_priority_score,
            unavailable_reason_count=self.unavailable_reason_count,
            strategy_learning_weight=self.strategy_learning_weight,
        ):
            raise ValueError("strategy_learning_score must combine source scores")
        if self.priority != _strategy_priority(
            self.strategy_learning_score,
            self.status,
        ):
            raise ValueError("priority must match score and status")
        if self.status == "guarded" and not self.hitl_required:
            raise ValueError("guarded strategy learning requires HITL posture")
        return self


class StrategyLearningPlan(BaseModel):
    """Bounded V6.1 advisory strategy learning plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["strategy_learning"] = "strategy_learning"
    serialization_version: Literal["strategy_learning_plan.v1"] = (
        STRATEGY_LEARNING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=STRATEGY_LEARNING_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_adaptive_strategy_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_adaptive_learning_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    patterns: tuple[StrategyLearningPattern, ...] = Field(min_length=4, max_length=4)
    pattern_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    learnable_pattern_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    review_required_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    guarded_pattern_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    standard_priority_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    elevated_priority_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
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
    applied_strategy_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    pattern_count: int = Field(ge=4, le=4)
    review_required_pattern_count: int = Field(ge=0, le=4)
    guarded_pattern_count: int = Field(ge=0, le=4)
    hitl_required_pattern_count: int = Field(ge=0, le=4)
    highest_strategy_learning_score: int = Field(ge=0, le=1_000)
    overall_strategy_learning_score: int = Field(ge=0, le=1_000)
    overall_strategy_learning_posture: StrategyLearningPosture
    selected_source_strategy_id: str = Field(min_length=1, max_length=180)
    applied_source_strategy_id: None = None
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    strategy_learning_implemented: Literal[True] = True
    strategy_pattern_metadata_implemented: Literal[True] = True
    adaptive_strategy_metadata_used: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    strategy_application_implemented: Literal[False] = False
    strategy_selection_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    local_model_inventory_scan_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
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
        if len(set(derived_pattern_ids)) != len(derived_pattern_ids):
            raise ValueError("pattern_ids must be unique")
        if self.pattern_ids != derived_pattern_ids:
            raise ValueError("pattern_ids must match patterns")
        if self.pattern_count != len(self.patterns):
            raise ValueError("pattern_count must match patterns")
        if self.learnable_pattern_ids != _pattern_ids_for_status(
            self.patterns,
            "learnable",
        ):
            raise ValueError("learnable_pattern_ids must match patterns")
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
        if self.standard_priority_pattern_ids != _pattern_ids_for_priority(
            self.patterns,
            "standard",
        ):
            raise ValueError("standard_priority_pattern_ids must match patterns")
        if self.elevated_priority_pattern_ids != _pattern_ids_for_priority(
            self.patterns,
            "elevated",
        ):
            raise ValueError("elevated_priority_pattern_ids must match patterns")
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
        if self.applied_strategy_pattern_ids:
            raise ValueError("applied_strategy_pattern_ids must remain empty")
        if self.applied_source_strategy_id is not None:
            raise ValueError("applied_source_strategy_id must remain unset")
        if self.review_required_pattern_count != len(self.review_required_pattern_ids):
            raise ValueError("review_required_pattern_count must match patterns")
        if self.guarded_pattern_count != len(self.guarded_pattern_ids):
            raise ValueError("guarded_pattern_count must match patterns")
        if self.hitl_required_pattern_count != len(self.hitl_required_pattern_ids):
            raise ValueError("hitl_required_pattern_count must match patterns")
        if self.highest_strategy_learning_score != max(
            pattern.strategy_learning_score for pattern in self.patterns
        ):
            raise ValueError("highest_strategy_learning_score must match patterns")
        if self.overall_strategy_learning_score != _overall_strategy_learning_score(
            self.patterns,
        ):
            raise ValueError("overall_strategy_learning_score must match patterns")
        if self.overall_strategy_learning_posture != _overall_strategy_posture(
            self.patterns,
        ):
            raise ValueError("overall_strategy_learning_posture must match patterns")
        for pattern in self.patterns:
            if pattern.route_name != self.route_name:
                raise ValueError("pattern route_name must match plan")
        return self


def learn_strategies(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    adaptive_learning: AdaptiveLearningPlan | None = None,
    adaptive_strategy: AdaptiveExecutionStrategySelectionPlan | None = None,
) -> StrategyLearningPlan:
    """Derive strategy learning patterns without applying strategies."""

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
    strategy_plan = adaptive_strategy or select_dynamic_execution_strategy(
        route=route_name,
        task_type=learning_plan.task_type,
        execution_mode_id=normalized_mode,
    )
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")

    patterns = _patterns(
        route_name=route_name,
        task_type=learning_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        adaptive_learning=learning_plan,
        adaptive_strategy=strategy_plan,
    )
    return StrategyLearningPlan(
        route_name=route_name,
        task_type=learning_plan.task_type,
        source_adaptive_strategy_serialization_version=(
            strategy_plan.serialization_version
        ),
        source_adaptive_learning_serialization_version=(
            learning_plan.serialization_version
        ),
        execution_mode_ids=execution_modes.execution_mode_ids,
        patterns=patterns,
        pattern_ids=tuple(pattern.pattern_id for pattern in patterns),
        learnable_pattern_ids=_pattern_ids_for_status(patterns, "learnable"),
        review_required_pattern_ids=_pattern_ids_for_status(
            patterns,
            "review_required",
        ),
        guarded_pattern_ids=_pattern_ids_for_status(patterns, "guarded"),
        standard_priority_pattern_ids=_pattern_ids_for_priority(patterns, "standard"),
        elevated_priority_pattern_ids=_pattern_ids_for_priority(patterns, "elevated"),
        critical_priority_pattern_ids=_pattern_ids_for_priority(patterns, "critical"),
        guarded_priority_pattern_ids=_pattern_ids_for_priority(patterns, "guarded"),
        hitl_required_pattern_ids=tuple(
            pattern.pattern_id for pattern in patterns if pattern.hitl_required
        ),
        applied_strategy_pattern_ids=(),
        pattern_count=len(patterns),
        review_required_pattern_count=len(
            _pattern_ids_for_status(patterns, "review_required")
        ),
        guarded_pattern_count=len(_pattern_ids_for_status(patterns, "guarded")),
        hitl_required_pattern_count=sum(
            1 for pattern in patterns if pattern.hitl_required
        ),
        highest_strategy_learning_score=max(
            pattern.strategy_learning_score for pattern in patterns
        ),
        overall_strategy_learning_score=_overall_strategy_learning_score(patterns),
        overall_strategy_learning_posture=_overall_strategy_posture(patterns),
        selected_source_strategy_id=strategy_plan.selected_strategy_id,
        applied_source_strategy_id=strategy_plan.applied_strategy_id,
        advisory_actions=_plan_actions(patterns),
    )


def strategy_learning_pattern_by_id(
    pattern_id: str,
    plan: StrategyLearningPlan | None = None,
) -> StrategyLearningPattern | None:
    """Return one strategy learning pattern without applying it."""

    source_plan = plan or learn_strategies()
    for pattern in source_plan.patterns:
        if pattern.pattern_id == pattern_id:
            return pattern
    return None


def strategy_learning_patterns_for_status(
    status: StrategyLearningStatus,
    plan: StrategyLearningPlan | None = None,
) -> tuple[StrategyLearningPattern, ...]:
    """Return strategy learning patterns by advisory status."""

    source_plan = plan or learn_strategies()
    return tuple(
        pattern for pattern in source_plan.patterns if pattern.status == status
    )


def strategy_learning_patterns_for_priority(
    priority: StrategyLearningPriority,
    plan: StrategyLearningPlan | None = None,
) -> tuple[StrategyLearningPattern, ...]:
    """Return strategy learning patterns by derived priority."""

    source_plan = plan or learn_strategies()
    return tuple(
        pattern for pattern in source_plan.patterns if pattern.priority == priority
    )


def _patterns(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    adaptive_strategy: AdaptiveExecutionStrategySelectionPlan,
) -> tuple[StrategyLearningPattern, ...]:
    return (
        _pattern(
            kind="balanced_strategy_learning",
            strategy_id="adaptive_execution_strategy::balanced_hybrid_strategy",
            learning_signal_id="adaptive_learning::strategy_pattern_learning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            adaptive_strategy=adaptive_strategy,
            strategy_learning_weight=190,
        ),
        _pattern(
            kind="quality_strategy_learning",
            strategy_id="adaptive_execution_strategy::quality_priority_strategy",
            learning_signal_id="adaptive_learning::workflow_pattern_learning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            adaptive_strategy=adaptive_strategy,
            strategy_learning_weight=170,
        ),
        _pattern(
            kind="latency_strategy_learning",
            strategy_id="adaptive_execution_strategy::latency_priority_strategy",
            learning_signal_id="adaptive_learning::governance_feedback_learning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            adaptive_strategy=adaptive_strategy,
            strategy_learning_weight=150,
        ),
        _pattern(
            kind="human_guarded_strategy_learning",
            strategy_id=(
                "adaptive_execution_strategy::human_guarded_fallback_strategy"
            ),
            learning_signal_id="adaptive_learning::routing_boundary_learning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            adaptive_strategy=adaptive_strategy,
            strategy_learning_weight=220,
        ),
    )


def _pattern(
    *,
    kind: StrategyLearningPatternKind,
    strategy_id: str,
    learning_signal_id: str,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    adaptive_strategy: AdaptiveExecutionStrategySelectionPlan,
    strategy_learning_weight: int,
) -> StrategyLearningPattern:
    strategy = _required_strategy(strategy_id, adaptive_strategy)
    learning_signal = _required_learning_signal(learning_signal_id, adaptive_learning)
    status = _pattern_status(strategy, learning_signal)
    score = _strategy_learning_score(
        dynamic_strategy_score=strategy.dynamic_strategy_score,
        learning_priority_score=learning_signal.learning_priority_score,
        unavailable_reason_count=len(strategy.unavailable_reason_codes),
        strategy_learning_weight=strategy_learning_weight,
    )
    return StrategyLearningPattern(
        pattern_id=f"strategy_learning::{kind}",
        pattern_kind=kind,
        status=status,
        priority=_strategy_priority(score, status),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_strategy_id=strategy.strategy_id,
        source_learning_signal_id=learning_signal.signal_id,
        source_workflow_risk_factor_id=learning_signal.source_workflow_risk_factor_id,
        source_strategy_status=strategy.status,
        source_strategy_kind=strategy.strategy_kind,
        dynamic_strategy_score=strategy.dynamic_strategy_score,
        learning_priority_score=learning_signal.learning_priority_score,
        unavailable_reason_count=len(strategy.unavailable_reason_codes),
        strategy_learning_weight=strategy_learning_weight,
        strategy_learning_score=score,
        hitl_required=strategy.hitl_required or learning_signal.hitl_required,
        provider_sequence=tuple(strategy.provider_sequence),
        model_profile_sequence=tuple(strategy.model_profile_sequence),
        strategy_pattern_tags=_strategy_pattern_tags(kind),
        strategy_summary=_strategy_summary(kind, status),
        advisory_actions=_pattern_actions(kind),
        evidence=(
            f"adaptive_strategy:{strategy.strategy_id}",
            f"learning_signal:{learning_signal.signal_id}",
            f"dynamic_strategy_score:{strategy.dynamic_strategy_score}",
            f"learning_priority_score:{learning_signal.learning_priority_score}",
            f"strategy_status:{strategy.status}",
            f"hitl_required:{strategy.hitl_required or learning_signal.hitl_required}",
        ),
    )


def _strategy_learning_score(
    *,
    dynamic_strategy_score: int,
    learning_priority_score: int,
    unavailable_reason_count: int,
    strategy_learning_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            dynamic_strategy_score
            + learning_priority_score // 2
            + strategy_learning_weight
            - unavailable_reason_count * 35,
        ),
    )


def _pattern_status(
    strategy: AdaptiveExecutionStrategyCandidate,
    learning_signal: AdaptiveLearningSignal,
) -> StrategyLearningStatus:
    if strategy.status == "guardrail" or learning_signal.status == "guardrail":
        return "guarded"
    if strategy.hitl_required or learning_signal.hitl_required:
        return "review_required"
    return "learnable"


def _strategy_priority(
    score: int,
    status: StrategyLearningStatus,
) -> StrategyLearningPriority:
    if status == "guarded":
        return "guarded"
    if score >= 850:
        return "critical"
    if score >= 620:
        return "elevated"
    return "standard"


def _pattern_ids_for_status(
    patterns: tuple[StrategyLearningPattern, ...],
    status: StrategyLearningStatus,
) -> tuple[str, ...]:
    return tuple(pattern.pattern_id for pattern in patterns if pattern.status == status)


def _pattern_ids_for_priority(
    patterns: tuple[StrategyLearningPattern, ...],
    priority: StrategyLearningPriority,
) -> tuple[str, ...]:
    return tuple(
        pattern.pattern_id for pattern in patterns if pattern.priority == priority
    )


def _overall_strategy_learning_score(
    patterns: tuple[StrategyLearningPattern, ...],
) -> int:
    return sum(pattern.strategy_learning_score for pattern in patterns) // len(patterns)


def _overall_strategy_posture(
    patterns: tuple[StrategyLearningPattern, ...],
) -> StrategyLearningPosture:
    if any(pattern.status == "guarded" for pattern in patterns):
        return "guarded"
    if any(pattern.hitl_required for pattern in patterns):
        return "review_required"
    return "learnable"


def _required_strategy(
    strategy_id: str,
    plan: AdaptiveExecutionStrategySelectionPlan,
) -> AdaptiveExecutionStrategyCandidate:
    strategy = adaptive_execution_strategy_by_id(strategy_id, plan)
    if strategy is None:
        raise ValueError("required strategy learning source metadata is missing")
    return strategy


def _required_learning_signal(
    signal_id: str,
    plan: AdaptiveLearningPlan,
) -> AdaptiveLearningSignal:
    for signal in plan.signals:
        if signal.signal_id == signal_id:
            return signal
    raise ValueError("required strategy learning adaptive metadata is missing")


def _strategy_pattern_tags(
    kind: StrategyLearningPatternKind,
) -> tuple[str, ...]:
    return {
        "balanced_strategy_learning": (
            "balanced_hybrid_strategy",
            "selected_strategy",
            "strategy_pattern",
        ),
        "quality_strategy_learning": (
            "quality_priority_strategy",
            "fallback_strategy",
            "quality_pattern",
        ),
        "latency_strategy_learning": (
            "latency_priority_strategy",
            "fallback_strategy",
            "latency_pattern",
        ),
        "human_guarded_strategy_learning": (
            "human_guarded_strategy",
            "guardrail_strategy",
            "routing_safety",
        ),
    }[kind]


def _strategy_summary(
    kind: StrategyLearningPatternKind,
    status: StrategyLearningStatus,
) -> str:
    if status == "guarded":
        return f"Surface {kind} as guarded strategy metadata without application."
    if status == "review_required":
        return f"Surface {kind} for review before future strategy learning behavior."
    return f"Surface {kind} as learnable strategy metadata only."


def _pattern_actions(kind: StrategyLearningPatternKind) -> tuple[str, ...]:
    return (
        f"Expose {kind} as derived strategy learning metadata.",
        "Keep strategy application, strategy mutation, routing, provider "
        "execution, HITL emission, workflow control, storage, Runtime "
        "Evolution, and output mutation disabled.",
    )


def _plan_actions(
    patterns: tuple[StrategyLearningPattern, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose strategy learning patterns as advisory metadata only.",
        "Keep applied strategy pattern ids empty.",
        "Preserve strategy application, strategy mutation, routing, provider, "
        "workflow, storage, output, and Runtime Evolution boundaries.",
    ]
    if any(pattern.hitl_required for pattern in patterns):
        actions.append("Require review before any future strategy learning behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
