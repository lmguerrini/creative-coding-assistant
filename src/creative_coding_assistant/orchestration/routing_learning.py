"""V6.1 advisory routing learning."""

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
    EstimatedCostBand,
    EstimatedLatencyBand,
    EstimatedQualityBand,
    ExecutionModeId,
    RoutingRiskBand,
    TaskAwareRoutingDecision,
    TaskAwareRoutingRegistry,
    TaskRoutingType,
    routing_execution_mode_registry,
    task_aware_routing_registry,
)

RoutingLearningPatternKind = Literal[
    "creative_route_learning",
    "quality_route_learning",
    "fast_route_learning",
    "guarded_route_learning",
]
RoutingLearningStatus = Literal["learnable", "review_required", "guarded"]
RoutingLearningPriority = Literal["standard", "elevated", "critical", "guarded"]
RoutingLearningPosture = Literal["learnable", "review_required", "guarded"]

ROUTING_LEARNING_PATTERN_SERIALIZATION_VERSION = "routing_learning_pattern.v1"
ROUTING_LEARNING_PLAN_SERIALIZATION_VERSION = "routing_learning_plan.v1"
ROUTING_LEARNING_AUTHORITY_BOUNDARY = (
    "V6.1 routing learning derives route patterns from read-only V5.2 "
    "task-aware routing metadata and adaptive learning signals only; it does "
    "not apply routing, switch configured providers or models, execute "
    "providers, download models, assume API keys, probe local runtimes, scan "
    "local model inventory, emit HITL requests, control workflows, mutate "
    "workflow graphs, compile graphs, trigger retries or refinements, mutate "
    "prompts, write storage, modify generated output, or apply Runtime "
    "Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "routing_application",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
    "automatic_model_download",
    "automatic_api_key_assumption",
    "provider_execution",
    "local_runtime_probe",
    "local_model_inventory_scan",
    "hitl_request_emission",
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


class RoutingLearningPattern(BaseModel):
    """One advisory routing learning pattern."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    pattern_id: str = Field(min_length=1, max_length=180)
    pattern_kind: RoutingLearningPatternKind
    status: RoutingLearningStatus
    priority: RoutingLearningPriority
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_routing_decision_id: str = Field(min_length=1, max_length=180)
    source_routing_task_type: TaskRoutingType
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    source_workflow_risk_factor_id: str = Field(min_length=1, max_length=180)
    recommended_model_profile_id: str = Field(min_length=1, max_length=120)
    fallback_model_profile_id: str = Field(min_length=1, max_length=120)
    available_model_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    routing_risk_band: RoutingRiskBand
    estimated_quality: EstimatedQualityBand
    estimated_cost: EstimatedCostBand
    estimated_latency: EstimatedLatencyBand
    confidence_score: float = Field(ge=0, le=1)
    unavailable_reason_count: int = Field(ge=0, le=6)
    learning_priority_score: int = Field(ge=0, le=1_000)
    routing_learning_weight: int = Field(ge=0, le=240)
    routing_learning_score: int = Field(ge=0, le=1_000)
    hitl_required: bool
    routing_pattern_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    routing_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    routing_learning_implemented: Literal[True] = True
    routing_pattern_metadata_implemented: Literal[True] = True
    task_aware_routing_metadata_used: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    routing_application_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    local_model_inventory_scan_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["routing_learning_pattern.v1"] = (
        ROUTING_LEARNING_PATTERN_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _pattern_matches_contract(self) -> Self:
        if self.pattern_id != f"routing_learning::{self.pattern_kind}":
            raise ValueError("pattern_id must match pattern_kind")
        if self.recommended_model_profile_id not in self.available_model_profile_ids:
            raise ValueError("recommended_model_profile_id must be available")
        if self.fallback_model_profile_id not in self.available_model_profile_ids:
            raise ValueError("fallback_model_profile_id must be available")
        if self.routing_learning_score != _routing_learning_score(
            confidence_score=self.confidence_score,
            estimated_quality=self.estimated_quality,
            estimated_cost=self.estimated_cost,
            estimated_latency=self.estimated_latency,
            routing_risk_band=self.routing_risk_band,
            learning_priority_score=self.learning_priority_score,
            unavailable_reason_count=self.unavailable_reason_count,
            routing_learning_weight=self.routing_learning_weight,
        ):
            raise ValueError("routing_learning_score must combine source scores")
        if self.priority != _routing_priority(self.routing_learning_score, self.status):
            raise ValueError("priority must match score and status")
        if self.status == "guarded" and not self.hitl_required:
            raise ValueError("guarded routing learning requires HITL posture")
        return self


class RoutingLearningPlan(BaseModel):
    """Bounded V6.1 advisory routing learning plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["routing_learning"] = "routing_learning"
    serialization_version: Literal["routing_learning_plan.v1"] = (
        ROUTING_LEARNING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ROUTING_LEARNING_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_task_routing_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_adaptive_learning_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    patterns: tuple[RoutingLearningPattern, ...] = Field(min_length=4, max_length=4)
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
    applied_routing_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    pattern_count: int = Field(ge=4, le=4)
    review_required_pattern_count: int = Field(ge=0, le=4)
    guarded_pattern_count: int = Field(ge=0, le=4)
    hitl_required_pattern_count: int = Field(ge=0, le=4)
    highest_routing_learning_score: int = Field(ge=0, le=1_000)
    overall_routing_learning_score: int = Field(ge=0, le=1_000)
    overall_routing_learning_posture: RoutingLearningPosture
    recommended_model_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    fallback_model_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    routing_learning_implemented: Literal[True] = True
    routing_pattern_metadata_implemented: Literal[True] = True
    task_aware_routing_metadata_used: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    routing_application_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    local_model_inventory_scan_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
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
        if self.pattern_ids != derived_pattern_ids:
            raise ValueError("pattern_ids must match patterns")
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
        if self.applied_routing_pattern_ids:
            raise ValueError("applied_routing_pattern_ids must remain empty")
        if self.pattern_count != len(self.patterns):
            raise ValueError("pattern_count must match patterns")
        if self.review_required_pattern_count != len(self.review_required_pattern_ids):
            raise ValueError("review_required_pattern_count must match patterns")
        if self.guarded_pattern_count != len(self.guarded_pattern_ids):
            raise ValueError("guarded_pattern_count must match patterns")
        if self.hitl_required_pattern_count != len(self.hitl_required_pattern_ids):
            raise ValueError("hitl_required_pattern_count must match patterns")
        if self.highest_routing_learning_score != max(
            pattern.routing_learning_score for pattern in self.patterns
        ):
            raise ValueError("highest_routing_learning_score must match patterns")
        if self.overall_routing_learning_score != _overall_routing_learning_score(
            self.patterns,
        ):
            raise ValueError("overall_routing_learning_score must match patterns")
        if self.overall_routing_learning_posture != _overall_routing_posture(
            self.patterns,
        ):
            raise ValueError("overall_routing_learning_posture must match patterns")
        if self.recommended_model_profile_ids != _unique_model_profile_ids(
            pattern.recommended_model_profile_id for pattern in self.patterns
        ):
            raise ValueError("recommended_model_profile_ids must match patterns")
        if self.fallback_model_profile_ids != _unique_model_profile_ids(
            pattern.fallback_model_profile_id for pattern in self.patterns
        ):
            raise ValueError("fallback_model_profile_ids must match patterns")
        for pattern in self.patterns:
            if pattern.route_name != self.route_name:
                raise ValueError("pattern route_name must match plan")
            if pattern.task_type != self.task_type:
                raise ValueError("pattern task_type must match plan")
        return self


def learn_routing(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    adaptive_learning: AdaptiveLearningPlan | None = None,
    task_routing: TaskAwareRoutingRegistry | None = None,
) -> RoutingLearningPlan:
    """Derive routing learning patterns without applying provider/model routing."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    learning_plan = adaptive_learning or evaluate_adaptive_learning_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    routing_registry = task_routing or task_aware_routing_registry()
    normalized_mode = str(
        execution_mode_id or learning_plan.signals[0].execution_mode_id
    )
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")
    patterns = _patterns(
        route_name=route_name,
        task_type=learning_plan.task_type,
        adaptive_learning=learning_plan,
        task_routing=routing_registry,
    )
    return RoutingLearningPlan(
        route_name=route_name,
        task_type=learning_plan.task_type,
        source_task_routing_serialization_version=(
            routing_registry.serialization_version
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
        applied_routing_pattern_ids=(),
        pattern_count=len(patterns),
        review_required_pattern_count=len(
            _pattern_ids_for_status(patterns, "review_required")
        ),
        guarded_pattern_count=len(_pattern_ids_for_status(patterns, "guarded")),
        hitl_required_pattern_count=sum(
            1 for pattern in patterns if pattern.hitl_required
        ),
        highest_routing_learning_score=max(
            pattern.routing_learning_score for pattern in patterns
        ),
        overall_routing_learning_score=_overall_routing_learning_score(patterns),
        overall_routing_learning_posture=_overall_routing_posture(patterns),
        recommended_model_profile_ids=_unique_model_profile_ids(
            pattern.recommended_model_profile_id for pattern in patterns
        ),
        fallback_model_profile_ids=_unique_model_profile_ids(
            pattern.fallback_model_profile_id for pattern in patterns
        ),
        advisory_actions=_plan_actions(patterns),
    )


def routing_learning_pattern_by_id(
    pattern_id: str,
    plan: RoutingLearningPlan | None = None,
) -> RoutingLearningPattern | None:
    source_plan = plan or learn_routing()
    for pattern in source_plan.patterns:
        if pattern.pattern_id == pattern_id:
            return pattern
    return None


def routing_learning_patterns_for_status(
    status: RoutingLearningStatus,
    plan: RoutingLearningPlan | None = None,
) -> tuple[RoutingLearningPattern, ...]:
    source_plan = plan or learn_routing()
    return tuple(
        pattern for pattern in source_plan.patterns if pattern.status == status
    )


def routing_learning_patterns_for_priority(
    priority: RoutingLearningPriority,
    plan: RoutingLearningPlan | None = None,
) -> tuple[RoutingLearningPattern, ...]:
    source_plan = plan or learn_routing()
    return tuple(
        pattern for pattern in source_plan.patterns if pattern.priority == priority
    )


def _patterns(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    adaptive_learning: AdaptiveLearningPlan,
    task_routing: TaskAwareRoutingRegistry,
) -> tuple[RoutingLearningPattern, ...]:
    return (
        _pattern(
            kind="creative_route_learning",
            decision=_required_routing_decision("creative_coding", task_routing),
            learning_signal_id="adaptive_learning::workflow_pattern_learning",
            weight=180,
            route_name=route_name,
            task_type=task_type,
            adaptive_learning=adaptive_learning,
        ),
        _pattern(
            kind="quality_route_learning",
            decision=_required_routing_decision(
                "maximum_quality_execution",
                task_routing,
            ),
            learning_signal_id="adaptive_learning::strategy_pattern_learning",
            weight=210,
            route_name=route_name,
            task_type=task_type,
            adaptive_learning=adaptive_learning,
        ),
        _pattern(
            kind="fast_route_learning",
            decision=_required_routing_decision("fast_draft", task_routing),
            learning_signal_id="adaptive_learning::governance_feedback_learning",
            weight=170,
            route_name=route_name,
            task_type=task_type,
            adaptive_learning=adaptive_learning,
        ),
        _pattern(
            kind="guarded_route_learning",
            decision=_required_routing_decision("long_context_reasoning", task_routing),
            learning_signal_id="adaptive_learning::routing_boundary_learning",
            weight=230,
            route_name=route_name,
            task_type=task_type,
            adaptive_learning=adaptive_learning,
        ),
    )


def _pattern(
    *,
    kind: RoutingLearningPatternKind,
    decision: TaskAwareRoutingDecision,
    learning_signal_id: str,
    weight: int,
    route_name: RouteName,
    task_type: TaskRoutingType,
    adaptive_learning: AdaptiveLearningPlan,
) -> RoutingLearningPattern:
    learning_signal = _required_learning_signal(learning_signal_id, adaptive_learning)
    status = _pattern_status(decision, learning_signal)
    score = _routing_learning_score(
        confidence_score=decision.confidence_score,
        estimated_quality=decision.estimated_quality,
        estimated_cost=decision.estimated_cost,
        estimated_latency=decision.estimated_latency,
        routing_risk_band=decision.risk_band,
        learning_priority_score=learning_signal.learning_priority_score,
        unavailable_reason_count=len(decision.unavailable_reason_codes),
        routing_learning_weight=weight,
    )
    hitl_required = (
        learning_signal.hitl_required or decision.hitl_required or status == "guarded"
    )
    return RoutingLearningPattern(
        pattern_id=f"routing_learning::{kind}",
        pattern_kind=kind,
        status=status,
        priority=_routing_priority(score, status),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=decision.execution_mode_id,
        source_routing_decision_id=decision.decision_id,
        source_routing_task_type=decision.task_type,
        source_learning_signal_id=learning_signal.signal_id,
        source_workflow_risk_factor_id=learning_signal.source_workflow_risk_factor_id,
        recommended_model_profile_id=decision.recommended_model_profile_id,
        fallback_model_profile_id=decision.fallback_model_profile_id,
        available_model_profile_ids=decision.available_model_profile_ids,
        routing_risk_band=decision.risk_band,
        estimated_quality=decision.estimated_quality,
        estimated_cost=decision.estimated_cost,
        estimated_latency=decision.estimated_latency,
        confidence_score=decision.confidence_score,
        unavailable_reason_count=len(decision.unavailable_reason_codes),
        learning_priority_score=learning_signal.learning_priority_score,
        routing_learning_weight=weight,
        routing_learning_score=score,
        hitl_required=hitl_required,
        routing_pattern_tags=(decision.task_type, decision.execution_mode_id, kind),
        routing_summary=_routing_summary(kind, status),
        advisory_actions=_pattern_actions(kind),
        evidence=(
            f"routing_decision:{decision.decision_id}",
            f"recommended_model:{decision.recommended_model_profile_id}",
            f"fallback_model:{decision.fallback_model_profile_id}",
            f"risk_band:{decision.risk_band}",
            f"confidence:{decision.confidence_score:.2f}",
            f"learning_signal:{learning_signal.signal_id}",
            f"learning_priority_score:{learning_signal.learning_priority_score}",
        ),
    )


def _routing_learning_score(
    *,
    confidence_score: float,
    estimated_quality: EstimatedQualityBand,
    estimated_cost: EstimatedCostBand,
    estimated_latency: EstimatedLatencyBand,
    routing_risk_band: RoutingRiskBand,
    learning_priority_score: int,
    unavailable_reason_count: int,
    routing_learning_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            int(confidence_score * 500)
            + _quality_bonus(estimated_quality)
            + _cost_bonus(estimated_cost)
            + _latency_bonus(estimated_latency)
            + learning_priority_score // 3
            + routing_learning_weight
            - _risk_penalty(routing_risk_band)
            - unavailable_reason_count * 60,
        ),
    )


def _quality_bonus(value: EstimatedQualityBand) -> int:
    return {"low": 20, "medium": 100, "high": 170, "maximum": 240}[value]


def _cost_bonus(value: EstimatedCostBand) -> int:
    return {"low": 120, "medium": 60, "high": 0}[value]


def _latency_bonus(value: EstimatedLatencyBand) -> int:
    return {"fast": 120, "moderate": 60, "slow": 0}[value]


def _risk_penalty(value: RoutingRiskBand) -> int:
    return {"low": 0, "medium": 80, "high": 180}[value]


def _pattern_status(
    decision: TaskAwareRoutingDecision,
    learning_signal: AdaptiveLearningSignal,
) -> RoutingLearningStatus:
    if learning_signal.status == "guardrail" or decision.risk_band == "high":
        return "guarded"
    if learning_signal.hitl_required or decision.hitl_required:
        return "review_required"
    return "learnable"


def _routing_priority(
    score: int,
    status: RoutingLearningStatus,
) -> RoutingLearningPriority:
    if status == "guarded":
        return "guarded"
    if score >= 840:
        return "critical"
    if score >= 620:
        return "elevated"
    return "standard"


def _pattern_ids_for_status(
    patterns: tuple[RoutingLearningPattern, ...],
    status: RoutingLearningStatus,
) -> tuple[str, ...]:
    return tuple(pattern.pattern_id for pattern in patterns if pattern.status == status)


def _pattern_ids_for_priority(
    patterns: tuple[RoutingLearningPattern, ...],
    priority: RoutingLearningPriority,
) -> tuple[str, ...]:
    return tuple(
        pattern.pattern_id for pattern in patterns if pattern.priority == priority
    )


def _overall_routing_learning_score(
    patterns: tuple[RoutingLearningPattern, ...],
) -> int:
    return sum(pattern.routing_learning_score for pattern in patterns) // len(patterns)


def _overall_routing_posture(
    patterns: tuple[RoutingLearningPattern, ...],
) -> RoutingLearningPosture:
    if any(pattern.status == "guarded" for pattern in patterns):
        return "guarded"
    if any(pattern.hitl_required for pattern in patterns):
        return "review_required"
    return "learnable"


def _unique_model_profile_ids(values: object) -> tuple[str, ...]:
    unique: list[str] = []
    for value in values:
        model_profile_id = str(value)
        if model_profile_id not in unique:
            unique.append(model_profile_id)
    return tuple(unique)


def _required_learning_signal(
    signal_id: str,
    plan: AdaptiveLearningPlan,
) -> AdaptiveLearningSignal:
    for signal in plan.signals:
        if signal.signal_id == signal_id:
            return signal
    raise ValueError("required routing learning adaptive metadata is missing")


def _required_routing_decision(
    task_type: TaskRoutingType,
    registry: TaskAwareRoutingRegistry,
) -> TaskAwareRoutingDecision:
    for decision in registry.decisions:
        if decision.task_type == task_type:
            return decision
    raise ValueError("required task-aware routing metadata is missing")


def _routing_summary(
    kind: RoutingLearningPatternKind,
    status: RoutingLearningStatus,
) -> str:
    if status == "guarded":
        return f"Surface {kind} as guarded routing metadata without route application."
    if status == "review_required":
        return f"Surface {kind} for review before future routing learning behavior."
    return f"Surface {kind} as learnable routing metadata only."


def _pattern_actions(kind: RoutingLearningPatternKind) -> tuple[str, ...]:
    return (
        f"Expose {kind} as derived routing learning metadata.",
        "Keep route application, provider/model switching, provider execution, "
        "HITL emission, workflow control, storage, Runtime Evolution, and "
        "output mutation disabled.",
    )


def _plan_actions(patterns: tuple[RoutingLearningPattern, ...]) -> tuple[str, ...]:
    actions = [
        "Expose routing learning patterns as advisory metadata only.",
        "Keep applied routing pattern ids empty.",
        "Preserve routing application, provider/model switching, provider "
        "execution, local probing, HITL emission, workflow, storage, output, "
        "and Runtime Evolution boundaries.",
    ]
    if any(pattern.hitl_required for pattern in patterns):
        actions.append("Require review before any future routing learning behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
