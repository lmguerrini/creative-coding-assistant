"""V6.1 advisory creative success learning metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)
from creative_coding_assistant.orchestration.success_pattern_discovery import (
    SuccessPattern,
    SuccessPatternDiscoveryPlan,
    discover_success_patterns,
    success_pattern_by_id,
)
from creative_coding_assistant.orchestration.workflow_success_tracking import (
    WorkflowSuccessIndicator,
    WorkflowSuccessTrackingPlan,
    track_workflow_success,
    workflow_success_indicator_by_id,
)

CreativeSuccessPatternKind = Literal[
    "creative_quality_success",
    "artifact_aesthetic_success",
    "usefulness_success",
    "originality_success",
]
CreativeSuccessPosture = Literal["candidate", "review_required", "guarded"]

CREATIVE_SUCCESS_PATTERN_SERIALIZATION_VERSION = "creative_success_pattern.v1"
CREATIVE_SUCCESS_LEARNING_PLAN_SERIALIZATION_VERSION = (
    "creative_success_learning_plan.v1"
)
CREATIVE_SUCCESS_LEARNING_AUTHORITY_BOUNDARY = (
    "V6.1 creative success learning specializes workflow success and success "
    "pattern metadata for creative coding workflows only; it does not mutate "
    "generated output, automatically mutate user preferences, write storage, "
    "execute providers, apply feedback, control workflows, or apply Runtime "
    "Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "generated_output_modification",
    "automatic_preference_mutation",
    "persistent_storage_write",
    "learning_feedback_application",
    "provider_execution",
    "workflow_control",
    "workflow_graph_mutation",
    "runtime_evolution_application",
)

_PATTERN_MAP: tuple[
    tuple[CreativeSuccessPatternKind, str, int],
    ...,
] = (
    ("creative_quality_success", "success_pattern::evaluation_success_pattern", 220),
    ("artifact_aesthetic_success", "success_pattern::artifact_success_pattern", 210),
    ("usefulness_success", "success_pattern::execution_success_pattern", 190),
    ("originality_success", "success_pattern::routing_success_pattern", 180),
)


class CreativeSuccessPattern(BaseModel):
    """One advisory creative success learning pattern."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    pattern_id: str = Field(min_length=1, max_length=200)
    pattern_kind: CreativeSuccessPatternKind
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_workflow_success_indicator_id: str = Field(min_length=1, max_length=180)
    source_success_pattern_id: str = Field(min_length=1, max_length=180)
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    source_success_confidence_band: str = Field(min_length=1, max_length=80)
    workflow_success_score: int = Field(ge=0, le=1_000)
    success_pattern_score: int = Field(ge=0, le=1_000)
    creative_quality_signals: tuple[str, ...] = Field(min_length=1, max_length=8)
    artifact_dimension_score: int = Field(ge=0, le=100)
    aesthetic_dimension_score: int = Field(ge=0, le=100)
    usefulness_dimension_score: int = Field(ge=0, le=100)
    originality_dimension_score: int = Field(ge=0, le=100)
    creative_success_weight: int = Field(ge=0, le=260)
    creative_success_score: int = Field(ge=0, le=1_000)
    explainability: str = Field(min_length=1, max_length=420)
    hitl_required: bool
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    creative_success_learning_implemented: Literal[True] = True
    creative_success_pattern_metadata_implemented: Literal[True] = True
    workflow_success_tracking_metadata_used: Literal[True] = True
    success_pattern_discovery_metadata_used: Literal[True] = True
    generated_output_mutation_implemented: Literal[False] = False
    automatic_preference_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    learning_feedback_application_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["creative_success_pattern.v1"] = (
        CREATIVE_SUCCESS_PATTERN_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _pattern_matches_contract(self) -> Self:
        if self.pattern_id != f"creative_success::{self.pattern_kind}":
            raise ValueError("pattern_id must match pattern_kind")
        if self.creative_success_score != _creative_success_score(
            workflow_success_score=self.workflow_success_score,
            success_pattern_score=self.success_pattern_score,
            artifact_dimension_score=self.artifact_dimension_score,
            aesthetic_dimension_score=self.aesthetic_dimension_score,
            usefulness_dimension_score=self.usefulness_dimension_score,
            originality_dimension_score=self.originality_dimension_score,
            creative_success_weight=self.creative_success_weight,
        ):
            raise ValueError("creative_success_score must combine source scores")
        return self


class CreativeSuccessLearningPlan(BaseModel):
    """Bounded V6.1 advisory creative success learning plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_success_learning"] = "creative_success_learning"
    serialization_version: Literal["creative_success_learning_plan.v1"] = (
        CREATIVE_SUCCESS_LEARNING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_SUCCESS_LEARNING_AUTHORITY_BOUNDARY,
        max_length=1400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_workflow_success_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_success_pattern_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    patterns: tuple[CreativeSuccessPattern, ...] = Field(min_length=4, max_length=4)
    pattern_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    hitl_required_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    pattern_count: int = Field(ge=4, le=4)
    highest_creative_success_score: int = Field(ge=0, le=1_000)
    overall_creative_success_score: int = Field(ge=0, le=1_000)
    overall_creative_success_posture: CreativeSuccessPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    creative_success_learning_implemented: Literal[True] = True
    creative_success_pattern_metadata_implemented: Literal[True] = True
    workflow_success_tracking_metadata_used: Literal[True] = True
    success_pattern_discovery_metadata_used: Literal[True] = True
    generated_output_mutation_implemented: Literal[False] = False
    automatic_preference_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    learning_feedback_application_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_patterns(self) -> Self:
        derived_ids = tuple(pattern.pattern_id for pattern in self.patterns)
        if len(set(derived_ids)) != len(derived_ids):
            raise ValueError("pattern_ids must be unique")
        if self.pattern_ids != derived_ids:
            raise ValueError("pattern_ids must match patterns")
        if self.hitl_required_pattern_ids != tuple(
            pattern.pattern_id for pattern in self.patterns if pattern.hitl_required
        ):
            raise ValueError("hitl_required_pattern_ids must match patterns")
        if self.pattern_count != len(self.patterns):
            raise ValueError("pattern_count must match patterns")
        if self.highest_creative_success_score != max(
            pattern.creative_success_score for pattern in self.patterns
        ):
            raise ValueError("highest_creative_success_score must match patterns")
        if self.overall_creative_success_score != _overall_success_score(
            self.patterns,
        ):
            raise ValueError("overall_creative_success_score must match patterns")
        if self.overall_creative_success_posture != _overall_success_posture(
            self.patterns,
        ):
            raise ValueError("overall_creative_success_posture must match patterns")
        return self


def learn_creative_success(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    workflow_success: WorkflowSuccessTrackingPlan | None = None,
    success_patterns: SuccessPatternDiscoveryPlan | None = None,
) -> CreativeSuccessLearningPlan:
    """Specialize creative success learning without mutating preferences."""

    route_name = _resolve_route(route)
    success_plan = workflow_success or track_workflow_success(
        route=route_name,
        task_type=str(task_type).strip(),
        execution_mode_id=execution_mode_id,
    )
    pattern_plan = success_patterns or discover_success_patterns(
        route=route_name,
        task_type=success_plan.task_type,
        execution_mode_id=execution_mode_id,
        workflow_success=success_plan,
    )
    normalized_mode = str(
        execution_mode_id or success_plan.indicators[0].execution_mode_id
    )
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")
    patterns = _patterns(
        route_name=route_name,
        task_type=success_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        workflow_success=success_plan,
        success_patterns=pattern_plan,
    )
    return CreativeSuccessLearningPlan(
        route_name=route_name,
        task_type=success_plan.task_type,
        source_workflow_success_serialization_version=success_plan.serialization_version,
        source_success_pattern_serialization_version=pattern_plan.serialization_version,
        execution_mode_ids=execution_modes.execution_mode_ids,
        patterns=patterns,
        pattern_ids=tuple(pattern.pattern_id for pattern in patterns),
        hitl_required_pattern_ids=tuple(
            pattern.pattern_id for pattern in patterns if pattern.hitl_required
        ),
        pattern_count=len(patterns),
        highest_creative_success_score=max(
            pattern.creative_success_score for pattern in patterns
        ),
        overall_creative_success_score=_overall_success_score(patterns),
        overall_creative_success_posture=_overall_success_posture(patterns),
        advisory_actions=_plan_actions(patterns),
    )


def creative_success_pattern_by_id(
    pattern_id: str,
    plan: CreativeSuccessLearningPlan | None = None,
) -> CreativeSuccessPattern | None:
    """Return one creative success pattern without applying it."""

    source_plan = plan or learn_creative_success()
    normalized_id = str(pattern_id).strip()
    for pattern in source_plan.patterns:
        if pattern.pattern_id == normalized_id:
            return pattern
    return None


def creative_success_patterns_for_quality_signal(
    quality_signal: str,
    plan: CreativeSuccessLearningPlan | None = None,
) -> tuple[CreativeSuccessPattern, ...]:
    """Return creative success patterns by quality signal."""

    source_plan = plan or learn_creative_success()
    normalized_signal = str(quality_signal).strip()
    return tuple(
        pattern
        for pattern in source_plan.patterns
        if normalized_signal in pattern.creative_quality_signals
    )


def _patterns(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    workflow_success: WorkflowSuccessTrackingPlan,
    success_patterns: SuccessPatternDiscoveryPlan,
) -> tuple[CreativeSuccessPattern, ...]:
    return tuple(
        _pattern(
            kind=kind,
            success_pattern=_required_success_pattern(pattern_id, success_patterns),
            workflow_success=workflow_success,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            weight=weight,
        )
        for kind, pattern_id, weight in _PATTERN_MAP
    )


def _pattern(
    *,
    kind: CreativeSuccessPatternKind,
    success_pattern: SuccessPattern,
    workflow_success: WorkflowSuccessTrackingPlan,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    weight: int,
) -> CreativeSuccessPattern:
    indicator = _required_success_indicator(
        success_pattern.source_success_indicator_id,
        workflow_success,
    )
    dimensions = _dimension_scores(kind, indicator, success_pattern)
    score = _creative_success_score(
        workflow_success_score=indicator.workflow_success_score,
        success_pattern_score=success_pattern.success_pattern_score,
        artifact_dimension_score=dimensions[0],
        aesthetic_dimension_score=dimensions[1],
        usefulness_dimension_score=dimensions[2],
        originality_dimension_score=dimensions[3],
        creative_success_weight=weight,
    )
    return CreativeSuccessPattern(
        pattern_id=f"creative_success::{kind}",
        pattern_kind=kind,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_workflow_success_indicator_id=indicator.indicator_id,
        source_success_pattern_id=success_pattern.pattern_id,
        source_learning_signal_id=success_pattern.source_learning_signal_id,
        source_success_confidence_band=success_pattern.success_confidence_band,
        workflow_success_score=indicator.workflow_success_score,
        success_pattern_score=success_pattern.success_pattern_score,
        creative_quality_signals=_quality_signals(kind),
        artifact_dimension_score=dimensions[0],
        aesthetic_dimension_score=dimensions[1],
        usefulness_dimension_score=dimensions[2],
        originality_dimension_score=dimensions[3],
        creative_success_weight=weight,
        creative_success_score=score,
        explainability=_explainability(kind, score),
        hitl_required=success_pattern.hitl_required or indicator.hitl_required,
        advisory_actions=_pattern_actions(kind),
        evidence=(
            f"workflow_success:{indicator.indicator_id}",
            f"success_pattern:{success_pattern.pattern_id}",
            f"learning_signal:{success_pattern.source_learning_signal_id}",
            f"creative_success_score:{score}",
        ),
    )


def _required_success_pattern(
    pattern_id: str,
    plan: SuccessPatternDiscoveryPlan,
) -> SuccessPattern:
    pattern = success_pattern_by_id(pattern_id, plan)
    if pattern is None:
        raise ValueError("required success pattern metadata is missing")
    return pattern


def _required_success_indicator(
    indicator_id: str,
    plan: WorkflowSuccessTrackingPlan,
) -> WorkflowSuccessIndicator:
    indicator = workflow_success_indicator_by_id(indicator_id, plan)
    if indicator is None:
        raise ValueError("required workflow success metadata is missing")
    return indicator


def _creative_success_score(
    *,
    workflow_success_score: int,
    success_pattern_score: int,
    artifact_dimension_score: int,
    aesthetic_dimension_score: int,
    usefulness_dimension_score: int,
    originality_dimension_score: int,
    creative_success_weight: int,
) -> int:
    dimension_score = (
        artifact_dimension_score
        + aesthetic_dimension_score
        + usefulness_dimension_score
        + originality_dimension_score
    ) * 2
    return min(
        1_000,
        max(
            0,
            workflow_success_score // 3
            + success_pattern_score // 3
            + dimension_score
            + creative_success_weight,
        ),
    )


def _dimension_scores(
    kind: CreativeSuccessPatternKind,
    indicator: WorkflowSuccessIndicator,
    success_pattern: SuccessPattern,
) -> tuple[int, int, int, int]:
    base = min(100, max(0, success_pattern.success_pattern_score // 10))
    confidence_bonus = 10 if indicator.confidence_band in {"strong", "moderate"} else 0
    return {
        "creative_quality_success": (
            min(100, base + 8),
            min(100, base + 10),
            min(100, base + confidence_bonus),
            min(100, base + 6),
        ),
        "artifact_aesthetic_success": (
            min(100, base + 12),
            min(100, base + 15),
            min(100, base + 4),
            min(100, base + 8),
        ),
        "usefulness_success": (
            min(100, base + 6),
            min(100, base + 2),
            min(100, base + 16),
            min(100, base + 4),
        ),
        "originality_success": (
            min(100, base + 8),
            min(100, base + 8),
            min(100, base + 5),
            min(100, base + 18),
        ),
    }[kind]


def _quality_signals(kind: CreativeSuccessPatternKind) -> tuple[str, ...]:
    return {
        "creative_quality_success": (
            "creative_quality",
            "aesthetic_coherence",
            "evaluation_alignment",
        ),
        "artifact_aesthetic_success": (
            "artifact_fit",
            "visual_aesthetic",
            "composition_quality",
        ),
        "usefulness_success": (
            "workflow_usefulness",
            "interactive_value",
            "implementation_clarity",
        ),
        "originality_success": (
            "creative_originality",
            "novel_pattern",
            "distinctive_interaction",
        ),
    }[kind]


def _overall_success_score(patterns: tuple[CreativeSuccessPattern, ...]) -> int:
    return sum(pattern.creative_success_score for pattern in patterns) // len(patterns)


def _overall_success_posture(
    patterns: tuple[CreativeSuccessPattern, ...],
) -> CreativeSuccessPosture:
    if any(pattern.source_success_confidence_band == "guarded" for pattern in patterns):
        return "guarded"
    if any(pattern.hitl_required for pattern in patterns):
        return "review_required"
    return "candidate"


def _explainability(kind: CreativeSuccessPatternKind, score: int) -> str:
    return (
        f"{kind} explains creative success as advisory metadata with score {score}; "
        "artifact, aesthetic, usefulness, and originality dimensions are not "
        "applied to generated output or preferences."
    )


def _pattern_actions(kind: CreativeSuccessPatternKind) -> tuple[str, ...]:
    return (
        f"Expose {kind} as creative success metadata.",
        "Keep generated output, user preferences, storage, feedback, and "
        "Runtime Evolution unchanged.",
    )


def _plan_actions(patterns: tuple[CreativeSuccessPattern, ...]) -> tuple[str, ...]:
    actions = [
        "Expose creative success learning as advisory metadata only.",
        "Keep generated output, preferences, storage, feedback, workflow control, "
        "and Runtime Evolution unchanged.",
    ]
    if any(pattern.hitl_required for pattern in patterns):
        actions.append("Require HITL before any future creative success application.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
