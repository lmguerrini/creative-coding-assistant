"""V6.1 advisory creative failure learning metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.failure_pattern_discovery import (
    FailurePattern,
    FailurePatternDiscoveryPlan,
    discover_failure_patterns,
    failure_pattern_by_id,
)
from creative_coding_assistant.orchestration.failure_tracking import (
    FailureTrackingIndicator,
    FailureTrackingPlan,
    failure_tracking_indicator_by_id,
    track_failures,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

CreativeFailurePatternKind = Literal[
    "artifact_preview_failure",
    "runtime_execution_failure",
    "aesthetic_prompt_failure",
    "retrieval_context_failure",
]
CreativeFailurePosture = Literal["observed_metadata", "review_required", "guarded"]

CREATIVE_FAILURE_PATTERN_SERIALIZATION_VERSION = "creative_failure_pattern.v1"
CREATIVE_FAILURE_LEARNING_PLAN_SERIALIZATION_VERSION = (
    "creative_failure_learning_plan.v1"
)
CREATIVE_FAILURE_LEARNING_AUTHORITY_BOUNDARY = (
    "V6.1 creative failure learning specializes failure tracking and failure "
    "pattern metadata for creative coding workflows only; it does not mutate "
    "generated output, automatically remediate failures, write storage, execute "
    "providers, control workflows, route terminal failures, or apply Runtime "
    "Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "generated_output_modification",
    "automatic_remediation",
    "persistent_storage_write",
    "provider_execution",
    "workflow_control",
    "workflow_graph_mutation",
    "terminal_failure_routing",
    "runtime_evolution_application",
)

_PATTERN_MAP: tuple[
    tuple[CreativeFailurePatternKind, str, int],
    ...,
] = (
    ("artifact_preview_failure", "failure_pattern::retry_failure_pattern", 220),
    ("runtime_execution_failure", "failure_pattern::performance_failure_pattern", 230),
    ("aesthetic_prompt_failure", "failure_pattern::langgraph_failure_pattern", 200),
    ("retrieval_context_failure", "failure_pattern::routing_failure_pattern", 190),
)


class CreativeFailurePattern(BaseModel):
    """One advisory creative failure learning pattern."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    pattern_id: str = Field(min_length=1, max_length=200)
    pattern_kind: CreativeFailurePatternKind
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_failure_indicator_id: str = Field(min_length=1, max_length=180)
    source_failure_pattern_id: str = Field(min_length=1, max_length=180)
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    failure_tracking_score: int = Field(ge=0, le=1_000)
    failure_pattern_score: int = Field(ge=0, le=1_000)
    common_creative_failure_modes: tuple[str, ...] = Field(min_length=1, max_length=8)
    artifact_dimension_score: int = Field(ge=0, le=100)
    preview_dimension_score: int = Field(ge=0, le=100)
    runtime_dimension_score: int = Field(ge=0, le=100)
    aesthetic_dimension_score: int = Field(ge=0, le=100)
    prompt_dimension_score: int = Field(ge=0, le=100)
    retrieval_dimension_score: int = Field(ge=0, le=100)
    creative_failure_weight: int = Field(ge=0, le=260)
    creative_failure_score: int = Field(ge=0, le=1_000)
    explainability: str = Field(min_length=1, max_length=460)
    hitl_required: bool
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    creative_failure_learning_implemented: Literal[True] = True
    creative_failure_pattern_metadata_implemented: Literal[True] = True
    failure_tracking_metadata_used: Literal[True] = True
    failure_pattern_discovery_metadata_used: Literal[True] = True
    generated_output_mutation_implemented: Literal[False] = False
    automatic_remediation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    terminal_failure_routing_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["creative_failure_pattern.v1"] = (
        CREATIVE_FAILURE_PATTERN_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _pattern_matches_contract(self) -> Self:
        if self.pattern_id != f"creative_failure::{self.pattern_kind}":
            raise ValueError("pattern_id must match pattern_kind")
        if self.creative_failure_score != _creative_failure_score(
            failure_tracking_score=self.failure_tracking_score,
            failure_pattern_score=self.failure_pattern_score,
            artifact_dimension_score=self.artifact_dimension_score,
            preview_dimension_score=self.preview_dimension_score,
            runtime_dimension_score=self.runtime_dimension_score,
            aesthetic_dimension_score=self.aesthetic_dimension_score,
            prompt_dimension_score=self.prompt_dimension_score,
            retrieval_dimension_score=self.retrieval_dimension_score,
            creative_failure_weight=self.creative_failure_weight,
        ):
            raise ValueError("creative_failure_score must combine source scores")
        return self


class CreativeFailureLearningPlan(BaseModel):
    """Bounded V6.1 advisory creative failure learning plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_failure_learning"] = "creative_failure_learning"
    serialization_version: Literal["creative_failure_learning_plan.v1"] = (
        CREATIVE_FAILURE_LEARNING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_FAILURE_LEARNING_AUTHORITY_BOUNDARY,
        max_length=1400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_failure_tracking_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_failure_pattern_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    patterns: tuple[CreativeFailurePattern, ...] = Field(min_length=4, max_length=4)
    pattern_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    hitl_required_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    pattern_count: int = Field(ge=4, le=4)
    highest_creative_failure_score: int = Field(ge=0, le=1_000)
    overall_creative_failure_score: int = Field(ge=0, le=1_000)
    overall_creative_failure_posture: CreativeFailurePosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    creative_failure_learning_implemented: Literal[True] = True
    creative_failure_pattern_metadata_implemented: Literal[True] = True
    failure_tracking_metadata_used: Literal[True] = True
    failure_pattern_discovery_metadata_used: Literal[True] = True
    generated_output_mutation_implemented: Literal[False] = False
    automatic_remediation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    terminal_failure_routing_implemented: Literal[False] = False
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
        if self.highest_creative_failure_score != max(
            pattern.creative_failure_score for pattern in self.patterns
        ):
            raise ValueError("highest_creative_failure_score must match patterns")
        if self.overall_creative_failure_score != _overall_failure_score(
            self.patterns,
        ):
            raise ValueError("overall_creative_failure_score must match patterns")
        if self.overall_creative_failure_posture != _overall_failure_posture(
            self.patterns,
        ):
            raise ValueError("overall_creative_failure_posture must match patterns")
        return self


def learn_creative_failures(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    failure_tracking: FailureTrackingPlan | None = None,
    failure_patterns: FailurePatternDiscoveryPlan | None = None,
) -> CreativeFailureLearningPlan:
    """Specialize creative failure learning without remediation."""

    route_name = _resolve_route(route)
    failure_plan = failure_tracking or track_failures(
        route=route_name,
        task_type=str(task_type).strip(),
        execution_mode_id=execution_mode_id,
    )
    pattern_plan = failure_patterns or discover_failure_patterns(
        route=route_name,
        task_type=failure_plan.task_type,
        execution_mode_id=execution_mode_id,
        failure_tracking=failure_plan,
    )
    normalized_mode = str(
        execution_mode_id or failure_plan.indicators[0].execution_mode_id
    )
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")
    patterns = _patterns(
        route_name=route_name,
        task_type=failure_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        failure_tracking=failure_plan,
        failure_patterns=pattern_plan,
    )
    return CreativeFailureLearningPlan(
        route_name=route_name,
        task_type=failure_plan.task_type,
        source_failure_tracking_serialization_version=failure_plan.serialization_version,
        source_failure_pattern_serialization_version=pattern_plan.serialization_version,
        execution_mode_ids=execution_modes.execution_mode_ids,
        patterns=patterns,
        pattern_ids=tuple(pattern.pattern_id for pattern in patterns),
        hitl_required_pattern_ids=tuple(
            pattern.pattern_id for pattern in patterns if pattern.hitl_required
        ),
        pattern_count=len(patterns),
        highest_creative_failure_score=max(
            pattern.creative_failure_score for pattern in patterns
        ),
        overall_creative_failure_score=_overall_failure_score(patterns),
        overall_creative_failure_posture=_overall_failure_posture(patterns),
        advisory_actions=_plan_actions(patterns),
    )


def creative_failure_pattern_by_id(
    pattern_id: str,
    plan: CreativeFailureLearningPlan | None = None,
) -> CreativeFailurePattern | None:
    """Return one creative failure pattern without applying remediation."""

    source_plan = plan or learn_creative_failures()
    normalized_id = str(pattern_id).strip()
    for pattern in source_plan.patterns:
        if pattern.pattern_id == normalized_id:
            return pattern
    return None


def creative_failure_patterns_for_mode(
    failure_mode: str,
    plan: CreativeFailureLearningPlan | None = None,
) -> tuple[CreativeFailurePattern, ...]:
    """Return creative failure patterns by common failure mode."""

    source_plan = plan or learn_creative_failures()
    normalized_mode = str(failure_mode).strip()
    return tuple(
        pattern
        for pattern in source_plan.patterns
        if normalized_mode in pattern.common_creative_failure_modes
    )


def _patterns(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    failure_tracking: FailureTrackingPlan,
    failure_patterns: FailurePatternDiscoveryPlan,
) -> tuple[CreativeFailurePattern, ...]:
    return tuple(
        _pattern(
            kind=kind,
            failure_pattern=_required_failure_pattern(pattern_id, failure_patterns),
            failure_tracking=failure_tracking,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            weight=weight,
        )
        for kind, pattern_id, weight in _PATTERN_MAP
    )


def _pattern(
    *,
    kind: CreativeFailurePatternKind,
    failure_pattern: FailurePattern,
    failure_tracking: FailureTrackingPlan,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    weight: int,
) -> CreativeFailurePattern:
    indicator = _required_failure_indicator(
        failure_pattern.source_failure_indicator_id,
        failure_tracking,
    )
    dimensions = _dimension_scores(kind, indicator, failure_pattern)
    score = _creative_failure_score(
        failure_tracking_score=indicator.failure_tracking_score,
        failure_pattern_score=failure_pattern.failure_pattern_score,
        artifact_dimension_score=dimensions[0],
        preview_dimension_score=dimensions[1],
        runtime_dimension_score=dimensions[2],
        aesthetic_dimension_score=dimensions[3],
        prompt_dimension_score=dimensions[4],
        retrieval_dimension_score=dimensions[5],
        creative_failure_weight=weight,
    )
    return CreativeFailurePattern(
        pattern_id=f"creative_failure::{kind}",
        pattern_kind=kind,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_failure_indicator_id=indicator.indicator_id,
        source_failure_pattern_id=failure_pattern.pattern_id,
        source_learning_signal_id=failure_pattern.source_learning_signal_id,
        failure_tracking_score=indicator.failure_tracking_score,
        failure_pattern_score=failure_pattern.failure_pattern_score,
        common_creative_failure_modes=_failure_modes(kind),
        artifact_dimension_score=dimensions[0],
        preview_dimension_score=dimensions[1],
        runtime_dimension_score=dimensions[2],
        aesthetic_dimension_score=dimensions[3],
        prompt_dimension_score=dimensions[4],
        retrieval_dimension_score=dimensions[5],
        creative_failure_weight=weight,
        creative_failure_score=score,
        explainability=_explainability(kind, score),
        hitl_required=failure_pattern.hitl_required or indicator.hitl_required,
        advisory_actions=_pattern_actions(kind),
        evidence=(
            f"failure_tracking:{indicator.indicator_id}",
            f"failure_pattern:{failure_pattern.pattern_id}",
            f"learning_signal:{failure_pattern.source_learning_signal_id}",
            f"creative_failure_score:{score}",
        ),
    )


def _required_failure_pattern(
    pattern_id: str,
    plan: FailurePatternDiscoveryPlan,
) -> FailurePattern:
    pattern = failure_pattern_by_id(pattern_id, plan)
    if pattern is None:
        raise ValueError("required failure pattern metadata is missing")
    return pattern


def _required_failure_indicator(
    indicator_id: str,
    plan: FailureTrackingPlan,
) -> FailureTrackingIndicator:
    indicator = failure_tracking_indicator_by_id(indicator_id, plan)
    if indicator is None:
        raise ValueError("required failure tracking metadata is missing")
    return indicator


def _creative_failure_score(
    *,
    failure_tracking_score: int,
    failure_pattern_score: int,
    artifact_dimension_score: int,
    preview_dimension_score: int,
    runtime_dimension_score: int,
    aesthetic_dimension_score: int,
    prompt_dimension_score: int,
    retrieval_dimension_score: int,
    creative_failure_weight: int,
) -> int:
    dimension_score = (
        artifact_dimension_score
        + preview_dimension_score
        + runtime_dimension_score
        + aesthetic_dimension_score
        + prompt_dimension_score
        + retrieval_dimension_score
    )
    return min(
        1_000,
        max(
            0,
            failure_tracking_score // 3
            + failure_pattern_score // 3
            + dimension_score
            + creative_failure_weight,
        ),
    )


def _dimension_scores(
    kind: CreativeFailurePatternKind,
    indicator: FailureTrackingIndicator,
    failure_pattern: FailurePattern,
) -> tuple[int, int, int, int, int, int]:
    base = min(100, max(0, failure_pattern.failure_pattern_score // 10))
    severity_bonus = min(12, indicator.failure_signal_count // 20)
    return {
        "artifact_preview_failure": (
            min(100, base + 14),
            min(100, base + 16),
            min(100, base + 6),
            min(100, base + 10),
            min(100, base + 4),
            min(100, base + 4),
        ),
        "runtime_execution_failure": (
            min(100, base + 5),
            min(100, base + 8),
            min(100, base + 18 + severity_bonus),
            min(100, base + 2),
            min(100, base + 4),
            min(100, base + 2),
        ),
        "aesthetic_prompt_failure": (
            min(100, base + 6),
            min(100, base + 4),
            min(100, base + 2),
            min(100, base + 16),
            min(100, base + 14),
            min(100, base + 4),
        ),
        "retrieval_context_failure": (
            min(100, base + 4),
            min(100, base + 4),
            min(100, base + 6),
            min(100, base + 6),
            min(100, base + 8),
            min(100, base + 18 + severity_bonus),
        ),
    }[kind]


def _failure_modes(kind: CreativeFailurePatternKind) -> tuple[str, ...]:
    return {
        "artifact_preview_failure": (
            "artifact_shape_mismatch",
            "preview_surface_mismatch",
            "interactive_affordance_gap",
        ),
        "runtime_execution_failure": (
            "runtime_capability_gap",
            "dependency_boundary",
            "performance_regression",
        ),
        "aesthetic_prompt_failure": (
            "aesthetic_direction_mismatch",
            "prompt_specificity_gap",
            "visual_coherence_risk",
        ),
        "retrieval_context_failure": (
            "retrieval_context_gap",
            "reference_alignment_gap",
            "domain_context_mismatch",
        ),
    }[kind]


def _overall_failure_score(patterns: tuple[CreativeFailurePattern, ...]) -> int:
    return sum(pattern.creative_failure_score for pattern in patterns) // len(patterns)


def _overall_failure_posture(
    patterns: tuple[CreativeFailurePattern, ...],
) -> CreativeFailurePosture:
    if any(pattern.hitl_required for pattern in patterns):
        return "guarded"
    return "observed_metadata"


def _explainability(kind: CreativeFailurePatternKind, score: int) -> str:
    return (
        f"{kind} explains creative failure risk as advisory metadata with score "
        f"{score}; artifact, preview, runtime, aesthetic, prompt, and retrieval "
        "dimensions do not trigger remediation or mutate output."
    )


def _pattern_actions(kind: CreativeFailurePatternKind) -> tuple[str, ...]:
    return (
        f"Expose {kind} as creative failure metadata.",
        "Keep remediation, generated output, storage, routing, workflow "
        "control, and Runtime Evolution unchanged.",
    )


def _plan_actions(patterns: tuple[CreativeFailurePattern, ...]) -> tuple[str, ...]:
    actions = [
        "Expose creative failure learning as advisory metadata only.",
        "Keep remediation, storage, terminal routing, workflow control, generated "
        "output, and Runtime Evolution unchanged.",
    ]
    if any(pattern.hitl_required for pattern in patterns):
        actions.append("Require HITL before any future creative failure remediation.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
