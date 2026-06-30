"""V6.1 advisory technique learning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration.adaptive_learning_engine import (
    AdaptiveLearningPlan,
    AdaptiveLearningSignal,
    evaluate_adaptive_learning_engine,
)
from creative_coding_assistant.orchestration.creative_technique import (
    CreativeTechniqueId,
    CreativeTechniqueProfile,
    TechniquePressure,
    derive_creative_technique_profile,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

TechniqueLearningPatternKind = Literal[
    "primary_technique_learning",
    "secondary_technique_learning",
    "tertiary_technique_learning",
    "fallback_technique_learning",
]
TechniqueLearningStatus = Literal["learnable", "review_required", "guarded"]
TechniqueLearningPriority = Literal["standard", "elevated", "critical", "guarded"]
TechniqueLearningPosture = Literal["learnable", "review_required", "guarded"]

TECHNIQUE_LEARNING_PATTERN_SERIALIZATION_VERSION = "technique_learning_pattern.v1"
TECHNIQUE_LEARNING_PLAN_SERIALIZATION_VERSION = "technique_learning_plan.v1"
TECHNIQUE_LEARNING_AUTHORITY_BOUNDARY = (
    "V6.1 technique learning derives technique patterns from read-only "
    "creative technique metadata and adaptive learning signals only; it does "
    "not render prompts, mutate prompts, select runtimes, change provider or "
    "model routing, execute providers, execute artifacts, invoke agents, emit "
    "HITL requests, control workflows, mutate workflow graphs, write storage, "
    "modify generated output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "technique_application",
    "prompt_rendering",
    "prompt_mutation",
    "runtime_selection",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
    "provider_execution",
    "artifact_execution",
    "agent_invocation",
    "hitl_request_emission",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "retry_or_refinement_triggering",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class TechniqueLearningPattern(BaseModel):
    """One advisory technique learning pattern."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    pattern_id: str = Field(min_length=1, max_length=180)
    pattern_kind: TechniqueLearningPatternKind
    status: TechniqueLearningStatus
    priority: TechniqueLearningPriority
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_technique_role: str = Field(min_length=1, max_length=120)
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    source_workflow_risk_factor_id: str = Field(min_length=1, max_length=180)
    technique_id: CreativeTechniqueId
    technique_confidence: float = Field(ge=0, le=1)
    technique_compatibility: str = Field(min_length=1, max_length=80)
    complexity_pressure: TechniquePressure
    performance_pressure: TechniquePressure
    learning_priority_score: int = Field(ge=0, le=1_000)
    technique_learning_weight: int = Field(ge=0, le=240)
    technique_learning_score: int = Field(ge=0, le=1_000)
    hitl_required: bool
    technique_pattern_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    technique_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    technique_learning_implemented: Literal[True] = True
    technique_pattern_metadata_implemented: Literal[True] = True
    creative_technique_metadata_used: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    technique_application_implemented: Literal[False] = False
    prompt_rendering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["technique_learning_pattern.v1"] = (
        TECHNIQUE_LEARNING_PATTERN_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _pattern_matches_contract(self) -> Self:
        if self.pattern_id != f"technique_learning::{self.pattern_kind}":
            raise ValueError("pattern_id must match pattern_kind")
        if self.technique_learning_score != _technique_learning_score(
            technique_confidence=self.technique_confidence,
            learning_priority_score=self.learning_priority_score,
            complexity_pressure=self.complexity_pressure,
            performance_pressure=self.performance_pressure,
            technique_learning_weight=self.technique_learning_weight,
        ):
            raise ValueError("technique_learning_score must combine source scores")
        if self.priority != _technique_priority(
            self.technique_learning_score,
            self.status,
        ):
            raise ValueError("priority must match score and status")
        if self.status == "guarded" and not self.hitl_required:
            raise ValueError("guarded technique learning requires HITL posture")
        return self


class TechniqueLearningPlan(BaseModel):
    """Bounded V6.1 advisory technique learning plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["technique_learning"] = "technique_learning"
    serialization_version: Literal["technique_learning_plan.v1"] = (
        TECHNIQUE_LEARNING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=TECHNIQUE_LEARNING_AUTHORITY_BOUNDARY,
        max_length=1600,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_creative_technique_role: str = Field(min_length=1, max_length=120)
    source_adaptive_learning_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    patterns: tuple[TechniqueLearningPattern, ...] = Field(min_length=4, max_length=4)
    pattern_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    learnable_pattern_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    review_required_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    guarded_pattern_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
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
    applied_technique_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    pattern_count: int = Field(ge=4, le=4)
    review_required_pattern_count: int = Field(ge=0, le=4)
    guarded_pattern_count: int = Field(ge=0, le=4)
    hitl_required_pattern_count: int = Field(ge=0, le=4)
    highest_technique_learning_score: int = Field(ge=0, le=1_000)
    overall_technique_learning_score: int = Field(ge=0, le=1_000)
    overall_technique_learning_posture: TechniqueLearningPosture
    primary_technique_id: CreativeTechniqueId
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    technique_learning_implemented: Literal[True] = True
    technique_pattern_metadata_implemented: Literal[True] = True
    creative_technique_metadata_used: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    technique_application_implemented: Literal[False] = False
    prompt_rendering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
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
        if self.applied_technique_pattern_ids:
            raise ValueError("applied_technique_pattern_ids must remain empty")
        if self.review_required_pattern_count != len(self.review_required_pattern_ids):
            raise ValueError("review_required_pattern_count must match patterns")
        if self.guarded_pattern_count != len(self.guarded_pattern_ids):
            raise ValueError("guarded_pattern_count must match patterns")
        if self.hitl_required_pattern_count != len(self.hitl_required_pattern_ids):
            raise ValueError("hitl_required_pattern_count must match patterns")
        if self.highest_technique_learning_score != max(
            pattern.technique_learning_score for pattern in self.patterns
        ):
            raise ValueError("highest_technique_learning_score must match patterns")
        if self.overall_technique_learning_score != _overall_technique_learning_score(
            self.patterns,
        ):
            raise ValueError("overall_technique_learning_score must match patterns")
        if self.overall_technique_learning_posture != _overall_technique_posture(
            self.patterns,
        ):
            raise ValueError("overall_technique_learning_posture must match patterns")
        for pattern in self.patterns:
            if pattern.route_name != self.route_name:
                raise ValueError("pattern route_name must match plan")
        return self


def learn_techniques(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    adaptive_learning: AdaptiveLearningPlan | None = None,
    technique_profile: CreativeTechniqueProfile | None = None,
    request: AssistantRequest | None = None,
) -> TechniqueLearningPlan:
    """Derive technique learning patterns without applying techniques."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    learning_plan = adaptive_learning or evaluate_adaptive_learning_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    profile = technique_profile or derive_creative_technique_profile(
        request=request or _default_technique_request(),
        route_decision=None,
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
        technique_profile=profile,
    )
    return TechniqueLearningPlan(
        route_name=route_name,
        task_type=learning_plan.task_type,
        source_creative_technique_role=profile.role,
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
        elevated_priority_pattern_ids=_pattern_ids_for_priority(patterns, "elevated"),
        critical_priority_pattern_ids=_pattern_ids_for_priority(patterns, "critical"),
        guarded_priority_pattern_ids=_pattern_ids_for_priority(patterns, "guarded"),
        hitl_required_pattern_ids=tuple(
            pattern.pattern_id for pattern in patterns if pattern.hitl_required
        ),
        applied_technique_pattern_ids=(),
        pattern_count=len(patterns),
        review_required_pattern_count=len(
            _pattern_ids_for_status(patterns, "review_required")
        ),
        guarded_pattern_count=len(_pattern_ids_for_status(patterns, "guarded")),
        hitl_required_pattern_count=sum(
            1 for pattern in patterns if pattern.hitl_required
        ),
        highest_technique_learning_score=max(
            pattern.technique_learning_score for pattern in patterns
        ),
        overall_technique_learning_score=_overall_technique_learning_score(patterns),
        overall_technique_learning_posture=_overall_technique_posture(patterns),
        primary_technique_id=profile.primary_technique,
        advisory_actions=_plan_actions(patterns),
    )


def technique_learning_pattern_by_id(
    pattern_id: str,
    plan: TechniqueLearningPlan | None = None,
) -> TechniqueLearningPattern | None:
    """Return one technique learning pattern without applying it."""

    source_plan = plan or learn_techniques()
    for pattern in source_plan.patterns:
        if pattern.pattern_id == pattern_id:
            return pattern
    return None


def technique_learning_patterns_for_status(
    status: TechniqueLearningStatus,
    plan: TechniqueLearningPlan | None = None,
) -> tuple[TechniqueLearningPattern, ...]:
    """Return technique learning patterns by advisory status."""

    source_plan = plan or learn_techniques()
    return tuple(
        pattern for pattern in source_plan.patterns if pattern.status == status
    )


def technique_learning_patterns_for_priority(
    priority: TechniqueLearningPriority,
    plan: TechniqueLearningPlan | None = None,
) -> tuple[TechniqueLearningPattern, ...]:
    """Return technique learning patterns by derived priority."""

    source_plan = plan or learn_techniques()
    return tuple(
        pattern for pattern in source_plan.patterns if pattern.priority == priority
    )


def _patterns(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    technique_profile: CreativeTechniqueProfile,
) -> tuple[TechniqueLearningPattern, ...]:
    alternatives = technique_profile.alternative_techniques
    if len(alternatives) < 3:
        raise ValueError("technique learning requires three alternatives")
    return (
        _pattern(
            kind="primary_technique_learning",
            technique_id=technique_profile.primary_technique,
            technique_confidence=technique_profile.confidence,
            learning_signal_id="adaptive_learning::workflow_pattern_learning",
            weight=200,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            technique_profile=technique_profile,
        ),
        _pattern(
            kind="secondary_technique_learning",
            technique_id=alternatives[0].technique,
            technique_confidence=alternatives[0].confidence,
            learning_signal_id="adaptive_learning::strategy_pattern_learning",
            weight=180,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            technique_profile=technique_profile,
        ),
        _pattern(
            kind="tertiary_technique_learning",
            technique_id=alternatives[1].technique,
            technique_confidence=alternatives[1].confidence,
            learning_signal_id="adaptive_learning::governance_feedback_learning",
            weight=160,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            technique_profile=technique_profile,
        ),
        _pattern(
            kind="fallback_technique_learning",
            technique_id=alternatives[2].technique,
            technique_confidence=alternatives[2].confidence,
            learning_signal_id="adaptive_learning::routing_boundary_learning",
            weight=210,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            technique_profile=technique_profile,
        ),
    )


def _pattern(
    *,
    kind: TechniqueLearningPatternKind,
    technique_id: CreativeTechniqueId,
    technique_confidence: float,
    learning_signal_id: str,
    weight: int,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    technique_profile: CreativeTechniqueProfile,
) -> TechniqueLearningPattern:
    learning_signal = _required_learning_signal(learning_signal_id, adaptive_learning)
    status = _pattern_status(learning_signal)
    score = _technique_learning_score(
        technique_confidence=technique_confidence,
        learning_priority_score=learning_signal.learning_priority_score,
        complexity_pressure=technique_profile.complexity_pressure,
        performance_pressure=technique_profile.performance_pressure,
        technique_learning_weight=weight,
    )
    return TechniqueLearningPattern(
        pattern_id=f"technique_learning::{kind}",
        pattern_kind=kind,
        status=status,
        priority=_technique_priority(score, status),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_technique_role=technique_profile.role,
        source_learning_signal_id=learning_signal.signal_id,
        source_workflow_risk_factor_id=learning_signal.source_workflow_risk_factor_id,
        technique_id=technique_id,
        technique_confidence=technique_confidence,
        technique_compatibility=technique_profile.compatibility,
        complexity_pressure=technique_profile.complexity_pressure,
        performance_pressure=technique_profile.performance_pressure,
        learning_priority_score=learning_signal.learning_priority_score,
        technique_learning_weight=weight,
        technique_learning_score=score,
        hitl_required=learning_signal.hitl_required,
        technique_pattern_tags=_technique_pattern_tags(kind, technique_id),
        technique_summary=_technique_summary(kind, status),
        advisory_actions=_pattern_actions(kind),
        evidence=(
            f"technique:{technique_id}",
            f"technique_confidence:{technique_confidence:.2f}",
            f"learning_signal:{learning_signal.signal_id}",
            f"learning_priority_score:{learning_signal.learning_priority_score}",
            f"complexity_pressure:{technique_profile.complexity_pressure}",
            f"performance_pressure:{technique_profile.performance_pressure}",
        ),
    )


def _technique_learning_score(
    *,
    technique_confidence: float,
    learning_priority_score: int,
    complexity_pressure: TechniquePressure,
    performance_pressure: TechniquePressure,
    technique_learning_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            int(technique_confidence * 600)
            + learning_priority_score // 3
            + technique_learning_weight
            - _pressure_penalty(complexity_pressure)
            - _pressure_penalty(performance_pressure),
        ),
    )


def _pressure_penalty(pressure: TechniquePressure) -> int:
    return {"low": 0, "medium": 40, "high": 100}[pressure]


def _pattern_status(
    learning_signal: AdaptiveLearningSignal,
) -> TechniqueLearningStatus:
    if learning_signal.status == "guardrail":
        return "guarded"
    if learning_signal.hitl_required:
        return "review_required"
    return "learnable"


def _technique_priority(
    score: int,
    status: TechniqueLearningStatus,
) -> TechniqueLearningPriority:
    if status == "guarded":
        return "guarded"
    if score >= 820:
        return "critical"
    if score >= 600:
        return "elevated"
    return "standard"


def _pattern_ids_for_status(
    patterns: tuple[TechniqueLearningPattern, ...],
    status: TechniqueLearningStatus,
) -> tuple[str, ...]:
    return tuple(pattern.pattern_id for pattern in patterns if pattern.status == status)


def _pattern_ids_for_priority(
    patterns: tuple[TechniqueLearningPattern, ...],
    priority: TechniqueLearningPriority,
) -> tuple[str, ...]:
    return tuple(
        pattern.pattern_id for pattern in patterns if pattern.priority == priority
    )


def _overall_technique_learning_score(
    patterns: tuple[TechniqueLearningPattern, ...],
) -> int:
    return sum(pattern.technique_learning_score for pattern in patterns) // len(
        patterns
    )


def _overall_technique_posture(
    patterns: tuple[TechniqueLearningPattern, ...],
) -> TechniqueLearningPosture:
    if any(pattern.status == "guarded" for pattern in patterns):
        return "guarded"
    if any(pattern.hitl_required for pattern in patterns):
        return "review_required"
    return "learnable"


def _required_learning_signal(
    signal_id: str,
    plan: AdaptiveLearningPlan,
) -> AdaptiveLearningSignal:
    for signal in plan.signals:
        if signal.signal_id == signal_id:
            return signal
    raise ValueError("required technique learning adaptive metadata is missing")


def _technique_pattern_tags(
    kind: TechniqueLearningPatternKind,
    technique_id: CreativeTechniqueId,
) -> tuple[str, ...]:
    return (
        str(technique_id),
        kind.removesuffix("_learning"),
        "creative_technique_pattern",
    )


def _technique_summary(
    kind: TechniqueLearningPatternKind,
    status: TechniqueLearningStatus,
) -> str:
    if status == "guarded":
        return f"Surface {kind} as guarded technique metadata without application."
    if status == "review_required":
        return f"Surface {kind} for review before future technique learning behavior."
    return f"Surface {kind} as learnable technique metadata only."


def _pattern_actions(kind: TechniqueLearningPatternKind) -> tuple[str, ...]:
    return (
        f"Expose {kind} as derived technique learning metadata.",
        "Keep technique application, prompt rendering or mutation, runtime "
        "selection, routing, provider execution, artifact execution, storage, "
        "Runtime Evolution, and output mutation disabled.",
    )


def _plan_actions(
    patterns: tuple[TechniqueLearningPattern, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose technique learning patterns as advisory metadata only.",
        "Keep applied technique pattern ids empty.",
        "Preserve technique application, prompt, runtime, routing, provider, "
        "artifact, workflow, storage, output, and Runtime Evolution boundaries.",
    ]
    if any(pattern.hitl_required for pattern in patterns):
        actions.append("Require review before any future technique learning behavior.")
    return tuple(actions)


def _default_technique_request() -> AssistantRequest:
    return AssistantRequest(
        query=(
            "Create an audio reactive particle system with noise fields and "
            "recursive geometry."
        ),
        mode=AssistantMode.GENERATE,
        domains=(CreativeCodingDomain.P5_JS,),
    )


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
