"""V6.1 advisory runtime learning."""

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
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityCandidate,
    RuntimeCapabilityId,
    RuntimeCapabilityProfile,
    derive_runtime_capability_profile,
)

RuntimeLearningPatternKind = Literal[
    "primary_runtime_learning",
    "secondary_runtime_learning",
    "audio_runtime_learning",
    "shader_runtime_learning",
]
RuntimeLearningStatus = Literal["learnable", "review_required", "guarded"]
RuntimeLearningPriority = Literal["standard", "elevated", "critical", "guarded"]
RuntimeLearningPosture = Literal["learnable", "review_required", "guarded"]

RUNTIME_LEARNING_PATTERN_SERIALIZATION_VERSION = "runtime_learning_pattern.v1"
RUNTIME_LEARNING_PLAN_SERIALIZATION_VERSION = "runtime_learning_plan.v1"
RUNTIME_LEARNING_AUTHORITY_BOUNDARY = (
    "V6.1 runtime learning derives runtime patterns from read-only runtime "
    "capability metadata and adaptive learning signals only; it does not "
    "select runtimes, create execution profiles, probe local runtimes, install "
    "dependencies, download models, change provider or model routing, execute "
    "providers or artifacts, change preview behavior, write storage, modify "
    "generated output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_selection",
    "execution_profile_creation",
    "local_runtime_probe",
    "runtime_installation",
    "automatic_model_download",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
    "provider_execution",
    "artifact_execution",
    "preview_behavior_change",
    "hitl_request_emission",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class RuntimeLearningPattern(BaseModel):
    """One advisory runtime learning pattern."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    pattern_id: str = Field(min_length=1, max_length=180)
    pattern_kind: RuntimeLearningPatternKind
    status: RuntimeLearningStatus
    priority: RuntimeLearningPriority
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_runtime_role: str = Field(min_length=1, max_length=120)
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    source_workflow_risk_factor_id: str = Field(min_length=1, max_length=180)
    runtime_id: RuntimeCapabilityId
    runtime_suitability: str = Field(min_length=1, max_length=80)
    runtime_confidence: float = Field(ge=0, le=1)
    implementation_complexity: str = Field(min_length=1, max_length=80)
    performance_pressure: str = Field(min_length=1, max_length=80)
    preview_support: str = Field(min_length=1, max_length=120)
    learning_priority_score: int = Field(ge=0, le=1_000)
    runtime_learning_weight: int = Field(ge=0, le=240)
    runtime_learning_score: int = Field(ge=0, le=1_000)
    hitl_required: bool
    runtime_pattern_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    runtime_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    runtime_learning_implemented: Literal[True] = True
    runtime_pattern_metadata_implemented: Literal[True] = True
    runtime_capability_metadata_used: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    runtime_selection_implemented: Literal[False] = False
    execution_profile_creation_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    runtime_installation_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    preview_behavior_change_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["runtime_learning_pattern.v1"] = (
        RUNTIME_LEARNING_PATTERN_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _pattern_matches_contract(self) -> Self:
        if self.pattern_id != f"runtime_learning::{self.pattern_kind}":
            raise ValueError("pattern_id must match pattern_kind")
        if self.runtime_learning_score != _runtime_learning_score(
            runtime_confidence=self.runtime_confidence,
            runtime_suitability=self.runtime_suitability,
            implementation_complexity=self.implementation_complexity,
            performance_pressure=self.performance_pressure,
            learning_priority_score=self.learning_priority_score,
            runtime_learning_weight=self.runtime_learning_weight,
        ):
            raise ValueError("runtime_learning_score must combine source scores")
        if self.priority != _runtime_priority(self.runtime_learning_score, self.status):
            raise ValueError("priority must match score and status")
        if self.status == "guarded" and not self.hitl_required:
            raise ValueError("guarded runtime learning requires HITL posture")
        return self


class RuntimeLearningPlan(BaseModel):
    """Bounded V6.1 advisory runtime learning plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["runtime_learning"] = "runtime_learning"
    serialization_version: Literal["runtime_learning_plan.v1"] = (
        RUNTIME_LEARNING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RUNTIME_LEARNING_AUTHORITY_BOUNDARY,
        max_length=1600,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_runtime_capability_role: str = Field(min_length=1, max_length=120)
    source_adaptive_learning_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    patterns: tuple[RuntimeLearningPattern, ...] = Field(min_length=4, max_length=4)
    pattern_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
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
    applied_runtime_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    pattern_count: int = Field(ge=4, le=4)
    review_required_pattern_count: int = Field(ge=0, le=4)
    guarded_pattern_count: int = Field(ge=0, le=4)
    hitl_required_pattern_count: int = Field(ge=0, le=4)
    highest_runtime_learning_score: int = Field(ge=0, le=1_000)
    overall_runtime_learning_score: int = Field(ge=0, le=1_000)
    overall_runtime_learning_posture: RuntimeLearningPosture
    likely_runtime_ids: tuple[RuntimeCapabilityId, ...] = Field(
        min_length=1,
        max_length=3,
    )
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    runtime_learning_implemented: Literal[True] = True
    runtime_pattern_metadata_implemented: Literal[True] = True
    runtime_capability_metadata_used: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    runtime_selection_implemented: Literal[False] = False
    execution_profile_creation_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    runtime_installation_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    preview_behavior_change_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
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
        if self.applied_runtime_pattern_ids:
            raise ValueError("applied_runtime_pattern_ids must remain empty")
        if self.pattern_count != len(self.patterns):
            raise ValueError("pattern_count must match patterns")
        if self.review_required_pattern_count != len(self.review_required_pattern_ids):
            raise ValueError("review_required_pattern_count must match patterns")
        if self.guarded_pattern_count != len(self.guarded_pattern_ids):
            raise ValueError("guarded_pattern_count must match patterns")
        if self.hitl_required_pattern_count != len(self.hitl_required_pattern_ids):
            raise ValueError("hitl_required_pattern_count must match patterns")
        if self.highest_runtime_learning_score != max(
            pattern.runtime_learning_score for pattern in self.patterns
        ):
            raise ValueError("highest_runtime_learning_score must match patterns")
        if self.overall_runtime_learning_score != _overall_runtime_learning_score(
            self.patterns,
        ):
            raise ValueError("overall_runtime_learning_score must match patterns")
        if self.overall_runtime_learning_posture != _overall_runtime_posture(
            self.patterns,
        ):
            raise ValueError("overall_runtime_learning_posture must match patterns")
        for pattern in self.patterns:
            if pattern.route_name != self.route_name:
                raise ValueError("pattern route_name must match plan")
        return self


def learn_runtimes(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    adaptive_learning: AdaptiveLearningPlan | None = None,
    runtime_profile: RuntimeCapabilityProfile | None = None,
    request: AssistantRequest | None = None,
) -> RuntimeLearningPlan:
    """Derive runtime learning patterns without selecting runtimes."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    learning_plan = adaptive_learning or evaluate_adaptive_learning_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    profile = runtime_profile or derive_runtime_capability_profile(
        request=request or _default_runtime_request(),
        route_decision=None,
        creative_translation=None,
        creative_strategy=None,
        creative_techniques=None,
        creative_plan=None,
        creative_constraints=None,
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
        runtime_profile=profile,
    )
    return RuntimeLearningPlan(
        route_name=route_name,
        task_type=learning_plan.task_type,
        source_runtime_capability_role=profile.role,
        source_adaptive_learning_serialization_version=(
            learning_plan.serialization_version
        ),
        execution_mode_ids=execution_modes.execution_mode_ids,
        patterns=patterns,
        pattern_ids=tuple(pattern.pattern_id for pattern in patterns),
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
        applied_runtime_pattern_ids=(),
        pattern_count=len(patterns),
        review_required_pattern_count=len(
            _pattern_ids_for_status(patterns, "review_required")
        ),
        guarded_pattern_count=len(_pattern_ids_for_status(patterns, "guarded")),
        hitl_required_pattern_count=sum(
            1 for pattern in patterns if pattern.hitl_required
        ),
        highest_runtime_learning_score=max(
            pattern.runtime_learning_score for pattern in patterns
        ),
        overall_runtime_learning_score=_overall_runtime_learning_score(patterns),
        overall_runtime_learning_posture=_overall_runtime_posture(patterns),
        likely_runtime_ids=profile.likely_candidates,
        advisory_actions=_plan_actions(patterns),
    )


def runtime_learning_pattern_by_id(
    pattern_id: str,
    plan: RuntimeLearningPlan | None = None,
) -> RuntimeLearningPattern | None:
    source_plan = plan or learn_runtimes()
    for pattern in source_plan.patterns:
        if pattern.pattern_id == pattern_id:
            return pattern
    return None


def runtime_learning_patterns_for_status(
    status: RuntimeLearningStatus,
    plan: RuntimeLearningPlan | None = None,
) -> tuple[RuntimeLearningPattern, ...]:
    source_plan = plan or learn_runtimes()
    return tuple(
        pattern for pattern in source_plan.patterns if pattern.status == status
    )


def runtime_learning_patterns_for_priority(
    priority: RuntimeLearningPriority,
    plan: RuntimeLearningPlan | None = None,
) -> tuple[RuntimeLearningPattern, ...]:
    source_plan = plan or learn_runtimes()
    return tuple(
        pattern for pattern in source_plan.patterns if pattern.priority == priority
    )


def _patterns(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    runtime_profile: RuntimeCapabilityProfile,
) -> tuple[RuntimeLearningPattern, ...]:
    candidates = runtime_profile.candidate_runtimes
    return (
        _pattern(
            kind="primary_runtime_learning",
            candidate=candidates[0],
            learning_signal_id="adaptive_learning::workflow_pattern_learning",
            weight=220,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            runtime_profile=runtime_profile,
        ),
        _pattern(
            kind="secondary_runtime_learning",
            candidate=candidates[1],
            learning_signal_id="adaptive_learning::strategy_pattern_learning",
            weight=190,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            runtime_profile=runtime_profile,
        ),
        _pattern(
            kind="audio_runtime_learning",
            candidate=candidates[2],
            learning_signal_id="adaptive_learning::governance_feedback_learning",
            weight=170,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            runtime_profile=runtime_profile,
        ),
        _pattern(
            kind="shader_runtime_learning",
            candidate=candidates[3],
            learning_signal_id="adaptive_learning::runtime_guardrail_learning",
            weight=210,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            runtime_profile=runtime_profile,
        ),
    )


def _pattern(
    *,
    kind: RuntimeLearningPatternKind,
    candidate: RuntimeCapabilityCandidate,
    learning_signal_id: str,
    weight: int,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    runtime_profile: RuntimeCapabilityProfile,
) -> RuntimeLearningPattern:
    learning_signal = _required_learning_signal(learning_signal_id, adaptive_learning)
    status = _pattern_status(learning_signal)
    score = _runtime_learning_score(
        runtime_confidence=candidate.confidence,
        runtime_suitability=candidate.suitability,
        implementation_complexity=candidate.implementation_complexity,
        performance_pressure=candidate.performance_pressure,
        learning_priority_score=learning_signal.learning_priority_score,
        runtime_learning_weight=weight,
    )
    return RuntimeLearningPattern(
        pattern_id=f"runtime_learning::{kind}",
        pattern_kind=kind,
        status=status,
        priority=_runtime_priority(score, status),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_runtime_role=runtime_profile.role,
        source_learning_signal_id=learning_signal.signal_id,
        source_workflow_risk_factor_id=learning_signal.source_workflow_risk_factor_id,
        runtime_id=candidate.runtime,
        runtime_suitability=candidate.suitability,
        runtime_confidence=candidate.confidence,
        implementation_complexity=candidate.implementation_complexity,
        performance_pressure=candidate.performance_pressure,
        preview_support=candidate.preview_support,
        learning_priority_score=learning_signal.learning_priority_score,
        runtime_learning_weight=weight,
        runtime_learning_score=score,
        hitl_required=learning_signal.hitl_required,
        runtime_pattern_tags=(candidate.runtime, kind.removesuffix("_learning")),
        runtime_summary=_runtime_summary(kind, status),
        advisory_actions=_pattern_actions(kind),
        evidence=(
            f"runtime:{candidate.runtime}",
            f"confidence:{candidate.confidence:.2f}",
            f"suitability:{candidate.suitability}",
            f"learning_signal:{learning_signal.signal_id}",
            f"learning_priority_score:{learning_signal.learning_priority_score}",
            f"preview_support:{candidate.preview_support}",
        ),
    )


def _runtime_learning_score(
    *,
    runtime_confidence: float,
    runtime_suitability: str,
    implementation_complexity: str,
    performance_pressure: str,
    learning_priority_score: int,
    runtime_learning_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            int(runtime_confidence * 520)
            + _fit_bonus(runtime_suitability)
            + learning_priority_score // 3
            + runtime_learning_weight
            - _pressure_penalty(implementation_complexity)
            - _pressure_penalty(performance_pressure),
        ),
    )


def _fit_bonus(value: str) -> int:
    return {"strong": 160, "moderate": 90, "weak": 20}[value]


def _pressure_penalty(value: str) -> int:
    return {"low": 0, "medium": 40, "high": 100}[value]


def _pattern_status(learning_signal: AdaptiveLearningSignal) -> RuntimeLearningStatus:
    if learning_signal.status == "guardrail":
        return "guarded"
    if learning_signal.hitl_required:
        return "review_required"
    return "learnable"


def _runtime_priority(
    score: int,
    status: RuntimeLearningStatus,
) -> RuntimeLearningPriority:
    if status == "guarded":
        return "guarded"
    if score >= 840:
        return "critical"
    if score >= 620:
        return "elevated"
    return "standard"


def _pattern_ids_for_status(
    patterns: tuple[RuntimeLearningPattern, ...],
    status: RuntimeLearningStatus,
) -> tuple[str, ...]:
    return tuple(pattern.pattern_id for pattern in patterns if pattern.status == status)


def _pattern_ids_for_priority(
    patterns: tuple[RuntimeLearningPattern, ...],
    priority: RuntimeLearningPriority,
) -> tuple[str, ...]:
    return tuple(
        pattern.pattern_id for pattern in patterns if pattern.priority == priority
    )


def _overall_runtime_learning_score(
    patterns: tuple[RuntimeLearningPattern, ...],
) -> int:
    return sum(pattern.runtime_learning_score for pattern in patterns) // len(patterns)


def _overall_runtime_posture(
    patterns: tuple[RuntimeLearningPattern, ...],
) -> RuntimeLearningPosture:
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
    raise ValueError("required runtime learning adaptive metadata is missing")


def _runtime_summary(
    kind: RuntimeLearningPatternKind,
    status: RuntimeLearningStatus,
) -> str:
    if status == "guarded":
        return f"Surface {kind} as guarded runtime metadata without selection."
    if status == "review_required":
        return f"Surface {kind} for review before future runtime learning behavior."
    return f"Surface {kind} as learnable runtime metadata only."


def _pattern_actions(kind: RuntimeLearningPatternKind) -> tuple[str, ...]:
    return (
        f"Expose {kind} as derived runtime learning metadata.",
        "Keep runtime selection, execution profile creation, probing, installs, "
        "routing, provider or artifact execution, preview changes, storage, "
        "Runtime Evolution, and output mutation disabled.",
    )


def _plan_actions(patterns: tuple[RuntimeLearningPattern, ...]) -> tuple[str, ...]:
    actions = [
        "Expose runtime learning patterns as advisory metadata only.",
        "Keep applied runtime pattern ids empty.",
        "Preserve runtime selection, probing, installation, routing, provider, "
        "artifact, preview, storage, output, and Runtime Evolution boundaries.",
    ]
    if any(pattern.hitl_required for pattern in patterns):
        actions.append("Require review before any future runtime learning behavior.")
    return tuple(actions)


def _default_runtime_request() -> AssistantRequest:
    return AssistantRequest(
        query="Build a realtime p5.js and three.js audio reactive particle shader.",
        mode=AssistantMode.GENERATE,
        domains=(CreativeCodingDomain.P5_JS,),
    )


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
