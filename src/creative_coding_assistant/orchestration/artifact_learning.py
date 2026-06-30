"""V6.1 advisory artifact learning."""

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
from creative_coding_assistant.orchestration.artifact_capability_matrix import (
    ArtifactCapabilityFit,
    ArtifactCapabilityMatrix,
    derive_artifact_capability_matrix,
)
from creative_coding_assistant.orchestration.artifact_planner import (
    ArtifactFamily,
    ArtifactPlan,
    ArtifactType,
    derive_artifact_plan,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityId,
    derive_runtime_capability_profile,
)

ArtifactLearningPatternKind = Literal[
    "artifact_shape_learning",
    "capability_fit_learning",
    "artifact_risk_learning",
    "artifact_guardrail_learning",
]
ArtifactLearningStatus = Literal["learnable", "review_required", "guarded"]
ArtifactLearningPriority = Literal["standard", "elevated", "critical", "guarded"]
ArtifactLearningPosture = Literal["learnable", "review_required", "guarded"]

ARTIFACT_LEARNING_PATTERN_SERIALIZATION_VERSION = "artifact_learning_pattern.v1"
ARTIFACT_LEARNING_PLAN_SERIALIZATION_VERSION = "artifact_learning_plan.v1"
ARTIFACT_LEARNING_AUTHORITY_BOUNDARY = (
    "V6.1 artifact learning derives artifact patterns from read-only artifact "
    "planning, artifact capability, and adaptive learning metadata only; it "
    "does not select artifacts, mutate artifacts, generate artifacts, execute "
    "artifacts, execute runtimes, change preview behavior, merge artifacts, "
    "export artifacts, route providers or models, switch providers or models, "
    "emit HITL requests, control workflows, mutate workflow graphs, trigger "
    "retries or refinements, mutate prompts, write storage, modify generated "
    "output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "artifact_selection",
    "artifact_mutation",
    "artifact_generation",
    "artifact_execution",
    "runtime_execution",
    "preview_behavior_change",
    "artifact_merge",
    "artifact_export",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
    "provider_execution",
    "local_runtime_probe",
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


class ArtifactLearningPattern(BaseModel):
    """One advisory artifact learning pattern."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    pattern_id: str = Field(min_length=1, max_length=180)
    pattern_kind: ArtifactLearningPatternKind
    status: ArtifactLearningStatus
    priority: ArtifactLearningPriority
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_artifact_plan_role: str = Field(min_length=1, max_length=120)
    source_artifact_capability_role: str = Field(min_length=1, max_length=120)
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    source_workflow_risk_factor_id: str = Field(min_length=1, max_length=180)
    artifact_type: ArtifactType
    artifact_family: ArtifactFamily
    target_runtime_id: RuntimeCapabilityId
    artifact_fit: ArtifactCapabilityFit
    creative_fit: ArtifactCapabilityFit
    generative_fit: ArtifactCapabilityFit
    interaction_fit: ArtifactCapabilityFit
    audiovisual_fit: ArtifactCapabilityFit
    export_fit: ArtifactCapabilityFit
    interoperability_fit: ArtifactCapabilityFit
    portability_fit: ArtifactCapabilityFit
    capability_confidence: float = Field(ge=0, le=1)
    missing_information_count: int = Field(ge=0, le=20)
    hitl_question_count: int = Field(ge=0, le=20)
    capability_risk_count: int = Field(ge=0, le=20)
    learning_priority_score: int = Field(ge=0, le=1_000)
    artifact_learning_weight: int = Field(ge=0, le=240)
    artifact_learning_score: int = Field(ge=0, le=1_000)
    hitl_required: bool
    artifact_pattern_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    artifact_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    artifact_learning_implemented: Literal[True] = True
    artifact_pattern_metadata_implemented: Literal[True] = True
    artifact_planning_metadata_used: Literal[True] = True
    artifact_capability_metadata_used: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    artifact_selection_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    artifact_generation_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    runtime_execution_implemented: Literal[False] = False
    preview_behavior_change_implemented: Literal[False] = False
    artifact_merge_implemented: Literal[False] = False
    artifact_export_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["artifact_learning_pattern.v1"] = (
        ARTIFACT_LEARNING_PATTERN_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _pattern_matches_contract(self) -> Self:
        if self.pattern_id != f"artifact_learning::{self.pattern_kind}":
            raise ValueError("pattern_id must match pattern_kind")
        if self.artifact_learning_score != _artifact_learning_score(
            artifact_fit=self.artifact_fit,
            creative_fit=self.creative_fit,
            generative_fit=self.generative_fit,
            interaction_fit=self.interaction_fit,
            audiovisual_fit=self.audiovisual_fit,
            export_fit=self.export_fit,
            interoperability_fit=self.interoperability_fit,
            portability_fit=self.portability_fit,
            capability_confidence=self.capability_confidence,
            learning_priority_score=self.learning_priority_score,
            missing_information_count=self.missing_information_count,
            hitl_question_count=self.hitl_question_count,
            capability_risk_count=self.capability_risk_count,
            artifact_learning_weight=self.artifact_learning_weight,
        ):
            raise ValueError("artifact_learning_score must combine source scores")
        if self.priority != _artifact_priority(
            self.artifact_learning_score,
            self.status,
        ):
            raise ValueError("priority must match score and status")
        if self.status == "guarded" and not self.hitl_required:
            raise ValueError("guarded artifact learning requires HITL posture")
        return self


class ArtifactLearningPlan(BaseModel):
    """Bounded V6.1 advisory artifact learning plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["artifact_learning"] = "artifact_learning"
    serialization_version: Literal["artifact_learning_plan.v1"] = (
        ARTIFACT_LEARNING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ARTIFACT_LEARNING_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_artifact_plan_role: str = Field(min_length=1, max_length=120)
    source_artifact_capability_role: str = Field(min_length=1, max_length=120)
    source_adaptive_learning_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    patterns: tuple[ArtifactLearningPattern, ...] = Field(min_length=4, max_length=4)
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
    applied_artifact_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    pattern_count: int = Field(ge=4, le=4)
    review_required_pattern_count: int = Field(ge=0, le=4)
    guarded_pattern_count: int = Field(ge=0, le=4)
    hitl_required_pattern_count: int = Field(ge=0, le=4)
    highest_artifact_learning_score: int = Field(ge=0, le=1_000)
    overall_artifact_learning_score: int = Field(ge=0, le=1_000)
    overall_artifact_learning_posture: ArtifactLearningPosture
    artifact_family: ArtifactFamily
    strongest_target_runtime_ids: tuple[RuntimeCapabilityId, ...] = Field(
        min_length=1,
        max_length=3,
    )
    weakest_target_runtime_ids: tuple[RuntimeCapabilityId, ...] = Field(
        min_length=1,
        max_length=3,
    )
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    artifact_learning_implemented: Literal[True] = True
    artifact_pattern_metadata_implemented: Literal[True] = True
    artifact_planning_metadata_used: Literal[True] = True
    artifact_capability_metadata_used: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    artifact_selection_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    artifact_generation_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    runtime_execution_implemented: Literal[False] = False
    preview_behavior_change_implemented: Literal[False] = False
    artifact_merge_implemented: Literal[False] = False
    artifact_export_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
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
        if self.applied_artifact_pattern_ids:
            raise ValueError("applied_artifact_pattern_ids must remain empty")
        if self.pattern_count != len(self.patterns):
            raise ValueError("pattern_count must match patterns")
        if self.review_required_pattern_count != len(self.review_required_pattern_ids):
            raise ValueError("review_required_pattern_count must match patterns")
        if self.guarded_pattern_count != len(self.guarded_pattern_ids):
            raise ValueError("guarded_pattern_count must match patterns")
        if self.hitl_required_pattern_count != len(self.hitl_required_pattern_ids):
            raise ValueError("hitl_required_pattern_count must match patterns")
        if self.highest_artifact_learning_score != max(
            pattern.artifact_learning_score for pattern in self.patterns
        ):
            raise ValueError("highest_artifact_learning_score must match patterns")
        if self.overall_artifact_learning_score != _overall_artifact_learning_score(
            self.patterns,
        ):
            raise ValueError("overall_artifact_learning_score must match patterns")
        if self.overall_artifact_learning_posture != _overall_artifact_posture(
            self.patterns,
        ):
            raise ValueError("overall_artifact_learning_posture must match patterns")
        for pattern in self.patterns:
            if pattern.route_name != self.route_name:
                raise ValueError("pattern route_name must match plan")
            if pattern.task_type != self.task_type:
                raise ValueError("pattern task_type must match plan")
            if pattern.artifact_family != self.artifact_family:
                raise ValueError("pattern artifact_family must match plan")
        return self


def learn_artifacts(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    adaptive_learning: AdaptiveLearningPlan | None = None,
    artifact_plan: ArtifactPlan | None = None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None = None,
    request: AssistantRequest | None = None,
) -> ArtifactLearningPlan:
    """Derive artifact learning patterns without selecting or mutating artifacts."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    learning_plan = adaptive_learning or evaluate_adaptive_learning_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    source_request = request or _default_artifact_request()
    plan = artifact_plan or derive_artifact_plan(
        request=source_request,
        route_decision=None,
    )
    matrix = artifact_capability_matrix or derive_artifact_capability_matrix(
        request=source_request,
        route_decision=None,
        artifact_plan=plan,
        artifact_dependency_graph=None,
        runtime_capabilities=derive_runtime_capability_profile(
            request=source_request,
            route_decision=None,
            creative_translation=None,
            creative_strategy=None,
            creative_techniques=None,
            creative_plan=None,
            creative_constraints=None,
        ),
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
        artifact_plan=plan,
        artifact_capability_matrix=matrix,
    )
    return ArtifactLearningPlan(
        route_name=route_name,
        task_type=learning_plan.task_type,
        source_artifact_plan_role=plan.role,
        source_artifact_capability_role=matrix.role,
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
        applied_artifact_pattern_ids=(),
        pattern_count=len(patterns),
        review_required_pattern_count=len(
            _pattern_ids_for_status(patterns, "review_required")
        ),
        guarded_pattern_count=len(_pattern_ids_for_status(patterns, "guarded")),
        hitl_required_pattern_count=sum(
            1 for pattern in patterns if pattern.hitl_required
        ),
        highest_artifact_learning_score=max(
            pattern.artifact_learning_score for pattern in patterns
        ),
        overall_artifact_learning_score=_overall_artifact_learning_score(patterns),
        overall_artifact_learning_posture=_overall_artifact_posture(patterns),
        artifact_family=plan.artifact_family,
        strongest_target_runtime_ids=matrix.strongest_targets,
        weakest_target_runtime_ids=matrix.weakest_targets,
        advisory_actions=_plan_actions(patterns),
    )


def artifact_learning_pattern_by_id(
    pattern_id: str,
    plan: ArtifactLearningPlan | None = None,
) -> ArtifactLearningPattern | None:
    source_plan = plan or learn_artifacts()
    for pattern in source_plan.patterns:
        if pattern.pattern_id == pattern_id:
            return pattern
    return None


def artifact_learning_patterns_for_status(
    status: ArtifactLearningStatus,
    plan: ArtifactLearningPlan | None = None,
) -> tuple[ArtifactLearningPattern, ...]:
    source_plan = plan or learn_artifacts()
    return tuple(
        pattern for pattern in source_plan.patterns if pattern.status == status
    )


def artifact_learning_patterns_for_priority(
    priority: ArtifactLearningPriority,
    plan: ArtifactLearningPlan | None = None,
) -> tuple[ArtifactLearningPattern, ...]:
    source_plan = plan or learn_artifacts()
    return tuple(
        pattern for pattern in source_plan.patterns if pattern.priority == priority
    )


def _patterns(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    artifact_plan: ArtifactPlan,
    artifact_capability_matrix: ArtifactCapabilityMatrix,
) -> tuple[ArtifactLearningPattern, ...]:
    targets = (
        artifact_capability_matrix.strongest_targets[0],
        artifact_capability_matrix.strongest_targets[1],
        artifact_capability_matrix.strongest_targets[2],
        artifact_capability_matrix.weakest_targets[0],
    )
    return (
        _pattern(
            kind="artifact_shape_learning",
            target=targets[0],
            learning_signal_id="adaptive_learning::workflow_pattern_learning",
            weight=180,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            artifact_plan=artifact_plan,
            artifact_capability_matrix=artifact_capability_matrix,
        ),
        _pattern(
            kind="capability_fit_learning",
            target=targets[1],
            learning_signal_id="adaptive_learning::strategy_pattern_learning",
            weight=160,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            artifact_plan=artifact_plan,
            artifact_capability_matrix=artifact_capability_matrix,
        ),
        _pattern(
            kind="artifact_risk_learning",
            target=targets[2],
            learning_signal_id="adaptive_learning::governance_feedback_learning",
            weight=150,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            artifact_plan=artifact_plan,
            artifact_capability_matrix=artifact_capability_matrix,
        ),
        _pattern(
            kind="artifact_guardrail_learning",
            target=targets[3],
            learning_signal_id="adaptive_learning::runtime_guardrail_learning",
            weight=220,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            artifact_plan=artifact_plan,
            artifact_capability_matrix=artifact_capability_matrix,
        ),
    )


def _pattern(
    *,
    kind: ArtifactLearningPatternKind,
    target: RuntimeCapabilityId,
    learning_signal_id: str,
    weight: int,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    artifact_plan: ArtifactPlan,
    artifact_capability_matrix: ArtifactCapabilityMatrix,
) -> ArtifactLearningPattern:
    learning_signal = _required_learning_signal(learning_signal_id, adaptive_learning)
    profile = _required_target_profile(target, artifact_capability_matrix)
    missing_count = len(artifact_plan.missing_information) + len(
        artifact_capability_matrix.missing_capability_information
    )
    hitl_count = len(artifact_plan.hitl_questions) + len(
        artifact_capability_matrix.hitl_questions
    )
    risk_count = len(artifact_plan.implementation_risks) + len(
        artifact_capability_matrix.capability_risks
    )
    status = _pattern_status(
        kind=kind,
        learning_signal=learning_signal,
        missing_information_count=missing_count,
        hitl_question_count=hitl_count,
    )
    score = _artifact_learning_score(
        artifact_fit=profile.artifact_fit,
        creative_fit=profile.creative_fit,
        generative_fit=profile.generative_fit,
        interaction_fit=profile.interaction_fit,
        audiovisual_fit=profile.audiovisual_fit,
        export_fit=profile.export_fit,
        interoperability_fit=profile.interoperability_fit,
        portability_fit=profile.portability_fit,
        capability_confidence=profile.capability_confidence,
        learning_priority_score=learning_signal.learning_priority_score,
        missing_information_count=missing_count,
        hitl_question_count=hitl_count,
        capability_risk_count=risk_count,
        artifact_learning_weight=weight,
    )
    return ArtifactLearningPattern(
        pattern_id=f"artifact_learning::{kind}",
        pattern_kind=kind,
        status=status,
        priority=_artifact_priority(score, status),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_artifact_plan_role=artifact_plan.role,
        source_artifact_capability_role=artifact_capability_matrix.role,
        source_learning_signal_id=learning_signal.signal_id,
        source_workflow_risk_factor_id=learning_signal.source_workflow_risk_factor_id,
        artifact_type=artifact_plan.artifact_type,
        artifact_family=artifact_plan.artifact_family,
        target_runtime_id=profile.target,
        artifact_fit=profile.artifact_fit,
        creative_fit=profile.creative_fit,
        generative_fit=profile.generative_fit,
        interaction_fit=profile.interaction_fit,
        audiovisual_fit=profile.audiovisual_fit,
        export_fit=profile.export_fit,
        interoperability_fit=profile.interoperability_fit,
        portability_fit=profile.portability_fit,
        capability_confidence=profile.capability_confidence,
        missing_information_count=missing_count,
        hitl_question_count=hitl_count,
        capability_risk_count=risk_count,
        learning_priority_score=learning_signal.learning_priority_score,
        artifact_learning_weight=weight,
        artifact_learning_score=score,
        hitl_required=learning_signal.hitl_required or hitl_count > 0,
        artifact_pattern_tags=(
            artifact_plan.artifact_family,
            profile.target,
            kind.removesuffix("_learning"),
        ),
        artifact_summary=_artifact_summary(kind, status),
        advisory_actions=_pattern_actions(kind),
        evidence=(
            f"artifact_family:{artifact_plan.artifact_family}",
            f"artifact_type:{artifact_plan.artifact_type}",
            f"target_runtime:{profile.target}",
            f"artifact_fit:{profile.artifact_fit}",
            f"capability_confidence:{profile.capability_confidence:.2f}",
            f"learning_signal:{learning_signal.signal_id}",
            f"learning_priority_score:{learning_signal.learning_priority_score}",
        ),
    )


def _artifact_learning_score(
    *,
    artifact_fit: ArtifactCapabilityFit,
    creative_fit: ArtifactCapabilityFit,
    generative_fit: ArtifactCapabilityFit,
    interaction_fit: ArtifactCapabilityFit,
    audiovisual_fit: ArtifactCapabilityFit,
    export_fit: ArtifactCapabilityFit,
    interoperability_fit: ArtifactCapabilityFit,
    portability_fit: ArtifactCapabilityFit,
    capability_confidence: float,
    learning_priority_score: int,
    missing_information_count: int,
    hitl_question_count: int,
    capability_risk_count: int,
    artifact_learning_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            _fit_score(artifact_fit) * 100
            + _fit_score(creative_fit) * 70
            + _fit_score(generative_fit) * 70
            + _fit_score(interaction_fit) * 55
            + _fit_score(audiovisual_fit) * 40
            + _fit_score(export_fit) * 35
            + _fit_score(interoperability_fit) * 35
            + _fit_score(portability_fit) * 35
            + int(capability_confidence * 260)
            + learning_priority_score // 4
            + artifact_learning_weight
            - missing_information_count * 25
            - hitl_question_count * 15
            - capability_risk_count * 20,
        ),
    )


def _fit_score(value: ArtifactCapabilityFit) -> int:
    return {"strong": 3, "moderate": 2, "weak": 1, "unsupported": 0}[value]


def _pattern_status(
    *,
    kind: ArtifactLearningPatternKind,
    learning_signal: AdaptiveLearningSignal,
    missing_information_count: int,
    hitl_question_count: int,
) -> ArtifactLearningStatus:
    if learning_signal.status == "guardrail" or kind == "artifact_guardrail_learning":
        return "guarded"
    if (
        learning_signal.hitl_required
        or missing_information_count
        or hitl_question_count
    ):
        return "review_required"
    return "learnable"


def _artifact_priority(
    score: int,
    status: ArtifactLearningStatus,
) -> ArtifactLearningPriority:
    if status == "guarded":
        return "guarded"
    if score >= 840:
        return "critical"
    if score >= 620:
        return "elevated"
    return "standard"


def _pattern_ids_for_status(
    patterns: tuple[ArtifactLearningPattern, ...],
    status: ArtifactLearningStatus,
) -> tuple[str, ...]:
    return tuple(pattern.pattern_id for pattern in patterns if pattern.status == status)


def _pattern_ids_for_priority(
    patterns: tuple[ArtifactLearningPattern, ...],
    priority: ArtifactLearningPriority,
) -> tuple[str, ...]:
    return tuple(
        pattern.pattern_id for pattern in patterns if pattern.priority == priority
    )


def _overall_artifact_learning_score(
    patterns: tuple[ArtifactLearningPattern, ...],
) -> int:
    return sum(pattern.artifact_learning_score for pattern in patterns) // len(patterns)


def _overall_artifact_posture(
    patterns: tuple[ArtifactLearningPattern, ...],
) -> ArtifactLearningPosture:
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
    raise ValueError("required artifact learning adaptive metadata is missing")


def _required_target_profile(
    target: RuntimeCapabilityId,
    matrix: ArtifactCapabilityMatrix,
):
    for profile in matrix.capability_profiles:
        if profile.target == target:
            return profile
    raise ValueError("required artifact capability metadata is missing")


def _artifact_summary(
    kind: ArtifactLearningPatternKind,
    status: ArtifactLearningStatus,
) -> str:
    if status == "guarded":
        return f"Surface {kind} as guarded artifact metadata without mutation."
    if status == "review_required":
        return f"Surface {kind} for review before future artifact learning behavior."
    return f"Surface {kind} as learnable artifact metadata only."


def _pattern_actions(kind: ArtifactLearningPatternKind) -> tuple[str, ...]:
    return (
        f"Expose {kind} as derived artifact learning metadata.",
        "Keep artifact selection, mutation, generation, execution, preview, "
        "merge, export, routing, workflow, storage, Runtime Evolution, and "
        "output mutation disabled.",
    )


def _plan_actions(patterns: tuple[ArtifactLearningPattern, ...]) -> tuple[str, ...]:
    actions = [
        "Expose artifact learning patterns as advisory metadata only.",
        "Keep applied artifact pattern ids empty.",
        "Preserve artifact selection, mutation, generation, execution, preview, "
        "merge, export, routing, workflow, storage, output, and Runtime "
        "Evolution boundaries.",
    ]
    if any(pattern.hitl_required for pattern in patterns):
        actions.append("Require review before any future artifact learning behavior.")
    return tuple(actions)


def _default_artifact_request() -> AssistantRequest:
    return AssistantRequest(
        query=(
            "Build a realtime p5.js and three.js audio reactive particle shader "
            "with export notes."
        ),
        mode=AssistantMode.GENERATE,
        domains=(CreativeCodingDomain.P5_JS,),
    )


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
