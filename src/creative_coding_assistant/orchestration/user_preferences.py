"""V6.2 advisory user preference metadata."""

from __future__ import annotations

from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.long_term_creative_memory import (
    LongTermCreativeMemoryPlan,
    build_long_term_creative_memory,
    long_term_creative_memory_record_by_id,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

UserPreferenceKind = Literal[
    "visual_style_preference",
    "interaction_preference",
    "constraint_preference",
    "technical_stack_preference",
    "review_depth_preference",
]
UserPreferenceStatus = Literal["candidate", "review_required", "guarded"]
UserPreferenceConfidence = Literal["low", "medium", "high", "guarded"]
UserPreferencePosture = Literal["candidate", "review_required", "guarded"]
UserPreferenceScope = Literal[
    "style",
    "interaction",
    "constraints",
    "technical_stack",
    "review_depth",
]

USER_PREFERENCE_SIGNAL_SERIALIZATION_VERSION = "user_preference_signal.v1"
USER_PREFERENCES_PLAN_SERIALIZATION_VERSION = "user_preferences_plan.v1"
USER_PREFERENCES_AUTHORITY_BOUNDARY = (
    "V6.2 user preferences models explicit and inferred preference posture as "
    "inspectable advisory metadata only; it does not write preference storage, "
    "create preference records, update preference records, learn preferences, "
    "mutate preferences, apply personalization, execute memory retrieval, "
    "change provider or model routing, execute providers, invoke agents, "
    "control workflows, mutate workflow graphs, trigger retries or refinements, "
    "mutate prompts, modify generated output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "preference_storage_write",
    "preference_record_creation",
    "preference_record_update",
    "preference_record_deletion",
    "automatic_preference_learning",
    "automatic_preference_mutation",
    "automatic_personalization_application",
    "memory_retrieval_execution",
    "memory_storage_write",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
    "provider_execution",
    "agent_invocation",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class UserPreferenceSignal(BaseModel):
    """One advisory user preference signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    preference_id: str = Field(min_length=1, max_length=180)
    preference_kind: UserPreferenceKind
    status: UserPreferenceStatus
    confidence: UserPreferenceConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    preference_scope: UserPreferenceScope
    source_long_term_memory_record_id: str = Field(min_length=1, max_length=180)
    preference_statement: str = Field(min_length=1, max_length=320)
    explicitness_score: int = Field(ge=0, le=100)
    consistency_score: int = Field(ge=0, le=100)
    conflict_risk_score: int = Field(ge=0, le=100)
    sensitivity_weight: int = Field(ge=0, le=240)
    preference_governance_score: int = Field(ge=0, le=1_000)
    hitl_required_before_mutation: bool
    applicable_context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    user_preferences_implemented: Literal[True] = True
    preference_signal_metadata_implemented: Literal[True] = True
    long_term_memory_source_used: Literal[True] = True
    preference_storage_write_implemented: Literal[False] = False
    preference_record_creation_implemented: Literal[False] = False
    preference_record_update_implemented: Literal[False] = False
    preference_record_deletion_implemented: Literal[False] = False
    automatic_preference_learning_implemented: Literal[False] = False
    preference_mutation_implemented: Literal[False] = False
    personalization_application_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_storage_write_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["user_preference_signal.v1"] = (
        USER_PREFERENCE_SIGNAL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.preference_id != f"user_preferences::{self.preference_kind}":
            raise ValueError("preference_id must match preference_kind")
        if self.preference_governance_score != _preference_governance_score(
            explicitness_score=self.explicitness_score,
            consistency_score=self.consistency_score,
            conflict_risk_score=self.conflict_risk_score,
            sensitivity_weight=self.sensitivity_weight,
        ):
            raise ValueError(
                "preference_governance_score must combine source scores"
            )
        if self.status != _preference_status(self.preference_governance_score):
            raise ValueError("status must match preference_governance_score")
        if self.confidence != _preference_confidence(
            self.preference_governance_score
        ):
            raise ValueError("confidence must match preference_governance_score")
        if not self.hitl_required_before_mutation:
            raise ValueError("preference mutation requires HITL posture")
        return self


class UserPreferencesPlan(BaseModel):
    """Bounded V6.2 advisory user preferences plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["user_preferences"] = "user_preferences"
    serialization_version: Literal["user_preferences_plan.v1"] = (
        USER_PREFERENCES_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=USER_PREFERENCES_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_long_term_memory_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_long_term_memory_record_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    preferences: tuple[UserPreferenceSignal, ...] = Field(
        min_length=5,
        max_length=5,
    )
    preference_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    candidate_preference_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    review_required_preference_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_preference_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    high_confidence_preference_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    hitl_required_preference_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    learned_preference_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_preference_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    personalized_preference_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    preference_count: int = Field(ge=5, le=5)
    candidate_preference_count: int = Field(ge=0, le=5)
    review_required_preference_count: int = Field(ge=0, le=5)
    guarded_preference_count: int = Field(ge=0, le=5)
    high_confidence_preference_count: int = Field(ge=0, le=5)
    hitl_required_preference_count: int = Field(ge=0, le=5)
    highest_preference_governance_score: int = Field(ge=0, le=1_000)
    overall_preference_governance_score: int = Field(ge=0, le=1_000)
    overall_preference_posture: UserPreferencePosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    user_preferences_implemented: Literal[True] = True
    preference_signal_metadata_implemented: Literal[True] = True
    long_term_memory_source_used: Literal[True] = True
    preference_storage_write_implemented: Literal[False] = False
    preference_record_creation_implemented: Literal[False] = False
    preference_record_update_implemented: Literal[False] = False
    preference_record_deletion_implemented: Literal[False] = False
    automatic_preference_learning_implemented: Literal[False] = False
    preference_mutation_implemented: Literal[False] = False
    personalization_application_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_storage_write_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
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
    def _plan_matches_preferences(self) -> Self:
        derived_preference_ids = tuple(
            preference.preference_id for preference in self.preferences
        )
        if len(set(derived_preference_ids)) != len(derived_preference_ids):
            raise ValueError("preference_ids must be unique")
        if self.preference_ids != derived_preference_ids:
            raise ValueError("preference_ids must match preferences")
        if self.candidate_preference_ids != _preference_ids_for_status(
            self.preferences,
            "candidate",
        ):
            raise ValueError("candidate_preference_ids must match preferences")
        if self.review_required_preference_ids != _preference_ids_for_status(
            self.preferences,
            "review_required",
        ):
            raise ValueError("review_required_preference_ids must match preferences")
        if self.guarded_preference_ids != _preference_ids_for_status(
            self.preferences,
            "guarded",
        ):
            raise ValueError("guarded_preference_ids must match preferences")
        if self.high_confidence_preference_ids != _preference_ids_for_confidence(
            self.preferences,
            "high",
            "guarded",
        ):
            raise ValueError("high_confidence_preference_ids must match preferences")
        if self.hitl_required_preference_ids != tuple(
            preference.preference_id
            for preference in self.preferences
            if preference.hitl_required_before_mutation
        ):
            raise ValueError("hitl_required_preference_ids must match preferences")
        if self.learned_preference_ids:
            raise ValueError("learned_preference_ids must remain empty")
        if self.mutated_preference_ids:
            raise ValueError("mutated_preference_ids must remain empty")
        if self.personalized_preference_ids:
            raise ValueError("personalized_preference_ids must remain empty")
        if self.preference_count != len(self.preferences):
            raise ValueError("preference_count must match preferences")
        if self.candidate_preference_count != len(self.candidate_preference_ids):
            raise ValueError("candidate_preference_count must match preferences")
        if self.review_required_preference_count != len(
            self.review_required_preference_ids
        ):
            raise ValueError("review_required_preference_count must match preferences")
        if self.guarded_preference_count != len(self.guarded_preference_ids):
            raise ValueError("guarded_preference_count must match preferences")
        if self.high_confidence_preference_count != len(
            self.high_confidence_preference_ids
        ):
            raise ValueError("high_confidence_preference_count must match preferences")
        if self.hitl_required_preference_count != len(
            self.hitl_required_preference_ids
        ):
            raise ValueError("hitl_required_preference_count must match preferences")
        if self.highest_preference_governance_score != max(
            preference.preference_governance_score
            for preference in self.preferences
        ):
            raise ValueError("highest_preference_governance_score must match")
        if self.overall_preference_governance_score != (
            _overall_preference_governance_score(self.preferences)
        ):
            raise ValueError("overall_preference_governance_score must match")
        if self.overall_preference_posture != _overall_preference_posture(
            self.preferences
        ):
            raise ValueError("overall_preference_posture must match preferences")
        for preference in self.preferences:
            if preference.route_name != self.route_name:
                raise ValueError("preference route_name must match plan")
            if (
                preference.source_long_term_memory_record_id
                not in self.source_long_term_memory_record_ids
            ):
                raise ValueError("source long-term memory record must be declared")
        return self


def build_user_preferences(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    long_term_memory: LongTermCreativeMemoryPlan | None = None,
) -> UserPreferencesPlan:
    """Build user preference metadata without learning or mutation."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    memory_plan = long_term_memory or build_long_term_creative_memory(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    preferences = _preferences(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        long_term_memory=memory_plan,
    )
    return UserPreferencesPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        source_long_term_memory_serialization_version=(
            memory_plan.serialization_version
        ),
        source_long_term_memory_record_ids=memory_plan.record_ids,
        execution_mode_ids=execution_modes.execution_mode_ids,
        preferences=preferences,
        preference_ids=tuple(preference.preference_id for preference in preferences),
        candidate_preference_ids=_preference_ids_for_status(
            preferences,
            "candidate",
        ),
        review_required_preference_ids=_preference_ids_for_status(
            preferences,
            "review_required",
        ),
        guarded_preference_ids=_preference_ids_for_status(preferences, "guarded"),
        high_confidence_preference_ids=_preference_ids_for_confidence(
            preferences,
            "high",
            "guarded",
        ),
        hitl_required_preference_ids=tuple(
            preference.preference_id
            for preference in preferences
            if preference.hitl_required_before_mutation
        ),
        learned_preference_ids=(),
        mutated_preference_ids=(),
        personalized_preference_ids=(),
        preference_count=len(preferences),
        candidate_preference_count=len(
            _preference_ids_for_status(preferences, "candidate")
        ),
        review_required_preference_count=len(
            _preference_ids_for_status(preferences, "review_required")
        ),
        guarded_preference_count=len(
            _preference_ids_for_status(preferences, "guarded")
        ),
        high_confidence_preference_count=len(
            _preference_ids_for_confidence(preferences, "high", "guarded")
        ),
        hitl_required_preference_count=sum(
            1 for preference in preferences if preference.hitl_required_before_mutation
        ),
        highest_preference_governance_score=max(
            preference.preference_governance_score for preference in preferences
        ),
        overall_preference_governance_score=_overall_preference_governance_score(
            preferences
        ),
        overall_preference_posture=_overall_preference_posture(preferences),
        advisory_actions=_plan_actions(preferences),
    )


def user_preference_by_id(
    preference_id: str,
    plan: UserPreferencesPlan | None = None,
) -> UserPreferenceSignal | None:
    """Return one user preference signal without preference retrieval."""

    source_plan = plan or build_user_preferences()
    for preference in source_plan.preferences:
        if preference.preference_id == preference_id:
            return preference
    return None


def user_preferences_for_status(
    status: UserPreferenceStatus,
    plan: UserPreferencesPlan | None = None,
) -> tuple[UserPreferenceSignal, ...]:
    """Return user preferences by advisory status."""

    source_plan = plan or build_user_preferences()
    return tuple(
        preference
        for preference in source_plan.preferences
        if preference.status == status
    )


def user_preferences_for_confidence(
    confidence: UserPreferenceConfidence,
    plan: UserPreferencesPlan | None = None,
) -> tuple[UserPreferenceSignal, ...]:
    """Return user preferences by confidence band."""

    source_plan = plan or build_user_preferences()
    return tuple(
        preference
        for preference in source_plan.preferences
        if preference.confidence == confidence
    )


def _preferences(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    long_term_memory: LongTermCreativeMemoryPlan,
) -> tuple[UserPreferenceSignal, ...]:
    return (
        _preference(
            kind="visual_style_preference",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            preference_scope="style",
            source_record_id="long_term_creative_memory::style_pattern_memory",
            long_term_memory=long_term_memory,
            explicitness_score=86,
            consistency_score=78,
            conflict_risk_score=52,
            sensitivity_weight=160,
        ),
        _preference(
            kind="interaction_preference",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            preference_scope="interaction",
            source_record_id="long_term_creative_memory::creative_intent_memory",
            long_term_memory=long_term_memory,
            explicitness_score=74,
            consistency_score=72,
            conflict_risk_score=40,
            sensitivity_weight=130,
        ),
        _preference(
            kind="constraint_preference",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            preference_scope="constraints",
            source_record_id="long_term_creative_memory::project_context_memory",
            long_term_memory=long_term_memory,
            explicitness_score=82,
            consistency_score=76,
            conflict_risk_score=58,
            sensitivity_weight=150,
        ),
        _preference(
            kind="technical_stack_preference",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            preference_scope="technical_stack",
            source_record_id="long_term_creative_memory::artifact_lineage_memory",
            long_term_memory=long_term_memory,
            explicitness_score=64,
            consistency_score=70,
            conflict_risk_score=34,
            sensitivity_weight=110,
        ),
        _preference(
            kind="review_depth_preference",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            preference_scope="review_depth",
            source_record_id="long_term_creative_memory::preference_signal_memory",
            long_term_memory=long_term_memory,
            explicitness_score=48,
            consistency_score=56,
            conflict_risk_score=20,
            sensitivity_weight=80,
        ),
    )


def _preference(
    *,
    kind: UserPreferenceKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    preference_scope: UserPreferenceScope,
    source_record_id: str,
    long_term_memory: LongTermCreativeMemoryPlan,
    explicitness_score: int,
    consistency_score: int,
    conflict_risk_score: int,
    sensitivity_weight: int,
) -> UserPreferenceSignal:
    source_record = long_term_creative_memory_record_by_id(
        source_record_id,
        long_term_memory,
    )
    if source_record is None:
        raise ValueError("source long-term memory record must exist")
    governance_score = _preference_governance_score(
        explicitness_score=explicitness_score,
        consistency_score=consistency_score,
        conflict_risk_score=conflict_risk_score,
        sensitivity_weight=sensitivity_weight,
    )
    status = _preference_status(governance_score)
    confidence = _preference_confidence(governance_score)
    return UserPreferenceSignal(
        preference_id=f"user_preferences::{kind}",
        preference_kind=kind,
        status=status,
        confidence=confidence,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        preference_scope=preference_scope,
        source_long_term_memory_record_id=source_record.record_id,
        preference_statement=_preference_statement(kind),
        explicitness_score=explicitness_score,
        consistency_score=consistency_score,
        conflict_risk_score=conflict_risk_score,
        sensitivity_weight=sensitivity_weight,
        preference_governance_score=governance_score,
        hitl_required_before_mutation=True,
        applicable_context_tags=_context_tags(kind, preference_scope),
        explainability_notes=_explainability_notes(kind, source_record.record_id),
        advisory_actions=_preference_actions(kind),
        evidence=(
            f"source_long_term_memory:{source_record.record_id}",
            f"source_memory_status:{source_record.status}",
            f"source_memory_sensitivity:{source_record.sensitivity}",
            f"preference_scope:{preference_scope}",
            f"explicitness_score:{explicitness_score}",
            f"consistency_score:{consistency_score}",
            f"conflict_risk_score:{conflict_risk_score}",
            "hitl_required_before_mutation:true",
        ),
    )


def _preference_governance_score(
    *,
    explicitness_score: int,
    consistency_score: int,
    conflict_risk_score: int,
    sensitivity_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            explicitness_score * 3
            + consistency_score * 3
            + conflict_risk_score * 4
            + sensitivity_weight,
        ),
    )


def _preference_status(score: int) -> UserPreferenceStatus:
    if score >= 760:
        return "guarded"
    if score >= 600:
        return "review_required"
    return "candidate"


def _preference_confidence(score: int) -> UserPreferenceConfidence:
    if score >= 760:
        return "guarded"
    if score >= 680:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_preference_governance_score(
    preferences: tuple[UserPreferenceSignal, ...],
) -> int:
    base = sum(
        preference.preference_governance_score for preference in preferences
    ) // len(preferences)
    guarded_count = len(_preference_ids_for_status(preferences, "guarded"))
    review_count = len(_preference_ids_for_status(preferences, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_preference_posture(
    preferences: tuple[UserPreferenceSignal, ...],
) -> UserPreferencePosture:
    if any(preference.status == "guarded" for preference in preferences):
        return "guarded"
    if any(preference.status == "review_required" for preference in preferences):
        return "review_required"
    return "candidate"


def _preference_ids_for_status(
    preferences: tuple[UserPreferenceSignal, ...],
    status: UserPreferenceStatus,
) -> tuple[str, ...]:
    return tuple(
        preference.preference_id
        for preference in preferences
        if preference.status == status
    )


def _preference_ids_for_confidence(
    preferences: tuple[UserPreferenceSignal, ...],
    *confidences: UserPreferenceConfidence,
) -> tuple[str, ...]:
    return tuple(
        preference.preference_id
        for preference in preferences
        if preference.confidence in confidences
    )


def _plan_actions(
    preferences: tuple[UserPreferenceSignal, ...],
) -> tuple[str, ...]:
    guarded_preference_count = len(
        _preference_ids_for_status(preferences, "guarded")
    )
    return (
        "inspect_user_preference_candidates",
        "require_hitl_before_preference_mutation",
        "keep_preference_learning_and_personalization_non_executing",
        f"review_guarded_preference_count:{guarded_preference_count}",
    )


def _preference_statement(kind: UserPreferenceKind) -> str:
    statements = {
        "visual_style_preference": (
            "User preference posture favors recurring visual style continuity."
        ),
        "interaction_preference": (
            "User preference posture favors interaction expectations and cadence."
        ),
        "constraint_preference": (
            "User preference posture favors project constraints before novelty."
        ),
        "technical_stack_preference": (
            "User preference posture favors known technical-stack continuity."
        ),
        "review_depth_preference": (
            "User preference posture favors calibrated review depth."
        ),
    }
    return statements[kind]


def _context_tags(
    kind: UserPreferenceKind,
    preference_scope: UserPreferenceScope,
) -> tuple[str, ...]:
    return (
        "creative_memory",
        "user_preferences",
        preference_scope,
        kind.removesuffix("_preference"),
    )


def _explainability_notes(
    kind: UserPreferenceKind,
    source_record_id: str,
) -> tuple[str, ...]:
    return (
        f"preference_kind:{kind}",
        f"source_record:{source_record_id}",
        "score_inputs:explicitness,consistency,conflict_risk,sensitivity",
        "mutation_boundary:HITL_required_before_preference_change",
        "personalization_boundary:metadata_only_no_application",
    )


def _preference_actions(kind: UserPreferenceKind) -> tuple[str, ...]:
    return (
        f"review_{kind}",
        "inspect_source_memory_before_preference_change",
        "preserve_no_automatic_personalization_boundary",
    )


def _resolve_route(route: RouteName | str) -> RouteName:
    return route if isinstance(route, RouteName) else RouteName(str(route).strip())


def _resolve_task_type(task_type: TaskRoutingType | str) -> TaskRoutingType:
    normalized = str(task_type).strip()
    if normalized not in get_args(TaskRoutingType):
        raise ValueError("task_type must be a known routing task type")
    return cast(TaskRoutingType, normalized)


def _resolve_execution_mode(
    execution_mode_id: ExecutionModeId | str,
    allowed_execution_mode_ids: tuple[ExecutionModeId, ...],
) -> ExecutionModeId:
    normalized = str(execution_mode_id).strip()
    if normalized not in allowed_execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")
    return cast(ExecutionModeId, normalized)
