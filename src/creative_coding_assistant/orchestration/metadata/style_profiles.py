"""V6.2 advisory style profile metadata."""

from __future__ import annotations

from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)
from creative_coding_assistant.orchestration.user_preferences import (
    UserPreferencesPlan,
    build_user_preferences,
    user_preference_by_id,
)

StyleProfileKind = Literal[
    "palette_profile",
    "motion_profile",
    "composition_profile",
    "typography_profile",
    "material_profile",
]
StyleProfileStatus = Literal["candidate", "review_required", "guarded"]
StyleProfileFidelity = Literal["low", "medium", "high", "guarded"]
StyleProfilePosture = Literal["candidate", "review_required", "guarded"]
StyleAxis = Literal["color", "motion", "composition", "typography", "material"]

STYLE_PROFILE_SERIALIZATION_VERSION = "style_profile.v1"
STYLE_PROFILE_PLAN_SERIALIZATION_VERSION = "style_profile_plan.v1"
STYLE_PROFILE_AUTHORITY_BOUNDARY = (
    "V6.2 style profiles model creative style continuity as inspectable "
    "advisory metadata only; they do not write style storage, create style "
    "profiles, update style profiles, learn styles automatically, apply style "
    "profiles to prompts or generated output, execute memory retrieval, mutate "
    "preferences, apply personalization, change provider or model routing, "
    "execute providers, invoke agents, control workflows, mutate workflow "
    "graphs, trigger retries or refinements, mutate prompts, modify generated "
    "output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "style_profile_storage_write",
    "style_profile_creation",
    "style_profile_update",
    "style_profile_deletion",
    "automatic_style_learning",
    "style_profile_application",
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


class StyleProfile(BaseModel):
    """One advisory style profile candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1, max_length=180)
    profile_kind: StyleProfileKind
    status: StyleProfileStatus
    fidelity: StyleProfileFidelity
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    style_axis: StyleAxis
    source_user_preference_id: str = Field(min_length=1, max_length=180)
    profile_summary: str = Field(min_length=1, max_length=340)
    preference_alignment_score: int = Field(ge=0, le=100)
    evidence_strength_score: int = Field(ge=0, le=100)
    conflict_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    style_profile_score: int = Field(ge=0, le=1_000)
    hitl_required_before_application: bool
    style_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    style_profiles_implemented: Literal[True] = True
    style_profile_metadata_implemented: Literal[True] = True
    user_preferences_source_used: Literal[True] = True
    style_profile_storage_write_implemented: Literal[False] = False
    style_profile_creation_implemented: Literal[False] = False
    style_profile_update_implemented: Literal[False] = False
    style_profile_deletion_implemented: Literal[False] = False
    automatic_style_learning_implemented: Literal[False] = False
    style_profile_application_implemented: Literal[False] = False
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
    serialization_version: Literal["style_profile.v1"] = (
        STYLE_PROFILE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _profile_matches_contract(self) -> Self:
        if self.profile_id != f"style_profiles::{self.profile_kind}":
            raise ValueError("profile_id must match profile_kind")
        if self.style_profile_score != _style_profile_score(
            preference_alignment_score=self.preference_alignment_score,
            evidence_strength_score=self.evidence_strength_score,
            conflict_risk_score=self.conflict_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("style_profile_score must combine source scores")
        if self.status != _style_profile_status(self.style_profile_score):
            raise ValueError("status must match style_profile_score")
        if self.fidelity != _style_profile_fidelity(self.style_profile_score):
            raise ValueError("fidelity must match style_profile_score")
        if not self.hitl_required_before_application:
            raise ValueError("style profile application requires HITL posture")
        return self


class StyleProfilePlan(BaseModel):
    """Bounded V6.2 advisory style profile plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["style_profiles"] = "style_profiles"
    serialization_version: Literal["style_profile_plan.v1"] = (
        STYLE_PROFILE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=STYLE_PROFILE_AUTHORITY_BOUNDARY,
        max_length=1800,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_user_preferences_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_user_preference_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    profiles: tuple[StyleProfile, ...] = Field(min_length=5, max_length=5)
    profile_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    candidate_profile_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    review_required_profile_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_profile_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    high_fidelity_profile_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    hitl_required_profile_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    created_profile_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    updated_profile_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    applied_profile_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    profile_count: int = Field(ge=5, le=5)
    candidate_profile_count: int = Field(ge=0, le=5)
    review_required_profile_count: int = Field(ge=0, le=5)
    guarded_profile_count: int = Field(ge=0, le=5)
    high_fidelity_profile_count: int = Field(ge=0, le=5)
    hitl_required_profile_count: int = Field(ge=0, le=5)
    highest_style_profile_score: int = Field(ge=0, le=1_000)
    overall_style_profile_score: int = Field(ge=0, le=1_000)
    overall_style_profile_posture: StyleProfilePosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    style_profiles_implemented: Literal[True] = True
    style_profile_metadata_implemented: Literal[True] = True
    user_preferences_source_used: Literal[True] = True
    style_profile_storage_write_implemented: Literal[False] = False
    style_profile_creation_implemented: Literal[False] = False
    style_profile_update_implemented: Literal[False] = False
    style_profile_deletion_implemented: Literal[False] = False
    automatic_style_learning_implemented: Literal[False] = False
    style_profile_application_implemented: Literal[False] = False
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
    def _plan_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(profile.profile_id for profile in self.profiles)
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("profile_ids must be unique")
        if self.profile_ids != derived_profile_ids:
            raise ValueError("profile_ids must match profiles")
        if self.candidate_profile_ids != _profile_ids_for_status(
            self.profiles,
            "candidate",
        ):
            raise ValueError("candidate_profile_ids must match profiles")
        if self.review_required_profile_ids != _profile_ids_for_status(
            self.profiles,
            "review_required",
        ):
            raise ValueError("review_required_profile_ids must match profiles")
        if self.guarded_profile_ids != _profile_ids_for_status(
            self.profiles,
            "guarded",
        ):
            raise ValueError("guarded_profile_ids must match profiles")
        if self.high_fidelity_profile_ids != _profile_ids_for_fidelity(
            self.profiles,
            "high",
            "guarded",
        ):
            raise ValueError("high_fidelity_profile_ids must match profiles")
        if self.hitl_required_profile_ids != tuple(
            profile.profile_id
            for profile in self.profiles
            if profile.hitl_required_before_application
        ):
            raise ValueError("hitl_required_profile_ids must match profiles")
        if self.created_profile_ids:
            raise ValueError("created_profile_ids must remain empty")
        if self.updated_profile_ids:
            raise ValueError("updated_profile_ids must remain empty")
        if self.applied_profile_ids:
            raise ValueError("applied_profile_ids must remain empty")
        if self.profile_count != len(self.profiles):
            raise ValueError("profile_count must match profiles")
        if self.candidate_profile_count != len(self.candidate_profile_ids):
            raise ValueError("candidate_profile_count must match profiles")
        if self.review_required_profile_count != len(self.review_required_profile_ids):
            raise ValueError("review_required_profile_count must match profiles")
        if self.guarded_profile_count != len(self.guarded_profile_ids):
            raise ValueError("guarded_profile_count must match profiles")
        if self.high_fidelity_profile_count != len(self.high_fidelity_profile_ids):
            raise ValueError("high_fidelity_profile_count must match profiles")
        if self.hitl_required_profile_count != len(self.hitl_required_profile_ids):
            raise ValueError("hitl_required_profile_count must match profiles")
        if self.highest_style_profile_score != max(
            profile.style_profile_score for profile in self.profiles
        ):
            raise ValueError("highest_style_profile_score must match profiles")
        if self.overall_style_profile_score != _overall_style_profile_score(
            self.profiles
        ):
            raise ValueError("overall_style_profile_score must match profiles")
        if self.overall_style_profile_posture != _overall_style_profile_posture(
            self.profiles
        ):
            raise ValueError("overall_style_profile_posture must match profiles")
        for profile in self.profiles:
            if profile.route_name != self.route_name:
                raise ValueError("profile route_name must match plan")
            if profile.source_user_preference_id not in self.source_user_preference_ids:
                raise ValueError("source user preference must be declared")
        return self


def build_style_profiles(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    user_preferences: UserPreferencesPlan | None = None,
) -> StyleProfilePlan:
    """Build style profile metadata without applying style profiles."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    preference_plan = user_preferences or build_user_preferences(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    profiles = _profiles(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        user_preferences=preference_plan,
    )
    return StyleProfilePlan(
        route_name=route_name,
        task_type=normalized_task_type,
        source_user_preferences_serialization_version=(
            preference_plan.serialization_version
        ),
        source_user_preference_ids=preference_plan.preference_ids,
        execution_mode_ids=execution_modes.execution_mode_ids,
        profiles=profiles,
        profile_ids=tuple(profile.profile_id for profile in profiles),
        candidate_profile_ids=_profile_ids_for_status(profiles, "candidate"),
        review_required_profile_ids=_profile_ids_for_status(
            profiles,
            "review_required",
        ),
        guarded_profile_ids=_profile_ids_for_status(profiles, "guarded"),
        high_fidelity_profile_ids=_profile_ids_for_fidelity(
            profiles,
            "high",
            "guarded",
        ),
        hitl_required_profile_ids=tuple(
            profile.profile_id
            for profile in profiles
            if profile.hitl_required_before_application
        ),
        created_profile_ids=(),
        updated_profile_ids=(),
        applied_profile_ids=(),
        profile_count=len(profiles),
        candidate_profile_count=len(_profile_ids_for_status(profiles, "candidate")),
        review_required_profile_count=len(
            _profile_ids_for_status(profiles, "review_required")
        ),
        guarded_profile_count=len(_profile_ids_for_status(profiles, "guarded")),
        high_fidelity_profile_count=len(
            _profile_ids_for_fidelity(profiles, "high", "guarded")
        ),
        hitl_required_profile_count=sum(
            1 for profile in profiles if profile.hitl_required_before_application
        ),
        highest_style_profile_score=max(
            profile.style_profile_score for profile in profiles
        ),
        overall_style_profile_score=_overall_style_profile_score(profiles),
        overall_style_profile_posture=_overall_style_profile_posture(profiles),
        advisory_actions=_plan_actions(profiles),
    )


def style_profile_by_id(
    profile_id: str,
    plan: StyleProfilePlan | None = None,
) -> StyleProfile | None:
    """Return one style profile without style application."""

    source_plan = plan or build_style_profiles()
    for profile in source_plan.profiles:
        if profile.profile_id == profile_id:
            return profile
    return None


def style_profiles_for_status(
    status: StyleProfileStatus,
    plan: StyleProfilePlan | None = None,
) -> tuple[StyleProfile, ...]:
    """Return style profiles by advisory status."""

    source_plan = plan or build_style_profiles()
    return tuple(
        profile for profile in source_plan.profiles if profile.status == status
    )


def style_profiles_for_fidelity(
    fidelity: StyleProfileFidelity,
    plan: StyleProfilePlan | None = None,
) -> tuple[StyleProfile, ...]:
    """Return style profiles by fidelity band."""

    source_plan = plan or build_style_profiles()
    return tuple(
        profile for profile in source_plan.profiles if profile.fidelity == fidelity
    )


def _profiles(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    user_preferences: UserPreferencesPlan,
) -> tuple[StyleProfile, ...]:
    return (
        _profile(
            kind="palette_profile",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            style_axis="color",
            source_preference_id="user_preferences::visual_style_preference",
            user_preferences=user_preferences,
            preference_alignment_score=86,
            evidence_strength_score=84,
            conflict_risk_score=48,
            governance_weight=150,
        ),
        _profile(
            kind="motion_profile",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            style_axis="motion",
            source_preference_id="user_preferences::interaction_preference",
            user_preferences=user_preferences,
            preference_alignment_score=72,
            evidence_strength_score=70,
            conflict_risk_score=42,
            governance_weight=130,
        ),
        _profile(
            kind="composition_profile",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            style_axis="composition",
            source_preference_id="user_preferences::constraint_preference",
            user_preferences=user_preferences,
            preference_alignment_score=78,
            evidence_strength_score=76,
            conflict_risk_score=50,
            governance_weight=145,
        ),
        _profile(
            kind="typography_profile",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            style_axis="typography",
            source_preference_id="user_preferences::review_depth_preference",
            user_preferences=user_preferences,
            preference_alignment_score=60,
            evidence_strength_score=66,
            conflict_risk_score=28,
            governance_weight=105,
        ),
        _profile(
            kind="material_profile",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            style_axis="material",
            source_preference_id="user_preferences::technical_stack_preference",
            user_preferences=user_preferences,
            preference_alignment_score=64,
            evidence_strength_score=68,
            conflict_risk_score=36,
            governance_weight=115,
        ),
    )


def _profile(
    *,
    kind: StyleProfileKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    style_axis: StyleAxis,
    source_preference_id: str,
    user_preferences: UserPreferencesPlan,
    preference_alignment_score: int,
    evidence_strength_score: int,
    conflict_risk_score: int,
    governance_weight: int,
) -> StyleProfile:
    source_preference = user_preference_by_id(source_preference_id, user_preferences)
    if source_preference is None:
        raise ValueError("source user preference must exist")
    profile_score = _style_profile_score(
        preference_alignment_score=preference_alignment_score,
        evidence_strength_score=evidence_strength_score,
        conflict_risk_score=conflict_risk_score,
        governance_weight=governance_weight,
    )
    status = _style_profile_status(profile_score)
    fidelity = _style_profile_fidelity(profile_score)
    return StyleProfile(
        profile_id=f"style_profiles::{kind}",
        profile_kind=kind,
        status=status,
        fidelity=fidelity,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        style_axis=style_axis,
        source_user_preference_id=source_preference.preference_id,
        profile_summary=_profile_summary(kind),
        preference_alignment_score=preference_alignment_score,
        evidence_strength_score=evidence_strength_score,
        conflict_risk_score=conflict_risk_score,
        governance_weight=governance_weight,
        style_profile_score=profile_score,
        hitl_required_before_application=True,
        style_tags=_style_tags(kind, style_axis),
        explainability_notes=_explainability_notes(
            kind,
            source_preference.preference_id,
        ),
        advisory_actions=_profile_actions(kind),
        evidence=(
            f"source_user_preference:{source_preference.preference_id}",
            f"source_preference_status:{source_preference.status}",
            f"source_preference_confidence:{source_preference.confidence}",
            f"style_axis:{style_axis}",
            f"preference_alignment_score:{preference_alignment_score}",
            f"evidence_strength_score:{evidence_strength_score}",
            f"conflict_risk_score:{conflict_risk_score}",
            "hitl_required_before_application:true",
        ),
    )


def _style_profile_score(
    *,
    preference_alignment_score: int,
    evidence_strength_score: int,
    conflict_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            preference_alignment_score * 3
            + evidence_strength_score * 3
            + conflict_risk_score * 4
            + governance_weight,
        ),
    )


def _style_profile_status(score: int) -> StyleProfileStatus:
    if score >= 760:
        return "guarded"
    if score >= 600:
        return "review_required"
    return "candidate"


def _style_profile_fidelity(score: int) -> StyleProfileFidelity:
    if score >= 760:
        return "guarded"
    if score >= 680:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_style_profile_score(
    profiles: tuple[StyleProfile, ...],
) -> int:
    base = sum(profile.style_profile_score for profile in profiles) // len(profiles)
    guarded_count = len(_profile_ids_for_status(profiles, "guarded"))
    review_count = len(_profile_ids_for_status(profiles, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_style_profile_posture(
    profiles: tuple[StyleProfile, ...],
) -> StyleProfilePosture:
    if any(profile.status == "guarded" for profile in profiles):
        return "guarded"
    if any(profile.status == "review_required" for profile in profiles):
        return "review_required"
    return "candidate"


def _profile_ids_for_status(
    profiles: tuple[StyleProfile, ...],
    status: StyleProfileStatus,
) -> tuple[str, ...]:
    return tuple(profile.profile_id for profile in profiles if profile.status == status)


def _profile_ids_for_fidelity(
    profiles: tuple[StyleProfile, ...],
    *fidelities: StyleProfileFidelity,
) -> tuple[str, ...]:
    return tuple(
        profile.profile_id for profile in profiles if profile.fidelity in fidelities
    )


def _plan_actions(profiles: tuple[StyleProfile, ...]) -> tuple[str, ...]:
    guarded_profile_count = len(_profile_ids_for_status(profiles, "guarded"))
    return (
        "inspect_style_profile_candidates",
        "require_hitl_before_style_profile_application",
        "keep_style_learning_and_generated_output_mutation_non_executing",
        f"review_guarded_profile_count:{guarded_profile_count}",
    )


def _profile_summary(kind: StyleProfileKind) -> str:
    summaries = {
        "palette_profile": "Models palette continuity and color tendency metadata.",
        "motion_profile": "Models motion cadence and interaction rhythm metadata.",
        "composition_profile": "Models layout and composition tendency metadata.",
        "typography_profile": "Models text density and typography posture metadata.",
        "material_profile": "Models texture, medium, and material posture metadata.",
    }
    return summaries[kind]


def _style_tags(kind: StyleProfileKind, style_axis: StyleAxis) -> tuple[str, ...]:
    return (
        "creative_memory",
        "style_profiles",
        style_axis,
        kind.removesuffix("_profile"),
    )


def _explainability_notes(
    kind: StyleProfileKind,
    source_preference_id: str,
) -> tuple[str, ...]:
    return (
        f"profile_kind:{kind}",
        f"source_preference:{source_preference_id}",
        "score_inputs:preference_alignment,evidence_strength,conflict_risk,governance",
        "application_boundary:HITL_required_before_style_application",
        "output_boundary:metadata_only_no_generated_output_mutation",
    )


def _profile_actions(kind: StyleProfileKind) -> tuple[str, ...]:
    return (
        f"review_{kind}",
        "inspect_source_preference_before_style_application",
        "preserve_no_generated_output_mutation_boundary",
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
