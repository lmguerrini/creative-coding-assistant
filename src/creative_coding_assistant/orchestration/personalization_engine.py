"""V6.2 advisory personalization engine metadata."""

from __future__ import annotations

from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.creative_dna import (
    CreativeDNAPlan,
    build_creative_dna,
    creative_dna_signature_by_id,
)
from creative_coding_assistant.orchestration.project_memory import (
    ProjectMemoryPlan,
    build_project_memory,
    project_memory_signal_by_id,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)
from creative_coding_assistant.orchestration.style_profiles import (
    StyleProfilePlan,
    build_style_profiles,
    style_profile_by_id,
)
from creative_coding_assistant.orchestration.user_preferences import (
    UserPreferencesPlan,
    build_user_preferences,
    user_preference_by_id,
)

PersonalizationRecommendationKind = Literal[
    "style_personalization",
    "interaction_personalization",
    "constraint_personalization",
    "technical_personalization",
    "review_depth_personalization",
]
PersonalizationStatus = Literal["candidate", "review_required", "guarded"]
PersonalizationConfidence = Literal["low", "medium", "high", "guarded"]
PersonalizationPosture = Literal["candidate", "review_required", "guarded"]
PersonalizationScope = Literal[
    "style",
    "interaction",
    "constraints",
    "technical_stack",
    "review_depth",
]

PERSONALIZATION_RECOMMENDATION_SERIALIZATION_VERSION = (
    "personalization_recommendation.v1"
)
PERSONALIZATION_ENGINE_PLAN_SERIALIZATION_VERSION = (
    "personalization_engine_plan.v1"
)
PERSONALIZATION_ENGINE_AUTHORITY_BOUNDARY = (
    "V6.2 Personalization Engine models governed personalization posture as "
    "inspectable advisory metadata only; it does not write personalization "
    "storage, create personalization rules, update personalization rules, "
    "delete personalization rules, learn personalization automatically, apply "
    "personalization to prompts or generated output, apply Creative DNA, apply "
    "style profiles, execute memory retrieval, write memory storage, write "
    "project memory storage, mutate preferences, change provider or model "
    "routing, execute providers, invoke agents, control workflows, mutate "
    "workflow graphs, trigger retries or refinements, mutate prompts, modify "
    "generated output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "personalization_storage_write",
    "personalization_rule_creation",
    "personalization_rule_update",
    "personalization_rule_deletion",
    "automatic_personalization_learning",
    "automatic_personalization_application",
    "creative_dna_application",
    "style_profile_application",
    "automatic_preference_mutation",
    "memory_retrieval_execution",
    "memory_storage_write",
    "project_memory_storage_write",
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


class PersonalizationRecommendation(BaseModel):
    """One advisory personalization recommendation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    personalization_id: str = Field(min_length=1, max_length=180)
    recommendation_kind: PersonalizationRecommendationKind
    status: PersonalizationStatus
    confidence: PersonalizationConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    personalization_scope: PersonalizationScope
    source_creative_dna_id: str = Field(min_length=1, max_length=180)
    source_user_preference_id: str = Field(min_length=1, max_length=180)
    source_style_profile_id: str = Field(min_length=1, max_length=180)
    source_project_memory_signal_id: str = Field(min_length=1, max_length=180)
    personalization_summary: str = Field(min_length=1, max_length=360)
    preference_alignment_score: int = Field(ge=0, le=100)
    creative_dna_alignment_score: int = Field(ge=0, le=100)
    project_fit_score: int = Field(ge=0, le=100)
    safety_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    personalization_score: int = Field(ge=0, le=1_000)
    hitl_required_before_application: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    personalization_engine_implemented: Literal[True] = True
    personalization_metadata_implemented: Literal[True] = True
    creative_dna_source_used: Literal[True] = True
    user_preferences_source_used: Literal[True] = True
    style_profile_source_used: Literal[True] = True
    project_memory_source_used: Literal[True] = True
    personalization_storage_write_implemented: Literal[False] = False
    personalization_rule_creation_implemented: Literal[False] = False
    personalization_rule_update_implemented: Literal[False] = False
    personalization_rule_deletion_implemented: Literal[False] = False
    automatic_personalization_learning_implemented: Literal[False] = False
    personalization_application_implemented: Literal[False] = False
    creative_dna_application_implemented: Literal[False] = False
    style_profile_application_implemented: Literal[False] = False
    preference_mutation_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_storage_write_implemented: Literal[False] = False
    project_memory_storage_write_implemented: Literal[False] = False
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
    serialization_version: Literal["personalization_recommendation.v1"] = (
        PERSONALIZATION_RECOMMENDATION_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _recommendation_matches_contract(self) -> Self:
        expected_id = f"personalization_engine::{self.recommendation_kind}"
        if self.personalization_id != expected_id:
            raise ValueError("personalization_id must match recommendation_kind")
        if self.personalization_score != _personalization_score(
            preference_alignment_score=self.preference_alignment_score,
            creative_dna_alignment_score=self.creative_dna_alignment_score,
            project_fit_score=self.project_fit_score,
            safety_risk_score=self.safety_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("personalization_score must combine source scores")
        if self.status != _personalization_status(self.personalization_score):
            raise ValueError("status must match personalization_score")
        if self.confidence != _personalization_confidence(
            self.personalization_score
        ):
            raise ValueError("confidence must match personalization_score")
        if not self.hitl_required_before_application:
            raise ValueError("personalization application requires HITL posture")
        return self


class PersonalizationEnginePlan(BaseModel):
    """Bounded V6.2 advisory personalization engine plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["personalization_engine"] = "personalization_engine"
    serialization_version: Literal["personalization_engine_plan.v1"] = (
        PERSONALIZATION_ENGINE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PERSONALIZATION_ENGINE_AUTHORITY_BOUNDARY,
        max_length=1900,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_creative_dna_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_user_preferences_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_style_profile_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_project_memory_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_creative_dna_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    source_user_preference_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    source_style_profile_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    source_project_memory_signal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    recommendations: tuple[PersonalizationRecommendation, ...] = Field(
        min_length=5,
        max_length=5,
    )
    recommendation_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    candidate_recommendation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    review_required_recommendation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_recommendation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    high_confidence_recommendation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    hitl_required_recommendation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    persisted_personalization_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    learned_personalization_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    applied_personalization_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_preference_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    recommendation_count: int = Field(ge=5, le=5)
    candidate_recommendation_count: int = Field(ge=0, le=5)
    review_required_recommendation_count: int = Field(ge=0, le=5)
    guarded_recommendation_count: int = Field(ge=0, le=5)
    high_confidence_recommendation_count: int = Field(ge=0, le=5)
    hitl_required_recommendation_count: int = Field(ge=0, le=5)
    highest_personalization_score: int = Field(ge=0, le=1_000)
    overall_personalization_score: int = Field(ge=0, le=1_000)
    overall_personalization_posture: PersonalizationPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    personalization_engine_implemented: Literal[True] = True
    personalization_metadata_implemented: Literal[True] = True
    creative_dna_source_used: Literal[True] = True
    user_preferences_source_used: Literal[True] = True
    style_profile_source_used: Literal[True] = True
    project_memory_source_used: Literal[True] = True
    personalization_storage_write_implemented: Literal[False] = False
    personalization_rule_creation_implemented: Literal[False] = False
    personalization_rule_update_implemented: Literal[False] = False
    personalization_rule_deletion_implemented: Literal[False] = False
    automatic_personalization_learning_implemented: Literal[False] = False
    personalization_application_implemented: Literal[False] = False
    creative_dna_application_implemented: Literal[False] = False
    style_profile_application_implemented: Literal[False] = False
    preference_mutation_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_storage_write_implemented: Literal[False] = False
    project_memory_storage_write_implemented: Literal[False] = False
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
    def _plan_matches_recommendations(self) -> Self:
        derived_recommendation_ids = tuple(
            recommendation.personalization_id
            for recommendation in self.recommendations
        )
        if len(set(derived_recommendation_ids)) != len(derived_recommendation_ids):
            raise ValueError("recommendation_ids must be unique")
        if self.recommendation_ids != derived_recommendation_ids:
            raise ValueError("recommendation_ids must match recommendations")
        if self.candidate_recommendation_ids != _recommendation_ids_for_status(
            self.recommendations,
            "candidate",
        ):
            raise ValueError("candidate_recommendation_ids must match")
        if self.review_required_recommendation_ids != _recommendation_ids_for_status(
            self.recommendations,
            "review_required",
        ):
            raise ValueError("review_required_recommendation_ids must match")
        if self.guarded_recommendation_ids != _recommendation_ids_for_status(
            self.recommendations,
            "guarded",
        ):
            raise ValueError("guarded_recommendation_ids must match")
        if self.high_confidence_recommendation_ids != (
            _recommendation_ids_for_confidence(
                self.recommendations,
                "high",
                "guarded",
            )
        ):
            raise ValueError("high_confidence_recommendation_ids must match")
        if self.hitl_required_recommendation_ids != tuple(
            recommendation.personalization_id
            for recommendation in self.recommendations
            if recommendation.hitl_required_before_application
        ):
            raise ValueError("hitl_required_recommendation_ids must match")
        if self.persisted_personalization_ids:
            raise ValueError("persisted_personalization_ids must remain empty")
        if self.learned_personalization_ids:
            raise ValueError("learned_personalization_ids must remain empty")
        if self.applied_personalization_ids:
            raise ValueError("applied_personalization_ids must remain empty")
        if self.mutated_preference_ids:
            raise ValueError("mutated_preference_ids must remain empty")
        if self.recommendation_count != len(self.recommendations):
            raise ValueError("recommendation_count must match recommendations")
        if self.candidate_recommendation_count != len(
            self.candidate_recommendation_ids
        ):
            raise ValueError("candidate_recommendation_count must match")
        if self.review_required_recommendation_count != len(
            self.review_required_recommendation_ids
        ):
            raise ValueError("review_required_recommendation_count must match")
        if self.guarded_recommendation_count != len(
            self.guarded_recommendation_ids
        ):
            raise ValueError("guarded_recommendation_count must match")
        if self.high_confidence_recommendation_count != len(
            self.high_confidence_recommendation_ids
        ):
            raise ValueError("high_confidence_recommendation_count must match")
        if self.hitl_required_recommendation_count != len(
            self.hitl_required_recommendation_ids
        ):
            raise ValueError("hitl_required_recommendation_count must match")
        if self.highest_personalization_score != max(
            recommendation.personalization_score
            for recommendation in self.recommendations
        ):
            raise ValueError("highest_personalization_score must match")
        if self.overall_personalization_score != _overall_personalization_score(
            self.recommendations
        ):
            raise ValueError("overall_personalization_score must match")
        if self.overall_personalization_posture != (
            _overall_personalization_posture(self.recommendations)
        ):
            raise ValueError("overall_personalization_posture must match")
        for recommendation in self.recommendations:
            if recommendation.route_name != self.route_name:
                raise ValueError("recommendation route_name must match plan")
            if (
                recommendation.source_creative_dna_id
                not in self.source_creative_dna_ids
            ):
                raise ValueError("source Creative DNA signature must be declared")
            if (
                recommendation.source_user_preference_id
                not in self.source_user_preference_ids
            ):
                raise ValueError("source user preference must be declared")
            if (
                recommendation.source_style_profile_id
                not in self.source_style_profile_ids
            ):
                raise ValueError("source style profile must be declared")
            if (
                recommendation.source_project_memory_signal_id
                not in self.source_project_memory_signal_ids
            ):
                raise ValueError("source project memory signal must be declared")
        return self


def build_personalization_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    creative_dna: CreativeDNAPlan | None = None,
    user_preferences: UserPreferencesPlan | None = None,
    style_profiles: StyleProfilePlan | None = None,
    project_memory: ProjectMemoryPlan | None = None,
) -> PersonalizationEnginePlan:
    """Build personalization metadata without applying personalization."""

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
    style_plan = style_profiles or build_style_profiles(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        user_preferences=preference_plan,
    )
    project_plan = project_memory or build_project_memory(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        style_profiles=style_plan,
    )
    dna_plan = creative_dna or build_creative_dna(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        user_preferences=preference_plan,
        style_profiles=style_plan,
        project_memory=project_plan,
    )
    recommendations = _recommendations(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        creative_dna=dna_plan,
        user_preferences=preference_plan,
        style_profiles=style_plan,
        project_memory=project_plan,
    )
    return PersonalizationEnginePlan(
        route_name=route_name,
        task_type=normalized_task_type,
        source_creative_dna_serialization_version=dna_plan.serialization_version,
        source_user_preferences_serialization_version=(
            preference_plan.serialization_version
        ),
        source_style_profile_serialization_version=style_plan.serialization_version,
        source_project_memory_serialization_version=project_plan.serialization_version,
        source_creative_dna_ids=dna_plan.signature_ids,
        source_user_preference_ids=preference_plan.preference_ids,
        source_style_profile_ids=style_plan.profile_ids,
        source_project_memory_signal_ids=project_plan.signal_ids,
        execution_mode_ids=execution_modes.execution_mode_ids,
        recommendations=recommendations,
        recommendation_ids=tuple(
            recommendation.personalization_id
            for recommendation in recommendations
        ),
        candidate_recommendation_ids=_recommendation_ids_for_status(
            recommendations,
            "candidate",
        ),
        review_required_recommendation_ids=_recommendation_ids_for_status(
            recommendations,
            "review_required",
        ),
        guarded_recommendation_ids=_recommendation_ids_for_status(
            recommendations,
            "guarded",
        ),
        high_confidence_recommendation_ids=_recommendation_ids_for_confidence(
            recommendations,
            "high",
            "guarded",
        ),
        hitl_required_recommendation_ids=tuple(
            recommendation.personalization_id
            for recommendation in recommendations
            if recommendation.hitl_required_before_application
        ),
        persisted_personalization_ids=(),
        learned_personalization_ids=(),
        applied_personalization_ids=(),
        mutated_preference_ids=(),
        recommendation_count=len(recommendations),
        candidate_recommendation_count=len(
            _recommendation_ids_for_status(recommendations, "candidate")
        ),
        review_required_recommendation_count=len(
            _recommendation_ids_for_status(recommendations, "review_required")
        ),
        guarded_recommendation_count=len(
            _recommendation_ids_for_status(recommendations, "guarded")
        ),
        high_confidence_recommendation_count=len(
            _recommendation_ids_for_confidence(recommendations, "high", "guarded")
        ),
        hitl_required_recommendation_count=sum(
            1
            for recommendation in recommendations
            if recommendation.hitl_required_before_application
        ),
        highest_personalization_score=max(
            recommendation.personalization_score
            for recommendation in recommendations
        ),
        overall_personalization_score=_overall_personalization_score(
            recommendations
        ),
        overall_personalization_posture=_overall_personalization_posture(
            recommendations
        ),
        advisory_actions=_plan_actions(recommendations),
    )


def personalization_recommendation_by_id(
    recommendation_id: str,
    plan: PersonalizationEnginePlan | None = None,
) -> PersonalizationRecommendation | None:
    """Return one personalization recommendation without applying it."""

    source_plan = plan or build_personalization_engine()
    for recommendation in source_plan.recommendations:
        if recommendation.personalization_id == recommendation_id:
            return recommendation
    return None


def personalization_recommendations_for_status(
    status: PersonalizationStatus,
    plan: PersonalizationEnginePlan | None = None,
) -> tuple[PersonalizationRecommendation, ...]:
    """Return personalization recommendations by advisory status."""

    source_plan = plan or build_personalization_engine()
    return tuple(
        recommendation
        for recommendation in source_plan.recommendations
        if recommendation.status == status
    )


def personalization_recommendations_for_confidence(
    confidence: PersonalizationConfidence,
    plan: PersonalizationEnginePlan | None = None,
) -> tuple[PersonalizationRecommendation, ...]:
    """Return personalization recommendations by confidence band."""

    source_plan = plan or build_personalization_engine()
    return tuple(
        recommendation
        for recommendation in source_plan.recommendations
        if recommendation.confidence == confidence
    )


def _recommendations(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    creative_dna: CreativeDNAPlan,
    user_preferences: UserPreferencesPlan,
    style_profiles: StyleProfilePlan,
    project_memory: ProjectMemoryPlan,
) -> tuple[PersonalizationRecommendation, ...]:
    return (
        _recommendation(
            kind="style_personalization",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            personalization_scope="style",
            source_dna_id="creative_dna::style_dna",
            source_preference_id="user_preferences::visual_style_preference",
            source_profile_id="style_profiles::palette_profile",
            source_project_signal_id="project_memory::project_style_memory",
            creative_dna=creative_dna,
            user_preferences=user_preferences,
            style_profiles=style_profiles,
            project_memory=project_memory,
            preference_alignment_score=88,
            creative_dna_alignment_score=86,
            project_fit_score=82,
            safety_risk_score=48,
            governance_weight=150,
        ),
        _recommendation(
            kind="interaction_personalization",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            personalization_scope="interaction",
            source_dna_id="creative_dna::interaction_dna",
            source_preference_id="user_preferences::interaction_preference",
            source_profile_id="style_profiles::motion_profile",
            source_project_signal_id="project_memory::project_decision_memory",
            creative_dna=creative_dna,
            user_preferences=user_preferences,
            style_profiles=style_profiles,
            project_memory=project_memory,
            preference_alignment_score=76,
            creative_dna_alignment_score=74,
            project_fit_score=68,
            safety_risk_score=40,
            governance_weight=130,
        ),
        _recommendation(
            kind="constraint_personalization",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            personalization_scope="constraints",
            source_dna_id="creative_dna::constraint_dna",
            source_preference_id="user_preferences::constraint_preference",
            source_profile_id="style_profiles::composition_profile",
            source_project_signal_id="project_memory::project_constraint_memory",
            creative_dna=creative_dna,
            user_preferences=user_preferences,
            style_profiles=style_profiles,
            project_memory=project_memory,
            preference_alignment_score=82,
            creative_dna_alignment_score=78,
            project_fit_score=76,
            safety_risk_score=52,
            governance_weight=145,
        ),
        _recommendation(
            kind="technical_personalization",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            personalization_scope="technical_stack",
            source_dna_id="creative_dna::lineage_dna",
            source_preference_id="user_preferences::technical_stack_preference",
            source_profile_id="style_profiles::material_profile",
            source_project_signal_id="project_memory::project_technical_memory",
            creative_dna=creative_dna,
            user_preferences=user_preferences,
            style_profiles=style_profiles,
            project_memory=project_memory,
            preference_alignment_score=64,
            creative_dna_alignment_score=60,
            project_fit_score=66,
            safety_risk_score=32,
            governance_weight=110,
        ),
        _recommendation(
            kind="review_depth_personalization",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            personalization_scope="review_depth",
            source_dna_id="creative_dna::intent_dna",
            source_preference_id="user_preferences::review_depth_preference",
            source_profile_id="style_profiles::typography_profile",
            source_project_signal_id="project_memory::project_goal_memory",
            creative_dna=creative_dna,
            user_preferences=user_preferences,
            style_profiles=style_profiles,
            project_memory=project_memory,
            preference_alignment_score=48,
            creative_dna_alignment_score=52,
            project_fit_score=50,
            safety_risk_score=18,
            governance_weight=85,
        ),
    )


def _recommendation(
    *,
    kind: PersonalizationRecommendationKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    personalization_scope: PersonalizationScope,
    source_dna_id: str,
    source_preference_id: str,
    source_profile_id: str,
    source_project_signal_id: str,
    creative_dna: CreativeDNAPlan,
    user_preferences: UserPreferencesPlan,
    style_profiles: StyleProfilePlan,
    project_memory: ProjectMemoryPlan,
    preference_alignment_score: int,
    creative_dna_alignment_score: int,
    project_fit_score: int,
    safety_risk_score: int,
    governance_weight: int,
) -> PersonalizationRecommendation:
    source_dna = creative_dna_signature_by_id(source_dna_id, creative_dna)
    source_preference = user_preference_by_id(source_preference_id, user_preferences)
    source_profile = style_profile_by_id(source_profile_id, style_profiles)
    source_project_signal = project_memory_signal_by_id(
        source_project_signal_id,
        project_memory,
    )
    if source_dna is None:
        raise ValueError("source Creative DNA signature must exist")
    if source_preference is None:
        raise ValueError("source user preference must exist")
    if source_profile is None:
        raise ValueError("source style profile must exist")
    if source_project_signal is None:
        raise ValueError("source project memory signal must exist")
    score = _personalization_score(
        preference_alignment_score=preference_alignment_score,
        creative_dna_alignment_score=creative_dna_alignment_score,
        project_fit_score=project_fit_score,
        safety_risk_score=safety_risk_score,
        governance_weight=governance_weight,
    )
    status = _personalization_status(score)
    confidence = _personalization_confidence(score)
    return PersonalizationRecommendation(
        personalization_id=f"personalization_engine::{kind}",
        recommendation_kind=kind,
        status=status,
        confidence=confidence,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        personalization_scope=personalization_scope,
        source_creative_dna_id=source_dna.creative_dna_id,
        source_user_preference_id=source_preference.preference_id,
        source_style_profile_id=source_profile.profile_id,
        source_project_memory_signal_id=source_project_signal.project_memory_id,
        personalization_summary=_personalization_summary(kind),
        preference_alignment_score=preference_alignment_score,
        creative_dna_alignment_score=creative_dna_alignment_score,
        project_fit_score=project_fit_score,
        safety_risk_score=safety_risk_score,
        governance_weight=governance_weight,
        personalization_score=score,
        hitl_required_before_application=True,
        context_tags=_context_tags(kind, personalization_scope),
        explainability_notes=_explainability_notes(
            kind,
            source_dna.creative_dna_id,
            source_preference.preference_id,
            source_profile.profile_id,
            source_project_signal.project_memory_id,
        ),
        advisory_actions=_recommendation_actions(kind),
        evidence=(
            f"source_creative_dna:{source_dna.creative_dna_id}",
            f"source_user_preference:{source_preference.preference_id}",
            f"source_style_profile:{source_profile.profile_id}",
            f"source_project_memory:{source_project_signal.project_memory_id}",
            f"personalization_scope:{personalization_scope}",
            f"preference_alignment_score:{preference_alignment_score}",
            f"creative_dna_alignment_score:{creative_dna_alignment_score}",
            f"project_fit_score:{project_fit_score}",
            f"safety_risk_score:{safety_risk_score}",
            "hitl_required_before_application:true",
        ),
    )


def _personalization_score(
    *,
    preference_alignment_score: int,
    creative_dna_alignment_score: int,
    project_fit_score: int,
    safety_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            preference_alignment_score * 3
            + creative_dna_alignment_score * 3
            + project_fit_score * 2
            + safety_risk_score * 3
            + governance_weight,
        ),
    )


def _personalization_status(score: int) -> PersonalizationStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _personalization_confidence(score: int) -> PersonalizationConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_personalization_score(
    recommendations: tuple[PersonalizationRecommendation, ...],
) -> int:
    base = sum(
        recommendation.personalization_score
        for recommendation in recommendations
    ) // len(recommendations)
    guarded_count = len(_recommendation_ids_for_status(recommendations, "guarded"))
    review_count = len(
        _recommendation_ids_for_status(recommendations, "review_required")
    )
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_personalization_posture(
    recommendations: tuple[PersonalizationRecommendation, ...],
) -> PersonalizationPosture:
    if any(recommendation.status == "guarded" for recommendation in recommendations):
        return "guarded"
    if any(
        recommendation.status == "review_required"
        for recommendation in recommendations
    ):
        return "review_required"
    return "candidate"


def _recommendation_ids_for_status(
    recommendations: tuple[PersonalizationRecommendation, ...],
    status: PersonalizationStatus,
) -> tuple[str, ...]:
    return tuple(
        recommendation.personalization_id
        for recommendation in recommendations
        if recommendation.status == status
    )


def _recommendation_ids_for_confidence(
    recommendations: tuple[PersonalizationRecommendation, ...],
    *confidences: PersonalizationConfidence,
) -> tuple[str, ...]:
    return tuple(
        recommendation.personalization_id
        for recommendation in recommendations
        if recommendation.confidence in confidences
    )


def _plan_actions(
    recommendations: tuple[PersonalizationRecommendation, ...],
) -> tuple[str, ...]:
    guarded_recommendation_count = len(
        _recommendation_ids_for_status(recommendations, "guarded")
    )
    return (
        "inspect_personalization_recommendations",
        "require_hitl_before_personalization_application",
        "keep_personalization_non_executing",
        f"review_guarded_personalization_count:{guarded_recommendation_count}",
    )


def _personalization_summary(kind: PersonalizationRecommendationKind) -> str:
    summaries = {
        "style_personalization": "Models advisory style personalization posture.",
        "interaction_personalization": (
            "Models advisory interaction personalization posture."
        ),
        "constraint_personalization": (
            "Models advisory constraint personalization posture."
        ),
        "technical_personalization": (
            "Models advisory technical-stack personalization posture."
        ),
        "review_depth_personalization": (
            "Models advisory review-depth personalization posture."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: PersonalizationRecommendationKind,
    personalization_scope: PersonalizationScope,
) -> tuple[str, ...]:
    return (
        "creative_memory",
        "personalization_engine",
        personalization_scope,
        kind.removesuffix("_personalization"),
    )


def _explainability_notes(
    kind: PersonalizationRecommendationKind,
    source_dna_id: str,
    source_preference_id: str,
    source_profile_id: str,
    source_project_signal_id: str,
) -> tuple[str, ...]:
    return (
        f"personalization_kind:{kind}",
        f"source_creative_dna:{source_dna_id}",
        f"source_preference:{source_preference_id}",
        f"source_style_profile:{source_profile_id}",
        f"source_project_memory:{source_project_signal_id}",
        "score_inputs:preference_alignment,creative_dna_alignment,project_fit,safety_risk,governance",
        "application_boundary:HITL_required_before_personalization_application",
    )


def _recommendation_actions(
    kind: PersonalizationRecommendationKind,
) -> tuple[str, ...]:
    return (
        f"review_{kind}",
        "inspect_sources_before_personalization_application",
        "preserve_no_personalization_application_boundary",
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
