"""V6.2 advisory creative DNA metadata."""

from __future__ import annotations

from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.long_term_creative_memory import (
    LongTermCreativeMemoryPlan,
    build_long_term_creative_memory,
    long_term_creative_memory_record_by_id,
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

CreativeDNAFeatureKind = Literal[
    "intent_dna",
    "style_dna",
    "constraint_dna",
    "interaction_dna",
    "lineage_dna",
]
CreativeDNAStatus = Literal["candidate", "review_required", "guarded"]
CreativeDNAConfidence = Literal["low", "medium", "high", "guarded"]
CreativeDNAPosture = Literal["candidate", "review_required", "guarded"]
CreativeDNAExpressionAxis = Literal[
    "intent",
    "style",
    "constraints",
    "interaction",
    "lineage",
]

CREATIVE_DNA_SIGNATURE_SERIALIZATION_VERSION = "creative_dna_signature.v1"
CREATIVE_DNA_PLAN_SERIALIZATION_VERSION = "creative_dna_plan.v1"
CREATIVE_DNA_AUTHORITY_BOUNDARY = (
    "V6.2 Creative DNA models durable creative identity signatures as "
    "inspectable advisory metadata only; it does not write Creative DNA "
    "storage, create Creative DNA signatures, update Creative DNA signatures, "
    "delete Creative DNA signatures, learn Creative DNA automatically, apply "
    "Creative DNA to prompts or generated output, execute memory retrieval, "
    "write memory storage, write project memory storage, apply style profiles, "
    "mutate preferences, apply personalization, change provider or model "
    "routing, execute providers, invoke agents, control workflows, mutate "
    "workflow graphs, trigger retries or refinements, mutate prompts, modify "
    "generated output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "creative_dna_storage_write",
    "creative_dna_signature_creation",
    "creative_dna_signature_update",
    "creative_dna_signature_deletion",
    "automatic_creative_dna_learning",
    "creative_dna_application",
    "automatic_preference_mutation",
    "automatic_personalization_application",
    "memory_retrieval_execution",
    "memory_storage_write",
    "project_memory_storage_write",
    "style_profile_application",
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


class CreativeDNASignature(BaseModel):
    """One advisory creative DNA signature."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    creative_dna_id: str = Field(min_length=1, max_length=180)
    feature_kind: CreativeDNAFeatureKind
    status: CreativeDNAStatus
    confidence: CreativeDNAConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    expression_axis: CreativeDNAExpressionAxis
    source_long_term_memory_record_id: str = Field(min_length=1, max_length=180)
    source_user_preference_id: str = Field(min_length=1, max_length=180)
    source_style_profile_id: str = Field(min_length=1, max_length=180)
    source_project_memory_signal_id: str = Field(min_length=1, max_length=180)
    dna_statement: str = Field(min_length=1, max_length=360)
    source_alignment_score: int = Field(ge=0, le=100)
    style_consistency_score: int = Field(ge=0, le=100)
    project_continuity_score: int = Field(ge=0, le=100)
    conflict_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    creative_dna_score: int = Field(ge=0, le=1_000)
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
    creative_dna_implemented: Literal[True] = True
    creative_dna_metadata_implemented: Literal[True] = True
    long_term_memory_source_used: Literal[True] = True
    user_preferences_source_used: Literal[True] = True
    style_profile_source_used: Literal[True] = True
    project_memory_source_used: Literal[True] = True
    creative_dna_storage_write_implemented: Literal[False] = False
    creative_dna_signature_creation_implemented: Literal[False] = False
    creative_dna_signature_update_implemented: Literal[False] = False
    creative_dna_signature_deletion_implemented: Literal[False] = False
    automatic_creative_dna_learning_implemented: Literal[False] = False
    creative_dna_application_implemented: Literal[False] = False
    preference_mutation_implemented: Literal[False] = False
    personalization_application_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_storage_write_implemented: Literal[False] = False
    project_memory_storage_write_implemented: Literal[False] = False
    style_profile_application_implemented: Literal[False] = False
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
    serialization_version: Literal["creative_dna_signature.v1"] = (
        CREATIVE_DNA_SIGNATURE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signature_matches_contract(self) -> Self:
        if self.creative_dna_id != f"creative_dna::{self.feature_kind}":
            raise ValueError("creative_dna_id must match feature_kind")
        if self.creative_dna_score != _creative_dna_score(
            source_alignment_score=self.source_alignment_score,
            style_consistency_score=self.style_consistency_score,
            project_continuity_score=self.project_continuity_score,
            conflict_risk_score=self.conflict_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("creative_dna_score must combine source scores")
        if self.status != _creative_dna_status(self.creative_dna_score):
            raise ValueError("status must match creative_dna_score")
        if self.confidence != _creative_dna_confidence(self.creative_dna_score):
            raise ValueError("confidence must match creative_dna_score")
        if not self.hitl_required_before_application:
            raise ValueError("Creative DNA application requires HITL posture")
        return self


class CreativeDNAPlan(BaseModel):
    """Bounded V6.2 advisory creative DNA plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_dna"] = "creative_dna"
    serialization_version: Literal["creative_dna_plan.v1"] = (
        CREATIVE_DNA_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_DNA_AUTHORITY_BOUNDARY,
        max_length=1900,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_long_term_memory_serialization_version: str = Field(
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
    source_long_term_memory_record_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_user_preference_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    source_style_profile_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    source_project_memory_signal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signatures: tuple[CreativeDNASignature, ...] = Field(min_length=5, max_length=5)
    signature_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    candidate_signature_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    review_required_signature_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_signature_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    high_confidence_signature_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    hitl_required_signature_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    persisted_creative_dna_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    learned_creative_dna_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    applied_creative_dna_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    personalized_creative_dna_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    signature_count: int = Field(ge=5, le=5)
    candidate_signature_count: int = Field(ge=0, le=5)
    review_required_signature_count: int = Field(ge=0, le=5)
    guarded_signature_count: int = Field(ge=0, le=5)
    high_confidence_signature_count: int = Field(ge=0, le=5)
    hitl_required_signature_count: int = Field(ge=0, le=5)
    highest_creative_dna_score: int = Field(ge=0, le=1_000)
    overall_creative_dna_score: int = Field(ge=0, le=1_000)
    overall_creative_dna_posture: CreativeDNAPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    creative_dna_implemented: Literal[True] = True
    creative_dna_metadata_implemented: Literal[True] = True
    long_term_memory_source_used: Literal[True] = True
    user_preferences_source_used: Literal[True] = True
    style_profile_source_used: Literal[True] = True
    project_memory_source_used: Literal[True] = True
    creative_dna_storage_write_implemented: Literal[False] = False
    creative_dna_signature_creation_implemented: Literal[False] = False
    creative_dna_signature_update_implemented: Literal[False] = False
    creative_dna_signature_deletion_implemented: Literal[False] = False
    automatic_creative_dna_learning_implemented: Literal[False] = False
    creative_dna_application_implemented: Literal[False] = False
    preference_mutation_implemented: Literal[False] = False
    personalization_application_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_storage_write_implemented: Literal[False] = False
    project_memory_storage_write_implemented: Literal[False] = False
    style_profile_application_implemented: Literal[False] = False
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
    def _plan_matches_signatures(self) -> Self:
        derived_signature_ids = tuple(
            signature.creative_dna_id for signature in self.signatures
        )
        if len(set(derived_signature_ids)) != len(derived_signature_ids):
            raise ValueError("signature_ids must be unique")
        if self.signature_ids != derived_signature_ids:
            raise ValueError("signature_ids must match signatures")
        if self.candidate_signature_ids != _signature_ids_for_status(
            self.signatures,
            "candidate",
        ):
            raise ValueError("candidate_signature_ids must match signatures")
        if self.review_required_signature_ids != _signature_ids_for_status(
            self.signatures,
            "review_required",
        ):
            raise ValueError("review_required_signature_ids must match signatures")
        if self.guarded_signature_ids != _signature_ids_for_status(
            self.signatures,
            "guarded",
        ):
            raise ValueError("guarded_signature_ids must match signatures")
        if self.high_confidence_signature_ids != _signature_ids_for_confidence(
            self.signatures,
            "high",
            "guarded",
        ):
            raise ValueError("high_confidence_signature_ids must match signatures")
        if self.hitl_required_signature_ids != tuple(
            signature.creative_dna_id
            for signature in self.signatures
            if signature.hitl_required_before_application
        ):
            raise ValueError("hitl_required_signature_ids must match signatures")
        if self.persisted_creative_dna_ids:
            raise ValueError("persisted_creative_dna_ids must remain empty")
        if self.learned_creative_dna_ids:
            raise ValueError("learned_creative_dna_ids must remain empty")
        if self.applied_creative_dna_ids:
            raise ValueError("applied_creative_dna_ids must remain empty")
        if self.personalized_creative_dna_ids:
            raise ValueError("personalized_creative_dna_ids must remain empty")
        if self.signature_count != len(self.signatures):
            raise ValueError("signature_count must match signatures")
        if self.candidate_signature_count != len(self.candidate_signature_ids):
            raise ValueError("candidate_signature_count must match signatures")
        if self.review_required_signature_count != len(
            self.review_required_signature_ids
        ):
            raise ValueError("review_required_signature_count must match signatures")
        if self.guarded_signature_count != len(self.guarded_signature_ids):
            raise ValueError("guarded_signature_count must match signatures")
        if self.high_confidence_signature_count != len(
            self.high_confidence_signature_ids
        ):
            raise ValueError("high_confidence_signature_count must match signatures")
        if self.hitl_required_signature_count != len(
            self.hitl_required_signature_ids
        ):
            raise ValueError("hitl_required_signature_count must match signatures")
        if self.highest_creative_dna_score != max(
            signature.creative_dna_score for signature in self.signatures
        ):
            raise ValueError("highest_creative_dna_score must match signatures")
        if self.overall_creative_dna_score != _overall_creative_dna_score(
            self.signatures
        ):
            raise ValueError("overall_creative_dna_score must match signatures")
        if self.overall_creative_dna_posture != _overall_creative_dna_posture(
            self.signatures
        ):
            raise ValueError("overall_creative_dna_posture must match signatures")
        for signature in self.signatures:
            if signature.route_name != self.route_name:
                raise ValueError("signature route_name must match plan")
            if (
                signature.source_long_term_memory_record_id
                not in self.source_long_term_memory_record_ids
            ):
                raise ValueError("source long-term memory record must be declared")
            if (
                signature.source_user_preference_id
                not in self.source_user_preference_ids
            ):
                raise ValueError("source user preference must be declared")
            if signature.source_style_profile_id not in self.source_style_profile_ids:
                raise ValueError("source style profile must be declared")
            if (
                signature.source_project_memory_signal_id
                not in self.source_project_memory_signal_ids
            ):
                raise ValueError("source project memory signal must be declared")
        return self


def build_creative_dna(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    long_term_memory: LongTermCreativeMemoryPlan | None = None,
    user_preferences: UserPreferencesPlan | None = None,
    style_profiles: StyleProfilePlan | None = None,
    project_memory: ProjectMemoryPlan | None = None,
) -> CreativeDNAPlan:
    """Build Creative DNA metadata without persistence or application."""

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
    preference_plan = user_preferences or build_user_preferences(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        long_term_memory=memory_plan,
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
        long_term_memory=memory_plan,
        style_profiles=style_plan,
    )
    signatures = _signatures(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        long_term_memory=memory_plan,
        user_preferences=preference_plan,
        style_profiles=style_plan,
        project_memory=project_plan,
    )
    return CreativeDNAPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        source_long_term_memory_serialization_version=(
            memory_plan.serialization_version
        ),
        source_user_preferences_serialization_version=(
            preference_plan.serialization_version
        ),
        source_style_profile_serialization_version=style_plan.serialization_version,
        source_project_memory_serialization_version=project_plan.serialization_version,
        source_long_term_memory_record_ids=memory_plan.record_ids,
        source_user_preference_ids=preference_plan.preference_ids,
        source_style_profile_ids=style_plan.profile_ids,
        source_project_memory_signal_ids=project_plan.signal_ids,
        execution_mode_ids=execution_modes.execution_mode_ids,
        signatures=signatures,
        signature_ids=tuple(signature.creative_dna_id for signature in signatures),
        candidate_signature_ids=_signature_ids_for_status(signatures, "candidate"),
        review_required_signature_ids=_signature_ids_for_status(
            signatures,
            "review_required",
        ),
        guarded_signature_ids=_signature_ids_for_status(signatures, "guarded"),
        high_confidence_signature_ids=_signature_ids_for_confidence(
            signatures,
            "high",
            "guarded",
        ),
        hitl_required_signature_ids=tuple(
            signature.creative_dna_id
            for signature in signatures
            if signature.hitl_required_before_application
        ),
        persisted_creative_dna_ids=(),
        learned_creative_dna_ids=(),
        applied_creative_dna_ids=(),
        personalized_creative_dna_ids=(),
        signature_count=len(signatures),
        candidate_signature_count=len(
            _signature_ids_for_status(signatures, "candidate")
        ),
        review_required_signature_count=len(
            _signature_ids_for_status(signatures, "review_required")
        ),
        guarded_signature_count=len(_signature_ids_for_status(signatures, "guarded")),
        high_confidence_signature_count=len(
            _signature_ids_for_confidence(signatures, "high", "guarded")
        ),
        hitl_required_signature_count=sum(
            1 for signature in signatures if signature.hitl_required_before_application
        ),
        highest_creative_dna_score=max(
            signature.creative_dna_score for signature in signatures
        ),
        overall_creative_dna_score=_overall_creative_dna_score(signatures),
        overall_creative_dna_posture=_overall_creative_dna_posture(signatures),
        advisory_actions=_plan_actions(signatures),
    )


def creative_dna_signature_by_id(
    signature_id: str,
    plan: CreativeDNAPlan | None = None,
) -> CreativeDNASignature | None:
    """Return one Creative DNA signature without applying it."""

    source_plan = plan or build_creative_dna()
    for signature in source_plan.signatures:
        if signature.creative_dna_id == signature_id:
            return signature
    return None


def creative_dna_signatures_for_status(
    status: CreativeDNAStatus,
    plan: CreativeDNAPlan | None = None,
) -> tuple[CreativeDNASignature, ...]:
    """Return Creative DNA signatures by advisory status."""

    source_plan = plan or build_creative_dna()
    return tuple(
        signature for signature in source_plan.signatures if signature.status == status
    )


def creative_dna_signatures_for_confidence(
    confidence: CreativeDNAConfidence,
    plan: CreativeDNAPlan | None = None,
) -> tuple[CreativeDNASignature, ...]:
    """Return Creative DNA signatures by confidence band."""

    source_plan = plan or build_creative_dna()
    return tuple(
        signature
        for signature in source_plan.signatures
        if signature.confidence == confidence
    )


def _signatures(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    long_term_memory: LongTermCreativeMemoryPlan,
    user_preferences: UserPreferencesPlan,
    style_profiles: StyleProfilePlan,
    project_memory: ProjectMemoryPlan,
) -> tuple[CreativeDNASignature, ...]:
    return (
        _signature(
            kind="intent_dna",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            expression_axis="intent",
            source_record_id="long_term_creative_memory::creative_intent_memory",
            source_preference_id="user_preferences::interaction_preference",
            source_profile_id="style_profiles::composition_profile",
            source_project_signal_id="project_memory::project_goal_memory",
            long_term_memory=long_term_memory,
            user_preferences=user_preferences,
            style_profiles=style_profiles,
            project_memory=project_memory,
            source_alignment_score=86,
            style_consistency_score=82,
            project_continuity_score=84,
            conflict_risk_score=46,
            governance_weight=150,
        ),
        _signature(
            kind="style_dna",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            expression_axis="style",
            source_record_id="long_term_creative_memory::style_pattern_memory",
            source_preference_id="user_preferences::visual_style_preference",
            source_profile_id="style_profiles::palette_profile",
            source_project_signal_id="project_memory::project_style_memory",
            long_term_memory=long_term_memory,
            user_preferences=user_preferences,
            style_profiles=style_profiles,
            project_memory=project_memory,
            source_alignment_score=82,
            style_consistency_score=88,
            project_continuity_score=78,
            conflict_risk_score=44,
            governance_weight=140,
        ),
        _signature(
            kind="constraint_dna",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            expression_axis="constraints",
            source_record_id="long_term_creative_memory::project_context_memory",
            source_preference_id="user_preferences::constraint_preference",
            source_profile_id="style_profiles::composition_profile",
            source_project_signal_id="project_memory::project_constraint_memory",
            long_term_memory=long_term_memory,
            user_preferences=user_preferences,
            style_profiles=style_profiles,
            project_memory=project_memory,
            source_alignment_score=74,
            style_consistency_score=76,
            project_continuity_score=72,
            conflict_risk_score=42,
            governance_weight=120,
        ),
        _signature(
            kind="interaction_dna",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            expression_axis="interaction",
            source_record_id="long_term_creative_memory::preference_signal_memory",
            source_preference_id="user_preferences::interaction_preference",
            source_profile_id="style_profiles::motion_profile",
            source_project_signal_id="project_memory::project_decision_memory",
            long_term_memory=long_term_memory,
            user_preferences=user_preferences,
            style_profiles=style_profiles,
            project_memory=project_memory,
            source_alignment_score=66,
            style_consistency_score=70,
            project_continuity_score=62,
            conflict_risk_score=32,
            governance_weight=110,
        ),
        _signature(
            kind="lineage_dna",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            expression_axis="lineage",
            source_record_id="long_term_creative_memory::artifact_lineage_memory",
            source_preference_id="user_preferences::technical_stack_preference",
            source_profile_id="style_profiles::material_profile",
            source_project_signal_id="project_memory::project_technical_memory",
            long_term_memory=long_term_memory,
            user_preferences=user_preferences,
            style_profiles=style_profiles,
            project_memory=project_memory,
            source_alignment_score=50,
            style_consistency_score=56,
            project_continuity_score=54,
            conflict_risk_score=20,
            governance_weight=85,
        ),
    )


def _signature(
    *,
    kind: CreativeDNAFeatureKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    expression_axis: CreativeDNAExpressionAxis,
    source_record_id: str,
    source_preference_id: str,
    source_profile_id: str,
    source_project_signal_id: str,
    long_term_memory: LongTermCreativeMemoryPlan,
    user_preferences: UserPreferencesPlan,
    style_profiles: StyleProfilePlan,
    project_memory: ProjectMemoryPlan,
    source_alignment_score: int,
    style_consistency_score: int,
    project_continuity_score: int,
    conflict_risk_score: int,
    governance_weight: int,
) -> CreativeDNASignature:
    source_record = long_term_creative_memory_record_by_id(
        source_record_id,
        long_term_memory,
    )
    source_preference = user_preference_by_id(source_preference_id, user_preferences)
    source_profile = style_profile_by_id(source_profile_id, style_profiles)
    source_project_signal = project_memory_signal_by_id(
        source_project_signal_id,
        project_memory,
    )
    if source_record is None:
        raise ValueError("source long-term memory record must exist")
    if source_preference is None:
        raise ValueError("source user preference must exist")
    if source_profile is None:
        raise ValueError("source style profile must exist")
    if source_project_signal is None:
        raise ValueError("source project memory signal must exist")
    score = _creative_dna_score(
        source_alignment_score=source_alignment_score,
        style_consistency_score=style_consistency_score,
        project_continuity_score=project_continuity_score,
        conflict_risk_score=conflict_risk_score,
        governance_weight=governance_weight,
    )
    status = _creative_dna_status(score)
    confidence = _creative_dna_confidence(score)
    return CreativeDNASignature(
        creative_dna_id=f"creative_dna::{kind}",
        feature_kind=kind,
        status=status,
        confidence=confidence,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        expression_axis=expression_axis,
        source_long_term_memory_record_id=source_record.record_id,
        source_user_preference_id=source_preference.preference_id,
        source_style_profile_id=source_profile.profile_id,
        source_project_memory_signal_id=source_project_signal.project_memory_id,
        dna_statement=_dna_statement(kind),
        source_alignment_score=source_alignment_score,
        style_consistency_score=style_consistency_score,
        project_continuity_score=project_continuity_score,
        conflict_risk_score=conflict_risk_score,
        governance_weight=governance_weight,
        creative_dna_score=score,
        hitl_required_before_application=True,
        context_tags=_context_tags(kind, expression_axis),
        explainability_notes=_explainability_notes(
            kind,
            source_record.record_id,
            source_preference.preference_id,
            source_profile.profile_id,
            source_project_signal.project_memory_id,
        ),
        advisory_actions=_signature_actions(kind),
        evidence=(
            f"source_long_term_memory:{source_record.record_id}",
            f"source_user_preference:{source_preference.preference_id}",
            f"source_style_profile:{source_profile.profile_id}",
            f"source_project_memory:{source_project_signal.project_memory_id}",
            f"expression_axis:{expression_axis}",
            f"source_alignment_score:{source_alignment_score}",
            f"style_consistency_score:{style_consistency_score}",
            f"project_continuity_score:{project_continuity_score}",
            f"conflict_risk_score:{conflict_risk_score}",
            "hitl_required_before_application:true",
        ),
    )


def _creative_dna_score(
    *,
    source_alignment_score: int,
    style_consistency_score: int,
    project_continuity_score: int,
    conflict_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            source_alignment_score * 3
            + style_consistency_score * 3
            + project_continuity_score * 2
            + conflict_risk_score * 3
            + governance_weight,
        ),
    )


def _creative_dna_status(score: int) -> CreativeDNAStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _creative_dna_confidence(score: int) -> CreativeDNAConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_creative_dna_score(
    signatures: tuple[CreativeDNASignature, ...],
) -> int:
    base = sum(signature.creative_dna_score for signature in signatures) // len(
        signatures
    )
    guarded_count = len(_signature_ids_for_status(signatures, "guarded"))
    review_count = len(_signature_ids_for_status(signatures, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_creative_dna_posture(
    signatures: tuple[CreativeDNASignature, ...],
) -> CreativeDNAPosture:
    if any(signature.status == "guarded" for signature in signatures):
        return "guarded"
    if any(signature.status == "review_required" for signature in signatures):
        return "review_required"
    return "candidate"


def _signature_ids_for_status(
    signatures: tuple[CreativeDNASignature, ...],
    status: CreativeDNAStatus,
) -> tuple[str, ...]:
    return tuple(
        signature.creative_dna_id
        for signature in signatures
        if signature.status == status
    )


def _signature_ids_for_confidence(
    signatures: tuple[CreativeDNASignature, ...],
    *confidences: CreativeDNAConfidence,
) -> tuple[str, ...]:
    return tuple(
        signature.creative_dna_id
        for signature in signatures
        if signature.confidence in confidences
    )


def _plan_actions(signatures: tuple[CreativeDNASignature, ...]) -> tuple[str, ...]:
    guarded_signature_count = len(_signature_ids_for_status(signatures, "guarded"))
    return (
        "inspect_creative_dna_signatures",
        "require_hitl_before_creative_dna_application",
        "keep_creative_dna_non_executing",
        f"review_guarded_creative_dna_count:{guarded_signature_count}",
    )


def _dna_statement(kind: CreativeDNAFeatureKind) -> str:
    statements = {
        "intent_dna": "Models recurring creative intent posture.",
        "style_dna": "Models recurring visual and stylistic identity.",
        "constraint_dna": "Models recurring constraint and project-shaping posture.",
        "interaction_dna": "Models recurring interaction and review posture.",
        "lineage_dna": "Models recurring artifact lineage and technical continuity.",
    }
    return statements[kind]


def _context_tags(
    kind: CreativeDNAFeatureKind,
    expression_axis: CreativeDNAExpressionAxis,
) -> tuple[str, ...]:
    return (
        "creative_memory",
        "creative_dna",
        expression_axis,
        kind.removesuffix("_dna"),
    )


def _explainability_notes(
    kind: CreativeDNAFeatureKind,
    source_record_id: str,
    source_preference_id: str,
    source_profile_id: str,
    source_project_signal_id: str,
) -> tuple[str, ...]:
    return (
        f"creative_dna_kind:{kind}",
        f"source_record:{source_record_id}",
        f"source_preference:{source_preference_id}",
        f"source_style_profile:{source_profile_id}",
        f"source_project_memory:{source_project_signal_id}",
        "score_inputs:source_alignment,style_consistency,project_continuity,conflict_risk,governance",
        "application_boundary:HITL_required_before_creative_dna_application",
    )


def _signature_actions(kind: CreativeDNAFeatureKind) -> tuple[str, ...]:
    return (
        f"review_{kind}",
        "inspect_sources_before_creative_dna_application",
        "preserve_no_creative_dna_application_boundary",
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
