"""V6.2 advisory project memory metadata."""

from __future__ import annotations

from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.memory import ProjectMemoryKind
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
from creative_coding_assistant.orchestration.style_profiles import (
    StyleProfilePlan,
    build_style_profiles,
    style_profile_by_id,
)

ProjectMemoryFacetKind = Literal[
    "project_goal_memory",
    "project_constraint_memory",
    "project_decision_memory",
    "project_style_memory",
    "project_technical_memory",
]
ProjectMemoryStatus = Literal["candidate", "review_required", "guarded"]
ProjectMemoryConfidence = Literal["low", "medium", "high", "guarded"]
ProjectMemoryPosture = Literal["candidate", "review_required", "guarded"]
ProjectMemoryScope = Literal["goal", "constraint", "decision", "style", "technical"]

PROJECT_MEMORY_SIGNAL_SERIALIZATION_VERSION = "project_memory_signal.v1"
PROJECT_MEMORY_PLAN_SERIALIZATION_VERSION = "project_memory_plan.v1"
PROJECT_MEMORY_AUTHORITY_BOUNDARY = (
    "V6.2 project memory models durable project context as inspectable "
    "advisory metadata only; it does not write project memory storage, create "
    "project memory records, update project memory records, delete project "
    "memory records, execute memory retrieval, consolidate memory, apply style "
    "profiles, mutate preferences, apply personalization, change provider or "
    "model routing, execute providers, invoke agents, control workflows, "
    "mutate workflow graphs, trigger retries or refinements, mutate prompts, "
    "modify generated output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "project_memory_storage_write",
    "project_memory_record_creation",
    "project_memory_record_update",
    "project_memory_record_deletion",
    "memory_retrieval_execution",
    "automatic_memory_consolidation",
    "style_profile_application",
    "automatic_preference_mutation",
    "automatic_personalization_application",
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


class ProjectMemorySignal(BaseModel):
    """One advisory project memory signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    project_memory_id: str = Field(min_length=1, max_length=180)
    facet_kind: ProjectMemoryFacetKind
    status: ProjectMemoryStatus
    confidence: ProjectMemoryConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    memory_scope: ProjectMemoryScope
    project_memory_kind: ProjectMemoryKind
    source_long_term_memory_record_id: str = Field(min_length=1, max_length=180)
    source_style_profile_id: str = Field(min_length=1, max_length=180)
    project_memory_summary: str = Field(min_length=1, max_length=360)
    continuity_score: int = Field(ge=0, le=100)
    specificity_score: int = Field(ge=0, le=100)
    conflict_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    project_memory_score: int = Field(ge=0, le=1_000)
    hitl_required_before_storage: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    project_memory_implemented: Literal[True] = True
    project_memory_metadata_implemented: Literal[True] = True
    long_term_memory_source_used: Literal[True] = True
    style_profile_source_used: Literal[True] = True
    project_memory_storage_write_implemented: Literal[False] = False
    project_memory_record_creation_implemented: Literal[False] = False
    project_memory_record_update_implemented: Literal[False] = False
    project_memory_record_deletion_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_consolidation_implemented: Literal[False] = False
    style_profile_application_implemented: Literal[False] = False
    preference_mutation_implemented: Literal[False] = False
    personalization_application_implemented: Literal[False] = False
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
    serialization_version: Literal["project_memory_signal.v1"] = (
        PROJECT_MEMORY_SIGNAL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.project_memory_id != f"project_memory::{self.facet_kind}":
            raise ValueError("project_memory_id must match facet_kind")
        if self.project_memory_score != _project_memory_score(
            continuity_score=self.continuity_score,
            specificity_score=self.specificity_score,
            conflict_risk_score=self.conflict_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("project_memory_score must combine source scores")
        if self.status != _project_memory_status(self.project_memory_score):
            raise ValueError("status must match project_memory_score")
        if self.confidence != _project_memory_confidence(self.project_memory_score):
            raise ValueError("confidence must match project_memory_score")
        if not self.hitl_required_before_storage:
            raise ValueError("project memory storage requires HITL posture")
        return self


class ProjectMemoryPlan(BaseModel):
    """Bounded V6.2 advisory project memory plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["project_memory"] = "project_memory"
    serialization_version: Literal["project_memory_plan.v1"] = (
        PROJECT_MEMORY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PROJECT_MEMORY_AUTHORITY_BOUNDARY,
        max_length=1800,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_long_term_memory_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_style_profile_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_long_term_memory_record_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_style_profile_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[ProjectMemorySignal, ...] = Field(min_length=5, max_length=5)
    signal_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    candidate_signal_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    review_required_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_signal_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    high_confidence_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    hitl_required_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    stored_project_memory_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    retrieved_project_memory_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    consolidated_project_memory_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    signal_count: int = Field(ge=5, le=5)
    candidate_signal_count: int = Field(ge=0, le=5)
    review_required_signal_count: int = Field(ge=0, le=5)
    guarded_signal_count: int = Field(ge=0, le=5)
    high_confidence_signal_count: int = Field(ge=0, le=5)
    hitl_required_signal_count: int = Field(ge=0, le=5)
    highest_project_memory_score: int = Field(ge=0, le=1_000)
    overall_project_memory_score: int = Field(ge=0, le=1_000)
    overall_project_memory_posture: ProjectMemoryPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    project_memory_implemented: Literal[True] = True
    project_memory_metadata_implemented: Literal[True] = True
    long_term_memory_source_used: Literal[True] = True
    style_profile_source_used: Literal[True] = True
    project_memory_storage_write_implemented: Literal[False] = False
    project_memory_record_creation_implemented: Literal[False] = False
    project_memory_record_update_implemented: Literal[False] = False
    project_memory_record_deletion_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_consolidation_implemented: Literal[False] = False
    style_profile_application_implemented: Literal[False] = False
    preference_mutation_implemented: Literal[False] = False
    personalization_application_implemented: Literal[False] = False
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
    def _plan_matches_signals(self) -> Self:
        derived_signal_ids = tuple(signal.project_memory_id for signal in self.signals)
        if len(set(derived_signal_ids)) != len(derived_signal_ids):
            raise ValueError("signal_ids must be unique")
        if self.signal_ids != derived_signal_ids:
            raise ValueError("signal_ids must match signals")
        if self.candidate_signal_ids != _signal_ids_for_status(
            self.signals,
            "candidate",
        ):
            raise ValueError("candidate_signal_ids must match signals")
        if self.review_required_signal_ids != _signal_ids_for_status(
            self.signals,
            "review_required",
        ):
            raise ValueError("review_required_signal_ids must match signals")
        if self.guarded_signal_ids != _signal_ids_for_status(self.signals, "guarded"):
            raise ValueError("guarded_signal_ids must match signals")
        if self.high_confidence_signal_ids != _signal_ids_for_confidence(
            self.signals,
            "high",
            "guarded",
        ):
            raise ValueError("high_confidence_signal_ids must match signals")
        if self.hitl_required_signal_ids != tuple(
            signal.project_memory_id
            for signal in self.signals
            if signal.hitl_required_before_storage
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.stored_project_memory_ids:
            raise ValueError("stored_project_memory_ids must remain empty")
        if self.retrieved_project_memory_ids:
            raise ValueError("retrieved_project_memory_ids must remain empty")
        if self.consolidated_project_memory_ids:
            raise ValueError("consolidated_project_memory_ids must remain empty")
        if self.signal_count != len(self.signals):
            raise ValueError("signal_count must match signals")
        if self.candidate_signal_count != len(self.candidate_signal_ids):
            raise ValueError("candidate_signal_count must match signals")
        if self.review_required_signal_count != len(self.review_required_signal_ids):
            raise ValueError("review_required_signal_count must match signals")
        if self.guarded_signal_count != len(self.guarded_signal_ids):
            raise ValueError("guarded_signal_count must match signals")
        if self.high_confidence_signal_count != len(self.high_confidence_signal_ids):
            raise ValueError("high_confidence_signal_count must match signals")
        if self.hitl_required_signal_count != len(self.hitl_required_signal_ids):
            raise ValueError("hitl_required_signal_count must match signals")
        if self.highest_project_memory_score != max(
            signal.project_memory_score for signal in self.signals
        ):
            raise ValueError("highest_project_memory_score must match signals")
        if self.overall_project_memory_score != _overall_project_memory_score(
            self.signals
        ):
            raise ValueError("overall_project_memory_score must match signals")
        if self.overall_project_memory_posture != _overall_project_memory_posture(
            self.signals
        ):
            raise ValueError("overall_project_memory_posture must match signals")
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if (
                signal.source_long_term_memory_record_id
                not in self.source_long_term_memory_record_ids
            ):
                raise ValueError("source long-term memory record must be declared")
            if signal.source_style_profile_id not in self.source_style_profile_ids:
                raise ValueError("source style profile must be declared")
        return self


def build_project_memory(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    long_term_memory: LongTermCreativeMemoryPlan | None = None,
    style_profiles: StyleProfilePlan | None = None,
) -> ProjectMemoryPlan:
    """Build project memory metadata without storage writes or retrieval."""

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
    style_plan = style_profiles or build_style_profiles(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        long_term_memory=memory_plan,
        style_profiles=style_plan,
    )
    return ProjectMemoryPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        source_long_term_memory_serialization_version=memory_plan.serialization_version,
        source_style_profile_serialization_version=style_plan.serialization_version,
        source_long_term_memory_record_ids=memory_plan.record_ids,
        source_style_profile_ids=style_plan.profile_ids,
        execution_mode_ids=execution_modes.execution_mode_ids,
        signals=signals,
        signal_ids=tuple(signal.project_memory_id for signal in signals),
        candidate_signal_ids=_signal_ids_for_status(signals, "candidate"),
        review_required_signal_ids=_signal_ids_for_status(
            signals,
            "review_required",
        ),
        guarded_signal_ids=_signal_ids_for_status(signals, "guarded"),
        high_confidence_signal_ids=_signal_ids_for_confidence(
            signals,
            "high",
            "guarded",
        ),
        hitl_required_signal_ids=tuple(
            signal.project_memory_id
            for signal in signals
            if signal.hitl_required_before_storage
        ),
        stored_project_memory_ids=(),
        retrieved_project_memory_ids=(),
        consolidated_project_memory_ids=(),
        signal_count=len(signals),
        candidate_signal_count=len(_signal_ids_for_status(signals, "candidate")),
        review_required_signal_count=len(
            _signal_ids_for_status(signals, "review_required")
        ),
        guarded_signal_count=len(_signal_ids_for_status(signals, "guarded")),
        high_confidence_signal_count=len(
            _signal_ids_for_confidence(signals, "high", "guarded")
        ),
        hitl_required_signal_count=sum(
            1 for signal in signals if signal.hitl_required_before_storage
        ),
        highest_project_memory_score=max(
            signal.project_memory_score for signal in signals
        ),
        overall_project_memory_score=_overall_project_memory_score(signals),
        overall_project_memory_posture=_overall_project_memory_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def project_memory_signal_by_id(
    signal_id: str,
    plan: ProjectMemoryPlan | None = None,
) -> ProjectMemorySignal | None:
    """Return one project memory signal without retrieval execution."""

    source_plan = plan or build_project_memory()
    for signal in source_plan.signals:
        if signal.project_memory_id == signal_id:
            return signal
    return None


def project_memory_signals_for_status(
    status: ProjectMemoryStatus,
    plan: ProjectMemoryPlan | None = None,
) -> tuple[ProjectMemorySignal, ...]:
    """Return project memory signals by advisory status."""

    source_plan = plan or build_project_memory()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def project_memory_signals_for_confidence(
    confidence: ProjectMemoryConfidence,
    plan: ProjectMemoryPlan | None = None,
) -> tuple[ProjectMemorySignal, ...]:
    """Return project memory signals by confidence band."""

    source_plan = plan or build_project_memory()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    long_term_memory: LongTermCreativeMemoryPlan,
    style_profiles: StyleProfilePlan,
) -> tuple[ProjectMemorySignal, ...]:
    return (
        _signal(
            kind="project_goal_memory",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            memory_scope="goal",
            project_memory_kind=ProjectMemoryKind.GOAL,
            source_record_id="long_term_creative_memory::creative_intent_memory",
            source_profile_id="style_profiles::composition_profile",
            long_term_memory=long_term_memory,
            style_profiles=style_profiles,
            continuity_score=86,
            specificity_score=82,
            conflict_risk_score=44,
            governance_weight=150,
        ),
        _signal(
            kind="project_constraint_memory",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            memory_scope="constraint",
            project_memory_kind=ProjectMemoryKind.CONSTRAINT,
            source_record_id="long_term_creative_memory::project_context_memory",
            source_profile_id="style_profiles::composition_profile",
            long_term_memory=long_term_memory,
            style_profiles=style_profiles,
            continuity_score=78,
            specificity_score=80,
            conflict_risk_score=50,
            governance_weight=145,
        ),
        _signal(
            kind="project_decision_memory",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            memory_scope="decision",
            project_memory_kind=ProjectMemoryKind.DECISION,
            source_record_id="long_term_creative_memory::artifact_lineage_memory",
            source_profile_id="style_profiles::material_profile",
            long_term_memory=long_term_memory,
            style_profiles=style_profiles,
            continuity_score=70,
            specificity_score=72,
            conflict_risk_score=40,
            governance_weight=120,
        ),
        _signal(
            kind="project_style_memory",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            memory_scope="style",
            project_memory_kind=ProjectMemoryKind.STYLE,
            source_record_id="long_term_creative_memory::style_pattern_memory",
            source_profile_id="style_profiles::palette_profile",
            long_term_memory=long_term_memory,
            style_profiles=style_profiles,
            continuity_score=72,
            specificity_score=80,
            conflict_risk_score=24,
            governance_weight=120,
        ),
        _signal(
            kind="project_technical_memory",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            memory_scope="technical",
            project_memory_kind=ProjectMemoryKind.TECHNICAL_NOTE,
            source_record_id="long_term_creative_memory::project_context_memory",
            source_profile_id="style_profiles::material_profile",
            long_term_memory=long_term_memory,
            style_profiles=style_profiles,
            continuity_score=54,
            specificity_score=60,
            conflict_risk_score=20,
            governance_weight=80,
        ),
    )


def _signal(
    *,
    kind: ProjectMemoryFacetKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    memory_scope: ProjectMemoryScope,
    project_memory_kind: ProjectMemoryKind,
    source_record_id: str,
    source_profile_id: str,
    long_term_memory: LongTermCreativeMemoryPlan,
    style_profiles: StyleProfilePlan,
    continuity_score: int,
    specificity_score: int,
    conflict_risk_score: int,
    governance_weight: int,
) -> ProjectMemorySignal:
    source_record = long_term_creative_memory_record_by_id(
        source_record_id,
        long_term_memory,
    )
    source_profile = style_profile_by_id(source_profile_id, style_profiles)
    if source_record is None:
        raise ValueError("source long-term memory record must exist")
    if source_profile is None:
        raise ValueError("source style profile must exist")
    score = _project_memory_score(
        continuity_score=continuity_score,
        specificity_score=specificity_score,
        conflict_risk_score=conflict_risk_score,
        governance_weight=governance_weight,
    )
    status = _project_memory_status(score)
    confidence = _project_memory_confidence(score)
    return ProjectMemorySignal(
        project_memory_id=f"project_memory::{kind}",
        facet_kind=kind,
        status=status,
        confidence=confidence,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        memory_scope=memory_scope,
        project_memory_kind=project_memory_kind,
        source_long_term_memory_record_id=source_record.record_id,
        source_style_profile_id=source_profile.profile_id,
        project_memory_summary=_memory_summary(kind),
        continuity_score=continuity_score,
        specificity_score=specificity_score,
        conflict_risk_score=conflict_risk_score,
        governance_weight=governance_weight,
        project_memory_score=score,
        hitl_required_before_storage=True,
        context_tags=_context_tags(kind, memory_scope),
        explainability_notes=_explainability_notes(
            kind,
            source_record.record_id,
            source_profile.profile_id,
        ),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"source_long_term_memory:{source_record.record_id}",
            f"source_style_profile:{source_profile.profile_id}",
            f"project_memory_kind:{project_memory_kind.value}",
            f"memory_scope:{memory_scope}",
            f"continuity_score:{continuity_score}",
            f"specificity_score:{specificity_score}",
            f"conflict_risk_score:{conflict_risk_score}",
            "hitl_required_before_storage:true",
        ),
    )


def _project_memory_score(
    *,
    continuity_score: int,
    specificity_score: int,
    conflict_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            continuity_score * 3
            + specificity_score * 3
            + conflict_risk_score * 4
            + governance_weight,
        ),
    )


def _project_memory_status(score: int) -> ProjectMemoryStatus:
    if score >= 760:
        return "guarded"
    if score >= 600:
        return "review_required"
    return "candidate"


def _project_memory_confidence(score: int) -> ProjectMemoryConfidence:
    if score >= 760:
        return "guarded"
    if score >= 680:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_project_memory_score(
    signals: tuple[ProjectMemorySignal, ...],
) -> int:
    base = sum(signal.project_memory_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_project_memory_posture(
    signals: tuple[ProjectMemorySignal, ...],
) -> ProjectMemoryPosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[ProjectMemorySignal, ...],
    status: ProjectMemoryStatus,
) -> tuple[str, ...]:
    return tuple(
        signal.project_memory_id for signal in signals if signal.status == status
    )


def _signal_ids_for_confidence(
    signals: tuple[ProjectMemorySignal, ...],
    *confidences: ProjectMemoryConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.project_memory_id
        for signal in signals
        if signal.confidence in confidences
    )


def _plan_actions(signals: tuple[ProjectMemorySignal, ...]) -> tuple[str, ...]:
    guarded_signal_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_project_memory_candidates",
        "require_hitl_before_project_memory_storage",
        "keep_project_memory_retrieval_and_consolidation_non_executing",
        f"review_guarded_project_memory_count:{guarded_signal_count}",
    )


def _memory_summary(kind: ProjectMemoryFacetKind) -> str:
    summaries = {
        "project_goal_memory": "Models durable project goals as advisory memory.",
        "project_constraint_memory": "Models durable project constraints.",
        "project_decision_memory": "Models durable project decisions.",
        "project_style_memory": "Models durable project style context.",
        "project_technical_memory": "Models durable project technical notes.",
    }
    return summaries[kind]


def _context_tags(
    kind: ProjectMemoryFacetKind,
    memory_scope: ProjectMemoryScope,
) -> tuple[str, ...]:
    return (
        "creative_memory",
        "project_memory",
        memory_scope,
        kind.removesuffix("_memory"),
    )


def _explainability_notes(
    kind: ProjectMemoryFacetKind,
    source_record_id: str,
    source_profile_id: str,
) -> tuple[str, ...]:
    return (
        f"project_memory_kind:{kind}",
        f"source_record:{source_record_id}",
        f"source_style_profile:{source_profile_id}",
        "score_inputs:continuity,specificity,conflict_risk,governance",
        "storage_boundary:HITL_required_before_project_memory_write",
    )


def _signal_actions(kind: ProjectMemoryFacetKind) -> tuple[str, ...]:
    return (
        f"review_{kind}",
        "inspect_sources_before_project_memory_storage",
        "preserve_no_memory_retrieval_execution_boundary",
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
