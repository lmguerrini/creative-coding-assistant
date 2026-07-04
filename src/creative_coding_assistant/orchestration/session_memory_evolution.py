"""V6.2 advisory session memory evolution metadata."""

from __future__ import annotations

from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.creative_dna import (
    CreativeDNAPlan,
    build_creative_dna,
    creative_dna_signature_by_id,
)
from creative_coding_assistant.orchestration.long_term_creative_memory import (
    LongTermCreativeMemoryPlan,
    build_long_term_creative_memory,
    long_term_creative_memory_record_by_id,
)
from creative_coding_assistant.orchestration.personalization_engine import (
    PersonalizationEnginePlan,
    build_personalization_engine,
    personalization_recommendation_by_id,
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

SessionMemoryEvolutionKind = Literal[
    "intent_evolution",
    "style_evolution",
    "constraint_evolution",
    "technical_evolution",
    "review_evolution",
]
SessionMemoryEvolutionStatus = Literal["candidate", "review_required", "guarded"]
SessionMemoryEvolutionConfidence = Literal["low", "medium", "high", "guarded"]
SessionMemoryEvolutionPosture = Literal["candidate", "review_required", "guarded"]
SessionMemoryEvolutionScope = Literal[
    "intent",
    "style",
    "constraints",
    "technical_stack",
    "review_depth",
]

SESSION_MEMORY_EVOLUTION_SIGNAL_SERIALIZATION_VERSION = (
    "session_memory_evolution_signal.v1"
)
SESSION_MEMORY_EVOLUTION_PLAN_SERIALIZATION_VERSION = "session_memory_evolution_plan.v1"
SESSION_MEMORY_EVOLUTION_AUTHORITY_BOUNDARY = (
    "V6.2 Session Memory Evolution models session-to-memory change posture as "
    "inspectable advisory metadata only; it does not write session memory "
    "storage, create session memory records, update session memory records, "
    "delete session memory records, apply session memory evolution, record "
    "sessions, execute session replay, execute memory retrieval, write memory "
    "storage, consolidate memory, apply personalization, apply Creative DNA, "
    "mutate preferences, change provider or model routing, execute providers, "
    "invoke agents, control workflows, mutate workflow graphs, trigger retries "
    "or refinements, mutate prompts, modify generated output, or apply Runtime "
    "Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "session_memory_storage_write",
    "session_memory_record_creation",
    "session_memory_record_update",
    "session_memory_record_deletion",
    "session_memory_evolution_application",
    "session_recording",
    "session_replay_execution",
    "memory_retrieval_execution",
    "memory_storage_write",
    "automatic_memory_consolidation",
    "automatic_personalization_application",
    "creative_dna_application",
    "automatic_preference_mutation",
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


class SessionMemoryEvolutionSignal(BaseModel):
    """One advisory session memory evolution signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    evolution_id: str = Field(min_length=1, max_length=180)
    evolution_kind: SessionMemoryEvolutionKind
    status: SessionMemoryEvolutionStatus
    confidence: SessionMemoryEvolutionConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    evolution_scope: SessionMemoryEvolutionScope
    source_personalization_id: str = Field(min_length=1, max_length=180)
    source_creative_dna_id: str = Field(min_length=1, max_length=180)
    source_long_term_memory_record_id: str = Field(min_length=1, max_length=180)
    source_project_memory_signal_id: str = Field(min_length=1, max_length=180)
    evolution_summary: str = Field(min_length=1, max_length=360)
    session_continuity_score: int = Field(ge=0, le=100)
    personalization_alignment_score: int = Field(ge=0, le=100)
    memory_stability_score: int = Field(ge=0, le=100)
    drift_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    session_evolution_score: int = Field(ge=0, le=1_000)
    hitl_required_before_session_memory_update: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    session_memory_evolution_implemented: Literal[True] = True
    session_memory_evolution_metadata_implemented: Literal[True] = True
    personalization_source_used: Literal[True] = True
    creative_dna_source_used: Literal[True] = True
    long_term_memory_source_used: Literal[True] = True
    project_memory_source_used: Literal[True] = True
    session_memory_storage_write_implemented: Literal[False] = False
    session_memory_record_creation_implemented: Literal[False] = False
    session_memory_record_update_implemented: Literal[False] = False
    session_memory_record_deletion_implemented: Literal[False] = False
    session_memory_evolution_application_implemented: Literal[False] = False
    session_recording_implemented: Literal[False] = False
    session_replay_execution_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_storage_write_implemented: Literal[False] = False
    memory_consolidation_implemented: Literal[False] = False
    personalization_application_implemented: Literal[False] = False
    creative_dna_application_implemented: Literal[False] = False
    preference_mutation_implemented: Literal[False] = False
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
    serialization_version: Literal["session_memory_evolution_signal.v1"] = (
        SESSION_MEMORY_EVOLUTION_SIGNAL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.evolution_id != f"session_memory_evolution::{self.evolution_kind}":
            raise ValueError("evolution_id must match evolution_kind")
        if self.session_evolution_score != _session_evolution_score(
            session_continuity_score=self.session_continuity_score,
            personalization_alignment_score=self.personalization_alignment_score,
            memory_stability_score=self.memory_stability_score,
            drift_risk_score=self.drift_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("session_evolution_score must combine source scores")
        if self.status != _session_evolution_status(self.session_evolution_score):
            raise ValueError("status must match session_evolution_score")
        if self.confidence != _session_evolution_confidence(
            self.session_evolution_score
        ):
            raise ValueError("confidence must match session_evolution_score")
        if not self.hitl_required_before_session_memory_update:
            raise ValueError("session memory updates require HITL posture")
        return self


class SessionMemoryEvolutionPlan(BaseModel):
    """Bounded V6.2 advisory session memory evolution plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["session_memory_evolution"] = "session_memory_evolution"
    serialization_version: Literal["session_memory_evolution_plan.v1"] = (
        SESSION_MEMORY_EVOLUTION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=SESSION_MEMORY_EVOLUTION_AUTHORITY_BOUNDARY,
        max_length=1900,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_personalization_engine_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_creative_dna_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_long_term_memory_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_project_memory_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_personalization_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_creative_dna_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    source_long_term_memory_record_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_project_memory_signal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[SessionMemoryEvolutionSignal, ...] = Field(
        min_length=5,
        max_length=5,
    )
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
    evolved_session_memory_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    persisted_session_memory_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    replayed_session_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    consolidated_memory_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    signal_count: int = Field(ge=5, le=5)
    candidate_signal_count: int = Field(ge=0, le=5)
    review_required_signal_count: int = Field(ge=0, le=5)
    guarded_signal_count: int = Field(ge=0, le=5)
    high_confidence_signal_count: int = Field(ge=0, le=5)
    hitl_required_signal_count: int = Field(ge=0, le=5)
    highest_session_evolution_score: int = Field(ge=0, le=1_000)
    overall_session_evolution_score: int = Field(ge=0, le=1_000)
    overall_session_evolution_posture: SessionMemoryEvolutionPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    session_memory_evolution_implemented: Literal[True] = True
    session_memory_evolution_metadata_implemented: Literal[True] = True
    personalization_source_used: Literal[True] = True
    creative_dna_source_used: Literal[True] = True
    long_term_memory_source_used: Literal[True] = True
    project_memory_source_used: Literal[True] = True
    session_memory_storage_write_implemented: Literal[False] = False
    session_memory_record_creation_implemented: Literal[False] = False
    session_memory_record_update_implemented: Literal[False] = False
    session_memory_record_deletion_implemented: Literal[False] = False
    session_memory_evolution_application_implemented: Literal[False] = False
    session_recording_implemented: Literal[False] = False
    session_replay_execution_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_storage_write_implemented: Literal[False] = False
    memory_consolidation_implemented: Literal[False] = False
    personalization_application_implemented: Literal[False] = False
    creative_dna_application_implemented: Literal[False] = False
    preference_mutation_implemented: Literal[False] = False
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
        derived_signal_ids = tuple(signal.evolution_id for signal in self.signals)
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
            signal.evolution_id
            for signal in self.signals
            if signal.hitl_required_before_session_memory_update
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.evolved_session_memory_ids:
            raise ValueError("evolved_session_memory_ids must remain empty")
        if self.persisted_session_memory_ids:
            raise ValueError("persisted_session_memory_ids must remain empty")
        if self.replayed_session_ids:
            raise ValueError("replayed_session_ids must remain empty")
        if self.consolidated_memory_ids:
            raise ValueError("consolidated_memory_ids must remain empty")
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
        if self.highest_session_evolution_score != max(
            signal.session_evolution_score for signal in self.signals
        ):
            raise ValueError("highest_session_evolution_score must match signals")
        if self.overall_session_evolution_score != _overall_session_evolution_score(
            self.signals
        ):
            raise ValueError("overall_session_evolution_score must match signals")
        if self.overall_session_evolution_posture != (
            _overall_session_evolution_posture(self.signals)
        ):
            raise ValueError("overall_session_evolution_posture must match signals")
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.source_personalization_id not in self.source_personalization_ids:
                raise ValueError("source personalization must be declared")
            if signal.source_creative_dna_id not in self.source_creative_dna_ids:
                raise ValueError("source Creative DNA signature must be declared")
            if (
                signal.source_long_term_memory_record_id
                not in self.source_long_term_memory_record_ids
            ):
                raise ValueError("source long-term memory record must be declared")
            if (
                signal.source_project_memory_signal_id
                not in self.source_project_memory_signal_ids
            ):
                raise ValueError("source project memory signal must be declared")
        return self


def build_session_memory_evolution(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    personalization_engine: PersonalizationEnginePlan | None = None,
    creative_dna: CreativeDNAPlan | None = None,
    long_term_memory: LongTermCreativeMemoryPlan | None = None,
    project_memory: ProjectMemoryPlan | None = None,
) -> SessionMemoryEvolutionPlan:
    """Build session memory evolution metadata without session writes."""

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
    project_plan = project_memory or build_project_memory(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        long_term_memory=memory_plan,
    )
    dna_plan = creative_dna or build_creative_dna(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        long_term_memory=memory_plan,
        project_memory=project_plan,
    )
    personalization_plan = personalization_engine or build_personalization_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        creative_dna=dna_plan,
        project_memory=project_plan,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        personalization_engine=personalization_plan,
        creative_dna=dna_plan,
        long_term_memory=memory_plan,
        project_memory=project_plan,
    )
    return SessionMemoryEvolutionPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        source_personalization_engine_serialization_version=(
            personalization_plan.serialization_version
        ),
        source_creative_dna_serialization_version=dna_plan.serialization_version,
        source_long_term_memory_serialization_version=memory_plan.serialization_version,
        source_project_memory_serialization_version=project_plan.serialization_version,
        source_personalization_ids=personalization_plan.recommendation_ids,
        source_creative_dna_ids=dna_plan.signature_ids,
        source_long_term_memory_record_ids=memory_plan.record_ids,
        source_project_memory_signal_ids=project_plan.signal_ids,
        execution_mode_ids=execution_modes.execution_mode_ids,
        signals=signals,
        signal_ids=tuple(signal.evolution_id for signal in signals),
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
            signal.evolution_id
            for signal in signals
            if signal.hitl_required_before_session_memory_update
        ),
        evolved_session_memory_ids=(),
        persisted_session_memory_ids=(),
        replayed_session_ids=(),
        consolidated_memory_ids=(),
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
            1 for signal in signals if signal.hitl_required_before_session_memory_update
        ),
        highest_session_evolution_score=max(
            signal.session_evolution_score for signal in signals
        ),
        overall_session_evolution_score=_overall_session_evolution_score(signals),
        overall_session_evolution_posture=_overall_session_evolution_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def session_memory_evolution_signal_by_id(
    signal_id: str,
    plan: SessionMemoryEvolutionPlan | None = None,
) -> SessionMemoryEvolutionSignal | None:
    """Return one session memory evolution signal without applying it."""

    source_plan = plan or build_session_memory_evolution()
    for signal in source_plan.signals:
        if signal.evolution_id == signal_id:
            return signal
    return None


def session_memory_evolution_signals_for_status(
    status: SessionMemoryEvolutionStatus,
    plan: SessionMemoryEvolutionPlan | None = None,
) -> tuple[SessionMemoryEvolutionSignal, ...]:
    """Return session memory evolution signals by advisory status."""

    source_plan = plan or build_session_memory_evolution()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def session_memory_evolution_signals_for_confidence(
    confidence: SessionMemoryEvolutionConfidence,
    plan: SessionMemoryEvolutionPlan | None = None,
) -> tuple[SessionMemoryEvolutionSignal, ...]:
    """Return session memory evolution signals by confidence band."""

    source_plan = plan or build_session_memory_evolution()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    personalization_engine: PersonalizationEnginePlan,
    creative_dna: CreativeDNAPlan,
    long_term_memory: LongTermCreativeMemoryPlan,
    project_memory: ProjectMemoryPlan,
) -> tuple[SessionMemoryEvolutionSignal, ...]:
    return (
        _signal(
            kind="intent_evolution",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            evolution_scope="intent",
            source_personalization_id=(
                "personalization_engine::interaction_personalization"
            ),
            source_dna_id="creative_dna::intent_dna",
            source_record_id="long_term_creative_memory::creative_intent_memory",
            source_project_signal_id="project_memory::project_goal_memory",
            personalization_engine=personalization_engine,
            creative_dna=creative_dna,
            long_term_memory=long_term_memory,
            project_memory=project_memory,
            session_continuity_score=86,
            personalization_alignment_score=82,
            memory_stability_score=84,
            drift_risk_score=46,
            governance_weight=150,
        ),
        _signal(
            kind="style_evolution",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            evolution_scope="style",
            source_personalization_id="personalization_engine::style_personalization",
            source_dna_id="creative_dna::style_dna",
            source_record_id="long_term_creative_memory::style_pattern_memory",
            source_project_signal_id="project_memory::project_style_memory",
            personalization_engine=personalization_engine,
            creative_dna=creative_dna,
            long_term_memory=long_term_memory,
            project_memory=project_memory,
            session_continuity_score=82,
            personalization_alignment_score=88,
            memory_stability_score=78,
            drift_risk_score=44,
            governance_weight=140,
        ),
        _signal(
            kind="constraint_evolution",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            evolution_scope="constraints",
            source_personalization_id=(
                "personalization_engine::constraint_personalization"
            ),
            source_dna_id="creative_dna::constraint_dna",
            source_record_id="long_term_creative_memory::project_context_memory",
            source_project_signal_id="project_memory::project_constraint_memory",
            personalization_engine=personalization_engine,
            creative_dna=creative_dna,
            long_term_memory=long_term_memory,
            project_memory=project_memory,
            session_continuity_score=74,
            personalization_alignment_score=76,
            memory_stability_score=72,
            drift_risk_score=42,
            governance_weight=120,
        ),
        _signal(
            kind="technical_evolution",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            evolution_scope="technical_stack",
            source_personalization_id=(
                "personalization_engine::technical_personalization"
            ),
            source_dna_id="creative_dna::lineage_dna",
            source_record_id="long_term_creative_memory::artifact_lineage_memory",
            source_project_signal_id="project_memory::project_technical_memory",
            personalization_engine=personalization_engine,
            creative_dna=creative_dna,
            long_term_memory=long_term_memory,
            project_memory=project_memory,
            session_continuity_score=64,
            personalization_alignment_score=60,
            memory_stability_score=66,
            drift_risk_score=32,
            governance_weight=110,
        ),
        _signal(
            kind="review_evolution",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            evolution_scope="review_depth",
            source_personalization_id=(
                "personalization_engine::review_depth_personalization"
            ),
            source_dna_id="creative_dna::intent_dna",
            source_record_id="long_term_creative_memory::preference_signal_memory",
            source_project_signal_id="project_memory::project_decision_memory",
            personalization_engine=personalization_engine,
            creative_dna=creative_dna,
            long_term_memory=long_term_memory,
            project_memory=project_memory,
            session_continuity_score=50,
            personalization_alignment_score=52,
            memory_stability_score=54,
            drift_risk_score=18,
            governance_weight=85,
        ),
    )


def _signal(
    *,
    kind: SessionMemoryEvolutionKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    evolution_scope: SessionMemoryEvolutionScope,
    source_personalization_id: str,
    source_dna_id: str,
    source_record_id: str,
    source_project_signal_id: str,
    personalization_engine: PersonalizationEnginePlan,
    creative_dna: CreativeDNAPlan,
    long_term_memory: LongTermCreativeMemoryPlan,
    project_memory: ProjectMemoryPlan,
    session_continuity_score: int,
    personalization_alignment_score: int,
    memory_stability_score: int,
    drift_risk_score: int,
    governance_weight: int,
) -> SessionMemoryEvolutionSignal:
    source_personalization = personalization_recommendation_by_id(
        source_personalization_id,
        personalization_engine,
    )
    source_dna = creative_dna_signature_by_id(source_dna_id, creative_dna)
    source_record = long_term_creative_memory_record_by_id(
        source_record_id,
        long_term_memory,
    )
    source_project_signal = project_memory_signal_by_id(
        source_project_signal_id,
        project_memory,
    )
    if source_personalization is None:
        raise ValueError("source personalization must exist")
    if source_dna is None:
        raise ValueError("source Creative DNA signature must exist")
    if source_record is None:
        raise ValueError("source long-term memory record must exist")
    if source_project_signal is None:
        raise ValueError("source project memory signal must exist")
    score = _session_evolution_score(
        session_continuity_score=session_continuity_score,
        personalization_alignment_score=personalization_alignment_score,
        memory_stability_score=memory_stability_score,
        drift_risk_score=drift_risk_score,
        governance_weight=governance_weight,
    )
    status = _session_evolution_status(score)
    confidence = _session_evolution_confidence(score)
    return SessionMemoryEvolutionSignal(
        evolution_id=f"session_memory_evolution::{kind}",
        evolution_kind=kind,
        status=status,
        confidence=confidence,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        evolution_scope=evolution_scope,
        source_personalization_id=source_personalization.personalization_id,
        source_creative_dna_id=source_dna.creative_dna_id,
        source_long_term_memory_record_id=source_record.record_id,
        source_project_memory_signal_id=source_project_signal.project_memory_id,
        evolution_summary=_evolution_summary(kind),
        session_continuity_score=session_continuity_score,
        personalization_alignment_score=personalization_alignment_score,
        memory_stability_score=memory_stability_score,
        drift_risk_score=drift_risk_score,
        governance_weight=governance_weight,
        session_evolution_score=score,
        hitl_required_before_session_memory_update=True,
        context_tags=_context_tags(kind, evolution_scope),
        explainability_notes=_explainability_notes(
            kind,
            source_personalization.personalization_id,
            source_dna.creative_dna_id,
            source_record.record_id,
            source_project_signal.project_memory_id,
        ),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"source_personalization:{source_personalization.personalization_id}",
            f"source_creative_dna:{source_dna.creative_dna_id}",
            f"source_long_term_memory:{source_record.record_id}",
            f"source_project_memory:{source_project_signal.project_memory_id}",
            f"evolution_scope:{evolution_scope}",
            f"session_continuity_score:{session_continuity_score}",
            f"personalization_alignment_score:{personalization_alignment_score}",
            f"memory_stability_score:{memory_stability_score}",
            f"drift_risk_score:{drift_risk_score}",
            "hitl_required_before_session_memory_update:true",
        ),
    )


def _session_evolution_score(
    *,
    session_continuity_score: int,
    personalization_alignment_score: int,
    memory_stability_score: int,
    drift_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            session_continuity_score * 3
            + personalization_alignment_score * 3
            + memory_stability_score * 2
            + drift_risk_score * 3
            + governance_weight,
        ),
    )


def _session_evolution_status(score: int) -> SessionMemoryEvolutionStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _session_evolution_confidence(score: int) -> SessionMemoryEvolutionConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_session_evolution_score(
    signals: tuple[SessionMemoryEvolutionSignal, ...],
) -> int:
    base = sum(signal.session_evolution_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_session_evolution_posture(
    signals: tuple[SessionMemoryEvolutionSignal, ...],
) -> SessionMemoryEvolutionPosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[SessionMemoryEvolutionSignal, ...],
    status: SessionMemoryEvolutionStatus,
) -> tuple[str, ...]:
    return tuple(signal.evolution_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[SessionMemoryEvolutionSignal, ...],
    *confidences: SessionMemoryEvolutionConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.evolution_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(signals: tuple[SessionMemoryEvolutionSignal, ...]) -> tuple[str, ...]:
    guarded_signal_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_session_memory_evolution_signals",
        "require_hitl_before_session_memory_update",
        "keep_session_memory_evolution_non_executing",
        f"review_guarded_session_evolution_count:{guarded_signal_count}",
    )


def _evolution_summary(kind: SessionMemoryEvolutionKind) -> str:
    summaries = {
        "intent_evolution": "Models advisory session-to-intent memory evolution.",
        "style_evolution": "Models advisory session-to-style memory evolution.",
        "constraint_evolution": (
            "Models advisory session-to-constraint memory evolution."
        ),
        "technical_evolution": (
            "Models advisory session-to-technical memory evolution."
        ),
        "review_evolution": "Models advisory session-to-review memory evolution.",
    }
    return summaries[kind]


def _context_tags(
    kind: SessionMemoryEvolutionKind,
    evolution_scope: SessionMemoryEvolutionScope,
) -> tuple[str, ...]:
    return (
        "creative_memory",
        "session_memory_evolution",
        evolution_scope,
        kind.removesuffix("_evolution"),
    )


def _explainability_notes(
    kind: SessionMemoryEvolutionKind,
    source_personalization_id: str,
    source_dna_id: str,
    source_record_id: str,
    source_project_signal_id: str,
) -> tuple[str, ...]:
    return (
        f"session_evolution_kind:{kind}",
        f"source_personalization:{source_personalization_id}",
        f"source_creative_dna:{source_dna_id}",
        f"source_record:{source_record_id}",
        f"source_project_memory:{source_project_signal_id}",
        "score_inputs:session_continuity,personalization_alignment,memory_stability,drift_risk,governance",
        "update_boundary:HITL_required_before_session_memory_update",
    )


def _signal_actions(kind: SessionMemoryEvolutionKind) -> tuple[str, ...]:
    return (
        f"review_{kind}",
        "inspect_sources_before_session_memory_update",
        "preserve_no_session_memory_evolution_application_boundary",
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
