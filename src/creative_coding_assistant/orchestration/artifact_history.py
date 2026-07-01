"""V6.2 advisory artifact history metadata."""

from __future__ import annotations

from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.long_term_creative_memory import (
    LongTermCreativeMemoryPlan,
    build_long_term_creative_memory,
    long_term_creative_memory_record_by_id,
)
from creative_coding_assistant.orchestration.multimodal_studio import (
    MultimodalWorkspaceHistoryRegistry,
    multimodal_workspace_history_profile_by_id,
    multimodal_workspace_history_registry,
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
from creative_coding_assistant.orchestration.session_memory_evolution import (
    SessionMemoryEvolutionPlan,
    build_session_memory_evolution,
    session_memory_evolution_signal_by_id,
)

ArtifactHistoryKind = Literal[
    "iteration_history",
    "decision_history",
    "style_history",
    "runtime_history",
    "review_history",
]
ArtifactHistoryStatus = Literal["candidate", "review_required", "guarded"]
ArtifactHistoryConfidence = Literal["low", "medium", "high", "guarded"]
ArtifactHistoryPosture = Literal["candidate", "review_required", "guarded"]
ArtifactHistoryScope = Literal["iteration", "decision", "style", "runtime", "review"]

ARTIFACT_HISTORY_RECORD_SERIALIZATION_VERSION = "artifact_history_record.v1"
ARTIFACT_HISTORY_PLAN_SERIALIZATION_VERSION = "artifact_history_plan.v1"
ARTIFACT_HISTORY_AUTHORITY_BOUNDARY = (
    "V6.2 Artifact History models artifact history posture as inspectable "
    "advisory metadata only; it does not write artifact history storage, create "
    "artifact history records, update artifact history records, delete artifact "
    "history records, reconstruct artifact timelines, apply artifact history, "
    "record workspace history, persist workspace history, mutate artifacts, "
    "record sessions, execute session replay, execute memory retrieval, write "
    "memory storage, consolidate memory, apply personalization, change provider "
    "or model routing, execute providers, invoke agents, control workflows, "
    "mutate workflow graphs, trigger retries or refinements, mutate prompts, "
    "modify generated output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "artifact_history_storage_write",
    "artifact_history_record_creation",
    "artifact_history_record_update",
    "artifact_history_record_deletion",
    "artifact_history_reconstruction",
    "artifact_history_application",
    "workspace_history_recording",
    "workspace_history_persistence",
    "artifact_mutation",
    "session_recording",
    "session_replay_execution",
    "memory_retrieval_execution",
    "memory_storage_write",
    "automatic_memory_consolidation",
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


class ArtifactHistoryRecord(BaseModel):
    """One advisory artifact history record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    artifact_history_id: str = Field(min_length=1, max_length=180)
    history_kind: ArtifactHistoryKind
    status: ArtifactHistoryStatus
    confidence: ArtifactHistoryConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    history_scope: ArtifactHistoryScope
    source_session_evolution_signal_id: str = Field(min_length=1, max_length=180)
    source_project_memory_signal_id: str = Field(min_length=1, max_length=180)
    source_long_term_memory_record_id: str = Field(min_length=1, max_length=180)
    source_workspace_history_profile_id: str = Field(min_length=1, max_length=180)
    history_summary: str = Field(min_length=1, max_length=360)
    history_continuity_score: int = Field(ge=0, le=100)
    provenance_strength_score: int = Field(ge=0, le=100)
    session_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    artifact_history_score: int = Field(ge=0, le=1_000)
    hitl_required_before_history_persistence: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=30,
    )
    artifact_history_implemented: Literal[True] = True
    artifact_history_metadata_implemented: Literal[True] = True
    session_memory_evolution_source_used: Literal[True] = True
    project_memory_source_used: Literal[True] = True
    long_term_memory_source_used: Literal[True] = True
    workspace_history_source_used: Literal[True] = True
    artifact_history_storage_write_implemented: Literal[False] = False
    artifact_history_record_creation_implemented: Literal[False] = False
    artifact_history_record_update_implemented: Literal[False] = False
    artifact_history_record_deletion_implemented: Literal[False] = False
    artifact_history_reconstruction_implemented: Literal[False] = False
    artifact_history_application_implemented: Literal[False] = False
    workspace_history_recording_implemented: Literal[False] = False
    workspace_history_persistence_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    session_recording_implemented: Literal[False] = False
    session_replay_execution_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_storage_write_implemented: Literal[False] = False
    memory_consolidation_implemented: Literal[False] = False
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
    serialization_version: Literal["artifact_history_record.v1"] = (
        ARTIFACT_HISTORY_RECORD_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_matches_contract(self) -> Self:
        if self.artifact_history_id != f"artifact_history::{self.history_kind}":
            raise ValueError("artifact_history_id must match history_kind")
        if self.artifact_history_score != _artifact_history_score(
            history_continuity_score=self.history_continuity_score,
            provenance_strength_score=self.provenance_strength_score,
            session_alignment_score=self.session_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("artifact_history_score must combine source scores")
        if self.status != _artifact_history_status(self.artifact_history_score):
            raise ValueError("status must match artifact_history_score")
        if self.confidence != _artifact_history_confidence(
            self.artifact_history_score
        ):
            raise ValueError("confidence must match artifact_history_score")
        if not self.hitl_required_before_history_persistence:
            raise ValueError("artifact history persistence requires HITL posture")
        return self


class ArtifactHistoryPlan(BaseModel):
    """Bounded V6.2 advisory artifact history plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["artifact_history"] = "artifact_history"
    serialization_version: Literal["artifact_history_plan.v1"] = (
        ARTIFACT_HISTORY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ARTIFACT_HISTORY_AUTHORITY_BOUNDARY,
        max_length=1900,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_session_evolution_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_project_memory_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_long_term_memory_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_workspace_history_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_session_evolution_signal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_project_memory_signal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_long_term_memory_record_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_workspace_history_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    records: tuple[ArtifactHistoryRecord, ...] = Field(min_length=5, max_length=5)
    record_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    candidate_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    review_required_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    high_confidence_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    hitl_required_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    persisted_artifact_history_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    reconstructed_artifact_history_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_artifact_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    recorded_workspace_history_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    record_count: int = Field(ge=5, le=5)
    candidate_record_count: int = Field(ge=0, le=5)
    review_required_record_count: int = Field(ge=0, le=5)
    guarded_record_count: int = Field(ge=0, le=5)
    high_confidence_record_count: int = Field(ge=0, le=5)
    hitl_required_record_count: int = Field(ge=0, le=5)
    highest_artifact_history_score: int = Field(ge=0, le=1_000)
    overall_artifact_history_score: int = Field(ge=0, le=1_000)
    overall_artifact_history_posture: ArtifactHistoryPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=30,
    )
    artifact_history_implemented: Literal[True] = True
    artifact_history_metadata_implemented: Literal[True] = True
    session_memory_evolution_source_used: Literal[True] = True
    project_memory_source_used: Literal[True] = True
    long_term_memory_source_used: Literal[True] = True
    workspace_history_source_used: Literal[True] = True
    artifact_history_storage_write_implemented: Literal[False] = False
    artifact_history_record_creation_implemented: Literal[False] = False
    artifact_history_record_update_implemented: Literal[False] = False
    artifact_history_record_deletion_implemented: Literal[False] = False
    artifact_history_reconstruction_implemented: Literal[False] = False
    artifact_history_application_implemented: Literal[False] = False
    workspace_history_recording_implemented: Literal[False] = False
    workspace_history_persistence_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    session_recording_implemented: Literal[False] = False
    session_replay_execution_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_storage_write_implemented: Literal[False] = False
    memory_consolidation_implemented: Literal[False] = False
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
    def _plan_matches_records(self) -> Self:
        derived_record_ids = tuple(
            record.artifact_history_id for record in self.records
        )
        if len(set(derived_record_ids)) != len(derived_record_ids):
            raise ValueError("record_ids must be unique")
        if self.record_ids != derived_record_ids:
            raise ValueError("record_ids must match records")
        if self.candidate_record_ids != _record_ids_for_status(
            self.records,
            "candidate",
        ):
            raise ValueError("candidate_record_ids must match records")
        if self.review_required_record_ids != _record_ids_for_status(
            self.records,
            "review_required",
        ):
            raise ValueError("review_required_record_ids must match records")
        if self.guarded_record_ids != _record_ids_for_status(self.records, "guarded"):
            raise ValueError("guarded_record_ids must match records")
        if self.high_confidence_record_ids != _record_ids_for_confidence(
            self.records,
            "high",
            "guarded",
        ):
            raise ValueError("high_confidence_record_ids must match records")
        if self.hitl_required_record_ids != tuple(
            record.artifact_history_id
            for record in self.records
            if record.hitl_required_before_history_persistence
        ):
            raise ValueError("hitl_required_record_ids must match records")
        if self.persisted_artifact_history_ids:
            raise ValueError("persisted_artifact_history_ids must remain empty")
        if self.reconstructed_artifact_history_ids:
            raise ValueError("reconstructed_artifact_history_ids must remain empty")
        if self.mutated_artifact_ids:
            raise ValueError("mutated_artifact_ids must remain empty")
        if self.recorded_workspace_history_ids:
            raise ValueError("recorded_workspace_history_ids must remain empty")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.candidate_record_count != len(self.candidate_record_ids):
            raise ValueError("candidate_record_count must match records")
        if self.review_required_record_count != len(
            self.review_required_record_ids
        ):
            raise ValueError("review_required_record_count must match records")
        if self.guarded_record_count != len(self.guarded_record_ids):
            raise ValueError("guarded_record_count must match records")
        if self.high_confidence_record_count != len(self.high_confidence_record_ids):
            raise ValueError("high_confidence_record_count must match records")
        if self.hitl_required_record_count != len(self.hitl_required_record_ids):
            raise ValueError("hitl_required_record_count must match records")
        if self.highest_artifact_history_score != max(
            record.artifact_history_score for record in self.records
        ):
            raise ValueError("highest_artifact_history_score must match records")
        if self.overall_artifact_history_score != _overall_artifact_history_score(
            self.records
        ):
            raise ValueError("overall_artifact_history_score must match records")
        if self.overall_artifact_history_posture != _overall_artifact_history_posture(
            self.records
        ):
            raise ValueError("overall_artifact_history_posture must match records")
        for record in self.records:
            if record.route_name != self.route_name:
                raise ValueError("record route_name must match plan")
            if (
                record.source_session_evolution_signal_id
                not in self.source_session_evolution_signal_ids
            ):
                raise ValueError("source session evolution signal must be declared")
            if (
                record.source_project_memory_signal_id
                not in self.source_project_memory_signal_ids
            ):
                raise ValueError("source project memory signal must be declared")
            if (
                record.source_long_term_memory_record_id
                not in self.source_long_term_memory_record_ids
            ):
                raise ValueError("source long-term memory record must be declared")
            if (
                record.source_workspace_history_profile_id
                not in self.source_workspace_history_profile_ids
            ):
                raise ValueError("source workspace history profile must be declared")
        return self


def build_artifact_history(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    session_memory_evolution: SessionMemoryEvolutionPlan | None = None,
    project_memory: ProjectMemoryPlan | None = None,
    long_term_memory: LongTermCreativeMemoryPlan | None = None,
    workspace_history: MultimodalWorkspaceHistoryRegistry | None = None,
) -> ArtifactHistoryPlan:
    """Build artifact history metadata without recording or reconstructing history."""

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
    session_plan = session_memory_evolution or build_session_memory_evolution(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        long_term_memory=memory_plan,
        project_memory=project_plan,
    )
    workspace_registry = workspace_history or multimodal_workspace_history_registry()
    records = _records(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        session_memory_evolution=session_plan,
        project_memory=project_plan,
        long_term_memory=memory_plan,
        workspace_history=workspace_registry,
    )
    return ArtifactHistoryPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        source_session_evolution_serialization_version=(
            session_plan.serialization_version
        ),
        source_project_memory_serialization_version=project_plan.serialization_version,
        source_long_term_memory_serialization_version=memory_plan.serialization_version,
        source_workspace_history_serialization_version=(
            workspace_registry.serialization_version
        ),
        source_session_evolution_signal_ids=session_plan.signal_ids,
        source_project_memory_signal_ids=project_plan.signal_ids,
        source_long_term_memory_record_ids=memory_plan.record_ids,
        source_workspace_history_profile_ids=workspace_registry.profile_ids,
        execution_mode_ids=execution_modes.execution_mode_ids,
        records=records,
        record_ids=tuple(record.artifact_history_id for record in records),
        candidate_record_ids=_record_ids_for_status(records, "candidate"),
        review_required_record_ids=_record_ids_for_status(
            records,
            "review_required",
        ),
        guarded_record_ids=_record_ids_for_status(records, "guarded"),
        high_confidence_record_ids=_record_ids_for_confidence(
            records,
            "high",
            "guarded",
        ),
        hitl_required_record_ids=tuple(
            record.artifact_history_id
            for record in records
            if record.hitl_required_before_history_persistence
        ),
        persisted_artifact_history_ids=(),
        reconstructed_artifact_history_ids=(),
        mutated_artifact_ids=(),
        recorded_workspace_history_ids=(),
        record_count=len(records),
        candidate_record_count=len(_record_ids_for_status(records, "candidate")),
        review_required_record_count=len(
            _record_ids_for_status(records, "review_required")
        ),
        guarded_record_count=len(_record_ids_for_status(records, "guarded")),
        high_confidence_record_count=len(
            _record_ids_for_confidence(records, "high", "guarded")
        ),
        hitl_required_record_count=sum(
            1 for record in records if record.hitl_required_before_history_persistence
        ),
        highest_artifact_history_score=max(
            record.artifact_history_score for record in records
        ),
        overall_artifact_history_score=_overall_artifact_history_score(records),
        overall_artifact_history_posture=_overall_artifact_history_posture(records),
        advisory_actions=_plan_actions(records),
    )


def artifact_history_record_by_id(
    record_id: str,
    plan: ArtifactHistoryPlan | None = None,
) -> ArtifactHistoryRecord | None:
    """Return one artifact history record without reconstructing history."""

    source_plan = plan or build_artifact_history()
    for record in source_plan.records:
        if record.artifact_history_id == record_id:
            return record
    return None


def artifact_history_records_for_status(
    status: ArtifactHistoryStatus,
    plan: ArtifactHistoryPlan | None = None,
) -> tuple[ArtifactHistoryRecord, ...]:
    """Return artifact history records by advisory status."""

    source_plan = plan or build_artifact_history()
    return tuple(record for record in source_plan.records if record.status == status)


def artifact_history_records_for_confidence(
    confidence: ArtifactHistoryConfidence,
    plan: ArtifactHistoryPlan | None = None,
) -> tuple[ArtifactHistoryRecord, ...]:
    """Return artifact history records by confidence band."""

    source_plan = plan or build_artifact_history()
    return tuple(
        record for record in source_plan.records if record.confidence == confidence
    )


def _records(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    session_memory_evolution: SessionMemoryEvolutionPlan,
    project_memory: ProjectMemoryPlan,
    long_term_memory: LongTermCreativeMemoryPlan,
    workspace_history: MultimodalWorkspaceHistoryRegistry,
) -> tuple[ArtifactHistoryRecord, ...]:
    return (
        _record(
            kind="iteration_history",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            history_scope="iteration",
            source_session_signal_id="session_memory_evolution::intent_evolution",
            source_project_signal_id="project_memory::project_decision_memory",
            source_record_id="long_term_creative_memory::artifact_lineage_memory",
            source_workspace_profile_id="artifact_board_workspace_history",
            session_memory_evolution=session_memory_evolution,
            project_memory=project_memory,
            long_term_memory=long_term_memory,
            workspace_history=workspace_history,
            history_continuity_score=86,
            provenance_strength_score=82,
            session_alignment_score=84,
            mutation_risk_score=46,
            governance_weight=150,
        ),
        _record(
            kind="decision_history",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            history_scope="decision",
            source_session_signal_id="session_memory_evolution::constraint_evolution",
            source_project_signal_id="project_memory::project_decision_memory",
            source_record_id="long_term_creative_memory::artifact_lineage_memory",
            source_workspace_profile_id="snapshot_workspace_history",
            session_memory_evolution=session_memory_evolution,
            project_memory=project_memory,
            long_term_memory=long_term_memory,
            workspace_history=workspace_history,
            history_continuity_score=82,
            provenance_strength_score=78,
            session_alignment_score=80,
            mutation_risk_score=44,
            governance_weight=140,
        ),
        _record(
            kind="style_history",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            history_scope="style",
            source_session_signal_id="session_memory_evolution::style_evolution",
            source_project_signal_id="project_memory::project_style_memory",
            source_record_id="long_term_creative_memory::style_pattern_memory",
            source_workspace_profile_id="artifact_board_workspace_history",
            session_memory_evolution=session_memory_evolution,
            project_memory=project_memory,
            long_term_memory=long_term_memory,
            workspace_history=workspace_history,
            history_continuity_score=74,
            provenance_strength_score=76,
            session_alignment_score=72,
            mutation_risk_score=42,
            governance_weight=120,
        ),
        _record(
            kind="runtime_history",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            history_scope="runtime",
            source_session_signal_id="session_memory_evolution::technical_evolution",
            source_project_signal_id="project_memory::project_technical_memory",
            source_record_id="long_term_creative_memory::project_context_memory",
            source_workspace_profile_id="runtime_event_workspace_history",
            session_memory_evolution=session_memory_evolution,
            project_memory=project_memory,
            long_term_memory=long_term_memory,
            workspace_history=workspace_history,
            history_continuity_score=64,
            provenance_strength_score=60,
            session_alignment_score=66,
            mutation_risk_score=32,
            governance_weight=110,
        ),
        _record(
            kind="review_history",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            history_scope="review",
            source_session_signal_id="session_memory_evolution::review_evolution",
            source_project_signal_id="project_memory::project_goal_memory",
            source_record_id="long_term_creative_memory::preference_signal_memory",
            source_workspace_profile_id="session_record_workspace_history",
            session_memory_evolution=session_memory_evolution,
            project_memory=project_memory,
            long_term_memory=long_term_memory,
            workspace_history=workspace_history,
            history_continuity_score=50,
            provenance_strength_score=52,
            session_alignment_score=54,
            mutation_risk_score=18,
            governance_weight=85,
        ),
    )


def _record(
    *,
    kind: ArtifactHistoryKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    history_scope: ArtifactHistoryScope,
    source_session_signal_id: str,
    source_project_signal_id: str,
    source_record_id: str,
    source_workspace_profile_id: str,
    session_memory_evolution: SessionMemoryEvolutionPlan,
    project_memory: ProjectMemoryPlan,
    long_term_memory: LongTermCreativeMemoryPlan,
    workspace_history: MultimodalWorkspaceHistoryRegistry,
    history_continuity_score: int,
    provenance_strength_score: int,
    session_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> ArtifactHistoryRecord:
    source_session_signal = session_memory_evolution_signal_by_id(
        source_session_signal_id,
        session_memory_evolution,
    )
    source_project_signal = project_memory_signal_by_id(
        source_project_signal_id,
        project_memory,
    )
    source_record = long_term_creative_memory_record_by_id(
        source_record_id,
        long_term_memory,
    )
    source_workspace_profile = multimodal_workspace_history_profile_by_id(
        source_workspace_profile_id,
        workspace_history,
    )
    if source_session_signal is None:
        raise ValueError("source session evolution signal must exist")
    if source_project_signal is None:
        raise ValueError("source project memory signal must exist")
    if source_record is None:
        raise ValueError("source long-term memory record must exist")
    if source_workspace_profile is None:
        raise ValueError("source workspace history profile must exist")
    score = _artifact_history_score(
        history_continuity_score=history_continuity_score,
        provenance_strength_score=provenance_strength_score,
        session_alignment_score=session_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    status = _artifact_history_status(score)
    confidence = _artifact_history_confidence(score)
    return ArtifactHistoryRecord(
        artifact_history_id=f"artifact_history::{kind}",
        history_kind=kind,
        status=status,
        confidence=confidence,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        history_scope=history_scope,
        source_session_evolution_signal_id=source_session_signal.evolution_id,
        source_project_memory_signal_id=source_project_signal.project_memory_id,
        source_long_term_memory_record_id=source_record.record_id,
        source_workspace_history_profile_id=source_workspace_profile.profile_id,
        history_summary=_history_summary(kind),
        history_continuity_score=history_continuity_score,
        provenance_strength_score=provenance_strength_score,
        session_alignment_score=session_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        artifact_history_score=score,
        hitl_required_before_history_persistence=True,
        context_tags=_context_tags(kind, history_scope),
        explainability_notes=_explainability_notes(
            kind,
            source_session_signal.evolution_id,
            source_project_signal.project_memory_id,
            source_record.record_id,
            source_workspace_profile.profile_id,
        ),
        advisory_actions=_record_actions(kind),
        evidence=(
            f"source_session_evolution:{source_session_signal.evolution_id}",
            f"source_project_memory:{source_project_signal.project_memory_id}",
            f"source_long_term_memory:{source_record.record_id}",
            f"source_workspace_history:{source_workspace_profile.profile_id}",
            f"history_scope:{history_scope}",
            f"history_continuity_score:{history_continuity_score}",
            f"provenance_strength_score:{provenance_strength_score}",
            f"session_alignment_score:{session_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_history_persistence:true",
        ),
    )


def _artifact_history_score(
    *,
    history_continuity_score: int,
    provenance_strength_score: int,
    session_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            history_continuity_score * 3
            + provenance_strength_score * 3
            + session_alignment_score * 2
            + mutation_risk_score * 3
            + governance_weight,
        ),
    )


def _artifact_history_status(score: int) -> ArtifactHistoryStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _artifact_history_confidence(score: int) -> ArtifactHistoryConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_artifact_history_score(
    records: tuple[ArtifactHistoryRecord, ...],
) -> int:
    base = sum(record.artifact_history_score for record in records) // len(records)
    guarded_count = len(_record_ids_for_status(records, "guarded"))
    review_count = len(_record_ids_for_status(records, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_artifact_history_posture(
    records: tuple[ArtifactHistoryRecord, ...],
) -> ArtifactHistoryPosture:
    if any(record.status == "guarded" for record in records):
        return "guarded"
    if any(record.status == "review_required" for record in records):
        return "review_required"
    return "candidate"


def _record_ids_for_status(
    records: tuple[ArtifactHistoryRecord, ...],
    status: ArtifactHistoryStatus,
) -> tuple[str, ...]:
    return tuple(
        record.artifact_history_id for record in records if record.status == status
    )


def _record_ids_for_confidence(
    records: tuple[ArtifactHistoryRecord, ...],
    *confidences: ArtifactHistoryConfidence,
) -> tuple[str, ...]:
    return tuple(
        record.artifact_history_id
        for record in records
        if record.confidence in confidences
    )


def _plan_actions(records: tuple[ArtifactHistoryRecord, ...]) -> tuple[str, ...]:
    guarded_record_count = len(_record_ids_for_status(records, "guarded"))
    return (
        "inspect_artifact_history_records",
        "require_hitl_before_artifact_history_persistence",
        "keep_artifact_history_non_reconstructing",
        f"review_guarded_artifact_history_count:{guarded_record_count}",
    )


def _history_summary(kind: ArtifactHistoryKind) -> str:
    summaries = {
        "iteration_history": "Models advisory artifact iteration history.",
        "decision_history": "Models advisory artifact decision history.",
        "style_history": "Models advisory artifact style history.",
        "runtime_history": "Models advisory artifact runtime history.",
        "review_history": "Models advisory artifact review history.",
    }
    return summaries[kind]


def _context_tags(
    kind: ArtifactHistoryKind,
    history_scope: ArtifactHistoryScope,
) -> tuple[str, ...]:
    return (
        "creative_memory",
        "artifact_history",
        history_scope,
        kind.removesuffix("_history"),
    )


def _explainability_notes(
    kind: ArtifactHistoryKind,
    source_session_signal_id: str,
    source_project_signal_id: str,
    source_record_id: str,
    source_workspace_profile_id: str,
) -> tuple[str, ...]:
    return (
        f"artifact_history_kind:{kind}",
        f"source_session_evolution:{source_session_signal_id}",
        f"source_project_memory:{source_project_signal_id}",
        f"source_record:{source_record_id}",
        f"source_workspace_history:{source_workspace_profile_id}",
        "score_inputs:history_continuity,provenance_strength,session_alignment,mutation_risk,governance",
        "persistence_boundary:HITL_required_before_artifact_history_persistence",
    )


def _record_actions(kind: ArtifactHistoryKind) -> tuple[str, ...]:
    return (
        f"review_{kind}",
        "inspect_sources_before_artifact_history_persistence",
        "preserve_no_artifact_history_reconstruction_boundary",
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
