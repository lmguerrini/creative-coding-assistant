"""V6.2 advisory creative lineage metadata."""

from __future__ import annotations

from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.artifact_history import (
    ArtifactHistoryPlan,
    artifact_history_record_by_id,
    build_artifact_history,
)
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
from creative_coding_assistant.orchestration.multimodal_studio import (
    MultimodalArtifactLineageRegistry,
    multimodal_artifact_lineage_profile_by_id,
    multimodal_artifact_lineage_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

CreativeLineageKind = Literal[
    "artifact_dependency_lineage",
    "timeline_stage_lineage",
    "source_transition_lineage",
    "style_evolution_lineage",
    "gap_review_lineage",
]
CreativeLineageStatus = Literal["candidate", "review_required", "guarded"]
CreativeLineageConfidence = Literal["low", "medium", "high", "guarded"]
CreativeLineagePosture = Literal["candidate", "review_required", "guarded"]
CreativeLineageAxis = Literal[
    "dependency",
    "timeline",
    "source_transition",
    "style_evolution",
    "gap_review",
]

CREATIVE_LINEAGE_RECORD_SERIALIZATION_VERSION = "creative_lineage_record.v1"
CREATIVE_LINEAGE_PLAN_SERIALIZATION_VERSION = "creative_lineage_plan.v1"
CREATIVE_LINEAGE_AUTHORITY_BOUNDARY = (
    "V6.2 Creative Lineage models creative lineage posture as inspectable "
    "advisory metadata only; it does not write creative lineage storage, create "
    "creative lineage records, update creative lineage records, delete creative "
    "lineage records, infer lineage, persist lineage, reconstruct timelines, "
    "materialize dependency graphs, record provenance, mutate artifacts, record "
    "sessions, execute session replay, execute memory retrieval, write memory "
    "storage, consolidate memory, apply Creative DNA, apply personalization, "
    "change provider or model routing, execute providers, invoke agents, control "
    "workflows, mutate workflow graphs, trigger retries or refinements, mutate "
    "prompts, modify generated output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "creative_lineage_storage_write",
    "creative_lineage_record_creation",
    "creative_lineage_record_update",
    "creative_lineage_record_deletion",
    "lineage_inference",
    "lineage_persistence",
    "timeline_reconstruction",
    "dependency_graph_materialization",
    "provenance_recording",
    "artifact_mutation",
    "session_recording",
    "session_replay_execution",
    "memory_retrieval_execution",
    "memory_storage_write",
    "automatic_memory_consolidation",
    "creative_dna_application",
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


class CreativeLineageRecord(BaseModel):
    """One advisory creative lineage record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    creative_lineage_id: str = Field(min_length=1, max_length=180)
    lineage_kind: CreativeLineageKind
    status: CreativeLineageStatus
    confidence: CreativeLineageConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    lineage_axis: CreativeLineageAxis
    source_artifact_history_record_id: str = Field(min_length=1, max_length=180)
    source_creative_dna_signature_id: str = Field(min_length=1, max_length=180)
    source_long_term_memory_record_id: str = Field(min_length=1, max_length=180)
    source_artifact_lineage_profile_id: str = Field(min_length=1, max_length=180)
    lineage_summary: str = Field(min_length=1, max_length=360)
    continuity_score: int = Field(ge=0, le=100)
    dependency_visibility_score: int = Field(ge=0, le=100)
    provenance_trace_score: int = Field(ge=0, le=100)
    governance_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    creative_lineage_score: int = Field(ge=0, le=1_000)
    hitl_required_before_lineage_persistence: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=34,
    )
    creative_lineage_implemented: Literal[True] = True
    creative_lineage_metadata_implemented: Literal[True] = True
    artifact_history_source_used: Literal[True] = True
    creative_dna_source_used: Literal[True] = True
    long_term_memory_source_used: Literal[True] = True
    artifact_lineage_registry_source_used: Literal[True] = True
    creative_lineage_storage_write_implemented: Literal[False] = False
    creative_lineage_record_creation_implemented: Literal[False] = False
    creative_lineage_record_update_implemented: Literal[False] = False
    creative_lineage_record_deletion_implemented: Literal[False] = False
    lineage_inference_implemented: Literal[False] = False
    lineage_persistence_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    dependency_graph_materialization_implemented: Literal[False] = False
    provenance_recording_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    session_recording_implemented: Literal[False] = False
    session_replay_execution_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_storage_write_implemented: Literal[False] = False
    memory_consolidation_implemented: Literal[False] = False
    creative_dna_application_implemented: Literal[False] = False
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
    serialization_version: Literal["creative_lineage_record.v1"] = (
        CREATIVE_LINEAGE_RECORD_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_matches_contract(self) -> Self:
        if self.creative_lineage_id != f"creative_lineage::{self.lineage_kind}":
            raise ValueError("creative_lineage_id must match lineage_kind")
        if self.creative_lineage_score != _creative_lineage_score(
            continuity_score=self.continuity_score,
            dependency_visibility_score=self.dependency_visibility_score,
            provenance_trace_score=self.provenance_trace_score,
            governance_risk_score=self.governance_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("creative_lineage_score must combine source scores")
        if self.status != _creative_lineage_status(self.creative_lineage_score):
            raise ValueError("status must match creative_lineage_score")
        if self.confidence != _creative_lineage_confidence(self.creative_lineage_score):
            raise ValueError("confidence must match creative_lineage_score")
        if not self.hitl_required_before_lineage_persistence:
            raise ValueError("creative lineage persistence requires HITL posture")
        return self


class CreativeLineagePlan(BaseModel):
    """Bounded V6.2 advisory creative lineage plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_lineage"] = "creative_lineage"
    serialization_version: Literal["creative_lineage_plan.v1"] = (
        CREATIVE_LINEAGE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_LINEAGE_AUTHORITY_BOUNDARY,
        max_length=2100,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_artifact_history_serialization_version: str = Field(
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
    source_artifact_lineage_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_artifact_history_record_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_creative_dna_signature_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_long_term_memory_record_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_artifact_lineage_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    records: tuple[CreativeLineageRecord, ...] = Field(min_length=5, max_length=5)
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
    persisted_creative_lineage_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    inferred_creative_lineage_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    reconstructed_lineage_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    materialized_dependency_graph_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_artifact_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    record_count: int = Field(ge=5, le=5)
    candidate_record_count: int = Field(ge=0, le=5)
    review_required_record_count: int = Field(ge=0, le=5)
    guarded_record_count: int = Field(ge=0, le=5)
    high_confidence_record_count: int = Field(ge=0, le=5)
    hitl_required_record_count: int = Field(ge=0, le=5)
    highest_creative_lineage_score: int = Field(ge=0, le=1_000)
    overall_creative_lineage_score: int = Field(ge=0, le=1_000)
    overall_creative_lineage_posture: CreativeLineagePosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=34,
    )
    creative_lineage_implemented: Literal[True] = True
    creative_lineage_metadata_implemented: Literal[True] = True
    artifact_history_source_used: Literal[True] = True
    creative_dna_source_used: Literal[True] = True
    long_term_memory_source_used: Literal[True] = True
    artifact_lineage_registry_source_used: Literal[True] = True
    creative_lineage_storage_write_implemented: Literal[False] = False
    creative_lineage_record_creation_implemented: Literal[False] = False
    creative_lineage_record_update_implemented: Literal[False] = False
    creative_lineage_record_deletion_implemented: Literal[False] = False
    lineage_inference_implemented: Literal[False] = False
    lineage_persistence_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    dependency_graph_materialization_implemented: Literal[False] = False
    provenance_recording_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    session_recording_implemented: Literal[False] = False
    session_replay_execution_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_storage_write_implemented: Literal[False] = False
    memory_consolidation_implemented: Literal[False] = False
    creative_dna_application_implemented: Literal[False] = False
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
            record.creative_lineage_id for record in self.records
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
            record.creative_lineage_id
            for record in self.records
            if record.hitl_required_before_lineage_persistence
        ):
            raise ValueError("hitl_required_record_ids must match records")
        if self.persisted_creative_lineage_ids:
            raise ValueError("persisted_creative_lineage_ids must remain empty")
        if self.inferred_creative_lineage_ids:
            raise ValueError("inferred_creative_lineage_ids must remain empty")
        if self.reconstructed_lineage_ids:
            raise ValueError("reconstructed_lineage_ids must remain empty")
        if self.materialized_dependency_graph_ids:
            raise ValueError("materialized_dependency_graph_ids must remain empty")
        if self.mutated_artifact_ids:
            raise ValueError("mutated_artifact_ids must remain empty")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.candidate_record_count != len(self.candidate_record_ids):
            raise ValueError("candidate_record_count must match records")
        if self.review_required_record_count != len(self.review_required_record_ids):
            raise ValueError("review_required_record_count must match records")
        if self.guarded_record_count != len(self.guarded_record_ids):
            raise ValueError("guarded_record_count must match records")
        if self.high_confidence_record_count != len(self.high_confidence_record_ids):
            raise ValueError("high_confidence_record_count must match records")
        if self.hitl_required_record_count != len(self.hitl_required_record_ids):
            raise ValueError("hitl_required_record_count must match records")
        if self.highest_creative_lineage_score != max(
            record.creative_lineage_score for record in self.records
        ):
            raise ValueError("highest_creative_lineage_score must match records")
        if self.overall_creative_lineage_score != _overall_creative_lineage_score(
            self.records
        ):
            raise ValueError("overall_creative_lineage_score must match records")
        if self.overall_creative_lineage_posture != _overall_creative_lineage_posture(
            self.records
        ):
            raise ValueError("overall_creative_lineage_posture must match records")
        for record in self.records:
            if record.route_name != self.route_name:
                raise ValueError("record route_name must match plan")
            if (
                record.source_artifact_history_record_id
                not in self.source_artifact_history_record_ids
            ):
                raise ValueError("source artifact history record must be declared")
            if (
                record.source_creative_dna_signature_id
                not in self.source_creative_dna_signature_ids
            ):
                raise ValueError("source Creative DNA signature must be declared")
            if (
                record.source_long_term_memory_record_id
                not in self.source_long_term_memory_record_ids
            ):
                raise ValueError("source long-term memory record must be declared")
            if (
                record.source_artifact_lineage_profile_id
                not in self.source_artifact_lineage_profile_ids
            ):
                raise ValueError("source artifact lineage profile must be declared")
        return self


def build_creative_lineage(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    artifact_history: ArtifactHistoryPlan | None = None,
    creative_dna: CreativeDNAPlan | None = None,
    long_term_memory: LongTermCreativeMemoryPlan | None = None,
    artifact_lineage: MultimodalArtifactLineageRegistry | None = None,
) -> CreativeLineagePlan:
    """Build creative lineage metadata without inferring or persisting lineage."""

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
    dna_plan = creative_dna or build_creative_dna(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        long_term_memory=memory_plan,
    )
    history_plan = artifact_history or build_artifact_history(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        long_term_memory=memory_plan,
    )
    lineage_registry = artifact_lineage or multimodal_artifact_lineage_registry()
    records = _records(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        artifact_history=history_plan,
        creative_dna=dna_plan,
        long_term_memory=memory_plan,
        artifact_lineage=lineage_registry,
    )
    return CreativeLineagePlan(
        route_name=route_name,
        task_type=normalized_task_type,
        source_artifact_history_serialization_version=(
            history_plan.serialization_version
        ),
        source_creative_dna_serialization_version=dna_plan.serialization_version,
        source_long_term_memory_serialization_version=memory_plan.serialization_version,
        source_artifact_lineage_serialization_version=(
            lineage_registry.serialization_version
        ),
        source_artifact_history_record_ids=history_plan.record_ids,
        source_creative_dna_signature_ids=dna_plan.signature_ids,
        source_long_term_memory_record_ids=memory_plan.record_ids,
        source_artifact_lineage_profile_ids=lineage_registry.profile_ids,
        execution_mode_ids=execution_modes.execution_mode_ids,
        records=records,
        record_ids=tuple(record.creative_lineage_id for record in records),
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
            record.creative_lineage_id
            for record in records
            if record.hitl_required_before_lineage_persistence
        ),
        persisted_creative_lineage_ids=(),
        inferred_creative_lineage_ids=(),
        reconstructed_lineage_ids=(),
        materialized_dependency_graph_ids=(),
        mutated_artifact_ids=(),
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
            1 for record in records if record.hitl_required_before_lineage_persistence
        ),
        highest_creative_lineage_score=max(
            record.creative_lineage_score for record in records
        ),
        overall_creative_lineage_score=_overall_creative_lineage_score(records),
        overall_creative_lineage_posture=_overall_creative_lineage_posture(records),
        advisory_actions=_plan_actions(records),
    )


def creative_lineage_record_by_id(
    record_id: str,
    plan: CreativeLineagePlan | None = None,
) -> CreativeLineageRecord | None:
    """Return one creative lineage record without inferring lineage."""

    source_plan = plan or build_creative_lineage()
    for record in source_plan.records:
        if record.creative_lineage_id == record_id:
            return record
    return None


def creative_lineage_records_for_status(
    status: CreativeLineageStatus,
    plan: CreativeLineagePlan | None = None,
) -> tuple[CreativeLineageRecord, ...]:
    """Return creative lineage records by advisory status."""

    source_plan = plan or build_creative_lineage()
    return tuple(record for record in source_plan.records if record.status == status)


def creative_lineage_records_for_confidence(
    confidence: CreativeLineageConfidence,
    plan: CreativeLineagePlan | None = None,
) -> tuple[CreativeLineageRecord, ...]:
    """Return creative lineage records by confidence band."""

    source_plan = plan or build_creative_lineage()
    return tuple(
        record for record in source_plan.records if record.confidence == confidence
    )


def _records(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    artifact_history: ArtifactHistoryPlan,
    creative_dna: CreativeDNAPlan,
    long_term_memory: LongTermCreativeMemoryPlan,
    artifact_lineage: MultimodalArtifactLineageRegistry,
) -> tuple[CreativeLineageRecord, ...]:
    return (
        _record(
            kind="artifact_dependency_lineage",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            lineage_axis="dependency",
            source_history_record_id="artifact_history::iteration_history",
            source_dna_signature_id="creative_dna::lineage_dna",
            source_memory_record_id="long_term_creative_memory::artifact_lineage_memory",
            source_lineage_profile_id="dependency_graph_artifact_lineage",
            artifact_history=artifact_history,
            creative_dna=creative_dna,
            long_term_memory=long_term_memory,
            artifact_lineage=artifact_lineage,
            continuity_score=86,
            dependency_visibility_score=84,
            provenance_trace_score=82,
            governance_risk_score=46,
            governance_weight=150,
        ),
        _record(
            kind="timeline_stage_lineage",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            lineage_axis="timeline",
            source_history_record_id="artifact_history::decision_history",
            source_dna_signature_id="creative_dna::constraint_dna",
            source_memory_record_id="long_term_creative_memory::project_context_memory",
            source_lineage_profile_id="timeline_stage_artifact_lineage",
            artifact_history=artifact_history,
            creative_dna=creative_dna,
            long_term_memory=long_term_memory,
            artifact_lineage=artifact_lineage,
            continuity_score=82,
            dependency_visibility_score=78,
            provenance_trace_score=80,
            governance_risk_score=44,
            governance_weight=140,
        ),
        _record(
            kind="source_transition_lineage",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            lineage_axis="source_transition",
            source_history_record_id="artifact_history::runtime_history",
            source_dna_signature_id="creative_dna::intent_dna",
            source_memory_record_id=(
                "long_term_creative_memory::creative_intent_memory"
            ),
            source_lineage_profile_id="source_transition_artifact_lineage",
            artifact_history=artifact_history,
            creative_dna=creative_dna,
            long_term_memory=long_term_memory,
            artifact_lineage=artifact_lineage,
            continuity_score=74,
            dependency_visibility_score=76,
            provenance_trace_score=72,
            governance_risk_score=42,
            governance_weight=120,
        ),
        _record(
            kind="style_evolution_lineage",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            lineage_axis="style_evolution",
            source_history_record_id="artifact_history::style_history",
            source_dna_signature_id="creative_dna::style_dna",
            source_memory_record_id="long_term_creative_memory::style_pattern_memory",
            source_lineage_profile_id="source_transition_artifact_lineage",
            artifact_history=artifact_history,
            creative_dna=creative_dna,
            long_term_memory=long_term_memory,
            artifact_lineage=artifact_lineage,
            continuity_score=64,
            dependency_visibility_score=60,
            provenance_trace_score=66,
            governance_risk_score=32,
            governance_weight=110,
        ),
        _record(
            kind="gap_review_lineage",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            lineage_axis="gap_review",
            source_history_record_id="artifact_history::review_history",
            source_dna_signature_id="creative_dna::interaction_dna",
            source_memory_record_id=(
                "long_term_creative_memory::preference_signal_memory"
            ),
            source_lineage_profile_id="missing_artifact_lineage",
            artifact_history=artifact_history,
            creative_dna=creative_dna,
            long_term_memory=long_term_memory,
            artifact_lineage=artifact_lineage,
            continuity_score=50,
            dependency_visibility_score=52,
            provenance_trace_score=54,
            governance_risk_score=18,
            governance_weight=85,
        ),
    )


def _record(
    *,
    kind: CreativeLineageKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    lineage_axis: CreativeLineageAxis,
    source_history_record_id: str,
    source_dna_signature_id: str,
    source_memory_record_id: str,
    source_lineage_profile_id: str,
    artifact_history: ArtifactHistoryPlan,
    creative_dna: CreativeDNAPlan,
    long_term_memory: LongTermCreativeMemoryPlan,
    artifact_lineage: MultimodalArtifactLineageRegistry,
    continuity_score: int,
    dependency_visibility_score: int,
    provenance_trace_score: int,
    governance_risk_score: int,
    governance_weight: int,
) -> CreativeLineageRecord:
    source_history_record = artifact_history_record_by_id(
        source_history_record_id,
        artifact_history,
    )
    source_dna_signature = creative_dna_signature_by_id(
        source_dna_signature_id,
        creative_dna,
    )
    source_memory_record = long_term_creative_memory_record_by_id(
        source_memory_record_id,
        long_term_memory,
    )
    source_lineage_profile = multimodal_artifact_lineage_profile_by_id(
        source_lineage_profile_id,
        artifact_lineage,
    )
    if source_history_record is None:
        raise ValueError("source artifact history record must exist")
    if source_dna_signature is None:
        raise ValueError("source Creative DNA signature must exist")
    if source_memory_record is None:
        raise ValueError("source long-term memory record must exist")
    if source_lineage_profile is None:
        raise ValueError("source artifact lineage profile must exist")
    score = _creative_lineage_score(
        continuity_score=continuity_score,
        dependency_visibility_score=dependency_visibility_score,
        provenance_trace_score=provenance_trace_score,
        governance_risk_score=governance_risk_score,
        governance_weight=governance_weight,
    )
    status = _creative_lineage_status(score)
    confidence = _creative_lineage_confidence(score)
    return CreativeLineageRecord(
        creative_lineage_id=f"creative_lineage::{kind}",
        lineage_kind=kind,
        status=status,
        confidence=confidence,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        lineage_axis=lineage_axis,
        source_artifact_history_record_id=(source_history_record.artifact_history_id),
        source_creative_dna_signature_id=source_dna_signature.creative_dna_id,
        source_long_term_memory_record_id=source_memory_record.record_id,
        source_artifact_lineage_profile_id=source_lineage_profile.profile_id,
        lineage_summary=_lineage_summary(kind),
        continuity_score=continuity_score,
        dependency_visibility_score=dependency_visibility_score,
        provenance_trace_score=provenance_trace_score,
        governance_risk_score=governance_risk_score,
        governance_weight=governance_weight,
        creative_lineage_score=score,
        hitl_required_before_lineage_persistence=True,
        context_tags=_context_tags(kind, lineage_axis),
        explainability_notes=_explainability_notes(
            kind,
            source_history_record.artifact_history_id,
            source_dna_signature.creative_dna_id,
            source_memory_record.record_id,
            source_lineage_profile.profile_id,
        ),
        advisory_actions=_record_actions(kind),
        evidence=(
            f"source_artifact_history:{source_history_record.artifact_history_id}",
            f"source_creative_dna:{source_dna_signature.creative_dna_id}",
            f"source_long_term_memory:{source_memory_record.record_id}",
            f"source_artifact_lineage:{source_lineage_profile.profile_id}",
            f"lineage_axis:{lineage_axis}",
            f"continuity_score:{continuity_score}",
            f"dependency_visibility_score:{dependency_visibility_score}",
            f"provenance_trace_score:{provenance_trace_score}",
            f"governance_risk_score:{governance_risk_score}",
            "hitl_required_before_lineage_persistence:true",
        ),
    )


def _creative_lineage_score(
    *,
    continuity_score: int,
    dependency_visibility_score: int,
    provenance_trace_score: int,
    governance_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            continuity_score * 3
            + dependency_visibility_score * 2
            + provenance_trace_score * 3
            + governance_risk_score * 2
            + governance_weight,
        ),
    )


def _creative_lineage_status(score: int) -> CreativeLineageStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _creative_lineage_confidence(score: int) -> CreativeLineageConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_creative_lineage_score(
    records: tuple[CreativeLineageRecord, ...],
) -> int:
    base = sum(record.creative_lineage_score for record in records) // len(records)
    guarded_count = len(_record_ids_for_status(records, "guarded"))
    review_count = len(_record_ids_for_status(records, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_creative_lineage_posture(
    records: tuple[CreativeLineageRecord, ...],
) -> CreativeLineagePosture:
    if any(record.status == "guarded" for record in records):
        return "guarded"
    if any(record.status == "review_required" for record in records):
        return "review_required"
    return "candidate"


def _record_ids_for_status(
    records: tuple[CreativeLineageRecord, ...],
    status: CreativeLineageStatus,
) -> tuple[str, ...]:
    return tuple(
        record.creative_lineage_id for record in records if record.status == status
    )


def _record_ids_for_confidence(
    records: tuple[CreativeLineageRecord, ...],
    *confidences: CreativeLineageConfidence,
) -> tuple[str, ...]:
    return tuple(
        record.creative_lineage_id
        for record in records
        if record.confidence in confidences
    )


def _plan_actions(records: tuple[CreativeLineageRecord, ...]) -> tuple[str, ...]:
    guarded_record_count = len(_record_ids_for_status(records, "guarded"))
    return (
        "inspect_creative_lineage_records",
        "require_hitl_before_creative_lineage_persistence",
        "keep_creative_lineage_non_inferential",
        f"review_guarded_creative_lineage_count:{guarded_record_count}",
    )


def _lineage_summary(kind: CreativeLineageKind) -> str:
    summaries = {
        "artifact_dependency_lineage": (
            "Models advisory creative lineage dependency posture."
        ),
        "timeline_stage_lineage": (
            "Models advisory creative lineage timeline-stage posture."
        ),
        "source_transition_lineage": (
            "Models advisory creative lineage source-transition posture."
        ),
        "style_evolution_lineage": (
            "Models advisory creative lineage style-evolution posture."
        ),
        "gap_review_lineage": ("Models advisory creative lineage gap-review posture."),
    }
    return summaries[kind]


def _context_tags(
    kind: CreativeLineageKind,
    lineage_axis: CreativeLineageAxis,
) -> tuple[str, ...]:
    return (
        "creative_memory",
        "creative_lineage",
        lineage_axis,
        kind.removesuffix("_lineage"),
    )


def _explainability_notes(
    kind: CreativeLineageKind,
    source_history_record_id: str,
    source_dna_signature_id: str,
    source_memory_record_id: str,
    source_lineage_profile_id: str,
) -> tuple[str, ...]:
    return (
        f"creative_lineage_kind:{kind}",
        f"source_artifact_history:{source_history_record_id}",
        f"source_creative_dna:{source_dna_signature_id}",
        f"source_record:{source_memory_record_id}",
        f"source_artifact_lineage:{source_lineage_profile_id}",
        "score_inputs:continuity,dependency_visibility,provenance_trace,governance_risk,governance",
        "persistence_boundary:HITL_required_before_creative_lineage_persistence",
    )


def _record_actions(kind: CreativeLineageKind) -> tuple[str, ...]:
    return (
        f"review_{kind}",
        "inspect_sources_before_creative_lineage_persistence",
        "preserve_no_lineage_inference_boundary",
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
