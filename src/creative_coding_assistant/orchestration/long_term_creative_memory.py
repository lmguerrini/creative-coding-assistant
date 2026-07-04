"""V6.2 advisory long-term creative memory metadata."""

from __future__ import annotations

from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.memory import ProjectMemoryKind
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

LongTermCreativeMemoryKind = Literal[
    "creative_intent_memory",
    "style_pattern_memory",
    "project_context_memory",
    "artifact_lineage_memory",
    "preference_signal_memory",
]
LongTermCreativeMemoryStatus = Literal["candidate", "review_required", "guarded"]
LongTermCreativeMemorySensitivity = Literal["low", "medium", "high", "guarded"]
LongTermCreativeMemoryPosture = Literal["candidate", "review_required", "guarded"]
LongTermCreativeMemoryScope = Literal[
    "user",
    "project",
    "style",
    "artifact",
    "session_evolution",
]

LONG_TERM_CREATIVE_MEMORY_RECORD_SERIALIZATION_VERSION = (
    "long_term_creative_memory_record.v1"
)
LONG_TERM_CREATIVE_MEMORY_PLAN_SERIALIZATION_VERSION = (
    "long_term_creative_memory_plan.v1"
)
LONG_TERM_CREATIVE_MEMORY_AUTHORITY_BOUNDARY = (
    "V6.2 long-term creative memory models durable creative continuity as "
    "inspectable advisory metadata only; it does not write memory storage, "
    "create memory records, update memory records, delete memory records, "
    "execute memory retrieval, consolidate memory, mutate user preferences, "
    "apply personalization, change provider or model routing, execute "
    "providers, invoke agents, control workflows, mutate workflow graphs, "
    "trigger retries or refinements, mutate prompts, modify generated output, "
    "or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "memory_storage_write",
    "memory_record_creation",
    "memory_record_update",
    "memory_record_deletion",
    "memory_retrieval_execution",
    "automatic_memory_consolidation",
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


class LongTermCreativeMemoryRecord(BaseModel):
    """One advisory long-term creative memory record candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(min_length=1, max_length=180)
    memory_kind: LongTermCreativeMemoryKind
    status: LongTermCreativeMemoryStatus
    sensitivity: LongTermCreativeMemorySensitivity
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    retention_scope: LongTermCreativeMemoryScope
    source_memory_kind: ProjectMemoryKind
    source_surface: str = Field(min_length=1, max_length=160)
    stability_score: int = Field(ge=0, le=100)
    evidence_strength_score: int = Field(ge=0, le=100)
    recency_resilience_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    memory_governance_score: int = Field(ge=0, le=1_000)
    hitl_required_before_persistence: bool
    retrieval_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    memory_summary: str = Field(min_length=1, max_length=360)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    long_term_creative_memory_implemented: Literal[True] = True
    memory_record_metadata_implemented: Literal[True] = True
    memory_governance_metadata_implemented: Literal[True] = True
    memory_storage_write_implemented: Literal[False] = False
    memory_record_creation_implemented: Literal[False] = False
    memory_record_update_implemented: Literal[False] = False
    memory_record_deletion_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_consolidation_implemented: Literal[False] = False
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
    serialization_version: Literal["long_term_creative_memory_record.v1"] = (
        LONG_TERM_CREATIVE_MEMORY_RECORD_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_matches_contract(self) -> Self:
        if self.record_id != f"long_term_creative_memory::{self.memory_kind}":
            raise ValueError("record_id must match memory_kind")
        if self.memory_governance_score != _memory_governance_score(
            stability_score=self.stability_score,
            evidence_strength_score=self.evidence_strength_score,
            recency_resilience_score=self.recency_resilience_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("memory_governance_score must combine source scores")
        if self.status != _record_status(self.memory_governance_score):
            raise ValueError("status must match memory_governance_score")
        if self.sensitivity != _record_sensitivity(self.memory_governance_score):
            raise ValueError("sensitivity must match memory_governance_score")
        if not self.hitl_required_before_persistence:
            raise ValueError("long-term memory persistence requires HITL posture")
        return self


class LongTermCreativeMemoryPlan(BaseModel):
    """Bounded V6.2 advisory long-term creative memory plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["long_term_creative_memory"] = "long_term_creative_memory"
    serialization_version: Literal["long_term_creative_memory_plan.v1"] = (
        LONG_TERM_CREATIVE_MEMORY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=LONG_TERM_CREATIVE_MEMORY_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    records: tuple[LongTermCreativeMemoryRecord, ...] = Field(
        min_length=5,
        max_length=5,
    )
    record_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    candidate_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    review_required_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    high_sensitivity_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    hitl_required_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    persisted_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    retrieved_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    personalized_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    record_count: int = Field(ge=5, le=5)
    candidate_record_count: int = Field(ge=0, le=5)
    review_required_record_count: int = Field(ge=0, le=5)
    guarded_record_count: int = Field(ge=0, le=5)
    high_sensitivity_record_count: int = Field(ge=0, le=5)
    hitl_required_record_count: int = Field(ge=0, le=5)
    highest_memory_governance_score: int = Field(ge=0, le=1_000)
    overall_memory_governance_score: int = Field(ge=0, le=1_000)
    overall_memory_posture: LongTermCreativeMemoryPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    long_term_creative_memory_implemented: Literal[True] = True
    memory_record_metadata_implemented: Literal[True] = True
    memory_governance_metadata_implemented: Literal[True] = True
    memory_storage_write_implemented: Literal[False] = False
    memory_record_creation_implemented: Literal[False] = False
    memory_record_update_implemented: Literal[False] = False
    memory_record_deletion_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_consolidation_implemented: Literal[False] = False
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
    def _plan_matches_records(self) -> Self:
        derived_record_ids = tuple(record.record_id for record in self.records)
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
        if self.high_sensitivity_record_ids != _record_ids_for_sensitivity(
            self.records,
            "high",
            "guarded",
        ):
            raise ValueError("high_sensitivity_record_ids must match records")
        if self.hitl_required_record_ids != tuple(
            record.record_id
            for record in self.records
            if record.hitl_required_before_persistence
        ):
            raise ValueError("hitl_required_record_ids must match records")
        if self.persisted_record_ids:
            raise ValueError("persisted_record_ids must remain empty")
        if self.retrieved_record_ids:
            raise ValueError("retrieved_record_ids must remain empty")
        if self.personalized_record_ids:
            raise ValueError("personalized_record_ids must remain empty")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.candidate_record_count != len(self.candidate_record_ids):
            raise ValueError("candidate_record_count must match records")
        if self.review_required_record_count != len(self.review_required_record_ids):
            raise ValueError("review_required_record_count must match records")
        if self.guarded_record_count != len(self.guarded_record_ids):
            raise ValueError("guarded_record_count must match records")
        if self.high_sensitivity_record_count != len(self.high_sensitivity_record_ids):
            raise ValueError("high_sensitivity_record_count must match records")
        if self.hitl_required_record_count != len(self.hitl_required_record_ids):
            raise ValueError("hitl_required_record_count must match records")
        if self.highest_memory_governance_score != max(
            record.memory_governance_score for record in self.records
        ):
            raise ValueError("highest_memory_governance_score must match records")
        if self.overall_memory_governance_score != _overall_memory_governance_score(
            self.records,
        ):
            raise ValueError("overall_memory_governance_score must match records")
        if self.overall_memory_posture != _overall_memory_posture(self.records):
            raise ValueError("overall_memory_posture must match records")
        for record in self.records:
            if record.route_name != self.route_name:
                raise ValueError("record route_name must match plan")
        return self


def build_long_term_creative_memory(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> LongTermCreativeMemoryPlan:
    """Build long-term creative memory metadata without storage mutation."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    records = _records(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    return LongTermCreativeMemoryPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_ids=execution_modes.execution_mode_ids,
        records=records,
        record_ids=tuple(record.record_id for record in records),
        candidate_record_ids=_record_ids_for_status(records, "candidate"),
        review_required_record_ids=_record_ids_for_status(records, "review_required"),
        guarded_record_ids=_record_ids_for_status(records, "guarded"),
        high_sensitivity_record_ids=_record_ids_for_sensitivity(
            records,
            "high",
            "guarded",
        ),
        hitl_required_record_ids=tuple(
            record.record_id
            for record in records
            if record.hitl_required_before_persistence
        ),
        persisted_record_ids=(),
        retrieved_record_ids=(),
        personalized_record_ids=(),
        record_count=len(records),
        candidate_record_count=len(_record_ids_for_status(records, "candidate")),
        review_required_record_count=len(
            _record_ids_for_status(records, "review_required")
        ),
        guarded_record_count=len(_record_ids_for_status(records, "guarded")),
        high_sensitivity_record_count=len(
            _record_ids_for_sensitivity(records, "high", "guarded")
        ),
        hitl_required_record_count=sum(
            1 for record in records if record.hitl_required_before_persistence
        ),
        highest_memory_governance_score=max(
            record.memory_governance_score for record in records
        ),
        overall_memory_governance_score=_overall_memory_governance_score(records),
        overall_memory_posture=_overall_memory_posture(records),
        advisory_actions=_plan_actions(records),
    )


def long_term_creative_memory_record_by_id(
    record_id: str,
    plan: LongTermCreativeMemoryPlan | None = None,
) -> LongTermCreativeMemoryRecord | None:
    """Return one long-term creative memory record without retrieval execution."""

    source_plan = plan or build_long_term_creative_memory()
    for record in source_plan.records:
        if record.record_id == record_id:
            return record
    return None


def long_term_creative_memory_records_for_status(
    status: LongTermCreativeMemoryStatus,
    plan: LongTermCreativeMemoryPlan | None = None,
) -> tuple[LongTermCreativeMemoryRecord, ...]:
    """Return long-term creative memory records by advisory status."""

    source_plan = plan or build_long_term_creative_memory()
    return tuple(record for record in source_plan.records if record.status == status)


def long_term_creative_memory_records_for_sensitivity(
    sensitivity: LongTermCreativeMemorySensitivity,
    plan: LongTermCreativeMemoryPlan | None = None,
) -> tuple[LongTermCreativeMemoryRecord, ...]:
    """Return long-term creative memory records by sensitivity band."""

    source_plan = plan or build_long_term_creative_memory()
    return tuple(
        record for record in source_plan.records if record.sensitivity == sensitivity
    )


def _records(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
) -> tuple[LongTermCreativeMemoryRecord, ...]:
    return (
        _record(
            kind="creative_intent_memory",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            retention_scope="project",
            source_memory_kind=ProjectMemoryKind.GOAL,
            source_surface="creative_intent_and_project_goal_continuity",
            stability_score=78,
            evidence_strength_score=82,
            recency_resilience_score=76,
            governance_weight=160,
        ),
        _record(
            kind="style_pattern_memory",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            retention_scope="style",
            source_memory_kind=ProjectMemoryKind.STYLE,
            source_surface="style_profile_and_visual_taste_continuity",
            stability_score=72,
            evidence_strength_score=88,
            recency_resilience_score=70,
            governance_weight=140,
        ),
        _record(
            kind="project_context_memory",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            retention_scope="project",
            source_memory_kind=ProjectMemoryKind.TECHNICAL_NOTE,
            source_surface="project_constraints_and_technical_notes",
            stability_score=62,
            evidence_strength_score=70,
            recency_resilience_score=68,
            governance_weight=120,
        ),
        _record(
            kind="artifact_lineage_memory",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            retention_scope="artifact",
            source_memory_kind=ProjectMemoryKind.DECISION,
            source_surface="artifact_history_and_lineage_signals",
            stability_score=58,
            evidence_strength_score=66,
            recency_resilience_score=62,
            governance_weight=110,
        ),
        _record(
            kind="preference_signal_memory",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            retention_scope="user",
            source_memory_kind=ProjectMemoryKind.PREFERENCE,
            source_surface="explicit_user_preference_signals",
            stability_score=42,
            evidence_strength_score=55,
            recency_resilience_score=50,
            governance_weight=95,
        ),
    )


def _record(
    *,
    kind: LongTermCreativeMemoryKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    retention_scope: LongTermCreativeMemoryScope,
    source_memory_kind: ProjectMemoryKind,
    source_surface: str,
    stability_score: int,
    evidence_strength_score: int,
    recency_resilience_score: int,
    governance_weight: int,
) -> LongTermCreativeMemoryRecord:
    governance_score = _memory_governance_score(
        stability_score=stability_score,
        evidence_strength_score=evidence_strength_score,
        recency_resilience_score=recency_resilience_score,
        governance_weight=governance_weight,
    )
    status = _record_status(governance_score)
    sensitivity = _record_sensitivity(governance_score)
    return LongTermCreativeMemoryRecord(
        record_id=f"long_term_creative_memory::{kind}",
        memory_kind=kind,
        status=status,
        sensitivity=sensitivity,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        retention_scope=retention_scope,
        source_memory_kind=source_memory_kind,
        source_surface=source_surface,
        stability_score=stability_score,
        evidence_strength_score=evidence_strength_score,
        recency_resilience_score=recency_resilience_score,
        governance_weight=governance_weight,
        memory_governance_score=governance_score,
        hitl_required_before_persistence=True,
        retrieval_tags=_retrieval_tags(kind, retention_scope),
        memory_summary=_memory_summary(kind, status),
        explainability_notes=_explainability_notes(kind),
        advisory_actions=_record_actions(kind),
        evidence=(
            f"source_memory_kind:{source_memory_kind.value}",
            f"source_surface:{source_surface}",
            f"retention_scope:{retention_scope}",
            f"stability_score:{stability_score}",
            f"evidence_strength_score:{evidence_strength_score}",
            f"recency_resilience_score:{recency_resilience_score}",
            "hitl_required_before_persistence:true",
        ),
    )


def _memory_governance_score(
    *,
    stability_score: int,
    evidence_strength_score: int,
    recency_resilience_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            stability_score * 4
            + evidence_strength_score * 3
            + recency_resilience_score * 2
            + governance_weight,
        ),
    )


def _record_status(score: int) -> LongTermCreativeMemoryStatus:
    if score >= 760:
        return "guarded"
    if score >= 600:
        return "review_required"
    return "candidate"


def _record_sensitivity(score: int) -> LongTermCreativeMemorySensitivity:
    if score >= 760:
        return "guarded"
    if score >= 680:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_memory_governance_score(
    records: tuple[LongTermCreativeMemoryRecord, ...],
) -> int:
    base = sum(record.memory_governance_score for record in records) // len(records)
    guarded_count = len(_record_ids_for_status(records, "guarded"))
    review_count = len(_record_ids_for_status(records, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_memory_posture(
    records: tuple[LongTermCreativeMemoryRecord, ...],
) -> LongTermCreativeMemoryPosture:
    if any(record.status == "guarded" for record in records):
        return "guarded"
    if any(record.status == "review_required" for record in records):
        return "review_required"
    return "candidate"


def _record_ids_for_status(
    records: tuple[LongTermCreativeMemoryRecord, ...],
    status: LongTermCreativeMemoryStatus,
) -> tuple[str, ...]:
    return tuple(record.record_id for record in records if record.status == status)


def _record_ids_for_sensitivity(
    records: tuple[LongTermCreativeMemoryRecord, ...],
    *sensitivities: LongTermCreativeMemorySensitivity,
) -> tuple[str, ...]:
    return tuple(
        record.record_id for record in records if record.sensitivity in sensitivities
    )


def _plan_actions(
    records: tuple[LongTermCreativeMemoryRecord, ...],
) -> tuple[str, ...]:
    guarded_record_count = len(_record_ids_for_status(records, "guarded"))
    return (
        "inspect_long_term_creative_memory_candidates",
        "require_hitl_before_any_memory_persistence",
        "keep_memory_retrieval_and_personalization_non_executing",
        f"review_guarded_record_count:{guarded_record_count}",
    )


def _retrieval_tags(
    kind: LongTermCreativeMemoryKind,
    retention_scope: LongTermCreativeMemoryScope,
) -> tuple[str, ...]:
    return (
        "creative_memory",
        "long_term",
        retention_scope,
        kind.removesuffix("_memory"),
    )


def _memory_summary(
    kind: LongTermCreativeMemoryKind,
    status: LongTermCreativeMemoryStatus,
) -> str:
    summaries = {
        "creative_intent_memory": (
            "Tracks durable creative goals and intent as reviewable metadata."
        ),
        "style_pattern_memory": (
            "Tracks recurring style and taste patterns as reviewable metadata."
        ),
        "project_context_memory": (
            "Tracks project-level constraints and technical context metadata."
        ),
        "artifact_lineage_memory": (
            "Tracks artifact evolution decisions as lineage-ready metadata."
        ),
        "preference_signal_memory": (
            "Tracks explicit preference signals without mutating preferences."
        ),
    }
    return f"{summaries[kind]} Status: {status}."


def _explainability_notes(
    kind: LongTermCreativeMemoryKind,
) -> tuple[str, ...]:
    return (
        f"record_kind:{kind}",
        "score_inputs:stability,evidence_strength,recency_resilience,governance",
        "persistence_boundary:HITL_required_before_storage_write",
        "retrieval_boundary:metadata_only_no_retrieval_execution",
    )


def _record_actions(
    kind: LongTermCreativeMemoryKind,
) -> tuple[str, ...]:
    return (
        f"review_{kind}",
        "inspect_evidence_before_persistence",
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
