"""V6.4 advisory research decomposer metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.research_planner import (
    ResearchPlannerPlan,
    build_research_planner,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

ResearchDecompositionKind = Literal[
    "research_objective_decomposition",
    "evidence_thread_decomposition",
    "source_validation_decomposition",
    "distillation_sequence_decomposition",
    "decomposition_governance_gate",
]
ResearchDecompositionStatus = Literal["candidate", "review_required", "guarded"]
ResearchDecompositionConfidence = Literal["low", "medium", "high", "guarded"]
ResearchDecompositionPosture = Literal["candidate", "review_required", "guarded"]
ResearchDecompositionAxis = Literal[
    "objective_decomposition",
    "evidence_threads",
    "source_validation",
    "distillation_sequence",
    "governance_gate",
]

RESEARCH_DECOMPOSER_ENTRY_SERIALIZATION_VERSION = "research_decomposer_entry.v1"
RESEARCH_DECOMPOSER_PLAN_SERIALIZATION_VERSION = "research_decomposer_plan.v1"
RESEARCH_PLANNER_PLAN_SERIALIZATION_VERSION = "research_planner_plan.v1"

RESEARCH_DECOMPOSER_AUTHORITY_BOUNDARY = (
    "V6.4 Research Decomposer exposes research objective, evidence-thread, "
    "source-validation, distillation-sequence, and decomposition governance "
    "posture as inspectable advisory metadata only; it does not execute "
    "research decomposition, create subtasks, mutate workflows, build workflow "
    "graphs, perform paper research, perform web research, fetch external "
    "sources, validate sources live, compute source credibility, detect "
    "contradictions, enrich the KB, write storage, mutate retrieval "
    "configuration, execute retrieval, mutate ranking, provision providers, "
    "infer API keys, route providers or models, execute providers, modify "
    "generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Research Decomposer",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "research_decomposition_execution",
    "subtask_creation_execution",
    "research_plan_execution",
    "workflow_graph_mutation",
    "workflow_control",
    "workflow_execution",
    "paper_research_execution",
    "web_research_execution",
    "external_source_fetch",
    "live_source_validation",
    "source_credibility_scoring",
    "contradiction_detection_execution",
    "kb_enrichment_execution",
    "kb_storage_write",
    "retrieval_configuration_mutation",
    "retrieval_execution",
    "ranking_mutation",
    "provider_provisioning",
    "api_key_inference",
    "provider_or_model_routing",
    "provider_execution",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class ResearchDecompositionEntry(BaseModel):
    """One advisory research decomposition entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=180)
    decomposition_kind: ResearchDecompositionKind
    status: ResearchDecompositionStatus
    confidence: ResearchDecompositionConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    decomposition_axis: ResearchDecompositionAxis
    research_planner_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    research_planner_entry_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    decomposition_summary: str = Field(min_length=1, max_length=360)
    objective_structure_score: int = Field(ge=0, le=100)
    evidence_thread_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    decomposition_score: int = Field(ge=0, le=1_000)
    hitl_required_before_decomposition_execution: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    research_decomposer_capability_implemented: Literal[True] = True
    research_decomposer_metadata_implemented: Literal[True] = True
    research_planner_metadata_used: Literal[True] = True
    research_decomposition_execution_implemented: Literal[False] = False
    subtask_creation_execution_implemented: Literal[False] = False
    research_plan_execution_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    paper_research_execution_implemented: Literal[False] = False
    web_research_execution_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    live_source_validation_implemented: Literal[False] = False
    source_credibility_scoring_implemented: Literal[False] = False
    contradiction_detection_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    api_key_inference_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["research_decomposer_entry.v1"] = (
        RESEARCH_DECOMPOSER_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.entry_id != f"research_decomposer::{self.decomposition_kind}":
            raise ValueError("entry_id must match decomposition_kind")
        if self.research_planner_entry_count != len(self.research_planner_entry_ids):
            raise ValueError("research_planner_entry_count must match planner ids")
        if self.decomposition_score != _decomposition_score(
            objective_structure_score=self.objective_structure_score,
            evidence_thread_score=self.evidence_thread_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("decomposition_score must combine source scores")
        if self.status != _decomposition_status(self.decomposition_score):
            raise ValueError("status must match decomposition_score")
        if self.confidence != _decomposition_confidence(self.decomposition_score):
            raise ValueError("confidence must match decomposition_score")
        if not self.hitl_required_before_decomposition_execution:
            raise ValueError("research decomposition execution requires HITL posture")
        return self


class ResearchDecomposerPlan(BaseModel):
    """Bounded V6.4 advisory research decomposer plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["research_decomposer"] = "research_decomposer"
    serialization_version: Literal["research_decomposer_plan.v1"] = (
        RESEARCH_DECOMPOSER_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RESEARCH_DECOMPOSER_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    research_planner_role: Literal["research_planner"] = "research_planner"
    research_planner_serialization_version: Literal["research_planner_plan.v1"]
    research_planner_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    research_planner_entry_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[ResearchDecompositionEntry, ...] = Field(
        min_length=5,
        max_length=5,
    )
    entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    entry_count: int = Field(ge=5, le=5)
    candidate_entry_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    review_required_entry_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_entry_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    high_confidence_entry_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    hitl_required_entry_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    planned_decomposition_execution_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    generated_subtask_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_workflow_graph_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    fetched_external_source_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_storage_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_decomposition_score: int = Field(ge=0, le=1_000)
    overall_decomposition_score: int = Field(ge=0, le=1_000)
    overall_decomposition_posture: ResearchDecompositionPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    research_decomposer_capability_implemented: Literal[True] = True
    research_decomposer_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    research_planner_metadata_used: Literal[True] = True
    research_decomposition_execution_implemented: Literal[False] = False
    subtask_creation_execution_implemented: Literal[False] = False
    research_plan_execution_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    paper_research_execution_implemented: Literal[False] = False
    web_research_execution_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    live_source_validation_implemented: Literal[False] = False
    source_credibility_scoring_implemented: Literal[False] = False
    contradiction_detection_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    api_key_inference_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @field_validator("checked_at")
    @classmethod
    def _checked_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("checked_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _plan_matches_entries(self) -> Self:
        derived_entry_ids = tuple(entry.entry_id for entry in self.entries)
        if len(set(derived_entry_ids)) != len(derived_entry_ids):
            raise ValueError("entry_ids must be unique")
        if self.entry_ids != derived_entry_ids:
            raise ValueError("entry_ids must match entries")
        if self.candidate_entry_ids != _entry_ids_for_status(
            self.entries,
            "candidate",
        ):
            raise ValueError("candidate_entry_ids must match entries")
        if self.review_required_entry_ids != _entry_ids_for_status(
            self.entries,
            "review_required",
        ):
            raise ValueError("review_required_entry_ids must match entries")
        if self.guarded_entry_ids != _entry_ids_for_status(self.entries, "guarded"):
            raise ValueError("guarded_entry_ids must match entries")
        if self.high_confidence_entry_ids != _entry_ids_for_confidence(
            self.entries,
            "high",
            "guarded",
        ):
            raise ValueError("high_confidence_entry_ids must match entries")
        if self.hitl_required_entry_ids != tuple(
            entry.entry_id
            for entry in self.entries
            if entry.hitl_required_before_decomposition_execution
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.planned_decomposition_execution_ids:
            raise ValueError("planned_decomposition_execution_ids must remain empty")
        if self.generated_subtask_ids:
            raise ValueError("generated_subtask_ids must remain empty")
        if self.mutated_workflow_graph_ids:
            raise ValueError("mutated_workflow_graph_ids must remain empty")
        if self.fetched_external_source_ids:
            raise ValueError("fetched_external_source_ids must remain empty")
        if self.written_storage_record_ids:
            raise ValueError("written_storage_record_ids must remain empty")
        if self.research_planner_entry_count != len(self.research_planner_entry_ids):
            raise ValueError("research_planner_entry_count must match planner ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 3 roadmap")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap items")
        if self.entry_count != len(self.entries):
            raise ValueError("entry_count must match entries")
        if self.candidate_entry_count != len(self.candidate_entry_ids):
            raise ValueError("candidate_entry_count must match entries")
        if self.review_required_entry_count != len(self.review_required_entry_ids):
            raise ValueError("review_required_entry_count must match entries")
        if self.guarded_entry_count != len(self.guarded_entry_ids):
            raise ValueError("guarded_entry_count must match entries")
        if self.high_confidence_entry_count != len(self.high_confidence_entry_ids):
            raise ValueError("high_confidence_entry_count must match entries")
        if self.hitl_required_entry_count != len(self.hitl_required_entry_ids):
            raise ValueError("hitl_required_entry_count must match entries")
        if self.highest_decomposition_score != max(
            entry.decomposition_score for entry in self.entries
        ):
            raise ValueError("highest_decomposition_score must match entries")
        if self.overall_decomposition_score != _overall_decomposition_score(
            self.entries
        ):
            raise ValueError("overall_decomposition_score must match entries")
        if self.overall_decomposition_posture != _overall_decomposition_posture(
            self.entries
        ):
            raise ValueError("overall_decomposition_posture must match entries")
        planner_entry_ids = set(self.research_planner_entry_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if entry.source_count != self.source_count:
                raise ValueError("entry source_count must match plan")
            if entry.domain_count != self.domain_count:
                raise ValueError("entry domain_count must match plan")
            if not set(entry.research_planner_entry_ids).issubset(planner_entry_ids):
                raise ValueError("entry planner ids must be declared")
        return self


def build_research_decomposer(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    research_planner: ResearchPlannerPlan | None = None,
) -> ResearchDecomposerPlan:
    """Build V6.4 Task 3 research decomposer metadata without decomposition."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    planner = research_planner or build_research_planner(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        planner=planner,
    )
    return ResearchDecomposerPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=planner.checked_at,
        research_planner_serialization_version=planner.serialization_version,
        research_planner_entry_ids=planner.entry_ids,
        research_planner_entry_count=len(planner.entry_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=planner.source_count,
        domain_count=planner.domain_count,
        execution_mode_ids=execution_modes.execution_mode_ids,
        entries=entries,
        entry_ids=tuple(entry.entry_id for entry in entries),
        entry_count=len(entries),
        candidate_entry_ids=_entry_ids_for_status(entries, "candidate"),
        review_required_entry_ids=_entry_ids_for_status(entries, "review_required"),
        guarded_entry_ids=_entry_ids_for_status(entries, "guarded"),
        high_confidence_entry_ids=_entry_ids_for_confidence(
            entries,
            "high",
            "guarded",
        ),
        hitl_required_entry_ids=tuple(
            entry.entry_id
            for entry in entries
            if entry.hitl_required_before_decomposition_execution
        ),
        candidate_entry_count=len(_entry_ids_for_status(entries, "candidate")),
        review_required_entry_count=len(
            _entry_ids_for_status(entries, "review_required")
        ),
        guarded_entry_count=len(_entry_ids_for_status(entries, "guarded")),
        high_confidence_entry_count=len(
            _entry_ids_for_confidence(entries, "high", "guarded")
        ),
        hitl_required_entry_count=sum(
            1 for entry in entries if entry.hitl_required_before_decomposition_execution
        ),
        highest_decomposition_score=max(
            entry.decomposition_score for entry in entries
        ),
        overall_decomposition_score=_overall_decomposition_score(entries),
        overall_decomposition_posture=_overall_decomposition_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def research_decomposition_entry_by_id(
    entry_id: str,
    plan: ResearchDecomposerPlan | None = None,
) -> ResearchDecompositionEntry | None:
    """Return one research decomposition entry without executing decomposition."""

    source_plan = plan or build_research_decomposer()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def research_decomposition_entries_for_status(
    status: ResearchDecompositionStatus,
    plan: ResearchDecomposerPlan | None = None,
) -> tuple[ResearchDecompositionEntry, ...]:
    """Return research decomposition entries by advisory status."""

    source_plan = plan or build_research_decomposer()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def research_decomposition_entries_for_confidence(
    confidence: ResearchDecompositionConfidence,
    plan: ResearchDecomposerPlan | None = None,
) -> tuple[ResearchDecompositionEntry, ...]:
    """Return research decomposition entries by confidence band."""

    source_plan = plan or build_research_decomposer()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    planner: ResearchPlannerPlan,
) -> tuple[ResearchDecompositionEntry, ...]:
    return (
        _entry(
            kind="research_objective_decomposition",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="objective_decomposition",
            planner_entry_ids=planner.entry_ids,
            planner=planner,
            objective_structure_score=90,
            evidence_thread_score=80,
            governance_alignment_score=88,
            mutation_risk_score=50,
            governance_weight=110,
        ),
        _entry(
            kind="evidence_thread_decomposition",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="evidence_threads",
            planner_entry_ids=(
                "research_planner::source_strategy_planning",
                "research_planner::validation_strategy_planning",
            ),
            planner=planner,
            objective_structure_score=82,
            evidence_thread_score=78,
            governance_alignment_score=84,
            mutation_risk_score=44,
            governance_weight=100,
        ),
        _entry(
            kind="source_validation_decomposition",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="source_validation",
            planner_entry_ids=(
                "research_planner::validation_strategy_planning",
                "research_planner::governance_gate_planning",
            ),
            planner=planner,
            objective_structure_score=76,
            evidence_thread_score=72,
            governance_alignment_score=86,
            mutation_risk_score=42,
            governance_weight=95,
        ),
        _entry(
            kind="distillation_sequence_decomposition",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="distillation_sequence",
            planner_entry_ids=(
                "research_planner::distillation_report_planning",
                "research_planner::research_scope_framing",
            ),
            planner=planner,
            objective_structure_score=68,
            evidence_thread_score=66,
            governance_alignment_score=78,
            mutation_risk_score=34,
            governance_weight=85,
        ),
        _entry(
            kind="decomposition_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            planner_entry_ids=planner.entry_ids,
            planner=planner,
            objective_structure_score=44,
            evidence_thread_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _entry(
    *,
    kind: ResearchDecompositionKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: ResearchDecompositionAxis,
    planner_entry_ids: tuple[str, ...],
    planner: ResearchPlannerPlan,
    objective_structure_score: int,
    evidence_thread_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> ResearchDecompositionEntry:
    score = _decomposition_score(
        objective_structure_score=objective_structure_score,
        evidence_thread_score=evidence_thread_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return ResearchDecompositionEntry(
        entry_id=f"research_decomposer::{kind}",
        decomposition_kind=kind,
        status=_decomposition_status(score),
        confidence=_decomposition_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        decomposition_axis=axis,
        research_planner_entry_ids=planner_entry_ids,
        research_planner_entry_count=len(planner_entry_ids),
        source_count=planner.source_count,
        domain_count=planner.domain_count,
        decomposition_summary=_decomposition_summary(kind),
        objective_structure_score=objective_structure_score,
        evidence_thread_score=evidence_thread_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        decomposition_score=score,
        hitl_required_before_decomposition_execution=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, planner_entry_ids),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"research_planner_entry_count:{len(planner_entry_ids)}",
            f"source_count:{planner.source_count}",
            f"domain_count:{planner.domain_count}",
            f"decomposition_axis:{axis}",
            f"status:{_decomposition_status(score)}",
            f"confidence:{_decomposition_confidence(score)}",
            "hitl_required_before_decomposition_execution:true",
        ),
    )


def _decomposition_score(
    *,
    objective_structure_score: int,
    evidence_thread_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            objective_structure_score * 3
            + evidence_thread_score * 2
            + governance_alignment_score * 3
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _decomposition_status(score: int) -> ResearchDecompositionStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _decomposition_confidence(score: int) -> ResearchDecompositionConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_decomposition_score(
    entries: tuple[ResearchDecompositionEntry, ...],
) -> int:
    return round(sum(entry.decomposition_score for entry in entries) / len(entries))


def _overall_decomposition_posture(
    entries: tuple[ResearchDecompositionEntry, ...],
) -> ResearchDecompositionPosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[ResearchDecompositionEntry, ...],
    status: ResearchDecompositionStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[ResearchDecompositionEntry, ...],
    *confidences: ResearchDecompositionConfidence,
) -> tuple[str, ...]:
    return tuple(
        entry.entry_id for entry in entries if entry.confidence in confidences
    )


def _plan_actions(entries: tuple[ResearchDecompositionEntry, ...]) -> tuple[str, ...]:
    return (
        f"review_research_decomposition_entries:{len(entries)}",
        "confirm_decomposition_before_execution",
        "confirm_no_subtask_creation_without_hitl",
        "confirm_no_workflow_mutation",
        "request_hitl_before_decomposition_execution",
    )


def _entry_actions(kind: ResearchDecompositionKind) -> tuple[str, ...]:
    actions: dict[ResearchDecompositionKind, tuple[str, ...]] = {
        "research_objective_decomposition": (
            "review_objective_breakdown",
            "confirm_question_boundaries",
            "confirm_evidence_threads",
        ),
        "evidence_thread_decomposition": (
            "review_evidence_thread_map",
            "confirm_source_role_boundaries",
            "confirm_comparison_points",
        ),
        "source_validation_decomposition": (
            "review_validation_thread_map",
            "confirm_provenance_requirements",
            "confirm_no_live_validation",
        ),
        "distillation_sequence_decomposition": (
            "review_distillation_sequence",
            "confirm_report_dependency_order",
            "confirm_confidence_disclosure_points",
        ),
        "decomposition_governance_gate": (
            "review_decomposition_hitl_gate",
            "confirm_no_subtask_creation",
            "confirm_no_workflow_graph_mutation",
        ),
    }
    return actions[kind]


def _decomposition_summary(kind: ResearchDecompositionKind) -> str:
    summaries: dict[ResearchDecompositionKind, str] = {
        "research_objective_decomposition": (
            "Frames how a research objective could be split into bounded "
            "questions without creating executable subtasks."
        ),
        "evidence_thread_decomposition": (
            "Maps advisory evidence threads from planner metadata without "
            "fetching sources or executing comparisons."
        ),
        "source_validation_decomposition": (
            "Describes validation and provenance threads without validating "
            "sources live or computing credibility."
        ),
        "distillation_sequence_decomposition": (
            "Models a distillation/report sequence without generating reports "
            "or mutating generated output."
        ),
        "decomposition_governance_gate": (
            "Models the HITL gate required before decomposition execution, "
            "subtask creation, or workflow mutation."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: ResearchDecompositionKind,
    axis: ResearchDecompositionAxis,
) -> tuple[str, ...]:
    return (
        "research_decomposer",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: ResearchDecompositionKind,
    planner_entry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"decomposition_kind:{kind}",
        f"planner_entry_count:{len(planner_entry_ids)}",
        "research_planner_metadata_only",
        "no_decomposition_execution_performed",
    )


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route).strip())


def _resolve_task_type(task_type: TaskRoutingType | str) -> TaskRoutingType:
    candidate = str(task_type).strip()
    if candidate not in get_args(TaskRoutingType):
        raise ValueError(f"Unknown task routing type: {task_type!r}")
    return cast(TaskRoutingType, candidate)


def _resolve_execution_mode(
    execution_mode_id: ExecutionModeId | str,
    allowed_modes: tuple[ExecutionModeId, ...],
) -> ExecutionModeId:
    candidate = str(execution_mode_id).strip()
    if candidate not in allowed_modes:
        raise ValueError(f"Unknown execution mode id: {execution_mode_id!r}")
    return cast(ExecutionModeId, candidate)
