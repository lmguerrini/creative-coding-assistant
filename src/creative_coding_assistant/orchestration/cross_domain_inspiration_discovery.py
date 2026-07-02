"""V6.4 advisory cross-domain inspiration discovery metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.creative_research_engine import (
    CREATIVE_RESEARCH_PLAN_SERIALIZATION_VERSION,
    CreativeResearchPlan,
    build_creative_research_engine,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

CrossDomainInspirationKind = Literal[
    "domain_analogy_mapping",
    "medium_transfer_review",
    "pattern_translation_review",
    "inspiration_provenance_review",
    "cross_domain_inspiration_governance_gate",
]
CrossDomainInspirationStatus = Literal["candidate", "review_required", "guarded"]
CrossDomainInspirationConfidence = Literal["low", "medium", "high", "guarded"]
CrossDomainInspirationPosture = Literal["candidate", "review_required", "guarded"]
CrossDomainInspirationAxis = Literal[
    "domain_analogy",
    "medium_transfer",
    "pattern_translation",
    "inspiration_provenance",
    "governance_gate",
]

CROSS_DOMAIN_INSPIRATION_ENTRY_SERIALIZATION_VERSION = (
    "cross_domain_inspiration_entry.v1"
)
CROSS_DOMAIN_INSPIRATION_PLAN_SERIALIZATION_VERSION = (
    "cross_domain_inspiration_plan.v1"
)

CROSS_DOMAIN_INSPIRATION_AUTHORITY_BOUNDARY = (
    "V6.4 Cross-domain Inspiration Discovery exposes domain analogy mapping, "
    "medium transfer review, pattern translation posture, inspiration "
    "provenance review, and inspiration governance as inspectable advisory "
    "metadata only; it does not execute inspiration discovery, perform live "
    "cross-domain search, fetch external sources, browse the web, download "
    "papers, generate creative outputs, mutate generated output, generate "
    "creative assets, write inspiration records, generate research "
    "recommendations, execute recommendations, emit HITL requests, apply "
    "HITL decisions, execute HITL gates, apply research execution policy, "
    "authorize research execution, execute research, mutate research plans, "
    "control workflows, mutate workflow graphs, execute workflows, mutate "
    "source registries, execute retrieval, mutate retrieval configuration, "
    "mutate vector indexes, enrich the KB, write storage, provision "
    "providers, infer API keys, route providers or models, execute providers, "
    "modify generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Cross-domain Inspiration Discovery",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "inspiration_discovery_execution",
    "live_cross_domain_search",
    "external_source_fetch",
    "web_browsing",
    "paper_download",
    "creative_output_generation",
    "creative_output_mutation",
    "creative_asset_generation",
    "creative_asset_write",
    "inspiration_record_write",
    "research_recommendation_generation",
    "recommendation_execution",
    "recommendation_record_write",
    "hitl_request_emission",
    "hitl_decision_application",
    "hitl_gate_execution",
    "research_execution_policy_application",
    "research_execution_authorization",
    "research_execution",
    "research_plan_mutation",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "source_registry_mutation",
    "retrieval_execution",
    "retrieval_configuration_mutation",
    "vector_index_mutation",
    "kb_enrichment_execution",
    "kb_storage_write",
    "persistent_storage_write",
    "provider_provisioning",
    "api_key_inference",
    "provider_or_model_routing",
    "provider_execution",
    "generated_output_modification",
    "runtime_evolution_application",
)


class CrossDomainInspirationEntry(BaseModel):
    """One advisory cross-domain inspiration entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=220)
    inspiration_kind: CrossDomainInspirationKind
    status: CrossDomainInspirationStatus
    confidence: CrossDomainInspirationConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    inspiration_axis: CrossDomainInspirationAxis
    creative_research_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    creative_research_entry_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    inspiration_summary: str = Field(min_length=1, max_length=420)
    source_domain_distance_score: int = Field(ge=0, le=100)
    analogy_quality_score: int = Field(ge=0, le=100)
    transferability_score: int = Field(ge=0, le=100)
    provenance_traceability_score: int = Field(ge=0, le=100)
    creative_research_alignment_score: int = Field(ge=0, le=100)
    hitl_policy_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    inspiration_score: int = Field(ge=0, le=1_000)
    hitl_required_before_inspiration_discovery_execution: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=44,
    )
    cross_domain_inspiration_discovery_capability_implemented: Literal[True] = True
    cross_domain_inspiration_discovery_metadata_implemented: Literal[True] = True
    creative_research_metadata_used: Literal[True] = True
    inspiration_discovery_execution_implemented: Literal[False] = False
    live_cross_domain_search_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    creative_output_generation_implemented: Literal[False] = False
    creative_output_mutation_implemented: Literal[False] = False
    creative_asset_generation_implemented: Literal[False] = False
    creative_asset_write_implemented: Literal[False] = False
    inspiration_record_write_implemented: Literal[False] = False
    research_recommendation_generation_implemented: Literal[False] = False
    recommendation_execution_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    hitl_decision_application_implemented: Literal[False] = False
    hitl_gate_execution_implemented: Literal[False] = False
    research_execution_policy_application_implemented: Literal[False] = False
    research_execution_authorization_implemented: Literal[False] = False
    research_execution_implemented: Literal[False] = False
    research_plan_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    vector_index_mutation_implemented: Literal[False] = False
    kb_enrichment_execution_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    api_key_inference_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["cross_domain_inspiration_entry.v1"] = (
        CROSS_DOMAIN_INSPIRATION_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.entry_id != f"cross_domain_inspiration::{self.inspiration_kind}":
            raise ValueError("entry_id must match inspiration_kind")
        if self.creative_research_entry_count != len(
            self.creative_research_entry_ids
        ):
            raise ValueError(
                "creative_research_entry_count must match creative research ids"
            )
        if self.inspiration_score != _inspiration_score(
            source_domain_distance_score=self.source_domain_distance_score,
            analogy_quality_score=self.analogy_quality_score,
            transferability_score=self.transferability_score,
            provenance_traceability_score=self.provenance_traceability_score,
            creative_research_alignment_score=self.creative_research_alignment_score,
            hitl_policy_alignment_score=self.hitl_policy_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("inspiration_score must combine source scores")
        if self.status != _inspiration_status(self.inspiration_score):
            raise ValueError("status must match inspiration_score")
        if self.confidence != _inspiration_confidence(self.inspiration_score):
            raise ValueError("confidence must match inspiration_score")
        if not self.hitl_required_before_inspiration_discovery_execution:
            raise ValueError("inspiration discovery execution requires HITL")
        return self


class CrossDomainInspirationPlan(BaseModel):
    """Bounded V6.4 advisory cross-domain inspiration plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cross_domain_inspiration_discovery"] = (
        "cross_domain_inspiration_discovery"
    )
    serialization_version: Literal["cross_domain_inspiration_plan.v1"] = (
        CROSS_DOMAIN_INSPIRATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CROSS_DOMAIN_INSPIRATION_AUTHORITY_BOUNDARY,
        max_length=2800,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    creative_research_role: Literal["creative_research_engine"] = (
        "creative_research_engine"
    )
    creative_research_serialization_version: Literal["creative_research_plan.v1"]
    creative_research_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    creative_research_entry_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[CrossDomainInspirationEntry, ...] = Field(
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
    executed_inspiration_discovery_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    live_cross_domain_search_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_inspiration_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    created_creative_output_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_generated_output_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    executed_research_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    emitted_hitl_request_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_inspiration_score: int = Field(ge=0, le=1_000)
    overall_inspiration_score: int = Field(ge=0, le=1_000)
    overall_inspiration_posture: CrossDomainInspirationPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=44,
    )
    cross_domain_inspiration_discovery_capability_implemented: Literal[True] = True
    cross_domain_inspiration_discovery_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    creative_research_metadata_used: Literal[True] = True
    inspiration_discovery_execution_implemented: Literal[False] = False
    live_cross_domain_search_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    creative_output_generation_implemented: Literal[False] = False
    creative_output_mutation_implemented: Literal[False] = False
    creative_asset_generation_implemented: Literal[False] = False
    creative_asset_write_implemented: Literal[False] = False
    inspiration_record_write_implemented: Literal[False] = False
    research_recommendation_generation_implemented: Literal[False] = False
    recommendation_execution_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    hitl_decision_application_implemented: Literal[False] = False
    hitl_gate_execution_implemented: Literal[False] = False
    research_execution_policy_application_implemented: Literal[False] = False
    research_execution_authorization_implemented: Literal[False] = False
    research_execution_implemented: Literal[False] = False
    research_plan_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    vector_index_mutation_implemented: Literal[False] = False
    kb_enrichment_execution_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    api_key_inference_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
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
            if entry.hitl_required_before_inspiration_discovery_execution
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.executed_inspiration_discovery_ids:
            raise ValueError("executed_inspiration_discovery_ids must remain empty")
        if self.live_cross_domain_search_ids:
            raise ValueError("live_cross_domain_search_ids must remain empty")
        if self.written_inspiration_record_ids:
            raise ValueError("written_inspiration_record_ids must remain empty")
        if self.created_creative_output_ids:
            raise ValueError("created_creative_output_ids must remain empty")
        if self.mutated_generated_output_ids:
            raise ValueError("mutated_generated_output_ids must remain empty")
        if self.executed_research_ids:
            raise ValueError("executed_research_ids must remain empty")
        if self.emitted_hitl_request_ids:
            raise ValueError("emitted_hitl_request_ids must remain empty")
        if self.creative_research_entry_count != len(
            self.creative_research_entry_ids
        ):
            raise ValueError(
                "creative_research_entry_count must match creative research ids"
            )
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 20 roadmap")
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
        if self.highest_inspiration_score != max(
            entry.inspiration_score for entry in self.entries
        ):
            raise ValueError("highest_inspiration_score must match entries")
        if self.overall_inspiration_score != _overall_inspiration_score(
            self.entries
        ):
            raise ValueError("overall_inspiration_score must match entries")
        if self.overall_inspiration_posture != _overall_inspiration_posture(
            self.entries
        ):
            raise ValueError("overall_inspiration_posture must match entries")
        creative_research_ids = set(self.creative_research_entry_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if entry.source_count != self.source_count:
                raise ValueError("entry source_count must match plan")
            if entry.domain_count != self.domain_count:
                raise ValueError("entry domain_count must match plan")
            if not set(entry.creative_research_entry_ids).issubset(
                creative_research_ids
            ):
                raise ValueError("entry creative research ids must be declared")
        return self


def build_cross_domain_inspiration_discovery(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    creative_research: CreativeResearchPlan | None = None,
) -> CrossDomainInspirationPlan:
    """Build V6.4 Task 20 inspiration metadata without live discovery."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    creative_research_plan = creative_research or build_creative_research_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        creative_research=creative_research_plan,
    )
    return CrossDomainInspirationPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=creative_research_plan.checked_at,
        creative_research_serialization_version=(
            CREATIVE_RESEARCH_PLAN_SERIALIZATION_VERSION
        ),
        creative_research_entry_ids=creative_research_plan.entry_ids,
        creative_research_entry_count=len(creative_research_plan.entry_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=creative_research_plan.source_count,
        domain_count=creative_research_plan.domain_count,
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
            if entry.hitl_required_before_inspiration_discovery_execution
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
            1
            for entry in entries
            if entry.hitl_required_before_inspiration_discovery_execution
        ),
        highest_inspiration_score=max(entry.inspiration_score for entry in entries),
        overall_inspiration_score=_overall_inspiration_score(entries),
        overall_inspiration_posture=_overall_inspiration_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def cross_domain_inspiration_entry_by_id(
    entry_id: str,
    plan: CrossDomainInspirationPlan | None = None,
) -> CrossDomainInspirationEntry | None:
    """Return one cross-domain inspiration entry without live discovery."""

    source_plan = plan or build_cross_domain_inspiration_discovery()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def cross_domain_inspiration_entries_for_status(
    status: CrossDomainInspirationStatus,
    plan: CrossDomainInspirationPlan | None = None,
) -> tuple[CrossDomainInspirationEntry, ...]:
    """Return cross-domain inspiration entries by advisory status."""

    source_plan = plan or build_cross_domain_inspiration_discovery()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def cross_domain_inspiration_entries_for_confidence(
    confidence: CrossDomainInspirationConfidence,
    plan: CrossDomainInspirationPlan | None = None,
) -> tuple[CrossDomainInspirationEntry, ...]:
    """Return cross-domain inspiration entries by confidence band."""

    source_plan = plan or build_cross_domain_inspiration_discovery()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    creative_research: CreativeResearchPlan,
) -> tuple[CrossDomainInspirationEntry, ...]:
    return (
        _entry(
            kind="domain_analogy_mapping",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="domain_analogy",
            creative_research_entry_ids=creative_research.entry_ids,
            creative_research=creative_research,
            source_domain_distance_score=82,
            analogy_quality_score=78,
            transferability_score=76,
            provenance_traceability_score=80,
            creative_research_alignment_score=84,
            hitl_policy_alignment_score=82,
            mutation_risk_score=12,
            governance_weight=15,
        ),
        _entry(
            kind="medium_transfer_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="medium_transfer",
            creative_research_entry_ids=(
                "creative_research_engine::medium_constraint_alignment",
                "creative_research_engine::aesthetic_evidence_mapping",
            ),
            creative_research=creative_research,
            source_domain_distance_score=72,
            analogy_quality_score=84,
            transferability_score=88,
            provenance_traceability_score=78,
            creative_research_alignment_score=80,
            hitl_policy_alignment_score=76,
            mutation_risk_score=12,
            governance_weight=20,
        ),
        _entry(
            kind="pattern_translation_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="pattern_translation",
            creative_research_entry_ids=(
                "creative_research_engine::creative_question_framing",
                "creative_research_engine::medium_constraint_alignment",
            ),
            creative_research=creative_research,
            source_domain_distance_score=70,
            analogy_quality_score=76,
            transferability_score=82,
            provenance_traceability_score=82,
            creative_research_alignment_score=78,
            hitl_policy_alignment_score=80,
            mutation_risk_score=10,
            governance_weight=20,
        ),
        _entry(
            kind="inspiration_provenance_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="inspiration_provenance",
            creative_research_entry_ids=(
                "creative_research_engine::aesthetic_evidence_mapping",
                "creative_research_engine::creative_risk_review",
            ),
            creative_research=creative_research,
            source_domain_distance_score=60,
            analogy_quality_score=64,
            transferability_score=68,
            provenance_traceability_score=74,
            creative_research_alignment_score=76,
            hitl_policy_alignment_score=88,
            mutation_risk_score=8,
            governance_weight=25,
        ),
        _entry(
            kind="cross_domain_inspiration_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            creative_research_entry_ids=creative_research.entry_ids,
            creative_research=creative_research,
            source_domain_distance_score=38,
            analogy_quality_score=40,
            transferability_score=42,
            provenance_traceability_score=54,
            creative_research_alignment_score=50,
            hitl_policy_alignment_score=92,
            mutation_risk_score=6,
            governance_weight=30,
        ),
    )


def _entry(
    *,
    kind: CrossDomainInspirationKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: CrossDomainInspirationAxis,
    creative_research_entry_ids: tuple[str, ...],
    creative_research: CreativeResearchPlan,
    source_domain_distance_score: int,
    analogy_quality_score: int,
    transferability_score: int,
    provenance_traceability_score: int,
    creative_research_alignment_score: int,
    hitl_policy_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> CrossDomainInspirationEntry:
    score = _inspiration_score(
        source_domain_distance_score=source_domain_distance_score,
        analogy_quality_score=analogy_quality_score,
        transferability_score=transferability_score,
        provenance_traceability_score=provenance_traceability_score,
        creative_research_alignment_score=creative_research_alignment_score,
        hitl_policy_alignment_score=hitl_policy_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return CrossDomainInspirationEntry(
        entry_id=f"cross_domain_inspiration::{kind}",
        inspiration_kind=kind,
        status=_inspiration_status(score),
        confidence=_inspiration_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        inspiration_axis=axis,
        creative_research_entry_ids=creative_research_entry_ids,
        creative_research_entry_count=len(creative_research_entry_ids),
        source_count=creative_research.source_count,
        domain_count=creative_research.domain_count,
        inspiration_summary=_inspiration_summary(kind),
        source_domain_distance_score=source_domain_distance_score,
        analogy_quality_score=analogy_quality_score,
        transferability_score=transferability_score,
        provenance_traceability_score=provenance_traceability_score,
        creative_research_alignment_score=creative_research_alignment_score,
        hitl_policy_alignment_score=hitl_policy_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        inspiration_score=score,
        hitl_required_before_inspiration_discovery_execution=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(
            kind,
            creative_research_entry_ids,
        ),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"creative_research_entry_count:{len(creative_research_entry_ids)}",
            f"source_count:{creative_research.source_count}",
            f"domain_count:{creative_research.domain_count}",
            f"inspiration_axis:{axis}",
            f"status:{_inspiration_status(score)}",
            f"confidence:{_inspiration_confidence(score)}",
            "hitl_required_before_inspiration_discovery_execution:true",
            "inspiration_discovery_execution_implemented:false",
        ),
    )


def _inspiration_score(
    *,
    source_domain_distance_score: int,
    analogy_quality_score: int,
    transferability_score: int,
    provenance_traceability_score: int,
    creative_research_alignment_score: int,
    hitl_policy_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            source_domain_distance_score * 2
            + analogy_quality_score * 2
            + transferability_score * 2
            + provenance_traceability_score * 2
            + creative_research_alignment_score * 2
            + hitl_policy_alignment_score * 2
            + mutation_risk_score
            + governance_weight,
        ),
    )


def _inspiration_status(score: int) -> CrossDomainInspirationStatus:
    if score >= 900:
        return "guarded"
    if score >= 700:
        return "review_required"
    return "candidate"


def _inspiration_confidence(score: int) -> CrossDomainInspirationConfidence:
    if score >= 900:
        return "guarded"
    if score >= 780:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_inspiration_score(
    entries: tuple[CrossDomainInspirationEntry, ...],
) -> int:
    return round(sum(entry.inspiration_score for entry in entries) / len(entries))


def _overall_inspiration_posture(
    entries: tuple[CrossDomainInspirationEntry, ...],
) -> CrossDomainInspirationPosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[CrossDomainInspirationEntry, ...],
    status: CrossDomainInspirationStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[CrossDomainInspirationEntry, ...],
    *confidences: CrossDomainInspirationConfidence,
) -> tuple[str, ...]:
    return tuple(
        entry.entry_id for entry in entries if entry.confidence in confidences
    )


def _plan_actions(
    entries: tuple[CrossDomainInspirationEntry, ...],
) -> tuple[str, ...]:
    return (
        f"review_cross_domain_inspiration_entries:{len(entries)}",
        "confirm_inspiration_discovery_scope_before_execution",
        "confirm_no_live_cross_domain_search",
        "confirm_no_creative_output_generation",
        "request_hitl_before_inspiration_discovery_execution",
    )


def _entry_actions(kind: CrossDomainInspirationKind) -> tuple[str, ...]:
    actions: dict[CrossDomainInspirationKind, tuple[str, ...]] = {
        "domain_analogy_mapping": (
            "review_domain_analogy_boundary",
            "confirm_analogy_mapping_is_advisory",
            "confirm_no_live_cross_domain_search",
        ),
        "medium_transfer_review": (
            "review_medium_transfer_boundary",
            "confirm_no_prototype_generation",
            "confirm_no_generated_output_mutation",
        ),
        "pattern_translation_review": (
            "review_pattern_translation_boundary",
            "confirm_no_creative_asset_generation",
            "confirm_no_workflow_mutation",
        ),
        "inspiration_provenance_review": (
            "review_inspiration_provenance_boundary",
            "confirm_no_external_source_fetch",
            "confirm_no_inspiration_record_write",
        ),
        "cross_domain_inspiration_governance_gate": (
            "review_cross_domain_inspiration_governance_gate",
            "confirm_hitl_required_before_execution",
            "confirm_no_runtime_evolution",
        ),
    }
    return actions[kind]


def _inspiration_summary(kind: CrossDomainInspirationKind) -> str:
    summaries: dict[CrossDomainInspirationKind, str] = {
        "domain_analogy_mapping": (
            "Models source and target domain analogy posture without live "
            "cross-domain search or research execution."
        ),
        "medium_transfer_review": (
            "Frames how medium-specific ideas could transfer without "
            "generating prototypes, creative assets, or output changes."
        ),
        "pattern_translation_review": (
            "Describes pattern translation readiness without workflow "
            "mutation, generated-output mutation, or provider execution."
        ),
        "inspiration_provenance_review": (
            "Surfaces provenance and traceability posture without fetching "
            "sources or writing inspiration records."
        ),
        "cross_domain_inspiration_governance_gate": (
            "Models the HITL gate required before inspiration discovery "
            "execution, live search, creative output generation, or Runtime "
            "Evolution."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: CrossDomainInspirationKind,
    axis: CrossDomainInspirationAxis,
) -> tuple[str, ...]:
    return (
        "cross_domain_inspiration_discovery",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: CrossDomainInspirationKind,
    creative_research_entry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"inspiration_kind:{kind}",
        f"creative_research_entry_count:{len(creative_research_entry_ids)}",
        "creative_research_metadata_used:true",
        "inspiration_discovery_execution_implemented:false",
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
