"""V6.4 advisory creative research engine metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.research_hitl_policies import (
    RESEARCH_HITL_POLICY_PLAN_SERIALIZATION_VERSION,
    ResearchHITLPoliciesPlan,
    build_research_hitl_policies,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

CreativeResearchKind = Literal[
    "creative_question_framing",
    "aesthetic_evidence_mapping",
    "medium_constraint_alignment",
    "creative_risk_review",
    "creative_research_governance_gate",
]
CreativeResearchStatus = Literal["candidate", "review_required", "guarded"]
CreativeResearchConfidence = Literal["low", "medium", "high", "guarded"]
CreativeResearchPosture = Literal["candidate", "review_required", "guarded"]
CreativeResearchAxis = Literal[
    "question_framing",
    "aesthetic_evidence",
    "medium_constraints",
    "creative_risk",
    "governance_gate",
]

CREATIVE_RESEARCH_ENTRY_SERIALIZATION_VERSION = "creative_research_entry.v1"
CREATIVE_RESEARCH_PLAN_SERIALIZATION_VERSION = "creative_research_plan.v1"

CREATIVE_RESEARCH_AUTHORITY_BOUNDARY = (
    "V6.4 Creative Research Engine exposes creative question framing, "
    "aesthetic evidence mapping, medium constraint alignment, creative risk "
    "review, and creative research governance as inspectable advisory "
    "metadata only; it does not generate creative outputs, mutate generated "
    "output, does not discover cross-domain inspiration, execute inspiration "
    "discovery, or write inspiration records; Cross-domain Inspiration "
    "Discovery remains separate Task 20 coverage. It does not generate "
    "research recommendations, "
    "execute recommendations, emit HITL requests, apply HITL decisions, "
    "execute HITL gates, apply research execution policy, authorize research "
    "execution, execute research, mutate research plans, control workflows, "
    "mutate workflow graphs, execute workflows, fetch external sources, browse "
    "the web, download papers, mutate source registries, execute retrieval, "
    "mutate retrieval configuration, mutate vector indexes, enrich the KB, "
    "write storage, provision providers, infer API keys, route providers or "
    "models, execute providers, modify generated output, or apply Runtime "
    "Evolution."
)

_ROADMAP_ITEMS = ("Creative Research Engine",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "creative_output_generation",
    "creative_output_mutation",
    "creative_asset_generation",
    "creative_asset_write",
    "prototype_generation",
    "prototype_execution",
    "cross_domain_inspiration_discovery",
    "inspiration_discovery_execution",
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
    "external_source_fetch",
    "web_browsing",
    "paper_download",
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


class CreativeResearchEntry(BaseModel):
    """One advisory creative research entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=200)
    creative_research_kind: CreativeResearchKind
    status: CreativeResearchStatus
    confidence: CreativeResearchConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    creative_axis: CreativeResearchAxis
    hitl_policy_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    hitl_policy_entry_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    creative_research_summary: str = Field(min_length=1, max_length=420)
    question_framing_score: int = Field(ge=0, le=100)
    source_grounding_score: int = Field(ge=0, le=100)
    creative_novelty_score: int = Field(ge=0, le=100)
    constraint_alignment_score: int = Field(ge=0, le=100)
    provenance_traceability_score: int = Field(ge=0, le=100)
    hitl_policy_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    creative_research_score: int = Field(ge=0, le=1_000)
    hitl_required_before_creative_research_execution: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=44,
    )
    creative_research_engine_capability_implemented: Literal[True] = True
    creative_research_engine_metadata_implemented: Literal[True] = True
    research_hitl_policy_metadata_used: Literal[True] = True
    creative_output_generation_implemented: Literal[False] = False
    creative_output_mutation_implemented: Literal[False] = False
    creative_asset_generation_implemented: Literal[False] = False
    creative_asset_write_implemented: Literal[False] = False
    prototype_generation_implemented: Literal[False] = False
    prototype_execution_implemented: Literal[False] = False
    cross_domain_inspiration_discovery_implemented: Literal[False] = False
    inspiration_discovery_execution_implemented: Literal[False] = False
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
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
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
    serialization_version: Literal["creative_research_entry.v1"] = (
        CREATIVE_RESEARCH_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.entry_id != f"creative_research_engine::{self.creative_research_kind}":
            raise ValueError("entry_id must match creative_research_kind")
        if self.hitl_policy_entry_count != len(self.hitl_policy_entry_ids):
            raise ValueError("hitl_policy_entry_count must match HITL policy ids")
        if self.creative_research_score != _creative_research_score(
            question_framing_score=self.question_framing_score,
            source_grounding_score=self.source_grounding_score,
            creative_novelty_score=self.creative_novelty_score,
            constraint_alignment_score=self.constraint_alignment_score,
            provenance_traceability_score=self.provenance_traceability_score,
            hitl_policy_alignment_score=self.hitl_policy_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("creative_research_score must combine source scores")
        if self.status != _creative_research_status(self.creative_research_score):
            raise ValueError("status must match creative_research_score")
        if self.confidence != _creative_research_confidence(
            self.creative_research_score
        ):
            raise ValueError("confidence must match creative_research_score")
        if not self.hitl_required_before_creative_research_execution:
            raise ValueError("creative research execution requires HITL")
        return self


class CreativeResearchPlan(BaseModel):
    """Bounded V6.4 advisory creative research plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_research_engine"] = "creative_research_engine"
    serialization_version: Literal["creative_research_plan.v1"] = (
        CREATIVE_RESEARCH_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_RESEARCH_AUTHORITY_BOUNDARY,
        max_length=2600,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    research_hitl_policy_role: Literal["research_hitl_policies"] = (
        "research_hitl_policies"
    )
    research_hitl_policy_serialization_version: Literal["research_hitl_policy_plan.v1"]
    hitl_policy_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    hitl_policy_entry_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[CreativeResearchEntry, ...] = Field(
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
    created_creative_output_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_generated_output_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    discovered_inspiration_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    generated_recommendation_ids: tuple[str, ...] = Field(
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
    highest_creative_research_score: int = Field(ge=0, le=1_000)
    overall_creative_research_score: int = Field(ge=0, le=1_000)
    overall_creative_research_posture: CreativeResearchPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=44,
    )
    creative_research_engine_capability_implemented: Literal[True] = True
    creative_research_engine_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    research_hitl_policy_metadata_used: Literal[True] = True
    creative_output_generation_implemented: Literal[False] = False
    creative_output_mutation_implemented: Literal[False] = False
    creative_asset_generation_implemented: Literal[False] = False
    creative_asset_write_implemented: Literal[False] = False
    prototype_generation_implemented: Literal[False] = False
    prototype_execution_implemented: Literal[False] = False
    cross_domain_inspiration_discovery_implemented: Literal[False] = False
    inspiration_discovery_execution_implemented: Literal[False] = False
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
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
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
            if entry.hitl_required_before_creative_research_execution
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.created_creative_output_ids:
            raise ValueError("created_creative_output_ids must remain empty")
        if self.mutated_generated_output_ids:
            raise ValueError("mutated_generated_output_ids must remain empty")
        if self.discovered_inspiration_ids:
            raise ValueError("discovered_inspiration_ids must remain empty")
        if self.generated_recommendation_ids:
            raise ValueError("generated_recommendation_ids must remain empty")
        if self.executed_research_ids:
            raise ValueError("executed_research_ids must remain empty")
        if self.emitted_hitl_request_ids:
            raise ValueError("emitted_hitl_request_ids must remain empty")
        if self.hitl_policy_entry_count != len(self.hitl_policy_entry_ids):
            raise ValueError("hitl_policy_entry_count must match HITL policy ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 19 roadmap")
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
        if self.highest_creative_research_score != max(
            entry.creative_research_score for entry in self.entries
        ):
            raise ValueError("highest_creative_research_score must match entries")
        if self.overall_creative_research_score != _overall_creative_research_score(
            self.entries
        ):
            raise ValueError("overall_creative_research_score must match entries")
        if self.overall_creative_research_posture != _overall_creative_research_posture(
            self.entries
        ):
            raise ValueError("overall_creative_research_posture must match entries")
        hitl_policy_ids = set(self.hitl_policy_entry_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if entry.source_count != self.source_count:
                raise ValueError("entry source_count must match plan")
            if entry.domain_count != self.domain_count:
                raise ValueError("entry domain_count must match plan")
            if not set(entry.hitl_policy_entry_ids).issubset(hitl_policy_ids):
                raise ValueError("entry HITL policy ids must be declared")
        return self


def build_creative_research_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    hitl_policy: ResearchHITLPoliciesPlan | None = None,
) -> CreativeResearchPlan:
    """Build V6.4 Task 19 creative research metadata without execution."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    hitl_policy_plan = hitl_policy or build_research_hitl_policies(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        hitl_policy=hitl_policy_plan,
    )
    return CreativeResearchPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=hitl_policy_plan.checked_at,
        research_hitl_policy_serialization_version=(
            RESEARCH_HITL_POLICY_PLAN_SERIALIZATION_VERSION
        ),
        hitl_policy_entry_ids=hitl_policy_plan.entry_ids,
        hitl_policy_entry_count=len(hitl_policy_plan.entry_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=hitl_policy_plan.source_count,
        domain_count=hitl_policy_plan.domain_count,
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
            if entry.hitl_required_before_creative_research_execution
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
            if entry.hitl_required_before_creative_research_execution
        ),
        highest_creative_research_score=max(
            entry.creative_research_score for entry in entries
        ),
        overall_creative_research_score=_overall_creative_research_score(entries),
        overall_creative_research_posture=_overall_creative_research_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def creative_research_entry_by_id(
    entry_id: str,
    plan: CreativeResearchPlan | None = None,
) -> CreativeResearchEntry | None:
    """Return one creative research entry without executing research."""

    source_plan = plan or build_creative_research_engine()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def creative_research_entries_for_status(
    status: CreativeResearchStatus,
    plan: CreativeResearchPlan | None = None,
) -> tuple[CreativeResearchEntry, ...]:
    """Return creative research entries by advisory status."""

    source_plan = plan or build_creative_research_engine()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def creative_research_entries_for_confidence(
    confidence: CreativeResearchConfidence,
    plan: CreativeResearchPlan | None = None,
) -> tuple[CreativeResearchEntry, ...]:
    """Return creative research entries by confidence band."""

    source_plan = plan or build_creative_research_engine()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    hitl_policy: ResearchHITLPoliciesPlan,
) -> tuple[CreativeResearchEntry, ...]:
    return (
        _entry(
            kind="creative_question_framing",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="question_framing",
            hitl_policy_entry_ids=hitl_policy.entry_ids,
            hitl_policy=hitl_policy,
            question_framing_score=82,
            source_grounding_score=78,
            creative_novelty_score=76,
            constraint_alignment_score=80,
            provenance_traceability_score=75,
            hitl_policy_alignment_score=84,
            mutation_risk_score=12,
            governance_weight=15,
        ),
        _entry(
            kind="aesthetic_evidence_mapping",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="aesthetic_evidence",
            hitl_policy_entry_ids=(
                "research_hitl_policies::source_access_review_policy",
                "research_hitl_policies::execution_approval_policy",
            ),
            hitl_policy=hitl_policy,
            question_framing_score=72,
            source_grounding_score=86,
            creative_novelty_score=82,
            constraint_alignment_score=76,
            provenance_traceability_score=84,
            hitl_policy_alignment_score=78,
            mutation_risk_score=12,
            governance_weight=20,
        ),
        _entry(
            kind="medium_constraint_alignment",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="medium_constraints",
            hitl_policy_entry_ids=(
                "research_hitl_policies::mutation_review_policy",
                "research_hitl_policies::execution_approval_policy",
            ),
            hitl_policy=hitl_policy,
            question_framing_score=68,
            source_grounding_score=74,
            creative_novelty_score=72,
            constraint_alignment_score=90,
            provenance_traceability_score=78,
            hitl_policy_alignment_score=82,
            mutation_risk_score=10,
            governance_weight=20,
        ),
        _entry(
            kind="creative_risk_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="creative_risk",
            hitl_policy_entry_ids=(
                "research_hitl_policies::recommendation_review_policy",
                "research_hitl_policies::mutation_review_policy",
            ),
            hitl_policy=hitl_policy,
            question_framing_score=60,
            source_grounding_score=66,
            creative_novelty_score=70,
            constraint_alignment_score=72,
            provenance_traceability_score=74,
            hitl_policy_alignment_score=88,
            mutation_risk_score=8,
            governance_weight=25,
        ),
        _entry(
            kind="creative_research_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            hitl_policy_entry_ids=hitl_policy.entry_ids,
            hitl_policy=hitl_policy,
            question_framing_score=38,
            source_grounding_score=40,
            creative_novelty_score=42,
            constraint_alignment_score=44,
            provenance_traceability_score=52,
            hitl_policy_alignment_score=92,
            mutation_risk_score=6,
            governance_weight=30,
        ),
    )


def _entry(
    *,
    kind: CreativeResearchKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: CreativeResearchAxis,
    hitl_policy_entry_ids: tuple[str, ...],
    hitl_policy: ResearchHITLPoliciesPlan,
    question_framing_score: int,
    source_grounding_score: int,
    creative_novelty_score: int,
    constraint_alignment_score: int,
    provenance_traceability_score: int,
    hitl_policy_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> CreativeResearchEntry:
    score = _creative_research_score(
        question_framing_score=question_framing_score,
        source_grounding_score=source_grounding_score,
        creative_novelty_score=creative_novelty_score,
        constraint_alignment_score=constraint_alignment_score,
        provenance_traceability_score=provenance_traceability_score,
        hitl_policy_alignment_score=hitl_policy_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return CreativeResearchEntry(
        entry_id=f"creative_research_engine::{kind}",
        creative_research_kind=kind,
        status=_creative_research_status(score),
        confidence=_creative_research_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        creative_axis=axis,
        hitl_policy_entry_ids=hitl_policy_entry_ids,
        hitl_policy_entry_count=len(hitl_policy_entry_ids),
        source_count=hitl_policy.source_count,
        domain_count=hitl_policy.domain_count,
        creative_research_summary=_creative_research_summary(kind),
        question_framing_score=question_framing_score,
        source_grounding_score=source_grounding_score,
        creative_novelty_score=creative_novelty_score,
        constraint_alignment_score=constraint_alignment_score,
        provenance_traceability_score=provenance_traceability_score,
        hitl_policy_alignment_score=hitl_policy_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        creative_research_score=score,
        hitl_required_before_creative_research_execution=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(
            kind,
            hitl_policy_entry_ids,
        ),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"hitl_policy_entry_count:{len(hitl_policy_entry_ids)}",
            f"source_count:{hitl_policy.source_count}",
            f"domain_count:{hitl_policy.domain_count}",
            f"creative_axis:{axis}",
            f"status:{_creative_research_status(score)}",
            f"confidence:{_creative_research_confidence(score)}",
            "hitl_required_before_creative_research_execution:true",
            "cross_domain_inspiration_discovery_implemented:false",
        ),
    )


def _creative_research_score(
    *,
    question_framing_score: int,
    source_grounding_score: int,
    creative_novelty_score: int,
    constraint_alignment_score: int,
    provenance_traceability_score: int,
    hitl_policy_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            question_framing_score * 2
            + source_grounding_score * 2
            + creative_novelty_score * 2
            + constraint_alignment_score * 2
            + provenance_traceability_score * 2
            + hitl_policy_alignment_score * 2
            + mutation_risk_score
            + governance_weight,
        ),
    )


def _creative_research_status(score: int) -> CreativeResearchStatus:
    if score >= 900:
        return "guarded"
    if score >= 700:
        return "review_required"
    return "candidate"


def _creative_research_confidence(score: int) -> CreativeResearchConfidence:
    if score >= 900:
        return "guarded"
    if score >= 780:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_creative_research_score(
    entries: tuple[CreativeResearchEntry, ...],
) -> int:
    return round(sum(entry.creative_research_score for entry in entries) / len(entries))


def _overall_creative_research_posture(
    entries: tuple[CreativeResearchEntry, ...],
) -> CreativeResearchPosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[CreativeResearchEntry, ...],
    status: CreativeResearchStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[CreativeResearchEntry, ...],
    *confidences: CreativeResearchConfidence,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.confidence in confidences)


def _plan_actions(entries: tuple[CreativeResearchEntry, ...]) -> tuple[str, ...]:
    return (
        f"review_creative_research_entries:{len(entries)}",
        "confirm_creative_research_scope_before_execution",
        "confirm_no_creative_output_generation",
        "confirm_no_cross_domain_inspiration_discovery",
        "request_hitl_before_creative_research_execution",
    )


def _entry_actions(kind: CreativeResearchKind) -> tuple[str, ...]:
    actions: dict[CreativeResearchKind, tuple[str, ...]] = {
        "creative_question_framing": (
            "review_creative_question_boundary",
            "confirm_question_frame_is_advisory",
            "confirm_no_research_execution",
        ),
        "aesthetic_evidence_mapping": (
            "review_aesthetic_evidence_boundary",
            "confirm_no_external_source_fetch",
            "confirm_no_generated_output_mutation",
        ),
        "medium_constraint_alignment": (
            "review_medium_constraint_boundary",
            "confirm_no_prototype_generation",
            "confirm_no_workflow_mutation",
        ),
        "creative_risk_review": (
            "review_creative_risk_boundary",
            "confirm_no_recommendation_generation",
            "confirm_no_cross_domain_inspiration_discovery",
        ),
        "creative_research_governance_gate": (
            "review_creative_research_governance_gate",
            "confirm_hitl_required_before_execution",
            "confirm_no_runtime_evolution",
        ),
    }
    return actions[kind]


def _creative_research_summary(kind: CreativeResearchKind) -> str:
    summaries: dict[CreativeResearchKind, str] = {
        "creative_question_framing": (
            "Frames creative research questions without executing research, "
            "generating outputs, or mutating research plans."
        ),
        "aesthetic_evidence_mapping": (
            "Models how aesthetic evidence should stay source-aware and "
            "provenance-aware without fetching external sources."
        ),
        "medium_constraint_alignment": (
            "Describes medium and implementation constraints without "
            "generating prototypes, assets, or workflow changes."
        ),
        "creative_risk_review": (
            "Surfaces creative research risks without recommendations, "
            "cross-domain inspiration discovery, or output mutation."
        ),
        "creative_research_governance_gate": (
            "Models the HITL gate required before creative research execution, "
            "creative output generation, inspiration discovery, or Runtime "
            "Evolution."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: CreativeResearchKind,
    axis: CreativeResearchAxis,
) -> tuple[str, ...]:
    return (
        "creative_research_engine",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: CreativeResearchKind,
    hitl_policy_entry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"creative_research_kind:{kind}",
        f"hitl_policy_entry_count:{len(hitl_policy_entry_ids)}",
        "research_hitl_policy_metadata_used:true",
        "cross_domain_inspiration_discovery_implemented:false",
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
