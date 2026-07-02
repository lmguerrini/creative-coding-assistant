"""V6.4 advisory research HITL policy metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.research_execution_policy import (
    RESEARCH_EXECUTION_POLICY_PLAN_SERIALIZATION_VERSION,
    ResearchExecutionPolicyPlan,
    build_research_execution_policy,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

ResearchHITLPolicyKind = Literal[
    "execution_approval_policy",
    "source_access_review_policy",
    "mutation_review_policy",
    "recommendation_review_policy",
    "hitl_governance_gate",
]
ResearchHITLPolicyStatus = Literal["candidate", "review_required", "guarded"]
ResearchHITLPolicyConfidence = Literal["low", "medium", "high", "guarded"]
ResearchHITLPolicyPosture = Literal["candidate", "review_required", "guarded"]
ResearchHITLPolicyAxis = Literal[
    "execution_approval",
    "source_access_review",
    "mutation_review",
    "recommendation_review",
    "governance_gate",
]

RESEARCH_HITL_POLICY_ENTRY_SERIALIZATION_VERSION = "research_hitl_policy_entry.v1"
RESEARCH_HITL_POLICY_PLAN_SERIALIZATION_VERSION = "research_hitl_policy_plan.v1"

RESEARCH_HITL_POLICY_AUTHORITY_BOUNDARY = (
    "V6.4 Research HITL Policies expose execution-approval posture, source "
    "access review readiness, mutation review posture, recommendation review "
    "posture, and HITL governance readiness as inspectable advisory metadata "
    "only; it does not emit HITL requests, apply HITL decisions, execute HITL "
    "gates, apply research execution policy, authorize research execution, "
    "execute research, execute recommendations, generate research "
    "recommendations, create research tasks, mutate research plans, control "
    "workflows, mutate workflow graphs, execute workflows, fetch external "
    "sources, browse the web, download papers, mutate source registries, "
    "execute retrieval, mutate retrieval configuration, mutate vector indexes, "
    "enrich the KB, write storage, provision providers, infer API keys, route "
    "providers or models, execute providers, modify generated output, or apply "
    "Runtime Evolution."
)

_ROADMAP_ITEMS = ("Research HITL Policies",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "hitl_request_emission",
    "hitl_decision_application",
    "hitl_gate_execution",
    "research_execution_policy_application",
    "research_execution_authorization",
    "research_execution",
    "recommendation_execution",
    "research_recommendation_generation",
    "recommendation_record_write",
    "research_task_creation",
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


class ResearchHITLPolicyEntry(BaseModel):
    """One advisory research HITL policy entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=200)
    hitl_policy_kind: ResearchHITLPolicyKind
    status: ResearchHITLPolicyStatus
    confidence: ResearchHITLPolicyConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    hitl_axis: ResearchHITLPolicyAxis
    execution_policy_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    execution_policy_entry_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    hitl_summary: str = Field(min_length=1, max_length=380)
    approval_threshold_score: int = Field(ge=0, le=100)
    source_review_score: int = Field(ge=0, le=100)
    mutation_review_score: int = Field(ge=0, le=100)
    recommendation_review_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    hitl_policy_score: int = Field(ge=0, le=1_000)
    hitl_required_before_policy_activation: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=34,
    )
    research_hitl_policies_capability_implemented: Literal[True] = True
    research_hitl_policies_metadata_implemented: Literal[True] = True
    research_execution_policy_metadata_used: Literal[True] = True
    hitl_request_emission_implemented: Literal[False] = False
    hitl_decision_application_implemented: Literal[False] = False
    hitl_gate_execution_implemented: Literal[False] = False
    research_execution_policy_application_implemented: Literal[False] = False
    research_execution_authorization_implemented: Literal[False] = False
    research_execution_implemented: Literal[False] = False
    recommendation_execution_implemented: Literal[False] = False
    research_recommendation_generation_implemented: Literal[False] = False
    recommendation_record_write_implemented: Literal[False] = False
    research_task_creation_implemented: Literal[False] = False
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
    serialization_version: Literal["research_hitl_policy_entry.v1"] = (
        RESEARCH_HITL_POLICY_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.entry_id != f"research_hitl_policies::{self.hitl_policy_kind}":
            raise ValueError("entry_id must match hitl_policy_kind")
        if self.execution_policy_entry_count != len(self.execution_policy_entry_ids):
            raise ValueError(
                "execution_policy_entry_count must match execution policy ids"
            )
        if self.hitl_policy_score != _hitl_policy_score(
            approval_threshold_score=self.approval_threshold_score,
            source_review_score=self.source_review_score,
            mutation_review_score=self.mutation_review_score,
            recommendation_review_score=self.recommendation_review_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("hitl_policy_score must combine source scores")
        if self.status != _hitl_policy_status(self.hitl_policy_score):
            raise ValueError("status must match hitl_policy_score")
        if self.confidence != _hitl_policy_confidence(self.hitl_policy_score):
            raise ValueError("confidence must match hitl_policy_score")
        if not self.hitl_required_before_policy_activation:
            raise ValueError("research HITL policy activation requires HITL")
        return self


class ResearchHITLPoliciesPlan(BaseModel):
    """Bounded V6.4 advisory research HITL policy plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["research_hitl_policies"] = "research_hitl_policies"
    serialization_version: Literal["research_hitl_policy_plan.v1"] = (
        RESEARCH_HITL_POLICY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RESEARCH_HITL_POLICY_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    research_execution_policy_role: Literal["research_execution_policy"] = (
        "research_execution_policy"
    )
    research_execution_policy_serialization_version: Literal[
        "research_execution_policy_plan.v1"
    ]
    execution_policy_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    execution_policy_entry_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[ResearchHITLPolicyEntry, ...] = Field(
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
    emitted_hitl_request_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    applied_hitl_decision_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    executed_hitl_gate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    applied_execution_policy_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    authorized_research_execution_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_workflow_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_hitl_policy_score: int = Field(ge=0, le=1_000)
    overall_hitl_policy_score: int = Field(ge=0, le=1_000)
    overall_hitl_policy_posture: ResearchHITLPolicyPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=34,
    )
    research_hitl_policies_capability_implemented: Literal[True] = True
    research_hitl_policies_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    research_execution_policy_metadata_used: Literal[True] = True
    hitl_request_emission_implemented: Literal[False] = False
    hitl_decision_application_implemented: Literal[False] = False
    hitl_gate_execution_implemented: Literal[False] = False
    research_execution_policy_application_implemented: Literal[False] = False
    research_execution_authorization_implemented: Literal[False] = False
    research_execution_implemented: Literal[False] = False
    recommendation_execution_implemented: Literal[False] = False
    research_recommendation_generation_implemented: Literal[False] = False
    recommendation_record_write_implemented: Literal[False] = False
    research_task_creation_implemented: Literal[False] = False
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
            if entry.hitl_required_before_policy_activation
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.emitted_hitl_request_ids:
            raise ValueError("emitted_hitl_request_ids must remain empty")
        if self.applied_hitl_decision_ids:
            raise ValueError("applied_hitl_decision_ids must remain empty")
        if self.executed_hitl_gate_ids:
            raise ValueError("executed_hitl_gate_ids must remain empty")
        if self.applied_execution_policy_ids:
            raise ValueError("applied_execution_policy_ids must remain empty")
        if self.authorized_research_execution_ids:
            raise ValueError("authorized_research_execution_ids must remain empty")
        if self.mutated_workflow_ids:
            raise ValueError("mutated_workflow_ids must remain empty")
        if self.execution_policy_entry_count != len(
            self.execution_policy_entry_ids
        ):
            raise ValueError(
                "execution_policy_entry_count must match execution policy ids"
            )
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 18 roadmap")
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
        if self.highest_hitl_policy_score != max(
            entry.hitl_policy_score for entry in self.entries
        ):
            raise ValueError("highest_hitl_policy_score must match entries")
        if self.overall_hitl_policy_score != _overall_hitl_policy_score(
            self.entries
        ):
            raise ValueError("overall_hitl_policy_score must match entries")
        if self.overall_hitl_policy_posture != _overall_hitl_policy_posture(
            self.entries
        ):
            raise ValueError("overall_hitl_policy_posture must match entries")
        execution_policy_ids = set(self.execution_policy_entry_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if entry.source_count != self.source_count:
                raise ValueError("entry source_count must match plan")
            if entry.domain_count != self.domain_count:
                raise ValueError("entry domain_count must match plan")
            if not set(entry.execution_policy_entry_ids).issubset(
                execution_policy_ids
            ):
                raise ValueError("entry execution policy ids must be declared")
        return self


def build_research_hitl_policies(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    execution_policy: ResearchExecutionPolicyPlan | None = None,
) -> ResearchHITLPoliciesPlan:
    """Build V6.4 Task 18 HITL policy metadata without emitting HITL."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    execution_policy_plan = execution_policy or build_research_execution_policy(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        execution_policy=execution_policy_plan,
    )
    return ResearchHITLPoliciesPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=execution_policy_plan.checked_at,
        research_execution_policy_serialization_version=(
            RESEARCH_EXECUTION_POLICY_PLAN_SERIALIZATION_VERSION
        ),
        execution_policy_entry_ids=execution_policy_plan.entry_ids,
        execution_policy_entry_count=len(execution_policy_plan.entry_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=execution_policy_plan.source_count,
        domain_count=execution_policy_plan.domain_count,
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
            if entry.hitl_required_before_policy_activation
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
            1 for entry in entries if entry.hitl_required_before_policy_activation
        ),
        highest_hitl_policy_score=max(entry.hitl_policy_score for entry in entries),
        overall_hitl_policy_score=_overall_hitl_policy_score(entries),
        overall_hitl_policy_posture=_overall_hitl_policy_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def research_hitl_policy_entry_by_id(
    entry_id: str,
    plan: ResearchHITLPoliciesPlan | None = None,
) -> ResearchHITLPolicyEntry | None:
    """Return one research HITL policy entry without emitting HITL."""

    source_plan = plan or build_research_hitl_policies()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def research_hitl_policy_entries_for_status(
    status: ResearchHITLPolicyStatus,
    plan: ResearchHITLPoliciesPlan | None = None,
) -> tuple[ResearchHITLPolicyEntry, ...]:
    """Return research HITL policy entries by advisory status."""

    source_plan = plan or build_research_hitl_policies()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def research_hitl_policy_entries_for_confidence(
    confidence: ResearchHITLPolicyConfidence,
    plan: ResearchHITLPoliciesPlan | None = None,
) -> tuple[ResearchHITLPolicyEntry, ...]:
    """Return research HITL policy entries by confidence band."""

    source_plan = plan or build_research_hitl_policies()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    execution_policy: ResearchExecutionPolicyPlan,
) -> tuple[ResearchHITLPolicyEntry, ...]:
    return (
        _entry(
            kind="execution_approval_policy",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="execution_approval",
            execution_policy_entry_ids=execution_policy.entry_ids,
            execution_policy=execution_policy,
            approval_threshold_score=80,
            source_review_score=76,
            mutation_review_score=78,
            recommendation_review_score=70,
            governance_alignment_score=82,
            mutation_risk_score=16,
            governance_weight=20,
        ),
        _entry(
            kind="source_access_review_policy",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="source_access_review",
            execution_policy_entry_ids=(
                "research_execution_policy::source_access_policy",
                "research_execution_policy::execution_scope_policy",
            ),
            execution_policy=execution_policy,
            approval_threshold_score=74,
            source_review_score=88,
            mutation_review_score=76,
            recommendation_review_score=68,
            governance_alignment_score=80,
            mutation_risk_score=16,
            governance_weight=25,
        ),
        _entry(
            kind="mutation_review_policy",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="mutation_review",
            execution_policy_entry_ids=(
                "research_execution_policy::mutation_boundary_policy",
                "research_execution_policy::execution_policy_governance_gate",
            ),
            execution_policy=execution_policy,
            approval_threshold_score=70,
            source_review_score=72,
            mutation_review_score=90,
            recommendation_review_score=70,
            governance_alignment_score=82,
            mutation_risk_score=14,
            governance_weight=20,
        ),
        _entry(
            kind="recommendation_review_policy",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="recommendation_review",
            execution_policy_entry_ids=(
                "research_execution_policy::recommendation_execution_policy",
                "research_execution_policy::execution_scope_policy",
            ),
            execution_policy=execution_policy,
            approval_threshold_score=58,
            source_review_score=62,
            mutation_review_score=64,
            recommendation_review_score=76,
            governance_alignment_score=72,
            mutation_risk_score=12,
            governance_weight=20,
        ),
        _entry(
            kind="hitl_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            execution_policy_entry_ids=execution_policy.entry_ids,
            execution_policy=execution_policy,
            approval_threshold_score=35,
            source_review_score=36,
            mutation_review_score=39,
            recommendation_review_score=40,
            governance_alignment_score=88,
            mutation_risk_score=8,
            governance_weight=30,
        ),
    )


def _entry(
    *,
    kind: ResearchHITLPolicyKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: ResearchHITLPolicyAxis,
    execution_policy_entry_ids: tuple[str, ...],
    execution_policy: ResearchExecutionPolicyPlan,
    approval_threshold_score: int,
    source_review_score: int,
    mutation_review_score: int,
    recommendation_review_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> ResearchHITLPolicyEntry:
    score = _hitl_policy_score(
        approval_threshold_score=approval_threshold_score,
        source_review_score=source_review_score,
        mutation_review_score=mutation_review_score,
        recommendation_review_score=recommendation_review_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return ResearchHITLPolicyEntry(
        entry_id=f"research_hitl_policies::{kind}",
        hitl_policy_kind=kind,
        status=_hitl_policy_status(score),
        confidence=_hitl_policy_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        hitl_axis=axis,
        execution_policy_entry_ids=execution_policy_entry_ids,
        execution_policy_entry_count=len(execution_policy_entry_ids),
        source_count=execution_policy.source_count,
        domain_count=execution_policy.domain_count,
        hitl_summary=_hitl_summary(kind),
        approval_threshold_score=approval_threshold_score,
        source_review_score=source_review_score,
        mutation_review_score=mutation_review_score,
        recommendation_review_score=recommendation_review_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        hitl_policy_score=score,
        hitl_required_before_policy_activation=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(
            kind,
            execution_policy_entry_ids,
        ),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"execution_policy_entry_count:{len(execution_policy_entry_ids)}",
            f"source_count:{execution_policy.source_count}",
            f"domain_count:{execution_policy.domain_count}",
            f"hitl_axis:{axis}",
            f"status:{_hitl_policy_status(score)}",
            f"confidence:{_hitl_policy_confidence(score)}",
            "hitl_required_before_policy_activation:true",
        ),
    )


def _hitl_policy_score(
    *,
    approval_threshold_score: int,
    source_review_score: int,
    mutation_review_score: int,
    recommendation_review_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            approval_threshold_score * 3
            + source_review_score * 2
            + mutation_review_score * 3
            + recommendation_review_score * 2
            + governance_alignment_score * 2
            + mutation_risk_score
            + governance_weight,
        ),
    )


def _hitl_policy_status(score: int) -> ResearchHITLPolicyStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _hitl_policy_confidence(score: int) -> ResearchHITLPolicyConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_hitl_policy_score(
    entries: tuple[ResearchHITLPolicyEntry, ...],
) -> int:
    return round(sum(entry.hitl_policy_score for entry in entries) / len(entries))


def _overall_hitl_policy_posture(
    entries: tuple[ResearchHITLPolicyEntry, ...],
) -> ResearchHITLPolicyPosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[ResearchHITLPolicyEntry, ...],
    status: ResearchHITLPolicyStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[ResearchHITLPolicyEntry, ...],
    *confidences: ResearchHITLPolicyConfidence,
) -> tuple[str, ...]:
    return tuple(
        entry.entry_id for entry in entries if entry.confidence in confidences
    )


def _plan_actions(entries: tuple[ResearchHITLPolicyEntry, ...]) -> tuple[str, ...]:
    return (
        f"review_research_hitl_policy_entries:{len(entries)}",
        "confirm_hitl_policy_scope_before_activation",
        "confirm_no_hitl_request_emission",
        "confirm_no_hitl_decision_application",
        "request_hitl_before_hitl_policy_activation",
    )


def _entry_actions(kind: ResearchHITLPolicyKind) -> tuple[str, ...]:
    actions: dict[ResearchHITLPolicyKind, tuple[str, ...]] = {
        "execution_approval_policy": (
            "review_execution_approval_boundary",
            "confirm_hitl_metadata_is_advisory",
            "confirm_no_research_execution_authorization",
        ),
        "source_access_review_policy": (
            "review_source_access_hitl_boundary",
            "confirm_no_external_source_fetch",
            "confirm_no_retrieval_execution",
        ),
        "mutation_review_policy": (
            "review_mutation_hitl_boundary",
            "confirm_no_workflow_mutation",
            "confirm_no_storage_write",
        ),
        "recommendation_review_policy": (
            "review_recommendation_hitl_boundary",
            "confirm_no_recommendation_execution",
            "confirm_no_task_creation",
        ),
        "hitl_governance_gate": (
            "review_hitl_governance_gate",
            "confirm_no_hitl_request_emission",
            "confirm_no_runtime_evolution",
        ),
    }
    return actions[kind]


def _hitl_summary(kind: ResearchHITLPolicyKind) -> str:
    summaries: dict[ResearchHITLPolicyKind, str] = {
        "execution_approval_policy": (
            "Frames execution approval posture without emitting HITL requests "
            "or authorizing research execution."
        ),
        "source_access_review_policy": (
            "Models HITL source-access review readiness without fetching "
            "sources, browsing, paper download, or retrieval execution."
        ),
        "mutation_review_policy": (
            "Describes HITL mutation review posture without workflow, research "
            "plan, retrieval, KB, or storage mutation."
        ),
        "recommendation_review_policy": (
            "Models HITL recommendation review posture without executing "
            "recommendations or creating research tasks."
        ),
        "hitl_governance_gate": (
            "Models the HITL gate required before HITL request emission, "
            "decision application, execution authorization, or Runtime "
            "Evolution."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: ResearchHITLPolicyKind,
    axis: ResearchHITLPolicyAxis,
) -> tuple[str, ...]:
    return (
        "research_hitl_policies",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: ResearchHITLPolicyKind,
    execution_policy_entry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"hitl_policy_kind:{kind}",
        f"execution_policy_entry_count:{len(execution_policy_entry_ids)}",
        "research_execution_policy_metadata_used:true",
        "no_hitl_request_emitted",
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
