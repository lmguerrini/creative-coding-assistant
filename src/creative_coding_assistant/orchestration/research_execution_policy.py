"""V6.4 advisory research execution policy metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.research_recommendation_engine import (
    RESEARCH_RECOMMENDATION_PLAN_SERIALIZATION_VERSION,
    ResearchRecommendationPlan,
    build_research_recommendation_engine,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

ResearchExecutionPolicyKind = Literal[
    "execution_scope_policy",
    "source_access_policy",
    "mutation_boundary_policy",
    "recommendation_execution_policy",
    "execution_policy_governance_gate",
]
ResearchExecutionPolicyStatus = Literal["candidate", "review_required", "guarded"]
ResearchExecutionPolicyConfidence = Literal["low", "medium", "high", "guarded"]
ResearchExecutionPolicyPosture = Literal["candidate", "review_required", "guarded"]
ResearchExecutionPolicyAxis = Literal[
    "execution_scope",
    "source_access",
    "mutation_boundary",
    "recommendation_execution",
    "governance_gate",
]

RESEARCH_EXECUTION_POLICY_ENTRY_SERIALIZATION_VERSION = (
    "research_execution_policy_entry.v1"
)
RESEARCH_EXECUTION_POLICY_PLAN_SERIALIZATION_VERSION = (
    "research_execution_policy_plan.v1"
)

RESEARCH_EXECUTION_POLICY_AUTHORITY_BOUNDARY = (
    "V6.4 Research Execution Policy exposes execution-scope posture, source "
    "access policy readiness, mutation boundary posture, recommendation "
    "execution policy posture, and governance readiness as inspectable "
    "advisory metadata only; it does not apply research execution policy, "
    "authorize research execution, execute research, execute recommendations, "
    "generate research recommendations, create research tasks, mutate research "
    "plans, control workflows, mutate workflow graphs, execute workflows, "
    "fetch external sources, browse the web, download papers, mutate source "
    "registries, execute retrieval, mutate retrieval configuration, mutate "
    "vector indexes, enrich the KB, write storage, provision providers, infer "
    "API keys, route providers or models, execute providers, modify generated "
    "output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Research Execution Policy",)

_BLOCKED_RUNTIME_BEHAVIORS = (
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
    "source_credibility_scoring_execution",
    "contradiction_detection_execution",
    "research_confidence_scoring_execution",
    "research_gap_discovery_execution",
    "generated_output_modification",
    "runtime_evolution_application",
)


class ResearchExecutionPolicyEntry(BaseModel):
    """One advisory research execution policy entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=200)
    policy_kind: ResearchExecutionPolicyKind
    status: ResearchExecutionPolicyStatus
    confidence: ResearchExecutionPolicyConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    policy_axis: ResearchExecutionPolicyAxis
    recommendation_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    recommendation_entry_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    policy_summary: str = Field(min_length=1, max_length=380)
    execution_scope_score: int = Field(ge=0, le=100)
    source_access_score: int = Field(ge=0, le=100)
    mutation_boundary_score: int = Field(ge=0, le=100)
    recommendation_execution_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    execution_policy_score: int = Field(ge=0, le=1_000)
    hitl_required_before_policy_application: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=34,
    )
    research_execution_policy_capability_implemented: Literal[True] = True
    research_execution_policy_metadata_implemented: Literal[True] = True
    research_recommendation_metadata_used: Literal[True] = True
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
    source_credibility_scoring_execution_implemented: Literal[False] = False
    contradiction_detection_execution_implemented: Literal[False] = False
    research_confidence_scoring_execution_implemented: Literal[False] = False
    research_gap_discovery_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["research_execution_policy_entry.v1"] = (
        RESEARCH_EXECUTION_POLICY_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.entry_id != f"research_execution_policy::{self.policy_kind}":
            raise ValueError("entry_id must match policy_kind")
        if self.recommendation_entry_count != len(self.recommendation_entry_ids):
            raise ValueError("recommendation_entry_count must match recommendation ids")
        if self.execution_policy_score != _execution_policy_score(
            execution_scope_score=self.execution_scope_score,
            source_access_score=self.source_access_score,
            mutation_boundary_score=self.mutation_boundary_score,
            recommendation_execution_score=self.recommendation_execution_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("execution_policy_score must combine source scores")
        if self.status != _policy_status(self.execution_policy_score):
            raise ValueError("status must match execution_policy_score")
        if self.confidence != _policy_confidence(self.execution_policy_score):
            raise ValueError("confidence must match execution_policy_score")
        if not self.hitl_required_before_policy_application:
            raise ValueError("research execution policy application requires HITL")
        return self


class ResearchExecutionPolicyPlan(BaseModel):
    """Bounded V6.4 advisory research execution policy plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["research_execution_policy"] = "research_execution_policy"
    serialization_version: Literal["research_execution_policy_plan.v1"] = (
        RESEARCH_EXECUTION_POLICY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RESEARCH_EXECUTION_POLICY_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    research_recommendation_role: Literal["research_recommendation_engine"] = (
        "research_recommendation_engine"
    )
    research_recommendation_serialization_version: Literal[
        "research_recommendation_plan.v1"
    ]
    recommendation_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    recommendation_entry_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[ResearchExecutionPolicyEntry, ...] = Field(
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
    applied_execution_policy_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    authorized_research_execution_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    executed_research_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    executed_recommendation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_workflow_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_research_plan_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_execution_policy_score: int = Field(ge=0, le=1_000)
    overall_execution_policy_score: int = Field(ge=0, le=1_000)
    overall_execution_policy_posture: ResearchExecutionPolicyPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=34,
    )
    research_execution_policy_capability_implemented: Literal[True] = True
    research_execution_policy_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    research_recommendation_metadata_used: Literal[True] = True
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
    source_credibility_scoring_execution_implemented: Literal[False] = False
    contradiction_detection_execution_implemented: Literal[False] = False
    research_confidence_scoring_execution_implemented: Literal[False] = False
    research_gap_discovery_execution_implemented: Literal[False] = False
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
            if entry.hitl_required_before_policy_application
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.applied_execution_policy_ids:
            raise ValueError("applied_execution_policy_ids must remain empty")
        if self.authorized_research_execution_ids:
            raise ValueError("authorized_research_execution_ids must remain empty")
        if self.executed_research_ids:
            raise ValueError("executed_research_ids must remain empty")
        if self.executed_recommendation_ids:
            raise ValueError("executed_recommendation_ids must remain empty")
        if self.mutated_workflow_ids:
            raise ValueError("mutated_workflow_ids must remain empty")
        if self.mutated_research_plan_ids:
            raise ValueError("mutated_research_plan_ids must remain empty")
        if self.recommendation_entry_count != len(self.recommendation_entry_ids):
            raise ValueError("recommendation_entry_count must match recommendation ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 17 roadmap")
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
        if self.highest_execution_policy_score != max(
            entry.execution_policy_score for entry in self.entries
        ):
            raise ValueError("highest_execution_policy_score must match entries")
        if self.overall_execution_policy_score != _overall_policy_score(self.entries):
            raise ValueError("overall_execution_policy_score must match entries")
        if self.overall_execution_policy_posture != _overall_policy_posture(
            self.entries
        ):
            raise ValueError("overall_execution_policy_posture must match entries")
        recommendation_ids = set(self.recommendation_entry_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if entry.source_count != self.source_count:
                raise ValueError("entry source_count must match plan")
            if entry.domain_count != self.domain_count:
                raise ValueError("entry domain_count must match plan")
            if not set(entry.recommendation_entry_ids).issubset(recommendation_ids):
                raise ValueError("entry recommendation ids must be declared")
        return self


def build_research_execution_policy(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    recommendations: ResearchRecommendationPlan | None = None,
) -> ResearchExecutionPolicyPlan:
    """Build V6.4 Task 17 policy metadata without applying policy."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    recommendation_plan = recommendations or build_research_recommendation_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        recommendations=recommendation_plan,
    )
    return ResearchExecutionPolicyPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=recommendation_plan.checked_at,
        research_recommendation_serialization_version=(
            RESEARCH_RECOMMENDATION_PLAN_SERIALIZATION_VERSION
        ),
        recommendation_entry_ids=recommendation_plan.entry_ids,
        recommendation_entry_count=len(recommendation_plan.entry_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=recommendation_plan.source_count,
        domain_count=recommendation_plan.domain_count,
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
            if entry.hitl_required_before_policy_application
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
            1 for entry in entries if entry.hitl_required_before_policy_application
        ),
        highest_execution_policy_score=max(
            entry.execution_policy_score for entry in entries
        ),
        overall_execution_policy_score=_overall_policy_score(entries),
        overall_execution_policy_posture=_overall_policy_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def research_execution_policy_entry_by_id(
    entry_id: str,
    plan: ResearchExecutionPolicyPlan | None = None,
) -> ResearchExecutionPolicyEntry | None:
    """Return one research execution policy entry without applying policy."""

    source_plan = plan or build_research_execution_policy()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def research_execution_policy_entries_for_status(
    status: ResearchExecutionPolicyStatus,
    plan: ResearchExecutionPolicyPlan | None = None,
) -> tuple[ResearchExecutionPolicyEntry, ...]:
    """Return execution policy entries by advisory status."""

    source_plan = plan or build_research_execution_policy()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def research_execution_policy_entries_for_confidence(
    confidence: ResearchExecutionPolicyConfidence,
    plan: ResearchExecutionPolicyPlan | None = None,
) -> tuple[ResearchExecutionPolicyEntry, ...]:
    """Return execution policy entries by confidence band."""

    source_plan = plan or build_research_execution_policy()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    recommendations: ResearchRecommendationPlan,
) -> tuple[ResearchExecutionPolicyEntry, ...]:
    return (
        _entry(
            kind="execution_scope_policy",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="execution_scope",
            recommendation_entry_ids=recommendations.entry_ids,
            recommendations=recommendations,
            execution_scope_score=82,
            source_access_score=78,
            mutation_boundary_score=80,
            recommendation_execution_score=70,
            governance_alignment_score=82,
            mutation_risk_score=18,
            governance_weight=20,
        ),
        _entry(
            kind="source_access_policy",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="source_access",
            recommendation_entry_ids=(
                "research_recommendation_engine::source_followup_review",
                "research_recommendation_engine::gap_prioritization_review",
            ),
            recommendations=recommendations,
            execution_scope_score=74,
            source_access_score=88,
            mutation_boundary_score=78,
            recommendation_execution_score=68,
            governance_alignment_score=80,
            mutation_risk_score=16,
            governance_weight=25,
        ),
        _entry(
            kind="mutation_boundary_policy",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="mutation_boundary",
            recommendation_entry_ids=(
                "research_recommendation_engine::execution_readiness_review",
                "research_recommendation_engine::recommendation_governance_gate",
            ),
            recommendations=recommendations,
            execution_scope_score=70,
            source_access_score=72,
            mutation_boundary_score=90,
            recommendation_execution_score=72,
            governance_alignment_score=82,
            mutation_risk_score=14,
            governance_weight=20,
        ),
        _entry(
            kind="recommendation_execution_policy",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="recommendation_execution",
            recommendation_entry_ids=(
                "research_recommendation_engine::confidence_improvement_review",
                "research_recommendation_engine::execution_readiness_review",
            ),
            recommendations=recommendations,
            execution_scope_score=58,
            source_access_score=60,
            mutation_boundary_score=62,
            recommendation_execution_score=76,
            governance_alignment_score=72,
            mutation_risk_score=12,
            governance_weight=20,
        ),
        _entry(
            kind="execution_policy_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            recommendation_entry_ids=recommendations.entry_ids,
            recommendations=recommendations,
            execution_scope_score=36,
            source_access_score=38,
            mutation_boundary_score=40,
            recommendation_execution_score=40,
            governance_alignment_score=88,
            mutation_risk_score=8,
            governance_weight=30,
        ),
    )


def _entry(
    *,
    kind: ResearchExecutionPolicyKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: ResearchExecutionPolicyAxis,
    recommendation_entry_ids: tuple[str, ...],
    recommendations: ResearchRecommendationPlan,
    execution_scope_score: int,
    source_access_score: int,
    mutation_boundary_score: int,
    recommendation_execution_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> ResearchExecutionPolicyEntry:
    score = _execution_policy_score(
        execution_scope_score=execution_scope_score,
        source_access_score=source_access_score,
        mutation_boundary_score=mutation_boundary_score,
        recommendation_execution_score=recommendation_execution_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return ResearchExecutionPolicyEntry(
        entry_id=f"research_execution_policy::{kind}",
        policy_kind=kind,
        status=_policy_status(score),
        confidence=_policy_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        policy_axis=axis,
        recommendation_entry_ids=recommendation_entry_ids,
        recommendation_entry_count=len(recommendation_entry_ids),
        source_count=recommendations.source_count,
        domain_count=recommendations.domain_count,
        policy_summary=_policy_summary(kind),
        execution_scope_score=execution_scope_score,
        source_access_score=source_access_score,
        mutation_boundary_score=mutation_boundary_score,
        recommendation_execution_score=recommendation_execution_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        execution_policy_score=score,
        hitl_required_before_policy_application=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(
            kind,
            recommendation_entry_ids,
        ),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"recommendation_entry_count:{len(recommendation_entry_ids)}",
            f"source_count:{recommendations.source_count}",
            f"domain_count:{recommendations.domain_count}",
            f"policy_axis:{axis}",
            f"status:{_policy_status(score)}",
            f"confidence:{_policy_confidence(score)}",
            "hitl_required_before_policy_application:true",
        ),
    )


def _execution_policy_score(
    *,
    execution_scope_score: int,
    source_access_score: int,
    mutation_boundary_score: int,
    recommendation_execution_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            execution_scope_score * 3
            + source_access_score * 2
            + mutation_boundary_score * 3
            + recommendation_execution_score * 2
            + governance_alignment_score * 2
            + mutation_risk_score
            + governance_weight,
        ),
    )


def _policy_status(score: int) -> ResearchExecutionPolicyStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _policy_confidence(score: int) -> ResearchExecutionPolicyConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_policy_score(
    entries: tuple[ResearchExecutionPolicyEntry, ...],
) -> int:
    return round(sum(entry.execution_policy_score for entry in entries) / len(entries))


def _overall_policy_posture(
    entries: tuple[ResearchExecutionPolicyEntry, ...],
) -> ResearchExecutionPolicyPosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[ResearchExecutionPolicyEntry, ...],
    status: ResearchExecutionPolicyStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[ResearchExecutionPolicyEntry, ...],
    *confidences: ResearchExecutionPolicyConfidence,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.confidence in confidences)


def _plan_actions(
    entries: tuple[ResearchExecutionPolicyEntry, ...],
) -> tuple[str, ...]:
    return (
        f"review_research_execution_policy_entries:{len(entries)}",
        "confirm_execution_policy_scope_before_application",
        "confirm_no_research_execution_authorization",
        "confirm_no_workflow_or_plan_mutation",
        "request_hitl_before_policy_application",
    )


def _entry_actions(kind: ResearchExecutionPolicyKind) -> tuple[str, ...]:
    actions: dict[ResearchExecutionPolicyKind, tuple[str, ...]] = {
        "execution_scope_policy": (
            "review_execution_scope_boundary",
            "confirm_policy_metadata_is_advisory",
            "confirm_no_research_execution",
        ),
        "source_access_policy": (
            "review_source_access_boundary",
            "confirm_no_external_source_fetch",
            "confirm_no_retrieval_execution",
        ),
        "mutation_boundary_policy": (
            "review_mutation_boundary_policy",
            "confirm_no_workflow_mutation",
            "confirm_no_storage_write",
        ),
        "recommendation_execution_policy": (
            "review_recommendation_execution_policy",
            "confirm_no_recommendation_execution",
            "confirm_no_task_creation",
        ),
        "execution_policy_governance_gate": (
            "review_execution_policy_hitl_gate",
            "confirm_no_policy_application",
            "confirm_no_runtime_evolution",
        ),
    }
    return actions[kind]


def _policy_summary(kind: ResearchExecutionPolicyKind) -> str:
    summaries: dict[ResearchExecutionPolicyKind, str] = {
        "execution_scope_policy": (
            "Frames execution-scope policy posture without authorizing or "
            "executing research."
        ),
        "source_access_policy": (
            "Models source-access policy readiness without fetching sources, "
            "browsing, downloading papers, or executing retrieval."
        ),
        "mutation_boundary_policy": (
            "Describes mutation-boundary policy posture without workflow, "
            "research plan, retrieval, KB, or storage mutation."
        ),
        "recommendation_execution_policy": (
            "Models recommendation execution policy posture without executing "
            "recommendations or creating research tasks."
        ),
        "execution_policy_governance_gate": (
            "Models the HITL gate required before policy application, research "
            "execution authorization, recommendation execution, or Runtime "
            "Evolution."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: ResearchExecutionPolicyKind,
    axis: ResearchExecutionPolicyAxis,
) -> tuple[str, ...]:
    return (
        "research_execution_policy",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: ResearchExecutionPolicyKind,
    recommendation_entry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"policy_kind:{kind}",
        f"recommendation_entry_count:{len(recommendation_entry_ids)}",
        "research_recommendation_metadata_used:true",
        "no_research_execution_policy_applied",
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
