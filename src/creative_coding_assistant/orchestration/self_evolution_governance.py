"""V6.5 advisory governance and safety metadata for self evolution."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)
from creative_coding_assistant.orchestration.self_evolution_common import (
    CROSS_CUTTING_CONTRACTS,
    UPSTREAM_CAPABILITIES,
)
from creative_coding_assistant.orchestration.self_evolution_core_surface import (
    CORE_ROADMAP_ITEMS,
)
from creative_coding_assistant.orchestration.self_evolution_secondary_surface import (
    SECONDARY_REPORT_SECTIONS,
    SelfEvolutionAdvisoryReportEntry,
    SelfEvolutionSecondarySurfacePlan,
    build_self_evolution_secondary_surface,
)

SelfEvolutionGovernanceStatus = Literal["blocked", "review_required", "guarded"]
SelfEvolutionGovernancePriority = Literal[
    "standard",
    "elevated",
    "critical",
    "guarded",
]
SelfEvolutionGovernancePosture = Literal["blocked", "review_required", "guarded"]

SELF_EVOLUTION_GOVERNANCE_BOUNDARY_SERIALIZATION_VERSION = (
    "self_evolution_governance_boundary.v1"
)
SELF_EVOLUTION_GOVERNANCE_PLAN_SERIALIZATION_VERSION = (
    "self_evolution_governance_plan.v1"
)
SELF_EVOLUTION_GOVERNANCE_AUTHORITY_BOUNDARY = (
    "V6.5 Self Evolution Governance and Safety exposes governance, safety, "
    "HITL, explainability, capability ownership, downstream impact, rollback "
    "feasibility, and no-automation boundaries as inspectable metadata "
    "composed from the V6.5 core and secondary surfaces. It may explain why "
    "a proposal exists, which V6.1 through V6.4 capabilities produced the "
    "signal, and which downstream systems may be affected; it does not "
    "enforce governance policies, enforce safety policies, emit HITL "
    "requests, request human input, activate automation, apply proposals, "
    "execute rollback, generate report artifacts, write storage, apply "
    "Runtime Evolution, rewrite prompts, mutate workflows, mutate routing, "
    "mutate memory, mutate retrieval, execute providers, invoke agents, "
    "modify generated output, or apply HITL decisions."
)

GOVERNANCE_BLOCKED_RUNTIME_BEHAVIORS = (
    "governance_policy_enforcement",
    "safety_policy_enforcement",
    "hitl_request_emission",
    "human_input_request",
    "automation_activation",
    "proposal_application",
    "rollback_execution",
    "report_artifact_generation",
    "runtime_evolution_application",
    "autonomous_code_mutation",
    "prompt_rewriting",
    "workflow_mutation",
    "routing_mutation",
    "memory_mutation",
    "retrieval_mutation",
    "storage_write",
    "provider_execution",
    "agent_invocation",
    "generated_output_mutation",
    "hitl_decision_application",
)

_SOURCE_SURFACE_ROLES = (
    "self_evolution_core_surface",
    "self_evolution_secondary_surface",
)


class SelfEvolutionGovernanceBoundary(BaseModel):
    """One advisory governance boundary for a V6.5 roadmap item."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    boundary_id: str = Field(min_length=1, max_length=220)
    status: SelfEvolutionGovernanceStatus
    priority: SelfEvolutionGovernancePriority
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    roadmap_item: str = Field(min_length=1, max_length=120)
    plan_role: str = Field(min_length=1, max_length=120)
    report_id: str = Field(min_length=1, max_length=220)
    source_surface_roles: tuple[str, ...] = Field(min_length=2, max_length=2)
    source_serialization_versions: tuple[str, ...] = Field(
        min_length=2,
        max_length=2,
    )
    source_item_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_item_count: int = Field(ge=6, le=6)
    proposal_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    proposal_count: int = Field(ge=5, le=5)
    hitl_required_proposal_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    hitl_requirement_count: int = Field(ge=5, le=5)
    explainability_signal_count: int = Field(ge=6, le=6)
    ownership_boundary_check_count: int = Field(ge=4, le=8)
    cross_capability_governance_check_count: int = Field(ge=4, le=8)
    no_automation_weight: int = Field(ge=0, le=240)
    safety_weight: int = Field(ge=0, le=240)
    governance_score: int = Field(ge=0, le=1_000)
    hitl_required_before_governance_application: Literal[True] = True
    upstream_capabilities: tuple[str, ...] = Field(min_length=4, max_length=4)
    upstream_signal_source_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    downstream_systems: tuple[str, ...] = Field(min_length=1, max_length=24)
    report_sections: tuple[str, ...] = Field(min_length=6, max_length=6)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    review_requirement: str = Field(min_length=1, max_length=520)
    explainability_requirement: str = Field(min_length=1, max_length=520)
    ownership_boundary: str = Field(min_length=1, max_length=520)
    downstream_impact_boundary: str = Field(min_length=1, max_length=520)
    rollback_boundary: str = Field(min_length=1, max_length=520)
    no_automation_boundary: str = Field(min_length=1, max_length=520)
    safety_boundary: str = Field(min_length=1, max_length=520)
    governance_tags: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=16)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=20,
        max_length=20,
    )
    applied_governance_boundary_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    enforced_safety_policy_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    emitted_hitl_request_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    requested_human_input_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    activated_automation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    applied_evolution_proposal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    executed_rollback_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    generated_report_artifact_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_storage_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_prompt_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_workflow_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_routing_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_memory_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_retrieval_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    provider_execution_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_output_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    self_evolution_governance_implemented: Literal[True] = True
    governance_boundary_metadata_implemented: Literal[True] = True
    hitl_boundary_metadata_implemented: Literal[True] = True
    explainability_boundary_metadata_implemented: Literal[True] = True
    no_automation_boundary_metadata_implemented: Literal[True] = True
    safety_boundary_metadata_implemented: Literal[True] = True
    all_roadmap_items_traceable: Literal[True] = True
    all_proposals_traceable: Literal[True] = True
    core_surface_foundation_used: Literal[True] = True
    secondary_surface_foundation_used: Literal[True] = True
    evolution_proposal_contract_implemented: Literal[True] = True
    evolution_graph_metadata_implemented: Literal[True] = True
    evolution_explainability_report_implemented: Literal[True] = True
    proposal_impact_model_implemented: Literal[True] = True
    cost_benefit_model_implemented: Literal[True] = True
    risk_model_implemented: Literal[True] = True
    rollback_strategy_model_implemented: Literal[True] = True
    capability_ownership_boundary_check_implemented: Literal[True] = True
    cross_capability_governance_check_implemented: Literal[True] = True
    governance_policy_enforcement_implemented: Literal[False] = False
    safety_policy_enforcement_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    automation_activation_implemented: Literal[False] = False
    proposal_application_implemented: Literal[False] = False
    rollback_execution_implemented: Literal[False] = False
    report_artifact_generation_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    prompt_rewriting_implemented: Literal[False] = False
    workflow_mutation_implemented: Literal[False] = False
    routing_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    hitl_decision_application_implemented: Literal[False] = False
    serialization_version: Literal["self_evolution_governance_boundary.v1"] = (
        SELF_EVOLUTION_GOVERNANCE_BOUNDARY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _boundary_matches_contract(self) -> Self:
        if self.boundary_id != f"self_evolution_governance::{self.plan_role}":
            raise ValueError("boundary_id must match plan_role")
        if self.report_id != f"self_evolution_advisory_report::{self.plan_role}":
            raise ValueError("report_id must match plan_role")
        if self.roadmap_item not in CORE_ROADMAP_ITEMS:
            raise ValueError("roadmap_item must be a V6.5 roadmap item")
        if self.source_surface_roles != _SOURCE_SURFACE_ROLES:
            raise ValueError("source_surface_roles must match V6.5 surfaces")
        if self.source_item_count != len(self.source_item_ids):
            raise ValueError("source_item_count must match source_item_ids")
        if self.proposal_count != len(self.proposal_ids):
            raise ValueError("proposal_count must match proposal_ids")
        if self.hitl_requirement_count != len(self.hitl_required_proposal_ids):
            raise ValueError("hitl_requirement_count must match proposal ids")
        if self.upstream_capabilities != UPSTREAM_CAPABILITIES:
            raise ValueError("upstream_capabilities must include V6.1 through V6.4")
        if self.report_sections != SECONDARY_REPORT_SECTIONS:
            raise ValueError("report_sections must match report contract")
        if self.cross_cutting_contracts != CROSS_CUTTING_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.5 contracts")
        if self.blocked_runtime_behaviors != GOVERNANCE_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match governance boundary")
        if self.governance_score != self_evolution_governance_score(
            source_item_count=self.source_item_count,
            hitl_requirement_count=self.hitl_requirement_count,
            explainability_signal_count=self.explainability_signal_count,
            ownership_boundary_check_count=self.ownership_boundary_check_count,
            cross_capability_governance_check_count=(
                self.cross_capability_governance_check_count
            ),
            no_automation_weight=self.no_automation_weight,
            safety_weight=self.safety_weight,
        ):
            raise ValueError("governance_score must combine boundary inputs")
        if self.status != self_evolution_governance_status(self.governance_score):
            raise ValueError("status must match governance_score")
        if self.priority != self_evolution_governance_priority(
            self.governance_score,
            self.status,
        ):
            raise ValueError("priority must match governance_score")
        empty_fields = (
            self.applied_governance_boundary_ids,
            self.enforced_safety_policy_ids,
            self.emitted_hitl_request_ids,
            self.requested_human_input_ids,
            self.activated_automation_ids,
            self.applied_evolution_proposal_ids,
            self.executed_rollback_ids,
            self.generated_report_artifact_ids,
            self.written_storage_record_ids,
            self.mutated_prompt_ids,
            self.mutated_workflow_ids,
            self.mutated_routing_ids,
            self.mutated_memory_ids,
            self.mutated_retrieval_ids,
            self.provider_execution_ids,
            self.mutated_output_ids,
        )
        if any(empty_fields):
            raise ValueError("governance boundary mutation ids must be empty")
        return self


class SelfEvolutionGovernancePlan(BaseModel):
    """Advisory governance and safety plan over V6.5 evolution metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["self_evolution_governance_safety"] = (
        "self_evolution_governance_safety"
    )
    serialization_version: Literal["self_evolution_governance_plan.v1"] = (
        SELF_EVOLUTION_GOVERNANCE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=SELF_EVOLUTION_GOVERNANCE_AUTHORITY_BOUNDARY,
        max_length=3000,
    )
    source_surface_roles: tuple[str, ...] = Field(min_length=2, max_length=2)
    source_surface_serialization_versions: tuple[str, ...] = Field(
        min_length=2,
        max_length=2,
    )
    source_core_surface_plan_count: int = Field(ge=22, le=22)
    source_secondary_report_count: int = Field(ge=22, le=22)
    source_proposal_count: int = Field(ge=110, le=110)
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    covered_roadmap_items: tuple[str, ...] = Field(min_length=22, max_length=22)
    covered_roadmap_item_count: int = Field(ge=22, le=22)
    proposal_ids: tuple[str, ...] = Field(min_length=110, max_length=110)
    proposal_count: int = Field(ge=110, le=110)
    hitl_required_proposal_ids: tuple[str, ...] = Field(min_length=110, max_length=110)
    hitl_required_proposal_count: int = Field(ge=110, le=110)
    upstream_capabilities: tuple[str, ...] = Field(min_length=4, max_length=4)
    upstream_signal_source_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    upstream_signal_source_count: int = Field(ge=4, le=4)
    governance_boundaries: tuple[SelfEvolutionGovernanceBoundary, ...] = Field(
        min_length=22,
        max_length=22,
    )
    governance_boundary_ids: tuple[str, ...] = Field(min_length=22, max_length=22)
    governance_boundary_count: int = Field(ge=22, le=22)
    guarded_boundary_ids: tuple[str, ...] = Field(min_length=22, max_length=22)
    guarded_boundary_count: int = Field(ge=22, le=22)
    hitl_required_boundary_ids: tuple[str, ...] = Field(min_length=22, max_length=22)
    hitl_required_boundary_count: int = Field(ge=22, le=22)
    highest_governance_score: int = Field(ge=0, le=1_000)
    overall_governance_score: int = Field(ge=0, le=1_000)
    overall_governance_posture: SelfEvolutionGovernancePosture
    all_downstream_systems: tuple[str, ...] = Field(min_length=1, max_length=120)
    report_sections: tuple[str, ...] = Field(min_length=6, max_length=6)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=20,
        max_length=20,
    )
    applied_governance_boundary_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=22,
    )
    enforced_safety_policy_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=22,
    )
    emitted_hitl_request_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=22,
    )
    requested_human_input_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=22,
    )
    activated_automation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=22,
    )
    applied_evolution_proposal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    executed_rollback_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    generated_report_artifact_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    written_storage_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    mutated_prompt_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=110)
    mutated_workflow_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=110)
    mutated_routing_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=110)
    mutated_memory_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=110)
    mutated_retrieval_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    provider_execution_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    mutated_output_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=110)
    self_evolution_governance_implemented: Literal[True] = True
    governance_boundary_metadata_implemented: Literal[True] = True
    hitl_boundary_metadata_implemented: Literal[True] = True
    explainability_boundary_metadata_implemented: Literal[True] = True
    no_automation_boundary_metadata_implemented: Literal[True] = True
    safety_boundary_metadata_implemented: Literal[True] = True
    all_roadmap_items_traceable: Literal[True] = True
    all_proposals_traceable: Literal[True] = True
    core_surface_foundation_used: Literal[True] = True
    secondary_surface_foundation_used: Literal[True] = True
    evolution_proposal_contract_implemented: Literal[True] = True
    evolution_graph_metadata_implemented: Literal[True] = True
    evolution_explainability_report_implemented: Literal[True] = True
    proposal_impact_model_implemented: Literal[True] = True
    cost_benefit_model_implemented: Literal[True] = True
    risk_model_implemented: Literal[True] = True
    rollback_strategy_model_implemented: Literal[True] = True
    capability_ownership_boundary_check_implemented: Literal[True] = True
    cross_capability_governance_check_implemented: Literal[True] = True
    governance_policy_enforcement_implemented: Literal[False] = False
    safety_policy_enforcement_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    automation_activation_implemented: Literal[False] = False
    proposal_application_implemented: Literal[False] = False
    rollback_execution_implemented: Literal[False] = False
    report_artifact_generation_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    prompt_rewriting_implemented: Literal[False] = False
    workflow_mutation_implemented: Literal[False] = False
    routing_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    hitl_decision_application_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_governance_boundaries(self) -> Self:
        if self.source_surface_roles != _SOURCE_SURFACE_ROLES:
            raise ValueError("source_surface_roles must match V6.5 surfaces")
        if self.governance_boundary_ids != tuple(
            boundary.boundary_id for boundary in self.governance_boundaries
        ):
            raise ValueError("governance_boundary_ids must match boundaries")
        if len(set(self.governance_boundary_ids)) != len(
            self.governance_boundary_ids
        ):
            raise ValueError("governance_boundary_ids must be unique")
        if self.governance_boundary_count != len(self.governance_boundaries):
            raise ValueError("governance_boundary_count must match boundaries")
        if self.covered_roadmap_items != tuple(
            boundary.roadmap_item for boundary in self.governance_boundaries
        ):
            raise ValueError("covered_roadmap_items must match boundaries")
        if self.covered_roadmap_items != CORE_ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.5 roadmap")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.proposal_ids != tuple(
            proposal_id
            for boundary in self.governance_boundaries
            for proposal_id in boundary.proposal_ids
        ):
            raise ValueError("proposal_ids must match boundaries")
        if len(set(self.proposal_ids)) != len(self.proposal_ids):
            raise ValueError("proposal_ids must be unique")
        if self.proposal_count != len(self.proposal_ids):
            raise ValueError("proposal_count must match proposal_ids")
        if self.hitl_required_proposal_count != len(
            self.hitl_required_proposal_ids
        ):
            raise ValueError("hitl_required_proposal_count must match proposals")
        if self.guarded_boundary_ids != tuple(
            boundary.boundary_id
            for boundary in self.governance_boundaries
            if boundary.status == "guarded"
        ):
            raise ValueError("guarded_boundary_ids must match boundaries")
        if self.guarded_boundary_count != len(self.guarded_boundary_ids):
            raise ValueError("guarded_boundary_count must match boundaries")
        if self.hitl_required_boundary_ids != tuple(
            boundary.boundary_id
            for boundary in self.governance_boundaries
            if boundary.hitl_required_before_governance_application
        ):
            raise ValueError("hitl_required_boundary_ids must match boundaries")
        if self.hitl_required_boundary_count != len(self.hitl_required_boundary_ids):
            raise ValueError("hitl_required_boundary_count must match boundaries")
        if self.highest_governance_score != max(
            boundary.governance_score for boundary in self.governance_boundaries
        ):
            raise ValueError("highest_governance_score must match boundaries")
        if self.overall_governance_score != _overall_governance_score(
            self.governance_boundaries
        ):
            raise ValueError("overall_governance_score must match boundaries")
        if self.overall_governance_posture != _overall_governance_posture(
            self.governance_boundaries
        ):
            raise ValueError("overall_governance_posture must match boundaries")
        if self.upstream_capabilities != UPSTREAM_CAPABILITIES:
            raise ValueError("upstream_capabilities must include V6.1 through V6.4")
        if self.upstream_signal_source_count != len(self.upstream_signal_source_ids):
            raise ValueError("upstream_signal_source_count must match sources")
        if self.all_downstream_systems != _unique_strings(
            tuple(
                downstream
                for boundary in self.governance_boundaries
                for downstream in boundary.downstream_systems
            )
        ):
            raise ValueError("all_downstream_systems must match boundaries")
        if self.report_sections != SECONDARY_REPORT_SECTIONS:
            raise ValueError("report_sections must match report contract")
        if self.cross_cutting_contracts != CROSS_CUTTING_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.5 contracts")
        if self.blocked_runtime_behaviors != GOVERNANCE_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match governance boundary")
        empty_fields = (
            self.applied_governance_boundary_ids,
            self.enforced_safety_policy_ids,
            self.emitted_hitl_request_ids,
            self.requested_human_input_ids,
            self.activated_automation_ids,
            self.applied_evolution_proposal_ids,
            self.executed_rollback_ids,
            self.generated_report_artifact_ids,
            self.written_storage_record_ids,
            self.mutated_prompt_ids,
            self.mutated_workflow_ids,
            self.mutated_routing_ids,
            self.mutated_memory_ids,
            self.mutated_retrieval_ids,
            self.provider_execution_ids,
            self.mutated_output_ids,
        )
        if any(empty_fields):
            raise ValueError("governance plan mutation ids must be empty")
        if any(not boundary.advisory_only for boundary in self.governance_boundaries):
            raise ValueError("all governance boundaries must be advisory only")
        return self


def build_self_evolution_governance(
    secondary_surface: SelfEvolutionSecondarySurfacePlan | None = None,
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> SelfEvolutionGovernancePlan:
    """Build advisory governance metadata without enforcing it."""

    surface = secondary_surface or build_self_evolution_secondary_surface(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    boundaries = tuple(
        _build_governance_boundary(entry) for entry in surface.report_entries
    )
    return SelfEvolutionGovernancePlan(
        source_surface_roles=_SOURCE_SURFACE_ROLES,
        source_surface_serialization_versions=(
            surface.source_core_surface_serialization_version,
            surface.serialization_version,
        ),
        source_core_surface_plan_count=surface.source_core_surface_plan_count,
        source_secondary_report_count=surface.report_entry_count,
        source_proposal_count=surface.proposal_count,
        route_name=surface.route_name,
        task_type=surface.task_type,
        execution_mode_id=surface.execution_mode_id,
        covered_roadmap_items=surface.covered_roadmap_items,
        covered_roadmap_item_count=surface.covered_roadmap_item_count,
        proposal_ids=surface.proposal_ids,
        proposal_count=surface.proposal_count,
        hitl_required_proposal_ids=surface.hitl_required_proposal_ids,
        hitl_required_proposal_count=surface.hitl_required_proposal_count,
        upstream_capabilities=surface.upstream_capabilities,
        upstream_signal_source_ids=surface.upstream_signal_source_ids,
        upstream_signal_source_count=surface.upstream_signal_source_count,
        governance_boundaries=boundaries,
        governance_boundary_ids=tuple(
            boundary.boundary_id for boundary in boundaries
        ),
        governance_boundary_count=len(boundaries),
        guarded_boundary_ids=tuple(
            boundary.boundary_id
            for boundary in boundaries
            if boundary.status == "guarded"
        ),
        guarded_boundary_count=sum(
            1 for boundary in boundaries if boundary.status == "guarded"
        ),
        hitl_required_boundary_ids=tuple(
            boundary.boundary_id
            for boundary in boundaries
            if boundary.hitl_required_before_governance_application
        ),
        hitl_required_boundary_count=sum(
            1
            for boundary in boundaries
            if boundary.hitl_required_before_governance_application
        ),
        highest_governance_score=max(
            boundary.governance_score for boundary in boundaries
        ),
        overall_governance_score=_overall_governance_score(boundaries),
        overall_governance_posture=_overall_governance_posture(boundaries),
        all_downstream_systems=surface.all_downstream_systems,
        report_sections=SECONDARY_REPORT_SECTIONS,
        cross_cutting_contracts=CROSS_CUTTING_CONTRACTS,
        blocked_runtime_behaviors=GOVERNANCE_BLOCKED_RUNTIME_BEHAVIORS,
    )


def self_evolution_governance_boundary_by_id(
    boundary_id: str,
    plan: SelfEvolutionGovernancePlan | None = None,
) -> SelfEvolutionGovernanceBoundary | None:
    """Return one governance boundary without applying it."""

    source_plan = plan or build_self_evolution_governance()
    for boundary in source_plan.governance_boundaries:
        if boundary.boundary_id == boundary_id:
            return boundary
    return None


def self_evolution_governance_boundary_by_roadmap_item(
    roadmap_item: str,
    plan: SelfEvolutionGovernancePlan | None = None,
) -> SelfEvolutionGovernanceBoundary | None:
    """Return one governance boundary for a roadmap item."""

    source_plan = plan or build_self_evolution_governance()
    for boundary in source_plan.governance_boundaries:
        if boundary.roadmap_item == roadmap_item:
            return boundary
    return None


def self_evolution_governance_boundaries_for_status(
    status: SelfEvolutionGovernanceStatus,
    plan: SelfEvolutionGovernancePlan | None = None,
) -> tuple[SelfEvolutionGovernanceBoundary, ...]:
    """Return governance boundaries by advisory status."""

    source_plan = plan or build_self_evolution_governance()
    return tuple(
        boundary
        for boundary in source_plan.governance_boundaries
        if boundary.status == status
    )


def self_evolution_governance_boundaries_for_priority(
    priority: SelfEvolutionGovernancePriority,
    plan: SelfEvolutionGovernancePlan | None = None,
) -> tuple[SelfEvolutionGovernanceBoundary, ...]:
    """Return governance boundaries by advisory priority."""

    source_plan = plan or build_self_evolution_governance()
    return tuple(
        boundary
        for boundary in source_plan.governance_boundaries
        if boundary.priority == priority
    )


def self_evolution_governance_score(
    *,
    source_item_count: int,
    hitl_requirement_count: int,
    explainability_signal_count: int,
    ownership_boundary_check_count: int,
    cross_capability_governance_check_count: int,
    no_automation_weight: int,
    safety_weight: int,
) -> int:
    """Score governance metadata without enforcing it."""

    return min(
        1_000,
        max(
            0,
            source_item_count * 20
            + hitl_requirement_count * 65
            + explainability_signal_count * 25
            + ownership_boundary_check_count * 20
            + cross_capability_governance_check_count * 20
            + no_automation_weight
            + safety_weight,
        ),
    )


def self_evolution_governance_status(
    score: int,
) -> SelfEvolutionGovernanceStatus:
    """Classify governance posture from score."""

    if score >= 900:
        return "guarded"
    if score >= 700:
        return "review_required"
    return "blocked"


def self_evolution_governance_priority(
    score: int,
    status: SelfEvolutionGovernanceStatus,
) -> SelfEvolutionGovernancePriority:
    """Classify governance priority from score and status."""

    if status == "guarded":
        return "guarded"
    if score >= 800:
        return "critical"
    if score >= 700:
        return "elevated"
    return "standard"


def _build_governance_boundary(
    entry: SelfEvolutionAdvisoryReportEntry,
) -> SelfEvolutionGovernanceBoundary:
    score = self_evolution_governance_score(
        source_item_count=1 + entry.proposal_count,
        hitl_requirement_count=entry.hitl_required_proposal_count,
        explainability_signal_count=len(entry.report_sections),
        ownership_boundary_check_count=len(entry.ownership_boundary_checks),
        cross_capability_governance_check_count=len(entry.governance_checks),
        no_automation_weight=180,
        safety_weight=180,
    )
    return SelfEvolutionGovernanceBoundary(
        boundary_id=f"self_evolution_governance::{entry.plan_role}",
        status=self_evolution_governance_status(score),
        priority=self_evolution_governance_priority(
            score,
            self_evolution_governance_status(score),
        ),
        route_name=entry.route_name,
        task_type=entry.task_type,
        execution_mode_id=entry.execution_mode_id,
        roadmap_item=entry.roadmap_item,
        plan_role=entry.plan_role,
        report_id=entry.report_id,
        source_surface_roles=_SOURCE_SURFACE_ROLES,
        source_serialization_versions=(
            entry.source_core_surface_serialization_version,
            "self_evolution_secondary_surface.v1",
        ),
        source_item_ids=(entry.report_id, *entry.proposal_ids),
        source_item_count=1 + entry.proposal_count,
        proposal_ids=entry.proposal_ids,
        proposal_count=entry.proposal_count,
        hitl_required_proposal_ids=entry.hitl_required_proposal_ids,
        hitl_requirement_count=entry.hitl_required_proposal_count,
        explainability_signal_count=len(entry.report_sections),
        ownership_boundary_check_count=len(entry.ownership_boundary_checks),
        cross_capability_governance_check_count=len(entry.governance_checks),
        no_automation_weight=180,
        safety_weight=180,
        governance_score=score,
        upstream_capabilities=entry.upstream_capabilities,
        upstream_signal_source_ids=entry.upstream_signal_source_ids,
        downstream_systems=entry.downstream_systems,
        report_sections=entry.report_sections,
        cross_cutting_contracts=entry.cross_cutting_contracts,
        review_requirement=(
            f"{entry.roadmap_item} proposals require human review before "
            "governance, safety, rollback, or evolution application."
        ),
        explainability_requirement=(
            "Governance metadata must preserve why each proposal exists, which "
            "V6.1 through V6.4 signal sources contributed, and which "
            "downstream systems may be affected."
        ),
        ownership_boundary=(
            f"{entry.roadmap_item} remains advisory; source capabilities keep "
            "ownership of learning, memory, knowledge, and research state."
        ),
        downstream_impact_boundary=(
            "Downstream systems are named for review only and are not mutated: "
            + ", ".join(entry.downstream_systems)
            + "."
        ),
        rollback_boundary=(
            "Rollback feasibility is recorded as metadata only; rollback "
            "execution is outside V6.5 without explicit HITL approval."
        ),
        no_automation_boundary=(
            "No automation is activated by this boundary, including proposal "
            "application, governance enforcement, report generation, or "
            "Runtime Evolution."
        ),
        safety_boundary=(
            "Safety posture is inspectable metadata only and cannot enforce "
            "policy, request humans, execute providers, or mutate outputs."
        ),
        governance_tags=(
            "hitl_required",
            "explainable",
            "advisory_only",
            "no_automation",
            "ownership_preserved",
        ),
        advisory_actions=(
            "surface_governance_boundary_for_review",
            "preserve_human_controlled_application_boundary",
            "verify_no_automation_before_any_future_application",
        ),
        evidence=(
            f"report_id:{entry.report_id}",
            f"roadmap_item:{entry.roadmap_item}",
            f"proposal_count:{entry.proposal_count}",
            f"hitl_requirement_count:{entry.hitl_required_proposal_count}",
            f"explainability_signal_count:{len(entry.report_sections)}",
            f"governance_score:{score}",
            "governance_policy_enforcement:false",
            "safety_policy_enforcement:false",
            "hitl_request_emission:false",
            "automation_activation:false",
            "proposal_application:false",
            "runtime_evolution_application:false",
        ),
        blocked_runtime_behaviors=GOVERNANCE_BLOCKED_RUNTIME_BEHAVIORS,
    )


def _overall_governance_score(
    boundaries: tuple[SelfEvolutionGovernanceBoundary, ...],
) -> int:
    base = sum(boundary.governance_score for boundary in boundaries) // len(boundaries)
    guarded_count = sum(1 for boundary in boundaries if boundary.status == "guarded")
    return min(1_000, base + guarded_count * 5)


def _overall_governance_posture(
    boundaries: tuple[SelfEvolutionGovernanceBoundary, ...],
) -> SelfEvolutionGovernancePosture:
    if any(boundary.status == "guarded" for boundary in boundaries):
        return "guarded"
    if any(boundary.status == "review_required" for boundary in boundaries):
        return "review_required"
    return "blocked"


def _unique_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return tuple(unique)
