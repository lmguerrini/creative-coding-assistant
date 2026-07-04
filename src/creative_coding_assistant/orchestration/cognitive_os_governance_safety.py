"""V6.6 Cognitive OS governance and safety boundary metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
    COGNITIVE_OS_ROADMAP_ITEMS,
    CognitiveOSCapability,
    CognitiveOSLayer,
)
from creative_coding_assistant.orchestration.cognitive_os_secondary_surface import (
    COGNITIVE_OS_FOUNDATION_SYSTEMS,
    COGNITIVE_OS_SECONDARY_REPORT_SECTIONS,
    CognitiveOSSecondarySurfaceEntry,
    CognitiveOSSecondarySurfacePlan,
    build_cognitive_os_secondary_surface,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

CognitiveOSGovernanceStatus = Literal["blocked", "review_required", "guarded"]
CognitiveOSGovernancePriority = Literal[
    "standard",
    "elevated",
    "critical",
    "guarded",
]
CognitiveOSGovernancePosture = Literal["blocked", "review_required", "guarded"]

COGNITIVE_OS_GOVERNANCE_BOUNDARY_SERIALIZATION_VERSION = (
    "cognitive_os_governance_boundary.v1"
)
COGNITIVE_OS_GOVERNANCE_PLAN_SERIALIZATION_VERSION = "cognitive_os_governance_safety.v1"
COGNITIVE_OS_GOVERNANCE_TASK_ITEM = "Governance and Safety"
COGNITIVE_OS_GOVERNANCE_SOURCE_ROLES = (
    "cognitive_os_core_surface",
    "cognitive_os_secondary_surface",
)
COGNITIVE_OS_GOVERNANCE_SOURCE_SERIALIZATION_VERSIONS = (
    "cognitive_os_core_surface.v1",
    "cognitive_os_secondary_surface.v1",
)
COGNITIVE_OS_GOVERNANCE_BLOCKED_RUNTIME_BEHAVIORS = (
    "governance_policy_enforcement",
    "safety_policy_enforcement",
    "hitl_request_emission",
    "human_input_request",
    "automation_activation",
    "core_surface_activation",
    "secondary_surface_activation",
    "execution_graph_application",
    "runtime_evolution_application",
    "autonomous_code_mutation",
    "workflow_mutation",
    "routing_mutation",
    "prompt_mutation",
    "memory_mutation",
    "retrieval_mutation",
    "storage_write",
    "provider_execution",
    "agent_invocation",
    "generated_output_mutation",
    "hitl_decision_application",
)
COGNITIVE_OS_GOVERNANCE_AUTHORITY_BOUNDARY = (
    "V6.6 Cognitive OS Governance and Safety exposes governance, safety, "
    "HITL, explainability, capability ownership, dependency traceability, "
    "and no-automation boundaries as inspectable metadata composed from the "
    "V6.6 core and secondary surfaces. It preserves all 24 roadmap items, "
    "V5 Decision Engine context, V6 Learning, Memory, Knowledge, Research, "
    "Self Evolution, and Cognitive Core sequence metadata. It does not "
    "enforce governance or safety policies, emit HITL requests, request "
    "human input, activate automation, activate core or secondary surfaces, "
    "execute workflow nodes, traverse execution edges, apply routing, write "
    "storage, execute providers, invoke agents, mutate prompts, workflows, "
    "memory, retrieval, generated output, runtime state, apply HITL "
    "decisions, or apply Runtime Evolution."
)


class CognitiveOSGovernanceBoundary(BaseModel):
    """One advisory governance boundary for a Cognitive OS capability surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    boundary_id: str = Field(min_length=1, max_length=190)
    status: CognitiveOSGovernanceStatus
    priority: CognitiveOSGovernancePriority
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    source_surface_roles: tuple[str, ...] = Field(min_length=2, max_length=2)
    source_serialization_versions: tuple[str, ...] = Field(min_length=2, max_length=2)
    source_core_surface_id: str = Field(min_length=1, max_length=190)
    source_secondary_surface_id: str = Field(min_length=1, max_length=190)
    consolidation_unit_id: str = Field(min_length=1, max_length=190)
    execution_node_id: str = Field(min_length=1, max_length=190)
    hitl_id: str = Field(min_length=1, max_length=190)
    safety_id: str = Field(min_length=1, max_length=190)
    explanation_id: str = Field(min_length=1, max_length=190)
    route_decision_id: str = Field(min_length=1, max_length=190)
    plan_id: str = Field(min_length=1, max_length=190)
    schedule_id: str = Field(min_length=1, max_length=190)
    source_item_ids: tuple[str, ...] = Field(min_length=10, max_length=10)
    source_item_count: int = Field(ge=10, le=10)
    capability_id: str = Field(min_length=1, max_length=80)
    capability_name: CognitiveOSCapability
    cognitive_layer: CognitiveOSLayer
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    surface_sequence_position: int = Field(ge=1, le=6)
    dependency_depth: int = Field(ge=0, le=5)
    source_core_surface_score: int = Field(ge=0, le=100)
    source_secondary_surface_score: int = Field(ge=0, le=100)
    governed_roadmap_items: tuple[str, ...] = Field(min_length=24, max_length=24)
    governed_roadmap_item_count: int = Field(ge=24, le=24)
    foundation_systems: tuple[str, ...] = Field(min_length=7, max_length=7)
    report_sections: tuple[str, ...] = Field(min_length=6, max_length=6)
    hitl_requirement_count: int = Field(ge=5, le=5)
    explainability_signal_count: int = Field(ge=6, le=6)
    no_automation_weight: int = Field(ge=0, le=240)
    safety_weight: int = Field(ge=0, le=240)
    governance_score: int = Field(ge=0, le=1_000)
    hitl_required_before_governance_application: Literal[True] = True
    review_requirement: str = Field(min_length=1, max_length=560)
    explainability_requirement: str = Field(min_length=1, max_length=560)
    ownership_boundary: str = Field(min_length=1, max_length=560)
    dependency_boundary: str = Field(min_length=1, max_length=560)
    hitl_boundary: str = Field(min_length=1, max_length=560)
    no_automation_boundary: str = Field(min_length=1, max_length=560)
    safety_boundary: str = Field(min_length=1, max_length=560)
    governance_tags: tuple[str, ...] = Field(min_length=6, max_length=10)
    advisory_actions: tuple[str, ...] = Field(min_length=3, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=6, max_length=14)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=20, max_length=20)
    applied_governance_boundary_ids: tuple[str, ...] = Field(default_factory=tuple)
    enforced_safety_policy_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    requested_human_input_ids: tuple[str, ...] = Field(default_factory=tuple)
    activated_automation_ids: tuple[str, ...] = Field(default_factory=tuple)
    activated_core_surface_ids: tuple[str, ...] = Field(default_factory=tuple)
    activated_secondary_surface_ids: tuple[str, ...] = Field(default_factory=tuple)
    executed_node_ids: tuple[str, ...] = Field(default_factory=tuple)
    traversed_edge_ids: tuple[str, ...] = Field(default_factory=tuple)
    applied_route_decision_ids: tuple[str, ...] = Field(default_factory=tuple)
    written_storage_record_ids: tuple[str, ...] = Field(default_factory=tuple)
    provider_execution_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_prompt_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_workflow_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_memory_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_retrieval_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_output_ids: tuple[str, ...] = Field(default_factory=tuple)
    applied_hitl_decision_ids: tuple[str, ...] = Field(default_factory=tuple)
    cognitive_os_governance_safety_implemented: Literal[True] = True
    governance_boundary_metadata_implemented: Literal[True] = True
    hitl_boundary_metadata_implemented: Literal[True] = True
    explainability_boundary_metadata_implemented: Literal[True] = True
    no_automation_boundary_metadata_implemented: Literal[True] = True
    safety_boundary_metadata_implemented: Literal[True] = True
    all_roadmap_items_traceable: Literal[True] = True
    core_surface_foundation_used: Literal[True] = True
    secondary_surface_foundation_used: Literal[True] = True
    governance_policy_enforcement_implemented: Literal[False] = False
    safety_policy_enforcement_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    automation_activation_implemented: Literal[False] = False
    core_surface_activation_implemented: Literal[False] = False
    secondary_surface_activation_implemented: Literal[False] = False
    execution_application_implemented: Literal[False] = False
    routing_application_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    workflow_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    hitl_decision_application_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["cognitive_os_governance_boundary.v1"] = (
        COGNITIVE_OS_GOVERNANCE_BOUNDARY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _boundary_matches_sources_and_contract(self) -> Self:
        expected_boundary_id = f"cognitive_os_governance::{self.capability_id}"
        if self.boundary_id != expected_boundary_id:
            raise ValueError("boundary_id must match capability_id")
        expected_core_id = f"cognitive_os_core::{self.capability_id}"
        if self.source_core_surface_id != expected_core_id:
            raise ValueError("source_core_surface_id must match capability_id")
        expected_secondary_id = f"cognitive_os_secondary::{self.capability_id}"
        if self.source_secondary_surface_id != expected_secondary_id:
            raise ValueError("source_secondary_surface_id must match capability_id")
        expected_unit_id = f"core_os::{self.capability_id}"
        if self.consolidation_unit_id != expected_unit_id:
            raise ValueError("consolidation_unit_id must match capability_id")
        expected_execution_id = f"unified_execution::{self.capability_id}"
        if self.execution_node_id != expected_execution_id:
            raise ValueError("execution_node_id must match capability_id")
        expected_hitl_id = f"cognitive_hitl::{self.capability_id}"
        if self.hitl_id != expected_hitl_id:
            raise ValueError("hitl_id must match capability_id")
        expected_safety_id = f"cognitive_safety::{self.capability_id}"
        if self.safety_id != expected_safety_id:
            raise ValueError("safety_id must match capability_id")
        expected_explanation_id = f"cognitive_explanation::{self.capability_id}"
        if self.explanation_id != expected_explanation_id:
            raise ValueError("explanation_id must match capability_id")
        expected_route_id = f"cognitive_router::{self.capability_id}"
        if self.route_decision_id != expected_route_id:
            raise ValueError("route_decision_id must match capability_id")
        expected_plan_id = f"cognitive_planner::{self.capability_id}"
        if self.plan_id != expected_plan_id:
            raise ValueError("plan_id must match capability_id")
        expected_schedule_id = f"cognitive_scheduler::{self.capability_id}"
        if self.schedule_id != expected_schedule_id:
            raise ValueError("schedule_id must match capability_id")
        if self.source_surface_roles != COGNITIVE_OS_GOVERNANCE_SOURCE_ROLES:
            raise ValueError("source_surface_roles must match V6.6 surfaces")
        if (
            self.source_serialization_versions
            != COGNITIVE_OS_GOVERNANCE_SOURCE_SERIALIZATION_VERSIONS
        ):
            raise ValueError("source_serialization_versions must match surfaces")
        if self.source_item_ids != (
            self.source_core_surface_id,
            self.source_secondary_surface_id,
            self.consolidation_unit_id,
            self.execution_node_id,
            self.hitl_id,
            self.safety_id,
            self.explanation_id,
            self.route_decision_id,
            self.plan_id,
            self.schedule_id,
        ):
            raise ValueError("source_item_ids must match boundary sources")
        if self.source_item_count != len(self.source_item_ids):
            raise ValueError("source_item_count must match source_item_ids")
        if self.governed_roadmap_items != COGNITIVE_OS_ROADMAP_ITEMS:
            raise ValueError("governed_roadmap_items must match V6.6 roadmap")
        if self.governed_roadmap_item_count != len(self.governed_roadmap_items):
            raise ValueError("governed_roadmap_item_count must match roadmap")
        if self.foundation_systems != COGNITIVE_OS_FOUNDATION_SYSTEMS:
            raise ValueError("foundation_systems must match V5/V6 foundations")
        if self.report_sections != COGNITIVE_OS_SECONDARY_REPORT_SECTIONS:
            raise ValueError("report_sections must match secondary surface contract")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if (
            self.blocked_runtime_behaviors
            != COGNITIVE_OS_GOVERNANCE_BLOCKED_RUNTIME_BEHAVIORS
        ):
            raise ValueError("blocked_runtime_behaviors must match governance boundary")
        if self.governance_score != cognitive_os_governance_score(
            source_item_count=self.source_item_count,
            governed_roadmap_item_count=self.governed_roadmap_item_count,
            hitl_requirement_count=self.hitl_requirement_count,
            explainability_signal_count=self.explainability_signal_count,
            no_automation_weight=self.no_automation_weight,
            safety_weight=self.safety_weight,
        ):
            raise ValueError("governance_score must combine boundary inputs")
        if self.status != cognitive_os_governance_status(self.governance_score):
            raise ValueError("status must match governance_score")
        if self.priority != cognitive_os_governance_priority(
            self.governance_score,
            self.status,
        ):
            raise ValueError("priority must match governance_score")
        if any(
            (
                self.applied_governance_boundary_ids,
                self.enforced_safety_policy_ids,
                self.emitted_hitl_request_ids,
                self.requested_human_input_ids,
                self.activated_automation_ids,
                self.activated_core_surface_ids,
                self.activated_secondary_surface_ids,
                self.executed_node_ids,
                self.traversed_edge_ids,
                self.applied_route_decision_ids,
                self.written_storage_record_ids,
                self.provider_execution_ids,
                self.mutated_prompt_ids,
                self.mutated_workflow_ids,
                self.mutated_memory_ids,
                self.mutated_retrieval_ids,
                self.mutated_output_ids,
                self.applied_hitl_decision_ids,
            )
        ):
            raise ValueError("governance boundary mutation ids must be empty")
        return self


class CognitiveOSGovernanceSafetyPlan(BaseModel):
    """Advisory governance and safety plan over V6.6 OS surfaces."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cognitive_os_governance_safety"] = "cognitive_os_governance_safety"
    serialization_version: Literal["cognitive_os_governance_safety.v1"] = (
        COGNITIVE_OS_GOVERNANCE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COGNITIVE_OS_GOVERNANCE_AUTHORITY_BOUNDARY,
        max_length=3600,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    source_surface_roles: tuple[str, ...] = Field(min_length=2, max_length=2)
    source_surface_serialization_versions: tuple[str, ...] = Field(
        min_length=2,
        max_length=2,
    )
    source_core_surface_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_core_surface_count: int = Field(ge=6, le=6)
    source_secondary_surface_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_secondary_surface_count: int = Field(ge=6, le=6)
    source_hitl_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_hitl_count: int = Field(ge=6, le=6)
    source_safety_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_safety_count: int = Field(ge=6, le=6)
    source_explanation_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_explanation_count: int = Field(ge=6, le=6)
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    foundation_systems: tuple[str, ...] = Field(min_length=7, max_length=7)
    foundation_system_count: int = Field(ge=7, le=7)
    governed_roadmap_items: tuple[str, ...] = Field(min_length=24, max_length=24)
    governed_roadmap_item_count: int = Field(ge=24, le=24)
    governance_boundaries: tuple[CognitiveOSGovernanceBoundary, ...] = Field(
        min_length=6,
        max_length=6,
    )
    governance_boundary_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    governance_boundary_count: int = Field(ge=6, le=6)
    guarded_boundary_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    guarded_boundary_count: int = Field(ge=6, le=6)
    hitl_required_boundary_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    hitl_required_boundary_count: int = Field(ge=6, le=6)
    highest_governance_score: int = Field(ge=0, le=1_000)
    overall_governance_score: int = Field(ge=0, le=1_000)
    overall_governance_posture: CognitiveOSGovernancePosture
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    report_sections: tuple[str, ...] = Field(min_length=6, max_length=6)
    covered_task_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_task_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=20, max_length=20)
    applied_governance_boundary_ids: tuple[str, ...] = Field(default_factory=tuple)
    enforced_safety_policy_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    requested_human_input_ids: tuple[str, ...] = Field(default_factory=tuple)
    activated_automation_ids: tuple[str, ...] = Field(default_factory=tuple)
    activated_core_surface_ids: tuple[str, ...] = Field(default_factory=tuple)
    activated_secondary_surface_ids: tuple[str, ...] = Field(default_factory=tuple)
    executed_node_ids: tuple[str, ...] = Field(default_factory=tuple)
    traversed_edge_ids: tuple[str, ...] = Field(default_factory=tuple)
    applied_route_decision_ids: tuple[str, ...] = Field(default_factory=tuple)
    written_storage_record_ids: tuple[str, ...] = Field(default_factory=tuple)
    provider_execution_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_prompt_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_workflow_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_memory_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_retrieval_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_output_ids: tuple[str, ...] = Field(default_factory=tuple)
    applied_hitl_decision_ids: tuple[str, ...] = Field(default_factory=tuple)
    cognitive_os_governance_safety_implemented: Literal[True] = True
    governance_boundary_metadata_implemented: Literal[True] = True
    hitl_boundary_metadata_implemented: Literal[True] = True
    explainability_boundary_metadata_implemented: Literal[True] = True
    no_automation_boundary_metadata_implemented: Literal[True] = True
    safety_boundary_metadata_implemented: Literal[True] = True
    all_capability_surfaces_traceable: Literal[True] = True
    all_roadmap_items_traceable: Literal[True] = True
    core_surface_foundation_used: Literal[True] = True
    secondary_surface_foundation_used: Literal[True] = True
    governance_policy_enforcement_implemented: Literal[False] = False
    safety_policy_enforcement_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    automation_activation_implemented: Literal[False] = False
    core_surface_activation_implemented: Literal[False] = False
    secondary_surface_activation_implemented: Literal[False] = False
    execution_application_implemented: Literal[False] = False
    routing_application_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    workflow_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    hitl_decision_application_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_boundaries_and_contract(self) -> Self:
        if self.source_surface_roles != COGNITIVE_OS_GOVERNANCE_SOURCE_ROLES:
            raise ValueError("source_surface_roles must match V6.6 surfaces")
        if (
            self.source_surface_serialization_versions
            != COGNITIVE_OS_GOVERNANCE_SOURCE_SERIALIZATION_VERSIONS
        ):
            raise ValueError("source_surface_serialization_versions must match")
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.foundation_systems != COGNITIVE_OS_FOUNDATION_SYSTEMS:
            raise ValueError("foundation_systems must match V5/V6 foundations")
        if self.foundation_system_count != len(self.foundation_systems):
            raise ValueError("foundation_system_count must match foundations")
        if self.governed_roadmap_items != COGNITIVE_OS_ROADMAP_ITEMS:
            raise ValueError("governed_roadmap_items must match V6.6 roadmap")
        if self.governed_roadmap_item_count != len(self.governed_roadmap_items):
            raise ValueError("governed_roadmap_item_count must match roadmap")
        if self.source_core_surface_ids != tuple(
            boundary.source_core_surface_id for boundary in self.governance_boundaries
        ):
            raise ValueError("source_core_surface_ids must match boundaries")
        if self.source_secondary_surface_ids != tuple(
            boundary.source_secondary_surface_id
            for boundary in self.governance_boundaries
        ):
            raise ValueError("source_secondary_surface_ids must match boundaries")
        count_fields = (
            (self.source_core_surface_count, self.source_core_surface_ids),
            (self.source_secondary_surface_count, self.source_secondary_surface_ids),
            (self.source_hitl_count, self.source_hitl_ids),
            (self.source_safety_count, self.source_safety_ids),
            (self.source_explanation_count, self.source_explanation_ids),
        )
        if any(count != len(ids) for count, ids in count_fields):
            raise ValueError("source counts must match source ids")
        if self.governance_boundary_ids != tuple(
            boundary.boundary_id for boundary in self.governance_boundaries
        ):
            raise ValueError("governance_boundary_ids must match boundaries")
        if len(set(self.governance_boundary_ids)) != len(self.governance_boundary_ids):
            raise ValueError("governance_boundary_ids must be unique")
        if self.governance_boundary_count != len(self.governance_boundaries):
            raise ValueError("governance_boundary_count must match boundaries")
        if self.guarded_boundary_ids != tuple(
            boundary.boundary_id
            for boundary in self.governance_boundaries
            if boundary.status == "guarded"
        ):
            raise ValueError("guarded_boundary_ids must match boundaries")
        if self.hitl_required_boundary_ids != tuple(
            boundary.boundary_id
            for boundary in self.governance_boundaries
            if boundary.hitl_required_before_governance_application
        ):
            raise ValueError("hitl_required_boundary_ids must match boundaries")
        if self.guarded_boundary_count != len(self.guarded_boundary_ids):
            raise ValueError("guarded_boundary_count must match ids")
        if self.hitl_required_boundary_count != len(self.hitl_required_boundary_ids):
            raise ValueError("hitl_required_boundary_count must match ids")
        if self.highest_governance_score != max(
            boundary.governance_score for boundary in self.governance_boundaries
        ):
            raise ValueError("highest_governance_score must match boundaries")
        expected_average = round(
            sum(boundary.governance_score for boundary in self.governance_boundaries)
            / len(self.governance_boundaries)
        )
        if self.overall_governance_score != expected_average:
            raise ValueError("overall_governance_score must match boundaries")
        if self.overall_governance_posture != cognitive_os_governance_posture(
            self.governance_boundaries
        ):
            raise ValueError("overall_governance_posture must match boundaries")
        declared_capabilities = set(self.capability_ids)
        declared_hitl = set(self.source_hitl_ids)
        declared_safety = set(self.source_safety_ids)
        declared_explanations = set(self.source_explanation_ids)
        declared_agents = set(self.linked_agent_ids)
        for boundary in self.governance_boundaries:
            if boundary.capability_id not in declared_capabilities:
                raise ValueError("boundary capability_id must be declared")
            if boundary.hitl_id not in declared_hitl:
                raise ValueError("boundary hitl_id must be declared")
            if boundary.safety_id not in declared_safety:
                raise ValueError("boundary safety_id must be declared")
            if boundary.explanation_id not in declared_explanations:
                raise ValueError("boundary explanation_id must be declared")
            if not set(boundary.linked_agent_ids).issubset(declared_agents):
                raise ValueError("boundary linked_agent_ids must be declared")
        if self.report_sections != COGNITIVE_OS_SECONDARY_REPORT_SECTIONS:
            raise ValueError("report_sections must match secondary surface contract")
        if self.covered_task_items != (COGNITIVE_OS_GOVERNANCE_TASK_ITEM,):
            raise ValueError("covered_task_items must be Task 28 only")
        if self.covered_task_item_count != len(self.covered_task_items):
            raise ValueError("covered_task_item_count must match tasks")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if (
            self.blocked_runtime_behaviors
            != COGNITIVE_OS_GOVERNANCE_BLOCKED_RUNTIME_BEHAVIORS
        ):
            raise ValueError("blocked_runtime_behaviors must match governance boundary")
        if any(
            (
                self.applied_governance_boundary_ids,
                self.enforced_safety_policy_ids,
                self.emitted_hitl_request_ids,
                self.requested_human_input_ids,
                self.activated_automation_ids,
                self.activated_core_surface_ids,
                self.activated_secondary_surface_ids,
                self.executed_node_ids,
                self.traversed_edge_ids,
                self.applied_route_decision_ids,
                self.written_storage_record_ids,
                self.provider_execution_ids,
                self.mutated_prompt_ids,
                self.mutated_workflow_ids,
                self.mutated_memory_ids,
                self.mutated_retrieval_ids,
                self.mutated_output_ids,
                self.applied_hitl_decision_ids,
            )
        ):
            raise ValueError("governance plan mutation ids must be empty")
        if not all(boundary.advisory_only for boundary in self.governance_boundaries):
            raise ValueError("all governance boundaries must be advisory")
        return self


def build_cognitive_os_governance_safety(
    cognitive_os_secondary_surface: CognitiveOSSecondarySurfacePlan | None = None,
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> CognitiveOSGovernanceSafetyPlan:
    """Build advisory Cognitive OS governance and safety boundary metadata."""

    secondary_surface = (
        cognitive_os_secondary_surface
        or build_cognitive_os_secondary_surface(
            route=route,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
        )
    )
    boundaries = _governance_boundaries(secondary_surface)
    boundary_ids = tuple(boundary.boundary_id for boundary in boundaries)
    guarded_ids = tuple(
        boundary.boundary_id for boundary in boundaries if boundary.status == "guarded"
    )
    hitl_required_ids = tuple(
        boundary.boundary_id
        for boundary in boundaries
        if boundary.hitl_required_before_governance_application
    )
    return CognitiveOSGovernanceSafetyPlan(
        route_name=secondary_surface.route_name,
        task_type=secondary_surface.task_type,
        execution_mode_ids=secondary_surface.execution_mode_ids,
        source_surface_roles=COGNITIVE_OS_GOVERNANCE_SOURCE_ROLES,
        source_surface_serialization_versions=(
            COGNITIVE_OS_GOVERNANCE_SOURCE_SERIALIZATION_VERSIONS
        ),
        source_core_surface_ids=secondary_surface.source_core_surface_ids,
        source_core_surface_count=secondary_surface.source_core_surface_count,
        source_secondary_surface_ids=secondary_surface.secondary_surface_ids,
        source_secondary_surface_count=secondary_surface.secondary_surface_count,
        source_hitl_ids=secondary_surface.source_hitl_ids,
        source_hitl_count=secondary_surface.source_hitl_count,
        source_safety_ids=secondary_surface.source_safety_ids,
        source_safety_count=secondary_surface.source_safety_count,
        source_explanation_ids=secondary_surface.source_explanation_ids,
        source_explanation_count=secondary_surface.source_explanation_count,
        layer_order=secondary_surface.layer_order,
        capabilities=secondary_surface.capabilities,
        capability_ids=secondary_surface.capability_ids,
        capability_count=secondary_surface.capability_count,
        foundation_systems=secondary_surface.foundation_systems,
        foundation_system_count=secondary_surface.foundation_system_count,
        governed_roadmap_items=COGNITIVE_OS_ROADMAP_ITEMS,
        governed_roadmap_item_count=len(COGNITIVE_OS_ROADMAP_ITEMS),
        governance_boundaries=boundaries,
        governance_boundary_ids=boundary_ids,
        governance_boundary_count=len(boundaries),
        guarded_boundary_ids=guarded_ids,
        guarded_boundary_count=len(guarded_ids),
        hitl_required_boundary_ids=hitl_required_ids,
        hitl_required_boundary_count=len(hitl_required_ids),
        highest_governance_score=max(
            boundary.governance_score for boundary in boundaries
        ),
        overall_governance_score=round(
            sum(boundary.governance_score for boundary in boundaries) / len(boundaries)
        ),
        overall_governance_posture=cognitive_os_governance_posture(boundaries),
        linked_agent_ids=secondary_surface.linked_agent_ids,
        report_sections=secondary_surface.report_sections,
        covered_task_items=(COGNITIVE_OS_GOVERNANCE_TASK_ITEM,),
        covered_task_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_GOVERNANCE_BLOCKED_RUNTIME_BEHAVIORS,
    )


def cognitive_os_governance_boundary_by_id(
    boundary_id: str,
    plan: CognitiveOSGovernanceSafetyPlan | None = None,
) -> CognitiveOSGovernanceBoundary | None:
    """Return one governance boundary without applying it."""

    source_plan = plan or build_cognitive_os_governance_safety()
    for boundary in source_plan.governance_boundaries:
        if boundary.boundary_id == boundary_id:
            return boundary
    return None


def cognitive_os_governance_boundaries_for_layer(
    cognitive_layer: CognitiveOSLayer,
    plan: CognitiveOSGovernanceSafetyPlan | None = None,
) -> tuple[CognitiveOSGovernanceBoundary, ...]:
    """Return governance boundaries for one Cognitive OS layer."""

    source_plan = plan or build_cognitive_os_governance_safety()
    return tuple(
        boundary
        for boundary in source_plan.governance_boundaries
        if boundary.cognitive_layer == cognitive_layer
    )


def cognitive_os_governance_boundaries_for_agent(
    agent_id: str,
    plan: CognitiveOSGovernanceSafetyPlan | None = None,
) -> tuple[CognitiveOSGovernanceBoundary, ...]:
    """Return governance boundaries linked to one agent."""

    source_plan = plan or build_cognitive_os_governance_safety()
    return tuple(
        boundary
        for boundary in source_plan.governance_boundaries
        if agent_id in boundary.linked_agent_ids
    )


def cognitive_os_governance_boundaries_for_status(
    status: CognitiveOSGovernanceStatus,
    plan: CognitiveOSGovernanceSafetyPlan | None = None,
) -> tuple[CognitiveOSGovernanceBoundary, ...]:
    """Return governance boundaries by review status."""

    source_plan = plan or build_cognitive_os_governance_safety()
    return tuple(
        boundary
        for boundary in source_plan.governance_boundaries
        if boundary.status == status
    )


def cognitive_os_governance_boundaries_for_priority(
    priority: CognitiveOSGovernancePriority,
    plan: CognitiveOSGovernanceSafetyPlan | None = None,
) -> tuple[CognitiveOSGovernanceBoundary, ...]:
    """Return governance boundaries by review priority."""

    source_plan = plan or build_cognitive_os_governance_safety()
    return tuple(
        boundary
        for boundary in source_plan.governance_boundaries
        if boundary.priority == priority
    )


def cognitive_os_governance_score(
    *,
    source_item_count: int,
    governed_roadmap_item_count: int,
    hitl_requirement_count: int,
    explainability_signal_count: int,
    no_automation_weight: int,
    safety_weight: int,
) -> int:
    """Score governance coverage without enforcing it."""

    return min(
        1_000,
        source_item_count * 18
        + governed_roadmap_item_count * 8
        + hitl_requirement_count * 45
        + explainability_signal_count * 35
        + no_automation_weight
        + safety_weight,
    )


def cognitive_os_governance_status(
    governance_score: int,
) -> CognitiveOSGovernanceStatus:
    """Classify governance status from advisory score."""

    if governance_score >= 900:
        return "guarded"
    if governance_score >= 700:
        return "review_required"
    return "blocked"


def cognitive_os_governance_priority(
    governance_score: int,
    status: CognitiveOSGovernanceStatus,
) -> CognitiveOSGovernancePriority:
    """Classify governance review priority from advisory status."""

    if status == "guarded":
        return "guarded"
    if governance_score >= 800:
        return "critical"
    if governance_score >= 700:
        return "elevated"
    return "standard"


def cognitive_os_governance_posture(
    boundaries: tuple[CognitiveOSGovernanceBoundary, ...],
) -> CognitiveOSGovernancePosture:
    """Summarize advisory governance posture across boundaries."""

    if all(boundary.status == "guarded" for boundary in boundaries):
        return "guarded"
    if any(boundary.status == "blocked" for boundary in boundaries):
        return "blocked"
    return "review_required"


def _governance_boundaries(
    secondary_surface: CognitiveOSSecondarySurfacePlan,
) -> tuple[CognitiveOSGovernanceBoundary, ...]:
    return tuple(
        _governance_boundary(entry, secondary_surface)
        for entry in secondary_surface.secondary_surface_entries
    )


def _governance_boundary(
    entry: CognitiveOSSecondarySurfaceEntry,
    secondary_surface: CognitiveOSSecondarySurfacePlan,
) -> CognitiveOSGovernanceBoundary:
    source_item_ids = (
        entry.source_core_surface_id,
        entry.secondary_surface_id,
        entry.consolidation_unit_id,
        entry.execution_node_id,
        entry.hitl_id,
        entry.safety_id,
        entry.explanation_id,
        entry.route_decision_id,
        entry.plan_id,
        entry.schedule_id,
    )
    score = cognitive_os_governance_score(
        source_item_count=len(source_item_ids),
        governed_roadmap_item_count=len(COGNITIVE_OS_ROADMAP_ITEMS),
        hitl_requirement_count=5,
        explainability_signal_count=len(COGNITIVE_OS_SECONDARY_REPORT_SECTIONS),
        no_automation_weight=220,
        safety_weight=220,
    )
    status = cognitive_os_governance_status(score)
    return CognitiveOSGovernanceBoundary(
        boundary_id=f"cognitive_os_governance::{entry.capability_id}",
        status=status,
        priority=cognitive_os_governance_priority(score, status),
        route_name=secondary_surface.route_name,
        task_type=secondary_surface.task_type,
        execution_mode_ids=secondary_surface.execution_mode_ids,
        source_surface_roles=COGNITIVE_OS_GOVERNANCE_SOURCE_ROLES,
        source_serialization_versions=(
            COGNITIVE_OS_GOVERNANCE_SOURCE_SERIALIZATION_VERSIONS
        ),
        source_core_surface_id=entry.source_core_surface_id,
        source_secondary_surface_id=entry.secondary_surface_id,
        consolidation_unit_id=entry.consolidation_unit_id,
        execution_node_id=entry.execution_node_id,
        hitl_id=entry.hitl_id,
        safety_id=entry.safety_id,
        explanation_id=entry.explanation_id,
        route_decision_id=entry.route_decision_id,
        plan_id=entry.plan_id,
        schedule_id=entry.schedule_id,
        source_item_ids=source_item_ids,
        source_item_count=len(source_item_ids),
        capability_id=entry.capability_id,
        capability_name=entry.capability_name,
        cognitive_layer=entry.cognitive_layer,
        linked_agent_ids=entry.linked_agent_ids,
        surface_sequence_position=entry.surface_sequence_position,
        dependency_depth=entry.dependency_depth,
        source_core_surface_score=entry.source_core_surface_score,
        source_secondary_surface_score=entry.secondary_surface_score,
        governed_roadmap_items=COGNITIVE_OS_ROADMAP_ITEMS,
        governed_roadmap_item_count=len(COGNITIVE_OS_ROADMAP_ITEMS),
        foundation_systems=COGNITIVE_OS_FOUNDATION_SYSTEMS,
        report_sections=COGNITIVE_OS_SECONDARY_REPORT_SECTIONS,
        hitl_requirement_count=5,
        explainability_signal_count=len(COGNITIVE_OS_SECONDARY_REPORT_SECTIONS),
        no_automation_weight=220,
        safety_weight=220,
        governance_score=score,
        review_requirement=(
            f"{entry.capability_name} governance requires HITL review before "
            "any policy enforcement, runtime activation, or execution action."
        ),
        explainability_requirement=(
            f"{entry.capability_name} must preserve ownership, dependency, "
            "safety, HITL, and report-view explanations before any future "
            "action can be considered."
        ),
        ownership_boundary=(
            f"{entry.capability_name} remains owned by its Cognitive OS "
            "capability surface; governance metadata cannot assume upstream "
            "learning, memory, knowledge, research, or evolution ownership."
        ),
        dependency_boundary=(
            f"{entry.capability_name} governance cites core, secondary, "
            "execution, routing, planning, HITL, safety, and explanation "
            "sources without traversing or applying them."
        ),
        hitl_boundary=(
            f"{entry.capability_name} requires HITL before governance, safety, "
            "routing, execution, activation, or Runtime Evolution application."
        ),
        no_automation_boundary=(
            f"No automation is activated for {entry.capability_name}; "
            "automation, routing, execution, storage, provider calls, and "
            "surface activation remain metadata-only."
        ),
        safety_boundary=(
            f"{entry.capability_name} safety posture is advisory; safety "
            "policies are described but not enforced, and generated output "
            "or runtime state is not mutated."
        ),
        governance_tags=(
            "cognitive_os_governance_safety",
            "governance_boundary",
            "safety_boundary",
            "hitl_required",
            "explainability_boundary",
            "no_automation",
        ),
        advisory_actions=(
            "inspect_governance_boundary_metadata",
            "preserve_hitl_explainability_and_safety_boundaries",
            "keep_runtime_automation_disabled",
        ),
        evidence=(
            entry.source_core_surface_id,
            entry.secondary_surface_id,
            entry.hitl_id,
            entry.safety_id,
            entry.explanation_id,
            f"governed_roadmap_item_count:{len(COGNITIVE_OS_ROADMAP_ITEMS)}",
        ),
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=(COGNITIVE_OS_GOVERNANCE_BLOCKED_RUNTIME_BEHAVIORS),
    )
