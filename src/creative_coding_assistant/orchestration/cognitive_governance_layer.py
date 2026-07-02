"""V6.6 Cognitive Governance Layer metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
    CognitiveOSCapability,
    CognitiveOSLayer,
    CognitiveOSPosture,
)
from creative_coding_assistant.orchestration.meta_planning_layer import (
    MetaPlanningLayerPlan,
    build_meta_planning_layer,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

COGNITIVE_GOVERNANCE_LAYER_SERIALIZATION_VERSION = (
    "cognitive_governance_layer.v1"
)
COGNITIVE_GOVERNANCE_LAYER_ROADMAP_ITEM = "Cognitive Governance Layer"
COGNITIVE_GOVERNANCE_LAYER_AUTHORITY_BOUNDARY = (
    "V6.6 Cognitive Governance Layer projects meta-planning metadata into "
    "read-only governance policies for ownership, dependency traceability, "
    "explainability, HITL posture, and mutation boundaries. It exposes "
    "governance readiness only; it does not enforce policies, block or "
    "execute workflows, mutate plans, prompts, memory, retrieval, storage, "
    "provider selection, generated output, runtime state, or apply Runtime "
    "Evolution."
)
COGNITIVE_GOVERNANCE_CONTROLS = (
    "ownership preservation",
    "dependency traceability",
    "explainability preservation",
    "HITL governance readiness",
    "mutation boundary preservation",
)


class CognitiveGovernancePolicy(BaseModel):
    """One read-only governance policy projection."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    governance_id: str = Field(min_length=1, max_length=190)
    planning_id: str = Field(min_length=1, max_length=170)
    reasoning_id: str = Field(min_length=1, max_length=170)
    profile_id: str = Field(min_length=1, max_length=170)
    state_id: str = Field(min_length=1, max_length=160)
    capability_id: str = Field(min_length=1, max_length=80)
    capability_name: CognitiveOSCapability
    cognitive_layer: CognitiveOSLayer
    source_optimization_signal_id: str = Field(min_length=1, max_length=180)
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    governance_controls: tuple[str, ...] = Field(min_length=5, max_length=5)
    planning_posture: CognitiveOSPosture
    governance_posture: CognitiveOSPosture
    hitl_required_before_application: Literal[True] = True
    governance_summary: str = Field(min_length=1, max_length=520)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _policy_matches_sources_and_boundary(self) -> Self:
        expected_governance_id = f"cognitive_governance::{self.capability_id}"
        if self.governance_id != expected_governance_id:
            raise ValueError("governance_id must match capability_id")
        expected_planning_id = f"meta_planning::{self.capability_id}"
        if self.planning_id != expected_planning_id:
            raise ValueError("planning_id must match capability_id")
        expected_reasoning_id = f"meta_reasoning::{self.capability_id}"
        if self.reasoning_id != expected_reasoning_id:
            raise ValueError("reasoning_id must match capability_id")
        expected_profile_id = f"cognitive_profile::{self.capability_id}"
        if self.profile_id != expected_profile_id:
            raise ValueError("profile_id must match capability_id")
        expected_state_id = f"cognitive_state::{self.capability_id}"
        if self.state_id != expected_state_id:
            raise ValueError("state_id must match capability_id")
        expected_optimization_id = f"cross_system_optimization::{self.capability_id}"
        if self.source_optimization_signal_id != expected_optimization_id:
            raise ValueError("source_optimization_signal_id must match capability")
        expected_learning_id = f"cross_system_learning::{self.capability_id}"
        if self.source_learning_signal_id != expected_learning_id:
            raise ValueError("source_learning_signal_id must match capability")
        if self.governance_controls != COGNITIVE_GOVERNANCE_CONTROLS:
            raise ValueError("governance_controls must match V6.6 governance")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class CognitiveGovernanceLayerPlan(BaseModel):
    """Read-only Cognitive OS governance layer over meta-planning."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cognitive_governance_layer"] = "cognitive_governance_layer"
    serialization_version: Literal["cognitive_governance_layer.v1"] = (
        COGNITIVE_GOVERNANCE_LAYER_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COGNITIVE_GOVERNANCE_LAYER_AUTHORITY_BOUNDARY,
        max_length=2000,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    meta_planning_layer_role: Literal["meta_planning_layer"]
    meta_planning_layer_serialization_version: Literal["meta_planning_layer.v1"]
    meta_reasoning_layer_role: Literal["meta_reasoning_layer"]
    profile_engine_role: Literal["cognitive_profile_engine"]
    state_engine_role: Literal["cognitive_state_engine"]
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    source_planning_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_planning_count: int = Field(ge=6, le=6)
    source_reasoning_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_reasoning_count: int = Field(ge=6, le=6)
    source_profile_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_profile_count: int = Field(ge=6, le=6)
    source_state_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_state_count: int = Field(ge=6, le=6)
    source_optimization_signal_ids: tuple[str, ...] = Field(
        min_length=6,
        max_length=6,
    )
    source_optimization_signal_count: int = Field(ge=6, le=6)
    source_learning_signal_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_learning_signal_count: int = Field(ge=6, le=6)
    governance_policies: tuple[CognitiveGovernancePolicy, ...] = Field(
        min_length=6,
        max_length=6,
    )
    governance_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    governance_count: int = Field(ge=6, le=6)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    cognitive_governance_layer_implemented: Literal[True] = True
    meta_planning_layer_integrated: Literal[True] = True
    governance_policy_contract_implemented: Literal[True] = True
    governance_dependency_traceability_implemented: Literal[True] = True
    governance_explainability_contract_implemented: Literal[True] = True
    governance_hitl_contract_implemented: Literal[True] = True
    policy_enforcement_implemented: Literal[False] = False
    workflow_blocking_implemented: Literal[False] = False
    autonomous_workflow_planning_implemented: Literal[False] = False
    plan_mutation_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    enforced_governance_ids: tuple[str, ...] = Field(default_factory=tuple)
    blocked_workflow_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_governance_policy_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _governance_layer_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.source_planning_count != len(self.source_planning_ids):
            raise ValueError("source_planning_count must match planning ids")
        if self.source_reasoning_count != len(self.source_reasoning_ids):
            raise ValueError("source_reasoning_count must match reasoning ids")
        if self.source_profile_count != len(self.source_profile_ids):
            raise ValueError("source_profile_count must match profile ids")
        if self.source_state_count != len(self.source_state_ids):
            raise ValueError("source_state_count must match state ids")
        if self.source_optimization_signal_count != len(
            self.source_optimization_signal_ids
        ):
            raise ValueError("source_optimization_signal_count must match signals")
        if self.source_learning_signal_count != len(self.source_learning_signal_ids):
            raise ValueError("source_learning_signal_count must match signals")
        if self.governance_ids != tuple(
            policy.governance_id for policy in self.governance_policies
        ):
            raise ValueError("governance_ids must match policies")
        if self.governance_count != len(self.governance_policies):
            raise ValueError("governance_count must match policies")
        if len(set(self.governance_ids)) != len(self.governance_ids):
            raise ValueError("governance_ids must be unique")
        declared_capabilities = set(self.capability_ids)
        declared_planning = set(self.source_planning_ids)
        declared_reasoning = set(self.source_reasoning_ids)
        declared_profiles = set(self.source_profile_ids)
        declared_states = set(self.source_state_ids)
        declared_optimization_signals = set(self.source_optimization_signal_ids)
        declared_learning_signals = set(self.source_learning_signal_ids)
        declared_agents = set(self.linked_agent_ids)
        for policy in self.governance_policies:
            if policy.capability_id not in declared_capabilities:
                raise ValueError("policy capability_id must be declared")
            if policy.planning_id not in declared_planning:
                raise ValueError("policy planning_id must be declared")
            if policy.reasoning_id not in declared_reasoning:
                raise ValueError("policy reasoning_id must be declared")
            if policy.profile_id not in declared_profiles:
                raise ValueError("policy profile_id must be declared")
            if policy.state_id not in declared_states:
                raise ValueError("policy state_id must be declared")
            if (
                policy.source_optimization_signal_id
                not in declared_optimization_signals
            ):
                raise ValueError("policy optimization signal must be declared")
            if policy.source_learning_signal_id not in declared_learning_signals:
                raise ValueError("policy learning signal must be declared")
            if not set(policy.linked_agent_ids).issubset(declared_agents):
                raise ValueError("policy linked_agent_ids must be declared")
        if self.covered_roadmap_items != (
            COGNITIVE_GOVERNANCE_LAYER_ROADMAP_ITEM,
        ):
            raise ValueError("covered_roadmap_items must be Task 13 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.enforced_governance_ids,
                self.blocked_workflow_ids,
                self.mutated_governance_policy_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "governance enforcement, workflow blocking, mutation, and "
                "HITL ids must be empty",
            )
        if not all(policy.advisory_only for policy in self.governance_policies):
            raise ValueError("all cognitive governance policies must be advisory only")
        return self


def build_cognitive_governance_layer(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    meta_planning_layer: MetaPlanningLayerPlan | None = None,
) -> CognitiveGovernanceLayerPlan:
    """Build read-only cognitive governance metadata."""

    planning = meta_planning_layer or build_meta_planning_layer(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    policies = _governance_policies(planning)
    return CognitiveGovernanceLayerPlan(
        route_name=planning.route_name,
        task_type=planning.task_type,
        execution_mode_ids=planning.execution_mode_ids,
        meta_planning_layer_role=planning.role,
        meta_planning_layer_serialization_version=planning.serialization_version,
        meta_reasoning_layer_role=planning.meta_reasoning_layer_role,
        profile_engine_role=planning.profile_engine_role,
        state_engine_role=planning.state_engine_role,
        layer_order=planning.layer_order,
        capabilities=planning.capabilities,
        capability_ids=planning.capability_ids,
        capability_count=planning.capability_count,
        source_planning_ids=planning.planning_ids,
        source_planning_count=planning.planning_count,
        source_reasoning_ids=planning.source_reasoning_ids,
        source_reasoning_count=planning.source_reasoning_count,
        source_profile_ids=planning.source_profile_ids,
        source_profile_count=planning.source_profile_count,
        source_state_ids=planning.source_state_ids,
        source_state_count=planning.source_state_count,
        source_optimization_signal_ids=planning.source_optimization_signal_ids,
        source_optimization_signal_count=planning.source_optimization_signal_count,
        source_learning_signal_ids=planning.source_learning_signal_ids,
        source_learning_signal_count=planning.source_learning_signal_count,
        governance_policies=policies,
        governance_ids=tuple(policy.governance_id for policy in policies),
        governance_count=len(policies),
        linked_agent_ids=planning.linked_agent_ids,
        covered_roadmap_items=(COGNITIVE_GOVERNANCE_LAYER_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture=planning.graph_posture,
    )


def cognitive_governance_policy_by_id(
    governance_id: str,
    layer: CognitiveGovernanceLayerPlan | None = None,
) -> CognitiveGovernancePolicy | None:
    """Return one cognitive governance policy without enforcing it."""

    source_layer = layer or build_cognitive_governance_layer()
    for policy in source_layer.governance_policies:
        if policy.governance_id == governance_id:
            return policy
    return None


def cognitive_governance_policies_for_layer(
    cognitive_layer: CognitiveOSLayer,
    layer: CognitiveGovernanceLayerPlan | None = None,
) -> tuple[CognitiveGovernancePolicy, ...]:
    """Return cognitive governance policies for one Cognitive OS layer."""

    source_layer = layer or build_cognitive_governance_layer()
    return tuple(
        policy
        for policy in source_layer.governance_policies
        if policy.cognitive_layer == cognitive_layer
    )


def cognitive_governance_policies_for_agent(
    agent_id: str,
    layer: CognitiveGovernanceLayerPlan | None = None,
) -> tuple[CognitiveGovernancePolicy, ...]:
    """Return cognitive governance policies linked to one agent."""

    source_layer = layer or build_cognitive_governance_layer()
    return tuple(
        policy
        for policy in source_layer.governance_policies
        if agent_id in policy.linked_agent_ids
    )


def _governance_policies(
    planning_layer: MetaPlanningLayerPlan,
) -> tuple[CognitiveGovernancePolicy, ...]:
    return tuple(
        CognitiveGovernancePolicy(
            governance_id=f"cognitive_governance::{projection.capability_id}",
            planning_id=projection.planning_id,
            reasoning_id=projection.reasoning_id,
            profile_id=projection.profile_id,
            state_id=projection.state_id,
            capability_id=projection.capability_id,
            capability_name=projection.capability_name,
            cognitive_layer=projection.cognitive_layer,
            source_optimization_signal_id=(
                projection.source_optimization_signal_id
            ),
            source_learning_signal_id=projection.source_learning_signal_id,
            linked_agent_ids=projection.linked_agent_ids,
            governance_controls=COGNITIVE_GOVERNANCE_CONTROLS,
            planning_posture=projection.planning_posture,
            governance_posture=projection.planning_posture,
            governance_summary=(
                f"Read-only governance policy for "
                f"{projection.capability_name}; preserves ownership, "
                f"dependencies, explainability, and HITL posture from "
                f"{projection.planning_id} without enforcement authority."
            ),
            dependency_contracts=(
                "governance policy follows meta-planning projection",
                f"meta-planning projection:{projection.planning_id}",
                f"meta-reasoning assessment:{projection.reasoning_id}",
            ),
            governance_contracts=(
                "governance policy does not enforce runtime behavior",
                "governance policy does not block workflows or mutate state",
                "HITL required before any governance-driven application",
            ),
            explanation_contracts=(
                "governance policy cites planning, reasoning, profile, and state",
                "governance policy preserves capability and layer ownership",
                "governance policy explains why no enforcement is applied",
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for projection in planning_layer.planning_projections
    )
