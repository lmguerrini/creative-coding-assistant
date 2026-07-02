"""V6.6 Creative Identity Layer metadata."""

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
from creative_coding_assistant.orchestration.creative_cognition_layer import (
    CreativeCognitionLayerPlan,
    build_creative_cognition_layer,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

CREATIVE_IDENTITY_LAYER_SERIALIZATION_VERSION = "creative_identity_layer.v1"
CREATIVE_IDENTITY_LAYER_ROADMAP_ITEM = "Creative Identity Layer"
CREATIVE_IDENTITY_LAYER_AUTHORITY_BOUNDARY = (
    "V6.6 Creative Identity Layer projects creative cognition signals into "
    "read-only identity profiles for style continuity, intent continuity, "
    "constraint memory posture, agent role alignment, and governed identity "
    "readiness. It exposes creative identity metadata only; it does not "
    "persist identities, personalize runtime behavior, mutate prompts, "
    "memory, retrieval, storage, provider selection, generated output, "
    "runtime state, or apply Runtime Evolution."
)
CREATIVE_IDENTITY_DIMENSIONS = (
    "style continuity",
    "intent continuity",
    "constraint memory posture",
    "agent role alignment",
    "governed identity readiness",
)


class CreativeIdentityProfile(BaseModel):
    """One read-only creative identity profile."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    identity_id: str = Field(min_length=1, max_length=190)
    cognition_id: str = Field(min_length=1, max_length=190)
    governance_id: str = Field(min_length=1, max_length=190)
    planning_id: str = Field(min_length=1, max_length=170)
    reasoning_id: str = Field(min_length=1, max_length=170)
    profile_id: str = Field(min_length=1, max_length=170)
    state_id: str = Field(min_length=1, max_length=160)
    capability_id: str = Field(min_length=1, max_length=80)
    capability_name: CognitiveOSCapability
    cognitive_layer: CognitiveOSLayer
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    identity_dimensions: tuple[str, ...] = Field(min_length=5, max_length=5)
    cognition_posture: CognitiveOSPosture
    identity_posture: CognitiveOSPosture
    persistent_identity_storage_authorized: Literal[False] = False
    identity_summary: str = Field(min_length=1, max_length=540)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _identity_matches_sources_and_boundary(self) -> Self:
        expected_identity_id = f"creative_identity::{self.capability_id}"
        if self.identity_id != expected_identity_id:
            raise ValueError("identity_id must match capability_id")
        expected_cognition_id = f"creative_cognition::{self.capability_id}"
        if self.cognition_id != expected_cognition_id:
            raise ValueError("cognition_id must match capability_id")
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
        if self.identity_dimensions != CREATIVE_IDENTITY_DIMENSIONS:
            raise ValueError("identity_dimensions must match V6.6 identity")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class CreativeIdentityLayerPlan(BaseModel):
    """Read-only creative identity layer over creative cognition."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_identity_layer"] = "creative_identity_layer"
    serialization_version: Literal["creative_identity_layer.v1"] = (
        CREATIVE_IDENTITY_LAYER_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_IDENTITY_LAYER_AUTHORITY_BOUNDARY,
        max_length=2000,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    creative_cognition_layer_role: Literal["creative_cognition_layer"]
    creative_cognition_layer_serialization_version: Literal[
        "creative_cognition_layer.v1"
    ]
    cognitive_governance_layer_role: Literal["cognitive_governance_layer"]
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    source_cognition_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_cognition_count: int = Field(ge=6, le=6)
    source_governance_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_governance_count: int = Field(ge=6, le=6)
    source_planning_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_planning_count: int = Field(ge=6, le=6)
    source_reasoning_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_reasoning_count: int = Field(ge=6, le=6)
    source_profile_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_profile_count: int = Field(ge=6, le=6)
    source_state_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_state_count: int = Field(ge=6, le=6)
    identity_profiles: tuple[CreativeIdentityProfile, ...] = Field(
        min_length=6,
        max_length=6,
    )
    identity_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    identity_count: int = Field(ge=6, le=6)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    creative_identity_layer_implemented: Literal[True] = True
    creative_cognition_layer_integrated: Literal[True] = True
    identity_profile_contract_implemented: Literal[True] = True
    identity_dependency_traceability_implemented: Literal[True] = True
    identity_governance_contract_implemented: Literal[True] = True
    identity_explainability_contract_implemented: Literal[True] = True
    identity_persistence_implemented: Literal[False] = False
    identity_mutation_implemented: Literal[False] = False
    personalized_runtime_behavior_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    persisted_identity_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_identity_ids: tuple[str, ...] = Field(default_factory=tuple)
    personalized_identity_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _creative_identity_layer_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.source_cognition_count != len(self.source_cognition_ids):
            raise ValueError("source_cognition_count must match cognition ids")
        if self.source_governance_count != len(self.source_governance_ids):
            raise ValueError("source_governance_count must match governance ids")
        if self.source_planning_count != len(self.source_planning_ids):
            raise ValueError("source_planning_count must match planning ids")
        if self.source_reasoning_count != len(self.source_reasoning_ids):
            raise ValueError("source_reasoning_count must match reasoning ids")
        if self.source_profile_count != len(self.source_profile_ids):
            raise ValueError("source_profile_count must match profile ids")
        if self.source_state_count != len(self.source_state_ids):
            raise ValueError("source_state_count must match state ids")
        if self.identity_ids != tuple(
            profile.identity_id for profile in self.identity_profiles
        ):
            raise ValueError("identity_ids must match profiles")
        if self.identity_count != len(self.identity_profiles):
            raise ValueError("identity_count must match profiles")
        if len(set(self.identity_ids)) != len(self.identity_ids):
            raise ValueError("identity_ids must be unique")
        declared_capabilities = set(self.capability_ids)
        declared_cognition = set(self.source_cognition_ids)
        declared_governance = set(self.source_governance_ids)
        declared_planning = set(self.source_planning_ids)
        declared_reasoning = set(self.source_reasoning_ids)
        declared_profiles = set(self.source_profile_ids)
        declared_states = set(self.source_state_ids)
        declared_agents = set(self.linked_agent_ids)
        for profile in self.identity_profiles:
            if profile.capability_id not in declared_capabilities:
                raise ValueError("profile capability_id must be declared")
            if profile.cognition_id not in declared_cognition:
                raise ValueError("profile cognition_id must be declared")
            if profile.governance_id not in declared_governance:
                raise ValueError("profile governance_id must be declared")
            if profile.planning_id not in declared_planning:
                raise ValueError("profile planning_id must be declared")
            if profile.reasoning_id not in declared_reasoning:
                raise ValueError("profile reasoning_id must be declared")
            if profile.profile_id not in declared_profiles:
                raise ValueError("profile profile_id must be declared")
            if profile.state_id not in declared_states:
                raise ValueError("profile state_id must be declared")
            if not set(profile.linked_agent_ids).issubset(declared_agents):
                raise ValueError("profile linked_agent_ids must be declared")
        if self.covered_roadmap_items != (CREATIVE_IDENTITY_LAYER_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 15 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.persisted_identity_ids,
                self.mutated_identity_ids,
                self.personalized_identity_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "identity persistence, mutation, personalization, and HITL ids "
                "must be empty",
            )
        if not all(profile.advisory_only for profile in self.identity_profiles):
            raise ValueError("all creative identity profiles must be advisory only")
        return self


def build_creative_identity_layer(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    creative_cognition_layer: CreativeCognitionLayerPlan | None = None,
) -> CreativeIdentityLayerPlan:
    """Build read-only creative identity metadata."""

    cognition = creative_cognition_layer or build_creative_cognition_layer(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    identities = _creative_identity_profiles(cognition)
    return CreativeIdentityLayerPlan(
        route_name=cognition.route_name,
        task_type=cognition.task_type,
        execution_mode_ids=cognition.execution_mode_ids,
        creative_cognition_layer_role=cognition.role,
        creative_cognition_layer_serialization_version=(
            cognition.serialization_version
        ),
        cognitive_governance_layer_role=cognition.cognitive_governance_layer_role,
        layer_order=cognition.layer_order,
        capabilities=cognition.capabilities,
        capability_ids=cognition.capability_ids,
        capability_count=cognition.capability_count,
        source_cognition_ids=cognition.cognition_ids,
        source_cognition_count=cognition.cognition_count,
        source_governance_ids=cognition.source_governance_ids,
        source_governance_count=cognition.source_governance_count,
        source_planning_ids=cognition.source_planning_ids,
        source_planning_count=cognition.source_planning_count,
        source_reasoning_ids=cognition.source_reasoning_ids,
        source_reasoning_count=cognition.source_reasoning_count,
        source_profile_ids=cognition.source_profile_ids,
        source_profile_count=cognition.source_profile_count,
        source_state_ids=cognition.source_state_ids,
        source_state_count=cognition.source_state_count,
        identity_profiles=identities,
        identity_ids=tuple(profile.identity_id for profile in identities),
        identity_count=len(identities),
        linked_agent_ids=cognition.linked_agent_ids,
        covered_roadmap_items=(CREATIVE_IDENTITY_LAYER_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture=cognition.graph_posture,
    )


def creative_identity_profile_by_id(
    identity_id: str,
    layer: CreativeIdentityLayerPlan | None = None,
) -> CreativeIdentityProfile | None:
    """Return one creative identity profile without persisting it."""

    source_layer = layer or build_creative_identity_layer()
    for profile in source_layer.identity_profiles:
        if profile.identity_id == identity_id:
            return profile
    return None


def creative_identity_profiles_for_layer(
    cognitive_layer: CognitiveOSLayer,
    layer: CreativeIdentityLayerPlan | None = None,
) -> tuple[CreativeIdentityProfile, ...]:
    """Return creative identity profiles for one Cognitive OS layer."""

    source_layer = layer or build_creative_identity_layer()
    return tuple(
        profile
        for profile in source_layer.identity_profiles
        if profile.cognitive_layer == cognitive_layer
    )


def creative_identity_profiles_for_agent(
    agent_id: str,
    layer: CreativeIdentityLayerPlan | None = None,
) -> tuple[CreativeIdentityProfile, ...]:
    """Return creative identity profiles linked to one agent."""

    source_layer = layer or build_creative_identity_layer()
    return tuple(
        profile
        for profile in source_layer.identity_profiles
        if agent_id in profile.linked_agent_ids
    )


def _creative_identity_profiles(
    cognition_layer: CreativeCognitionLayerPlan,
) -> tuple[CreativeIdentityProfile, ...]:
    return tuple(
        CreativeIdentityProfile(
            identity_id=f"creative_identity::{signal.capability_id}",
            cognition_id=signal.cognition_id,
            governance_id=signal.governance_id,
            planning_id=signal.planning_id,
            reasoning_id=signal.reasoning_id,
            profile_id=signal.profile_id,
            state_id=signal.state_id,
            capability_id=signal.capability_id,
            capability_name=signal.capability_name,
            cognitive_layer=signal.cognitive_layer,
            linked_agent_ids=signal.linked_agent_ids,
            identity_dimensions=CREATIVE_IDENTITY_DIMENSIONS,
            cognition_posture=signal.cognition_posture,
            identity_posture=signal.cognition_posture,
            identity_summary=(
                f"Read-only creative identity profile for "
                f"{signal.capability_name}; preserves style, intent, "
                f"constraints, agent alignment, and governance from "
                f"{signal.cognition_id} without persistence authority."
            ),
            dependency_contracts=(
                "creative identity profile follows creative cognition signal",
                f"creative cognition signal:{signal.cognition_id}",
                f"cognitive governance policy:{signal.governance_id}",
            ),
            governance_contracts=(
                "creative identity does not persist identity profiles",
                "creative identity does not personalize runtime behavior",
                "HITL required before any identity-driven behavior",
            ),
            explanation_contracts=(
                "creative identity cites cognition and governance sources",
                "creative identity preserves capability and layer ownership",
                "creative identity explains why no personalization is applied",
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for signal in cognition_layer.creative_cognition_signals
    )
