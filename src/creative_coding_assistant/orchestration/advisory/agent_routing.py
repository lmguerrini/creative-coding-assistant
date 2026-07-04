"""Passive V4.2 dynamic agent routing metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_contracts import AGENT_CONTRACTS
from creative_coding_assistant.orchestration.agent_metadata import AGENT_METADATA
from creative_coding_assistant.orchestration.routing import RouteName

AgentRoutingStage = Literal["v4_2_dynamic_agent_routing"]
AgentRoutingAuthority = Literal["metadata_only_advisory"]
AgentRoutingPriorityBand = Literal[
    "foundational_context",
    "domain_context",
    "execution_context",
    "quality_review",
    "refinement_context",
    "final_synthesis",
]

AGENT_ROUTING_SERIALIZATION_VERSION = "agent_routing.v1"
AGENT_ROUTING_REGISTRY_SERIALIZATION_VERSION = "agent_routing_registry.v1"
AGENT_ROUTING_REGISTRY_AUTHORITY_BOUNDARY = (
    "Agent routing metadata describes future dynamic agent candidate routing, "
    "routing inputs, routing outputs, route applicability, and decision "
    "boundaries only; it does not execute agents, change workflow routing, "
    "route providers or models, select runtimes, trigger retries, mutate "
    "memory, call external systems, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "agent_instantiation",
    "agent_invocation",
    "active_dynamic_agent_routing",
    "workflow_routing_change",
    "provider_or_model_routing",
    "runtime_selection",
    "retry_or_refinement_triggering",
    "memory_storage_or_mutation",
    "external_system_calling",
    "generated_output_modification",
)

_ROUTING_INPUTS = (
    "assistant_request_metadata",
    "route_decision_metadata",
    "domain_selection_metadata",
    "agent_contract_registry",
    "agent_metadata_registry",
)

_ROUTING_OUTPUTS = (
    "agent_route_candidate_metadata",
    "agent_handoff_requirement_metadata",
    "routing_decision_boundary_metadata",
    "routing_guardrail_notes",
)

_DECISION_BOUNDARY = (
    "Routing profiles are inspectable candidate descriptions for future "
    "orchestration only. Consumers may read route candidates, required inputs, "
    "and produced metadata, but must not instantiate agents, reorder workflow "
    "nodes, route providers or models, select runtimes, trigger retries, or "
    "modify generated output."
)

_ROUTE_CANDIDATES_BY_AGENT_ID: dict[str, tuple[RouteName, ...]] = {
    "planner_agent": tuple(RouteName),
    "research_agent": (
        RouteName.GENERATE,
        RouteName.EXPLAIN,
        RouteName.DEBUG,
        RouteName.DESIGN,
        RouteName.REVIEW,
    ),
    "style_agent": (
        RouteName.GENERATE,
        RouteName.DESIGN,
        RouteName.PREVIEW,
    ),
    "runtime_agent": (
        RouteName.GENERATE,
        RouteName.DEBUG,
        RouteName.REVIEW,
        RouteName.PREVIEW,
    ),
    "artifact_agent": (
        RouteName.GENERATE,
        RouteName.DEBUG,
        RouteName.REVIEW,
        RouteName.PREVIEW,
    ),
    "art_direction_agent": (
        RouteName.GENERATE,
        RouteName.DESIGN,
        RouteName.PREVIEW,
    ),
    "aesthetic_critic_agent": (
        RouteName.GENERATE,
        RouteName.DESIGN,
        RouteName.REVIEW,
    ),
    "narrative_symbolic_agent": (
        RouteName.GENERATE,
        RouteName.DESIGN,
        RouteName.REVIEW,
    ),
    "creative_curator_agent": (
        RouteName.GENERATE,
        RouteName.DESIGN,
        RouteName.REVIEW,
    ),
    "critic_agent": (
        RouteName.GENERATE,
        RouteName.DEBUG,
        RouteName.REVIEW,
    ),
    "refiner_agent": (
        RouteName.GENERATE,
        RouteName.DEBUG,
        RouteName.REVIEW,
    ),
    "final_synthesizer_agent": tuple(RouteName),
}

_PRIORITY_BAND_BY_AGENT_ID: dict[str, AgentRoutingPriorityBand] = {
    "planner_agent": "foundational_context",
    "research_agent": "domain_context",
    "style_agent": "domain_context",
    "runtime_agent": "execution_context",
    "artifact_agent": "execution_context",
    "art_direction_agent": "domain_context",
    "aesthetic_critic_agent": "quality_review",
    "narrative_symbolic_agent": "domain_context",
    "creative_curator_agent": "quality_review",
    "critic_agent": "quality_review",
    "refiner_agent": "refinement_context",
    "final_synthesizer_agent": "final_synthesis",
}


class AgentRoutingProfile(BaseModel):
    """Metadata-only route applicability profile for one passive agent."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    role_id: str = Field(min_length=1, max_length=80)
    routing_stage: AgentRoutingStage = "v4_2_dynamic_agent_routing"
    routing_authority: AgentRoutingAuthority = "metadata_only_advisory"
    priority_band: AgentRoutingPriorityBand
    route_candidates: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    routing_inputs: tuple[str, ...] = Field(min_length=5, max_length=5)
    routing_outputs: tuple[str, ...] = Field(min_length=4, max_length=4)
    decision_signals: tuple[str, ...] = Field(min_length=1, max_length=18)
    required_metadata_inputs: tuple[str, ...] = Field(min_length=1, max_length=24)
    produced_metadata_outputs: tuple[str, ...] = Field(min_length=1, max_length=18)
    decision_boundary: str = Field(min_length=1, max_length=900)
    source_contract_registries: tuple[str, ...] = Field(min_length=1, max_length=16)
    source_agent_contract_version: str = Field(min_length=1, max_length=24)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    provider_model_routing_implemented: Literal[False] = False
    workflow_routing_implemented: Literal[False] = False
    agent_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_routing.v1"] = (
        AGENT_ROUTING_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentRoutingRegistry(BaseModel):
    """Stable registry for passive V4.2 dynamic agent routing metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_routing_registry"] = "agent_routing_registry"
    serialization_version: Literal["agent_routing_registry.v1"] = (
        AGENT_ROUTING_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_ROUTING_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    routing_profiles: tuple[AgentRoutingProfile, ...] = Field(
        min_length=12,
        max_length=12,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=12, le=12)
    routing_inputs: tuple[str, ...] = Field(min_length=5, max_length=5)
    routing_outputs: tuple[str, ...] = Field(min_length=4, max_length=4)
    source_registries: tuple[str, ...] = Field(min_length=3, max_length=3)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    provider_model_routing_implemented: Literal[False] = False
    workflow_routing_implemented: Literal[False] = False
    agent_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_agent_ids = tuple(profile.agent_id for profile in self.routing_profiles)
        if len(set(derived_agent_ids)) != len(derived_agent_ids):
            raise ValueError("agent_ids must be unique")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match routing_profiles")
        if self.profile_count != len(self.routing_profiles):
            raise ValueError("profile_count must match routing_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        known_routes = set(self.route_names)
        for profile in self.routing_profiles:
            if profile.routing_inputs != self.routing_inputs:
                raise ValueError("routing_inputs must match registry")
            if profile.routing_outputs != self.routing_outputs:
                raise ValueError("routing_outputs must match registry")
            if not set(profile.route_candidates).issubset(known_routes):
                raise ValueError("route_candidates must be known route names")
        return self


def agent_routing_registry() -> AgentRoutingRegistry:
    """Return passive V4.2 dynamic agent routing metadata."""

    return AGENT_ROUTING_REGISTRY


def agent_routing_profile_by_agent_id(
    agent_id: str,
    registry: AgentRoutingRegistry | None = None,
) -> AgentRoutingProfile | None:
    """Return one routing profile without selecting or invoking an agent."""

    source_registry = registry or AGENT_ROUTING_REGISTRY
    for profile in source_registry.routing_profiles:
        if profile.agent_id == agent_id:
            return profile
    return None


def agent_routing_profiles_for_route(
    route: RouteName | str,
    registry: AgentRoutingRegistry | None = None,
) -> tuple[AgentRoutingProfile, ...]:
    """Return passive profiles applicable to a route without workflow changes."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or AGENT_ROUTING_REGISTRY
    return tuple(
        profile
        for profile in source_registry.routing_profiles
        if route_name in profile.route_candidates
    )


def _profile(agent_id: str) -> AgentRoutingProfile:
    contract = next(
        contract for contract in AGENT_CONTRACTS if contract.agent_id == agent_id
    )
    metadata = next(item for item in AGENT_METADATA if item.agent_id == agent_id)
    return AgentRoutingProfile(
        agent_id=agent_id,
        role_id=contract.role_id,
        priority_band=_PRIORITY_BAND_BY_AGENT_ID[agent_id],
        route_candidates=_ROUTE_CANDIDATES_BY_AGENT_ID[agent_id],
        routing_inputs=_ROUTING_INPUTS,
        routing_outputs=_ROUTING_OUTPUTS,
        decision_signals=contract.produced_signals
        + (
            metadata.parallelization_support,
            metadata.future_orchestration_readiness,
        ),
        required_metadata_inputs=contract.required_inputs + contract.optional_inputs,
        produced_metadata_outputs=contract.produced_metadata,
        decision_boundary=_DECISION_BOUNDARY,
        source_contract_registries=tuple(
            dict.fromkeys(
                (
                    "agent_contract_registry",
                    "agent_role_registry",
                    "agent_metadata_registry",
                    *contract.source_contract_registries,
                )
            )
        ),
        source_agent_contract_version=contract.agent_version,
        provider_model_routing_implemented=False,
        workflow_routing_implemented=False,
        agent_execution_implemented=False,
        retry_triggering_implemented=False,
        generated_output_mutation_implemented=False,
    )


AGENT_ROUTING_PROFILES = tuple(
    _profile(contract.agent_id) for contract in AGENT_CONTRACTS
)
AGENT_ROUTING_REGISTRY = AgentRoutingRegistry(
    routing_profiles=AGENT_ROUTING_PROFILES,
    agent_ids=tuple(profile.agent_id for profile in AGENT_ROUTING_PROFILES),
    route_names=tuple(RouteName),
    profile_count=len(AGENT_ROUTING_PROFILES),
    routing_inputs=_ROUTING_INPUTS,
    routing_outputs=_ROUTING_OUTPUTS,
    source_registries=(
        "agent_contract_registry",
        "agent_role_registry",
        "agent_metadata_registry",
    ),
)
