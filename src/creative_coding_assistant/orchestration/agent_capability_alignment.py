"""Passive V4.2 agent capability alignment metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_roles import AGENT_ROLES

V42CapabilityId = Literal[
    "dynamic_agent_routing",
    "blackboard_memory",
    "shared_context_view",
    "dependency_graph",
    "parallel_scheduling",
    "agent_coordination",
    "agent_debate",
    "consensus_builder",
    "agent_escalation_signals",
    "agent_lifecycle",
    "agent_state_synchronization",
    "workflow_agent_handoff",
    "orchestration_contract_integration",
]

CAPABILITY_ALIGNMENT_PROFILE_SERIALIZATION_VERSION = "agent_capability_alignment.v1"
CAPABILITY_ALIGNMENT_REGISTRY_SERIALIZATION_VERSION = (
    "agent_capability_alignment_registry.v1"
)
CAPABILITY_ALIGNMENT_REGISTRY_AUTHORITY_BOUNDARY = (
    "Agent capability alignment metadata maps V4.1 roles to V4.2 "
    "orchestration capability registries only; it does not activate "
    "capabilities, route runtime work, change prompts, invoke agents, execute "
    "coordination, run debates, execute voting, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "capability_activation",
    "runtime_work_routing",
    "prompt_change",
    "agent_invocation",
    "coordination_execution",
    "debate_execution",
    "voting_execution",
    "generated_output_modification",
)

_BASE_CAPABILITIES: tuple[V42CapabilityId, ...] = (
    "dynamic_agent_routing",
    "blackboard_memory",
    "shared_context_view",
    "dependency_graph",
    "parallel_scheduling",
    "agent_coordination",
)

_LATER_BASE_CAPABILITIES: tuple[V42CapabilityId, ...] = (
    "agent_escalation_signals",
    "agent_lifecycle",
    "agent_state_synchronization",
    "workflow_agent_handoff",
    "orchestration_contract_integration",
)

_SOURCE_ORCHESTRATION_REGISTRIES = (
    "agent_routing_registry",
    "blackboard_memory_registry",
    "shared_context_view_registry",
    "agent_dependency_graph_registry",
    "parallel_scheduling_registry",
    "agent_coordination_registry",
    "agent_debate_registry",
    "consensus_builder_registry",
    "agent_escalation_signal_registry",
    "agent_lifecycle_registry",
    "agent_state_synchronization_registry",
    "workflow_agent_handoff_registry",
    "orchestration_contract_integration_registry",
)

_DEBATE_AGENT_IDS = {
    "planner_agent",
    "runtime_agent",
    "artifact_agent",
    "art_direction_agent",
    "style_agent",
    "aesthetic_critic_agent",
    "creative_curator_agent",
    "critic_agent",
    "refiner_agent",
    "final_synthesizer_agent",
}

_CONSENSUS_AGENT_IDS = {
    "planner_agent",
    "critic_agent",
    "refiner_agent",
    "final_synthesizer_agent",
}


class AgentCapabilityAlignmentProfile(BaseModel):
    """Metadata-only mapping from one V4.1 role to V4.2 capabilities."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    role_id: str = Field(min_length=1, max_length=80)
    role_name: str = Field(min_length=1, max_length=140)
    capability_ids: tuple[V42CapabilityId, ...] = Field(min_length=1, max_length=13)
    source_role_registry: Literal["agent_role_registry"] = "agent_role_registry"
    source_orchestration_registries: tuple[str, ...] = Field(
        min_length=13,
        max_length=13,
    )
    alignment_boundary: str = Field(min_length=1, max_length=700)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    capabilities_activated: Literal[False] = False
    runtime_work_routing_implemented: Literal[False] = False
    prompt_changes_implemented: Literal[False] = False
    serialization_version: Literal["agent_capability_alignment.v1"] = (
        CAPABILITY_ALIGNMENT_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentCapabilityAlignmentRegistry(BaseModel):
    """Stable passive V4.2 capability alignment registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_capability_alignment_registry"] = (
        "agent_capability_alignment_registry"
    )
    serialization_version: Literal["agent_capability_alignment_registry.v1"] = (
        CAPABILITY_ALIGNMENT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CAPABILITY_ALIGNMENT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    alignments: tuple[AgentCapabilityAlignmentProfile, ...] = Field(
        min_length=12,
        max_length=12,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    role_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    capability_ids: tuple[V42CapabilityId, ...] = Field(min_length=13, max_length=13)
    alignment_count: int = Field(ge=12, le=12)
    source_registries: tuple[str, ...] = Field(min_length=15, max_length=15)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    capabilities_activated: Literal[False] = False
    runtime_work_routing_implemented: Literal[False] = False
    prompt_changes_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_alignments(self) -> Self:
        derived_agent_ids = tuple(alignment.agent_id for alignment in self.alignments)
        derived_role_ids = tuple(alignment.role_id for alignment in self.alignments)
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match alignments")
        if self.role_ids != derived_role_ids:
            raise ValueError("role_ids must match alignments")
        if self.alignment_count != len(self.alignments):
            raise ValueError("alignment_count must match alignments")
        base_capabilities = set(_BASE_CAPABILITIES + _LATER_BASE_CAPABILITIES)
        for alignment in self.alignments:
            if not base_capabilities.issubset(set(alignment.capability_ids)):
                raise ValueError(
                    "capability_ids must include base orchestration capabilities"
                )
            if alignment.source_orchestration_registries != (
                _SOURCE_ORCHESTRATION_REGISTRIES
            ):
                raise ValueError(
                    "source_orchestration_registries must match V4.2 coverage"
                )
        covered_capabilities = tuple(
            dict.fromkeys(
                capability
                for alignment in self.alignments
                for capability in alignment.capability_ids
            )
        )
        if self.capability_ids != covered_capabilities:
            raise ValueError("capability_ids must match alignment coverage")
        if not set(_SOURCE_ORCHESTRATION_REGISTRIES).issubset(
            set(self.source_registries)
        ):
            raise ValueError(
                "source_registries must include V4.2 orchestration sources"
            )
        return self


def agent_capability_alignment_registry() -> AgentCapabilityAlignmentRegistry:
    """Return passive V4.2 role-to-capability alignment metadata."""

    return AGENT_CAPABILITY_ALIGNMENT_REGISTRY


def agent_capability_alignment_by_agent_id(
    agent_id: str,
    registry: AgentCapabilityAlignmentRegistry | None = None,
) -> AgentCapabilityAlignmentProfile | None:
    """Return one alignment profile without activating capabilities."""

    source_registry = registry or AGENT_CAPABILITY_ALIGNMENT_REGISTRY
    for alignment in source_registry.alignments:
        if alignment.agent_id == agent_id:
            return alignment
    return None


def _capabilities_for_agent(agent_id: str) -> tuple[V42CapabilityId, ...]:
    capabilities: list[V42CapabilityId] = list(_BASE_CAPABILITIES)
    if agent_id in _DEBATE_AGENT_IDS:
        capabilities.append("agent_debate")
    if agent_id in _CONSENSUS_AGENT_IDS:
        capabilities.append("consensus_builder")
    capabilities.extend(_LATER_BASE_CAPABILITIES)
    return tuple(capabilities)


def _alignment(agent_id: str) -> AgentCapabilityAlignmentProfile:
    role = next(role for role in AGENT_ROLES if role.agent_id == agent_id)
    return AgentCapabilityAlignmentProfile(
        agent_id=agent_id,
        role_id=role.role_id,
        role_name=role.role_name,
        capability_ids=_capabilities_for_agent(agent_id),
        source_orchestration_registries=_SOURCE_ORCHESTRATION_REGISTRIES,
        alignment_boundary=(
            "Capability alignment is export-only metadata; it does not "
            "activate capabilities, route runtime work, change prompts, "
            "execute orchestration, or modify generated output."
        ),
    )


AGENT_CAPABILITY_ALIGNMENTS = tuple(_alignment(role.agent_id) for role in AGENT_ROLES)
AGENT_CAPABILITY_ALIGNMENT_REGISTRY = AgentCapabilityAlignmentRegistry(
    alignments=AGENT_CAPABILITY_ALIGNMENTS,
    agent_ids=tuple(alignment.agent_id for alignment in AGENT_CAPABILITY_ALIGNMENTS),
    role_ids=tuple(alignment.role_id for alignment in AGENT_CAPABILITY_ALIGNMENTS),
    capability_ids=tuple(
        dict.fromkeys(
            capability
            for alignment in AGENT_CAPABILITY_ALIGNMENTS
            for capability in alignment.capability_ids
        )
    ),
    alignment_count=len(AGENT_CAPABILITY_ALIGNMENTS),
    source_registries=(
        "agent_role_registry",
        "agent_capability_registry",
        *_SOURCE_ORCHESTRATION_REGISTRIES,
    ),
)
