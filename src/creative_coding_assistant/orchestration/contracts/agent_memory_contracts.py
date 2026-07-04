"""Passive V4.1 agent memory access contracts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_identities import AGENT_IDENTITIES

AgentMemorySurface = Literal[
    "session",
    "artifact",
    "evaluation",
    "provenance",
    "future_blackboard",
]
AgentMemoryReadAccess = Literal["none", "metadata_only"]
AgentMemoryWriteAccess = Literal["none", "future_metadata_only"]
AgentMemoryReferenceAccess = Literal["none", "metadata_only"]

AGENT_MEMORY_SURFACE_CONTRACT_SERIALIZATION_VERSION = "agent_memory_surface_contract.v1"
AGENT_MEMORY_CONTRACT_SERIALIZATION_VERSION = "agent_memory_contract.v1"
AGENT_MEMORY_CONTRACT_REGISTRY_SERIALIZATION_VERSION = (
    "agent_memory_contract_registry.v1"
)
AGENT_MEMORY_CONTRACT_REGISTRY_AUTHORITY_BOUNDARY = (
    "Agent memory contracts describe what each V4.1 agent may read, write, "
    "and reference as future metadata boundaries across session, artifact, "
    "evaluation, provenance, and future blackboard surfaces only; they do not "
    "implement persistence, blackboard memory, shared context views, retrieval "
    "side effects, storage writes, workflow control, provider calls, agent "
    "execution, or generated output changes."
)

AGENT_MEMORY_SURFACE_ORDER: tuple[AgentMemorySurface, ...] = (
    "session",
    "artifact",
    "evaluation",
    "provenance",
    "future_blackboard",
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "memory_persistence",
    "blackboard_memory",
    "shared_context_view_materialization",
    "retrieval_side_effects",
    "storage_writes",
    "agent_invocation",
    "dynamic_agent_routing",
    "workflow_control",
    "provider_or_model_routing",
    "runtime_selection",
    "artifact_execution",
    "generated_output_modification",
)

_SURFACE_READ_METADATA = {
    "session": (
        "assistant_request_metadata",
        "conversation_summary_metadata",
        "workflow_status_metadata",
    ),
    "artifact": (
        "artifact_plan_metadata",
        "artifact_contract_metadata",
        "artifact_risk_metadata",
    ),
    "evaluation": (
        "evaluation_report_metadata",
        "quality_signal_metadata",
        "confidence_signal_metadata",
    ),
    "provenance": (
        "source_trace_metadata",
        "workflow_event_metadata",
        "metadata_origin_metadata",
    ),
    "future_blackboard": (),
}

_SURFACE_REFERENCE_METADATA = {
    "session": ("session_context_reference",),
    "artifact": ("artifact_context_reference",),
    "evaluation": ("evaluation_context_reference",),
    "provenance": ("provenance_context_reference",),
    "future_blackboard": (
        "future_blackboard_contract_reference",
        "future_shared_context_view_reference",
    ),
}

_AGENT_READ_SURFACES: dict[str, tuple[AgentMemorySurface, ...]] = {
    "planner_agent": ("session", "artifact", "evaluation", "provenance"),
    "research_agent": ("session", "provenance"),
    "style_agent": ("session", "artifact", "provenance"),
    "runtime_agent": ("session", "artifact", "provenance"),
    "artifact_agent": ("session", "artifact", "provenance"),
    "art_direction_agent": ("session", "artifact", "provenance"),
    "aesthetic_critic_agent": (
        "session",
        "artifact",
        "evaluation",
        "provenance",
    ),
    "narrative_symbolic_agent": ("session", "artifact", "provenance"),
    "creative_curator_agent": (
        "session",
        "artifact",
        "evaluation",
        "provenance",
    ),
    "critic_agent": ("session", "artifact", "evaluation", "provenance"),
    "refiner_agent": ("session", "artifact", "evaluation", "provenance"),
    "final_synthesizer_agent": (
        "session",
        "artifact",
        "evaluation",
        "provenance",
    ),
}

_AGENT_FUTURE_BLACKBOARD_WRITES = {
    "planner_agent": ("planning_context_packet", "planning_gap_summary"),
    "research_agent": ("research_context_notes", "source_gap_summary"),
    "style_agent": ("style_context_notes", "visual_style_constraints"),
    "runtime_agent": ("runtime_fit_notes", "compatibility_caveats"),
    "artifact_agent": ("artifact_readiness_notes", "artifact_risk_summary"),
    "art_direction_agent": ("art_direction_notes", "composition_intent_notes"),
    "aesthetic_critic_agent": (
        "aesthetic_review_notes",
        "visual_quality_signals",
    ),
    "narrative_symbolic_agent": (
        "narrative_context_notes",
        "symbolic_mapping_notes",
    ),
    "creative_curator_agent": (
        "curation_context_notes",
        "selection_rationale_notes",
    ),
    "critic_agent": ("critique_context_notes", "quality_review_signals"),
    "refiner_agent": ("refinement_context_notes", "revision_candidate_notes"),
    "final_synthesizer_agent": (
        "synthesis_context_notes",
        "final_handoff_summary",
    ),
}


class AgentMemorySurfaceContract(BaseModel):
    """Metadata-only access boundary for one agent memory surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    surface: AgentMemorySurface
    read_access: AgentMemoryReadAccess
    write_access: AgentMemoryWriteAccess
    reference_access: AgentMemoryReferenceAccess
    readable_metadata: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    writable_metadata: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    referenceable_metadata: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    authority_boundary: str = Field(min_length=1, max_length=900)
    storage_implemented: Literal[False] = False
    retrieval_side_effects: Literal[False] = False
    blackboard_implemented: Literal[False] = False
    serialization_version: Literal["agent_memory_surface_contract.v1"] = (
        AGENT_MEMORY_SURFACE_CONTRACT_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentMemoryContract(BaseModel):
    """Passive memory access contract for one future V4.1 agent."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    memory_contract_id: str = Field(min_length=1, max_length=120)
    surfaces: tuple[AgentMemorySurfaceContract, ...] = Field(
        min_length=5,
        max_length=5,
    )
    readable_memory_surfaces: tuple[str, ...] = Field(max_length=5)
    writable_memory_surfaces: tuple[str, ...] = Field(max_length=5)
    referenceable_memory_surfaces: tuple[str, ...] = Field(max_length=5)
    future_blackboard_hooks: tuple[str, ...] = Field(min_length=1, max_length=8)
    future_shared_context_hooks: tuple[str, ...] = Field(min_length=1, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    persistence_implemented: Literal[False] = False
    blackboard_implemented: Literal[False] = False
    retrieval_side_effects: Literal[False] = False
    serialization_version: Literal["agent_memory_contract.v1"] = (
        AGENT_MEMORY_CONTRACT_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _surface_metadata_matches_access(self) -> Self:
        surface_order = tuple(surface.surface for surface in self.surfaces)
        if surface_order != AGENT_MEMORY_SURFACE_ORDER:
            raise ValueError("surfaces must match memory surface order")

        readable = tuple(
            surface.surface
            for surface in self.surfaces
            if surface.read_access != "none"
        )
        writable = tuple(
            surface.surface
            for surface in self.surfaces
            if surface.write_access != "none"
        )
        referenceable = tuple(
            surface.surface
            for surface in self.surfaces
            if surface.reference_access != "none"
        )
        if self.readable_memory_surfaces != readable:
            raise ValueError("readable_memory_surfaces must match surfaces")
        if self.writable_memory_surfaces != writable:
            raise ValueError("writable_memory_surfaces must match surfaces")
        if self.referenceable_memory_surfaces != referenceable:
            raise ValueError("referenceable_memory_surfaces must match surfaces")
        return self


class AgentMemoryContractRegistry(BaseModel):
    """Stable registry of passive V4.1 agent memory contracts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_memory_contract_registry"] = "agent_memory_contract_registry"
    serialization_version: Literal["agent_memory_contract_registry.v1"] = (
        AGENT_MEMORY_CONTRACT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_MEMORY_CONTRACT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    contracts: tuple[AgentMemoryContract, ...] = Field(
        min_length=12,
        max_length=12,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    contract_count: int = Field(ge=12, le=12)
    memory_surfaces: tuple[AgentMemorySurface, ...] = Field(
        default=AGENT_MEMORY_SURFACE_ORDER,
        min_length=5,
        max_length=5,
    )
    source_identity_registry: Literal["agent_identity_registry"] = (
        "agent_identity_registry"
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    persistence_implemented: Literal[False] = False
    blackboard_implemented: Literal[False] = False
    retrieval_side_effects: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_contracts(self) -> Self:
        derived_agent_ids = tuple(contract.agent_id for contract in self.contracts)
        if len(set(derived_agent_ids)) != len(derived_agent_ids):
            raise ValueError("agent_ids must be unique")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match contracts")
        if self.contract_count != len(self.contracts):
            raise ValueError("contract_count must match contracts")
        if self.memory_surfaces != AGENT_MEMORY_SURFACE_ORDER:
            raise ValueError("memory_surfaces must match declared surface order")
        return self


def agent_memory_contract_registry() -> AgentMemoryContractRegistry:
    """Return the static passive V4.1 agent memory contract registry."""

    return AGENT_MEMORY_CONTRACT_REGISTRY


def agent_memory_contract_by_agent_id(
    agent_id: str,
) -> AgentMemoryContract | None:
    """Return one memory contract without reading, writing, or retrieving memory."""

    for contract in AGENT_MEMORY_CONTRACTS:
        if contract.agent_id == agent_id:
            return contract
    return None


def _surface_contract(
    *,
    surface: AgentMemorySurface,
    readable_metadata: tuple[str, ...],
    writable_metadata: tuple[str, ...],
    referenceable_metadata: tuple[str, ...],
) -> AgentMemorySurfaceContract:
    return AgentMemorySurfaceContract(
        surface=surface,
        read_access="metadata_only" if readable_metadata else "none",
        write_access="future_metadata_only" if writable_metadata else "none",
        reference_access="metadata_only" if referenceable_metadata else "none",
        readable_metadata=readable_metadata,
        writable_metadata=writable_metadata,
        referenceable_metadata=referenceable_metadata,
        authority_boundary=(
            "This memory surface contract describes passive metadata access "
            "only; it does not read, write, persist, retrieve, materialize "
            "blackboard memory, or mutate shared context."
        ),
    )


def _contract(agent_id: str) -> AgentMemoryContract:
    read_surfaces = _AGENT_READ_SURFACES[agent_id]
    surfaces = tuple(
        _surface_contract(
            surface=surface,
            readable_metadata=(
                _SURFACE_READ_METADATA[surface] if surface in read_surfaces else ()
            ),
            writable_metadata=(
                _AGENT_FUTURE_BLACKBOARD_WRITES[agent_id]
                if surface == "future_blackboard"
                else ()
            ),
            referenceable_metadata=_SURFACE_REFERENCE_METADATA[surface],
        )
        for surface in AGENT_MEMORY_SURFACE_ORDER
    )
    return AgentMemoryContract(
        agent_id=agent_id,
        memory_contract_id=f"{agent_id}_memory_contract",
        surfaces=surfaces,
        readable_memory_surfaces=tuple(
            surface.surface for surface in surfaces if surface.read_access != "none"
        ),
        writable_memory_surfaces=tuple(
            surface.surface for surface in surfaces if surface.write_access != "none"
        ),
        referenceable_memory_surfaces=tuple(
            surface.surface
            for surface in surfaces
            if surface.reference_access != "none"
        ),
        future_blackboard_hooks=(f"{agent_id}_future_blackboard_metadata",),
        future_shared_context_hooks=(f"{agent_id}_shared_context_view_metadata",),
    )


AGENT_MEMORY_CONTRACTS = tuple(
    _contract(identity.agent_id) for identity in AGENT_IDENTITIES
)
AGENT_MEMORY_CONTRACT_REGISTRY = AgentMemoryContractRegistry(
    contracts=AGENT_MEMORY_CONTRACTS,
    agent_ids=tuple(contract.agent_id for contract in AGENT_MEMORY_CONTRACTS),
    contract_count=len(AGENT_MEMORY_CONTRACTS),
)
