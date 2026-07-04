"""Passive V4.1 agent authority boundary metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_contracts import AGENT_CONTRACTS

AGENT_BOUNDARY_SERIALIZATION_VERSION = "agent_boundary.v1"
AGENT_BOUNDARY_REGISTRY_SERIALIZATION_VERSION = "agent_boundary_registry.v1"
AGENT_BOUNDARY_REGISTRY_AUTHORITY_BOUNDARY = (
    "Agent boundary metadata formalizes V4.1 role-specific allowed inputs, "
    "allowed outputs, forbidden behaviors, and boundary rationale only; it "
    "does not implement enforcement runtime, change workflow behavior, add "
    "autonomous escalation, route providers or models, select runtimes, "
    "trigger retries, execute artifacts, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "enforcement_runtime",
    "workflow_behavior_change",
    "autonomous_escalation",
    "agent_execution",
    "dynamic_task_routing",
    "provider_or_model_routing",
    "runtime_selection",
    "retry_or_refinement_triggering",
    "artifact_execution",
    "generated_output_modification",
)

_BOUNDARY_RATIONALE = {
    "planner_agent": (
        "Planner boundaries keep planning metadata inspectable while leaving "
        "V3 planning behavior and workflow order unchanged."
    ),
    "research_agent": (
        "Research boundaries preserve existing retrieval/source behavior while "
        "documenting source-context metadata for future handoff."
    ),
    "style_agent": (
        "Style boundaries describe style interpretation metadata without "
        "changing style generation or style engine behavior."
    ),
    "runtime_agent": (
        "Runtime boundaries expose compatibility metadata without selecting "
        "or executing runtime decisions."
    ),
    "artifact_agent": (
        "Artifact boundaries describe artifact intelligence metadata without "
        "changing generation, export, or artifact execution."
    ),
    "art_direction_agent": (
        "Art direction boundaries describe composition and scene metadata "
        "without autonomous direction behavior."
    ),
    "aesthetic_critic_agent": (
        "Aesthetic critic boundaries isolate visual critique metadata from "
        "Creative Critic scoring and critique loops."
    ),
    "narrative_symbolic_agent": (
        "Narrative boundaries expose meaning-layer metadata without symbolic "
        "generation or prompt semantic changes."
    ),
    "creative_curator_agent": (
        "Curator boundaries document selection preferences without autonomous "
        "final output selection or final synthesis changes."
    ),
    "critic_agent": (
        "Critic boundaries describe broad review metadata without scoring "
        "changes, retries, or evaluation engine execution."
    ),
    "refiner_agent": (
        "Refiner boundaries describe improvement planning metadata without "
        "refinement loop execution or generation changes."
    ),
    "final_synthesizer_agent": (
        "Final synthesizer boundaries describe finalization handoff metadata "
        "without final response generation or provider call changes."
    ),
}


class AgentBoundaryMetadata(BaseModel):
    """Role-specific passive boundary metadata for one V4.1 agent."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    role_id: str = Field(min_length=1, max_length=80)
    authority_boundary: str = Field(min_length=1, max_length=900)
    boundary_rationale: str = Field(min_length=1, max_length=360)
    allowed_inputs: tuple[str, ...] = Field(min_length=1, max_length=40)
    allowed_outputs: tuple[str, ...] = Field(min_length=1, max_length=24)
    forbidden_behaviors: tuple[str, ...] = Field(min_length=1, max_length=24)
    passive_contract_refs: tuple[str, ...] = Field(min_length=1, max_length=8)
    enforcement_runtime_implemented: Literal[False] = False
    workflow_behavior_changed: Literal[False] = False
    autonomous_escalation_added: Literal[False] = False
    serialization_version: Literal["agent_boundary.v1"] = (
        AGENT_BOUNDARY_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentBoundaryRegistry(BaseModel):
    """Stable registry of all passive V4.1 agent authority boundaries."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_boundary_registry"] = "agent_boundary_registry"
    serialization_version: Literal["agent_boundary_registry.v1"] = (
        AGENT_BOUNDARY_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_BOUNDARY_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    boundaries: tuple[AgentBoundaryMetadata, ...] = Field(
        min_length=12,
        max_length=12,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    boundary_count: int = Field(ge=12, le=12)
    source_contract_registry: Literal["agent_contract_registry"] = (
        "agent_contract_registry"
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    enforcement_runtime_implemented: Literal[False] = False
    workflow_behavior_changed: Literal[False] = False
    autonomous_escalation_added: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_boundaries(self) -> Self:
        derived_agent_ids = tuple(boundary.agent_id for boundary in self.boundaries)
        if len(set(derived_agent_ids)) != len(derived_agent_ids):
            raise ValueError("agent_ids must be unique")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match boundaries")
        if self.boundary_count != len(self.boundaries):
            raise ValueError("boundary_count must match boundaries")
        return self


def agent_boundary_registry() -> AgentBoundaryRegistry:
    """Return the static passive V4.1 boundary registry."""

    return AGENT_BOUNDARY_REGISTRY


def agent_boundary_by_agent_id(agent_id: str) -> AgentBoundaryMetadata | None:
    """Return one boundary metadata entry without enforcing runtime behavior."""

    for boundary in AGENT_BOUNDARIES:
        if boundary.agent_id == agent_id:
            return boundary
    return None


def _boundary(agent_id: str) -> AgentBoundaryMetadata:
    contract = next(
        contract for contract in AGENT_CONTRACTS if contract.agent_id == agent_id
    )
    return AgentBoundaryMetadata(
        agent_id=agent_id,
        role_id=contract.role_id,
        authority_boundary=contract.authority_boundary,
        boundary_rationale=_BOUNDARY_RATIONALE[agent_id],
        allowed_inputs=contract.required_inputs + contract.optional_inputs,
        allowed_outputs=contract.produced_outputs,
        forbidden_behaviors=contract.prohibited_actions,
        passive_contract_refs=(
            contract.serialization_version,
            "agent_identity.v1",
            "agent_memory_contract.v1",
        ),
    )


AGENT_BOUNDARIES = tuple(_boundary(contract.agent_id) for contract in AGENT_CONTRACTS)
AGENT_BOUNDARY_REGISTRY = AgentBoundaryRegistry(
    boundaries=AGENT_BOUNDARIES,
    agent_ids=tuple(boundary.agent_id for boundary in AGENT_BOUNDARIES),
    boundary_count=len(AGENT_BOUNDARIES),
)
