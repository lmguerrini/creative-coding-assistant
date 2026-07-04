"""Stable passive V4.1 agent identity metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

AgentRoleFamily = Literal[
    "planning",
    "research",
    "style",
    "runtime",
    "artifact",
    "art_direction",
    "critique",
    "narrative",
    "curation",
    "refinement",
    "synthesis",
]
AgentCapabilityClass = Literal[
    "planning_strategy",
    "source_context",
    "visual_style",
    "runtime_compatibility",
    "artifact_implementation",
    "creative_direction",
    "aesthetic_evaluation",
    "symbolic_narrative",
    "creative_selection",
    "quality_review",
    "refinement_planning",
    "final_response_synthesis",
]
AgentAuthorityScope = Literal["metadata_only_contract"]
AgentIdentityVisibility = Literal["inspectable_registry_metadata"]

AGENT_IDENTITY_SERIALIZATION_VERSION = "agent_identity.v1"
AGENT_IDENTITY_REGISTRY_SERIALIZATION_VERSION = "agent_identity_registry.v1"
AGENT_IDENTITY_REGISTRY_AUTHORITY_BOUNDARY = (
    "Agent identity metadata defines stable V4.1 agent names, role families, "
    "purposes, capability classes, authority scopes, visibility, and versions "
    "only; it does not create persistent user identities, hidden agent state, "
    "agent execution, routing, provider calls, runtime selection, retries, "
    "memory mutation, artifact execution, or generated output changes."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "persistent_user_identity",
    "hidden_agent_state",
    "agent_instantiation",
    "agent_invocation",
    "dynamic_agent_routing",
    "provider_or_model_routing",
    "runtime_selection",
    "retry_or_refinement_triggering",
    "memory_storage_or_mutation",
    "artifact_execution",
    "generated_output_modification",
)


class AgentIdentityMetadata(BaseModel):
    """Deterministic identity metadata for one passive V4.1 agent."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    agent_name: str = Field(min_length=1, max_length=140)
    role_family: AgentRoleFamily
    purpose: str = Field(min_length=1, max_length=260)
    capability_class: AgentCapabilityClass
    authority_scope: AgentAuthorityScope = "metadata_only_contract"
    visibility: AgentIdentityVisibility = "inspectable_registry_metadata"
    identity_version: Literal["v4.1"] = "v4.1"
    contract_hook: str = Field(min_length=1, max_length=120)
    persistent_user_identity: Literal[False] = False
    hidden_state: Literal[False] = False
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    serialization_version: Literal["agent_identity.v1"] = (
        AGENT_IDENTITY_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentIdentityRegistry(BaseModel):
    """Stable registry of passive V4.1 agent identity metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_identity_registry"] = "agent_identity_registry"
    serialization_version: Literal["agent_identity_registry.v1"] = (
        AGENT_IDENTITY_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_IDENTITY_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    identities: tuple[AgentIdentityMetadata, ...] = Field(
        min_length=12,
        max_length=12,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    identity_count: int = Field(ge=12, le=12)
    role_families: tuple[str, ...] = Field(min_length=11, max_length=11)
    identity_version: Literal["v4.1"] = "v4.1"
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_identities(self) -> Self:
        derived_agent_ids = tuple(identity.agent_id for identity in self.identities)
        if len(set(derived_agent_ids)) != len(derived_agent_ids):
            raise ValueError("agent_ids must be unique")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match identities")
        if self.identity_count != len(self.identities):
            raise ValueError("identity_count must match identities")
        derived_role_families = tuple(
            dict.fromkeys(identity.role_family for identity in self.identities)
        )
        if self.role_families != derived_role_families:
            raise ValueError("role_families must match identities")
        return self


def agent_identity_registry() -> AgentIdentityRegistry:
    """Return the static passive V4.1 agent identity registry."""

    return AGENT_IDENTITY_REGISTRY


def agent_identity_by_id(agent_id: str) -> AgentIdentityMetadata | None:
    """Return one identity without creating or invoking an agent."""

    for identity in AGENT_IDENTITIES:
        if identity.agent_id == agent_id:
            return identity
    return None


def agent_identities_by_role_family(
    role_family: str,
) -> tuple[AgentIdentityMetadata, ...]:
    """Return identities in one family without changing runtime behavior."""

    return tuple(
        identity for identity in AGENT_IDENTITIES if identity.role_family == role_family
    )


def _identity(
    *,
    agent_id: str,
    agent_name: str,
    role_family: AgentRoleFamily,
    purpose: str,
    capability_class: AgentCapabilityClass,
) -> AgentIdentityMetadata:
    return AgentIdentityMetadata(
        agent_id=agent_id,
        agent_name=agent_name,
        role_family=role_family,
        purpose=purpose,
        capability_class=capability_class,
        contract_hook=f"{agent_id}_contract",
    )


AGENT_IDENTITIES = (
    _identity(
        agent_id="planner_agent",
        agent_name="Planner Agent",
        role_family="planning",
        purpose=(
            "Frames request requirements, planning gaps, and handoff metadata "
            "for future agent orchestration."
        ),
        capability_class="planning_strategy",
    ),
    _identity(
        agent_id="research_agent",
        agent_name="Research Agent",
        role_family="research",
        purpose=(
            "Tracks source context and evidence metadata needed by later "
            "research-aware agent contracts."
        ),
        capability_class="source_context",
    ),
    _identity(
        agent_id="style_agent",
        agent_name="Style Agent",
        role_family="style",
        purpose=(
            "Describes visual style metadata responsibilities for future "
            "creative styling contracts."
        ),
        capability_class="visual_style",
    ),
    _identity(
        agent_id="runtime_agent",
        agent_name="Runtime Agent",
        role_family="runtime",
        purpose=(
            "Identifies runtime compatibility metadata responsibilities for "
            "future passive runtime contracts."
        ),
        capability_class="runtime_compatibility",
    ),
    _identity(
        agent_id="artifact_agent",
        agent_name="Artifact Agent",
        role_family="artifact",
        purpose=(
            "Identifies artifact implementation metadata responsibilities "
            "without creating, executing, or modifying artifacts."
        ),
        capability_class="artifact_implementation",
    ),
    _identity(
        agent_id="art_direction_agent",
        agent_name="Art Direction Agent",
        role_family="art_direction",
        purpose=(
            "Describes creative direction metadata responsibilities for later "
            "art direction contracts."
        ),
        capability_class="creative_direction",
    ),
    _identity(
        agent_id="aesthetic_critic_agent",
        agent_name="Aesthetic Critic Agent",
        role_family="critique",
        purpose=(
            "Identifies aesthetic evaluation metadata responsibilities for "
            "future critique contracts."
        ),
        capability_class="aesthetic_evaluation",
    ),
    _identity(
        agent_id="narrative_symbolic_agent",
        agent_name="Narrative & Symbolic Agent",
        role_family="narrative",
        purpose=(
            "Describes symbolic narrative metadata responsibilities for future "
            "narrative-aware contracts."
        ),
        capability_class="symbolic_narrative",
    ),
    _identity(
        agent_id="creative_curator_agent",
        agent_name="Creative Curator Agent",
        role_family="curation",
        purpose=(
            "Identifies creative selection and curation metadata "
            "responsibilities for future curator contracts."
        ),
        capability_class="creative_selection",
    ),
    _identity(
        agent_id="critic_agent",
        agent_name="Critic Agent",
        role_family="critique",
        purpose=(
            "Describes quality review metadata responsibilities for later "
            "general critic contracts."
        ),
        capability_class="quality_review",
    ),
    _identity(
        agent_id="refiner_agent",
        agent_name="Refiner Agent",
        role_family="refinement",
        purpose=(
            "Identifies refinement planning metadata responsibilities without "
            "triggering revisions or retries."
        ),
        capability_class="refinement_planning",
    ),
    _identity(
        agent_id="final_synthesizer_agent",
        agent_name="Final Synthesizer Agent",
        role_family="synthesis",
        purpose=(
            "Describes final response synthesis metadata responsibilities for "
            "future synthesis contracts."
        ),
        capability_class="final_response_synthesis",
    ),
)

AGENT_IDENTITY_REGISTRY = AgentIdentityRegistry(
    identities=AGENT_IDENTITIES,
    agent_ids=tuple(identity.agent_id for identity in AGENT_IDENTITIES),
    identity_count=len(AGENT_IDENTITIES),
    role_families=tuple(
        dict.fromkeys(identity.role_family for identity in AGENT_IDENTITIES)
    ),
)
