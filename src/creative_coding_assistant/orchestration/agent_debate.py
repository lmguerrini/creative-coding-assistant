"""Passive V4.2 advisory agent debate metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_contracts import AGENT_CONTRACTS

DebateRole = Literal["claimant", "counterclaimant", "evidence_reviewer"]
DebateTopicId = Literal[
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
]

DEBATE_PARTICIPANT_SERIALIZATION_VERSION = "agent_debate_participant.v1"
DEBATE_CLAIM_SERIALIZATION_VERSION = "agent_debate_claim.v1"
DEBATE_ROUND_SERIALIZATION_VERSION = "agent_debate_round.v1"
DEBATE_REGISTRY_SERIALIZATION_VERSION = "agent_debate_registry.v1"
DEBATE_REGISTRY_AUTHORITY_BOUNDARY = (
    "Agent debate metadata describes bounded advisory participants, claims, "
    "counterclaims, evidence surfaces, and non-autonomous debate boundaries "
    "only; it does not execute debate loops, trigger retries, change generated "
    "output, invoke agents, route providers or models, schedule work, or mutate "
    "state."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "debate_loop_execution",
    "retry_triggering",
    "generated_output_modification",
    "agent_invocation",
    "provider_or_model_routing",
    "workflow_control",
    "state_mutation",
)

_DEBATE_TOPICS: tuple[DebateTopicId, ...] = (
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
)

_CLAIMS: dict[DebateTopicId, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "planning_execution_fit": (
        "planner_agent",
        ("runtime_agent", "artifact_agent"),
        ("planning_context_packet", "runtime_fit_notes", "artifact_readiness_notes"),
    ),
    "style_aesthetic_alignment": (
        "art_direction_agent",
        ("style_agent", "aesthetic_critic_agent", "critic_agent"),
        ("style_context_notes", "art_direction_notes", "aesthetic_review_notes"),
    ),
    "curation_refinement_need": (
        "creative_curator_agent",
        ("critic_agent", "refiner_agent"),
        ("curation_context_notes", "quality_review_signals", "revision_candidate_notes"),
    ),
    "final_synthesis_readiness": (
        "final_synthesizer_agent",
        ("planner_agent", "critic_agent", "refiner_agent"),
        ("final_handoff_summary", "planning_gap_summary", "refinement_context_notes"),
    ),
}


class AgentDebateParticipant(BaseModel):
    """Passive debate participant metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    participant_id: str = Field(min_length=1, max_length=140)
    agent_id: str = Field(min_length=1, max_length=80)
    debate_roles: tuple[DebateRole, ...] = Field(min_length=1, max_length=3)
    evidence_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    debate_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_debate_participant.v1"] = (
        DEBATE_PARTICIPANT_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentDebateClaimContract(BaseModel):
    """Advisory claim and counterclaim metadata for one debate topic."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    claim_id: str = Field(min_length=1, max_length=140)
    topic_id: DebateTopicId
    claimant_agent_id: str = Field(min_length=1, max_length=80)
    counterclaim_agent_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    claim_surface: str = Field(min_length=1, max_length=240)
    counterclaim_surface: str = Field(min_length=1, max_length=240)
    evidence_surfaces: tuple[str, ...] = Field(min_length=1, max_length=12)
    advisory_output_keys: tuple[str, ...] = Field(min_length=1, max_length=8)
    debate_boundary: str = Field(min_length=1, max_length=600)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    debate_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_debate_claim.v1"] = (
        DEBATE_CLAIM_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentDebateRoundContract(BaseModel):
    """Bounded advisory debate round metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    round_id: str = Field(min_length=1, max_length=140)
    topic_id: DebateTopicId
    participant_agent_ids: tuple[str, ...] = Field(min_length=2, max_length=8)
    claim_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    max_exchange_count: int = Field(ge=1, le=2)
    evidence_surfaces: tuple[str, ...] = Field(min_length=1, max_length=12)
    round_boundary: str = Field(min_length=1, max_length=600)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    debate_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_debate_round.v1"] = (
        DEBATE_ROUND_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentDebateRegistry(BaseModel):
    """Stable passive V4.2 debate metadata registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_debate_registry"] = "agent_debate_registry"
    serialization_version: Literal["agent_debate_registry.v1"] = (
        DEBATE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=DEBATE_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    participants: tuple[AgentDebateParticipant, ...] = Field(
        min_length=12,
        max_length=12,
    )
    claims: tuple[AgentDebateClaimContract, ...] = Field(min_length=4, max_length=4)
    rounds: tuple[AgentDebateRoundContract, ...] = Field(min_length=4, max_length=4)
    participant_agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    claim_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    round_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[DebateTopicId, ...] = Field(min_length=4, max_length=4)
    max_rounds: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=2, max_length=2)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    debate_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_debate_contracts(self) -> Self:
        derived_participant_ids = tuple(
            participant.agent_id for participant in self.participants
        )
        derived_claim_ids = tuple(claim.claim_id for claim in self.claims)
        derived_round_ids = tuple(round_contract.round_id for round_contract in self.rounds)
        derived_topic_ids = tuple(round_contract.topic_id for round_contract in self.rounds)
        if self.participant_agent_ids != derived_participant_ids:
            raise ValueError("participant_agent_ids must match participants")
        if self.claim_ids != derived_claim_ids:
            raise ValueError("claim_ids must match claims")
        if self.round_ids != derived_round_ids:
            raise ValueError("round_ids must match rounds")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match rounds")
        if derived_topic_ids != _DEBATE_TOPICS:
            raise ValueError("rounds must preserve deterministic topic order")
        known_agents = set(self.participant_agent_ids)
        known_claims = set(self.claim_ids)
        for claim in self.claims:
            if claim.claimant_agent_id not in known_agents:
                raise ValueError("claimant_agent_id must be a known participant")
            if not set(claim.counterclaim_agent_ids).issubset(known_agents):
                raise ValueError("counterclaim_agent_ids must be known participants")
        for round_contract in self.rounds:
            if not set(round_contract.participant_agent_ids).issubset(known_agents):
                raise ValueError("round participant_agent_ids must be known participants")
            if not set(round_contract.claim_ids).issubset(known_claims):
                raise ValueError("round claim_ids must be known claims")
        return self


def agent_debate_registry() -> AgentDebateRegistry:
    """Return passive V4.2 advisory debate metadata."""

    return AGENT_DEBATE_REGISTRY


def debate_claim_by_id(
    claim_id: str,
    registry: AgentDebateRegistry | None = None,
) -> AgentDebateClaimContract | None:
    """Return one debate claim without executing debate behavior."""

    source_registry = registry or AGENT_DEBATE_REGISTRY
    for claim in source_registry.claims:
        if claim.claim_id == claim_id:
            return claim
    return None


def debate_round_by_topic(
    topic_id: DebateTopicId,
    registry: AgentDebateRegistry | None = None,
) -> AgentDebateRoundContract | None:
    """Return one debate round without triggering retries or generation."""

    source_registry = registry or AGENT_DEBATE_REGISTRY
    for round_contract in source_registry.rounds:
        if round_contract.topic_id == topic_id:
            return round_contract
    return None


def debate_participant_by_agent_id(
    agent_id: str,
    registry: AgentDebateRegistry | None = None,
) -> AgentDebateParticipant | None:
    """Return one participant without invoking an agent."""

    source_registry = registry or AGENT_DEBATE_REGISTRY
    for participant in source_registry.participants:
        if participant.agent_id == agent_id:
            return participant
    return None


def _participant(agent_id: str) -> AgentDebateParticipant:
    roles: list[DebateRole] = ["evidence_reviewer"]
    for claimant, counterclaims, _evidence in _CLAIMS.values():
        if agent_id == claimant and "claimant" not in roles:
            roles.append("claimant")
        if agent_id in counterclaims and "counterclaimant" not in roles:
            roles.append("counterclaimant")
    return AgentDebateParticipant(
        participant_id=f"debate_participant::{agent_id}",
        agent_id=agent_id,
        debate_roles=tuple(roles),
        evidence_surface_ids=(f"{agent_id}_shared_context_view",),
    )


def _claim(topic_id: DebateTopicId) -> AgentDebateClaimContract:
    claimant, counterclaims, evidence = _CLAIMS[topic_id]
    return AgentDebateClaimContract(
        claim_id=f"debate_claim::{topic_id}",
        topic_id=topic_id,
        claimant_agent_id=claimant,
        counterclaim_agent_ids=counterclaims,
        claim_surface=f"{topic_id}_claim_metadata",
        counterclaim_surface=f"{topic_id}_counterclaim_metadata",
        evidence_surfaces=evidence,
        advisory_output_keys=(
            f"{topic_id}_claim_summary",
            f"{topic_id}_counterclaim_summary",
            f"{topic_id}_evidence_notes",
        ),
        debate_boundary=(
            "Debate claim metadata is advisory only; it does not execute a "
            "debate loop, trigger retries, invoke agents, or change generated "
            "output."
        ),
    )


def _round(topic_id: DebateTopicId) -> AgentDebateRoundContract:
    claim = _claim(topic_id)
    participants = (claim.claimant_agent_id,) + claim.counterclaim_agent_ids
    return AgentDebateRoundContract(
        round_id=f"debate_round::{topic_id}",
        topic_id=topic_id,
        participant_agent_ids=participants,
        claim_ids=(claim.claim_id,),
        max_exchange_count=2,
        evidence_surfaces=claim.evidence_surfaces,
        round_boundary=(
            "Debate round metadata is bounded and advisory only; it does not "
            "run debate turns, trigger retries, or mutate generated output."
        ),
    )


DEBATE_PARTICIPANTS = tuple(
    _participant(contract.agent_id) for contract in AGENT_CONTRACTS
)
DEBATE_CLAIMS = tuple(_claim(topic_id) for topic_id in _DEBATE_TOPICS)
DEBATE_ROUNDS = tuple(_round(topic_id) for topic_id in _DEBATE_TOPICS)
AGENT_DEBATE_REGISTRY = AgentDebateRegistry(
    participants=DEBATE_PARTICIPANTS,
    claims=DEBATE_CLAIMS,
    rounds=DEBATE_ROUNDS,
    participant_agent_ids=tuple(participant.agent_id for participant in DEBATE_PARTICIPANTS),
    claim_ids=tuple(claim.claim_id for claim in DEBATE_CLAIMS),
    round_ids=tuple(round_contract.round_id for round_contract in DEBATE_ROUNDS),
    topic_ids=tuple(round_contract.topic_id for round_contract in DEBATE_ROUNDS),
    max_rounds=len(DEBATE_ROUNDS),
    source_registries=(
        "agent_contract_registry",
        "shared_context_view_registry",
    ),
)
