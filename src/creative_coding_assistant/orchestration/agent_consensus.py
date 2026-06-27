"""Passive V4.2 consensus builder metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_debate import (
    DEBATE_ROUNDS,
    DebateTopicId,
)

ConsensusAggregationMode = Literal["placeholder_only"]

CONSENSUS_VOTING_INPUT_SERIALIZATION_VERSION = "consensus_voting_input.v1"
CONSENSUS_AGREEMENT_SURFACE_SERIALIZATION_VERSION = "consensus_agreement_surface.v1"
CONSENSUS_REGISTRY_SERIALIZATION_VERSION = "consensus_builder_registry.v1"
CONSENSUS_REGISTRY_AUTHORITY_BOUNDARY = (
    "Consensus builder metadata describes voting inputs, confidence "
    "aggregation placeholders, agreement surfaces, disagreement surfaces, and "
    "non-executable consensus boundaries only; it does not execute voting, "
    "select final answers, mutate final synthesis, invoke agents, route "
    "providers or models, trigger retries, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "voting_execution",
    "final_answer_selection",
    "final_synthesis_mutation",
    "agent_invocation",
    "provider_or_model_routing",
    "retry_or_refinement_triggering",
    "generated_output_modification",
)

_VOTING_DIMENSIONS = (
    "agreement",
    "confidence",
    "risk",
    "evidence_coverage",
)


class ConsensusVotingInputContract(BaseModel):
    """Passive voting input metadata for one debate topic."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    voting_input_id: str = Field(min_length=1, max_length=140)
    topic_id: DebateTopicId
    participant_agent_ids: tuple[str, ...] = Field(min_length=2, max_length=8)
    claim_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    voting_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    confidence_placeholder_key: str = Field(min_length=1, max_length=120)
    aggregation_mode: ConsensusAggregationMode = "placeholder_only"
    input_boundary: str = Field(min_length=1, max_length=600)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    voting_execution_implemented: Literal[False] = False
    final_answer_selection_implemented: Literal[False] = False
    final_synthesis_mutation_implemented: Literal[False] = False
    serialization_version: Literal["consensus_voting_input.v1"] = (
        CONSENSUS_VOTING_INPUT_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class ConsensusAgreementSurface(BaseModel):
    """Passive agreement and disagreement surface metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    surface_id: str = Field(min_length=1, max_length=140)
    topic_id: DebateTopicId
    agreement_metadata_keys: tuple[str, ...] = Field(min_length=1, max_length=8)
    disagreement_metadata_keys: tuple[str, ...] = Field(min_length=1, max_length=8)
    unresolved_risk_keys: tuple[str, ...] = Field(min_length=1, max_length=8)
    confidence_placeholder_key: str = Field(min_length=1, max_length=120)
    surface_boundary: str = Field(min_length=1, max_length=600)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    voting_execution_implemented: Literal[False] = False
    final_answer_selection_implemented: Literal[False] = False
    final_synthesis_mutation_implemented: Literal[False] = False
    serialization_version: Literal["consensus_agreement_surface.v1"] = (
        CONSENSUS_AGREEMENT_SURFACE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class ConsensusBuilderRegistry(BaseModel):
    """Stable passive V4.2 consensus builder metadata registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["consensus_builder_registry"] = "consensus_builder_registry"
    serialization_version: Literal["consensus_builder_registry.v1"] = (
        CONSENSUS_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CONSENSUS_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    voting_inputs: tuple[ConsensusVotingInputContract, ...] = Field(
        min_length=4,
        max_length=4,
    )
    agreement_surfaces: tuple[ConsensusAgreementSurface, ...] = Field(
        min_length=4,
        max_length=4,
    )
    voting_input_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    agreement_surface_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[DebateTopicId, ...] = Field(min_length=4, max_length=4)
    aggregation_mode: ConsensusAggregationMode = "placeholder_only"
    source_registries: tuple[str, ...] = Field(min_length=1, max_length=2)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    voting_execution_implemented: Literal[False] = False
    final_answer_selection_implemented: Literal[False] = False
    final_synthesis_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_consensus_contracts(self) -> Self:
        derived_input_ids = tuple(item.voting_input_id for item in self.voting_inputs)
        derived_surface_ids = tuple(surface.surface_id for surface in self.agreement_surfaces)
        derived_topic_ids = tuple(item.topic_id for item in self.voting_inputs)
        if self.voting_input_ids != derived_input_ids:
            raise ValueError("voting_input_ids must match voting_inputs")
        if self.agreement_surface_ids != derived_surface_ids:
            raise ValueError("agreement_surface_ids must match agreement_surfaces")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match voting_inputs")
        if tuple(surface.topic_id for surface in self.agreement_surfaces) != self.topic_ids:
            raise ValueError("agreement_surfaces must match voting input topics")
        for item in self.voting_inputs:
            if item.aggregation_mode != self.aggregation_mode:
                raise ValueError("aggregation_mode must match voting inputs")
        return self


def consensus_builder_registry() -> ConsensusBuilderRegistry:
    """Return passive V4.2 consensus builder metadata."""

    return CONSENSUS_BUILDER_REGISTRY


def consensus_voting_input_by_topic(
    topic_id: DebateTopicId,
    registry: ConsensusBuilderRegistry | None = None,
) -> ConsensusVotingInputContract | None:
    """Return one voting input without executing voting."""

    source_registry = registry or CONSENSUS_BUILDER_REGISTRY
    for item in source_registry.voting_inputs:
        if item.topic_id == topic_id:
            return item
    return None


def consensus_agreement_surface_by_topic(
    topic_id: DebateTopicId,
    registry: ConsensusBuilderRegistry | None = None,
) -> ConsensusAgreementSurface | None:
    """Return one agreement surface without selecting final answers."""

    source_registry = registry or CONSENSUS_BUILDER_REGISTRY
    for surface in source_registry.agreement_surfaces:
        if surface.topic_id == topic_id:
            return surface
    return None


def _voting_input(round_index: int) -> ConsensusVotingInputContract:
    round_contract = DEBATE_ROUNDS[round_index]
    topic_id = round_contract.topic_id
    return ConsensusVotingInputContract(
        voting_input_id=f"consensus_voting_input::{topic_id}",
        topic_id=topic_id,
        participant_agent_ids=round_contract.participant_agent_ids,
        claim_ids=round_contract.claim_ids,
        voting_dimensions=_VOTING_DIMENSIONS,
        confidence_placeholder_key=f"{topic_id}_confidence_placeholder",
        input_boundary=(
            "Consensus voting input metadata is placeholder-only; it does not "
            "execute voting, select final answers, mutate final synthesis, or "
            "change generated output."
        ),
    )


def _agreement_surface(round_index: int) -> ConsensusAgreementSurface:
    topic_id = DEBATE_ROUNDS[round_index].topic_id
    return ConsensusAgreementSurface(
        surface_id=f"consensus_agreement_surface::{topic_id}",
        topic_id=topic_id,
        agreement_metadata_keys=(
            f"{topic_id}_agreement_points",
            f"{topic_id}_shared_evidence",
        ),
        disagreement_metadata_keys=(
            f"{topic_id}_disagreement_points",
            f"{topic_id}_counterevidence",
        ),
        unresolved_risk_keys=(
            f"{topic_id}_unresolved_risks",
            f"{topic_id}_review_notes",
        ),
        confidence_placeholder_key=f"{topic_id}_confidence_placeholder",
        surface_boundary=(
            "Consensus agreement surface metadata records advisory agreement "
            "and disagreement placeholders only; it does not select final "
            "answers or mutate synthesis."
        ),
    )


CONSENSUS_VOTING_INPUTS = tuple(
    _voting_input(index) for index in range(len(DEBATE_ROUNDS))
)
CONSENSUS_AGREEMENT_SURFACES = tuple(
    _agreement_surface(index) for index in range(len(DEBATE_ROUNDS))
)
CONSENSUS_BUILDER_REGISTRY = ConsensusBuilderRegistry(
    voting_inputs=CONSENSUS_VOTING_INPUTS,
    agreement_surfaces=CONSENSUS_AGREEMENT_SURFACES,
    voting_input_ids=tuple(item.voting_input_id for item in CONSENSUS_VOTING_INPUTS),
    agreement_surface_ids=tuple(surface.surface_id for surface in CONSENSUS_AGREEMENT_SURFACES),
    topic_ids=tuple(item.topic_id for item in CONSENSUS_VOTING_INPUTS),
    source_registries=("agent_debate_registry",),
)
