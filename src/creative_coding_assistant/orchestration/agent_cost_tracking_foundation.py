"""Passive V4.6 agent cost tracking foundation metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_contracts import (
    AGENT_CONTRACT_REGISTRY,
    AgentContract,
    agent_contract_by_id,
)
from creative_coding_assistant.orchestration.agent_metadata import (
    AgentOperationalMetadata,
    agent_metadata_by_agent_id,
    agent_metadata_registry,
)
from creative_coding_assistant.orchestration.engine_contract_consistency import (
    engine_contract_consistency_registry,
)
from creative_coding_assistant.orchestration.hybrid_agentic_workflow import (
    cost_threshold_routing_registry,
    creative_exploration_budget_registry,
)
from creative_coding_assistant.orchestration.hybrid_studio import (
    cost_profile_registry,
)

AgentCostTrackingFoundationStage = Literal["v4_6_agent_cost_tracking_foundation"]
AgentCostTrackingFoundationStatus = Literal["pass"]

AGENT_COST_TRACKING_FOUNDATION_PROFILE_SERIALIZATION_VERSION = (
    "agent_cost_tracking_foundation_profile.v1"
)
AGENT_COST_TRACKING_FOUNDATION_REGISTRY_SERIALIZATION_VERSION = (
    "agent_cost_tracking_foundation_registry.v1"
)
AGENT_COST_TRACKING_FOUNDATION_REGISTRY_AUTHORITY_BOUNDARY = (
    "V4.6 agent cost tracking foundation metadata describes passive agent "
    "cost class declarations, cost basis text, cache sensitivity notes, "
    "creative exploration budget references, cost threshold references, "
    "Hybrid Studio cost profile references, engine contract cost consistency "
    "references, and cost boundary flags only; it does not meter cost, "
    "calculate pricing, call providers, enforce budgets, optimize execution, "
    "route by cost, select providers or models, invoke agents, control "
    "workflows, trigger retries, mutate prompts, write storage, or modify "
    "generated output."
)

_SOURCE_COST_REGISTRIES = (
    "agent_contract_registry",
    "agent_metadata_registry",
    "creative_exploration_budget_registry",
    "cost_threshold_routing_registry",
    "cost_profile_registry",
    "engine_contract_consistency_registry",
)
_COST_DIMENSIONS = (
    "agent_identity",
    "role_identity",
    "contract_cost_class",
    "metadata_cost_class",
    "cost_basis",
    "cache_sensitivity",
    "external_provider_call_boundary",
    "creative_budget_reference",
    "cost_threshold_reference",
    "studio_cost_profile_reference",
    "engine_contract_cost_consistency",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "cost_tracking_engine_blocked",
    "cost_metering_blocked",
    "pricing_lookup_blocked",
    "budget_enforcement_blocked",
    "cost_based_routing_blocked",
    "execution_optimization_blocked",
    "provider_model_routing_blocked",
    "agent_invocation_blocked",
    "workflow_control_blocked",
    "generated_output_mutation_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "cost_tracking_engine",
    "cost_metering",
    "pricing_lookup",
    "budget_enforcement",
    "cost_based_routing",
    "execution_optimization",
    "provider_or_model_routing",
    "agent_invocation",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)
_FOUNDATION_FINDINGS = (
    "contract_cost_metadata_confirmed",
    "agent_metadata_cost_alignment_confirmed",
    "creative_budget_references_confirmed",
    "cost_threshold_references_confirmed",
    "studio_cost_profile_references_confirmed",
    "engine_contract_cost_consistency_confirmed",
    "runtime_cost_tracking_blocks_confirmed",
)


class AgentCostTrackingFoundationProfile(BaseModel):
    """One passive cost tracking foundation profile for an agent."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    role_id: str = Field(min_length=1, max_length=80)
    cost_tracking_stage: AgentCostTrackingFoundationStage = (
        "v4_6_agent_cost_tracking_foundation"
    )
    contract_serialization_version: str = Field(min_length=1, max_length=80)
    metadata_serialization_version: str = Field(min_length=1, max_length=80)
    contract_cost_class: str = Field(min_length=1, max_length=40)
    metadata_cost_class: str = Field(min_length=1, max_length=40)
    contract_cost_basis: str = Field(min_length=1, max_length=260)
    metadata_cost_basis: str = Field(min_length=1, max_length=260)
    contract_cache_sensitivity: str = Field(min_length=1, max_length=260)
    external_provider_calls_declared: Literal[False] = False
    budget_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    budget_postures: tuple[str, ...] = Field(min_length=4, max_length=4)
    cost_threshold_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    cost_threshold_bands: tuple[str, ...] = Field(min_length=4, max_length=4)
    cost_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    studio_cost_bands: tuple[str, ...] = Field(min_length=4, max_length=4)
    consistency_family_ids: tuple[str, ...] = Field(min_length=3, max_length=3)
    cost_source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    cost_dimensions: tuple[str, ...] = Field(min_length=11, max_length=11)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=10, max_length=10)
    foundation_findings: tuple[str, ...] = Field(min_length=7, max_length=7)
    missing_coverage_items: tuple[str, ...] = Field(default_factory=tuple, max_length=20)
    contract_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=16,
    )
    metadata_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=16,
    )
    budget_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    threshold_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    profile_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=16,
    )
    consistency_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    foundation_status: AgentCostTrackingFoundationStatus = "pass"
    metadata_only_declared: Literal[True] = True
    cost_metadata_alignment_present: Literal[True] = True
    budget_threshold_reference_present: Literal[True] = True
    studio_cost_profile_reference_present: Literal[True] = True
    cost_tracking_implemented: Literal[False] = False
    cost_metering_implemented: Literal[False] = False
    pricing_lookup_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    cost_based_routing_implemented: Literal[False] = False
    execution_optimization_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    external_provider_calls_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_cost_tracking_foundation_profile.v1"] = (
        AGENT_COST_TRACKING_FOUNDATION_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentCostTrackingFoundationRegistry(BaseModel):
    """Stable passive V4.6 registry for agent cost tracking metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_cost_tracking_foundation_registry"] = (
        "agent_cost_tracking_foundation_registry"
    )
    serialization_version: Literal["agent_cost_tracking_foundation_registry.v1"] = (
        AGENT_COST_TRACKING_FOUNDATION_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_COST_TRACKING_FOUNDATION_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    cost_tracking_stage: AgentCostTrackingFoundationStage = (
        "v4_6_agent_cost_tracking_foundation"
    )
    profiles: tuple[AgentCostTrackingFoundationProfile, ...] = Field(
        min_length=12,
        max_length=12,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    profile_count: int = Field(ge=12, le=12)
    source_agent_contract_registry: Literal["agent_contract_registry"] = (
        "agent_contract_registry"
    )
    source_agent_metadata_registry: Literal["agent_metadata_registry"] = (
        "agent_metadata_registry"
    )
    source_creative_exploration_budget_registry: Literal[
        "creative_exploration_budget_registry"
    ] = "creative_exploration_budget_registry"
    source_cost_threshold_routing_registry: Literal[
        "cost_threshold_routing_registry"
    ] = "cost_threshold_routing_registry"
    source_cost_profile_registry: Literal["cost_profile_registry"] = (
        "cost_profile_registry"
    )
    source_engine_contract_consistency_registry: Literal[
        "engine_contract_consistency_registry"
    ] = "engine_contract_consistency_registry"
    cost_source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    cost_classes: tuple[str, ...] = Field(min_length=1, max_length=4)
    budget_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    budget_postures: tuple[str, ...] = Field(min_length=4, max_length=4)
    cost_threshold_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    cost_threshold_bands: tuple[str, ...] = Field(min_length=4, max_length=4)
    cost_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    studio_cost_bands: tuple[str, ...] = Field(min_length=4, max_length=4)
    consistency_family_ids: tuple[str, ...] = Field(min_length=3, max_length=3)
    cost_dimensions: tuple[str, ...] = Field(min_length=11, max_length=11)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    all_agents_covered: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    cost_tracking_engine_implemented: Literal[False] = False
    cost_metering_implemented: Literal[False] = False
    pricing_lookup_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    cost_based_routing_implemented: Literal[False] = False
    execution_optimization_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    external_provider_calls_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_agent_ids = tuple(profile.agent_id for profile in self.profiles)
        if len(set(derived_agent_ids)) != len(derived_agent_ids):
            raise ValueError("agent_ids must be unique")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match profiles")
        if self.profile_count != len(self.profiles):
            raise ValueError("profile_count must match profiles")
        if self.cost_source_registries != _SOURCE_COST_REGISTRIES:
            raise ValueError("cost_source_registries must match sources")
        if self.cost_dimensions != _COST_DIMENSIONS:
            raise ValueError("cost_dimensions must match foundation dimensions")
        if self.passive_boundary_flags != _PASSIVE_BOUNDARY_FLAGS:
            raise ValueError("passive_boundary_flags must match foundation flags")
        if any(
            (
                self.cost_tracking_engine_implemented,
                self.cost_metering_implemented,
                self.pricing_lookup_implemented,
                self.budget_enforcement_implemented,
                self.cost_based_routing_implemented,
                self.execution_optimization_implemented,
                self.provider_model_routing_implemented,
                self.external_provider_calls_implemented,
                self.agent_invocation_implemented,
                self.workflow_control_implemented,
                self.retry_triggering_implemented,
                self.prompt_mutation_implemented,
                self.persistent_storage_write_implemented,
                self.generated_output_mutation_implemented,
            )
        ):
            raise ValueError("cost tracking foundation must remain passive")

        for profile in self.profiles:
            if profile.cost_tracking_stage != self.cost_tracking_stage:
                raise ValueError("cost_tracking_stage must match registry")
            if profile.contract_cost_class not in self.cost_classes:
                raise ValueError("contract_cost_class must be known")
            if profile.metadata_cost_class != profile.contract_cost_class:
                raise ValueError("metadata_cost_class must match contract")
            if profile.metadata_cost_basis != profile.contract_cost_basis:
                raise ValueError("metadata_cost_basis must match contract")
            if profile.budget_profile_ids != self.budget_profile_ids:
                raise ValueError("budget_profile_ids must match registry")
            if profile.budget_postures != self.budget_postures:
                raise ValueError("budget_postures must match registry")
            if profile.cost_threshold_profile_ids != self.cost_threshold_profile_ids:
                raise ValueError("cost_threshold_profile_ids must match registry")
            if profile.cost_threshold_bands != self.cost_threshold_bands:
                raise ValueError("cost_threshold_bands must match registry")
            if profile.cost_profile_ids != self.cost_profile_ids:
                raise ValueError("cost_profile_ids must match registry")
            if profile.studio_cost_bands != self.studio_cost_bands:
                raise ValueError("studio_cost_bands must match registry")
            if profile.consistency_family_ids != self.consistency_family_ids:
                raise ValueError("consistency_family_ids must match registry")
            if profile.cost_source_registries != self.cost_source_registries:
                raise ValueError("cost_source_registries must match registry")
            if profile.cost_dimensions != self.cost_dimensions:
                raise ValueError("cost_dimensions must match registry")
            if profile.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if profile.missing_coverage_items:
                raise ValueError("profiles must not contain missing coverage")
            if any(
                (
                    profile.external_provider_calls_declared,
                    profile.cost_tracking_implemented,
                    profile.cost_metering_implemented,
                    profile.pricing_lookup_implemented,
                    profile.budget_enforcement_implemented,
                    profile.cost_based_routing_implemented,
                    profile.execution_optimization_implemented,
                    profile.provider_model_routing_implemented,
                    profile.external_provider_calls_implemented,
                    profile.agent_invocation_implemented,
                    profile.workflow_control_implemented,
                    profile.retry_triggering_implemented,
                    profile.prompt_mutation_implemented,
                    profile.persistent_storage_write_implemented,
                    profile.generated_output_mutation_implemented,
                )
            ):
                raise ValueError("cost tracking profiles must remain passive")
        return self


def agent_cost_tracking_foundation_registry() -> (
    AgentCostTrackingFoundationRegistry
):
    """Return passive V4.6 agent cost tracking foundation metadata."""

    return AGENT_COST_TRACKING_FOUNDATION_REGISTRY


def agent_cost_tracking_profile_by_agent_id(
    agent_id: str,
    registry: AgentCostTrackingFoundationRegistry | None = None,
) -> AgentCostTrackingFoundationProfile | None:
    """Return one cost tracking foundation profile by agent id."""

    source_registry = registry or AGENT_COST_TRACKING_FOUNDATION_REGISTRY
    for profile in source_registry.profiles:
        if profile.agent_id == agent_id:
            return profile
    return None


def agent_cost_tracking_profiles_for_cost_class(
    cost_class: str,
    registry: AgentCostTrackingFoundationRegistry | None = None,
) -> tuple[AgentCostTrackingFoundationProfile, ...]:
    """Return passive cost tracking profiles for one declared cost class."""

    source_registry = registry or AGENT_COST_TRACKING_FOUNDATION_REGISTRY
    normalized_cost_class = str(cost_class).strip()
    return tuple(
        profile
        for profile in source_registry.profiles
        if profile.contract_cost_class == normalized_cost_class
    )


def agent_cost_tracking_profiles_for_cost_profile(
    cost_profile_id: str,
    registry: AgentCostTrackingFoundationRegistry | None = None,
) -> tuple[AgentCostTrackingFoundationProfile, ...]:
    """Return passive profiles that reference one Hybrid Studio cost profile."""

    source_registry = registry or AGENT_COST_TRACKING_FOUNDATION_REGISTRY
    normalized_profile_id = str(cost_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.profiles
        if normalized_profile_id in profile.cost_profile_ids
    )


def _missing_coverage_items(
    *,
    contract: AgentContract,
    metadata: AgentOperationalMetadata,
) -> tuple[str, ...]:
    metadata_registry = agent_metadata_registry()
    budget = creative_exploration_budget_registry()
    threshold = cost_threshold_routing_registry()
    studio_cost = cost_profile_registry()
    consistency = engine_contract_consistency_registry()
    missing: list[str] = []
    if metadata.estimated_cost_class != contract.estimated_cost_metadata.relative_cost:
        missing.append("agent_metadata_cost_class_alignment_missing")
    if metadata.estimated_cost_basis != contract.estimated_cost_metadata.cost_basis:
        missing.append("agent_metadata_cost_basis_alignment_missing")
    if contract.estimated_cost_metadata.external_provider_calls:
        missing.append("external_provider_call_boundary_missing")
    if not contract.estimated_cost_metadata.cache_sensitivity:
        missing.append("cache_sensitivity_reference_missing")
    if not budget.budget_profile_ids:
        missing.append("creative_budget_reference_missing")
    if not threshold.cost_threshold_profile_ids:
        missing.append("cost_threshold_reference_missing")
    if budget.budget_profile_ids != threshold.budget_profile_ids:
        missing.append("budget_threshold_reference_alignment_missing")
    if not studio_cost.cost_profile_ids:
        missing.append("studio_cost_profile_reference_missing")
    if studio_cost.cost_threshold_profile_ids != threshold.cost_threshold_profile_ids:
        missing.append("studio_threshold_reference_alignment_missing")
    if "cost_metadata" not in consistency.shared_contract_concepts:
        missing.append("engine_contract_cost_consistency_missing")
    if not all(
        (
            contract.metadata_only,
            metadata.metadata_only,
            metadata_registry.metadata_only,
            budget.metadata_only,
            threshold.metadata_only,
            studio_cost.metadata_only,
            consistency.metadata_only,
        )
    ):
        missing.append("metadata_only_declaration_missing")
    if "provider_or_model_routing" not in contract.blocked_runtime_behaviors:
        missing.append("contract_provider_model_routing_block_missing")
    if "cost_or_latency_routing" not in metadata_registry.blocked_runtime_behaviors:
        missing.append("metadata_cost_routing_block_missing")
    if "budget_enforcement" not in budget.blocked_runtime_behaviors:
        missing.append("budget_enforcement_block_missing")
    if "cost_routing" not in budget.blocked_runtime_behaviors:
        missing.append("budget_cost_routing_block_missing")
    if "cost_based_routing" not in threshold.blocked_runtime_behaviors:
        missing.append("threshold_cost_routing_block_missing")
    if "budget_enforcement" not in threshold.blocked_runtime_behaviors:
        missing.append("threshold_budget_enforcement_block_missing")
    if "cost_scoring" not in studio_cost.blocked_runtime_behaviors:
        missing.append("studio_cost_scoring_block_missing")
    if "pricing_lookup" not in studio_cost.blocked_runtime_behaviors:
        missing.append("studio_pricing_lookup_block_missing")
    if "provider_or_model_routing" not in consistency.blocked_runtime_behaviors:
        missing.append("consistency_provider_model_routing_block_missing")
    if budget.budget_enforcement_implemented:
        missing.append("budget_enforcement_enabled")
    if threshold.cost_based_routing_implemented:
        missing.append("cost_threshold_routing_enabled")
    if threshold.budget_enforcement_implemented:
        missing.append("cost_threshold_budget_enforcement_enabled")
    if studio_cost.cost_scoring_implemented:
        missing.append("studio_cost_scoring_enabled")
    if studio_cost.pricing_lookup_implemented:
        missing.append("studio_pricing_lookup_enabled")
    if studio_cost.cost_based_routing_implemented:
        missing.append("studio_cost_routing_enabled")
    if studio_cost.execution_optimization_implemented:
        missing.append("studio_execution_optimization_enabled")
    return tuple(missing)


def _profile(agent_id: str) -> AgentCostTrackingFoundationProfile:
    contract = agent_contract_by_id(agent_id, AGENT_CONTRACT_REGISTRY)
    metadata = agent_metadata_by_agent_id(agent_id)
    metadata_registry = agent_metadata_registry()
    budget = creative_exploration_budget_registry()
    threshold = cost_threshold_routing_registry()
    studio_cost = cost_profile_registry()
    consistency = engine_contract_consistency_registry()
    if contract is None:
        raise ValueError(f"missing agent contract for {agent_id}")
    if metadata is None:
        raise ValueError(f"missing agent metadata for {agent_id}")

    return AgentCostTrackingFoundationProfile(
        agent_id=agent_id,
        role_id=contract.role_id,
        contract_serialization_version=contract.serialization_version,
        metadata_serialization_version=metadata.serialization_version,
        contract_cost_class=contract.estimated_cost_metadata.relative_cost,
        metadata_cost_class=metadata.estimated_cost_class,
        contract_cost_basis=contract.estimated_cost_metadata.cost_basis,
        metadata_cost_basis=metadata.estimated_cost_basis,
        contract_cache_sensitivity=(
            contract.estimated_cost_metadata.cache_sensitivity
        ),
        external_provider_calls_declared=(
            contract.estimated_cost_metadata.external_provider_calls
        ),
        budget_profile_ids=budget.budget_profile_ids,
        budget_postures=tuple(str(posture) for posture in budget.budget_postures),
        cost_threshold_profile_ids=threshold.cost_threshold_profile_ids,
        cost_threshold_bands=tuple(str(band) for band in threshold.cost_bands),
        cost_profile_ids=studio_cost.cost_profile_ids,
        studio_cost_bands=tuple(str(band) for band in studio_cost.cost_bands),
        consistency_family_ids=consistency.family_ids,
        cost_source_registries=_SOURCE_COST_REGISTRIES,
        cost_dimensions=_COST_DIMENSIONS,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        foundation_findings=_FOUNDATION_FINDINGS,
        missing_coverage_items=_missing_coverage_items(
            contract=contract,
            metadata=metadata,
        ),
        contract_blocked_runtime_behaviors=contract.blocked_runtime_behaviors,
        metadata_blocked_runtime_behaviors=(
            metadata_registry.blocked_runtime_behaviors
        ),
        budget_blocked_runtime_behaviors=budget.blocked_runtime_behaviors,
        threshold_blocked_runtime_behaviors=threshold.blocked_runtime_behaviors,
        profile_blocked_runtime_behaviors=studio_cost.blocked_runtime_behaviors,
        consistency_blocked_runtime_behaviors=consistency.blocked_runtime_behaviors,
        metadata_only_declared=(
            contract.metadata_only
            and metadata.metadata_only
            and metadata_registry.metadata_only
            and budget.metadata_only
            and threshold.metadata_only
            and studio_cost.metadata_only
            and consistency.metadata_only
        ),
    )


def _unique_contract_cost_classes() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            contract.estimated_cost_metadata.relative_cost
            for contract in AGENT_CONTRACT_REGISTRY.contracts
        )
    )


AGENT_COST_TRACKING_FOUNDATION_PROFILES = tuple(
    _profile(agent_id) for agent_id in AGENT_CONTRACT_REGISTRY.agent_ids
)
AGENT_COST_TRACKING_FOUNDATION_REGISTRY = AgentCostTrackingFoundationRegistry(
    profiles=AGENT_COST_TRACKING_FOUNDATION_PROFILES,
    agent_ids=tuple(
        profile.agent_id for profile in AGENT_COST_TRACKING_FOUNDATION_PROFILES
    ),
    profile_count=len(AGENT_COST_TRACKING_FOUNDATION_PROFILES),
    cost_source_registries=_SOURCE_COST_REGISTRIES,
    cost_classes=_unique_contract_cost_classes(),
    budget_profile_ids=creative_exploration_budget_registry().budget_profile_ids,
    budget_postures=tuple(
        str(posture)
        for posture in creative_exploration_budget_registry().budget_postures
    ),
    cost_threshold_profile_ids=(
        cost_threshold_routing_registry().cost_threshold_profile_ids
    ),
    cost_threshold_bands=tuple(
        str(band) for band in cost_threshold_routing_registry().cost_bands
    ),
    cost_profile_ids=cost_profile_registry().cost_profile_ids,
    studio_cost_bands=tuple(str(band) for band in cost_profile_registry().cost_bands),
    consistency_family_ids=engine_contract_consistency_registry().family_ids,
    cost_dimensions=_COST_DIMENSIONS,
    passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
)
