"""Passive V4.6 agent performance tracking foundation metadata."""

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
from creative_coding_assistant.orchestration.agent_parallel_scheduling import (
    ParallelSchedulingGroup,
    parallel_scheduling_group_for_agent,
    parallel_scheduling_registry,
)
from creative_coding_assistant.orchestration.engine_contract_consistency import (
    engine_contract_consistency_registry,
)
from creative_coding_assistant.orchestration.hybrid_agentic_workflow import (
    latency_threshold_routing_registry,
)
from creative_coding_assistant.orchestration.hybrid_studio import (
    cloud_model_registry,
    execution_simulator_registry,
    local_model_registry,
    model_profile_registry,
)

AgentPerformanceTrackingFoundationStage = Literal[
    "v4_6_agent_performance_tracking_foundation"
]
AgentPerformanceTrackingFoundationStatus = Literal["pass"]

AGENT_PERFORMANCE_TRACKING_FOUNDATION_PROFILE_SERIALIZATION_VERSION = (
    "agent_performance_tracking_foundation_profile.v1"
)
AGENT_PERFORMANCE_TRACKING_FOUNDATION_REGISTRY_SERIALIZATION_VERSION = (
    "agent_performance_tracking_foundation_registry.v1"
)
AGENT_PERFORMANCE_TRACKING_FOUNDATION_REGISTRY_AUTHORITY_BOUNDARY = (
    "V4.6 agent performance tracking foundation metadata describes passive "
    "agent latency declarations, latency basis text, blocking input "
    "references, parallelization support, scheduling group references, "
    "latency threshold references, model profile references, execution "
    "simulator references, local and cloud latency posture references, engine "
    "contract latency consistency references, and performance boundary flags "
    "only; it does not measure latency, execute simulations, run work in "
    "parallel, optimize execution, route by latency, select runtimes, select "
    "providers or models, call providers, invoke agents, control workflows, "
    "trigger retries, mutate prompts, write storage, or modify generated "
    "output."
)

_SOURCE_PERFORMANCE_REGISTRIES = (
    "agent_contract_registry",
    "agent_metadata_registry",
    "latency_threshold_routing_registry",
    "model_profile_registry",
    "execution_simulator_registry",
    "local_model_registry",
    "cloud_model_registry",
    "parallel_scheduling_registry",
    "engine_contract_consistency_registry",
)
_PERFORMANCE_DIMENSIONS = (
    "agent_identity",
    "role_identity",
    "contract_latency_class",
    "metadata_latency_class",
    "latency_basis",
    "blocking_inputs",
    "parallelization_support",
    "scheduling_group_reference",
    "latency_threshold_reference",
    "model_profile_reference",
    "execution_simulation_reference",
    "local_model_latency_posture",
    "cloud_model_latency_posture",
    "engine_contract_latency_consistency",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "performance_tracking_engine_blocked",
    "latency_measurement_blocked",
    "latency_threshold_evaluation_blocked",
    "latency_based_routing_blocked",
    "runtime_selection_blocked",
    "model_selection_blocked",
    "execution_simulation_blocked",
    "parallel_execution_blocked",
    "provider_model_routing_blocked",
    "workflow_timing_change_blocked",
    "generated_output_mutation_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "performance_tracking_engine",
    "latency_measurement",
    "latency_threshold_evaluation",
    "latency_based_routing",
    "runtime_selection",
    "model_selection",
    "execution_simulation",
    "parallel_task_execution",
    "execution_optimization",
    "provider_or_model_routing",
    "provider_execution",
    "agent_invocation",
    "workflow_control",
    "workflow_timing_change",
    "retry_or_refinement_triggering",
    "generated_output_modification",
)
_FOUNDATION_FINDINGS = (
    "contract_latency_metadata_confirmed",
    "agent_metadata_latency_alignment_confirmed",
    "parallel_scheduling_reference_confirmed",
    "latency_threshold_references_confirmed",
    "model_profile_references_confirmed",
    "execution_simulation_references_confirmed",
    "local_cloud_latency_postures_confirmed",
    "engine_contract_latency_consistency_confirmed",
    "runtime_performance_tracking_blocks_confirmed",
)


class AgentPerformanceTrackingFoundationProfile(BaseModel):
    """One passive performance tracking foundation profile for an agent."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    role_id: str = Field(min_length=1, max_length=80)
    performance_tracking_stage: AgentPerformanceTrackingFoundationStage = (
        "v4_6_agent_performance_tracking_foundation"
    )
    contract_serialization_version: str = Field(min_length=1, max_length=80)
    metadata_serialization_version: str = Field(min_length=1, max_length=80)
    contract_latency_class: str = Field(min_length=1, max_length=40)
    metadata_latency_class: str = Field(min_length=1, max_length=40)
    contract_latency_basis: str = Field(min_length=1, max_length=260)
    metadata_latency_basis: str = Field(min_length=1, max_length=260)
    contract_blocking_inputs: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=16,
    )
    contract_network_required_declared: Literal[False] = False
    metadata_parallelization_support: str = Field(min_length=1, max_length=80)
    scheduling_group_id: str = Field(min_length=1, max_length=120)
    scheduling_hint: str = Field(min_length=1, max_length=80)
    max_parallel_agents: int = Field(ge=1, le=6)
    latency_threshold_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    latency_bands: tuple[str, ...] = Field(min_length=4, max_length=4)
    latency_metadata_sources: tuple[str, ...] = Field(min_length=4, max_length=4)
    model_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    model_profile_kinds: tuple[str, ...] = Field(min_length=4, max_length=4)
    execution_simulation_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    simulation_scopes: tuple[str, ...] = Field(min_length=4, max_length=4)
    local_latency_postures: tuple[str, ...] = Field(min_length=4, max_length=4)
    cloud_latency_postures: tuple[str, ...] = Field(min_length=4, max_length=4)
    consistency_family_ids: tuple[str, ...] = Field(min_length=3, max_length=3)
    performance_source_registries: tuple[str, ...] = Field(
        min_length=9,
        max_length=9,
    )
    performance_dimensions: tuple[str, ...] = Field(min_length=14, max_length=14)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=11, max_length=11)
    foundation_findings: tuple[str, ...] = Field(min_length=9, max_length=9)
    missing_coverage_items: tuple[str, ...] = Field(
        default_factory=tuple, max_length=24
    )
    contract_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=16,
    )
    metadata_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=16,
    )
    latency_threshold_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    model_profile_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=14,
    )
    execution_simulator_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=14,
    )
    local_model_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    cloud_model_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    scheduling_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    consistency_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    foundation_status: AgentPerformanceTrackingFoundationStatus = "pass"
    metadata_only_declared: Literal[True] = True
    latency_metadata_alignment_present: Literal[True] = True
    scheduling_reference_present: Literal[True] = True
    latency_threshold_reference_present: Literal[True] = True
    execution_simulation_reference_present: Literal[True] = True
    model_latency_posture_reference_present: Literal[True] = True
    performance_tracking_implemented: Literal[False] = False
    latency_measurement_implemented: Literal[False] = False
    latency_threshold_evaluation_implemented: Literal[False] = False
    latency_based_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    execution_simulation_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    execution_optimization_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    external_provider_calls_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal[
        "agent_performance_tracking_foundation_profile.v1"
    ] = AGENT_PERFORMANCE_TRACKING_FOUNDATION_PROFILE_SERIALIZATION_VERSION
    metadata_only: Literal[True] = True


class AgentPerformanceTrackingFoundationRegistry(BaseModel):
    """Stable passive V4.6 registry for agent performance tracking metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_performance_tracking_foundation_registry"] = (
        "agent_performance_tracking_foundation_registry"
    )
    serialization_version: Literal[
        "agent_performance_tracking_foundation_registry.v1"
    ] = AGENT_PERFORMANCE_TRACKING_FOUNDATION_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=AGENT_PERFORMANCE_TRACKING_FOUNDATION_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1300,
    )
    performance_tracking_stage: AgentPerformanceTrackingFoundationStage = (
        "v4_6_agent_performance_tracking_foundation"
    )
    profiles: tuple[AgentPerformanceTrackingFoundationProfile, ...] = Field(
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
    source_latency_threshold_routing_registry: Literal[
        "latency_threshold_routing_registry"
    ] = "latency_threshold_routing_registry"
    source_model_profile_registry: Literal["model_profile_registry"] = (
        "model_profile_registry"
    )
    source_execution_simulator_registry: Literal["execution_simulator_registry"] = (
        "execution_simulator_registry"
    )
    source_local_model_registry: Literal["local_model_registry"] = (
        "local_model_registry"
    )
    source_cloud_model_registry: Literal["cloud_model_registry"] = (
        "cloud_model_registry"
    )
    source_parallel_scheduling_registry: Literal["parallel_scheduling_registry"] = (
        "parallel_scheduling_registry"
    )
    source_engine_contract_consistency_registry: Literal[
        "engine_contract_consistency_registry"
    ] = "engine_contract_consistency_registry"
    performance_source_registries: tuple[str, ...] = Field(
        min_length=9,
        max_length=9,
    )
    latency_classes: tuple[str, ...] = Field(min_length=1, max_length=4)
    scheduling_group_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    latency_threshold_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    latency_bands: tuple[str, ...] = Field(min_length=4, max_length=4)
    latency_metadata_sources: tuple[str, ...] = Field(min_length=4, max_length=4)
    model_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    model_profile_kinds: tuple[str, ...] = Field(min_length=4, max_length=4)
    execution_simulation_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    simulation_scopes: tuple[str, ...] = Field(min_length=4, max_length=4)
    local_latency_postures: tuple[str, ...] = Field(min_length=4, max_length=4)
    cloud_latency_postures: tuple[str, ...] = Field(min_length=4, max_length=4)
    consistency_family_ids: tuple[str, ...] = Field(min_length=3, max_length=3)
    performance_dimensions: tuple[str, ...] = Field(min_length=14, max_length=14)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=11, max_length=11)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    all_agents_covered: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    performance_tracking_engine_implemented: Literal[False] = False
    latency_measurement_implemented: Literal[False] = False
    latency_threshold_evaluation_implemented: Literal[False] = False
    latency_based_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    execution_simulation_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    execution_optimization_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    external_provider_calls_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
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
        if self.performance_source_registries != _SOURCE_PERFORMANCE_REGISTRIES:
            raise ValueError("performance_source_registries must match sources")
        if self.performance_dimensions != _PERFORMANCE_DIMENSIONS:
            raise ValueError("performance_dimensions must match foundation dimensions")
        if self.passive_boundary_flags != _PASSIVE_BOUNDARY_FLAGS:
            raise ValueError("passive_boundary_flags must match foundation flags")
        if any(
            (
                self.performance_tracking_engine_implemented,
                self.latency_measurement_implemented,
                self.latency_threshold_evaluation_implemented,
                self.latency_based_routing_implemented,
                self.runtime_selection_implemented,
                self.model_selection_implemented,
                self.execution_simulation_implemented,
                self.parallel_execution_implemented,
                self.execution_optimization_implemented,
                self.provider_model_routing_implemented,
                self.provider_execution_implemented,
                self.external_provider_calls_implemented,
                self.agent_invocation_implemented,
                self.workflow_control_implemented,
                self.workflow_timing_change_implemented,
                self.retry_triggering_implemented,
                self.prompt_mutation_implemented,
                self.persistent_storage_write_implemented,
                self.generated_output_mutation_implemented,
            )
        ):
            raise ValueError("performance tracking foundation must remain passive")

        known_group_ids = set(self.scheduling_group_ids)
        for profile in self.profiles:
            if profile.performance_tracking_stage != self.performance_tracking_stage:
                raise ValueError("performance_tracking_stage must match registry")
            if profile.contract_latency_class not in self.latency_classes:
                raise ValueError("contract_latency_class must be known")
            if profile.metadata_latency_class != profile.contract_latency_class:
                raise ValueError("metadata_latency_class must match contract")
            if profile.metadata_latency_basis != profile.contract_latency_basis:
                raise ValueError("metadata_latency_basis must match contract")
            if profile.scheduling_group_id not in known_group_ids:
                raise ValueError("scheduling_group_id must be known")
            if profile.latency_threshold_profile_ids != (
                self.latency_threshold_profile_ids
            ):
                raise ValueError("latency_threshold_profile_ids must match registry")
            if profile.latency_bands != self.latency_bands:
                raise ValueError("latency_bands must match registry")
            if profile.latency_metadata_sources != self.latency_metadata_sources:
                raise ValueError("latency_metadata_sources must match registry")
            if profile.model_profile_ids != self.model_profile_ids:
                raise ValueError("model_profile_ids must match registry")
            if profile.model_profile_kinds != self.model_profile_kinds:
                raise ValueError("model_profile_kinds must match registry")
            if profile.execution_simulation_profile_ids != (
                self.execution_simulation_profile_ids
            ):
                raise ValueError("execution_simulation_profile_ids must match registry")
            if profile.simulation_scopes != self.simulation_scopes:
                raise ValueError("simulation_scopes must match registry")
            if profile.local_latency_postures != self.local_latency_postures:
                raise ValueError("local_latency_postures must match registry")
            if profile.cloud_latency_postures != self.cloud_latency_postures:
                raise ValueError("cloud_latency_postures must match registry")
            if profile.consistency_family_ids != self.consistency_family_ids:
                raise ValueError("consistency_family_ids must match registry")
            if profile.performance_source_registries != (
                self.performance_source_registries
            ):
                raise ValueError("performance_source_registries must match registry")
            if profile.performance_dimensions != self.performance_dimensions:
                raise ValueError("performance_dimensions must match registry")
            if profile.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if profile.missing_coverage_items:
                raise ValueError("profiles must not contain missing coverage")
            if any(
                (
                    profile.contract_network_required_declared,
                    profile.performance_tracking_implemented,
                    profile.latency_measurement_implemented,
                    profile.latency_threshold_evaluation_implemented,
                    profile.latency_based_routing_implemented,
                    profile.runtime_selection_implemented,
                    profile.model_selection_implemented,
                    profile.execution_simulation_implemented,
                    profile.parallel_execution_implemented,
                    profile.execution_optimization_implemented,
                    profile.provider_model_routing_implemented,
                    profile.provider_execution_implemented,
                    profile.external_provider_calls_implemented,
                    profile.agent_invocation_implemented,
                    profile.workflow_control_implemented,
                    profile.workflow_timing_change_implemented,
                    profile.retry_triggering_implemented,
                    profile.prompt_mutation_implemented,
                    profile.persistent_storage_write_implemented,
                    profile.generated_output_mutation_implemented,
                )
            ):
                raise ValueError("performance tracking profiles must remain passive")
        return self


def agent_performance_tracking_foundation_registry() -> (
    AgentPerformanceTrackingFoundationRegistry
):
    """Return passive V4.6 agent performance tracking foundation metadata."""

    return AGENT_PERFORMANCE_TRACKING_FOUNDATION_REGISTRY


def agent_performance_tracking_profile_by_agent_id(
    agent_id: str,
    registry: AgentPerformanceTrackingFoundationRegistry | None = None,
) -> AgentPerformanceTrackingFoundationProfile | None:
    """Return one performance tracking foundation profile by agent id."""

    source_registry = registry or AGENT_PERFORMANCE_TRACKING_FOUNDATION_REGISTRY
    for profile in source_registry.profiles:
        if profile.agent_id == agent_id:
            return profile
    return None


def agent_performance_tracking_profiles_for_latency_class(
    latency_class: str,
    registry: AgentPerformanceTrackingFoundationRegistry | None = None,
) -> tuple[AgentPerformanceTrackingFoundationProfile, ...]:
    """Return passive performance profiles for one declared latency class."""

    source_registry = registry or AGENT_PERFORMANCE_TRACKING_FOUNDATION_REGISTRY
    normalized_latency_class = str(latency_class).strip()
    return tuple(
        profile
        for profile in source_registry.profiles
        if profile.contract_latency_class == normalized_latency_class
    )


def agent_performance_tracking_profiles_for_latency_threshold(
    latency_threshold_profile_id: str,
    registry: AgentPerformanceTrackingFoundationRegistry | None = None,
) -> tuple[AgentPerformanceTrackingFoundationProfile, ...]:
    """Return passive profiles that reference one latency threshold profile."""

    source_registry = registry or AGENT_PERFORMANCE_TRACKING_FOUNDATION_REGISTRY
    normalized_profile_id = str(latency_threshold_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.profiles
        if normalized_profile_id in profile.latency_threshold_profile_ids
    )


def _missing_coverage_items(
    *,
    contract: AgentContract,
    metadata: AgentOperationalMetadata,
    scheduling_group: ParallelSchedulingGroup,
) -> tuple[str, ...]:
    metadata_registry = agent_metadata_registry()
    latency = latency_threshold_routing_registry()
    models = model_profile_registry()
    simulator = execution_simulator_registry()
    local_models = local_model_registry()
    cloud_models = cloud_model_registry()
    scheduling = parallel_scheduling_registry()
    consistency = engine_contract_consistency_registry()
    missing: list[str] = []
    if metadata.estimated_latency_class != (
        contract.estimated_latency_metadata.relative_latency
    ):
        missing.append("agent_metadata_latency_class_alignment_missing")
    if metadata.estimated_latency_basis != (
        contract.estimated_latency_metadata.latency_basis
    ):
        missing.append("agent_metadata_latency_basis_alignment_missing")
    if contract.estimated_latency_metadata.network_required:
        missing.append("network_required_boundary_missing")
    if not contract.estimated_latency_metadata.latency_basis:
        missing.append("latency_basis_reference_missing")
    if not scheduling_group.group_id:
        missing.append("parallel_scheduling_reference_missing")
    if scheduling_group.group_id not in scheduling.group_ids:
        missing.append("parallel_scheduling_group_unknown")
    if not latency.latency_threshold_profile_ids:
        missing.append("latency_threshold_reference_missing")
    if not latency.latency_metadata_sources:
        missing.append("latency_metadata_source_reference_missing")
    if not models.model_profile_ids:
        missing.append("model_profile_reference_missing")
    if not simulator.execution_simulation_profile_ids:
        missing.append("execution_simulation_reference_missing")
    if not local_models.surface_ids:
        missing.append("local_model_latency_posture_missing")
    if not cloud_models.surface_ids:
        missing.append("cloud_model_latency_posture_missing")
    if "latency_metadata" not in consistency.shared_contract_concepts:
        missing.append("engine_contract_latency_consistency_missing")
    if not all(
        (
            contract.metadata_only,
            metadata.metadata_only,
            metadata_registry.metadata_only,
            latency.metadata_only,
            models.metadata_only,
            simulator.metadata_only,
            local_models.metadata_only,
            cloud_models.metadata_only,
            scheduling.metadata_only,
            consistency.metadata_only,
        )
    ):
        missing.append("metadata_only_declaration_missing")
    if "runtime_selection" not in contract.blocked_runtime_behaviors:
        missing.append("contract_runtime_selection_block_missing")
    if "cost_or_latency_routing" not in metadata_registry.blocked_runtime_behaviors:
        missing.append("metadata_latency_routing_block_missing")
    if "latency_based_routing" not in latency.blocked_runtime_behaviors:
        missing.append("latency_routing_block_missing")
    if "runtime_selection" not in latency.blocked_runtime_behaviors:
        missing.append("latency_runtime_selection_block_missing")
    if "execution_optimization" not in models.blocked_runtime_behaviors:
        missing.append("model_execution_optimization_block_missing")
    if "model_selection" not in models.blocked_runtime_behaviors:
        missing.append("model_selection_block_missing")
    if "simulation_runtime_execution" not in simulator.blocked_runtime_behaviors:
        missing.append("execution_simulation_block_missing")
    if "local_provider_execution" not in local_models.blocked_runtime_behaviors:
        missing.append("local_provider_execution_block_missing")
    if "cloud_provider_execution" not in cloud_models.blocked_runtime_behaviors:
        missing.append("cloud_provider_execution_block_missing")
    if "pricing_or_latency_optimization" not in cloud_models.blocked_runtime_behaviors:
        missing.append("cloud_latency_optimization_block_missing")
    if "parallel_task_execution" not in scheduling.blocked_runtime_behaviors:
        missing.append("parallel_execution_block_missing")
    if "runtime_selection" not in consistency.blocked_runtime_behaviors:
        missing.append("consistency_runtime_selection_block_missing")
    if latency.latency_based_routing_implemented:
        missing.append("latency_routing_enabled")
    if latency.runtime_selection_implemented:
        missing.append("latency_runtime_selection_enabled")
    if models.model_selection_implemented:
        missing.append("model_selection_enabled")
    if models.execution_optimization_implemented:
        missing.append("model_execution_optimization_enabled")
    if simulator.simulation_runtime_execution_implemented:
        missing.append("execution_simulation_enabled")
    if local_models.local_provider_execution_implemented:
        missing.append("local_provider_execution_enabled")
    if cloud_models.cloud_provider_execution_implemented:
        missing.append("cloud_provider_execution_enabled")
    if cloud_models.pricing_latency_optimization_implemented:
        missing.append("cloud_latency_optimization_enabled")
    if scheduling.parallel_execution_implemented:
        missing.append("parallel_execution_enabled")
    if scheduling.async_behavior_changed:
        missing.append("async_behavior_changed")
    return tuple(missing)


def _profile(agent_id: str) -> AgentPerformanceTrackingFoundationProfile:
    contract = agent_contract_by_id(agent_id, AGENT_CONTRACT_REGISTRY)
    metadata = agent_metadata_by_agent_id(agent_id)
    scheduling_group = parallel_scheduling_group_for_agent(
        agent_id,
        parallel_scheduling_registry(),
    )
    metadata_registry = agent_metadata_registry()
    latency = latency_threshold_routing_registry()
    models = model_profile_registry()
    simulator = execution_simulator_registry()
    local_models = local_model_registry()
    cloud_models = cloud_model_registry()
    consistency = engine_contract_consistency_registry()
    if contract is None:
        raise ValueError(f"missing agent contract for {agent_id}")
    if metadata is None:
        raise ValueError(f"missing agent metadata for {agent_id}")
    if scheduling_group is None:
        raise ValueError(f"missing scheduling group for {agent_id}")

    return AgentPerformanceTrackingFoundationProfile(
        agent_id=agent_id,
        role_id=contract.role_id,
        contract_serialization_version=contract.serialization_version,
        metadata_serialization_version=metadata.serialization_version,
        contract_latency_class=contract.estimated_latency_metadata.relative_latency,
        metadata_latency_class=metadata.estimated_latency_class,
        contract_latency_basis=contract.estimated_latency_metadata.latency_basis,
        metadata_latency_basis=metadata.estimated_latency_basis,
        contract_blocking_inputs=contract.estimated_latency_metadata.blocking_inputs,
        contract_network_required_declared=(
            contract.estimated_latency_metadata.network_required
        ),
        metadata_parallelization_support=metadata.parallelization_support,
        scheduling_group_id=scheduling_group.group_id,
        scheduling_hint=scheduling_group.scheduling_hint,
        max_parallel_agents=scheduling_group.max_parallel_agents,
        latency_threshold_profile_ids=latency.latency_threshold_profile_ids,
        latency_bands=tuple(str(band) for band in latency.latency_bands),
        latency_metadata_sources=latency.latency_metadata_sources,
        model_profile_ids=models.model_profile_ids,
        model_profile_kinds=tuple(str(kind) for kind in models.model_profile_kinds),
        execution_simulation_profile_ids=(simulator.execution_simulation_profile_ids),
        simulation_scopes=tuple(str(scope) for scope in simulator.simulation_scopes),
        local_latency_postures=tuple(
            str(surface.latency_posture) for surface in local_models.model_surfaces
        ),
        cloud_latency_postures=tuple(
            str(surface.latency_posture) for surface in cloud_models.model_surfaces
        ),
        consistency_family_ids=consistency.family_ids,
        performance_source_registries=_SOURCE_PERFORMANCE_REGISTRIES,
        performance_dimensions=_PERFORMANCE_DIMENSIONS,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        foundation_findings=_FOUNDATION_FINDINGS,
        missing_coverage_items=_missing_coverage_items(
            contract=contract,
            metadata=metadata,
            scheduling_group=scheduling_group,
        ),
        contract_blocked_runtime_behaviors=contract.blocked_runtime_behaviors,
        metadata_blocked_runtime_behaviors=(
            metadata_registry.blocked_runtime_behaviors
        ),
        latency_threshold_blocked_runtime_behaviors=(latency.blocked_runtime_behaviors),
        model_profile_blocked_runtime_behaviors=models.blocked_runtime_behaviors,
        execution_simulator_blocked_runtime_behaviors=(
            simulator.blocked_runtime_behaviors
        ),
        local_model_blocked_runtime_behaviors=local_models.blocked_runtime_behaviors,
        cloud_model_blocked_runtime_behaviors=cloud_models.blocked_runtime_behaviors,
        scheduling_blocked_runtime_behaviors=scheduling_group.blocked_runtime_behaviors,
        consistency_blocked_runtime_behaviors=consistency.blocked_runtime_behaviors,
        metadata_only_declared=(
            contract.metadata_only
            and metadata.metadata_only
            and metadata_registry.metadata_only
            and latency.metadata_only
            and models.metadata_only
            and simulator.metadata_only
            and local_models.metadata_only
            and cloud_models.metadata_only
            and scheduling_group.metadata_only
            and consistency.metadata_only
        ),
    )


def _unique_contract_latency_classes() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            contract.estimated_latency_metadata.relative_latency
            for contract in AGENT_CONTRACT_REGISTRY.contracts
        )
    )


AGENT_PERFORMANCE_TRACKING_FOUNDATION_PROFILES = tuple(
    _profile(agent_id) for agent_id in AGENT_CONTRACT_REGISTRY.agent_ids
)
AGENT_PERFORMANCE_TRACKING_FOUNDATION_REGISTRY = (
    AgentPerformanceTrackingFoundationRegistry(
        profiles=AGENT_PERFORMANCE_TRACKING_FOUNDATION_PROFILES,
        agent_ids=tuple(
            profile.agent_id
            for profile in AGENT_PERFORMANCE_TRACKING_FOUNDATION_PROFILES
        ),
        profile_count=len(AGENT_PERFORMANCE_TRACKING_FOUNDATION_PROFILES),
        performance_source_registries=_SOURCE_PERFORMANCE_REGISTRIES,
        latency_classes=_unique_contract_latency_classes(),
        scheduling_group_ids=parallel_scheduling_registry().group_ids,
        latency_threshold_profile_ids=(
            latency_threshold_routing_registry().latency_threshold_profile_ids
        ),
        latency_bands=tuple(
            str(band) for band in latency_threshold_routing_registry().latency_bands
        ),
        latency_metadata_sources=(
            latency_threshold_routing_registry().latency_metadata_sources
        ),
        model_profile_ids=model_profile_registry().model_profile_ids,
        model_profile_kinds=tuple(
            str(kind) for kind in model_profile_registry().model_profile_kinds
        ),
        execution_simulation_profile_ids=(
            execution_simulator_registry().execution_simulation_profile_ids
        ),
        simulation_scopes=tuple(
            str(scope) for scope in execution_simulator_registry().simulation_scopes
        ),
        local_latency_postures=tuple(
            str(surface.latency_posture)
            for surface in local_model_registry().model_surfaces
        ),
        cloud_latency_postures=tuple(
            str(surface.latency_posture)
            for surface in cloud_model_registry().model_surfaces
        ),
        consistency_family_ids=engine_contract_consistency_registry().family_ids,
        performance_dimensions=_PERFORMANCE_DIMENSIONS,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
    )
)
