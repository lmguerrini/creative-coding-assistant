"""Passive cross-family engine contract consistency metadata for V3.6."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.orchestration.artifact_engine_contracts import (
    ARTIFACT_ENGINE_CONTRACT_REGISTRY_SERIALIZATION_VERSION,
    ARTIFACT_ENGINE_CONTRACT_SERIALIZATION_VERSION,
    artifact_intelligence_engine_contracts,
)
from creative_coding_assistant.orchestration.evaluation_engine_contracts import (
    EVALUATION_ENGINE_CONTRACT_REGISTRY_SERIALIZATION_VERSION,
    EVALUATION_ENGINE_CONTRACT_SERIALIZATION_VERSION,
    evaluation_engine_contracts,
)
from creative_coding_assistant.orchestration.workstation_contracts import (
    WORKSTATION_ENGINE_CONTRACT_REGISTRY_SERIALIZATION_VERSION,
    WORKSTATION_ENGINE_CONTRACT_SERIALIZATION_VERSION,
    workstation_engine_contracts,
)

EngineContractConsistencyFamily = Literal[
    "artifact_intelligence",
    "creative_evaluation",
    "creative_workstation",
]
EngineContractItemKind = Literal["engine", "surface"]

ENGINE_CONTRACT_FAMILY_CONSISTENCY_SERIALIZATION_VERSION = (
    "engine_contract_family_consistency.v1"
)
ENGINE_CONTRACT_CONSISTENCY_REGISTRY_SERIALIZATION_VERSION = (
    "engine_contract_consistency_registry.v1"
)
ENGINE_CONTRACT_CONSISTENCY_REGISTRY_AUTHORITY_BOUNDARY = (
    "Engine contract consistency metadata normalizes existing V3 artifact, "
    "evaluation, and workstation contract registry surfaces for audit only; "
    "it does not change workflow graph order, generation behavior, provider "
    "or model routing, runtime selection, retries, artifact execution, "
    "preview execution, contract payload streaming, prompts, or generated "
    "output."
)

_SHARED_CONTRACT_CONCEPTS = (
    "identity",
    "category",
    "version",
    "authority_boundary",
    "required_inputs",
    "optional_inputs",
    "metadata_surface",
    "signal_surface",
    "quality_or_stability_signals",
    "dependency_surface",
    "cacheability",
    "cost_metadata",
    "latency_metadata",
    "serialization_version",
    "future_agent_hooks",
    "future_execution_hooks",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "workflow_graph_mutation",
    "generation_behavior_change",
    "provider_or_model_routing",
    "runtime_selection",
    "retry_or_refinement_triggering",
    "prompt_rendering_change",
    "contract_payload_streaming_change",
    "artifact_execution",
    "preview_execution",
    "generated_output_modification",
)


class EngineContractFamilyConsistencyProfile(BaseModel):
    """Metadata-only consistency profile for one contract registry family."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    family_id: EngineContractConsistencyFamily
    family_name: str = Field(min_length=1, max_length=140)
    registry_role: str = Field(min_length=1, max_length=120)
    registry_serialization_version: str = Field(min_length=1, max_length=80)
    contract_serialization_version: str = Field(min_length=1, max_length=80)
    contract_category: str = Field(min_length=1, max_length=80)
    contract_item_kind: EngineContractItemKind
    contract_ids: tuple[str, ...] = Field(min_length=1, max_length=32)
    contract_count: int = Field(ge=1, le=32)
    identity_fields: tuple[str, ...] = Field(min_length=4, max_length=4)
    input_fields: tuple[str, ...] = Field(min_length=2, max_length=2)
    output_fields: tuple[str, ...] = Field(min_length=2, max_length=2)
    quality_signal_fields: tuple[str, ...] = Field(min_length=2, max_length=5)
    relationship_fields: tuple[str, ...] = Field(min_length=2, max_length=2)
    execution_boundary_fields: tuple[str, ...] = Field(min_length=2, max_length=4)
    performance_fields: tuple[str, ...] = Field(min_length=3, max_length=5)
    future_hook_fields: tuple[str, ...] = Field(min_length=2, max_length=3)
    normalized_concepts: tuple[str, ...] = Field(min_length=16, max_length=16)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=1, max_length=12)
    serialization_version: Literal["engine_contract_family_consistency.v1"] = (
        ENGINE_CONTRACT_FAMILY_CONSISTENCY_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class EngineContractConsistencyRegistry(BaseModel):
    """Stable metadata registry for cross-family engine contract consistency."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["engine_contract_consistency_registry"] = (
        "engine_contract_consistency_registry"
    )
    serialization_version: Literal[
        "engine_contract_consistency_registry.v1"
    ] = ENGINE_CONTRACT_CONSISTENCY_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=ENGINE_CONTRACT_CONSISTENCY_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    families: tuple[EngineContractFamilyConsistencyProfile, ...] = Field(
        min_length=3,
        max_length=3,
    )
    family_ids: tuple[str, ...] = Field(min_length=3, max_length=3)
    family_count: int = Field(ge=3, le=3)
    shared_contract_concepts: tuple[str, ...] = Field(min_length=16, max_length=16)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=1, max_length=12)
    metadata_only: Literal[True] = True


def engine_contract_consistency_registry() -> EngineContractConsistencyRegistry:
    """Return static metadata that normalizes the V3 engine contract families."""

    return ENGINE_CONTRACT_CONSISTENCY_REGISTRY


def engine_contract_family_consistency_by_id(
    family_id: str,
) -> EngineContractFamilyConsistencyProfile | None:
    """Return one consistency profile without changing engine behavior."""

    for family in ENGINE_CONTRACT_FAMILY_CONSISTENCY_PROFILES:
        if family.family_id == family_id:
            return family
    return None


def _artifact_profile() -> EngineContractFamilyConsistencyProfile:
    registry = artifact_intelligence_engine_contracts()
    return EngineContractFamilyConsistencyProfile(
        family_id="artifact_intelligence",
        family_name="Artifact Intelligence Engine Contracts",
        registry_role=registry.role,
        registry_serialization_version=(
            ARTIFACT_ENGINE_CONTRACT_REGISTRY_SERIALIZATION_VERSION
        ),
        contract_serialization_version=ARTIFACT_ENGINE_CONTRACT_SERIALIZATION_VERSION,
        contract_category=registry.engine_category,
        contract_item_kind="engine",
        contract_ids=registry.engine_ids,
        contract_count=registry.contract_count,
        identity_fields=(
            "engine_id",
            "engine_name",
            "engine_version",
            "engine_category",
        ),
        input_fields=("required_inputs", "optional_inputs"),
        output_fields=("produced_metadata", "produced_signals"),
        quality_signal_fields=(
            "confidence_signals",
            "ambiguity_signals",
            "risk_signals",
            "escalation_candidates",
        ),
        relationship_fields=("downstream_dependencies", "upstream_dependencies"),
        execution_boundary_fields=(
            "authority_boundary",
            "future_execution_hooks",
            "future_agent_hooks",
        ),
        performance_fields=(
            "cacheability",
            "parallelization_support",
            "estimated_cost_metadata",
            "estimated_latency_metadata",
        ),
        future_hook_fields=("future_agent_hooks", "future_execution_hooks"),
        normalized_concepts=_SHARED_CONTRACT_CONCEPTS,
        blocked_runtime_behaviors=_BLOCKED_RUNTIME_BEHAVIORS,
    )


def _evaluation_profile() -> EngineContractFamilyConsistencyProfile:
    registry = evaluation_engine_contracts()
    return EngineContractFamilyConsistencyProfile(
        family_id="creative_evaluation",
        family_name="Creative Evaluation Engine Contracts",
        registry_role=registry.role,
        registry_serialization_version=(
            EVALUATION_ENGINE_CONTRACT_REGISTRY_SERIALIZATION_VERSION
        ),
        contract_serialization_version=EVALUATION_ENGINE_CONTRACT_SERIALIZATION_VERSION,
        contract_category=registry.engine_category,
        contract_item_kind="engine",
        contract_ids=registry.engine_ids,
        contract_count=registry.contract_count,
        identity_fields=(
            "engine_id",
            "engine_name",
            "engine_version",
            "engine_category",
        ),
        input_fields=("required_inputs", "optional_inputs"),
        output_fields=("produced_metadata", "produced_signals"),
        quality_signal_fields=(
            "confidence_signals",
            "ambiguity_signals",
            "risk_signals",
            "evidence_contract",
        ),
        relationship_fields=("downstream_dependencies", "upstream_dependencies"),
        execution_boundary_fields=(
            "authority_boundary",
            "future_execution_hooks",
            "future_agent_hooks",
        ),
        performance_fields=(
            "cacheability",
            "parallelization_support",
            "estimated_cost_metadata",
            "estimated_latency_metadata",
        ),
        future_hook_fields=("future_agent_hooks", "future_execution_hooks"),
        normalized_concepts=_SHARED_CONTRACT_CONCEPTS,
        blocked_runtime_behaviors=_BLOCKED_RUNTIME_BEHAVIORS,
    )


def _workstation_profile() -> EngineContractFamilyConsistencyProfile:
    registry = workstation_engine_contracts()
    return EngineContractFamilyConsistencyProfile(
        family_id="creative_workstation",
        family_name="Creative Workstation Engine Contracts",
        registry_role=registry.role,
        registry_serialization_version=(
            WORKSTATION_ENGINE_CONTRACT_REGISTRY_SERIALIZATION_VERSION
        ),
        contract_serialization_version=WORKSTATION_ENGINE_CONTRACT_SERIALIZATION_VERSION,
        contract_category=registry.surface_category,
        contract_item_kind="surface",
        contract_ids=registry.surface_ids,
        contract_count=registry.contract_count,
        identity_fields=(
            "surface_id",
            "surface_name",
            "surface_version",
            "surface_category",
        ),
        input_fields=("required_inputs", "optional_inputs"),
        output_fields=("exposed_metadata", "exposed_signals"),
        quality_signal_fields=("stability_signals", "missing_metadata_behavior"),
        relationship_fields=("downstream_consumers", "upstream_dependencies"),
        execution_boundary_fields=(
            "authority_boundary",
            "hydration_mode",
            "future_execution_hooks",
            "future_agent_hooks",
        ),
        performance_fields=(
            "cacheability",
            "hydration_mode",
            "estimated_cost_metadata",
            "estimated_latency_metadata",
        ),
        future_hook_fields=(
            "future_agent_hooks",
            "future_execution_hooks",
            "future_evolution_hooks",
        ),
        normalized_concepts=_SHARED_CONTRACT_CONCEPTS,
        blocked_runtime_behaviors=_BLOCKED_RUNTIME_BEHAVIORS,
    )


ENGINE_CONTRACT_FAMILY_CONSISTENCY_PROFILES = (
    _artifact_profile(),
    _evaluation_profile(),
    _workstation_profile(),
)

ENGINE_CONTRACT_CONSISTENCY_REGISTRY = EngineContractConsistencyRegistry(
    families=ENGINE_CONTRACT_FAMILY_CONSISTENCY_PROFILES,
    family_ids=tuple(
        family.family_id for family in ENGINE_CONTRACT_FAMILY_CONSISTENCY_PROFILES
    ),
    family_count=len(ENGINE_CONTRACT_FAMILY_CONSISTENCY_PROFILES),
    shared_contract_concepts=_SHARED_CONTRACT_CONCEPTS,
    blocked_runtime_behaviors=_BLOCKED_RUNTIME_BEHAVIORS,
)
