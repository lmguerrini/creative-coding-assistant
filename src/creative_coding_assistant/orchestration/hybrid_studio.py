"""Passive V4.4 Hybrid Studio metadata surfaces."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_capabilities import (
    AGENT_CAPABILITY_REGISTRY,
)
from creative_coding_assistant.orchestration.agent_identities import (
    AGENT_IDENTITY_REGISTRY,
)
from creative_coding_assistant.orchestration.agent_memory_contracts import (
    AGENT_MEMORY_CONTRACT_REGISTRY,
)
from creative_coding_assistant.orchestration.agent_metadata import (
    AGENT_METADATA_REGISTRY,
)
from creative_coding_assistant.orchestration.agent_roles import AGENT_ROLE_REGISTRY
from creative_coding_assistant.orchestration.hybrid_agentic_workflow import (
    COST_THRESHOLD_ROUTING_REGISTRY,
    QUALITY_ESCALATION_REGISTRY,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.shared_context_views import (
    SHARED_CONTEXT_VIEW_REGISTRY,
)
from creative_coding_assistant.orchestration.workflow_agent_handoff import (
    WORKFLOW_AGENT_HANDOFF_REGISTRY,
)

LocalModelRuntimeKind = Literal[
    "ollama",
    "lm_studio",
    "llama_cpp",
    "local_transformers",
]
LocalModelExecutionSurface = Literal[
    "localhost_http",
    "local_process_binding",
    "user_managed_runtime",
]
LocalModelCapabilityBand = Literal[
    "general_chat",
    "creative_reasoning",
    "code_assistance",
    "multimodal_inspection",
]
LocalModelContextWindowBand = Literal["small", "medium", "large", "model_declared"]
LocalModelLatencyPosture = Literal[
    "hardware_dependent",
    "interactive_when_loaded",
    "startup_sensitive",
]

LOCAL_MODEL_SURFACE_SERIALIZATION_VERSION = "local_model_surface.v1"
LOCAL_MODEL_REGISTRY_SERIALIZATION_VERSION = "local_model_registry.v1"
LOCAL_MODEL_REGISTRY_AUTHORITY_BOUNDARY = (
    "Local model metadata describes passive candidate surfaces for V4.4 Hybrid "
    "Studio inspection only; it does not discover installed models, start "
    "local runtimes, execute local providers, route providers or models, "
    "select models automatically, call external providers, trigger retries, "
    "write replay storage, or modify generated output."
)

_LOCAL_MODEL_SOURCE_REGISTRIES = (
    "settings_generation_provider_config",
    "generation_provider_factory",
    "generation_provider_contract",
    "agent_routing_registry",
    "hybrid_agentic_workflow_registry",
)

_LOCAL_MODEL_STUDIO_SURFACES = (
    "local_model_catalog",
    "local_model_readiness_inspector",
    "provider_selection_metadata",
    "execution_simulator_metadata",
    "local_cloud_comparison_metadata",
)

_LOCAL_MODEL_OBSERVABILITY_SURFACES = (
    "surface_id",
    "runtime_kind",
    "execution_surface",
    "route_applicability",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_LOCAL_MODEL_BLOCKED_RUNTIME_BEHAVIORS = (
    "local_runtime_discovery",
    "local_provider_execution",
    "provider_or_model_routing",
    "automatic_model_selection",
    "external_provider_calling",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_replay_storage",
    "generated_output_modification",
)


class LocalModelSurface(BaseModel):
    """Inspectable metadata for one candidate local model surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    surface_id: str = Field(min_length=1, max_length=80)
    surface_name: str = Field(min_length=1, max_length=120)
    runtime_kind: LocalModelRuntimeKind
    execution_surface: LocalModelExecutionSurface
    capability_band: LocalModelCapabilityBand
    context_window_band: LocalModelContextWindowBand
    latency_posture: LocalModelLatencyPosture
    cost_posture: Literal["local_hardware_only"] = "local_hardware_only"
    privacy_posture: Literal["local_operator_boundary"] = "local_operator_boundary"
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    supported_payloads: tuple[str, ...] = Field(min_length=1, max_length=8)
    readiness_signals: tuple[str, ...] = Field(min_length=1, max_length=8)
    studio_surface_refs: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    authority_boundary: str = Field(
        default=LOCAL_MODEL_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_LOCAL_MODEL_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    local_runtime_discovery_implemented: Literal[False] = False
    local_provider_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    external_provider_calls_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    serialization_version: Literal["local_model_surface.v1"] = (
        LOCAL_MODEL_SURFACE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class LocalModelRegistry(BaseModel):
    """Stable passive registry for V4.4 Hybrid Studio local model metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["local_model_registry"] = "local_model_registry"
    serialization_version: Literal["local_model_registry.v1"] = (
        LOCAL_MODEL_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=LOCAL_MODEL_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    model_surfaces: tuple[LocalModelSurface, ...] = Field(
        min_length=4,
        max_length=4,
    )
    surface_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    runtime_kinds: tuple[LocalModelRuntimeKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    studio_surface_refs: tuple[str, ...] = Field(min_length=5, max_length=5)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_LOCAL_MODEL_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    local_runtime_discovery_implemented: Literal[False] = False
    local_provider_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    external_provider_calls_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_surfaces(self) -> Self:
        derived_surface_ids = tuple(
            surface.surface_id for surface in self.model_surfaces
        )
        if len(set(derived_surface_ids)) != len(derived_surface_ids):
            raise ValueError("surface_ids must be unique")
        if self.surface_ids != derived_surface_ids:
            raise ValueError("surface_ids must match model_surfaces")
        if self.profile_count != len(self.model_surfaces):
            raise ValueError("profile_count must match model_surfaces")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")

        known_routes = set(self.route_names)
        known_studio_surfaces = set(self.studio_surface_refs)
        profile_sources = {
            source_registry
            for surface in self.model_surfaces
            for source_registry in surface.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match local model sources")

        for surface in self.model_surfaces:
            if surface.source_registries != self.source_registries:
                raise ValueError("surface source_registries must match registry")
            if surface.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(surface.studio_surface_refs).issubset(known_studio_surfaces):
                raise ValueError("studio_surface_refs must be known registry surfaces")
            if not set(surface.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
        return self


def local_model_registry() -> LocalModelRegistry:
    """Return passive V4.4 Hybrid Studio local model metadata."""

    return LOCAL_MODEL_REGISTRY


def local_model_surface_by_id(
    surface_id: str,
    registry: LocalModelRegistry | None = None,
) -> LocalModelSurface | None:
    """Return one local model surface without routing or executing it."""

    source_registry = registry or LOCAL_MODEL_REGISTRY
    for surface in source_registry.model_surfaces:
        if surface.surface_id == surface_id:
            return surface
    return None


def local_model_surfaces_for_runtime(
    runtime_kind: LocalModelRuntimeKind | str,
    registry: LocalModelRegistry | None = None,
) -> tuple[LocalModelSurface, ...]:
    """Return passive local model surfaces for a runtime kind."""

    source_registry = registry or LOCAL_MODEL_REGISTRY
    runtime_value = str(runtime_kind).strip()
    return tuple(
        surface
        for surface in source_registry.model_surfaces
        if surface.runtime_kind == runtime_value
    )


def _surface(
    *,
    surface_id: str,
    surface_name: str,
    runtime_kind: LocalModelRuntimeKind,
    execution_surface: LocalModelExecutionSurface,
    capability_band: LocalModelCapabilityBand,
    context_window_band: LocalModelContextWindowBand,
    latency_posture: LocalModelLatencyPosture,
    route_applicability: tuple[RouteName, ...],
    supported_payloads: tuple[str, ...],
    readiness_signals: tuple[str, ...],
    studio_surface_refs: tuple[str, ...],
) -> LocalModelSurface:
    return LocalModelSurface(
        surface_id=surface_id,
        surface_name=surface_name,
        runtime_kind=runtime_kind,
        execution_surface=execution_surface,
        capability_band=capability_band,
        context_window_band=context_window_band,
        latency_posture=latency_posture,
        route_applicability=route_applicability,
        supported_payloads=supported_payloads,
        readiness_signals=readiness_signals,
        studio_surface_refs=studio_surface_refs,
        source_registries=_LOCAL_MODEL_SOURCE_REGISTRIES,
        observability_surfaces=_LOCAL_MODEL_OBSERVABILITY_SURFACES,
    )


LOCAL_MODEL_SURFACES = (
    _surface(
        surface_id="ollama_chat_surface",
        surface_name="Ollama Chat Surface",
        runtime_kind="ollama",
        execution_surface="localhost_http",
        capability_band="general_chat",
        context_window_band="model_declared",
        latency_posture="interactive_when_loaded",
        route_applicability=tuple(RouteName),
        supported_payloads=(
            "text_prompt",
            "code_context",
            "retrieval_context_metadata",
        ),
        readiness_signals=(
            "runtime_process_user_managed",
            "model_identifier_user_supplied",
            "localhost_endpoint_metadata_only",
        ),
        studio_surface_refs=(
            "local_model_catalog",
            "local_model_readiness_inspector",
            "provider_selection_metadata",
        ),
    ),
    _surface(
        surface_id="lm_studio_chat_surface",
        surface_name="LM Studio Chat Surface",
        runtime_kind="lm_studio",
        execution_surface="localhost_http",
        capability_band="creative_reasoning",
        context_window_band="large",
        latency_posture="interactive_when_loaded",
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        supported_payloads=(
            "text_prompt",
            "creative_brief_metadata",
            "artifact_context_metadata",
        ),
        readiness_signals=(
            "runtime_process_user_managed",
            "chat_completion_shape_metadata",
            "localhost_endpoint_metadata_only",
        ),
        studio_surface_refs=(
            "local_model_catalog",
            "provider_selection_metadata",
            "local_cloud_comparison_metadata",
        ),
    ),
    _surface(
        surface_id="llama_cpp_completion_surface",
        surface_name="llama.cpp Completion Surface",
        runtime_kind="llama_cpp",
        execution_surface="local_process_binding",
        capability_band="code_assistance",
        context_window_band="medium",
        latency_posture="hardware_dependent",
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DEBUG,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        supported_payloads=(
            "text_prompt",
            "code_context",
            "runtime_diagnostic_metadata",
        ),
        readiness_signals=(
            "binary_or_binding_user_managed",
            "model_file_path_metadata_only",
            "hardware_fit_metadata_only",
        ),
        studio_surface_refs=(
            "local_model_readiness_inspector",
            "execution_simulator_metadata",
            "local_cloud_comparison_metadata",
        ),
    ),
    _surface(
        surface_id="local_transformers_multimodal_surface",
        surface_name="Local Transformers Multimodal Surface",
        runtime_kind="local_transformers",
        execution_surface="user_managed_runtime",
        capability_band="multimodal_inspection",
        context_window_band="small",
        latency_posture="startup_sensitive",
        route_applicability=(
            RouteName.EXPLAIN,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        supported_payloads=(
            "text_prompt",
            "image_reference_metadata",
            "multimodal_attachment_metadata",
        ),
        readiness_signals=(
            "runtime_environment_user_managed",
            "model_checkpoint_metadata_only",
            "device_memory_metadata_only",
        ),
        studio_surface_refs=(
            "local_model_catalog",
            "local_model_readiness_inspector",
            "execution_simulator_metadata",
            "local_cloud_comparison_metadata",
        ),
    ),
)

LOCAL_MODEL_REGISTRY = LocalModelRegistry(
    model_surfaces=LOCAL_MODEL_SURFACES,
    surface_ids=tuple(surface.surface_id for surface in LOCAL_MODEL_SURFACES),
    runtime_kinds=tuple(surface.runtime_kind for surface in LOCAL_MODEL_SURFACES),
    route_names=tuple(RouteName),
    profile_count=len(LOCAL_MODEL_SURFACES),
    source_registries=_LOCAL_MODEL_SOURCE_REGISTRIES,
    studio_surface_refs=_LOCAL_MODEL_STUDIO_SURFACES,
    observability_surfaces=_LOCAL_MODEL_OBSERVABILITY_SURFACES,
)

CloudModelProviderKind = Literal["openai"]
CloudModelCapabilityBand = Literal[
    "assistant_generation",
    "retrieval_embedding",
    "evaluation_scoring",
    "provider_response_metadata",
]
CloudModelConfigurationSource = Literal[
    "openai_model",
    "openai_embedding_model",
    "eval_ragas_model",
    "provider_response_model",
]
CloudModelLatencyPosture = Literal[
    "network_dependent",
    "provider_reported",
    "batch_sensitive",
]

CLOUD_MODEL_SURFACE_SERIALIZATION_VERSION = "cloud_model_surface.v1"
CLOUD_MODEL_REGISTRY_SERIALIZATION_VERSION = "cloud_model_registry.v1"
CLOUD_MODEL_REGISTRY_AUTHORITY_BOUNDARY = (
    "Cloud model metadata describes passive candidate cloud surfaces for V4.4 "
    "Hybrid Studio inspection only; it does not call cloud providers, change "
    "provider or model routing, select models automatically, optimize by cost "
    "or latency, trigger retries, mutate prompts, write replay storage, or "
    "modify generated output."
)

_CLOUD_MODEL_SOURCE_REGISTRIES = (
    "settings_generation_provider_config",
    "generation_provider_factory",
    "generation_provider_contract",
    "openai_provider_adapter",
    "provider_telemetry_metadata",
    "local_model_registry",
)

_CLOUD_MODEL_STUDIO_SURFACES = (
    "cloud_model_catalog",
    "cloud_model_readiness_inspector",
    "provider_selection_metadata",
    "execution_simulator_metadata",
    "local_cloud_comparison_metadata",
)

_CLOUD_MODEL_OBSERVABILITY_SURFACES = (
    "surface_id",
    "provider_kind",
    "configuration_source",
    "route_applicability",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_CLOUD_MODEL_BLOCKED_RUNTIME_BEHAVIORS = (
    "cloud_provider_execution",
    "provider_or_model_routing",
    "automatic_model_selection",
    "external_provider_calling",
    "pricing_or_latency_optimization",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_replay_storage",
    "generated_output_modification",
)


class CloudModelSurface(BaseModel):
    """Inspectable metadata for one candidate cloud model surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    surface_id: str = Field(min_length=1, max_length=80)
    surface_name: str = Field(min_length=1, max_length=120)
    provider_kind: CloudModelProviderKind
    capability_band: CloudModelCapabilityBand
    configuration_source: CloudModelConfigurationSource
    latency_posture: CloudModelLatencyPosture
    cost_posture: Literal["provider_metered"] = "provider_metered"
    privacy_posture: Literal["external_provider_boundary"] = (
        "external_provider_boundary"
    )
    route_applicability: tuple[RouteName, ...] = Field(
        default_factory=tuple, max_length=6
    )
    supported_payloads: tuple[str, ...] = Field(min_length=1, max_length=8)
    readiness_signals: tuple[str, ...] = Field(min_length=1, max_length=8)
    studio_surface_refs: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    authority_boundary: str = Field(
        default=CLOUD_MODEL_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CLOUD_MODEL_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    cloud_provider_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    external_provider_calls_implemented: Literal[False] = False
    pricing_latency_optimization_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    serialization_version: Literal["cloud_model_surface.v1"] = (
        CLOUD_MODEL_SURFACE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class CloudModelRegistry(BaseModel):
    """Stable passive registry for V4.4 Hybrid Studio cloud model metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cloud_model_registry"] = "cloud_model_registry"
    serialization_version: Literal["cloud_model_registry.v1"] = (
        CLOUD_MODEL_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CLOUD_MODEL_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    model_surfaces: tuple[CloudModelSurface, ...] = Field(
        min_length=4,
        max_length=4,
    )
    surface_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    provider_kinds: tuple[CloudModelProviderKind, ...] = Field(
        min_length=1,
        max_length=4,
    )
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    studio_surface_refs: tuple[str, ...] = Field(min_length=5, max_length=5)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CLOUD_MODEL_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    cloud_provider_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    external_provider_calls_implemented: Literal[False] = False
    pricing_latency_optimization_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_surfaces(self) -> Self:
        derived_surface_ids = tuple(
            surface.surface_id for surface in self.model_surfaces
        )
        if len(set(derived_surface_ids)) != len(derived_surface_ids):
            raise ValueError("surface_ids must be unique")
        if self.surface_ids != derived_surface_ids:
            raise ValueError("surface_ids must match model_surfaces")
        if self.profile_count != len(self.model_surfaces):
            raise ValueError("profile_count must match model_surfaces")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.provider_kinds != tuple(
            dict.fromkeys(surface.provider_kind for surface in self.model_surfaces)
        ):
            raise ValueError("provider_kinds must match model_surfaces")

        known_routes = set(self.route_names)
        known_studio_surfaces = set(self.studio_surface_refs)
        profile_sources = {
            source_registry
            for surface in self.model_surfaces
            for source_registry in surface.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match cloud model sources")

        for surface in self.model_surfaces:
            if surface.source_registries != self.source_registries:
                raise ValueError("surface source_registries must match registry")
            if surface.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(surface.studio_surface_refs).issubset(known_studio_surfaces):
                raise ValueError("studio_surface_refs must be known registry surfaces")
            if not set(surface.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
        return self


def cloud_model_registry() -> CloudModelRegistry:
    """Return passive V4.4 Hybrid Studio cloud model metadata."""

    return CLOUD_MODEL_REGISTRY


def cloud_model_surface_by_id(
    surface_id: str,
    registry: CloudModelRegistry | None = None,
) -> CloudModelSurface | None:
    """Return one cloud model surface without routing or executing it."""

    source_registry = registry or CLOUD_MODEL_REGISTRY
    for surface in source_registry.model_surfaces:
        if surface.surface_id == surface_id:
            return surface
    return None


def cloud_model_surfaces_for_provider(
    provider_kind: CloudModelProviderKind | str,
    registry: CloudModelRegistry | None = None,
) -> tuple[CloudModelSurface, ...]:
    """Return passive cloud model surfaces for a provider kind."""

    source_registry = registry or CLOUD_MODEL_REGISTRY
    provider_value = str(provider_kind).strip()
    return tuple(
        surface
        for surface in source_registry.model_surfaces
        if surface.provider_kind == provider_value
    )


def _cloud_surface(
    *,
    surface_id: str,
    surface_name: str,
    provider_kind: CloudModelProviderKind,
    capability_band: CloudModelCapabilityBand,
    configuration_source: CloudModelConfigurationSource,
    latency_posture: CloudModelLatencyPosture,
    supported_payloads: tuple[str, ...],
    readiness_signals: tuple[str, ...],
    studio_surface_refs: tuple[str, ...],
    route_applicability: tuple[RouteName, ...] = (),
) -> CloudModelSurface:
    return CloudModelSurface(
        surface_id=surface_id,
        surface_name=surface_name,
        provider_kind=provider_kind,
        capability_band=capability_band,
        configuration_source=configuration_source,
        latency_posture=latency_posture,
        route_applicability=route_applicability,
        supported_payloads=supported_payloads,
        readiness_signals=readiness_signals,
        studio_surface_refs=studio_surface_refs,
        source_registries=_CLOUD_MODEL_SOURCE_REGISTRIES,
        observability_surfaces=_CLOUD_MODEL_OBSERVABILITY_SURFACES,
    )


CLOUD_MODEL_SURFACES = (
    _cloud_surface(
        surface_id="openai_generation_model_surface",
        surface_name="OpenAI Generation Model Surface",
        provider_kind="openai",
        capability_band="assistant_generation",
        configuration_source="openai_model",
        latency_posture="network_dependent",
        route_applicability=tuple(RouteName),
        supported_payloads=(
            "provider_ready_messages",
            "rendered_prompt_sections",
            "retrieval_context_metadata",
        ),
        readiness_signals=(
            "default_generation_provider_configured",
            "openai_model_setting_available",
            "api_key_required_at_execution_time",
        ),
        studio_surface_refs=(
            "cloud_model_catalog",
            "cloud_model_readiness_inspector",
            "provider_selection_metadata",
        ),
    ),
    _cloud_surface(
        surface_id="openai_embedding_model_surface",
        surface_name="OpenAI Embedding Model Surface",
        provider_kind="openai",
        capability_band="retrieval_embedding",
        configuration_source="openai_embedding_model",
        latency_posture="batch_sensitive",
        supported_payloads=(
            "knowledge_base_chunks",
            "retrieval_query_text",
            "embedding_metadata",
        ),
        readiness_signals=(
            "openai_embedding_model_setting_available",
            "retrieval_pipeline_metadata_only",
            "api_key_required_at_execution_time",
        ),
        studio_surface_refs=(
            "cloud_model_catalog",
            "cloud_model_readiness_inspector",
            "local_cloud_comparison_metadata",
        ),
    ),
    _cloud_surface(
        surface_id="ragas_evaluator_model_surface",
        surface_name="RAGAs Evaluator Model Surface",
        provider_kind="openai",
        capability_band="evaluation_scoring",
        configuration_source="eval_ragas_model",
        latency_posture="batch_sensitive",
        route_applicability=(RouteName.REVIEW,),
        supported_payloads=(
            "evaluation_dataset_rows",
            "provider_metadata_snapshots",
            "retrieval_grounding_metadata",
        ),
        readiness_signals=(
            "eval_ragas_model_setting_available",
            "provider_calls_require_explicit_opt_in",
            "dry_run_boundary_available",
        ),
        studio_surface_refs=(
            "cloud_model_readiness_inspector",
            "execution_simulator_metadata",
            "local_cloud_comparison_metadata",
        ),
    ),
    _cloud_surface(
        surface_id="provider_reported_response_model_surface",
        surface_name="Provider Reported Response Model Surface",
        provider_kind="openai",
        capability_band="provider_response_metadata",
        configuration_source="provider_response_model",
        latency_posture="provider_reported",
        route_applicability=tuple(RouteName),
        supported_payloads=(
            "provider_response_id",
            "provider_model_name",
            "token_usage_metadata",
        ),
        readiness_signals=(
            "generation_response_metadata_available",
            "provider_telemetry_metadata_only",
            "no_model_selection_authority",
        ),
        studio_surface_refs=(
            "provider_selection_metadata",
            "execution_simulator_metadata",
            "local_cloud_comparison_metadata",
        ),
    ),
)

CLOUD_MODEL_REGISTRY = CloudModelRegistry(
    model_surfaces=CLOUD_MODEL_SURFACES,
    surface_ids=tuple(surface.surface_id for surface in CLOUD_MODEL_SURFACES),
    provider_kinds=tuple(
        dict.fromkeys(surface.provider_kind for surface in CLOUD_MODEL_SURFACES)
    ),
    route_names=tuple(RouteName),
    profile_count=len(CLOUD_MODEL_SURFACES),
    source_registries=_CLOUD_MODEL_SOURCE_REGISTRIES,
    studio_surface_refs=_CLOUD_MODEL_STUDIO_SURFACES,
    observability_surfaces=_CLOUD_MODEL_OBSERVABILITY_SURFACES,
)

HybridExecutionStrategy = Literal[
    "local_first_advisory",
    "cloud_first_advisory",
    "side_by_side_advisory",
    "operator_selected_advisory",
]

HYBRID_EXECUTION_PROFILE_SERIALIZATION_VERSION = "hybrid_execution_profile.v1"
HYBRID_EXECUTION_REGISTRY_SERIALIZATION_VERSION = "hybrid_execution_registry.v1"
HYBRID_EXECUTION_REGISTRY_AUTHORITY_BOUNDARY = (
    "Hybrid execution metadata describes passive advisory coordination between "
    "local and cloud model surfaces for V4.4 Hybrid Studio inspection only; "
    "it does not execute local or cloud providers, route providers or models, "
    "run fallback, run parallel model calls, select models automatically, "
    "trigger retries, mutate prompts, write replay storage, or modify "
    "generated output."
)

_HYBRID_EXECUTION_SOURCE_REGISTRIES = (
    "local_model_registry",
    "cloud_model_registry",
    "generation_provider_contract",
    "agent_routing_registry",
    "workflow_agent_handoff_registry",
    "hybrid_agentic_workflow_registry",
)

_HYBRID_EXECUTION_STUDIO_SURFACES = (
    "hybrid_execution_panel",
    "provider_selection_metadata",
    "execution_simulator_metadata",
    "local_cloud_comparison_metadata",
)

_HYBRID_EXECUTION_OBSERVABILITY_SURFACES = (
    "execution_profile_id",
    "coordination_strategy",
    "source_local_surface_ids",
    "source_cloud_surface_ids",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_HYBRID_EXECUTION_BLOCKED_RUNTIME_BEHAVIORS = (
    "hybrid_execution",
    "local_provider_execution",
    "cloud_provider_execution",
    "provider_or_model_routing",
    "parallel_model_execution",
    "fallback_execution",
    "automatic_model_selection",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_replay_storage",
    "generated_output_modification",
)


class HybridExecutionProfile(BaseModel):
    """Inspectable advisory profile for local/cloud execution context."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    execution_profile_id: str = Field(min_length=1, max_length=100)
    profile_name: str = Field(min_length=1, max_length=140)
    coordination_strategy: HybridExecutionStrategy
    source_local_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_cloud_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    decision_inputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    studio_surface_refs: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    authority_boundary: str = Field(
        default=HYBRID_EXECUTION_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_HYBRID_EXECUTION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    hybrid_execution_implemented: Literal[False] = False
    local_provider_execution_implemented: Literal[False] = False
    cloud_provider_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    parallel_model_execution_implemented: Literal[False] = False
    fallback_execution_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    serialization_version: Literal["hybrid_execution_profile.v1"] = (
        HYBRID_EXECUTION_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class HybridExecutionRegistry(BaseModel):
    """Stable passive registry for V4.4 Hybrid Studio execution metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["hybrid_execution_registry"] = "hybrid_execution_registry"
    serialization_version: Literal["hybrid_execution_registry.v1"] = (
        HYBRID_EXECUTION_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=HYBRID_EXECUTION_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    execution_profiles: tuple[HybridExecutionProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    execution_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    coordination_strategies: tuple[HybridExecutionStrategy, ...] = Field(
        min_length=4,
        max_length=4,
    )
    local_surface_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    cloud_surface_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    studio_surface_refs: tuple[str, ...] = Field(min_length=4, max_length=4)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_HYBRID_EXECUTION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    hybrid_execution_implemented: Literal[False] = False
    local_provider_execution_implemented: Literal[False] = False
    cloud_provider_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    parallel_model_execution_implemented: Literal[False] = False
    fallback_execution_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.execution_profile_id for profile in self.execution_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("execution_profile_ids must be unique")
        if self.execution_profile_ids != derived_profile_ids:
            raise ValueError("execution_profile_ids must match execution_profiles")
        if self.profile_count != len(self.execution_profiles):
            raise ValueError("profile_count must match execution_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.coordination_strategies != tuple(
            profile.coordination_strategy for profile in self.execution_profiles
        ):
            raise ValueError("coordination_strategies must match execution_profiles")

        known_routes = set(self.route_names)
        known_local_surfaces = set(self.local_surface_ids)
        known_cloud_surfaces = set(self.cloud_surface_ids)
        known_studio_surfaces = set(self.studio_surface_refs)
        profile_sources = {
            source_registry
            for profile in self.execution_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match hybrid execution sources")

        for profile in self.execution_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("profile source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.source_local_surface_ids).issubset(known_local_surfaces):
                raise ValueError("source_local_surface_ids must be known local models")
            if not set(profile.source_cloud_surface_ids).issubset(known_cloud_surfaces):
                raise ValueError("source_cloud_surface_ids must be known cloud models")
            if not set(profile.studio_surface_refs).issubset(known_studio_surfaces):
                raise ValueError("studio_surface_refs must be known registry surfaces")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
        return self


def hybrid_execution_registry() -> HybridExecutionRegistry:
    """Return passive V4.4 Hybrid Studio execution metadata."""

    return HYBRID_EXECUTION_REGISTRY


def hybrid_execution_profile_by_id(
    execution_profile_id: str,
    registry: HybridExecutionRegistry | None = None,
) -> HybridExecutionProfile | None:
    """Return one hybrid execution profile without running it."""

    source_registry = registry or HYBRID_EXECUTION_REGISTRY
    for profile in source_registry.execution_profiles:
        if profile.execution_profile_id == execution_profile_id:
            return profile
    return None


def hybrid_execution_profiles_for_route(
    route: RouteName | str,
    registry: HybridExecutionRegistry | None = None,
) -> tuple[HybridExecutionProfile, ...]:
    """Return passive hybrid execution profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or HYBRID_EXECUTION_REGISTRY
    return tuple(
        profile
        for profile in source_registry.execution_profiles
        if route_name in profile.route_applicability
    )


def _hybrid_execution_profile(
    *,
    execution_profile_id: str,
    profile_name: str,
    coordination_strategy: HybridExecutionStrategy,
    source_local_surface_ids: tuple[str, ...],
    source_cloud_surface_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    decision_inputs: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
    studio_surface_refs: tuple[str, ...],
) -> HybridExecutionProfile:
    return HybridExecutionProfile(
        execution_profile_id=execution_profile_id,
        profile_name=profile_name,
        coordination_strategy=coordination_strategy,
        source_local_surface_ids=source_local_surface_ids,
        source_cloud_surface_ids=source_cloud_surface_ids,
        route_applicability=route_applicability,
        decision_inputs=decision_inputs,
        advisory_outputs=advisory_outputs,
        studio_surface_refs=studio_surface_refs,
        source_registries=_HYBRID_EXECUTION_SOURCE_REGISTRIES,
        observability_surfaces=_HYBRID_EXECUTION_OBSERVABILITY_SURFACES,
    )


HYBRID_EXECUTION_PROFILES = (
    _hybrid_execution_profile(
        execution_profile_id="local_first_context_profile",
        profile_name="Local First Context Profile",
        coordination_strategy="local_first_advisory",
        source_local_surface_ids=(
            "ollama_chat_surface",
            "llama_cpp_completion_surface",
        ),
        source_cloud_surface_ids=("openai_generation_model_surface",),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DEBUG,
            RouteName.PREVIEW,
        ),
        decision_inputs=(
            "operator_local_preference_metadata",
            "privacy_posture_metadata",
            "local_readiness_metadata",
        ),
        advisory_outputs=(
            "local_first_candidate_summary",
            "cloud_context_backup_note",
            "execution_boundary_notice",
        ),
        studio_surface_refs=(
            "hybrid_execution_panel",
            "provider_selection_metadata",
            "local_cloud_comparison_metadata",
        ),
    ),
    _hybrid_execution_profile(
        execution_profile_id="cloud_first_context_profile",
        profile_name="Cloud First Context Profile",
        coordination_strategy="cloud_first_advisory",
        source_local_surface_ids=("lm_studio_chat_surface",),
        source_cloud_surface_ids=(
            "openai_generation_model_surface",
            "provider_reported_response_model_surface",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        decision_inputs=(
            "operator_cloud_preference_metadata",
            "provider_configuration_metadata",
            "quality_expectation_metadata",
        ),
        advisory_outputs=(
            "cloud_first_candidate_summary",
            "local_context_review_note",
            "execution_boundary_notice",
        ),
        studio_surface_refs=(
            "hybrid_execution_panel",
            "provider_selection_metadata",
            "execution_simulator_metadata",
        ),
    ),
    _hybrid_execution_profile(
        execution_profile_id="side_by_side_comparison_profile",
        profile_name="Side By Side Comparison Profile",
        coordination_strategy="side_by_side_advisory",
        source_local_surface_ids=(
            "ollama_chat_surface",
            "local_transformers_multimodal_surface",
        ),
        source_cloud_surface_ids=(
            "openai_generation_model_surface",
            "ragas_evaluator_model_surface",
        ),
        route_applicability=(
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        decision_inputs=(
            "local_cloud_comparison_metadata",
            "evaluation_context_metadata",
            "operator_review_metadata",
        ),
        advisory_outputs=(
            "side_by_side_candidate_summary",
            "comparison_only_boundary_notice",
            "manual_review_prompt_metadata",
        ),
        studio_surface_refs=(
            "hybrid_execution_panel",
            "execution_simulator_metadata",
            "local_cloud_comparison_metadata",
        ),
    ),
    _hybrid_execution_profile(
        execution_profile_id="operator_selected_context_profile",
        profile_name="Operator Selected Context Profile",
        coordination_strategy="operator_selected_advisory",
        source_local_surface_ids=tuple(LOCAL_MODEL_REGISTRY.surface_ids),
        source_cloud_surface_ids=tuple(CLOUD_MODEL_REGISTRY.surface_ids),
        route_applicability=tuple(RouteName),
        decision_inputs=(
            "explicit_operator_selection_metadata",
            "model_catalog_metadata",
            "provider_boundary_metadata",
        ),
        advisory_outputs=(
            "operator_selected_candidate_summary",
            "manual_selection_boundary_notice",
            "no_automatic_execution_notice",
        ),
        studio_surface_refs=(
            "hybrid_execution_panel",
            "provider_selection_metadata",
            "execution_simulator_metadata",
            "local_cloud_comparison_metadata",
        ),
    ),
)

HYBRID_EXECUTION_REGISTRY = HybridExecutionRegistry(
    execution_profiles=HYBRID_EXECUTION_PROFILES,
    execution_profile_ids=tuple(
        profile.execution_profile_id for profile in HYBRID_EXECUTION_PROFILES
    ),
    coordination_strategies=tuple(
        profile.coordination_strategy for profile in HYBRID_EXECUTION_PROFILES
    ),
    local_surface_ids=tuple(LOCAL_MODEL_REGISTRY.surface_ids),
    cloud_surface_ids=tuple(CLOUD_MODEL_REGISTRY.surface_ids),
    route_names=tuple(RouteName),
    profile_count=len(HYBRID_EXECUTION_PROFILES),
    source_registries=_HYBRID_EXECUTION_SOURCE_REGISTRIES,
    studio_surface_refs=_HYBRID_EXECUTION_STUDIO_SURFACES,
    observability_surfaces=_HYBRID_EXECUTION_OBSERVABILITY_SURFACES,
)

AutoModePosture = Literal[
    "observe_only",
    "suggestion_only",
    "simulation_only",
    "operator_confirmed",
]

AUTO_MODE_PROFILE_SERIALIZATION_VERSION = "auto_mode_profile.v1"
AUTO_MODE_REGISTRY_SERIALIZATION_VERSION = "auto_mode_registry.v1"
AUTO_MODE_REGISTRY_AUTHORITY_BOUNDARY = (
    "Auto Mode metadata describes passive V4.4 Hybrid Studio advisory mode "
    "postures only; it does not execute workflows, route providers or models, "
    "select models automatically, run hybrid execution, trigger retries, ask "
    "for human input automatically, mutate prompts, write replay storage, or "
    "modify generated output."
)

_AUTO_MODE_SOURCE_REGISTRIES = (
    "local_model_registry",
    "cloud_model_registry",
    "hybrid_execution_registry",
    "agent_routing_registry",
    "settings_generation_provider_config",
    "hybrid_agentic_workflow_registry",
)

_AUTO_MODE_STUDIO_SURFACES = (
    "auto_mode_panel",
    "hybrid_execution_panel",
    "provider_selection_metadata",
    "execution_simulator_metadata",
    "local_cloud_comparison_metadata",
)

_AUTO_MODE_OBSERVABILITY_SURFACES = (
    "auto_mode_profile_id",
    "auto_mode_posture",
    "source_execution_profile_ids",
    "route_applicability",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_AUTO_MODE_BLOCKED_RUNTIME_BEHAVIORS = (
    "auto_mode_execution",
    "automatic_provider_selection",
    "automatic_model_selection",
    "hybrid_execution",
    "provider_or_model_routing",
    "human_input_request",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_replay_storage",
    "generated_output_modification",
)


class AutoModeProfile(BaseModel):
    """Inspectable advisory Auto Mode profile for Hybrid Studio."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    auto_mode_profile_id: str = Field(min_length=1, max_length=100)
    profile_name: str = Field(min_length=1, max_length=140)
    auto_mode_posture: AutoModePosture
    source_execution_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    advisory_inputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    operator_controls: tuple[str, ...] = Field(min_length=1, max_length=8)
    studio_surface_refs: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    authority_boundary: str = Field(
        default=AUTO_MODE_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_AUTO_MODE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    auto_mode_execution_implemented: Literal[False] = False
    automatic_provider_selection_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    hybrid_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    serialization_version: Literal["auto_mode_profile.v1"] = (
        AUTO_MODE_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AutoModeRegistry(BaseModel):
    """Stable passive registry for V4.4 Hybrid Studio Auto Mode metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["auto_mode_registry"] = "auto_mode_registry"
    serialization_version: Literal["auto_mode_registry.v1"] = (
        AUTO_MODE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AUTO_MODE_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    auto_mode_profiles: tuple[AutoModeProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    auto_mode_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    auto_mode_postures: tuple[AutoModePosture, ...] = Field(
        min_length=4,
        max_length=4,
    )
    execution_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    studio_surface_refs: tuple[str, ...] = Field(min_length=5, max_length=5)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_AUTO_MODE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    auto_mode_execution_implemented: Literal[False] = False
    automatic_provider_selection_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    hybrid_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.auto_mode_profile_id for profile in self.auto_mode_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("auto_mode_profile_ids must be unique")
        if self.auto_mode_profile_ids != derived_profile_ids:
            raise ValueError("auto_mode_profile_ids must match auto_mode_profiles")
        if self.profile_count != len(self.auto_mode_profiles):
            raise ValueError("profile_count must match auto_mode_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.auto_mode_postures != tuple(
            profile.auto_mode_posture for profile in self.auto_mode_profiles
        ):
            raise ValueError("auto_mode_postures must match auto_mode_profiles")

        known_routes = set(self.route_names)
        known_execution_profiles = set(self.execution_profile_ids)
        known_studio_surfaces = set(self.studio_surface_refs)
        profile_sources = {
            source_registry
            for profile in self.auto_mode_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match auto mode sources")

        for profile in self.auto_mode_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("profile source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.source_execution_profile_ids).issubset(
                known_execution_profiles
            ):
                raise ValueError(
                    "source_execution_profile_ids must be known execution profiles"
                )
            if not set(profile.studio_surface_refs).issubset(known_studio_surfaces):
                raise ValueError("studio_surface_refs must be known registry surfaces")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
        return self


def auto_mode_registry() -> AutoModeRegistry:
    """Return passive V4.4 Hybrid Studio Auto Mode metadata."""

    return AUTO_MODE_REGISTRY


def auto_mode_profile_by_id(
    auto_mode_profile_id: str,
    registry: AutoModeRegistry | None = None,
) -> AutoModeProfile | None:
    """Return one Auto Mode profile without executing it."""

    source_registry = registry or AUTO_MODE_REGISTRY
    for profile in source_registry.auto_mode_profiles:
        if profile.auto_mode_profile_id == auto_mode_profile_id:
            return profile
    return None


def auto_mode_profiles_for_route(
    route: RouteName | str,
    registry: AutoModeRegistry | None = None,
) -> tuple[AutoModeProfile, ...]:
    """Return passive Auto Mode profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or AUTO_MODE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.auto_mode_profiles
        if route_name in profile.route_applicability
    )


def _auto_mode_profile(
    *,
    auto_mode_profile_id: str,
    profile_name: str,
    auto_mode_posture: AutoModePosture,
    source_execution_profile_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    advisory_inputs: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
    operator_controls: tuple[str, ...],
    studio_surface_refs: tuple[str, ...],
) -> AutoModeProfile:
    return AutoModeProfile(
        auto_mode_profile_id=auto_mode_profile_id,
        profile_name=profile_name,
        auto_mode_posture=auto_mode_posture,
        source_execution_profile_ids=source_execution_profile_ids,
        route_applicability=route_applicability,
        advisory_inputs=advisory_inputs,
        advisory_outputs=advisory_outputs,
        operator_controls=operator_controls,
        studio_surface_refs=studio_surface_refs,
        source_registries=_AUTO_MODE_SOURCE_REGISTRIES,
        observability_surfaces=_AUTO_MODE_OBSERVABILITY_SURFACES,
    )


AUTO_MODE_PROFILES = (
    _auto_mode_profile(
        auto_mode_profile_id="auto_mode_observe_only_profile",
        profile_name="Auto Mode Observe Only Profile",
        auto_mode_posture="observe_only",
        source_execution_profile_ids=("operator_selected_context_profile",),
        route_applicability=tuple(RouteName),
        advisory_inputs=(
            "workflow_route_metadata",
            "model_catalog_metadata",
            "operator_visibility_metadata",
        ),
        advisory_outputs=(
            "auto_mode_observation_summary",
            "available_surface_snapshot",
            "no_automatic_action_notice",
        ),
        operator_controls=(
            "view_only_toggle",
            "manual_provider_override",
            "manual_model_override",
        ),
        studio_surface_refs=(
            "auto_mode_panel",
            "hybrid_execution_panel",
            "provider_selection_metadata",
        ),
    ),
    _auto_mode_profile(
        auto_mode_profile_id="auto_mode_suggestion_profile",
        profile_name="Auto Mode Suggestion Profile",
        auto_mode_posture="suggestion_only",
        source_execution_profile_ids=(
            "local_first_context_profile",
            "cloud_first_context_profile",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        advisory_inputs=(
            "route_applicability_metadata",
            "local_readiness_metadata",
            "cloud_configuration_metadata",
        ),
        advisory_outputs=(
            "suggested_execution_profile_metadata",
            "manual_confirmation_requirement",
            "routing_boundary_notice",
        ),
        operator_controls=(
            "accept_suggestion_control",
            "dismiss_suggestion_control",
            "manual_selection_control",
        ),
        studio_surface_refs=(
            "auto_mode_panel",
            "provider_selection_metadata",
            "local_cloud_comparison_metadata",
        ),
    ),
    _auto_mode_profile(
        auto_mode_profile_id="auto_mode_simulation_profile",
        profile_name="Auto Mode Simulation Profile",
        auto_mode_posture="simulation_only",
        source_execution_profile_ids=("side_by_side_comparison_profile",),
        route_applicability=(
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        advisory_inputs=(
            "execution_profile_metadata",
            "comparison_context_metadata",
            "evaluation_surface_metadata",
        ),
        advisory_outputs=(
            "simulated_execution_plan_metadata",
            "side_by_side_preview_metadata",
            "no_execution_notice",
        ),
        operator_controls=(
            "run_simulation_control",
            "clear_simulation_control",
            "manual_review_control",
        ),
        studio_surface_refs=(
            "auto_mode_panel",
            "execution_simulator_metadata",
            "local_cloud_comparison_metadata",
        ),
    ),
    _auto_mode_profile(
        auto_mode_profile_id="auto_mode_operator_confirmed_profile",
        profile_name="Auto Mode Operator Confirmed Profile",
        auto_mode_posture="operator_confirmed",
        source_execution_profile_ids=tuple(
            HYBRID_EXECUTION_REGISTRY.execution_profile_ids
        ),
        route_applicability=tuple(RouteName),
        advisory_inputs=(
            "explicit_operator_confirmation_metadata",
            "selected_execution_profile_metadata",
            "authority_boundary_metadata",
        ),
        advisory_outputs=(
            "operator_confirmed_selection_metadata",
            "manual_execution_boundary_notice",
            "audit_ready_mode_snapshot",
        ),
        operator_controls=(
            "confirm_selection_control",
            "cancel_selection_control",
            "manual_override_control",
        ),
        studio_surface_refs=(
            "auto_mode_panel",
            "hybrid_execution_panel",
            "provider_selection_metadata",
            "execution_simulator_metadata",
        ),
    ),
)

AUTO_MODE_REGISTRY = AutoModeRegistry(
    auto_mode_profiles=AUTO_MODE_PROFILES,
    auto_mode_profile_ids=tuple(
        profile.auto_mode_profile_id for profile in AUTO_MODE_PROFILES
    ),
    auto_mode_postures=tuple(
        profile.auto_mode_posture for profile in AUTO_MODE_PROFILES
    ),
    execution_profile_ids=tuple(HYBRID_EXECUTION_REGISTRY.execution_profile_ids),
    route_names=tuple(RouteName),
    profile_count=len(AUTO_MODE_PROFILES),
    source_registries=_AUTO_MODE_SOURCE_REGISTRIES,
    studio_surface_refs=_AUTO_MODE_STUDIO_SURFACES,
    observability_surfaces=_AUTO_MODE_OBSERVABILITY_SURFACES,
)

StudioModePosture = Literal[
    "inspect",
    "compare",
    "simulate",
    "operator_review",
]

STUDIO_MODE_PROFILE_SERIALIZATION_VERSION = "studio_mode_profile.v1"
STUDIO_MODE_REGISTRY_SERIALIZATION_VERSION = "studio_mode_registry.v1"
STUDIO_MODE_REGISTRY_AUTHORITY_BOUNDARY = (
    "Studio Mode metadata describes passive V4.4 Hybrid Studio operator views "
    "and inspectable mode surfaces only; it does not control workflows, run "
    "Auto Mode, execute hybrid profiles, route providers or models, execute "
    "artifacts, trigger retries, request human input automatically, mutate "
    "prompts, write replay storage, or modify generated output."
)

_STUDIO_MODE_SOURCE_REGISTRIES = (
    "auto_mode_registry",
    "hybrid_execution_registry",
    "local_model_registry",
    "cloud_model_registry",
    "workstation_engine_contract_registry",
    "hybrid_agentic_workflow_registry",
)

_STUDIO_MODE_SURFACES = (
    "studio_mode_shell",
    "model_catalog_panel",
    "auto_mode_panel",
    "hybrid_execution_panel",
    "comparison_panel",
)

_STUDIO_MODE_OBSERVABILITY_SURFACES = (
    "studio_mode_profile_id",
    "studio_mode_posture",
    "source_auto_mode_profile_ids",
    "source_execution_profile_ids",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_STUDIO_MODE_BLOCKED_RUNTIME_BEHAVIORS = (
    "studio_mode_runtime_control",
    "auto_mode_execution",
    "hybrid_execution",
    "workflow_control",
    "provider_or_model_routing",
    "artifact_execution",
    "human_input_request",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_replay_storage",
    "generated_output_modification",
)


class StudioModeProfile(BaseModel):
    """Inspectable passive Studio Mode surface profile."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    studio_mode_profile_id: str = Field(min_length=1, max_length=100)
    profile_name: str = Field(min_length=1, max_length=140)
    studio_mode_posture: StudioModePosture
    source_auto_mode_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_execution_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    visible_surface_refs: tuple[str, ...] = Field(min_length=1, max_length=5)
    operator_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    authority_boundary: str = Field(
        default=STUDIO_MODE_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_STUDIO_MODE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    studio_mode_runtime_control_implemented: Literal[False] = False
    auto_mode_execution_implemented: Literal[False] = False
    hybrid_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    serialization_version: Literal["studio_mode_profile.v1"] = (
        STUDIO_MODE_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class StudioModeRegistry(BaseModel):
    """Stable passive registry for V4.4 Hybrid Studio Mode metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["studio_mode_registry"] = "studio_mode_registry"
    serialization_version: Literal["studio_mode_registry.v1"] = (
        STUDIO_MODE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=STUDIO_MODE_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    studio_mode_profiles: tuple[StudioModeProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    studio_mode_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    studio_mode_postures: tuple[StudioModePosture, ...] = Field(
        min_length=4,
        max_length=4,
    )
    auto_mode_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    execution_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    visible_surface_refs: tuple[str, ...] = Field(min_length=5, max_length=5)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_STUDIO_MODE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    studio_mode_runtime_control_implemented: Literal[False] = False
    auto_mode_execution_implemented: Literal[False] = False
    hybrid_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.studio_mode_profile_id for profile in self.studio_mode_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("studio_mode_profile_ids must be unique")
        if self.studio_mode_profile_ids != derived_profile_ids:
            raise ValueError("studio_mode_profile_ids must match studio_mode_profiles")
        if self.profile_count != len(self.studio_mode_profiles):
            raise ValueError("profile_count must match studio_mode_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.studio_mode_postures != tuple(
            profile.studio_mode_posture for profile in self.studio_mode_profiles
        ):
            raise ValueError("studio_mode_postures must match studio_mode_profiles")

        known_routes = set(self.route_names)
        known_auto_profiles = set(self.auto_mode_profile_ids)
        known_execution_profiles = set(self.execution_profile_ids)
        known_visible_surfaces = set(self.visible_surface_refs)
        profile_sources = {
            source_registry
            for profile in self.studio_mode_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match studio mode sources")

        for profile in self.studio_mode_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("profile source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.source_auto_mode_profile_ids).issubset(
                known_auto_profiles
            ):
                raise ValueError(
                    "source_auto_mode_profile_ids must be known Auto Mode profiles"
                )
            if not set(profile.source_execution_profile_ids).issubset(
                known_execution_profiles
            ):
                raise ValueError(
                    "source_execution_profile_ids must be known execution profiles"
                )
            if not set(profile.visible_surface_refs).issubset(known_visible_surfaces):
                raise ValueError("visible_surface_refs must be known registry surfaces")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
        return self


def studio_mode_registry() -> StudioModeRegistry:
    """Return passive V4.4 Hybrid Studio Mode metadata."""

    return STUDIO_MODE_REGISTRY


def studio_mode_profile_by_id(
    studio_mode_profile_id: str,
    registry: StudioModeRegistry | None = None,
) -> StudioModeProfile | None:
    """Return one Studio Mode profile without activating it."""

    source_registry = registry or STUDIO_MODE_REGISTRY
    for profile in source_registry.studio_mode_profiles:
        if profile.studio_mode_profile_id == studio_mode_profile_id:
            return profile
    return None


def studio_mode_profiles_for_route(
    route: RouteName | str,
    registry: StudioModeRegistry | None = None,
) -> tuple[StudioModeProfile, ...]:
    """Return passive Studio Mode profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or STUDIO_MODE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.studio_mode_profiles
        if route_name in profile.route_applicability
    )


def _studio_mode_profile(
    *,
    studio_mode_profile_id: str,
    profile_name: str,
    studio_mode_posture: StudioModePosture,
    source_auto_mode_profile_ids: tuple[str, ...],
    source_execution_profile_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    visible_surface_refs: tuple[str, ...],
    operator_actions: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> StudioModeProfile:
    return StudioModeProfile(
        studio_mode_profile_id=studio_mode_profile_id,
        profile_name=profile_name,
        studio_mode_posture=studio_mode_posture,
        source_auto_mode_profile_ids=source_auto_mode_profile_ids,
        source_execution_profile_ids=source_execution_profile_ids,
        route_applicability=route_applicability,
        visible_surface_refs=visible_surface_refs,
        operator_actions=operator_actions,
        advisory_outputs=advisory_outputs,
        source_registries=_STUDIO_MODE_SOURCE_REGISTRIES,
        observability_surfaces=_STUDIO_MODE_OBSERVABILITY_SURFACES,
    )


STUDIO_MODE_PROFILES = (
    _studio_mode_profile(
        studio_mode_profile_id="studio_mode_inspection_profile",
        profile_name="Studio Mode Inspection Profile",
        studio_mode_posture="inspect",
        source_auto_mode_profile_ids=("auto_mode_observe_only_profile",),
        source_execution_profile_ids=("operator_selected_context_profile",),
        route_applicability=tuple(RouteName),
        visible_surface_refs=(
            "studio_mode_shell",
            "model_catalog_panel",
            "auto_mode_panel",
        ),
        operator_actions=(
            "inspect_model_catalog_metadata",
            "inspect_auto_mode_metadata",
            "inspect_execution_boundaries",
        ),
        advisory_outputs=(
            "studio_inspection_summary",
            "metadata_surface_inventory",
            "authority_boundary_snapshot",
        ),
    ),
    _studio_mode_profile(
        studio_mode_profile_id="studio_mode_comparison_profile",
        profile_name="Studio Mode Comparison Profile",
        studio_mode_posture="compare",
        source_auto_mode_profile_ids=("auto_mode_suggestion_profile",),
        source_execution_profile_ids=(
            "local_first_context_profile",
            "cloud_first_context_profile",
            "side_by_side_comparison_profile",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        visible_surface_refs=(
            "studio_mode_shell",
            "model_catalog_panel",
            "comparison_panel",
        ),
        operator_actions=(
            "compare_local_cloud_metadata",
            "compare_provider_boundary_metadata",
            "select_metadata_view",
        ),
        advisory_outputs=(
            "comparison_view_metadata",
            "surface_difference_summary",
            "manual_selection_context",
        ),
    ),
    _studio_mode_profile(
        studio_mode_profile_id="studio_mode_simulation_profile",
        profile_name="Studio Mode Simulation Profile",
        studio_mode_posture="simulate",
        source_auto_mode_profile_ids=("auto_mode_simulation_profile",),
        source_execution_profile_ids=("side_by_side_comparison_profile",),
        route_applicability=(
            RouteName.EXPLAIN,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        visible_surface_refs=(
            "studio_mode_shell",
            "hybrid_execution_panel",
            "comparison_panel",
        ),
        operator_actions=(
            "open_simulation_metadata_view",
            "review_simulated_plan_metadata",
            "clear_simulation_metadata",
        ),
        advisory_outputs=(
            "simulation_view_metadata",
            "non_execution_notice",
            "manual_review_context",
        ),
    ),
    _studio_mode_profile(
        studio_mode_profile_id="studio_mode_operator_review_profile",
        profile_name="Studio Mode Operator Review Profile",
        studio_mode_posture="operator_review",
        source_auto_mode_profile_ids=("auto_mode_operator_confirmed_profile",),
        source_execution_profile_ids=tuple(
            HYBRID_EXECUTION_REGISTRY.execution_profile_ids
        ),
        route_applicability=tuple(RouteName),
        visible_surface_refs=tuple(_STUDIO_MODE_SURFACES),
        operator_actions=(
            "review_operator_selection_metadata",
            "review_authority_boundaries",
            "record_manual_review_note_metadata",
        ),
        advisory_outputs=(
            "operator_review_snapshot",
            "manual_confirmation_context",
            "audit_ready_studio_metadata",
        ),
    ),
)

STUDIO_MODE_REGISTRY = StudioModeRegistry(
    studio_mode_profiles=STUDIO_MODE_PROFILES,
    studio_mode_profile_ids=tuple(
        profile.studio_mode_profile_id for profile in STUDIO_MODE_PROFILES
    ),
    studio_mode_postures=tuple(
        profile.studio_mode_posture for profile in STUDIO_MODE_PROFILES
    ),
    auto_mode_profile_ids=tuple(AUTO_MODE_REGISTRY.auto_mode_profile_ids),
    execution_profile_ids=tuple(HYBRID_EXECUTION_REGISTRY.execution_profile_ids),
    visible_surface_refs=_STUDIO_MODE_SURFACES,
    route_names=tuple(RouteName),
    profile_count=len(STUDIO_MODE_PROFILES),
    source_registries=_STUDIO_MODE_SOURCE_REGISTRIES,
    observability_surfaces=_STUDIO_MODE_OBSERVABILITY_SURFACES,
)

HitlDecisionPosture = Literal[
    "visible_only",
    "confirmation_advised",
    "risk_review_advised",
    "final_review_advised",
]

HITL_DECISION_PROFILE_SERIALIZATION_VERSION = "hitl_decision_profile.v1"
HITL_DECISION_REGISTRY_SERIALIZATION_VERSION = "hitl_decision_registry.v1"
HITL_DECISION_REGISTRY_AUTHORITY_BOUNDARY = (
    "HITL decision metadata describes passive human-review decision surfaces "
    "for V4.4 Hybrid Studio inspection only; it does not request human input, "
    "approve escalation, interrupt workflows, route providers or models, run "
    "Auto Mode, execute hybrid profiles, trigger retries, mutate prompts, "
    "write replay storage, or modify generated output."
)

_HITL_DECISION_SOURCE_REGISTRIES = (
    "studio_mode_registry",
    "auto_mode_registry",
    "hybrid_execution_registry",
    "hitl_escalation_gate_registry",
    "workflow_agent_handoff_registry",
    "agent_escalation_signal_registry",
)

_HITL_DECISION_SURFACES = (
    "hitl_decision_panel",
    "studio_mode_shell",
    "auto_mode_panel",
    "hybrid_execution_panel",
    "operator_review_panel",
)

_HITL_DECISION_OBSERVABILITY_SURFACES = (
    "hitl_decision_profile_id",
    "hitl_decision_posture",
    "source_studio_mode_profile_ids",
    "source_auto_mode_profile_ids",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_HITL_DECISION_BLOCKED_RUNTIME_BEHAVIORS = (
    "hitl_decision_execution",
    "human_input_request",
    "escalation_approval",
    "workflow_interruption",
    "workflow_control",
    "auto_mode_execution",
    "hybrid_execution",
    "provider_or_model_routing",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_replay_storage",
    "generated_output_modification",
)


class HitlDecisionProfile(BaseModel):
    """Inspectable passive HITL decision profile for Hybrid Studio."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    hitl_decision_profile_id: str = Field(min_length=1, max_length=100)
    profile_name: str = Field(min_length=1, max_length=140)
    hitl_decision_posture: HitlDecisionPosture
    source_studio_mode_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_auto_mode_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_execution_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    decision_inputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_decision_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    human_review_surfaces: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    authority_boundary: str = Field(
        default=HITL_DECISION_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_HITL_DECISION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    hitl_decision_execution_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    escalation_approval_implemented: Literal[False] = False
    workflow_interruption_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    auto_mode_execution_implemented: Literal[False] = False
    hybrid_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    serialization_version: Literal["hitl_decision_profile.v1"] = (
        HITL_DECISION_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class HitlDecisionRegistry(BaseModel):
    """Stable passive registry for V4.4 Hybrid Studio HITL decisions."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["hitl_decision_registry"] = "hitl_decision_registry"
    serialization_version: Literal["hitl_decision_registry.v1"] = (
        HITL_DECISION_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=HITL_DECISION_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    hitl_decision_profiles: tuple[HitlDecisionProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    hitl_decision_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    hitl_decision_postures: tuple[HitlDecisionPosture, ...] = Field(
        min_length=4,
        max_length=4,
    )
    studio_mode_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    auto_mode_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    execution_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    human_review_surfaces: tuple[str, ...] = Field(min_length=5, max_length=5)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_HITL_DECISION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    hitl_decision_execution_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    escalation_approval_implemented: Literal[False] = False
    workflow_interruption_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    auto_mode_execution_implemented: Literal[False] = False
    hybrid_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.hitl_decision_profile_id for profile in self.hitl_decision_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("hitl_decision_profile_ids must be unique")
        if self.hitl_decision_profile_ids != derived_profile_ids:
            raise ValueError(
                "hitl_decision_profile_ids must match hitl_decision_profiles"
            )
        if self.profile_count != len(self.hitl_decision_profiles):
            raise ValueError("profile_count must match hitl_decision_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.hitl_decision_postures != tuple(
            profile.hitl_decision_posture for profile in self.hitl_decision_profiles
        ):
            raise ValueError("hitl_decision_postures must match hitl_decision_profiles")

        known_routes = set(self.route_names)
        known_studio_profiles = set(self.studio_mode_profile_ids)
        known_auto_profiles = set(self.auto_mode_profile_ids)
        known_execution_profiles = set(self.execution_profile_ids)
        known_review_surfaces = set(self.human_review_surfaces)
        profile_sources = {
            source_registry
            for profile in self.hitl_decision_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match HITL decision sources")

        for profile in self.hitl_decision_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("profile source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.source_studio_mode_profile_ids).issubset(
                known_studio_profiles
            ):
                raise ValueError(
                    "source_studio_mode_profile_ids must be known Studio Mode profiles"
                )
            if not set(profile.source_auto_mode_profile_ids).issubset(
                known_auto_profiles
            ):
                raise ValueError(
                    "source_auto_mode_profile_ids must be known Auto Mode profiles"
                )
            if not set(profile.source_execution_profile_ids).issubset(
                known_execution_profiles
            ):
                raise ValueError(
                    "source_execution_profile_ids must be known execution profiles"
                )
            if not set(profile.human_review_surfaces).issubset(known_review_surfaces):
                raise ValueError(
                    "human_review_surfaces must be known registry surfaces"
                )
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
        return self


def hitl_decision_registry() -> HitlDecisionRegistry:
    """Return passive V4.4 Hybrid Studio HITL decision metadata."""

    return HITL_DECISION_REGISTRY


def hitl_decision_profile_by_id(
    hitl_decision_profile_id: str,
    registry: HitlDecisionRegistry | None = None,
) -> HitlDecisionProfile | None:
    """Return one HITL decision profile without requesting input."""

    source_registry = registry or HITL_DECISION_REGISTRY
    for profile in source_registry.hitl_decision_profiles:
        if profile.hitl_decision_profile_id == hitl_decision_profile_id:
            return profile
    return None


def hitl_decision_profiles_for_route(
    route: RouteName | str,
    registry: HitlDecisionRegistry | None = None,
) -> tuple[HitlDecisionProfile, ...]:
    """Return passive HITL decision profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or HITL_DECISION_REGISTRY
    return tuple(
        profile
        for profile in source_registry.hitl_decision_profiles
        if route_name in profile.route_applicability
    )


def _hitl_decision_profile(
    *,
    hitl_decision_profile_id: str,
    profile_name: str,
    hitl_decision_posture: HitlDecisionPosture,
    source_studio_mode_profile_ids: tuple[str, ...],
    source_auto_mode_profile_ids: tuple[str, ...],
    source_execution_profile_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    decision_inputs: tuple[str, ...],
    advisory_decision_outputs: tuple[str, ...],
    human_review_surfaces: tuple[str, ...],
) -> HitlDecisionProfile:
    return HitlDecisionProfile(
        hitl_decision_profile_id=hitl_decision_profile_id,
        profile_name=profile_name,
        hitl_decision_posture=hitl_decision_posture,
        source_studio_mode_profile_ids=source_studio_mode_profile_ids,
        source_auto_mode_profile_ids=source_auto_mode_profile_ids,
        source_execution_profile_ids=source_execution_profile_ids,
        route_applicability=route_applicability,
        decision_inputs=decision_inputs,
        advisory_decision_outputs=advisory_decision_outputs,
        human_review_surfaces=human_review_surfaces,
        source_registries=_HITL_DECISION_SOURCE_REGISTRIES,
        observability_surfaces=_HITL_DECISION_OBSERVABILITY_SURFACES,
    )


HITL_DECISION_PROFILES = (
    _hitl_decision_profile(
        hitl_decision_profile_id="hitl_visibility_decision_profile",
        profile_name="HITL Visibility Decision Profile",
        hitl_decision_posture="visible_only",
        source_studio_mode_profile_ids=("studio_mode_inspection_profile",),
        source_auto_mode_profile_ids=("auto_mode_observe_only_profile",),
        source_execution_profile_ids=("operator_selected_context_profile",),
        route_applicability=tuple(RouteName),
        decision_inputs=(
            "operator_visibility_metadata",
            "studio_surface_inventory",
            "authority_boundary_snapshot",
        ),
        advisory_decision_outputs=(
            "human_review_visibility_metadata",
            "no_interruption_notice",
            "manual_review_optional_context",
        ),
        human_review_surfaces=(
            "hitl_decision_panel",
            "studio_mode_shell",
            "operator_review_panel",
        ),
    ),
    _hitl_decision_profile(
        hitl_decision_profile_id="hitl_confirmation_decision_profile",
        profile_name="HITL Confirmation Decision Profile",
        hitl_decision_posture="confirmation_advised",
        source_studio_mode_profile_ids=("studio_mode_operator_review_profile",),
        source_auto_mode_profile_ids=("auto_mode_operator_confirmed_profile",),
        source_execution_profile_ids=tuple(
            HYBRID_EXECUTION_REGISTRY.execution_profile_ids
        ),
        route_applicability=tuple(RouteName),
        decision_inputs=(
            "operator_confirmed_selection_metadata",
            "manual_execution_boundary_notice",
            "selected_execution_profile_metadata",
        ),
        advisory_decision_outputs=(
            "confirmation_recommended_metadata",
            "manual_confirmation_context",
            "no_automatic_approval_notice",
        ),
        human_review_surfaces=(
            "hitl_decision_panel",
            "auto_mode_panel",
            "operator_review_panel",
        ),
    ),
    _hitl_decision_profile(
        hitl_decision_profile_id="hitl_risk_review_decision_profile",
        profile_name="HITL Risk Review Decision Profile",
        hitl_decision_posture="risk_review_advised",
        source_studio_mode_profile_ids=(
            "studio_mode_comparison_profile",
            "studio_mode_simulation_profile",
        ),
        source_auto_mode_profile_ids=(
            "auto_mode_suggestion_profile",
            "auto_mode_simulation_profile",
        ),
        source_execution_profile_ids=("side_by_side_comparison_profile",),
        route_applicability=(
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        decision_inputs=(
            "risk_review_metadata",
            "comparison_view_metadata",
            "simulation_view_metadata",
        ),
        advisory_decision_outputs=(
            "risk_review_recommended_metadata",
            "operator_attention_context",
            "no_workflow_interruption_notice",
        ),
        human_review_surfaces=(
            "hitl_decision_panel",
            "hybrid_execution_panel",
            "operator_review_panel",
        ),
    ),
    _hitl_decision_profile(
        hitl_decision_profile_id="hitl_final_review_decision_profile",
        profile_name="HITL Final Review Decision Profile",
        hitl_decision_posture="final_review_advised",
        source_studio_mode_profile_ids=("studio_mode_operator_review_profile",),
        source_auto_mode_profile_ids=("auto_mode_operator_confirmed_profile",),
        source_execution_profile_ids=tuple(
            HYBRID_EXECUTION_REGISTRY.execution_profile_ids
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        decision_inputs=(
            "final_review_metadata",
            "operator_review_snapshot",
            "audit_ready_studio_metadata",
        ),
        advisory_decision_outputs=(
            "final_review_recommended_metadata",
            "manual_signoff_context",
            "no_output_mutation_notice",
        ),
        human_review_surfaces=tuple(_HITL_DECISION_SURFACES),
    ),
)

HITL_DECISION_REGISTRY = HitlDecisionRegistry(
    hitl_decision_profiles=HITL_DECISION_PROFILES,
    hitl_decision_profile_ids=tuple(
        profile.hitl_decision_profile_id for profile in HITL_DECISION_PROFILES
    ),
    hitl_decision_postures=tuple(
        profile.hitl_decision_posture for profile in HITL_DECISION_PROFILES
    ),
    studio_mode_profile_ids=tuple(STUDIO_MODE_REGISTRY.studio_mode_profile_ids),
    auto_mode_profile_ids=tuple(AUTO_MODE_REGISTRY.auto_mode_profile_ids),
    execution_profile_ids=tuple(HYBRID_EXECUTION_REGISTRY.execution_profile_ids),
    human_review_surfaces=_HITL_DECISION_SURFACES,
    route_names=tuple(RouteName),
    profile_count=len(HITL_DECISION_PROFILES),
    source_registries=_HITL_DECISION_SOURCE_REGISTRIES,
    observability_surfaces=_HITL_DECISION_OBSERVABILITY_SURFACES,
)

ProviderSelectionPosture = Literal[
    "current_config_visibility",
    "local_candidate_visibility",
    "cloud_candidate_visibility",
    "operator_override_visibility",
]

PROVIDER_SELECTION_PROFILE_SERIALIZATION_VERSION = "provider_selection_profile.v1"
PROVIDER_SELECTION_REGISTRY_SERIALIZATION_VERSION = "provider_selection_registry.v1"
PROVIDER_SELECTION_REGISTRY_AUTHORITY_BOUNDARY = (
    "Provider Selection metadata describes passive provider and model candidate "
    "visibility for V4.4 Hybrid Studio inspection only; it does not select "
    "providers automatically, switch models, alter build_generation_provider, "
    "route providers or models, execute local or cloud providers, trigger "
    "retries, request human input automatically, mutate prompts, write replay "
    "storage, or modify generated output."
)

_PROVIDER_SELECTION_SOURCE_REGISTRIES = (
    "local_model_registry",
    "cloud_model_registry",
    "auto_mode_registry",
    "hitl_decision_registry",
    "generation_provider_factory",
    "settings_generation_provider_config",
)

_PROVIDER_SELECTION_SURFACES = (
    "provider_selection_panel",
    "model_catalog_panel",
    "auto_mode_panel",
    "hitl_decision_panel",
    "operator_override_panel",
)

_PROVIDER_SELECTION_OBSERVABILITY_SURFACES = (
    "provider_selection_profile_id",
    "provider_selection_posture",
    "provider_candidate_ids",
    "route_applicability",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_PROVIDER_SELECTION_BLOCKED_RUNTIME_BEHAVIORS = (
    "provider_selection_execution",
    "automatic_provider_selection",
    "automatic_model_selection",
    "provider_or_model_routing",
    "model_switching",
    "local_provider_execution",
    "cloud_provider_execution",
    "workflow_control",
    "human_input_request",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_replay_storage",
    "generated_output_modification",
)


class ProviderSelectionProfile(BaseModel):
    """Inspectable passive provider selection metadata profile."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    provider_selection_profile_id: str = Field(min_length=1, max_length=110)
    profile_name: str = Field(min_length=1, max_length=150)
    provider_selection_posture: ProviderSelectionPosture
    provider_candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_local_surface_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=4
    )
    source_cloud_surface_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=4
    )
    source_auto_mode_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_hitl_decision_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    selection_inputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    selection_surface_refs: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    authority_boundary: str = Field(
        default=PROVIDER_SELECTION_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1100,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_PROVIDER_SELECTION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    provider_selection_execution_implemented: Literal[False] = False
    automatic_provider_selection_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_switching_implemented: Literal[False] = False
    local_provider_execution_implemented: Literal[False] = False
    cloud_provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    serialization_version: Literal["provider_selection_profile.v1"] = (
        PROVIDER_SELECTION_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class ProviderSelectionRegistry(BaseModel):
    """Stable passive registry for V4.4 Hybrid Studio provider selection."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["provider_selection_registry"] = "provider_selection_registry"
    serialization_version: Literal["provider_selection_registry.v1"] = (
        PROVIDER_SELECTION_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PROVIDER_SELECTION_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1100,
    )
    provider_selection_profiles: tuple[ProviderSelectionProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    provider_selection_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    provider_selection_postures: tuple[ProviderSelectionPosture, ...] = Field(
        min_length=4,
        max_length=4,
    )
    provider_candidate_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    local_surface_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    cloud_surface_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    auto_mode_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    hitl_decision_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    selection_surface_refs: tuple[str, ...] = Field(min_length=5, max_length=5)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_PROVIDER_SELECTION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    provider_selection_execution_implemented: Literal[False] = False
    automatic_provider_selection_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_switching_implemented: Literal[False] = False
    local_provider_execution_implemented: Literal[False] = False
    cloud_provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.provider_selection_profile_id
            for profile in self.provider_selection_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("provider_selection_profile_ids must be unique")
        if self.provider_selection_profile_ids != derived_profile_ids:
            raise ValueError(
                "provider_selection_profile_ids must match provider_selection_profiles"
            )
        if self.profile_count != len(self.provider_selection_profiles):
            raise ValueError("profile_count must match provider_selection_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.provider_selection_postures != tuple(
            profile.provider_selection_posture
            for profile in self.provider_selection_profiles
        ):
            raise ValueError(
                "provider_selection_postures must match provider_selection_profiles"
            )

        known_routes = set(self.route_names)
        known_provider_candidates = set(self.provider_candidate_ids)
        known_local_surfaces = set(self.local_surface_ids)
        known_cloud_surfaces = set(self.cloud_surface_ids)
        known_auto_profiles = set(self.auto_mode_profile_ids)
        known_hitl_profiles = set(self.hitl_decision_profile_ids)
        known_selection_surfaces = set(self.selection_surface_refs)
        profile_sources = {
            source_registry
            for profile in self.provider_selection_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match provider selection sources")

        for profile in self.provider_selection_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("profile source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.provider_candidate_ids).issubset(
                known_provider_candidates
            ):
                raise ValueError("provider_candidate_ids must be known providers")
            if not set(profile.source_local_surface_ids).issubset(known_local_surfaces):
                raise ValueError("source_local_surface_ids must be known local models")
            if not set(profile.source_cloud_surface_ids).issubset(known_cloud_surfaces):
                raise ValueError("source_cloud_surface_ids must be known cloud models")
            if not set(profile.source_auto_mode_profile_ids).issubset(
                known_auto_profiles
            ):
                raise ValueError(
                    "source_auto_mode_profile_ids must be known Auto Mode profiles"
                )
            if not set(profile.source_hitl_decision_profile_ids).issubset(
                known_hitl_profiles
            ):
                raise ValueError(
                    "source_hitl_decision_profile_ids must be known HITL profiles"
                )
            if not set(profile.selection_surface_refs).issubset(
                known_selection_surfaces
            ):
                raise ValueError(
                    "selection_surface_refs must be known registry surfaces"
                )
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
        return self


def provider_selection_registry() -> ProviderSelectionRegistry:
    """Return passive V4.4 Hybrid Studio provider selection metadata."""

    return PROVIDER_SELECTION_REGISTRY


def provider_selection_profile_by_id(
    provider_selection_profile_id: str,
    registry: ProviderSelectionRegistry | None = None,
) -> ProviderSelectionProfile | None:
    """Return one provider selection profile without selecting a provider."""

    source_registry = registry or PROVIDER_SELECTION_REGISTRY
    for profile in source_registry.provider_selection_profiles:
        if profile.provider_selection_profile_id == provider_selection_profile_id:
            return profile
    return None


def provider_selection_profiles_for_route(
    route: RouteName | str,
    registry: ProviderSelectionRegistry | None = None,
) -> tuple[ProviderSelectionProfile, ...]:
    """Return passive provider selection profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or PROVIDER_SELECTION_REGISTRY
    return tuple(
        profile
        for profile in source_registry.provider_selection_profiles
        if route_name in profile.route_applicability
    )


def _provider_selection_profile(
    *,
    provider_selection_profile_id: str,
    profile_name: str,
    provider_selection_posture: ProviderSelectionPosture,
    provider_candidate_ids: tuple[str, ...],
    source_local_surface_ids: tuple[str, ...],
    source_cloud_surface_ids: tuple[str, ...],
    source_auto_mode_profile_ids: tuple[str, ...],
    source_hitl_decision_profile_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    selection_inputs: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
    selection_surface_refs: tuple[str, ...],
) -> ProviderSelectionProfile:
    return ProviderSelectionProfile(
        provider_selection_profile_id=provider_selection_profile_id,
        profile_name=profile_name,
        provider_selection_posture=provider_selection_posture,
        provider_candidate_ids=provider_candidate_ids,
        source_local_surface_ids=source_local_surface_ids,
        source_cloud_surface_ids=source_cloud_surface_ids,
        source_auto_mode_profile_ids=source_auto_mode_profile_ids,
        source_hitl_decision_profile_ids=source_hitl_decision_profile_ids,
        route_applicability=route_applicability,
        selection_inputs=selection_inputs,
        advisory_outputs=advisory_outputs,
        selection_surface_refs=selection_surface_refs,
        source_registries=_PROVIDER_SELECTION_SOURCE_REGISTRIES,
        observability_surfaces=_PROVIDER_SELECTION_OBSERVABILITY_SURFACES,
    )


PROVIDER_SELECTION_CANDIDATE_IDS = (
    "openai",
    "ollama",
    "lm_studio",
    "llama_cpp",
    "local_transformers",
)

PROVIDER_SELECTION_PROFILES = (
    _provider_selection_profile(
        provider_selection_profile_id="current_config_provider_visibility_profile",
        profile_name="Current Config Provider Visibility Profile",
        provider_selection_posture="current_config_visibility",
        provider_candidate_ids=("openai",),
        source_local_surface_ids=(),
        source_cloud_surface_ids=("openai_generation_model_surface",),
        source_auto_mode_profile_ids=("auto_mode_observe_only_profile",),
        source_hitl_decision_profile_ids=("hitl_visibility_decision_profile",),
        route_applicability=tuple(RouteName),
        selection_inputs=(
            "default_generation_provider_config",
            "openai_model_setting",
            "provider_factory_metadata",
        ),
        advisory_outputs=(
            "current_provider_visibility_metadata",
            "configured_model_visibility_metadata",
            "no_provider_switch_notice",
        ),
        selection_surface_refs=(
            "provider_selection_panel",
            "model_catalog_panel",
            "operator_override_panel",
        ),
    ),
    _provider_selection_profile(
        provider_selection_profile_id="local_candidate_provider_visibility_profile",
        profile_name="Local Candidate Provider Visibility Profile",
        provider_selection_posture="local_candidate_visibility",
        provider_candidate_ids=(
            "ollama",
            "lm_studio",
            "llama_cpp",
            "local_transformers",
        ),
        source_local_surface_ids=tuple(LOCAL_MODEL_REGISTRY.surface_ids),
        source_cloud_surface_ids=(),
        source_auto_mode_profile_ids=("auto_mode_suggestion_profile",),
        source_hitl_decision_profile_ids=("hitl_confirmation_decision_profile",),
        route_applicability=tuple(RouteName),
        selection_inputs=(
            "local_model_catalog_metadata",
            "local_runtime_readiness_metadata",
            "operator_local_preference_metadata",
        ),
        advisory_outputs=(
            "local_candidate_visibility_metadata",
            "manual_local_selection_context",
            "no_local_execution_notice",
        ),
        selection_surface_refs=(
            "provider_selection_panel",
            "model_catalog_panel",
            "auto_mode_panel",
        ),
    ),
    _provider_selection_profile(
        provider_selection_profile_id="cloud_candidate_provider_visibility_profile",
        profile_name="Cloud Candidate Provider Visibility Profile",
        provider_selection_posture="cloud_candidate_visibility",
        provider_candidate_ids=("openai",),
        source_local_surface_ids=(),
        source_cloud_surface_ids=tuple(CLOUD_MODEL_REGISTRY.surface_ids),
        source_auto_mode_profile_ids=("auto_mode_suggestion_profile",),
        source_hitl_decision_profile_ids=(
            "hitl_confirmation_decision_profile",
            "hitl_risk_review_decision_profile",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        selection_inputs=(
            "cloud_model_catalog_metadata",
            "provider_configuration_metadata",
            "provider_boundary_metadata",
        ),
        advisory_outputs=(
            "cloud_candidate_visibility_metadata",
            "manual_cloud_selection_context",
            "no_cloud_execution_notice",
        ),
        selection_surface_refs=(
            "provider_selection_panel",
            "model_catalog_panel",
            "hitl_decision_panel",
        ),
    ),
    _provider_selection_profile(
        provider_selection_profile_id="operator_override_provider_visibility_profile",
        profile_name="Operator Override Provider Visibility Profile",
        provider_selection_posture="operator_override_visibility",
        provider_candidate_ids=PROVIDER_SELECTION_CANDIDATE_IDS,
        source_local_surface_ids=tuple(LOCAL_MODEL_REGISTRY.surface_ids),
        source_cloud_surface_ids=tuple(CLOUD_MODEL_REGISTRY.surface_ids),
        source_auto_mode_profile_ids=("auto_mode_operator_confirmed_profile",),
        source_hitl_decision_profile_ids=(
            "hitl_confirmation_decision_profile",
            "hitl_final_review_decision_profile",
        ),
        route_applicability=tuple(RouteName),
        selection_inputs=(
            "explicit_operator_override_metadata",
            "provider_candidate_visibility_metadata",
            "hitl_confirmation_metadata",
        ),
        advisory_outputs=(
            "operator_override_visibility_metadata",
            "manual_selection_boundary_notice",
            "no_automatic_provider_selection_notice",
        ),
        selection_surface_refs=tuple(_PROVIDER_SELECTION_SURFACES),
    ),
)

PROVIDER_SELECTION_REGISTRY = ProviderSelectionRegistry(
    provider_selection_profiles=PROVIDER_SELECTION_PROFILES,
    provider_selection_profile_ids=tuple(
        profile.provider_selection_profile_id for profile in PROVIDER_SELECTION_PROFILES
    ),
    provider_selection_postures=tuple(
        profile.provider_selection_posture for profile in PROVIDER_SELECTION_PROFILES
    ),
    provider_candidate_ids=PROVIDER_SELECTION_CANDIDATE_IDS,
    local_surface_ids=tuple(LOCAL_MODEL_REGISTRY.surface_ids),
    cloud_surface_ids=tuple(CLOUD_MODEL_REGISTRY.surface_ids),
    auto_mode_profile_ids=tuple(AUTO_MODE_REGISTRY.auto_mode_profile_ids),
    hitl_decision_profile_ids=tuple(HITL_DECISION_REGISTRY.hitl_decision_profile_ids),
    selection_surface_refs=_PROVIDER_SELECTION_SURFACES,
    route_names=tuple(RouteName),
    profile_count=len(PROVIDER_SELECTION_PROFILES),
    source_registries=_PROVIDER_SELECTION_SOURCE_REGISTRIES,
    observability_surfaces=_PROVIDER_SELECTION_OBSERVABILITY_SURFACES,
)

ExecutionSimulationScope = Literal[
    "route_preview",
    "local_cloud_comparison",
    "hitl_review",
    "provider_selection",
]

EXECUTION_SIMULATION_PROFILE_SERIALIZATION_VERSION = "execution_simulation_profile.v1"
EXECUTION_SIMULATOR_REGISTRY_SERIALIZATION_VERSION = "execution_simulator_registry.v1"
EXECUTION_SIMULATOR_REGISTRY_AUTHORITY_BOUNDARY = (
    "Execution Simulator metadata describes passive simulated execution plans "
    "and comparison surfaces for V4.4 Hybrid Studio inspection only; it does "
    "not execute providers, execute artifacts, route providers or models, run "
    "workflow transitions, trigger retries, request human input automatically, "
    "mutate prompts, write replay storage, or modify generated output."
)

_EXECUTION_SIMULATOR_SOURCE_REGISTRIES = (
    "provider_selection_registry",
    "hitl_decision_registry",
    "auto_mode_registry",
    "studio_mode_registry",
    "hybrid_execution_registry",
    "workflow_agent_handoff_registry",
)

_EXECUTION_SIMULATOR_SURFACES = (
    "execution_simulator_panel",
    "provider_selection_panel",
    "hitl_decision_panel",
    "hybrid_execution_panel",
    "comparison_panel",
)

_EXECUTION_SIMULATOR_OBSERVABILITY_SURFACES = (
    "execution_simulation_profile_id",
    "simulation_scope",
    "source_provider_selection_profile_ids",
    "source_execution_profile_ids",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_EXECUTION_SIMULATOR_BLOCKED_RUNTIME_BEHAVIORS = (
    "simulation_runtime_execution",
    "provider_execution",
    "artifact_execution",
    "provider_or_model_routing",
    "workflow_transition_execution",
    "human_input_request",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_replay_storage",
    "generated_output_modification",
)


class ExecutionSimulationProfile(BaseModel):
    """Inspectable passive execution simulation profile."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    execution_simulation_profile_id: str = Field(min_length=1, max_length=120)
    profile_name: str = Field(min_length=1, max_length=150)
    simulation_scope: ExecutionSimulationScope
    source_provider_selection_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_hitl_decision_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_auto_mode_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_execution_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    simulated_inputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    simulated_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    simulation_surface_refs: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    authority_boundary: str = Field(
        default=EXECUTION_SIMULATOR_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1100,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_EXECUTION_SIMULATOR_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    simulation_runtime_execution_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_transition_execution_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    serialization_version: Literal["execution_simulation_profile.v1"] = (
        EXECUTION_SIMULATION_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class ExecutionSimulatorRegistry(BaseModel):
    """Stable passive registry for V4.4 Hybrid Studio execution simulation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["execution_simulator_registry"] = "execution_simulator_registry"
    serialization_version: Literal["execution_simulator_registry.v1"] = (
        EXECUTION_SIMULATOR_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=EXECUTION_SIMULATOR_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1100,
    )
    simulation_profiles: tuple[ExecutionSimulationProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    execution_simulation_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    simulation_scopes: tuple[ExecutionSimulationScope, ...] = Field(
        min_length=4,
        max_length=4,
    )
    provider_selection_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    hitl_decision_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    auto_mode_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    execution_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    simulation_surface_refs: tuple[str, ...] = Field(min_length=5, max_length=5)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_EXECUTION_SIMULATOR_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    simulation_runtime_execution_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_transition_execution_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.execution_simulation_profile_id
            for profile in self.simulation_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("execution_simulation_profile_ids must be unique")
        if self.execution_simulation_profile_ids != derived_profile_ids:
            raise ValueError(
                "execution_simulation_profile_ids must match simulation_profiles"
            )
        if self.profile_count != len(self.simulation_profiles):
            raise ValueError("profile_count must match simulation_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.simulation_scopes != tuple(
            profile.simulation_scope for profile in self.simulation_profiles
        ):
            raise ValueError("simulation_scopes must match simulation_profiles")

        known_routes = set(self.route_names)
        known_provider_profiles = set(self.provider_selection_profile_ids)
        known_hitl_profiles = set(self.hitl_decision_profile_ids)
        known_auto_profiles = set(self.auto_mode_profile_ids)
        known_execution_profiles = set(self.execution_profile_ids)
        known_surfaces = set(self.simulation_surface_refs)
        profile_sources = {
            source_registry
            for profile in self.simulation_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match simulator sources")

        for profile in self.simulation_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("profile source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.source_provider_selection_profile_ids).issubset(
                known_provider_profiles
            ):
                raise ValueError(
                    "source_provider_selection_profile_ids must be known provider profiles"
                )
            if not set(profile.source_hitl_decision_profile_ids).issubset(
                known_hitl_profiles
            ):
                raise ValueError(
                    "source_hitl_decision_profile_ids must be known HITL profiles"
                )
            if not set(profile.source_auto_mode_profile_ids).issubset(
                known_auto_profiles
            ):
                raise ValueError(
                    "source_auto_mode_profile_ids must be known Auto Mode profiles"
                )
            if not set(profile.source_execution_profile_ids).issubset(
                known_execution_profiles
            ):
                raise ValueError(
                    "source_execution_profile_ids must be known execution profiles"
                )
            if not set(profile.simulation_surface_refs).issubset(known_surfaces):
                raise ValueError(
                    "simulation_surface_refs must be known registry surfaces"
                )
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
        return self


def execution_simulator_registry() -> ExecutionSimulatorRegistry:
    """Return passive V4.4 Hybrid Studio execution simulator metadata."""

    return EXECUTION_SIMULATOR_REGISTRY


def execution_simulation_profile_by_id(
    execution_simulation_profile_id: str,
    registry: ExecutionSimulatorRegistry | None = None,
) -> ExecutionSimulationProfile | None:
    """Return one execution simulation profile without running it."""

    source_registry = registry or EXECUTION_SIMULATOR_REGISTRY
    for profile in source_registry.simulation_profiles:
        if profile.execution_simulation_profile_id == execution_simulation_profile_id:
            return profile
    return None


def execution_simulation_profiles_for_route(
    route: RouteName | str,
    registry: ExecutionSimulatorRegistry | None = None,
) -> tuple[ExecutionSimulationProfile, ...]:
    """Return passive execution simulation profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or EXECUTION_SIMULATOR_REGISTRY
    return tuple(
        profile
        for profile in source_registry.simulation_profiles
        if route_name in profile.route_applicability
    )


def _execution_simulation_profile(
    *,
    execution_simulation_profile_id: str,
    profile_name: str,
    simulation_scope: ExecutionSimulationScope,
    source_provider_selection_profile_ids: tuple[str, ...],
    source_hitl_decision_profile_ids: tuple[str, ...],
    source_auto_mode_profile_ids: tuple[str, ...],
    source_execution_profile_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    simulated_inputs: tuple[str, ...],
    simulated_outputs: tuple[str, ...],
    simulation_surface_refs: tuple[str, ...],
) -> ExecutionSimulationProfile:
    return ExecutionSimulationProfile(
        execution_simulation_profile_id=execution_simulation_profile_id,
        profile_name=profile_name,
        simulation_scope=simulation_scope,
        source_provider_selection_profile_ids=source_provider_selection_profile_ids,
        source_hitl_decision_profile_ids=source_hitl_decision_profile_ids,
        source_auto_mode_profile_ids=source_auto_mode_profile_ids,
        source_execution_profile_ids=source_execution_profile_ids,
        route_applicability=route_applicability,
        simulated_inputs=simulated_inputs,
        simulated_outputs=simulated_outputs,
        simulation_surface_refs=simulation_surface_refs,
        source_registries=_EXECUTION_SIMULATOR_SOURCE_REGISTRIES,
        observability_surfaces=_EXECUTION_SIMULATOR_OBSERVABILITY_SURFACES,
    )


EXECUTION_SIMULATION_PROFILES = (
    _execution_simulation_profile(
        execution_simulation_profile_id="route_preview_simulation_profile",
        profile_name="Route Preview Simulation Profile",
        simulation_scope="route_preview",
        source_provider_selection_profile_ids=(
            "current_config_provider_visibility_profile",
        ),
        source_hitl_decision_profile_ids=("hitl_visibility_decision_profile",),
        source_auto_mode_profile_ids=("auto_mode_observe_only_profile",),
        source_execution_profile_ids=("operator_selected_context_profile",),
        route_applicability=tuple(RouteName),
        simulated_inputs=(
            "route_decision_metadata",
            "current_provider_visibility_metadata",
            "studio_inspection_summary",
        ),
        simulated_outputs=(
            "route_preview_simulation_metadata",
            "provider_visibility_snapshot",
            "no_execution_notice",
        ),
        simulation_surface_refs=(
            "execution_simulator_panel",
            "provider_selection_panel",
            "comparison_panel",
        ),
    ),
    _execution_simulation_profile(
        execution_simulation_profile_id="local_cloud_comparison_simulation_profile",
        profile_name="Local Cloud Comparison Simulation Profile",
        simulation_scope="local_cloud_comparison",
        source_provider_selection_profile_ids=(
            "local_candidate_provider_visibility_profile",
            "cloud_candidate_provider_visibility_profile",
        ),
        source_hitl_decision_profile_ids=("hitl_risk_review_decision_profile",),
        source_auto_mode_profile_ids=("auto_mode_simulation_profile",),
        source_execution_profile_ids=("side_by_side_comparison_profile",),
        route_applicability=(
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        simulated_inputs=(
            "local_candidate_visibility_metadata",
            "cloud_candidate_visibility_metadata",
            "comparison_view_metadata",
        ),
        simulated_outputs=(
            "local_cloud_comparison_simulation_metadata",
            "side_by_side_non_execution_plan",
            "manual_review_context",
        ),
        simulation_surface_refs=(
            "execution_simulator_panel",
            "hybrid_execution_panel",
            "comparison_panel",
        ),
    ),
    _execution_simulation_profile(
        execution_simulation_profile_id="hitl_review_simulation_profile",
        profile_name="HITL Review Simulation Profile",
        simulation_scope="hitl_review",
        source_provider_selection_profile_ids=(
            "operator_override_provider_visibility_profile",
        ),
        source_hitl_decision_profile_ids=(
            "hitl_confirmation_decision_profile",
            "hitl_final_review_decision_profile",
        ),
        source_auto_mode_profile_ids=("auto_mode_operator_confirmed_profile",),
        source_execution_profile_ids=tuple(
            HYBRID_EXECUTION_REGISTRY.execution_profile_ids
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        simulated_inputs=(
            "operator_override_visibility_metadata",
            "confirmation_recommended_metadata",
            "final_review_recommended_metadata",
        ),
        simulated_outputs=(
            "hitl_review_simulation_metadata",
            "manual_signoff_context",
            "no_human_request_notice",
        ),
        simulation_surface_refs=(
            "execution_simulator_panel",
            "hitl_decision_panel",
            "provider_selection_panel",
        ),
    ),
    _execution_simulation_profile(
        execution_simulation_profile_id="provider_selection_simulation_profile",
        profile_name="Provider Selection Simulation Profile",
        simulation_scope="provider_selection",
        source_provider_selection_profile_ids=tuple(
            PROVIDER_SELECTION_REGISTRY.provider_selection_profile_ids
        ),
        source_hitl_decision_profile_ids=(
            "hitl_visibility_decision_profile",
            "hitl_confirmation_decision_profile",
        ),
        source_auto_mode_profile_ids=(
            "auto_mode_suggestion_profile",
            "auto_mode_operator_confirmed_profile",
        ),
        source_execution_profile_ids=tuple(
            HYBRID_EXECUTION_REGISTRY.execution_profile_ids
        ),
        route_applicability=tuple(RouteName),
        simulated_inputs=(
            "provider_candidate_visibility_metadata",
            "manual_selection_boundary_notice",
            "auto_mode_suggestion_metadata",
        ),
        simulated_outputs=(
            "provider_selection_simulation_metadata",
            "candidate_selection_plan_metadata",
            "no_provider_switch_notice",
        ),
        simulation_surface_refs=tuple(_EXECUTION_SIMULATOR_SURFACES),
    ),
)

EXECUTION_SIMULATOR_REGISTRY = ExecutionSimulatorRegistry(
    simulation_profiles=EXECUTION_SIMULATION_PROFILES,
    execution_simulation_profile_ids=tuple(
        profile.execution_simulation_profile_id
        for profile in EXECUTION_SIMULATION_PROFILES
    ),
    simulation_scopes=tuple(
        profile.simulation_scope for profile in EXECUTION_SIMULATION_PROFILES
    ),
    provider_selection_profile_ids=tuple(
        PROVIDER_SELECTION_REGISTRY.provider_selection_profile_ids
    ),
    hitl_decision_profile_ids=tuple(HITL_DECISION_REGISTRY.hitl_decision_profile_ids),
    auto_mode_profile_ids=tuple(AUTO_MODE_REGISTRY.auto_mode_profile_ids),
    execution_profile_ids=tuple(HYBRID_EXECUTION_REGISTRY.execution_profile_ids),
    simulation_surface_refs=_EXECUTION_SIMULATOR_SURFACES,
    route_names=tuple(RouteName),
    profile_count=len(EXECUTION_SIMULATION_PROFILES),
    source_registries=_EXECUTION_SIMULATOR_SOURCE_REGISTRIES,
    observability_surfaces=_EXECUTION_SIMULATOR_OBSERVABILITY_SURFACES,
)

ModelProfileKind = Literal[
    "fast_iteration",
    "creative_reasoning",
    "code_assistance",
    "evaluation_review",
]

MODEL_PROFILE_SERIALIZATION_VERSION = "model_profile.v1"
MODEL_PROFILE_REGISTRY_SERIALIZATION_VERSION = "model_profile_registry.v1"
MODEL_PROFILE_REGISTRY_AUTHORITY_BOUNDARY = (
    "Model Profiles metadata describes passive local and cloud model capability "
    "profiles for V4.4 Hybrid Studio inspection only; it does not select "
    "models, route providers or models, execute providers, calculate cost or "
    "quality scores, optimize execution, trigger retries, mutate prompts, "
    "write replay storage, or modify generated output."
)

_MODEL_PROFILE_SOURCE_REGISTRIES = (
    "local_model_registry",
    "cloud_model_registry",
    "provider_selection_registry",
    "execution_simulator_registry",
    "settings_generation_provider_config",
    "hybrid_agentic_workflow_registry",
)

_MODEL_PROFILE_SURFACES = (
    "model_profile_panel",
    "model_catalog_panel",
    "provider_selection_panel",
    "execution_simulator_panel",
)

_MODEL_PROFILE_OBSERVABILITY_SURFACES = (
    "model_profile_id",
    "model_profile_kind",
    "source_local_surface_ids",
    "source_cloud_surface_ids",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_MODEL_PROFILE_BLOCKED_RUNTIME_BEHAVIORS = (
    "model_profile_execution",
    "model_selection",
    "provider_or_model_routing",
    "provider_execution",
    "cost_scoring",
    "quality_scoring",
    "execution_optimization",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_replay_storage",
    "generated_output_modification",
)


class ModelProfile(BaseModel):
    """Inspectable passive model capability profile."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    model_profile_id: str = Field(min_length=1, max_length=100)
    profile_name: str = Field(min_length=1, max_length=140)
    model_profile_kind: ModelProfileKind
    source_local_surface_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=4
    )
    source_cloud_surface_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=4
    )
    provider_candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    capability_dimensions: tuple[str, ...] = Field(min_length=1, max_length=10)
    profile_inputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    profile_surface_refs: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    authority_boundary: str = Field(
        default=MODEL_PROFILE_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1100,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_MODEL_PROFILE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    model_profile_execution_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    cost_scoring_implemented: Literal[False] = False
    quality_scoring_implemented: Literal[False] = False
    execution_optimization_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    serialization_version: Literal["model_profile.v1"] = (
        MODEL_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class ModelProfileRegistry(BaseModel):
    """Stable passive registry for V4.4 Hybrid Studio model profiles."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["model_profile_registry"] = "model_profile_registry"
    serialization_version: Literal["model_profile_registry.v1"] = (
        MODEL_PROFILE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=MODEL_PROFILE_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1100,
    )
    model_profiles: tuple[ModelProfile, ...] = Field(min_length=4, max_length=4)
    model_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    model_profile_kinds: tuple[ModelProfileKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    local_surface_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    cloud_surface_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    provider_candidate_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    profile_surface_refs: tuple[str, ...] = Field(min_length=4, max_length=4)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_MODEL_PROFILE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    model_profile_execution_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    cost_scoring_implemented: Literal[False] = False
    quality_scoring_implemented: Literal[False] = False
    execution_optimization_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.model_profile_id for profile in self.model_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("model_profile_ids must be unique")
        if self.model_profile_ids != derived_profile_ids:
            raise ValueError("model_profile_ids must match model_profiles")
        if self.profile_count != len(self.model_profiles):
            raise ValueError("profile_count must match model_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.model_profile_kinds != tuple(
            profile.model_profile_kind for profile in self.model_profiles
        ):
            raise ValueError("model_profile_kinds must match model_profiles")

        known_routes = set(self.route_names)
        known_local_surfaces = set(self.local_surface_ids)
        known_cloud_surfaces = set(self.cloud_surface_ids)
        known_provider_candidates = set(self.provider_candidate_ids)
        known_surfaces = set(self.profile_surface_refs)
        profile_sources = {
            source_registry
            for profile in self.model_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match model profile sources")

        for profile in self.model_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("profile source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.source_local_surface_ids).issubset(known_local_surfaces):
                raise ValueError("source_local_surface_ids must be known local models")
            if not set(profile.source_cloud_surface_ids).issubset(known_cloud_surfaces):
                raise ValueError("source_cloud_surface_ids must be known cloud models")
            if not set(profile.provider_candidate_ids).issubset(
                known_provider_candidates
            ):
                raise ValueError("provider_candidate_ids must be known providers")
            if not set(profile.profile_surface_refs).issubset(known_surfaces):
                raise ValueError("profile_surface_refs must be known registry surfaces")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
        return self


def model_profile_registry() -> ModelProfileRegistry:
    """Return passive V4.4 Hybrid Studio model profile metadata."""

    return MODEL_PROFILE_REGISTRY


def model_profile_by_id(
    model_profile_id: str,
    registry: ModelProfileRegistry | None = None,
) -> ModelProfile | None:
    """Return one model profile without selecting or executing it."""

    source_registry = registry or MODEL_PROFILE_REGISTRY
    for profile in source_registry.model_profiles:
        if profile.model_profile_id == model_profile_id:
            return profile
    return None


def model_profiles_for_route(
    route: RouteName | str,
    registry: ModelProfileRegistry | None = None,
) -> tuple[ModelProfile, ...]:
    """Return passive model profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or MODEL_PROFILE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.model_profiles
        if route_name in profile.route_applicability
    )


def _model_profile(
    *,
    model_profile_id: str,
    profile_name: str,
    model_profile_kind: ModelProfileKind,
    source_local_surface_ids: tuple[str, ...],
    source_cloud_surface_ids: tuple[str, ...],
    provider_candidate_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    capability_dimensions: tuple[str, ...],
    profile_inputs: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
    profile_surface_refs: tuple[str, ...],
) -> ModelProfile:
    return ModelProfile(
        model_profile_id=model_profile_id,
        profile_name=profile_name,
        model_profile_kind=model_profile_kind,
        source_local_surface_ids=source_local_surface_ids,
        source_cloud_surface_ids=source_cloud_surface_ids,
        provider_candidate_ids=provider_candidate_ids,
        route_applicability=route_applicability,
        capability_dimensions=capability_dimensions,
        profile_inputs=profile_inputs,
        advisory_outputs=advisory_outputs,
        profile_surface_refs=profile_surface_refs,
        source_registries=_MODEL_PROFILE_SOURCE_REGISTRIES,
        observability_surfaces=_MODEL_PROFILE_OBSERVABILITY_SURFACES,
    )


MODEL_PROFILE_SURFACES = (
    "model_profile_panel",
    "model_catalog_panel",
    "provider_selection_panel",
    "execution_simulator_panel",
)

MODEL_PROFILES = (
    _model_profile(
        model_profile_id="fast_iteration_model_profile",
        profile_name="Fast Iteration Model Profile",
        model_profile_kind="fast_iteration",
        source_local_surface_ids=(
            "ollama_chat_surface",
            "llama_cpp_completion_surface",
        ),
        source_cloud_surface_ids=("openai_generation_model_surface",),
        provider_candidate_ids=("ollama", "llama_cpp", "openai"),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DEBUG,
            RouteName.PREVIEW,
        ),
        capability_dimensions=(
            "iteration_latency_metadata",
            "local_readiness_metadata",
            "route_preview_metadata",
        ),
        profile_inputs=(
            "local_candidate_visibility_metadata",
            "route_preview_simulation_metadata",
            "provider_boundary_metadata",
        ),
        advisory_outputs=(
            "fast_iteration_capability_profile",
            "manual_fast_path_context",
            "no_selection_notice",
        ),
        profile_surface_refs=("model_profile_panel", "model_catalog_panel"),
    ),
    _model_profile(
        model_profile_id="creative_reasoning_model_profile",
        profile_name="Creative Reasoning Model Profile",
        model_profile_kind="creative_reasoning",
        source_local_surface_ids=("lm_studio_chat_surface",),
        source_cloud_surface_ids=("openai_generation_model_surface",),
        provider_candidate_ids=("lm_studio", "openai"),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        capability_dimensions=(
            "creative_reasoning_metadata",
            "context_window_metadata",
            "studio_comparison_metadata",
        ),
        profile_inputs=(
            "cloud_candidate_visibility_metadata",
            "local_cloud_comparison_simulation_metadata",
            "creative_route_metadata",
        ),
        advisory_outputs=(
            "creative_reasoning_capability_profile",
            "manual_reasoning_selection_context",
            "no_quality_scoring_notice",
        ),
        profile_surface_refs=(
            "model_profile_panel",
            "provider_selection_panel",
            "execution_simulator_panel",
        ),
    ),
    _model_profile(
        model_profile_id="code_assistance_model_profile",
        profile_name="Code Assistance Model Profile",
        model_profile_kind="code_assistance",
        source_local_surface_ids=("llama_cpp_completion_surface",),
        source_cloud_surface_ids=("openai_generation_model_surface",),
        provider_candidate_ids=("llama_cpp", "openai"),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        capability_dimensions=(
            "code_context_metadata",
            "runtime_diagnostic_metadata",
            "debug_route_metadata",
        ),
        profile_inputs=(
            "local_runtime_readiness_metadata",
            "provider_selection_simulation_metadata",
            "runtime_route_metadata",
        ),
        advisory_outputs=(
            "code_assistance_capability_profile",
            "manual_debug_model_context",
            "no_runtime_execution_notice",
        ),
        profile_surface_refs=("model_profile_panel", "execution_simulator_panel"),
    ),
    _model_profile(
        model_profile_id="evaluation_review_model_profile",
        profile_name="Evaluation Review Model Profile",
        model_profile_kind="evaluation_review",
        source_local_surface_ids=("local_transformers_multimodal_surface",),
        source_cloud_surface_ids=(
            "ragas_evaluator_model_surface",
            "provider_reported_response_model_surface",
        ),
        provider_candidate_ids=("local_transformers", "openai"),
        route_applicability=(
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        capability_dimensions=(
            "review_metadata",
            "evaluation_surface_metadata",
            "provider_response_metadata",
        ),
        profile_inputs=(
            "hitl_review_simulation_metadata",
            "evaluation_dataset_metadata",
            "provider_telemetry_metadata",
        ),
        advisory_outputs=(
            "evaluation_review_capability_profile",
            "manual_review_model_context",
            "no_evaluator_call_notice",
        ),
        profile_surface_refs=(
            "model_profile_panel",
            "model_catalog_panel",
            "execution_simulator_panel",
        ),
    ),
)

MODEL_PROFILE_REGISTRY = ModelProfileRegistry(
    model_profiles=MODEL_PROFILES,
    model_profile_ids=tuple(profile.model_profile_id for profile in MODEL_PROFILES),
    model_profile_kinds=tuple(profile.model_profile_kind for profile in MODEL_PROFILES),
    local_surface_ids=tuple(LOCAL_MODEL_REGISTRY.surface_ids),
    cloud_surface_ids=tuple(CLOUD_MODEL_REGISTRY.surface_ids),
    provider_candidate_ids=PROVIDER_SELECTION_CANDIDATE_IDS,
    profile_surface_refs=MODEL_PROFILE_SURFACES,
    route_names=tuple(RouteName),
    profile_count=len(MODEL_PROFILES),
    source_registries=_MODEL_PROFILE_SOURCE_REGISTRIES,
    observability_surfaces=_MODEL_PROFILE_OBSERVABILITY_SURFACES,
)

CostProfileKind = Literal[
    "planning_iteration_budget",
    "creative_reasoning_budget",
    "curation_refinement_budget",
    "final_review_budget",
]
CostProfileBand = Literal["medium", "high", "guarded", "low"]

COST_PROFILE_SERIALIZATION_VERSION = "cost_profile.v1"
COST_PROFILE_REGISTRY_SERIALIZATION_VERSION = "cost_profile_registry.v1"
COST_PROFILE_REGISTRY_AUTHORITY_BOUNDARY = (
    "Cost Profiles metadata describes passive cost posture, budget context, "
    "and source cost-threshold references for V4.4 Hybrid Studio inspection "
    "only; it does not calculate cost scores, look up provider pricing, "
    "enforce budgets, optimize execution, route by cost, select providers or "
    "models, execute providers, trigger retries, mutate prompts, write replay "
    "storage, or modify generated output."
)

_COST_PROFILE_SOURCE_REGISTRIES = (
    "model_profile_registry",
    "provider_selection_registry",
    "cost_threshold_routing_registry",
    "local_model_registry",
    "cloud_model_registry",
    "execution_simulator_registry",
)

_COST_PROFILE_SURFACES = (
    "cost_profile_panel",
    "model_profile_panel",
    "provider_selection_panel",
    "execution_simulator_panel",
    "budget_boundary_panel",
)

_COST_PROFILE_OBSERVABILITY_SURFACES = (
    "cost_profile_id",
    "cost_profile_kind",
    "cost_band",
    "source_model_profile_ids",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_COST_PROFILE_BLOCKED_RUNTIME_BEHAVIORS = (
    "cost_profile_execution",
    "cost_scoring",
    "pricing_lookup",
    "budget_enforcement",
    "cost_based_routing",
    "provider_or_model_routing",
    "provider_execution",
    "model_selection",
    "execution_optimization",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_replay_storage",
    "generated_output_modification",
)


class CostProfile(BaseModel):
    """Inspectable passive cost profile for Hybrid Studio."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    cost_profile_id: str = Field(min_length=1, max_length=100)
    profile_name: str = Field(min_length=1, max_length=140)
    cost_profile_kind: CostProfileKind
    cost_band: CostProfileBand
    advisory_cost_range: tuple[int, int]
    source_model_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_provider_selection_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_cost_threshold_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_local_surface_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    source_cloud_surface_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    cost_dimensions: tuple[str, ...] = Field(min_length=1, max_length=10)
    cost_inputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    cost_surface_refs: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    authority_boundary: str = Field(
        default=COST_PROFILE_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_COST_PROFILE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    cost_profile_execution_implemented: Literal[False] = False
    cost_scoring_implemented: Literal[False] = False
    pricing_lookup_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    cost_based_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    execution_optimization_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    serialization_version: Literal["cost_profile.v1"] = (
        COST_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class CostProfileRegistry(BaseModel):
    """Stable passive registry for V4.4 Hybrid Studio cost profiles."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cost_profile_registry"] = "cost_profile_registry"
    serialization_version: Literal["cost_profile_registry.v1"] = (
        COST_PROFILE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COST_PROFILE_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    cost_profiles: tuple[CostProfile, ...] = Field(min_length=4, max_length=4)
    cost_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    cost_profile_kinds: tuple[CostProfileKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    cost_bands: tuple[CostProfileBand, ...] = Field(min_length=4, max_length=4)
    model_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    provider_selection_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    cost_threshold_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    local_surface_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    cloud_surface_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    cost_surface_refs: tuple[str, ...] = Field(min_length=5, max_length=5)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_COST_PROFILE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    cost_profile_execution_implemented: Literal[False] = False
    cost_scoring_implemented: Literal[False] = False
    pricing_lookup_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    cost_based_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    execution_optimization_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.cost_profile_id for profile in self.cost_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("cost_profile_ids must be unique")
        if self.cost_profile_ids != derived_profile_ids:
            raise ValueError("cost_profile_ids must match cost_profiles")
        if self.profile_count != len(self.cost_profiles):
            raise ValueError("profile_count must match cost_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.cost_profile_kinds != tuple(
            profile.cost_profile_kind for profile in self.cost_profiles
        ):
            raise ValueError("cost_profile_kinds must match cost_profiles")
        if self.cost_bands != tuple(
            profile.cost_band for profile in self.cost_profiles
        ):
            raise ValueError("cost_bands must match cost_profiles")

        known_routes = set(self.route_names)
        known_model_profiles = set(self.model_profile_ids)
        known_provider_profiles = set(self.provider_selection_profile_ids)
        known_cost_thresholds = set(self.cost_threshold_profile_ids)
        known_local_surfaces = set(self.local_surface_ids)
        known_cloud_surfaces = set(self.cloud_surface_ids)
        known_cost_surfaces = set(self.cost_surface_refs)
        profile_sources = {
            source_registry
            for profile in self.cost_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match cost profile sources")

        for profile in self.cost_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("profile source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.source_model_profile_ids).issubset(known_model_profiles):
                raise ValueError("source_model_profile_ids must be known profiles")
            if not set(profile.source_provider_selection_profile_ids).issubset(
                known_provider_profiles
            ):
                raise ValueError(
                    "source_provider_selection_profile_ids must be known profiles"
                )
            if not set(profile.source_cost_threshold_profile_ids).issubset(
                known_cost_thresholds
            ):
                raise ValueError(
                    "source_cost_threshold_profile_ids must be known profiles"
                )
            if not set(profile.source_local_surface_ids).issubset(known_local_surfaces):
                raise ValueError("source_local_surface_ids must be known local models")
            if not set(profile.source_cloud_surface_ids).issubset(known_cloud_surfaces):
                raise ValueError("source_cloud_surface_ids must be known cloud models")
            if not set(profile.cost_surface_refs).issubset(known_cost_surfaces):
                raise ValueError("cost_surface_refs must be known registry surfaces")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
            cost_low, cost_high = profile.advisory_cost_range
            if not 0 <= cost_low <= cost_high:
                raise ValueError("advisory cost range must be non-negative")
        return self


def cost_profile_registry() -> CostProfileRegistry:
    """Return passive V4.4 Hybrid Studio cost profile metadata."""

    return COST_PROFILE_REGISTRY


def cost_profile_by_id(
    cost_profile_id: str,
    registry: CostProfileRegistry | None = None,
) -> CostProfile | None:
    """Return one cost profile without scoring or enforcing it."""

    source_registry = registry or COST_PROFILE_REGISTRY
    for profile in source_registry.cost_profiles:
        if profile.cost_profile_id == cost_profile_id:
            return profile
    return None


def cost_profiles_for_route(
    route: RouteName | str,
    registry: CostProfileRegistry | None = None,
) -> tuple[CostProfile, ...]:
    """Return passive cost profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or COST_PROFILE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.cost_profiles
        if route_name in profile.route_applicability
    )


def cost_profiles_for_band(
    cost_band: CostProfileBand | str,
    registry: CostProfileRegistry | None = None,
) -> tuple[CostProfile, ...]:
    """Return passive cost profiles for an advisory cost band."""

    band_value = str(cost_band).strip()
    source_registry = registry or COST_PROFILE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.cost_profiles
        if profile.cost_band == band_value
    )


def _cost_profile(
    *,
    cost_profile_id: str,
    profile_name: str,
    cost_profile_kind: CostProfileKind,
    cost_band: CostProfileBand,
    advisory_cost_range: tuple[int, int],
    source_model_profile_ids: tuple[str, ...],
    source_provider_selection_profile_ids: tuple[str, ...],
    source_cost_threshold_profile_ids: tuple[str, ...],
    source_local_surface_ids: tuple[str, ...],
    source_cloud_surface_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    cost_dimensions: tuple[str, ...],
    cost_inputs: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
    cost_surface_refs: tuple[str, ...],
) -> CostProfile:
    return CostProfile(
        cost_profile_id=cost_profile_id,
        profile_name=profile_name,
        cost_profile_kind=cost_profile_kind,
        cost_band=cost_band,
        advisory_cost_range=advisory_cost_range,
        source_model_profile_ids=source_model_profile_ids,
        source_provider_selection_profile_ids=source_provider_selection_profile_ids,
        source_cost_threshold_profile_ids=source_cost_threshold_profile_ids,
        source_local_surface_ids=source_local_surface_ids,
        source_cloud_surface_ids=source_cloud_surface_ids,
        route_applicability=route_applicability,
        cost_dimensions=cost_dimensions,
        cost_inputs=cost_inputs,
        advisory_outputs=advisory_outputs,
        cost_surface_refs=cost_surface_refs,
        source_registries=_COST_PROFILE_SOURCE_REGISTRIES,
        observability_surfaces=_COST_PROFILE_OBSERVABILITY_SURFACES,
    )


COST_PROFILE_SURFACES = (
    "cost_profile_panel",
    "model_profile_panel",
    "provider_selection_panel",
    "execution_simulator_panel",
    "budget_boundary_panel",
)

COST_PROFILES = (
    _cost_profile(
        cost_profile_id="planning_iteration_cost_profile",
        profile_name="Planning Iteration Cost Profile",
        cost_profile_kind="planning_iteration_budget",
        cost_band="medium",
        advisory_cost_range=(2, 4),
        source_model_profile_ids=("fast_iteration_model_profile",),
        source_provider_selection_profile_ids=(
            "current_config_provider_visibility_profile",
            "local_candidate_provider_visibility_profile",
        ),
        source_cost_threshold_profile_ids=(
            "cost_threshold_routing::planning_execution_fit",
        ),
        source_local_surface_ids=(
            "ollama_chat_surface",
            "llama_cpp_completion_surface",
        ),
        source_cloud_surface_ids=("openai_generation_model_surface",),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DEBUG,
            RouteName.PREVIEW,
        ),
        cost_dimensions=(
            "iteration_budget_metadata",
            "local_hardware_cost_posture",
            "cloud_metered_fallback_context",
        ),
        cost_inputs=(
            "fast_iteration_capability_profile",
            "planning_token_budget_context",
            "current_provider_visibility_metadata",
        ),
        advisory_outputs=(
            "planning_iteration_cost_context",
            "manual_budget_review_hint",
            "no_cost_scoring_notice",
        ),
        cost_surface_refs=("cost_profile_panel", "model_profile_panel"),
    ),
    _cost_profile(
        cost_profile_id="creative_reasoning_cost_profile",
        profile_name="Creative Reasoning Cost Profile",
        cost_profile_kind="creative_reasoning_budget",
        cost_band="high",
        advisory_cost_range=(4, 7),
        source_model_profile_ids=("creative_reasoning_model_profile",),
        source_provider_selection_profile_ids=(
            "local_candidate_provider_visibility_profile",
            "cloud_candidate_provider_visibility_profile",
        ),
        source_cost_threshold_profile_ids=(
            "cost_threshold_routing::style_aesthetic_alignment",
        ),
        source_local_surface_ids=("lm_studio_chat_surface",),
        source_cloud_surface_ids=("openai_generation_model_surface",),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        cost_dimensions=(
            "creative_reasoning_budget_metadata",
            "style_variant_budget_context",
            "provider_metered_cost_posture",
        ),
        cost_inputs=(
            "creative_reasoning_capability_profile",
            "style_variant_budget_context",
            "cloud_candidate_visibility_metadata",
        ),
        advisory_outputs=(
            "creative_reasoning_cost_context",
            "manual_high_cost_review_hint",
            "no_pricing_lookup_notice",
        ),
        cost_surface_refs=(
            "cost_profile_panel",
            "provider_selection_panel",
            "budget_boundary_panel",
        ),
    ),
    _cost_profile(
        cost_profile_id="curation_refinement_cost_profile",
        profile_name="Curation Refinement Cost Profile",
        cost_profile_kind="curation_refinement_budget",
        cost_band="guarded",
        advisory_cost_range=(3, 5),
        source_model_profile_ids=(
            "code_assistance_model_profile",
            "evaluation_review_model_profile",
        ),
        source_provider_selection_profile_ids=(
            "cloud_candidate_provider_visibility_profile",
            "operator_override_provider_visibility_profile",
        ),
        source_cost_threshold_profile_ids=(
            "cost_threshold_routing::curation_refinement_need",
        ),
        source_local_surface_ids=(
            "llama_cpp_completion_surface",
            "local_transformers_multimodal_surface",
        ),
        source_cloud_surface_ids=(
            "openai_generation_model_surface",
            "ragas_evaluator_model_surface",
            "provider_reported_response_model_surface",
        ),
        route_applicability=(
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        cost_dimensions=(
            "refinement_budget_metadata",
            "evaluation_review_budget_context",
            "operator_override_cost_visibility",
        ),
        cost_inputs=(
            "code_assistance_capability_profile",
            "evaluation_review_capability_profile",
            "refinement_budget_context",
        ),
        advisory_outputs=(
            "curation_refinement_cost_context",
            "manual_guarded_budget_review_hint",
            "no_budget_enforcement_notice",
        ),
        cost_surface_refs=(
            "cost_profile_panel",
            "execution_simulator_panel",
            "budget_boundary_panel",
        ),
    ),
    _cost_profile(
        cost_profile_id="final_review_cost_profile",
        profile_name="Final Review Cost Profile",
        cost_profile_kind="final_review_budget",
        cost_band="low",
        advisory_cost_range=(0, 2),
        source_model_profile_ids=("evaluation_review_model_profile",),
        source_provider_selection_profile_ids=(
            "current_config_provider_visibility_profile",
            "operator_override_provider_visibility_profile",
        ),
        source_cost_threshold_profile_ids=(
            "cost_threshold_routing::final_synthesis_readiness",
        ),
        source_local_surface_ids=("local_transformers_multimodal_surface",),
        source_cloud_surface_ids=(
            "ragas_evaluator_model_surface",
            "provider_reported_response_model_surface",
        ),
        route_applicability=(
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        cost_dimensions=(
            "final_review_budget_metadata",
            "final_synthesis_budget_context",
            "evaluation_call_boundary_metadata",
        ),
        cost_inputs=(
            "evaluation_review_capability_profile",
            "final_cost_routing_context",
            "provider_response_metadata",
        ),
        advisory_outputs=(
            "final_review_cost_context",
            "manual_low_cost_review_hint",
            "no_evaluator_call_notice",
        ),
        cost_surface_refs=(
            "cost_profile_panel",
            "model_profile_panel",
            "budget_boundary_panel",
        ),
    ),
)

COST_PROFILE_REGISTRY = CostProfileRegistry(
    cost_profiles=COST_PROFILES,
    cost_profile_ids=tuple(profile.cost_profile_id for profile in COST_PROFILES),
    cost_profile_kinds=tuple(profile.cost_profile_kind for profile in COST_PROFILES),
    cost_bands=tuple(profile.cost_band for profile in COST_PROFILES),
    model_profile_ids=tuple(MODEL_PROFILE_REGISTRY.model_profile_ids),
    provider_selection_profile_ids=tuple(
        PROVIDER_SELECTION_REGISTRY.provider_selection_profile_ids
    ),
    cost_threshold_profile_ids=tuple(
        COST_THRESHOLD_ROUTING_REGISTRY.cost_threshold_profile_ids
    ),
    local_surface_ids=tuple(LOCAL_MODEL_REGISTRY.surface_ids),
    cloud_surface_ids=tuple(CLOUD_MODEL_REGISTRY.surface_ids),
    cost_surface_refs=COST_PROFILE_SURFACES,
    route_names=tuple(RouteName),
    profile_count=len(COST_PROFILES),
    source_registries=_COST_PROFILE_SOURCE_REGISTRIES,
    observability_surfaces=_COST_PROFILE_OBSERVABILITY_SURFACES,
)

QualityProfileKind = Literal[
    "planning_quality_review",
    "creative_quality_review",
    "refinement_quality_review",
    "final_review_quality",
]
QualityProfileLevel = Literal["medium", "high", "critical", "low"]

QUALITY_PROFILE_SERIALIZATION_VERSION = "quality_profile.v1"
QUALITY_PROFILE_REGISTRY_SERIALIZATION_VERSION = "quality_profile_registry.v1"
QUALITY_PROFILE_REGISTRY_AUTHORITY_BOUNDARY = (
    "Quality Profiles metadata describes passive quality review context, "
    "quality escalation source references, and studio-visible quality signals "
    "for V4.4 Hybrid Studio inspection only; it does not calculate quality "
    "scores, evaluate quality, execute quality escalation, trigger refinement, "
    "route providers or models, select models, execute providers, request "
    "human input, trigger retries, mutate prompts, write replay storage, or "
    "modify generated output."
)

_QUALITY_PROFILE_SOURCE_REGISTRIES = (
    "model_profile_registry",
    "cost_profile_registry",
    "provider_selection_registry",
    "quality_escalation_registry",
    "execution_simulator_registry",
    "hitl_decision_registry",
)

_QUALITY_PROFILE_SURFACES = (
    "quality_profile_panel",
    "model_profile_panel",
    "cost_profile_panel",
    "execution_simulator_panel",
    "hitl_decision_panel",
)

_QUALITY_PROFILE_OBSERVABILITY_SURFACES = (
    "quality_profile_id",
    "quality_profile_kind",
    "quality_level",
    "source_model_profile_ids",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_QUALITY_PROFILE_BLOCKED_RUNTIME_BEHAVIORS = (
    "quality_profile_execution",
    "quality_scoring",
    "quality_evaluation",
    "quality_escalation",
    "refinement_triggering",
    "provider_or_model_routing",
    "provider_execution",
    "model_selection",
    "cost_optimization",
    "workflow_control",
    "human_input_request",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_replay_storage",
    "generated_output_modification",
)


class QualityProfile(BaseModel):
    """Inspectable passive quality profile for Hybrid Studio."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    quality_profile_id: str = Field(min_length=1, max_length=110)
    profile_name: str = Field(min_length=1, max_length=140)
    quality_profile_kind: QualityProfileKind
    quality_level: QualityProfileLevel
    source_model_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_cost_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_provider_selection_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_quality_escalation_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_execution_simulation_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_hitl_decision_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    quality_dimensions: tuple[str, ...] = Field(min_length=1, max_length=10)
    quality_inputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    quality_surface_refs: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    authority_boundary: str = Field(
        default=QUALITY_PROFILE_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1300,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_QUALITY_PROFILE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    quality_profile_execution_implemented: Literal[False] = False
    quality_scoring_implemented: Literal[False] = False
    quality_evaluation_implemented: Literal[False] = False
    quality_escalation_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    cost_optimization_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    serialization_version: Literal["quality_profile.v1"] = (
        QUALITY_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class QualityProfileRegistry(BaseModel):
    """Stable passive registry for V4.4 Hybrid Studio quality profiles."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["quality_profile_registry"] = "quality_profile_registry"
    serialization_version: Literal["quality_profile_registry.v1"] = (
        QUALITY_PROFILE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=QUALITY_PROFILE_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1300,
    )
    quality_profiles: tuple[QualityProfile, ...] = Field(min_length=4, max_length=4)
    quality_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    quality_profile_kinds: tuple[QualityProfileKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    quality_levels: tuple[QualityProfileLevel, ...] = Field(
        min_length=4,
        max_length=4,
    )
    model_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    cost_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    provider_selection_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    quality_escalation_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    execution_simulation_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    hitl_decision_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    quality_surface_refs: tuple[str, ...] = Field(min_length=5, max_length=5)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_QUALITY_PROFILE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    quality_profile_execution_implemented: Literal[False] = False
    quality_scoring_implemented: Literal[False] = False
    quality_evaluation_implemented: Literal[False] = False
    quality_escalation_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    cost_optimization_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.quality_profile_id for profile in self.quality_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("quality_profile_ids must be unique")
        if self.quality_profile_ids != derived_profile_ids:
            raise ValueError("quality_profile_ids must match quality_profiles")
        if self.profile_count != len(self.quality_profiles):
            raise ValueError("profile_count must match quality_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.quality_profile_kinds != tuple(
            profile.quality_profile_kind for profile in self.quality_profiles
        ):
            raise ValueError("quality_profile_kinds must match quality_profiles")
        if self.quality_levels != tuple(
            profile.quality_level for profile in self.quality_profiles
        ):
            raise ValueError("quality_levels must match quality_profiles")

        known_routes = set(self.route_names)
        known_model_profiles = set(self.model_profile_ids)
        known_cost_profiles = set(self.cost_profile_ids)
        known_provider_profiles = set(self.provider_selection_profile_ids)
        known_quality_escalations = set(self.quality_escalation_profile_ids)
        known_simulation_profiles = set(self.execution_simulation_profile_ids)
        known_hitl_profiles = set(self.hitl_decision_profile_ids)
        known_quality_surfaces = set(self.quality_surface_refs)
        profile_sources = {
            source_registry
            for profile in self.quality_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match quality profile sources")

        for profile in self.quality_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("profile source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.source_model_profile_ids).issubset(known_model_profiles):
                raise ValueError("source_model_profile_ids must be known profiles")
            if not set(profile.source_cost_profile_ids).issubset(known_cost_profiles):
                raise ValueError("source_cost_profile_ids must be known profiles")
            if not set(profile.source_provider_selection_profile_ids).issubset(
                known_provider_profiles
            ):
                raise ValueError(
                    "source_provider_selection_profile_ids must be known profiles"
                )
            if not set(profile.source_quality_escalation_profile_ids).issubset(
                known_quality_escalations
            ):
                raise ValueError(
                    "source_quality_escalation_profile_ids must be known profiles"
                )
            if not set(profile.source_execution_simulation_profile_ids).issubset(
                known_simulation_profiles
            ):
                raise ValueError(
                    "source_execution_simulation_profile_ids must be known profiles"
                )
            if not set(profile.source_hitl_decision_profile_ids).issubset(
                known_hitl_profiles
            ):
                raise ValueError(
                    "source_hitl_decision_profile_ids must be known profiles"
                )
            if not set(profile.quality_surface_refs).issubset(known_quality_surfaces):
                raise ValueError("quality_surface_refs must be known registry surfaces")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
        return self


def quality_profile_registry() -> QualityProfileRegistry:
    """Return passive V4.4 Hybrid Studio quality profile metadata."""

    return QUALITY_PROFILE_REGISTRY


def quality_profile_by_id(
    quality_profile_id: str,
    registry: QualityProfileRegistry | None = None,
) -> QualityProfile | None:
    """Return one quality profile without evaluating or scoring it."""

    source_registry = registry or QUALITY_PROFILE_REGISTRY
    for profile in source_registry.quality_profiles:
        if profile.quality_profile_id == quality_profile_id:
            return profile
    return None


def quality_profiles_for_route(
    route: RouteName | str,
    registry: QualityProfileRegistry | None = None,
) -> tuple[QualityProfile, ...]:
    """Return passive quality profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or QUALITY_PROFILE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.quality_profiles
        if route_name in profile.route_applicability
    )


def quality_profiles_for_level(
    quality_level: QualityProfileLevel | str,
    registry: QualityProfileRegistry | None = None,
) -> tuple[QualityProfile, ...]:
    """Return passive quality profiles for an advisory quality level."""

    level_value = str(quality_level).strip()
    source_registry = registry or QUALITY_PROFILE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.quality_profiles
        if profile.quality_level == level_value
    )


def _quality_profile(
    *,
    quality_profile_id: str,
    profile_name: str,
    quality_profile_kind: QualityProfileKind,
    quality_level: QualityProfileLevel,
    source_model_profile_ids: tuple[str, ...],
    source_cost_profile_ids: tuple[str, ...],
    source_provider_selection_profile_ids: tuple[str, ...],
    source_quality_escalation_profile_ids: tuple[str, ...],
    source_execution_simulation_profile_ids: tuple[str, ...],
    source_hitl_decision_profile_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    quality_dimensions: tuple[str, ...],
    quality_inputs: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
    quality_surface_refs: tuple[str, ...],
) -> QualityProfile:
    return QualityProfile(
        quality_profile_id=quality_profile_id,
        profile_name=profile_name,
        quality_profile_kind=quality_profile_kind,
        quality_level=quality_level,
        source_model_profile_ids=source_model_profile_ids,
        source_cost_profile_ids=source_cost_profile_ids,
        source_provider_selection_profile_ids=source_provider_selection_profile_ids,
        source_quality_escalation_profile_ids=source_quality_escalation_profile_ids,
        source_execution_simulation_profile_ids=(
            source_execution_simulation_profile_ids
        ),
        source_hitl_decision_profile_ids=source_hitl_decision_profile_ids,
        route_applicability=route_applicability,
        quality_dimensions=quality_dimensions,
        quality_inputs=quality_inputs,
        advisory_outputs=advisory_outputs,
        quality_surface_refs=quality_surface_refs,
        source_registries=_QUALITY_PROFILE_SOURCE_REGISTRIES,
        observability_surfaces=_QUALITY_PROFILE_OBSERVABILITY_SURFACES,
    )


QUALITY_PROFILE_SURFACES = (
    "quality_profile_panel",
    "model_profile_panel",
    "cost_profile_panel",
    "execution_simulator_panel",
    "hitl_decision_panel",
)

QUALITY_PROFILES = (
    _quality_profile(
        quality_profile_id="planning_quality_profile",
        profile_name="Planning Quality Profile",
        quality_profile_kind="planning_quality_review",
        quality_level="medium",
        source_model_profile_ids=("fast_iteration_model_profile",),
        source_cost_profile_ids=("planning_iteration_cost_profile",),
        source_provider_selection_profile_ids=(
            "current_config_provider_visibility_profile",
            "local_candidate_provider_visibility_profile",
        ),
        source_quality_escalation_profile_ids=(
            "quality_escalation::planning_execution_fit",
        ),
        source_execution_simulation_profile_ids=("route_preview_simulation_profile",),
        source_hitl_decision_profile_ids=("hitl_visibility_decision_profile",),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DEBUG,
            RouteName.PREVIEW,
        ),
        quality_dimensions=(
            "planning_execution_fit_metadata",
            "route_preview_quality_context",
            "iteration_readiness_quality",
        ),
        quality_inputs=(
            "fast_iteration_capability_profile",
            "planning_quality_context",
            "route_preview_simulation_metadata",
        ),
        advisory_outputs=(
            "planning_quality_review_context",
            "manual_quality_review_hint",
            "no_quality_scoring_notice",
        ),
        quality_surface_refs=("quality_profile_panel", "model_profile_panel"),
    ),
    _quality_profile(
        quality_profile_id="creative_quality_profile",
        profile_name="Creative Quality Profile",
        quality_profile_kind="creative_quality_review",
        quality_level="high",
        source_model_profile_ids=("creative_reasoning_model_profile",),
        source_cost_profile_ids=("creative_reasoning_cost_profile",),
        source_provider_selection_profile_ids=(
            "local_candidate_provider_visibility_profile",
            "cloud_candidate_provider_visibility_profile",
        ),
        source_quality_escalation_profile_ids=(
            "quality_escalation::style_aesthetic_alignment",
        ),
        source_execution_simulation_profile_ids=(
            "local_cloud_comparison_simulation_profile",
        ),
        source_hitl_decision_profile_ids=("hitl_risk_review_decision_profile",),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        quality_dimensions=(
            "creative_reasoning_quality_metadata",
            "style_aesthetic_alignment_context",
            "local_cloud_quality_comparison",
        ),
        quality_inputs=(
            "creative_reasoning_capability_profile",
            "aesthetic_quality_context",
            "local_cloud_comparison_simulation_metadata",
        ),
        advisory_outputs=(
            "creative_quality_review_context",
            "manual_aesthetic_quality_hint",
            "no_quality_evaluation_notice",
        ),
        quality_surface_refs=(
            "quality_profile_panel",
            "cost_profile_panel",
            "execution_simulator_panel",
        ),
    ),
    _quality_profile(
        quality_profile_id="refinement_quality_profile",
        profile_name="Refinement Quality Profile",
        quality_profile_kind="refinement_quality_review",
        quality_level="critical",
        source_model_profile_ids=(
            "code_assistance_model_profile",
            "evaluation_review_model_profile",
        ),
        source_cost_profile_ids=("curation_refinement_cost_profile",),
        source_provider_selection_profile_ids=(
            "cloud_candidate_provider_visibility_profile",
            "operator_override_provider_visibility_profile",
        ),
        source_quality_escalation_profile_ids=(
            "quality_escalation::curation_refinement_need",
        ),
        source_execution_simulation_profile_ids=(
            "local_cloud_comparison_simulation_profile",
            "hitl_review_simulation_profile",
        ),
        source_hitl_decision_profile_ids=(
            "hitl_risk_review_decision_profile",
            "hitl_final_review_decision_profile",
        ),
        route_applicability=(
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        quality_dimensions=(
            "curation_refinement_quality_metadata",
            "review_quality_signal_context",
            "evaluation_boundary_quality_context",
        ),
        quality_inputs=(
            "code_assistance_capability_profile",
            "evaluation_review_capability_profile",
            "refinement_quality_context",
        ),
        advisory_outputs=(
            "refinement_quality_review_context",
            "manual_critical_quality_hint",
            "no_refinement_trigger_notice",
        ),
        quality_surface_refs=(
            "quality_profile_panel",
            "execution_simulator_panel",
            "hitl_decision_panel",
        ),
    ),
    _quality_profile(
        quality_profile_id="final_review_quality_profile",
        profile_name="Final Review Quality Profile",
        quality_profile_kind="final_review_quality",
        quality_level="low",
        source_model_profile_ids=("evaluation_review_model_profile",),
        source_cost_profile_ids=("final_review_cost_profile",),
        source_provider_selection_profile_ids=(
            "current_config_provider_visibility_profile",
            "operator_override_provider_visibility_profile",
        ),
        source_quality_escalation_profile_ids=(
            "quality_escalation::final_synthesis_readiness",
        ),
        source_execution_simulation_profile_ids=("hitl_review_simulation_profile",),
        source_hitl_decision_profile_ids=("hitl_final_review_decision_profile",),
        route_applicability=(
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        quality_dimensions=(
            "final_synthesis_quality_metadata",
            "evaluation_review_quality_context",
            "manual_signoff_quality_context",
        ),
        quality_inputs=(
            "evaluation_review_capability_profile",
            "final_quality_context",
            "hitl_review_simulation_metadata",
        ),
        advisory_outputs=(
            "final_review_quality_context",
            "manual_final_quality_hint",
            "no_output_mutation_notice",
        ),
        quality_surface_refs=(
            "quality_profile_panel",
            "cost_profile_panel",
            "hitl_decision_panel",
        ),
    ),
)

QUALITY_PROFILE_REGISTRY = QualityProfileRegistry(
    quality_profiles=QUALITY_PROFILES,
    quality_profile_ids=tuple(
        profile.quality_profile_id for profile in QUALITY_PROFILES
    ),
    quality_profile_kinds=tuple(
        profile.quality_profile_kind for profile in QUALITY_PROFILES
    ),
    quality_levels=tuple(profile.quality_level for profile in QUALITY_PROFILES),
    model_profile_ids=tuple(MODEL_PROFILE_REGISTRY.model_profile_ids),
    cost_profile_ids=tuple(COST_PROFILE_REGISTRY.cost_profile_ids),
    provider_selection_profile_ids=tuple(
        PROVIDER_SELECTION_REGISTRY.provider_selection_profile_ids
    ),
    quality_escalation_profile_ids=tuple(
        QUALITY_ESCALATION_REGISTRY.quality_profile_ids
    ),
    execution_simulation_profile_ids=tuple(
        EXECUTION_SIMULATOR_REGISTRY.execution_simulation_profile_ids
    ),
    hitl_decision_profile_ids=tuple(HITL_DECISION_REGISTRY.hitl_decision_profile_ids),
    quality_surface_refs=QUALITY_PROFILE_SURFACES,
    route_names=tuple(RouteName),
    profile_count=len(QUALITY_PROFILES),
    source_registries=_QUALITY_PROFILE_SOURCE_REGISTRIES,
    observability_surfaces=_QUALITY_PROFILE_OBSERVABILITY_SURFACES,
)

LocalCloudComparisonKind = Literal[
    "generation_route_comparison",
    "creative_reasoning_comparison",
    "code_review_comparison",
    "evaluation_review_comparison",
]

LOCAL_CLOUD_COMPARISON_PROFILE_SERIALIZATION_VERSION = (
    "local_cloud_comparison_profile.v1"
)
LOCAL_CLOUD_COMPARISON_REGISTRY_SERIALIZATION_VERSION = (
    "local_cloud_comparison_registry.v1"
)
LOCAL_CLOUD_COMPARISON_REGISTRY_AUTHORITY_BOUNDARY = (
    "Local/Cloud Comparison Layer metadata describes passive side-by-side "
    "inspection context for local and cloud model surfaces in V4.4 Hybrid "
    "Studio only; it does not execute local or cloud providers, run parallel "
    "model calls, select comparison winners, route providers or models, score "
    "cost or quality, run fallback, trigger retries, mutate prompts, write "
    "replay storage, or modify generated output."
)

_LOCAL_CLOUD_COMPARISON_SOURCE_REGISTRIES = (
    "local_model_registry",
    "cloud_model_registry",
    "hybrid_execution_registry",
    "provider_selection_registry",
    "execution_simulator_registry",
    "model_profile_registry",
    "cost_profile_registry",
    "quality_profile_registry",
)

_LOCAL_CLOUD_COMPARISON_SURFACES = (
    "local_cloud_comparison_panel",
    "model_catalog_panel",
    "provider_selection_panel",
    "execution_simulator_panel",
    "cost_profile_panel",
    "quality_profile_panel",
)

_LOCAL_CLOUD_COMPARISON_OBSERVABILITY_SURFACES = (
    "comparison_profile_id",
    "comparison_kind",
    "source_local_surface_ids",
    "source_cloud_surface_ids",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_LOCAL_CLOUD_COMPARISON_BLOCKED_RUNTIME_BEHAVIORS = (
    "comparison_runtime_execution",
    "local_provider_execution",
    "cloud_provider_execution",
    "parallel_model_execution",
    "provider_or_model_routing",
    "model_selection",
    "cost_scoring",
    "quality_scoring",
    "winner_selection",
    "fallback_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_replay_storage",
    "generated_output_modification",
)


class LocalCloudComparisonProfile(BaseModel):
    """Inspectable passive local/cloud comparison profile."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    comparison_profile_id: str = Field(min_length=1, max_length=120)
    profile_name: str = Field(min_length=1, max_length=150)
    comparison_kind: LocalCloudComparisonKind
    source_local_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_cloud_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_execution_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_provider_selection_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_execution_simulation_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_model_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_cost_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_quality_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    comparison_dimensions: tuple[str, ...] = Field(min_length=1, max_length=10)
    comparison_inputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    comparison_surface_refs: tuple[str, ...] = Field(min_length=1, max_length=6)
    source_registries: tuple[str, ...] = Field(min_length=8, max_length=8)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    authority_boundary: str = Field(
        default=LOCAL_CLOUD_COMPARISON_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1300,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_LOCAL_CLOUD_COMPARISON_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    comparison_runtime_execution_implemented: Literal[False] = False
    local_provider_execution_implemented: Literal[False] = False
    cloud_provider_execution_implemented: Literal[False] = False
    parallel_model_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    cost_scoring_implemented: Literal[False] = False
    quality_scoring_implemented: Literal[False] = False
    winner_selection_implemented: Literal[False] = False
    fallback_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    serialization_version: Literal["local_cloud_comparison_profile.v1"] = (
        LOCAL_CLOUD_COMPARISON_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class LocalCloudComparisonRegistry(BaseModel):
    """Stable passive registry for V4.4 local/cloud comparison metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["local_cloud_comparison_registry"] = "local_cloud_comparison_registry"
    serialization_version: Literal["local_cloud_comparison_registry.v1"] = (
        LOCAL_CLOUD_COMPARISON_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=LOCAL_CLOUD_COMPARISON_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1300,
    )
    comparison_profiles: tuple[LocalCloudComparisonProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    comparison_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    comparison_kinds: tuple[LocalCloudComparisonKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    local_surface_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    cloud_surface_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    execution_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    provider_selection_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    execution_simulation_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    model_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    cost_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    quality_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    comparison_surface_refs: tuple[str, ...] = Field(min_length=6, max_length=6)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=8, max_length=8)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_LOCAL_CLOUD_COMPARISON_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    comparison_runtime_execution_implemented: Literal[False] = False
    local_provider_execution_implemented: Literal[False] = False
    cloud_provider_execution_implemented: Literal[False] = False
    parallel_model_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    cost_scoring_implemented: Literal[False] = False
    quality_scoring_implemented: Literal[False] = False
    winner_selection_implemented: Literal[False] = False
    fallback_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.comparison_profile_id for profile in self.comparison_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("comparison_profile_ids must be unique")
        if self.comparison_profile_ids != derived_profile_ids:
            raise ValueError("comparison_profile_ids must match comparison_profiles")
        if self.profile_count != len(self.comparison_profiles):
            raise ValueError("profile_count must match comparison_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.comparison_kinds != tuple(
            profile.comparison_kind for profile in self.comparison_profiles
        ):
            raise ValueError("comparison_kinds must match comparison_profiles")

        known_routes = set(self.route_names)
        known_local_surfaces = set(self.local_surface_ids)
        known_cloud_surfaces = set(self.cloud_surface_ids)
        known_execution_profiles = set(self.execution_profile_ids)
        known_provider_profiles = set(self.provider_selection_profile_ids)
        known_simulation_profiles = set(self.execution_simulation_profile_ids)
        known_model_profiles = set(self.model_profile_ids)
        known_cost_profiles = set(self.cost_profile_ids)
        known_quality_profiles = set(self.quality_profile_ids)
        known_comparison_surfaces = set(self.comparison_surface_refs)
        profile_sources = {
            source_registry
            for profile in self.comparison_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match comparison sources")

        for profile in self.comparison_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("profile source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.source_local_surface_ids).issubset(known_local_surfaces):
                raise ValueError("source_local_surface_ids must be known local models")
            if not set(profile.source_cloud_surface_ids).issubset(known_cloud_surfaces):
                raise ValueError("source_cloud_surface_ids must be known cloud models")
            if not set(profile.source_execution_profile_ids).issubset(
                known_execution_profiles
            ):
                raise ValueError("source_execution_profile_ids must be known profiles")
            if not set(profile.source_provider_selection_profile_ids).issubset(
                known_provider_profiles
            ):
                raise ValueError(
                    "source_provider_selection_profile_ids must be known profiles"
                )
            if not set(profile.source_execution_simulation_profile_ids).issubset(
                known_simulation_profiles
            ):
                raise ValueError(
                    "source_execution_simulation_profile_ids must be known profiles"
                )
            if not set(profile.source_model_profile_ids).issubset(known_model_profiles):
                raise ValueError("source_model_profile_ids must be known profiles")
            if not set(profile.source_cost_profile_ids).issubset(known_cost_profiles):
                raise ValueError("source_cost_profile_ids must be known profiles")
            if not set(profile.source_quality_profile_ids).issubset(
                known_quality_profiles
            ):
                raise ValueError("source_quality_profile_ids must be known profiles")
            if not set(profile.comparison_surface_refs).issubset(
                known_comparison_surfaces
            ):
                raise ValueError(
                    "comparison_surface_refs must be known registry surfaces"
                )
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
        return self


def local_cloud_comparison_registry() -> LocalCloudComparisonRegistry:
    """Return passive V4.4 Hybrid Studio local/cloud comparison metadata."""

    return LOCAL_CLOUD_COMPARISON_REGISTRY


def local_cloud_comparison_profile_by_id(
    comparison_profile_id: str,
    registry: LocalCloudComparisonRegistry | None = None,
) -> LocalCloudComparisonProfile | None:
    """Return one comparison profile without executing either side."""

    source_registry = registry or LOCAL_CLOUD_COMPARISON_REGISTRY
    for profile in source_registry.comparison_profiles:
        if profile.comparison_profile_id == comparison_profile_id:
            return profile
    return None


def local_cloud_comparison_profiles_for_route(
    route: RouteName | str,
    registry: LocalCloudComparisonRegistry | None = None,
) -> tuple[LocalCloudComparisonProfile, ...]:
    """Return passive comparison profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or LOCAL_CLOUD_COMPARISON_REGISTRY
    return tuple(
        profile
        for profile in source_registry.comparison_profiles
        if route_name in profile.route_applicability
    )


def _local_cloud_comparison_profile(
    *,
    comparison_profile_id: str,
    profile_name: str,
    comparison_kind: LocalCloudComparisonKind,
    source_local_surface_ids: tuple[str, ...],
    source_cloud_surface_ids: tuple[str, ...],
    source_execution_profile_ids: tuple[str, ...],
    source_provider_selection_profile_ids: tuple[str, ...],
    source_execution_simulation_profile_ids: tuple[str, ...],
    source_model_profile_ids: tuple[str, ...],
    source_cost_profile_ids: tuple[str, ...],
    source_quality_profile_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    comparison_dimensions: tuple[str, ...],
    comparison_inputs: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
    comparison_surface_refs: tuple[str, ...],
) -> LocalCloudComparisonProfile:
    return LocalCloudComparisonProfile(
        comparison_profile_id=comparison_profile_id,
        profile_name=profile_name,
        comparison_kind=comparison_kind,
        source_local_surface_ids=source_local_surface_ids,
        source_cloud_surface_ids=source_cloud_surface_ids,
        source_execution_profile_ids=source_execution_profile_ids,
        source_provider_selection_profile_ids=source_provider_selection_profile_ids,
        source_execution_simulation_profile_ids=(
            source_execution_simulation_profile_ids
        ),
        source_model_profile_ids=source_model_profile_ids,
        source_cost_profile_ids=source_cost_profile_ids,
        source_quality_profile_ids=source_quality_profile_ids,
        route_applicability=route_applicability,
        comparison_dimensions=comparison_dimensions,
        comparison_inputs=comparison_inputs,
        advisory_outputs=advisory_outputs,
        comparison_surface_refs=comparison_surface_refs,
        source_registries=_LOCAL_CLOUD_COMPARISON_SOURCE_REGISTRIES,
        observability_surfaces=_LOCAL_CLOUD_COMPARISON_OBSERVABILITY_SURFACES,
    )


LOCAL_CLOUD_COMPARISON_SURFACES = (
    "local_cloud_comparison_panel",
    "model_catalog_panel",
    "provider_selection_panel",
    "execution_simulator_panel",
    "cost_profile_panel",
    "quality_profile_panel",
)

LOCAL_CLOUD_COMPARISON_PROFILES = (
    _local_cloud_comparison_profile(
        comparison_profile_id="generation_route_comparison_profile",
        profile_name="Generation Route Comparison Profile",
        comparison_kind="generation_route_comparison",
        source_local_surface_ids=(
            "ollama_chat_surface",
            "llama_cpp_completion_surface",
        ),
        source_cloud_surface_ids=("openai_generation_model_surface",),
        source_execution_profile_ids=(
            "local_first_context_profile",
            "side_by_side_comparison_profile",
        ),
        source_provider_selection_profile_ids=(
            "current_config_provider_visibility_profile",
            "local_candidate_provider_visibility_profile",
            "cloud_candidate_provider_visibility_profile",
        ),
        source_execution_simulation_profile_ids=(
            "route_preview_simulation_profile",
            "provider_selection_simulation_profile",
        ),
        source_model_profile_ids=(
            "fast_iteration_model_profile",
            "code_assistance_model_profile",
        ),
        source_cost_profile_ids=("planning_iteration_cost_profile",),
        source_quality_profile_ids=("planning_quality_profile",),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DEBUG,
            RouteName.PREVIEW,
        ),
        comparison_dimensions=(
            "route_fit_metadata",
            "latency_cost_posture_metadata",
            "provider_boundary_visibility",
        ),
        comparison_inputs=(
            "local_candidate_visibility_metadata",
            "cloud_candidate_visibility_metadata",
            "planning_quality_review_context",
        ),
        advisory_outputs=(
            "generation_route_comparison_context",
            "manual_route_comparison_hint",
            "no_winner_selection_notice",
        ),
        comparison_surface_refs=(
            "local_cloud_comparison_panel",
            "model_catalog_panel",
            "provider_selection_panel",
        ),
    ),
    _local_cloud_comparison_profile(
        comparison_profile_id="creative_reasoning_comparison_profile",
        profile_name="Creative Reasoning Comparison Profile",
        comparison_kind="creative_reasoning_comparison",
        source_local_surface_ids=("lm_studio_chat_surface",),
        source_cloud_surface_ids=("openai_generation_model_surface",),
        source_execution_profile_ids=(
            "cloud_first_context_profile",
            "side_by_side_comparison_profile",
        ),
        source_provider_selection_profile_ids=(
            "local_candidate_provider_visibility_profile",
            "cloud_candidate_provider_visibility_profile",
            "operator_override_provider_visibility_profile",
        ),
        source_execution_simulation_profile_ids=(
            "local_cloud_comparison_simulation_profile",
            "provider_selection_simulation_profile",
        ),
        source_model_profile_ids=("creative_reasoning_model_profile",),
        source_cost_profile_ids=("creative_reasoning_cost_profile",),
        source_quality_profile_ids=("creative_quality_profile",),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        comparison_dimensions=(
            "creative_reasoning_metadata",
            "context_window_metadata",
            "quality_cost_visibility",
        ),
        comparison_inputs=(
            "creative_reasoning_capability_profile",
            "creative_reasoning_cost_context",
            "creative_quality_review_context",
        ),
        advisory_outputs=(
            "creative_reasoning_comparison_context",
            "manual_creative_comparison_hint",
            "no_parallel_execution_notice",
        ),
        comparison_surface_refs=(
            "local_cloud_comparison_panel",
            "execution_simulator_panel",
            "quality_profile_panel",
        ),
    ),
    _local_cloud_comparison_profile(
        comparison_profile_id="code_review_comparison_profile",
        profile_name="Code Review Comparison Profile",
        comparison_kind="code_review_comparison",
        source_local_surface_ids=("llama_cpp_completion_surface",),
        source_cloud_surface_ids=(
            "openai_generation_model_surface",
            "provider_reported_response_model_surface",
        ),
        source_execution_profile_ids=(
            "local_first_context_profile",
            "side_by_side_comparison_profile",
        ),
        source_provider_selection_profile_ids=(
            "local_candidate_provider_visibility_profile",
            "cloud_candidate_provider_visibility_profile",
            "operator_override_provider_visibility_profile",
        ),
        source_execution_simulation_profile_ids=(
            "local_cloud_comparison_simulation_profile",
            "hitl_review_simulation_profile",
        ),
        source_model_profile_ids=("code_assistance_model_profile",),
        source_cost_profile_ids=("curation_refinement_cost_profile",),
        source_quality_profile_ids=("refinement_quality_profile",),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        comparison_dimensions=(
            "code_context_metadata",
            "runtime_diagnostic_metadata",
            "review_quality_visibility",
        ),
        comparison_inputs=(
            "code_assistance_capability_profile",
            "curation_refinement_cost_context",
            "refinement_quality_review_context",
        ),
        advisory_outputs=(
            "code_review_comparison_context",
            "manual_debug_comparison_hint",
            "no_provider_execution_notice",
        ),
        comparison_surface_refs=(
            "local_cloud_comparison_panel",
            "cost_profile_panel",
            "quality_profile_panel",
        ),
    ),
    _local_cloud_comparison_profile(
        comparison_profile_id="evaluation_review_comparison_profile",
        profile_name="Evaluation Review Comparison Profile",
        comparison_kind="evaluation_review_comparison",
        source_local_surface_ids=("local_transformers_multimodal_surface",),
        source_cloud_surface_ids=(
            "ragas_evaluator_model_surface",
            "provider_reported_response_model_surface",
        ),
        source_execution_profile_ids=(
            "side_by_side_comparison_profile",
            "operator_selected_context_profile",
        ),
        source_provider_selection_profile_ids=(
            "cloud_candidate_provider_visibility_profile",
            "operator_override_provider_visibility_profile",
        ),
        source_execution_simulation_profile_ids=(
            "local_cloud_comparison_simulation_profile",
            "hitl_review_simulation_profile",
        ),
        source_model_profile_ids=("evaluation_review_model_profile",),
        source_cost_profile_ids=("final_review_cost_profile",),
        source_quality_profile_ids=("final_review_quality_profile",),
        route_applicability=(
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        comparison_dimensions=(
            "evaluation_review_metadata",
            "provider_response_metadata",
            "human_review_visibility",
        ),
        comparison_inputs=(
            "evaluation_review_capability_profile",
            "final_review_cost_context",
            "final_review_quality_context",
        ),
        advisory_outputs=(
            "evaluation_review_comparison_context",
            "manual_evaluation_comparison_hint",
            "no_evaluator_call_notice",
        ),
        comparison_surface_refs=(
            "local_cloud_comparison_panel",
            "execution_simulator_panel",
            "quality_profile_panel",
        ),
    ),
)

LOCAL_CLOUD_COMPARISON_REGISTRY = LocalCloudComparisonRegistry(
    comparison_profiles=LOCAL_CLOUD_COMPARISON_PROFILES,
    comparison_profile_ids=tuple(
        profile.comparison_profile_id for profile in LOCAL_CLOUD_COMPARISON_PROFILES
    ),
    comparison_kinds=tuple(
        profile.comparison_kind for profile in LOCAL_CLOUD_COMPARISON_PROFILES
    ),
    local_surface_ids=tuple(LOCAL_MODEL_REGISTRY.surface_ids),
    cloud_surface_ids=tuple(CLOUD_MODEL_REGISTRY.surface_ids),
    execution_profile_ids=tuple(HYBRID_EXECUTION_REGISTRY.execution_profile_ids),
    provider_selection_profile_ids=tuple(
        PROVIDER_SELECTION_REGISTRY.provider_selection_profile_ids
    ),
    execution_simulation_profile_ids=tuple(
        EXECUTION_SIMULATOR_REGISTRY.execution_simulation_profile_ids
    ),
    model_profile_ids=tuple(MODEL_PROFILE_REGISTRY.model_profile_ids),
    cost_profile_ids=tuple(COST_PROFILE_REGISTRY.cost_profile_ids),
    quality_profile_ids=tuple(QUALITY_PROFILE_REGISTRY.quality_profile_ids),
    comparison_surface_refs=LOCAL_CLOUD_COMPARISON_SURFACES,
    route_names=tuple(RouteName),
    profile_count=len(LOCAL_CLOUD_COMPARISON_PROFILES),
    source_registries=_LOCAL_CLOUD_COMPARISON_SOURCE_REGISTRIES,
    observability_surfaces=_LOCAL_CLOUD_COMPARISON_OBSERVABILITY_SURFACES,
)

AgentWorkspaceKind = Literal[
    "planning_context_workspace",
    "artifact_runtime_workspace",
    "critique_curation_workspace",
    "refinement_synthesis_workspace",
]

AGENT_WORKSPACE_PROFILE_SERIALIZATION_VERSION = "agent_workspace_profile.v1"
AGENT_WORKSPACE_REGISTRY_SERIALIZATION_VERSION = "agent_workspace_registry.v1"
AGENT_WORKSPACE_REGISTRY_AUTHORITY_BOUNDARY = (
    "Agent Workspace metadata describes passive Studio-visible groupings of "
    "agent identities, roles, capability readiness, comparison context, "
    "quality context, and HITL context for V4.4 inspection only; it does not "
    "instantiate agents, invoke agents, orchestrate multiple agents, mutate "
    "workspace state, write memory, route providers or models, control "
    "workflow transitions, request human input, trigger retries, write replay "
    "storage, or modify generated output."
)

_AGENT_WORKSPACE_SOURCE_REGISTRIES = (
    "agent_identity_registry",
    "agent_role_registry",
    "agent_metadata_registry",
    "agent_capability_registry",
    "local_cloud_comparison_registry",
    "quality_profile_registry",
    "hitl_decision_registry",
    "studio_mode_registry",
)

_AGENT_WORKSPACE_SURFACES = (
    "agent_workspace_panel",
    "agent_roster_panel",
    "agent_role_matrix",
    "comparison_context_panel",
    "quality_context_panel",
    "hitl_review_panel",
)

_AGENT_WORKSPACE_OBSERVABILITY_SURFACES = (
    "workspace_profile_id",
    "workspace_kind",
    "source_agent_ids",
    "source_role_ids",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_AGENT_WORKSPACE_BLOCKED_RUNTIME_BEHAVIORS = (
    "workspace_execution",
    "agent_instantiation",
    "agent_invocation",
    "multi_agent_orchestration",
    "workspace_state_mutation",
    "memory_write",
    "provider_or_model_routing",
    "workflow_control",
    "human_input_request",
    "retry_or_refinement_triggering",
    "persistent_replay_storage",
    "generated_output_modification",
)


class AgentWorkspaceProfile(BaseModel):
    """Inspectable passive agent workspace grouping for Hybrid Studio."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    workspace_profile_id: str = Field(min_length=1, max_length=120)
    profile_name: str = Field(min_length=1, max_length=150)
    workspace_kind: AgentWorkspaceKind
    source_agent_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_role_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_agent_metadata_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_capability_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_comparison_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_quality_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_hitl_decision_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    workspace_surfaces: tuple[str, ...] = Field(min_length=1, max_length=6)
    visible_context_fields: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_registries: tuple[str, ...] = Field(min_length=8, max_length=8)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    authority_boundary: str = Field(
        default=AGENT_WORKSPACE_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1300,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_AGENT_WORKSPACE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    workspace_execution_implemented: Literal[False] = False
    agent_instantiation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    multi_agent_orchestration_implemented: Literal[False] = False
    workspace_state_mutation_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    serialization_version: Literal["agent_workspace_profile.v1"] = (
        AGENT_WORKSPACE_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentWorkspaceRegistry(BaseModel):
    """Stable passive registry for V4.4 Hybrid Studio agent workspaces."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_workspace_registry"] = "agent_workspace_registry"
    serialization_version: Literal["agent_workspace_registry.v1"] = (
        AGENT_WORKSPACE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_WORKSPACE_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1300,
    )
    workspace_profiles: tuple[AgentWorkspaceProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    workspace_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    workspace_kinds: tuple[AgentWorkspaceKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    role_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    agent_metadata_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    comparison_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    quality_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    hitl_decision_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    workspace_surface_refs: tuple[str, ...] = Field(min_length=6, max_length=6)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=8, max_length=8)
    observability_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_AGENT_WORKSPACE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    workspace_execution_implemented: Literal[False] = False
    agent_instantiation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    multi_agent_orchestration_implemented: Literal[False] = False
    workspace_state_mutation_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.workspace_profile_id for profile in self.workspace_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("workspace_profile_ids must be unique")
        if self.workspace_profile_ids != derived_profile_ids:
            raise ValueError("workspace_profile_ids must match workspace_profiles")
        if self.profile_count != len(self.workspace_profiles):
            raise ValueError("profile_count must match workspace_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.workspace_kinds != tuple(
            profile.workspace_kind for profile in self.workspace_profiles
        ):
            raise ValueError("workspace_kinds must match workspace_profiles")

        known_routes = set(self.route_names)
        known_agents = set(self.agent_ids)
        known_roles = set(self.role_ids)
        known_agent_metadata = set(self.agent_metadata_ids)
        known_capabilities = set(self.capability_ids)
        known_comparisons = set(self.comparison_profile_ids)
        known_quality_profiles = set(self.quality_profile_ids)
        known_hitl_profiles = set(self.hitl_decision_profile_ids)
        known_surfaces = set(self.workspace_surface_refs)
        profile_sources = {
            source_registry
            for profile in self.workspace_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match agent workspace sources")

        for profile in self.workspace_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("profile source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.source_agent_ids).issubset(known_agents):
                raise ValueError("source_agent_ids must be known agents")
            if not set(profile.source_role_ids).issubset(known_roles):
                raise ValueError("source_role_ids must be known roles")
            if not set(profile.source_agent_metadata_ids).issubset(
                known_agent_metadata
            ):
                raise ValueError(
                    "source_agent_metadata_ids must be known metadata entries"
                )
            if not set(profile.source_capability_ids).issubset(known_capabilities):
                raise ValueError("source_capability_ids must be known capabilities")
            if not set(profile.source_comparison_profile_ids).issubset(
                known_comparisons
            ):
                raise ValueError("source_comparison_profile_ids must be known profiles")
            if not set(profile.source_quality_profile_ids).issubset(
                known_quality_profiles
            ):
                raise ValueError("source_quality_profile_ids must be known profiles")
            if not set(profile.source_hitl_decision_profile_ids).issubset(
                known_hitl_profiles
            ):
                raise ValueError(
                    "source_hitl_decision_profile_ids must be known profiles"
                )
            if not set(profile.workspace_surfaces).issubset(known_surfaces):
                raise ValueError("workspace_surfaces must be known registry surfaces")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
        return self


def agent_workspace_registry() -> AgentWorkspaceRegistry:
    """Return passive V4.4 Hybrid Studio agent workspace metadata."""

    return AGENT_WORKSPACE_REGISTRY


def agent_workspace_profile_by_id(
    workspace_profile_id: str,
    registry: AgentWorkspaceRegistry | None = None,
) -> AgentWorkspaceProfile | None:
    """Return one agent workspace profile without creating a workspace."""

    source_registry = registry or AGENT_WORKSPACE_REGISTRY
    for profile in source_registry.workspace_profiles:
        if profile.workspace_profile_id == workspace_profile_id:
            return profile
    return None


def agent_workspace_profiles_for_route(
    route: RouteName | str,
    registry: AgentWorkspaceRegistry | None = None,
) -> tuple[AgentWorkspaceProfile, ...]:
    """Return passive agent workspaces applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or AGENT_WORKSPACE_REGISTRY
    return tuple(
        profile
        for profile in source_registry.workspace_profiles
        if route_name in profile.route_applicability
    )


def agent_workspace_profiles_for_agent_id(
    agent_id: str,
    registry: AgentWorkspaceRegistry | None = None,
) -> tuple[AgentWorkspaceProfile, ...]:
    """Return passive workspace profiles containing an agent id."""

    source_registry = registry or AGENT_WORKSPACE_REGISTRY
    agent_id_value = str(agent_id).strip()
    return tuple(
        profile
        for profile in source_registry.workspace_profiles
        if agent_id_value in profile.source_agent_ids
    )


def _agent_workspace_profile(
    *,
    workspace_profile_id: str,
    profile_name: str,
    workspace_kind: AgentWorkspaceKind,
    source_agent_ids: tuple[str, ...],
    source_role_ids: tuple[str, ...],
    source_capability_ids: tuple[str, ...],
    source_comparison_profile_ids: tuple[str, ...],
    source_quality_profile_ids: tuple[str, ...],
    source_hitl_decision_profile_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    workspace_surfaces: tuple[str, ...],
    visible_context_fields: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> AgentWorkspaceProfile:
    return AgentWorkspaceProfile(
        workspace_profile_id=workspace_profile_id,
        profile_name=profile_name,
        workspace_kind=workspace_kind,
        source_agent_ids=source_agent_ids,
        source_role_ids=source_role_ids,
        source_agent_metadata_ids=source_agent_ids,
        source_capability_ids=source_capability_ids,
        source_comparison_profile_ids=source_comparison_profile_ids,
        source_quality_profile_ids=source_quality_profile_ids,
        source_hitl_decision_profile_ids=source_hitl_decision_profile_ids,
        route_applicability=route_applicability,
        workspace_surfaces=workspace_surfaces,
        visible_context_fields=visible_context_fields,
        advisory_outputs=advisory_outputs,
        source_registries=_AGENT_WORKSPACE_SOURCE_REGISTRIES,
        observability_surfaces=_AGENT_WORKSPACE_OBSERVABILITY_SURFACES,
    )


AGENT_WORKSPACE_SURFACES = (
    "agent_workspace_panel",
    "agent_roster_panel",
    "agent_role_matrix",
    "comparison_context_panel",
    "quality_context_panel",
    "hitl_review_panel",
)

AGENT_WORKSPACE_PROFILES = (
    _agent_workspace_profile(
        workspace_profile_id="planning_context_agent_workspace",
        profile_name="Planning Context Agent Workspace",
        workspace_kind="planning_context_workspace",
        source_agent_ids=(
            "planner_agent",
            "research_agent",
            "style_agent",
        ),
        source_role_ids=("planner", "research", "style"),
        source_capability_ids=("v4_planner_agent", "v4_agent_router"),
        source_comparison_profile_ids=(
            "generation_route_comparison_profile",
            "creative_reasoning_comparison_profile",
        ),
        source_quality_profile_ids=(
            "planning_quality_profile",
            "creative_quality_profile",
        ),
        source_hitl_decision_profile_ids=(
            "hitl_visibility_decision_profile",
            "hitl_confirmation_decision_profile",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.PREVIEW,
        ),
        workspace_surfaces=(
            "agent_workspace_panel",
            "agent_roster_panel",
            "comparison_context_panel",
        ),
        visible_context_fields=(
            "agent_identity_metadata",
            "role_family_metadata",
            "planning_comparison_context",
        ),
        advisory_outputs=(
            "planning_agent_workspace_context",
            "manual_planning_handoff_hint",
            "no_agent_invocation_notice",
        ),
    ),
    _agent_workspace_profile(
        workspace_profile_id="artifact_runtime_agent_workspace",
        profile_name="Artifact Runtime Agent Workspace",
        workspace_kind="artifact_runtime_workspace",
        source_agent_ids=(
            "runtime_agent",
            "artifact_agent",
            "art_direction_agent",
        ),
        source_role_ids=("runtime", "artifact", "art_direction"),
        source_capability_ids=("v4_artifact_agent", "v4_runtime_agent"),
        source_comparison_profile_ids=(
            "generation_route_comparison_profile",
            "code_review_comparison_profile",
        ),
        source_quality_profile_ids=(
            "planning_quality_profile",
            "refinement_quality_profile",
        ),
        source_hitl_decision_profile_ids=(
            "hitl_confirmation_decision_profile",
            "hitl_risk_review_decision_profile",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.PREVIEW,
        ),
        workspace_surfaces=(
            "agent_workspace_panel",
            "agent_role_matrix",
            "comparison_context_panel",
            "quality_context_panel",
        ),
        visible_context_fields=(
            "runtime_agent_metadata",
            "artifact_agent_metadata",
            "implementation_comparison_context",
        ),
        advisory_outputs=(
            "artifact_runtime_agent_workspace_context",
            "manual_runtime_handoff_hint",
            "no_artifact_execution_notice",
        ),
    ),
    _agent_workspace_profile(
        workspace_profile_id="critique_curation_agent_workspace",
        profile_name="Critique Curation Agent Workspace",
        workspace_kind="critique_curation_workspace",
        source_agent_ids=(
            "aesthetic_critic_agent",
            "narrative_symbolic_agent",
            "creative_curator_agent",
            "critic_agent",
        ),
        source_role_ids=(
            "aesthetic_critic",
            "narrative_symbolic",
            "creative_curator",
            "critic",
        ),
        source_capability_ids=("v4_agentic_studio", "adaptive_multi_agent_escalation"),
        source_comparison_profile_ids=(
            "creative_reasoning_comparison_profile",
            "evaluation_review_comparison_profile",
        ),
        source_quality_profile_ids=(
            "creative_quality_profile",
            "refinement_quality_profile",
            "final_review_quality_profile",
        ),
        source_hitl_decision_profile_ids=(
            "hitl_risk_review_decision_profile",
            "hitl_final_review_decision_profile",
        ),
        route_applicability=(
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        workspace_surfaces=(
            "agent_workspace_panel",
            "agent_role_matrix",
            "quality_context_panel",
            "hitl_review_panel",
        ),
        visible_context_fields=(
            "critique_agent_metadata",
            "quality_profile_context",
            "hitl_review_context",
        ),
        advisory_outputs=(
            "critique_curation_agent_workspace_context",
            "manual_critique_review_hint",
            "no_multi_agent_orchestration_notice",
        ),
    ),
    _agent_workspace_profile(
        workspace_profile_id="refinement_synthesis_agent_workspace",
        profile_name="Refinement Synthesis Agent Workspace",
        workspace_kind="refinement_synthesis_workspace",
        source_agent_ids=(
            "refiner_agent",
            "final_synthesizer_agent",
        ),
        source_role_ids=("refiner", "final_synthesizer"),
        source_capability_ids=("v4_agentic_studio", "adaptive_multi_agent_escalation"),
        source_comparison_profile_ids=(
            "code_review_comparison_profile",
            "evaluation_review_comparison_profile",
        ),
        source_quality_profile_ids=(
            "refinement_quality_profile",
            "final_review_quality_profile",
        ),
        source_hitl_decision_profile_ids=(
            "hitl_confirmation_decision_profile",
            "hitl_final_review_decision_profile",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        workspace_surfaces=(
            "agent_workspace_panel",
            "agent_roster_panel",
            "quality_context_panel",
            "hitl_review_panel",
        ),
        visible_context_fields=(
            "refiner_agent_metadata",
            "synthesis_agent_metadata",
            "final_review_quality_context",
        ),
        advisory_outputs=(
            "refinement_synthesis_agent_workspace_context",
            "manual_synthesis_review_hint",
            "no_output_mutation_notice",
        ),
    ),
)

AGENT_WORKSPACE_REGISTRY = AgentWorkspaceRegistry(
    workspace_profiles=AGENT_WORKSPACE_PROFILES,
    workspace_profile_ids=tuple(
        profile.workspace_profile_id for profile in AGENT_WORKSPACE_PROFILES
    ),
    workspace_kinds=tuple(
        profile.workspace_kind for profile in AGENT_WORKSPACE_PROFILES
    ),
    agent_ids=tuple(AGENT_IDENTITY_REGISTRY.agent_ids),
    role_ids=tuple(AGENT_ROLE_REGISTRY.role_ids),
    agent_metadata_ids=tuple(AGENT_METADATA_REGISTRY.agent_ids),
    capability_ids=tuple(AGENT_CAPABILITY_REGISTRY.capability_ids),
    comparison_profile_ids=tuple(
        LOCAL_CLOUD_COMPARISON_REGISTRY.comparison_profile_ids
    ),
    quality_profile_ids=tuple(QUALITY_PROFILE_REGISTRY.quality_profile_ids),
    hitl_decision_profile_ids=tuple(HITL_DECISION_REGISTRY.hitl_decision_profile_ids),
    workspace_surface_refs=AGENT_WORKSPACE_SURFACES,
    route_names=tuple(RouteName),
    profile_count=len(AGENT_WORKSPACE_PROFILES),
    source_registries=_AGENT_WORKSPACE_SOURCE_REGISTRIES,
    observability_surfaces=_AGENT_WORKSPACE_OBSERVABILITY_SURFACES,
)

AgentConversationViewKind = Literal[
    "workspace_thread_view",
    "agent_handoff_view",
    "review_discussion_view",
    "audit_trail_view",
]

AGENT_CONVERSATION_VIEW_PROFILE_SERIALIZATION_VERSION = (
    "agent_conversation_view_profile.v1"
)
AGENT_CONVERSATION_VIEW_REGISTRY_SERIALIZATION_VERSION = (
    "agent_conversation_view_registry.v1"
)
AGENT_CONVERSATION_VIEW_REGISTRY_AUTHORITY_BOUNDARY = (
    "Agent Conversation View metadata describes passive Studio-visible "
    "conversation thread, handoff, review, and audit surfaces over existing "
    "agent workspace, shared context, memory contract, workflow handoff, and "
    "HITL decision metadata for V4.4 inspection only; it does not record "
    "conversations, generate messages, invoke agents, persist conversation "
    "state, write memory, mutate workspace state, control workflow "
    "transitions, request human input, route providers or models, trigger "
    "retries, write replay storage, or modify generated output."
)

_AGENT_CONVERSATION_VIEW_SOURCE_REGISTRIES = (
    "agent_workspace_registry",
    "agent_identity_registry",
    "agent_role_registry",
    "shared_context_view_registry",
    "agent_memory_contract_registry",
    "workflow_agent_handoff_registry",
    "hitl_decision_registry",
    "studio_mode_registry",
)

_AGENT_CONVERSATION_VIEW_SURFACES = (
    "agent_conversation_panel",
    "conversation_thread_list",
    "agent_message_timeline",
    "handoff_context_panel",
    "shared_context_scope_panel",
    "hitl_conversation_review_panel",
)

_AGENT_CONVERSATION_VIEW_OBSERVABILITY_SURFACES = (
    "conversation_view_profile_id",
    "conversation_view_kind",
    "source_workspace_profile_ids",
    "source_agent_ids",
    "source_shared_context_view_ids",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_AGENT_CONVERSATION_VIEW_BLOCKED_RUNTIME_BEHAVIORS = (
    "conversation_execution",
    "conversation_persistence",
    "agent_message_generation",
    "agent_invocation",
    "memory_write",
    "workspace_state_mutation",
    "workflow_control",
    "human_input_request",
    "provider_or_model_routing",
    "retry_or_refinement_triggering",
    "persistent_replay_storage",
    "generated_output_modification",
)


class AgentConversationViewProfile(BaseModel):
    """Inspectable passive agent conversation view for Hybrid Studio."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    conversation_view_profile_id: str = Field(min_length=1, max_length=140)
    profile_name: str = Field(min_length=1, max_length=160)
    conversation_view_kind: AgentConversationViewKind
    source_workspace_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_agent_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_role_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_shared_context_view_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=8,
    )
    source_memory_contract_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_handoff_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_hitl_decision_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    conversation_surfaces: tuple[str, ...] = Field(min_length=1, max_length=6)
    visible_thread_fields: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_registries: tuple[str, ...] = Field(min_length=8, max_length=8)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=AGENT_CONVERSATION_VIEW_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1400,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_AGENT_CONVERSATION_VIEW_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    conversation_execution_implemented: Literal[False] = False
    conversation_persistence_implemented: Literal[False] = False
    agent_message_generation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    workspace_state_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    serialization_version: Literal["agent_conversation_view_profile.v1"] = (
        AGENT_CONVERSATION_VIEW_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentConversationViewRegistry(BaseModel):
    """Stable passive registry for V4.4 Hybrid Studio conversation views."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_conversation_view_registry"] = (
        "agent_conversation_view_registry"
    )
    serialization_version: Literal["agent_conversation_view_registry.v1"] = (
        AGENT_CONVERSATION_VIEW_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_CONVERSATION_VIEW_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1400,
    )
    conversation_view_profiles: tuple[AgentConversationViewProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    conversation_view_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    conversation_view_kinds: tuple[AgentConversationViewKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    workspace_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    role_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    shared_context_view_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    memory_contract_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    handoff_profile_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    hitl_decision_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    conversation_surface_refs: tuple[str, ...] = Field(min_length=6, max_length=6)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=8, max_length=8)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_AGENT_CONVERSATION_VIEW_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    conversation_execution_implemented: Literal[False] = False
    conversation_persistence_implemented: Literal[False] = False
    agent_message_generation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    workspace_state_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.conversation_view_profile_id
            for profile in self.conversation_view_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("conversation_view_profile_ids must be unique")
        if self.conversation_view_profile_ids != derived_profile_ids:
            raise ValueError(
                "conversation_view_profile_ids must match conversation_view_profiles"
            )
        if self.profile_count != len(self.conversation_view_profiles):
            raise ValueError("profile_count must match conversation_view_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.conversation_view_kinds != tuple(
            profile.conversation_view_kind
            for profile in self.conversation_view_profiles
        ):
            raise ValueError(
                "conversation_view_kinds must match conversation_view_profiles"
            )

        known_routes = set(self.route_names)
        known_workspaces = set(self.workspace_profile_ids)
        known_agents = set(self.agent_ids)
        known_roles = set(self.role_ids)
        known_shared_context_views = set(self.shared_context_view_ids)
        known_memory_contracts = set(self.memory_contract_ids)
        known_handoff_profiles = set(self.handoff_profile_ids)
        known_hitl_profiles = set(self.hitl_decision_profile_ids)
        known_surfaces = set(self.conversation_surface_refs)
        profile_sources = {
            source_registry
            for profile in self.conversation_view_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match conversation view sources")

        for profile in self.conversation_view_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("profile source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.source_workspace_profile_ids).issubset(known_workspaces):
                raise ValueError(
                    "source_workspace_profile_ids must be known workspace profiles"
                )
            if not set(profile.source_agent_ids).issubset(known_agents):
                raise ValueError("source_agent_ids must be known agents")
            if not set(profile.source_role_ids).issubset(known_roles):
                raise ValueError("source_role_ids must be known roles")
            if not set(profile.source_shared_context_view_ids).issubset(
                known_shared_context_views
            ):
                raise ValueError(
                    "source_shared_context_view_ids must be known context views"
                )
            if not set(profile.source_memory_contract_ids).issubset(
                known_memory_contracts
            ):
                raise ValueError(
                    "source_memory_contract_ids must be known memory contracts"
                )
            if not set(profile.source_handoff_profile_ids).issubset(
                known_handoff_profiles
            ):
                raise ValueError(
                    "source_handoff_profile_ids must be known handoff profiles"
                )
            if not set(profile.source_hitl_decision_profile_ids).issubset(
                known_hitl_profiles
            ):
                raise ValueError(
                    "source_hitl_decision_profile_ids must be known profiles"
                )
            if not set(profile.conversation_surfaces).issubset(known_surfaces):
                raise ValueError(
                    "conversation_surfaces must be known registry surfaces"
                )
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
        return self


def agent_conversation_view_registry() -> AgentConversationViewRegistry:
    """Return passive V4.4 Hybrid Studio agent conversation view metadata."""

    return AGENT_CONVERSATION_VIEW_REGISTRY


def agent_conversation_view_profile_by_id(
    conversation_view_profile_id: str,
    registry: AgentConversationViewRegistry | None = None,
) -> AgentConversationViewProfile | None:
    """Return one conversation view profile without recording a conversation."""

    source_registry = registry or AGENT_CONVERSATION_VIEW_REGISTRY
    for profile in source_registry.conversation_view_profiles:
        if profile.conversation_view_profile_id == conversation_view_profile_id:
            return profile
    return None


def agent_conversation_view_profiles_for_route(
    route: RouteName | str,
    registry: AgentConversationViewRegistry | None = None,
) -> tuple[AgentConversationViewProfile, ...]:
    """Return passive conversation views applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or AGENT_CONVERSATION_VIEW_REGISTRY
    return tuple(
        profile
        for profile in source_registry.conversation_view_profiles
        if route_name in profile.route_applicability
    )


def agent_conversation_view_profiles_for_workspace(
    workspace_profile_id: str,
    registry: AgentConversationViewRegistry | None = None,
) -> tuple[AgentConversationViewProfile, ...]:
    """Return passive conversation views for a workspace profile id."""

    source_registry = registry or AGENT_CONVERSATION_VIEW_REGISTRY
    workspace_id = str(workspace_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.conversation_view_profiles
        if workspace_id in profile.source_workspace_profile_ids
    )


def agent_conversation_view_profiles_for_agent_id(
    agent_id: str,
    registry: AgentConversationViewRegistry | None = None,
) -> tuple[AgentConversationViewProfile, ...]:
    """Return passive conversation views containing an agent id."""

    source_registry = registry or AGENT_CONVERSATION_VIEW_REGISTRY
    agent_id_value = str(agent_id).strip()
    return tuple(
        profile
        for profile in source_registry.conversation_view_profiles
        if agent_id_value in profile.source_agent_ids
    )


def _role_ids_for_conversation_agent_ids(agent_ids: tuple[str, ...]) -> tuple[str, ...]:
    role_by_agent_id = {
        role.agent_id: role.role_id for role in AGENT_ROLE_REGISTRY.roles
    }
    return tuple(role_by_agent_id[agent_id] for agent_id in agent_ids)


def _shared_context_view_ids_for_agent_ids(
    agent_ids: tuple[str, ...],
) -> tuple[str, ...]:
    view_by_agent_id = {
        view.agent_id: view.view_id for view in SHARED_CONTEXT_VIEW_REGISTRY.views
    }
    return tuple(view_by_agent_id[agent_id] for agent_id in agent_ids)


def _memory_contract_ids_for_agent_ids(agent_ids: tuple[str, ...]) -> tuple[str, ...]:
    contract_by_agent_id = {
        contract.agent_id: contract.memory_contract_id
        for contract in AGENT_MEMORY_CONTRACT_REGISTRY.contracts
    }
    return tuple(contract_by_agent_id[agent_id] for agent_id in agent_ids)


def _handoff_profile_ids_for_agent_ids(agent_ids: tuple[str, ...]) -> tuple[str, ...]:
    profile_by_agent_id = {
        profile.agent_id: profile.handoff_profile_id
        for profile in WORKFLOW_AGENT_HANDOFF_REGISTRY.profiles
    }
    return tuple(profile_by_agent_id[agent_id] for agent_id in agent_ids)


def _agent_conversation_view_profile(
    *,
    conversation_view_profile_id: str,
    profile_name: str,
    conversation_view_kind: AgentConversationViewKind,
    source_workspace_profile_ids: tuple[str, ...],
    source_agent_ids: tuple[str, ...],
    source_hitl_decision_profile_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    conversation_surfaces: tuple[str, ...],
    visible_thread_fields: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> AgentConversationViewProfile:
    return AgentConversationViewProfile(
        conversation_view_profile_id=conversation_view_profile_id,
        profile_name=profile_name,
        conversation_view_kind=conversation_view_kind,
        source_workspace_profile_ids=source_workspace_profile_ids,
        source_agent_ids=source_agent_ids,
        source_role_ids=_role_ids_for_conversation_agent_ids(source_agent_ids),
        source_shared_context_view_ids=_shared_context_view_ids_for_agent_ids(
            source_agent_ids
        ),
        source_memory_contract_ids=_memory_contract_ids_for_agent_ids(source_agent_ids),
        source_handoff_profile_ids=_handoff_profile_ids_for_agent_ids(source_agent_ids),
        source_hitl_decision_profile_ids=source_hitl_decision_profile_ids,
        route_applicability=route_applicability,
        conversation_surfaces=conversation_surfaces,
        visible_thread_fields=visible_thread_fields,
        advisory_outputs=advisory_outputs,
        source_registries=_AGENT_CONVERSATION_VIEW_SOURCE_REGISTRIES,
        observability_surfaces=_AGENT_CONVERSATION_VIEW_OBSERVABILITY_SURFACES,
    )


AGENT_CONVERSATION_VIEW_SURFACES = (
    "agent_conversation_panel",
    "conversation_thread_list",
    "agent_message_timeline",
    "handoff_context_panel",
    "shared_context_scope_panel",
    "hitl_conversation_review_panel",
)

AGENT_CONVERSATION_VIEW_PROFILES = (
    _agent_conversation_view_profile(
        conversation_view_profile_id="workspace_thread_conversation_view",
        profile_name="Workspace Thread Conversation View",
        conversation_view_kind="workspace_thread_view",
        source_workspace_profile_ids=(
            "planning_context_agent_workspace",
            "artifact_runtime_agent_workspace",
        ),
        source_agent_ids=(
            "planner_agent",
            "research_agent",
            "runtime_agent",
            "artifact_agent",
        ),
        source_hitl_decision_profile_ids=(
            "hitl_visibility_decision_profile",
            "hitl_confirmation_decision_profile",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DEBUG,
            RouteName.PREVIEW,
        ),
        conversation_surfaces=(
            "agent_conversation_panel",
            "conversation_thread_list",
            "agent_message_timeline",
            "shared_context_scope_panel",
        ),
        visible_thread_fields=(
            "workspace_profile_metadata",
            "agent_identity_summary",
            "route_context_metadata",
            "shared_context_scope_metadata",
        ),
        advisory_outputs=(
            "workspace_thread_conversation_context",
            "manual_agent_note_hint",
            "no_agent_message_generation_notice",
        ),
    ),
    _agent_conversation_view_profile(
        conversation_view_profile_id="agent_handoff_conversation_view",
        profile_name="Agent Handoff Conversation View",
        conversation_view_kind="agent_handoff_view",
        source_workspace_profile_ids=(
            "planning_context_agent_workspace",
            "artifact_runtime_agent_workspace",
            "critique_curation_agent_workspace",
        ),
        source_agent_ids=(
            "planner_agent",
            "runtime_agent",
            "artifact_agent",
            "critic_agent",
            "final_synthesizer_agent",
        ),
        source_hitl_decision_profile_ids=(
            "hitl_confirmation_decision_profile",
            "hitl_risk_review_decision_profile",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        conversation_surfaces=(
            "agent_conversation_panel",
            "agent_message_timeline",
            "handoff_context_panel",
            "shared_context_scope_panel",
        ),
        visible_thread_fields=(
            "handoff_profile_metadata",
            "accepted_surface_metadata",
            "source_workflow_field_metadata",
            "shared_context_scope_metadata",
        ),
        advisory_outputs=(
            "agent_handoff_conversation_context",
            "manual_handoff_review_hint",
            "no_runtime_handoff_notice",
        ),
    ),
    _agent_conversation_view_profile(
        conversation_view_profile_id="review_conversation_view",
        profile_name="Review Conversation View",
        conversation_view_kind="review_discussion_view",
        source_workspace_profile_ids=(
            "critique_curation_agent_workspace",
            "refinement_synthesis_agent_workspace",
        ),
        source_agent_ids=(
            "aesthetic_critic_agent",
            "creative_curator_agent",
            "critic_agent",
            "refiner_agent",
            "final_synthesizer_agent",
        ),
        source_hitl_decision_profile_ids=(
            "hitl_risk_review_decision_profile",
            "hitl_final_review_decision_profile",
        ),
        route_applicability=(
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        conversation_surfaces=(
            "agent_conversation_panel",
            "agent_message_timeline",
            "shared_context_scope_panel",
            "hitl_conversation_review_panel",
        ),
        visible_thread_fields=(
            "review_agent_metadata",
            "quality_signal_context",
            "hitl_decision_context",
            "refinement_handoff_metadata",
        ),
        advisory_outputs=(
            "review_conversation_context",
            "manual_review_handoff_hint",
            "no_output_mutation_notice",
        ),
    ),
    _agent_conversation_view_profile(
        conversation_view_profile_id="audit_trail_conversation_view",
        profile_name="Audit Trail Conversation View",
        conversation_view_kind="audit_trail_view",
        source_workspace_profile_ids=tuple(
            AGENT_WORKSPACE_REGISTRY.workspace_profile_ids
        ),
        source_agent_ids=(
            "planner_agent",
            "artifact_agent",
            "critic_agent",
            "refiner_agent",
            "final_synthesizer_agent",
        ),
        source_hitl_decision_profile_ids=tuple(
            HITL_DECISION_REGISTRY.hitl_decision_profile_ids
        ),
        route_applicability=tuple(RouteName),
        conversation_surfaces=tuple(_AGENT_CONVERSATION_VIEW_SURFACES),
        visible_thread_fields=(
            "conversation_profile_metadata",
            "workspace_source_metadata",
            "handoff_profile_metadata",
            "hitl_decision_metadata",
            "authority_boundary_snapshot",
        ),
        advisory_outputs=(
            "audit_trail_conversation_context",
            "manual_audit_trace_hint",
            "no_persistent_replay_storage_notice",
        ),
    ),
)

AGENT_CONVERSATION_VIEW_REGISTRY = AgentConversationViewRegistry(
    conversation_view_profiles=AGENT_CONVERSATION_VIEW_PROFILES,
    conversation_view_profile_ids=tuple(
        profile.conversation_view_profile_id
        for profile in AGENT_CONVERSATION_VIEW_PROFILES
    ),
    conversation_view_kinds=tuple(
        profile.conversation_view_kind for profile in AGENT_CONVERSATION_VIEW_PROFILES
    ),
    workspace_profile_ids=tuple(AGENT_WORKSPACE_REGISTRY.workspace_profile_ids),
    agent_ids=tuple(AGENT_IDENTITY_REGISTRY.agent_ids),
    role_ids=tuple(AGENT_ROLE_REGISTRY.role_ids),
    shared_context_view_ids=tuple(SHARED_CONTEXT_VIEW_REGISTRY.view_ids),
    memory_contract_ids=tuple(
        contract.memory_contract_id
        for contract in AGENT_MEMORY_CONTRACT_REGISTRY.contracts
    ),
    handoff_profile_ids=tuple(WORKFLOW_AGENT_HANDOFF_REGISTRY.profile_ids),
    hitl_decision_profile_ids=tuple(HITL_DECISION_REGISTRY.hitl_decision_profile_ids),
    conversation_surface_refs=AGENT_CONVERSATION_VIEW_SURFACES,
    route_names=tuple(RouteName),
    profile_count=len(AGENT_CONVERSATION_VIEW_PROFILES),
    source_registries=_AGENT_CONVERSATION_VIEW_SOURCE_REGISTRIES,
    observability_surfaces=_AGENT_CONVERSATION_VIEW_OBSERVABILITY_SURFACES,
)

WorkspaceSnapshotKind = Literal[
    "studio_overview_snapshot",
    "agent_context_snapshot",
    "execution_context_snapshot",
    "review_audit_snapshot",
]

WORKSPACE_SNAPSHOT_PROFILE_SERIALIZATION_VERSION = "workspace_snapshot_profile.v1"
WORKSPACE_SNAPSHOT_REGISTRY_SERIALIZATION_VERSION = "workspace_snapshot_registry.v1"
WORKSPACE_SNAPSHOT_REGISTRY_AUTHORITY_BOUNDARY = (
    "Workspace Snapshot metadata describes passive Studio-visible summary "
    "snapshots over existing agent workspace, conversation view, local/cloud "
    "comparison, execution simulation, quality, and HITL decision metadata "
    "for V4.4 inspection only; it does not capture live workspace state, "
    "persist snapshots, record conversations, invoke agents, read or write "
    "memory, mutate workspace state, control workflow transitions, request "
    "human input, route providers or models, trigger retries, write replay "
    "storage, or modify generated output."
)

_WORKSPACE_SNAPSHOT_SOURCE_REGISTRIES = (
    "agent_workspace_registry",
    "agent_conversation_view_registry",
    "local_cloud_comparison_registry",
    "execution_simulator_registry",
    "quality_profile_registry",
    "hitl_decision_registry",
    "studio_mode_registry",
)

_WORKSPACE_SNAPSHOT_SURFACES = (
    "workspace_snapshot_panel",
    "snapshot_summary_strip",
    "snapshot_context_matrix",
    "conversation_snapshot_panel",
    "execution_snapshot_panel",
    "review_snapshot_panel",
)

_WORKSPACE_SNAPSHOT_OBSERVABILITY_SURFACES = (
    "workspace_snapshot_profile_id",
    "snapshot_kind",
    "source_workspace_profile_ids",
    "source_conversation_view_profile_ids",
    "snapshot_context_fields",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_WORKSPACE_SNAPSHOT_BLOCKED_RUNTIME_BEHAVIORS = (
    "live_workspace_capture",
    "runtime_state_capture",
    "snapshot_persistence",
    "conversation_recording",
    "agent_invocation",
    "memory_read",
    "memory_write",
    "workspace_state_mutation",
    "workflow_control",
    "human_input_request",
    "provider_or_model_routing",
    "retry_or_refinement_triggering",
    "persistent_replay_storage",
    "generated_output_modification",
)


class WorkspaceSnapshotProfile(BaseModel):
    """Inspectable passive Studio workspace snapshot metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    workspace_snapshot_profile_id: str = Field(min_length=1, max_length=140)
    profile_name: str = Field(min_length=1, max_length=160)
    snapshot_kind: WorkspaceSnapshotKind
    source_workspace_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_conversation_view_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_comparison_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_execution_simulation_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_quality_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_hitl_decision_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    snapshot_surfaces: tuple[str, ...] = Field(min_length=1, max_length=6)
    snapshot_context_fields: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=WORKSPACE_SNAPSHOT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1400,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_WORKSPACE_SNAPSHOT_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    snapshot_capture_implemented: Literal[False] = False
    snapshot_persistence_implemented: Literal[False] = False
    conversation_recording_implemented: Literal[False] = False
    live_workspace_state_read_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    memory_read_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    workspace_state_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    serialization_version: Literal["workspace_snapshot_profile.v1"] = (
        WORKSPACE_SNAPSHOT_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class WorkspaceSnapshotRegistry(BaseModel):
    """Stable passive registry for V4.4 Hybrid Studio workspace snapshots."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["workspace_snapshot_registry"] = "workspace_snapshot_registry"
    serialization_version: Literal["workspace_snapshot_registry.v1"] = (
        WORKSPACE_SNAPSHOT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=WORKSPACE_SNAPSHOT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1400,
    )
    snapshot_profiles: tuple[WorkspaceSnapshotProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    workspace_snapshot_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    snapshot_kinds: tuple[WorkspaceSnapshotKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    workspace_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    conversation_view_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    comparison_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    execution_simulation_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    quality_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    hitl_decision_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    snapshot_surface_refs: tuple[str, ...] = Field(min_length=6, max_length=6)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_WORKSPACE_SNAPSHOT_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    snapshot_capture_implemented: Literal[False] = False
    snapshot_persistence_implemented: Literal[False] = False
    conversation_recording_implemented: Literal[False] = False
    live_workspace_state_read_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    memory_read_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    workspace_state_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.workspace_snapshot_profile_id for profile in self.snapshot_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("workspace_snapshot_profile_ids must be unique")
        if self.workspace_snapshot_profile_ids != derived_profile_ids:
            raise ValueError(
                "workspace_snapshot_profile_ids must match snapshot_profiles"
            )
        if self.profile_count != len(self.snapshot_profiles):
            raise ValueError("profile_count must match snapshot_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.snapshot_kinds != tuple(
            profile.snapshot_kind for profile in self.snapshot_profiles
        ):
            raise ValueError("snapshot_kinds must match snapshot_profiles")

        known_routes = set(self.route_names)
        known_workspaces = set(self.workspace_profile_ids)
        known_conversation_views = set(self.conversation_view_profile_ids)
        known_comparisons = set(self.comparison_profile_ids)
        known_simulation_profiles = set(self.execution_simulation_profile_ids)
        known_quality_profiles = set(self.quality_profile_ids)
        known_hitl_profiles = set(self.hitl_decision_profile_ids)
        known_surfaces = set(self.snapshot_surface_refs)
        profile_sources = {
            source_registry
            for profile in self.snapshot_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match workspace snapshot sources")

        for profile in self.snapshot_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("profile source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.source_workspace_profile_ids).issubset(known_workspaces):
                raise ValueError(
                    "source_workspace_profile_ids must be known workspace profiles"
                )
            if not set(profile.source_conversation_view_profile_ids).issubset(
                known_conversation_views
            ):
                raise ValueError(
                    "source_conversation_view_profile_ids must be known views"
                )
            if not set(profile.source_comparison_profile_ids).issubset(
                known_comparisons
            ):
                raise ValueError("source_comparison_profile_ids must be known profiles")
            if not set(profile.source_execution_simulation_profile_ids).issubset(
                known_simulation_profiles
            ):
                raise ValueError(
                    "source_execution_simulation_profile_ids must be known profiles"
                )
            if not set(profile.source_quality_profile_ids).issubset(
                known_quality_profiles
            ):
                raise ValueError("source_quality_profile_ids must be known profiles")
            if not set(profile.source_hitl_decision_profile_ids).issubset(
                known_hitl_profiles
            ):
                raise ValueError(
                    "source_hitl_decision_profile_ids must be known profiles"
                )
            if not set(profile.snapshot_surfaces).issubset(known_surfaces):
                raise ValueError("snapshot_surfaces must be known registry surfaces")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
        return self


def workspace_snapshot_registry() -> WorkspaceSnapshotRegistry:
    """Return passive V4.4 Hybrid Studio workspace snapshot metadata."""

    return WORKSPACE_SNAPSHOT_REGISTRY


def workspace_snapshot_profile_by_id(
    workspace_snapshot_profile_id: str,
    registry: WorkspaceSnapshotRegistry | None = None,
) -> WorkspaceSnapshotProfile | None:
    """Return one workspace snapshot profile without capturing runtime state."""

    source_registry = registry or WORKSPACE_SNAPSHOT_REGISTRY
    for profile in source_registry.snapshot_profiles:
        if profile.workspace_snapshot_profile_id == workspace_snapshot_profile_id:
            return profile
    return None


def workspace_snapshot_profiles_for_route(
    route: RouteName | str,
    registry: WorkspaceSnapshotRegistry | None = None,
) -> tuple[WorkspaceSnapshotProfile, ...]:
    """Return passive workspace snapshots applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or WORKSPACE_SNAPSHOT_REGISTRY
    return tuple(
        profile
        for profile in source_registry.snapshot_profiles
        if route_name in profile.route_applicability
    )


def workspace_snapshot_profiles_for_workspace(
    workspace_profile_id: str,
    registry: WorkspaceSnapshotRegistry | None = None,
) -> tuple[WorkspaceSnapshotProfile, ...]:
    """Return passive snapshot profiles for a workspace profile id."""

    source_registry = registry or WORKSPACE_SNAPSHOT_REGISTRY
    workspace_id = str(workspace_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.snapshot_profiles
        if workspace_id in profile.source_workspace_profile_ids
    )


def workspace_snapshot_profiles_for_conversation_view(
    conversation_view_profile_id: str,
    registry: WorkspaceSnapshotRegistry | None = None,
) -> tuple[WorkspaceSnapshotProfile, ...]:
    """Return passive snapshot profiles for a conversation view id."""

    source_registry = registry or WORKSPACE_SNAPSHOT_REGISTRY
    view_id = str(conversation_view_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.snapshot_profiles
        if view_id in profile.source_conversation_view_profile_ids
    )


def _workspace_snapshot_profile(
    *,
    workspace_snapshot_profile_id: str,
    profile_name: str,
    snapshot_kind: WorkspaceSnapshotKind,
    source_workspace_profile_ids: tuple[str, ...],
    source_conversation_view_profile_ids: tuple[str, ...],
    source_comparison_profile_ids: tuple[str, ...],
    source_execution_simulation_profile_ids: tuple[str, ...],
    source_quality_profile_ids: tuple[str, ...],
    source_hitl_decision_profile_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    snapshot_surfaces: tuple[str, ...],
    snapshot_context_fields: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> WorkspaceSnapshotProfile:
    return WorkspaceSnapshotProfile(
        workspace_snapshot_profile_id=workspace_snapshot_profile_id,
        profile_name=profile_name,
        snapshot_kind=snapshot_kind,
        source_workspace_profile_ids=source_workspace_profile_ids,
        source_conversation_view_profile_ids=source_conversation_view_profile_ids,
        source_comparison_profile_ids=source_comparison_profile_ids,
        source_execution_simulation_profile_ids=(
            source_execution_simulation_profile_ids
        ),
        source_quality_profile_ids=source_quality_profile_ids,
        source_hitl_decision_profile_ids=source_hitl_decision_profile_ids,
        route_applicability=route_applicability,
        snapshot_surfaces=snapshot_surfaces,
        snapshot_context_fields=snapshot_context_fields,
        advisory_outputs=advisory_outputs,
        source_registries=_WORKSPACE_SNAPSHOT_SOURCE_REGISTRIES,
        observability_surfaces=_WORKSPACE_SNAPSHOT_OBSERVABILITY_SURFACES,
    )


WORKSPACE_SNAPSHOT_SURFACES = (
    "workspace_snapshot_panel",
    "snapshot_summary_strip",
    "snapshot_context_matrix",
    "conversation_snapshot_panel",
    "execution_snapshot_panel",
    "review_snapshot_panel",
)

WORKSPACE_SNAPSHOT_PROFILES = (
    _workspace_snapshot_profile(
        workspace_snapshot_profile_id="studio_overview_workspace_snapshot",
        profile_name="Studio Overview Workspace Snapshot",
        snapshot_kind="studio_overview_snapshot",
        source_workspace_profile_ids=(
            "planning_context_agent_workspace",
            "artifact_runtime_agent_workspace",
        ),
        source_conversation_view_profile_ids=(
            "workspace_thread_conversation_view",
            "audit_trail_conversation_view",
        ),
        source_comparison_profile_ids=(
            "generation_route_comparison_profile",
            "creative_reasoning_comparison_profile",
        ),
        source_execution_simulation_profile_ids=(
            "route_preview_simulation_profile",
            "local_cloud_comparison_simulation_profile",
        ),
        source_quality_profile_ids=(
            "planning_quality_profile",
            "creative_quality_profile",
        ),
        source_hitl_decision_profile_ids=(
            "hitl_visibility_decision_profile",
            "hitl_confirmation_decision_profile",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.PREVIEW,
        ),
        snapshot_surfaces=(
            "workspace_snapshot_panel",
            "snapshot_summary_strip",
            "conversation_snapshot_panel",
            "execution_snapshot_panel",
        ),
        snapshot_context_fields=(
            "workspace_profile_metadata",
            "conversation_view_metadata",
            "route_applicability_metadata",
            "manual_visibility_metadata",
        ),
        advisory_outputs=(
            "studio_overview_snapshot_context",
            "manual_snapshot_review_hint",
            "no_live_workspace_capture_notice",
        ),
    ),
    _workspace_snapshot_profile(
        workspace_snapshot_profile_id="agent_context_workspace_snapshot",
        profile_name="Agent Context Workspace Snapshot",
        snapshot_kind="agent_context_snapshot",
        source_workspace_profile_ids=(
            "planning_context_agent_workspace",
            "artifact_runtime_agent_workspace",
            "critique_curation_agent_workspace",
        ),
        source_conversation_view_profile_ids=(
            "workspace_thread_conversation_view",
            "agent_handoff_conversation_view",
        ),
        source_comparison_profile_ids=(
            "generation_route_comparison_profile",
            "code_review_comparison_profile",
        ),
        source_execution_simulation_profile_ids=(
            "route_preview_simulation_profile",
            "provider_selection_simulation_profile",
        ),
        source_quality_profile_ids=(
            "planning_quality_profile",
            "refinement_quality_profile",
        ),
        source_hitl_decision_profile_ids=(
            "hitl_confirmation_decision_profile",
            "hitl_risk_review_decision_profile",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.DESIGN,
        ),
        snapshot_surfaces=(
            "workspace_snapshot_panel",
            "snapshot_context_matrix",
            "conversation_snapshot_panel",
        ),
        snapshot_context_fields=(
            "agent_workspace_metadata",
            "handoff_conversation_metadata",
            "shared_context_scope_metadata",
            "role_matrix_snapshot_metadata",
        ),
        advisory_outputs=(
            "agent_context_snapshot_context",
            "manual_agent_context_review_hint",
            "no_agent_invocation_notice",
        ),
    ),
    _workspace_snapshot_profile(
        workspace_snapshot_profile_id="execution_context_workspace_snapshot",
        profile_name="Execution Context Workspace Snapshot",
        snapshot_kind="execution_context_snapshot",
        source_workspace_profile_ids=(
            "artifact_runtime_agent_workspace",
            "refinement_synthesis_agent_workspace",
        ),
        source_conversation_view_profile_ids=(
            "agent_handoff_conversation_view",
            "audit_trail_conversation_view",
        ),
        source_comparison_profile_ids=(
            "generation_route_comparison_profile",
            "code_review_comparison_profile",
            "evaluation_review_comparison_profile",
        ),
        source_execution_simulation_profile_ids=(
            "local_cloud_comparison_simulation_profile",
            "provider_selection_simulation_profile",
            "hitl_review_simulation_profile",
        ),
        source_quality_profile_ids=(
            "refinement_quality_profile",
            "final_review_quality_profile",
        ),
        source_hitl_decision_profile_ids=(
            "hitl_confirmation_decision_profile",
            "hitl_risk_review_decision_profile",
            "hitl_final_review_decision_profile",
        ),
        route_applicability=(
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        snapshot_surfaces=(
            "workspace_snapshot_panel",
            "snapshot_context_matrix",
            "execution_snapshot_panel",
            "review_snapshot_panel",
        ),
        snapshot_context_fields=(
            "execution_simulation_metadata",
            "local_cloud_comparison_metadata",
            "quality_review_metadata",
            "hitl_decision_metadata",
        ),
        advisory_outputs=(
            "execution_context_snapshot_context",
            "manual_execution_snapshot_review_hint",
            "no_provider_execution_notice",
        ),
    ),
    _workspace_snapshot_profile(
        workspace_snapshot_profile_id="review_audit_workspace_snapshot",
        profile_name="Review Audit Workspace Snapshot",
        snapshot_kind="review_audit_snapshot",
        source_workspace_profile_ids=(
            "critique_curation_agent_workspace",
            "refinement_synthesis_agent_workspace",
        ),
        source_conversation_view_profile_ids=(
            "review_conversation_view",
            "audit_trail_conversation_view",
        ),
        source_comparison_profile_ids=(
            "creative_reasoning_comparison_profile",
            "code_review_comparison_profile",
            "evaluation_review_comparison_profile",
        ),
        source_execution_simulation_profile_ids=(
            "local_cloud_comparison_simulation_profile",
            "hitl_review_simulation_profile",
        ),
        source_quality_profile_ids=(
            "creative_quality_profile",
            "refinement_quality_profile",
            "final_review_quality_profile",
        ),
        source_hitl_decision_profile_ids=(
            "hitl_risk_review_decision_profile",
            "hitl_final_review_decision_profile",
        ),
        route_applicability=(
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        snapshot_surfaces=(
            "workspace_snapshot_panel",
            "snapshot_summary_strip",
            "conversation_snapshot_panel",
            "review_snapshot_panel",
        ),
        snapshot_context_fields=(
            "review_conversation_metadata",
            "audit_boundary_metadata",
            "final_review_quality_metadata",
            "manual_signoff_context_metadata",
        ),
        advisory_outputs=(
            "review_audit_snapshot_context",
            "manual_audit_snapshot_review_hint",
            "no_snapshot_persistence_notice",
        ),
    ),
)

WORKSPACE_SNAPSHOT_REGISTRY = WorkspaceSnapshotRegistry(
    snapshot_profiles=WORKSPACE_SNAPSHOT_PROFILES,
    workspace_snapshot_profile_ids=tuple(
        profile.workspace_snapshot_profile_id for profile in WORKSPACE_SNAPSHOT_PROFILES
    ),
    snapshot_kinds=tuple(
        profile.snapshot_kind for profile in WORKSPACE_SNAPSHOT_PROFILES
    ),
    workspace_profile_ids=tuple(AGENT_WORKSPACE_REGISTRY.workspace_profile_ids),
    conversation_view_profile_ids=tuple(
        AGENT_CONVERSATION_VIEW_REGISTRY.conversation_view_profile_ids
    ),
    comparison_profile_ids=tuple(
        LOCAL_CLOUD_COMPARISON_REGISTRY.comparison_profile_ids
    ),
    execution_simulation_profile_ids=tuple(
        EXECUTION_SIMULATOR_REGISTRY.execution_simulation_profile_ids
    ),
    quality_profile_ids=tuple(QUALITY_PROFILE_REGISTRY.quality_profile_ids),
    hitl_decision_profile_ids=tuple(HITL_DECISION_REGISTRY.hitl_decision_profile_ids),
    snapshot_surface_refs=WORKSPACE_SNAPSHOT_SURFACES,
    route_names=tuple(RouteName),
    profile_count=len(WORKSPACE_SNAPSHOT_PROFILES),
    source_registries=_WORKSPACE_SNAPSHOT_SOURCE_REGISTRIES,
    observability_surfaces=_WORKSPACE_SNAPSHOT_OBSERVABILITY_SURFACES,
)

SessionReplayKind = Literal[
    "session_overview_replay",
    "conversation_timeline_replay",
    "snapshot_transition_replay",
    "review_decision_replay",
]

SESSION_REPLAY_PROFILE_SERIALIZATION_VERSION = "session_replay_profile.v1"
SESSION_REPLAY_REGISTRY_SERIALIZATION_VERSION = "session_replay_registry.v1"
SESSION_REPLAY_REGISTRY_AUTHORITY_BOUNDARY = (
    "Session Replay metadata describes passive Studio-visible replay views "
    "over existing workspace snapshot, conversation view, workspace, HITL, "
    "Studio Mode, and Auto Mode metadata for V4.4 inspection only; it does "
    "not record sessions, reconstruct timelines, persist replay data, replay "
    "runtime events, persist conversations, capture snapshots, invoke agents, "
    "read or write memory, mutate workspace state, control workflow "
    "transitions, request human input, route providers or models, trigger "
    "retries, write replay storage, or modify generated output."
)

_SESSION_REPLAY_SOURCE_REGISTRIES = (
    "workspace_snapshot_registry",
    "agent_conversation_view_registry",
    "agent_workspace_registry",
    "hitl_decision_registry",
    "studio_mode_registry",
    "auto_mode_registry",
)

_SESSION_REPLAY_SURFACES = (
    "session_replay_panel",
    "session_timeline_strip",
    "conversation_replay_panel",
    "snapshot_replay_panel",
    "decision_replay_panel",
    "replay_boundary_panel",
)

_SESSION_REPLAY_OBSERVABILITY_SURFACES = (
    "session_replay_profile_id",
    "session_replay_kind",
    "source_workspace_snapshot_profile_ids",
    "source_conversation_view_profile_ids",
    "route_applicability",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_SESSION_REPLAY_BLOCKED_RUNTIME_BEHAVIORS = (
    "session_replay_execution",
    "session_recording",
    "timeline_reconstruction",
    "replay_persistence",
    "conversation_persistence",
    "snapshot_capture",
    "agent_invocation",
    "memory_read",
    "memory_write",
    "workspace_state_mutation",
    "workflow_control",
    "human_input_request",
    "provider_or_model_routing",
    "retry_or_refinement_triggering",
    "persistent_replay_storage",
    "generated_output_modification",
)


class SessionReplayProfile(BaseModel):
    """Inspectable passive Studio session replay metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    session_replay_profile_id: str = Field(min_length=1, max_length=140)
    profile_name: str = Field(min_length=1, max_length=160)
    session_replay_kind: SessionReplayKind
    source_workspace_snapshot_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_conversation_view_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_workspace_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_hitl_decision_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_studio_mode_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_auto_mode_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    replay_surfaces: tuple[str, ...] = Field(min_length=1, max_length=6)
    replay_context_fields: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=SESSION_REPLAY_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_SESSION_REPLAY_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    session_replay_execution_implemented: Literal[False] = False
    session_recording_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    replay_persistence_implemented: Literal[False] = False
    conversation_persistence_implemented: Literal[False] = False
    snapshot_capture_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    memory_read_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    workspace_state_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    serialization_version: Literal["session_replay_profile.v1"] = (
        SESSION_REPLAY_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class SessionReplayRegistry(BaseModel):
    """Stable passive registry for V4.4 Hybrid Studio session replay metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["session_replay_registry"] = "session_replay_registry"
    serialization_version: Literal["session_replay_registry.v1"] = (
        SESSION_REPLAY_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=SESSION_REPLAY_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    session_replay_profiles: tuple[SessionReplayProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    session_replay_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    session_replay_kinds: tuple[SessionReplayKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    workspace_snapshot_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    conversation_view_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    workspace_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    hitl_decision_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    studio_mode_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    auto_mode_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    replay_surface_refs: tuple[str, ...] = Field(min_length=6, max_length=6)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_SESSION_REPLAY_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    session_replay_execution_implemented: Literal[False] = False
    session_recording_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    replay_persistence_implemented: Literal[False] = False
    conversation_persistence_implemented: Literal[False] = False
    snapshot_capture_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    memory_read_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    workspace_state_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.session_replay_profile_id
            for profile in self.session_replay_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("session_replay_profile_ids must be unique")
        if self.session_replay_profile_ids != derived_profile_ids:
            raise ValueError(
                "session_replay_profile_ids must match session_replay_profiles"
            )
        if self.profile_count != len(self.session_replay_profiles):
            raise ValueError("profile_count must match session_replay_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.session_replay_kinds != tuple(
            profile.session_replay_kind for profile in self.session_replay_profiles
        ):
            raise ValueError("session_replay_kinds must match session_replay_profiles")

        known_routes = set(self.route_names)
        known_snapshots = set(self.workspace_snapshot_profile_ids)
        known_conversation_views = set(self.conversation_view_profile_ids)
        known_workspaces = set(self.workspace_profile_ids)
        known_hitl_profiles = set(self.hitl_decision_profile_ids)
        known_studio_profiles = set(self.studio_mode_profile_ids)
        known_auto_profiles = set(self.auto_mode_profile_ids)
        known_surfaces = set(self.replay_surface_refs)
        profile_sources = {
            source_registry
            for profile in self.session_replay_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match session replay sources")

        for profile in self.session_replay_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("profile source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.source_workspace_snapshot_profile_ids).issubset(
                known_snapshots
            ):
                raise ValueError(
                    "source_workspace_snapshot_profile_ids must be known snapshots"
                )
            if not set(profile.source_conversation_view_profile_ids).issubset(
                known_conversation_views
            ):
                raise ValueError(
                    "source_conversation_view_profile_ids must be known views"
                )
            if not set(profile.source_workspace_profile_ids).issubset(known_workspaces):
                raise ValueError(
                    "source_workspace_profile_ids must be known workspace profiles"
                )
            if not set(profile.source_hitl_decision_profile_ids).issubset(
                known_hitl_profiles
            ):
                raise ValueError(
                    "source_hitl_decision_profile_ids must be known profiles"
                )
            if not set(profile.source_studio_mode_profile_ids).issubset(
                known_studio_profiles
            ):
                raise ValueError(
                    "source_studio_mode_profile_ids must be known Studio Mode profiles"
                )
            if not set(profile.source_auto_mode_profile_ids).issubset(
                known_auto_profiles
            ):
                raise ValueError(
                    "source_auto_mode_profile_ids must be known Auto Mode profiles"
                )
            if not set(profile.replay_surfaces).issubset(known_surfaces):
                raise ValueError("replay_surfaces must be known registry surfaces")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
        return self


def session_replay_registry() -> SessionReplayRegistry:
    """Return passive V4.4 Hybrid Studio session replay metadata."""

    return SESSION_REPLAY_REGISTRY


def session_replay_profile_by_id(
    session_replay_profile_id: str,
    registry: SessionReplayRegistry | None = None,
) -> SessionReplayProfile | None:
    """Return one session replay profile without replaying runtime events."""

    source_registry = registry or SESSION_REPLAY_REGISTRY
    for profile in source_registry.session_replay_profiles:
        if profile.session_replay_profile_id == session_replay_profile_id:
            return profile
    return None


def session_replay_profiles_for_route(
    route: RouteName | str,
    registry: SessionReplayRegistry | None = None,
) -> tuple[SessionReplayProfile, ...]:
    """Return passive session replay profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or SESSION_REPLAY_REGISTRY
    return tuple(
        profile
        for profile in source_registry.session_replay_profiles
        if route_name in profile.route_applicability
    )


def session_replay_profiles_for_workspace_snapshot(
    workspace_snapshot_profile_id: str,
    registry: SessionReplayRegistry | None = None,
) -> tuple[SessionReplayProfile, ...]:
    """Return passive session replay profiles for a workspace snapshot id."""

    source_registry = registry or SESSION_REPLAY_REGISTRY
    snapshot_id = str(workspace_snapshot_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.session_replay_profiles
        if snapshot_id in profile.source_workspace_snapshot_profile_ids
    )


def session_replay_profiles_for_conversation_view(
    conversation_view_profile_id: str,
    registry: SessionReplayRegistry | None = None,
) -> tuple[SessionReplayProfile, ...]:
    """Return passive session replay profiles for a conversation view id."""

    source_registry = registry or SESSION_REPLAY_REGISTRY
    view_id = str(conversation_view_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.session_replay_profiles
        if view_id in profile.source_conversation_view_profile_ids
    )


def _session_replay_profile(
    *,
    session_replay_profile_id: str,
    profile_name: str,
    session_replay_kind: SessionReplayKind,
    source_workspace_snapshot_profile_ids: tuple[str, ...],
    source_conversation_view_profile_ids: tuple[str, ...],
    source_workspace_profile_ids: tuple[str, ...],
    source_hitl_decision_profile_ids: tuple[str, ...],
    source_studio_mode_profile_ids: tuple[str, ...],
    source_auto_mode_profile_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    replay_surfaces: tuple[str, ...],
    replay_context_fields: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> SessionReplayProfile:
    return SessionReplayProfile(
        session_replay_profile_id=session_replay_profile_id,
        profile_name=profile_name,
        session_replay_kind=session_replay_kind,
        source_workspace_snapshot_profile_ids=source_workspace_snapshot_profile_ids,
        source_conversation_view_profile_ids=source_conversation_view_profile_ids,
        source_workspace_profile_ids=source_workspace_profile_ids,
        source_hitl_decision_profile_ids=source_hitl_decision_profile_ids,
        source_studio_mode_profile_ids=source_studio_mode_profile_ids,
        source_auto_mode_profile_ids=source_auto_mode_profile_ids,
        route_applicability=route_applicability,
        replay_surfaces=replay_surfaces,
        replay_context_fields=replay_context_fields,
        advisory_outputs=advisory_outputs,
        source_registries=_SESSION_REPLAY_SOURCE_REGISTRIES,
        observability_surfaces=_SESSION_REPLAY_OBSERVABILITY_SURFACES,
    )


SESSION_REPLAY_SURFACES = (
    "session_replay_panel",
    "session_timeline_strip",
    "conversation_replay_panel",
    "snapshot_replay_panel",
    "decision_replay_panel",
    "replay_boundary_panel",
)

SESSION_REPLAY_PROFILES = (
    _session_replay_profile(
        session_replay_profile_id="session_overview_replay_profile",
        profile_name="Session Overview Replay Profile",
        session_replay_kind="session_overview_replay",
        source_workspace_snapshot_profile_ids=(
            "studio_overview_workspace_snapshot",
            "agent_context_workspace_snapshot",
        ),
        source_conversation_view_profile_ids=(
            "workspace_thread_conversation_view",
            "audit_trail_conversation_view",
        ),
        source_workspace_profile_ids=(
            "planning_context_agent_workspace",
            "artifact_runtime_agent_workspace",
        ),
        source_hitl_decision_profile_ids=(
            "hitl_visibility_decision_profile",
            "hitl_confirmation_decision_profile",
        ),
        source_studio_mode_profile_ids=(
            "studio_mode_inspection_profile",
            "studio_mode_operator_review_profile",
        ),
        source_auto_mode_profile_ids=(
            "auto_mode_observe_only_profile",
            "auto_mode_operator_confirmed_profile",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.PREVIEW,
        ),
        replay_surfaces=(
            "session_replay_panel",
            "session_timeline_strip",
            "snapshot_replay_panel",
            "replay_boundary_panel",
        ),
        replay_context_fields=(
            "session_metadata_summary",
            "workspace_snapshot_refs",
            "conversation_view_refs",
            "operator_visibility_refs",
        ),
        advisory_outputs=(
            "session_overview_replay_context",
            "manual_session_review_hint",
            "no_session_recording_notice",
        ),
    ),
    _session_replay_profile(
        session_replay_profile_id="conversation_timeline_replay_profile",
        profile_name="Conversation Timeline Replay Profile",
        session_replay_kind="conversation_timeline_replay",
        source_workspace_snapshot_profile_ids=(
            "agent_context_workspace_snapshot",
            "review_audit_workspace_snapshot",
        ),
        source_conversation_view_profile_ids=(
            "workspace_thread_conversation_view",
            "agent_handoff_conversation_view",
            "review_conversation_view",
        ),
        source_workspace_profile_ids=(
            "planning_context_agent_workspace",
            "critique_curation_agent_workspace",
            "refinement_synthesis_agent_workspace",
        ),
        source_hitl_decision_profile_ids=(
            "hitl_confirmation_decision_profile",
            "hitl_risk_review_decision_profile",
            "hitl_final_review_decision_profile",
        ),
        source_studio_mode_profile_ids=(
            "studio_mode_inspection_profile",
            "studio_mode_comparison_profile",
            "studio_mode_operator_review_profile",
        ),
        source_auto_mode_profile_ids=(
            "auto_mode_observe_only_profile",
            "auto_mode_suggestion_profile",
            "auto_mode_operator_confirmed_profile",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        replay_surfaces=(
            "session_replay_panel",
            "session_timeline_strip",
            "conversation_replay_panel",
        ),
        replay_context_fields=(
            "conversation_view_sequence_metadata",
            "agent_handoff_refs",
            "shared_context_scope_refs",
            "manual_note_refs",
        ),
        advisory_outputs=(
            "conversation_timeline_replay_context",
            "manual_conversation_trace_hint",
            "no_timeline_reconstruction_notice",
        ),
    ),
    _session_replay_profile(
        session_replay_profile_id="snapshot_transition_replay_profile",
        profile_name="Snapshot Transition Replay Profile",
        session_replay_kind="snapshot_transition_replay",
        source_workspace_snapshot_profile_ids=tuple(
            WORKSPACE_SNAPSHOT_REGISTRY.workspace_snapshot_profile_ids
        ),
        source_conversation_view_profile_ids=(
            "agent_handoff_conversation_view",
            "audit_trail_conversation_view",
        ),
        source_workspace_profile_ids=tuple(
            AGENT_WORKSPACE_REGISTRY.workspace_profile_ids
        ),
        source_hitl_decision_profile_ids=tuple(
            HITL_DECISION_REGISTRY.hitl_decision_profile_ids
        ),
        source_studio_mode_profile_ids=(
            "studio_mode_comparison_profile",
            "studio_mode_simulation_profile",
            "studio_mode_operator_review_profile",
        ),
        source_auto_mode_profile_ids=(
            "auto_mode_suggestion_profile",
            "auto_mode_simulation_profile",
            "auto_mode_operator_confirmed_profile",
        ),
        route_applicability=(
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        replay_surfaces=(
            "session_replay_panel",
            "session_timeline_strip",
            "snapshot_replay_panel",
            "replay_boundary_panel",
        ),
        replay_context_fields=(
            "workspace_snapshot_order_metadata",
            "snapshot_transition_refs",
            "execution_context_snapshot_refs",
            "review_snapshot_refs",
        ),
        advisory_outputs=(
            "snapshot_transition_replay_context",
            "manual_transition_review_hint",
            "no_snapshot_capture_notice",
        ),
    ),
    _session_replay_profile(
        session_replay_profile_id="review_decision_replay_profile",
        profile_name="Review Decision Replay Profile",
        session_replay_kind="review_decision_replay",
        source_workspace_snapshot_profile_ids=(
            "execution_context_workspace_snapshot",
            "review_audit_workspace_snapshot",
        ),
        source_conversation_view_profile_ids=(
            "review_conversation_view",
            "audit_trail_conversation_view",
        ),
        source_workspace_profile_ids=(
            "critique_curation_agent_workspace",
            "refinement_synthesis_agent_workspace",
        ),
        source_hitl_decision_profile_ids=(
            "hitl_risk_review_decision_profile",
            "hitl_final_review_decision_profile",
        ),
        source_studio_mode_profile_ids=(
            "studio_mode_simulation_profile",
            "studio_mode_operator_review_profile",
        ),
        source_auto_mode_profile_ids=(
            "auto_mode_simulation_profile",
            "auto_mode_operator_confirmed_profile",
        ),
        route_applicability=(
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        replay_surfaces=(
            "session_replay_panel",
            "conversation_replay_panel",
            "snapshot_replay_panel",
            "decision_replay_panel",
            "replay_boundary_panel",
        ),
        replay_context_fields=(
            "hitl_decision_refs",
            "review_snapshot_refs",
            "audit_boundary_refs",
            "manual_signoff_refs",
        ),
        advisory_outputs=(
            "review_decision_replay_context",
            "manual_decision_trace_hint",
            "no_human_input_request_notice",
        ),
    ),
)

SESSION_REPLAY_REGISTRY = SessionReplayRegistry(
    session_replay_profiles=SESSION_REPLAY_PROFILES,
    session_replay_profile_ids=tuple(
        profile.session_replay_profile_id for profile in SESSION_REPLAY_PROFILES
    ),
    session_replay_kinds=tuple(
        profile.session_replay_kind for profile in SESSION_REPLAY_PROFILES
    ),
    workspace_snapshot_profile_ids=tuple(
        WORKSPACE_SNAPSHOT_REGISTRY.workspace_snapshot_profile_ids
    ),
    conversation_view_profile_ids=tuple(
        AGENT_CONVERSATION_VIEW_REGISTRY.conversation_view_profile_ids
    ),
    workspace_profile_ids=tuple(AGENT_WORKSPACE_REGISTRY.workspace_profile_ids),
    hitl_decision_profile_ids=tuple(HITL_DECISION_REGISTRY.hitl_decision_profile_ids),
    studio_mode_profile_ids=tuple(STUDIO_MODE_REGISTRY.studio_mode_profile_ids),
    auto_mode_profile_ids=tuple(AUTO_MODE_REGISTRY.auto_mode_profile_ids),
    replay_surface_refs=SESSION_REPLAY_SURFACES,
    route_names=tuple(RouteName),
    profile_count=len(SESSION_REPLAY_PROFILES),
    source_registries=_SESSION_REPLAY_SOURCE_REGISTRIES,
    observability_surfaces=_SESSION_REPLAY_OBSERVABILITY_SURFACES,
)

ExecutionReplayKind = Literal[
    "route_execution_replay",
    "provider_selection_replay",
    "local_cloud_execution_replay",
    "quality_review_execution_replay",
]

EXECUTION_REPLAY_PROFILE_SERIALIZATION_VERSION = "execution_replay_profile.v1"
EXECUTION_REPLAY_REGISTRY_SERIALIZATION_VERSION = "execution_replay_registry.v1"
EXECUTION_REPLAY_REGISTRY_AUTHORITY_BOUNDARY = (
    "Execution Replay metadata describes passive Studio-visible execution "
    "replay views over existing session replay, execution simulation, hybrid "
    "execution, provider selection, model, cost, quality, and local/cloud "
    "comparison metadata for V4.4 inspection only; it does not execute "
    "providers, reconstruct execution traces, replay runtime events, persist "
    "replay data, select providers or models, calculate cost or quality "
    "scores, control workflows, request human input, trigger retries, write "
    "replay storage, or modify generated output."
)

_EXECUTION_REPLAY_SOURCE_REGISTRIES = (
    "session_replay_registry",
    "execution_simulator_registry",
    "hybrid_execution_registry",
    "provider_selection_registry",
    "model_profile_registry",
    "cost_profile_registry",
    "quality_profile_registry",
    "local_cloud_comparison_registry",
)

_EXECUTION_REPLAY_SURFACES = (
    "execution_replay_panel",
    "execution_trace_timeline",
    "simulation_replay_panel",
    "provider_selection_replay_panel",
    "cost_quality_replay_panel",
    "replay_boundary_panel",
)

_EXECUTION_REPLAY_OBSERVABILITY_SURFACES = (
    "execution_replay_profile_id",
    "execution_replay_kind",
    "source_session_replay_profile_ids",
    "source_execution_simulation_profile_ids",
    "route_applicability",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_EXECUTION_REPLAY_BLOCKED_RUNTIME_BEHAVIORS = (
    "execution_replay_execution",
    "provider_execution",
    "local_provider_execution",
    "cloud_provider_execution",
    "model_selection",
    "provider_or_model_routing",
    "execution_trace_reconstruction",
    "replay_persistence",
    "session_replay_execution",
    "cost_scoring",
    "quality_scoring",
    "quality_evaluation",
    "workflow_control",
    "human_input_request",
    "retry_or_refinement_triggering",
    "persistent_replay_storage",
    "generated_output_modification",
)


class ExecutionReplayProfile(BaseModel):
    """Inspectable passive Studio execution replay metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    execution_replay_profile_id: str = Field(min_length=1, max_length=140)
    profile_name: str = Field(min_length=1, max_length=160)
    execution_replay_kind: ExecutionReplayKind
    source_session_replay_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_execution_simulation_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_execution_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_provider_selection_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    source_model_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_cost_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_quality_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_comparison_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    execution_replay_surfaces: tuple[str, ...] = Field(min_length=1, max_length=6)
    replay_context_fields: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_registries: tuple[str, ...] = Field(min_length=8, max_length=8)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=EXECUTION_REPLAY_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_EXECUTION_REPLAY_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    execution_replay_execution_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_provider_execution_implemented: Literal[False] = False
    cloud_provider_execution_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    execution_trace_reconstruction_implemented: Literal[False] = False
    replay_persistence_implemented: Literal[False] = False
    session_replay_execution_implemented: Literal[False] = False
    cost_scoring_implemented: Literal[False] = False
    quality_scoring_implemented: Literal[False] = False
    quality_evaluation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    serialization_version: Literal["execution_replay_profile.v1"] = (
        EXECUTION_REPLAY_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class ExecutionReplayRegistry(BaseModel):
    """Stable passive registry for V4.4 Hybrid Studio execution replay metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["execution_replay_registry"] = "execution_replay_registry"
    serialization_version: Literal["execution_replay_registry.v1"] = (
        EXECUTION_REPLAY_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=EXECUTION_REPLAY_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    execution_replay_profiles: tuple[ExecutionReplayProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    execution_replay_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    execution_replay_kinds: tuple[ExecutionReplayKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    session_replay_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    execution_simulation_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    execution_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    provider_selection_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    model_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    cost_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    quality_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    comparison_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    execution_replay_surface_refs: tuple[str, ...] = Field(
        min_length=6,
        max_length=6,
    )
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=8, max_length=8)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_EXECUTION_REPLAY_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    execution_replay_execution_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_provider_execution_implemented: Literal[False] = False
    cloud_provider_execution_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    execution_trace_reconstruction_implemented: Literal[False] = False
    replay_persistence_implemented: Literal[False] = False
    session_replay_execution_implemented: Literal[False] = False
    cost_scoring_implemented: Literal[False] = False
    quality_scoring_implemented: Literal[False] = False
    quality_evaluation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.execution_replay_profile_id
            for profile in self.execution_replay_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("execution_replay_profile_ids must be unique")
        if self.execution_replay_profile_ids != derived_profile_ids:
            raise ValueError(
                "execution_replay_profile_ids must match execution_replay_profiles"
            )
        if self.profile_count != len(self.execution_replay_profiles):
            raise ValueError("profile_count must match execution_replay_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.execution_replay_kinds != tuple(
            profile.execution_replay_kind for profile in self.execution_replay_profiles
        ):
            raise ValueError(
                "execution_replay_kinds must match execution_replay_profiles"
            )

        known_routes = set(self.route_names)
        known_session_replays = set(self.session_replay_profile_ids)
        known_simulation_profiles = set(self.execution_simulation_profile_ids)
        known_execution_profiles = set(self.execution_profile_ids)
        known_provider_profiles = set(self.provider_selection_profile_ids)
        known_model_profiles = set(self.model_profile_ids)
        known_cost_profiles = set(self.cost_profile_ids)
        known_quality_profiles = set(self.quality_profile_ids)
        known_comparison_profiles = set(self.comparison_profile_ids)
        known_surfaces = set(self.execution_replay_surface_refs)
        profile_sources = {
            source_registry
            for profile in self.execution_replay_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match execution replay sources")

        for profile in self.execution_replay_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("profile source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.source_session_replay_profile_ids).issubset(
                known_session_replays
            ):
                raise ValueError(
                    "source_session_replay_profile_ids must be known session replays"
                )
            if not set(profile.source_execution_simulation_profile_ids).issubset(
                known_simulation_profiles
            ):
                raise ValueError(
                    "source_execution_simulation_profile_ids must be known profiles"
                )
            if not set(profile.source_execution_profile_ids).issubset(
                known_execution_profiles
            ):
                raise ValueError(
                    "source_execution_profile_ids must be known execution profiles"
                )
            if not set(profile.source_provider_selection_profile_ids).issubset(
                known_provider_profiles
            ):
                raise ValueError(
                    "source_provider_selection_profile_ids must be known profiles"
                )
            if not set(profile.source_model_profile_ids).issubset(known_model_profiles):
                raise ValueError("source_model_profile_ids must be known profiles")
            if not set(profile.source_cost_profile_ids).issubset(known_cost_profiles):
                raise ValueError("source_cost_profile_ids must be known profiles")
            if not set(profile.source_quality_profile_ids).issubset(
                known_quality_profiles
            ):
                raise ValueError("source_quality_profile_ids must be known profiles")
            if not set(profile.source_comparison_profile_ids).issubset(
                known_comparison_profiles
            ):
                raise ValueError("source_comparison_profile_ids must be known profiles")
            if not set(profile.execution_replay_surfaces).issubset(known_surfaces):
                raise ValueError(
                    "execution_replay_surfaces must be known registry surfaces"
                )
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
        return self


def execution_replay_registry() -> ExecutionReplayRegistry:
    """Return passive V4.4 Hybrid Studio execution replay metadata."""

    return EXECUTION_REPLAY_REGISTRY


def execution_replay_profile_by_id(
    execution_replay_profile_id: str,
    registry: ExecutionReplayRegistry | None = None,
) -> ExecutionReplayProfile | None:
    """Return one execution replay profile without executing or replaying it."""

    source_registry = registry or EXECUTION_REPLAY_REGISTRY
    for profile in source_registry.execution_replay_profiles:
        if profile.execution_replay_profile_id == execution_replay_profile_id:
            return profile
    return None


def execution_replay_profiles_for_route(
    route: RouteName | str,
    registry: ExecutionReplayRegistry | None = None,
) -> tuple[ExecutionReplayProfile, ...]:
    """Return passive execution replay profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or EXECUTION_REPLAY_REGISTRY
    return tuple(
        profile
        for profile in source_registry.execution_replay_profiles
        if route_name in profile.route_applicability
    )


def execution_replay_profiles_for_session_replay(
    session_replay_profile_id: str,
    registry: ExecutionReplayRegistry | None = None,
) -> tuple[ExecutionReplayProfile, ...]:
    """Return passive execution replays for a session replay profile id."""

    source_registry = registry or EXECUTION_REPLAY_REGISTRY
    replay_id = str(session_replay_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.execution_replay_profiles
        if replay_id in profile.source_session_replay_profile_ids
    )


def execution_replay_profiles_for_execution_simulation(
    execution_simulation_profile_id: str,
    registry: ExecutionReplayRegistry | None = None,
) -> tuple[ExecutionReplayProfile, ...]:
    """Return passive execution replays for an execution simulation profile id."""

    source_registry = registry or EXECUTION_REPLAY_REGISTRY
    simulation_id = str(execution_simulation_profile_id).strip()
    return tuple(
        profile
        for profile in source_registry.execution_replay_profiles
        if simulation_id in profile.source_execution_simulation_profile_ids
    )


def _execution_replay_profile(
    *,
    execution_replay_profile_id: str,
    profile_name: str,
    execution_replay_kind: ExecutionReplayKind,
    source_session_replay_profile_ids: tuple[str, ...],
    source_execution_simulation_profile_ids: tuple[str, ...],
    source_execution_profile_ids: tuple[str, ...],
    source_provider_selection_profile_ids: tuple[str, ...],
    source_model_profile_ids: tuple[str, ...],
    source_cost_profile_ids: tuple[str, ...],
    source_quality_profile_ids: tuple[str, ...],
    source_comparison_profile_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    execution_replay_surfaces: tuple[str, ...],
    replay_context_fields: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> ExecutionReplayProfile:
    return ExecutionReplayProfile(
        execution_replay_profile_id=execution_replay_profile_id,
        profile_name=profile_name,
        execution_replay_kind=execution_replay_kind,
        source_session_replay_profile_ids=source_session_replay_profile_ids,
        source_execution_simulation_profile_ids=(
            source_execution_simulation_profile_ids
        ),
        source_execution_profile_ids=source_execution_profile_ids,
        source_provider_selection_profile_ids=source_provider_selection_profile_ids,
        source_model_profile_ids=source_model_profile_ids,
        source_cost_profile_ids=source_cost_profile_ids,
        source_quality_profile_ids=source_quality_profile_ids,
        source_comparison_profile_ids=source_comparison_profile_ids,
        route_applicability=route_applicability,
        execution_replay_surfaces=execution_replay_surfaces,
        replay_context_fields=replay_context_fields,
        advisory_outputs=advisory_outputs,
        source_registries=_EXECUTION_REPLAY_SOURCE_REGISTRIES,
        observability_surfaces=_EXECUTION_REPLAY_OBSERVABILITY_SURFACES,
    )


EXECUTION_REPLAY_SURFACES = (
    "execution_replay_panel",
    "execution_trace_timeline",
    "simulation_replay_panel",
    "provider_selection_replay_panel",
    "cost_quality_replay_panel",
    "replay_boundary_panel",
)

EXECUTION_REPLAY_PROFILES = (
    _execution_replay_profile(
        execution_replay_profile_id="route_execution_replay_profile",
        profile_name="Route Execution Replay Profile",
        execution_replay_kind="route_execution_replay",
        source_session_replay_profile_ids=(
            "session_overview_replay_profile",
            "conversation_timeline_replay_profile",
        ),
        source_execution_simulation_profile_ids=(
            "route_preview_simulation_profile",
            "provider_selection_simulation_profile",
        ),
        source_execution_profile_ids=(
            "operator_selected_context_profile",
            "local_first_context_profile",
        ),
        source_provider_selection_profile_ids=(
            "current_config_provider_visibility_profile",
            "local_candidate_provider_visibility_profile",
        ),
        source_model_profile_ids=("fast_iteration_model_profile",),
        source_cost_profile_ids=("planning_iteration_cost_profile",),
        source_quality_profile_ids=("planning_quality_profile",),
        source_comparison_profile_ids=("generation_route_comparison_profile",),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.PREVIEW,
        ),
        execution_replay_surfaces=(
            "execution_replay_panel",
            "execution_trace_timeline",
            "simulation_replay_panel",
            "replay_boundary_panel",
        ),
        replay_context_fields=(
            "route_simulation_metadata",
            "operator_selection_metadata",
            "fast_iteration_model_refs",
            "planning_cost_quality_refs",
        ),
        advisory_outputs=(
            "route_execution_replay_context",
            "manual_route_replay_review_hint",
            "no_execution_replay_notice",
        ),
    ),
    _execution_replay_profile(
        execution_replay_profile_id="provider_selection_execution_replay_profile",
        profile_name="Provider Selection Execution Replay Profile",
        execution_replay_kind="provider_selection_replay",
        source_session_replay_profile_ids=(
            "session_overview_replay_profile",
            "snapshot_transition_replay_profile",
        ),
        source_execution_simulation_profile_ids=(
            "provider_selection_simulation_profile",
            "local_cloud_comparison_simulation_profile",
        ),
        source_execution_profile_ids=(
            "local_first_context_profile",
            "cloud_first_context_profile",
            "operator_selected_context_profile",
        ),
        source_provider_selection_profile_ids=tuple(
            PROVIDER_SELECTION_REGISTRY.provider_selection_profile_ids
        ),
        source_model_profile_ids=(
            "fast_iteration_model_profile",
            "creative_reasoning_model_profile",
        ),
        source_cost_profile_ids=(
            "planning_iteration_cost_profile",
            "creative_reasoning_cost_profile",
        ),
        source_quality_profile_ids=(
            "planning_quality_profile",
            "creative_quality_profile",
        ),
        source_comparison_profile_ids=(
            "generation_route_comparison_profile",
            "creative_reasoning_comparison_profile",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.DESIGN,
        ),
        execution_replay_surfaces=(
            "execution_replay_panel",
            "provider_selection_replay_panel",
            "simulation_replay_panel",
            "replay_boundary_panel",
        ),
        replay_context_fields=(
            "provider_visibility_metadata",
            "operator_override_metadata",
            "model_profile_refs",
            "provider_selection_boundary_refs",
        ),
        advisory_outputs=(
            "provider_selection_execution_replay_context",
            "manual_provider_replay_review_hint",
            "no_provider_selection_notice",
        ),
    ),
    _execution_replay_profile(
        execution_replay_profile_id="local_cloud_execution_replay_profile",
        profile_name="Local Cloud Execution Replay Profile",
        execution_replay_kind="local_cloud_execution_replay",
        source_session_replay_profile_ids=("snapshot_transition_replay_profile",),
        source_execution_simulation_profile_ids=(
            "local_cloud_comparison_simulation_profile",
            "provider_selection_simulation_profile",
        ),
        source_execution_profile_ids=(
            "local_first_context_profile",
            "cloud_first_context_profile",
            "side_by_side_comparison_profile",
        ),
        source_provider_selection_profile_ids=(
            "local_candidate_provider_visibility_profile",
            "cloud_candidate_provider_visibility_profile",
            "operator_override_provider_visibility_profile",
        ),
        source_model_profile_ids=(
            "creative_reasoning_model_profile",
            "code_assistance_model_profile",
        ),
        source_cost_profile_ids=(
            "creative_reasoning_cost_profile",
            "curation_refinement_cost_profile",
        ),
        source_quality_profile_ids=(
            "creative_quality_profile",
            "refinement_quality_profile",
        ),
        source_comparison_profile_ids=(
            "creative_reasoning_comparison_profile",
            "code_review_comparison_profile",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        execution_replay_surfaces=(
            "execution_replay_panel",
            "execution_trace_timeline",
            "simulation_replay_panel",
            "cost_quality_replay_panel",
        ),
        replay_context_fields=(
            "local_cloud_comparison_refs",
            "side_by_side_execution_metadata",
            "provider_boundary_metadata",
            "cost_quality_context_refs",
        ),
        advisory_outputs=(
            "local_cloud_execution_replay_context",
            "manual_local_cloud_replay_hint",
            "no_parallel_provider_execution_notice",
        ),
    ),
    _execution_replay_profile(
        execution_replay_profile_id="quality_review_execution_replay_profile",
        profile_name="Quality Review Execution Replay Profile",
        execution_replay_kind="quality_review_execution_replay",
        source_session_replay_profile_ids=(
            "conversation_timeline_replay_profile",
            "snapshot_transition_replay_profile",
            "review_decision_replay_profile",
        ),
        source_execution_simulation_profile_ids=(
            "hitl_review_simulation_profile",
            "local_cloud_comparison_simulation_profile",
        ),
        source_execution_profile_ids=(
            "side_by_side_comparison_profile",
            "operator_selected_context_profile",
        ),
        source_provider_selection_profile_ids=(
            "cloud_candidate_provider_visibility_profile",
            "operator_override_provider_visibility_profile",
            "current_config_provider_visibility_profile",
        ),
        source_model_profile_ids=(
            "code_assistance_model_profile",
            "evaluation_review_model_profile",
        ),
        source_cost_profile_ids=(
            "curation_refinement_cost_profile",
            "final_review_cost_profile",
        ),
        source_quality_profile_ids=(
            "refinement_quality_profile",
            "final_review_quality_profile",
        ),
        source_comparison_profile_ids=(
            "code_review_comparison_profile",
            "evaluation_review_comparison_profile",
        ),
        route_applicability=(
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        execution_replay_surfaces=(
            "execution_replay_panel",
            "execution_trace_timeline",
            "simulation_replay_panel",
            "cost_quality_replay_panel",
            "replay_boundary_panel",
        ),
        replay_context_fields=(
            "hitl_review_simulation_refs",
            "quality_profile_refs",
            "final_review_cost_refs",
            "manual_review_decision_refs",
        ),
        advisory_outputs=(
            "quality_review_execution_replay_context",
            "manual_quality_replay_review_hint",
            "no_quality_evaluation_notice",
        ),
    ),
)

EXECUTION_REPLAY_REGISTRY = ExecutionReplayRegistry(
    execution_replay_profiles=EXECUTION_REPLAY_PROFILES,
    execution_replay_profile_ids=tuple(
        profile.execution_replay_profile_id for profile in EXECUTION_REPLAY_PROFILES
    ),
    execution_replay_kinds=tuple(
        profile.execution_replay_kind for profile in EXECUTION_REPLAY_PROFILES
    ),
    session_replay_profile_ids=tuple(
        SESSION_REPLAY_REGISTRY.session_replay_profile_ids
    ),
    execution_simulation_profile_ids=tuple(
        EXECUTION_SIMULATOR_REGISTRY.execution_simulation_profile_ids
    ),
    execution_profile_ids=tuple(HYBRID_EXECUTION_REGISTRY.execution_profile_ids),
    provider_selection_profile_ids=tuple(
        PROVIDER_SELECTION_REGISTRY.provider_selection_profile_ids
    ),
    model_profile_ids=tuple(MODEL_PROFILE_REGISTRY.model_profile_ids),
    cost_profile_ids=tuple(COST_PROFILE_REGISTRY.cost_profile_ids),
    quality_profile_ids=tuple(QUALITY_PROFILE_REGISTRY.quality_profile_ids),
    comparison_profile_ids=tuple(
        LOCAL_CLOUD_COMPARISON_REGISTRY.comparison_profile_ids
    ),
    execution_replay_surface_refs=EXECUTION_REPLAY_SURFACES,
    route_names=tuple(RouteName),
    profile_count=len(EXECUTION_REPLAY_PROFILES),
    source_registries=_EXECUTION_REPLAY_SOURCE_REGISTRIES,
    observability_surfaces=_EXECUTION_REPLAY_OBSERVABILITY_SURFACES,
)

HybridStudioIntegrationKind = Literal[
    "model_execution_integration",
    "agent_workspace_integration",
    "snapshot_replay_integration",
    "operator_review_integration",
]

HYBRID_STUDIO_INTEGRATION_PROFILE_SERIALIZATION_VERSION = (
    "hybrid_studio_integration_profile.v1"
)
HYBRID_STUDIO_INTEGRATION_REGISTRY_SERIALIZATION_VERSION = (
    "hybrid_studio_integration_registry.v1"
)
HYBRID_STUDIO_INTEGRATION_REGISTRY_AUTHORITY_BOUNDARY = (
    "Hybrid Studio Integration metadata describes a passive V4.4 inventory of "
    "Studio-visible local model, cloud model, hybrid execution, Auto Mode, "
    "Studio Mode, HITL, provider selection, simulator, model, cost, quality, "
    "comparison, agent workspace, conversation, snapshot, session replay, and "
    "execution replay registries only; it does not activate Studio runtime, "
    "select providers or models, execute providers, invoke agents, control "
    "workflows, request human input, trigger retries, mutate storage, write "
    "replay storage, or modify generated output."
)

_HYBRID_STUDIO_INTEGRATION_SOURCE_REGISTRIES = (
    "local_model_registry",
    "cloud_model_registry",
    "hybrid_execution_registry",
    "auto_mode_registry",
    "studio_mode_registry",
    "hitl_decision_registry",
    "provider_selection_registry",
    "execution_simulator_registry",
    "model_profile_registry",
    "cost_profile_registry",
    "quality_profile_registry",
    "local_cloud_comparison_registry",
    "agent_workspace_registry",
    "agent_conversation_view_registry",
    "workspace_snapshot_registry",
    "session_replay_registry",
    "execution_replay_registry",
)

_HYBRID_STUDIO_INTEGRATION_PROFILE_GROUPS = (
    "local_model_surfaces",
    "cloud_model_surfaces",
    "hybrid_execution_profiles",
    "auto_mode_profiles",
    "studio_mode_profiles",
    "hitl_decision_profiles",
    "provider_selection_profiles",
    "execution_simulation_profiles",
    "model_profiles",
    "cost_profiles",
    "quality_profiles",
    "comparison_profiles",
    "agent_workspace_profiles",
    "conversation_view_profiles",
    "workspace_snapshot_profiles",
    "session_replay_profiles",
    "execution_replay_profiles",
)

_HYBRID_STUDIO_INTEGRATION_SURFACES = (
    "hybrid_studio_shell",
    "model_execution_surface",
    "agent_workspace_surface",
    "snapshot_replay_surface",
    "operator_review_surface",
    "integration_boundary_panel",
)

_HYBRID_STUDIO_INTEGRATION_OBSERVABILITY_SURFACES = (
    "integration_profile_id",
    "integration_kind",
    "source_registry_names",
    "linked_profile_group_refs",
    "route_applicability",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_HYBRID_STUDIO_INTEGRATION_BLOCKED_RUNTIME_BEHAVIORS = (
    "studio_runtime_activation",
    "runtime_selection",
    "provider_or_model_routing",
    "provider_execution",
    "agent_invocation",
    "workflow_control",
    "human_input_request",
    "retry_or_refinement_triggering",
    "storage_mutation",
    "persistent_replay_storage",
    "generated_output_modification",
)


class HybridStudioIntegrationProfile(BaseModel):
    """Inspectable passive integration profile for V4.4 Hybrid Studio metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    integration_profile_id: str = Field(min_length=1, max_length=140)
    profile_name: str = Field(min_length=1, max_length=160)
    integration_kind: HybridStudioIntegrationKind
    source_registry_names: tuple[str, ...] = Field(min_length=1, max_length=17)
    linked_profile_group_refs: tuple[str, ...] = Field(min_length=1, max_length=17)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    integration_surfaces: tuple[str, ...] = Field(min_length=1, max_length=6)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_registries: tuple[str, ...] = Field(min_length=17, max_length=17)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=HYBRID_STUDIO_INTEGRATION_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_HYBRID_STUDIO_INTEGRATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    studio_runtime_activation_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    serialization_version: Literal["hybrid_studio_integration_profile.v1"] = (
        HYBRID_STUDIO_INTEGRATION_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class HybridStudioIntegrationRegistry(BaseModel):
    """Stable passive registry integrating V4.4 Hybrid Studio metadata surfaces."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["hybrid_studio_integration_registry"] = (
        "hybrid_studio_integration_registry"
    )
    serialization_version: Literal["hybrid_studio_integration_registry.v1"] = (
        HYBRID_STUDIO_INTEGRATION_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=HYBRID_STUDIO_INTEGRATION_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    integration_profiles: tuple[HybridStudioIntegrationProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    integration_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    integration_kinds: tuple[HybridStudioIntegrationKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    local_surface_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    cloud_surface_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    execution_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    auto_mode_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    studio_mode_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    hitl_decision_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    provider_selection_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    execution_simulation_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    model_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    cost_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    quality_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    comparison_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    workspace_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    conversation_view_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    workspace_snapshot_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    session_replay_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    execution_replay_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    profile_group_refs: tuple[str, ...] = Field(min_length=17, max_length=17)
    integration_surface_refs: tuple[str, ...] = Field(min_length=6, max_length=6)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=17, max_length=17)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_HYBRID_STUDIO_INTEGRATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    studio_runtime_activation_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.integration_profile_id for profile in self.integration_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("integration_profile_ids must be unique")
        if self.integration_profile_ids != derived_profile_ids:
            raise ValueError("integration_profile_ids must match integration_profiles")
        if self.profile_count != len(self.integration_profiles):
            raise ValueError("profile_count must match integration_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.integration_kinds != tuple(
            profile.integration_kind for profile in self.integration_profiles
        ):
            raise ValueError("integration_kinds must match integration_profiles")

        known_routes = set(self.route_names)
        known_sources = set(self.source_registries)
        known_profile_groups = set(self.profile_group_refs)
        known_surfaces = set(self.integration_surface_refs)
        for profile in self.integration_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("profile source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.source_registry_names).issubset(known_sources):
                raise ValueError("source_registry_names must be known registries")
            if not set(profile.linked_profile_group_refs).issubset(
                known_profile_groups
            ):
                raise ValueError("linked_profile_group_refs must be known groups")
            if not set(profile.integration_surfaces).issubset(known_surfaces):
                raise ValueError("integration_surfaces must be known registry surfaces")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must be known route names")
        return self


def hybrid_studio_integration_registry() -> HybridStudioIntegrationRegistry:
    """Return passive V4.4 Hybrid Studio integration metadata."""

    return HYBRID_STUDIO_INTEGRATION_REGISTRY


def hybrid_studio_integration_profile_by_id(
    integration_profile_id: str,
    registry: HybridStudioIntegrationRegistry | None = None,
) -> HybridStudioIntegrationProfile | None:
    """Return one integration profile without activating Studio behavior."""

    source_registry = registry or HYBRID_STUDIO_INTEGRATION_REGISTRY
    for profile in source_registry.integration_profiles:
        if profile.integration_profile_id == integration_profile_id:
            return profile
    return None


def hybrid_studio_integration_profiles_for_route(
    route: RouteName | str,
    registry: HybridStudioIntegrationRegistry | None = None,
) -> tuple[HybridStudioIntegrationProfile, ...]:
    """Return passive integration profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or HYBRID_STUDIO_INTEGRATION_REGISTRY
    return tuple(
        profile
        for profile in source_registry.integration_profiles
        if route_name in profile.route_applicability
    )


def hybrid_studio_integration_profiles_for_source_registry(
    source_registry_name: str,
    registry: HybridStudioIntegrationRegistry | None = None,
) -> tuple[HybridStudioIntegrationProfile, ...]:
    """Return passive integration profiles referencing one source registry."""

    source_registry = registry or HYBRID_STUDIO_INTEGRATION_REGISTRY
    source_name = str(source_registry_name).strip()
    return tuple(
        profile
        for profile in source_registry.integration_profiles
        if source_name in profile.source_registry_names
    )


def _hybrid_studio_integration_profile(
    *,
    integration_profile_id: str,
    profile_name: str,
    integration_kind: HybridStudioIntegrationKind,
    source_registry_names: tuple[str, ...],
    linked_profile_group_refs: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    integration_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> HybridStudioIntegrationProfile:
    return HybridStudioIntegrationProfile(
        integration_profile_id=integration_profile_id,
        profile_name=profile_name,
        integration_kind=integration_kind,
        source_registry_names=source_registry_names,
        linked_profile_group_refs=linked_profile_group_refs,
        route_applicability=route_applicability,
        integration_surfaces=integration_surfaces,
        advisory_outputs=advisory_outputs,
        source_registries=_HYBRID_STUDIO_INTEGRATION_SOURCE_REGISTRIES,
        observability_surfaces=_HYBRID_STUDIO_INTEGRATION_OBSERVABILITY_SURFACES,
    )


HYBRID_STUDIO_INTEGRATION_PROFILES = (
    _hybrid_studio_integration_profile(
        integration_profile_id="model_execution_studio_integration",
        profile_name="Model Execution Studio Integration",
        integration_kind="model_execution_integration",
        source_registry_names=(
            "local_model_registry",
            "cloud_model_registry",
            "hybrid_execution_registry",
            "provider_selection_registry",
            "execution_simulator_registry",
            "model_profile_registry",
            "cost_profile_registry",
            "quality_profile_registry",
            "local_cloud_comparison_registry",
            "execution_replay_registry",
        ),
        linked_profile_group_refs=(
            "local_model_surfaces",
            "cloud_model_surfaces",
            "hybrid_execution_profiles",
            "provider_selection_profiles",
            "execution_simulation_profiles",
            "model_profiles",
            "cost_profiles",
            "quality_profiles",
            "comparison_profiles",
            "execution_replay_profiles",
        ),
        route_applicability=tuple(RouteName),
        integration_surfaces=(
            "hybrid_studio_shell",
            "model_execution_surface",
            "integration_boundary_panel",
        ),
        advisory_outputs=(
            "model_execution_integration_inventory",
            "manual_model_execution_review_hint",
            "no_provider_execution_notice",
        ),
    ),
    _hybrid_studio_integration_profile(
        integration_profile_id="agent_workspace_studio_integration",
        profile_name="Agent Workspace Studio Integration",
        integration_kind="agent_workspace_integration",
        source_registry_names=(
            "agent_workspace_registry",
            "agent_conversation_view_registry",
            "workspace_snapshot_registry",
            "session_replay_registry",
            "hitl_decision_registry",
            "quality_profile_registry",
            "studio_mode_registry",
        ),
        linked_profile_group_refs=(
            "agent_workspace_profiles",
            "conversation_view_profiles",
            "workspace_snapshot_profiles",
            "session_replay_profiles",
            "hitl_decision_profiles",
            "quality_profiles",
            "studio_mode_profiles",
        ),
        route_applicability=(
            RouteName.GENERATE,
            RouteName.EXPLAIN,
            RouteName.DESIGN,
            RouteName.REVIEW,
        ),
        integration_surfaces=(
            "hybrid_studio_shell",
            "agent_workspace_surface",
            "operator_review_surface",
            "integration_boundary_panel",
        ),
        advisory_outputs=(
            "agent_workspace_integration_inventory",
            "manual_agent_workspace_review_hint",
            "no_agent_invocation_notice",
        ),
    ),
    _hybrid_studio_integration_profile(
        integration_profile_id="snapshot_replay_studio_integration",
        profile_name="Snapshot Replay Studio Integration",
        integration_kind="snapshot_replay_integration",
        source_registry_names=(
            "workspace_snapshot_registry",
            "session_replay_registry",
            "execution_replay_registry",
            "execution_simulator_registry",
            "local_cloud_comparison_registry",
            "quality_profile_registry",
        ),
        linked_profile_group_refs=(
            "workspace_snapshot_profiles",
            "session_replay_profiles",
            "execution_replay_profiles",
            "execution_simulation_profiles",
            "comparison_profiles",
            "quality_profiles",
        ),
        route_applicability=(
            RouteName.DEBUG,
            RouteName.DESIGN,
            RouteName.REVIEW,
            RouteName.PREVIEW,
        ),
        integration_surfaces=(
            "hybrid_studio_shell",
            "snapshot_replay_surface",
            "model_execution_surface",
            "integration_boundary_panel",
        ),
        advisory_outputs=(
            "snapshot_replay_integration_inventory",
            "manual_replay_review_hint",
            "no_replay_execution_notice",
        ),
    ),
    _hybrid_studio_integration_profile(
        integration_profile_id="operator_review_studio_integration",
        profile_name="Operator Review Studio Integration",
        integration_kind="operator_review_integration",
        source_registry_names=(
            "studio_mode_registry",
            "auto_mode_registry",
            "hitl_decision_registry",
            "provider_selection_registry",
            "workspace_snapshot_registry",
            "session_replay_registry",
            "execution_replay_registry",
        ),
        linked_profile_group_refs=(
            "studio_mode_profiles",
            "auto_mode_profiles",
            "hitl_decision_profiles",
            "provider_selection_profiles",
            "workspace_snapshot_profiles",
            "session_replay_profiles",
            "execution_replay_profiles",
        ),
        route_applicability=tuple(RouteName),
        integration_surfaces=(
            "hybrid_studio_shell",
            "operator_review_surface",
            "snapshot_replay_surface",
            "integration_boundary_panel",
        ),
        advisory_outputs=(
            "operator_review_integration_inventory",
            "manual_operator_review_hint",
            "no_human_input_request_notice",
        ),
    ),
)

HYBRID_STUDIO_INTEGRATION_REGISTRY = HybridStudioIntegrationRegistry(
    integration_profiles=HYBRID_STUDIO_INTEGRATION_PROFILES,
    integration_profile_ids=tuple(
        profile.integration_profile_id for profile in HYBRID_STUDIO_INTEGRATION_PROFILES
    ),
    integration_kinds=tuple(
        profile.integration_kind for profile in HYBRID_STUDIO_INTEGRATION_PROFILES
    ),
    local_surface_ids=tuple(LOCAL_MODEL_REGISTRY.surface_ids),
    cloud_surface_ids=tuple(CLOUD_MODEL_REGISTRY.surface_ids),
    execution_profile_ids=tuple(HYBRID_EXECUTION_REGISTRY.execution_profile_ids),
    auto_mode_profile_ids=tuple(AUTO_MODE_REGISTRY.auto_mode_profile_ids),
    studio_mode_profile_ids=tuple(STUDIO_MODE_REGISTRY.studio_mode_profile_ids),
    hitl_decision_profile_ids=tuple(HITL_DECISION_REGISTRY.hitl_decision_profile_ids),
    provider_selection_profile_ids=tuple(
        PROVIDER_SELECTION_REGISTRY.provider_selection_profile_ids
    ),
    execution_simulation_profile_ids=tuple(
        EXECUTION_SIMULATOR_REGISTRY.execution_simulation_profile_ids
    ),
    model_profile_ids=tuple(MODEL_PROFILE_REGISTRY.model_profile_ids),
    cost_profile_ids=tuple(COST_PROFILE_REGISTRY.cost_profile_ids),
    quality_profile_ids=tuple(QUALITY_PROFILE_REGISTRY.quality_profile_ids),
    comparison_profile_ids=tuple(
        LOCAL_CLOUD_COMPARISON_REGISTRY.comparison_profile_ids
    ),
    workspace_profile_ids=tuple(AGENT_WORKSPACE_REGISTRY.workspace_profile_ids),
    conversation_view_profile_ids=tuple(
        AGENT_CONVERSATION_VIEW_REGISTRY.conversation_view_profile_ids
    ),
    workspace_snapshot_profile_ids=tuple(
        WORKSPACE_SNAPSHOT_REGISTRY.workspace_snapshot_profile_ids
    ),
    session_replay_profile_ids=tuple(
        SESSION_REPLAY_REGISTRY.session_replay_profile_ids
    ),
    execution_replay_profile_ids=tuple(
        EXECUTION_REPLAY_REGISTRY.execution_replay_profile_ids
    ),
    profile_group_refs=_HYBRID_STUDIO_INTEGRATION_PROFILE_GROUPS,
    integration_surface_refs=_HYBRID_STUDIO_INTEGRATION_SURFACES,
    route_names=tuple(RouteName),
    profile_count=len(HYBRID_STUDIO_INTEGRATION_PROFILES),
    source_registries=_HYBRID_STUDIO_INTEGRATION_SOURCE_REGISTRIES,
    observability_surfaces=_HYBRID_STUDIO_INTEGRATION_OBSERVABILITY_SURFACES,
)
