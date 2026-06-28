"""Passive V4.4 Hybrid Studio metadata surfaces."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.hybrid_agentic_workflow import (
    COST_THRESHOLD_ROUTING_REGISTRY,
)
from creative_coding_assistant.orchestration.routing import RouteName

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
