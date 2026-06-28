"""Passive V4.4 Hybrid Studio metadata surfaces."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
