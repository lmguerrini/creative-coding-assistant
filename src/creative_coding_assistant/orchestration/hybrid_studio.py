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
