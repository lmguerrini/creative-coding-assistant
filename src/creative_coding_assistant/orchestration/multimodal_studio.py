"""Passive V4.5 Multimodal Studio metadata surfaces."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.preview import PreviewTarget

LivePreviewProfileKind = Literal[
    "browser_sandbox_preview",
    "media_asset_preview",
    "structured_panel_preview",
    "runtime_status_preview",
]

LIVE_PREVIEW_PROFILE_SERIALIZATION_VERSION = "multimodal_live_preview_profile.v1"
LIVE_PREVIEW_REGISTRY_SERIALIZATION_VERSION = "multimodal_live_preview_registry.v1"
LIVE_PREVIEW_AUTHORITY_BOUNDARY = (
    "Live Preview metadata describes passive V4.5 Multimodal Studio surfaces "
    "for inspection only; it does not execute rendering, change browser canvas "
    "runtime behavior, route providers or models, call external providers, "
    "trigger retries, mutate generated outputs, open networking, persist "
    "collaboration state, or activate Studio runtime behavior."
)

_LIVE_PREVIEW_SOURCE_REGISTRIES = (
    "preview_contracts",
    "workflow_artifact_preview_preparation",
    "nextjs_preview_targets",
    "nextjs_preview_renderers",
    "nextjs_preview_runtime_adapters",
    "nextjs_preview_sandbox_runtime",
)

_LIVE_PREVIEW_SOURCE_REFERENCES = (
    "preview.contracts.PreviewTarget",
    "preview.contracts.PreviewRequest",
    "preview.contracts.PreviewResult",
    "preview.contracts.PreviewStatus",
    "orchestration.artifacts.prepare_workflow_preview_results",
    "clients.nextjs.preview_targets.derivePreviewTargetIdFromArtifact",
    "clients.nextjs.preview_renderers.creativePreviewRendererRegistry",
    "clients.nextjs.preview_runtime_adapters.PreviewRuntimeStatus",
    "clients.nextjs.preview_sandbox_runtime.mountPreviewSandboxRuntime",
)

_LIVE_PREVIEW_SURFACES = (
    "live_preview_shelf",
    "live_preview_target_panel",
    "preview_renderer_match_panel",
    "preview_source_metadata_panel",
    "preview_status_panel",
    "preview_boundary_panel",
)

_LIVE_PREVIEW_OBSERVABILITY_SURFACES = (
    "profile_id",
    "surface_kind",
    "preview_targets",
    "renderer_contract_refs",
    "source_reference_ids",
    "blocked_runtime_behaviors",
    "authority_boundary",
)

_LIVE_PREVIEW_BLOCKED_RUNTIME_BEHAVIORS = (
    "rendering_execution",
    "browser_canvas_runtime_change",
    "provider_or_model_routing",
    "external_provider_calling",
    "retry_triggering",
    "generated_output_mutation",
    "networking",
    "persistent_collaboration_storage",
    "active_studio_runtime_behavior",
)


class LivePreviewProfile(BaseModel):
    """Inspectable metadata for one passive Live Preview surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1, max_length=120)
    profile_name: str = Field(min_length=1, max_length=140)
    surface_kind: LivePreviewProfileKind
    preview_targets: tuple[PreviewTarget, ...] = Field(min_length=1, max_length=6)
    renderer_contract_refs: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_reference_ids: tuple[str, ...] = Field(min_length=1, max_length=9)
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    live_preview_surfaces: tuple[str, ...] = Field(min_length=1, max_length=6)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    authority_boundary: str = Field(
        default=LIVE_PREVIEW_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_LIVE_PREVIEW_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    rendering_execution_implemented: Literal[False] = False
    browser_canvas_runtime_change_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    external_provider_calls_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_storage_implemented: Literal[False] = False
    active_studio_behavior_implemented: Literal[False] = False
    serialization_version: Literal["multimodal_live_preview_profile.v1"] = (
        LIVE_PREVIEW_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class MultimodalLivePreviewRegistry(BaseModel):
    """Stable passive registry for V4.5 Multimodal Studio Live Preview."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["multimodal_live_preview_registry"] = (
        "multimodal_live_preview_registry"
    )
    serialization_version: Literal["multimodal_live_preview_registry.v1"] = (
        LIVE_PREVIEW_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=LIVE_PREVIEW_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    live_preview_profiles: tuple[LivePreviewProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    surface_kinds: tuple[LivePreviewProfileKind, ...] = Field(
        min_length=4,
        max_length=4,
    )
    preview_targets: tuple[PreviewTarget, ...] = Field(min_length=6, max_length=6)
    route_names: tuple[RouteName, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_reference_ids: tuple[str, ...] = Field(min_length=9, max_length=9)
    live_preview_surface_refs: tuple[str, ...] = Field(min_length=6, max_length=6)
    observability_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_LIVE_PREVIEW_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    rendering_execution_implemented: Literal[False] = False
    browser_canvas_runtime_change_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    external_provider_calls_implemented: Literal[False] = False
    networking_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_storage_implemented: Literal[False] = False
    active_studio_behavior_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_profile_ids = tuple(
            profile.profile_id for profile in self.live_preview_profiles
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("profile_ids must be unique")
        if self.profile_ids != derived_profile_ids:
            raise ValueError("profile_ids must match live_preview_profiles")
        if self.profile_count != len(self.live_preview_profiles):
            raise ValueError("profile_count must match live_preview_profiles")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.preview_targets != tuple(PreviewTarget):
            raise ValueError("preview_targets must match preview target enum order")

        derived_surface_kinds = _ordered_unique(
            profile.surface_kind for profile in self.live_preview_profiles
        )
        if self.surface_kinds != derived_surface_kinds:
            raise ValueError("surface_kinds must match live preview profiles")

        profile_source_references = {
            source_reference
            for profile in self.live_preview_profiles
            for source_reference in profile.source_reference_ids
        }
        if set(self.source_reference_ids) != profile_source_references:
            raise ValueError(
                "source_reference_ids must match profile source references"
            )

        known_routes = set(self.route_names)
        known_targets = set(self.preview_targets)
        known_surfaces = set(self.live_preview_surface_refs)
        known_source_references = set(self.source_reference_ids)
        for profile in self.live_preview_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("source_registries must match registry")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if not set(profile.route_applicability).issubset(known_routes):
                raise ValueError("route_applicability must use known routes")
            if not set(profile.preview_targets).issubset(known_targets):
                raise ValueError("preview_targets must use known targets")
            if not set(profile.live_preview_surfaces).issubset(known_surfaces):
                raise ValueError("live_preview_surfaces must be known surfaces")
            if not set(profile.source_reference_ids).issubset(
                known_source_references
            ):
                raise ValueError(
                    "source_reference_ids must be known registry references"
                )
        return self


def multimodal_live_preview_registry() -> MultimodalLivePreviewRegistry:
    """Return passive V4.5 Multimodal Studio Live Preview metadata."""

    return MULTIMODAL_LIVE_PREVIEW_REGISTRY


def multimodal_live_preview_profile_by_id(
    profile_id: str,
    registry: MultimodalLivePreviewRegistry | None = None,
) -> LivePreviewProfile | None:
    """Return one Live Preview profile without executing preview behavior."""

    source_registry = registry or MULTIMODAL_LIVE_PREVIEW_REGISTRY
    normalized_profile_id = str(profile_id).strip()
    for profile in source_registry.live_preview_profiles:
        if profile.profile_id == normalized_profile_id:
            return profile
    return None


def multimodal_live_preview_profiles_for_route(
    route: RouteName | str,
    registry: MultimodalLivePreviewRegistry | None = None,
) -> tuple[LivePreviewProfile, ...]:
    """Return passive Live Preview profiles applicable to a route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_registry = registry or MULTIMODAL_LIVE_PREVIEW_REGISTRY
    return tuple(
        profile
        for profile in source_registry.live_preview_profiles
        if route_name in profile.route_applicability
    )


def multimodal_live_preview_profiles_for_target(
    target: PreviewTarget | str,
    registry: MultimodalLivePreviewRegistry | None = None,
) -> tuple[LivePreviewProfile, ...]:
    """Return passive Live Preview profiles covering a preview target."""

    preview_target = (
        target if isinstance(target, PreviewTarget) else PreviewTarget(str(target))
    )
    source_registry = registry or MULTIMODAL_LIVE_PREVIEW_REGISTRY
    return tuple(
        profile
        for profile in source_registry.live_preview_profiles
        if preview_target in profile.preview_targets
    )


def multimodal_live_preview_profiles_for_source_reference(
    source_reference_id: str,
    registry: MultimodalLivePreviewRegistry | None = None,
) -> tuple[LivePreviewProfile, ...]:
    """Return passive Live Preview profiles referencing one source surface."""

    source_registry = registry or MULTIMODAL_LIVE_PREVIEW_REGISTRY
    source_reference = str(source_reference_id).strip()
    return tuple(
        profile
        for profile in source_registry.live_preview_profiles
        if source_reference in profile.source_reference_ids
    )


def _live_preview_profile(
    *,
    profile_id: str,
    profile_name: str,
    surface_kind: LivePreviewProfileKind,
    preview_targets: tuple[PreviewTarget, ...],
    renderer_contract_refs: tuple[str, ...],
    source_reference_ids: tuple[str, ...],
    route_applicability: tuple[RouteName, ...],
    live_preview_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> LivePreviewProfile:
    return LivePreviewProfile(
        profile_id=profile_id,
        profile_name=profile_name,
        surface_kind=surface_kind,
        preview_targets=preview_targets,
        renderer_contract_refs=renderer_contract_refs,
        source_reference_ids=source_reference_ids,
        route_applicability=route_applicability,
        live_preview_surfaces=live_preview_surfaces,
        advisory_outputs=advisory_outputs,
        source_registries=_LIVE_PREVIEW_SOURCE_REGISTRIES,
        observability_surfaces=_LIVE_PREVIEW_OBSERVABILITY_SURFACES,
    )


_T = TypeVar("_T")


def _ordered_unique(values: Iterable[_T]) -> tuple[_T, ...]:
    resolved: list[_T] = []
    for value in values:
        if value not in resolved:
            resolved.append(value)
    return tuple(resolved)


MULTIMODAL_LIVE_PREVIEW_PROFILES = (
    _live_preview_profile(
        profile_id="browser_sandbox_live_preview",
        profile_name="Browser Sandbox Live Preview",
        surface_kind="browser_sandbox_preview",
        preview_targets=(PreviewTarget.BROWSER_SANDBOX,),
        renderer_contract_refs=(
            "surface.p5",
            "surface.three",
            "surface.glsl",
            "surface.hydra",
            "surface.tone",
            "surface.gsap",
            "surface.svg",
            "surface.canvas",
        ),
        source_reference_ids=(
            "preview.contracts.PreviewTarget",
            "preview.contracts.PreviewRequest",
            "orchestration.artifacts.prepare_workflow_preview_results",
            "clients.nextjs.preview_renderers.creativePreviewRendererRegistry",
            "clients.nextjs.preview_sandbox_runtime.mountPreviewSandboxRuntime",
        ),
        route_applicability=tuple(RouteName),
        live_preview_surfaces=(
            "live_preview_shelf",
            "live_preview_target_panel",
            "preview_renderer_match_panel",
            "preview_source_metadata_panel",
            "preview_boundary_panel",
        ),
        advisory_outputs=(
            "browser_sandbox_preview_inventory",
            "manual_preview_target_review_hint",
            "no_rendering_execution_notice",
        ),
    ),
    _live_preview_profile(
        profile_id="media_asset_live_preview",
        profile_name="Media Asset Live Preview",
        surface_kind="media_asset_preview",
        preview_targets=(
            PreviewTarget.IMAGE_ASSET,
            PreviewTarget.AUDIO_ASSET,
            PreviewTarget.VIDEO_ASSET,
        ),
        renderer_contract_refs=(
            "image_asset_surface",
            "audio_asset_surface",
            "video_asset_surface",
        ),
        source_reference_ids=(
            "preview.contracts.PreviewTarget",
            "preview.contracts.PreviewResult",
            "clients.nextjs.preview_targets.derivePreviewTargetIdFromArtifact",
        ),
        route_applicability=tuple(RouteName),
        live_preview_surfaces=(
            "live_preview_shelf",
            "live_preview_target_panel",
            "preview_source_metadata_panel",
            "preview_boundary_panel",
        ),
        advisory_outputs=(
            "media_asset_preview_inventory",
            "manual_media_target_review_hint",
            "no_asset_mutation_notice",
        ),
    ),
    _live_preview_profile(
        profile_id="structured_panel_live_preview",
        profile_name="Structured Panel Live Preview",
        surface_kind="structured_panel_preview",
        preview_targets=(
            PreviewTarget.TEXT_PANEL,
            PreviewTarget.JSON_PANEL,
        ),
        renderer_contract_refs=(
            "text_panel_surface",
            "json_panel_surface",
        ),
        source_reference_ids=(
            "preview.contracts.PreviewTarget",
            "preview.contracts.PreviewResult",
            "clients.nextjs.preview_targets.derivePreviewTargetIdFromArtifact",
            "clients.nextjs.preview_renderers.creativePreviewRendererRegistry",
        ),
        route_applicability=tuple(RouteName),
        live_preview_surfaces=(
            "live_preview_shelf",
            "live_preview_target_panel",
            "preview_renderer_match_panel",
            "preview_source_metadata_panel",
            "preview_boundary_panel",
        ),
        advisory_outputs=(
            "structured_panel_preview_inventory",
            "manual_panel_review_hint",
            "no_generated_output_mutation_notice",
        ),
    ),
    _live_preview_profile(
        profile_id="runtime_status_live_preview",
        profile_name="Runtime Status Live Preview",
        surface_kind="runtime_status_preview",
        preview_targets=(
            PreviewTarget.BROWSER_SANDBOX,
            PreviewTarget.TEXT_PANEL,
            PreviewTarget.JSON_PANEL,
        ),
        renderer_contract_refs=(
            "preview_runtime_status",
            "preview_runtime_source",
        ),
        source_reference_ids=(
            "preview.contracts.PreviewResult",
            "preview.contracts.PreviewStatus",
            "clients.nextjs.preview_runtime_adapters.PreviewRuntimeStatus",
            "clients.nextjs.preview_sandbox_runtime.mountPreviewSandboxRuntime",
        ),
        route_applicability=tuple(RouteName),
        live_preview_surfaces=(
            "live_preview_shelf",
            "preview_status_panel",
            "preview_source_metadata_panel",
            "preview_boundary_panel",
        ),
        advisory_outputs=(
            "runtime_status_preview_inventory",
            "manual_runtime_status_review_hint",
            "no_runtime_control_notice",
        ),
    ),
)

MULTIMODAL_LIVE_PREVIEW_REGISTRY = MultimodalLivePreviewRegistry(
    live_preview_profiles=MULTIMODAL_LIVE_PREVIEW_PROFILES,
    profile_ids=tuple(
        profile.profile_id for profile in MULTIMODAL_LIVE_PREVIEW_PROFILES
    ),
    surface_kinds=tuple(
        profile.surface_kind for profile in MULTIMODAL_LIVE_PREVIEW_PROFILES
    ),
    preview_targets=tuple(PreviewTarget),
    route_names=tuple(RouteName),
    profile_count=len(MULTIMODAL_LIVE_PREVIEW_PROFILES),
    source_registries=_LIVE_PREVIEW_SOURCE_REGISTRIES,
    source_reference_ids=_LIVE_PREVIEW_SOURCE_REFERENCES,
    live_preview_surface_refs=_LIVE_PREVIEW_SURFACES,
    observability_surfaces=_LIVE_PREVIEW_OBSERVABILITY_SURFACES,
)
