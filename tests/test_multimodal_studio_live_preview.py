import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    MultimodalLivePreviewRegistry,
    multimodal_live_preview_profile_by_id,
    multimodal_live_preview_profiles_for_route,
    multimodal_live_preview_profiles_for_source_reference,
    multimodal_live_preview_profiles_for_target,
    multimodal_live_preview_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.preview import PreviewTarget

EXPECTED_PROFILE_IDS = (
    "browser_sandbox_live_preview",
    "media_asset_live_preview",
    "structured_panel_live_preview",
    "runtime_status_live_preview",
)
EXPECTED_SURFACE_KINDS = (
    "browser_sandbox_preview",
    "media_asset_preview",
    "structured_panel_preview",
    "runtime_status_preview",
)
EXPECTED_SOURCE_REGISTRIES = (
    "preview_contracts",
    "workflow_artifact_preview_preparation",
    "nextjs_preview_targets",
    "nextjs_preview_renderers",
    "nextjs_preview_runtime_adapters",
    "nextjs_preview_sandbox_runtime",
)
EXPECTED_SOURCE_REFERENCES = (
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
EXPECTED_LIVE_PREVIEW_SURFACES = (
    "live_preview_shelf",
    "live_preview_target_panel",
    "preview_renderer_match_panel",
    "preview_source_metadata_panel",
    "preview_status_panel",
    "preview_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "profile_id",
    "profile_name",
    "surface_kind",
    "preview_targets",
    "renderer_contract_refs",
    "source_reference_ids",
    "route_applicability",
    "live_preview_surfaces",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "rendering_execution_implemented",
    "browser_canvas_runtime_change_implemented",
    "provider_model_routing_implemented",
    "external_provider_calls_implemented",
    "networking_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "persistent_storage_implemented",
    "active_studio_behavior_implemented",
    "serialization_version",
    "metadata_only",
}


class MultimodalStudioLivePreviewTests(unittest.TestCase):
    def test_live_preview_registry_covers_expected_sources(self) -> None:
        registry = multimodal_live_preview_registry()

        self.assertEqual(registry.role, "multimodal_live_preview_registry")
        self.assertEqual(
            registry.serialization_version,
            "multimodal_live_preview_registry.v1",
        )
        self.assertEqual(registry.profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.surface_kinds, EXPECTED_SURFACE_KINDS)
        self.assertEqual(registry.preview_targets, tuple(PreviewTarget))
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertEqual(registry.source_reference_ids, EXPECTED_SOURCE_REFERENCES)
        self.assertEqual(
            registry.live_preview_surface_refs,
            EXPECTED_LIVE_PREVIEW_SURFACES,
        )
        self.assertIn("does not execute rendering", registry.authority_boundary)
        self.assertIn("rendering_execution", registry.blocked_runtime_behaviors)
        self.assertIn(
            "browser_canvas_runtime_change",
            registry.blocked_runtime_behaviors,
        )
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.rendering_execution_implemented)
        self.assertFalse(registry.browser_canvas_runtime_change_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.external_provider_calls_implemented)
        self.assertFalse(registry.networking_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_storage_implemented)
        self.assertFalse(registry.active_studio_behavior_implemented)

    def test_live_preview_profiles_are_passive_and_source_aligned(self) -> None:
        registry = multimodal_live_preview_registry()
        known_routes = set(registry.route_names)
        known_targets = set(registry.preview_targets)
        known_surfaces = set(registry.live_preview_surface_refs)
        known_source_references = set(registry.source_reference_ids)
        covered_source_references: set[str] = set()

        for profile in registry.live_preview_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "multimodal_live_preview_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(set(profile.preview_targets).issubset(known_targets))
            self.assertTrue(set(profile.live_preview_surfaces).issubset(known_surfaces))
            self.assertTrue(
                set(profile.source_reference_ids).issubset(known_source_references)
            )
            covered_source_references.update(profile.source_reference_ids)
            self.assertIn(
                "rendering_execution",
                profile.blocked_runtime_behaviors,
            )
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.rendering_execution_implemented)
            self.assertFalse(profile.browser_canvas_runtime_change_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.external_provider_calls_implemented)
            self.assertFalse(profile.networking_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_storage_implemented)
            self.assertFalse(profile.active_studio_behavior_implemented)

        self.assertEqual(covered_source_references, known_source_references)

    def test_live_preview_lookup_helpers_are_stable(self) -> None:
        profile = multimodal_live_preview_profile_by_id(
            "browser_sandbox_live_preview"
        )
        missing_profile = multimodal_live_preview_profile_by_id("missing_profile")
        browser_profiles = multimodal_live_preview_profiles_for_target(
            PreviewTarget.BROWSER_SANDBOX
        )
        preview_route_profiles = multimodal_live_preview_profiles_for_route(
            RouteName.PREVIEW
        )
        runtime_status_profiles = (
            multimodal_live_preview_profiles_for_source_reference(
                "clients.nextjs.preview_runtime_adapters.PreviewRuntimeStatus"
            )
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.surface_kind, "browser_sandbox_preview")
        self.assertIn("surface.p5", profile.renderer_contract_refs)
        self.assertIn(
            "no_rendering_execution_notice",
            profile.advisory_outputs,
        )
        self.assertEqual(
            tuple(item.profile_id for item in browser_profiles),
            (
                "browser_sandbox_live_preview",
                "runtime_status_live_preview",
            ),
        )
        self.assertEqual(
            tuple(item.profile_id for item in preview_route_profiles),
            EXPECTED_PROFILE_IDS,
        )
        self.assertEqual(
            tuple(item.profile_id for item in runtime_status_profiles),
            ("runtime_status_live_preview",),
        )

    def test_registry_rejects_mismatched_live_preview_metadata(self) -> None:
        registry = multimodal_live_preview_registry()
        first_profile = registry.live_preview_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Live Preview"}
        )
        unknown_source_profile = first_profile.model_copy(
            update={"source_reference_ids": ("unknown.preview.source",)}
        )

        duplicate_kwargs = _registry_kwargs(registry)
        duplicate_kwargs["live_preview_profiles"] = (
            first_profile,
            duplicate_profile,
        ) + registry.live_preview_profiles[2:]
        with self.assertRaisesRegex(ValueError, "profile_ids must be unique"):
            MultimodalLivePreviewRegistry(**duplicate_kwargs)

        unknown_source_kwargs = _registry_kwargs(registry)
        unknown_source_kwargs["live_preview_profiles"] = (
            unknown_source_profile,
        ) + registry.live_preview_profiles[1:]
        with self.assertRaisesRegex(ValueError, "source_reference_ids"):
            MultimodalLivePreviewRegistry(**unknown_source_kwargs)

    def test_live_preview_metadata_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        multimodal_live_preview_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")


def _registry_kwargs(
    registry: MultimodalLivePreviewRegistry,
) -> dict[str, object]:
    return {
        "live_preview_profiles": registry.live_preview_profiles,
        "profile_ids": registry.profile_ids,
        "surface_kinds": registry.surface_kinds,
        "preview_targets": registry.preview_targets,
        "route_names": registry.route_names,
        "profile_count": registry.profile_count,
        "source_registries": registry.source_registries,
        "source_reference_ids": registry.source_reference_ids,
        "live_preview_surface_refs": registry.live_preview_surface_refs,
        "observability_surfaces": registry.observability_surfaces,
    }


if __name__ == "__main__":
    unittest.main()
