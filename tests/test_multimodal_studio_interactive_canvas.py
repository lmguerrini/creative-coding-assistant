import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    MultimodalInteractiveCanvasRegistry,
    multimodal_interactive_canvas_profile_by_id,
    multimodal_interactive_canvas_profiles_for_live_preview_profile,
    multimodal_interactive_canvas_profiles_for_multi_preview_profile,
    multimodal_interactive_canvas_profiles_for_route,
    multimodal_interactive_canvas_profiles_for_surface_kind,
    multimodal_interactive_canvas_registry,
    multimodal_live_preview_registry,
    multimodal_multi_preview_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.preview import PreviewTarget

EXPECTED_PROFILE_IDS = (
    "canvas_2d_interactive_canvas",
    "webgl_interactive_canvas",
    "input_boundary_interactive_canvas",
    "runtime_status_interactive_canvas",
)
EXPECTED_PROFILE_KINDS = (
    "canvas_surface_inspection",
    "webgl_canvas_inspection",
    "input_boundary_inspection",
    "canvas_status_inspection",
)
EXPECTED_SURFACE_KINDS = (
    "canvas_2d",
    "webgl_canvas",
    "input_boundary",
    "runtime_status",
)
EXPECTED_SOURCE_REGISTRIES = (
    "multimodal_live_preview_registry",
    "multimodal_multi_preview_registry",
    "nextjs_svg_canvas_runtime",
    "nextjs_preview_runtime_adapters",
    "nextjs_preview_sandbox_runtime",
    "nextjs_preview_renderers",
)
EXPECTED_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_LIVE_PREVIEW_REGISTRY",
    "multimodal_studio.MULTIMODAL_MULTI_PREVIEW_REGISTRY",
    "clients.nextjs.svg_canvas_runtime.hasCanvasPreviewSignal",
    "clients.nextjs.svg_canvas_runtime.getCanvasRuntimeSupportIssue",
    "clients.nextjs.preview_runtime_adapters.buildPreviewRuntimeSource",
    "clients.nextjs.preview_runtime_adapters.mountPreviewRuntime",
    "clients.nextjs.preview_sandbox_runtime.buildPreviewSandboxDocument",
    "clients.nextjs.preview_renderers.surface.canvas",
)
EXPECTED_CANVAS_SURFACES = (
    "interactive_canvas_panel",
    "canvas_surface_contract_panel",
    "canvas_input_boundary_panel",
    "canvas_runtime_status_panel",
    "canvas_source_guardrail_panel",
    "canvas_fallback_panel",
    "interactive_canvas_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "profile_id",
    "profile_name",
    "canvas_profile_kind",
    "canvas_surface_kind",
    "preview_targets",
    "source_live_preview_profile_ids",
    "source_multi_preview_profile_ids",
    "canvas_signal_refs",
    "source_reference_ids",
    "route_applicability",
    "interactive_canvas_surfaces",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "rendering_execution_implemented",
    "interactive_input_binding_implemented",
    "browser_canvas_runtime_change_implemented",
    "canvas_context_mutation_implemented",
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


class MultimodalStudioInteractiveCanvasTests(unittest.TestCase):
    def test_interactive_canvas_registry_covers_expected_sources(self) -> None:
        registry = multimodal_interactive_canvas_registry()

        self.assertEqual(registry.role, "multimodal_interactive_canvas_registry")
        self.assertEqual(
            registry.serialization_version,
            "multimodal_interactive_canvas_registry.v1",
        )
        self.assertEqual(registry.profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.canvas_profile_kinds, EXPECTED_PROFILE_KINDS)
        self.assertEqual(registry.canvas_surface_kinds, EXPECTED_SURFACE_KINDS)
        self.assertEqual(registry.preview_targets, (PreviewTarget.BROWSER_SANDBOX,))
        self.assertEqual(
            registry.live_preview_profile_ids,
            multimodal_live_preview_registry().profile_ids,
        )
        self.assertEqual(
            registry.multi_preview_profile_ids,
            multimodal_multi_preview_registry().profile_ids,
        )
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertEqual(registry.source_reference_ids, EXPECTED_SOURCE_REFERENCES)
        self.assertEqual(
            registry.interactive_canvas_surface_refs,
            EXPECTED_CANVAS_SURFACES,
        )
        self.assertIn("does not execute rendering", registry.authority_boundary)
        self.assertIn("bind interactive input", registry.authority_boundary)
        self.assertIn("interactive_input_binding", registry.blocked_runtime_behaviors)
        self.assertIn("canvas_context_mutation", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.rendering_execution_implemented)
        self.assertFalse(registry.interactive_input_binding_implemented)
        self.assertFalse(registry.browser_canvas_runtime_change_implemented)
        self.assertFalse(registry.canvas_context_mutation_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.external_provider_calls_implemented)
        self.assertFalse(registry.networking_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_storage_implemented)
        self.assertFalse(registry.active_studio_behavior_implemented)

    def test_interactive_canvas_profiles_are_passive_and_source_aligned(self) -> None:
        registry = multimodal_interactive_canvas_registry()
        known_routes = set(registry.route_names)
        known_targets = set(registry.preview_targets)
        known_live_profiles = set(registry.live_preview_profile_ids)
        known_multi_profiles = set(registry.multi_preview_profile_ids)
        known_surfaces = set(registry.interactive_canvas_surface_refs)
        known_source_references = set(registry.source_reference_ids)
        covered_source_references: set[str] = set()

        for profile in registry.interactive_canvas_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "multimodal_interactive_canvas_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(set(profile.preview_targets).issubset(known_targets))
            self.assertTrue(
                set(profile.source_live_preview_profile_ids).issubset(
                    known_live_profiles
                )
            )
            self.assertTrue(
                set(profile.source_multi_preview_profile_ids).issubset(
                    known_multi_profiles
                )
            )
            self.assertTrue(
                set(profile.interactive_canvas_surfaces).issubset(known_surfaces)
            )
            self.assertTrue(
                set(profile.source_reference_ids).issubset(known_source_references)
            )
            covered_source_references.update(profile.source_reference_ids)
            self.assertIn(
                "interactive_input_binding",
                profile.blocked_runtime_behaviors,
            )
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.rendering_execution_implemented)
            self.assertFalse(profile.interactive_input_binding_implemented)
            self.assertFalse(profile.browser_canvas_runtime_change_implemented)
            self.assertFalse(profile.canvas_context_mutation_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.external_provider_calls_implemented)
            self.assertFalse(profile.networking_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_storage_implemented)
            self.assertFalse(profile.active_studio_behavior_implemented)

        self.assertEqual(covered_source_references, known_source_references)

    def test_interactive_canvas_lookup_helpers_are_stable(self) -> None:
        profile = multimodal_interactive_canvas_profile_by_id(
            "input_boundary_interactive_canvas"
        )
        missing_profile = multimodal_interactive_canvas_profile_by_id("missing_profile")
        canvas_profiles = multimodal_interactive_canvas_profiles_for_surface_kind(
            "canvas_2d"
        )
        preview_route_profiles = multimodal_interactive_canvas_profiles_for_route(
            RouteName.PREVIEW
        )
        runtime_status_profiles = (
            multimodal_interactive_canvas_profiles_for_live_preview_profile(
                "runtime_status_live_preview"
            )
        )
        split_profiles = (
            multimodal_interactive_canvas_profiles_for_multi_preview_profile(
                "split_comparison_multi_preview"
            )
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.canvas_profile_kind, "input_boundary_inspection")
        self.assertIn("getCanvasRuntimeSupportIssue", profile.canvas_signal_refs)
        self.assertIn(
            "no_interactive_input_binding_notice",
            profile.advisory_outputs,
        )
        self.assertEqual(
            tuple(item.profile_id for item in canvas_profiles),
            ("canvas_2d_interactive_canvas",),
        )
        self.assertEqual(
            tuple(item.profile_id for item in preview_route_profiles),
            EXPECTED_PROFILE_IDS,
        )
        self.assertEqual(
            tuple(item.profile_id for item in runtime_status_profiles),
            EXPECTED_PROFILE_IDS,
        )
        self.assertEqual(
            tuple(item.profile_id for item in split_profiles),
            ("input_boundary_interactive_canvas",),
        )

    def test_registry_rejects_mismatched_interactive_canvas_metadata(self) -> None:
        registry = multimodal_interactive_canvas_registry()
        first_profile = registry.interactive_canvas_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Interactive Canvas"}
        )
        unknown_multi_profile = first_profile.model_copy(
            update={"source_multi_preview_profile_ids": ("unknown_multi_preview",)}
        )

        duplicate_kwargs = _registry_kwargs(registry)
        duplicate_kwargs["interactive_canvas_profiles"] = (
            first_profile,
            duplicate_profile,
        ) + registry.interactive_canvas_profiles[2:]
        with self.assertRaisesRegex(ValueError, "profile_ids must be unique"):
            MultimodalInteractiveCanvasRegistry(**duplicate_kwargs)

        unknown_multi_kwargs = _registry_kwargs(registry)
        unknown_multi_kwargs["interactive_canvas_profiles"] = (
            unknown_multi_profile,
        ) + registry.interactive_canvas_profiles[1:]
        with self.assertRaisesRegex(
            ValueError,
            "source_multi_preview_profile_ids",
        ):
            MultimodalInteractiveCanvasRegistry(**unknown_multi_kwargs)

    def test_interactive_canvas_metadata_does_not_change_provider_factory(
        self,
    ) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        multimodal_interactive_canvas_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")


def _registry_kwargs(
    registry: MultimodalInteractiveCanvasRegistry,
) -> dict[str, object]:
    return {
        "interactive_canvas_profiles": registry.interactive_canvas_profiles,
        "profile_ids": registry.profile_ids,
        "canvas_profile_kinds": registry.canvas_profile_kinds,
        "canvas_surface_kinds": registry.canvas_surface_kinds,
        "preview_targets": registry.preview_targets,
        "live_preview_profile_ids": registry.live_preview_profile_ids,
        "multi_preview_profile_ids": registry.multi_preview_profile_ids,
        "route_names": registry.route_names,
        "profile_count": registry.profile_count,
        "source_registries": registry.source_registries,
        "source_reference_ids": registry.source_reference_ids,
        "interactive_canvas_surface_refs": registry.interactive_canvas_surface_refs,
        "observability_surfaces": registry.observability_surfaces,
    }


if __name__ == "__main__":
    unittest.main()
