import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    MultimodalMultiPreviewRegistry,
    multimodal_live_preview_registry,
    multimodal_multi_preview_profile_by_id,
    multimodal_multi_preview_profiles_for_layout,
    multimodal_multi_preview_profiles_for_live_preview_profile,
    multimodal_multi_preview_profiles_for_route,
    multimodal_multi_preview_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "candidate_grid_multi_preview",
    "split_comparison_multi_preview",
    "recommended_candidate_multi_preview",
    "fallback_multi_preview",
)
EXPECTED_PREVIEW_KINDS = (
    "candidate_grid_preview",
    "split_comparison_preview",
    "recommended_candidate_preview",
    "comparison_fallback_preview",
)
EXPECTED_LAYOUTS = ("empty", "single", "split", "grid")
EXPECTED_OUTPUT_KINDS = ("visual", "audio", "audiovisual", "code")
EXPECTED_SOURCE_REGISTRIES = (
    "multimodal_live_preview_registry",
    "nextjs_multi_preview_comparison",
    "nextjs_multi_preview_workspace",
    "nextjs_artifact_comparison",
    "nextjs_preview_renderers",
    "nextjs_preview_runtime_adapters",
)
EXPECTED_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_LIVE_PREVIEW_REGISTRY",
    "clients.nextjs.multi_preview_comparison.buildMultiPreviewComparisonModel",
    "clients.nextjs.multi_preview_comparison.resolveMultiPreviewLayout",
    "clients.nextjs.multi_preview_comparison.MultiPreviewCandidate",
    "clients.nextjs.components.MultiPreviewComparisonWorkspace",
    "clients.nextjs.artifact_comparison.buildArtifactComparisonModel",
    "clients.nextjs.preview_renderers.buildPreviewRendererRoute",
    "clients.nextjs.preview_runtime_adapters.buildPreviewRuntimeSource",
)
EXPECTED_MULTI_PREVIEW_SURFACES = (
    "multi_preview_workspace",
    "multi_preview_candidate_grid",
    "multi_preview_split_layout",
    "candidate_preview_card",
    "comparison_fallback_panel",
    "recommendation_summary_panel",
    "multi_preview_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "profile_id",
    "profile_name",
    "preview_kind",
    "comparison_layouts",
    "output_kinds",
    "source_live_preview_profile_ids",
    "candidate_state_fields",
    "source_reference_ids",
    "route_applicability",
    "multi_preview_surfaces",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "rendering_execution_implemented",
    "candidate_selection_execution_implemented",
    "artifact_selection_mutation_implemented",
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


class MultimodalStudioMultiPreviewTests(unittest.TestCase):
    def test_multi_preview_registry_covers_expected_sources(self) -> None:
        registry = multimodal_multi_preview_registry()

        self.assertEqual(registry.role, "multimodal_multi_preview_registry")
        self.assertEqual(
            registry.serialization_version,
            "multimodal_multi_preview_registry.v1",
        )
        self.assertEqual(registry.profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.preview_kinds, EXPECTED_PREVIEW_KINDS)
        self.assertEqual(registry.comparison_layouts, EXPECTED_LAYOUTS)
        self.assertEqual(registry.output_kinds, EXPECTED_OUTPUT_KINDS)
        self.assertEqual(
            registry.live_preview_profile_ids,
            multimodal_live_preview_registry().profile_ids,
        )
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertEqual(registry.source_reference_ids, EXPECTED_SOURCE_REFERENCES)
        self.assertEqual(
            registry.multi_preview_surface_refs,
            EXPECTED_MULTI_PREVIEW_SURFACES,
        )
        self.assertIn("does not execute rendering", registry.authority_boundary)
        self.assertIn("select artifacts", registry.authority_boundary)
        self.assertIn(
            "candidate_selection_execution", registry.blocked_runtime_behaviors
        )
        self.assertIn("artifact_selection_mutation", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.rendering_execution_implemented)
        self.assertFalse(registry.candidate_selection_execution_implemented)
        self.assertFalse(registry.artifact_selection_mutation_implemented)
        self.assertFalse(registry.browser_canvas_runtime_change_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.external_provider_calls_implemented)
        self.assertFalse(registry.networking_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_storage_implemented)
        self.assertFalse(registry.active_studio_behavior_implemented)

    def test_multi_preview_profiles_are_passive_and_source_aligned(self) -> None:
        registry = multimodal_multi_preview_registry()
        known_routes = set(registry.route_names)
        known_layouts = set(registry.comparison_layouts)
        known_outputs = set(registry.output_kinds)
        known_live_profiles = set(registry.live_preview_profile_ids)
        known_surfaces = set(registry.multi_preview_surface_refs)
        known_source_references = set(registry.source_reference_ids)
        covered_source_references: set[str] = set()

        for profile in registry.multi_preview_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "multimodal_multi_preview_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(set(profile.comparison_layouts).issubset(known_layouts))
            self.assertTrue(set(profile.output_kinds).issubset(known_outputs))
            self.assertTrue(
                set(profile.source_live_preview_profile_ids).issubset(
                    known_live_profiles
                )
            )
            self.assertTrue(
                set(profile.multi_preview_surfaces).issubset(known_surfaces)
            )
            self.assertTrue(
                set(profile.source_reference_ids).issubset(known_source_references)
            )
            covered_source_references.update(profile.source_reference_ids)
            self.assertIn(
                "candidate_selection_execution",
                profile.blocked_runtime_behaviors,
            )
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.rendering_execution_implemented)
            self.assertFalse(profile.candidate_selection_execution_implemented)
            self.assertFalse(profile.artifact_selection_mutation_implemented)
            self.assertFalse(profile.browser_canvas_runtime_change_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.external_provider_calls_implemented)
            self.assertFalse(profile.networking_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_storage_implemented)
            self.assertFalse(profile.active_studio_behavior_implemented)

        self.assertEqual(covered_source_references, known_source_references)

    def test_multi_preview_lookup_helpers_are_stable(self) -> None:
        profile = multimodal_multi_preview_profile_by_id(
            "recommended_candidate_multi_preview"
        )
        missing_profile = multimodal_multi_preview_profile_by_id("missing_profile")
        grid_profiles = multimodal_multi_preview_profiles_for_layout("grid")
        preview_route_profiles = multimodal_multi_preview_profiles_for_route(
            RouteName.PREVIEW
        )
        runtime_status_profiles = (
            multimodal_multi_preview_profiles_for_live_preview_profile(
                "runtime_status_live_preview"
            )
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.preview_kind, "recommended_candidate_preview")
        self.assertIn("recommended_reason", profile.candidate_state_fields)
        self.assertIn(
            "no_candidate_selection_execution_notice",
            profile.advisory_outputs,
        )
        self.assertEqual(
            tuple(item.profile_id for item in grid_profiles),
            (
                "candidate_grid_multi_preview",
                "recommended_candidate_multi_preview",
                "fallback_multi_preview",
            ),
        )
        self.assertEqual(
            tuple(item.profile_id for item in preview_route_profiles),
            EXPECTED_PROFILE_IDS,
        )
        self.assertEqual(
            tuple(item.profile_id for item in runtime_status_profiles),
            (
                "candidate_grid_multi_preview",
                "recommended_candidate_multi_preview",
                "fallback_multi_preview",
            ),
        )

    def test_registry_rejects_mismatched_multi_preview_metadata(self) -> None:
        registry = multimodal_multi_preview_registry()
        first_profile = registry.multi_preview_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Multi Preview"}
        )
        unknown_live_profile = first_profile.model_copy(
            update={"source_live_preview_profile_ids": ("unknown_live_preview",)}
        )

        duplicate_kwargs = _registry_kwargs(registry)
        duplicate_kwargs["multi_preview_profiles"] = (
            first_profile,
            duplicate_profile,
        ) + registry.multi_preview_profiles[2:]
        with self.assertRaisesRegex(ValueError, "profile_ids must be unique"):
            MultimodalMultiPreviewRegistry(**duplicate_kwargs)

        unknown_live_kwargs = _registry_kwargs(registry)
        unknown_live_kwargs["multi_preview_profiles"] = (
            unknown_live_profile,
        ) + registry.multi_preview_profiles[1:]
        with self.assertRaisesRegex(
            ValueError,
            "source_live_preview_profile_ids",
        ):
            MultimodalMultiPreviewRegistry(**unknown_live_kwargs)

    def test_multi_preview_metadata_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        multimodal_multi_preview_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")


def _registry_kwargs(
    registry: MultimodalMultiPreviewRegistry,
) -> dict[str, object]:
    return {
        "multi_preview_profiles": registry.multi_preview_profiles,
        "profile_ids": registry.profile_ids,
        "preview_kinds": registry.preview_kinds,
        "comparison_layouts": registry.comparison_layouts,
        "output_kinds": registry.output_kinds,
        "live_preview_profile_ids": registry.live_preview_profile_ids,
        "route_names": registry.route_names,
        "profile_count": registry.profile_count,
        "source_registries": registry.source_registries,
        "source_reference_ids": registry.source_reference_ids,
        "multi_preview_surface_refs": registry.multi_preview_surface_refs,
        "observability_surfaces": registry.observability_surfaces,
    }


if __name__ == "__main__":
    unittest.main()
