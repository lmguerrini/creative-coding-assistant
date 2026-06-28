import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    CloudModelRegistry,
    cloud_model_registry,
    cloud_model_surface_by_id,
    cloud_model_surfaces_for_provider,
    local_model_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_SURFACE_IDS = (
    "openai_generation_model_surface",
    "openai_embedding_model_surface",
    "ragas_evaluator_model_surface",
    "provider_reported_response_model_surface",
)
EXPECTED_PROVIDER_KINDS = ("openai",)
EXPECTED_SOURCE_REGISTRIES = (
    "settings_generation_provider_config",
    "generation_provider_factory",
    "generation_provider_contract",
    "openai_provider_adapter",
    "provider_telemetry_metadata",
    "local_model_registry",
)
EXPECTED_STUDIO_SURFACES = (
    "cloud_model_catalog",
    "cloud_model_readiness_inspector",
    "provider_selection_metadata",
    "execution_simulator_metadata",
    "local_cloud_comparison_metadata",
)
REQUIRED_SURFACE_FIELDS = {
    "surface_id",
    "surface_name",
    "provider_kind",
    "capability_band",
    "configuration_source",
    "latency_posture",
    "cost_posture",
    "privacy_posture",
    "route_applicability",
    "supported_payloads",
    "readiness_signals",
    "studio_surface_refs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "cloud_provider_execution_implemented",
    "provider_model_routing_implemented",
    "automatic_model_selection_implemented",
    "external_provider_calls_implemented",
    "pricing_latency_optimization_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "persistent_replay_storage_implemented",
    "serialization_version",
    "metadata_only",
}


class HybridStudioCloudModelTests(unittest.TestCase):
    def test_cloud_model_registry_covers_expected_surfaces(self) -> None:
        registry = cloud_model_registry()

        self.assertEqual(registry.role, "cloud_model_registry")
        self.assertEqual(
            registry.serialization_version,
            "cloud_model_registry.v1",
        )
        self.assertEqual(registry.surface_ids, EXPECTED_SURFACE_IDS)
        self.assertEqual(registry.provider_kinds, EXPECTED_PROVIDER_KINDS)
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertEqual(registry.studio_surface_refs, EXPECTED_STUDIO_SURFACES)
        self.assertIn("does not call cloud providers", registry.authority_boundary)
        self.assertIn("provider_or_model_routing", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.cloud_provider_execution_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.automatic_model_selection_implemented)
        self.assertFalse(registry.external_provider_calls_implemented)
        self.assertFalse(registry.pricing_latency_optimization_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_replay_storage_implemented)

    def test_cloud_model_surfaces_are_passive_metadata(self) -> None:
        registry = cloud_model_registry()
        known_routes = set(registry.route_names)
        known_studio_surfaces = set(registry.studio_surface_refs)

        for surface in registry.model_surfaces:
            dumped = surface.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SURFACE_FIELDS)
            self.assertEqual(surface.serialization_version, "cloud_model_surface.v1")
            self.assertEqual(surface.source_registries, registry.source_registries)
            self.assertEqual(
                surface.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(surface.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(surface.studio_surface_refs).issubset(known_studio_surfaces)
            )
            self.assertEqual(surface.provider_kind, "openai")
            self.assertEqual(surface.cost_posture, "provider_metered")
            self.assertEqual(surface.privacy_posture, "external_provider_boundary")
            self.assertIn("cloud_provider_execution", surface.blocked_runtime_behaviors)
            self.assertTrue(surface.metadata_only)
            self.assertFalse(surface.cloud_provider_execution_implemented)
            self.assertFalse(surface.provider_model_routing_implemented)
            self.assertFalse(surface.automatic_model_selection_implemented)
            self.assertFalse(surface.external_provider_calls_implemented)
            self.assertFalse(surface.pricing_latency_optimization_implemented)
            self.assertFalse(surface.retry_triggering_implemented)
            self.assertFalse(surface.generated_output_mutation_implemented)
            self.assertFalse(surface.persistent_replay_storage_implemented)

    def test_cloud_model_sources_align_with_local_model_registry(self) -> None:
        cloud_registry = cloud_model_registry()
        local_registry = local_model_registry()

        self.assertIn("local_model_registry", cloud_registry.source_registries)
        self.assertIn(
            "local_cloud_comparison_metadata",
            cloud_registry.studio_surface_refs,
        )
        self.assertIn(
            "local_cloud_comparison_metadata",
            local_registry.studio_surface_refs,
        )
        self.assertEqual(
            {
                source
                for surface in cloud_registry.model_surfaces
                for source in surface.source_registries
            },
            set(cloud_registry.source_registries),
        )

    def test_cloud_model_lookup_helpers_are_stable(self) -> None:
        generation_surface = cloud_model_surface_by_id(
            "openai_generation_model_surface"
        )
        missing_surface = cloud_model_surface_by_id("missing_surface")
        openai_surfaces = cloud_model_surfaces_for_provider("openai")
        missing_provider_surfaces = cloud_model_surfaces_for_provider("missing")

        self.assertIsNone(missing_surface)
        self.assertEqual(missing_provider_surfaces, ())
        self.assertIsNotNone(generation_surface)
        assert generation_surface is not None
        self.assertEqual(generation_surface.provider_kind, "openai")
        self.assertEqual(generation_surface.configuration_source, "openai_model")
        self.assertEqual(generation_surface.route_applicability, tuple(RouteName))
        self.assertEqual(
            tuple(surface.surface_id for surface in openai_surfaces),
            EXPECTED_SURFACE_IDS,
        )

    def test_registry_rejects_mismatched_sources_or_surfaces(self) -> None:
        registry = cloud_model_registry()
        first_surface = registry.model_surfaces[0]
        duplicate_surface = first_surface.model_copy(
            update={"surface_name": "Duplicate Surface"}
        )
        mismatched_source_surface = first_surface.model_copy(
            update={
                "source_registries": (
                    "settings_generation_provider_config",
                    "generation_provider_factory",
                    "generation_provider_contract",
                    "openai_provider_adapter",
                    "provider_telemetry_metadata",
                    "unknown_registry",
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "surface_ids must be unique"):
            CloudModelRegistry(
                model_surfaces=(first_surface, duplicate_surface)
                + registry.model_surfaces[2:],
                surface_ids=registry.surface_ids,
                provider_kinds=registry.provider_kinds,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                studio_surface_refs=registry.studio_surface_refs,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "surface_ids must match"):
            CloudModelRegistry(
                model_surfaces=registry.model_surfaces,
                surface_ids=("other_surface",) + registry.surface_ids[1:],
                provider_kinds=registry.provider_kinds,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                studio_surface_refs=registry.studio_surface_refs,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_registries must match"):
            CloudModelRegistry(
                model_surfaces=(mismatched_source_surface,)
                + registry.model_surfaces[1:],
                surface_ids=registry.surface_ids,
                provider_kinds=registry.provider_kinds,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                studio_surface_refs=registry.studio_surface_refs,
                observability_surfaces=registry.observability_surfaces,
            )

    def test_cloud_model_metadata_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        cloud_model_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_cloud_model_metadata_does_not_declare_active_execution(self) -> None:
        registry = cloud_model_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.source_registries,
                *registry.studio_surface_refs,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for surface in registry.model_surfaces
                    for field in (
                        surface.surface_id,
                        surface.provider_kind,
                        surface.capability_band,
                        surface.configuration_source,
                        *surface.readiness_signals,
                        *surface.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "route_provider",
            "provider_route",
            "execute_provider",
            "call_provider_now",
            "auto_select_model",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
