import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    LocalCloudComparisonRegistry,
    cloud_model_registry,
    cost_profile_registry,
    execution_simulator_registry,
    hybrid_execution_registry,
    local_cloud_comparison_profile_by_id,
    local_cloud_comparison_profiles_for_route,
    local_cloud_comparison_registry,
    local_model_registry,
    model_profile_registry,
    provider_selection_registry,
    quality_profile_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "generation_route_comparison_profile",
    "creative_reasoning_comparison_profile",
    "code_review_comparison_profile",
    "evaluation_review_comparison_profile",
)
EXPECTED_KINDS = (
    "generation_route_comparison",
    "creative_reasoning_comparison",
    "code_review_comparison",
    "evaluation_review_comparison",
)
EXPECTED_SOURCE_REGISTRIES = (
    "local_model_registry",
    "cloud_model_registry",
    "hybrid_execution_registry",
    "provider_selection_registry",
    "execution_simulator_registry",
    "model_profile_registry",
    "cost_profile_registry",
    "quality_profile_registry",
)
EXPECTED_COMPARISON_SURFACES = (
    "local_cloud_comparison_panel",
    "model_catalog_panel",
    "provider_selection_panel",
    "execution_simulator_panel",
    "cost_profile_panel",
    "quality_profile_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "comparison_profile_id",
    "profile_name",
    "comparison_kind",
    "source_local_surface_ids",
    "source_cloud_surface_ids",
    "source_execution_profile_ids",
    "source_provider_selection_profile_ids",
    "source_execution_simulation_profile_ids",
    "source_model_profile_ids",
    "source_cost_profile_ids",
    "source_quality_profile_ids",
    "route_applicability",
    "comparison_dimensions",
    "comparison_inputs",
    "advisory_outputs",
    "comparison_surface_refs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "comparison_runtime_execution_implemented",
    "local_provider_execution_implemented",
    "cloud_provider_execution_implemented",
    "parallel_model_execution_implemented",
    "provider_model_routing_implemented",
    "model_selection_implemented",
    "cost_scoring_implemented",
    "quality_scoring_implemented",
    "winner_selection_implemented",
    "fallback_execution_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "persistent_replay_storage_implemented",
    "serialization_version",
    "metadata_only",
}


class HybridStudioLocalCloudComparisonTests(unittest.TestCase):
    def test_comparison_registry_covers_expected_profiles(self) -> None:
        registry = local_cloud_comparison_registry()

        self.assertEqual(registry.role, "local_cloud_comparison_registry")
        self.assertEqual(
            registry.serialization_version,
            "local_cloud_comparison_registry.v1",
        )
        self.assertEqual(registry.comparison_profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.comparison_kinds, EXPECTED_KINDS)
        self.assertEqual(registry.local_surface_ids, local_model_registry().surface_ids)
        self.assertEqual(registry.cloud_surface_ids, cloud_model_registry().surface_ids)
        self.assertEqual(
            registry.execution_profile_ids,
            hybrid_execution_registry().execution_profile_ids,
        )
        self.assertEqual(
            registry.provider_selection_profile_ids,
            provider_selection_registry().provider_selection_profile_ids,
        )
        self.assertEqual(
            registry.execution_simulation_profile_ids,
            execution_simulator_registry().execution_simulation_profile_ids,
        )
        self.assertEqual(
            registry.model_profile_ids,
            model_profile_registry().model_profile_ids,
        )
        self.assertEqual(
            registry.cost_profile_ids,
            cost_profile_registry().cost_profile_ids,
        )
        self.assertEqual(
            registry.quality_profile_ids,
            quality_profile_registry().quality_profile_ids,
        )
        self.assertEqual(
            registry.comparison_surface_refs,
            EXPECTED_COMPARISON_SURFACES,
        )
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertIn(
            "does not execute local or cloud providers",
            registry.authority_boundary,
        )
        self.assertIn("winner_selection", registry.blocked_runtime_behaviors)
        self.assertIn("parallel_model_execution", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.comparison_runtime_execution_implemented)
        self.assertFalse(registry.local_provider_execution_implemented)
        self.assertFalse(registry.cloud_provider_execution_implemented)
        self.assertFalse(registry.parallel_model_execution_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.model_selection_implemented)
        self.assertFalse(registry.cost_scoring_implemented)
        self.assertFalse(registry.quality_scoring_implemented)
        self.assertFalse(registry.winner_selection_implemented)
        self.assertFalse(registry.fallback_execution_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_replay_storage_implemented)

    def test_comparison_profiles_are_passive_and_source_aligned(self) -> None:
        registry = local_cloud_comparison_registry()
        known_routes = set(registry.route_names)
        known_local_surfaces = set(local_model_registry().surface_ids)
        known_cloud_surfaces = set(cloud_model_registry().surface_ids)
        known_execution_profiles = set(
            hybrid_execution_registry().execution_profile_ids
        )
        known_provider_profiles = set(
            provider_selection_registry().provider_selection_profile_ids
        )
        known_simulations = set(
            execution_simulator_registry().execution_simulation_profile_ids
        )
        known_model_profiles = set(model_profile_registry().model_profile_ids)
        known_cost_profiles = set(cost_profile_registry().cost_profile_ids)
        known_quality_profiles = set(quality_profile_registry().quality_profile_ids)
        known_comparison_surfaces = set(registry.comparison_surface_refs)

        for profile in registry.comparison_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "local_cloud_comparison_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_local_surface_ids).issubset(known_local_surfaces)
            )
            self.assertTrue(
                set(profile.source_cloud_surface_ids).issubset(known_cloud_surfaces)
            )
            self.assertTrue(
                set(profile.source_execution_profile_ids).issubset(
                    known_execution_profiles
                )
            )
            self.assertTrue(
                set(profile.source_provider_selection_profile_ids).issubset(
                    known_provider_profiles
                )
            )
            self.assertTrue(
                set(profile.source_execution_simulation_profile_ids).issubset(
                    known_simulations
                )
            )
            self.assertTrue(
                set(profile.source_model_profile_ids).issubset(known_model_profiles)
            )
            self.assertTrue(
                set(profile.source_cost_profile_ids).issubset(known_cost_profiles)
            )
            self.assertTrue(
                set(profile.source_quality_profile_ids).issubset(known_quality_profiles)
            )
            self.assertTrue(
                set(profile.comparison_surface_refs).issubset(known_comparison_surfaces)
            )
            self.assertIn(
                "provider_or_model_routing", profile.blocked_runtime_behaviors
            )
            self.assertIn("winner_selection", profile.blocked_runtime_behaviors)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.comparison_runtime_execution_implemented)
            self.assertFalse(profile.local_provider_execution_implemented)
            self.assertFalse(profile.cloud_provider_execution_implemented)
            self.assertFalse(profile.parallel_model_execution_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.model_selection_implemented)
            self.assertFalse(profile.cost_scoring_implemented)
            self.assertFalse(profile.quality_scoring_implemented)
            self.assertFalse(profile.winner_selection_implemented)
            self.assertFalse(profile.fallback_execution_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_replay_storage_implemented)

    def test_comparison_lookup_helpers_are_stable(self) -> None:
        profile = local_cloud_comparison_profile_by_id(
            "creative_reasoning_comparison_profile"
        )
        missing_profile = local_cloud_comparison_profile_by_id("missing_profile")
        review_profiles = local_cloud_comparison_profiles_for_route(RouteName.REVIEW)
        preview_profiles = local_cloud_comparison_profiles_for_route("preview")

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.comparison_kind, "creative_reasoning_comparison")
        self.assertIn("lm_studio_chat_surface", profile.source_local_surface_ids)
        self.assertIn(
            "openai_generation_model_surface", profile.source_cloud_surface_ids
        )
        self.assertIn(
            "creative_reasoning_comparison_context",
            profile.advisory_outputs,
        )
        self.assertEqual(
            tuple(item.comparison_profile_id for item in preview_profiles),
            (
                "generation_route_comparison_profile",
                "code_review_comparison_profile",
            ),
        )
        self.assertEqual(
            tuple(item.comparison_profile_id for item in review_profiles),
            (
                "creative_reasoning_comparison_profile",
                "code_review_comparison_profile",
                "evaluation_review_comparison_profile",
            ),
        )

    def test_registry_rejects_mismatched_sources_or_comparison_metadata(self) -> None:
        registry = local_cloud_comparison_registry()
        first_profile = registry.comparison_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Profile"}
        )
        unknown_local_profile = first_profile.model_copy(
            update={"source_local_surface_ids": ("unknown_local_surface",)}
        )
        unknown_quality_profile = first_profile.model_copy(
            update={"source_quality_profile_ids": ("unknown_quality_profile",)}
        )

        with self.assertRaisesRegex(
            ValueError, "comparison_profile_ids must be unique"
        ):
            LocalCloudComparisonRegistry(
                comparison_profiles=(first_profile, duplicate_profile)
                + registry.comparison_profiles[2:],
                comparison_profile_ids=registry.comparison_profile_ids,
                comparison_kinds=registry.comparison_kinds,
                local_surface_ids=registry.local_surface_ids,
                cloud_surface_ids=registry.cloud_surface_ids,
                execution_profile_ids=registry.execution_profile_ids,
                provider_selection_profile_ids=registry.provider_selection_profile_ids,
                execution_simulation_profile_ids=(
                    registry.execution_simulation_profile_ids
                ),
                model_profile_ids=registry.model_profile_ids,
                cost_profile_ids=registry.cost_profile_ids,
                quality_profile_ids=registry.quality_profile_ids,
                comparison_surface_refs=registry.comparison_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_local_surface_ids"):
            LocalCloudComparisonRegistry(
                comparison_profiles=(unknown_local_profile,)
                + registry.comparison_profiles[1:],
                comparison_profile_ids=registry.comparison_profile_ids,
                comparison_kinds=registry.comparison_kinds,
                local_surface_ids=registry.local_surface_ids,
                cloud_surface_ids=registry.cloud_surface_ids,
                execution_profile_ids=registry.execution_profile_ids,
                provider_selection_profile_ids=registry.provider_selection_profile_ids,
                execution_simulation_profile_ids=(
                    registry.execution_simulation_profile_ids
                ),
                model_profile_ids=registry.model_profile_ids,
                cost_profile_ids=registry.cost_profile_ids,
                quality_profile_ids=registry.quality_profile_ids,
                comparison_surface_refs=registry.comparison_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_quality_profile_ids"):
            LocalCloudComparisonRegistry(
                comparison_profiles=(unknown_quality_profile,)
                + registry.comparison_profiles[1:],
                comparison_profile_ids=registry.comparison_profile_ids,
                comparison_kinds=registry.comparison_kinds,
                local_surface_ids=registry.local_surface_ids,
                cloud_surface_ids=registry.cloud_surface_ids,
                execution_profile_ids=registry.execution_profile_ids,
                provider_selection_profile_ids=registry.provider_selection_profile_ids,
                execution_simulation_profile_ids=(
                    registry.execution_simulation_profile_ids
                ),
                model_profile_ids=registry.model_profile_ids,
                cost_profile_ids=registry.cost_profile_ids,
                quality_profile_ids=registry.quality_profile_ids,
                comparison_surface_refs=registry.comparison_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

    def test_comparison_layer_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        local_cloud_comparison_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")
