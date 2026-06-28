import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    CostProfileRegistry,
    cloud_model_registry,
    cost_profile_by_id,
    cost_profile_registry,
    cost_profiles_for_band,
    cost_profiles_for_route,
    cost_threshold_routing_registry,
    local_model_registry,
    model_profile_registry,
    provider_selection_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "planning_iteration_cost_profile",
    "creative_reasoning_cost_profile",
    "curation_refinement_cost_profile",
    "final_review_cost_profile",
)
EXPECTED_KINDS = (
    "planning_iteration_budget",
    "creative_reasoning_budget",
    "curation_refinement_budget",
    "final_review_budget",
)
EXPECTED_COST_BANDS = ("medium", "high", "guarded", "low")
EXPECTED_SOURCE_REGISTRIES = (
    "model_profile_registry",
    "provider_selection_registry",
    "cost_threshold_routing_registry",
    "local_model_registry",
    "cloud_model_registry",
    "execution_simulator_registry",
)
EXPECTED_COST_SURFACES = (
    "cost_profile_panel",
    "model_profile_panel",
    "provider_selection_panel",
    "execution_simulator_panel",
    "budget_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "cost_profile_id",
    "profile_name",
    "cost_profile_kind",
    "cost_band",
    "advisory_cost_range",
    "source_model_profile_ids",
    "source_provider_selection_profile_ids",
    "source_cost_threshold_profile_ids",
    "source_local_surface_ids",
    "source_cloud_surface_ids",
    "route_applicability",
    "cost_dimensions",
    "cost_inputs",
    "advisory_outputs",
    "cost_surface_refs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "cost_profile_execution_implemented",
    "cost_scoring_implemented",
    "pricing_lookup_implemented",
    "budget_enforcement_implemented",
    "cost_based_routing_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "model_selection_implemented",
    "execution_optimization_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "persistent_replay_storage_implemented",
    "serialization_version",
    "metadata_only",
}


class HybridStudioCostProfileTests(unittest.TestCase):
    def test_cost_profile_registry_covers_expected_profiles(self) -> None:
        registry = cost_profile_registry()

        self.assertEqual(registry.role, "cost_profile_registry")
        self.assertEqual(registry.serialization_version, "cost_profile_registry.v1")
        self.assertEqual(registry.cost_profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.cost_profile_kinds, EXPECTED_KINDS)
        self.assertEqual(registry.cost_bands, EXPECTED_COST_BANDS)
        self.assertEqual(
            registry.cost_bands,
            cost_threshold_routing_registry().cost_bands,
        )
        self.assertEqual(
            registry.model_profile_ids,
            model_profile_registry().model_profile_ids,
        )
        self.assertEqual(
            registry.provider_selection_profile_ids,
            provider_selection_registry().provider_selection_profile_ids,
        )
        self.assertEqual(
            registry.cost_threshold_profile_ids,
            cost_threshold_routing_registry().cost_threshold_profile_ids,
        )
        self.assertEqual(registry.local_surface_ids, local_model_registry().surface_ids)
        self.assertEqual(registry.cloud_surface_ids, cloud_model_registry().surface_ids)
        self.assertEqual(registry.cost_surface_refs, EXPECTED_COST_SURFACES)
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertIn("does not calculate cost", registry.authority_boundary)
        self.assertIn("cost_scoring", registry.blocked_runtime_behaviors)
        self.assertIn("budget_enforcement", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.cost_profile_execution_implemented)
        self.assertFalse(registry.cost_scoring_implemented)
        self.assertFalse(registry.pricing_lookup_implemented)
        self.assertFalse(registry.budget_enforcement_implemented)
        self.assertFalse(registry.cost_based_routing_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.provider_execution_implemented)
        self.assertFalse(registry.model_selection_implemented)
        self.assertFalse(registry.execution_optimization_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_replay_storage_implemented)

    def test_cost_profiles_are_passive_and_source_aligned(self) -> None:
        registry = cost_profile_registry()
        known_routes = set(registry.route_names)
        known_model_profiles = set(model_profile_registry().model_profile_ids)
        known_provider_profiles = set(
            provider_selection_registry().provider_selection_profile_ids
        )
        known_cost_thresholds = set(
            cost_threshold_routing_registry().cost_threshold_profile_ids
        )
        known_local_surfaces = set(local_model_registry().surface_ids)
        known_cloud_surfaces = set(cloud_model_registry().surface_ids)
        known_cost_surfaces = set(registry.cost_surface_refs)

        for profile in registry.cost_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(profile.serialization_version, "cost_profile.v1")
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertIn(profile.cost_band, registry.cost_bands)
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_model_profile_ids).issubset(known_model_profiles)
            )
            self.assertTrue(
                set(profile.source_provider_selection_profile_ids).issubset(
                    known_provider_profiles
                )
            )
            self.assertTrue(
                set(profile.source_cost_threshold_profile_ids).issubset(
                    known_cost_thresholds
                )
            )
            self.assertTrue(
                set(profile.source_local_surface_ids).issubset(known_local_surfaces)
            )
            self.assertTrue(
                set(profile.source_cloud_surface_ids).issubset(known_cloud_surfaces)
            )
            self.assertTrue(
                set(profile.cost_surface_refs).issubset(known_cost_surfaces)
            )
            cost_low, cost_high = profile.advisory_cost_range
            self.assertGreaterEqual(cost_low, 0)
            self.assertGreaterEqual(cost_high, cost_low)
            self.assertIn("cost_based_routing", profile.blocked_runtime_behaviors)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.cost_profile_execution_implemented)
            self.assertFalse(profile.cost_scoring_implemented)
            self.assertFalse(profile.pricing_lookup_implemented)
            self.assertFalse(profile.budget_enforcement_implemented)
            self.assertFalse(profile.cost_based_routing_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.provider_execution_implemented)
            self.assertFalse(profile.model_selection_implemented)
            self.assertFalse(profile.execution_optimization_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_replay_storage_implemented)

    def test_cost_profile_lookup_helpers_are_stable(self) -> None:
        profile = cost_profile_by_id("creative_reasoning_cost_profile")
        missing_profile = cost_profile_by_id("missing_profile")
        review_profiles = cost_profiles_for_route(RouteName.REVIEW)
        preview_profiles = cost_profiles_for_route("preview")
        high_cost_profiles = cost_profiles_for_band("high")

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.cost_profile_kind, "creative_reasoning_budget")
        self.assertEqual(profile.cost_band, "high")
        self.assertEqual(profile.advisory_cost_range, (4, 7))
        self.assertEqual(
            tuple(item.cost_profile_id for item in preview_profiles),
            (
                "planning_iteration_cost_profile",
                "curation_refinement_cost_profile",
            ),
        )
        self.assertEqual(
            tuple(item.cost_profile_id for item in review_profiles),
            (
                "creative_reasoning_cost_profile",
                "curation_refinement_cost_profile",
                "final_review_cost_profile",
            ),
        )
        self.assertEqual(
            tuple(item.cost_profile_id for item in high_cost_profiles),
            ("creative_reasoning_cost_profile",),
        )

    def test_registry_rejects_mismatched_sources_or_cost_metadata(self) -> None:
        registry = cost_profile_registry()
        first_profile = registry.cost_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Profile"}
        )
        unknown_model_profile = first_profile.model_copy(
            update={"source_model_profile_ids": ("unknown_model_profile",)}
        )
        invalid_range_profile = first_profile.model_copy(
            update={"advisory_cost_range": (5, 2)}
        )

        with self.assertRaisesRegex(ValueError, "cost_profile_ids must be unique"):
            CostProfileRegistry(
                cost_profiles=(first_profile, duplicate_profile)
                + registry.cost_profiles[2:],
                cost_profile_ids=registry.cost_profile_ids,
                cost_profile_kinds=registry.cost_profile_kinds,
                cost_bands=registry.cost_bands,
                model_profile_ids=registry.model_profile_ids,
                provider_selection_profile_ids=registry.provider_selection_profile_ids,
                cost_threshold_profile_ids=registry.cost_threshold_profile_ids,
                local_surface_ids=registry.local_surface_ids,
                cloud_surface_ids=registry.cloud_surface_ids,
                cost_surface_refs=registry.cost_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_model_profile_ids"):
            CostProfileRegistry(
                cost_profiles=(unknown_model_profile,) + registry.cost_profiles[1:],
                cost_profile_ids=registry.cost_profile_ids,
                cost_profile_kinds=registry.cost_profile_kinds,
                cost_bands=registry.cost_bands,
                model_profile_ids=registry.model_profile_ids,
                provider_selection_profile_ids=registry.provider_selection_profile_ids,
                cost_threshold_profile_ids=registry.cost_threshold_profile_ids,
                local_surface_ids=registry.local_surface_ids,
                cloud_surface_ids=registry.cloud_surface_ids,
                cost_surface_refs=registry.cost_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "advisory cost range"):
            CostProfileRegistry(
                cost_profiles=(invalid_range_profile,) + registry.cost_profiles[1:],
                cost_profile_ids=registry.cost_profile_ids,
                cost_profile_kinds=registry.cost_profile_kinds,
                cost_bands=registry.cost_bands,
                model_profile_ids=registry.model_profile_ids,
                provider_selection_profile_ids=registry.provider_selection_profile_ids,
                cost_threshold_profile_ids=registry.cost_threshold_profile_ids,
                local_surface_ids=registry.local_surface_ids,
                cloud_surface_ids=registry.cloud_surface_ids,
                cost_surface_refs=registry.cost_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

    def test_cost_profiles_do_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        cost_profile_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")
