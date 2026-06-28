import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    ModelProfileRegistry,
    cloud_model_registry,
    local_model_registry,
    model_profile_by_id,
    model_profile_registry,
    model_profiles_for_route,
    provider_selection_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "fast_iteration_model_profile",
    "creative_reasoning_model_profile",
    "code_assistance_model_profile",
    "evaluation_review_model_profile",
)
EXPECTED_KINDS = (
    "fast_iteration",
    "creative_reasoning",
    "code_assistance",
    "evaluation_review",
)
EXPECTED_SOURCE_REGISTRIES = (
    "local_model_registry",
    "cloud_model_registry",
    "provider_selection_registry",
    "execution_simulator_registry",
    "settings_generation_provider_config",
    "hybrid_agentic_workflow_registry",
)
EXPECTED_PROFILE_SURFACES = (
    "model_profile_panel",
    "model_catalog_panel",
    "provider_selection_panel",
    "execution_simulator_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "model_profile_id",
    "profile_name",
    "model_profile_kind",
    "source_local_surface_ids",
    "source_cloud_surface_ids",
    "provider_candidate_ids",
    "route_applicability",
    "capability_dimensions",
    "profile_inputs",
    "advisory_outputs",
    "profile_surface_refs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "model_profile_execution_implemented",
    "model_selection_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "cost_scoring_implemented",
    "quality_scoring_implemented",
    "execution_optimization_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "persistent_replay_storage_implemented",
    "serialization_version",
    "metadata_only",
}


class HybridStudioModelProfileTests(unittest.TestCase):
    def test_model_profile_registry_covers_expected_profiles(self) -> None:
        registry = model_profile_registry()

        self.assertEqual(registry.role, "model_profile_registry")
        self.assertEqual(registry.serialization_version, "model_profile_registry.v1")
        self.assertEqual(registry.model_profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.model_profile_kinds, EXPECTED_KINDS)
        self.assertEqual(registry.local_surface_ids, local_model_registry().surface_ids)
        self.assertEqual(registry.cloud_surface_ids, cloud_model_registry().surface_ids)
        self.assertEqual(
            registry.provider_candidate_ids,
            provider_selection_registry().provider_candidate_ids,
        )
        self.assertEqual(registry.profile_surface_refs, EXPECTED_PROFILE_SURFACES)
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertIn("does not select models", registry.authority_boundary)
        self.assertIn("model_selection", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.model_profile_execution_implemented)
        self.assertFalse(registry.model_selection_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.provider_execution_implemented)
        self.assertFalse(registry.cost_scoring_implemented)
        self.assertFalse(registry.quality_scoring_implemented)
        self.assertFalse(registry.execution_optimization_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_replay_storage_implemented)

    def test_model_profiles_are_passive_and_source_aligned(self) -> None:
        registry = model_profile_registry()
        known_routes = set(registry.route_names)
        known_local_surfaces = set(local_model_registry().surface_ids)
        known_cloud_surfaces = set(cloud_model_registry().surface_ids)
        known_providers = set(provider_selection_registry().provider_candidate_ids)
        known_profile_surfaces = set(registry.profile_surface_refs)

        for profile in registry.model_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(profile.serialization_version, "model_profile.v1")
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
                set(profile.provider_candidate_ids).issubset(known_providers)
            )
            self.assertTrue(
                set(profile.profile_surface_refs).issubset(known_profile_surfaces)
            )
            self.assertIn(
                "provider_or_model_routing", profile.blocked_runtime_behaviors
            )
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.model_profile_execution_implemented)
            self.assertFalse(profile.model_selection_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.provider_execution_implemented)
            self.assertFalse(profile.cost_scoring_implemented)
            self.assertFalse(profile.quality_scoring_implemented)
            self.assertFalse(profile.execution_optimization_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_replay_storage_implemented)

    def test_model_profile_lookup_helpers_are_stable(self) -> None:
        profile = model_profile_by_id("creative_reasoning_model_profile")
        missing_profile = model_profile_by_id("missing_profile")
        review_profiles = model_profiles_for_route(RouteName.REVIEW)
        preview_profiles = model_profiles_for_route("preview")

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.model_profile_kind, "creative_reasoning")
        self.assertIn("creative_reasoning_capability_profile", profile.advisory_outputs)
        self.assertEqual(
            tuple(item.model_profile_id for item in preview_profiles),
            (
                "fast_iteration_model_profile",
                "code_assistance_model_profile",
            ),
        )
        self.assertEqual(
            tuple(item.model_profile_id for item in review_profiles),
            (
                "creative_reasoning_model_profile",
                "code_assistance_model_profile",
                "evaluation_review_model_profile",
            ),
        )

    def test_registry_rejects_mismatched_sources_or_surface_refs(self) -> None:
        registry = model_profile_registry()
        first_profile = registry.model_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Profile"}
        )
        unknown_local_profile = first_profile.model_copy(
            update={"source_local_surface_ids": ("unknown_local_surface",)}
        )
        mismatched_source_profile = first_profile.model_copy(
            update={
                "source_registries": (
                    "local_model_registry",
                    "cloud_model_registry",
                    "provider_selection_registry",
                    "execution_simulator_registry",
                    "settings_generation_provider_config",
                    "unknown_registry",
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "model_profile_ids must be unique"):
            ModelProfileRegistry(
                model_profiles=(first_profile, duplicate_profile)
                + registry.model_profiles[2:],
                model_profile_ids=registry.model_profile_ids,
                model_profile_kinds=registry.model_profile_kinds,
                local_surface_ids=registry.local_surface_ids,
                cloud_surface_ids=registry.cloud_surface_ids,
                provider_candidate_ids=registry.provider_candidate_ids,
                profile_surface_refs=registry.profile_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "known local models"):
            ModelProfileRegistry(
                model_profiles=(unknown_local_profile,) + registry.model_profiles[1:],
                model_profile_ids=registry.model_profile_ids,
                model_profile_kinds=registry.model_profile_kinds,
                local_surface_ids=registry.local_surface_ids,
                cloud_surface_ids=registry.cloud_surface_ids,
                provider_candidate_ids=registry.provider_candidate_ids,
                profile_surface_refs=registry.profile_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_registries must match"):
            ModelProfileRegistry(
                model_profiles=(mismatched_source_profile,)
                + registry.model_profiles[1:],
                model_profile_ids=registry.model_profile_ids,
                model_profile_kinds=registry.model_profile_kinds,
                local_surface_ids=registry.local_surface_ids,
                cloud_surface_ids=registry.cloud_surface_ids,
                provider_candidate_ids=registry.provider_candidate_ids,
                profile_surface_refs=registry.profile_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

    def test_model_profiles_do_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        model_profile_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_model_profiles_do_not_declare_active_selection_or_scoring(self) -> None:
        registry = model_profile_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.source_registries,
                *registry.profile_surface_refs,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for profile in registry.model_profiles
                    for field in (
                        profile.model_profile_id,
                        profile.model_profile_kind,
                        *profile.capability_dimensions,
                        *profile.advisory_outputs,
                        *profile.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "route_provider",
            "provider_route",
            "execute_provider",
            "select_model_now",
            "score_quality_now",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
