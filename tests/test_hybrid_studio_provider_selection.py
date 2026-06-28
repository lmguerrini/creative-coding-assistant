import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    ProviderSelectionRegistry,
    auto_mode_registry,
    cloud_model_registry,
    hitl_decision_registry,
    local_model_registry,
    provider_selection_profile_by_id,
    provider_selection_profiles_for_route,
    provider_selection_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "current_config_provider_visibility_profile",
    "local_candidate_provider_visibility_profile",
    "cloud_candidate_provider_visibility_profile",
    "operator_override_provider_visibility_profile",
)
EXPECTED_POSTURES = (
    "current_config_visibility",
    "local_candidate_visibility",
    "cloud_candidate_visibility",
    "operator_override_visibility",
)
EXPECTED_PROVIDER_CANDIDATES = (
    "openai",
    "ollama",
    "lm_studio",
    "llama_cpp",
    "local_transformers",
)
EXPECTED_SOURCE_REGISTRIES = (
    "local_model_registry",
    "cloud_model_registry",
    "auto_mode_registry",
    "hitl_decision_registry",
    "generation_provider_factory",
    "settings_generation_provider_config",
)
EXPECTED_SELECTION_SURFACES = (
    "provider_selection_panel",
    "model_catalog_panel",
    "auto_mode_panel",
    "hitl_decision_panel",
    "operator_override_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "provider_selection_profile_id",
    "profile_name",
    "provider_selection_posture",
    "provider_candidate_ids",
    "source_local_surface_ids",
    "source_cloud_surface_ids",
    "source_auto_mode_profile_ids",
    "source_hitl_decision_profile_ids",
    "route_applicability",
    "selection_inputs",
    "advisory_outputs",
    "selection_surface_refs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "provider_selection_execution_implemented",
    "automatic_provider_selection_implemented",
    "automatic_model_selection_implemented",
    "provider_model_routing_implemented",
    "model_switching_implemented",
    "local_provider_execution_implemented",
    "cloud_provider_execution_implemented",
    "workflow_control_implemented",
    "human_input_request_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "persistent_replay_storage_implemented",
    "serialization_version",
    "metadata_only",
}


class HybridStudioProviderSelectionTests(unittest.TestCase):
    def test_provider_selection_registry_covers_expected_profiles(self) -> None:
        registry = provider_selection_registry()

        self.assertEqual(registry.role, "provider_selection_registry")
        self.assertEqual(
            registry.serialization_version,
            "provider_selection_registry.v1",
        )
        self.assertEqual(registry.provider_selection_profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.provider_selection_postures, EXPECTED_POSTURES)
        self.assertEqual(registry.provider_candidate_ids, EXPECTED_PROVIDER_CANDIDATES)
        self.assertEqual(registry.local_surface_ids, local_model_registry().surface_ids)
        self.assertEqual(registry.cloud_surface_ids, cloud_model_registry().surface_ids)
        self.assertEqual(
            registry.auto_mode_profile_ids,
            auto_mode_registry().auto_mode_profile_ids,
        )
        self.assertEqual(
            registry.hitl_decision_profile_ids,
            hitl_decision_registry().hitl_decision_profile_ids,
        )
        self.assertEqual(registry.selection_surface_refs, EXPECTED_SELECTION_SURFACES)
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertIn("does not select providers", registry.authority_boundary)
        self.assertIn(
            "provider_selection_execution", registry.blocked_runtime_behaviors
        )
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.provider_selection_execution_implemented)
        self.assertFalse(registry.automatic_provider_selection_implemented)
        self.assertFalse(registry.automatic_model_selection_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.model_switching_implemented)
        self.assertFalse(registry.local_provider_execution_implemented)
        self.assertFalse(registry.cloud_provider_execution_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_replay_storage_implemented)

    def test_provider_selection_profiles_are_passive_and_source_aligned(self) -> None:
        registry = provider_selection_registry()
        known_routes = set(registry.route_names)
        known_providers = set(registry.provider_candidate_ids)
        known_local_surfaces = set(local_model_registry().surface_ids)
        known_cloud_surfaces = set(cloud_model_registry().surface_ids)
        known_auto_profiles = set(auto_mode_registry().auto_mode_profile_ids)
        known_hitl_profiles = set(hitl_decision_registry().hitl_decision_profile_ids)
        known_selection_surfaces = set(registry.selection_surface_refs)

        for profile in registry.provider_selection_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "provider_selection_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.provider_candidate_ids).issubset(known_providers)
            )
            self.assertTrue(
                set(profile.source_local_surface_ids).issubset(known_local_surfaces)
            )
            self.assertTrue(
                set(profile.source_cloud_surface_ids).issubset(known_cloud_surfaces)
            )
            self.assertTrue(
                set(profile.source_auto_mode_profile_ids).issubset(known_auto_profiles)
            )
            self.assertTrue(
                set(profile.source_hitl_decision_profile_ids).issubset(
                    known_hitl_profiles
                )
            )
            self.assertTrue(
                set(profile.selection_surface_refs).issubset(known_selection_surfaces)
            )
            self.assertIn(
                "provider_or_model_routing", profile.blocked_runtime_behaviors
            )
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.provider_selection_execution_implemented)
            self.assertFalse(profile.automatic_provider_selection_implemented)
            self.assertFalse(profile.automatic_model_selection_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.model_switching_implemented)
            self.assertFalse(profile.local_provider_execution_implemented)
            self.assertFalse(profile.cloud_provider_execution_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_replay_storage_implemented)

    def test_provider_selection_lookup_helpers_are_stable(self) -> None:
        profile = provider_selection_profile_by_id(
            "operator_override_provider_visibility_profile"
        )
        missing_profile = provider_selection_profile_by_id("missing_profile")
        review_profiles = provider_selection_profiles_for_route(RouteName.REVIEW)
        preview_profiles = provider_selection_profiles_for_route("preview")

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(
            profile.provider_selection_posture, "operator_override_visibility"
        )
        self.assertEqual(profile.provider_candidate_ids, EXPECTED_PROVIDER_CANDIDATES)
        self.assertEqual(
            tuple(item.provider_selection_profile_id for item in preview_profiles),
            (
                "current_config_provider_visibility_profile",
                "local_candidate_provider_visibility_profile",
                "operator_override_provider_visibility_profile",
            ),
        )
        self.assertEqual(
            tuple(item.provider_selection_profile_id for item in review_profiles),
            EXPECTED_PROFILE_IDS,
        )

    def test_registry_rejects_mismatched_sources_or_provider_refs(self) -> None:
        registry = provider_selection_registry()
        first_profile = registry.provider_selection_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Profile"}
        )
        unknown_provider_profile = first_profile.model_copy(
            update={"provider_candidate_ids": ("unknown_provider",)}
        )
        mismatched_source_profile = first_profile.model_copy(
            update={
                "source_registries": (
                    "local_model_registry",
                    "cloud_model_registry",
                    "auto_mode_registry",
                    "hitl_decision_registry",
                    "generation_provider_factory",
                    "unknown_registry",
                )
            }
        )

        with self.assertRaisesRegex(
            ValueError, "provider_selection_profile_ids must be unique"
        ):
            ProviderSelectionRegistry(
                provider_selection_profiles=(first_profile, duplicate_profile)
                + registry.provider_selection_profiles[2:],
                provider_selection_profile_ids=registry.provider_selection_profile_ids,
                provider_selection_postures=registry.provider_selection_postures,
                provider_candidate_ids=registry.provider_candidate_ids,
                local_surface_ids=registry.local_surface_ids,
                cloud_surface_ids=registry.cloud_surface_ids,
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                selection_surface_refs=registry.selection_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "known providers"):
            ProviderSelectionRegistry(
                provider_selection_profiles=(unknown_provider_profile,)
                + registry.provider_selection_profiles[1:],
                provider_selection_profile_ids=registry.provider_selection_profile_ids,
                provider_selection_postures=registry.provider_selection_postures,
                provider_candidate_ids=registry.provider_candidate_ids,
                local_surface_ids=registry.local_surface_ids,
                cloud_surface_ids=registry.cloud_surface_ids,
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                selection_surface_refs=registry.selection_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_registries must match"):
            ProviderSelectionRegistry(
                provider_selection_profiles=(mismatched_source_profile,)
                + registry.provider_selection_profiles[1:],
                provider_selection_profile_ids=registry.provider_selection_profile_ids,
                provider_selection_postures=registry.provider_selection_postures,
                provider_candidate_ids=registry.provider_candidate_ids,
                local_surface_ids=registry.local_surface_ids,
                cloud_surface_ids=registry.cloud_surface_ids,
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                selection_surface_refs=registry.selection_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

    def test_provider_selection_metadata_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        provider_selection_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_provider_selection_metadata_does_not_declare_active_selection(
        self,
    ) -> None:
        registry = provider_selection_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.source_registries,
                *registry.selection_surface_refs,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for profile in registry.provider_selection_profiles
                    for field in (
                        profile.provider_selection_profile_id,
                        profile.provider_selection_posture,
                        *profile.provider_candidate_ids,
                        *profile.selection_inputs,
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
            "auto_select_provider",
            "switch_model_now",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
