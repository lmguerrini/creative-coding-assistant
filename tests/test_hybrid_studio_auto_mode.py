import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    AutoModeRegistry,
    auto_mode_profile_by_id,
    auto_mode_profiles_for_route,
    auto_mode_registry,
    hybrid_execution_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "auto_mode_observe_only_profile",
    "auto_mode_suggestion_profile",
    "auto_mode_simulation_profile",
    "auto_mode_operator_confirmed_profile",
)
EXPECTED_POSTURES = (
    "observe_only",
    "suggestion_only",
    "simulation_only",
    "operator_confirmed",
)
EXPECTED_SOURCE_REGISTRIES = (
    "local_model_registry",
    "cloud_model_registry",
    "hybrid_execution_registry",
    "agent_routing_registry",
    "settings_generation_provider_config",
    "hybrid_agentic_workflow_registry",
)
EXPECTED_STUDIO_SURFACES = (
    "auto_mode_panel",
    "hybrid_execution_panel",
    "provider_selection_metadata",
    "execution_simulator_metadata",
    "local_cloud_comparison_metadata",
)
REQUIRED_PROFILE_FIELDS = {
    "auto_mode_profile_id",
    "profile_name",
    "auto_mode_posture",
    "source_execution_profile_ids",
    "route_applicability",
    "advisory_inputs",
    "advisory_outputs",
    "operator_controls",
    "studio_surface_refs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "auto_mode_execution_implemented",
    "automatic_provider_selection_implemented",
    "automatic_model_selection_implemented",
    "hybrid_execution_implemented",
    "provider_model_routing_implemented",
    "human_input_request_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "persistent_replay_storage_implemented",
    "serialization_version",
    "metadata_only",
}


class HybridStudioAutoModeTests(unittest.TestCase):
    def test_auto_mode_registry_covers_expected_profiles(self) -> None:
        registry = auto_mode_registry()

        self.assertEqual(registry.role, "auto_mode_registry")
        self.assertEqual(
            registry.serialization_version,
            "auto_mode_registry.v1",
        )
        self.assertEqual(registry.auto_mode_profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.auto_mode_postures, EXPECTED_POSTURES)
        self.assertEqual(
            registry.execution_profile_ids,
            hybrid_execution_registry().execution_profile_ids,
        )
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertEqual(registry.studio_surface_refs, EXPECTED_STUDIO_SURFACES)
        self.assertIn("does not execute workflows", registry.authority_boundary)
        self.assertIn("auto_mode_execution", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.auto_mode_execution_implemented)
        self.assertFalse(registry.automatic_provider_selection_implemented)
        self.assertFalse(registry.automatic_model_selection_implemented)
        self.assertFalse(registry.hybrid_execution_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_replay_storage_implemented)

    def test_auto_mode_profiles_are_passive_and_source_aligned(self) -> None:
        registry = auto_mode_registry()
        known_routes = set(registry.route_names)
        known_execution_profiles = set(
            hybrid_execution_registry().execution_profile_ids
        )
        known_studio_surfaces = set(registry.studio_surface_refs)

        for profile in registry.auto_mode_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(profile.serialization_version, "auto_mode_profile.v1")
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_execution_profile_ids).issubset(
                    known_execution_profiles
                )
            )
            self.assertTrue(
                set(profile.studio_surface_refs).issubset(known_studio_surfaces)
            )
            self.assertTrue(profile.operator_controls)
            self.assertIn(
                "automatic_model_selection",
                profile.blocked_runtime_behaviors,
            )
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.auto_mode_execution_implemented)
            self.assertFalse(profile.automatic_provider_selection_implemented)
            self.assertFalse(profile.automatic_model_selection_implemented)
            self.assertFalse(profile.hybrid_execution_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_replay_storage_implemented)

    def test_auto_mode_lookup_helpers_are_stable(self) -> None:
        profile = auto_mode_profile_by_id("auto_mode_suggestion_profile")
        missing_profile = auto_mode_profile_by_id("missing_profile")
        review_profiles = auto_mode_profiles_for_route(RouteName.REVIEW)
        preview_profiles = auto_mode_profiles_for_route("preview")

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.auto_mode_posture, "suggestion_only")
        self.assertIn("manual_confirmation_requirement", profile.advisory_outputs)
        self.assertEqual(
            tuple(item.auto_mode_profile_id for item in preview_profiles),
            (
                "auto_mode_observe_only_profile",
                "auto_mode_operator_confirmed_profile",
            ),
        )
        self.assertEqual(
            tuple(item.auto_mode_profile_id for item in review_profiles),
            (
                "auto_mode_observe_only_profile",
                "auto_mode_suggestion_profile",
                "auto_mode_simulation_profile",
                "auto_mode_operator_confirmed_profile",
            ),
        )

    def test_registry_rejects_mismatched_sources_or_execution_refs(self) -> None:
        registry = auto_mode_registry()
        first_profile = registry.auto_mode_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Profile"}
        )
        unknown_execution_profile = first_profile.model_copy(
            update={"source_execution_profile_ids": ("unknown_execution_profile",)}
        )
        mismatched_source_profile = first_profile.model_copy(
            update={
                "source_registries": (
                    "local_model_registry",
                    "cloud_model_registry",
                    "hybrid_execution_registry",
                    "agent_routing_registry",
                    "settings_generation_provider_config",
                    "unknown_registry",
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "auto_mode_profile_ids must be unique"):
            AutoModeRegistry(
                auto_mode_profiles=(first_profile, duplicate_profile)
                + registry.auto_mode_profiles[2:],
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                auto_mode_postures=registry.auto_mode_postures,
                execution_profile_ids=registry.execution_profile_ids,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                studio_surface_refs=registry.studio_surface_refs,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "known execution profiles"):
            AutoModeRegistry(
                auto_mode_profiles=(unknown_execution_profile,)
                + registry.auto_mode_profiles[1:],
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                auto_mode_postures=registry.auto_mode_postures,
                execution_profile_ids=registry.execution_profile_ids,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                studio_surface_refs=registry.studio_surface_refs,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_registries must match"):
            AutoModeRegistry(
                auto_mode_profiles=(mismatched_source_profile,)
                + registry.auto_mode_profiles[1:],
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                auto_mode_postures=registry.auto_mode_postures,
                execution_profile_ids=registry.execution_profile_ids,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                studio_surface_refs=registry.studio_surface_refs,
                observability_surfaces=registry.observability_surfaces,
            )

    def test_auto_mode_metadata_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        auto_mode_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_auto_mode_metadata_does_not_declare_active_automation(self) -> None:
        registry = auto_mode_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.source_registries,
                *registry.studio_surface_refs,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for profile in registry.auto_mode_profiles
                    for field in (
                        profile.auto_mode_profile_id,
                        profile.auto_mode_posture,
                        *profile.advisory_inputs,
                        *profile.advisory_outputs,
                        *profile.operator_controls,
                        *profile.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "route_provider",
            "provider_route",
            "execute_provider",
            "auto_select_model",
            "request_human_now",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
