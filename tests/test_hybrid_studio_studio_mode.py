import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    StudioModeRegistry,
    auto_mode_registry,
    hybrid_execution_registry,
    studio_mode_profile_by_id,
    studio_mode_profiles_for_route,
    studio_mode_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "studio_mode_inspection_profile",
    "studio_mode_comparison_profile",
    "studio_mode_simulation_profile",
    "studio_mode_operator_review_profile",
)
EXPECTED_POSTURES = ("inspect", "compare", "simulate", "operator_review")
EXPECTED_SOURCE_REGISTRIES = (
    "auto_mode_registry",
    "hybrid_execution_registry",
    "local_model_registry",
    "cloud_model_registry",
    "workstation_engine_contract_registry",
    "hybrid_agentic_workflow_registry",
)
EXPECTED_VISIBLE_SURFACES = (
    "studio_mode_shell",
    "model_catalog_panel",
    "auto_mode_panel",
    "hybrid_execution_panel",
    "comparison_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "studio_mode_profile_id",
    "profile_name",
    "studio_mode_posture",
    "source_auto_mode_profile_ids",
    "source_execution_profile_ids",
    "route_applicability",
    "visible_surface_refs",
    "operator_actions",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "studio_mode_runtime_control_implemented",
    "auto_mode_execution_implemented",
    "hybrid_execution_implemented",
    "workflow_control_implemented",
    "provider_model_routing_implemented",
    "artifact_execution_implemented",
    "human_input_request_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "persistent_replay_storage_implemented",
    "serialization_version",
    "metadata_only",
}


class HybridStudioStudioModeTests(unittest.TestCase):
    def test_studio_mode_registry_covers_expected_profiles(self) -> None:
        registry = studio_mode_registry()

        self.assertEqual(registry.role, "studio_mode_registry")
        self.assertEqual(
            registry.serialization_version,
            "studio_mode_registry.v1",
        )
        self.assertEqual(registry.studio_mode_profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.studio_mode_postures, EXPECTED_POSTURES)
        self.assertEqual(
            registry.auto_mode_profile_ids,
            auto_mode_registry().auto_mode_profile_ids,
        )
        self.assertEqual(
            registry.execution_profile_ids,
            hybrid_execution_registry().execution_profile_ids,
        )
        self.assertEqual(registry.visible_surface_refs, EXPECTED_VISIBLE_SURFACES)
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertIn("does not control workflows", registry.authority_boundary)
        self.assertIn("studio_mode_runtime_control", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.studio_mode_runtime_control_implemented)
        self.assertFalse(registry.auto_mode_execution_implemented)
        self.assertFalse(registry.hybrid_execution_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.artifact_execution_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_replay_storage_implemented)

    def test_studio_mode_profiles_are_passive_and_source_aligned(self) -> None:
        registry = studio_mode_registry()
        known_routes = set(registry.route_names)
        known_auto_profiles = set(auto_mode_registry().auto_mode_profile_ids)
        known_execution_profiles = set(
            hybrid_execution_registry().execution_profile_ids
        )
        known_visible_surfaces = set(registry.visible_surface_refs)

        for profile in registry.studio_mode_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(profile.serialization_version, "studio_mode_profile.v1")
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_auto_mode_profile_ids).issubset(known_auto_profiles)
            )
            self.assertTrue(
                set(profile.source_execution_profile_ids).issubset(
                    known_execution_profiles
                )
            )
            self.assertTrue(
                set(profile.visible_surface_refs).issubset(known_visible_surfaces)
            )
            self.assertTrue(profile.operator_actions)
            self.assertIn("workflow_control", profile.blocked_runtime_behaviors)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.studio_mode_runtime_control_implemented)
            self.assertFalse(profile.auto_mode_execution_implemented)
            self.assertFalse(profile.hybrid_execution_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.artifact_execution_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_replay_storage_implemented)

    def test_studio_mode_lookup_helpers_are_stable(self) -> None:
        profile = studio_mode_profile_by_id("studio_mode_comparison_profile")
        missing_profile = studio_mode_profile_by_id("missing_profile")
        review_profiles = studio_mode_profiles_for_route(RouteName.REVIEW)
        preview_profiles = studio_mode_profiles_for_route("preview")

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.studio_mode_posture, "compare")
        self.assertIn("comparison_panel", profile.visible_surface_refs)
        self.assertEqual(
            tuple(item.studio_mode_profile_id for item in preview_profiles),
            (
                "studio_mode_inspection_profile",
                "studio_mode_operator_review_profile",
            ),
        )
        self.assertEqual(
            tuple(item.studio_mode_profile_id for item in review_profiles),
            (
                "studio_mode_inspection_profile",
                "studio_mode_comparison_profile",
                "studio_mode_simulation_profile",
                "studio_mode_operator_review_profile",
            ),
        )

    def test_registry_rejects_mismatched_sources_or_surface_refs(self) -> None:
        registry = studio_mode_registry()
        first_profile = registry.studio_mode_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Profile"}
        )
        unknown_auto_profile = first_profile.model_copy(
            update={"source_auto_mode_profile_ids": ("unknown_auto_profile",)}
        )
        mismatched_source_profile = first_profile.model_copy(
            update={
                "source_registries": (
                    "auto_mode_registry",
                    "hybrid_execution_registry",
                    "local_model_registry",
                    "cloud_model_registry",
                    "workstation_engine_contract_registry",
                    "unknown_registry",
                )
            }
        )

        with self.assertRaisesRegex(
            ValueError, "studio_mode_profile_ids must be unique"
        ):
            StudioModeRegistry(
                studio_mode_profiles=(first_profile, duplicate_profile)
                + registry.studio_mode_profiles[2:],
                studio_mode_profile_ids=registry.studio_mode_profile_ids,
                studio_mode_postures=registry.studio_mode_postures,
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                execution_profile_ids=registry.execution_profile_ids,
                visible_surface_refs=registry.visible_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "known Auto Mode profiles"):
            StudioModeRegistry(
                studio_mode_profiles=(unknown_auto_profile,)
                + registry.studio_mode_profiles[1:],
                studio_mode_profile_ids=registry.studio_mode_profile_ids,
                studio_mode_postures=registry.studio_mode_postures,
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                execution_profile_ids=registry.execution_profile_ids,
                visible_surface_refs=registry.visible_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_registries must match"):
            StudioModeRegistry(
                studio_mode_profiles=(mismatched_source_profile,)
                + registry.studio_mode_profiles[1:],
                studio_mode_profile_ids=registry.studio_mode_profile_ids,
                studio_mode_postures=registry.studio_mode_postures,
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                execution_profile_ids=registry.execution_profile_ids,
                visible_surface_refs=registry.visible_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

    def test_studio_mode_metadata_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        studio_mode_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_studio_mode_metadata_does_not_declare_runtime_control(self) -> None:
        registry = studio_mode_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.source_registries,
                *registry.visible_surface_refs,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for profile in registry.studio_mode_profiles
                    for field in (
                        profile.studio_mode_profile_id,
                        profile.studio_mode_posture,
                        *profile.operator_actions,
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
            "control_workflow_now",
            "request_human_now",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
