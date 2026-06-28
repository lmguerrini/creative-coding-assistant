import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    ExecutionSimulatorRegistry,
    auto_mode_registry,
    execution_simulation_profile_by_id,
    execution_simulation_profiles_for_route,
    execution_simulator_registry,
    hitl_decision_registry,
    hybrid_execution_registry,
    provider_selection_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "route_preview_simulation_profile",
    "local_cloud_comparison_simulation_profile",
    "hitl_review_simulation_profile",
    "provider_selection_simulation_profile",
)
EXPECTED_SCOPES = (
    "route_preview",
    "local_cloud_comparison",
    "hitl_review",
    "provider_selection",
)
EXPECTED_SOURCE_REGISTRIES = (
    "provider_selection_registry",
    "hitl_decision_registry",
    "auto_mode_registry",
    "studio_mode_registry",
    "hybrid_execution_registry",
    "workflow_agent_handoff_registry",
)
EXPECTED_SURFACES = (
    "execution_simulator_panel",
    "provider_selection_panel",
    "hitl_decision_panel",
    "hybrid_execution_panel",
    "comparison_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "execution_simulation_profile_id",
    "profile_name",
    "simulation_scope",
    "source_provider_selection_profile_ids",
    "source_hitl_decision_profile_ids",
    "source_auto_mode_profile_ids",
    "source_execution_profile_ids",
    "route_applicability",
    "simulated_inputs",
    "simulated_outputs",
    "simulation_surface_refs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "simulation_runtime_execution_implemented",
    "provider_execution_implemented",
    "artifact_execution_implemented",
    "provider_model_routing_implemented",
    "workflow_transition_execution_implemented",
    "human_input_request_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "persistent_replay_storage_implemented",
    "serialization_version",
    "metadata_only",
}


class HybridStudioExecutionSimulatorTests(unittest.TestCase):
    def test_execution_simulator_registry_covers_expected_profiles(self) -> None:
        registry = execution_simulator_registry()

        self.assertEqual(registry.role, "execution_simulator_registry")
        self.assertEqual(
            registry.serialization_version,
            "execution_simulator_registry.v1",
        )
        self.assertEqual(
            registry.execution_simulation_profile_ids, EXPECTED_PROFILE_IDS
        )
        self.assertEqual(registry.simulation_scopes, EXPECTED_SCOPES)
        self.assertEqual(
            registry.provider_selection_profile_ids,
            provider_selection_registry().provider_selection_profile_ids,
        )
        self.assertEqual(
            registry.hitl_decision_profile_ids,
            hitl_decision_registry().hitl_decision_profile_ids,
        )
        self.assertEqual(
            registry.auto_mode_profile_ids,
            auto_mode_registry().auto_mode_profile_ids,
        )
        self.assertEqual(
            registry.execution_profile_ids,
            hybrid_execution_registry().execution_profile_ids,
        )
        self.assertEqual(registry.simulation_surface_refs, EXPECTED_SURFACES)
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertIn("does not execute providers", registry.authority_boundary)
        self.assertIn(
            "simulation_runtime_execution", registry.blocked_runtime_behaviors
        )
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.simulation_runtime_execution_implemented)
        self.assertFalse(registry.provider_execution_implemented)
        self.assertFalse(registry.artifact_execution_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.workflow_transition_execution_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_replay_storage_implemented)

    def test_simulation_profiles_are_passive_and_source_aligned(self) -> None:
        registry = execution_simulator_registry()
        known_routes = set(registry.route_names)
        known_provider_profiles = set(
            provider_selection_registry().provider_selection_profile_ids
        )
        known_hitl_profiles = set(hitl_decision_registry().hitl_decision_profile_ids)
        known_auto_profiles = set(auto_mode_registry().auto_mode_profile_ids)
        known_execution_profiles = set(
            hybrid_execution_registry().execution_profile_ids
        )
        known_surfaces = set(registry.simulation_surface_refs)

        for profile in registry.simulation_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "execution_simulation_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_provider_selection_profile_ids).issubset(
                    known_provider_profiles
                )
            )
            self.assertTrue(
                set(profile.source_hitl_decision_profile_ids).issubset(
                    known_hitl_profiles
                )
            )
            self.assertTrue(
                set(profile.source_auto_mode_profile_ids).issubset(known_auto_profiles)
            )
            self.assertTrue(
                set(profile.source_execution_profile_ids).issubset(
                    known_execution_profiles
                )
            )
            self.assertTrue(
                set(profile.simulation_surface_refs).issubset(known_surfaces)
            )
            self.assertIn(
                "provider_or_model_routing", profile.blocked_runtime_behaviors
            )
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.simulation_runtime_execution_implemented)
            self.assertFalse(profile.provider_execution_implemented)
            self.assertFalse(profile.artifact_execution_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.workflow_transition_execution_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_replay_storage_implemented)

    def test_execution_simulator_lookup_helpers_are_stable(self) -> None:
        profile = execution_simulation_profile_by_id(
            "provider_selection_simulation_profile"
        )
        missing_profile = execution_simulation_profile_by_id("missing_profile")
        review_profiles = execution_simulation_profiles_for_route(RouteName.REVIEW)
        preview_profiles = execution_simulation_profiles_for_route("preview")

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.simulation_scope, "provider_selection")
        self.assertIn("candidate_selection_plan_metadata", profile.simulated_outputs)
        self.assertEqual(
            tuple(item.execution_simulation_profile_id for item in preview_profiles),
            (
                "route_preview_simulation_profile",
                "provider_selection_simulation_profile",
            ),
        )
        self.assertEqual(
            tuple(item.execution_simulation_profile_id for item in review_profiles),
            EXPECTED_PROFILE_IDS,
        )

    def test_registry_rejects_mismatched_sources_or_profile_refs(self) -> None:
        registry = execution_simulator_registry()
        first_profile = registry.simulation_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Profile"}
        )
        unknown_provider_profile = first_profile.model_copy(
            update={
                "source_provider_selection_profile_ids": ("unknown_provider_profile",)
            }
        )
        mismatched_source_profile = first_profile.model_copy(
            update={
                "source_registries": (
                    "provider_selection_registry",
                    "hitl_decision_registry",
                    "auto_mode_registry",
                    "studio_mode_registry",
                    "hybrid_execution_registry",
                    "unknown_registry",
                )
            }
        )

        with self.assertRaisesRegex(
            ValueError, "execution_simulation_profile_ids must be unique"
        ):
            ExecutionSimulatorRegistry(
                simulation_profiles=(first_profile, duplicate_profile)
                + registry.simulation_profiles[2:],
                execution_simulation_profile_ids=(
                    registry.execution_simulation_profile_ids
                ),
                simulation_scopes=registry.simulation_scopes,
                provider_selection_profile_ids=registry.provider_selection_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                execution_profile_ids=registry.execution_profile_ids,
                simulation_surface_refs=registry.simulation_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "known provider profiles"):
            ExecutionSimulatorRegistry(
                simulation_profiles=(unknown_provider_profile,)
                + registry.simulation_profiles[1:],
                execution_simulation_profile_ids=(
                    registry.execution_simulation_profile_ids
                ),
                simulation_scopes=registry.simulation_scopes,
                provider_selection_profile_ids=registry.provider_selection_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                execution_profile_ids=registry.execution_profile_ids,
                simulation_surface_refs=registry.simulation_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_registries must match"):
            ExecutionSimulatorRegistry(
                simulation_profiles=(mismatched_source_profile,)
                + registry.simulation_profiles[1:],
                execution_simulation_profile_ids=(
                    registry.execution_simulation_profile_ids
                ),
                simulation_scopes=registry.simulation_scopes,
                provider_selection_profile_ids=registry.provider_selection_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                execution_profile_ids=registry.execution_profile_ids,
                simulation_surface_refs=registry.simulation_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

    def test_execution_simulator_metadata_does_not_change_provider_factory(
        self,
    ) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        execution_simulator_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_execution_simulator_metadata_does_not_execute(self) -> None:
        registry = execution_simulator_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.source_registries,
                *registry.simulation_surface_refs,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for profile in registry.simulation_profiles
                    for field in (
                        profile.execution_simulation_profile_id,
                        profile.simulation_scope,
                        *profile.simulated_inputs,
                        *profile.simulated_outputs,
                        *profile.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "route_provider",
            "provider_route",
            "execute_provider",
            "run_workflow_now",
            "request_human_now",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
