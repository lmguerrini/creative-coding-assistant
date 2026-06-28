import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    HybridExecutionRegistry,
    cloud_model_registry,
    hybrid_execution_profile_by_id,
    hybrid_execution_profiles_for_route,
    hybrid_execution_registry,
    local_model_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "local_first_context_profile",
    "cloud_first_context_profile",
    "side_by_side_comparison_profile",
    "operator_selected_context_profile",
)
EXPECTED_COORDINATION_STRATEGIES = (
    "local_first_advisory",
    "cloud_first_advisory",
    "side_by_side_advisory",
    "operator_selected_advisory",
)
EXPECTED_SOURCE_REGISTRIES = (
    "local_model_registry",
    "cloud_model_registry",
    "generation_provider_contract",
    "agent_routing_registry",
    "workflow_agent_handoff_registry",
    "hybrid_agentic_workflow_registry",
)
EXPECTED_STUDIO_SURFACES = (
    "hybrid_execution_panel",
    "provider_selection_metadata",
    "execution_simulator_metadata",
    "local_cloud_comparison_metadata",
)
REQUIRED_PROFILE_FIELDS = {
    "execution_profile_id",
    "profile_name",
    "coordination_strategy",
    "source_local_surface_ids",
    "source_cloud_surface_ids",
    "route_applicability",
    "decision_inputs",
    "advisory_outputs",
    "studio_surface_refs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "hybrid_execution_implemented",
    "local_provider_execution_implemented",
    "cloud_provider_execution_implemented",
    "provider_model_routing_implemented",
    "parallel_model_execution_implemented",
    "fallback_execution_implemented",
    "automatic_model_selection_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "persistent_replay_storage_implemented",
    "serialization_version",
    "metadata_only",
}


class HybridStudioHybridExecutionTests(unittest.TestCase):
    def test_hybrid_execution_registry_covers_expected_profiles(self) -> None:
        registry = hybrid_execution_registry()

        self.assertEqual(registry.role, "hybrid_execution_registry")
        self.assertEqual(
            registry.serialization_version,
            "hybrid_execution_registry.v1",
        )
        self.assertEqual(registry.execution_profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(
            registry.coordination_strategies,
            EXPECTED_COORDINATION_STRATEGIES,
        )
        self.assertEqual(registry.local_surface_ids, local_model_registry().surface_ids)
        self.assertEqual(registry.cloud_surface_ids, cloud_model_registry().surface_ids)
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertEqual(registry.studio_surface_refs, EXPECTED_STUDIO_SURFACES)
        self.assertIn("does not execute local or cloud", registry.authority_boundary)
        self.assertIn("hybrid_execution", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.hybrid_execution_implemented)
        self.assertFalse(registry.local_provider_execution_implemented)
        self.assertFalse(registry.cloud_provider_execution_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.parallel_model_execution_implemented)
        self.assertFalse(registry.fallback_execution_implemented)
        self.assertFalse(registry.automatic_model_selection_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_replay_storage_implemented)

    def test_profiles_reference_known_local_and_cloud_surfaces(self) -> None:
        registry = hybrid_execution_registry()
        known_routes = set(registry.route_names)
        known_local_surfaces = set(local_model_registry().surface_ids)
        known_cloud_surfaces = set(cloud_model_registry().surface_ids)
        known_studio_surfaces = set(registry.studio_surface_refs)

        for profile in registry.execution_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "hybrid_execution_profile.v1",
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
                set(profile.studio_surface_refs).issubset(known_studio_surfaces)
            )
            self.assertIn(
                "provider_or_model_routing", profile.blocked_runtime_behaviors
            )
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.hybrid_execution_implemented)
            self.assertFalse(profile.local_provider_execution_implemented)
            self.assertFalse(profile.cloud_provider_execution_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.parallel_model_execution_implemented)
            self.assertFalse(profile.fallback_execution_implemented)
            self.assertFalse(profile.automatic_model_selection_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_replay_storage_implemented)

    def test_hybrid_execution_lookup_helpers_are_stable(self) -> None:
        profile = hybrid_execution_profile_by_id("side_by_side_comparison_profile")
        missing_profile = hybrid_execution_profile_by_id("missing_profile")
        review_profiles = hybrid_execution_profiles_for_route(RouteName.REVIEW)
        preview_profiles = hybrid_execution_profiles_for_route("preview")

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.coordination_strategy, "side_by_side_advisory")
        self.assertIn("ragas_evaluator_model_surface", profile.source_cloud_surface_ids)
        self.assertEqual(
            tuple(item.execution_profile_id for item in preview_profiles),
            (
                "local_first_context_profile",
                "operator_selected_context_profile",
            ),
        )
        self.assertEqual(
            tuple(item.execution_profile_id for item in review_profiles),
            (
                "cloud_first_context_profile",
                "side_by_side_comparison_profile",
                "operator_selected_context_profile",
            ),
        )

    def test_registry_rejects_mismatched_sources_or_surface_refs(self) -> None:
        registry = hybrid_execution_registry()
        first_profile = registry.execution_profiles[0]
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
                    "generation_provider_contract",
                    "agent_routing_registry",
                    "workflow_agent_handoff_registry",
                    "unknown_registry",
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "execution_profile_ids must be unique"):
            HybridExecutionRegistry(
                execution_profiles=(first_profile, duplicate_profile)
                + registry.execution_profiles[2:],
                execution_profile_ids=registry.execution_profile_ids,
                coordination_strategies=registry.coordination_strategies,
                local_surface_ids=registry.local_surface_ids,
                cloud_surface_ids=registry.cloud_surface_ids,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                studio_surface_refs=registry.studio_surface_refs,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "known local models"):
            HybridExecutionRegistry(
                execution_profiles=(unknown_local_profile,)
                + registry.execution_profiles[1:],
                execution_profile_ids=registry.execution_profile_ids,
                coordination_strategies=registry.coordination_strategies,
                local_surface_ids=registry.local_surface_ids,
                cloud_surface_ids=registry.cloud_surface_ids,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                studio_surface_refs=registry.studio_surface_refs,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_registries must match"):
            HybridExecutionRegistry(
                execution_profiles=(mismatched_source_profile,)
                + registry.execution_profiles[1:],
                execution_profile_ids=registry.execution_profile_ids,
                coordination_strategies=registry.coordination_strategies,
                local_surface_ids=registry.local_surface_ids,
                cloud_surface_ids=registry.cloud_surface_ids,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                studio_surface_refs=registry.studio_surface_refs,
                observability_surfaces=registry.observability_surfaces,
            )

    def test_hybrid_execution_metadata_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        hybrid_execution_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_hybrid_execution_metadata_does_not_declare_active_execution(self) -> None:
        registry = hybrid_execution_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.source_registries,
                *registry.studio_surface_refs,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for profile in registry.execution_profiles
                    for field in (
                        profile.execution_profile_id,
                        profile.coordination_strategy,
                        *profile.decision_inputs,
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
            "run_parallel_models",
            "auto_select_model",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
