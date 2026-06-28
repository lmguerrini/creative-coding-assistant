import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    ExecutionReplayRegistry,
    cost_profile_registry,
    execution_replay_profile_by_id,
    execution_replay_profiles_for_execution_simulation,
    execution_replay_profiles_for_route,
    execution_replay_profiles_for_session_replay,
    execution_replay_registry,
    execution_simulator_registry,
    hybrid_execution_registry,
    local_cloud_comparison_registry,
    model_profile_registry,
    provider_selection_registry,
    quality_profile_registry,
    session_replay_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "route_execution_replay_profile",
    "provider_selection_execution_replay_profile",
    "local_cloud_execution_replay_profile",
    "quality_review_execution_replay_profile",
)
EXPECTED_KINDS = (
    "route_execution_replay",
    "provider_selection_replay",
    "local_cloud_execution_replay",
    "quality_review_execution_replay",
)
EXPECTED_SOURCE_REGISTRIES = (
    "session_replay_registry",
    "execution_simulator_registry",
    "hybrid_execution_registry",
    "provider_selection_registry",
    "model_profile_registry",
    "cost_profile_registry",
    "quality_profile_registry",
    "local_cloud_comparison_registry",
)
EXPECTED_REPLAY_SURFACES = (
    "execution_replay_panel",
    "execution_trace_timeline",
    "simulation_replay_panel",
    "provider_selection_replay_panel",
    "cost_quality_replay_panel",
    "replay_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "execution_replay_profile_id",
    "profile_name",
    "execution_replay_kind",
    "source_session_replay_profile_ids",
    "source_execution_simulation_profile_ids",
    "source_execution_profile_ids",
    "source_provider_selection_profile_ids",
    "source_model_profile_ids",
    "source_cost_profile_ids",
    "source_quality_profile_ids",
    "source_comparison_profile_ids",
    "route_applicability",
    "execution_replay_surfaces",
    "replay_context_fields",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "execution_replay_execution_implemented",
    "provider_execution_implemented",
    "local_provider_execution_implemented",
    "cloud_provider_execution_implemented",
    "model_selection_implemented",
    "provider_model_routing_implemented",
    "execution_trace_reconstruction_implemented",
    "replay_persistence_implemented",
    "session_replay_execution_implemented",
    "cost_scoring_implemented",
    "quality_scoring_implemented",
    "quality_evaluation_implemented",
    "workflow_control_implemented",
    "human_input_request_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "persistent_replay_storage_implemented",
    "serialization_version",
    "metadata_only",
}


class HybridStudioExecutionReplayTests(unittest.TestCase):
    def test_execution_replay_registry_covers_expected_profiles(self) -> None:
        registry = execution_replay_registry()

        self.assertEqual(registry.role, "execution_replay_registry")
        self.assertEqual(registry.serialization_version, "execution_replay_registry.v1")
        self.assertEqual(registry.execution_replay_profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.execution_replay_kinds, EXPECTED_KINDS)
        self.assertEqual(
            registry.session_replay_profile_ids,
            session_replay_registry().session_replay_profile_ids,
        )
        self.assertEqual(
            registry.execution_simulation_profile_ids,
            execution_simulator_registry().execution_simulation_profile_ids,
        )
        self.assertEqual(
            registry.execution_profile_ids,
            hybrid_execution_registry().execution_profile_ids,
        )
        self.assertEqual(
            registry.provider_selection_profile_ids,
            provider_selection_registry().provider_selection_profile_ids,
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
            registry.comparison_profile_ids,
            local_cloud_comparison_registry().comparison_profile_ids,
        )
        self.assertEqual(
            registry.execution_replay_surface_refs,
            EXPECTED_REPLAY_SURFACES,
        )
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertIn("does not execute providers", registry.authority_boundary)
        self.assertIn(
            "execution_trace_reconstruction",
            registry.blocked_runtime_behaviors,
        )
        self.assertIn("provider_execution", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.execution_replay_execution_implemented)
        self.assertFalse(registry.provider_execution_implemented)
        self.assertFalse(registry.local_provider_execution_implemented)
        self.assertFalse(registry.cloud_provider_execution_implemented)
        self.assertFalse(registry.model_selection_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.execution_trace_reconstruction_implemented)
        self.assertFalse(registry.replay_persistence_implemented)
        self.assertFalse(registry.session_replay_execution_implemented)
        self.assertFalse(registry.cost_scoring_implemented)
        self.assertFalse(registry.quality_scoring_implemented)
        self.assertFalse(registry.quality_evaluation_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_replay_storage_implemented)

    def test_execution_replay_profiles_are_passive_and_source_aligned(self) -> None:
        registry = execution_replay_registry()
        known_routes = set(registry.route_names)
        known_session_replays = set(
            session_replay_registry().session_replay_profile_ids
        )
        known_simulations = set(
            execution_simulator_registry().execution_simulation_profile_ids
        )
        known_execution_profiles = set(
            hybrid_execution_registry().execution_profile_ids
        )
        known_provider_profiles = set(
            provider_selection_registry().provider_selection_profile_ids
        )
        known_model_profiles = set(model_profile_registry().model_profile_ids)
        known_cost_profiles = set(cost_profile_registry().cost_profile_ids)
        known_quality_profiles = set(quality_profile_registry().quality_profile_ids)
        known_comparisons = set(
            local_cloud_comparison_registry().comparison_profile_ids
        )
        known_surfaces = set(registry.execution_replay_surface_refs)

        for profile in registry.execution_replay_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "execution_replay_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_session_replay_profile_ids).issubset(
                    known_session_replays
                )
            )
            self.assertTrue(
                set(profile.source_execution_simulation_profile_ids).issubset(
                    known_simulations
                )
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
                set(profile.source_model_profile_ids).issubset(known_model_profiles)
            )
            self.assertTrue(
                set(profile.source_cost_profile_ids).issubset(known_cost_profiles)
            )
            self.assertTrue(
                set(profile.source_quality_profile_ids).issubset(known_quality_profiles)
            )
            self.assertTrue(
                set(profile.source_comparison_profile_ids).issubset(known_comparisons)
            )
            self.assertTrue(
                set(profile.execution_replay_surfaces).issubset(known_surfaces)
            )
            self.assertIn("provider_execution", profile.blocked_runtime_behaviors)
            self.assertIn(
                "execution_trace_reconstruction",
                profile.blocked_runtime_behaviors,
            )
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.execution_replay_execution_implemented)
            self.assertFalse(profile.provider_execution_implemented)
            self.assertFalse(profile.local_provider_execution_implemented)
            self.assertFalse(profile.cloud_provider_execution_implemented)
            self.assertFalse(profile.model_selection_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.execution_trace_reconstruction_implemented)
            self.assertFalse(profile.replay_persistence_implemented)
            self.assertFalse(profile.session_replay_execution_implemented)
            self.assertFalse(profile.cost_scoring_implemented)
            self.assertFalse(profile.quality_scoring_implemented)
            self.assertFalse(profile.quality_evaluation_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_replay_storage_implemented)

    def test_execution_replay_lookup_helpers_are_stable(self) -> None:
        profile = execution_replay_profile_by_id("local_cloud_execution_replay_profile")
        missing_profile = execution_replay_profile_by_id("missing_profile")
        preview_profiles = execution_replay_profiles_for_route("preview")
        review_profiles = execution_replay_profiles_for_route(RouteName.REVIEW)
        transition_profiles = execution_replay_profiles_for_session_replay(
            "snapshot_transition_replay_profile"
        )
        local_cloud_simulation_profiles = (
            execution_replay_profiles_for_execution_simulation(
                "local_cloud_comparison_simulation_profile"
            )
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.execution_replay_kind, "local_cloud_execution_replay")
        self.assertIn("no_parallel_provider_execution_notice", profile.advisory_outputs)
        self.assertEqual(
            tuple(item.execution_replay_profile_id for item in preview_profiles),
            (
                "route_execution_replay_profile",
                "local_cloud_execution_replay_profile",
            ),
        )
        self.assertEqual(
            tuple(item.execution_replay_profile_id for item in review_profiles),
            (
                "local_cloud_execution_replay_profile",
                "quality_review_execution_replay_profile",
            ),
        )
        self.assertEqual(
            tuple(item.execution_replay_profile_id for item in transition_profiles),
            (
                "provider_selection_execution_replay_profile",
                "local_cloud_execution_replay_profile",
                "quality_review_execution_replay_profile",
            ),
        )
        self.assertEqual(
            tuple(
                item.execution_replay_profile_id
                for item in local_cloud_simulation_profiles
            ),
            (
                "provider_selection_execution_replay_profile",
                "local_cloud_execution_replay_profile",
                "quality_review_execution_replay_profile",
            ),
        )

    def test_registry_rejects_mismatched_sources_or_replay_metadata(self) -> None:
        registry = execution_replay_registry()
        first_profile = registry.execution_replay_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Execution Replay"}
        )
        unknown_session_profile = first_profile.model_copy(
            update={"source_session_replay_profile_ids": ("unknown_session_replay",)}
        )
        unknown_provider_profile = first_profile.model_copy(
            update={
                "source_provider_selection_profile_ids": ("unknown_provider_selection",)
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "execution_replay_profile_ids must be unique",
        ):
            ExecutionReplayRegistry(
                execution_replay_profiles=(first_profile, duplicate_profile)
                + registry.execution_replay_profiles[2:],
                execution_replay_profile_ids=registry.execution_replay_profile_ids,
                execution_replay_kinds=registry.execution_replay_kinds,
                session_replay_profile_ids=registry.session_replay_profile_ids,
                execution_simulation_profile_ids=(
                    registry.execution_simulation_profile_ids
                ),
                execution_profile_ids=registry.execution_profile_ids,
                provider_selection_profile_ids=registry.provider_selection_profile_ids,
                model_profile_ids=registry.model_profile_ids,
                cost_profile_ids=registry.cost_profile_ids,
                quality_profile_ids=registry.quality_profile_ids,
                comparison_profile_ids=registry.comparison_profile_ids,
                execution_replay_surface_refs=registry.execution_replay_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_session_replay_profile_ids"):
            ExecutionReplayRegistry(
                execution_replay_profiles=(unknown_session_profile,)
                + registry.execution_replay_profiles[1:],
                execution_replay_profile_ids=registry.execution_replay_profile_ids,
                execution_replay_kinds=registry.execution_replay_kinds,
                session_replay_profile_ids=registry.session_replay_profile_ids,
                execution_simulation_profile_ids=(
                    registry.execution_simulation_profile_ids
                ),
                execution_profile_ids=registry.execution_profile_ids,
                provider_selection_profile_ids=registry.provider_selection_profile_ids,
                model_profile_ids=registry.model_profile_ids,
                cost_profile_ids=registry.cost_profile_ids,
                quality_profile_ids=registry.quality_profile_ids,
                comparison_profile_ids=registry.comparison_profile_ids,
                execution_replay_surface_refs=registry.execution_replay_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(
            ValueError,
            "source_provider_selection_profile_ids",
        ):
            ExecutionReplayRegistry(
                execution_replay_profiles=(unknown_provider_profile,)
                + registry.execution_replay_profiles[1:],
                execution_replay_profile_ids=registry.execution_replay_profile_ids,
                execution_replay_kinds=registry.execution_replay_kinds,
                session_replay_profile_ids=registry.session_replay_profile_ids,
                execution_simulation_profile_ids=(
                    registry.execution_simulation_profile_ids
                ),
                execution_profile_ids=registry.execution_profile_ids,
                provider_selection_profile_ids=registry.provider_selection_profile_ids,
                model_profile_ids=registry.model_profile_ids,
                cost_profile_ids=registry.cost_profile_ids,
                quality_profile_ids=registry.quality_profile_ids,
                comparison_profile_ids=registry.comparison_profile_ids,
                execution_replay_surface_refs=registry.execution_replay_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

    def test_execution_replay_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        execution_replay_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")
