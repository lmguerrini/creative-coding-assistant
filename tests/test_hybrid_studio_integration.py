import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    HybridStudioIntegrationRegistry,
    agent_conversation_view_registry,
    agent_workspace_registry,
    auto_mode_registry,
    cloud_model_registry,
    cost_profile_registry,
    execution_replay_registry,
    execution_simulator_registry,
    hitl_decision_registry,
    hybrid_execution_registry,
    hybrid_studio_integration_profile_by_id,
    hybrid_studio_integration_profiles_for_route,
    hybrid_studio_integration_profiles_for_source_registry,
    hybrid_studio_integration_registry,
    local_cloud_comparison_registry,
    local_model_registry,
    model_profile_registry,
    provider_selection_registry,
    quality_profile_registry,
    session_replay_registry,
    studio_mode_registry,
    workspace_snapshot_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "model_execution_studio_integration",
    "agent_workspace_studio_integration",
    "snapshot_replay_studio_integration",
    "operator_review_studio_integration",
)
EXPECTED_KINDS = (
    "model_execution_integration",
    "agent_workspace_integration",
    "snapshot_replay_integration",
    "operator_review_integration",
)
EXPECTED_SOURCE_REGISTRIES = (
    "local_model_registry",
    "cloud_model_registry",
    "hybrid_execution_registry",
    "auto_mode_registry",
    "studio_mode_registry",
    "hitl_decision_registry",
    "provider_selection_registry",
    "execution_simulator_registry",
    "model_profile_registry",
    "cost_profile_registry",
    "quality_profile_registry",
    "local_cloud_comparison_registry",
    "agent_workspace_registry",
    "agent_conversation_view_registry",
    "workspace_snapshot_registry",
    "session_replay_registry",
    "execution_replay_registry",
)
EXPECTED_PROFILE_GROUPS = (
    "local_model_surfaces",
    "cloud_model_surfaces",
    "hybrid_execution_profiles",
    "auto_mode_profiles",
    "studio_mode_profiles",
    "hitl_decision_profiles",
    "provider_selection_profiles",
    "execution_simulation_profiles",
    "model_profiles",
    "cost_profiles",
    "quality_profiles",
    "comparison_profiles",
    "agent_workspace_profiles",
    "conversation_view_profiles",
    "workspace_snapshot_profiles",
    "session_replay_profiles",
    "execution_replay_profiles",
)
EXPECTED_INTEGRATION_SURFACES = (
    "hybrid_studio_shell",
    "model_execution_surface",
    "agent_workspace_surface",
    "snapshot_replay_surface",
    "operator_review_surface",
    "integration_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "integration_profile_id",
    "profile_name",
    "integration_kind",
    "source_registry_names",
    "linked_profile_group_refs",
    "route_applicability",
    "integration_surfaces",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "studio_runtime_activation_implemented",
    "runtime_selection_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "human_input_request_implemented",
    "retry_triggering_implemented",
    "storage_mutation_implemented",
    "generated_output_mutation_implemented",
    "persistent_replay_storage_implemented",
    "serialization_version",
    "metadata_only",
}


class HybridStudioIntegrationTests(unittest.TestCase):
    def test_integration_registry_covers_all_v4_4_sources(self) -> None:
        registry = hybrid_studio_integration_registry()

        self.assertEqual(registry.role, "hybrid_studio_integration_registry")
        self.assertEqual(
            registry.serialization_version,
            "hybrid_studio_integration_registry.v1",
        )
        self.assertEqual(registry.integration_profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.integration_kinds, EXPECTED_KINDS)
        self.assertEqual(registry.local_surface_ids, local_model_registry().surface_ids)
        self.assertEqual(registry.cloud_surface_ids, cloud_model_registry().surface_ids)
        self.assertEqual(
            registry.execution_profile_ids,
            hybrid_execution_registry().execution_profile_ids,
        )
        self.assertEqual(
            registry.auto_mode_profile_ids,
            auto_mode_registry().auto_mode_profile_ids,
        )
        self.assertEqual(
            registry.studio_mode_profile_ids,
            studio_mode_registry().studio_mode_profile_ids,
        )
        self.assertEqual(
            registry.hitl_decision_profile_ids,
            hitl_decision_registry().hitl_decision_profile_ids,
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
            registry.cost_profile_ids, cost_profile_registry().cost_profile_ids
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
            registry.workspace_profile_ids,
            agent_workspace_registry().workspace_profile_ids,
        )
        self.assertEqual(
            registry.conversation_view_profile_ids,
            agent_conversation_view_registry().conversation_view_profile_ids,
        )
        self.assertEqual(
            registry.workspace_snapshot_profile_ids,
            workspace_snapshot_registry().workspace_snapshot_profile_ids,
        )
        self.assertEqual(
            registry.session_replay_profile_ids,
            session_replay_registry().session_replay_profile_ids,
        )
        self.assertEqual(
            registry.execution_replay_profile_ids,
            execution_replay_registry().execution_replay_profile_ids,
        )
        self.assertEqual(registry.profile_group_refs, EXPECTED_PROFILE_GROUPS)
        self.assertEqual(
            registry.integration_surface_refs, EXPECTED_INTEGRATION_SURFACES
        )
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertIn("does not activate Studio runtime", registry.authority_boundary)
        self.assertIn("provider_or_model_routing", registry.blocked_runtime_behaviors)
        self.assertIn("studio_runtime_activation", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.studio_runtime_activation_implemented)
        self.assertFalse(registry.runtime_selection_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.provider_execution_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.storage_mutation_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_replay_storage_implemented)

    def test_integration_profiles_are_passive_and_source_aligned(self) -> None:
        registry = hybrid_studio_integration_registry()
        known_routes = set(registry.route_names)
        known_sources = set(registry.source_registries)
        known_profile_groups = set(registry.profile_group_refs)
        known_surfaces = set(registry.integration_surface_refs)

        for profile in registry.integration_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "hybrid_studio_integration_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(set(profile.source_registry_names).issubset(known_sources))
            self.assertTrue(
                set(profile.linked_profile_group_refs).issubset(known_profile_groups)
            )
            self.assertTrue(set(profile.integration_surfaces).issubset(known_surfaces))
            self.assertIn(
                "studio_runtime_activation", profile.blocked_runtime_behaviors
            )
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.studio_runtime_activation_implemented)
            self.assertFalse(profile.runtime_selection_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.provider_execution_implemented)
            self.assertFalse(profile.agent_invocation_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.storage_mutation_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_replay_storage_implemented)

    def test_integration_lookup_helpers_are_stable(self) -> None:
        profile = hybrid_studio_integration_profile_by_id(
            "snapshot_replay_studio_integration"
        )
        missing_profile = hybrid_studio_integration_profile_by_id("missing_profile")
        preview_profiles = hybrid_studio_integration_profiles_for_route("preview")
        review_profiles = hybrid_studio_integration_profiles_for_route(RouteName.REVIEW)
        execution_replay_profiles = (
            hybrid_studio_integration_profiles_for_source_registry(
                "execution_replay_registry"
            )
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.integration_kind, "snapshot_replay_integration")
        self.assertIn("execution_replay_registry", profile.source_registry_names)
        self.assertIn("no_replay_execution_notice", profile.advisory_outputs)
        self.assertEqual(
            tuple(item.integration_profile_id for item in preview_profiles),
            (
                "model_execution_studio_integration",
                "snapshot_replay_studio_integration",
                "operator_review_studio_integration",
            ),
        )
        self.assertEqual(
            tuple(item.integration_profile_id for item in review_profiles),
            EXPECTED_PROFILE_IDS,
        )
        self.assertEqual(
            tuple(item.integration_profile_id for item in execution_replay_profiles),
            (
                "model_execution_studio_integration",
                "snapshot_replay_studio_integration",
                "operator_review_studio_integration",
            ),
        )

    def test_registry_rejects_mismatched_integration_metadata(self) -> None:
        registry = hybrid_studio_integration_registry()
        first_profile = registry.integration_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Integration"}
        )
        unknown_source_profile = first_profile.model_copy(
            update={"source_registry_names": ("unknown_registry",)}
        )
        unknown_group_profile = first_profile.model_copy(
            update={"linked_profile_group_refs": ("unknown_group",)}
        )

        with self.assertRaisesRegex(
            ValueError,
            "integration_profile_ids must be unique",
        ):
            HybridStudioIntegrationRegistry(
                integration_profiles=(first_profile, duplicate_profile)
                + registry.integration_profiles[2:],
                integration_profile_ids=registry.integration_profile_ids,
                integration_kinds=registry.integration_kinds,
                local_surface_ids=registry.local_surface_ids,
                cloud_surface_ids=registry.cloud_surface_ids,
                execution_profile_ids=registry.execution_profile_ids,
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                studio_mode_profile_ids=registry.studio_mode_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                provider_selection_profile_ids=(
                    registry.provider_selection_profile_ids
                ),
                execution_simulation_profile_ids=(
                    registry.execution_simulation_profile_ids
                ),
                model_profile_ids=registry.model_profile_ids,
                cost_profile_ids=registry.cost_profile_ids,
                quality_profile_ids=registry.quality_profile_ids,
                comparison_profile_ids=registry.comparison_profile_ids,
                workspace_profile_ids=registry.workspace_profile_ids,
                conversation_view_profile_ids=registry.conversation_view_profile_ids,
                workspace_snapshot_profile_ids=registry.workspace_snapshot_profile_ids,
                session_replay_profile_ids=registry.session_replay_profile_ids,
                execution_replay_profile_ids=registry.execution_replay_profile_ids,
                profile_group_refs=registry.profile_group_refs,
                integration_surface_refs=registry.integration_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_registry_names"):
            HybridStudioIntegrationRegistry(
                integration_profiles=(unknown_source_profile,)
                + registry.integration_profiles[1:],
                integration_profile_ids=registry.integration_profile_ids,
                integration_kinds=registry.integration_kinds,
                local_surface_ids=registry.local_surface_ids,
                cloud_surface_ids=registry.cloud_surface_ids,
                execution_profile_ids=registry.execution_profile_ids,
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                studio_mode_profile_ids=registry.studio_mode_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                provider_selection_profile_ids=(
                    registry.provider_selection_profile_ids
                ),
                execution_simulation_profile_ids=(
                    registry.execution_simulation_profile_ids
                ),
                model_profile_ids=registry.model_profile_ids,
                cost_profile_ids=registry.cost_profile_ids,
                quality_profile_ids=registry.quality_profile_ids,
                comparison_profile_ids=registry.comparison_profile_ids,
                workspace_profile_ids=registry.workspace_profile_ids,
                conversation_view_profile_ids=registry.conversation_view_profile_ids,
                workspace_snapshot_profile_ids=registry.workspace_snapshot_profile_ids,
                session_replay_profile_ids=registry.session_replay_profile_ids,
                execution_replay_profile_ids=registry.execution_replay_profile_ids,
                profile_group_refs=registry.profile_group_refs,
                integration_surface_refs=registry.integration_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "linked_profile_group_refs"):
            HybridStudioIntegrationRegistry(
                integration_profiles=(unknown_group_profile,)
                + registry.integration_profiles[1:],
                integration_profile_ids=registry.integration_profile_ids,
                integration_kinds=registry.integration_kinds,
                local_surface_ids=registry.local_surface_ids,
                cloud_surface_ids=registry.cloud_surface_ids,
                execution_profile_ids=registry.execution_profile_ids,
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                studio_mode_profile_ids=registry.studio_mode_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                provider_selection_profile_ids=(
                    registry.provider_selection_profile_ids
                ),
                execution_simulation_profile_ids=(
                    registry.execution_simulation_profile_ids
                ),
                model_profile_ids=registry.model_profile_ids,
                cost_profile_ids=registry.cost_profile_ids,
                quality_profile_ids=registry.quality_profile_ids,
                comparison_profile_ids=registry.comparison_profile_ids,
                workspace_profile_ids=registry.workspace_profile_ids,
                conversation_view_profile_ids=registry.conversation_view_profile_ids,
                workspace_snapshot_profile_ids=registry.workspace_snapshot_profile_ids,
                session_replay_profile_ids=registry.session_replay_profile_ids,
                execution_replay_profile_ids=registry.execution_replay_profile_ids,
                profile_group_refs=registry.profile_group_refs,
                integration_surface_refs=registry.integration_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

    def test_integration_metadata_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        hybrid_studio_integration_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")
