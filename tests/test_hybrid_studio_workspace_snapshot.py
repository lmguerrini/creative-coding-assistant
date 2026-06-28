import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    WorkspaceSnapshotRegistry,
    agent_conversation_view_registry,
    agent_workspace_registry,
    execution_simulator_registry,
    hitl_decision_registry,
    local_cloud_comparison_registry,
    quality_profile_registry,
    workspace_snapshot_profile_by_id,
    workspace_snapshot_profiles_for_conversation_view,
    workspace_snapshot_profiles_for_route,
    workspace_snapshot_profiles_for_workspace,
    workspace_snapshot_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "studio_overview_workspace_snapshot",
    "agent_context_workspace_snapshot",
    "execution_context_workspace_snapshot",
    "review_audit_workspace_snapshot",
)
EXPECTED_KINDS = (
    "studio_overview_snapshot",
    "agent_context_snapshot",
    "execution_context_snapshot",
    "review_audit_snapshot",
)
EXPECTED_SOURCE_REGISTRIES = (
    "agent_workspace_registry",
    "agent_conversation_view_registry",
    "local_cloud_comparison_registry",
    "execution_simulator_registry",
    "quality_profile_registry",
    "hitl_decision_registry",
    "studio_mode_registry",
)
EXPECTED_SNAPSHOT_SURFACES = (
    "workspace_snapshot_panel",
    "snapshot_summary_strip",
    "snapshot_context_matrix",
    "conversation_snapshot_panel",
    "execution_snapshot_panel",
    "review_snapshot_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "workspace_snapshot_profile_id",
    "profile_name",
    "snapshot_kind",
    "source_workspace_profile_ids",
    "source_conversation_view_profile_ids",
    "source_comparison_profile_ids",
    "source_execution_simulation_profile_ids",
    "source_quality_profile_ids",
    "source_hitl_decision_profile_ids",
    "route_applicability",
    "snapshot_surfaces",
    "snapshot_context_fields",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "snapshot_capture_implemented",
    "snapshot_persistence_implemented",
    "conversation_recording_implemented",
    "live_workspace_state_read_implemented",
    "agent_invocation_implemented",
    "memory_read_implemented",
    "memory_write_implemented",
    "workspace_state_mutation_implemented",
    "workflow_control_implemented",
    "human_input_request_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "persistent_replay_storage_implemented",
    "serialization_version",
    "metadata_only",
}


class HybridStudioWorkspaceSnapshotTests(unittest.TestCase):
    def test_workspace_snapshot_registry_covers_expected_profiles(self) -> None:
        registry = workspace_snapshot_registry()

        self.assertEqual(registry.role, "workspace_snapshot_registry")
        self.assertEqual(
            registry.serialization_version,
            "workspace_snapshot_registry.v1",
        )
        self.assertEqual(registry.workspace_snapshot_profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.snapshot_kinds, EXPECTED_KINDS)
        self.assertEqual(
            registry.workspace_profile_ids,
            agent_workspace_registry().workspace_profile_ids,
        )
        self.assertEqual(
            registry.conversation_view_profile_ids,
            agent_conversation_view_registry().conversation_view_profile_ids,
        )
        self.assertEqual(
            registry.comparison_profile_ids,
            local_cloud_comparison_registry().comparison_profile_ids,
        )
        self.assertEqual(
            registry.execution_simulation_profile_ids,
            execution_simulator_registry().execution_simulation_profile_ids,
        )
        self.assertEqual(
            registry.quality_profile_ids,
            quality_profile_registry().quality_profile_ids,
        )
        self.assertEqual(
            registry.hitl_decision_profile_ids,
            hitl_decision_registry().hitl_decision_profile_ids,
        )
        self.assertEqual(registry.snapshot_surface_refs, EXPECTED_SNAPSHOT_SURFACES)
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertIn(
            "does not capture live workspace state", registry.authority_boundary
        )
        self.assertIn("snapshot_persistence", registry.blocked_runtime_behaviors)
        self.assertIn("memory_read", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.snapshot_capture_implemented)
        self.assertFalse(registry.snapshot_persistence_implemented)
        self.assertFalse(registry.conversation_recording_implemented)
        self.assertFalse(registry.live_workspace_state_read_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.memory_read_implemented)
        self.assertFalse(registry.memory_write_implemented)
        self.assertFalse(registry.workspace_state_mutation_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_replay_storage_implemented)

    def test_workspace_snapshot_profiles_are_passive_and_source_aligned(self) -> None:
        registry = workspace_snapshot_registry()
        known_routes = set(registry.route_names)
        known_workspaces = set(agent_workspace_registry().workspace_profile_ids)
        known_conversation_views = set(
            agent_conversation_view_registry().conversation_view_profile_ids
        )
        known_comparisons = set(
            local_cloud_comparison_registry().comparison_profile_ids
        )
        known_simulations = set(
            execution_simulator_registry().execution_simulation_profile_ids
        )
        known_quality_profiles = set(quality_profile_registry().quality_profile_ids)
        known_hitl_profiles = set(hitl_decision_registry().hitl_decision_profile_ids)
        known_surfaces = set(registry.snapshot_surface_refs)

        for profile in registry.snapshot_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "workspace_snapshot_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_workspace_profile_ids).issubset(known_workspaces)
            )
            self.assertTrue(
                set(profile.source_conversation_view_profile_ids).issubset(
                    known_conversation_views
                )
            )
            self.assertTrue(
                set(profile.source_comparison_profile_ids).issubset(known_comparisons)
            )
            self.assertTrue(
                set(profile.source_execution_simulation_profile_ids).issubset(
                    known_simulations
                )
            )
            self.assertTrue(
                set(profile.source_quality_profile_ids).issubset(known_quality_profiles)
            )
            self.assertTrue(
                set(profile.source_hitl_decision_profile_ids).issubset(
                    known_hitl_profiles
                )
            )
            self.assertTrue(set(profile.snapshot_surfaces).issubset(known_surfaces))
            self.assertIn("live_workspace_capture", profile.blocked_runtime_behaviors)
            self.assertIn("snapshot_persistence", profile.blocked_runtime_behaviors)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.snapshot_capture_implemented)
            self.assertFalse(profile.snapshot_persistence_implemented)
            self.assertFalse(profile.conversation_recording_implemented)
            self.assertFalse(profile.live_workspace_state_read_implemented)
            self.assertFalse(profile.agent_invocation_implemented)
            self.assertFalse(profile.memory_read_implemented)
            self.assertFalse(profile.memory_write_implemented)
            self.assertFalse(profile.workspace_state_mutation_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_replay_storage_implemented)

    def test_workspace_snapshot_lookup_helpers_are_stable(self) -> None:
        profile = workspace_snapshot_profile_by_id(
            "execution_context_workspace_snapshot"
        )
        missing_profile = workspace_snapshot_profile_by_id("missing_profile")
        preview_profiles = workspace_snapshot_profiles_for_route("preview")
        review_profiles = workspace_snapshot_profiles_for_route(RouteName.REVIEW)
        artifact_workspace_profiles = workspace_snapshot_profiles_for_workspace(
            "artifact_runtime_agent_workspace"
        )
        audit_view_profiles = workspace_snapshot_profiles_for_conversation_view(
            "audit_trail_conversation_view"
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.snapshot_kind, "execution_context_snapshot")
        self.assertIn(
            "hitl_review_simulation_profile",
            profile.source_execution_simulation_profile_ids,
        )
        self.assertIn("no_provider_execution_notice", profile.advisory_outputs)
        self.assertEqual(
            tuple(item.workspace_snapshot_profile_id for item in preview_profiles),
            (
                "studio_overview_workspace_snapshot",
                "execution_context_workspace_snapshot",
            ),
        )
        self.assertEqual(
            tuple(item.workspace_snapshot_profile_id for item in review_profiles),
            (
                "execution_context_workspace_snapshot",
                "review_audit_workspace_snapshot",
            ),
        )
        self.assertEqual(
            tuple(
                item.workspace_snapshot_profile_id
                for item in artifact_workspace_profiles
            ),
            (
                "studio_overview_workspace_snapshot",
                "agent_context_workspace_snapshot",
                "execution_context_workspace_snapshot",
            ),
        )
        self.assertEqual(
            tuple(item.workspace_snapshot_profile_id for item in audit_view_profiles),
            (
                "studio_overview_workspace_snapshot",
                "execution_context_workspace_snapshot",
                "review_audit_workspace_snapshot",
            ),
        )

    def test_registry_rejects_mismatched_sources_or_snapshot_metadata(self) -> None:
        registry = workspace_snapshot_registry()
        first_profile = registry.snapshot_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Workspace Snapshot"}
        )
        unknown_conversation_profile = first_profile.model_copy(
            update={
                "source_conversation_view_profile_ids": ("unknown_conversation_view",)
            }
        )
        unknown_simulation_profile = first_profile.model_copy(
            update={
                "source_execution_simulation_profile_ids": (
                    "unknown_simulation_profile",
                )
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "workspace_snapshot_profile_ids must be unique",
        ):
            WorkspaceSnapshotRegistry(
                snapshot_profiles=(first_profile, duplicate_profile)
                + registry.snapshot_profiles[2:],
                workspace_snapshot_profile_ids=registry.workspace_snapshot_profile_ids,
                snapshot_kinds=registry.snapshot_kinds,
                workspace_profile_ids=registry.workspace_profile_ids,
                conversation_view_profile_ids=registry.conversation_view_profile_ids,
                comparison_profile_ids=registry.comparison_profile_ids,
                execution_simulation_profile_ids=(
                    registry.execution_simulation_profile_ids
                ),
                quality_profile_ids=registry.quality_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                snapshot_surface_refs=registry.snapshot_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(
            ValueError,
            "source_conversation_view_profile_ids",
        ):
            WorkspaceSnapshotRegistry(
                snapshot_profiles=(unknown_conversation_profile,)
                + registry.snapshot_profiles[1:],
                workspace_snapshot_profile_ids=registry.workspace_snapshot_profile_ids,
                snapshot_kinds=registry.snapshot_kinds,
                workspace_profile_ids=registry.workspace_profile_ids,
                conversation_view_profile_ids=registry.conversation_view_profile_ids,
                comparison_profile_ids=registry.comparison_profile_ids,
                execution_simulation_profile_ids=(
                    registry.execution_simulation_profile_ids
                ),
                quality_profile_ids=registry.quality_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                snapshot_surface_refs=registry.snapshot_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(
            ValueError,
            "source_execution_simulation_profile_ids",
        ):
            WorkspaceSnapshotRegistry(
                snapshot_profiles=(unknown_simulation_profile,)
                + registry.snapshot_profiles[1:],
                workspace_snapshot_profile_ids=registry.workspace_snapshot_profile_ids,
                snapshot_kinds=registry.snapshot_kinds,
                workspace_profile_ids=registry.workspace_profile_ids,
                conversation_view_profile_ids=registry.conversation_view_profile_ids,
                comparison_profile_ids=registry.comparison_profile_ids,
                execution_simulation_profile_ids=(
                    registry.execution_simulation_profile_ids
                ),
                quality_profile_ids=registry.quality_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                snapshot_surface_refs=registry.snapshot_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

    def test_workspace_snapshot_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        workspace_snapshot_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")
