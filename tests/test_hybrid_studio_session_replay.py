import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    SessionReplayRegistry,
    agent_conversation_view_registry,
    agent_workspace_registry,
    auto_mode_registry,
    hitl_decision_registry,
    session_replay_profile_by_id,
    session_replay_profiles_for_conversation_view,
    session_replay_profiles_for_route,
    session_replay_profiles_for_workspace_snapshot,
    session_replay_registry,
    studio_mode_registry,
    workspace_snapshot_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "session_overview_replay_profile",
    "conversation_timeline_replay_profile",
    "snapshot_transition_replay_profile",
    "review_decision_replay_profile",
)
EXPECTED_KINDS = (
    "session_overview_replay",
    "conversation_timeline_replay",
    "snapshot_transition_replay",
    "review_decision_replay",
)
EXPECTED_SOURCE_REGISTRIES = (
    "workspace_snapshot_registry",
    "agent_conversation_view_registry",
    "agent_workspace_registry",
    "hitl_decision_registry",
    "studio_mode_registry",
    "auto_mode_registry",
)
EXPECTED_REPLAY_SURFACES = (
    "session_replay_panel",
    "session_timeline_strip",
    "conversation_replay_panel",
    "snapshot_replay_panel",
    "decision_replay_panel",
    "replay_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "session_replay_profile_id",
    "profile_name",
    "session_replay_kind",
    "source_workspace_snapshot_profile_ids",
    "source_conversation_view_profile_ids",
    "source_workspace_profile_ids",
    "source_hitl_decision_profile_ids",
    "source_studio_mode_profile_ids",
    "source_auto_mode_profile_ids",
    "route_applicability",
    "replay_surfaces",
    "replay_context_fields",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "session_replay_execution_implemented",
    "session_recording_implemented",
    "timeline_reconstruction_implemented",
    "replay_persistence_implemented",
    "conversation_persistence_implemented",
    "snapshot_capture_implemented",
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


class HybridStudioSessionReplayTests(unittest.TestCase):
    def test_session_replay_registry_covers_expected_profiles(self) -> None:
        registry = session_replay_registry()

        self.assertEqual(registry.role, "session_replay_registry")
        self.assertEqual(registry.serialization_version, "session_replay_registry.v1")
        self.assertEqual(registry.session_replay_profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.session_replay_kinds, EXPECTED_KINDS)
        self.assertEqual(
            registry.workspace_snapshot_profile_ids,
            workspace_snapshot_registry().workspace_snapshot_profile_ids,
        )
        self.assertEqual(
            registry.conversation_view_profile_ids,
            agent_conversation_view_registry().conversation_view_profile_ids,
        )
        self.assertEqual(
            registry.workspace_profile_ids,
            agent_workspace_registry().workspace_profile_ids,
        )
        self.assertEqual(
            registry.hitl_decision_profile_ids,
            hitl_decision_registry().hitl_decision_profile_ids,
        )
        self.assertEqual(
            registry.studio_mode_profile_ids,
            studio_mode_registry().studio_mode_profile_ids,
        )
        self.assertEqual(
            registry.auto_mode_profile_ids,
            auto_mode_registry().auto_mode_profile_ids,
        )
        self.assertEqual(registry.replay_surface_refs, EXPECTED_REPLAY_SURFACES)
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertIn("does not record sessions", registry.authority_boundary)
        self.assertIn("timeline_reconstruction", registry.blocked_runtime_behaviors)
        self.assertIn("replay_persistence", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.session_replay_execution_implemented)
        self.assertFalse(registry.session_recording_implemented)
        self.assertFalse(registry.timeline_reconstruction_implemented)
        self.assertFalse(registry.replay_persistence_implemented)
        self.assertFalse(registry.conversation_persistence_implemented)
        self.assertFalse(registry.snapshot_capture_implemented)
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

    def test_session_replay_profiles_are_passive_and_source_aligned(self) -> None:
        registry = session_replay_registry()
        known_routes = set(registry.route_names)
        known_snapshots = set(
            workspace_snapshot_registry().workspace_snapshot_profile_ids
        )
        known_conversation_views = set(
            agent_conversation_view_registry().conversation_view_profile_ids
        )
        known_workspaces = set(agent_workspace_registry().workspace_profile_ids)
        known_hitl_profiles = set(hitl_decision_registry().hitl_decision_profile_ids)
        known_studio_profiles = set(studio_mode_registry().studio_mode_profile_ids)
        known_auto_profiles = set(auto_mode_registry().auto_mode_profile_ids)
        known_surfaces = set(registry.replay_surface_refs)

        for profile in registry.session_replay_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "session_replay_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_workspace_snapshot_profile_ids).issubset(
                    known_snapshots
                )
            )
            self.assertTrue(
                set(profile.source_conversation_view_profile_ids).issubset(
                    known_conversation_views
                )
            )
            self.assertTrue(
                set(profile.source_workspace_profile_ids).issubset(known_workspaces)
            )
            self.assertTrue(
                set(profile.source_hitl_decision_profile_ids).issubset(
                    known_hitl_profiles
                )
            )
            self.assertTrue(
                set(profile.source_studio_mode_profile_ids).issubset(
                    known_studio_profiles
                )
            )
            self.assertTrue(
                set(profile.source_auto_mode_profile_ids).issubset(known_auto_profiles)
            )
            self.assertTrue(set(profile.replay_surfaces).issubset(known_surfaces))
            self.assertIn("session_recording", profile.blocked_runtime_behaviors)
            self.assertIn("timeline_reconstruction", profile.blocked_runtime_behaviors)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.session_replay_execution_implemented)
            self.assertFalse(profile.session_recording_implemented)
            self.assertFalse(profile.timeline_reconstruction_implemented)
            self.assertFalse(profile.replay_persistence_implemented)
            self.assertFalse(profile.conversation_persistence_implemented)
            self.assertFalse(profile.snapshot_capture_implemented)
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

    def test_session_replay_lookup_helpers_are_stable(self) -> None:
        profile = session_replay_profile_by_id("snapshot_transition_replay_profile")
        missing_profile = session_replay_profile_by_id("missing_profile")
        preview_profiles = session_replay_profiles_for_route("preview")
        review_profiles = session_replay_profiles_for_route(RouteName.REVIEW)
        review_snapshot_profiles = session_replay_profiles_for_workspace_snapshot(
            "review_audit_workspace_snapshot"
        )
        audit_view_profiles = session_replay_profiles_for_conversation_view(
            "audit_trail_conversation_view"
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.session_replay_kind, "snapshot_transition_replay")
        self.assertIn("no_snapshot_capture_notice", profile.advisory_outputs)
        self.assertEqual(
            tuple(item.session_replay_profile_id for item in preview_profiles),
            (
                "session_overview_replay_profile",
                "snapshot_transition_replay_profile",
            ),
        )
        self.assertEqual(
            tuple(item.session_replay_profile_id for item in review_profiles),
            (
                "conversation_timeline_replay_profile",
                "snapshot_transition_replay_profile",
                "review_decision_replay_profile",
            ),
        )
        self.assertEqual(
            tuple(item.session_replay_profile_id for item in review_snapshot_profiles),
            (
                "conversation_timeline_replay_profile",
                "snapshot_transition_replay_profile",
                "review_decision_replay_profile",
            ),
        )
        self.assertEqual(
            tuple(item.session_replay_profile_id for item in audit_view_profiles),
            (
                "session_overview_replay_profile",
                "snapshot_transition_replay_profile",
                "review_decision_replay_profile",
            ),
        )

    def test_registry_rejects_mismatched_sources_or_replay_metadata(self) -> None:
        registry = session_replay_registry()
        first_profile = registry.session_replay_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Session Replay"}
        )
        unknown_snapshot_profile = first_profile.model_copy(
            update={
                "source_workspace_snapshot_profile_ids": ("unknown_workspace_snapshot",)
            }
        )
        unknown_studio_profile = first_profile.model_copy(
            update={"source_studio_mode_profile_ids": ("unknown_studio_profile",)}
        )

        with self.assertRaisesRegex(
            ValueError,
            "session_replay_profile_ids must be unique",
        ):
            SessionReplayRegistry(
                session_replay_profiles=(first_profile, duplicate_profile)
                + registry.session_replay_profiles[2:],
                session_replay_profile_ids=registry.session_replay_profile_ids,
                session_replay_kinds=registry.session_replay_kinds,
                workspace_snapshot_profile_ids=(
                    registry.workspace_snapshot_profile_ids
                ),
                conversation_view_profile_ids=registry.conversation_view_profile_ids,
                workspace_profile_ids=registry.workspace_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                studio_mode_profile_ids=registry.studio_mode_profile_ids,
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                replay_surface_refs=registry.replay_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(
            ValueError,
            "source_workspace_snapshot_profile_ids",
        ):
            SessionReplayRegistry(
                session_replay_profiles=(unknown_snapshot_profile,)
                + registry.session_replay_profiles[1:],
                session_replay_profile_ids=registry.session_replay_profile_ids,
                session_replay_kinds=registry.session_replay_kinds,
                workspace_snapshot_profile_ids=(
                    registry.workspace_snapshot_profile_ids
                ),
                conversation_view_profile_ids=registry.conversation_view_profile_ids,
                workspace_profile_ids=registry.workspace_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                studio_mode_profile_ids=registry.studio_mode_profile_ids,
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                replay_surface_refs=registry.replay_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_studio_mode_profile_ids"):
            SessionReplayRegistry(
                session_replay_profiles=(unknown_studio_profile,)
                + registry.session_replay_profiles[1:],
                session_replay_profile_ids=registry.session_replay_profile_ids,
                session_replay_kinds=registry.session_replay_kinds,
                workspace_snapshot_profile_ids=(
                    registry.workspace_snapshot_profile_ids
                ),
                conversation_view_profile_ids=registry.conversation_view_profile_ids,
                workspace_profile_ids=registry.workspace_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                studio_mode_profile_ids=registry.studio_mode_profile_ids,
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                replay_surface_refs=registry.replay_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

    def test_session_replay_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        session_replay_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")
