import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    MultimodalWorkspaceHistoryRegistry,
    multimodal_runtime_collaboration_registry,
    multimodal_shared_artifact_board_registry,
    multimodal_workspace_history_profile_by_id,
    multimodal_workspace_history_profiles_for_route,
    multimodal_workspace_history_profiles_for_surface_kind,
    multimodal_workspace_history_profiles_for_workspace_snapshot_profile,
    multimodal_workspace_history_registry,
    session_replay_registry,
    workspace_snapshot_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "session_record_workspace_history",
    "snapshot_workspace_history",
    "artifact_board_workspace_history",
    "runtime_event_workspace_history",
)
EXPECTED_PROFILE_KINDS = (
    "session_record_history",
    "snapshot_history",
    "artifact_board_history",
    "runtime_event_history",
)
EXPECTED_SURFACE_KINDS = (
    "session_record",
    "snapshot",
    "artifact_board",
    "runtime_event",
)
EXPECTED_SOURCE_REGISTRIES = (
    "multimodal_shared_artifact_board_registry",
    "multimodal_runtime_collaboration_registry",
    "workspace_snapshot_registry",
    "session_replay_registry",
    "nextjs_workspace_persistence",
    "nextjs_creative_timeline",
    "nextjs_workflow_runtime",
    "nextjs_workstation_shell",
)
EXPECTED_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY",
    "multimodal_studio.MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY",
    "hybrid_studio.WORKSPACE_SNAPSHOT_REGISTRY",
    "hybrid_studio.SESSION_REPLAY_REGISTRY",
    "clients.nextjs.workspace_persistence.WorkspaceSessionRecord",
    "clients.nextjs.workspace_persistence.createWorkspaceSessionRecord",
    "clients.nextjs.workspace_persistence.snapshotFromWorkspaceSessionRecord",
    "clients.nextjs.creative_timeline.buildCreativeTimelineModel",
    "clients.nextjs.workflow_runtime.WorkflowRuntimeTraceEvent",
    "clients.nextjs.workstation_shell.WorkstationShell",
)
EXPECTED_HISTORY_SURFACES = (
    "workspace_history_panel",
    "session_record_history_surface",
    "snapshot_history_surface",
    "artifact_board_history_surface",
    "runtime_event_history_surface",
    "history_summary_surface",
    "workspace_history_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "profile_id",
    "profile_name",
    "history_profile_kind",
    "history_surface_kind",
    "source_shared_artifact_board_profile_ids",
    "source_runtime_collaboration_profile_ids",
    "source_workspace_snapshot_profile_ids",
    "source_session_replay_profile_ids",
    "history_context_fields",
    "source_reference_ids",
    "route_applicability",
    "workspace_history_surfaces",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "history_recording_implemented",
    "snapshot_capture_implemented",
    "timeline_reconstruction_implemented",
    "history_persistence_implemented",
    "session_replay_execution_implemented",
    "runtime_event_replay_implemented",
    "workspace_state_mutation_implemented",
    "artifact_mutation_implemented",
    "generated_output_mutation_implemented",
    "rendering_execution_implemented",
    "workflow_control_implemented",
    "human_input_request_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "networking_implemented",
    "serialization_version",
    "metadata_only",
}


class MultimodalStudioWorkspaceHistoryTests(unittest.TestCase):
    def test_workspace_history_registry_covers_expected_sources(self) -> None:
        registry = multimodal_workspace_history_registry()

        self.assertEqual(registry.role, "multimodal_workspace_history_registry")
        self.assertEqual(
            registry.serialization_version,
            "multimodal_workspace_history_registry.v1",
        )
        self.assertEqual(registry.profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.history_profile_kinds, EXPECTED_PROFILE_KINDS)
        self.assertEqual(registry.history_surface_kinds, EXPECTED_SURFACE_KINDS)
        self.assertEqual(
            registry.shared_artifact_board_profile_ids,
            multimodal_shared_artifact_board_registry().profile_ids,
        )
        self.assertEqual(
            registry.runtime_collaboration_profile_ids,
            multimodal_runtime_collaboration_registry().profile_ids,
        )
        self.assertEqual(
            registry.workspace_snapshot_profile_ids,
            workspace_snapshot_registry().workspace_snapshot_profile_ids,
        )
        self.assertEqual(
            registry.session_replay_profile_ids,
            session_replay_registry().session_replay_profile_ids,
        )
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertEqual(registry.source_reference_ids, EXPECTED_SOURCE_REFERENCES)
        self.assertEqual(
            registry.workspace_history_surface_refs,
            EXPECTED_HISTORY_SURFACES,
        )
        self.assertIn("does not record workspace history", registry.authority_boundary)
        self.assertIn("capture snapshots", registry.authority_boundary)
        self.assertIn("history_recording", registry.blocked_runtime_behaviors)
        self.assertIn("timeline_reconstruction", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.history_recording_implemented)
        self.assertFalse(registry.snapshot_capture_implemented)
        self.assertFalse(registry.timeline_reconstruction_implemented)
        self.assertFalse(registry.history_persistence_implemented)
        self.assertFalse(registry.session_replay_execution_implemented)
        self.assertFalse(registry.runtime_event_replay_implemented)
        self.assertFalse(registry.workspace_state_mutation_implemented)
        self.assertFalse(registry.artifact_mutation_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.rendering_execution_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.networking_implemented)

    def test_workspace_history_profiles_are_passive_and_source_aligned(self) -> None:
        registry = multimodal_workspace_history_registry()
        known_routes = set(registry.route_names)
        known_boards = set(registry.shared_artifact_board_profile_ids)
        known_runtime_profiles = set(registry.runtime_collaboration_profile_ids)
        known_snapshots = set(registry.workspace_snapshot_profile_ids)
        known_replays = set(registry.session_replay_profile_ids)
        known_surfaces = set(registry.workspace_history_surface_refs)
        known_source_references = set(registry.source_reference_ids)
        covered_source_references: set[str] = set()

        for profile in registry.workspace_history_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "multimodal_workspace_history_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_shared_artifact_board_profile_ids).issubset(
                    known_boards
                )
            )
            self.assertTrue(
                set(profile.source_runtime_collaboration_profile_ids).issubset(
                    known_runtime_profiles
                )
            )
            self.assertTrue(
                set(profile.source_workspace_snapshot_profile_ids).issubset(
                    known_snapshots
                )
            )
            self.assertTrue(
                set(profile.source_session_replay_profile_ids).issubset(
                    known_replays
                )
            )
            self.assertTrue(
                set(profile.workspace_history_surfaces).issubset(known_surfaces)
            )
            self.assertTrue(
                set(profile.source_reference_ids).issubset(known_source_references)
            )
            covered_source_references.update(profile.source_reference_ids)
            self.assertIn("history_recording", profile.blocked_runtime_behaviors)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.history_recording_implemented)
            self.assertFalse(profile.snapshot_capture_implemented)
            self.assertFalse(profile.timeline_reconstruction_implemented)
            self.assertFalse(profile.history_persistence_implemented)
            self.assertFalse(profile.session_replay_execution_implemented)
            self.assertFalse(profile.runtime_event_replay_implemented)
            self.assertFalse(profile.workspace_state_mutation_implemented)
            self.assertFalse(profile.artifact_mutation_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.rendering_execution_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.networking_implemented)

        self.assertEqual(covered_source_references, known_source_references)

    def test_workspace_history_lookup_helpers_are_stable(self) -> None:
        profile = multimodal_workspace_history_profile_by_id(
            "session_record_workspace_history"
        )
        missing_profile = multimodal_workspace_history_profile_by_id(
            "missing_profile"
        )
        session_profiles = multimodal_workspace_history_profiles_for_surface_kind(
            "session_record"
        )
        route_profiles = multimodal_workspace_history_profiles_for_route(
            RouteName.PREVIEW
        )
        review_snapshot_profiles = (
            multimodal_workspace_history_profiles_for_workspace_snapshot_profile(
                "review_audit_workspace_snapshot"
            )
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.history_profile_kind, "session_record_history")
        self.assertIn("WorkspaceSessionRecord", profile.history_context_fields)
        self.assertIn("no_history_recording_notice", profile.advisory_outputs)
        self.assertEqual(
            tuple(item.profile_id for item in session_profiles),
            ("session_record_workspace_history",),
        )
        self.assertEqual(tuple(item.profile_id for item in route_profiles), EXPECTED_PROFILE_IDS)
        self.assertEqual(
            tuple(item.profile_id for item in review_snapshot_profiles),
            (
                "snapshot_workspace_history",
                "artifact_board_workspace_history",
                "runtime_event_workspace_history",
            ),
        )

    def test_registry_rejects_mismatched_workspace_history_metadata(self) -> None:
        registry = multimodal_workspace_history_registry()
        first_profile = registry.workspace_history_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Workspace History"}
        )
        unknown_snapshot_profile = first_profile.model_copy(
            update={
                "source_workspace_snapshot_profile_ids": (
                    "unknown_workspace_snapshot",
                )
            }
        )

        duplicate_kwargs = _registry_kwargs(registry)
        duplicate_kwargs["workspace_history_profiles"] = (
            first_profile,
            duplicate_profile,
        ) + registry.workspace_history_profiles[2:]
        with self.assertRaisesRegex(ValueError, "profile_ids must be unique"):
            MultimodalWorkspaceHistoryRegistry(**duplicate_kwargs)

        unknown_snapshot_kwargs = _registry_kwargs(registry)
        unknown_snapshot_kwargs["workspace_history_profiles"] = (
            unknown_snapshot_profile,
        ) + registry.workspace_history_profiles[1:]
        with self.assertRaisesRegex(
            ValueError,
            "source_workspace_snapshot_profile_ids",
        ):
            MultimodalWorkspaceHistoryRegistry(**unknown_snapshot_kwargs)

    def test_workspace_history_metadata_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        multimodal_workspace_history_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")


def _registry_kwargs(
    registry: MultimodalWorkspaceHistoryRegistry,
) -> dict[str, object]:
    return {
        "workspace_history_profiles": registry.workspace_history_profiles,
        "profile_ids": registry.profile_ids,
        "history_profile_kinds": registry.history_profile_kinds,
        "history_surface_kinds": registry.history_surface_kinds,
        "shared_artifact_board_profile_ids": (
            registry.shared_artifact_board_profile_ids
        ),
        "runtime_collaboration_profile_ids": (
            registry.runtime_collaboration_profile_ids
        ),
        "workspace_snapshot_profile_ids": registry.workspace_snapshot_profile_ids,
        "session_replay_profile_ids": registry.session_replay_profile_ids,
        "route_names": registry.route_names,
        "profile_count": registry.profile_count,
        "source_registries": registry.source_registries,
        "source_reference_ids": registry.source_reference_ids,
        "workspace_history_surface_refs": registry.workspace_history_surface_refs,
        "observability_surfaces": registry.observability_surfaces,
    }


if __name__ == "__main__":
    unittest.main()
