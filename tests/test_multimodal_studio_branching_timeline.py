import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    MultimodalBranchingTimelineRegistry,
    multimodal_artifact_lineage_registry,
    multimodal_branching_timeline_profile_by_id,
    multimodal_branching_timeline_profiles_for_route,
    multimodal_branching_timeline_profiles_for_surface_kind,
    multimodal_branching_timeline_profiles_for_workspace_history_profile,
    multimodal_branching_timeline_registry,
    multimodal_runtime_collaboration_registry,
    multimodal_shared_artifact_board_registry,
    multimodal_workspace_history_registry,
    session_replay_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "workflow_branching_timeline",
    "artifact_variant_branching_timeline",
    "review_retry_branching_timeline",
    "fallback_failure_branching_timeline",
)
EXPECTED_PROFILE_KINDS = (
    "workflow_branch_timeline",
    "artifact_variant_branch_timeline",
    "review_retry_branch_timeline",
    "fallback_failure_branch_timeline",
)
EXPECTED_SURFACE_KINDS = (
    "workflow_branch",
    "artifact_variant",
    "review_retry",
    "fallback_failure",
)
EXPECTED_SOURCE_REGISTRIES = (
    "multimodal_workspace_history_registry",
    "multimodal_artifact_lineage_registry",
    "multimodal_shared_artifact_board_registry",
    "multimodal_runtime_collaboration_registry",
    "session_replay_registry",
    "nextjs_workflow_runtime",
    "nextjs_workflow_timeline",
    "nextjs_workstation_shell",
)
EXPECTED_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_WORKSPACE_HISTORY_REGISTRY",
    "multimodal_studio.MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY",
    "multimodal_studio.MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY",
    "multimodal_studio.MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY",
    "hybrid_studio.SESSION_REPLAY_REGISTRY",
    "clients.nextjs.workflow_runtime.WorkflowRuntimeVisualState",
    "clients.nextjs.workflow_runtime.deriveWorkflowVisualState",
    "clients.nextjs.workflow_timeline.buildWorkflowTimelineModel",
    "clients.nextjs.workflow_timeline.WorkflowTimelineEvent",
    "clients.nextjs.workstation_shell.WorkstationShell",
)
EXPECTED_BRANCHING_SURFACES = (
    "branching_timeline_panel",
    "workflow_branch_surface",
    "artifact_variant_branch_surface",
    "review_retry_branch_surface",
    "fallback_failure_branch_surface",
    "branch_summary_surface",
    "branching_timeline_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "profile_id",
    "profile_name",
    "branching_timeline_kind",
    "branch_surface_kind",
    "source_workspace_history_profile_ids",
    "source_artifact_lineage_profile_ids",
    "source_shared_artifact_board_profile_ids",
    "source_runtime_collaboration_profile_ids",
    "source_session_replay_profile_ids",
    "branch_context_fields",
    "source_reference_ids",
    "route_applicability",
    "branching_timeline_surfaces",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "branch_creation_implemented",
    "branch_routing_execution_implemented",
    "timeline_reconstruction_implemented",
    "runtime_event_replay_implemented",
    "retry_triggering_implemented",
    "refinement_triggering_implemented",
    "workflow_state_mutation_implemented",
    "workspace_state_mutation_implemented",
    "artifact_mutation_implemented",
    "generated_output_mutation_implemented",
    "branch_storage_persistence_implemented",
    "rendering_execution_implemented",
    "human_input_request_implemented",
    "provider_model_routing_implemented",
    "networking_implemented",
    "serialization_version",
    "metadata_only",
}


class MultimodalStudioBranchingTimelineTests(unittest.TestCase):
    def test_branching_timeline_registry_covers_expected_sources(self) -> None:
        registry = multimodal_branching_timeline_registry()

        self.assertEqual(registry.role, "multimodal_branching_timeline_registry")
        self.assertEqual(
            registry.serialization_version,
            "multimodal_branching_timeline_registry.v1",
        )
        self.assertEqual(registry.profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.branching_timeline_kinds, EXPECTED_PROFILE_KINDS)
        self.assertEqual(registry.branch_surface_kinds, EXPECTED_SURFACE_KINDS)
        self.assertEqual(
            registry.workspace_history_profile_ids,
            multimodal_workspace_history_registry().profile_ids,
        )
        self.assertEqual(
            registry.artifact_lineage_profile_ids,
            multimodal_artifact_lineage_registry().profile_ids,
        )
        self.assertEqual(
            registry.shared_artifact_board_profile_ids,
            multimodal_shared_artifact_board_registry().profile_ids,
        )
        self.assertEqual(
            registry.runtime_collaboration_profile_ids,
            multimodal_runtime_collaboration_registry().profile_ids,
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
            registry.branching_timeline_surface_refs,
            EXPECTED_BRANCHING_SURFACES,
        )
        self.assertIn("does not create branches", registry.authority_boundary)
        self.assertIn("execute branch routing", registry.authority_boundary)
        self.assertIn("branch_creation", registry.blocked_runtime_behaviors)
        self.assertIn("workflow_state_mutation", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.branch_creation_implemented)
        self.assertFalse(registry.branch_routing_execution_implemented)
        self.assertFalse(registry.timeline_reconstruction_implemented)
        self.assertFalse(registry.runtime_event_replay_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.refinement_triggering_implemented)
        self.assertFalse(registry.workflow_state_mutation_implemented)
        self.assertFalse(registry.workspace_state_mutation_implemented)
        self.assertFalse(registry.artifact_mutation_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.branch_storage_persistence_implemented)
        self.assertFalse(registry.rendering_execution_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.networking_implemented)

    def test_branching_timeline_profiles_are_passive_and_source_aligned(self) -> None:
        registry = multimodal_branching_timeline_registry()
        known_routes = set(registry.route_names)
        known_history = set(registry.workspace_history_profile_ids)
        known_lineage = set(registry.artifact_lineage_profile_ids)
        known_boards = set(registry.shared_artifact_board_profile_ids)
        known_runtime = set(registry.runtime_collaboration_profile_ids)
        known_replays = set(registry.session_replay_profile_ids)
        known_surfaces = set(registry.branching_timeline_surface_refs)
        known_source_references = set(registry.source_reference_ids)
        covered_source_references: set[str] = set()

        for profile in registry.branching_timeline_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "multimodal_branching_timeline_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_workspace_history_profile_ids).issubset(
                    known_history
                )
            )
            self.assertTrue(
                set(profile.source_artifact_lineage_profile_ids).issubset(
                    known_lineage
                )
            )
            self.assertTrue(
                set(profile.source_shared_artifact_board_profile_ids).issubset(
                    known_boards
                )
            )
            self.assertTrue(
                set(profile.source_runtime_collaboration_profile_ids).issubset(
                    known_runtime
                )
            )
            self.assertTrue(
                set(profile.source_session_replay_profile_ids).issubset(
                    known_replays
                )
            )
            self.assertTrue(
                set(profile.branching_timeline_surfaces).issubset(known_surfaces)
            )
            self.assertTrue(
                set(profile.source_reference_ids).issubset(known_source_references)
            )
            covered_source_references.update(profile.source_reference_ids)
            self.assertIn("branch_creation", profile.blocked_runtime_behaviors)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.branch_creation_implemented)
            self.assertFalse(profile.branch_routing_execution_implemented)
            self.assertFalse(profile.timeline_reconstruction_implemented)
            self.assertFalse(profile.runtime_event_replay_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.refinement_triggering_implemented)
            self.assertFalse(profile.workflow_state_mutation_implemented)
            self.assertFalse(profile.workspace_state_mutation_implemented)
            self.assertFalse(profile.artifact_mutation_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.branch_storage_persistence_implemented)
            self.assertFalse(profile.rendering_execution_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.networking_implemented)

        self.assertEqual(covered_source_references, known_source_references)

    def test_branching_timeline_lookup_helpers_are_stable(self) -> None:
        profile = multimodal_branching_timeline_profile_by_id(
            "workflow_branching_timeline"
        )
        missing_profile = multimodal_branching_timeline_profile_by_id(
            "missing_profile"
        )
        workflow_profiles = multimodal_branching_timeline_profiles_for_surface_kind(
            "workflow_branch"
        )
        route_profiles = multimodal_branching_timeline_profiles_for_route(
            RouteName.PREVIEW
        )
        runtime_history_profiles = (
            multimodal_branching_timeline_profiles_for_workspace_history_profile(
                "runtime_event_workspace_history"
            )
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.branching_timeline_kind, "workflow_branch_timeline")
        self.assertIn("WorkflowRuntimeVisualState", profile.branch_context_fields)
        self.assertIn(
            "no_branch_routing_execution_notice",
            profile.advisory_outputs,
        )
        self.assertEqual(
            tuple(item.profile_id for item in workflow_profiles),
            ("workflow_branching_timeline",),
        )
        self.assertEqual(
            tuple(item.profile_id for item in route_profiles),
            (
                "workflow_branching_timeline",
                "artifact_variant_branching_timeline",
            ),
        )
        self.assertEqual(
            tuple(item.profile_id for item in runtime_history_profiles),
            (
                "workflow_branching_timeline",
                "review_retry_branching_timeline",
                "fallback_failure_branching_timeline",
            ),
        )

    def test_registry_rejects_mismatched_branching_timeline_metadata(self) -> None:
        registry = multimodal_branching_timeline_registry()
        first_profile = registry.branching_timeline_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Branching Timeline"}
        )
        unknown_history_profile = first_profile.model_copy(
            update={
                "source_workspace_history_profile_ids": (
                    "unknown_workspace_history",
                )
            }
        )

        duplicate_kwargs = _registry_kwargs(registry)
        duplicate_kwargs["branching_timeline_profiles"] = (
            first_profile,
            duplicate_profile,
        ) + registry.branching_timeline_profiles[2:]
        with self.assertRaisesRegex(ValueError, "profile_ids must be unique"):
            MultimodalBranchingTimelineRegistry(**duplicate_kwargs)

        unknown_history_kwargs = _registry_kwargs(registry)
        unknown_history_kwargs["branching_timeline_profiles"] = (
            unknown_history_profile,
        ) + registry.branching_timeline_profiles[1:]
        with self.assertRaisesRegex(
            ValueError,
            "source_workspace_history_profile_ids",
        ):
            MultimodalBranchingTimelineRegistry(**unknown_history_kwargs)

    def test_branching_timeline_metadata_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        multimodal_branching_timeline_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")


def _registry_kwargs(
    registry: MultimodalBranchingTimelineRegistry,
) -> dict[str, object]:
    return {
        "branching_timeline_profiles": registry.branching_timeline_profiles,
        "profile_ids": registry.profile_ids,
        "branching_timeline_kinds": registry.branching_timeline_kinds,
        "branch_surface_kinds": registry.branch_surface_kinds,
        "workspace_history_profile_ids": registry.workspace_history_profile_ids,
        "artifact_lineage_profile_ids": registry.artifact_lineage_profile_ids,
        "shared_artifact_board_profile_ids": (
            registry.shared_artifact_board_profile_ids
        ),
        "runtime_collaboration_profile_ids": (
            registry.runtime_collaboration_profile_ids
        ),
        "session_replay_profile_ids": registry.session_replay_profile_ids,
        "route_names": registry.route_names,
        "profile_count": registry.profile_count,
        "source_registries": registry.source_registries,
        "source_reference_ids": registry.source_reference_ids,
        "branching_timeline_surface_refs": (
            registry.branching_timeline_surface_refs
        ),
        "observability_surfaces": registry.observability_surfaces,
    }


if __name__ == "__main__":
    unittest.main()
