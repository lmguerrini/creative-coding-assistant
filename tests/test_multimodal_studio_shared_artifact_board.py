import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    MultimodalSharedArtifactBoardRegistry,
    multimodal_artifact_collaboration_registry,
    multimodal_artifact_lineage_registry,
    multimodal_artifact_provenance_registry,
    multimodal_cross_agent_workspace_registry,
    multimodal_multi_preview_registry,
    multimodal_shared_artifact_board_profile_by_id,
    multimodal_shared_artifact_board_profiles_for_cross_agent_workspace_profile,
    multimodal_shared_artifact_board_profiles_for_route,
    multimodal_shared_artifact_board_profiles_for_surface_kind,
    multimodal_shared_artifact_board_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "selection_shared_artifact_board",
    "comparison_shared_artifact_board",
    "provenance_lineage_shared_artifact_board",
    "handoff_refinement_shared_artifact_board",
)
EXPECTED_PROFILE_KINDS = (
    "artifact_selection_board",
    "comparison_review_board",
    "provenance_lineage_board",
    "handoff_refinement_board",
)
EXPECTED_SURFACE_KINDS = (
    "selection",
    "comparison",
    "provenance_lineage",
    "handoff_refinement",
)
EXPECTED_SOURCE_REGISTRIES = (
    "multimodal_cross_agent_workspace_registry",
    "multimodal_artifact_collaboration_registry",
    "multimodal_multi_preview_registry",
    "multimodal_artifact_provenance_registry",
    "multimodal_artifact_lineage_registry",
    "nextjs_artifact_comparison",
    "nextjs_artifact_inspector",
    "nextjs_workstation_shell",
)
EXPECTED_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_CROSS_AGENT_WORKSPACE_REGISTRY",
    "multimodal_studio.MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY",
    "multimodal_studio.MULTIMODAL_MULTI_PREVIEW_REGISTRY",
    "multimodal_studio.MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY",
    "multimodal_studio.MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY",
    "clients.nextjs.artifact_comparison.buildArtifactComparisonModel",
    "clients.nextjs.artifact_comparison.ArtifactComparisonRow",
    "clients.nextjs.artifact_inspector.buildArtifactDocument",
    "clients.nextjs.artifact_inspector.highlightArtifactDocument",
    "clients.nextjs.workstation_shell.WorkstationShell",
)
EXPECTED_BOARD_SURFACES = (
    "shared_artifact_board_panel",
    "artifact_selection_board_surface",
    "artifact_comparison_board_surface",
    "artifact_provenance_board_surface",
    "artifact_lineage_board_surface",
    "artifact_handoff_board_surface",
    "shared_artifact_board_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "profile_id",
    "profile_name",
    "board_profile_kind",
    "board_surface_kind",
    "source_cross_agent_workspace_profile_ids",
    "source_artifact_collaboration_profile_ids",
    "source_multi_preview_profile_ids",
    "source_artifact_provenance_profile_ids",
    "source_artifact_lineage_profile_ids",
    "board_context_fields",
    "source_reference_ids",
    "route_applicability",
    "shared_artifact_board_surfaces",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "board_state_creation_implemented",
    "collaborative_board_persistence_implemented",
    "artifact_selection_mutation_implemented",
    "artifact_mutation_implemented",
    "generated_output_mutation_implemented",
    "rendering_execution_implemented",
    "agent_invocation_implemented",
    "shared_context_materialization_implemented",
    "workflow_control_implemented",
    "human_input_request_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "networking_implemented",
    "serialization_version",
    "metadata_only",
}


class MultimodalStudioSharedArtifactBoardTests(unittest.TestCase):
    def test_shared_artifact_board_registry_covers_expected_sources(self) -> None:
        registry = multimodal_shared_artifact_board_registry()

        self.assertEqual(registry.role, "multimodal_shared_artifact_board_registry")
        self.assertEqual(
            registry.serialization_version,
            "multimodal_shared_artifact_board_registry.v1",
        )
        self.assertEqual(registry.profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.board_profile_kinds, EXPECTED_PROFILE_KINDS)
        self.assertEqual(registry.board_surface_kinds, EXPECTED_SURFACE_KINDS)
        self.assertEqual(
            registry.cross_agent_workspace_profile_ids,
            multimodal_cross_agent_workspace_registry().profile_ids,
        )
        self.assertEqual(
            registry.artifact_collaboration_profile_ids,
            multimodal_artifact_collaboration_registry().profile_ids,
        )
        self.assertEqual(
            registry.multi_preview_profile_ids,
            multimodal_multi_preview_registry().profile_ids,
        )
        self.assertEqual(
            registry.artifact_provenance_profile_ids,
            multimodal_artifact_provenance_registry().profile_ids,
        )
        self.assertEqual(
            registry.artifact_lineage_profile_ids,
            multimodal_artifact_lineage_registry().profile_ids,
        )
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertEqual(registry.source_reference_ids, EXPECTED_SOURCE_REFERENCES)
        self.assertEqual(
            registry.shared_artifact_board_surface_refs,
            EXPECTED_BOARD_SURFACES,
        )
        self.assertIn("does not create collaborative board state", registry.authority_boundary)
        self.assertIn("change artifact selection", registry.authority_boundary)
        self.assertIn("board_state_creation", registry.blocked_runtime_behaviors)
        self.assertIn(
            "artifact_selection_mutation",
            registry.blocked_runtime_behaviors,
        )
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.board_state_creation_implemented)
        self.assertFalse(registry.collaborative_board_persistence_implemented)
        self.assertFalse(registry.artifact_selection_mutation_implemented)
        self.assertFalse(registry.artifact_mutation_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.rendering_execution_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.shared_context_materialization_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.networking_implemented)

    def test_shared_artifact_board_profiles_are_passive_and_source_aligned(self) -> None:
        registry = multimodal_shared_artifact_board_registry()
        known_routes = set(registry.route_names)
        known_cross_agent_workspaces = set(registry.cross_agent_workspace_profile_ids)
        known_artifact_collaboration = set(registry.artifact_collaboration_profile_ids)
        known_multi_preview = set(registry.multi_preview_profile_ids)
        known_provenance = set(registry.artifact_provenance_profile_ids)
        known_lineage = set(registry.artifact_lineage_profile_ids)
        known_surfaces = set(registry.shared_artifact_board_surface_refs)
        known_source_references = set(registry.source_reference_ids)
        covered_source_references: set[str] = set()

        for profile in registry.shared_artifact_board_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "multimodal_shared_artifact_board_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_cross_agent_workspace_profile_ids).issubset(
                    known_cross_agent_workspaces
                )
            )
            self.assertTrue(
                set(profile.source_artifact_collaboration_profile_ids).issubset(
                    known_artifact_collaboration
                )
            )
            self.assertTrue(
                set(profile.source_multi_preview_profile_ids).issubset(
                    known_multi_preview
                )
            )
            self.assertTrue(
                set(profile.source_artifact_provenance_profile_ids).issubset(
                    known_provenance
                )
            )
            self.assertTrue(
                set(profile.source_artifact_lineage_profile_ids).issubset(
                    known_lineage
                )
            )
            self.assertTrue(
                set(profile.shared_artifact_board_surfaces).issubset(known_surfaces)
            )
            self.assertTrue(
                set(profile.source_reference_ids).issubset(known_source_references)
            )
            covered_source_references.update(profile.source_reference_ids)
            self.assertIn("artifact_mutation", profile.blocked_runtime_behaviors)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.board_state_creation_implemented)
            self.assertFalse(profile.collaborative_board_persistence_implemented)
            self.assertFalse(profile.artifact_selection_mutation_implemented)
            self.assertFalse(profile.artifact_mutation_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.rendering_execution_implemented)
            self.assertFalse(profile.agent_invocation_implemented)
            self.assertFalse(profile.shared_context_materialization_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.networking_implemented)

        self.assertEqual(covered_source_references, known_source_references)

    def test_shared_artifact_board_lookup_helpers_are_stable(self) -> None:
        profile = multimodal_shared_artifact_board_profile_by_id(
            "selection_shared_artifact_board"
        )
        missing_profile = multimodal_shared_artifact_board_profile_by_id(
            "missing_profile"
        )
        selection_profiles = multimodal_shared_artifact_board_profiles_for_surface_kind(
            "selection"
        )
        route_profiles = multimodal_shared_artifact_board_profiles_for_route(
            RouteName.PREVIEW
        )
        artifact_runtime_profiles = (
            multimodal_shared_artifact_board_profiles_for_cross_agent_workspace_profile(
                "artifact_runtime_cross_agent_workspace"
            )
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.board_profile_kind, "artifact_selection_board")
        self.assertIn("activeArtifactId", profile.board_context_fields)
        self.assertIn(
            "no_artifact_selection_mutation_notice",
            profile.advisory_outputs,
        )
        self.assertEqual(
            tuple(item.profile_id for item in selection_profiles),
            ("selection_shared_artifact_board",),
        )
        self.assertEqual(
            tuple(item.profile_id for item in route_profiles),
            (
                "selection_shared_artifact_board",
                "comparison_shared_artifact_board",
                "provenance_lineage_shared_artifact_board",
            ),
        )
        self.assertEqual(
            tuple(item.profile_id for item in artifact_runtime_profiles),
            (
                "selection_shared_artifact_board",
                "comparison_shared_artifact_board",
                "handoff_refinement_shared_artifact_board",
            ),
        )

    def test_registry_rejects_mismatched_shared_artifact_board_metadata(self) -> None:
        registry = multimodal_shared_artifact_board_registry()
        first_profile = registry.shared_artifact_board_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Shared Artifact Board"}
        )
        unknown_workspace_profile = first_profile.model_copy(
            update={
                "source_cross_agent_workspace_profile_ids": (
                    "unknown_cross_agent_workspace",
                )
            }
        )

        duplicate_kwargs = _registry_kwargs(registry)
        duplicate_kwargs["shared_artifact_board_profiles"] = (
            first_profile,
            duplicate_profile,
        ) + registry.shared_artifact_board_profiles[2:]
        with self.assertRaisesRegex(ValueError, "profile_ids must be unique"):
            MultimodalSharedArtifactBoardRegistry(**duplicate_kwargs)

        unknown_workspace_kwargs = _registry_kwargs(registry)
        unknown_workspace_kwargs["shared_artifact_board_profiles"] = (
            unknown_workspace_profile,
        ) + registry.shared_artifact_board_profiles[1:]
        with self.assertRaisesRegex(
            ValueError,
            "source_cross_agent_workspace_profile_ids",
        ):
            MultimodalSharedArtifactBoardRegistry(**unknown_workspace_kwargs)

    def test_shared_artifact_board_metadata_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        multimodal_shared_artifact_board_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")


def _registry_kwargs(
    registry: MultimodalSharedArtifactBoardRegistry,
) -> dict[str, object]:
    return {
        "shared_artifact_board_profiles": registry.shared_artifact_board_profiles,
        "profile_ids": registry.profile_ids,
        "board_profile_kinds": registry.board_profile_kinds,
        "board_surface_kinds": registry.board_surface_kinds,
        "cross_agent_workspace_profile_ids": (
            registry.cross_agent_workspace_profile_ids
        ),
        "artifact_collaboration_profile_ids": (
            registry.artifact_collaboration_profile_ids
        ),
        "multi_preview_profile_ids": registry.multi_preview_profile_ids,
        "artifact_provenance_profile_ids": (
            registry.artifact_provenance_profile_ids
        ),
        "artifact_lineage_profile_ids": registry.artifact_lineage_profile_ids,
        "route_names": registry.route_names,
        "profile_count": registry.profile_count,
        "source_registries": registry.source_registries,
        "source_reference_ids": registry.source_reference_ids,
        "shared_artifact_board_surface_refs": (
            registry.shared_artifact_board_surface_refs
        ),
        "observability_surfaces": registry.observability_surfaces,
    }


if __name__ == "__main__":
    unittest.main()
