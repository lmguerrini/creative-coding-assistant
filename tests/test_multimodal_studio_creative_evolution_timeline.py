import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    MultimodalCreativeEvolutionTimelineRegistry,
    multimodal_artifact_lineage_registry,
    multimodal_artifact_provenance_registry,
    multimodal_branching_timeline_registry,
    multimodal_creative_evolution_timeline_profile_by_id,
    multimodal_creative_evolution_timeline_profiles_for_branching_timeline_profile,
    multimodal_creative_evolution_timeline_profiles_for_route,
    multimodal_creative_evolution_timeline_profiles_for_surface_kind,
    multimodal_creative_evolution_timeline_registry,
    multimodal_shared_artifact_board_registry,
    multimodal_workspace_history_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "intent_creative_evolution_timeline",
    "artifact_iteration_creative_evolution_timeline",
    "quality_refinement_creative_evolution_timeline",
    "final_synthesis_creative_evolution_timeline",
)
EXPECTED_PROFILE_KINDS = (
    "intent_evolution_timeline",
    "artifact_iteration_evolution_timeline",
    "quality_refinement_evolution_timeline",
    "final_synthesis_evolution_timeline",
)
EXPECTED_SURFACE_KINDS = (
    "intent_evolution",
    "artifact_iteration",
    "quality_refinement",
    "final_synthesis",
)
EXPECTED_SOURCE_REGISTRIES = (
    "multimodal_branching_timeline_registry",
    "multimodal_workspace_history_registry",
    "multimodal_shared_artifact_board_registry",
    "multimodal_artifact_lineage_registry",
    "multimodal_artifact_provenance_registry",
    "nextjs_creative_timeline",
    "nextjs_workflow_explorer",
    "nextjs_workstation_shell",
)
EXPECTED_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_BRANCHING_TIMELINE_REGISTRY",
    "multimodal_studio.MULTIMODAL_WORKSPACE_HISTORY_REGISTRY",
    "multimodal_studio.MULTIMODAL_SHARED_ARTIFACT_BOARD_REGISTRY",
    "multimodal_studio.MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY",
    "multimodal_studio.MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY",
    "clients.nextjs.creative_timeline.buildCreativeTimelineModel",
    "clients.nextjs.creative_timeline.CreativeTimelineEvent",
    "clients.nextjs.creative_timeline.provenanceSourceCount",
    "clients.nextjs.workflow_explorer.WorkflowExplorerStage",
    "clients.nextjs.workstation_shell.WorkstationShell",
)
EXPECTED_EVOLUTION_SURFACES = (
    "creative_evolution_timeline_panel",
    "intent_evolution_surface",
    "artifact_iteration_evolution_surface",
    "quality_refinement_evolution_surface",
    "final_synthesis_evolution_surface",
    "evolution_summary_surface",
    "creative_evolution_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "profile_id",
    "profile_name",
    "evolution_profile_kind",
    "evolution_surface_kind",
    "source_branching_timeline_profile_ids",
    "source_workspace_history_profile_ids",
    "source_shared_artifact_board_profile_ids",
    "source_artifact_lineage_profile_ids",
    "source_artifact_provenance_profile_ids",
    "evolution_context_fields",
    "source_reference_ids",
    "route_applicability",
    "creative_evolution_surfaces",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "evolution_generation_implemented",
    "timeline_reconstruction_implemented",
    "branch_creation_implemented",
    "artifact_mutation_implemented",
    "generated_output_mutation_implemented",
    "quality_score_mutation_implemented",
    "provenance_recording_implemented",
    "runtime_event_replay_implemented",
    "workflow_control_implemented",
    "human_input_request_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "networking_implemented",
    "rendering_execution_implemented",
    "serialization_version",
    "metadata_only",
}


class MultimodalStudioCreativeEvolutionTimelineTests(unittest.TestCase):
    def test_creative_evolution_registry_covers_expected_sources(self) -> None:
        registry = multimodal_creative_evolution_timeline_registry()

        self.assertEqual(
            registry.role,
            "multimodal_creative_evolution_timeline_registry",
        )
        self.assertEqual(
            registry.serialization_version,
            "multimodal_creative_evolution_timeline_registry.v1",
        )
        self.assertEqual(registry.profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.evolution_profile_kinds, EXPECTED_PROFILE_KINDS)
        self.assertEqual(registry.evolution_surface_kinds, EXPECTED_SURFACE_KINDS)
        self.assertEqual(
            registry.branching_timeline_profile_ids,
            multimodal_branching_timeline_registry().profile_ids,
        )
        self.assertEqual(
            registry.workspace_history_profile_ids,
            multimodal_workspace_history_registry().profile_ids,
        )
        self.assertEqual(
            registry.shared_artifact_board_profile_ids,
            multimodal_shared_artifact_board_registry().profile_ids,
        )
        self.assertEqual(
            registry.artifact_lineage_profile_ids,
            multimodal_artifact_lineage_registry().profile_ids,
        )
        self.assertEqual(
            registry.artifact_provenance_profile_ids,
            multimodal_artifact_provenance_registry().profile_ids,
        )
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertEqual(registry.source_reference_ids, EXPECTED_SOURCE_REFERENCES)
        self.assertEqual(
            registry.creative_evolution_surface_refs,
            EXPECTED_EVOLUTION_SURFACES,
        )
        self.assertIn(
            "does not generate creative evolution",
            registry.authority_boundary,
        )
        self.assertIn("reconstruct timelines", registry.authority_boundary)
        self.assertIn("evolution_generation", registry.blocked_runtime_behaviors)
        self.assertIn("quality_score_mutation", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.evolution_generation_implemented)
        self.assertFalse(registry.timeline_reconstruction_implemented)
        self.assertFalse(registry.branch_creation_implemented)
        self.assertFalse(registry.artifact_mutation_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.quality_score_mutation_implemented)
        self.assertFalse(registry.provenance_recording_implemented)
        self.assertFalse(registry.runtime_event_replay_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.networking_implemented)
        self.assertFalse(registry.rendering_execution_implemented)

    def test_creative_evolution_profiles_are_passive_and_source_aligned(
        self,
    ) -> None:
        registry = multimodal_creative_evolution_timeline_registry()
        known_routes = set(registry.route_names)
        known_branching = set(registry.branching_timeline_profile_ids)
        known_history = set(registry.workspace_history_profile_ids)
        known_boards = set(registry.shared_artifact_board_profile_ids)
        known_lineage = set(registry.artifact_lineage_profile_ids)
        known_provenance = set(registry.artifact_provenance_profile_ids)
        known_surfaces = set(registry.creative_evolution_surface_refs)
        known_source_references = set(registry.source_reference_ids)
        covered_source_references: set[str] = set()

        for profile in registry.creative_evolution_timeline_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "multimodal_creative_evolution_timeline_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_branching_timeline_profile_ids).issubset(
                    known_branching
                )
            )
            self.assertTrue(
                set(profile.source_workspace_history_profile_ids).issubset(
                    known_history
                )
            )
            self.assertTrue(
                set(profile.source_shared_artifact_board_profile_ids).issubset(
                    known_boards
                )
            )
            self.assertTrue(
                set(profile.source_artifact_lineage_profile_ids).issubset(
                    known_lineage
                )
            )
            self.assertTrue(
                set(profile.source_artifact_provenance_profile_ids).issubset(
                    known_provenance
                )
            )
            self.assertTrue(
                set(profile.creative_evolution_surfaces).issubset(known_surfaces)
            )
            self.assertTrue(
                set(profile.source_reference_ids).issubset(known_source_references)
            )
            covered_source_references.update(profile.source_reference_ids)
            self.assertIn("evolution_generation", profile.blocked_runtime_behaviors)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.evolution_generation_implemented)
            self.assertFalse(profile.timeline_reconstruction_implemented)
            self.assertFalse(profile.branch_creation_implemented)
            self.assertFalse(profile.artifact_mutation_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.quality_score_mutation_implemented)
            self.assertFalse(profile.provenance_recording_implemented)
            self.assertFalse(profile.runtime_event_replay_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.networking_implemented)
            self.assertFalse(profile.rendering_execution_implemented)

        self.assertEqual(covered_source_references, known_source_references)

    def test_creative_evolution_lookup_helpers_are_stable(self) -> None:
        profile = multimodal_creative_evolution_timeline_profile_by_id(
            "intent_creative_evolution_timeline"
        )
        missing_profile = multimodal_creative_evolution_timeline_profile_by_id(
            "missing_profile"
        )
        intent_profiles = (
            multimodal_creative_evolution_timeline_profiles_for_surface_kind(
                "intent_evolution"
            )
        )
        route_profiles = multimodal_creative_evolution_timeline_profiles_for_route(
            RouteName.PREVIEW
        )
        workflow_branch_profiles = (
            multimodal_creative_evolution_timeline_profiles_for_branching_timeline_profile(
                "workflow_branching_timeline"
            )
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.evolution_profile_kind, "intent_evolution_timeline")
        self.assertIn("creative_intelligence", profile.evolution_context_fields)
        self.assertIn("no_evolution_generation_notice", profile.advisory_outputs)
        self.assertEqual(
            tuple(item.profile_id for item in intent_profiles),
            ("intent_creative_evolution_timeline",),
        )
        self.assertEqual(
            tuple(item.profile_id for item in route_profiles),
            (
                "intent_creative_evolution_timeline",
                "artifact_iteration_creative_evolution_timeline",
            ),
        )
        self.assertEqual(
            tuple(item.profile_id for item in workflow_branch_profiles),
            (
                "intent_creative_evolution_timeline",
                "artifact_iteration_creative_evolution_timeline",
                "final_synthesis_creative_evolution_timeline",
            ),
        )

    def test_registry_rejects_mismatched_creative_evolution_metadata(
        self,
    ) -> None:
        registry = multimodal_creative_evolution_timeline_registry()
        first_profile = registry.creative_evolution_timeline_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Creative Evolution Timeline"}
        )
        unknown_branching_profile = first_profile.model_copy(
            update={
                "source_branching_timeline_profile_ids": (
                    "unknown_branching_timeline",
                )
            }
        )

        duplicate_kwargs = _registry_kwargs(registry)
        duplicate_kwargs["creative_evolution_timeline_profiles"] = (
            first_profile,
            duplicate_profile,
        ) + registry.creative_evolution_timeline_profiles[2:]
        with self.assertRaisesRegex(ValueError, "profile_ids must be unique"):
            MultimodalCreativeEvolutionTimelineRegistry(**duplicate_kwargs)

        unknown_branching_kwargs = _registry_kwargs(registry)
        unknown_branching_kwargs["creative_evolution_timeline_profiles"] = (
            unknown_branching_profile,
        ) + registry.creative_evolution_timeline_profiles[1:]
        with self.assertRaisesRegex(
            ValueError,
            "source_branching_timeline_profile_ids",
        ):
            MultimodalCreativeEvolutionTimelineRegistry(**unknown_branching_kwargs)

    def test_creative_evolution_metadata_does_not_change_provider_factory(
        self,
    ) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        multimodal_creative_evolution_timeline_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")


def _registry_kwargs(
    registry: MultimodalCreativeEvolutionTimelineRegistry,
) -> dict[str, object]:
    return {
        "creative_evolution_timeline_profiles": (
            registry.creative_evolution_timeline_profiles
        ),
        "profile_ids": registry.profile_ids,
        "evolution_profile_kinds": registry.evolution_profile_kinds,
        "evolution_surface_kinds": registry.evolution_surface_kinds,
        "branching_timeline_profile_ids": registry.branching_timeline_profile_ids,
        "workspace_history_profile_ids": registry.workspace_history_profile_ids,
        "shared_artifact_board_profile_ids": (
            registry.shared_artifact_board_profile_ids
        ),
        "artifact_lineage_profile_ids": registry.artifact_lineage_profile_ids,
        "artifact_provenance_profile_ids": (
            registry.artifact_provenance_profile_ids
        ),
        "route_names": registry.route_names,
        "profile_count": registry.profile_count,
        "source_registries": registry.source_registries,
        "source_reference_ids": registry.source_reference_ids,
        "creative_evolution_surface_refs": (
            registry.creative_evolution_surface_refs
        ),
        "observability_surfaces": registry.observability_surfaces,
    }


if __name__ == "__main__":
    unittest.main()
