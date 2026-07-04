import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    MultimodalArtifactCollaborationRegistry,
    multimodal_artifact_collaboration_profile_by_id,
    multimodal_artifact_collaboration_profiles_for_route,
    multimodal_artifact_collaboration_profiles_for_surface_kind,
    multimodal_artifact_collaboration_profiles_for_workspace_profile,
    multimodal_artifact_collaboration_registry,
    multimodal_runtime_collaboration_registry,
    multimodal_visual_workspace_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "selection_artifact_collaboration",
    "comparison_artifact_collaboration",
    "inspection_artifact_collaboration",
    "refinement_artifact_collaboration",
)
EXPECTED_PROFILE_KINDS = (
    "artifact_selection_collaboration",
    "artifact_comparison_collaboration",
    "artifact_inspection_collaboration",
    "artifact_refinement_collaboration",
)
EXPECTED_SURFACE_KINDS = ("selection", "comparison", "inspection", "refinement")
EXPECTED_SOURCE_REGISTRIES = (
    "multimodal_visual_workspace_registry",
    "multimodal_runtime_collaboration_registry",
    "nextjs_artifact_comparison",
    "nextjs_artifact_inspector",
    "nextjs_artifact_refinement",
    "nextjs_multi_preview_comparison",
    "nextjs_workstation_shell",
)
EXPECTED_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_VISUAL_WORKSPACE_REGISTRY",
    "multimodal_studio.MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY",
    "clients.nextjs.artifact_comparison.buildArtifactComparisonModel",
    "clients.nextjs.artifact_inspector.buildArtifactDocument",
    "clients.nextjs.artifact_inspector.highlightArtifactDocument",
    "clients.nextjs.artifact_refinement.enrichArtifactRefinementRequest",
    "clients.nextjs.multi_preview_comparison.buildMultiPreviewComparisonModel",
    "clients.nextjs.workstation_shell.handleArtifactRefine",
)
EXPECTED_ARTIFACT_SURFACES = (
    "artifact_collaboration_panel",
    "artifact_selection_surface",
    "artifact_comparison_surface",
    "artifact_inspection_surface",
    "artifact_refinement_surface",
    "artifact_action_feedback_surface",
    "artifact_collaboration_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "profile_id",
    "profile_name",
    "artifact_profile_kind",
    "artifact_surface_kind",
    "source_workspace_profile_ids",
    "source_runtime_collaboration_profile_ids",
    "artifact_context_fields",
    "source_reference_ids",
    "route_applicability",
    "artifact_collaboration_surfaces",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "artifact_mutation_implemented",
    "generated_output_mutation_implemented",
    "persistent_collaboration_storage_implemented",
    "rendering_execution_implemented",
    "workflow_control_implemented",
    "human_input_request_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "networking_implemented",
    "serialization_version",
    "metadata_only",
}


class MultimodalStudioArtifactCollaborationTests(unittest.TestCase):
    def test_artifact_collaboration_registry_covers_expected_sources(self) -> None:
        registry = multimodal_artifact_collaboration_registry()

        self.assertEqual(registry.role, "multimodal_artifact_collaboration_registry")
        self.assertEqual(
            registry.serialization_version,
            "multimodal_artifact_collaboration_registry.v1",
        )
        self.assertEqual(registry.profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.artifact_profile_kinds, EXPECTED_PROFILE_KINDS)
        self.assertEqual(registry.artifact_surface_kinds, EXPECTED_SURFACE_KINDS)
        self.assertEqual(
            registry.visual_workspace_profile_ids,
            multimodal_visual_workspace_registry().profile_ids,
        )
        self.assertEqual(
            registry.runtime_collaboration_profile_ids,
            multimodal_runtime_collaboration_registry().profile_ids,
        )
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertEqual(registry.source_reference_ids, EXPECTED_SOURCE_REFERENCES)
        self.assertEqual(
            registry.artifact_collaboration_surface_refs,
            EXPECTED_ARTIFACT_SURFACES,
        )
        self.assertIn("does not mutate artifacts", registry.authority_boundary)
        self.assertIn("persistent collaboration storage", registry.authority_boundary)
        self.assertIn("artifact_mutation", registry.blocked_runtime_behaviors)
        self.assertIn(
            "persistent_collaboration_storage",
            registry.blocked_runtime_behaviors,
        )
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.artifact_mutation_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_collaboration_storage_implemented)
        self.assertFalse(registry.rendering_execution_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.networking_implemented)

    def test_artifact_collaboration_profiles_are_passive_and_source_aligned(
        self,
    ) -> None:
        registry = multimodal_artifact_collaboration_registry()
        known_routes = set(registry.route_names)
        known_workspace_profiles = set(registry.visual_workspace_profile_ids)
        known_runtime_profiles = set(registry.runtime_collaboration_profile_ids)
        known_surfaces = set(registry.artifact_collaboration_surface_refs)
        known_source_references = set(registry.source_reference_ids)
        covered_source_references: set[str] = set()

        for profile in registry.artifact_collaboration_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "multimodal_artifact_collaboration_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_workspace_profile_ids).issubset(
                    known_workspace_profiles
                )
            )
            self.assertTrue(
                set(profile.source_runtime_collaboration_profile_ids).issubset(
                    known_runtime_profiles
                )
            )
            self.assertTrue(
                set(profile.artifact_collaboration_surfaces).issubset(known_surfaces)
            )
            self.assertTrue(
                set(profile.source_reference_ids).issubset(known_source_references)
            )
            covered_source_references.update(profile.source_reference_ids)
            self.assertIn("artifact_mutation", profile.blocked_runtime_behaviors)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.artifact_mutation_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_collaboration_storage_implemented)
            self.assertFalse(profile.rendering_execution_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.networking_implemented)

        self.assertEqual(covered_source_references, known_source_references)

    def test_artifact_collaboration_lookup_helpers_are_stable(self) -> None:
        profile = multimodal_artifact_collaboration_profile_by_id(
            "comparison_artifact_collaboration"
        )
        missing_profile = multimodal_artifact_collaboration_profile_by_id(
            "missing_profile"
        )
        comparison_profiles = (
            multimodal_artifact_collaboration_profiles_for_surface_kind("comparison")
        )
        route_profiles = multimodal_artifact_collaboration_profiles_for_route(
            RouteName.PREVIEW
        )
        artifact_workspace_profiles = (
            multimodal_artifact_collaboration_profiles_for_workspace_profile(
                "artifact_selection_visual_workspace"
            )
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(
            profile.artifact_profile_kind, "artifact_comparison_collaboration"
        )
        self.assertIn("comparisonRows", profile.artifact_context_fields)
        self.assertIn("no_generated_output_mutation_notice", profile.advisory_outputs)
        self.assertEqual(
            tuple(item.profile_id for item in comparison_profiles),
            ("comparison_artifact_collaboration",),
        )
        self.assertEqual(
            tuple(item.profile_id for item in route_profiles),
            EXPECTED_PROFILE_IDS,
        )
        self.assertEqual(
            tuple(item.profile_id for item in artifact_workspace_profiles),
            (
                "selection_artifact_collaboration",
                "comparison_artifact_collaboration",
                "inspection_artifact_collaboration",
                "refinement_artifact_collaboration",
            ),
        )

    def test_registry_rejects_mismatched_artifact_collaboration_metadata(
        self,
    ) -> None:
        registry = multimodal_artifact_collaboration_registry()
        first_profile = registry.artifact_collaboration_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Artifact Collaboration"}
        )
        unknown_workspace_profile = first_profile.model_copy(
            update={"source_workspace_profile_ids": ("unknown_workspace",)}
        )

        duplicate_kwargs = _registry_kwargs(registry)
        duplicate_kwargs["artifact_collaboration_profiles"] = (
            first_profile,
            duplicate_profile,
        ) + registry.artifact_collaboration_profiles[2:]
        with self.assertRaisesRegex(ValueError, "profile_ids must be unique"):
            MultimodalArtifactCollaborationRegistry(**duplicate_kwargs)

        unknown_workspace_kwargs = _registry_kwargs(registry)
        unknown_workspace_kwargs["artifact_collaboration_profiles"] = (
            unknown_workspace_profile,
        ) + registry.artifact_collaboration_profiles[1:]
        with self.assertRaisesRegex(ValueError, "source_workspace_profile_ids"):
            MultimodalArtifactCollaborationRegistry(**unknown_workspace_kwargs)

    def test_artifact_collaboration_metadata_does_not_change_provider_factory(
        self,
    ) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        multimodal_artifact_collaboration_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")


def _registry_kwargs(
    registry: MultimodalArtifactCollaborationRegistry,
) -> dict[str, object]:
    return {
        "artifact_collaboration_profiles": registry.artifact_collaboration_profiles,
        "profile_ids": registry.profile_ids,
        "artifact_profile_kinds": registry.artifact_profile_kinds,
        "artifact_surface_kinds": registry.artifact_surface_kinds,
        "visual_workspace_profile_ids": registry.visual_workspace_profile_ids,
        "runtime_collaboration_profile_ids": (
            registry.runtime_collaboration_profile_ids
        ),
        "route_names": registry.route_names,
        "profile_count": registry.profile_count,
        "source_registries": registry.source_registries,
        "source_reference_ids": registry.source_reference_ids,
        "artifact_collaboration_surface_refs": (
            registry.artifact_collaboration_surface_refs
        ),
        "observability_surfaces": registry.observability_surfaces,
    }


if __name__ == "__main__":
    unittest.main()
