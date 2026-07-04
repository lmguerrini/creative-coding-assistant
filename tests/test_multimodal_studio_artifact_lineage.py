import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    MultimodalArtifactLineageRegistry,
    multimodal_artifact_lineage_profile_by_id,
    multimodal_artifact_lineage_profiles_for_artifact_provenance_profile,
    multimodal_artifact_lineage_profiles_for_route,
    multimodal_artifact_lineage_profiles_for_surface_kind,
    multimodal_artifact_lineage_registry,
    multimodal_artifact_provenance_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "dependency_graph_artifact_lineage",
    "timeline_stage_artifact_lineage",
    "source_transition_artifact_lineage",
    "missing_artifact_lineage",
)
EXPECTED_PROFILE_KINDS = (
    "dependency_graph_lineage",
    "timeline_stage_lineage",
    "source_transition_lineage",
    "missing_lineage",
)
EXPECTED_SURFACE_KINDS = (
    "dependency_graph",
    "timeline_stage",
    "source_transition",
    "missing_lineage",
)
EXPECTED_SOURCE_REGISTRIES = (
    "multimodal_artifact_provenance_registry",
    "orchestration_artifact_dependency_graph",
    "nextjs_provenance_engine",
    "nextjs_creative_timeline",
    "nextjs_workflow_explorer",
    "nextjs_workflow_runtime",
    "nextjs_workstation_shell",
)
EXPECTED_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_ARTIFACT_PROVENANCE_REGISTRY",
    "orchestration.artifact_dependency_graph.ArtifactDependencyGraph",
    "orchestration.artifact_dependency_graph.ArtifactDependencyEdge",
    "clients.nextjs.provenance_engine.buildProvenanceEngineModel",
    "clients.nextjs.provenance_engine.ProvenanceSource",
    "clients.nextjs.creative_timeline.buildCreativeTimelineModel",
    "clients.nextjs.creative_timeline.provenanceSourceCount",
    "clients.nextjs.workflow_explorer.WorkflowExplorerStage",
    "clients.nextjs.workflow_runtime.WorkflowRuntimeTraceEvent",
    "clients.nextjs.workstation_shell.WorkstationShell",
)
EXPECTED_LINEAGE_SURFACES = (
    "artifact_lineage_panel",
    "dependency_graph_lineage_surface",
    "timeline_stage_lineage_surface",
    "source_transition_lineage_surface",
    "missing_lineage_surface",
    "lineage_summary_surface",
    "artifact_lineage_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "profile_id",
    "profile_name",
    "lineage_profile_kind",
    "lineage_surface_kind",
    "source_artifact_provenance_profile_ids",
    "lineage_context_fields",
    "source_reference_ids",
    "route_applicability",
    "artifact_lineage_surfaces",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "lineage_inference_implemented",
    "timeline_reconstruction_implemented",
    "provenance_recording_implemented",
    "persistent_lineage_storage_implemented",
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


class MultimodalStudioArtifactLineageTests(unittest.TestCase):
    def test_artifact_lineage_registry_covers_expected_sources(self) -> None:
        registry = multimodal_artifact_lineage_registry()

        self.assertEqual(registry.role, "multimodal_artifact_lineage_registry")
        self.assertEqual(
            registry.serialization_version,
            "multimodal_artifact_lineage_registry.v1",
        )
        self.assertEqual(registry.profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.lineage_profile_kinds, EXPECTED_PROFILE_KINDS)
        self.assertEqual(registry.lineage_surface_kinds, EXPECTED_SURFACE_KINDS)
        self.assertEqual(
            registry.artifact_provenance_profile_ids,
            multimodal_artifact_provenance_registry().profile_ids,
        )
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertEqual(registry.source_reference_ids, EXPECTED_SOURCE_REFERENCES)
        self.assertEqual(
            registry.artifact_lineage_surface_refs,
            EXPECTED_LINEAGE_SURFACES,
        )
        self.assertIn("does not infer lineage", registry.authority_boundary)
        self.assertIn("reconstruct timelines", registry.authority_boundary)
        self.assertIn("lineage_inference", registry.blocked_runtime_behaviors)
        self.assertIn(
            "persistent_lineage_storage",
            registry.blocked_runtime_behaviors,
        )
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.lineage_inference_implemented)
        self.assertFalse(registry.timeline_reconstruction_implemented)
        self.assertFalse(registry.provenance_recording_implemented)
        self.assertFalse(registry.persistent_lineage_storage_implemented)
        self.assertFalse(registry.artifact_mutation_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.rendering_execution_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.networking_implemented)

    def test_artifact_lineage_profiles_are_passive_and_source_aligned(self) -> None:
        registry = multimodal_artifact_lineage_registry()
        known_routes = set(registry.route_names)
        known_provenance_profiles = set(registry.artifact_provenance_profile_ids)
        known_surfaces = set(registry.artifact_lineage_surface_refs)
        known_source_references = set(registry.source_reference_ids)
        covered_source_references: set[str] = set()

        for profile in registry.artifact_lineage_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "multimodal_artifact_lineage_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_artifact_provenance_profile_ids).issubset(
                    known_provenance_profiles
                )
            )
            self.assertTrue(
                set(profile.artifact_lineage_surfaces).issubset(known_surfaces)
            )
            self.assertTrue(
                set(profile.source_reference_ids).issubset(known_source_references)
            )
            covered_source_references.update(profile.source_reference_ids)
            self.assertIn("lineage_inference", profile.blocked_runtime_behaviors)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.lineage_inference_implemented)
            self.assertFalse(profile.timeline_reconstruction_implemented)
            self.assertFalse(profile.provenance_recording_implemented)
            self.assertFalse(profile.persistent_lineage_storage_implemented)
            self.assertFalse(profile.artifact_mutation_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.rendering_execution_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.networking_implemented)

        self.assertEqual(covered_source_references, known_source_references)

    def test_artifact_lineage_lookup_helpers_are_stable(self) -> None:
        profile = multimodal_artifact_lineage_profile_by_id(
            "dependency_graph_artifact_lineage"
        )
        missing_profile = multimodal_artifact_lineage_profile_by_id("missing_profile")
        dependency_profiles = multimodal_artifact_lineage_profiles_for_surface_kind(
            "dependency_graph"
        )
        route_profiles = multimodal_artifact_lineage_profiles_for_route(
            RouteName.PREVIEW
        )
        payload_profiles = (
            multimodal_artifact_lineage_profiles_for_artifact_provenance_profile(
                "payload_artifact_provenance"
            )
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.lineage_profile_kind, "dependency_graph_lineage")
        self.assertIn(
            "artifact_dependency_graph.dependency_edges",
            profile.lineage_context_fields,
        )
        self.assertIn("no_lineage_inference_notice", profile.advisory_outputs)
        self.assertEqual(
            tuple(item.profile_id for item in dependency_profiles),
            ("dependency_graph_artifact_lineage",),
        )
        self.assertEqual(
            tuple(item.profile_id for item in route_profiles), EXPECTED_PROFILE_IDS
        )
        self.assertEqual(
            tuple(item.profile_id for item in payload_profiles),
            (
                "dependency_graph_artifact_lineage",
                "source_transition_artifact_lineage",
            ),
        )

    def test_registry_rejects_mismatched_artifact_lineage_metadata(self) -> None:
        registry = multimodal_artifact_lineage_registry()
        first_profile = registry.artifact_lineage_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Artifact Lineage"}
        )
        unknown_provenance_profile = first_profile.model_copy(
            update={
                "source_artifact_provenance_profile_ids": (
                    "unknown_artifact_provenance",
                )
            }
        )

        duplicate_kwargs = _registry_kwargs(registry)
        duplicate_kwargs["artifact_lineage_profiles"] = (
            first_profile,
            duplicate_profile,
        ) + registry.artifact_lineage_profiles[2:]
        with self.assertRaisesRegex(ValueError, "profile_ids must be unique"):
            MultimodalArtifactLineageRegistry(**duplicate_kwargs)

        unknown_provenance_kwargs = _registry_kwargs(registry)
        unknown_provenance_kwargs["artifact_lineage_profiles"] = (
            unknown_provenance_profile,
        ) + registry.artifact_lineage_profiles[1:]
        with self.assertRaisesRegex(
            ValueError,
            "source_artifact_provenance_profile_ids",
        ):
            MultimodalArtifactLineageRegistry(**unknown_provenance_kwargs)

    def test_artifact_lineage_metadata_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        multimodal_artifact_lineage_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")


def _registry_kwargs(
    registry: MultimodalArtifactLineageRegistry,
) -> dict[str, object]:
    return {
        "artifact_lineage_profiles": registry.artifact_lineage_profiles,
        "profile_ids": registry.profile_ids,
        "lineage_profile_kinds": registry.lineage_profile_kinds,
        "lineage_surface_kinds": registry.lineage_surface_kinds,
        "artifact_provenance_profile_ids": (registry.artifact_provenance_profile_ids),
        "route_names": registry.route_names,
        "profile_count": registry.profile_count,
        "source_registries": registry.source_registries,
        "source_reference_ids": registry.source_reference_ids,
        "artifact_lineage_surface_refs": registry.artifact_lineage_surface_refs,
        "observability_surfaces": registry.observability_surfaces,
    }


if __name__ == "__main__":
    unittest.main()
