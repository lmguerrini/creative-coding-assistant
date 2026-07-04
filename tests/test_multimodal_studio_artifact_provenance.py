import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    MultimodalArtifactProvenanceRegistry,
    multimodal_artifact_collaboration_registry,
    multimodal_artifact_provenance_profile_by_id,
    multimodal_artifact_provenance_profiles_for_artifact_collaboration_profile,
    multimodal_artifact_provenance_profiles_for_route,
    multimodal_artifact_provenance_profiles_for_surface_kind,
    multimodal_artifact_provenance_registry,
    multimodal_runtime_collaboration_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "evidence_artifact_provenance",
    "payload_artifact_provenance",
    "evaluation_artifact_provenance",
    "missing_source_artifact_provenance",
)
EXPECTED_PROFILE_KINDS = (
    "source_evidence_provenance",
    "artifact_payload_provenance",
    "evaluation_provenance",
    "missing_source_provenance",
)
EXPECTED_SURFACE_KINDS = (
    "evidence",
    "artifact_payload",
    "evaluation",
    "missing_source",
)
EXPECTED_SOURCE_REGISTRIES = (
    "multimodal_artifact_collaboration_registry",
    "multimodal_runtime_collaboration_registry",
    "nextjs_provenance_engine",
    "nextjs_v3_inspector_panels",
    "nextjs_workstation_state",
    "preview_contracts",
    "workflow_trace_events",
)
EXPECTED_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_ARTIFACT_COLLABORATION_REGISTRY",
    "multimodal_studio.MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY",
    "clients.nextjs.provenance_engine.buildProvenanceEngineModel",
    "clients.nextjs.provenance_engine.ProvenanceSource",
    "clients.nextjs.v3_inspector_panels.buildProvenancePanel",
    "clients.nextjs.workstation_state.buildWorkstationState",
    "preview.contracts.PreviewProvenance",
    "clients.nextjs.workflow_runtime.WorkflowRuntimeTraceEvent",
)
EXPECTED_PROVENANCE_SURFACES = (
    "artifact_provenance_panel",
    "evidence_source_surface",
    "artifact_payload_source_surface",
    "evaluation_source_surface",
    "missing_source_surface",
    "provenance_summary_surface",
    "artifact_provenance_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "profile_id",
    "profile_name",
    "provenance_profile_kind",
    "provenance_surface_kind",
    "source_artifact_collaboration_profile_ids",
    "source_runtime_collaboration_profile_ids",
    "provenance_context_fields",
    "source_reference_ids",
    "route_applicability",
    "artifact_provenance_surfaces",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "provenance_recording_implemented",
    "persistent_provenance_storage_implemented",
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


class MultimodalStudioArtifactProvenanceTests(unittest.TestCase):
    def test_artifact_provenance_registry_covers_expected_sources(self) -> None:
        registry = multimodal_artifact_provenance_registry()

        self.assertEqual(registry.role, "multimodal_artifact_provenance_registry")
        self.assertEqual(
            registry.serialization_version,
            "multimodal_artifact_provenance_registry.v1",
        )
        self.assertEqual(registry.profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.provenance_profile_kinds, EXPECTED_PROFILE_KINDS)
        self.assertEqual(registry.provenance_surface_kinds, EXPECTED_SURFACE_KINDS)
        self.assertEqual(
            registry.artifact_collaboration_profile_ids,
            multimodal_artifact_collaboration_registry().profile_ids,
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
            registry.artifact_provenance_surface_refs,
            EXPECTED_PROVENANCE_SURFACES,
        )
        self.assertIn("does not record provenance", registry.authority_boundary)
        self.assertIn("persist provenance storage", registry.authority_boundary)
        self.assertIn("provenance_recording", registry.blocked_runtime_behaviors)
        self.assertIn(
            "persistent_provenance_storage",
            registry.blocked_runtime_behaviors,
        )
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.provenance_recording_implemented)
        self.assertFalse(registry.persistent_provenance_storage_implemented)
        self.assertFalse(registry.artifact_mutation_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.rendering_execution_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.networking_implemented)

    def test_artifact_provenance_profiles_are_passive_and_source_aligned(self) -> None:
        registry = multimodal_artifact_provenance_registry()
        known_routes = set(registry.route_names)
        known_artifact_profiles = set(registry.artifact_collaboration_profile_ids)
        known_runtime_profiles = set(registry.runtime_collaboration_profile_ids)
        known_surfaces = set(registry.artifact_provenance_surface_refs)
        known_source_references = set(registry.source_reference_ids)
        covered_source_references: set[str] = set()

        for profile in registry.artifact_provenance_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "multimodal_artifact_provenance_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_artifact_collaboration_profile_ids).issubset(
                    known_artifact_profiles
                )
            )
            self.assertTrue(
                set(profile.source_runtime_collaboration_profile_ids).issubset(
                    known_runtime_profiles
                )
            )
            self.assertTrue(
                set(profile.artifact_provenance_surfaces).issubset(known_surfaces)
            )
            self.assertTrue(
                set(profile.source_reference_ids).issubset(known_source_references)
            )
            covered_source_references.update(profile.source_reference_ids)
            self.assertIn("provenance_recording", profile.blocked_runtime_behaviors)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.provenance_recording_implemented)
            self.assertFalse(profile.persistent_provenance_storage_implemented)
            self.assertFalse(profile.artifact_mutation_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.rendering_execution_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.networking_implemented)

        self.assertEqual(covered_source_references, known_source_references)

    def test_artifact_provenance_lookup_helpers_are_stable(self) -> None:
        profile = multimodal_artifact_provenance_profile_by_id(
            "evidence_artifact_provenance"
        )
        missing_profile = multimodal_artifact_provenance_profile_by_id(
            "missing_profile"
        )
        evidence_profiles = multimodal_artifact_provenance_profiles_for_surface_kind(
            "evidence"
        )
        route_profiles = multimodal_artifact_provenance_profiles_for_route(
            RouteName.PREVIEW
        )
        inspection_profiles = (
            multimodal_artifact_provenance_profiles_for_artifact_collaboration_profile(
                "inspection_artifact_collaboration"
            )
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.provenance_profile_kind, "source_evidence_provenance")
        self.assertIn("evidence_sources", profile.provenance_context_fields)
        self.assertIn("no_provenance_recording_notice", profile.advisory_outputs)
        self.assertEqual(
            tuple(item.profile_id for item in evidence_profiles),
            ("evidence_artifact_provenance",),
        )
        self.assertEqual(
            tuple(item.profile_id for item in route_profiles), EXPECTED_PROFILE_IDS
        )
        self.assertEqual(
            tuple(item.profile_id for item in inspection_profiles),
            (
                "evidence_artifact_provenance",
                "payload_artifact_provenance",
                "missing_source_artifact_provenance",
            ),
        )

    def test_registry_rejects_mismatched_artifact_provenance_metadata(self) -> None:
        registry = multimodal_artifact_provenance_registry()
        first_profile = registry.artifact_provenance_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Artifact Provenance"}
        )
        unknown_artifact_profile = first_profile.model_copy(
            update={
                "source_artifact_collaboration_profile_ids": (
                    "unknown_artifact_collaboration",
                )
            }
        )

        duplicate_kwargs = _registry_kwargs(registry)
        duplicate_kwargs["artifact_provenance_profiles"] = (
            first_profile,
            duplicate_profile,
        ) + registry.artifact_provenance_profiles[2:]
        with self.assertRaisesRegex(ValueError, "profile_ids must be unique"):
            MultimodalArtifactProvenanceRegistry(**duplicate_kwargs)

        unknown_artifact_kwargs = _registry_kwargs(registry)
        unknown_artifact_kwargs["artifact_provenance_profiles"] = (
            unknown_artifact_profile,
        ) + registry.artifact_provenance_profiles[1:]
        with self.assertRaisesRegex(
            ValueError,
            "source_artifact_collaboration_profile_ids",
        ):
            MultimodalArtifactProvenanceRegistry(**unknown_artifact_kwargs)

    def test_artifact_provenance_metadata_does_not_change_provider_factory(
        self,
    ) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        multimodal_artifact_provenance_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")


def _registry_kwargs(
    registry: MultimodalArtifactProvenanceRegistry,
) -> dict[str, object]:
    return {
        "artifact_provenance_profiles": registry.artifact_provenance_profiles,
        "profile_ids": registry.profile_ids,
        "provenance_profile_kinds": registry.provenance_profile_kinds,
        "provenance_surface_kinds": registry.provenance_surface_kinds,
        "artifact_collaboration_profile_ids": (
            registry.artifact_collaboration_profile_ids
        ),
        "runtime_collaboration_profile_ids": (
            registry.runtime_collaboration_profile_ids
        ),
        "route_names": registry.route_names,
        "profile_count": registry.profile_count,
        "source_registries": registry.source_registries,
        "source_reference_ids": registry.source_reference_ids,
        "artifact_provenance_surface_refs": (registry.artifact_provenance_surface_refs),
        "observability_surfaces": registry.observability_surfaces,
    }


if __name__ == "__main__":
    unittest.main()
