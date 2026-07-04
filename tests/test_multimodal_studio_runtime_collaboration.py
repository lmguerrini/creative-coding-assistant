import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    MultimodalRuntimeCollaborationRegistry,
    multimodal_runtime_collaboration_profile_by_id,
    multimodal_runtime_collaboration_profiles_for_route,
    multimodal_runtime_collaboration_profiles_for_surface_kind,
    multimodal_runtime_collaboration_profiles_for_visual_workspace_profile,
    multimodal_runtime_collaboration_registry,
    multimodal_visual_workspace_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "trace_runtime_collaboration",
    "console_runtime_collaboration",
    "stream_event_runtime_collaboration",
    "operator_context_runtime_collaboration",
)
EXPECTED_PROFILE_KINDS = (
    "runtime_trace_collaboration",
    "runtime_console_collaboration",
    "stream_event_collaboration",
    "operator_context_collaboration",
)
EXPECTED_SURFACE_KINDS = (
    "trace",
    "console",
    "stream",
    "operator_context",
)
EXPECTED_SOURCE_REGISTRIES = (
    "multimodal_visual_workspace_registry",
    "nextjs_workflow_runtime",
    "nextjs_runtime_console",
    "nextjs_assistant_stream",
    "nextjs_workstation_shell",
    "nextjs_provider_telemetry",
    "nextjs_session_intelligence",
)
EXPECTED_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_VISUAL_WORKSPACE_REGISTRY",
    "clients.nextjs.workflow_runtime.buildWorkflowRuntimeModel",
    "clients.nextjs.runtime_console.buildRuntimeConsoleModel",
    "clients.nextjs.assistant_stream.streamAssistantEvents",
    "clients.nextjs.workstation_shell.applyStreamEventToWorkspace",
    "clients.nextjs.provider_telemetry.buildProviderTelemetryModel",
    "clients.nextjs.session_intelligence.buildSessionIntelligenceModel",
    "clients.nextjs.session_intelligence.readSessionIntelligenceMetadata",
)
EXPECTED_COLLABORATION_SURFACES = (
    "runtime_collaboration_panel",
    "runtime_trace_surface",
    "runtime_console_surface",
    "stream_event_surface",
    "operator_context_surface",
    "runtime_health_surface",
    "runtime_collaboration_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "profile_id",
    "profile_name",
    "collaboration_profile_kind",
    "collaboration_surface_kind",
    "source_visual_workspace_profile_ids",
    "runtime_context_fields",
    "source_reference_ids",
    "route_applicability",
    "runtime_collaboration_surfaces",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "real_time_networking_implemented",
    "external_peer_synchronization_implemented",
    "persistent_collaboration_storage_implemented",
    "runtime_execution_implemented",
    "workflow_control_implemented",
    "human_input_request_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class MultimodalStudioRuntimeCollaborationTests(unittest.TestCase):
    def test_runtime_collaboration_registry_covers_expected_sources(self) -> None:
        registry = multimodal_runtime_collaboration_registry()

        self.assertEqual(registry.role, "multimodal_runtime_collaboration_registry")
        self.assertEqual(
            registry.serialization_version,
            "multimodal_runtime_collaboration_registry.v1",
        )
        self.assertEqual(registry.profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(
            registry.collaboration_profile_kinds,
            EXPECTED_PROFILE_KINDS,
        )
        self.assertEqual(
            registry.collaboration_surface_kinds,
            EXPECTED_SURFACE_KINDS,
        )
        self.assertEqual(
            registry.visual_workspace_profile_ids,
            multimodal_visual_workspace_registry().profile_ids,
        )
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertEqual(registry.source_reference_ids, EXPECTED_SOURCE_REFERENCES)
        self.assertEqual(
            registry.runtime_collaboration_surface_refs,
            EXPECTED_COLLABORATION_SURFACES,
        )
        self.assertIn("does not open real-time networking", registry.authority_boundary)
        self.assertIn("synchronize external peers", registry.authority_boundary)
        self.assertIn("real_time_networking", registry.blocked_runtime_behaviors)
        self.assertIn(
            "external_peer_synchronization",
            registry.blocked_runtime_behaviors,
        )
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.real_time_networking_implemented)
        self.assertFalse(registry.external_peer_synchronization_implemented)
        self.assertFalse(registry.persistent_collaboration_storage_implemented)
        self.assertFalse(registry.runtime_execution_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)

    def test_runtime_collaboration_profiles_are_passive_and_source_aligned(
        self,
    ) -> None:
        registry = multimodal_runtime_collaboration_registry()
        known_routes = set(registry.route_names)
        known_workspace_profiles = set(registry.visual_workspace_profile_ids)
        known_surfaces = set(registry.runtime_collaboration_surface_refs)
        known_source_references = set(registry.source_reference_ids)
        covered_source_references: set[str] = set()

        for profile in registry.runtime_collaboration_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "multimodal_runtime_collaboration_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_visual_workspace_profile_ids).issubset(
                    known_workspace_profiles
                )
            )
            self.assertTrue(
                set(profile.runtime_collaboration_surfaces).issubset(known_surfaces)
            )
            self.assertTrue(
                set(profile.source_reference_ids).issubset(known_source_references)
            )
            covered_source_references.update(profile.source_reference_ids)
            self.assertIn("real_time_networking", profile.blocked_runtime_behaviors)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.real_time_networking_implemented)
            self.assertFalse(profile.external_peer_synchronization_implemented)
            self.assertFalse(profile.persistent_collaboration_storage_implemented)
            self.assertFalse(profile.runtime_execution_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)

        self.assertEqual(covered_source_references, known_source_references)

    def test_runtime_collaboration_lookup_helpers_are_stable(self) -> None:
        profile = multimodal_runtime_collaboration_profile_by_id(
            "stream_event_runtime_collaboration"
        )
        missing_profile = multimodal_runtime_collaboration_profile_by_id(
            "missing_profile"
        )
        stream_profiles = multimodal_runtime_collaboration_profiles_for_surface_kind(
            "stream"
        )
        route_profiles = multimodal_runtime_collaboration_profiles_for_route(
            RouteName.PREVIEW
        )
        inspector_profiles = (
            multimodal_runtime_collaboration_profiles_for_visual_workspace_profile(
                "inspector_visual_workspace"
            )
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(
            profile.collaboration_profile_kind, "stream_event_collaboration"
        )
        self.assertIn("event_type", profile.runtime_context_fields)
        self.assertIn("no_real_time_networking_notice", profile.advisory_outputs)
        self.assertEqual(
            tuple(item.profile_id for item in stream_profiles),
            ("stream_event_runtime_collaboration",),
        )
        self.assertEqual(
            tuple(item.profile_id for item in route_profiles),
            EXPECTED_PROFILE_IDS,
        )
        self.assertEqual(
            tuple(item.profile_id for item in inspector_profiles),
            (
                "trace_runtime_collaboration",
                "console_runtime_collaboration",
                "operator_context_runtime_collaboration",
            ),
        )

    def test_registry_rejects_mismatched_runtime_collaboration_metadata(
        self,
    ) -> None:
        registry = multimodal_runtime_collaboration_registry()
        first_profile = registry.runtime_collaboration_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Runtime Collaboration"}
        )
        unknown_workspace_profile = first_profile.model_copy(
            update={"source_visual_workspace_profile_ids": ("unknown_workspace",)}
        )

        duplicate_kwargs = _registry_kwargs(registry)
        duplicate_kwargs["runtime_collaboration_profiles"] = (
            first_profile,
            duplicate_profile,
        ) + registry.runtime_collaboration_profiles[2:]
        with self.assertRaisesRegex(ValueError, "profile_ids must be unique"):
            MultimodalRuntimeCollaborationRegistry(**duplicate_kwargs)

        unknown_workspace_kwargs = _registry_kwargs(registry)
        unknown_workspace_kwargs["runtime_collaboration_profiles"] = (
            unknown_workspace_profile,
        ) + registry.runtime_collaboration_profiles[1:]
        with self.assertRaisesRegex(
            ValueError,
            "source_visual_workspace_profile_ids",
        ):
            MultimodalRuntimeCollaborationRegistry(**unknown_workspace_kwargs)

    def test_runtime_collaboration_metadata_does_not_change_provider_factory(
        self,
    ) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        multimodal_runtime_collaboration_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")


def _registry_kwargs(
    registry: MultimodalRuntimeCollaborationRegistry,
) -> dict[str, object]:
    return {
        "runtime_collaboration_profiles": registry.runtime_collaboration_profiles,
        "profile_ids": registry.profile_ids,
        "collaboration_profile_kinds": registry.collaboration_profile_kinds,
        "collaboration_surface_kinds": registry.collaboration_surface_kinds,
        "visual_workspace_profile_ids": registry.visual_workspace_profile_ids,
        "route_names": registry.route_names,
        "profile_count": registry.profile_count,
        "source_registries": registry.source_registries,
        "source_reference_ids": registry.source_reference_ids,
        "runtime_collaboration_surface_refs": (
            registry.runtime_collaboration_surface_refs
        ),
        "observability_surfaces": registry.observability_surfaces,
    }


if __name__ == "__main__":
    unittest.main()
