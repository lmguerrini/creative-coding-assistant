import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    MultimodalStudioIntegrationRegistry,
    multimodal_artifact_collaboration_registry,
    multimodal_artifact_lineage_registry,
    multimodal_artifact_provenance_registry,
    multimodal_branching_timeline_registry,
    multimodal_creative_evolution_timeline_registry,
    multimodal_cross_agent_workspace_registry,
    multimodal_interactive_canvas_registry,
    multimodal_live_preview_registry,
    multimodal_multi_preview_registry,
    multimodal_real_time_workflow_visualization_registry,
    multimodal_runtime_collaboration_registry,
    multimodal_shared_artifact_board_registry,
    multimodal_studio_integration_profile_by_id,
    multimodal_studio_integration_profiles_for_route,
    multimodal_studio_integration_profiles_for_source_registry,
    multimodal_studio_integration_registry,
    multimodal_visual_workspace_registry,
    multimodal_workspace_history_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "preview_workspace_multimodal_studio_integration",
    "collaboration_artifact_multimodal_studio_integration",
    "history_lineage_multimodal_studio_integration",
    "timeline_visualization_multimodal_studio_integration",
)
EXPECTED_KINDS = (
    "preview_workspace_integration",
    "collaboration_artifact_integration",
    "history_lineage_integration",
    "timeline_visualization_integration",
)
EXPECTED_SOURCE_REGISTRIES = (
    "multimodal_live_preview_registry",
    "multimodal_multi_preview_registry",
    "multimodal_interactive_canvas_registry",
    "multimodal_visual_workspace_registry",
    "multimodal_runtime_collaboration_registry",
    "multimodal_artifact_collaboration_registry",
    "multimodal_artifact_provenance_registry",
    "multimodal_artifact_lineage_registry",
    "multimodal_cross_agent_workspace_registry",
    "multimodal_shared_artifact_board_registry",
    "multimodal_workspace_history_registry",
    "multimodal_branching_timeline_registry",
    "multimodal_creative_evolution_timeline_registry",
    "multimodal_real_time_workflow_visualization_registry",
)
EXPECTED_PROFILE_GROUPS = (
    "live_preview_profiles",
    "multi_preview_profiles",
    "interactive_canvas_profiles",
    "visual_workspace_profiles",
    "runtime_collaboration_profiles",
    "artifact_collaboration_profiles",
    "artifact_provenance_profiles",
    "artifact_lineage_profiles",
    "cross_agent_workspace_profiles",
    "shared_artifact_board_profiles",
    "workspace_history_profiles",
    "branching_timeline_profiles",
    "creative_evolution_timeline_profiles",
    "real_time_workflow_visualization_profiles",
)
EXPECTED_INTEGRATION_SURFACES = (
    "multimodal_studio_shell",
    "preview_workspace_integration_surface",
    "collaboration_artifact_integration_surface",
    "history_lineage_integration_surface",
    "timeline_visualization_integration_surface",
    "integration_summary_surface",
    "multimodal_studio_integration_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "integration_profile_id",
    "profile_name",
    "integration_kind",
    "source_registry_names",
    "linked_profile_group_refs",
    "route_applicability",
    "integration_surfaces",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "studio_runtime_activation_implemented",
    "rendering_execution_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "artifact_mutation_implemented",
    "generated_output_mutation_implemented",
    "workflow_control_implemented",
    "human_input_request_implemented",
    "retry_triggering_implemented",
    "storage_mutation_implemented",
    "collaboration_storage_persistence_implemented",
    "timeline_reconstruction_implemented",
    "real_time_stream_subscription_implemented",
    "networking_implemented",
    "serialization_version",
    "metadata_only",
}


class MultimodalStudioIntegrationTests(unittest.TestCase):
    def test_integration_registry_covers_all_v4_5_sources(self) -> None:
        registry = multimodal_studio_integration_registry()

        self.assertEqual(registry.role, "multimodal_studio_integration_registry")
        self.assertEqual(
            registry.serialization_version,
            "multimodal_studio_integration_registry.v1",
        )
        self.assertEqual(registry.integration_profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.integration_kinds, EXPECTED_KINDS)
        self.assertEqual(
            registry.live_preview_profile_ids,
            multimodal_live_preview_registry().profile_ids,
        )
        self.assertEqual(
            registry.multi_preview_profile_ids,
            multimodal_multi_preview_registry().profile_ids,
        )
        self.assertEqual(
            registry.interactive_canvas_profile_ids,
            multimodal_interactive_canvas_registry().profile_ids,
        )
        self.assertEqual(
            registry.visual_workspace_profile_ids,
            multimodal_visual_workspace_registry().profile_ids,
        )
        self.assertEqual(
            registry.runtime_collaboration_profile_ids,
            multimodal_runtime_collaboration_registry().profile_ids,
        )
        self.assertEqual(
            registry.artifact_collaboration_profile_ids,
            multimodal_artifact_collaboration_registry().profile_ids,
        )
        self.assertEqual(
            registry.artifact_provenance_profile_ids,
            multimodal_artifact_provenance_registry().profile_ids,
        )
        self.assertEqual(
            registry.artifact_lineage_profile_ids,
            multimodal_artifact_lineage_registry().profile_ids,
        )
        self.assertEqual(
            registry.cross_agent_workspace_profile_ids,
            multimodal_cross_agent_workspace_registry().profile_ids,
        )
        self.assertEqual(
            registry.shared_artifact_board_profile_ids,
            multimodal_shared_artifact_board_registry().profile_ids,
        )
        self.assertEqual(
            registry.workspace_history_profile_ids,
            multimodal_workspace_history_registry().profile_ids,
        )
        self.assertEqual(
            registry.branching_timeline_profile_ids,
            multimodal_branching_timeline_registry().profile_ids,
        )
        self.assertEqual(
            registry.creative_evolution_timeline_profile_ids,
            multimodal_creative_evolution_timeline_registry().profile_ids,
        )
        self.assertEqual(
            registry.real_time_workflow_visualization_profile_ids,
            multimodal_real_time_workflow_visualization_registry().profile_ids,
        )
        self.assertEqual(registry.profile_group_refs, EXPECTED_PROFILE_GROUPS)
        self.assertEqual(
            registry.integration_surface_refs,
            EXPECTED_INTEGRATION_SURFACES,
        )
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertIn("does not activate Studio runtime", registry.authority_boundary)
        self.assertIn("execute rendering", registry.authority_boundary)
        self.assertIn("studio_runtime_activation", registry.blocked_runtime_behaviors)
        self.assertIn("rendering_execution", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.studio_runtime_activation_implemented)
        self.assertFalse(registry.rendering_execution_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.provider_execution_implemented)
        self.assertFalse(registry.artifact_mutation_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.storage_mutation_implemented)
        self.assertFalse(registry.collaboration_storage_persistence_implemented)
        self.assertFalse(registry.timeline_reconstruction_implemented)
        self.assertFalse(registry.real_time_stream_subscription_implemented)
        self.assertFalse(registry.networking_implemented)

    def test_integration_profiles_are_passive_and_source_aligned(self) -> None:
        registry = multimodal_studio_integration_registry()
        known_routes = set(registry.route_names)
        known_sources = set(registry.source_registries)
        known_profile_groups = set(registry.profile_group_refs)
        known_surfaces = set(registry.integration_surface_refs)

        for profile in registry.integration_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "multimodal_studio_integration_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(set(profile.source_registry_names).issubset(known_sources))
            self.assertTrue(
                set(profile.linked_profile_group_refs).issubset(known_profile_groups)
            )
            self.assertTrue(set(profile.integration_surfaces).issubset(known_surfaces))
            self.assertIn(
                "studio_runtime_activation",
                profile.blocked_runtime_behaviors,
            )
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.studio_runtime_activation_implemented)
            self.assertFalse(profile.rendering_execution_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.provider_execution_implemented)
            self.assertFalse(profile.artifact_mutation_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.storage_mutation_implemented)
            self.assertFalse(profile.collaboration_storage_persistence_implemented)
            self.assertFalse(profile.timeline_reconstruction_implemented)
            self.assertFalse(profile.real_time_stream_subscription_implemented)
            self.assertFalse(profile.networking_implemented)

    def test_integration_lookup_helpers_are_stable(self) -> None:
        profile = multimodal_studio_integration_profile_by_id(
            "timeline_visualization_multimodal_studio_integration"
        )
        missing_profile = multimodal_studio_integration_profile_by_id("missing_profile")
        preview_profiles = multimodal_studio_integration_profiles_for_route(
            RouteName.PREVIEW
        )
        review_profiles = multimodal_studio_integration_profiles_for_route("review")
        realtime_profiles = multimodal_studio_integration_profiles_for_source_registry(
            "multimodal_real_time_workflow_visualization_registry"
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.integration_kind, "timeline_visualization_integration")
        self.assertIn(
            "multimodal_real_time_workflow_visualization_registry",
            profile.source_registry_names,
        )
        self.assertIn(
            "no_real_time_stream_subscription_notice",
            profile.advisory_outputs,
        )
        self.assertEqual(
            tuple(item.integration_profile_id for item in preview_profiles),
            (
                "preview_workspace_multimodal_studio_integration",
                "collaboration_artifact_multimodal_studio_integration",
                "timeline_visualization_multimodal_studio_integration",
            ),
        )
        self.assertEqual(
            tuple(item.integration_profile_id for item in review_profiles),
            (
                "collaboration_artifact_multimodal_studio_integration",
                "history_lineage_multimodal_studio_integration",
                "timeline_visualization_multimodal_studio_integration",
            ),
        )
        self.assertEqual(
            tuple(item.integration_profile_id for item in realtime_profiles),
            (
                "preview_workspace_multimodal_studio_integration",
                "timeline_visualization_multimodal_studio_integration",
            ),
        )

    def test_registry_rejects_mismatched_integration_metadata(self) -> None:
        registry = multimodal_studio_integration_registry()
        first_profile = registry.integration_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Multimodal Integration"}
        )
        unknown_source_profile = first_profile.model_copy(
            update={"source_registry_names": ("unknown_registry",)}
        )
        unknown_group_profile = first_profile.model_copy(
            update={"linked_profile_group_refs": ("unknown_group",)}
        )

        duplicate_kwargs = _registry_kwargs(registry)
        duplicate_kwargs["integration_profiles"] = (
            first_profile,
            duplicate_profile,
        ) + registry.integration_profiles[2:]
        with self.assertRaisesRegex(
            ValueError,
            "integration_profile_ids must be unique",
        ):
            MultimodalStudioIntegrationRegistry(**duplicate_kwargs)

        unknown_source_kwargs = _registry_kwargs(registry)
        unknown_source_kwargs["integration_profiles"] = (
            unknown_source_profile,
        ) + registry.integration_profiles[1:]
        with self.assertRaisesRegex(ValueError, "source_registry_names"):
            MultimodalStudioIntegrationRegistry(**unknown_source_kwargs)

        unknown_group_kwargs = _registry_kwargs(registry)
        unknown_group_kwargs["integration_profiles"] = (
            unknown_group_profile,
        ) + registry.integration_profiles[1:]
        with self.assertRaisesRegex(ValueError, "linked_profile_group_refs"):
            MultimodalStudioIntegrationRegistry(**unknown_group_kwargs)

    def test_integration_metadata_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        multimodal_studio_integration_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")


def _registry_kwargs(
    registry: MultimodalStudioIntegrationRegistry,
) -> dict[str, object]:
    return {
        "integration_profiles": registry.integration_profiles,
        "integration_profile_ids": registry.integration_profile_ids,
        "integration_kinds": registry.integration_kinds,
        "live_preview_profile_ids": registry.live_preview_profile_ids,
        "multi_preview_profile_ids": registry.multi_preview_profile_ids,
        "interactive_canvas_profile_ids": registry.interactive_canvas_profile_ids,
        "visual_workspace_profile_ids": registry.visual_workspace_profile_ids,
        "runtime_collaboration_profile_ids": (
            registry.runtime_collaboration_profile_ids
        ),
        "artifact_collaboration_profile_ids": (
            registry.artifact_collaboration_profile_ids
        ),
        "artifact_provenance_profile_ids": registry.artifact_provenance_profile_ids,
        "artifact_lineage_profile_ids": registry.artifact_lineage_profile_ids,
        "cross_agent_workspace_profile_ids": (
            registry.cross_agent_workspace_profile_ids
        ),
        "shared_artifact_board_profile_ids": (
            registry.shared_artifact_board_profile_ids
        ),
        "workspace_history_profile_ids": registry.workspace_history_profile_ids,
        "branching_timeline_profile_ids": registry.branching_timeline_profile_ids,
        "creative_evolution_timeline_profile_ids": (
            registry.creative_evolution_timeline_profile_ids
        ),
        "real_time_workflow_visualization_profile_ids": (
            registry.real_time_workflow_visualization_profile_ids
        ),
        "profile_group_refs": registry.profile_group_refs,
        "integration_surface_refs": registry.integration_surface_refs,
        "route_names": registry.route_names,
        "profile_count": registry.profile_count,
        "source_registries": registry.source_registries,
        "observability_surfaces": registry.observability_surfaces,
    }


if __name__ == "__main__":
    unittest.main()
