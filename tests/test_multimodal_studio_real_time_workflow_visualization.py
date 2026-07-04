import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    MultimodalRealTimeWorkflowVisualizationRegistry,
    multimodal_branching_timeline_registry,
    multimodal_creative_evolution_timeline_registry,
    multimodal_real_time_workflow_visualization_profile_by_id,
    multimodal_real_time_workflow_visualization_profiles_for_creative_evolution_timeline_profile,
    multimodal_real_time_workflow_visualization_profiles_for_route,
    multimodal_real_time_workflow_visualization_profiles_for_surface_kind,
    multimodal_real_time_workflow_visualization_registry,
    multimodal_runtime_collaboration_registry,
    multimodal_workspace_history_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "runtime_state_real_time_workflow_visualization",
    "timeline_event_real_time_workflow_visualization",
    "metadata_stage_real_time_workflow_visualization",
    "console_health_real_time_workflow_visualization",
)
EXPECTED_PROFILE_KINDS = (
    "runtime_state_visualization",
    "timeline_event_visualization",
    "metadata_stage_visualization",
    "console_health_visualization",
)
EXPECTED_SURFACE_KINDS = (
    "runtime_state",
    "timeline_event",
    "metadata_stage",
    "console_health",
)
EXPECTED_SOURCE_REGISTRIES = (
    "multimodal_creative_evolution_timeline_registry",
    "multimodal_branching_timeline_registry",
    "multimodal_runtime_collaboration_registry",
    "multimodal_workspace_history_registry",
    "nextjs_workflow_runtime",
    "nextjs_workflow_timeline",
    "nextjs_workflow_explorer",
    "nextjs_runtime_console",
)
EXPECTED_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_CREATIVE_EVOLUTION_TIMELINE_REGISTRY",
    "multimodal_studio.MULTIMODAL_BRANCHING_TIMELINE_REGISTRY",
    "multimodal_studio.MULTIMODAL_RUNTIME_COLLABORATION_REGISTRY",
    "multimodal_studio.MULTIMODAL_WORKSPACE_HISTORY_REGISTRY",
    "clients.nextjs.workflow_runtime.WorkflowRuntimeModel",
    "clients.nextjs.workflow_runtime.WorkflowRuntimeVisualState",
    "clients.nextjs.workflow_timeline.WorkflowTimelineModel",
    "clients.nextjs.workflow_timeline.WorkflowTimelineEvent",
    "clients.nextjs.workflow_explorer.WorkflowExplorerStage",
    "clients.nextjs.runtime_console.RuntimeConsoleModel",
)
EXPECTED_WORKFLOW_VISUALIZATION_SURFACES = (
    "real_time_workflow_visualization_panel",
    "runtime_state_visual_surface",
    "timeline_event_visual_surface",
    "metadata_stage_visual_surface",
    "console_health_visual_surface",
    "workflow_visualization_summary_surface",
    "real_time_workflow_visualization_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "profile_id",
    "profile_name",
    "visualization_profile_kind",
    "visualization_surface_kind",
    "source_creative_evolution_timeline_profile_ids",
    "source_branching_timeline_profile_ids",
    "source_runtime_collaboration_profile_ids",
    "source_workspace_history_profile_ids",
    "visualization_context_fields",
    "source_reference_ids",
    "route_applicability",
    "workflow_visualization_surfaces",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "real_time_stream_subscription_implemented",
    "workflow_state_mutation_implemented",
    "timeline_reconstruction_implemented",
    "event_replay_implemented",
    "runtime_console_control_implemented",
    "preview_runtime_control_implemented",
    "artifact_mutation_implemented",
    "generated_output_mutation_implemented",
    "collaboration_storage_persistence_implemented",
    "rendering_execution_implemented",
    "workflow_control_implemented",
    "human_input_request_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "networking_implemented",
    "serialization_version",
    "metadata_only",
}


class MultimodalStudioRealTimeWorkflowVisualizationTests(unittest.TestCase):
    def test_workflow_visualization_registry_covers_expected_sources(self) -> None:
        registry = multimodal_real_time_workflow_visualization_registry()

        self.assertEqual(
            registry.role,
            "multimodal_real_time_workflow_visualization_registry",
        )
        self.assertEqual(
            registry.serialization_version,
            "multimodal_real_time_workflow_visualization_registry.v1",
        )
        self.assertEqual(registry.profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(
            registry.visualization_profile_kinds,
            EXPECTED_PROFILE_KINDS,
        )
        self.assertEqual(
            registry.visualization_surface_kinds,
            EXPECTED_SURFACE_KINDS,
        )
        self.assertEqual(
            registry.creative_evolution_timeline_profile_ids,
            multimodal_creative_evolution_timeline_registry().profile_ids,
        )
        self.assertEqual(
            registry.branching_timeline_profile_ids,
            multimodal_branching_timeline_registry().profile_ids,
        )
        self.assertEqual(
            registry.runtime_collaboration_profile_ids,
            multimodal_runtime_collaboration_registry().profile_ids,
        )
        self.assertEqual(
            registry.workspace_history_profile_ids,
            multimodal_workspace_history_registry().profile_ids,
        )
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertEqual(registry.source_reference_ids, EXPECTED_SOURCE_REFERENCES)
        self.assertEqual(
            registry.workflow_visualization_surface_refs,
            EXPECTED_WORKFLOW_VISUALIZATION_SURFACES,
        )
        self.assertIn("does not subscribe to live streams", registry.authority_boundary)
        self.assertIn("control runtime consoles", registry.authority_boundary)
        self.assertIn(
            "real_time_stream_subscription",
            registry.blocked_runtime_behaviors,
        )
        self.assertIn("runtime_console_control", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.real_time_stream_subscription_implemented)
        self.assertFalse(registry.workflow_state_mutation_implemented)
        self.assertFalse(registry.timeline_reconstruction_implemented)
        self.assertFalse(registry.event_replay_implemented)
        self.assertFalse(registry.runtime_console_control_implemented)
        self.assertFalse(registry.preview_runtime_control_implemented)
        self.assertFalse(registry.artifact_mutation_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.collaboration_storage_persistence_implemented)
        self.assertFalse(registry.rendering_execution_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.networking_implemented)

    def test_workflow_visualization_profiles_are_passive_and_source_aligned(
        self,
    ) -> None:
        registry = multimodal_real_time_workflow_visualization_registry()
        known_routes = set(registry.route_names)
        known_evolution = set(registry.creative_evolution_timeline_profile_ids)
        known_branching = set(registry.branching_timeline_profile_ids)
        known_runtime = set(registry.runtime_collaboration_profile_ids)
        known_history = set(registry.workspace_history_profile_ids)
        known_surfaces = set(registry.workflow_visualization_surface_refs)
        known_source_references = set(registry.source_reference_ids)
        covered_source_references: set[str] = set()

        for profile in registry.real_time_workflow_visualization_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "multimodal_real_time_workflow_visualization_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_creative_evolution_timeline_profile_ids).issubset(
                    known_evolution
                )
            )
            self.assertTrue(
                set(profile.source_branching_timeline_profile_ids).issubset(
                    known_branching
                )
            )
            self.assertTrue(
                set(profile.source_runtime_collaboration_profile_ids).issubset(
                    known_runtime
                )
            )
            self.assertTrue(
                set(profile.source_workspace_history_profile_ids).issubset(
                    known_history
                )
            )
            self.assertTrue(
                set(profile.workflow_visualization_surfaces).issubset(known_surfaces)
            )
            self.assertTrue(
                set(profile.source_reference_ids).issubset(known_source_references)
            )
            covered_source_references.update(profile.source_reference_ids)
            self.assertIn(
                "real_time_stream_subscription",
                profile.blocked_runtime_behaviors,
            )
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.real_time_stream_subscription_implemented)
            self.assertFalse(profile.workflow_state_mutation_implemented)
            self.assertFalse(profile.timeline_reconstruction_implemented)
            self.assertFalse(profile.event_replay_implemented)
            self.assertFalse(profile.runtime_console_control_implemented)
            self.assertFalse(profile.preview_runtime_control_implemented)
            self.assertFalse(profile.artifact_mutation_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.collaboration_storage_persistence_implemented)
            self.assertFalse(profile.rendering_execution_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.networking_implemented)

        self.assertEqual(covered_source_references, known_source_references)

    def test_workflow_visualization_lookup_helpers_are_stable(self) -> None:
        profile = multimodal_real_time_workflow_visualization_profile_by_id(
            "runtime_state_real_time_workflow_visualization"
        )
        missing_profile = multimodal_real_time_workflow_visualization_profile_by_id(
            "missing_profile"
        )
        runtime_profiles = (
            multimodal_real_time_workflow_visualization_profiles_for_surface_kind(
                "runtime_state"
            )
        )
        debug_profiles = multimodal_real_time_workflow_visualization_profiles_for_route(
            RouteName.DEBUG
        )
        quality_evolution_profiles = multimodal_real_time_workflow_visualization_profiles_for_creative_evolution_timeline_profile(  # noqa: E501
            "quality_refinement_creative_evolution_timeline"
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(
            profile.visualization_profile_kind,
            "runtime_state_visualization",
        )
        self.assertIn("WorkflowRuntimeModel", profile.visualization_context_fields)
        self.assertIn(
            "no_real_time_stream_subscription_notice",
            profile.advisory_outputs,
        )
        self.assertEqual(
            tuple(item.profile_id for item in runtime_profiles),
            ("runtime_state_real_time_workflow_visualization",),
        )
        self.assertEqual(
            tuple(item.profile_id for item in debug_profiles),
            (
                "runtime_state_real_time_workflow_visualization",
                "timeline_event_real_time_workflow_visualization",
                "console_health_real_time_workflow_visualization",
            ),
        )
        self.assertEqual(
            tuple(item.profile_id for item in quality_evolution_profiles),
            (
                "timeline_event_real_time_workflow_visualization",
                "console_health_real_time_workflow_visualization",
            ),
        )

    def test_registry_rejects_mismatched_workflow_visualization_metadata(
        self,
    ) -> None:
        registry = multimodal_real_time_workflow_visualization_registry()
        first_profile = registry.real_time_workflow_visualization_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Workflow Visualization"}
        )
        unknown_evolution_profile = first_profile.model_copy(
            update={
                "source_creative_evolution_timeline_profile_ids": (
                    "unknown_creative_evolution",
                )
            }
        )

        duplicate_kwargs = _registry_kwargs(registry)
        duplicate_kwargs["real_time_workflow_visualization_profiles"] = (
            first_profile,
            duplicate_profile,
        ) + registry.real_time_workflow_visualization_profiles[2:]
        with self.assertRaisesRegex(ValueError, "profile_ids must be unique"):
            MultimodalRealTimeWorkflowVisualizationRegistry(**duplicate_kwargs)

        unknown_evolution_kwargs = _registry_kwargs(registry)
        unknown_evolution_kwargs["real_time_workflow_visualization_profiles"] = (
            unknown_evolution_profile,
        ) + registry.real_time_workflow_visualization_profiles[1:]
        with self.assertRaisesRegex(
            ValueError,
            "source_creative_evolution_timeline_profile_ids",
        ):
            MultimodalRealTimeWorkflowVisualizationRegistry(**unknown_evolution_kwargs)

    def test_workflow_visualization_metadata_does_not_change_provider_factory(
        self,
    ) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        multimodal_real_time_workflow_visualization_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")


def _registry_kwargs(
    registry: MultimodalRealTimeWorkflowVisualizationRegistry,
) -> dict[str, object]:
    return {
        "real_time_workflow_visualization_profiles": (
            registry.real_time_workflow_visualization_profiles
        ),
        "profile_ids": registry.profile_ids,
        "visualization_profile_kinds": registry.visualization_profile_kinds,
        "visualization_surface_kinds": registry.visualization_surface_kinds,
        "creative_evolution_timeline_profile_ids": (
            registry.creative_evolution_timeline_profile_ids
        ),
        "branching_timeline_profile_ids": registry.branching_timeline_profile_ids,
        "runtime_collaboration_profile_ids": (
            registry.runtime_collaboration_profile_ids
        ),
        "workspace_history_profile_ids": registry.workspace_history_profile_ids,
        "route_names": registry.route_names,
        "profile_count": registry.profile_count,
        "source_registries": registry.source_registries,
        "source_reference_ids": registry.source_reference_ids,
        "workflow_visualization_surface_refs": (
            registry.workflow_visualization_surface_refs
        ),
        "observability_surfaces": registry.observability_surfaces,
    }


if __name__ == "__main__":
    unittest.main()
