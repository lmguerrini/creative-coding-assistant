import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    MultimodalCrossAgentWorkspaceRegistry,
    agent_workspace_registry,
    blackboard_memory_registry,
    multimodal_artifact_lineage_registry,
    multimodal_cross_agent_workspace_profile_by_id,
    multimodal_cross_agent_workspace_profiles_for_agent_workspace_profile,
    multimodal_cross_agent_workspace_profiles_for_route,
    multimodal_cross_agent_workspace_profiles_for_surface_kind,
    multimodal_cross_agent_workspace_registry,
    multimodal_visual_workspace_registry,
    shared_context_view_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "planning_cross_agent_workspace",
    "artifact_runtime_cross_agent_workspace",
    "critique_curation_cross_agent_workspace",
    "refinement_synthesis_cross_agent_workspace",
)
EXPECTED_PROFILE_KINDS = (
    "planning_cross_agent_workspace",
    "artifact_runtime_cross_agent_workspace",
    "critique_curation_cross_agent_workspace",
    "refinement_synthesis_cross_agent_workspace",
)
EXPECTED_SURFACE_KINDS = (
    "planning_context",
    "artifact_runtime",
    "critique_curation",
    "refinement_synthesis",
)
EXPECTED_SOURCE_REGISTRIES = (
    "multimodal_visual_workspace_registry",
    "multimodal_artifact_lineage_registry",
    "agent_workspace_registry",
    "shared_context_view_registry",
    "blackboard_memory_registry",
    "nextjs_workstation_shell",
    "nextjs_workstation_state",
)
EXPECTED_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_VISUAL_WORKSPACE_REGISTRY",
    "multimodal_studio.MULTIMODAL_ARTIFACT_LINEAGE_REGISTRY",
    "hybrid_studio.AGENT_WORKSPACE_REGISTRY",
    "shared_context_views.SHARED_CONTEXT_VIEW_REGISTRY",
    "blackboard_memory.BLACKBOARD_MEMORY_REGISTRY",
    "hybrid_studio.AgentWorkspaceProfile",
    "shared_context_views.SharedContextViewContract",
    "blackboard_memory.BlackboardMemoryChannelContract",
    "clients.nextjs.workstation_shell.WorkstationShell",
    "clients.nextjs.workstation_state.buildWorkstationState",
)
EXPECTED_WORKSPACE_SURFACES = (
    "cross_agent_workspace_panel",
    "cross_agent_roster_surface",
    "shared_context_scope_surface",
    "blackboard_channel_surface",
    "lineage_context_surface",
    "workspace_handoff_surface",
    "cross_agent_workspace_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "profile_id",
    "profile_name",
    "cross_agent_workspace_kind",
    "cross_agent_surface_kind",
    "source_agent_workspace_profile_ids",
    "source_visual_workspace_profile_ids",
    "source_artifact_lineage_profile_ids",
    "source_shared_context_view_ids",
    "source_blackboard_channel_ids",
    "workspace_context_fields",
    "source_reference_ids",
    "route_applicability",
    "cross_agent_workspace_surfaces",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "workspace_execution_implemented",
    "agent_instantiation_implemented",
    "agent_invocation_implemented",
    "multi_agent_orchestration_implemented",
    "shared_context_materialization_implemented",
    "blackboard_state_read_implemented",
    "blackboard_state_write_implemented",
    "workspace_state_mutation_implemented",
    "collaboration_storage_persistence_implemented",
    "rendering_execution_implemented",
    "workflow_control_implemented",
    "human_input_request_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class MultimodalStudioCrossAgentWorkspaceTests(unittest.TestCase):
    def test_cross_agent_workspace_registry_covers_expected_sources(self) -> None:
        registry = multimodal_cross_agent_workspace_registry()

        self.assertEqual(
            registry.role,
            "multimodal_cross_agent_workspace_registry",
        )
        self.assertEqual(
            registry.serialization_version,
            "multimodal_cross_agent_workspace_registry.v1",
        )
        self.assertEqual(registry.profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(
            registry.cross_agent_workspace_kinds,
            EXPECTED_PROFILE_KINDS,
        )
        self.assertEqual(registry.cross_agent_surface_kinds, EXPECTED_SURFACE_KINDS)
        self.assertEqual(
            registry.agent_workspace_profile_ids,
            agent_workspace_registry().workspace_profile_ids,
        )
        self.assertEqual(
            registry.visual_workspace_profile_ids,
            multimodal_visual_workspace_registry().profile_ids,
        )
        self.assertEqual(
            registry.artifact_lineage_profile_ids,
            multimodal_artifact_lineage_registry().profile_ids,
        )
        self.assertEqual(
            registry.shared_context_view_ids,
            shared_context_view_registry().view_ids,
        )
        self.assertEqual(
            registry.blackboard_channel_ids,
            blackboard_memory_registry().channel_ids,
        )
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertEqual(registry.source_reference_ids, EXPECTED_SOURCE_REFERENCES)
        self.assertEqual(
            registry.cross_agent_workspace_surface_refs,
            EXPECTED_WORKSPACE_SURFACES,
        )
        self.assertIn("does not instantiate agents", registry.authority_boundary)
        self.assertIn("read or write blackboard state", registry.authority_boundary)
        self.assertIn("agent_invocation", registry.blocked_runtime_behaviors)
        self.assertIn(
            "shared_context_materialization",
            registry.blocked_runtime_behaviors,
        )
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.workspace_execution_implemented)
        self.assertFalse(registry.agent_instantiation_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.multi_agent_orchestration_implemented)
        self.assertFalse(registry.shared_context_materialization_implemented)
        self.assertFalse(registry.blackboard_state_read_implemented)
        self.assertFalse(registry.blackboard_state_write_implemented)
        self.assertFalse(registry.workspace_state_mutation_implemented)
        self.assertFalse(registry.collaboration_storage_persistence_implemented)
        self.assertFalse(registry.rendering_execution_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)

    def test_cross_agent_workspace_profiles_are_passive_and_source_aligned(self) -> None:
        registry = multimodal_cross_agent_workspace_registry()
        known_routes = set(registry.route_names)
        known_agent_workspaces = set(registry.agent_workspace_profile_ids)
        known_visual_workspaces = set(registry.visual_workspace_profile_ids)
        known_lineage_profiles = set(registry.artifact_lineage_profile_ids)
        known_shared_context_views = set(registry.shared_context_view_ids)
        known_blackboard_channels = set(registry.blackboard_channel_ids)
        known_surfaces = set(registry.cross_agent_workspace_surface_refs)
        known_source_references = set(registry.source_reference_ids)
        covered_source_references: set[str] = set()

        for profile in registry.cross_agent_workspace_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "multimodal_cross_agent_workspace_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_agent_workspace_profile_ids).issubset(
                    known_agent_workspaces
                )
            )
            self.assertTrue(
                set(profile.source_visual_workspace_profile_ids).issubset(
                    known_visual_workspaces
                )
            )
            self.assertTrue(
                set(profile.source_artifact_lineage_profile_ids).issubset(
                    known_lineage_profiles
                )
            )
            self.assertTrue(
                set(profile.source_shared_context_view_ids).issubset(
                    known_shared_context_views
                )
            )
            self.assertTrue(
                set(profile.source_blackboard_channel_ids).issubset(
                    known_blackboard_channels
                )
            )
            self.assertTrue(
                set(profile.cross_agent_workspace_surfaces).issubset(known_surfaces)
            )
            self.assertTrue(
                set(profile.source_reference_ids).issubset(known_source_references)
            )
            covered_source_references.update(profile.source_reference_ids)
            self.assertIn("agent_invocation", profile.blocked_runtime_behaviors)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.workspace_execution_implemented)
            self.assertFalse(profile.agent_instantiation_implemented)
            self.assertFalse(profile.agent_invocation_implemented)
            self.assertFalse(profile.multi_agent_orchestration_implemented)
            self.assertFalse(profile.shared_context_materialization_implemented)
            self.assertFalse(profile.blackboard_state_read_implemented)
            self.assertFalse(profile.blackboard_state_write_implemented)
            self.assertFalse(profile.workspace_state_mutation_implemented)
            self.assertFalse(profile.collaboration_storage_persistence_implemented)
            self.assertFalse(profile.rendering_execution_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)

        self.assertEqual(covered_source_references, known_source_references)

    def test_cross_agent_workspace_lookup_helpers_are_stable(self) -> None:
        profile = multimodal_cross_agent_workspace_profile_by_id(
            "planning_cross_agent_workspace"
        )
        missing_profile = multimodal_cross_agent_workspace_profile_by_id(
            "missing_profile"
        )
        planning_profiles = (
            multimodal_cross_agent_workspace_profiles_for_surface_kind(
                "planning_context"
            )
        )
        route_profiles = multimodal_cross_agent_workspace_profiles_for_route(
            RouteName.PREVIEW
        )
        agent_workspace_profiles = (
            multimodal_cross_agent_workspace_profiles_for_agent_workspace_profile(
                "artifact_runtime_agent_workspace"
            )
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(
            profile.cross_agent_workspace_kind,
            "planning_cross_agent_workspace",
        )
        self.assertIn("planning_lineage_context", profile.workspace_context_fields)
        self.assertIn("no_agent_invocation_notice", profile.advisory_outputs)
        self.assertEqual(
            tuple(item.profile_id for item in planning_profiles),
            ("planning_cross_agent_workspace",),
        )
        self.assertEqual(
            tuple(item.profile_id for item in route_profiles),
            (
                "planning_cross_agent_workspace",
                "artifact_runtime_cross_agent_workspace",
            ),
        )
        self.assertEqual(
            tuple(item.profile_id for item in agent_workspace_profiles),
            ("artifact_runtime_cross_agent_workspace",),
        )

    def test_registry_rejects_mismatched_cross_agent_workspace_metadata(self) -> None:
        registry = multimodal_cross_agent_workspace_registry()
        first_profile = registry.cross_agent_workspace_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Cross-Agent Workspace"}
        )
        unknown_workspace_profile = first_profile.model_copy(
            update={
                "source_agent_workspace_profile_ids": (
                    "unknown_agent_workspace",
                )
            }
        )

        duplicate_kwargs = _registry_kwargs(registry)
        duplicate_kwargs["cross_agent_workspace_profiles"] = (
            first_profile,
            duplicate_profile,
        ) + registry.cross_agent_workspace_profiles[2:]
        with self.assertRaisesRegex(ValueError, "profile_ids must be unique"):
            MultimodalCrossAgentWorkspaceRegistry(**duplicate_kwargs)

        unknown_workspace_kwargs = _registry_kwargs(registry)
        unknown_workspace_kwargs["cross_agent_workspace_profiles"] = (
            unknown_workspace_profile,
        ) + registry.cross_agent_workspace_profiles[1:]
        with self.assertRaisesRegex(
            ValueError,
            "source_agent_workspace_profile_ids",
        ):
            MultimodalCrossAgentWorkspaceRegistry(**unknown_workspace_kwargs)

    def test_cross_agent_workspace_metadata_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        multimodal_cross_agent_workspace_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")


def _registry_kwargs(
    registry: MultimodalCrossAgentWorkspaceRegistry,
) -> dict[str, object]:
    return {
        "cross_agent_workspace_profiles": (
            registry.cross_agent_workspace_profiles
        ),
        "profile_ids": registry.profile_ids,
        "cross_agent_workspace_kinds": registry.cross_agent_workspace_kinds,
        "cross_agent_surface_kinds": registry.cross_agent_surface_kinds,
        "agent_workspace_profile_ids": registry.agent_workspace_profile_ids,
        "visual_workspace_profile_ids": registry.visual_workspace_profile_ids,
        "artifact_lineage_profile_ids": registry.artifact_lineage_profile_ids,
        "shared_context_view_ids": registry.shared_context_view_ids,
        "blackboard_channel_ids": registry.blackboard_channel_ids,
        "route_names": registry.route_names,
        "profile_count": registry.profile_count,
        "source_registries": registry.source_registries,
        "source_reference_ids": registry.source_reference_ids,
        "cross_agent_workspace_surface_refs": (
            registry.cross_agent_workspace_surface_refs
        ),
        "observability_surfaces": registry.observability_surfaces,
    }


if __name__ == "__main__":
    unittest.main()
