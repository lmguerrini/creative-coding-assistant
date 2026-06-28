import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    AgentWorkspaceRegistry,
    agent_capability_registry,
    agent_identity_registry,
    agent_metadata_registry,
    agent_role_registry,
    agent_workspace_profile_by_id,
    agent_workspace_profiles_for_agent_id,
    agent_workspace_profiles_for_route,
    agent_workspace_registry,
    hitl_decision_registry,
    local_cloud_comparison_registry,
    quality_profile_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "planning_context_agent_workspace",
    "artifact_runtime_agent_workspace",
    "critique_curation_agent_workspace",
    "refinement_synthesis_agent_workspace",
)
EXPECTED_KINDS = (
    "planning_context_workspace",
    "artifact_runtime_workspace",
    "critique_curation_workspace",
    "refinement_synthesis_workspace",
)
EXPECTED_SOURCE_REGISTRIES = (
    "agent_identity_registry",
    "agent_role_registry",
    "agent_metadata_registry",
    "agent_capability_registry",
    "local_cloud_comparison_registry",
    "quality_profile_registry",
    "hitl_decision_registry",
    "studio_mode_registry",
)
EXPECTED_WORKSPACE_SURFACES = (
    "agent_workspace_panel",
    "agent_roster_panel",
    "agent_role_matrix",
    "comparison_context_panel",
    "quality_context_panel",
    "hitl_review_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "workspace_profile_id",
    "profile_name",
    "workspace_kind",
    "source_agent_ids",
    "source_role_ids",
    "source_agent_metadata_ids",
    "source_capability_ids",
    "source_comparison_profile_ids",
    "source_quality_profile_ids",
    "source_hitl_decision_profile_ids",
    "route_applicability",
    "workspace_surfaces",
    "visible_context_fields",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "workspace_execution_implemented",
    "agent_instantiation_implemented",
    "agent_invocation_implemented",
    "multi_agent_orchestration_implemented",
    "workspace_state_mutation_implemented",
    "memory_write_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
    "human_input_request_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "persistent_replay_storage_implemented",
    "serialization_version",
    "metadata_only",
}


class HybridStudioAgentWorkspaceTests(unittest.TestCase):
    def test_agent_workspace_registry_covers_expected_profiles(self) -> None:
        registry = agent_workspace_registry()

        self.assertEqual(registry.role, "agent_workspace_registry")
        self.assertEqual(registry.serialization_version, "agent_workspace_registry.v1")
        self.assertEqual(registry.workspace_profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.workspace_kinds, EXPECTED_KINDS)
        self.assertEqual(registry.agent_ids, agent_identity_registry().agent_ids)
        self.assertEqual(registry.role_ids, agent_role_registry().role_ids)
        self.assertEqual(
            registry.agent_metadata_ids, agent_metadata_registry().agent_ids
        )
        self.assertEqual(
            registry.capability_ids,
            agent_capability_registry().capability_ids,
        )
        self.assertEqual(
            registry.comparison_profile_ids,
            local_cloud_comparison_registry().comparison_profile_ids,
        )
        self.assertEqual(
            registry.quality_profile_ids,
            quality_profile_registry().quality_profile_ids,
        )
        self.assertEqual(
            registry.hitl_decision_profile_ids,
            hitl_decision_registry().hitl_decision_profile_ids,
        )
        self.assertEqual(registry.workspace_surface_refs, EXPECTED_WORKSPACE_SURFACES)
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertIn("does not instantiate agents", registry.authority_boundary)
        self.assertIn("agent_invocation", registry.blocked_runtime_behaviors)
        self.assertIn("workspace_state_mutation", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.workspace_execution_implemented)
        self.assertFalse(registry.agent_instantiation_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.multi_agent_orchestration_implemented)
        self.assertFalse(registry.workspace_state_mutation_implemented)
        self.assertFalse(registry.memory_write_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_replay_storage_implemented)

    def test_agent_workspace_profiles_are_passive_and_source_aligned(self) -> None:
        registry = agent_workspace_registry()
        known_routes = set(registry.route_names)
        known_agents = set(agent_identity_registry().agent_ids)
        known_roles = set(agent_role_registry().role_ids)
        known_agent_metadata = set(agent_metadata_registry().agent_ids)
        known_capabilities = set(agent_capability_registry().capability_ids)
        known_comparisons = set(
            local_cloud_comparison_registry().comparison_profile_ids
        )
        known_quality_profiles = set(quality_profile_registry().quality_profile_ids)
        known_hitl_profiles = set(hitl_decision_registry().hitl_decision_profile_ids)
        known_surfaces = set(registry.workspace_surface_refs)

        for profile in registry.workspace_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "agent_workspace_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(set(profile.source_agent_ids).issubset(known_agents))
            self.assertTrue(set(profile.source_role_ids).issubset(known_roles))
            self.assertTrue(
                set(profile.source_agent_metadata_ids).issubset(known_agent_metadata)
            )
            self.assertTrue(
                set(profile.source_capability_ids).issubset(known_capabilities)
            )
            self.assertTrue(
                set(profile.source_comparison_profile_ids).issubset(known_comparisons)
            )
            self.assertTrue(
                set(profile.source_quality_profile_ids).issubset(known_quality_profiles)
            )
            self.assertTrue(
                set(profile.source_hitl_decision_profile_ids).issubset(
                    known_hitl_profiles
                )
            )
            self.assertTrue(set(profile.workspace_surfaces).issubset(known_surfaces))
            self.assertIn("agent_invocation", profile.blocked_runtime_behaviors)
            self.assertIn(
                "multi_agent_orchestration",
                profile.blocked_runtime_behaviors,
            )
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.workspace_execution_implemented)
            self.assertFalse(profile.agent_instantiation_implemented)
            self.assertFalse(profile.agent_invocation_implemented)
            self.assertFalse(profile.multi_agent_orchestration_implemented)
            self.assertFalse(profile.workspace_state_mutation_implemented)
            self.assertFalse(profile.memory_write_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_replay_storage_implemented)

    def test_agent_workspace_lookup_helpers_are_stable(self) -> None:
        profile = agent_workspace_profile_by_id("critique_curation_agent_workspace")
        missing_profile = agent_workspace_profile_by_id("missing_profile")
        review_profiles = agent_workspace_profiles_for_route(RouteName.REVIEW)
        preview_profiles = agent_workspace_profiles_for_route("preview")
        critic_profiles = agent_workspace_profiles_for_agent_id("critic_agent")

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.workspace_kind, "critique_curation_workspace")
        self.assertIn("critic_agent", profile.source_agent_ids)
        self.assertIn(
            "critique_curation_agent_workspace_context",
            profile.advisory_outputs,
        )
        self.assertEqual(
            tuple(item.workspace_profile_id for item in preview_profiles),
            (
                "planning_context_agent_workspace",
                "artifact_runtime_agent_workspace",
            ),
        )
        self.assertEqual(
            tuple(item.workspace_profile_id for item in review_profiles),
            (
                "critique_curation_agent_workspace",
                "refinement_synthesis_agent_workspace",
            ),
        )
        self.assertEqual(
            tuple(item.workspace_profile_id for item in critic_profiles),
            ("critique_curation_agent_workspace",),
        )

    def test_registry_rejects_mismatched_sources_or_workspace_metadata(self) -> None:
        registry = agent_workspace_registry()
        first_profile = registry.workspace_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Profile"}
        )
        unknown_agent_profile = first_profile.model_copy(
            update={"source_agent_ids": ("unknown_agent",)}
        )
        unknown_comparison_profile = first_profile.model_copy(
            update={"source_comparison_profile_ids": ("unknown_comparison",)}
        )

        with self.assertRaisesRegex(ValueError, "workspace_profile_ids must be unique"):
            AgentWorkspaceRegistry(
                workspace_profiles=(first_profile, duplicate_profile)
                + registry.workspace_profiles[2:],
                workspace_profile_ids=registry.workspace_profile_ids,
                workspace_kinds=registry.workspace_kinds,
                agent_ids=registry.agent_ids,
                role_ids=registry.role_ids,
                agent_metadata_ids=registry.agent_metadata_ids,
                capability_ids=registry.capability_ids,
                comparison_profile_ids=registry.comparison_profile_ids,
                quality_profile_ids=registry.quality_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                workspace_surface_refs=registry.workspace_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_agent_ids"):
            AgentWorkspaceRegistry(
                workspace_profiles=(unknown_agent_profile,)
                + registry.workspace_profiles[1:],
                workspace_profile_ids=registry.workspace_profile_ids,
                workspace_kinds=registry.workspace_kinds,
                agent_ids=registry.agent_ids,
                role_ids=registry.role_ids,
                agent_metadata_ids=registry.agent_metadata_ids,
                capability_ids=registry.capability_ids,
                comparison_profile_ids=registry.comparison_profile_ids,
                quality_profile_ids=registry.quality_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                workspace_surface_refs=registry.workspace_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_comparison_profile_ids"):
            AgentWorkspaceRegistry(
                workspace_profiles=(unknown_comparison_profile,)
                + registry.workspace_profiles[1:],
                workspace_profile_ids=registry.workspace_profile_ids,
                workspace_kinds=registry.workspace_kinds,
                agent_ids=registry.agent_ids,
                role_ids=registry.role_ids,
                agent_metadata_ids=registry.agent_metadata_ids,
                capability_ids=registry.capability_ids,
                comparison_profile_ids=registry.comparison_profile_ids,
                quality_profile_ids=registry.quality_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                workspace_surface_refs=registry.workspace_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

    def test_agent_workspace_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        agent_workspace_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")
