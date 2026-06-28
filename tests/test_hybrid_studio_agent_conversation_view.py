import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    AgentConversationViewRegistry,
    agent_conversation_view_profile_by_id,
    agent_conversation_view_profiles_for_agent_id,
    agent_conversation_view_profiles_for_route,
    agent_conversation_view_profiles_for_workspace,
    agent_conversation_view_registry,
    agent_identity_registry,
    agent_memory_contract_registry,
    agent_role_registry,
    agent_workspace_registry,
    hitl_decision_registry,
    shared_context_view_registry,
    workflow_agent_handoff_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "workspace_thread_conversation_view",
    "agent_handoff_conversation_view",
    "review_conversation_view",
    "audit_trail_conversation_view",
)
EXPECTED_KINDS = (
    "workspace_thread_view",
    "agent_handoff_view",
    "review_discussion_view",
    "audit_trail_view",
)
EXPECTED_SOURCE_REGISTRIES = (
    "agent_workspace_registry",
    "agent_identity_registry",
    "agent_role_registry",
    "shared_context_view_registry",
    "agent_memory_contract_registry",
    "workflow_agent_handoff_registry",
    "hitl_decision_registry",
    "studio_mode_registry",
)
EXPECTED_CONVERSATION_SURFACES = (
    "agent_conversation_panel",
    "conversation_thread_list",
    "agent_message_timeline",
    "handoff_context_panel",
    "shared_context_scope_panel",
    "hitl_conversation_review_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "conversation_view_profile_id",
    "profile_name",
    "conversation_view_kind",
    "source_workspace_profile_ids",
    "source_agent_ids",
    "source_role_ids",
    "source_shared_context_view_ids",
    "source_memory_contract_ids",
    "source_handoff_profile_ids",
    "source_hitl_decision_profile_ids",
    "route_applicability",
    "conversation_surfaces",
    "visible_thread_fields",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "conversation_execution_implemented",
    "conversation_persistence_implemented",
    "agent_message_generation_implemented",
    "agent_invocation_implemented",
    "memory_write_implemented",
    "workspace_state_mutation_implemented",
    "workflow_control_implemented",
    "human_input_request_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "persistent_replay_storage_implemented",
    "serialization_version",
    "metadata_only",
}


class HybridStudioAgentConversationViewTests(unittest.TestCase):
    def test_agent_conversation_view_registry_covers_expected_profiles(self) -> None:
        registry = agent_conversation_view_registry()

        self.assertEqual(registry.role, "agent_conversation_view_registry")
        self.assertEqual(
            registry.serialization_version,
            "agent_conversation_view_registry.v1",
        )
        self.assertEqual(registry.conversation_view_profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.conversation_view_kinds, EXPECTED_KINDS)
        self.assertEqual(
            registry.workspace_profile_ids,
            agent_workspace_registry().workspace_profile_ids,
        )
        self.assertEqual(registry.agent_ids, agent_identity_registry().agent_ids)
        self.assertEqual(registry.role_ids, agent_role_registry().role_ids)
        self.assertEqual(
            registry.shared_context_view_ids,
            shared_context_view_registry().view_ids,
        )
        self.assertEqual(
            registry.memory_contract_ids,
            tuple(
                contract.memory_contract_id
                for contract in agent_memory_contract_registry().contracts
            ),
        )
        self.assertEqual(
            registry.handoff_profile_ids,
            workflow_agent_handoff_registry().profile_ids,
        )
        self.assertEqual(
            registry.hitl_decision_profile_ids,
            hitl_decision_registry().hitl_decision_profile_ids,
        )
        self.assertEqual(
            registry.conversation_surface_refs,
            EXPECTED_CONVERSATION_SURFACES,
        )
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertIn("does not record conversations", registry.authority_boundary)
        self.assertIn(
            "agent_message_generation",
            registry.blocked_runtime_behaviors,
        )
        self.assertIn("conversation_persistence", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.conversation_execution_implemented)
        self.assertFalse(registry.conversation_persistence_implemented)
        self.assertFalse(registry.agent_message_generation_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.memory_write_implemented)
        self.assertFalse(registry.workspace_state_mutation_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_replay_storage_implemented)

    def test_agent_conversation_views_are_passive_and_source_aligned(self) -> None:
        registry = agent_conversation_view_registry()
        known_routes = set(registry.route_names)
        known_workspaces = set(agent_workspace_registry().workspace_profile_ids)
        known_agents = set(agent_identity_registry().agent_ids)
        known_roles = set(agent_role_registry().role_ids)
        known_context_views = set(shared_context_view_registry().view_ids)
        known_memory_contracts = {
            contract.memory_contract_id
            for contract in agent_memory_contract_registry().contracts
        }
        known_handoff_profiles = set(workflow_agent_handoff_registry().profile_ids)
        known_hitl_profiles = set(hitl_decision_registry().hitl_decision_profile_ids)
        known_surfaces = set(registry.conversation_surface_refs)

        for profile in registry.conversation_view_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "agent_conversation_view_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_workspace_profile_ids).issubset(known_workspaces)
            )
            self.assertTrue(set(profile.source_agent_ids).issubset(known_agents))
            self.assertTrue(set(profile.source_role_ids).issubset(known_roles))
            self.assertTrue(
                set(profile.source_shared_context_view_ids).issubset(
                    known_context_views
                )
            )
            self.assertTrue(
                set(profile.source_memory_contract_ids).issubset(known_memory_contracts)
            )
            self.assertTrue(
                set(profile.source_handoff_profile_ids).issubset(known_handoff_profiles)
            )
            self.assertTrue(
                set(profile.source_hitl_decision_profile_ids).issubset(
                    known_hitl_profiles
                )
            )
            self.assertTrue(set(profile.conversation_surfaces).issubset(known_surfaces))
            self.assertIn(
                "agent_message_generation",
                profile.blocked_runtime_behaviors,
            )
            self.assertIn(
                "conversation_persistence",
                profile.blocked_runtime_behaviors,
            )
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.conversation_execution_implemented)
            self.assertFalse(profile.conversation_persistence_implemented)
            self.assertFalse(profile.agent_message_generation_implemented)
            self.assertFalse(profile.agent_invocation_implemented)
            self.assertFalse(profile.memory_write_implemented)
            self.assertFalse(profile.workspace_state_mutation_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_replay_storage_implemented)

    def test_agent_conversation_view_lookup_helpers_are_stable(self) -> None:
        profile = agent_conversation_view_profile_by_id(
            "agent_handoff_conversation_view"
        )
        missing_profile = agent_conversation_view_profile_by_id("missing_profile")
        preview_profiles = agent_conversation_view_profiles_for_route("preview")
        review_profiles = agent_conversation_view_profiles_for_route(RouteName.REVIEW)
        artifact_workspace_profiles = agent_conversation_view_profiles_for_workspace(
            "artifact_runtime_agent_workspace"
        )
        critic_profiles = agent_conversation_view_profiles_for_agent_id("critic_agent")

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.conversation_view_kind, "agent_handoff_view")
        self.assertIn("critic_agent", profile.source_agent_ids)
        self.assertIn(
            "agent_handoff_conversation_context",
            profile.advisory_outputs,
        )
        self.assertEqual(
            tuple(item.conversation_view_profile_id for item in preview_profiles),
            (
                "workspace_thread_conversation_view",
                "audit_trail_conversation_view",
            ),
        )
        self.assertEqual(
            tuple(item.conversation_view_profile_id for item in review_profiles),
            (
                "agent_handoff_conversation_view",
                "review_conversation_view",
                "audit_trail_conversation_view",
            ),
        )
        self.assertEqual(
            tuple(
                item.conversation_view_profile_id
                for item in artifact_workspace_profiles
            ),
            (
                "workspace_thread_conversation_view",
                "agent_handoff_conversation_view",
                "audit_trail_conversation_view",
            ),
        )
        self.assertEqual(
            tuple(item.conversation_view_profile_id for item in critic_profiles),
            (
                "agent_handoff_conversation_view",
                "review_conversation_view",
                "audit_trail_conversation_view",
            ),
        )

    def test_registry_rejects_mismatched_sources_or_conversation_metadata(
        self,
    ) -> None:
        registry = agent_conversation_view_registry()
        first_profile = registry.conversation_view_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Conversation View"}
        )
        unknown_workspace_profile = first_profile.model_copy(
            update={"source_workspace_profile_ids": ("unknown_workspace",)}
        )
        unknown_handoff_profile = first_profile.model_copy(
            update={"source_handoff_profile_ids": ("unknown_handoff_profile",)}
        )

        with self.assertRaisesRegex(
            ValueError,
            "conversation_view_profile_ids must be unique",
        ):
            AgentConversationViewRegistry(
                conversation_view_profiles=(first_profile, duplicate_profile)
                + registry.conversation_view_profiles[2:],
                conversation_view_profile_ids=registry.conversation_view_profile_ids,
                conversation_view_kinds=registry.conversation_view_kinds,
                workspace_profile_ids=registry.workspace_profile_ids,
                agent_ids=registry.agent_ids,
                role_ids=registry.role_ids,
                shared_context_view_ids=registry.shared_context_view_ids,
                memory_contract_ids=registry.memory_contract_ids,
                handoff_profile_ids=registry.handoff_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                conversation_surface_refs=registry.conversation_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_workspace_profile_ids"):
            AgentConversationViewRegistry(
                conversation_view_profiles=(unknown_workspace_profile,)
                + registry.conversation_view_profiles[1:],
                conversation_view_profile_ids=registry.conversation_view_profile_ids,
                conversation_view_kinds=registry.conversation_view_kinds,
                workspace_profile_ids=registry.workspace_profile_ids,
                agent_ids=registry.agent_ids,
                role_ids=registry.role_ids,
                shared_context_view_ids=registry.shared_context_view_ids,
                memory_contract_ids=registry.memory_contract_ids,
                handoff_profile_ids=registry.handoff_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                conversation_surface_refs=registry.conversation_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_handoff_profile_ids"):
            AgentConversationViewRegistry(
                conversation_view_profiles=(unknown_handoff_profile,)
                + registry.conversation_view_profiles[1:],
                conversation_view_profile_ids=registry.conversation_view_profile_ids,
                conversation_view_kinds=registry.conversation_view_kinds,
                workspace_profile_ids=registry.workspace_profile_ids,
                agent_ids=registry.agent_ids,
                role_ids=registry.role_ids,
                shared_context_view_ids=registry.shared_context_view_ids,
                memory_contract_ids=registry.memory_contract_ids,
                handoff_profile_ids=registry.handoff_profile_ids,
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                conversation_surface_refs=registry.conversation_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

    def test_agent_conversation_view_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        agent_conversation_view_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")
