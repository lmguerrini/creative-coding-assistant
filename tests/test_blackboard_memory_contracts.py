import unittest

from creative_coding_assistant.contracts import AssistantMode, AssistantRequest
from creative_coding_assistant.orchestration import (
    BlackboardMemoryRegistry,
    RouteCapability,
    RouteDecision,
    RouteName,
    agent_memory_contract_registry,
    blackboard_channel_by_id,
    blackboard_channels_for_agent,
    blackboard_memory_registry,
    blackboard_permissions_by_agent_id,
    build_memory_context_request,
)

REQUIRED_CHANNEL_FIELDS = {
    "channel_id",
    "channel_name",
    "owner_agent_id",
    "owner_role_family",
    "memory_stage",
    "permitted_writer_agent_ids",
    "permitted_reader_agent_ids",
    "permitted_reference_agent_ids",
    "metadata_keys",
    "persistence_mode",
    "storage_boundary",
    "persistence_policy",
    "authority_boundary",
    "source_memory_contract_id",
    "blocked_runtime_behaviors",
    "persistence_implemented",
    "storage_backend_implemented",
    "database_schema_implemented",
    "runtime_read_implemented",
    "runtime_write_implemented",
    "memory_mutation_implemented",
    "serialization_version",
    "metadata_only",
}

REQUIRED_PERMISSION_FIELDS = {
    "agent_id",
    "permission_id",
    "memory_stage",
    "read_access",
    "write_access",
    "reference_access",
    "readable_channel_ids",
    "writable_channel_ids",
    "referenceable_channel_ids",
    "writable_metadata_keys",
    "permission_boundary",
    "source_memory_contract_id",
    "blocked_runtime_behaviors",
    "reads_runtime_blackboard",
    "writes_runtime_blackboard",
    "persists_blackboard_records",
    "creates_storage_backend",
    "mutates_shared_context",
    "serialization_version",
    "metadata_only",
}


class BlackboardMemoryContractTests(unittest.TestCase):
    def test_registry_covers_agent_memory_contracts(self) -> None:
        memory_registry = agent_memory_contract_registry()
        blackboard_registry = blackboard_memory_registry()

        self.assertEqual(blackboard_registry.role, "blackboard_memory_registry")
        self.assertEqual(
            blackboard_registry.serialization_version,
            "blackboard_memory_registry.v1",
        )
        self.assertEqual(blackboard_registry.agent_ids, memory_registry.agent_ids)
        self.assertEqual(blackboard_registry.channel_count, 12)
        self.assertEqual(blackboard_registry.permission_count, 12)
        self.assertEqual(
            blackboard_registry.source_registries,
            ("agent_memory_contract_registry", "agent_identity_registry"),
        )
        self.assertTrue(blackboard_registry.metadata_only)
        self.assertFalse(blackboard_registry.persistence_implemented)
        self.assertFalse(blackboard_registry.storage_backend_implemented)
        self.assertFalse(blackboard_registry.database_schema_implemented)
        self.assertFalse(blackboard_registry.runtime_read_implemented)
        self.assertFalse(blackboard_registry.runtime_write_implemented)
        self.assertFalse(blackboard_registry.memory_mutation_implemented)
        self.assertIn(
            "do not implement persistence",
            blackboard_registry.authority_boundary,
        )

    def test_channels_mark_persistence_and_storage_as_non_implemented(self) -> None:
        registry = blackboard_memory_registry()

        for channel in registry.channels:
            dumped = channel.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CHANNEL_FIELDS)
            self.assertEqual(channel.serialization_version, "blackboard_memory_channel.v1")
            self.assertEqual(channel.persistence_mode, "not_persisted")
            self.assertEqual(channel.storage_boundary, "no_storage_backend")
            self.assertEqual(channel.permitted_reader_agent_ids, registry.agent_ids)
            self.assertEqual(channel.permitted_reference_agent_ids, registry.agent_ids)
            self.assertEqual(channel.permitted_writer_agent_ids, (channel.owner_agent_id,))
            self.assertTrue(channel.metadata_keys)
            self.assertIn(
                "storage_backend_creation",
                channel.blocked_runtime_behaviors,
            )
            self.assertFalse(channel.persistence_implemented)
            self.assertFalse(channel.storage_backend_implemented)
            self.assertFalse(channel.database_schema_implemented)
            self.assertFalse(channel.runtime_read_implemented)
            self.assertFalse(channel.runtime_write_implemented)
            self.assertFalse(channel.memory_mutation_implemented)
            self.assertTrue(channel.metadata_only)

    def test_permissions_are_explicit_future_metadata_boundaries(self) -> None:
        registry = blackboard_memory_registry()

        for permission in registry.permissions:
            dumped = permission.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PERMISSION_FIELDS)
            self.assertEqual(
                permission.serialization_version,
                "blackboard_agent_permission.v1",
            )
            self.assertEqual(permission.read_access, "future_metadata_only")
            self.assertEqual(permission.write_access, "future_metadata_only")
            self.assertEqual(permission.reference_access, "metadata_reference_only")
            self.assertEqual(permission.readable_channel_ids, registry.channel_ids)
            self.assertEqual(permission.referenceable_channel_ids, registry.channel_ids)
            self.assertEqual(len(permission.writable_channel_ids), 1)
            self.assertIn("does not read or write", permission.permission_boundary)
            self.assertFalse(permission.reads_runtime_blackboard)
            self.assertFalse(permission.writes_runtime_blackboard)
            self.assertFalse(permission.persists_blackboard_records)
            self.assertFalse(permission.creates_storage_backend)
            self.assertFalse(permission.mutates_shared_context)
            self.assertTrue(permission.metadata_only)

    def test_blackboard_lookups_are_stable_and_passive(self) -> None:
        planner_channel = blackboard_channel_by_id("planner_agent_blackboard_channel")
        planner_permission = blackboard_permissions_by_agent_id("planner_agent")
        planner_channels = blackboard_channels_for_agent("planner_agent")

        self.assertIsNone(blackboard_channel_by_id("missing_channel"))
        self.assertIsNone(blackboard_permissions_by_agent_id("missing_agent"))
        self.assertEqual(blackboard_channels_for_agent("missing_agent"), ())
        self.assertIsNotNone(planner_channel)
        self.assertIsNotNone(planner_permission)
        assert planner_channel is not None
        assert planner_permission is not None
        self.assertEqual(planner_channel.owner_agent_id, "planner_agent")
        self.assertEqual(
            planner_permission.writable_channel_ids,
            ("planner_agent_blackboard_channel",),
        )
        self.assertIn("planning_context_packet", planner_channel.metadata_keys)
        self.assertEqual(len(planner_channels), 12)

    def test_registry_rejects_mismatched_channels_or_permissions(self) -> None:
        registry = blackboard_memory_registry()
        first_channel = registry.channels[0]
        duplicate_channel = first_channel.model_copy(
            update={"channel_name": "Duplicate Blackboard Channel"}
        )

        with self.assertRaisesRegex(ValueError, "channel_ids must be unique"):
            BlackboardMemoryRegistry(
                channels=(first_channel, duplicate_channel) + registry.channels[2:],
                permissions=registry.permissions,
                channel_ids=registry.channel_ids,
                agent_ids=registry.agent_ids,
                channel_count=12,
                permission_count=12,
                source_registries=registry.source_registries,
            )

        with self.assertRaisesRegex(ValueError, "agent_ids must match permissions"):
            BlackboardMemoryRegistry(
                channels=registry.channels,
                permissions=registry.permissions,
                channel_ids=registry.channel_ids,
                agent_ids=("other_agent",) + registry.agent_ids[1:],
                channel_count=12,
                permission_count=12,
                source_registries=registry.source_registries,
            )

    def test_blackboard_contracts_do_not_change_runtime_memory_requests(self) -> None:
        request = AssistantRequest(
            query="Continue the installation planning thread.",
            conversation_id="conversation-1",
            project_id="project-1",
            mode=AssistantMode.EXPLAIN,
        )
        route_decision = RouteDecision(
            route=RouteName.EXPLAIN,
            mode=AssistantMode.EXPLAIN,
            capabilities=(RouteCapability.MEMORY_CONTEXT,),
        )
        before = build_memory_context_request(request, route_decision)

        blackboard_memory_registry()
        blackboard_channel_by_id("planner_agent_blackboard_channel")
        blackboard_permissions_by_agent_id("planner_agent")
        after = build_memory_context_request(request, route_decision)

        self.assertEqual(after, before)

    def test_blackboard_contracts_do_not_declare_storage_side_effects(self) -> None:
        registry = blackboard_memory_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for channel in registry.channels
                    for field in (
                        channel.channel_id,
                        channel.persistence_mode,
                        channel.storage_boundary,
                        channel.persistence_policy,
                        channel.authority_boundary,
                        *channel.blocked_runtime_behaviors,
                    )
                ),
                *(
                    field
                    for permission in registry.permissions
                    for field in (
                        permission.permission_id,
                        permission.permission_boundary,
                        *permission.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "ChromaMemoryAdapter",
            "ConversationTurnRepository",
            "ProjectMemoryRepository",
            "sqlite",
            "upsert(",
            "external_provider_call",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
