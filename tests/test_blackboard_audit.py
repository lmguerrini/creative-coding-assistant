import unittest

from creative_coding_assistant.contracts import AssistantMode, AssistantRequest
from creative_coding_assistant.orchestration import (
    BlackboardAuditRegistry,
    RouteCapability,
    RouteDecision,
    RouteName,
    blackboard_audit_by_agent_id,
    blackboard_audit_by_channel_id,
    blackboard_audit_registry,
    blackboard_audits_for_source_registry,
    blackboard_memory_registry,
    build_memory_context_request,
)

REQUIRED_AUDIT_RECORD_FIELDS = {
    "agent_id",
    "channel_id",
    "permission_id",
    "owner_role_family",
    "audit_stage",
    "channel_serialization_version",
    "permission_serialization_version",
    "source_memory_contract_id",
    "source_registries",
    "metadata_keys",
    "readable_channel_ids",
    "writable_channel_ids",
    "referenceable_channel_ids",
    "validated_blackboard_surfaces",
    "passive_boundary_flags",
    "audit_findings",
    "missing_coverage_items",
    "channel_blocked_runtime_behaviors",
    "permission_blocked_runtime_behaviors",
    "blocked_runtime_behaviors",
    "audit_status",
    "metadata_only_declared",
    "persistence_implemented",
    "storage_backend_implemented",
    "database_schema_implemented",
    "runtime_read_implemented",
    "runtime_write_implemented",
    "memory_mutation_implemented",
    "shared_context_materialization_implemented",
    "serialization_version",
    "metadata_only",
}


class BlackboardAuditTests(unittest.TestCase):
    def test_audit_registry_covers_blackboard_channels_and_permissions(self) -> None:
        audit_registry = blackboard_audit_registry()
        blackboard_registry = blackboard_memory_registry()

        self.assertEqual(audit_registry.role, "blackboard_audit_registry")
        self.assertEqual(
            audit_registry.serialization_version,
            "blackboard_audit_registry.v1",
        )
        self.assertEqual(audit_registry.audit_stage, "v4_6_blackboard_hardening")
        self.assertEqual(audit_registry.agent_ids, blackboard_registry.agent_ids)
        self.assertEqual(audit_registry.channel_ids, blackboard_registry.channel_ids)
        self.assertEqual(audit_registry.audit_count, 12)
        self.assertEqual(
            audit_registry.source_registries,
            ("agent_memory_contract_registry", "agent_identity_registry"),
        )
        self.assertTrue(audit_registry.all_channels_covered)
        self.assertTrue(audit_registry.all_permissions_covered)
        self.assertTrue(audit_registry.no_missing_coverage)
        self.assertTrue(audit_registry.metadata_only)
        self.assertFalse(audit_registry.active_runtime_audit_implemented)
        self.assertFalse(audit_registry.persistence_implemented)
        self.assertFalse(audit_registry.storage_backend_implemented)
        self.assertFalse(audit_registry.runtime_read_implemented)
        self.assertFalse(audit_registry.runtime_write_implemented)
        self.assertFalse(audit_registry.memory_mutation_implemented)
        self.assertIn("does not create storage", audit_registry.authority_boundary)

    def test_audit_records_are_passive_and_permission_aligned(self) -> None:
        registry = blackboard_audit_registry()

        for record in registry.audit_records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_AUDIT_RECORD_FIELDS)
            self.assertEqual(record.serialization_version, "blackboard_audit_record.v1")
            self.assertEqual(record.audit_status, "pass")
            self.assertEqual(record.audit_stage, registry.audit_stage)
            self.assertEqual(record.source_registries, registry.source_registries)
            self.assertEqual(
                record.validated_blackboard_surfaces,
                registry.validated_blackboard_surfaces,
            )
            self.assertEqual(
                record.passive_boundary_flags, registry.passive_boundary_flags
            )
            self.assertFalse(record.missing_coverage_items)
            self.assertEqual(record.writable_channel_ids, (record.channel_id,))
            self.assertEqual(record.readable_channel_ids, registry.channel_ids)
            self.assertEqual(record.referenceable_channel_ids, registry.channel_ids)
            self.assertTrue(record.metadata_keys)
            self.assertIn(
                "storage_backend_creation",
                record.channel_blocked_runtime_behaviors,
            )
            self.assertIn(
                "runtime_blackboard_mutation",
                record.permission_blocked_runtime_behaviors,
            )
            self.assertTrue(record.metadata_only_declared)
            self.assertTrue(record.metadata_only)
            self.assertFalse(record.persistence_implemented)
            self.assertFalse(record.storage_backend_implemented)
            self.assertFalse(record.database_schema_implemented)
            self.assertFalse(record.runtime_read_implemented)
            self.assertFalse(record.runtime_write_implemented)
            self.assertFalse(record.memory_mutation_implemented)
            self.assertFalse(record.shared_context_materialization_implemented)

    def test_audit_lookup_and_source_filtering_are_stable(self) -> None:
        registry = blackboard_audit_registry()
        planner_audit = blackboard_audit_by_agent_id("planner_agent")
        channel_audit = blackboard_audit_by_channel_id(
            "planner_agent_blackboard_channel"
        )
        missing_audit = blackboard_audit_by_agent_id("missing_agent")
        memory_source_audits = blackboard_audits_for_source_registry(
            "agent_memory_contract_registry"
        )
        missing_source_audits = blackboard_audits_for_source_registry(
            "missing_registry"
        )

        self.assertIsNone(missing_audit)
        self.assertIsNotNone(planner_audit)
        self.assertIs(planner_audit, channel_audit)
        assert planner_audit is not None
        self.assertEqual(planner_audit.owner_role_family, "planning")
        self.assertIn("planning_context_packet", planner_audit.metadata_keys)
        self.assertEqual(len(memory_source_audits), registry.audit_count)
        self.assertEqual(missing_source_audits, ())
        self.assertIs(
            planner_audit,
            blackboard_audit_by_agent_id("planner_agent", registry),
        )

    def test_audit_registry_rejects_mismatched_or_incomplete_records(self) -> None:
        registry = blackboard_audit_registry()
        first_record = registry.audit_records[0]
        duplicate_record = first_record.model_copy(
            update={"permission_id": "duplicate_permission"}
        )
        mismatched_sources_record = first_record.model_copy(
            update={"source_registries": ("other_registry", "agent_identity_registry")}
        )
        incomplete_record = first_record.model_copy(
            update={"missing_coverage_items": ("runtime_write_enabled",)}
        )

        with self.assertRaisesRegex(ValueError, "agent_ids must be unique"):
            BlackboardAuditRegistry(
                audit_records=(first_record, duplicate_record)
                + registry.audit_records[2:],
                agent_ids=registry.agent_ids,
                channel_ids=registry.channel_ids,
                audit_count=registry.audit_count,
                source_registries=registry.source_registries,
                validated_blackboard_surfaces=registry.validated_blackboard_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "source_registries"):
            BlackboardAuditRegistry(
                audit_records=(mismatched_sources_record,) + registry.audit_records[1:],
                agent_ids=registry.agent_ids,
                channel_ids=registry.channel_ids,
                audit_count=registry.audit_count,
                source_registries=registry.source_registries,
                validated_blackboard_surfaces=registry.validated_blackboard_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "missing coverage"):
            BlackboardAuditRegistry(
                audit_records=(incomplete_record,) + registry.audit_records[1:],
                agent_ids=registry.agent_ids,
                channel_ids=registry.channel_ids,
                audit_count=registry.audit_count,
                source_registries=registry.source_registries,
                validated_blackboard_surfaces=registry.validated_blackboard_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

    def test_blackboard_audit_does_not_change_memory_context_requests(self) -> None:
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

        blackboard_audit_registry()
        blackboard_audit_by_agent_id("planner_agent")
        blackboard_audit_by_channel_id("planner_agent_blackboard_channel")
        after = build_memory_context_request(request, route_decision)

        self.assertEqual(after, before)

    def test_audit_metadata_does_not_declare_storage_side_effects(self) -> None:
        registry = blackboard_audit_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for record in registry.audit_records
                    for field in (
                        record.channel_id,
                        record.permission_id,
                        *record.audit_findings,
                        *record.passive_boundary_flags,
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
