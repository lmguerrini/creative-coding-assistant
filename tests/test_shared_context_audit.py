import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    SharedContextAuditRegistry,
    route_request,
    shared_context_audit_by_agent_id,
    shared_context_audit_by_view_id,
    shared_context_audit_registry,
    shared_context_audits_for_source_registry,
    shared_context_view_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_AUDIT_RECORD_FIELDS = {
    "agent_id",
    "view_id",
    "audit_stage",
    "view_serialization_version",
    "access_mode",
    "visible_memory_surfaces",
    "visible_blackboard_channel_ids",
    "hidden_blackboard_channel_ids",
    "visible_metadata_keys",
    "source_memory_contract_id",
    "source_blackboard_registry",
    "source_registries",
    "validated_context_surfaces",
    "passive_boundary_flags",
    "audit_findings",
    "missing_coverage_items",
    "view_blocked_runtime_behaviors",
    "blocked_runtime_behaviors",
    "audit_status",
    "metadata_only_declared",
    "unrestricted_global_state_exposed",
    "runtime_memory_implemented",
    "context_materialization_implemented",
    "context_mutation_implemented",
    "storage_backend_implemented",
    "blackboard_state_access_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class SharedContextAuditTests(unittest.TestCase):
    def test_audit_registry_covers_shared_context_views(self) -> None:
        audit_registry = shared_context_audit_registry()
        view_registry = shared_context_view_registry()

        self.assertEqual(audit_registry.role, "shared_context_audit_registry")
        self.assertEqual(
            audit_registry.serialization_version,
            "shared_context_audit_registry.v1",
        )
        self.assertEqual(audit_registry.audit_stage, "v4_6_shared_context_hardening")
        self.assertEqual(audit_registry.agent_ids, view_registry.agent_ids)
        self.assertEqual(audit_registry.view_ids, view_registry.view_ids)
        self.assertEqual(
            audit_registry.blackboard_channel_ids,
            view_registry.blackboard_channel_ids,
        )
        self.assertEqual(audit_registry.audit_count, 12)
        self.assertEqual(
            audit_registry.source_registries,
            ("agent_memory_contract_registry", "blackboard_memory_registry"),
        )
        self.assertTrue(audit_registry.all_views_covered)
        self.assertTrue(audit_registry.scoped_visibility_confirmed)
        self.assertTrue(audit_registry.no_missing_coverage)
        self.assertTrue(audit_registry.metadata_only)
        self.assertFalse(audit_registry.active_runtime_audit_implemented)
        self.assertFalse(audit_registry.unrestricted_global_state_exposed)
        self.assertFalse(audit_registry.runtime_memory_implemented)
        self.assertFalse(audit_registry.context_materialization_implemented)
        self.assertFalse(audit_registry.context_mutation_implemented)
        self.assertFalse(audit_registry.storage_backend_implemented)
        self.assertIn(
            "does not expose unrestricted global state",
            audit_registry.authority_boundary,
        )

    def test_audit_records_are_passive_and_scoped(self) -> None:
        registry = shared_context_audit_registry()
        known_channels = set(registry.blackboard_channel_ids)

        for record in registry.audit_records:
            dumped = record.model_dump(mode="json")
            visible = set(record.visible_blackboard_channel_ids)
            hidden = set(record.hidden_blackboard_channel_ids)

            self.assertEqual(set(dumped), REQUIRED_AUDIT_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "shared_context_audit_record.v1",
            )
            self.assertEqual(record.audit_status, "pass")
            self.assertEqual(record.audit_stage, registry.audit_stage)
            self.assertEqual(record.source_registries, registry.source_registries)
            self.assertEqual(
                record.validated_context_surfaces,
                registry.validated_context_surfaces,
            )
            self.assertEqual(
                record.passive_boundary_flags, registry.passive_boundary_flags
            )
            self.assertFalse(record.missing_coverage_items)
            self.assertEqual(record.access_mode, "scoped_metadata_view")
            self.assertTrue(record.visible_memory_surfaces)
            self.assertTrue(record.visible_metadata_keys)
            self.assertLess(len(visible), len(known_channels))
            self.assertEqual(hidden, known_channels - visible)
            self.assertIn(
                "unrestricted_global_state_access",
                record.view_blocked_runtime_behaviors,
            )
            self.assertIn(
                "generated_output_modification",
                record.view_blocked_runtime_behaviors,
            )
            self.assertTrue(record.metadata_only_declared)
            self.assertTrue(record.metadata_only)
            self.assertFalse(record.unrestricted_global_state_exposed)
            self.assertFalse(record.runtime_memory_implemented)
            self.assertFalse(record.context_materialization_implemented)
            self.assertFalse(record.context_mutation_implemented)
            self.assertFalse(record.storage_backend_implemented)
            self.assertFalse(record.blackboard_state_access_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)

    def test_audit_lookup_and_source_filtering_are_stable(self) -> None:
        registry = shared_context_audit_registry()
        planner_audit = shared_context_audit_by_agent_id("planner_agent")
        style_audit = shared_context_audit_by_view_id("style_agent_shared_context_view")
        missing_audit = shared_context_audit_by_agent_id("missing_agent")
        blackboard_source_audits = shared_context_audits_for_source_registry(
            "blackboard_memory_registry"
        )
        missing_source_audits = shared_context_audits_for_source_registry(
            "missing_registry"
        )

        self.assertIsNone(missing_audit)
        self.assertIsNotNone(planner_audit)
        self.assertIsNotNone(style_audit)
        assert planner_audit is not None
        assert style_audit is not None
        self.assertEqual(planner_audit.view_id, "planner_agent_shared_context_view")
        self.assertIn(
            "planner_agent_blackboard_channel",
            planner_audit.visible_blackboard_channel_ids,
        )
        self.assertIn("visual_style_constraints", style_audit.visible_metadata_keys)
        self.assertEqual(len(blackboard_source_audits), registry.audit_count)
        self.assertEqual(missing_source_audits, ())
        self.assertIs(
            planner_audit,
            shared_context_audit_by_agent_id("planner_agent", registry),
        )

    def test_audit_registry_rejects_mismatched_or_incomplete_records(self) -> None:
        registry = shared_context_audit_registry()
        first_record = registry.audit_records[0]
        duplicate_record = first_record.model_copy(update={"view_id": "duplicate_view"})
        mismatched_surface_record = first_record.model_copy(
            update={
                "validated_context_surfaces": (
                    "other_surface",
                    "source_memory_contract_alignment",
                    "source_blackboard_registry_alignment",
                    "visible_blackboard_channels",
                    "hidden_blackboard_channels",
                    "visible_metadata_keys",
                    "runtime_materialization_flags",
                )
            }
        )
        incomplete_record = first_record.model_copy(
            update={"missing_coverage_items": ("context_mutation_enabled",)}
        )

        with self.assertRaisesRegex(ValueError, "agent_ids must be unique"):
            SharedContextAuditRegistry(
                audit_records=(first_record, duplicate_record)
                + registry.audit_records[2:],
                agent_ids=registry.agent_ids,
                view_ids=registry.view_ids,
                blackboard_channel_ids=registry.blackboard_channel_ids,
                audit_count=registry.audit_count,
                source_registries=registry.source_registries,
                validated_context_surfaces=registry.validated_context_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "validated_context_surfaces"):
            SharedContextAuditRegistry(
                audit_records=(mismatched_surface_record,) + registry.audit_records[1:],
                agent_ids=registry.agent_ids,
                view_ids=registry.view_ids,
                blackboard_channel_ids=registry.blackboard_channel_ids,
                audit_count=registry.audit_count,
                source_registries=registry.source_registries,
                validated_context_surfaces=registry.validated_context_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "missing coverage"):
            SharedContextAuditRegistry(
                audit_records=(incomplete_record,) + registry.audit_records[1:],
                agent_ids=registry.agent_ids,
                view_ids=registry.view_ids,
                blackboard_channel_ids=registry.blackboard_channel_ids,
                audit_count=registry.audit_count,
                source_registries=registry.source_registries,
                validated_context_surfaces=registry.validated_context_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

    def test_shared_context_audit_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate a scoped shared-context sketch.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        shared_context_audit_registry()
        shared_context_audit_by_agent_id("planner_agent")
        shared_context_audit_by_view_id("planner_agent_shared_context_view")
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)
        self.assertNotIn(
            "shared_context_audit_registry",
            next_decision.model_dump_json(),
        )

    def test_audit_metadata_does_not_declare_context_mutation_terms(self) -> None:
        registry = shared_context_audit_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for record in registry.audit_records
                    for field in (
                        record.view_id,
                        record.access_mode,
                        *record.audit_findings,
                        *record.passive_boundary_flags,
                    )
                ),
            )
        )

        for forbidden_term in (
            "global_state_dump",
            "runtime_memory_write",
            "context_mutation_trigger",
            "storage_adapter",
            "execute_provider",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
