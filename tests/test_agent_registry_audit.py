import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    AgentRegistryAuditRegistry,
    agent_contract_registry,
    agent_registry_audit_by_registry_id,
    agent_registry_audit_registry,
    agent_registry_audits_for_kind,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_REGISTRY_IDS = (
    "agent_identity_registry",
    "agent_contract_registry",
    "agent_memory_contract_registry",
    "agent_role_registry",
    "agent_boundary_registry",
    "agent_metadata_registry",
    "agent_capability_registry",
    "agent_routing_registry",
    "agent_dependency_graph_registry",
    "parallel_scheduling_registry",
    "agent_coordination_registry",
    "agent_debate_registry",
    "consensus_builder_registry",
    "agent_capability_alignment_registry",
    "agent_escalation_signal_registry",
    "agent_lifecycle_registry",
    "agent_state_synchronization_registry",
    "workflow_agent_handoff_registry",
    "orchestration_contract_integration_registry",
    "agent_contract_audit_registry",
)

REQUIRED_AUDIT_ENTRY_FIELDS = {
    "registry_id",
    "registry_role",
    "registry_kind",
    "export_symbol",
    "registry_serialization_version",
    "linked_agent_ids",
    "source_registry_refs",
    "coverage_surfaces",
    "passive_boundary_flags",
    "audit_findings",
    "missing_coverage_items",
    "registry_blocked_runtime_behaviors",
    "blocked_runtime_behaviors",
    "audit_stage",
    "audit_status",
    "metadata_only_declared",
    "active_runtime_path_implemented",
    "agent_execution_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class AgentRegistryAuditTests(unittest.TestCase):
    def test_registry_audit_covers_agent_registry_surfaces(self) -> None:
        audit_registry = agent_registry_audit_registry()
        contract_registry = agent_contract_registry()

        self.assertEqual(audit_registry.role, "agent_registry_audit_registry")
        self.assertEqual(
            audit_registry.serialization_version,
            "agent_registry_audit_registry.v1",
        )
        self.assertEqual(audit_registry.audit_stage, "v4_6_agent_registry_hardening")
        self.assertEqual(audit_registry.registry_ids, EXPECTED_REGISTRY_IDS)
        self.assertEqual(audit_registry.audit_count, len(EXPECTED_REGISTRY_IDS))
        self.assertEqual(audit_registry.agent_ids, contract_registry.agent_ids)
        self.assertTrue(audit_registry.all_registries_covered)
        self.assertTrue(audit_registry.all_agent_ids_aligned)
        self.assertTrue(audit_registry.no_missing_coverage)
        self.assertTrue(audit_registry.metadata_only)
        self.assertFalse(audit_registry.active_runtime_audit_implemented)
        self.assertFalse(audit_registry.agent_execution_implemented)
        self.assertFalse(audit_registry.provider_model_routing_implemented)
        self.assertFalse(audit_registry.generated_output_mutation_implemented)
        self.assertIn("does not create agents", audit_registry.authority_boundary)

    def test_audit_entries_are_passive_and_agent_aligned(self) -> None:
        registry = agent_registry_audit_registry()
        known_agents = set(agent_contract_registry().agent_ids)

        for entry in registry.audit_entries:
            dumped = entry.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_AUDIT_ENTRY_FIELDS)
            self.assertEqual(
                entry.serialization_version,
                "agent_registry_audit_entry.v1",
            )
            self.assertEqual(entry.audit_status, "pass")
            self.assertEqual(entry.audit_stage, registry.audit_stage)
            self.assertEqual(entry.registry_id, entry.registry_role)
            self.assertEqual(entry.export_symbol, entry.registry_id)
            self.assertEqual(set(entry.linked_agent_ids), known_agents)
            self.assertEqual(entry.coverage_surfaces, registry.coverage_surfaces)
            self.assertEqual(entry.passive_boundary_flags, registry.passive_boundary_flags)
            self.assertFalse(entry.missing_coverage_items)
            self.assertTrue(
                {
                    "provider_or_model_routing",
                    "runtime_work_routing",
                    "dynamic_agent_routing",
                    "workflow_routing_change",
                }.intersection(entry.registry_blocked_runtime_behaviors)
            )
            self.assertIn(
                "generated_output_modification",
                entry.registry_blocked_runtime_behaviors,
            )
            self.assertTrue(entry.metadata_only_declared)
            self.assertTrue(entry.metadata_only)
            self.assertFalse(entry.active_runtime_path_implemented)
            self.assertFalse(entry.agent_execution_implemented)
            self.assertFalse(entry.provider_model_routing_implemented)
            self.assertFalse(entry.workflow_control_implemented)
            self.assertFalse(entry.retry_triggering_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)

    def test_audit_lookup_and_kind_filtering_are_stable(self) -> None:
        registry = agent_registry_audit_registry()
        contract_entry = agent_registry_audit_by_registry_id("agent_contract_registry")
        missing_entry = agent_registry_audit_by_registry_id("missing_registry")
        foundation_entries = agent_registry_audits_for_kind("v4_1_foundation")
        v4_2_entries = agent_registry_audits_for_kind("v4_2_agent_registry")
        missing_kind_entries = agent_registry_audits_for_kind("missing_kind")

        self.assertIsNone(missing_entry)
        self.assertIsNotNone(contract_entry)
        assert contract_entry is not None
        self.assertEqual(contract_entry.registry_kind, "v4_1_foundation")
        self.assertEqual(contract_entry.registry_serialization_version, "agent_contract_registry.v1")
        self.assertEqual(len(foundation_entries), 6)
        self.assertGreaterEqual(len(v4_2_entries), 10)
        self.assertEqual(missing_kind_entries, ())
        self.assertIs(
            contract_entry,
            agent_registry_audit_by_registry_id("agent_contract_registry", registry),
        )

    def test_registry_audit_rejects_mismatched_or_incomplete_entries(self) -> None:
        registry = agent_registry_audit_registry()
        first_entry = registry.audit_entries[0]
        duplicate_entry = first_entry.model_copy(
            update={"registry_serialization_version": "duplicate.v1"}
        )
        mismatched_flags_entry = first_entry.model_copy(
            update={
                "passive_boundary_flags": (
                    "other_flag",
                    "agent_execution_blocked",
                    "provider_model_routing_blocked",
                    "runtime_selection_blocked",
                    "retry_triggering_blocked",
                    "workflow_control_blocked",
                    "generated_output_mutation_blocked",
                )
            }
        )
        incomplete_entry = first_entry.model_copy(
            update={"missing_coverage_items": ("metadata_only_missing",)}
        )

        with self.assertRaisesRegex(ValueError, "registry_ids must be unique"):
            AgentRegistryAuditRegistry(
                audit_entries=(first_entry, duplicate_entry)
                + registry.audit_entries[2:],
                registry_ids=registry.registry_ids,
                agent_ids=registry.agent_ids,
                audit_count=registry.audit_count,
                coverage_surfaces=registry.coverage_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "passive_boundary_flags"):
            AgentRegistryAuditRegistry(
                audit_entries=(mismatched_flags_entry,) + registry.audit_entries[1:],
                registry_ids=registry.registry_ids,
                agent_ids=registry.agent_ids,
                audit_count=registry.audit_count,
                coverage_surfaces=registry.coverage_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "missing coverage"):
            AgentRegistryAuditRegistry(
                audit_entries=(incomplete_entry,) + registry.audit_entries[1:],
                registry_ids=registry.registry_ids,
                agent_ids=registry.agent_ids,
                audit_count=registry.audit_count,
                coverage_surfaces=registry.coverage_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

    def test_agent_registry_audit_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate a metadata-only creative coding sketch.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        agent_registry_audit_registry()
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)
        self.assertNotIn(
            "agent_registry_audit_registry",
            next_decision.model_dump_json(),
        )

    def test_audit_metadata_does_not_declare_active_agent_terms(self) -> None:
        registry = agent_registry_audit_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for entry in registry.audit_entries
                    for field in (
                        entry.registry_id,
                        entry.registry_kind,
                        *entry.audit_findings,
                        *entry.passive_boundary_flags,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_agent",
            "route_provider",
            "select_runtime",
            "trigger_retry",
            "write_memory",
            "modify_generated_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
