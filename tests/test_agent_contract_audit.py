import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    AgentContractAuditRegistry,
    agent_contract_audit_by_agent_id,
    agent_contract_audit_registry,
    agent_contract_audits_for_registry_ref,
    agent_contract_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_AGENT_IDS = (
    "planner_agent",
    "research_agent",
    "style_agent",
    "runtime_agent",
    "artifact_agent",
    "art_direction_agent",
    "aesthetic_critic_agent",
    "narrative_symbolic_agent",
    "creative_curator_agent",
    "critic_agent",
    "refiner_agent",
    "final_synthesizer_agent",
)

EXPECTED_AUDITED_REGISTRY_REFS = (
    "agent_contract_registry",
    "agent_role_registry",
    "agent_boundary_registry",
    "agent_metadata_registry",
    "agent_memory_contract_registry",
)

REQUIRED_AUDIT_RECORD_FIELDS = {
    "agent_id",
    "role_id",
    "contract_serialization_version",
    "audit_stage",
    "audited_registry_refs",
    "contract_source_registry_refs",
    "validated_contract_surfaces",
    "passive_boundary_flags",
    "audit_findings",
    "missing_coverage_items",
    "contract_blocked_runtime_behaviors",
    "blocked_runtime_behaviors",
    "audit_status",
    "metadata_only_declared",
    "active_agent_execution_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "memory_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class AgentContractAuditTests(unittest.TestCase):
    def test_audit_registry_covers_all_agent_contracts(self) -> None:
        registry = agent_contract_audit_registry()
        contract_registry = agent_contract_registry()

        self.assertEqual(registry.role, "agent_contract_audit_registry")
        self.assertEqual(
            registry.serialization_version,
            "agent_contract_audit_registry.v1",
        )
        self.assertEqual(registry.audit_stage, "v4_6_agent_contract_hardening")
        self.assertEqual(registry.agent_ids, EXPECTED_AGENT_IDS)
        self.assertEqual(registry.agent_ids, contract_registry.agent_ids)
        self.assertEqual(registry.audit_count, 12)
        self.assertEqual(registry.audited_registry_refs, EXPECTED_AUDITED_REGISTRY_REFS)
        self.assertTrue(registry.all_contracts_covered)
        self.assertTrue(registry.no_missing_coverage)
        self.assertFalse(registry.active_runtime_audit_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertTrue(registry.metadata_only)
        self.assertIn("does not execute agents", registry.authority_boundary)

    def test_audit_records_are_passive_and_complete(self) -> None:
        registry = agent_contract_audit_registry()

        for record in registry.audit_records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_AUDIT_RECORD_FIELDS)
            self.assertEqual(record.serialization_version, "agent_contract_audit.v1")
            self.assertEqual(record.audit_status, "pass")
            self.assertEqual(record.audit_stage, registry.audit_stage)
            self.assertEqual(record.audited_registry_refs, registry.audited_registry_refs)
            self.assertEqual(
                record.validated_contract_surfaces,
                registry.validated_contract_surfaces,
            )
            self.assertEqual(record.passive_boundary_flags, registry.passive_boundary_flags)
            self.assertFalse(record.missing_coverage_items)
            self.assertTrue(record.contract_source_registry_refs)
            self.assertIn(
                "provider_or_model_routing",
                record.contract_blocked_runtime_behaviors,
            )
            self.assertIn(
                "generated_output_modification",
                record.contract_blocked_runtime_behaviors,
            )
            self.assertTrue(record.metadata_only_declared)
            self.assertTrue(record.metadata_only)
            self.assertFalse(record.active_agent_execution_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.retry_triggering_implemented)
            self.assertFalse(record.memory_mutation_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)

    def test_audit_lookup_and_registry_ref_filtering_are_stable(self) -> None:
        registry = agent_contract_audit_registry()
        planner_audit = agent_contract_audit_by_agent_id("planner_agent")
        missing_audit = agent_contract_audit_by_agent_id("missing_agent")
        role_ref_audits = agent_contract_audits_for_registry_ref("agent_role_registry")
        missing_ref_audits = agent_contract_audits_for_registry_ref("missing_registry")

        self.assertIsNone(missing_audit)
        self.assertIsNotNone(planner_audit)
        assert planner_audit is not None
        self.assertEqual(planner_audit.role_id, "planner")
        self.assertEqual(len(role_ref_audits), registry.audit_count)
        self.assertEqual(missing_ref_audits, ())
        self.assertIs(
            planner_audit,
            agent_contract_audit_by_agent_id("planner_agent", registry),
        )

    def test_audit_registry_rejects_mismatched_or_incomplete_records(self) -> None:
        registry = agent_contract_audit_registry()
        first_record = registry.audit_records[0]
        duplicate_record = first_record.model_copy(update={"role_id": "duplicate"})
        mismatched_ref_record = first_record.model_copy(
            update={
                "audited_registry_refs": (
                    "other_registry",
                    "agent_role_registry",
                    "agent_boundary_registry",
                    "agent_metadata_registry",
                    "agent_memory_contract_registry",
                )
            }
        )
        incomplete_record = first_record.model_copy(
            update={"missing_coverage_items": ("missing_boundary_registry",)}
        )

        with self.assertRaisesRegex(ValueError, "agent_ids must be unique"):
            AgentContractAuditRegistry(
                audit_records=(first_record, duplicate_record)
                + registry.audit_records[2:],
                agent_ids=registry.agent_ids,
                audit_count=registry.audit_count,
                audited_registry_refs=registry.audited_registry_refs,
                validated_contract_surfaces=registry.validated_contract_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "audited_registry_refs must match"):
            AgentContractAuditRegistry(
                audit_records=(mismatched_ref_record,) + registry.audit_records[1:],
                agent_ids=registry.agent_ids,
                audit_count=registry.audit_count,
                audited_registry_refs=registry.audited_registry_refs,
                validated_contract_surfaces=registry.validated_contract_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "missing coverage"):
            AgentContractAuditRegistry(
                audit_records=(incomplete_record,) + registry.audit_records[1:],
                agent_ids=registry.agent_ids,
                audit_count=registry.audit_count,
                audited_registry_refs=registry.audited_registry_refs,
                validated_contract_surfaces=registry.validated_contract_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

    def test_agent_contract_audit_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate a p5.js particle field.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        agent_contract_audit_registry()
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)
        self.assertNotIn(
            "agent_contract_audit_registry",
            next_decision.model_dump_json(),
        )

    def test_audit_metadata_does_not_declare_active_runtime_terms(self) -> None:
        registry = agent_contract_audit_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for record in registry.audit_records
                    for field in (
                        record.agent_id,
                        record.role_id,
                        *record.audit_findings,
                        *record.passive_boundary_flags,
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
