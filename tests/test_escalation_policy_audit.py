import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    EscalationPolicyAuditRegistry,
    escalation_policy_audit_by_rule_id,
    escalation_policy_audit_registry,
    escalation_policy_audits_for_downstream_registry,
    escalation_policy_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_RULE_IDS = (
    "missing_information_review",
    "artifact_risk_review",
    "runtime_incompatibility_review",
    "evaluation_confidence_review",
    "future_agent_escalation_readiness",
)

EXPECTED_AUDITED_REGISTRY_REFS = (
    "escalation_policy_registry",
    "agent_escalation_signal_registry",
    "conditional_multi_agent_escalation_registry",
    "escalation_gate_registry",
    "creative_escalation_policy_registry",
    "reflection_escalation_registry",
    "escalation_trace_registry",
    "hitl_escalation_gate_registry",
    "ambiguity_escalation_registry",
    "risk_escalation_registry",
    "quality_escalation_registry",
    "adaptive_multi_agent_escalation_registry",
    "hybrid_agentic_workflow_registry",
)

REQUIRED_AUDIT_RECORD_FIELDS = {
    "rule_id",
    "policy_stage",
    "rule_serialization_version",
    "audit_stage",
    "audited_registry_refs",
    "rule_source_contract_registries",
    "downstream_registry_refs",
    "trigger_signals",
    "evidence_sources",
    "validated_policy_surfaces",
    "passive_boundary_flags",
    "audit_findings",
    "missing_coverage_items",
    "rule_blocked_runtime_behaviors",
    "blocked_runtime_behaviors",
    "audit_status",
    "metadata_only_declared",
    "policy_evaluation_implemented",
    "escalation_triggering_implemented",
    "agent_invocation_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class EscalationPolicyAuditTests(unittest.TestCase):
    def test_audit_registry_covers_all_escalation_policy_rules(self) -> None:
        audit_registry = escalation_policy_audit_registry()
        policy_registry = escalation_policy_registry()

        self.assertEqual(audit_registry.role, "escalation_policy_audit_registry")
        self.assertEqual(
            audit_registry.serialization_version,
            "escalation_policy_audit_registry.v1",
        )
        self.assertEqual(
            audit_registry.audit_stage,
            "v4_6_escalation_policy_hardening",
        )
        self.assertEqual(audit_registry.rule_ids, EXPECTED_RULE_IDS)
        self.assertEqual(audit_registry.rule_ids, policy_registry.rule_ids)
        self.assertEqual(audit_registry.audit_count, policy_registry.rule_count)
        self.assertEqual(
            audit_registry.audited_registry_refs,
            EXPECTED_AUDITED_REGISTRY_REFS,
        )
        self.assertEqual(
            audit_registry.source_contract_registries,
            policy_registry.source_contract_registries,
        )
        self.assertTrue(audit_registry.all_policy_rules_covered)
        self.assertTrue(audit_registry.no_missing_coverage)
        self.assertTrue(audit_registry.metadata_only)
        self.assertFalse(audit_registry.policy_evaluation_implemented)
        self.assertFalse(audit_registry.escalation_triggering_implemented)
        self.assertFalse(audit_registry.provider_model_routing_implemented)
        self.assertFalse(audit_registry.generated_output_mutation_implemented)
        self.assertIn("does not evaluate policy", audit_registry.authority_boundary)

    def test_audit_records_are_passive_and_downstream_linked(self) -> None:
        registry = escalation_policy_audit_registry()

        for record in registry.audit_records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_AUDIT_RECORD_FIELDS)
            self.assertEqual(record.serialization_version, "escalation_policy_audit.v1")
            self.assertEqual(record.audit_status, "pass")
            self.assertEqual(record.audit_stage, registry.audit_stage)
            self.assertEqual(
                record.audited_registry_refs, registry.audited_registry_refs
            )
            self.assertEqual(
                record.validated_policy_surfaces,
                registry.validated_policy_surfaces,
            )
            self.assertEqual(
                record.passive_boundary_flags, registry.passive_boundary_flags
            )
            self.assertFalse(record.missing_coverage_items)
            self.assertTrue(record.rule_source_contract_registries)
            self.assertTrue(record.downstream_registry_refs)
            self.assertIn(
                "hybrid_agentic_workflow_registry",
                record.downstream_registry_refs,
            )
            self.assertIn(
                "provider_or_model_routing",
                record.rule_blocked_runtime_behaviors,
            )
            self.assertIn(
                "generated_output_modification",
                record.rule_blocked_runtime_behaviors,
            )
            self.assertTrue(record.metadata_only_declared)
            self.assertTrue(record.metadata_only)
            self.assertFalse(record.policy_evaluation_implemented)
            self.assertFalse(record.escalation_triggering_implemented)
            self.assertFalse(record.agent_invocation_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.retry_triggering_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)

    def test_audit_lookup_and_downstream_filtering_are_stable(self) -> None:
        registry = escalation_policy_audit_registry()
        future_audit = escalation_policy_audit_by_rule_id(
            "future_agent_escalation_readiness"
        )
        missing_audit = escalation_policy_audit_by_rule_id("missing_rule")
        workflow_audits = escalation_policy_audits_for_downstream_registry(
            "hybrid_agentic_workflow_registry"
        )
        missing_registry_audits = escalation_policy_audits_for_downstream_registry(
            "missing_registry"
        )

        self.assertIsNone(missing_audit)
        self.assertIsNotNone(future_audit)
        assert future_audit is not None
        self.assertEqual(future_audit.policy_stage, "future_agent_advisory")
        self.assertIn(
            "agent_escalation_signal_registry",
            future_audit.downstream_registry_refs,
        )
        self.assertEqual(len(workflow_audits), registry.audit_count)
        self.assertEqual(missing_registry_audits, ())
        self.assertIs(
            future_audit,
            escalation_policy_audit_by_rule_id(
                "future_agent_escalation_readiness",
                registry,
            ),
        )

    def test_audit_registry_rejects_mismatched_or_incomplete_records(self) -> None:
        registry = escalation_policy_audit_registry()
        first_record = registry.audit_records[0]
        duplicate_record = first_record.model_copy(update={"policy_stage": "duplicate"})
        mismatched_surface_record = first_record.model_copy(
            update={
                "validated_policy_surfaces": (
                    "other_surface",
                    "policy_stage",
                    "source_contract_registries",
                    "trigger_signals",
                    "evidence_sources",
                    "advisory_outcome",
                    "blocked_runtime_behaviors",
                )
            }
        )
        incomplete_record = first_record.model_copy(
            update={"missing_coverage_items": ("downstream_reference_missing",)}
        )

        with self.assertRaisesRegex(ValueError, "rule_ids must be unique"):
            EscalationPolicyAuditRegistry(
                audit_records=(first_record, duplicate_record)
                + registry.audit_records[2:],
                rule_ids=registry.rule_ids,
                audit_count=registry.audit_count,
                audited_registry_refs=registry.audited_registry_refs,
                source_contract_registries=registry.source_contract_registries,
                validated_policy_surfaces=registry.validated_policy_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "validated_policy_surfaces"):
            EscalationPolicyAuditRegistry(
                audit_records=(mismatched_surface_record,) + registry.audit_records[1:],
                rule_ids=registry.rule_ids,
                audit_count=registry.audit_count,
                audited_registry_refs=registry.audited_registry_refs,
                source_contract_registries=registry.source_contract_registries,
                validated_policy_surfaces=registry.validated_policy_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "missing coverage"):
            EscalationPolicyAuditRegistry(
                audit_records=(incomplete_record,) + registry.audit_records[1:],
                rule_ids=registry.rule_ids,
                audit_count=registry.audit_count,
                audited_registry_refs=registry.audited_registry_refs,
                source_contract_registries=registry.source_contract_registries,
                validated_policy_surfaces=registry.validated_policy_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

    def test_escalation_policy_audit_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate an interactive shader sketch.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        escalation_policy_audit_registry()
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)
        self.assertNotIn(
            "escalation_policy_audit_registry",
            next_decision.model_dump_json(),
        )

    def test_audit_metadata_does_not_declare_active_escalation_terms(self) -> None:
        registry = escalation_policy_audit_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for record in registry.audit_records
                    for field in (
                        record.rule_id,
                        record.policy_stage,
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
