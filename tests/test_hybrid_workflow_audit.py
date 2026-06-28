import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    HybridWorkflowAuditRegistry,
    hybrid_agentic_workflow_registry,
    hybrid_workflow_audit_by_stage_id,
    hybrid_workflow_audit_registry,
    hybrid_workflow_audits_for_source_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_STAGE_IDS = (
    "intake_routing_context_readiness",
    "planning_reasoning_readiness",
    "generation_artifact_readiness",
    "review_refinement_readiness",
    "completion_guardrail_readiness",
)

REQUIRED_AUDIT_RECORD_FIELDS = {
    "stage_id",
    "stage_name",
    "stage_serialization_version",
    "audit_stage",
    "v3_workflow_nodes",
    "future_capability_ids",
    "escalation_rule_ids",
    "source_metadata_registries",
    "advisory_outputs",
    "validated_stage_surfaces",
    "passive_boundary_flags",
    "audit_findings",
    "missing_coverage_items",
    "stage_blocked_runtime_behaviors",
    "blocked_runtime_behaviors",
    "audit_status",
    "metadata_only_declared",
    "workflow_order_mutation_implemented",
    "agent_execution_implemented",
    "provider_model_routing_implemented",
    "runtime_selection_implemented",
    "retry_triggering_implemented",
    "artifact_execution_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class HybridWorkflowAuditTests(unittest.TestCase):
    def test_audit_registry_covers_hybrid_workflow_stages_and_sources(self) -> None:
        audit_registry = hybrid_workflow_audit_registry()
        workflow_registry = hybrid_agentic_workflow_registry()

        self.assertEqual(audit_registry.role, "hybrid_workflow_audit_registry")
        self.assertEqual(
            audit_registry.serialization_version,
            "hybrid_workflow_audit_registry.v1",
        )
        self.assertEqual(audit_registry.audit_stage, "v4_6_hybrid_workflow_hardening")
        self.assertEqual(audit_registry.stage_ids, EXPECTED_STAGE_IDS)
        self.assertEqual(audit_registry.stage_ids, workflow_registry.stage_ids)
        self.assertEqual(audit_registry.audit_count, workflow_registry.stage_count)
        self.assertEqual(
            audit_registry.source_metadata_registries,
            workflow_registry.source_metadata_registries,
        )
        self.assertEqual(len(audit_registry.source_metadata_registries), 43)
        self.assertIn(
            "adaptive_multi_agent_escalation_registry",
            audit_registry.source_metadata_registries,
        )
        self.assertIn(
            "hybrid_agentic_workflow_registry",
            audit_registry.source_metadata_registries,
        )
        self.assertTrue(audit_registry.all_stages_covered)
        self.assertTrue(audit_registry.all_sources_covered)
        self.assertTrue(audit_registry.no_missing_coverage)
        self.assertTrue(audit_registry.metadata_only)
        self.assertFalse(audit_registry.active_runtime_audit_implemented)
        self.assertFalse(audit_registry.workflow_order_mutation_implemented)
        self.assertFalse(audit_registry.agent_execution_implemented)
        self.assertFalse(audit_registry.provider_model_routing_implemented)
        self.assertFalse(audit_registry.generated_output_mutation_implemented)
        self.assertIn("does not change workflow graph", audit_registry.authority_boundary)

    def test_audit_records_are_passive_and_stage_aligned(self) -> None:
        registry = hybrid_workflow_audit_registry()

        for record in registry.audit_records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_AUDIT_RECORD_FIELDS)
            self.assertEqual(record.serialization_version, "hybrid_workflow_audit.v1")
            self.assertEqual(record.audit_status, "pass")
            self.assertEqual(record.audit_stage, registry.audit_stage)
            self.assertEqual(
                record.source_metadata_registries,
                registry.source_metadata_registries,
            )
            self.assertEqual(
                record.validated_stage_surfaces,
                registry.validated_stage_surfaces,
            )
            self.assertEqual(record.passive_boundary_flags, registry.passive_boundary_flags)
            self.assertFalse(record.missing_coverage_items)
            self.assertTrue(record.v3_workflow_nodes)
            self.assertTrue(record.future_capability_ids)
            self.assertTrue(record.escalation_rule_ids)
            self.assertTrue(record.advisory_outputs)
            self.assertIn(
                "provider_or_model_routing",
                record.stage_blocked_runtime_behaviors,
            )
            self.assertIn(
                "generated_output_modification",
                record.stage_blocked_runtime_behaviors,
            )
            self.assertTrue(record.metadata_only_declared)
            self.assertTrue(record.metadata_only)
            self.assertFalse(record.workflow_order_mutation_implemented)
            self.assertFalse(record.agent_execution_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.runtime_selection_implemented)
            self.assertFalse(record.retry_triggering_implemented)
            self.assertFalse(record.artifact_execution_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)

    def test_audit_lookup_and_source_filtering_are_stable(self) -> None:
        registry = hybrid_workflow_audit_registry()
        generation_audit = hybrid_workflow_audit_by_stage_id(
            "generation_artifact_readiness"
        )
        missing_audit = hybrid_workflow_audit_by_stage_id("missing_stage")
        agent_contract_audits = hybrid_workflow_audits_for_source_registry(
            "agent_contract_registry"
        )
        missing_source_audits = hybrid_workflow_audits_for_source_registry(
            "missing_registry"
        )

        self.assertIsNone(missing_audit)
        self.assertIsNotNone(generation_audit)
        assert generation_audit is not None
        self.assertIn("generation", generation_audit.v3_workflow_nodes)
        self.assertIn("v4_artifact_agent", generation_audit.future_capability_ids)
        self.assertIn(
            "runtime_incompatibility_review",
            generation_audit.escalation_rule_ids,
        )
        self.assertEqual(len(agent_contract_audits), registry.audit_count)
        self.assertEqual(missing_source_audits, ())
        self.assertIs(
            generation_audit,
            hybrid_workflow_audit_by_stage_id(
                "generation_artifact_readiness",
                registry,
            ),
        )

    def test_audit_registry_rejects_mismatched_or_incomplete_records(self) -> None:
        registry = hybrid_workflow_audit_registry()
        first_record = registry.audit_records[0]
        duplicate_record = first_record.model_copy(update={"stage_name": "Duplicate"})
        mismatched_source_record = first_record.model_copy(
            update={
                "source_metadata_registries": (
                    "other_registry",
                )
                + first_record.source_metadata_registries[1:]
            }
        )
        incomplete_record = first_record.model_copy(
            update={"missing_coverage_items": ("unknown_v3_workflow_node",)}
        )

        with self.assertRaisesRegex(ValueError, "stage_ids must be unique"):
            HybridWorkflowAuditRegistry(
                audit_records=(first_record, duplicate_record)
                + registry.audit_records[2:],
                stage_ids=registry.stage_ids,
                audit_count=registry.audit_count,
                source_metadata_registries=registry.source_metadata_registries,
                validated_stage_surfaces=registry.validated_stage_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "source_metadata_registries"):
            HybridWorkflowAuditRegistry(
                audit_records=(mismatched_source_record,) + registry.audit_records[1:],
                stage_ids=registry.stage_ids,
                audit_count=registry.audit_count,
                source_metadata_registries=registry.source_metadata_registries,
                validated_stage_surfaces=registry.validated_stage_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "missing coverage"):
            HybridWorkflowAuditRegistry(
                audit_records=(incomplete_record,) + registry.audit_records[1:],
                stage_ids=registry.stage_ids,
                audit_count=registry.audit_count,
                source_metadata_registries=registry.source_metadata_registries,
                validated_stage_surfaces=registry.validated_stage_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

    def test_hybrid_workflow_audit_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate a browser-based creative coding study.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        hybrid_workflow_audit_registry()
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)
        self.assertNotIn(
            "hybrid_workflow_audit_registry",
            next_decision.model_dump_json(),
        )

    def test_audit_metadata_does_not_declare_active_hybrid_terms(self) -> None:
        registry = hybrid_workflow_audit_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for record in registry.audit_records
                    for field in (
                        record.stage_id,
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
