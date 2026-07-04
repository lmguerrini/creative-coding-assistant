import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    AgentExplainabilityAuditRegistry,
    agent_contract_audit_registry,
    agent_contract_registry,
    agent_explainability_audit_by_agent_id,
    agent_explainability_audit_registry,
    agent_explainability_audits_for_memory_source,
    agent_explainability_audits_for_source_registry,
    decision_provenance_registry,
    escalation_trace_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_AUDIT_RECORD_FIELDS = {
    "agent_id",
    "role_id",
    "audit_stage",
    "contract_serialization_version",
    "contract_source_registry_refs",
    "memory_reference_sources",
    "explanation_metadata_keys",
    "explanation_signal_keys",
    "explanation_output_contracts",
    "decision_provenance_profile_ids",
    "escalation_trace_profile_ids",
    "validated_explainability_surfaces",
    "passive_boundary_flags",
    "audit_findings",
    "missing_coverage_items",
    "contract_blocked_runtime_behaviors",
    "blocked_runtime_behaviors",
    "audit_status",
    "metadata_only_declared",
    "provenance_memory_reference_present",
    "active_explanation_generation_implemented",
    "active_agent_execution_implemented",
    "provider_model_routing_implemented",
    "runtime_selection_implemented",
    "trace_capture_implemented",
    "memory_write_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


def _contract_source_refs() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            source_ref
            for contract in agent_contract_registry().contracts
            for source_ref in contract.source_contract_registries
        )
    )


def _contract_memory_sources() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            memory_source
            for contract in agent_contract_registry().contracts
            for memory_source in contract.memory_access.allowed_memory_sources
        )
    )


class AgentExplainabilityAuditTests(unittest.TestCase):
    def test_audit_registry_covers_agent_explainability_sources(self) -> None:
        registry = agent_explainability_audit_registry()
        contract_registry = agent_contract_registry()

        self.assertEqual(registry.role, "agent_explainability_audit_registry")
        self.assertEqual(
            registry.serialization_version,
            "agent_explainability_audit_registry.v1",
        )
        self.assertEqual(
            registry.audit_stage,
            "v4_6_agent_explainability_hardening",
        )
        self.assertEqual(registry.agent_ids, contract_registry.agent_ids)
        self.assertEqual(registry.agent_ids, agent_contract_audit_registry().agent_ids)
        self.assertEqual(registry.audit_count, 12)
        self.assertEqual(registry.source_contract_registry, "agent_contract_registry")
        self.assertEqual(
            registry.source_contract_audit_registry,
            "agent_contract_audit_registry",
        )
        self.assertEqual(registry.source_registry_refs, _contract_source_refs())
        self.assertEqual(registry.memory_reference_sources, _contract_memory_sources())
        self.assertEqual(
            registry.decision_provenance_profile_ids,
            decision_provenance_registry().provenance_profile_ids,
        )
        self.assertEqual(
            registry.escalation_trace_profile_ids,
            escalation_trace_registry().trace_profile_ids,
        )
        self.assertTrue(registry.all_agent_contracts_covered)
        self.assertTrue(registry.all_records_provenance_referenced)
        self.assertTrue(registry.no_missing_coverage)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.active_explanation_generation_implemented)
        self.assertFalse(registry.active_agent_execution_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.trace_capture_implemented)
        self.assertFalse(registry.memory_write_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertIn("does not generate explanations", registry.authority_boundary)

    def test_audit_records_are_passive_and_explainability_aligned(self) -> None:
        registry = agent_explainability_audit_registry()
        known_sources = set(registry.source_registry_refs)
        known_memory_sources = set(registry.memory_reference_sources)

        for record in registry.audit_records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_AUDIT_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "agent_explainability_audit_record.v1",
            )
            self.assertEqual(record.audit_status, "pass")
            self.assertEqual(record.audit_stage, registry.audit_stage)
            self.assertTrue(
                set(record.contract_source_registry_refs).issubset(known_sources)
            )
            self.assertTrue(
                set(record.memory_reference_sources).issubset(known_memory_sources)
            )
            self.assertEqual(
                record.decision_provenance_profile_ids,
                registry.decision_provenance_profile_ids,
            )
            self.assertEqual(
                record.escalation_trace_profile_ids,
                registry.escalation_trace_profile_ids,
            )
            self.assertEqual(
                record.validated_explainability_surfaces,
                registry.validated_explainability_surfaces,
            )
            self.assertEqual(
                record.passive_boundary_flags, registry.passive_boundary_flags
            )
            self.assertFalse(record.missing_coverage_items)
            self.assertTrue(record.explanation_metadata_keys)
            self.assertTrue(record.explanation_signal_keys)
            self.assertTrue(record.explanation_output_contracts)
            self.assertIn("provenance_metadata", record.memory_reference_sources)
            self.assertIn(
                "provider_or_model_routing",
                record.contract_blocked_runtime_behaviors,
            )
            self.assertIn(
                "generated_output_modification",
                record.contract_blocked_runtime_behaviors,
            )
            self.assertTrue(record.provenance_memory_reference_present)
            self.assertTrue(record.metadata_only_declared)
            self.assertTrue(record.metadata_only)
            self.assertFalse(record.active_explanation_generation_implemented)
            self.assertFalse(record.active_agent_execution_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.runtime_selection_implemented)
            self.assertFalse(record.trace_capture_implemented)
            self.assertFalse(record.memory_write_implemented)
            self.assertFalse(record.retry_triggering_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)

    def test_audit_lookup_source_and_memory_filtering_are_stable(self) -> None:
        registry = agent_explainability_audit_registry()
        research_audit = agent_explainability_audit_by_agent_id("research_agent")
        missing_audit = agent_explainability_audit_by_agent_id("missing_agent")
        memory_contract_audits = agent_explainability_audits_for_source_registry(
            "agent_memory_contract_registry"
        )
        provenance_memory_audits = agent_explainability_audits_for_memory_source(
            "provenance_metadata"
        )
        missing_source_audits = agent_explainability_audits_for_source_registry(
            "missing_registry"
        )

        self.assertIsNone(missing_audit)
        self.assertIsNotNone(research_audit)
        assert research_audit is not None
        self.assertEqual(research_audit.role_id, "research")
        self.assertIn(
            "evidence_summary_metadata", research_audit.explanation_metadata_keys
        )
        self.assertIn("evidence_density", research_audit.explanation_signal_keys)
        self.assertEqual(len(memory_contract_audits), registry.audit_count)
        self.assertEqual(len(provenance_memory_audits), registry.audit_count)
        self.assertEqual(missing_source_audits, ())
        self.assertIs(
            research_audit,
            agent_explainability_audit_by_agent_id("research_agent", registry),
        )

    def test_audit_registry_rejects_mismatched_or_incomplete_records(self) -> None:
        registry = agent_explainability_audit_registry()
        first_record = registry.audit_records[0]
        duplicate_record = first_record.model_copy(update={"role_id": "duplicate"})
        mismatched_surface_record = first_record.model_copy(
            update={
                "validated_explainability_surfaces": (
                    "other_surface",
                    "produced_signals",
                    "produced_outputs",
                    "source_registry_refs",
                    "memory_reference_sources",
                    "contract_audit_alignment",
                    "provenance_trace_registry_refs",
                    "blocked_runtime_behaviors",
                )
            }
        )
        incomplete_record = first_record.model_copy(
            update={"missing_coverage_items": ("provenance_reference_missing",)}
        )

        with self.assertRaisesRegex(ValueError, "agent_ids must be unique"):
            AgentExplainabilityAuditRegistry(
                audit_records=(first_record, duplicate_record)
                + registry.audit_records[2:],
                agent_ids=registry.agent_ids,
                audit_count=registry.audit_count,
                source_registry_refs=registry.source_registry_refs,
                memory_reference_sources=registry.memory_reference_sources,
                decision_provenance_profile_ids=registry.decision_provenance_profile_ids,
                escalation_trace_profile_ids=registry.escalation_trace_profile_ids,
                validated_explainability_surfaces=(
                    registry.validated_explainability_surfaces
                ),
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "validated_explainability_surfaces"):
            AgentExplainabilityAuditRegistry(
                audit_records=(mismatched_surface_record,) + registry.audit_records[1:],
                agent_ids=registry.agent_ids,
                audit_count=registry.audit_count,
                source_registry_refs=registry.source_registry_refs,
                memory_reference_sources=registry.memory_reference_sources,
                decision_provenance_profile_ids=registry.decision_provenance_profile_ids,
                escalation_trace_profile_ids=registry.escalation_trace_profile_ids,
                validated_explainability_surfaces=(
                    registry.validated_explainability_surfaces
                ),
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "missing coverage"):
            AgentExplainabilityAuditRegistry(
                audit_records=(incomplete_record,) + registry.audit_records[1:],
                agent_ids=registry.agent_ids,
                audit_count=registry.audit_count,
                source_registry_refs=registry.source_registry_refs,
                memory_reference_sources=registry.memory_reference_sources,
                decision_provenance_profile_ids=registry.decision_provenance_profile_ids,
                escalation_trace_profile_ids=registry.escalation_trace_profile_ids,
                validated_explainability_surfaces=(
                    registry.validated_explainability_surfaces
                ),
                passive_boundary_flags=registry.passive_boundary_flags,
            )

    def test_agent_explainability_audit_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Explain the reasoning for a visual sketch.",
            mode=AssistantMode.EXPLAIN,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        agent_explainability_audit_registry()
        agent_explainability_audit_by_agent_id("planner_agent")
        agent_explainability_audits_for_memory_source("provenance_metadata")
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.EXPLAIN)
        self.assertNotIn(
            "agent_explainability_audit_registry",
            next_decision.model_dump_json(),
        )

    def test_audit_metadata_does_not_declare_active_explanation_terms(self) -> None:
        registry = agent_explainability_audit_registry()
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
            "generate_explanation",
            "execute_agent",
            "route_provider",
            "select_runtime",
            "capture_runtime_trace",
            "write_memory",
            "modify_generated_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
