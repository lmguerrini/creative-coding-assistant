import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    AgentCollaborationAuditRegistry,
    agent_collaboration_audit_by_registry_id,
    agent_collaboration_audit_registry,
    agent_collaboration_audits_for_source_registry,
    agent_collaboration_audits_for_surface,
    agent_contract_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_REGISTRY_IDS = (
    "agent_coordination_registry",
    "agent_debate_registry",
    "consensus_builder_registry",
    "workflow_agent_handoff_registry",
)
EXPECTED_COLLABORATION_SURFACES = (
    "coordination",
    "debate",
    "consensus",
    "workflow_handoff",
)
EXPECTED_SOURCE_REGISTRY_REFS = (
    "parallel_scheduling_registry",
    "agent_dependency_graph_registry",
    "agent_contract_registry",
    "shared_context_view_registry",
    "agent_debate_registry",
    "workflow_state",
    "agent_role_registry",
)

REQUIRED_AUDIT_RECORD_FIELDS = {
    "registry_id",
    "registry_role",
    "collaboration_surface",
    "registry_serialization_version",
    "source_registry_refs",
    "participant_agent_ids",
    "collaboration_contract_ids",
    "collaboration_contract_count",
    "validated_collaboration_surfaces",
    "passive_boundary_flags",
    "audit_findings",
    "missing_coverage_items",
    "registry_blocked_runtime_behaviors",
    "blocked_runtime_behaviors",
    "audit_stage",
    "audit_status",
    "metadata_only_declared",
    "live_coordination_implemented",
    "debate_execution_implemented",
    "voting_execution_implemented",
    "runtime_handoff_implemented",
    "agent_invocation_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "state_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class AgentCollaborationAuditTests(unittest.TestCase):
    def test_audit_registry_covers_collaboration_registries_and_sources(self) -> None:
        audit_registry = agent_collaboration_audit_registry()
        contract_registry = agent_contract_registry()

        self.assertEqual(audit_registry.role, "agent_collaboration_audit_registry")
        self.assertEqual(
            audit_registry.serialization_version,
            "agent_collaboration_audit_registry.v1",
        )
        self.assertEqual(
            audit_registry.audit_stage,
            "v4_6_agent_collaboration_hardening",
        )
        self.assertEqual(audit_registry.registry_ids, EXPECTED_REGISTRY_IDS)
        self.assertEqual(
            audit_registry.collaboration_surfaces,
            EXPECTED_COLLABORATION_SURFACES,
        )
        self.assertEqual(
            audit_registry.source_collaboration_registries,
            EXPECTED_REGISTRY_IDS,
        )
        self.assertEqual(audit_registry.source_registry_refs, EXPECTED_SOURCE_REGISTRY_REFS)
        self.assertEqual(audit_registry.agent_ids, contract_registry.agent_ids)
        self.assertEqual(audit_registry.audit_count, 4)
        self.assertTrue(audit_registry.all_collaboration_registries_covered)
        self.assertTrue(audit_registry.all_agent_ids_aligned)
        self.assertTrue(audit_registry.no_missing_coverage)
        self.assertTrue(audit_registry.metadata_only)
        self.assertFalse(audit_registry.active_collaboration_execution_implemented)
        self.assertFalse(audit_registry.live_coordination_implemented)
        self.assertFalse(audit_registry.debate_execution_implemented)
        self.assertFalse(audit_registry.voting_execution_implemented)
        self.assertFalse(audit_registry.runtime_handoff_implemented)
        self.assertFalse(audit_registry.agent_invocation_implemented)
        self.assertFalse(audit_registry.generated_output_mutation_implemented)
        self.assertIn(
            "does not coordinate live agents",
            audit_registry.authority_boundary,
        )

    def test_audit_records_are_passive_and_source_aligned(self) -> None:
        registry = agent_collaboration_audit_registry()
        known_agents = set(agent_contract_registry().agent_ids)
        required_surface_blocks = {
            "coordination": "live_coordination",
            "debate": "debate_loop_execution",
            "consensus": "voting_execution",
            "workflow_handoff": "runtime_handoff_execution",
        }

        for record in registry.audit_records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_AUDIT_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "agent_collaboration_audit_record.v1",
            )
            self.assertEqual(record.audit_status, "pass")
            self.assertEqual(record.audit_stage, registry.audit_stage)
            self.assertEqual(
                record.validated_collaboration_surfaces,
                registry.validated_collaboration_surfaces,
            )
            self.assertEqual(record.passive_boundary_flags, registry.passive_boundary_flags)
            self.assertFalse(record.missing_coverage_items)
            self.assertEqual(record.registry_id, record.registry_role)
            self.assertTrue(record.source_registry_refs)
            self.assertTrue(set(record.participant_agent_ids).issubset(known_agents))
            self.assertEqual(
                record.collaboration_contract_count,
                len(record.collaboration_contract_ids),
            )
            self.assertIn(
                required_surface_blocks[record.collaboration_surface],
                record.registry_blocked_runtime_behaviors,
            )
            self.assertIn(
                "provider_or_model_routing",
                record.registry_blocked_runtime_behaviors,
            )
            self.assertIn(
                "generated_output_modification",
                record.registry_blocked_runtime_behaviors,
            )
            self.assertTrue(record.metadata_only_declared)
            self.assertTrue(record.metadata_only)
            self.assertFalse(record.live_coordination_implemented)
            self.assertFalse(record.debate_execution_implemented)
            self.assertFalse(record.voting_execution_implemented)
            self.assertFalse(record.runtime_handoff_implemented)
            self.assertFalse(record.agent_invocation_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.retry_triggering_implemented)
            self.assertFalse(record.state_mutation_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)

    def test_audit_lookup_surface_and_source_filtering_are_stable(self) -> None:
        registry = agent_collaboration_audit_registry()
        coordination_audit = agent_collaboration_audit_by_registry_id(
            "agent_coordination_registry"
        )
        missing_audit = agent_collaboration_audit_by_registry_id("missing_registry")
        debate_audits = agent_collaboration_audits_for_surface("debate")
        shared_context_audits = agent_collaboration_audits_for_source_registry(
            "shared_context_view_registry"
        )
        debate_source_audits = agent_collaboration_audits_for_source_registry(
            "agent_debate_registry"
        )
        missing_source_audits = agent_collaboration_audits_for_source_registry(
            "missing_registry"
        )

        self.assertIsNone(missing_audit)
        self.assertIsNotNone(coordination_audit)
        assert coordination_audit is not None
        self.assertEqual(coordination_audit.collaboration_surface, "coordination")
        self.assertIn(
            "coordination_handoff::parallel_group::foundational_context->"
            "parallel_group::domain_context",
            coordination_audit.collaboration_contract_ids,
        )
        self.assertEqual(len(debate_audits), 1)
        self.assertEqual(debate_audits[0].registry_id, "agent_debate_registry")
        self.assertEqual(
            tuple(record.registry_id for record in shared_context_audits),
            ("agent_debate_registry", "workflow_agent_handoff_registry"),
        )
        self.assertEqual(
            tuple(record.registry_id for record in debate_source_audits),
            ("consensus_builder_registry",),
        )
        self.assertEqual(missing_source_audits, ())
        self.assertIs(
            coordination_audit,
            agent_collaboration_audit_by_registry_id(
                "agent_coordination_registry",
                registry,
            ),
        )

    def test_audit_registry_rejects_mismatched_or_incomplete_records(self) -> None:
        registry = agent_collaboration_audit_registry()
        first_record = registry.audit_records[0]
        duplicate_record = first_record.model_copy(
            update={"registry_role": "duplicate_registry_role"}
        )
        mismatched_flags_record = first_record.model_copy(
            update={
                "passive_boundary_flags": (
                    "other_flag",
                    "debate_execution_blocked",
                    "voting_execution_blocked",
                    "runtime_handoff_blocked",
                    "agent_invocation_blocked",
                    "provider_model_routing_blocked",
                    "generated_output_mutation_blocked",
                )
            }
        )
        incomplete_record = first_record.model_copy(
            update={"missing_coverage_items": ("runtime_collaboration_enabled",)}
        )

        with self.assertRaisesRegex(ValueError, "registry_ids must be unique"):
            AgentCollaborationAuditRegistry(
                audit_records=(first_record, duplicate_record)
                + registry.audit_records[2:],
                registry_ids=registry.registry_ids,
                collaboration_surfaces=registry.collaboration_surfaces,
                agent_ids=registry.agent_ids,
                audit_count=registry.audit_count,
                source_collaboration_registries=registry.source_collaboration_registries,
                source_registry_refs=registry.source_registry_refs,
                validated_collaboration_surfaces=(
                    registry.validated_collaboration_surfaces
                ),
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "passive_boundary_flags"):
            AgentCollaborationAuditRegistry(
                audit_records=(mismatched_flags_record,) + registry.audit_records[1:],
                registry_ids=registry.registry_ids,
                collaboration_surfaces=registry.collaboration_surfaces,
                agent_ids=registry.agent_ids,
                audit_count=registry.audit_count,
                source_collaboration_registries=registry.source_collaboration_registries,
                source_registry_refs=registry.source_registry_refs,
                validated_collaboration_surfaces=(
                    registry.validated_collaboration_surfaces
                ),
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "missing coverage"):
            AgentCollaborationAuditRegistry(
                audit_records=(incomplete_record,) + registry.audit_records[1:],
                registry_ids=registry.registry_ids,
                collaboration_surfaces=registry.collaboration_surfaces,
                agent_ids=registry.agent_ids,
                audit_count=registry.audit_count,
                source_collaboration_registries=registry.source_collaboration_registries,
                source_registry_refs=registry.source_registry_refs,
                validated_collaboration_surfaces=(
                    registry.validated_collaboration_surfaces
                ),
                passive_boundary_flags=registry.passive_boundary_flags,
            )

    def test_agent_collaboration_audit_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate an agent collaboration planning sketch.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        agent_collaboration_audit_registry()
        agent_collaboration_audit_by_registry_id("agent_coordination_registry")
        agent_collaboration_audits_for_surface("consensus")
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)
        self.assertNotIn(
            "agent_collaboration_audit_registry",
            next_decision.model_dump_json(),
        )

    def test_audit_metadata_does_not_declare_active_collaboration_terms(self) -> None:
        registry = agent_collaboration_audit_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for record in registry.audit_records
                    for field in (
                        record.registry_id,
                        record.collaboration_surface,
                        *record.audit_findings,
                        *record.passive_boundary_flags,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_agent",
            "run_debate",
            "emit_coordination_event",
            "perform_handoff",
            "route_provider",
            "select_runtime",
            "trigger_retry",
            "mutate_generated_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
