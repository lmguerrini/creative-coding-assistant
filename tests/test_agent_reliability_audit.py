import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    AgentReliabilityAuditRegistry,
    agent_contract_registry,
    agent_escalation_signal_registry,
    agent_lifecycle_registry,
    agent_reliability_audit_by_agent_id,
    agent_reliability_audit_registry,
    agent_reliability_audits_for_consistency_family,
    agent_reliability_audits_for_escalation_category,
    agent_state_synchronization_registry,
    engine_contract_consistency_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_SOURCE_REGISTRIES = (
    "agent_lifecycle_registry",
    "agent_state_synchronization_registry",
    "agent_escalation_signal_registry",
    "engine_contract_consistency_registry",
)

REQUIRED_AUDIT_RECORD_FIELDS = {
    "agent_id",
    "audit_stage",
    "lifecycle_profile_id",
    "lifecycle_transition_ids",
    "lifecycle_terminal_states",
    "state_sync_profile_id",
    "sync_checkpoint_ids",
    "consistency_constraint_ids",
    "stale_warning_ids",
    "conflict_surface_ids",
    "escalation_signal_ids",
    "escalation_signal_categories",
    "consistency_family_ids",
    "reliability_source_registries",
    "validated_reliability_surfaces",
    "passive_boundary_flags",
    "audit_findings",
    "missing_coverage_items",
    "lifecycle_blocked_runtime_behaviors",
    "state_sync_blocked_runtime_behaviors",
    "escalation_blocked_runtime_behaviors",
    "consistency_blocked_runtime_behaviors",
    "blocked_runtime_behaviors",
    "audit_status",
    "metadata_only_declared",
    "lifecycle_profile_present",
    "state_sync_profile_present",
    "escalation_signal_coverage_present",
    "consistency_family_coverage_present",
    "runtime_lifecycle_engine_implemented",
    "runtime_state_synchronization_implemented",
    "stale_state_detection_implemented",
    "conflict_resolution_implemented",
    "escalation_execution_implemented",
    "retry_recovery_implemented",
    "provider_model_routing_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class AgentReliabilityAuditTests(unittest.TestCase):
    def test_audit_registry_covers_reliability_sources(self) -> None:
        registry = agent_reliability_audit_registry()
        lifecycle = agent_lifecycle_registry()
        state_sync = agent_state_synchronization_registry()
        escalation_signals = agent_escalation_signal_registry()
        consistency = engine_contract_consistency_registry()

        self.assertEqual(registry.role, "agent_reliability_audit_registry")
        self.assertEqual(
            registry.serialization_version,
            "agent_reliability_audit_registry.v1",
        )
        self.assertEqual(
            registry.audit_stage,
            "v4_6_agent_reliability_hardening",
        )
        self.assertEqual(registry.agent_ids, agent_contract_registry().agent_ids)
        self.assertEqual(registry.agent_ids, lifecycle.agent_ids)
        self.assertEqual(registry.agent_ids, state_sync.agent_ids)
        self.assertEqual(
            registry.lifecycle_profile_ids,
            tuple(profile.lifecycle_profile_id for profile in lifecycle.profiles),
        )
        self.assertEqual(
            registry.state_sync_profile_ids,
            tuple(profile.sync_profile_id for profile in state_sync.profiles),
        )
        self.assertEqual(registry.lifecycle_transition_ids, lifecycle.transition_ids)
        self.assertEqual(registry.sync_checkpoint_ids, state_sync.checkpoint_ids)
        self.assertEqual(
            registry.consistency_constraint_ids,
            state_sync.constraint_ids,
        )
        self.assertEqual(registry.stale_warning_ids, state_sync.stale_warning_ids)
        self.assertEqual(registry.conflict_surface_ids, state_sync.conflict_surface_ids)
        self.assertEqual(registry.escalation_signal_ids, escalation_signals.signal_ids)
        self.assertEqual(
            registry.escalation_signal_categories,
            escalation_signals.categories,
        )
        self.assertEqual(registry.consistency_family_ids, consistency.family_ids)
        self.assertEqual(
            registry.reliability_source_registries,
            EXPECTED_SOURCE_REGISTRIES,
        )
        self.assertEqual(registry.audit_count, 12)
        self.assertTrue(registry.all_agents_covered)
        self.assertTrue(registry.no_missing_coverage)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.runtime_reliability_engine_implemented)
        self.assertFalse(registry.runtime_lifecycle_engine_implemented)
        self.assertFalse(registry.runtime_state_synchronization_implemented)
        self.assertFalse(registry.stale_state_detection_implemented)
        self.assertFalse(registry.conflict_resolution_implemented)
        self.assertFalse(registry.escalation_execution_implemented)
        self.assertFalse(registry.retry_recovery_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertIn("does not run lifecycle transitions", registry.authority_boundary)

    def test_audit_records_are_passive_and_source_aligned(self) -> None:
        registry = agent_reliability_audit_registry()
        known_lifecycle_profiles = set(registry.lifecycle_profile_ids)
        known_state_sync_profiles = set(registry.state_sync_profile_ids)

        for record in registry.audit_records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_AUDIT_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "agent_reliability_audit_record.v1",
            )
            self.assertEqual(record.audit_status, "pass")
            self.assertEqual(record.audit_stage, registry.audit_stage)
            self.assertIn(record.lifecycle_profile_id, known_lifecycle_profiles)
            self.assertIn(record.state_sync_profile_id, known_state_sync_profiles)
            self.assertEqual(
                record.lifecycle_transition_ids,
                registry.lifecycle_transition_ids,
            )
            self.assertEqual(record.sync_checkpoint_ids, registry.sync_checkpoint_ids)
            self.assertEqual(
                record.consistency_constraint_ids,
                registry.consistency_constraint_ids,
            )
            self.assertEqual(record.stale_warning_ids, registry.stale_warning_ids)
            self.assertEqual(record.conflict_surface_ids, registry.conflict_surface_ids)
            self.assertEqual(
                record.escalation_signal_ids, registry.escalation_signal_ids
            )
            self.assertEqual(
                record.escalation_signal_categories,
                registry.escalation_signal_categories,
            )
            self.assertEqual(
                record.consistency_family_ids, registry.consistency_family_ids
            )
            self.assertEqual(
                record.reliability_source_registries,
                registry.reliability_source_registries,
            )
            self.assertEqual(
                record.validated_reliability_surfaces,
                registry.validated_reliability_surfaces,
            )
            self.assertEqual(
                record.passive_boundary_flags, registry.passive_boundary_flags
            )
            self.assertFalse(record.missing_coverage_items)
            self.assertIn(
                "runtime_lifecycle_engine",
                record.lifecycle_blocked_runtime_behaviors,
            )
            self.assertIn(
                "runtime_state_synchronization",
                record.state_sync_blocked_runtime_behaviors,
            )
            self.assertIn(
                "conflict_resolution",
                record.state_sync_blocked_runtime_behaviors,
            )
            self.assertIn(
                "escalation_execution",
                record.escalation_blocked_runtime_behaviors,
            )
            self.assertIn(
                "retry_or_refinement_triggering",
                record.consistency_blocked_runtime_behaviors,
            )
            self.assertIn(
                "generated_output_modification",
                record.consistency_blocked_runtime_behaviors,
            )
            self.assertTrue(record.metadata_only_declared)
            self.assertTrue(record.lifecycle_profile_present)
            self.assertTrue(record.state_sync_profile_present)
            self.assertTrue(record.escalation_signal_coverage_present)
            self.assertTrue(record.consistency_family_coverage_present)
            self.assertTrue(record.metadata_only)
            self.assertFalse(record.runtime_lifecycle_engine_implemented)
            self.assertFalse(record.runtime_state_synchronization_implemented)
            self.assertFalse(record.stale_state_detection_implemented)
            self.assertFalse(record.conflict_resolution_implemented)
            self.assertFalse(record.escalation_execution_implemented)
            self.assertFalse(record.retry_recovery_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)

    def test_audit_lookup_category_and_family_filtering_are_stable(self) -> None:
        registry = agent_reliability_audit_registry()
        planner_audit = agent_reliability_audit_by_agent_id("planner_agent")
        missing_audit = agent_reliability_audit_by_agent_id("missing_agent")
        confidence_audits = agent_reliability_audits_for_escalation_category(
            "confidence"
        )
        artifact_family_audits = agent_reliability_audits_for_consistency_family(
            "artifact_intelligence"
        )
        missing_category_audits = agent_reliability_audits_for_escalation_category(
            "missing_category"
        )
        missing_family_audits = agent_reliability_audits_for_consistency_family(
            "missing_family"
        )

        self.assertIsNone(missing_audit)
        self.assertIsNotNone(planner_audit)
        assert planner_audit is not None
        self.assertEqual(
            planner_audit.lifecycle_profile_id, "planner_agent_lifecycle_profile"
        )
        self.assertEqual(
            planner_audit.state_sync_profile_id, "planner_agent_state_sync_profile"
        )
        self.assertIn("quality", planner_audit.escalation_signal_categories)
        self.assertIn("creative_workstation", planner_audit.consistency_family_ids)
        self.assertEqual(len(confidence_audits), registry.audit_count)
        self.assertEqual(len(artifact_family_audits), registry.audit_count)
        self.assertEqual(missing_category_audits, ())
        self.assertEqual(missing_family_audits, ())
        self.assertIs(
            planner_audit,
            agent_reliability_audit_by_agent_id("planner_agent", registry),
        )

    def test_audit_registry_rejects_mismatched_or_incomplete_records(self) -> None:
        registry = agent_reliability_audit_registry()
        first_record = registry.audit_records[0]
        duplicate_record = first_record.model_copy(
            update={"lifecycle_profile_id": "duplicate_lifecycle_profile"}
        )
        mismatched_flags_record = first_record.model_copy(
            update={
                "passive_boundary_flags": (
                    "other_flag",
                    "runtime_state_synchronization_blocked",
                    "stale_state_detection_blocked",
                    "conflict_resolution_blocked",
                    "escalation_execution_blocked",
                    "retry_recovery_execution_blocked",
                    "provider_model_routing_blocked",
                    "generated_output_mutation_blocked",
                )
            }
        )
        incomplete_record = first_record.model_copy(
            update={"missing_coverage_items": ("retry_recovery_enabled",)}
        )

        with self.assertRaisesRegex(ValueError, "agent_ids must be unique"):
            AgentReliabilityAuditRegistry(
                audit_records=(first_record, duplicate_record)
                + registry.audit_records[2:],
                agent_ids=registry.agent_ids,
                audit_count=registry.audit_count,
                lifecycle_profile_ids=registry.lifecycle_profile_ids,
                state_sync_profile_ids=registry.state_sync_profile_ids,
                lifecycle_transition_ids=registry.lifecycle_transition_ids,
                sync_checkpoint_ids=registry.sync_checkpoint_ids,
                consistency_constraint_ids=registry.consistency_constraint_ids,
                stale_warning_ids=registry.stale_warning_ids,
                conflict_surface_ids=registry.conflict_surface_ids,
                escalation_signal_ids=registry.escalation_signal_ids,
                escalation_signal_categories=registry.escalation_signal_categories,
                consistency_family_ids=registry.consistency_family_ids,
                reliability_source_registries=registry.reliability_source_registries,
                validated_reliability_surfaces=registry.validated_reliability_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "passive_boundary_flags"):
            AgentReliabilityAuditRegistry(
                audit_records=(mismatched_flags_record,) + registry.audit_records[1:],
                agent_ids=registry.agent_ids,
                audit_count=registry.audit_count,
                lifecycle_profile_ids=registry.lifecycle_profile_ids,
                state_sync_profile_ids=registry.state_sync_profile_ids,
                lifecycle_transition_ids=registry.lifecycle_transition_ids,
                sync_checkpoint_ids=registry.sync_checkpoint_ids,
                consistency_constraint_ids=registry.consistency_constraint_ids,
                stale_warning_ids=registry.stale_warning_ids,
                conflict_surface_ids=registry.conflict_surface_ids,
                escalation_signal_ids=registry.escalation_signal_ids,
                escalation_signal_categories=registry.escalation_signal_categories,
                consistency_family_ids=registry.consistency_family_ids,
                reliability_source_registries=registry.reliability_source_registries,
                validated_reliability_surfaces=registry.validated_reliability_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "missing coverage"):
            AgentReliabilityAuditRegistry(
                audit_records=(incomplete_record,) + registry.audit_records[1:],
                agent_ids=registry.agent_ids,
                audit_count=registry.audit_count,
                lifecycle_profile_ids=registry.lifecycle_profile_ids,
                state_sync_profile_ids=registry.state_sync_profile_ids,
                lifecycle_transition_ids=registry.lifecycle_transition_ids,
                sync_checkpoint_ids=registry.sync_checkpoint_ids,
                consistency_constraint_ids=registry.consistency_constraint_ids,
                stale_warning_ids=registry.stale_warning_ids,
                conflict_surface_ids=registry.conflict_surface_ids,
                escalation_signal_ids=registry.escalation_signal_ids,
                escalation_signal_categories=registry.escalation_signal_categories,
                consistency_family_ids=registry.consistency_family_ids,
                reliability_source_registries=registry.reliability_source_registries,
                validated_reliability_surfaces=registry.validated_reliability_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

    def test_agent_reliability_audit_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate a reliable agent workflow metadata sketch.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        agent_reliability_audit_registry()
        agent_reliability_audit_by_agent_id("planner_agent")
        agent_reliability_audits_for_escalation_category("confidence")
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)
        self.assertNotIn(
            "agent_reliability_audit_registry",
            next_decision.model_dump_json(),
        )

    def test_audit_metadata_does_not_declare_active_reliability_terms(self) -> None:
        registry = agent_reliability_audit_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for record in registry.audit_records
                    for field in (
                        record.agent_id,
                        *record.audit_findings,
                        *record.passive_boundary_flags,
                    )
                ),
            )
        )

        for forbidden_term in (
            "trigger_runtime_retry",
            "perform_recovery",
            "synchronize_runtime_state",
            "resolve_runtime_conflict",
            "execute_agent",
            "route_provider",
            "mutate_generated_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
