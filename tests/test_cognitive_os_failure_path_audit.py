import unittest

from creative_coding_assistant.orchestration import (
    CognitiveOSFailurePathAuditRecord,
    CognitiveOSFailurePathAuditRegistry,
    build_cognitive_os_governance_safety,
    cognitive_os_failure_path_audit_by_id,
    cognitive_os_failure_path_audit_registry,
    cognitive_os_failure_path_audits_for_check,
    cognitive_os_failure_path_audits_for_surface,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_ROADMAP_ITEMS,
)
from creative_coding_assistant.orchestration.cognitive_os_failure_path_audit import (
    APPLICABLE_FAILURE_PATH_CHECKS,
    COGNITIVE_OS_FAILURE_PATH_BLOCKED_RUNTIME_BEHAVIORS,
    NOT_APPLICABLE_FAILURE_PATH_CHECKS,
    REQUIRED_FAILURE_PATH_CHECKS,
)
from creative_coding_assistant.orchestration.cognitive_os_governance_safety import (
    COGNITIVE_OS_GOVERNANCE_SOURCE_ROLES,
    COGNITIVE_OS_GOVERNANCE_SOURCE_SERIALIZATION_VERSIONS,
)
from creative_coding_assistant.orchestration.cognitive_os_secondary_surface import (
    COGNITIVE_OS_FOUNDATION_SYSTEMS,
)

FALSE_FLAG_FIELDS = (
    "audit_enforcement_implemented",
    "live_failure_observation_implemented",
    "live_error_classification_implemented",
    "terminal_failure_routing_implemented",
    "failure_handling_implemented",
    "failure_repair_implemented",
    "automatic_remediation_implemented",
    "governance_policy_enforcement_implemented",
    "safety_policy_enforcement_implemented",
    "hitl_request_emission_implemented",
    "human_review_request_implemented",
    "hitl_decision_application_implemented",
    "automation_activation_implemented",
    "core_surface_activation_implemented",
    "secondary_surface_activation_implemented",
    "execution_application_implemented",
    "routing_application_implemented",
    "storage_write_implemented",
    "prompt_rendering_implemented",
    "prompt_mutation_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "provider_model_routing_implemented",
    "memory_mutation_implemented",
    "retrieval_mutation_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "telemetry_collection_implemented",
    "runtime_probe_implemented",
    "dependency_installation_implemented",
    "retry_triggering_implemented",
    "refinement_triggering_implemented",
    "generated_output_evaluation_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
)


class CognitiveOSFailurePathAuditTests(unittest.TestCase):
    def test_registry_covers_runtime_failure_path_checklist(self) -> None:
        governance = build_cognitive_os_governance_safety()
        registry = cognitive_os_failure_path_audit_registry(governance)

        self.assertEqual(registry.role, "cognitive_os_failure_path_audit_registry")
        self.assertEqual(
            registry.serialization_version,
            "cognitive_os_failure_path_audit_registry.v1",
        )
        self.assertEqual(
            registry.checklist_source,
            "runtime/RUNTIME_FAILURE_PATH_AUDIT.md",
        )
        self.assertEqual(registry.route_name, governance.route_name)
        self.assertEqual(registry.task_type, governance.task_type)
        self.assertEqual(registry.execution_mode_ids, governance.execution_mode_ids)
        self.assertEqual(registry.source_surface_id_count, 21)
        self.assertEqual(
            registry.source_surface_ids[:3],
            (
                "cognitive_os_core_surface",
                "cognitive_os_secondary_surface",
                "cognitive_os_governance_safety",
            ),
        )
        self.assertIn(
            "cognitive_os_governance::v6_6_cognitive_core",
            registry.source_surface_ids,
        )
        self.assertEqual(len(registry.source_serialization_versions), 21)
        self.assertEqual(
            registry.source_surface_roles,
            COGNITIVE_OS_GOVERNANCE_SOURCE_ROLES,
        )
        self.assertEqual(
            registry.source_surface_serialization_versions,
            COGNITIVE_OS_GOVERNANCE_SOURCE_SERIALIZATION_VERSIONS,
        )
        self.assertEqual(registry.required_checks, REQUIRED_FAILURE_PATH_CHECKS)
        self.assertEqual(
            registry.applicable_required_checks,
            APPLICABLE_FAILURE_PATH_CHECKS,
        )
        self.assertEqual(
            registry.not_applicable_required_checks,
            NOT_APPLICABLE_FAILURE_PATH_CHECKS,
        )
        self.assertEqual(registry.check_kinds, APPLICABLE_FAILURE_PATH_CHECKS)
        self.assertEqual(registry.record_count, 17)
        self.assertEqual(registry.covered_roadmap_items, COGNITIVE_OS_ROADMAP_ITEMS)
        self.assertEqual(registry.covered_roadmap_item_count, 24)
        self.assertEqual(registry.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(registry.capability_ids, governance.capability_ids)
        self.assertEqual(registry.capability_count, 6)
        self.assertEqual(registry.foundation_systems, COGNITIVE_OS_FOUNDATION_SYSTEMS)
        self.assertEqual(registry.foundation_system_count, 7)
        self.assertEqual(
            registry.governance_boundary_ids,
            governance.governance_boundary_ids,
        )
        self.assertEqual(registry.governance_boundary_count, 6)
        self.assertEqual(registry.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            registry.blocked_runtime_behaviors,
            COGNITIVE_OS_FAILURE_PATH_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(registry.all_applicable_checks_covered)
        self.assertTrue(registry.all_roadmap_items_traceable)
        self.assertTrue(registry.all_capability_surfaces_traceable)
        self.assertTrue(registry.foundation_traceability_verified)
        self.assertTrue(registry.governance_safety_boundary_preserved)
        self.assertTrue(registry.runtime_failure_boundary_preserved)
        self.assertTrue(registry.workflow_state_integrity_boundary_preserved)
        self.assertTrue(registry.provider_model_routing_preserved)
        self.assertTrue(registry.generated_output_mutation_boundary_preserved)
        self.assertTrue(registry.passive_registry_activation_boundary_preserved)
        self.assertTrue(registry.metadata_only_rule_satisfied)
        self.assertTrue(registry.active_behavior_rule_satisfied)
        self.assertTrue(registry.runtime_evolution_not_applied)
        self.assertIn("V6.6 Cognitive OS", registry.authority_boundary)
        self.assertIn("metadata only", registry.authority_boundary)
        for field_name in FALSE_FLAG_FIELDS:
            self.assertFalse(getattr(registry, field_name))
        self.assertFalse(registry.applied_audit_fix_ids)
        self.assertFalse(registry.handled_failure_ids)
        self.assertFalse(registry.routed_terminal_failure_ids)
        self.assertFalse(registry.emitted_hitl_request_ids)
        self.assertFalse(registry.activated_core_surface_ids)
        self.assertFalse(registry.activated_secondary_surface_ids)
        self.assertFalse(registry.generated_report_artifact_ids)
        self.assertFalse(registry.written_storage_record_ids)
        self.assertFalse(registry.provider_execution_ids)
        self.assertFalse(registry.mutated_output_ids)
        self.assertTrue(registry.metadata_only)

    def test_records_are_metadata_only_and_trace_all_surfaces(self) -> None:
        registry = cognitive_os_failure_path_audit_registry()

        for record in registry.records:
            self.assertEqual(
                record.audit_id,
                f"cognitive_os_failure_path_audit::{record.check_kind}",
            )
            self.assertEqual(record.checklist_source, registry.checklist_source)
            self.assertEqual(record.source_surface_ids, registry.source_surface_ids)
            self.assertEqual(
                record.source_serialization_versions,
                registry.source_serialization_versions,
            )
            self.assertEqual(record.covered_roadmap_items, COGNITIVE_OS_ROADMAP_ITEMS)
            self.assertEqual(record.capability_ids, registry.capability_ids)
            self.assertEqual(record.capabilities, COGNITIVE_OS_CAPABILITIES)
            self.assertEqual(record.foundation_systems, COGNITIVE_OS_FOUNDATION_SYSTEMS)
            self.assertEqual(record.governance_boundary_count, 6)
            self.assertEqual(record.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
            self.assertEqual(record.audit_status, "pass")
            self.assertTrue(record.checklist_item_applicable)
            self.assertTrue(record.runtime_failure_path_audit_implemented)
            self.assertTrue(record.cognitive_os_layer_verified)
            self.assertTrue(record.governance_boundary_verified)
            self.assertTrue(record.all_roadmap_items_traceable)
            self.assertTrue(record.all_capability_surfaces_traceable)
            self.assertTrue(record.foundation_traceability_verified)
            self.assertTrue(record.metadata_only_rule_satisfied)
            self.assertTrue(record.no_automation_boundary_preserved)
            self.assertTrue(record.provider_model_routing_preserved)
            self.assertTrue(record.generated_output_mutation_boundary_preserved)
            self.assertTrue(record.passive_registry_activation_boundary_preserved)
            self.assertTrue(record.runtime_evolution_not_applied)
            self.assertIn("metadata coverage only", record.failure_response_boundary)
            for field_name in FALSE_FLAG_FIELDS:
                self.assertFalse(getattr(record, field_name))
            self.assertFalse(record.applied_audit_fix_ids)
            self.assertFalse(record.handled_failure_ids)
            self.assertFalse(record.routed_terminal_failure_ids)
            self.assertFalse(record.emitted_hitl_request_ids)
            self.assertFalse(record.activated_core_surface_ids)
            self.assertFalse(record.activated_secondary_surface_ids)
            self.assertFalse(record.generated_report_artifact_ids)
            self.assertFalse(record.written_storage_record_ids)
            self.assertFalse(record.provider_execution_ids)
            self.assertFalse(record.mutated_output_ids)
            self.assertTrue(record.metadata_only)

        provider_record = cognitive_os_failure_path_audit_by_id(
            "cognitive_os_failure_path_audit::provider_failures",
            registry,
        )
        self.assertIsNotNone(provider_record)
        assert provider_record is not None
        self.assertEqual(provider_record.check_kind, "provider_failures")
        self.assertEqual(
            cognitive_os_failure_path_audits_for_check(
                "serialization_failures",
                registry,
            ),
            (
                cognitive_os_failure_path_audit_by_id(
                    "cognitive_os_failure_path_audit::serialization_failures",
                    registry,
                ),
            ),
        )
        governance_records = cognitive_os_failure_path_audits_for_surface(
            "cognitive_os_governance::v6_6_cognitive_core",
            registry,
        )
        self.assertEqual(len(governance_records), 17)
        self.assertIsNone(cognitive_os_failure_path_audit_by_id("missing", registry))
        self.assertFalse(
            cognitive_os_failure_path_audits_for_surface("missing", registry),
        )

    def test_registry_rejects_mismatched_or_mutating_payloads(self) -> None:
        registry = cognitive_os_failure_path_audit_registry()
        payload = registry.model_dump(mode="json")
        payload["record_ids"] = ("missing",) + tuple(payload["record_ids"][1:])

        with self.assertRaisesRegex(ValueError, "record_ids must match records"):
            CognitiveOSFailurePathAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["handled_failure_ids"] = ("failure",)

        with self.assertRaisesRegex(
            ValueError,
            "failure path audit mutation ids must be empty",
        ):
            CognitiveOSFailurePathAuditRegistry(**payload)

        record_payload = registry.records[0].model_dump(mode="json")
        record_payload["emitted_hitl_request_ids"] = ("hitl_request",)

        with self.assertRaisesRegex(
            ValueError,
            "failure path audit mutation ids must be empty",
        ):
            CognitiveOSFailurePathAuditRecord(**record_payload)

    def test_registry_reuses_supplied_governance_plan(self) -> None:
        governance = build_cognitive_os_governance_safety(route="generate")
        registry = cognitive_os_failure_path_audit_registry(governance)

        self.assertEqual(registry.route_name, governance.route_name)
        self.assertEqual(registry.task_type, governance.task_type)
        self.assertEqual(registry.execution_mode_ids, governance.execution_mode_ids)
        self.assertEqual(registry.capability_ids, governance.capability_ids)
        self.assertEqual(
            registry.governance_boundary_ids,
            governance.governance_boundary_ids,
        )


if __name__ == "__main__":
    unittest.main()
