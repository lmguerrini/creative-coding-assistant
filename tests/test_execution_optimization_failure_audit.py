import unittest

from creative_coding_assistant.orchestration import (
    ExecutionOptimizationFailureAuditRegistry,
    execution_optimization_failure_audit_by_id,
    execution_optimization_failure_audit_registry,
    execution_optimization_failure_audits_for_check,
)

EXPECTED_SOURCE_SURFACES = (
    "execution_graph_analysis",
    "workflow_cost_analysis",
    "workflow_complexity_analysis",
    "creative_complexity_analysis",
    "context_budget_plan",
    "exploration_budget_plan",
    "context_routing_plan",
    "prompt_compression_result",
    "retrieval_compression_result",
    "memory_summarization_result",
    "execution_cache_lookup",
    "context_reuse_plan",
    "workflow_pruning_plan",
    "execution_cost_forecast",
    "execution_path_optimization_plan",
    "execution_strategy_selection",
)
EXPECTED_CHECK_KINDS = (
    "terminal_failure_routing",
    "provider_model_routing_preservation",
    "retry_failure_boundary",
    "planning_helper_validation",
    "serialization_guard",
    "cache_failure_mode",
    "budget_cost_prediction_boundary",
    "workflow_state_output_boundary",
    "metadata_activation_boundary",
)
EXPECTED_APPLICABLE_CHECKS = (
    "node_level_failure_paths",
    "terminal_failure_routing",
    "provider_failures",
    "model_routing_failures",
    "retry_failures",
    "planning_helper_failures",
    "serialization_failures",
    "cache_failures",
    "budget_cost_prediction_failures",
    "workflow_state_integrity_after_failure",
    "provider_model_routing_preservation",
    "generated_output_mutation_boundaries",
    "passive_registry_activation_boundaries",
)


class ExecutionOptimizationFailureAuditTests(unittest.TestCase):
    def test_failure_audit_covers_v5_1_runtime_failure_checks(self) -> None:
        registry = execution_optimization_failure_audit_registry()

        self.assertEqual(registry.role, "execution_optimization_failure_audit")
        self.assertEqual(
            registry.serialization_version,
            "execution_optimization_failure_audit.v1",
        )
        self.assertEqual(registry.source_surface_ids, EXPECTED_SOURCE_SURFACES)
        self.assertEqual(
            registry.applicable_required_checks, EXPECTED_APPLICABLE_CHECKS
        )
        self.assertEqual(
            registry.not_applicable_required_checks,
            (
                "stream_failures",
                "scheduling_failures",
                "preview_workstation_frontend_backend_failures",
                "telemetry_observability_failures",
            ),
        )
        self.assertEqual(registry.check_kinds, EXPECTED_CHECK_KINDS)
        self.assertEqual(registry.record_count, 9)
        self.assertTrue(registry.metadata_only_rule_satisfied)
        self.assertTrue(registry.active_behavior_rule_satisfied)
        self.assertTrue(registry.provider_model_routing_preserved)
        self.assertTrue(registry.generated_output_mutation_boundary_preserved)
        self.assertTrue(registry.runtime_failure_audit_implemented)
        self.assertFalse(registry.workflow_execution_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.budget_enforcement_implemented)
        self.assertFalse(registry.prompt_mutation_implemented)
        self.assertFalse(registry.persistent_storage_write_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertTrue(registry.metadata_only)
        self.assertIn("does not execute workflows", registry.authority_boundary)

    def test_failure_audit_records_are_passive_and_source_scoped(self) -> None:
        registry = execution_optimization_failure_audit_registry()

        for record in registry.records:
            self.assertTrue(
                record.audit_id.startswith("execution_optimization_failure::")
            )
            self.assertEqual(
                record.serialization_version,
                "execution_optimization_failure_audit_record.v1",
            )
            self.assertIn(record.check_kind, EXPECTED_CHECK_KINDS)
            self.assertTrue(record.evidence)
            self.assertTrue(record.invariant_assertions)
            self.assertEqual(record.audit_status, "pass")
            self.assertTrue(record.runtime_failure_audit_implemented)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.retry_triggering_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.budget_enforcement_implemented)
            self.assertFalse(record.prompt_mutation_implemented)
            self.assertFalse(record.persistent_storage_write_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)
            self.assertTrue(record.metadata_only)
            for source_surface_id in record.source_surface_ids:
                self.assertIn(source_surface_id, EXPECTED_SOURCE_SURFACES)

    def test_failure_audit_lookup_helpers_are_stable(self) -> None:
        registry = execution_optimization_failure_audit_registry()
        retry_records = execution_optimization_failure_audits_for_check(
            "retry_failure_boundary",
            registry,
        )
        retry_record = execution_optimization_failure_audit_by_id(
            "execution_optimization_failure::retry_failure_boundary",
            registry,
        )
        missing_record = execution_optimization_failure_audit_by_id("missing", registry)

        self.assertEqual(len(retry_records), 1)
        self.assertIs(retry_records[0], retry_record)
        self.assertIsNone(missing_record)

    def test_failure_audit_rejects_mismatched_coverage(self) -> None:
        registry = execution_optimization_failure_audit_registry()
        payload = registry.model_dump(mode="json")
        payload["audit_ids"] = ("missing",) + tuple(payload["audit_ids"][1:])

        with self.assertRaisesRegex(ValueError, "audit_ids must match"):
            ExecutionOptimizationFailureAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["source_surface_ids"] = tuple(payload["source_surface_ids"][1:])

        with self.assertRaisesRegex(ValueError, "source_surface_ids"):
            ExecutionOptimizationFailureAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["check_kinds"] = ("cache_failure_mode",) + tuple(
            payload["check_kinds"][1:]
        )

        with self.assertRaisesRegex(ValueError, "check_kinds"):
            ExecutionOptimizationFailureAuditRegistry(**payload)

    def test_failure_audit_does_not_declare_runtime_failure_actions(self) -> None:
        registry = execution_optimization_failure_audit_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for record in registry.records
                    for field in (
                        record.audit_id,
                        *record.evidence,
                        *record.invariant_assertions,
                        *record.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_workflow(",
            "trigger_retry(",
            "route_provider(",
            "route_model(",
            "enforce_budget(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
