import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    PerformanceFailurePathAuditRegistry,
    performance_failure_path_audit_by_id,
    performance_failure_path_audit_registry,
    performance_failure_path_audits_for_check,
    performance_failure_path_audits_for_surface,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_SURFACE_IDS = (
    "parallel_scheduler",
    "latency_optimizer",
    "async_execution",
    "streaming_optimizer",
    "retry_policies",
    "load_balancer",
    "execution_profiling",
    "workflow_replay_engine",
    "execution_replay_engine",
    "bottleneck_detection",
    "throughput_optimizer",
    "performance_prediction",
    "performance_benchmarking",
    "reasoning_budget_optimizer",
    "performance_regression_detection",
    "resource_utilization_optimizer",
)
EXPECTED_REQUIRED_CHECKS = (
    "node_level_failure_paths",
    "terminal_failure_routing",
    "provider_failures",
    "model_routing_failures",
    "stream_failures",
    "scheduling_failures",
    "retry_failures",
    "planning_helper_failures",
    "prompt_rendering_failures",
    "serialization_failures",
    "preview_workstation_frontend_backend_failures",
    "registry_import_loading_failures",
    "telemetry_observability_failures",
    "cache_failures",
    "budget_cost_prediction_failures",
    "workflow_state_integrity_after_failure",
    "provider_model_routing_preservation",
    "generated_output_mutation_boundaries",
    "passive_registry_activation_boundaries",
)
EXPECTED_APPLICABLE_CHECKS = (
    "node_level_failure_paths",
    "terminal_failure_routing",
    "provider_failures",
    "model_routing_failures",
    "stream_failures",
    "scheduling_failures",
    "retry_failures",
    "planning_helper_failures",
    "prompt_rendering_failures",
    "serialization_failures",
    "registry_import_loading_failures",
    "budget_cost_prediction_failures",
    "workflow_state_integrity_after_failure",
    "provider_model_routing_preservation",
    "generated_output_mutation_boundaries",
    "passive_registry_activation_boundaries",
)
EXPECTED_NOT_APPLICABLE_CHECKS = (
    "preview_workstation_frontend_backend_failures",
    "telemetry_observability_failures",
    "cache_failures",
)
REQUIRED_RECORD_FIELDS = {
    "audit_id",
    "check_kind",
    "checklist_source",
    "source_surface_ids",
    "evidence",
    "invariant_assertions",
    "failure_response_boundary",
    "audit_status",
    "blocked_runtime_behaviors",
    "checklist_item_applicable",
    "runtime_failure_path_audit_implemented",
    "runtime_performance_measurement_implemented",
    "runtime_profiling_implemented",
    "benchmark_execution_implemented",
    "resource_allocation_implemented",
    "capacity_enforcement_implemented",
    "budget_enforcement_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "provider_model_routing_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "metadata_only",
}


class PerformanceFailurePathAuditTests(unittest.TestCase):
    def test_registry_covers_runtime_failure_path_checklist(self) -> None:
        registry = performance_failure_path_audit_registry()

        self.assertEqual(registry.role, "performance_failure_path_audit_registry")
        self.assertEqual(
            registry.serialization_version,
            "performance_failure_path_audit_registry.v1",
        )
        self.assertEqual(
            registry.checklist_source,
            "runtime/RUNTIME_FAILURE_PATH_AUDIT.md",
        )
        self.assertEqual(
            registry.architecture_registry_serialization_version,
            "performance_architecture_consistency_registry.v1",
        )
        self.assertEqual(registry.architecture_registry_record_count, 16)
        self.assertEqual(registry.source_surface_ids, EXPECTED_SURFACE_IDS)
        self.assertEqual(registry.required_checks, EXPECTED_REQUIRED_CHECKS)
        self.assertEqual(
            registry.applicable_required_checks,
            EXPECTED_APPLICABLE_CHECKS,
        )
        self.assertEqual(
            registry.not_applicable_required_checks,
            EXPECTED_NOT_APPLICABLE_CHECKS,
        )
        self.assertEqual(registry.check_kinds, EXPECTED_APPLICABLE_CHECKS)
        self.assertEqual(registry.record_count, 16)
        self.assertTrue(registry.metadata_only_rule_satisfied)
        self.assertTrue(registry.active_behavior_rule_satisfied)
        self.assertTrue(registry.all_applicable_checks_covered)
        self.assertTrue(registry.provider_model_routing_preserved)
        self.assertTrue(registry.generated_output_mutation_boundary_preserved)
        self.assertTrue(registry.passive_registry_activation_boundary_preserved)
        self.assertTrue(registry.runtime_evolution_not_applied)
        self.assertIn("does not measure performance", registry.authority_boundary)
        self.assertFalse(registry.runtime_performance_measurement_implemented)
        self.assertFalse(registry.runtime_profiling_implemented)
        self.assertFalse(registry.benchmark_execution_implemented)
        self.assertFalse(registry.resource_allocation_implemented)
        self.assertFalse(registry.capacity_enforcement_implemented)
        self.assertFalse(registry.budget_enforcement_implemented)
        self.assertFalse(registry.workflow_execution_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.workflow_graph_mutation_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.node_handler_invocation_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.prompt_mutation_implemented)
        self.assertFalse(registry.persistent_storage_write_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.runtime_evolution_implemented)
        self.assertTrue(registry.metadata_only)

    def test_records_are_passive_and_source_scoped(self) -> None:
        registry = performance_failure_path_audit_registry()

        for record in registry.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertTrue(record.audit_id.startswith("performance_failure::"))
            self.assertEqual(record.checklist_source, registry.checklist_source)
            self.assertIn(record.check_kind, EXPECTED_APPLICABLE_CHECKS)
            self.assertEqual(
                record.serialization_version,
                "performance_failure_path_audit_record.v1",
            )
            self.assertTrue(record.source_surface_ids)
            self.assertTrue(record.evidence)
            self.assertTrue(record.invariant_assertions)
            self.assertTrue(record.failure_response_boundary)
            self.assertEqual(record.audit_status, "pass")
            self.assertTrue(record.checklist_item_applicable)
            self.assertTrue(record.runtime_failure_path_audit_implemented)
            self.assertFalse(record.runtime_performance_measurement_implemented)
            self.assertFalse(record.runtime_profiling_implemented)
            self.assertFalse(record.benchmark_execution_implemented)
            self.assertFalse(record.resource_allocation_implemented)
            self.assertFalse(record.capacity_enforcement_implemented)
            self.assertFalse(record.budget_enforcement_implemented)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.workflow_control_implemented)
            self.assertFalse(record.workflow_graph_mutation_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.agent_invocation_implemented)
            self.assertFalse(record.node_handler_invocation_implemented)
            self.assertFalse(record.retry_triggering_implemented)
            self.assertFalse(record.prompt_mutation_implemented)
            self.assertFalse(record.persistent_storage_write_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertTrue(record.metadata_only)
            for source_surface_id in record.source_surface_ids:
                self.assertIn(source_surface_id, EXPECTED_SURFACE_IDS)

    def test_lookup_helpers_are_stable(self) -> None:
        registry = performance_failure_path_audit_registry()
        stream_records = performance_failure_path_audits_for_check(
            "stream_failures",
            registry,
        )
        stream_record = performance_failure_path_audit_by_id(
            "performance_failure::stream_failures",
            registry,
        )
        throughput_records = performance_failure_path_audits_for_surface(
            "throughput_optimizer",
            registry,
        )
        missing_record = performance_failure_path_audit_by_id("missing", registry)
        missing_surface_records = performance_failure_path_audits_for_surface(
            "missing",
            registry,
        )

        self.assertEqual(len(stream_records), 1)
        self.assertIs(stream_records[0], stream_record)
        self.assertIsNone(missing_record)
        self.assertEqual(missing_surface_records, ())
        self.assertIn(
            "stream_failures",
            tuple(record.check_kind for record in throughput_records),
        )
        self.assertIn(
            "scheduling_failures",
            tuple(record.check_kind for record in throughput_records),
        )
        self.assertIn(
            "provider_model_routing_preservation",
            tuple(record.check_kind for record in throughput_records),
        )

    def test_registry_rejects_mismatched_coverage(self) -> None:
        registry = performance_failure_path_audit_registry()
        payload = registry.model_dump(mode="json")
        payload["audit_ids"] = ["missing", *payload["audit_ids"][1:]]

        with self.assertRaisesRegex(ValueError, "audit_ids must match"):
            PerformanceFailurePathAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["source_surface_ids"] = payload["source_surface_ids"][1:]

        with self.assertRaisesRegex(ValueError, "source_surface_ids"):
            PerformanceFailurePathAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["check_kinds"] = [
            "serialization_failures",
            *payload["check_kinds"][1:],
        ]

        with self.assertRaisesRegex(ValueError, "check_kinds"):
            PerformanceFailurePathAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["records"][0]["source_surface_ids"] = ["unknown"]

        with self.assertRaisesRegex(ValueError, "source_surface_ids"):
            PerformanceFailurePathAuditRegistry(**payload)

    def test_failure_audit_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate performance failure path audit metadata.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        performance_failure_path_audit_registry()
        performance_failure_path_audit_by_id("performance_failure::stream_failures")
        performance_failure_path_audits_for_check("serialization_failures")
        performance_failure_path_audits_for_surface("throughput_optimizer")
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)

    def test_failure_audit_metadata_does_not_declare_active_terms(self) -> None:
        registry = performance_failure_path_audit_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *registry.required_checks,
                *registry.applicable_required_checks,
                *registry.not_applicable_required_checks,
                *(
                    field
                    for record in registry.records
                    for field in (
                        record.audit_id,
                        record.check_kind,
                        record.failure_response_boundary,
                        *record.evidence,
                        *record.invariant_assertions,
                        *record.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "measure_performance(",
            "execute_workflow(",
            "execute_benchmark(",
            "allocate_resources(",
            "enforce_capacity(",
            "enforce_budget(",
            "route_provider(",
            "route_model(",
            "control_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
