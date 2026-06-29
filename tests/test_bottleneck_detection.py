import unittest

from creative_coding_assistant.orchestration import (
    BottleneckDetectionPlan,
    bottleneck_candidate_by_id,
    bottleneck_candidates_for_status,
    detect_bottlenecks,
    optimize_latency,
    plan_execution_profiling,
    plan_execution_replay,
    plan_load_balancer,
)

REQUIRED_BOTTLENECK_FIELDS = {
    "candidate_id",
    "bottleneck_id",
    "bottleneck_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_candidate_ids",
    "signal_count",
    "blocking_input_count",
    "advisory_severity_score",
    "severity",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "bottleneck_detection_planning_implemented",
    "runtime_bottleneck_detection_implemented",
    "runtime_latency_measurement_implemented",
    "profiler_hook_installation_implemented",
    "runtime_trace_collection_implemented",
    "load_balancing_runtime_implemented",
    "latency_based_routing_implemented",
    "provider_model_routing_implemented",
    "runtime_selection_implemented",
    "workflow_timing_change_implemented",
    "workflow_graph_mutation_implemented",
    "graph_compilation_implemented",
    "workflow_execution_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "execution_replay_execution_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class BottleneckDetectionTests(unittest.TestCase):
    def test_default_plan_derives_advisory_bottleneck_signals(self) -> None:
        latency = optimize_latency()
        load = plan_load_balancer(latency_optimization=latency)
        profiling = plan_execution_profiling(
            latency_optimization=latency,
            load_balancer=load,
        )
        replay = plan_execution_replay(execution_profiling=profiling)
        plan = detect_bottlenecks(
            latency_optimization=latency,
            load_balancer=load,
            execution_profiling=profiling,
            execution_replay=replay,
        )

        self.assertEqual(plan.role, "bottleneck_detector")
        self.assertEqual(
            plan.serialization_version,
            "bottleneck_detection_plan.v1",
        )
        self.assertEqual(
            plan.source_latency_optimization_serialization_version,
            latency.serialization_version,
        )
        self.assertEqual(
            plan.source_load_balancer_serialization_version,
            load.serialization_version,
        )
        self.assertEqual(
            plan.source_execution_profiling_serialization_version,
            profiling.serialization_version,
        )
        self.assertEqual(
            plan.source_execution_replay_serialization_version,
            replay.serialization_version,
        )
        self.assertEqual(plan.candidate_count, 5)
        self.assertEqual(
            plan.candidate_ids,
            (
                "bottleneck_detection::latency_pressure",
                "bottleneck_detection::load_pressure",
                "bottleneck_detection::profiling_pressure",
                "bottleneck_detection::replay_boundary",
                "bottleneck_detection::routing_boundary",
            ),
        )
        self.assertEqual(plan.bottleneck_candidate_count, 3)
        self.assertEqual(plan.boundary_guardrail_count, 1)
        self.assertEqual(plan.review_only_count, 1)
        self.assertGreater(plan.total_signal_count, 0)
        self.assertGreater(plan.total_blocking_input_count, 0)
        self.assertGreater(plan.highest_advisory_severity_score, 0)
        self.assertGreater(plan.total_advisory_severity_score, 0)
        self.assertEqual(plan.bottleneck_detection_severity, "guarded")
        self.assertIn("does not measure runtime latency", plan.authority_boundary)
        self.assertTrue(plan.bottleneck_detection_planning_implemented)
        self.assertFalse(plan.runtime_bottleneck_detection_implemented)
        self.assertFalse(plan.runtime_latency_measurement_implemented)
        self.assertFalse(plan.profiler_hook_installation_implemented)
        self.assertFalse(plan.runtime_trace_collection_implemented)
        self.assertFalse(plan.load_balancing_runtime_implemented)
        self.assertFalse(plan.latency_based_routing_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.runtime_selection_implemented)
        self.assertFalse(plan.workflow_timing_change_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.graph_compilation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.node_handler_invocation_implemented)
        self.assertFalse(plan.execution_replay_execution_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_candidates_preserve_detection_boundaries(self) -> None:
        plan = detect_bottlenecks()
        latency = bottleneck_candidate_by_id(
            "bottleneck_detection::latency_pressure",
            plan,
        )
        boundary = bottleneck_candidate_by_id(
            "bottleneck_detection::replay_boundary",
            plan,
        )

        self.assertIsNotNone(latency)
        self.assertIsNotNone(boundary)
        assert latency is not None
        assert boundary is not None
        self.assertEqual(latency.status, "bottleneck_candidate")
        self.assertGreater(latency.signal_count, 0)
        self.assertEqual(boundary.status, "boundary_guardrail")
        self.assertEqual(boundary.severity, "guarded")

        for candidate in plan.candidates:
            self.assertEqual(
                set(candidate.model_dump(mode="json")),
                REQUIRED_BOTTLENECK_FIELDS,
            )
            self.assertEqual(
                candidate.serialization_version,
                "bottleneck_candidate.v1",
            )
            self.assertIn(
                "runtime_bottleneck_detection",
                candidate.blocked_runtime_behaviors,
            )
            self.assertFalse(candidate.runtime_bottleneck_detection_implemented)
            self.assertFalse(candidate.runtime_latency_measurement_implemented)
            self.assertFalse(candidate.profiler_hook_installation_implemented)
            self.assertFalse(candidate.runtime_trace_collection_implemented)
            self.assertFalse(candidate.load_balancing_runtime_implemented)
            self.assertFalse(candidate.latency_based_routing_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.runtime_selection_implemented)
            self.assertFalse(candidate.workflow_timing_change_implemented)
            self.assertFalse(candidate.workflow_graph_mutation_implemented)
            self.assertFalse(candidate.graph_compilation_implemented)
            self.assertFalse(candidate.workflow_execution_implemented)
            self.assertFalse(candidate.agent_invocation_implemented)
            self.assertFalse(candidate.node_handler_invocation_implemented)
            self.assertFalse(candidate.execution_replay_execution_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.prompt_mutation_implemented)
            self.assertFalse(candidate.persistent_storage_write_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.advisory_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = detect_bottlenecks()
        bottlenecks = bottleneck_candidates_for_status(
            "bottleneck_candidate",
            plan,
        )
        guardrails = bottleneck_candidates_for_status("boundary_guardrail", plan)
        review = bottleneck_candidates_for_status("review_only", plan)
        missing = bottleneck_candidate_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in bottlenecks),
            plan.bottleneck_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in guardrails),
            plan.boundary_guardrail_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in review),
            plan.review_only_candidate_ids,
        )

    def test_plan_rejects_mismatched_candidate_totals(self) -> None:
        plan = detect_bottlenecks()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            BottleneckDetectionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["total_signal_count"] += 1

        with self.assertRaisesRegex(ValueError, "total_signal_count must match"):
            BottleneckDetectionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["bottleneck_detection_severity"] = "low"

        with self.assertRaisesRegex(
            ValueError,
            "bottleneck_detection_severity must match",
        ):
            BottleneckDetectionPlan(**payload)

    def test_plan_does_not_declare_runtime_detection_terms(self) -> None:
        plan = detect_bottlenecks()
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for candidate in plan.candidates
                    for field in (
                        candidate.candidate_id,
                        candidate.source_id,
                        *candidate.evidence,
                        *candidate.advisory_actions,
                        *candidate.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "measure_latency(",
            "install_profiler(",
            "collect_trace(",
            "rebalance_load(",
            "route_provider(",
            "select_runtime(",
            "execute_workflow(",
            "invoke_agent(",
            "invoke_node_handler(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
