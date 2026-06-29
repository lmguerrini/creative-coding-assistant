import unittest

from creative_coding_assistant.orchestration import (
    ThroughputOptimizationPlan,
    detect_bottlenecks,
    optimize_streaming,
    optimize_throughput,
    plan_async_execution,
    plan_load_balancer,
    throughput_optimization_candidate_by_id,
    throughput_optimization_candidates_for_status,
)

REQUIRED_THROUGHPUT_FIELDS = {
    "candidate_id",
    "throughput_id",
    "throughput_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_candidate_ids",
    "advisory_throughput_units",
    "advisory_capacity_units",
    "advisory_backpressure_units",
    "advisory_throughput_score",
    "throughput_pressure",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "throughput_optimizer_planning_implemented",
    "throughput_runtime_optimization_implemented",
    "throughput_measurement_implemented",
    "concurrency_limit_change_implemented",
    "queue_management_runtime_implemented",
    "stream_chunk_batching_runtime_implemented",
    "request_distribution_implemented",
    "load_balancing_runtime_implemented",
    "capacity_enforcement_implemented",
    "provider_model_routing_implemented",
    "workflow_timing_change_implemented",
    "workflow_graph_mutation_implemented",
    "graph_compilation_implemented",
    "workflow_execution_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class ThroughputOptimizerTests(unittest.TestCase):
    def test_default_plan_derives_advisory_throughput_candidates(self) -> None:
        async_plan = plan_async_execution()
        streaming = optimize_streaming(async_execution=async_plan)
        load = plan_load_balancer(async_execution=async_plan)
        bottlenecks = detect_bottlenecks(load_balancer=load)
        plan = optimize_throughput(
            async_execution=async_plan,
            streaming_optimization=streaming,
            load_balancer=load,
            bottleneck_detection=bottlenecks,
        )

        self.assertEqual(plan.role, "throughput_optimizer")
        self.assertEqual(
            plan.serialization_version,
            "throughput_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_async_execution_serialization_version,
            async_plan.serialization_version,
        )
        self.assertEqual(
            plan.source_streaming_optimization_serialization_version,
            streaming.serialization_version,
        )
        self.assertEqual(
            plan.source_load_balancer_serialization_version,
            load.serialization_version,
        )
        self.assertEqual(
            plan.source_bottleneck_detection_serialization_version,
            bottlenecks.serialization_version,
        )
        self.assertEqual(plan.candidate_count, 5)
        self.assertEqual(
            plan.candidate_ids,
            (
                "throughput_optimizer::async_slot_throughput",
                "throughput_optimizer::stream_batch_throughput",
                "throughput_optimizer::load_capacity_throughput",
                "throughput_optimizer::bottleneck_backpressure",
                "throughput_optimizer::routing_boundary",
            ),
        )
        self.assertEqual(plan.throughput_candidate_count, 3)
        self.assertEqual(plan.capacity_guardrail_count, 1)
        self.assertEqual(plan.boundary_guardrail_count, 1)
        self.assertGreater(plan.total_advisory_throughput_units, 0)
        self.assertGreater(plan.total_advisory_capacity_units, 0)
        self.assertGreater(plan.total_advisory_backpressure_units, 0)
        self.assertGreater(plan.highest_advisory_throughput_score, 0)
        self.assertGreater(plan.total_advisory_throughput_score, 0)
        self.assertEqual(plan.throughput_optimization_pressure, "guarded")
        self.assertIn("does not measure throughput", plan.authority_boundary)
        self.assertTrue(plan.throughput_optimizer_planning_implemented)
        self.assertFalse(plan.throughput_runtime_optimization_implemented)
        self.assertFalse(plan.throughput_measurement_implemented)
        self.assertFalse(plan.concurrency_limit_change_implemented)
        self.assertFalse(plan.queue_management_runtime_implemented)
        self.assertFalse(plan.stream_chunk_batching_runtime_implemented)
        self.assertFalse(plan.request_distribution_implemented)
        self.assertFalse(plan.load_balancing_runtime_implemented)
        self.assertFalse(plan.capacity_enforcement_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.workflow_timing_change_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.graph_compilation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.node_handler_invocation_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_candidates_preserve_throughput_boundaries(self) -> None:
        plan = optimize_throughput()
        async_candidate = throughput_optimization_candidate_by_id(
            "throughput_optimizer::async_slot_throughput",
            plan,
        )
        routing_boundary = throughput_optimization_candidate_by_id(
            "throughput_optimizer::routing_boundary",
            plan,
        )

        self.assertIsNotNone(async_candidate)
        self.assertIsNotNone(routing_boundary)
        assert async_candidate is not None
        assert routing_boundary is not None
        self.assertEqual(async_candidate.status, "throughput_candidate")
        self.assertGreater(async_candidate.advisory_throughput_units, 0)
        self.assertEqual(routing_boundary.status, "boundary_guardrail")
        self.assertEqual(routing_boundary.throughput_pressure, "guarded")

        for candidate in plan.candidates:
            self.assertEqual(
                set(candidate.model_dump(mode="json")),
                REQUIRED_THROUGHPUT_FIELDS,
            )
            self.assertEqual(
                candidate.serialization_version,
                "throughput_optimization_candidate.v1",
            )
            self.assertIn(
                "throughput_runtime_optimization",
                candidate.blocked_runtime_behaviors,
            )
            self.assertFalse(candidate.throughput_runtime_optimization_implemented)
            self.assertFalse(candidate.throughput_measurement_implemented)
            self.assertFalse(candidate.concurrency_limit_change_implemented)
            self.assertFalse(candidate.queue_management_runtime_implemented)
            self.assertFalse(candidate.stream_chunk_batching_runtime_implemented)
            self.assertFalse(candidate.request_distribution_implemented)
            self.assertFalse(candidate.load_balancing_runtime_implemented)
            self.assertFalse(candidate.capacity_enforcement_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.workflow_timing_change_implemented)
            self.assertFalse(candidate.workflow_graph_mutation_implemented)
            self.assertFalse(candidate.graph_compilation_implemented)
            self.assertFalse(candidate.workflow_execution_implemented)
            self.assertFalse(candidate.agent_invocation_implemented)
            self.assertFalse(candidate.node_handler_invocation_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.prompt_mutation_implemented)
            self.assertFalse(candidate.persistent_storage_write_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.advisory_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = optimize_throughput()
        throughput = throughput_optimization_candidates_for_status(
            "throughput_candidate",
            plan,
        )
        capacity = throughput_optimization_candidates_for_status(
            "capacity_guardrail",
            plan,
        )
        boundary = throughput_optimization_candidates_for_status(
            "boundary_guardrail",
            plan,
        )
        missing = throughput_optimization_candidate_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in throughput),
            plan.throughput_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in capacity),
            plan.capacity_guardrail_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in boundary),
            plan.boundary_guardrail_candidate_ids,
        )

    def test_plan_rejects_mismatched_candidate_totals(self) -> None:
        plan = optimize_throughput()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            ThroughputOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["total_advisory_throughput_units"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "total_advisory_throughput_units must match",
        ):
            ThroughputOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["throughput_optimization_pressure"] = "low"

        with self.assertRaisesRegex(
            ValueError,
            "throughput_optimization_pressure must match",
        ):
            ThroughputOptimizationPlan(**payload)

    def test_plan_does_not_declare_runtime_throughput_terms(self) -> None:
        plan = optimize_throughput()
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
            "optimize_runtime_throughput(",
            "measure_throughput(",
            "set_concurrency_limit(",
            "resize_queue(",
            "batch_stream_chunks(",
            "distribute_request(",
            "balance_load(",
            "enforce_capacity(",
            "route_provider(",
            "execute_workflow(",
            "invoke_agent(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
