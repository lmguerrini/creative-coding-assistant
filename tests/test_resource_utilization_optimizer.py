import unittest

from creative_coding_assistant.orchestration import (
    ResourceUtilizationOptimizationPlan,
    detect_performance_regressions,
    optimize_reasoning_budget,
    optimize_resource_utilization,
    optimize_throughput,
    plan_execution_profiling,
    plan_performance_benchmarking,
    resource_utilization_recommendation_by_id,
    resource_utilization_recommendations_for_status,
)

REQUIRED_RESOURCE_UTILIZATION_FIELDS = {
    "recommendation_id",
    "utilization_id",
    "utilization_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "advisory_resource_units",
    "advisory_reserve_units",
    "advisory_pressure_units",
    "advisory_utilization_score",
    "utilization_pressure",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "resource_utilization_optimizer_planning_implemented",
    "resource_allocation_implemented",
    "runtime_resource_measurement_implemented",
    "cpu_memory_measurement_implemented",
    "concurrency_limit_change_implemented",
    "queue_management_runtime_implemented",
    "autoscaling_implemented",
    "capacity_enforcement_implemented",
    "benchmark_execution_implemented",
    "runtime_profiling_implemented",
    "budget_enforcement_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
    "workflow_timing_change_implemented",
    "workflow_graph_mutation_implemented",
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


class ResourceUtilizationOptimizerTests(unittest.TestCase):
    def test_default_plan_derives_advisory_utilization_recommendations(
        self,
    ) -> None:
        throughput = optimize_throughput()
        profiling = plan_execution_profiling()
        benchmarking = plan_performance_benchmarking(
            throughput_optimization=throughput,
            execution_profiling=profiling,
        )
        reasoning = optimize_reasoning_budget(performance_benchmarking=benchmarking)
        regression = detect_performance_regressions(
            performance_benchmarking=benchmarking,
            reasoning_budget=reasoning,
        )
        plan = optimize_resource_utilization(
            throughput_optimization=throughput,
            execution_profiling=profiling,
            performance_benchmarking=benchmarking,
            reasoning_budget=reasoning,
            performance_regression=regression,
        )

        self.assertEqual(plan.role, "resource_utilization_optimizer")
        self.assertEqual(
            plan.serialization_version,
            "resource_utilization_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_throughput_optimization_serialization_version,
            throughput.serialization_version,
        )
        self.assertEqual(
            plan.source_execution_profiling_serialization_version,
            profiling.serialization_version,
        )
        self.assertEqual(
            plan.source_performance_benchmarking_serialization_version,
            benchmarking.serialization_version,
        )
        self.assertEqual(
            plan.source_reasoning_budget_serialization_version,
            reasoning.serialization_version,
        )
        self.assertEqual(
            plan.source_performance_regression_serialization_version,
            regression.serialization_version,
        )
        self.assertEqual(plan.recommendation_count, 6)
        self.assertEqual(
            plan.recommendation_ids,
            (
                "resource_utilization::throughput_capacity_utilization",
                "resource_utilization::profiling_scope_utilization",
                "resource_utilization::benchmark_workload_utilization",
                "resource_utilization::reasoning_budget_utilization",
                "resource_utilization::regression_baseline_utilization",
                "resource_utilization::runtime_resource_boundary",
            ),
        )
        self.assertEqual(plan.optimization_candidate_count, 2)
        self.assertEqual(plan.capacity_guardrail_count, 1)
        self.assertEqual(plan.review_guardrail_count, 2)
        self.assertEqual(plan.boundary_guardrail_count, 1)
        self.assertGreater(plan.total_advisory_resource_units, 0)
        self.assertGreater(plan.total_advisory_reserve_units, 0)
        self.assertGreater(plan.total_advisory_pressure_units, 0)
        self.assertGreater(plan.highest_advisory_utilization_score, 0)
        self.assertEqual(plan.resource_utilization_pressure, "guarded")
        self.assertIn("does not allocate resources", plan.authority_boundary)
        self.assertTrue(plan.resource_utilization_optimizer_planning_implemented)
        self.assertFalse(plan.resource_allocation_implemented)
        self.assertFalse(plan.runtime_resource_measurement_implemented)
        self.assertFalse(plan.cpu_memory_measurement_implemented)
        self.assertFalse(plan.concurrency_limit_change_implemented)
        self.assertFalse(plan.queue_management_runtime_implemented)
        self.assertFalse(plan.autoscaling_implemented)
        self.assertFalse(plan.capacity_enforcement_implemented)
        self.assertFalse(plan.benchmark_execution_implemented)
        self.assertFalse(plan.runtime_profiling_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_timing_change_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.node_handler_invocation_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_recommendations_preserve_resource_boundaries(self) -> None:
        plan = optimize_resource_utilization()
        throughput = resource_utilization_recommendation_by_id(
            "resource_utilization::throughput_capacity_utilization",
            plan,
        )
        boundary = resource_utilization_recommendation_by_id(
            "resource_utilization::runtime_resource_boundary",
            plan,
        )

        self.assertIsNotNone(throughput)
        self.assertIsNotNone(boundary)
        assert throughput is not None
        assert boundary is not None
        self.assertEqual(throughput.status, "optimization_candidate")
        self.assertGreater(throughput.advisory_resource_units, 0)
        self.assertGreater(throughput.advisory_utilization_score, 0)
        self.assertEqual(boundary.status, "boundary_guardrail")
        self.assertEqual(boundary.utilization_pressure, "guarded")
        self.assertEqual(boundary.advisory_utilization_score, 0)

        for recommendation in plan.recommendations:
            self.assertEqual(
                set(recommendation.model_dump(mode="json")),
                REQUIRED_RESOURCE_UTILIZATION_FIELDS,
            )
            self.assertEqual(
                recommendation.serialization_version,
                "resource_utilization_recommendation.v1",
            )
            self.assertIn(
                "resource_allocation",
                recommendation.blocked_runtime_behaviors,
            )
            self.assertTrue(
                recommendation.resource_utilization_optimizer_planning_implemented,
            )
            self.assertFalse(recommendation.resource_allocation_implemented)
            self.assertFalse(recommendation.runtime_resource_measurement_implemented)
            self.assertFalse(recommendation.cpu_memory_measurement_implemented)
            self.assertFalse(recommendation.concurrency_limit_change_implemented)
            self.assertFalse(recommendation.queue_management_runtime_implemented)
            self.assertFalse(recommendation.autoscaling_implemented)
            self.assertFalse(recommendation.capacity_enforcement_implemented)
            self.assertFalse(recommendation.benchmark_execution_implemented)
            self.assertFalse(recommendation.runtime_profiling_implemented)
            self.assertFalse(recommendation.budget_enforcement_implemented)
            self.assertFalse(recommendation.provider_model_routing_implemented)
            self.assertFalse(recommendation.workflow_control_implemented)
            self.assertFalse(recommendation.workflow_timing_change_implemented)
            self.assertFalse(recommendation.workflow_graph_mutation_implemented)
            self.assertFalse(recommendation.workflow_execution_implemented)
            self.assertFalse(recommendation.agent_invocation_implemented)
            self.assertFalse(recommendation.node_handler_invocation_implemented)
            self.assertFalse(recommendation.retry_triggering_implemented)
            self.assertFalse(recommendation.prompt_mutation_implemented)
            self.assertFalse(recommendation.persistent_storage_write_implemented)
            self.assertFalse(recommendation.generated_output_mutation_implemented)
            self.assertTrue(recommendation.advisory_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = optimize_resource_utilization()
        optimization = resource_utilization_recommendations_for_status(
            "optimization_candidate",
            plan,
        )
        capacity = resource_utilization_recommendations_for_status(
            "capacity_guardrail",
            plan,
        )
        review = resource_utilization_recommendations_for_status(
            "review_guardrail",
            plan,
        )
        boundary = resource_utilization_recommendations_for_status(
            "boundary_guardrail",
            plan,
        )
        missing = resource_utilization_recommendation_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(
            tuple(recommendation.recommendation_id for recommendation in optimization),
            plan.optimization_candidate_ids,
        )
        self.assertEqual(
            tuple(recommendation.recommendation_id for recommendation in capacity),
            plan.capacity_guardrail_ids,
        )
        self.assertEqual(
            tuple(recommendation.recommendation_id for recommendation in review),
            plan.review_guardrail_ids,
        )
        self.assertEqual(
            tuple(recommendation.recommendation_id for recommendation in boundary),
            plan.boundary_guardrail_ids,
        )

    def test_plan_rejects_mismatched_recommendation_totals(self) -> None:
        plan = optimize_resource_utilization()
        payload = plan.model_dump(mode="json")
        payload["recommendation_ids"] = (
            "missing",
            *tuple(payload["recommendation_ids"][1:]),
        )

        with self.assertRaisesRegex(ValueError, "recommendation_ids must match"):
            ResourceUtilizationOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["total_advisory_resource_units"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "total_advisory_resource_units must match",
        ):
            ResourceUtilizationOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["resource_utilization_pressure"] = "low"

        with self.assertRaisesRegex(
            ValueError,
            "resource_utilization_pressure must match",
        ):
            ResourceUtilizationOptimizationPlan(**payload)

    def test_plan_does_not_declare_runtime_resource_terms(self) -> None:
        plan = optimize_resource_utilization()
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for recommendation in plan.recommendations
                    for field in (
                        recommendation.recommendation_id,
                        recommendation.source_id,
                        *recommendation.evidence,
                        *recommendation.advisory_actions,
                        *recommendation.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "allocate_resources(",
            "measure_cpu(",
            "measure_memory(",
            "change_concurrency_limit(",
            "manage_queue(",
            "autoscale(",
            "enforce_capacity(",
            "execute_benchmark(",
            "install_profiler(",
            "enforce_budget(",
            "route_provider(",
            "control_workflow(",
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
