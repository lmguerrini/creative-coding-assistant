import unittest

from creative_coding_assistant.orchestration import (
    PerformanceBenchmarkingPlan,
    optimize_latency,
    optimize_throughput,
    performance_benchmark_scenario_by_id,
    performance_benchmark_scenarios_for_status,
    plan_execution_profiling,
    plan_performance_benchmarking,
    predict_performance,
)

REQUIRED_PERFORMANCE_BENCHMARK_FIELDS = {
    "scenario_id",
    "benchmark_id",
    "benchmark_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_candidate_ids",
    "advisory_sample_count",
    "advisory_benchmark_units",
    "baseline_reference_count",
    "benchmark_priority_score",
    "benchmark_readiness",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "performance_benchmarking_planning_implemented",
    "benchmark_execution_implemented",
    "runtime_performance_measurement_implemented",
    "timer_collection_implemented",
    "profiler_hook_installation_implemented",
    "runtime_trace_collection_implemented",
    "workload_execution_implemented",
    "workflow_replay_execution_implemented",
    "workflow_execution_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
    "workflow_timing_change_implemented",
    "workflow_graph_mutation_implemented",
    "graph_compilation_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class PerformanceBenchmarkingTests(unittest.TestCase):
    def test_default_plan_derives_advisory_benchmark_scenarios(self) -> None:
        latency = optimize_latency()
        profiling = plan_execution_profiling(latency_optimization=latency)
        throughput = optimize_throughput()
        prediction = predict_performance(
            throughput_optimization=throughput,
            latency_optimization=latency,
            execution_profiling=profiling,
        )
        plan = plan_performance_benchmarking(
            performance_prediction=prediction,
            throughput_optimization=throughput,
            latency_optimization=latency,
            execution_profiling=profiling,
        )

        self.assertEqual(plan.role, "performance_benchmark_planner")
        self.assertEqual(
            plan.serialization_version,
            "performance_benchmarking_plan.v1",
        )
        self.assertEqual(
            plan.source_performance_prediction_serialization_version,
            prediction.serialization_version,
        )
        self.assertEqual(
            plan.source_throughput_optimization_serialization_version,
            throughput.serialization_version,
        )
        self.assertEqual(
            plan.source_latency_optimization_serialization_version,
            latency.serialization_version,
        )
        self.assertEqual(
            plan.source_execution_profiling_serialization_version,
            profiling.serialization_version,
        )
        self.assertEqual(plan.scenario_count, 4)
        self.assertEqual(
            plan.scenario_ids,
            (
                "performance_benchmark::prediction_baseline",
                "performance_benchmark::throughput_benchmark",
                "performance_benchmark::latency_benchmark",
                "performance_benchmark::profiling_boundary",
            ),
        )
        self.assertEqual(plan.baseline_candidate_count, 1)
        self.assertEqual(plan.benchmark_candidate_count, 2)
        self.assertEqual(plan.guardrail_count, 1)
        self.assertGreater(plan.total_advisory_sample_count, 0)
        self.assertGreater(plan.total_advisory_benchmark_units, 0)
        self.assertGreater(plan.highest_benchmark_priority_score, 0)
        self.assertGreater(plan.total_benchmark_priority_score, 0)
        self.assertEqual(plan.benchmarking_readiness, "guarded")
        self.assertIn("does not execute benchmarks", plan.authority_boundary)
        self.assertTrue(plan.performance_benchmarking_planning_implemented)
        self.assertFalse(plan.benchmark_execution_implemented)
        self.assertFalse(plan.runtime_performance_measurement_implemented)
        self.assertFalse(plan.timer_collection_implemented)
        self.assertFalse(plan.profiler_hook_installation_implemented)
        self.assertFalse(plan.runtime_trace_collection_implemented)
        self.assertFalse(plan.workload_execution_implemented)
        self.assertFalse(plan.workflow_replay_execution_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_timing_change_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.graph_compilation_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.node_handler_invocation_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_scenarios_preserve_benchmark_boundaries(self) -> None:
        plan = plan_performance_benchmarking()
        throughput = performance_benchmark_scenario_by_id(
            "performance_benchmark::throughput_benchmark",
            plan,
        )
        guardrail = performance_benchmark_scenario_by_id(
            "performance_benchmark::profiling_boundary",
            plan,
        )

        self.assertIsNotNone(throughput)
        self.assertIsNotNone(guardrail)
        assert throughput is not None
        assert guardrail is not None
        self.assertEqual(throughput.status, "benchmark_candidate")
        self.assertGreater(throughput.advisory_sample_count, 0)
        self.assertEqual(guardrail.status, "guardrail")
        self.assertEqual(guardrail.benchmark_readiness, "guarded")

        for scenario in plan.scenarios:
            self.assertEqual(
                set(scenario.model_dump(mode="json")),
                REQUIRED_PERFORMANCE_BENCHMARK_FIELDS,
            )
            self.assertEqual(
                scenario.serialization_version,
                "performance_benchmark_scenario.v1",
            )
            self.assertIn("benchmark_execution", scenario.blocked_runtime_behaviors)
            self.assertTrue(scenario.performance_benchmarking_planning_implemented)
            self.assertFalse(scenario.benchmark_execution_implemented)
            self.assertFalse(scenario.runtime_performance_measurement_implemented)
            self.assertFalse(scenario.timer_collection_implemented)
            self.assertFalse(scenario.profiler_hook_installation_implemented)
            self.assertFalse(scenario.runtime_trace_collection_implemented)
            self.assertFalse(scenario.workload_execution_implemented)
            self.assertFalse(scenario.workflow_replay_execution_implemented)
            self.assertFalse(scenario.workflow_execution_implemented)
            self.assertFalse(scenario.provider_model_routing_implemented)
            self.assertFalse(scenario.workflow_control_implemented)
            self.assertFalse(scenario.workflow_timing_change_implemented)
            self.assertFalse(scenario.workflow_graph_mutation_implemented)
            self.assertFalse(scenario.graph_compilation_implemented)
            self.assertFalse(scenario.agent_invocation_implemented)
            self.assertFalse(scenario.node_handler_invocation_implemented)
            self.assertFalse(scenario.retry_triggering_implemented)
            self.assertFalse(scenario.prompt_mutation_implemented)
            self.assertFalse(scenario.persistent_storage_write_implemented)
            self.assertFalse(scenario.generated_output_mutation_implemented)
            self.assertTrue(scenario.advisory_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = plan_performance_benchmarking()
        baseline = performance_benchmark_scenarios_for_status(
            "baseline_candidate",
            plan,
        )
        benchmarks = performance_benchmark_scenarios_for_status(
            "benchmark_candidate",
            plan,
        )
        guardrails = performance_benchmark_scenarios_for_status("guardrail", plan)
        missing = performance_benchmark_scenario_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(
            tuple(scenario.scenario_id for scenario in baseline),
            plan.baseline_candidate_ids,
        )
        self.assertEqual(
            tuple(scenario.scenario_id for scenario in benchmarks),
            plan.benchmark_candidate_ids,
        )
        self.assertEqual(
            tuple(scenario.scenario_id for scenario in guardrails),
            plan.guardrail_scenario_ids,
        )

    def test_plan_rejects_mismatched_scenario_totals(self) -> None:
        plan = plan_performance_benchmarking()
        payload = plan.model_dump(mode="json")
        payload["scenario_ids"] = ("missing",) + tuple(payload["scenario_ids"][1:])

        with self.assertRaisesRegex(ValueError, "scenario_ids must match"):
            PerformanceBenchmarkingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["total_advisory_sample_count"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "total_advisory_sample_count must match",
        ):
            PerformanceBenchmarkingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["benchmarking_readiness"] = "low"

        with self.assertRaisesRegex(ValueError, "benchmarking_readiness must match"):
            PerformanceBenchmarkingPlan(**payload)

    def test_plan_does_not_declare_runtime_benchmark_terms(self) -> None:
        plan = plan_performance_benchmarking()
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for scenario in plan.scenarios
                    for field in (
                        scenario.scenario_id,
                        scenario.source_id,
                        *scenario.evidence,
                        *scenario.advisory_actions,
                        *scenario.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_benchmark(",
            "measure_performance(",
            "collect_timer(",
            "install_profiler(",
            "collect_trace(",
            "run_workload(",
            "replay_workflow(",
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
