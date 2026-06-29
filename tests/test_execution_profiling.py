import unittest

from creative_coding_assistant.orchestration import (
    ExecutionProfilingPlan,
    agent_performance_tracking_foundation_registry,
    analyze_assistant_execution_graph,
    execution_profile_candidate_by_id,
    execution_profile_candidates_for_status,
    optimize_latency,
    plan_execution_profiling,
    plan_load_balancer,
)

REQUIRED_EXECUTION_PROFILE_FIELDS = {
    "candidate_id",
    "profile_id",
    "profile_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "profiled_node_count",
    "profiled_agent_count",
    "blocking_input_count",
    "advisory_profile_score",
    "profile_pressure",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "execution_profiling_planning_implemented",
    "runtime_profiling_implemented",
    "timing_measurement_implemented",
    "profiler_hook_installation_implemented",
    "runtime_trace_collection_implemented",
    "latency_measurement_implemented",
    "latency_threshold_evaluation_implemented",
    "load_balancing_runtime_implemented",
    "workflow_timing_change_implemented",
    "workflow_graph_mutation_implemented",
    "graph_compilation_implemented",
    "workflow_execution_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "provider_model_routing_implemented",
    "runtime_selection_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class ExecutionProfilingTests(unittest.TestCase):
    def test_default_plan_derives_advisory_execution_profiles(self) -> None:
        graph = analyze_assistant_execution_graph()
        performance = agent_performance_tracking_foundation_registry()
        latency = optimize_latency()
        load = plan_load_balancer(latency_optimization=latency)
        plan = plan_execution_profiling(
            execution_graph=graph,
            performance_registry=performance,
            latency_optimization=latency,
            load_balancer=load,
        )

        self.assertEqual(plan.role, "execution_profiler")
        self.assertEqual(
            plan.serialization_version,
            "execution_profiling_plan.v1",
        )
        self.assertEqual(
            plan.source_graph_serialization_version,
            graph.serialization_version,
        )
        self.assertEqual(
            plan.source_performance_tracking_serialization_version,
            performance.serialization_version,
        )
        self.assertEqual(
            plan.source_latency_optimization_serialization_version,
            latency.serialization_version,
        )
        self.assertEqual(
            plan.source_load_balancer_serialization_version,
            load.serialization_version,
        )
        self.assertEqual(plan.source_graph_node_count, graph.node_count)
        self.assertEqual(plan.source_agent_profile_count, performance.profile_count)
        self.assertTrue(plan.failure_path_reachable)
        self.assertEqual(plan.candidate_count, 5)
        self.assertEqual(
            plan.candidate_ids,
            (
                "execution_profiling::critical_path_profile",
                "execution_profiling::agent_latency_profile",
                "execution_profiling::load_pressure_profile",
                "execution_profiling::failure_path_profile",
                "execution_profiling::measurement_boundary_profile",
            ),
        )
        self.assertEqual(plan.profile_candidate_count, 3)
        self.assertEqual(plan.failure_guardrail_count, 1)
        self.assertEqual(plan.measurement_guardrail_count, 1)
        self.assertGreater(plan.total_profiled_node_count, 0)
        self.assertEqual(plan.total_profiled_agent_count, performance.profile_count)
        self.assertGreaterEqual(
            plan.total_blocking_input_count,
            latency.total_blocking_input_count,
        )
        self.assertGreater(plan.highest_advisory_profile_score, 0)
        self.assertGreater(plan.total_advisory_profile_score, 0)
        self.assertEqual(plan.execution_profile_pressure, "guarded")
        self.assertIn("does not measure timing", plan.authority_boundary)
        self.assertTrue(plan.execution_profiling_planning_implemented)
        self.assertFalse(plan.runtime_profiling_implemented)
        self.assertFalse(plan.timing_measurement_implemented)
        self.assertFalse(plan.profiler_hook_installation_implemented)
        self.assertFalse(plan.runtime_trace_collection_implemented)
        self.assertFalse(plan.latency_measurement_implemented)
        self.assertFalse(plan.latency_threshold_evaluation_implemented)
        self.assertFalse(plan.load_balancing_runtime_implemented)
        self.assertFalse(plan.workflow_timing_change_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.graph_compilation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.node_handler_invocation_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.runtime_selection_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_candidates_preserve_measurement_and_execution_boundaries(self) -> None:
        plan = plan_execution_profiling()
        critical = execution_profile_candidate_by_id(
            "execution_profiling::critical_path_profile",
            plan,
        )
        measurement = execution_profile_candidate_by_id(
            "execution_profiling::measurement_boundary_profile",
            plan,
        )

        self.assertIsNotNone(critical)
        self.assertIsNotNone(measurement)
        assert critical is not None
        assert measurement is not None
        self.assertEqual(critical.status, "profile_candidate")
        self.assertGreater(critical.profiled_node_count, 0)
        self.assertEqual(measurement.status, "measurement_guardrail")
        self.assertEqual(measurement.advisory_profile_score, 0)
        self.assertEqual(measurement.profile_pressure, "guarded")

        for candidate in plan.candidates:
            self.assertEqual(
                set(candidate.model_dump(mode="json")),
                REQUIRED_EXECUTION_PROFILE_FIELDS,
            )
            self.assertEqual(
                candidate.serialization_version,
                "execution_profile_candidate.v1",
            )
            self.assertIn("runtime_profiling", candidate.blocked_runtime_behaviors)
            self.assertIn("timing_measurement", candidate.blocked_runtime_behaviors)
            self.assertTrue(candidate.execution_profiling_planning_implemented)
            self.assertFalse(candidate.runtime_profiling_implemented)
            self.assertFalse(candidate.timing_measurement_implemented)
            self.assertFalse(candidate.profiler_hook_installation_implemented)
            self.assertFalse(candidate.runtime_trace_collection_implemented)
            self.assertFalse(candidate.latency_measurement_implemented)
            self.assertFalse(candidate.latency_threshold_evaluation_implemented)
            self.assertFalse(candidate.load_balancing_runtime_implemented)
            self.assertFalse(candidate.workflow_timing_change_implemented)
            self.assertFalse(candidate.workflow_graph_mutation_implemented)
            self.assertFalse(candidate.graph_compilation_implemented)
            self.assertFalse(candidate.workflow_execution_implemented)
            self.assertFalse(candidate.agent_invocation_implemented)
            self.assertFalse(candidate.node_handler_invocation_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.runtime_selection_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.prompt_mutation_implemented)
            self.assertFalse(candidate.persistent_storage_write_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.advisory_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = plan_execution_profiling()
        profile_candidates = execution_profile_candidates_for_status(
            "profile_candidate",
            plan,
        )
        measurement = execution_profile_candidates_for_status(
            "measurement_guardrail",
            plan,
        )
        failure = execution_profile_candidates_for_status("failure_guardrail", plan)
        missing = execution_profile_candidate_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in profile_candidates),
            plan.profile_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in measurement),
            plan.measurement_guardrail_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in failure),
            plan.failure_guardrail_candidate_ids,
        )

    def test_plan_rejects_mismatched_candidate_totals(self) -> None:
        plan = plan_execution_profiling()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            ExecutionProfilingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["total_profiled_node_count"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "total_profiled_node_count must match",
        ):
            ExecutionProfilingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["execution_profile_pressure"] = "low"

        with self.assertRaisesRegex(
            ValueError,
            "execution_profile_pressure must match",
        ):
            ExecutionProfilingPlan(**payload)

    def test_plan_does_not_declare_runtime_profiling_terms(self) -> None:
        plan = plan_execution_profiling()
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
            "measure_timing(",
            "install_profiler(",
            "collect_trace(",
            "execute_workflow(",
            "invoke_agent(",
            "invoke_node_handler(",
            "route_provider(",
            "select_runtime(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
