import unittest

from creative_coding_assistant.orchestration import (
    ExecutionPathOptimizationPlan,
    analyze_assistant_execution_graph,
    analyze_workflow_cost,
    execution_path_candidate_by_id,
    execution_path_candidates_for_status,
    forecast_execution_cost,
    plan_execution_path_optimization,
    plan_workflow_pruning,
)

REQUIRED_PATH_CANDIDATE_FIELDS = {
    "candidate_id",
    "candidate_kind",
    "status",
    "node_ids",
    "edge_ids",
    "source_scenario_id",
    "forecast_tokens",
    "token_delta_from_worst_case",
    "advisory_rank",
    "optimization_score",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "execution_path_optimization_implemented",
    "execution_path_selection_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_order_mutation_implemented",
    "graph_compilation_implemented",
    "workflow_execution_implemented",
    "node_handler_invocation_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "planning_only",
}


class ExecutionPathOptimizationTests(unittest.TestCase):
    def test_default_plan_derives_advisory_path_candidates(self) -> None:
        graph = analyze_assistant_execution_graph()
        costs = analyze_workflow_cost(execution_graph=graph)
        pruning = plan_workflow_pruning(
            execution_graph=graph,
            cost_analysis=costs,
        )
        forecast = forecast_execution_cost(
            cost_analysis=costs,
            pruning_plan=pruning,
        )
        plan = plan_execution_path_optimization(
            execution_graph=graph,
            cost_analysis=costs,
            cost_forecast=forecast,
            pruning_plan=pruning,
        )

        self.assertEqual(plan.role, "execution_path_optimizer")
        self.assertEqual(
            plan.serialization_version,
            "execution_path_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_graph_serialization_version,
            graph.serialization_version,
        )
        self.assertEqual(
            plan.source_cost_serialization_version,
            costs.serialization_version,
        )
        self.assertEqual(
            plan.source_forecast_serialization_version,
            forecast.serialization_version,
        )
        self.assertEqual(
            plan.source_pruning_serialization_version,
            pruning.serialization_version,
        )
        self.assertEqual(plan.candidate_count, 5)
        self.assertEqual(
            plan.candidate_ids,
            (
                "execution_path::pruning_adjusted_path",
                "execution_path::minimum_success_path",
                "execution_path::single_retry_path",
                "execution_path::worst_case_bound_path",
                "execution_path::failure_normalization_path",
            ),
        )
        self.assertEqual(
            plan.largest_advisory_token_savings,
            max(
                max(0, -candidate.token_delta_from_worst_case)
                for candidate in plan.candidates
            ),
        )
        self.assertEqual(plan.optimization_pressure, "high")
        self.assertIn("does not select execution paths", plan.authority_boundary)
        self.assertTrue(plan.execution_path_optimization_implemented)
        self.assertFalse(plan.execution_path_selection_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_order_mutation_implemented)
        self.assertFalse(plan.graph_compilation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.node_handler_invocation_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.planning_only)

    def test_candidates_cover_path_surfaces_without_selection(self) -> None:
        graph = analyze_assistant_execution_graph()
        plan = plan_execution_path_optimization(execution_graph=graph)
        optimized = execution_path_candidate_by_id(
            "execution_path::pruning_adjusted_path",
            plan,
        )
        single_retry = execution_path_candidate_by_id(
            "execution_path::single_retry_path",
            plan,
        )
        failure = execution_path_candidate_by_id(
            "execution_path::failure_normalization_path",
            plan,
        )

        self.assertIsNotNone(optimized)
        self.assertIsNotNone(single_retry)
        self.assertIsNotNone(failure)
        assert optimized is not None
        assert single_retry is not None
        assert failure is not None
        self.assertEqual(optimized.status, "optimization_candidate")
        self.assertEqual(optimized.node_ids, graph.critical_path_node_ids)
        self.assertEqual(optimized.optimization_score, plan.highest_advisory_score)
        self.assertEqual(single_retry.status, "review")
        self.assertIn("review->refinement", single_retry.edge_ids)
        self.assertIn("refinement->generation", single_retry.edge_ids)
        self.assertEqual(failure.status, "retain")
        self.assertEqual(failure.node_ids, ("failure",))
        self.assertIn("failure->__end__", failure.edge_ids)

        for candidate in plan.candidates:
            self.assertEqual(
                set(candidate.model_dump(mode="json")),
                REQUIRED_PATH_CANDIDATE_FIELDS,
            )
            self.assertEqual(
                candidate.serialization_version,
                "execution_path_candidate.v1",
            )
            self.assertIn(
                "execution_path_selection",
                candidate.blocked_runtime_behaviors,
            )
            self.assertTrue(candidate.execution_path_optimization_implemented)
            self.assertFalse(candidate.execution_path_selection_implemented)
            self.assertFalse(candidate.workflow_graph_mutation_implemented)
            self.assertFalse(candidate.workflow_order_mutation_implemented)
            self.assertFalse(candidate.graph_compilation_implemented)
            self.assertFalse(candidate.workflow_execution_implemented)
            self.assertFalse(candidate.node_handler_invocation_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.workflow_control_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.prompt_mutation_implemented)
            self.assertFalse(candidate.persistent_storage_write_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.planning_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = plan_execution_path_optimization()
        optimization_candidates = execution_path_candidates_for_status(
            "optimization_candidate",
            plan,
        )
        retained = execution_path_candidates_for_status("retain", plan)
        missing = execution_path_candidate_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in optimization_candidates),
            plan.optimization_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in retained),
            plan.retained_candidate_ids,
        )
        self.assertIs(
            optimization_candidates[0],
            execution_path_candidate_by_id(
                optimization_candidates[0].candidate_id,
                plan,
            ),
        )

    def test_plan_rejects_mismatched_candidates_or_scores(self) -> None:
        plan = plan_execution_path_optimization()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            ExecutionPathOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["highest_advisory_score"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "highest_advisory_score must match",
        ):
            ExecutionPathOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["optimization_pressure"] = "low"

        with self.assertRaisesRegex(ValueError, "optimization_pressure must match"):
            ExecutionPathOptimizationPlan(**payload)

    def test_plan_does_not_declare_runtime_path_control_terms(self) -> None:
        plan = plan_execution_path_optimization()
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
                        *candidate.node_ids,
                        *candidate.edge_ids,
                        *candidate.evidence,
                        *candidate.advisory_actions,
                        *candidate.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "select_execution_path(",
            "mutate_graph(",
            "compile_graph(",
            "execute_workflow(",
            "invoke_node_handler(",
            "route_provider(",
            "trigger_retry(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
