import unittest

from creative_coding_assistant.orchestration import (
    ExecutionStrategySelection,
    execution_strategies_for_status,
    execution_strategy_by_id,
    forecast_execution_cost,
    plan_execution_path_optimization,
    plan_workflow_pruning,
    select_execution_strategy,
)

REQUIRED_STRATEGY_FIELDS = {
    "strategy_id",
    "strategy_kind",
    "status",
    "source_path_candidate_id",
    "forecast_tokens",
    "estimated_token_savings",
    "selection_score",
    "confidence",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "execution_strategy_selection_implemented",
    "execution_strategy_application_implemented",
    "execution_path_selection_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_order_mutation_implemented",
    "graph_compilation_implemented",
    "workflow_execution_implemented",
    "node_handler_invocation_implemented",
    "provider_model_routing_implemented",
    "budget_enforcement_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "selection_only",
}


class ExecutionStrategySelectionTests(unittest.TestCase):
    def test_default_selection_chooses_advisory_cost_guarded_strategy(self) -> None:
        pruning = plan_workflow_pruning()
        forecast = forecast_execution_cost(pruning_plan=pruning)
        path_plan = plan_execution_path_optimization(
            cost_forecast=forecast,
            pruning_plan=pruning,
        )
        selection = select_execution_strategy(
            path_optimization=path_plan,
            cost_forecast=forecast,
            pruning_plan=pruning,
        )

        self.assertEqual(selection.role, "execution_strategy_selector")
        self.assertEqual(
            selection.serialization_version,
            "execution_strategy_selection.v1",
        )
        self.assertEqual(
            selection.source_path_optimization_serialization_version,
            path_plan.serialization_version,
        )
        self.assertEqual(
            selection.source_cost_forecast_serialization_version,
            forecast.serialization_version,
        )
        self.assertEqual(
            selection.source_pruning_serialization_version,
            pruning.serialization_version,
        )
        self.assertEqual(selection.strategy_count, 4)
        self.assertEqual(
            selection.strategy_ids,
            (
                "execution_strategy::cost_guarded_pruning",
                "execution_strategy::baseline_success",
                "execution_strategy::retry_guarded_quality",
                "execution_strategy::failure_safe",
            ),
        )
        self.assertEqual(
            selection.selected_strategy_id,
            "execution_strategy::cost_guarded_pruning",
        )
        self.assertEqual(
            selection.selected_path_candidate_id,
            "execution_path::pruning_adjusted_path",
        )
        self.assertEqual(selection.selected_strategy_count, 1)
        self.assertGreater(selection.selected_strategy_score, 0)
        self.assertGreater(selection.selected_estimated_token_savings, 0)
        self.assertEqual(selection.selection_confidence, "high")
        self.assertIn("does not apply the strategy", selection.authority_boundary)
        self.assertTrue(selection.execution_strategy_selection_implemented)
        self.assertFalse(selection.execution_strategy_application_implemented)
        self.assertFalse(selection.execution_path_selection_implemented)
        self.assertFalse(selection.workflow_graph_mutation_implemented)
        self.assertFalse(selection.workflow_order_mutation_implemented)
        self.assertFalse(selection.graph_compilation_implemented)
        self.assertFalse(selection.workflow_execution_implemented)
        self.assertFalse(selection.node_handler_invocation_implemented)
        self.assertFalse(selection.provider_model_routing_implemented)
        self.assertFalse(selection.budget_enforcement_implemented)
        self.assertFalse(selection.workflow_control_implemented)
        self.assertFalse(selection.retry_triggering_implemented)
        self.assertFalse(selection.prompt_mutation_implemented)
        self.assertFalse(selection.persistent_storage_write_implemented)
        self.assertFalse(selection.generated_output_mutation_implemented)
        self.assertTrue(selection.selection_only)

    def test_strategies_cover_selected_fallback_deferred_and_guardrail(self) -> None:
        selection = select_execution_strategy()
        selected = execution_strategy_by_id(
            "execution_strategy::cost_guarded_pruning",
            selection,
        )
        baseline = execution_strategy_by_id(
            "execution_strategy::baseline_success",
            selection,
        )
        retry = execution_strategy_by_id(
            "execution_strategy::retry_guarded_quality",
            selection,
        )
        failure = execution_strategy_by_id("execution_strategy::failure_safe", selection)

        self.assertIsNotNone(selected)
        self.assertIsNotNone(baseline)
        self.assertIsNotNone(retry)
        self.assertIsNotNone(failure)
        assert selected is not None
        assert baseline is not None
        assert retry is not None
        assert failure is not None
        self.assertEqual(selected.status, "selected")
        self.assertEqual(baseline.status, "fallback")
        self.assertEqual(retry.status, "deferred")
        self.assertEqual(failure.status, "guardrail")
        self.assertEqual(failure.strategy_kind, "failure_safe")

        for strategy in selection.strategies:
            self.assertEqual(
                set(strategy.model_dump(mode="json")),
                REQUIRED_STRATEGY_FIELDS,
            )
            self.assertEqual(
                strategy.serialization_version,
                "execution_strategy_candidate.v1",
            )
            self.assertIn("strategy_application", strategy.blocked_runtime_behaviors)
            self.assertTrue(strategy.execution_strategy_selection_implemented)
            self.assertFalse(strategy.execution_strategy_application_implemented)
            self.assertFalse(strategy.execution_path_selection_implemented)
            self.assertFalse(strategy.workflow_graph_mutation_implemented)
            self.assertFalse(strategy.workflow_order_mutation_implemented)
            self.assertFalse(strategy.graph_compilation_implemented)
            self.assertFalse(strategy.workflow_execution_implemented)
            self.assertFalse(strategy.node_handler_invocation_implemented)
            self.assertFalse(strategy.provider_model_routing_implemented)
            self.assertFalse(strategy.budget_enforcement_implemented)
            self.assertFalse(strategy.workflow_control_implemented)
            self.assertFalse(strategy.retry_triggering_implemented)
            self.assertFalse(strategy.prompt_mutation_implemented)
            self.assertFalse(strategy.persistent_storage_write_implemented)
            self.assertFalse(strategy.generated_output_mutation_implemented)
            self.assertTrue(strategy.selection_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        selection = select_execution_strategy()
        selected = execution_strategies_for_status("selected", selection)
        fallbacks = execution_strategies_for_status("fallback", selection)
        missing = execution_strategy_by_id("missing", selection)

        self.assertIsNone(missing)
        self.assertEqual(len(selected), 1)
        self.assertEqual(
            tuple(strategy.strategy_id for strategy in fallbacks),
            selection.fallback_strategy_ids,
        )
        self.assertIs(
            selected[0],
            execution_strategy_by_id(selected[0].strategy_id, selection),
        )

    def test_selection_rejects_mismatched_strategies_or_selected_fields(self) -> None:
        selection = select_execution_strategy()
        payload = selection.model_dump(mode="json")
        payload["strategy_ids"] = ("missing",) + tuple(payload["strategy_ids"][1:])

        with self.assertRaisesRegex(ValueError, "strategy_ids must match"):
            ExecutionStrategySelection(**payload)

        payload = selection.model_dump(mode="json")
        payload["selected_strategy_id"] = "missing"

        with self.assertRaisesRegex(ValueError, "selected_strategy_id must match"):
            ExecutionStrategySelection(**payload)

        payload = selection.model_dump(mode="json")
        payload["selected_estimated_token_savings"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "selected_estimated_token_savings must match",
        ):
            ExecutionStrategySelection(**payload)

    def test_selection_does_not_declare_runtime_application_terms(self) -> None:
        selection = select_execution_strategy()
        combined_text = " ".join(
            (
                selection.authority_boundary,
                *selection.blocked_runtime_behaviors,
                *selection.advisory_actions,
                *(
                    field
                    for strategy in selection.strategies
                    for field in (
                        strategy.strategy_id,
                        strategy.source_path_candidate_id,
                        *strategy.evidence,
                        *strategy.advisory_actions,
                        *strategy.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "apply_strategy(",
            "select_execution_path(",
            "mutate_graph(",
            "compile_graph(",
            "execute_workflow(",
            "invoke_node_handler(",
            "route_provider(",
            "enforce_budget(",
            "trigger_retry(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
