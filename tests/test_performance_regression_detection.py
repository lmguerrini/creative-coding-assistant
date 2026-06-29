import unittest

from creative_coding_assistant.orchestration import (
    PerformanceRegressionDetectionPlan,
    detect_performance_regressions,
    optimize_reasoning_budget,
    performance_regression_signal_by_id,
    performance_regression_signals_for_status,
    plan_performance_benchmarking,
    predict_performance,
)

REQUIRED_PERFORMANCE_REGRESSION_FIELDS = {
    "signal_id",
    "regression_id",
    "regression_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "baseline_reference_count",
    "source_pressure_score",
    "advisory_regression_score",
    "regression_severity",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "performance_regression_detection_planning_implemented",
    "runtime_regression_detection_implemented",
    "benchmark_execution_implemented",
    "runtime_performance_measurement_implemented",
    "timer_collection_implemented",
    "threshold_enforcement_implemented",
    "alert_emission_implemented",
    "workflow_blocking_implemented",
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


class PerformanceRegressionDetectionTests(unittest.TestCase):
    def test_default_plan_derives_advisory_regression_signals(self) -> None:
        prediction = predict_performance()
        benchmarking = plan_performance_benchmarking(
            performance_prediction=prediction,
        )
        reasoning = optimize_reasoning_budget(
            performance_prediction=prediction,
            performance_benchmarking=benchmarking,
        )
        plan = detect_performance_regressions(
            performance_prediction=prediction,
            performance_benchmarking=benchmarking,
            reasoning_budget=reasoning,
        )

        self.assertEqual(plan.role, "performance_regression_detector")
        self.assertEqual(
            plan.serialization_version,
            "performance_regression_detection_plan.v1",
        )
        self.assertEqual(
            plan.source_performance_prediction_serialization_version,
            prediction.serialization_version,
        )
        self.assertEqual(
            plan.source_performance_benchmarking_serialization_version,
            benchmarking.serialization_version,
        )
        self.assertEqual(
            plan.source_reasoning_budget_serialization_version,
            reasoning.serialization_version,
        )
        self.assertEqual(plan.signal_count, 4)
        self.assertEqual(
            plan.signal_ids,
            (
                "performance_regression::prediction_regression_risk",
                "performance_regression::benchmark_regression_risk",
                "performance_regression::reasoning_budget_pressure",
                "performance_regression::measurement_boundary",
            ),
        )
        self.assertEqual(plan.regression_candidate_count, 2)
        self.assertEqual(plan.review_only_count, 1)
        self.assertEqual(plan.baseline_guardrail_count, 1)
        self.assertGreater(plan.total_baseline_reference_count, 0)
        self.assertGreater(plan.highest_advisory_regression_score, 0)
        self.assertGreater(plan.total_advisory_regression_score, 0)
        self.assertEqual(plan.regression_detection_severity, "guarded")
        self.assertIn("does not detect live regressions", plan.authority_boundary)
        self.assertTrue(plan.performance_regression_detection_planning_implemented)
        self.assertFalse(plan.runtime_regression_detection_implemented)
        self.assertFalse(plan.benchmark_execution_implemented)
        self.assertFalse(plan.runtime_performance_measurement_implemented)
        self.assertFalse(plan.timer_collection_implemented)
        self.assertFalse(plan.threshold_enforcement_implemented)
        self.assertFalse(plan.alert_emission_implemented)
        self.assertFalse(plan.workflow_blocking_implemented)
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

    def test_signals_preserve_regression_boundaries(self) -> None:
        plan = detect_performance_regressions()
        benchmark = performance_regression_signal_by_id(
            "performance_regression::benchmark_regression_risk",
            plan,
        )
        boundary = performance_regression_signal_by_id(
            "performance_regression::measurement_boundary",
            plan,
        )

        self.assertIsNotNone(benchmark)
        self.assertIsNotNone(boundary)
        assert benchmark is not None
        assert boundary is not None
        self.assertEqual(benchmark.status, "regression_candidate")
        self.assertGreater(benchmark.advisory_regression_score, 0)
        self.assertEqual(boundary.status, "baseline_guardrail")
        self.assertEqual(boundary.regression_severity, "guarded")

        for signal in plan.signals:
            self.assertEqual(
                set(signal.model_dump(mode="json")),
                REQUIRED_PERFORMANCE_REGRESSION_FIELDS,
            )
            self.assertEqual(
                signal.serialization_version,
                "performance_regression_signal.v1",
            )
            self.assertIn(
                "runtime_regression_detection",
                signal.blocked_runtime_behaviors,
            )
            self.assertTrue(
                signal.performance_regression_detection_planning_implemented,
            )
            self.assertFalse(signal.runtime_regression_detection_implemented)
            self.assertFalse(signal.benchmark_execution_implemented)
            self.assertFalse(signal.runtime_performance_measurement_implemented)
            self.assertFalse(signal.timer_collection_implemented)
            self.assertFalse(signal.threshold_enforcement_implemented)
            self.assertFalse(signal.alert_emission_implemented)
            self.assertFalse(signal.workflow_blocking_implemented)
            self.assertFalse(signal.provider_model_routing_implemented)
            self.assertFalse(signal.workflow_control_implemented)
            self.assertFalse(signal.workflow_timing_change_implemented)
            self.assertFalse(signal.workflow_graph_mutation_implemented)
            self.assertFalse(signal.workflow_execution_implemented)
            self.assertFalse(signal.agent_invocation_implemented)
            self.assertFalse(signal.node_handler_invocation_implemented)
            self.assertFalse(signal.retry_triggering_implemented)
            self.assertFalse(signal.prompt_mutation_implemented)
            self.assertFalse(signal.persistent_storage_write_implemented)
            self.assertFalse(signal.generated_output_mutation_implemented)
            self.assertTrue(signal.advisory_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = detect_performance_regressions()
        candidates = performance_regression_signals_for_status(
            "regression_candidate",
            plan,
        )
        guardrails = performance_regression_signals_for_status(
            "baseline_guardrail",
            plan,
        )
        review = performance_regression_signals_for_status("review_only", plan)
        missing = performance_regression_signal_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(
            tuple(signal.signal_id for signal in candidates),
            plan.regression_candidate_ids,
        )
        self.assertEqual(
            tuple(signal.signal_id for signal in guardrails),
            plan.baseline_guardrail_ids,
        )
        self.assertEqual(
            tuple(signal.signal_id for signal in review),
            plan.review_only_ids,
        )

    def test_plan_rejects_mismatched_signal_totals(self) -> None:
        plan = detect_performance_regressions()
        payload = plan.model_dump(mode="json")
        payload["signal_ids"] = ("missing",) + tuple(payload["signal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "signal_ids must match"):
            PerformanceRegressionDetectionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["total_baseline_reference_count"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "total_baseline_reference_count must match",
        ):
            PerformanceRegressionDetectionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["regression_detection_severity"] = "low"

        with self.assertRaisesRegex(
            ValueError,
            "regression_detection_severity must match",
        ):
            PerformanceRegressionDetectionPlan(**payload)

    def test_plan_does_not_declare_runtime_regression_terms(self) -> None:
        plan = detect_performance_regressions()
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for signal in plan.signals
                    for field in (
                        signal.signal_id,
                        signal.source_id,
                        *signal.evidence,
                        *signal.advisory_actions,
                        *signal.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "detect_live_regression(",
            "execute_benchmark(",
            "measure_performance(",
            "collect_timer(",
            "enforce_threshold(",
            "emit_alert(",
            "block_workflow(",
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
