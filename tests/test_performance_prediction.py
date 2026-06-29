import unittest

from creative_coding_assistant.orchestration import (
    PerformancePredictionPlan,
    detect_bottlenecks,
    optimize_latency,
    optimize_throughput,
    performance_prediction_by_id,
    performance_predictions_for_band,
    plan_execution_profiling,
    predict_performance,
)

REQUIRED_PERFORMANCE_PREDICTION_FIELDS = {
    "prediction_id",
    "prediction_focus",
    "status",
    "source_id",
    "source_serialization_version",
    "source_candidate_ids",
    "source_signal_score",
    "source_guardrail_count",
    "predicted_performance_band",
    "predicted_performance_range",
    "predicted_performance_midpoint",
    "prediction_confidence",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "performance_prediction_implemented",
    "advisory_performance_prediction_implemented",
    "relative_performance_units_only",
    "runtime_performance_measurement_implemented",
    "benchmark_execution_implemented",
    "profiler_hook_installation_implemented",
    "runtime_trace_collection_implemented",
    "regression_detection_implemented",
    "throughput_runtime_optimization_implemented",
    "concurrency_limit_change_implemented",
    "queue_management_runtime_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
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


class PerformancePredictionTests(unittest.TestCase):
    def test_default_plan_derives_advisory_performance_predictions(self) -> None:
        latency = optimize_latency()
        profiling = plan_execution_profiling(latency_optimization=latency)
        bottlenecks = detect_bottlenecks(
            latency_optimization=latency,
            execution_profiling=profiling,
        )
        throughput = optimize_throughput(bottleneck_detection=bottlenecks)
        plan = predict_performance(
            throughput_optimization=throughput,
            latency_optimization=latency,
            bottleneck_detection=bottlenecks,
            execution_profiling=profiling,
        )

        self.assertEqual(plan.role, "performance_predictor")
        self.assertEqual(
            plan.serialization_version,
            "performance_prediction_plan.v1",
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
            plan.source_bottleneck_detection_serialization_version,
            bottlenecks.serialization_version,
        )
        self.assertEqual(
            plan.source_execution_profiling_serialization_version,
            profiling.serialization_version,
        )
        self.assertEqual(plan.prediction_count, 4)
        self.assertEqual(
            plan.prediction_ids,
            (
                "performance_prediction::overall_throughput_posture",
                "performance_prediction::latency_pressure_posture",
                "performance_prediction::profile_readiness_posture",
                "performance_prediction::bottleneck_risk_posture",
            ),
        )
        self.assertEqual(
            plan.recommended_prediction_id,
            "performance_prediction::overall_throughput_posture",
        )
        self.assertEqual(plan.recommended_performance_band, "guarded")
        self.assertEqual(plan.recommended_performance_midpoint, 17)
        self.assertEqual(len(plan.fallback_prediction_ids), 2)
        self.assertEqual(len(plan.guardrail_prediction_ids), 1)
        self.assertGreaterEqual(plan.guarded_prediction_count, 1)
        self.assertGreaterEqual(plan.highest_predicted_performance_midpoint, 17)
        self.assertIn("does not measure performance", plan.authority_boundary)
        self.assertTrue(plan.performance_prediction_implemented)
        self.assertTrue(plan.advisory_performance_prediction_implemented)
        self.assertTrue(plan.relative_performance_units_only)
        self.assertFalse(plan.runtime_performance_measurement_implemented)
        self.assertFalse(plan.benchmark_execution_implemented)
        self.assertFalse(plan.profiler_hook_installation_implemented)
        self.assertFalse(plan.runtime_trace_collection_implemented)
        self.assertFalse(plan.regression_detection_implemented)
        self.assertFalse(plan.throughput_runtime_optimization_implemented)
        self.assertFalse(plan.concurrency_limit_change_implemented)
        self.assertFalse(plan.queue_management_runtime_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.workflow_control_implemented)
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

    def test_predictions_are_relative_and_advisory_only(self) -> None:
        plan = predict_performance()

        for prediction in plan.predictions:
            self.assertEqual(
                set(prediction.model_dump(mode="json")),
                REQUIRED_PERFORMANCE_PREDICTION_FIELDS,
            )
            self.assertEqual(
                prediction.serialization_version,
                "performance_prediction.v1",
            )
            self.assertEqual(
                prediction.predicted_performance_midpoint,
                (
                    prediction.predicted_performance_range[0]
                    + prediction.predicted_performance_range[1]
                )
                // 2,
            )
            self.assertIn(
                "runtime_performance_measurement",
                prediction.blocked_runtime_behaviors,
            )
            self.assertTrue(prediction.performance_prediction_implemented)
            self.assertTrue(prediction.advisory_performance_prediction_implemented)
            self.assertTrue(prediction.relative_performance_units_only)
            self.assertFalse(prediction.runtime_performance_measurement_implemented)
            self.assertFalse(prediction.benchmark_execution_implemented)
            self.assertFalse(prediction.profiler_hook_installation_implemented)
            self.assertFalse(prediction.runtime_trace_collection_implemented)
            self.assertFalse(prediction.regression_detection_implemented)
            self.assertFalse(prediction.throughput_runtime_optimization_implemented)
            self.assertFalse(prediction.concurrency_limit_change_implemented)
            self.assertFalse(prediction.queue_management_runtime_implemented)
            self.assertFalse(prediction.provider_model_routing_implemented)
            self.assertFalse(prediction.workflow_control_implemented)
            self.assertFalse(prediction.workflow_timing_change_implemented)
            self.assertFalse(prediction.workflow_graph_mutation_implemented)
            self.assertFalse(prediction.graph_compilation_implemented)
            self.assertFalse(prediction.workflow_execution_implemented)
            self.assertFalse(prediction.agent_invocation_implemented)
            self.assertFalse(prediction.node_handler_invocation_implemented)
            self.assertFalse(prediction.retry_triggering_implemented)
            self.assertFalse(prediction.prompt_mutation_implemented)
            self.assertFalse(prediction.persistent_storage_write_implemented)
            self.assertFalse(prediction.generated_output_mutation_implemented)
            self.assertTrue(prediction.advisory_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = predict_performance()
        recommended = performance_prediction_by_id(
            "performance_prediction::overall_throughput_posture",
            plan,
        )
        guarded = performance_predictions_for_band("guarded", plan)
        missing = performance_prediction_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(recommended.status, "recommended")
        self.assertEqual(recommended.predicted_performance_band, "guarded")
        self.assertIn(recommended, guarded)

    def test_plan_rejects_mismatched_predictions_or_recommendation(self) -> None:
        plan = predict_performance()
        payload = plan.model_dump(mode="json")
        payload["prediction_ids"] = ("missing",) + tuple(payload["prediction_ids"][1:])

        with self.assertRaisesRegex(ValueError, "prediction_ids must match"):
            PerformancePredictionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_prediction_id"] = "missing"

        with self.assertRaisesRegex(
            ValueError,
            "recommended_prediction_id must match",
        ):
            PerformancePredictionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_performance_midpoint"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "recommended_performance_midpoint must match",
        ):
            PerformancePredictionPlan(**payload)

    def test_plan_does_not_declare_runtime_prediction_terms(self) -> None:
        plan = predict_performance()
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for prediction in plan.predictions
                    for field in (
                        prediction.prediction_id,
                        prediction.source_id,
                        *prediction.evidence,
                        *prediction.advisory_actions,
                        *prediction.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "measure_performance(",
            "run_benchmark(",
            "install_profiler(",
            "collect_trace(",
            "detect_regression(",
            "optimize_runtime_throughput(",
            "set_concurrency_limit(",
            "resize_queue(",
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
