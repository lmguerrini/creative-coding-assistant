import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    AdaptiveLatencyPlan,
    adaptive_latency_candidate_by_id,
    adaptive_latency_candidates_for_posture,
    optimize_adaptive_latency,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_CANDIDATE_FIELDS = {
    "candidate_id",
    "source_latency_candidate_id",
    "source_hybrid_workflow_candidate_id",
    "stage_id",
    "route_name",
    "task_type",
    "execution_mode_id",
    "source_latency_band",
    "hybrid_estimated_latency",
    "resource_utilization_pressure",
    "predicted_performance_midpoint",
    "agent_activation_candidate_count",
    "latency_savings_score",
    "latency_pressure_score",
    "performance_weight",
    "resource_weight",
    "hybrid_latency_weight",
    "adaptive_latency_score",
    "adaptive_latency_posture",
    "status",
    "hitl_required",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "adaptive_latency_optimizer_implemented",
    "adaptive_latency_scoring_implemented",
    "latency_measurement_implemented",
    "latency_threshold_evaluation_implemented",
    "latency_based_routing_implemented",
    "runtime_selection_implemented",
    "parallel_execution_implemented",
    "async_execution_implemented",
    "workflow_timing_change_implemented",
    "workflow_graph_mutation_implemented",
    "graph_compilation_implemented",
    "workflow_execution_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class AdaptiveLatencyOptimizerTests(unittest.TestCase):
    def test_plan_combines_latency_performance_resource_and_v5_context(self) -> None:
        plan = optimize_adaptive_latency(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "adaptive_latency_optimizer")
        self.assertEqual(plan.serialization_version, "adaptive_latency_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_latency_optimization_serialization_version,
            "latency_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_performance_prediction_serialization_version,
            "performance_prediction_plan.v1",
        )
        self.assertEqual(
            plan.source_resource_utilization_serialization_version,
            "resource_utilization_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_hybrid_workflow_serialization_version,
            "adaptive_hybrid_workflow_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_agent_activation_serialization_version,
            "agent_activation_optimization_plan.v1",
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.candidate_count, 6)
        self.assertIn("does not measure latency", plan.authority_boundary)
        self.assertTrue(plan.adaptive_latency_optimizer_implemented)
        self.assertTrue(plan.adaptive_latency_scoring_implemented)
        self.assertFalse(plan.latency_measurement_implemented)
        self.assertFalse(plan.latency_threshold_evaluation_implemented)
        self.assertFalse(plan.latency_based_routing_implemented)
        self.assertFalse(plan.runtime_selection_implemented)
        self.assertFalse(plan.parallel_execution_implemented)
        self.assertFalse(plan.async_execution_implemented)
        self.assertFalse(plan.workflow_timing_change_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_candidates_are_relative_latency_recommendations(self) -> None:
        plan = optimize_adaptive_latency(route="generate")

        for candidate in plan.candidates:
            dumped = candidate.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CANDIDATE_FIELDS)
            self.assertEqual(
                candidate.serialization_version,
                "adaptive_latency_candidate.v1",
            )
            self.assertEqual(candidate.route_name, RouteName.GENERATE)
            self.assertGreaterEqual(candidate.predicted_performance_midpoint, 0)
            self.assertLessEqual(candidate.predicted_performance_midpoint, 100)
            self.assertEqual(
                candidate.adaptive_latency_score,
                min(
                    240,
                    candidate.performance_weight
                    + candidate.resource_weight
                    + candidate.hybrid_latency_weight
                    + candidate.latency_savings_score // 10
                    - candidate.latency_pressure_score // 100,
                ),
            )
            self.assertIn("latency_measurement", candidate.blocked_runtime_behaviors)
            self.assertFalse(candidate.latency_measurement_implemented)
            self.assertFalse(candidate.latency_threshold_evaluation_implemented)
            self.assertFalse(candidate.latency_based_routing_implemented)
            self.assertFalse(candidate.parallel_execution_implemented)
            self.assertFalse(candidate.async_execution_implemented)
            self.assertFalse(candidate.workflow_timing_change_implemented)
            self.assertFalse(candidate.workflow_execution_implemented)
            self.assertFalse(candidate.agent_invocation_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.provider_execution_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.advisory_only)

        recommended = adaptive_latency_candidate_by_id(plan.recommended_candidate_id, plan)
        balanced = adaptive_latency_candidates_for_posture("balanced_latency", plan)
        guarded = adaptive_latency_candidates_for_posture("guarded_latency", plan)
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(recommended.status, "recommended")
        self.assertIn(recommended, balanced)
        self.assertTrue(guarded)
        self.assertTrue(recommended.hitl_required)

    def test_plan_rejects_mismatched_candidate_metadata(self) -> None:
        plan = optimize_adaptive_latency()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            AdaptiveLatencyPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_candidate_id"] = "missing"

        with self.assertRaisesRegex(ValueError, "recommended_candidate_id must match"):
            AdaptiveLatencyPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_adaptive_latency_score"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "recommended_adaptive_latency_score must match",
        ):
            AdaptiveLatencyPlan(**payload)

    def test_optimizer_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Reduce perceived latency for a Three.js scene.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = optimize_adaptive_latency(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_optimizer_does_not_declare_runtime_application_terms(self) -> None:
        plan = optimize_adaptive_latency(route=RouteName.GENERATE)
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
                        candidate.source_latency_candidate_id,
                        candidate.source_hybrid_workflow_candidate_id,
                        candidate.stage_id,
                        *candidate.evidence,
                        *candidate.advisory_actions,
                        *candidate.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "measure_latency(",
            "evaluate_latency_threshold(",
            "route_by_latency(",
            "select_runtime(",
            "execute_parallel_tasks(",
            "create_async_task(",
            "change_workflow_timing(",
            "mutate_workflow_graph(",
            "compile_graph(",
            "execute_workflow(",
            "invoke_agent(",
            "route_provider(",
            "execute_provider(",
            "trigger_retry(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
