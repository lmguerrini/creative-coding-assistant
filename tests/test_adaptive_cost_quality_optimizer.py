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
    AdaptiveCostQualityPlan,
    adaptive_cost_quality_candidate_by_id,
    adaptive_cost_quality_candidates_for_posture,
    optimize_adaptive_cost_quality,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_CANDIDATE_FIELDS = {
    "candidate_id",
    "source_quality_cost_candidate_id",
    "source_model_profile_id",
    "source_hybrid_workflow_candidate_id",
    "route_name",
    "task_type",
    "execution_mode_id",
    "quality_level",
    "cost_band",
    "quality_cost_tradeoff_posture",
    "predicted_quality_midpoint",
    "predicted_cost_midpoint",
    "quality_weight",
    "cost_weight",
    "hybrid_context_bonus",
    "agent_context_bonus",
    "adaptive_score",
    "adaptive_posture",
    "status",
    "hitl_required",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "adaptive_cost_quality_optimizer_implemented",
    "adaptive_cost_quality_scoring_implemented",
    "quality_prediction_metadata_used",
    "cost_prediction_metadata_used",
    "generated_output_quality_evaluation_implemented",
    "provider_pricing_lookup_implemented",
    "live_usage_metering_implemented",
    "budget_enforcement_implemented",
    "cost_based_routing_implemented",
    "quality_based_routing_implemented",
    "provider_model_routing_implemented",
    "model_selection_implemented",
    "provider_execution_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class AdaptiveCostQualityOptimizerTests(unittest.TestCase):
    def test_plan_combines_quality_cost_prediction_and_v5_context(self) -> None:
        plan = optimize_adaptive_cost_quality(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "adaptive_cost_quality_optimizer")
        self.assertEqual(plan.serialization_version, "adaptive_cost_quality_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_quality_cost_serialization_version,
            "quality_cost_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_quality_prediction_serialization_version,
            "quality_prediction_plan.v1",
        )
        self.assertEqual(
            plan.source_cost_prediction_serialization_version,
            "cost_prediction_plan.v1",
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
        self.assertEqual(plan.candidate_count, 3)
        self.assertIn("does not evaluate generated output", plan.authority_boundary)
        self.assertTrue(plan.adaptive_cost_quality_optimizer_implemented)
        self.assertTrue(plan.adaptive_cost_quality_scoring_implemented)
        self.assertFalse(plan.generated_output_quality_evaluation_implemented)
        self.assertFalse(plan.provider_pricing_lookup_implemented)
        self.assertFalse(plan.live_usage_metering_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.cost_based_routing_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_candidates_are_relative_advisory_tradeoffs(self) -> None:
        plan = optimize_adaptive_cost_quality(route="generate")

        for candidate in plan.candidates:
            dumped = candidate.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CANDIDATE_FIELDS)
            self.assertEqual(
                candidate.serialization_version,
                "adaptive_cost_quality_candidate.v1",
            )
            self.assertEqual(candidate.route_name, RouteName.GENERATE)
            self.assertGreaterEqual(candidate.predicted_quality_midpoint, 0)
            self.assertLessEqual(candidate.predicted_quality_midpoint, 100)
            self.assertGreaterEqual(candidate.predicted_cost_midpoint, 0)
            self.assertLessEqual(candidate.predicted_cost_midpoint, 100)
            self.assertEqual(
                candidate.adaptive_score,
                min(
                    240,
                    candidate.quality_weight
                    + candidate.cost_weight
                    + candidate.hybrid_context_bonus
                    + candidate.agent_context_bonus,
                ),
            )
            self.assertTrue(candidate.quality_prediction_metadata_used)
            self.assertTrue(candidate.cost_prediction_metadata_used)
            self.assertFalse(candidate.generated_output_quality_evaluation_implemented)
            self.assertFalse(candidate.provider_pricing_lookup_implemented)
            self.assertFalse(candidate.live_usage_metering_implemented)
            self.assertFalse(candidate.budget_enforcement_implemented)
            self.assertFalse(candidate.quality_based_routing_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.provider_execution_implemented)
            self.assertFalse(candidate.workflow_control_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.prompt_mutation_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.advisory_only)

        recommended = adaptive_cost_quality_candidate_by_id(
            plan.recommended_candidate_id,
            plan,
        )
        quality_priority = adaptive_cost_quality_candidates_for_posture(
            "quality_priority",
            plan,
        )
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(recommended.status, "recommended")
        self.assertIn(recommended, quality_priority)
        self.assertTrue(recommended.hitl_required)

    def test_plan_rejects_mismatched_candidate_metadata(self) -> None:
        plan = optimize_adaptive_cost_quality()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            AdaptiveCostQualityPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_candidate_id"] = "missing"

        with self.assertRaisesRegex(ValueError, "recommended_candidate_id must match"):
            AdaptiveCostQualityPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_adaptive_score"] += 1

        with self.assertRaisesRegex(
            ValueError, "recommended_adaptive_score must match"
        ):
            AdaptiveCostQualityPlan(**payload)

    def test_optimizer_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Optimize cost and quality for a WebGL sketch.",
            mode=AssistantMode.DESIGN,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = optimize_adaptive_cost_quality(route=RouteName.DESIGN)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.DESIGN)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_optimizer_does_not_declare_runtime_application_terms(self) -> None:
        plan = optimize_adaptive_cost_quality(route=RouteName.GENERATE)
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
                        candidate.source_quality_cost_candidate_id,
                        candidate.source_model_profile_id,
                        candidate.source_hybrid_workflow_candidate_id,
                        *candidate.evidence,
                        *candidate.advisory_actions,
                        *candidate.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "evaluate_generated_output(",
            "lookup_provider_pricing(",
            "meter_live_usage(",
            "enforce_budget(",
            "route_by_cost(",
            "route_by_quality(",
            "select_model(",
            "route_provider(",
            "execute_provider(",
            "control_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
