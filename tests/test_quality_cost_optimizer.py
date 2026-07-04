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
    QualityCostOptimizationPlan,
    cost_profile_registry,
    optimize_quality_cost,
    quality_cost_candidate_by_id,
    quality_cost_candidates_for_posture,
    quality_profile_registry,
    route_hybrid_model_request,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_QUALITY_COST_CANDIDATE_FIELDS = {
    "candidate_id",
    "source_hybrid_decision_id",
    "source_model_profile_id",
    "route_name",
    "hybrid_mode",
    "source_quality_profile_ids",
    "source_cost_profile_ids",
    "quality_level",
    "cost_band",
    "advisory_cost_range",
    "quality_score",
    "cost_score",
    "hybrid_bonus",
    "optimization_score",
    "tradeoff_posture",
    "status",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "quality_cost_optimization_implemented",
    "quality_cost_scoring_implemented",
    "quality_prediction_implemented",
    "cost_prediction_implemented",
    "cost_estimation_implemented",
    "pricing_lookup_implemented",
    "budget_enforcement_implemented",
    "provider_model_routing_implemented",
    "model_selection_implemented",
    "provider_execution_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class QualityCostOptimizerTests(unittest.TestCase):
    def test_default_optimizer_uses_hybrid_quality_and_cost_metadata(self) -> None:
        hybrid_plan = route_hybrid_model_request(route=RouteName.GENERATE)
        plan = optimize_quality_cost(hybrid_routing=hybrid_plan)

        self.assertEqual(plan.role, "quality_cost_optimizer")
        self.assertEqual(
            plan.serialization_version,
            "quality_cost_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_hybrid_routing_serialization_version,
            hybrid_plan.serialization_version,
        )
        self.assertEqual(
            plan.source_quality_profile_serialization_version,
            quality_profile_registry().serialization_version,
        )
        self.assertEqual(
            plan.source_cost_profile_serialization_version,
            cost_profile_registry().serialization_version,
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(
            plan.recommended_candidate_id,
            "quality_cost::creative_reasoning_model_profile",
        )
        self.assertEqual(plan.recommended_tradeoff_posture, "quality_favored")
        self.assertEqual(plan.candidate_count, 3)
        self.assertEqual(plan.recommended_optimization_score, 120)
        self.assertIn("does not estimate live provider cost", plan.authority_boundary)
        self.assertTrue(plan.quality_cost_optimization_implemented)
        self.assertTrue(plan.quality_cost_scoring_implemented)
        self.assertFalse(plan.quality_prediction_implemented)
        self.assertFalse(plan.cost_prediction_implemented)
        self.assertFalse(plan.cost_estimation_implemented)
        self.assertFalse(plan.pricing_lookup_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.model_selection_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_quality_cost_candidates_remain_advisory_and_source_aligned(self) -> None:
        plan = optimize_quality_cost(route=RouteName.REVIEW)
        known_quality_ids = set(quality_profile_registry().quality_profile_ids)
        known_cost_ids = set(cost_profile_registry().cost_profile_ids)

        for candidate in plan.candidates:
            dumped = candidate.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_QUALITY_COST_CANDIDATE_FIELDS)
            self.assertEqual(
                candidate.serialization_version,
                "quality_cost_optimization_candidate.v1",
            )
            self.assertEqual(candidate.route_name, RouteName.REVIEW)
            self.assertTrue(
                set(candidate.source_quality_profile_ids).issubset(known_quality_ids)
            )
            self.assertTrue(
                set(candidate.source_cost_profile_ids).issubset(known_cost_ids)
            )
            self.assertEqual(
                candidate.optimization_score,
                candidate.quality_score + candidate.cost_score + candidate.hybrid_bonus,
            )
            self.assertIn(
                "provider_or_model_routing",
                candidate.blocked_runtime_behaviors,
            )
            self.assertTrue(candidate.quality_cost_optimization_implemented)
            self.assertTrue(candidate.quality_cost_scoring_implemented)
            self.assertFalse(candidate.quality_prediction_implemented)
            self.assertFalse(candidate.cost_prediction_implemented)
            self.assertFalse(candidate.cost_estimation_implemented)
            self.assertFalse(candidate.pricing_lookup_implemented)
            self.assertFalse(candidate.budget_enforcement_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.model_selection_implemented)
            self.assertFalse(candidate.provider_execution_implemented)
            self.assertFalse(candidate.workflow_control_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.prompt_mutation_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.advisory_only)

    def test_review_route_prioritizes_balanced_tradeoff_lookup(self) -> None:
        plan = optimize_quality_cost(route=RouteName.REVIEW)
        recommended = quality_cost_candidate_by_id(
            "quality_cost::evaluation_review_model_profile",
            plan,
        )
        balanced = quality_cost_candidates_for_posture("balanced_tradeoff", plan)
        missing = quality_cost_candidate_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(plan.recommended_tradeoff_posture, "balanced_tradeoff")
        self.assertEqual(recommended.status, "recommended")
        self.assertIn(recommended, balanced)

    def test_plan_rejects_mismatched_candidates_or_recommendation(self) -> None:
        plan = optimize_quality_cost(route=RouteName.GENERATE)
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            QualityCostOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_candidate_id"] = "missing"

        with self.assertRaisesRegex(ValueError, "recommended_candidate_id must match"):
            QualityCostOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_optimization_score"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "recommended_optimization_score must match",
        ):
            QualityCostOptimizationPlan(**payload)

    def test_optimizer_does_not_change_request_routing_or_provider_factory(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Design a refined p5.js study.",
            mode=AssistantMode.DESIGN,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = optimize_quality_cost(route_decision=baseline_decision)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, baseline_decision.route)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_optimizer_does_not_declare_runtime_application_terms(self) -> None:
        plan = optimize_quality_cost(route=RouteName.REVIEW)
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
                        candidate.source_hybrid_decision_id,
                        candidate.source_model_profile_id,
                        *candidate.source_quality_profile_ids,
                        *candidate.source_cost_profile_ids,
                        *candidate.evidence,
                        *candidate.advisory_actions,
                        *candidate.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "estimate_cost(",
            "lookup_pricing(",
            "predict_quality(",
            "predict_cost(",
            "enforce_budget(",
            "select_model(",
            "switch_model(",
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
