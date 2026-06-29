import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    CreativeDiversityPredictionPlan,
    creative_diversity_audit_registry,
    creative_diversity_prediction_by_id,
    creative_diversity_predictions_for_band,
    predict_creative_diversity,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_DIVERSITY_PREDICTION_FIELDS = {
    "prediction_id",
    "source_budget_profile_id",
    "source_topic_id",
    "source_audit_status",
    "predicted_diversity_band",
    "max_advisory_variants",
    "max_advisory_refinement_passes",
    "predicted_variant_range",
    "diversity_readiness_score",
    "source_trace_profile_id",
    "source_provenance_profile_id",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "status",
    "creative_diversity_predictor_implemented",
    "advisory_diversity_prediction_implemented",
    "active_diversity_generation_implemented",
    "budget_enforcement_implemented",
    "variant_generation_implemented",
    "refinement_triggering_implemented",
    "cost_routing_implemented",
    "agent_invocation_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class CreativeDiversityPredictorTests(unittest.TestCase):
    def test_predictor_uses_creative_diversity_audit_metadata(self) -> None:
        audit_registry = creative_diversity_audit_registry()
        plan = predict_creative_diversity(diversity_audit=audit_registry)

        self.assertEqual(plan.role, "creative_diversity_predictor")
        self.assertEqual(
            plan.serialization_version,
            "creative_diversity_prediction_plan.v1",
        )
        self.assertEqual(
            plan.source_creative_diversity_audit_serialization_version,
            audit_registry.serialization_version,
        )
        self.assertEqual(plan.prediction_count, audit_registry.audit_count)
        self.assertEqual(
            plan.recommended_prediction_id,
            (
                "creative_diversity_prediction::"
                "creative_exploration_budget::style_aesthetic_alignment"
            ),
        )
        self.assertEqual(plan.recommended_diversity_band, "broad")
        self.assertGreaterEqual(plan.recommended_diversity_readiness_score, 80)
        self.assertEqual(plan.broad_prediction_count, 1)
        self.assertEqual(plan.guarded_prediction_count, 1)
        self.assertIn("does not enforce budgets", plan.authority_boundary)
        self.assertTrue(plan.creative_diversity_predictor_implemented)
        self.assertTrue(plan.advisory_diversity_prediction_implemented)
        self.assertFalse(plan.active_diversity_generation_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.variant_generation_implemented)
        self.assertFalse(plan.refinement_triggering_implemented)
        self.assertFalse(plan.cost_routing_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_diversity_predictions_are_advisory_only(self) -> None:
        plan = predict_creative_diversity()

        for prediction in plan.predictions:
            dumped = prediction.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_DIVERSITY_PREDICTION_FIELDS)
            self.assertEqual(
                prediction.serialization_version,
                "creative_diversity_prediction.v1",
            )
            self.assertEqual(prediction.source_audit_status, "pass")
            self.assertEqual(
                prediction.predicted_variant_range,
                (0, prediction.max_advisory_variants),
            )
            self.assertIn("variant_generation", prediction.blocked_runtime_behaviors)
            self.assertTrue(prediction.creative_diversity_predictor_implemented)
            self.assertTrue(prediction.advisory_diversity_prediction_implemented)
            self.assertFalse(prediction.active_diversity_generation_implemented)
            self.assertFalse(prediction.budget_enforcement_implemented)
            self.assertFalse(prediction.variant_generation_implemented)
            self.assertFalse(prediction.refinement_triggering_implemented)
            self.assertFalse(prediction.cost_routing_implemented)
            self.assertFalse(prediction.agent_invocation_implemented)
            self.assertFalse(prediction.provider_model_routing_implemented)
            self.assertFalse(prediction.workflow_control_implemented)
            self.assertFalse(prediction.retry_triggering_implemented)
            self.assertFalse(prediction.prompt_mutation_implemented)
            self.assertFalse(prediction.generated_output_mutation_implemented)
            self.assertTrue(prediction.advisory_only)

    def test_lookup_helpers_return_predictions_without_generation(self) -> None:
        plan = predict_creative_diversity()
        recommended = creative_diversity_prediction_by_id(
            (
                "creative_diversity_prediction::"
                "creative_exploration_budget::style_aesthetic_alignment"
            ),
            plan,
        )
        broad_predictions = creative_diversity_predictions_for_band("broad", plan)
        missing = creative_diversity_prediction_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(recommended.status, "recommended")
        self.assertEqual(recommended.predicted_diversity_band, "broad")
        self.assertIn(recommended, broad_predictions)

    def test_plan_rejects_mismatched_predictions_or_recommendation(self) -> None:
        plan = predict_creative_diversity()
        payload = plan.model_dump(mode="json")
        payload["prediction_ids"] = ("missing",) + tuple(payload["prediction_ids"][1:])

        with self.assertRaisesRegex(ValueError, "prediction_ids must match"):
            CreativeDiversityPredictionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_prediction_id"] = "missing"

        with self.assertRaisesRegex(
            ValueError,
            "recommended_prediction_id must match",
        ):
            CreativeDiversityPredictionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_diversity_readiness_score"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "recommended_diversity_readiness_score must match",
        ):
            CreativeDiversityPredictionPlan(**payload)

    def test_diversity_prediction_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate a diverse family of visual studies.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        predict_creative_diversity()
        creative_diversity_predictions_for_band("broad")
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)

    def test_diversity_prediction_does_not_declare_generation_terms(self) -> None:
        plan = predict_creative_diversity()
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
                        prediction.source_budget_profile_id,
                        prediction.source_topic_id,
                        prediction.predicted_diversity_band,
                        *prediction.evidence,
                        *prediction.advisory_actions,
                        *prediction.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "enforce_runtime_budget(",
            "generate_variant(",
            "trigger_runtime_refinement(",
            "route_by_cost(",
            "execute_agent(",
            "route_provider(",
            "control_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
