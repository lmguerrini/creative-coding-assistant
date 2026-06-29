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
    QualityPredictionPlan,
    predict_quality_for_route,
    quality_prediction_by_id,
    quality_predictions_for_level,
    route_request,
)
from creative_coding_assistant.orchestration.hybrid_studio import (
    quality_profile_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_QUALITY_PREDICTION_FIELDS = {
    "prediction_id",
    "source_quality_profile_id",
    "source_model_profile_ids",
    "source_provider_selection_profile_ids",
    "route_name",
    "quality_profile_kind",
    "predicted_quality_level",
    "predicted_quality_range",
    "predicted_quality_midpoint",
    "prediction_confidence",
    "status",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "quality_prediction_engine_implemented",
    "advisory_quality_prediction_implemented",
    "relative_quality_units_only",
    "generated_output_quality_evaluation_implemented",
    "quality_scoring_implemented",
    "quality_escalation_implemented",
    "refinement_triggering_implemented",
    "provider_model_routing_implemented",
    "model_selection_implemented",
    "provider_execution_implemented",
    "human_input_request_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class QualityPredictionEngineTests(unittest.TestCase):
    def test_default_quality_prediction_uses_quality_profiles(self) -> None:
        registry = quality_profile_registry()
        plan = predict_quality_for_route(
            route=RouteName.GENERATE,
            quality_profiles=registry,
        )

        self.assertEqual(plan.role, "quality_prediction_engine")
        self.assertEqual(plan.serialization_version, "quality_prediction_plan.v1")
        self.assertEqual(
            plan.source_quality_profile_serialization_version,
            registry.serialization_version,
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(
            plan.recommended_prediction_id,
            "quality_prediction::creative_quality_profile",
        )
        self.assertEqual(plan.recommended_quality_level, "high")
        self.assertEqual(plan.recommended_quality_midpoint, 75)
        self.assertEqual(plan.prediction_count, 2)
        self.assertEqual(plan.high_or_critical_prediction_count, 1)
        self.assertEqual(plan.critical_prediction_count, 0)
        self.assertIn("does not evaluate generated output", plan.authority_boundary)
        self.assertTrue(plan.quality_prediction_engine_implemented)
        self.assertTrue(plan.advisory_quality_prediction_implemented)
        self.assertTrue(plan.relative_quality_units_only)
        self.assertFalse(plan.generated_output_quality_evaluation_implemented)
        self.assertFalse(plan.quality_scoring_implemented)
        self.assertFalse(plan.quality_escalation_implemented)
        self.assertFalse(plan.refinement_triggering_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.model_selection_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.human_input_request_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_quality_prediction_decisions_are_advisory_only(self) -> None:
        plan = predict_quality_for_route(route=RouteName.REVIEW)

        for prediction in plan.predictions:
            dumped = prediction.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_QUALITY_PREDICTION_FIELDS)
            self.assertEqual(
                prediction.serialization_version,
                "quality_prediction_decision.v1",
            )
            self.assertEqual(prediction.route_name, RouteName.REVIEW)
            self.assertEqual(
                prediction.predicted_quality_midpoint,
                (
                    prediction.predicted_quality_range[0]
                    + prediction.predicted_quality_range[1]
                )
                // 2,
            )
            self.assertIn(
                "generated_output_quality_evaluation",
                prediction.blocked_runtime_behaviors,
            )
            self.assertTrue(prediction.quality_prediction_engine_implemented)
            self.assertTrue(prediction.advisory_quality_prediction_implemented)
            self.assertTrue(prediction.relative_quality_units_only)
            self.assertFalse(
                prediction.generated_output_quality_evaluation_implemented,
            )
            self.assertFalse(prediction.quality_scoring_implemented)
            self.assertFalse(prediction.quality_escalation_implemented)
            self.assertFalse(prediction.refinement_triggering_implemented)
            self.assertFalse(prediction.provider_model_routing_implemented)
            self.assertFalse(prediction.model_selection_implemented)
            self.assertFalse(prediction.provider_execution_implemented)
            self.assertFalse(prediction.human_input_request_implemented)
            self.assertFalse(prediction.workflow_control_implemented)
            self.assertFalse(prediction.retry_triggering_implemented)
            self.assertFalse(prediction.prompt_mutation_implemented)
            self.assertFalse(prediction.generated_output_mutation_implemented)
            self.assertTrue(prediction.advisory_only)

    def test_lookup_helpers_return_predictions_without_evaluation(self) -> None:
        plan = predict_quality_for_route(route=RouteName.REVIEW)
        recommended = quality_prediction_by_id(
            "quality_prediction::refinement_quality_profile",
            plan,
        )
        critical = quality_predictions_for_level("critical", plan)
        missing = quality_prediction_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(recommended.status, "recommended")
        self.assertEqual(recommended.predicted_quality_level, "critical")
        self.assertIn(recommended, critical)

    def test_plan_rejects_mismatched_predictions_or_recommendation(self) -> None:
        plan = predict_quality_for_route(route=RouteName.GENERATE)
        payload = plan.model_dump(mode="json")
        payload["prediction_ids"] = ("missing",) + tuple(payload["prediction_ids"][1:])

        with self.assertRaisesRegex(ValueError, "prediction_ids must match"):
            QualityPredictionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_prediction_id"] = "missing"

        with self.assertRaisesRegex(
            ValueError,
            "recommended_prediction_id must match",
        ):
            QualityPredictionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_quality_midpoint"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "recommended_quality_midpoint must match",
        ):
            QualityPredictionPlan(**payload)

    def test_quality_prediction_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Predict review quality for a shader design.",
            mode=AssistantMode.REVIEW,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = predict_quality_for_route(route_decision=baseline_decision)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, baseline_decision.route)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_quality_prediction_does_not_declare_application_terms(self) -> None:
        plan = predict_quality_for_route(route=RouteName.REVIEW)
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
                        prediction.source_quality_profile_id,
                        *prediction.source_model_profile_ids,
                        *prediction.source_provider_selection_profile_ids,
                        *prediction.evidence,
                        *prediction.advisory_actions,
                        *prediction.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "evaluate_quality(",
            "score_quality(",
            "execute_quality_escalation(",
            "trigger_refinement(",
            "select_model(",
            "route_provider(",
            "execute_provider(",
            "request_hitl(",
            "control_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
