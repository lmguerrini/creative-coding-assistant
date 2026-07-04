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
    CreativeConsistencyPredictionPlan,
    CreativeQualityPrediction,
    CreativeQualitySignal,
    creative_consistency_prediction_by_id,
    creative_consistency_predictions_for_band,
    predict_creative_consistency,
    route_request,
)

REQUIRED_CREATIVE_CONSISTENCY_PREDICTION_FIELDS = {
    "prediction_id",
    "source_quality_prediction_role",
    "source_predicted_quality_level",
    "source_readiness_score",
    "source_quality_signal_dimension",
    "source_quality_signal_score",
    "predicted_consistency_band",
    "predicted_consistency_range",
    "predicted_consistency_midpoint",
    "prediction_confidence",
    "status",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "creative_consistency_predictor_implemented",
    "advisory_consistency_prediction_implemented",
    "source_quality_metadata_only",
    "generated_output_consistency_evaluation_implemented",
    "consistency_validation_execution_implemented",
    "artifact_scoring_implemented",
    "artifact_critique_implemented",
    "artifact_selection_implemented",
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


class CreativeConsistencyPredictorTests(unittest.TestCase):
    def test_predictor_uses_creative_quality_coherence_metadata(self) -> None:
        source = _quality_prediction()
        plan = predict_creative_consistency(creative_quality_prediction=source)

        self.assertEqual(plan.role, "creative_consistency_predictor")
        self.assertEqual(
            plan.serialization_version,
            "creative_consistency_prediction_plan.v1",
        )
        self.assertEqual(
            plan.source_creative_quality_prediction_role,
            source.role,
        )
        self.assertEqual(plan.source_predicted_quality_level, "promising")
        self.assertEqual(plan.source_readiness_score, 72)
        self.assertEqual(plan.prediction_count, 4)
        self.assertEqual(
            plan.recommended_prediction_id,
            "creative_consistency_prediction::emotional_coherence",
        )
        self.assertEqual(plan.recommended_consistency_band, "watch")
        self.assertEqual(plan.recommended_consistency_midpoint, 40)
        self.assertEqual(plan.strong_or_stable_prediction_count, 3)
        self.assertEqual(plan.watch_or_fragile_prediction_count, 1)
        self.assertEqual(plan.fragile_prediction_count, 0)
        self.assertIn(
            "does not evaluate generated output",
            plan.authority_boundary,
        )
        self.assertTrue(plan.creative_consistency_predictor_implemented)
        self.assertTrue(plan.advisory_consistency_prediction_implemented)
        self.assertTrue(plan.source_quality_metadata_only)
        self.assertFalse(plan.generated_output_consistency_evaluation_implemented)
        self.assertFalse(plan.consistency_validation_execution_implemented)
        self.assertFalse(plan.artifact_scoring_implemented)
        self.assertFalse(plan.artifact_critique_implemented)
        self.assertFalse(plan.artifact_selection_implemented)
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

    def test_consistency_predictions_are_advisory_only(self) -> None:
        plan = predict_creative_consistency(
            creative_quality_prediction=_quality_prediction(),
        )

        for prediction in plan.predictions:
            dumped = prediction.model_dump(mode="json")
            self.assertEqual(
                set(dumped),
                REQUIRED_CREATIVE_CONSISTENCY_PREDICTION_FIELDS,
            )
            self.assertEqual(
                prediction.serialization_version,
                "creative_consistency_prediction.v1",
            )
            self.assertEqual(
                prediction.predicted_consistency_midpoint,
                prediction.source_quality_signal_score * 10,
            )
            self.assertIn(
                "generated_output_consistency_evaluation",
                prediction.blocked_runtime_behaviors,
            )
            self.assertTrue(prediction.creative_consistency_predictor_implemented)
            self.assertTrue(prediction.advisory_consistency_prediction_implemented)
            self.assertTrue(prediction.source_quality_metadata_only)
            self.assertFalse(
                prediction.generated_output_consistency_evaluation_implemented,
            )
            self.assertFalse(prediction.consistency_validation_execution_implemented)
            self.assertFalse(prediction.artifact_scoring_implemented)
            self.assertFalse(prediction.artifact_critique_implemented)
            self.assertFalse(prediction.artifact_selection_implemented)
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

    def test_lookup_helpers_return_predictions_without_validation(self) -> None:
        plan = predict_creative_consistency(
            creative_quality_prediction=_quality_prediction(),
        )
        recommended = creative_consistency_prediction_by_id(
            "creative_consistency_prediction::emotional_coherence",
            plan,
        )
        stable_predictions = creative_consistency_predictions_for_band("stable", plan)
        missing = creative_consistency_prediction_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(recommended.status, "recommended")
        self.assertEqual(recommended.predicted_consistency_band, "watch")
        self.assertEqual(len(stable_predictions), 2)

    def test_plan_rejects_mismatched_predictions_or_recommendation(self) -> None:
        plan = predict_creative_consistency(
            creative_quality_prediction=_quality_prediction(),
        )
        payload = plan.model_dump(mode="json")
        payload["prediction_ids"] = ("missing",) + tuple(payload["prediction_ids"][1:])

        with self.assertRaisesRegex(ValueError, "prediction_ids must match"):
            CreativeConsistencyPredictionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_prediction_id"] = "missing"

        with self.assertRaisesRegex(
            ValueError,
            "recommended_prediction_id must match",
        ):
            CreativeConsistencyPredictionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_consistency_midpoint"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "recommended_consistency_midpoint must match",
        ):
            CreativeConsistencyPredictionPlan(**payload)

    def test_consistency_prediction_does_not_change_routing_or_provider(self) -> None:
        request = AssistantRequest(
            query="Predict creative consistency for a shader study.",
            mode=AssistantMode.REVIEW,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        predict_creative_consistency(creative_quality_prediction=_quality_prediction())
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_consistency_prediction_does_not_declare_application_terms(self) -> None:
        plan = predict_creative_consistency(
            creative_quality_prediction=_quality_prediction(),
        )
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
                        prediction.source_quality_signal_dimension,
                        prediction.predicted_consistency_band,
                        *prediction.evidence,
                        *prediction.advisory_actions,
                        *prediction.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "evaluate_generated_consistency(",
            "validate_consistency(",
            "score_artifact(",
            "critique_artifact(",
            "select_artifact(",
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


def _quality_prediction() -> CreativeQualityPrediction:
    return CreativeQualityPrediction(
        predicted_quality_level="promising",
        confidence=0.74,
        readiness_score=72,
        strongest_quality_signals=(
            CreativeQualitySignal(
                dimension="symbolic_coherence",
                score=8,
                summary="Symbolic direction has a clear anchor.",
                evidence=("Phoenix motif and mandala structure align.",),
            ),
            CreativeQualitySignal(
                dimension="narrative_coherence",
                score=7,
                summary="Narrative phases form a usable progression.",
                evidence=("Dissolve, threshold, and reintegration are sequenced.",),
            ),
        ),
        weakest_quality_signals=(
            CreativeQualitySignal(
                dimension="emotional_coherence",
                score=4,
                summary="Emotional tone needs a tighter dominant through-line.",
                evidence=("Awe, rupture, and serenity compete for priority.",),
            ),
            CreativeQualitySignal(
                dimension="aesthetic_coherence_potential",
                score=6,
                summary="Aesthetic direction is present but still broad.",
                evidence=("Gold and blue palette is named without material limits.",),
            ),
        ),
        likely_failure_modes=(
            "Competing emotional tones may dilute the final visual through-line.",
        ),
        suggested_improvements=(
            "Choose one dominant emotional phase before generation.",
        ),
        prompt_guidance=(
            "Keep symbolic, narrative, and emotional consistency visible.",
        ),
        evidence=("Synthetic source quality prediction for consistency projection.",),
    )


if __name__ == "__main__":
    unittest.main()
