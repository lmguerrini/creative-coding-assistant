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
    LearningConfidenceCalibrationPlan,
    calibrate_learning_confidence,
    learning_confidence_calibration_by_id,
    learning_confidence_calibrations_for_band,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")


class LearningConfidenceCalibrationTests(unittest.TestCase):
    def test_plan_calibrates_learning_confidence_without_training(self) -> None:
        plan = calibrate_learning_confidence(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "learning_confidence_calibration")
        self.assertEqual(
            plan.serialization_version,
            "learning_confidence_calibration_plan.v1",
        )
        self.assertEqual(
            plan.source_adaptive_learning_serialization_version,
            "adaptive_learning_plan.v1",
        )
        self.assertEqual(
            plan.source_execution_confidence_serialization_version,
            "execution_confidence_plan.v1",
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.record_count, 5)
        self.assertEqual(plan.overall_calibration_posture, "guarded")
        self.assertIn("does not train models", plan.authority_boundary)
        self.assertTrue(plan.learning_confidence_calibration_implemented)
        self.assertTrue(plan.calibration_metadata_implemented)
        self.assertTrue(plan.adaptive_learning_metadata_used)
        self.assertTrue(plan.execution_confidence_metadata_used)
        self.assertFalse(plan.model_training_implemented)
        self.assertFalse(plan.learning_feedback_application_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.runtime_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_records_map_confidence_and_require_hitl_for_risk(self) -> None:
        plan = calibrate_learning_confidence(route="generate")

        for record in plan.records:
            self.assertLessEqual(record.confidence_after, record.confidence_before)
            self.assertTrue(record.uncertainty_factors)
            self.assertTrue(record.calibration_rationale)
            self.assertTrue(record.hitl_required)
            self.assertIn("model_training", record.blocked_runtime_behaviors)
            self.assertFalse(record.model_training_implemented)
            self.assertFalse(record.learning_feedback_application_implemented)
            self.assertFalse(record.persistent_storage_write_implemented)
            self.assertFalse(record.runtime_mutation_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)
            self.assertFalse(record.runtime_evolution_implemented)

        guarded = learning_confidence_calibrations_for_band("guarded", plan)
        workflow = learning_confidence_calibration_by_id(
            "learning_confidence::adaptive_learning::workflow_pattern_learning",
            plan,
        )
        self.assertIsNotNone(workflow)
        self.assertGreaterEqual(len(guarded), 1)

    def test_plan_rejects_mismatched_calibration_metadata(self) -> None:
        plan = calibrate_learning_confidence()
        payload = plan.model_dump(mode="json")
        payload["calibration_ids"] = ("missing",) + tuple(
            payload["calibration_ids"][1:]
        )

        with self.assertRaisesRegex(ValueError, "calibration_ids must match"):
            LearningConfidenceCalibrationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["average_confidence_after"] -= 1

        with self.assertRaisesRegex(ValueError, "average_confidence_after must match"):
            LearningConfidenceCalibrationPlan(**payload)

    def test_calibration_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review learning confidence calibration for a creative workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
        )

        calibrate_learning_confidence(route=RouteName.GENERATE)
        provider = build_generation_provider(settings)
        after_decision = route_request(request)

        self.assertEqual(after_decision, baseline_decision)
        self.assertIsInstance(provider, OpenAIGenerationProvider)

    def test_calibration_metadata_does_not_declare_active_terms(self) -> None:
        plan = calibrate_learning_confidence(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for record in plan.records
                    for field in (
                        record.calibration_id,
                        record.calibration_rationale,
                        *record.uncertainty_factors,
                        *record.advisory_actions,
                        *record.evidence,
                        *record.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "train_model(",
            "apply_feedback(",
            "write_calibration(",
            "mutate_runtime(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
