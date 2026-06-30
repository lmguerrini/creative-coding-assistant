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
    CreativeSuccessLearningPlan,
    creative_success_pattern_by_id,
    creative_success_patterns_for_quality_signal,
    learn_creative_success,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")


class CreativeSuccessLearningTests(unittest.TestCase):
    def test_plan_specializes_success_learning_for_creative_workflows(self) -> None:
        plan = learn_creative_success(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "creative_success_learning")
        self.assertEqual(
            plan.serialization_version,
            "creative_success_learning_plan.v1",
        )
        self.assertEqual(
            plan.source_workflow_success_serialization_version,
            "workflow_success_tracking_plan.v1",
        )
        self.assertEqual(
            plan.source_success_pattern_serialization_version,
            "success_pattern_discovery_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.pattern_count, 4)
        self.assertEqual(plan.overall_creative_success_posture, "guarded")
        self.assertIn("creative coding workflows", plan.authority_boundary)
        self.assertTrue(plan.creative_success_learning_implemented)
        self.assertTrue(plan.workflow_success_tracking_metadata_used)
        self.assertTrue(plan.success_pattern_discovery_metadata_used)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.automatic_preference_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_patterns_score_creative_success_dimensions_without_mutation(self) -> None:
        plan = learn_creative_success(route="generate")

        for pattern in plan.patterns:
            dimension_score = (
                pattern.artifact_dimension_score
                + pattern.aesthetic_dimension_score
                + pattern.usefulness_dimension_score
                + pattern.originality_dimension_score
            ) * 2
            expected_score = min(
                1000,
                max(
                    0,
                    pattern.workflow_success_score // 3
                    + pattern.success_pattern_score // 3
                    + dimension_score
                    + pattern.creative_success_weight,
                ),
            )
            self.assertEqual(pattern.creative_success_score, expected_score)
            self.assertTrue(pattern.creative_quality_signals)
            self.assertTrue(pattern.explainability)
            self.assertTrue(pattern.hitl_required)
            self.assertFalse(pattern.generated_output_mutation_implemented)
            self.assertFalse(pattern.automatic_preference_mutation_implemented)
            self.assertFalse(pattern.persistent_storage_write_implemented)
            self.assertFalse(pattern.learning_feedback_application_implemented)
            self.assertFalse(pattern.runtime_evolution_implemented)

        aesthetic = creative_success_patterns_for_quality_signal(
            "visual_aesthetic",
            plan,
        )
        quality = creative_success_pattern_by_id(
            "creative_success::creative_quality_success",
            plan,
        )
        self.assertEqual(len(aesthetic), 1)
        self.assertIsNotNone(quality)

    def test_plan_rejects_mismatched_creative_success_metadata(self) -> None:
        plan = learn_creative_success()
        payload = plan.model_dump(mode="json")
        payload["pattern_ids"] = ("missing",) + tuple(payload["pattern_ids"][1:])

        with self.assertRaisesRegex(ValueError, "pattern_ids must match"):
            CreativeSuccessLearningPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_creative_success_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_creative_success_score must match",
        ):
            CreativeSuccessLearningPlan(**payload)

    def test_creative_success_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review creative success learning for a creative workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
        )

        learn_creative_success(route=RouteName.GENERATE)
        provider = build_generation_provider(settings)
        after_decision = route_request(request)

        self.assertEqual(after_decision, baseline_decision)
        self.assertIsInstance(provider, OpenAIGenerationProvider)

    def test_success_metadata_does_not_declare_active_terms(self) -> None:
        plan = learn_creative_success(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for pattern in plan.patterns
                    for field in (
                        pattern.pattern_id,
                        *pattern.creative_quality_signals,
                        pattern.explainability,
                        *pattern.advisory_actions,
                        *pattern.evidence,
                        *pattern.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "mutate_generated_output(",
            "mutate_user_preferences(",
            "write_creative_success(",
            "apply_feedback(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
