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
    CreativeFailureLearningPlan,
    creative_failure_pattern_by_id,
    creative_failure_patterns_for_mode,
    learn_creative_failures,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")


class CreativeFailureLearningTests(unittest.TestCase):
    def test_plan_specializes_failure_learning_for_creative_workflows(self) -> None:
        plan = learn_creative_failures(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "creative_failure_learning")
        self.assertEqual(
            plan.serialization_version,
            "creative_failure_learning_plan.v1",
        )
        self.assertEqual(
            plan.source_failure_tracking_serialization_version,
            "failure_tracking_plan.v1",
        )
        self.assertEqual(
            plan.source_failure_pattern_serialization_version,
            "failure_pattern_discovery_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.pattern_count, 4)
        self.assertEqual(plan.overall_creative_failure_posture, "guarded")
        self.assertIn("creative coding workflows", plan.authority_boundary)
        self.assertTrue(plan.creative_failure_learning_implemented)
        self.assertTrue(plan.failure_tracking_metadata_used)
        self.assertTrue(plan.failure_pattern_discovery_metadata_used)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.automatic_remediation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.terminal_failure_routing_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_patterns_score_creative_failure_dimensions(self) -> None:
        plan = learn_creative_failures(route="generate")

        for pattern in plan.patterns:
            dimension_score = (
                pattern.artifact_dimension_score
                + pattern.preview_dimension_score
                + pattern.runtime_dimension_score
                + pattern.aesthetic_dimension_score
                + pattern.prompt_dimension_score
                + pattern.retrieval_dimension_score
            )
            expected_score = min(
                1000,
                max(
                    0,
                    pattern.failure_tracking_score // 3
                    + pattern.failure_pattern_score // 3
                    + dimension_score
                    + pattern.creative_failure_weight,
                ),
            )
            self.assertEqual(pattern.creative_failure_score, expected_score)
            self.assertTrue(pattern.common_creative_failure_modes)
            self.assertTrue(pattern.explainability)
            self.assertTrue(pattern.hitl_required)
            self.assertFalse(pattern.generated_output_mutation_implemented)
            self.assertFalse(pattern.automatic_remediation_implemented)
            self.assertFalse(pattern.persistent_storage_write_implemented)
            self.assertFalse(pattern.terminal_failure_routing_implemented)
            self.assertFalse(pattern.runtime_evolution_implemented)

        preview = creative_failure_patterns_for_mode(
            "preview_surface_mismatch",
            plan,
        )
        runtime = creative_failure_pattern_by_id(
            "creative_failure::runtime_execution_failure",
            plan,
        )
        self.assertEqual(len(preview), 1)
        self.assertIsNotNone(runtime)

    def test_plan_rejects_mismatched_creative_failure_metadata(self) -> None:
        plan = learn_creative_failures()
        payload = plan.model_dump(mode="json")
        payload["pattern_ids"] = ("missing",) + tuple(payload["pattern_ids"][1:])

        with self.assertRaisesRegex(ValueError, "pattern_ids must match"):
            CreativeFailureLearningPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_creative_failure_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_creative_failure_score must match",
        ):
            CreativeFailureLearningPlan(**payload)

    def test_creative_failure_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review creative failure learning for a creative workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
        )

        learn_creative_failures(route=RouteName.GENERATE)
        provider = build_generation_provider(settings)
        after_decision = route_request(request)

        self.assertEqual(after_decision, baseline_decision)
        self.assertIsInstance(provider, OpenAIGenerationProvider)

    def test_failure_metadata_does_not_declare_active_terms(self) -> None:
        plan = learn_creative_failures(route=RouteName.GENERATE)
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
                        *pattern.common_creative_failure_modes,
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
            "remediate_failure(",
            "write_creative_failure(",
            "route_terminal_failure(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
