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
    TechniqueLearningPlan,
    learn_techniques,
    route_request,
    technique_learning_pattern_by_id,
    technique_learning_patterns_for_priority,
    technique_learning_patterns_for_status,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_PATTERN_FIELDS = {
    "pattern_id",
    "pattern_kind",
    "status",
    "priority",
    "route_name",
    "task_type",
    "execution_mode_id",
    "source_technique_role",
    "source_learning_signal_id",
    "source_workflow_risk_factor_id",
    "technique_id",
    "technique_confidence",
    "technique_compatibility",
    "complexity_pressure",
    "performance_pressure",
    "learning_priority_score",
    "technique_learning_weight",
    "technique_learning_score",
    "hitl_required",
    "technique_pattern_tags",
    "technique_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "technique_learning_implemented",
    "technique_pattern_metadata_implemented",
    "creative_technique_metadata_used",
    "adaptive_learning_metadata_used",
    "technique_application_implemented",
    "prompt_rendering_implemented",
    "prompt_mutation_implemented",
    "runtime_selection_implemented",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "provider_execution_implemented",
    "artifact_execution_implemented",
    "agent_invocation_implemented",
    "hitl_request_emitted",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "retry_triggering_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class TechniqueLearningTests(unittest.TestCase):
    def test_plan_derives_technique_patterns_from_read_only_sources(self) -> None:
        plan = learn_techniques(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "technique_learning")
        self.assertEqual(plan.serialization_version, "technique_learning_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_creative_technique_role,
            "creative_technique_selector",
        )
        self.assertEqual(
            plan.source_adaptive_learning_serialization_version,
            "adaptive_learning_plan.v1",
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.primary_technique_id, "recursive_geometry")
        self.assertEqual(plan.pattern_count, 4)
        self.assertEqual(plan.review_required_pattern_count, 3)
        self.assertEqual(plan.guarded_pattern_count, 1)
        self.assertEqual(plan.hitl_required_pattern_count, 4)
        self.assertFalse(plan.applied_technique_pattern_ids)
        self.assertEqual(plan.overall_technique_learning_posture, "guarded")
        self.assertIn("does not render prompts", plan.authority_boundary)
        self.assertTrue(plan.technique_learning_implemented)
        self.assertTrue(plan.technique_pattern_metadata_implemented)
        self.assertTrue(plan.creative_technique_metadata_used)
        self.assertTrue(plan.adaptive_learning_metadata_used)
        self.assertFalse(plan.technique_application_implemented)
        self.assertFalse(plan.prompt_rendering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.runtime_selection_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.artifact_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_patterns_score_technique_learning_without_application(self) -> None:
        plan = learn_techniques(route="generate")

        for pattern in plan.patterns:
            dumped = pattern.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PATTERN_FIELDS)
            self.assertEqual(
                pattern.serialization_version,
                "technique_learning_pattern.v1",
            )
            self.assertEqual(pattern.route_name, RouteName.GENERATE)
            self.assertEqual(
                pattern.pattern_id,
                f"technique_learning::{pattern.pattern_kind}",
            )
            pressure_penalty = {"low": 0, "medium": 40, "high": 100}
            self.assertEqual(
                pattern.technique_learning_score,
                min(
                    1000,
                    max(
                        0,
                        int(pattern.technique_confidence * 600)
                        + pattern.learning_priority_score // 3
                        + pattern.technique_learning_weight
                        - pressure_penalty[pattern.complexity_pressure]
                        - pressure_penalty[pattern.performance_pressure],
                    ),
                ),
            )
            self.assertIn("technique_application", pattern.blocked_runtime_behaviors)
            self.assertTrue(pattern.technique_pattern_tags)
            self.assertTrue(pattern.advisory_actions)
            self.assertTrue(pattern.evidence)
            self.assertTrue(pattern.hitl_required)
            self.assertTrue(pattern.technique_learning_implemented)
            self.assertTrue(pattern.creative_technique_metadata_used)
            self.assertTrue(pattern.adaptive_learning_metadata_used)
            self.assertFalse(pattern.technique_application_implemented)
            self.assertFalse(pattern.prompt_rendering_implemented)
            self.assertFalse(pattern.prompt_mutation_implemented)
            self.assertFalse(pattern.runtime_selection_implemented)
            self.assertFalse(pattern.provider_model_routing_implemented)
            self.assertFalse(pattern.provider_execution_implemented)
            self.assertFalse(pattern.artifact_execution_implemented)
            self.assertFalse(pattern.workflow_control_implemented)
            self.assertFalse(pattern.workflow_graph_mutation_implemented)
            self.assertFalse(pattern.workflow_execution_implemented)
            self.assertFalse(pattern.persistent_storage_write_implemented)
            self.assertFalse(pattern.generated_output_mutation_implemented)
            self.assertFalse(pattern.runtime_evolution_implemented)
            self.assertTrue(pattern.advisory_only)

        fallback = technique_learning_pattern_by_id(
            "technique_learning::fallback_technique_learning",
            plan,
        )
        critical = technique_learning_patterns_for_priority("critical", plan)
        review = technique_learning_patterns_for_status("review_required", plan)
        self.assertIsNotNone(fallback)
        assert fallback is not None
        self.assertEqual(fallback.status, "guarded")
        self.assertEqual(fallback.priority, "guarded")
        self.assertEqual(len(critical), 1)
        self.assertEqual(len(review), 3)

    def test_plan_rejects_mismatched_technique_learning_metadata(self) -> None:
        plan = learn_techniques()
        payload = plan.model_dump(mode="json")
        payload["pattern_ids"] = ("missing",) + tuple(payload["pattern_ids"][1:])

        with self.assertRaisesRegex(ValueError, "pattern_ids must match"):
            TechniqueLearningPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_technique_learning_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_technique_learning_score must match",
        ):
            TechniqueLearningPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_technique_pattern_ids"] = (plan.pattern_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_technique_pattern_ids must remain empty",
        ):
            TechniqueLearningPlan(**payload)

    def test_learning_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review technique learning signals for a creative workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = learn_techniques(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_learning_does_not_declare_runtime_application_terms(self) -> None:
        plan = learn_techniques(route=RouteName.GENERATE)
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
                        pattern.pattern_kind,
                        pattern.source_technique_role,
                        pattern.source_learning_signal_id,
                        pattern.source_workflow_risk_factor_id,
                        pattern.technique_id,
                        pattern.technique_compatibility,
                        pattern.complexity_pressure,
                        pattern.performance_pressure,
                        *pattern.technique_pattern_tags,
                        pattern.technique_summary,
                        *pattern.advisory_actions,
                        *pattern.evidence,
                        *pattern.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "apply_technique(",
            "render_prompt(",
            "mutate_prompt(",
            "select_runtime(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "execute_artifact(",
            "invoke_agent(",
            "emit_hitl_request(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
            "trigger_retry(",
            "write_storage(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
