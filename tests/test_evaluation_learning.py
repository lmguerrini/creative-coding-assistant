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
    EvaluationLearningPlan,
    evaluation_learning_pattern_by_id,
    evaluation_learning_patterns_for_priority,
    evaluation_learning_patterns_for_status,
    learn_evaluations,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_SOURCE_ENGINES = (
    "creative_critic",
    "self_evaluation",
    "creative_confidence",
    "evaluation_reports",
)
REQUIRED_PATTERN_FIELDS = {
    "pattern_id",
    "pattern_kind",
    "status",
    "priority",
    "route_name",
    "task_type",
    "execution_mode_id",
    "source_contract_registry_role",
    "source_engine_id",
    "source_engine_name",
    "source_learning_signal_id",
    "source_workflow_risk_factor_id",
    "required_input_count",
    "optional_input_count",
    "produced_signal_count",
    "confidence_signal_count",
    "ambiguity_signal_count",
    "risk_signal_count",
    "future_execution_hook_count",
    "relative_cost",
    "relative_latency",
    "cacheability",
    "parallelization_support",
    "learning_priority_score",
    "evaluation_learning_weight",
    "evaluation_learning_score",
    "hitl_required",
    "evaluation_pattern_tags",
    "evaluation_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "evaluation_learning_implemented",
    "evaluation_pattern_metadata_implemented",
    "evaluation_contract_metadata_used",
    "adaptive_learning_metadata_used",
    "evaluation_execution_implemented",
    "generated_output_evaluation_implemented",
    "score_mutation_implemented",
    "confidence_mutation_implemented",
    "reflection_loop_execution_implemented",
    "report_generation_implemented",
    "workflow_order_change_implemented",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "provider_execution_implemented",
    "runtime_selection_implemented",
    "artifact_execution_implemented",
    "hitl_request_emitted",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class EvaluationLearningTests(unittest.TestCase):
    def test_plan_derives_evaluation_patterns_from_contract_metadata(self) -> None:
        plan = learn_evaluations(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "evaluation_learning")
        self.assertEqual(plan.serialization_version, "evaluation_learning_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_contract_registry_role,
            "evaluation_engine_contract_registry",
        )
        self.assertEqual(
            plan.source_contract_registry_serialization_version,
            "evaluation_engine_contract_registry.v1",
        )
        self.assertEqual(
            plan.source_adaptive_learning_serialization_version,
            "adaptive_learning_plan.v1",
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.source_engine_ids, REQUIRED_SOURCE_ENGINES)
        self.assertEqual(plan.pattern_count, 4)
        self.assertEqual(plan.review_required_pattern_count, 3)
        self.assertEqual(plan.guarded_pattern_count, 1)
        self.assertEqual(plan.hitl_required_pattern_count, 4)
        self.assertFalse(plan.applied_evaluation_pattern_ids)
        self.assertEqual(plan.overall_evaluation_learning_posture, "guarded")
        self.assertIn("does not run evaluations", plan.authority_boundary)
        self.assertTrue(plan.evaluation_learning_implemented)
        self.assertTrue(plan.evaluation_pattern_metadata_implemented)
        self.assertTrue(plan.evaluation_contract_metadata_used)
        self.assertTrue(plan.adaptive_learning_metadata_used)
        self.assertFalse(plan.evaluation_execution_implemented)
        self.assertFalse(plan.generated_output_evaluation_implemented)
        self.assertFalse(plan.score_mutation_implemented)
        self.assertFalse(plan.confidence_mutation_implemented)
        self.assertFalse(plan.reflection_loop_execution_implemented)
        self.assertFalse(plan.report_generation_implemented)
        self.assertFalse(plan.workflow_order_change_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.runtime_selection_implemented)
        self.assertFalse(plan.artifact_execution_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_patterns_score_without_running_evaluation(self) -> None:
        plan = learn_evaluations(route="generate")

        cost_penalty = {"low": 0, "medium": 40}
        latency_penalty = {"low": 0, "medium": 40}
        for pattern in plan.patterns:
            dumped = pattern.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PATTERN_FIELDS)
            self.assertEqual(
                pattern.serialization_version,
                "evaluation_learning_pattern.v1",
            )
            self.assertEqual(pattern.route_name, RouteName.GENERATE)
            self.assertEqual(
                pattern.pattern_id,
                f"evaluation_learning::{pattern.pattern_kind}",
            )
            self.assertEqual(
                pattern.evaluation_learning_score,
                min(
                    1000,
                    max(
                        0,
                        pattern.produced_signal_count * 40
                        + pattern.confidence_signal_count * 45
                        + pattern.ambiguity_signal_count * 35
                        + pattern.risk_signal_count * 35
                        + pattern.future_execution_hook_count * 25
                        + pattern.learning_priority_score // 3
                        + pattern.evaluation_learning_weight
                        - pattern.required_input_count * 20
                        - pattern.optional_input_count * 8
                        - cost_penalty[pattern.relative_cost]
                        - latency_penalty[pattern.relative_latency],
                    ),
                ),
            )
            self.assertIn("evaluation_execution", pattern.blocked_runtime_behaviors)
            self.assertIn(pattern.source_engine_id, REQUIRED_SOURCE_ENGINES)
            self.assertTrue(pattern.evaluation_pattern_tags)
            self.assertTrue(pattern.advisory_actions)
            self.assertTrue(pattern.evidence)
            self.assertTrue(pattern.hitl_required)
            self.assertTrue(pattern.evaluation_learning_implemented)
            self.assertTrue(pattern.evaluation_contract_metadata_used)
            self.assertTrue(pattern.adaptive_learning_metadata_used)
            self.assertFalse(pattern.evaluation_execution_implemented)
            self.assertFalse(pattern.generated_output_evaluation_implemented)
            self.assertFalse(pattern.score_mutation_implemented)
            self.assertFalse(pattern.confidence_mutation_implemented)
            self.assertFalse(pattern.reflection_loop_execution_implemented)
            self.assertFalse(pattern.report_generation_implemented)
            self.assertFalse(pattern.workflow_order_change_implemented)
            self.assertFalse(pattern.provider_model_routing_implemented)
            self.assertFalse(pattern.provider_execution_implemented)
            self.assertFalse(pattern.runtime_selection_implemented)
            self.assertFalse(pattern.artifact_execution_implemented)
            self.assertFalse(pattern.workflow_control_implemented)
            self.assertFalse(pattern.workflow_graph_mutation_implemented)
            self.assertFalse(pattern.workflow_execution_implemented)
            self.assertFalse(pattern.persistent_storage_write_implemented)
            self.assertFalse(pattern.generated_output_mutation_implemented)
            self.assertFalse(pattern.runtime_evolution_implemented)
            self.assertTrue(pattern.advisory_only)

        guarded = evaluation_learning_pattern_by_id(
            "evaluation_learning::report_guardrail_learning",
            plan,
        )
        critical = evaluation_learning_patterns_for_priority("critical", plan)
        review = evaluation_learning_patterns_for_status("review_required", plan)
        self.assertIsNotNone(guarded)
        assert guarded is not None
        self.assertEqual(guarded.status, "guarded")
        self.assertEqual(guarded.priority, "guarded")
        self.assertEqual(len(critical), 3)
        self.assertEqual(len(review), 3)

    def test_plan_rejects_mismatched_evaluation_learning_metadata(self) -> None:
        plan = learn_evaluations()
        payload = plan.model_dump(mode="json")
        payload["pattern_ids"] = ("missing",) + tuple(payload["pattern_ids"][1:])

        with self.assertRaisesRegex(ValueError, "pattern_ids must match"):
            EvaluationLearningPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_evaluation_learning_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_evaluation_learning_score must match",
        ):
            EvaluationLearningPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_evaluation_pattern_ids"] = (plan.pattern_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_evaluation_pattern_ids must remain empty",
        ):
            EvaluationLearningPlan(**payload)

    def test_learning_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review evaluation learning signals for a creative workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = learn_evaluations(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_learning_does_not_declare_evaluation_execution_terms(self) -> None:
        plan = learn_evaluations(route=RouteName.GENERATE)
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
                        pattern.source_contract_registry_role,
                        pattern.source_engine_id,
                        pattern.source_engine_name,
                        pattern.source_learning_signal_id,
                        pattern.source_workflow_risk_factor_id,
                        pattern.relative_cost,
                        pattern.relative_latency,
                        pattern.cacheability,
                        pattern.parallelization_support,
                        *pattern.evaluation_pattern_tags,
                        pattern.evaluation_summary,
                        *pattern.advisory_actions,
                        *pattern.evidence,
                        *pattern.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "run_evaluation(",
            "evaluate_generated_output(",
            "mutate_score(",
            "mutate_confidence(",
            "execute_reflection_loop(",
            "generate_report(",
            "change_workflow_order(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "select_runtime(",
            "execute_artifact(",
            "emit_hitl_request(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
