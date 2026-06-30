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
    StrategyLearningPlan,
    learn_strategies,
    route_request,
    strategy_learning_pattern_by_id,
    strategy_learning_patterns_for_priority,
    strategy_learning_patterns_for_status,
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
    "source_strategy_id",
    "source_learning_signal_id",
    "source_workflow_risk_factor_id",
    "source_strategy_status",
    "source_strategy_kind",
    "dynamic_strategy_score",
    "learning_priority_score",
    "unavailable_reason_count",
    "strategy_learning_weight",
    "strategy_learning_score",
    "hitl_required",
    "provider_sequence",
    "model_profile_sequence",
    "strategy_pattern_tags",
    "strategy_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "strategy_learning_implemented",
    "strategy_pattern_metadata_implemented",
    "adaptive_strategy_metadata_used",
    "adaptive_learning_metadata_used",
    "strategy_application_implemented",
    "strategy_selection_mutation_implemented",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "automatic_model_download_implemented",
    "provider_execution_implemented",
    "local_runtime_probe_implemented",
    "local_model_inventory_scan_implemented",
    "agent_invocation_implemented",
    "hitl_request_emitted",
    "budget_enforcement_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "graph_compilation_implemented",
    "workflow_execution_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class StrategyLearningTests(unittest.TestCase):
    def test_plan_derives_strategy_patterns_from_advisory_sources(self) -> None:
        plan = learn_strategies(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "strategy_learning")
        self.assertEqual(plan.serialization_version, "strategy_learning_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_adaptive_strategy_serialization_version,
            "adaptive_execution_strategy_selection_plan.v1",
        )
        self.assertEqual(
            plan.source_adaptive_learning_serialization_version,
            "adaptive_learning_plan.v1",
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.pattern_count, 4)
        self.assertEqual(plan.review_required_pattern_count, 3)
        self.assertEqual(plan.guarded_pattern_count, 1)
        self.assertEqual(plan.hitl_required_pattern_count, 4)
        self.assertFalse(plan.applied_strategy_pattern_ids)
        self.assertIsNone(plan.applied_source_strategy_id)
        self.assertEqual(plan.overall_strategy_learning_posture, "guarded")
        self.assertIn("does not apply strategies", plan.authority_boundary)
        self.assertTrue(plan.strategy_learning_implemented)
        self.assertTrue(plan.strategy_pattern_metadata_implemented)
        self.assertTrue(plan.adaptive_strategy_metadata_used)
        self.assertTrue(plan.adaptive_learning_metadata_used)
        self.assertFalse(plan.strategy_application_implemented)
        self.assertFalse(plan.strategy_selection_mutation_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.graph_compilation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_patterns_score_strategy_learning_without_application(self) -> None:
        plan = learn_strategies(route="generate")

        for pattern in plan.patterns:
            dumped = pattern.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PATTERN_FIELDS)
            self.assertEqual(
                pattern.serialization_version,
                "strategy_learning_pattern.v1",
            )
            self.assertEqual(pattern.route_name, RouteName.GENERATE)
            self.assertEqual(
                pattern.pattern_id,
                f"strategy_learning::{pattern.pattern_kind}",
            )
            self.assertEqual(
                pattern.strategy_learning_score,
                min(
                    1000,
                    max(
                        0,
                        pattern.dynamic_strategy_score
                        + pattern.learning_priority_score // 2
                        + pattern.strategy_learning_weight
                        - pattern.unavailable_reason_count * 35,
                    ),
                ),
            )
            self.assertIn("strategy_application", pattern.blocked_runtime_behaviors)
            self.assertTrue(pattern.strategy_pattern_tags)
            self.assertTrue(pattern.advisory_actions)
            self.assertTrue(pattern.evidence)
            self.assertTrue(pattern.hitl_required)
            self.assertTrue(pattern.strategy_learning_implemented)
            self.assertTrue(pattern.adaptive_strategy_metadata_used)
            self.assertTrue(pattern.adaptive_learning_metadata_used)
            self.assertFalse(pattern.strategy_application_implemented)
            self.assertFalse(pattern.strategy_selection_mutation_implemented)
            self.assertFalse(pattern.provider_model_routing_implemented)
            self.assertFalse(pattern.provider_execution_implemented)
            self.assertFalse(pattern.workflow_control_implemented)
            self.assertFalse(pattern.workflow_graph_mutation_implemented)
            self.assertFalse(pattern.graph_compilation_implemented)
            self.assertFalse(pattern.workflow_execution_implemented)
            self.assertFalse(pattern.persistent_storage_write_implemented)
            self.assertFalse(pattern.generated_output_mutation_implemented)
            self.assertFalse(pattern.runtime_evolution_implemented)
            self.assertTrue(pattern.advisory_only)

        guarded = strategy_learning_pattern_by_id(
            "strategy_learning::human_guarded_strategy_learning",
            plan,
        )
        critical = strategy_learning_patterns_for_priority("critical", plan)
        review = strategy_learning_patterns_for_status("review_required", plan)
        self.assertIsNotNone(guarded)
        assert guarded is not None
        self.assertEqual(guarded.status, "guarded")
        self.assertEqual(guarded.priority, "guarded")
        self.assertEqual(len(critical), 2)
        self.assertEqual(len(review), 3)

    def test_plan_rejects_mismatched_strategy_learning_metadata(self) -> None:
        plan = learn_strategies()
        payload = plan.model_dump(mode="json")
        payload["pattern_ids"] = ("missing",) + tuple(payload["pattern_ids"][1:])

        with self.assertRaisesRegex(ValueError, "pattern_ids must match"):
            StrategyLearningPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_strategy_learning_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_strategy_learning_score must match",
        ):
            StrategyLearningPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_strategy_pattern_ids"] = (plan.pattern_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_strategy_pattern_ids must remain empty",
        ):
            StrategyLearningPlan(**payload)

    def test_learning_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review strategy learning signals for a guarded workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = learn_strategies(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_learning_does_not_declare_runtime_application_terms(self) -> None:
        plan = learn_strategies(route=RouteName.GENERATE)
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
                        pattern.source_strategy_id,
                        pattern.source_learning_signal_id,
                        pattern.source_workflow_risk_factor_id,
                        pattern.source_strategy_status,
                        pattern.source_strategy_kind,
                        *pattern.provider_sequence,
                        *pattern.model_profile_sequence,
                        *pattern.strategy_pattern_tags,
                        pattern.strategy_summary,
                        *pattern.advisory_actions,
                        *pattern.evidence,
                        *pattern.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "apply_strategy(",
            "mutate_strategy_selection(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "download_model(",
            "execute_provider(",
            "probe_local_runtime(",
            "scan_local_models(",
            "invoke_agent(",
            "emit_hitl_request(",
            "enforce_budget(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "compile_graph(",
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
