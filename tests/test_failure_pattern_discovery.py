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
    FailurePatternDiscoveryPlan,
    discover_failure_patterns,
    failure_pattern_by_id,
    failure_patterns_for_priority,
    failure_patterns_for_status,
    route_request,
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
    "source_failure_indicator_id",
    "source_failure_panel_id",
    "source_improvement_signal_id",
    "source_learning_signal_id",
    "source_workflow_risk_factor_id",
    "failure_signal_count",
    "guardrail_signal_count",
    "failure_tracking_score",
    "improvement_score",
    "learning_priority_score",
    "failure_pattern_weight",
    "failure_pattern_score",
    "hitl_required",
    "failure_pattern_tags",
    "failure_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "failure_pattern_discovery_implemented",
    "failure_pattern_metadata_implemented",
    "failure_tracking_metadata_used",
    "continuous_improvement_metadata_used",
    "adaptive_learning_metadata_used",
    "runtime_failure_observation_implemented",
    "live_error_classification_implemented",
    "terminal_failure_routing_implemented",
    "failure_handling_implemented",
    "failure_repair_implemented",
    "terminal_routing_mutation_implemented",
    "learning_feedback_application_implemented",
    "learning_memory_persistence_implemented",
    "learning_policy_update_implemented",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
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


class FailurePatternDiscoveryTests(unittest.TestCase):
    def test_plan_discovers_failure_patterns_from_read_only_metadata(self) -> None:
        plan = discover_failure_patterns(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "failure_pattern_discovery")
        self.assertEqual(
            plan.serialization_version,
            "failure_pattern_discovery_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_failure_tracking_serialization_version,
            "failure_tracking_plan.v1",
        )
        self.assertEqual(
            plan.source_continuous_improvement_serialization_version,
            "continuous_improvement_signal_plan.v1",
        )
        self.assertEqual(
            plan.source_adaptive_learning_serialization_version,
            "adaptive_learning_plan.v1",
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.pattern_count, 4)
        self.assertEqual(plan.guarded_pattern_count, 4)
        self.assertEqual(plan.hitl_required_pattern_count, 4)
        self.assertFalse(plan.applied_failure_pattern_ids)
        self.assertEqual(plan.overall_failure_pattern_posture, "guarded")
        self.assertIn("does not observe runtime failures", plan.authority_boundary)
        self.assertTrue(plan.failure_pattern_discovery_implemented)
        self.assertTrue(plan.failure_pattern_metadata_implemented)
        self.assertTrue(plan.failure_tracking_metadata_used)
        self.assertTrue(plan.continuous_improvement_metadata_used)
        self.assertTrue(plan.adaptive_learning_metadata_used)
        self.assertFalse(plan.runtime_failure_observation_implemented)
        self.assertFalse(plan.live_error_classification_implemented)
        self.assertFalse(plan.terminal_failure_routing_implemented)
        self.assertFalse(plan.failure_handling_implemented)
        self.assertFalse(plan.failure_repair_implemented)
        self.assertFalse(plan.terminal_routing_mutation_implemented)
        self.assertFalse(plan.learning_feedback_application_implemented)
        self.assertFalse(plan.learning_memory_persistence_implemented)
        self.assertFalse(plan.learning_policy_update_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_patterns_score_failure_discovery_without_handling(self) -> None:
        plan = discover_failure_patterns(route="generate")

        for pattern in plan.patterns:
            dumped = pattern.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PATTERN_FIELDS)
            self.assertEqual(pattern.serialization_version, "failure_pattern.v1")
            self.assertEqual(pattern.route_name, RouteName.GENERATE)
            self.assertEqual(
                pattern.pattern_id,
                f"failure_pattern::{pattern.pattern_kind}",
            )
            self.assertEqual(
                pattern.failure_pattern_score,
                min(
                    1000,
                    max(
                        0,
                        pattern.failure_tracking_score
                        + pattern.improvement_score // 3
                        + pattern.learning_priority_score // 4
                        + pattern.failure_signal_count * 3
                        + pattern.guardrail_signal_count * 8
                        + pattern.failure_pattern_weight,
                    ),
                ),
            )
            self.assertIn(
                "runtime_failure_observation",
                pattern.blocked_runtime_behaviors,
            )
            self.assertEqual(pattern.status, "guarded")
            self.assertEqual(pattern.priority, "guarded")
            self.assertTrue(pattern.failure_pattern_tags)
            self.assertTrue(pattern.advisory_actions)
            self.assertTrue(pattern.evidence)
            self.assertTrue(pattern.hitl_required)
            self.assertFalse(pattern.runtime_failure_observation_implemented)
            self.assertFalse(pattern.live_error_classification_implemented)
            self.assertFalse(pattern.terminal_failure_routing_implemented)
            self.assertFalse(pattern.failure_handling_implemented)
            self.assertFalse(pattern.failure_repair_implemented)
            self.assertFalse(pattern.terminal_routing_mutation_implemented)
            self.assertFalse(pattern.learning_feedback_application_implemented)
            self.assertFalse(pattern.workflow_control_implemented)
            self.assertFalse(pattern.workflow_graph_mutation_implemented)
            self.assertFalse(pattern.workflow_execution_implemented)
            self.assertFalse(pattern.persistent_storage_write_implemented)
            self.assertFalse(pattern.generated_output_mutation_implemented)
            self.assertFalse(pattern.runtime_evolution_implemented)
            self.assertTrue(pattern.advisory_only)

        guarded = failure_pattern_by_id(
            "failure_pattern::routing_failure_pattern",
            plan,
        )
        self.assertIsNotNone(guarded)
        self.assertEqual(len(failure_patterns_for_priority("guarded", plan)), 4)
        self.assertEqual(len(failure_patterns_for_status("guarded", plan)), 4)

    def test_plan_rejects_mismatched_failure_pattern_metadata(self) -> None:
        plan = discover_failure_patterns()
        payload = plan.model_dump(mode="json")
        payload["pattern_ids"] = ("missing",) + tuple(payload["pattern_ids"][1:])

        with self.assertRaisesRegex(ValueError, "pattern_ids must match"):
            FailurePatternDiscoveryPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_failure_pattern_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_failure_pattern_score must match",
        ):
            FailurePatternDiscoveryPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_failure_pattern_ids"] = (plan.pattern_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_failure_pattern_ids must remain empty",
        ):
            FailurePatternDiscoveryPlan(**payload)

    def test_learning_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review failure pattern discovery for a creative workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = discover_failure_patterns(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_learning_does_not_declare_failure_handling_terms(self) -> None:
        plan = discover_failure_patterns(route=RouteName.GENERATE)
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
                        pattern.source_failure_indicator_id,
                        pattern.source_failure_panel_id,
                        pattern.source_improvement_signal_id,
                        pattern.source_learning_signal_id,
                        pattern.source_workflow_risk_factor_id,
                        *pattern.failure_pattern_tags,
                        pattern.failure_summary,
                        *pattern.advisory_actions,
                        *pattern.evidence,
                        *pattern.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "observe_runtime_failure(",
            "classify_live_error(",
            "route_terminal_failure(",
            "handle_failure(",
            "repair_failure(",
            "mutate_terminal_routing(",
            "apply_feedback(",
            "persist_learning_memory(",
            "update_learning_policy(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "invoke_agent(",
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
