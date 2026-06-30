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
    SuccessPatternDiscoveryPlan,
    discover_success_patterns,
    route_request,
    success_pattern_by_id,
    success_patterns_for_priority,
    success_patterns_for_status,
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
    "source_success_indicator_id",
    "source_improvement_signal_id",
    "source_learning_signal_id",
    "source_workflow_risk_factor_id",
    "source_success_status",
    "source_improvement_status",
    "success_confidence_band",
    "workflow_success_score",
    "improvement_score",
    "learning_priority_score",
    "success_pattern_weight",
    "success_pattern_score",
    "hitl_required",
    "success_pattern_tags",
    "success_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "success_pattern_discovery_implemented",
    "success_pattern_metadata_implemented",
    "workflow_success_metadata_used",
    "continuous_improvement_metadata_used",
    "adaptive_learning_metadata_used",
    "runtime_success_observation_implemented",
    "live_telemetry_collection_implemented",
    "success_metric_persistence_implemented",
    "success_pattern_application_implemented",
    "learning_feedback_application_implemented",
    "learning_memory_persistence_implemented",
    "learning_policy_update_implemented",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "hitl_request_emitted",
    "generated_output_evaluation_implemented",
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


class SuccessPatternDiscoveryTests(unittest.TestCase):
    def test_plan_discovers_success_patterns_from_read_only_metadata(self) -> None:
        plan = discover_success_patterns(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "success_pattern_discovery")
        self.assertEqual(
            plan.serialization_version,
            "success_pattern_discovery_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_workflow_success_serialization_version,
            "workflow_success_tracking_plan.v1",
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
        self.assertEqual(plan.review_required_pattern_count, 2)
        self.assertEqual(plan.guarded_pattern_count, 2)
        self.assertEqual(plan.hitl_required_pattern_count, 4)
        self.assertFalse(plan.applied_success_pattern_ids)
        self.assertEqual(plan.overall_success_pattern_posture, "guarded")
        self.assertIn("does not observe runtime success", plan.authority_boundary)
        self.assertTrue(plan.success_pattern_discovery_implemented)
        self.assertTrue(plan.success_pattern_metadata_implemented)
        self.assertTrue(plan.workflow_success_metadata_used)
        self.assertTrue(plan.continuous_improvement_metadata_used)
        self.assertTrue(plan.adaptive_learning_metadata_used)
        self.assertFalse(plan.runtime_success_observation_implemented)
        self.assertFalse(plan.live_telemetry_collection_implemented)
        self.assertFalse(plan.success_metric_persistence_implemented)
        self.assertFalse(plan.success_pattern_application_implemented)
        self.assertFalse(plan.learning_feedback_application_implemented)
        self.assertFalse(plan.learning_memory_persistence_implemented)
        self.assertFalse(plan.learning_policy_update_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.generated_output_evaluation_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_patterns_score_success_discovery_without_application(self) -> None:
        plan = discover_success_patterns(route="generate")

        for pattern in plan.patterns:
            dumped = pattern.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PATTERN_FIELDS)
            self.assertEqual(pattern.serialization_version, "success_pattern.v1")
            self.assertEqual(pattern.route_name, RouteName.GENERATE)
            self.assertEqual(
                pattern.pattern_id,
                f"success_pattern::{pattern.pattern_kind}",
            )
            self.assertEqual(
                pattern.success_pattern_score,
                min(
                    1000,
                    max(
                        0,
                        pattern.workflow_success_score
                        + pattern.improvement_score // 2
                        + pattern.learning_priority_score // 4
                        + pattern.success_pattern_weight,
                    ),
                ),
            )
            self.assertIn(
                "runtime_success_observation",
                pattern.blocked_runtime_behaviors,
            )
            self.assertTrue(pattern.success_pattern_tags)
            self.assertTrue(pattern.advisory_actions)
            self.assertTrue(pattern.evidence)
            self.assertTrue(pattern.hitl_required)
            self.assertTrue(pattern.success_pattern_discovery_implemented)
            self.assertTrue(pattern.workflow_success_metadata_used)
            self.assertTrue(pattern.continuous_improvement_metadata_used)
            self.assertTrue(pattern.adaptive_learning_metadata_used)
            self.assertFalse(pattern.runtime_success_observation_implemented)
            self.assertFalse(pattern.live_telemetry_collection_implemented)
            self.assertFalse(pattern.success_metric_persistence_implemented)
            self.assertFalse(pattern.success_pattern_application_implemented)
            self.assertFalse(pattern.learning_feedback_application_implemented)
            self.assertFalse(pattern.learning_memory_persistence_implemented)
            self.assertFalse(pattern.learning_policy_update_implemented)
            self.assertFalse(pattern.provider_model_routing_implemented)
            self.assertFalse(pattern.provider_execution_implemented)
            self.assertFalse(pattern.agent_invocation_implemented)
            self.assertFalse(pattern.hitl_request_emitted)
            self.assertFalse(pattern.generated_output_evaluation_implemented)
            self.assertFalse(pattern.workflow_control_implemented)
            self.assertFalse(pattern.workflow_graph_mutation_implemented)
            self.assertFalse(pattern.workflow_execution_implemented)
            self.assertFalse(pattern.persistent_storage_write_implemented)
            self.assertFalse(pattern.generated_output_mutation_implemented)
            self.assertFalse(pattern.runtime_evolution_implemented)
            self.assertTrue(pattern.advisory_only)

        guarded = success_pattern_by_id(
            "success_pattern::routing_success_pattern",
            plan,
        )
        critical = success_patterns_for_priority("critical", plan)
        review = success_patterns_for_status("review_required", plan)
        self.assertIsNotNone(guarded)
        assert guarded is not None
        self.assertEqual(guarded.status, "guarded")
        self.assertEqual(guarded.priority, "guarded")
        self.assertEqual(len(critical), 2)
        self.assertEqual(len(review), 2)

    def test_plan_rejects_mismatched_success_pattern_metadata(self) -> None:
        plan = discover_success_patterns()
        payload = plan.model_dump(mode="json")
        payload["pattern_ids"] = ("missing",) + tuple(payload["pattern_ids"][1:])

        with self.assertRaisesRegex(ValueError, "pattern_ids must match"):
            SuccessPatternDiscoveryPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_success_pattern_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_success_pattern_score must match",
        ):
            SuccessPatternDiscoveryPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_success_pattern_ids"] = (plan.pattern_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_success_pattern_ids must remain empty",
        ):
            SuccessPatternDiscoveryPlan(**payload)

    def test_learning_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review success pattern discovery for a creative workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = discover_success_patterns(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_learning_does_not_declare_success_application_terms(self) -> None:
        plan = discover_success_patterns(route=RouteName.GENERATE)
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
                        pattern.source_success_indicator_id,
                        pattern.source_improvement_signal_id,
                        pattern.source_learning_signal_id,
                        pattern.source_workflow_risk_factor_id,
                        pattern.source_success_status,
                        pattern.source_improvement_status,
                        pattern.success_confidence_band,
                        *pattern.success_pattern_tags,
                        pattern.success_summary,
                        *pattern.advisory_actions,
                        *pattern.evidence,
                        *pattern.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "observe_runtime_success(",
            "collect_telemetry(",
            "persist_success_metric(",
            "apply_success_pattern(",
            "apply_feedback(",
            "persist_learning_memory(",
            "update_learning_policy(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "invoke_agent(",
            "emit_hitl_request(",
            "evaluate_generated_output(",
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
