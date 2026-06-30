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
    WorkflowSuccessTrackingPlan,
    route_request,
    track_workflow_success,
    workflow_success_indicator_by_id,
    workflow_success_indicators_for_confidence,
    workflow_success_indicators_for_status,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_INDICATOR_FIELDS = {
    "indicator_id",
    "indicator_kind",
    "status",
    "confidence_band",
    "route_name",
    "task_type",
    "execution_mode_id",
    "source_learning_signal_id",
    "source_workflow_risk_factor_id",
    "source_execution_confidence_signal_id",
    "source_self_tuning_policy_id",
    "workflow_risk_score",
    "execution_confidence_score",
    "self_tuning_score",
    "learning_priority_score",
    "unavailable_reason_count",
    "guardrail_signal_count",
    "success_weight",
    "workflow_success_score",
    "hitl_required",
    "success_pattern_tags",
    "success_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "workflow_success_tracking_implemented",
    "success_indicator_metadata_implemented",
    "adaptive_learning_metadata_used",
    "runtime_success_observation_implemented",
    "live_telemetry_collection_implemented",
    "generated_output_evaluation_implemented",
    "success_metric_persistence_implemented",
    "learning_feedback_application_implemented",
    "learning_policy_mutation_implemented",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "resource_allocation_implemented",
    "hitl_request_emitted",
    "budget_enforcement_implemented",
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


class WorkflowSuccessTrackingTests(unittest.TestCase):
    def test_plan_derives_success_indicators_from_learning_metadata(self) -> None:
        plan = track_workflow_success(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "workflow_success_tracking")
        self.assertEqual(
            plan.serialization_version,
            "workflow_success_tracking_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_adaptive_learning_serialization_version,
            "adaptive_learning_plan.v1",
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.indicator_count, 5)
        self.assertEqual(plan.review_required_indicator_count, 3)
        self.assertEqual(plan.guarded_indicator_count, 2)
        self.assertEqual(plan.hitl_required_indicator_count, 5)
        self.assertFalse(plan.persisted_success_indicator_ids)
        self.assertFalse(plan.applied_success_indicator_ids)
        self.assertEqual(plan.overall_success_posture, "guarded")
        self.assertIn(
            "does not observe live workflow outcomes",
            plan.authority_boundary,
        )
        self.assertTrue(plan.workflow_success_tracking_implemented)
        self.assertTrue(plan.success_indicator_metadata_implemented)
        self.assertTrue(plan.adaptive_learning_metadata_used)
        self.assertFalse(plan.runtime_success_observation_implemented)
        self.assertFalse(plan.live_telemetry_collection_implemented)
        self.assertFalse(plan.generated_output_evaluation_implemented)
        self.assertFalse(plan.success_metric_persistence_implemented)
        self.assertFalse(plan.learning_feedback_application_implemented)
        self.assertFalse(plan.learning_policy_mutation_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_indicators_score_success_without_runtime_tracking(self) -> None:
        plan = track_workflow_success(route="generate")

        for indicator in plan.indicators:
            dumped = indicator.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_INDICATOR_FIELDS)
            self.assertEqual(
                indicator.serialization_version,
                "workflow_success_indicator.v1",
            )
            self.assertEqual(indicator.route_name, RouteName.GENERATE)
            self.assertEqual(
                indicator.indicator_id,
                f"workflow_success::{indicator.indicator_kind}",
            )
            self.assertEqual(
                indicator.workflow_success_score,
                min(
                    1000,
                    max(
                        0,
                        indicator.execution_confidence_score * 5
                        + indicator.self_tuning_score // 2
                        + max(0, 1000 - indicator.workflow_risk_score) // 2
                        + indicator.success_weight
                        - indicator.learning_priority_score // 5
                        - indicator.unavailable_reason_count * 30
                        - indicator.guardrail_signal_count * 50,
                    ),
                ),
            )
            self.assertIn(
                "runtime_success_observation",
                indicator.blocked_runtime_behaviors,
            )
            self.assertTrue(indicator.success_pattern_tags)
            self.assertTrue(indicator.advisory_actions)
            self.assertTrue(indicator.evidence)
            self.assertTrue(indicator.hitl_required)
            self.assertTrue(indicator.workflow_success_tracking_implemented)
            self.assertTrue(indicator.adaptive_learning_metadata_used)
            self.assertFalse(indicator.runtime_success_observation_implemented)
            self.assertFalse(indicator.live_telemetry_collection_implemented)
            self.assertFalse(indicator.generated_output_evaluation_implemented)
            self.assertFalse(indicator.success_metric_persistence_implemented)
            self.assertFalse(indicator.learning_feedback_application_implemented)
            self.assertFalse(indicator.learning_policy_mutation_implemented)
            self.assertFalse(indicator.provider_model_routing_implemented)
            self.assertFalse(indicator.provider_execution_implemented)
            self.assertFalse(indicator.workflow_control_implemented)
            self.assertFalse(indicator.workflow_graph_mutation_implemented)
            self.assertFalse(indicator.workflow_execution_implemented)
            self.assertFalse(indicator.persistent_storage_write_implemented)
            self.assertFalse(indicator.generated_output_mutation_implemented)
            self.assertFalse(indicator.runtime_evolution_implemented)
            self.assertTrue(indicator.advisory_only)

        routing = workflow_success_indicator_by_id(
            "workflow_success::routing_safety_success",
            plan,
        )
        guarded = workflow_success_indicators_for_confidence("guarded", plan)
        review = workflow_success_indicators_for_status("review_required", plan)
        self.assertIsNotNone(routing)
        assert routing is not None
        self.assertEqual(routing.status, "guarded")
        self.assertEqual(routing.confidence_band, "guarded")
        self.assertEqual(len(guarded), 2)
        self.assertEqual(len(review), 3)

    def test_plan_rejects_mismatched_success_metadata(self) -> None:
        plan = track_workflow_success()
        payload = plan.model_dump(mode="json")
        payload["indicator_ids"] = ("missing",) + tuple(payload["indicator_ids"][1:])

        with self.assertRaisesRegex(ValueError, "indicator_ids must match"):
            WorkflowSuccessTrackingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_workflow_success_score"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_workflow_success_score must match",
        ):
            WorkflowSuccessTrackingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_success_indicator_ids"] = (plan.indicator_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_success_indicator_ids must remain empty",
        ):
            WorkflowSuccessTrackingPlan(**payload)

    def test_tracking_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Derive workflow success indicators for a coding workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = track_workflow_success(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_tracking_does_not_declare_runtime_application_terms(self) -> None:
        plan = track_workflow_success(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for indicator in plan.indicators
                    for field in (
                        indicator.indicator_id,
                        indicator.indicator_kind,
                        indicator.source_learning_signal_id,
                        indicator.source_workflow_risk_factor_id,
                        indicator.source_execution_confidence_signal_id,
                        indicator.source_self_tuning_policy_id,
                        *indicator.success_pattern_tags,
                        indicator.success_summary,
                        *indicator.advisory_actions,
                        *indicator.evidence,
                        *indicator.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "observe_runtime_success(",
            "collect_telemetry(",
            "evaluate_output(",
            "persist_success_metric(",
            "apply_learning(",
            "update_policy(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "invoke_agent(",
            "allocate_resource(",
            "emit_hitl_request(",
            "enforce_budget(",
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
