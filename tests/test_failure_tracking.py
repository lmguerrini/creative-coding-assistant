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
    FailureTrackingPlan,
    failure_tracking_indicator_by_id,
    failure_tracking_indicators_for_severity,
    failure_tracking_indicators_for_status,
    route_request,
    track_failures,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_INDICATOR_FIELDS = {
    "indicator_id",
    "indicator_kind",
    "status",
    "severity",
    "route_name",
    "task_type",
    "execution_mode_id",
    "source_failure_panel_id",
    "source_learning_signal_id",
    "source_workflow_risk_factor_id",
    "failure_signal_count",
    "guardrail_signal_count",
    "workflow_risk_score",
    "learning_priority_score",
    "failure_tracking_weight",
    "failure_tracking_score",
    "hitl_required",
    "failure_pattern_tags",
    "failure_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "failure_tracking_implemented",
    "failure_indicator_metadata_implemented",
    "failure_analysis_metadata_used",
    "adaptive_learning_metadata_used",
    "runtime_failure_observation_implemented",
    "live_error_classification_implemented",
    "terminal_failure_routing_implemented",
    "failure_handling_implemented",
    "alert_emission_implemented",
    "hitl_request_emitted",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "resource_allocation_implemented",
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


class FailureTrackingTests(unittest.TestCase):
    def test_plan_derives_failure_indicators_from_read_only_sources(self) -> None:
        plan = track_failures(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "failure_tracking")
        self.assertEqual(plan.serialization_version, "failure_tracking_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_failure_analysis_serialization_version,
            "failure_analysis.v1",
        )
        self.assertEqual(
            plan.source_adaptive_learning_serialization_version,
            "adaptive_learning_plan.v1",
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.indicator_count, 6)
        self.assertEqual(plan.guarded_indicator_count, 6)
        self.assertEqual(plan.hitl_required_indicator_count, 6)
        self.assertEqual(plan.total_failure_signal_count, 221)
        self.assertEqual(plan.total_guardrail_signal_count, 70)
        self.assertFalse(plan.observed_failure_indicator_ids)
        self.assertFalse(plan.handled_failure_indicator_ids)
        self.assertEqual(plan.overall_failure_tracking_posture, "guarded")
        self.assertIn("does not observe runtime failures", plan.authority_boundary)
        self.assertTrue(plan.failure_tracking_implemented)
        self.assertTrue(plan.failure_indicator_metadata_implemented)
        self.assertTrue(plan.failure_analysis_metadata_used)
        self.assertTrue(plan.adaptive_learning_metadata_used)
        self.assertFalse(plan.runtime_failure_observation_implemented)
        self.assertFalse(plan.live_error_classification_implemented)
        self.assertFalse(plan.terminal_failure_routing_implemented)
        self.assertFalse(plan.failure_handling_implemented)
        self.assertFalse(plan.alert_emission_implemented)
        self.assertFalse(plan.hitl_request_emitted)
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

    def test_indicators_score_failure_tracking_without_handling(self) -> None:
        plan = track_failures(route="generate")

        for indicator in plan.indicators:
            dumped = indicator.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_INDICATOR_FIELDS)
            self.assertEqual(
                indicator.serialization_version,
                "failure_tracking_indicator.v1",
            )
            self.assertEqual(indicator.route_name, RouteName.GENERATE)
            self.assertEqual(
                indicator.indicator_id,
                f"failure_tracking::{indicator.indicator_kind}",
            )
            self.assertEqual(
                indicator.failure_tracking_score,
                min(
                    1000,
                    max(
                        0,
                        indicator.failure_signal_count * 5
                        + indicator.guardrail_signal_count * 18
                        + indicator.workflow_risk_score // 3
                        + indicator.learning_priority_score // 4
                        + indicator.failure_tracking_weight,
                    ),
                ),
            )
            self.assertIn(
                "runtime_failure_observation",
                indicator.blocked_runtime_behaviors,
            )
            self.assertTrue(indicator.failure_pattern_tags)
            self.assertTrue(indicator.advisory_actions)
            self.assertTrue(indicator.evidence)
            self.assertTrue(indicator.hitl_required)
            self.assertTrue(indicator.failure_tracking_implemented)
            self.assertTrue(indicator.failure_analysis_metadata_used)
            self.assertTrue(indicator.adaptive_learning_metadata_used)
            self.assertFalse(indicator.runtime_failure_observation_implemented)
            self.assertFalse(indicator.live_error_classification_implemented)
            self.assertFalse(indicator.terminal_failure_routing_implemented)
            self.assertFalse(indicator.failure_handling_implemented)
            self.assertFalse(indicator.alert_emission_implemented)
            self.assertFalse(indicator.hitl_request_emitted)
            self.assertFalse(indicator.provider_model_routing_implemented)
            self.assertFalse(indicator.provider_execution_implemented)
            self.assertFalse(indicator.workflow_control_implemented)
            self.assertFalse(indicator.workflow_graph_mutation_implemented)
            self.assertFalse(indicator.workflow_execution_implemented)
            self.assertFalse(indicator.persistent_storage_write_implemented)
            self.assertFalse(indicator.generated_output_mutation_implemented)
            self.assertFalse(indicator.runtime_evolution_implemented)
            self.assertTrue(indicator.advisory_only)

        routing = failure_tracking_indicator_by_id(
            "failure_tracking::routing_failure_tracking",
            plan,
        )
        guarded = failure_tracking_indicators_for_severity("guarded", plan)
        guarded_status = failure_tracking_indicators_for_status("guarded", plan)
        self.assertIsNotNone(routing)
        assert routing is not None
        self.assertEqual(routing.status, "guarded")
        self.assertEqual(routing.severity, "guarded")
        self.assertEqual(len(guarded), 6)
        self.assertEqual(len(guarded_status), 6)

    def test_plan_rejects_mismatched_failure_tracking_metadata(self) -> None:
        plan = track_failures()
        payload = plan.model_dump(mode="json")
        payload["indicator_ids"] = ("missing",) + tuple(payload["indicator_ids"][1:])

        with self.assertRaisesRegex(ValueError, "indicator_ids must match"):
            FailureTrackingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_failure_tracking_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_failure_tracking_score must match",
        ):
            FailureTrackingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["observed_failure_indicator_ids"] = (plan.indicator_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "observed_failure_indicator_ids must remain empty",
        ):
            FailureTrackingPlan(**payload)

    def test_tracking_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Derive failure tracking indicators for a guarded workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = track_failures(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_tracking_does_not_declare_runtime_application_terms(self) -> None:
        plan = track_failures(route=RouteName.GENERATE)
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
                        indicator.source_failure_panel_id,
                        indicator.source_learning_signal_id,
                        indicator.source_workflow_risk_factor_id,
                        *indicator.failure_pattern_tags,
                        indicator.failure_summary,
                        *indicator.advisory_actions,
                        *indicator.evidence,
                        *indicator.blocked_runtime_behaviors,
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
            "emit_alert(",
            "emit_hitl_request(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "invoke_agent(",
            "allocate_resource(",
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
