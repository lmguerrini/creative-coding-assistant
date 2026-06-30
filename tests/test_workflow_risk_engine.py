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
    WorkflowRiskPlan,
    evaluate_workflow_risk,
    route_request,
    workflow_risk_factor_by_id,
    workflow_risk_factors_for_severity,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_PROVIDER_IDS = ("openai", "anthropic", "gemini", "local")
REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_HYBRID_DIRECTIONS = (
    "local_to_cloud",
    "cloud_to_local",
    "cloud_to_cloud",
    "local_to_local",
)
REQUIRED_FACTOR_FIELDS = {
    "factor_id",
    "factor_kind",
    "status",
    "severity",
    "route_name",
    "task_type",
    "execution_mode_id",
    "source_execution_confidence_signal_id",
    "source_escalation_decision_id",
    "source_dynamic_resource_allocation_id",
    "source_self_tuning_policy_id",
    "source_performance_regression_signal_id",
    "provider_sequence",
    "model_profile_sequence",
    "hybrid_policy_direction",
    "unavailable_reason_codes",
    "execution_confidence_score",
    "escalation_score",
    "dynamic_resource_score",
    "self_tuning_score",
    "performance_regression_score",
    "unavailable_reason_count",
    "guardrail_signal_count",
    "risk_weight",
    "workflow_risk_score",
    "hitl_required",
    "mitigation_summary",
    "fallback_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "workflow_risk_engine_implemented",
    "advisory_workflow_risk_metadata_implemented",
    "execution_confidence_metadata_used",
    "escalation_metadata_used",
    "dynamic_resource_allocation_metadata_used",
    "workflow_self_tuning_metadata_used",
    "performance_regression_metadata_used",
    "provider_intelligence_metadata_used",
    "availability_awareness_metadata_used",
    "manual_assisted_auto_mode_metadata_used",
    "hybrid_transition_metadata_used",
    "task_aware_category_metadata_used",
    "execution_simulation_metadata_used",
    "fallback_safety_metadata_used",
    "risk_decision_application_implemented",
    "risk_mitigation_execution_implemented",
    "workflow_blocking_implemented",
    "threshold_enforcement_implemented",
    "alert_emission_implemented",
    "escalation_triggering_implemented",
    "human_review_request_implemented",
    "hitl_request_emitted",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "provider_execution_implemented",
    "local_runtime_probe_implemented",
    "local_model_inventory_scan_implemented",
    "automatic_model_download_implemented",
    "agent_invocation_implemented",
    "resource_allocation_implemented",
    "self_tuning_application_implemented",
    "budget_enforcement_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "runtime_regression_detection_implemented",
    "benchmark_execution_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class WorkflowRiskEngineTests(unittest.TestCase):
    def test_plan_combines_workflow_risk_sources(self) -> None:
        plan = evaluate_workflow_risk(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "workflow_risk_engine")
        self.assertEqual(plan.serialization_version, "workflow_risk_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_execution_confidence_serialization_version,
            "execution_confidence_plan.v1",
        )
        self.assertEqual(
            plan.source_escalation_optimization_serialization_version,
            "adaptive_escalation_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_dynamic_resource_allocation_serialization_version,
            "dynamic_resource_allocation_plan.v1",
        )
        self.assertEqual(
            plan.source_workflow_self_tuning_serialization_version,
            "workflow_self_tuning_policy_plan.v1",
        )
        self.assertEqual(
            plan.source_performance_regression_serialization_version,
            "performance_regression_detection_plan.v1",
        )
        self.assertEqual(plan.provider_ids, REQUIRED_PROVIDER_IDS)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.hybrid_policy_directions, REQUIRED_HYBRID_DIRECTIONS)
        self.assertEqual(plan.factor_count, 6)
        self.assertEqual(plan.review_required_factor_count, 4)
        self.assertEqual(plan.guardrail_factor_count, 2)
        self.assertEqual(plan.hitl_required_factor_count, 6)
        self.assertFalse(plan.applied_mitigation_ids)
        self.assertEqual(plan.overall_workflow_risk_severity, "guarded")
        self.assertIn("does not apply risk decisions", plan.authority_boundary)
        self.assertTrue(plan.workflow_risk_engine_implemented)
        self.assertTrue(plan.advisory_workflow_risk_metadata_implemented)
        self.assertTrue(plan.execution_confidence_metadata_used)
        self.assertTrue(plan.escalation_metadata_used)
        self.assertTrue(plan.dynamic_resource_allocation_metadata_used)
        self.assertTrue(plan.workflow_self_tuning_metadata_used)
        self.assertTrue(plan.performance_regression_metadata_used)
        self.assertTrue(plan.provider_intelligence_metadata_used)
        self.assertTrue(plan.availability_awareness_metadata_used)
        self.assertTrue(plan.manual_assisted_auto_mode_metadata_used)
        self.assertTrue(plan.hybrid_transition_metadata_used)
        self.assertTrue(plan.task_aware_category_metadata_used)
        self.assertTrue(plan.execution_simulation_metadata_used)
        self.assertTrue(plan.fallback_safety_metadata_used)
        self.assertFalse(plan.risk_decision_application_implemented)
        self.assertFalse(plan.risk_mitigation_execution_implemented)
        self.assertFalse(plan.workflow_blocking_implemented)
        self.assertFalse(plan.threshold_enforcement_implemented)
        self.assertFalse(plan.alert_emission_implemented)
        self.assertFalse(plan.escalation_triggering_implemented)
        self.assertFalse(plan.human_review_request_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.automatic_provider_switching_implemented)
        self.assertFalse(plan.automatic_model_switching_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.local_runtime_probe_implemented)
        self.assertFalse(plan.local_model_inventory_scan_implemented)
        self.assertFalse(plan.automatic_model_download_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.resource_allocation_implemented)
        self.assertFalse(plan.self_tuning_application_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.runtime_regression_detection_implemented)
        self.assertFalse(plan.benchmark_execution_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_factors_score_workflow_risk_without_mitigation(self) -> None:
        plan = evaluate_workflow_risk(route="generate")

        for factor in plan.factors:
            dumped = factor.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_FACTOR_FIELDS)
            self.assertEqual(factor.serialization_version, "workflow_risk_factor.v1")
            self.assertEqual(factor.route_name, RouteName.GENERATE)
            self.assertEqual(
                factor.factor_id,
                f"workflow_risk::{factor.factor_kind}",
            )
            self.assertEqual(
                factor.workflow_risk_score,
                min(
                    1000,
                    max(
                        0,
                        (100 - factor.execution_confidence_score) * 4
                        + factor.escalation_score * 2
                        + max(0, 500 - factor.dynamic_resource_score) // 2
                        + factor.self_tuning_score // 3
                        + factor.performance_regression_score // 10
                        + factor.unavailable_reason_count * 35
                        + factor.guardrail_signal_count * 20
                        + factor.risk_weight,
                    ),
                ),
            )
            self.assertEqual(
                factor.unavailable_reason_count,
                len(factor.unavailable_reason_codes),
            )
            self.assertIn("risk_mitigation_execution", factor.blocked_runtime_behaviors)
            self.assertTrue(factor.workflow_risk_engine_implemented)
            self.assertTrue(factor.execution_confidence_metadata_used)
            self.assertTrue(factor.escalation_metadata_used)
            self.assertTrue(factor.dynamic_resource_allocation_metadata_used)
            self.assertTrue(factor.workflow_self_tuning_metadata_used)
            self.assertTrue(factor.performance_regression_metadata_used)
            self.assertTrue(factor.provider_intelligence_metadata_used)
            self.assertTrue(factor.availability_awareness_metadata_used)
            self.assertTrue(factor.manual_assisted_auto_mode_metadata_used)
            self.assertTrue(factor.hybrid_transition_metadata_used)
            self.assertTrue(factor.task_aware_category_metadata_used)
            self.assertTrue(factor.execution_simulation_metadata_used)
            self.assertTrue(factor.fallback_safety_metadata_used)
            self.assertFalse(factor.risk_decision_application_implemented)
            self.assertFalse(factor.risk_mitigation_execution_implemented)
            self.assertFalse(factor.workflow_blocking_implemented)
            self.assertFalse(factor.threshold_enforcement_implemented)
            self.assertFalse(factor.alert_emission_implemented)
            self.assertFalse(factor.escalation_triggering_implemented)
            self.assertFalse(factor.human_review_request_implemented)
            self.assertFalse(factor.hitl_request_emitted)
            self.assertFalse(factor.provider_model_routing_implemented)
            self.assertFalse(factor.automatic_provider_switching_implemented)
            self.assertFalse(factor.automatic_model_switching_implemented)
            self.assertFalse(factor.provider_execution_implemented)
            self.assertFalse(factor.local_runtime_probe_implemented)
            self.assertFalse(factor.local_model_inventory_scan_implemented)
            self.assertFalse(factor.automatic_model_download_implemented)
            self.assertFalse(factor.agent_invocation_implemented)
            self.assertFalse(factor.resource_allocation_implemented)
            self.assertFalse(factor.self_tuning_application_implemented)
            self.assertFalse(factor.workflow_control_implemented)
            self.assertFalse(factor.workflow_graph_mutation_implemented)
            self.assertFalse(factor.workflow_execution_implemented)
            self.assertFalse(factor.runtime_regression_detection_implemented)
            self.assertFalse(factor.benchmark_execution_implemented)
            self.assertFalse(factor.retry_triggering_implemented)
            self.assertFalse(factor.generated_output_mutation_implemented)
            self.assertTrue(factor.hitl_required)
            self.assertTrue(factor.advisory_only)

        provider = workflow_risk_factor_by_id(
            "workflow_risk::provider_fallback_risk",
            plan,
        )
        guarded = workflow_risk_factors_for_severity("guarded", plan)
        self.assertIsNotNone(provider)
        assert provider is not None
        self.assertEqual(provider.status, "guardrail")
        self.assertEqual(provider.severity, "guarded")
        self.assertIn("missing_api_key", provider.unavailable_reason_codes)
        self.assertEqual(provider.provider_sequence, ("local", "openai"))
        self.assertEqual(len(guarded), 2)

    def test_plan_rejects_mismatched_risk_metadata(self) -> None:
        plan = evaluate_workflow_risk()
        payload = plan.model_dump(mode="json")
        payload["factor_ids"] = ("missing",) + tuple(payload["factor_ids"][1:])

        with self.assertRaisesRegex(ValueError, "factor_ids must match"):
            WorkflowRiskPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["highest_workflow_risk_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "highest_workflow_risk_score must match",
        ):
            WorkflowRiskPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_mitigation_ids"] = (plan.factor_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_mitigation_ids must remain empty",
        ):
            WorkflowRiskPlan(**payload)

    def test_engine_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Evaluate workflow risk for a hybrid WebGL generation flow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = evaluate_workflow_risk(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_engine_does_not_declare_runtime_application_terms(self) -> None:
        plan = evaluate_workflow_risk(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for factor in plan.factors
                    for field in (
                        factor.factor_id,
                        factor.factor_kind,
                        factor.source_execution_confidence_signal_id,
                        factor.source_escalation_decision_id,
                        factor.source_dynamic_resource_allocation_id,
                        factor.source_self_tuning_policy_id,
                        factor.source_performance_regression_signal_id,
                        *factor.provider_sequence,
                        *factor.model_profile_sequence,
                        factor.mitigation_summary,
                        factor.fallback_summary,
                        *factor.advisory_actions,
                        *factor.evidence,
                        *factor.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "apply_risk_decision(",
            "execute_mitigation(",
            "block_workflow(",
            "enforce_threshold(",
            "emit_alert(",
            "trigger_escalation(",
            "emit_hitl_request(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "probe_local_runtime(",
            "scan_local_models(",
            "download_model(",
            "invoke_agent(",
            "allocate_resource(",
            "apply_self_tuning(",
            "enforce_budget(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
            "detect_live_regression(",
            "execute_benchmark(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
