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
    AdaptiveLearningPlan,
    adaptive_learning_signal_by_id,
    adaptive_learning_signals_for_priority,
    adaptive_learning_signals_for_status,
    evaluate_adaptive_learning_engine,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_SIGNAL_FIELDS = {
    "signal_id",
    "signal_kind",
    "status",
    "priority_band",
    "route_name",
    "task_type",
    "execution_mode_id",
    "source_workflow_risk_factor_id",
    "source_execution_confidence_signal_id",
    "source_self_tuning_policy_id",
    "workflow_risk_score",
    "execution_confidence_score",
    "self_tuning_score",
    "unavailable_reason_count",
    "guardrail_signal_count",
    "learning_weight",
    "learning_priority_score",
    "hitl_required",
    "pattern_tags",
    "learning_summary",
    "proposed_learning_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "adaptive_learning_engine_implemented",
    "learning_signal_metadata_implemented",
    "execution_confidence_metadata_used",
    "workflow_risk_metadata_used",
    "workflow_self_tuning_metadata_used",
    "learning_memory_persistence_implemented",
    "learning_feedback_application_implemented",
    "learning_policy_mutation_implemented",
    "strategy_mutation_implemented",
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


class AdaptiveLearningEngineTests(unittest.TestCase):
    def test_plan_combines_v5_learning_sources(self) -> None:
        plan = evaluate_adaptive_learning_engine(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "adaptive_learning_engine")
        self.assertEqual(plan.serialization_version, "adaptive_learning_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_execution_confidence_serialization_version,
            "execution_confidence_plan.v1",
        )
        self.assertEqual(
            plan.source_workflow_risk_serialization_version,
            "workflow_risk_plan.v1",
        )
        self.assertEqual(
            plan.source_workflow_self_tuning_serialization_version,
            "workflow_self_tuning_policy_plan.v1",
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.signal_count, 5)
        self.assertEqual(plan.review_required_signal_count, 3)
        self.assertEqual(plan.guardrail_signal_count, 2)
        self.assertEqual(plan.hitl_required_signal_count, 5)
        self.assertFalse(plan.persisted_learning_signal_ids)
        self.assertFalse(plan.applied_learning_signal_ids)
        self.assertEqual(plan.overall_learning_posture, "guarded")
        self.assertIn("does not persist learning memory", plan.authority_boundary)
        self.assertTrue(plan.adaptive_learning_engine_implemented)
        self.assertTrue(plan.learning_signal_metadata_implemented)
        self.assertTrue(plan.execution_confidence_metadata_used)
        self.assertTrue(plan.workflow_risk_metadata_used)
        self.assertTrue(plan.workflow_self_tuning_metadata_used)
        self.assertFalse(plan.learning_memory_persistence_implemented)
        self.assertFalse(plan.learning_feedback_application_implemented)
        self.assertFalse(plan.learning_policy_mutation_implemented)
        self.assertFalse(plan.strategy_mutation_implemented)
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
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_signals_score_learning_priority_without_application(self) -> None:
        plan = evaluate_adaptive_learning_engine(route="generate")

        for signal in plan.signals:
            dumped = signal.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SIGNAL_FIELDS)
            self.assertEqual(
                signal.serialization_version,
                "adaptive_learning_signal.v1",
            )
            self.assertEqual(signal.route_name, RouteName.GENERATE)
            self.assertEqual(
                signal.signal_id,
                f"adaptive_learning::{signal.signal_kind}",
            )
            self.assertEqual(
                signal.learning_priority_score,
                min(
                    1000,
                    max(
                        0,
                        signal.workflow_risk_score // 2
                        + max(0, 100 - signal.execution_confidence_score) * 4
                        + signal.self_tuning_score // 2
                        + signal.unavailable_reason_count * 35
                        + signal.guardrail_signal_count * 75
                        + signal.learning_weight,
                    ),
                ),
            )
            self.assertIn(
                "learning_memory_persistence",
                signal.blocked_runtime_behaviors,
            )
            self.assertTrue(signal.pattern_tags)
            self.assertTrue(signal.proposed_learning_actions)
            self.assertTrue(signal.evidence)
            self.assertTrue(signal.hitl_required)
            self.assertTrue(signal.adaptive_learning_engine_implemented)
            self.assertTrue(signal.execution_confidence_metadata_used)
            self.assertTrue(signal.workflow_risk_metadata_used)
            self.assertTrue(signal.workflow_self_tuning_metadata_used)
            self.assertFalse(signal.learning_memory_persistence_implemented)
            self.assertFalse(signal.learning_feedback_application_implemented)
            self.assertFalse(signal.learning_policy_mutation_implemented)
            self.assertFalse(signal.strategy_mutation_implemented)
            self.assertFalse(signal.provider_model_routing_implemented)
            self.assertFalse(signal.provider_execution_implemented)
            self.assertFalse(signal.workflow_control_implemented)
            self.assertFalse(signal.workflow_graph_mutation_implemented)
            self.assertFalse(signal.workflow_execution_implemented)
            self.assertFalse(signal.persistent_storage_write_implemented)
            self.assertFalse(signal.generated_output_mutation_implemented)
            self.assertFalse(signal.runtime_evolution_implemented)
            self.assertTrue(signal.advisory_only)

        routing = adaptive_learning_signal_by_id(
            "adaptive_learning::routing_boundary_learning",
            plan,
        )
        guarded = adaptive_learning_signals_for_priority("guarded", plan)
        review = adaptive_learning_signals_for_status("review_required", plan)
        self.assertIsNotNone(routing)
        assert routing is not None
        self.assertEqual(routing.status, "guardrail")
        self.assertEqual(routing.priority_band, "guarded")
        self.assertEqual(len(guarded), 2)
        self.assertEqual(len(review), 3)

    def test_plan_rejects_mismatched_learning_metadata(self) -> None:
        plan = evaluate_adaptive_learning_engine()
        payload = plan.model_dump(mode="json")
        payload["signal_ids"] = ("missing",) + tuple(payload["signal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "signal_ids must match"):
            AdaptiveLearningPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_learning_priority_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_learning_priority_score must match",
        ):
            AdaptiveLearningPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_learning_signal_ids"] = (plan.signal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_learning_signal_ids must remain empty",
        ):
            AdaptiveLearningPlan(**payload)

    def test_engine_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review learning signals for a controlled adaptive workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = evaluate_adaptive_learning_engine(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_engine_does_not_declare_runtime_application_terms(self) -> None:
        plan = evaluate_adaptive_learning_engine(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for signal in plan.signals
                    for field in (
                        signal.signal_id,
                        signal.signal_kind,
                        signal.source_workflow_risk_factor_id,
                        signal.source_execution_confidence_signal_id,
                        signal.source_self_tuning_policy_id,
                        *signal.pattern_tags,
                        signal.learning_summary,
                        *signal.proposed_learning_actions,
                        *signal.evidence,
                        *signal.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "persist_learning(",
            "apply_feedback(",
            "update_policy(",
            "mutate_strategy(",
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
