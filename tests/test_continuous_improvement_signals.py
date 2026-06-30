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
    ContinuousImprovementSignalPlan,
    continuous_improvement_signal_by_id,
    continuous_improvement_signals_for_priority,
    continuous_improvement_signals_for_status,
    derive_continuous_improvement_signals,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_SOURCE_PLAN_ROLES = (
    "workflow_success_tracking",
    "failure_tracking",
    "artifact_learning",
    "evaluation_learning",
)
REQUIRED_SIGNAL_FIELDS = {
    "signal_id",
    "signal_kind",
    "status",
    "priority",
    "route_name",
    "task_type",
    "execution_mode_id",
    "source_plan_role",
    "source_learning_signal_id",
    "source_workflow_risk_factor_id",
    "source_score",
    "source_item_count",
    "source_review_required_count",
    "source_guarded_count",
    "source_hitl_required_count",
    "learning_priority_score",
    "improvement_weight",
    "improvement_score",
    "hitl_required",
    "improvement_tags",
    "improvement_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "continuous_improvement_signals_implemented",
    "improvement_signal_metadata_implemented",
    "adaptive_learning_metadata_used",
    "source_learning_metadata_used",
    "learning_feedback_application_implemented",
    "learning_memory_persistence_implemented",
    "learning_policy_update_implemented",
    "strategy_mutation_implemented",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "resource_allocation_implemented",
    "hitl_request_emitted",
    "runtime_outcome_observation_implemented",
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


class ContinuousImprovementSignalsTests(unittest.TestCase):
    def test_plan_derives_improvement_signals_from_learning_metadata(self) -> None:
        plan = derive_continuous_improvement_signals(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "continuous_improvement_signals")
        self.assertEqual(
            plan.serialization_version,
            "continuous_improvement_signal_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_adaptive_learning_serialization_version,
            "adaptive_learning_plan.v1",
        )
        self.assertEqual(plan.source_plan_roles, REQUIRED_SOURCE_PLAN_ROLES)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.signal_count, 4)
        self.assertEqual(plan.review_required_signal_count, 2)
        self.assertEqual(plan.guarded_signal_count, 2)
        self.assertEqual(plan.hitl_required_signal_count, 4)
        self.assertFalse(plan.applied_improvement_signal_ids)
        self.assertEqual(plan.overall_improvement_posture, "guarded")
        self.assertIn("do not apply feedback", plan.authority_boundary)
        self.assertTrue(plan.continuous_improvement_signals_implemented)
        self.assertTrue(plan.improvement_signal_metadata_implemented)
        self.assertTrue(plan.adaptive_learning_metadata_used)
        self.assertTrue(plan.source_learning_metadata_used)
        self.assertFalse(plan.learning_feedback_application_implemented)
        self.assertFalse(plan.learning_memory_persistence_implemented)
        self.assertFalse(plan.learning_policy_update_implemented)
        self.assertFalse(plan.strategy_mutation_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.resource_allocation_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.runtime_outcome_observation_implemented)
        self.assertFalse(plan.generated_output_evaluation_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_signals_score_improvement_without_applying_feedback(self) -> None:
        plan = derive_continuous_improvement_signals(route="generate")

        for signal in plan.signals:
            dumped = signal.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SIGNAL_FIELDS)
            self.assertEqual(
                signal.serialization_version,
                "continuous_improvement_signal.v1",
            )
            self.assertEqual(signal.route_name, RouteName.GENERATE)
            self.assertEqual(
                signal.signal_id,
                f"continuous_improvement::{signal.signal_kind}",
            )
            self.assertEqual(
                signal.improvement_score,
                min(
                    1000,
                    max(
                        0,
                        signal.source_score
                        + signal.learning_priority_score // 3
                        + signal.improvement_weight
                        + signal.source_review_required_count * 40
                        + signal.source_guarded_count * 50
                        + signal.source_hitl_required_count * 20
                        - signal.source_item_count * 5,
                    ),
                ),
            )
            self.assertIn(
                "learning_feedback_application",
                signal.blocked_runtime_behaviors,
            )
            self.assertIn(signal.source_plan_role, REQUIRED_SOURCE_PLAN_ROLES)
            self.assertTrue(signal.improvement_tags)
            self.assertTrue(signal.advisory_actions)
            self.assertTrue(signal.evidence)
            self.assertTrue(signal.hitl_required)
            self.assertTrue(signal.continuous_improvement_signals_implemented)
            self.assertTrue(signal.source_learning_metadata_used)
            self.assertFalse(signal.learning_feedback_application_implemented)
            self.assertFalse(signal.learning_memory_persistence_implemented)
            self.assertFalse(signal.learning_policy_update_implemented)
            self.assertFalse(signal.strategy_mutation_implemented)
            self.assertFalse(signal.provider_model_routing_implemented)
            self.assertFalse(signal.provider_execution_implemented)
            self.assertFalse(signal.agent_invocation_implemented)
            self.assertFalse(signal.resource_allocation_implemented)
            self.assertFalse(signal.hitl_request_emitted)
            self.assertFalse(signal.runtime_outcome_observation_implemented)
            self.assertFalse(signal.generated_output_evaluation_implemented)
            self.assertFalse(signal.workflow_control_implemented)
            self.assertFalse(signal.workflow_graph_mutation_implemented)
            self.assertFalse(signal.workflow_execution_implemented)
            self.assertFalse(signal.persistent_storage_write_implemented)
            self.assertFalse(signal.generated_output_mutation_implemented)
            self.assertFalse(signal.runtime_evolution_implemented)
            self.assertTrue(signal.advisory_only)

        guarded = continuous_improvement_signal_by_id(
            "continuous_improvement::failure_prevention_signal",
            plan,
        )
        critical = continuous_improvement_signals_for_priority("critical", plan)
        review = continuous_improvement_signals_for_status("review_required", plan)
        self.assertIsNotNone(guarded)
        assert guarded is not None
        self.assertEqual(guarded.status, "guarded")
        self.assertEqual(guarded.priority, "guarded")
        self.assertEqual(len(critical), 2)
        self.assertEqual(len(review), 2)

    def test_plan_rejects_mismatched_improvement_metadata(self) -> None:
        plan = derive_continuous_improvement_signals()
        payload = plan.model_dump(mode="json")
        payload["signal_ids"] = ("missing",) + tuple(payload["signal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "signal_ids must match"):
            ContinuousImprovementSignalPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_improvement_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_improvement_score must match",
        ):
            ContinuousImprovementSignalPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_improvement_signal_ids"] = (plan.signal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_improvement_signal_ids must remain empty",
        ):
            ContinuousImprovementSignalPlan(**payload)

    def test_learning_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review continuous improvement signals for a creative workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = derive_continuous_improvement_signals(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_learning_does_not_declare_feedback_application_terms(self) -> None:
        plan = derive_continuous_improvement_signals(route=RouteName.GENERATE)
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
                        signal.source_plan_role,
                        signal.source_learning_signal_id,
                        signal.source_workflow_risk_factor_id,
                        *signal.improvement_tags,
                        signal.improvement_summary,
                        *signal.advisory_actions,
                        *signal.evidence,
                        *signal.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "apply_feedback(",
            "persist_learning_memory(",
            "update_learning_policy(",
            "mutate_strategy(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "invoke_agent(",
            "allocate_resource(",
            "emit_hitl_request(",
            "observe_runtime_outcome(",
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
