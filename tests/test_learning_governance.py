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
    LearningGovernancePlan,
    build_learning_governance,
    learning_governance_policies_for_status,
    learning_governance_policy_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_SOURCE_PLAN_ROLES = (
    "continuous_improvement_signals",
    "success_pattern_discovery",
    "failure_pattern_discovery",
    "adaptive_learning_engine",
)
REQUIRED_POLICY_FIELDS = {
    "policy_id",
    "policy_kind",
    "status",
    "priority",
    "route_name",
    "task_type",
    "execution_mode_id",
    "source_plan_roles",
    "governed_surface",
    "review_requirement",
    "explainability_requirement",
    "no_automation_boundary",
    "source_signal_count",
    "source_guarded_count",
    "source_hitl_required_count",
    "governance_weight",
    "governance_score",
    "hitl_required_before_application",
    "governance_tags",
    "governance_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "learning_governance_implemented",
    "governance_policy_metadata_implemented",
    "learning_memory_persistence_implemented",
    "learning_feedback_application_implemented",
    "learning_policy_update_implemented",
    "learning_policy_enforcement_implemented",
    "hitl_request_emitted",
    "human_input_request_implemented",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "resource_allocation_implemented",
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


class LearningGovernanceTests(unittest.TestCase):
    def test_plan_builds_governance_and_safety_boundaries(self) -> None:
        plan = build_learning_governance(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "learning_governance")
        self.assertEqual(plan.serialization_version, "learning_governance_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_adaptive_learning_serialization_version,
            "adaptive_learning_plan.v1",
        )
        self.assertEqual(plan.source_plan_roles, REQUIRED_SOURCE_PLAN_ROLES)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.policy_count, 5)
        self.assertEqual(plan.blocked_policy_count, 1)
        self.assertEqual(plan.review_required_policy_count, 2)
        self.assertEqual(plan.guarded_policy_count, 2)
        self.assertEqual(plan.hitl_required_policy_count, 5)
        self.assertFalse(plan.applied_governance_policy_ids)
        self.assertEqual(plan.overall_governance_posture, "guarded")
        self.assertIn("does not persist learning memory", plan.authority_boundary)
        self.assertTrue(plan.learning_governance_implemented)
        self.assertTrue(plan.governance_policy_metadata_implemented)
        self.assertFalse(plan.learning_memory_persistence_implemented)
        self.assertFalse(plan.learning_feedback_application_implemented)
        self.assertFalse(plan.learning_policy_update_implemented)
        self.assertFalse(plan.learning_policy_enforcement_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.human_input_request_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.resource_allocation_implemented)
        self.assertFalse(plan.runtime_outcome_observation_implemented)
        self.assertFalse(plan.generated_output_evaluation_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_policies_score_governance_without_enforcement(self) -> None:
        plan = build_learning_governance(route="generate")

        for policy in plan.policies:
            dumped = policy.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_POLICY_FIELDS)
            self.assertEqual(
                policy.serialization_version,
                "learning_governance_policy.v1",
            )
            self.assertEqual(policy.route_name, RouteName.GENERATE)
            self.assertEqual(
                policy.policy_id,
                f"learning_governance::{policy.policy_kind}",
            )
            self.assertEqual(
                policy.governance_score,
                min(
                    1000,
                    max(
                        0,
                        policy.source_signal_count * 20
                        + policy.source_guarded_count * 55
                        + policy.source_hitl_required_count * 35
                        + policy.governance_weight,
                    ),
                ),
            )
            self.assertIn(
                "learning_memory_persistence",
                policy.blocked_runtime_behaviors,
            )
            self.assertEqual(policy.source_plan_roles, REQUIRED_SOURCE_PLAN_ROLES)
            self.assertTrue(policy.governance_tags)
            self.assertTrue(policy.advisory_actions)
            self.assertTrue(policy.evidence)
            self.assertTrue(policy.hitl_required_before_application)
            self.assertFalse(policy.learning_memory_persistence_implemented)
            self.assertFalse(policy.learning_feedback_application_implemented)
            self.assertFalse(policy.learning_policy_update_implemented)
            self.assertFalse(policy.learning_policy_enforcement_implemented)
            self.assertFalse(policy.hitl_request_emitted)
            self.assertFalse(policy.human_input_request_implemented)
            self.assertFalse(policy.provider_model_routing_implemented)
            self.assertFalse(policy.provider_execution_implemented)
            self.assertFalse(policy.agent_invocation_implemented)
            self.assertFalse(policy.resource_allocation_implemented)
            self.assertFalse(policy.workflow_control_implemented)
            self.assertFalse(policy.workflow_graph_mutation_implemented)
            self.assertFalse(policy.workflow_execution_implemented)
            self.assertFalse(policy.persistent_storage_write_implemented)
            self.assertFalse(policy.generated_output_mutation_implemented)
            self.assertFalse(policy.runtime_evolution_implemented)
            self.assertTrue(policy.advisory_only)

        guarded = learning_governance_policy_by_id(
            "learning_governance::safety_no_automation_boundary",
            plan,
        )
        self.assertIsNotNone(guarded)
        self.assertEqual(
            len(learning_governance_policies_for_status("blocked", plan)),
            1,
        )
        self.assertEqual(
            len(learning_governance_policies_for_status("guarded", plan)),
            2,
        )

    def test_plan_rejects_mismatched_governance_metadata(self) -> None:
        plan = build_learning_governance()
        payload = plan.model_dump(mode="json")
        payload["policy_ids"] = ("missing",) + tuple(payload["policy_ids"][1:])

        with self.assertRaisesRegex(ValueError, "policy_ids must match"):
            LearningGovernancePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_governance_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_governance_score must match",
        ):
            LearningGovernancePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_governance_policy_ids"] = (plan.policy_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_governance_policy_ids must remain empty",
        ):
            LearningGovernancePlan(**payload)

    def test_learning_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review learning governance for a creative workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_learning_governance(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_learning_does_not_declare_governance_application_terms(self) -> None:
        plan = build_learning_governance(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for policy in plan.policies
                    for field in (
                        policy.policy_id,
                        policy.policy_kind,
                        policy.governed_surface,
                        policy.review_requirement,
                        policy.explainability_requirement,
                        policy.no_automation_boundary,
                        *policy.governance_tags,
                        policy.governance_summary,
                        *policy.advisory_actions,
                        *policy.evidence,
                        *policy.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "persist_learning_memory(",
            "apply_feedback(",
            "update_learning_policy(",
            "enforce_learning_policy(",
            "emit_hitl_request(",
            "request_human_input(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "invoke_agent(",
            "allocate_resource(",
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
