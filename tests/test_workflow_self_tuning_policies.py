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
    WorkflowSelfTuningPolicyPlan,
    plan_workflow_self_tuning_policies,
    route_request,
    workflow_self_tuning_policies_for_status,
    workflow_self_tuning_policy_by_id,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_POLICY_FIELDS = {
    "policy_id",
    "policy_kind",
    "status",
    "route_name",
    "task_type",
    "execution_mode_id",
    "source_retry_policy_id",
    "source_load_balance_candidate_id",
    "source_dynamic_resource_allocation_id",
    "source_dynamic_strategy_id",
    "source_dynamic_strategy_kind",
    "retry_policy_pressure",
    "load_balancing_pressure",
    "dynamic_resource_score",
    "dynamic_strategy_score",
    "retry_policy_score",
    "load_balance_score",
    "guardrail_penalty",
    "self_tuning_score",
    "hitl_required",
    "applied_policy",
    "policy_summary",
    "fallback_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "workflow_self_tuning_policy_implemented",
    "self_tuning_recommendation_implemented",
    "retry_policy_metadata_used",
    "load_balance_metadata_used",
    "dynamic_resource_allocation_metadata_used",
    "dynamic_execution_strategy_metadata_used",
    "workflow_self_tuning_application_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_order_mutation_implemented",
    "graph_compilation_implemented",
    "workflow_execution_implemented",
    "retry_triggering_implemented",
    "refinement_triggering_implemented",
    "request_distribution_implemented",
    "load_balancing_runtime_implemented",
    "resource_allocation_implemented",
    "runtime_resource_measurement_implemented",
    "capacity_enforcement_implemented",
    "budget_enforcement_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class WorkflowSelfTuningPolicyTests(unittest.TestCase):
    def test_plan_combines_retry_load_resource_and_strategy_sources(self) -> None:
        plan = plan_workflow_self_tuning_policies(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "workflow_self_tuning_policy_planner")
        self.assertEqual(
            plan.serialization_version,
            "workflow_self_tuning_policy_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_retry_policy_serialization_version,
            "retry_policy_plan.v1",
        )
        self.assertEqual(
            plan.source_load_balancer_serialization_version,
            "load_balancer_plan.v1",
        )
        self.assertEqual(
            plan.source_dynamic_resource_allocation_serialization_version,
            "dynamic_resource_allocation_plan.v1",
        )
        self.assertEqual(
            plan.source_dynamic_execution_strategy_serialization_version,
            "adaptive_execution_strategy_selection_plan.v1",
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.policy_count, 4)
        self.assertEqual(plan.recommended_policy_count, 2)
        self.assertEqual(len(plan.review_required_policy_ids), 1)
        self.assertEqual(len(plan.guardrail_policy_ids), 1)
        self.assertEqual(plan.hitl_required_policy_count, 4)
        self.assertFalse(plan.applied_policy_ids)
        self.assertIn("do not apply tuning policies", plan.authority_boundary)
        self.assertTrue(plan.workflow_self_tuning_policy_implemented)
        self.assertTrue(plan.self_tuning_recommendation_implemented)
        self.assertTrue(plan.retry_policy_metadata_used)
        self.assertTrue(plan.load_balance_metadata_used)
        self.assertTrue(plan.dynamic_resource_allocation_metadata_used)
        self.assertTrue(plan.dynamic_execution_strategy_metadata_used)
        self.assertFalse(plan.workflow_self_tuning_application_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_order_mutation_implemented)
        self.assertFalse(plan.graph_compilation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.refinement_triggering_implemented)
        self.assertFalse(plan.request_distribution_implemented)
        self.assertFalse(plan.load_balancing_runtime_implemented)
        self.assertFalse(plan.resource_allocation_implemented)
        self.assertFalse(plan.runtime_resource_measurement_implemented)
        self.assertFalse(plan.capacity_enforcement_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.node_handler_invocation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_policies_score_self_tuning_metadata_without_applying(self) -> None:
        plan = plan_workflow_self_tuning_policies(route="generate")

        for policy in plan.policies:
            dumped = policy.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_POLICY_FIELDS)
            self.assertEqual(
                policy.serialization_version,
                "workflow_self_tuning_policy.v1",
            )
            self.assertEqual(policy.route_name, RouteName.GENERATE)
            self.assertFalse(policy.applied_policy)
            self.assertEqual(
                policy.self_tuning_score,
                min(
                    600,
                    max(
                        0,
                        policy.retry_policy_score // 4
                        + policy.load_balance_score // 10
                        + policy.dynamic_resource_score // 3
                        + policy.dynamic_strategy_score // 8
                        - policy.guardrail_penalty,
                    ),
                ),
            )
            self.assertIn(
                "workflow_self_tuning_application",
                policy.blocked_runtime_behaviors,
            )
            self.assertTrue(policy.workflow_self_tuning_policy_implemented)
            self.assertTrue(policy.retry_policy_metadata_used)
            self.assertTrue(policy.load_balance_metadata_used)
            self.assertTrue(policy.dynamic_resource_allocation_metadata_used)
            self.assertTrue(policy.dynamic_execution_strategy_metadata_used)
            self.assertFalse(policy.workflow_self_tuning_application_implemented)
            self.assertFalse(policy.workflow_control_implemented)
            self.assertFalse(policy.workflow_graph_mutation_implemented)
            self.assertFalse(policy.workflow_execution_implemented)
            self.assertFalse(policy.retry_triggering_implemented)
            self.assertFalse(policy.refinement_triggering_implemented)
            self.assertFalse(policy.request_distribution_implemented)
            self.assertFalse(policy.load_balancing_runtime_implemented)
            self.assertFalse(policy.resource_allocation_implemented)
            self.assertFalse(policy.runtime_resource_measurement_implemented)
            self.assertFalse(policy.provider_model_routing_implemented)
            self.assertFalse(policy.provider_execution_implemented)
            self.assertFalse(policy.agent_invocation_implemented)
            self.assertFalse(policy.node_handler_invocation_implemented)
            self.assertFalse(policy.generated_output_mutation_implemented)
            self.assertTrue(policy.hitl_required)
            self.assertTrue(policy.advisory_only)

        load_policy = workflow_self_tuning_policy_by_id(
            "workflow_self_tuning::load_self_tuning_policy",
            plan,
        )
        recommended = workflow_self_tuning_policies_for_status("recommended", plan)
        guardrail = workflow_self_tuning_policies_for_status("guardrail", plan)
        self.assertIsNotNone(load_policy)
        assert load_policy is not None
        self.assertEqual(load_policy.status, "recommended")
        self.assertEqual(len(recommended), 2)
        self.assertEqual(len(guardrail), 1)

    def test_plan_rejects_mismatched_policy_metadata(self) -> None:
        plan = plan_workflow_self_tuning_policies()
        payload = plan.model_dump(mode="json")
        payload["policy_ids"] = ("missing",) + tuple(payload["policy_ids"][1:])

        with self.assertRaisesRegex(ValueError, "policy_ids must match"):
            WorkflowSelfTuningPolicyPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["highest_self_tuning_score"] += 1

        with self.assertRaisesRegex(ValueError, "highest_self_tuning_score must match"):
            WorkflowSelfTuningPolicyPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_policy_ids"] = (plan.policy_ids[0],)

        with self.assertRaisesRegex(ValueError, "applied_policy_ids must remain empty"):
            WorkflowSelfTuningPolicyPlan(**payload)

    def test_planner_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Plan workflow self tuning for a generative animation.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = plan_workflow_self_tuning_policies(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_planner_does_not_declare_runtime_application_terms(self) -> None:
        plan = plan_workflow_self_tuning_policies(route=RouteName.GENERATE)
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
                        policy.source_retry_policy_id,
                        policy.source_load_balance_candidate_id,
                        policy.source_dynamic_resource_allocation_id,
                        policy.source_dynamic_strategy_id,
                        policy.policy_summary,
                        policy.fallback_summary,
                        *policy.advisory_actions,
                        *policy.evidence,
                        *policy.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "apply_tuning_policy(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "change_workflow_order(",
            "compile_graph(",
            "execute_workflow(",
            "trigger_retry(",
            "trigger_refinement(",
            "distribute_requests(",
            "run_load_balancer(",
            "allocate_resource(",
            "measure_runtime_resources(",
            "enforce_capacity(",
            "enforce_budget(",
            "route_provider(",
            "execute_provider(",
            "invoke_agent(",
            "invoke_node_handler(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
