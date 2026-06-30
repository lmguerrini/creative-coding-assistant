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
    DynamicResourceAllocationPlan,
    allocate_dynamic_resources,
    dynamic_resource_allocation_by_id,
    dynamic_resource_allocations_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_CANDIDATE_FIELDS = {
    "allocation_id",
    "source_resource_recommendation_id",
    "source_resource_utilization_id",
    "source_resource_status",
    "route_name",
    "task_type",
    "execution_mode_id",
    "source_dynamic_agent_allocation_ids",
    "source_dynamic_strategy_id",
    "source_cost_quality_candidate_id",
    "source_latency_candidate_id",
    "source_budget_policy_id",
    "allocation_status",
    "resource_utilization_pressure",
    "adaptive_cost_quality_posture",
    "adaptive_latency_posture",
    "budget_posture",
    "advisory_resource_units",
    "advisory_reserve_units",
    "advisory_pressure_units",
    "applied_resource_units",
    "applied_reserve_units",
    "resource_utilization_score",
    "agent_allocation_score",
    "cost_quality_score",
    "latency_score",
    "budget_posture_weight",
    "guardrail_penalty",
    "dynamic_resource_score",
    "agent_allocation_count",
    "hitl_required_agent_allocation_count",
    "hitl_required",
    "fallback_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "dynamic_resource_allocation_implemented",
    "resource_allocation_recommendation_implemented",
    "resource_utilization_metadata_used",
    "dynamic_agent_allocation_metadata_used",
    "adaptive_cost_quality_metadata_used",
    "adaptive_latency_metadata_used",
    "budget_policy_metadata_used",
    "resource_allocation_implemented",
    "runtime_resource_measurement_implemented",
    "cpu_memory_measurement_implemented",
    "concurrency_limit_change_implemented",
    "queue_management_runtime_implemented",
    "autoscaling_implemented",
    "capacity_enforcement_implemented",
    "budget_enforcement_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "workflow_control_implemented",
    "workflow_timing_change_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class DynamicResourceAllocationTests(unittest.TestCase):
    def test_plan_combines_resource_agent_cost_latency_and_budget_posture(self) -> None:
        plan = allocate_dynamic_resources(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "dynamic_resource_allocator")
        self.assertEqual(plan.serialization_version, "dynamic_resource_allocation_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_resource_utilization_serialization_version,
            "resource_utilization_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_dynamic_agent_allocation_serialization_version,
            "dynamic_agent_allocation_plan.v1",
        )
        self.assertEqual(
            plan.source_adaptive_cost_quality_serialization_version,
            "adaptive_cost_quality_plan.v1",
        )
        self.assertEqual(
            plan.source_adaptive_latency_serialization_version,
            "adaptive_latency_plan.v1",
        )
        self.assertEqual(
            plan.source_budget_policy_serialization_version,
            "budget_policy_plan.v1",
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.allocation_count, 6)
        self.assertEqual(plan.recommended_allocation_count, 2)
        self.assertEqual(len(plan.capacity_guardrail_allocation_ids), 1)
        self.assertEqual(len(plan.review_required_allocation_ids), 2)
        self.assertEqual(len(plan.boundary_guardrail_allocation_ids), 1)
        self.assertEqual(plan.hitl_required_allocation_count, 6)
        self.assertFalse(plan.applied_resource_allocation_ids)
        self.assertEqual(plan.total_applied_resource_units, 0)
        self.assertIn("does not allocate resources", plan.authority_boundary)
        self.assertTrue(plan.dynamic_resource_allocation_implemented)
        self.assertTrue(plan.resource_allocation_recommendation_implemented)
        self.assertTrue(plan.resource_utilization_metadata_used)
        self.assertTrue(plan.dynamic_agent_allocation_metadata_used)
        self.assertTrue(plan.adaptive_cost_quality_metadata_used)
        self.assertTrue(plan.adaptive_latency_metadata_used)
        self.assertTrue(plan.budget_policy_metadata_used)
        self.assertFalse(plan.resource_allocation_implemented)
        self.assertFalse(plan.runtime_resource_measurement_implemented)
        self.assertFalse(plan.cpu_memory_measurement_implemented)
        self.assertFalse(plan.concurrency_limit_change_implemented)
        self.assertFalse(plan.queue_management_runtime_implemented)
        self.assertFalse(plan.autoscaling_implemented)
        self.assertFalse(plan.capacity_enforcement_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_timing_change_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.node_handler_invocation_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_allocations_are_advisory_resource_recommendations(self) -> None:
        plan = allocate_dynamic_resources(route="generate")

        for allocation in plan.allocations:
            dumped = allocation.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CANDIDATE_FIELDS)
            self.assertEqual(
                allocation.serialization_version,
                "dynamic_resource_allocation_candidate.v1",
            )
            self.assertEqual(allocation.route_name, RouteName.GENERATE)
            self.assertEqual(allocation.applied_resource_units, 0)
            self.assertEqual(allocation.applied_reserve_units, 0)
            self.assertEqual(
                allocation.source_resource_recommendation_id,
                f"resource_utilization::{allocation.source_resource_utilization_id}",
            )
            self.assertEqual(
                allocation.dynamic_resource_score,
                min(
                    500,
                    max(
                        0,
                        allocation.resource_utilization_score // 8
                        + allocation.agent_allocation_score // 3
                        + allocation.cost_quality_score // 2
                        + allocation.latency_score // 2
                        + allocation.budget_posture_weight
                        - allocation.guardrail_penalty,
                    ),
                ),
            )
            self.assertIn("resource_allocation", allocation.blocked_runtime_behaviors)
            self.assertTrue(allocation.dynamic_resource_allocation_implemented)
            self.assertTrue(allocation.resource_utilization_metadata_used)
            self.assertTrue(allocation.dynamic_agent_allocation_metadata_used)
            self.assertTrue(allocation.adaptive_cost_quality_metadata_used)
            self.assertTrue(allocation.adaptive_latency_metadata_used)
            self.assertTrue(allocation.budget_policy_metadata_used)
            self.assertFalse(allocation.resource_allocation_implemented)
            self.assertFalse(allocation.runtime_resource_measurement_implemented)
            self.assertFalse(allocation.cpu_memory_measurement_implemented)
            self.assertFalse(allocation.concurrency_limit_change_implemented)
            self.assertFalse(allocation.queue_management_runtime_implemented)
            self.assertFalse(allocation.autoscaling_implemented)
            self.assertFalse(allocation.capacity_enforcement_implemented)
            self.assertFalse(allocation.budget_enforcement_implemented)
            self.assertFalse(allocation.provider_model_routing_implemented)
            self.assertFalse(allocation.provider_execution_implemented)
            self.assertFalse(allocation.workflow_control_implemented)
            self.assertFalse(allocation.workflow_graph_mutation_implemented)
            self.assertFalse(allocation.workflow_execution_implemented)
            self.assertFalse(allocation.agent_invocation_implemented)
            self.assertFalse(allocation.node_handler_invocation_implemented)
            self.assertFalse(allocation.retry_triggering_implemented)
            self.assertFalse(allocation.generated_output_mutation_implemented)
            self.assertTrue(allocation.hitl_required)
            self.assertTrue(allocation.advisory_only)

        throughput = dynamic_resource_allocation_by_id(
            "dynamic_resource_allocation::throughput_capacity_utilization",
            plan,
        )
        recommended = dynamic_resource_allocations_for_status("recommended", plan)
        boundary = dynamic_resource_allocations_for_status("boundary_guardrail", plan)
        self.assertIsNotNone(throughput)
        assert throughput is not None
        self.assertEqual(throughput.allocation_status, "recommended")
        self.assertEqual(len(recommended), 2)
        self.assertEqual(len(boundary), 1)
        self.assertEqual(boundary[0].applied_resource_units, 0)

    def test_plan_rejects_mismatched_resource_allocation_metadata(self) -> None:
        plan = allocate_dynamic_resources()
        payload = plan.model_dump(mode="json")
        payload["allocation_ids"] = ("missing",) + tuple(payload["allocation_ids"][1:])

        with self.assertRaisesRegex(ValueError, "allocation_ids must match"):
            DynamicResourceAllocationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["highest_dynamic_resource_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "highest_dynamic_resource_score must match",
        ):
            DynamicResourceAllocationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_resource_allocation_ids"] = (plan.allocation_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_resource_allocation_ids must remain empty",
        ):
            DynamicResourceAllocationPlan(**payload)

    def test_allocator_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Plan resource posture for a multi-agent WebGL workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = allocate_dynamic_resources(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_allocator_does_not_declare_runtime_application_terms(self) -> None:
        plan = allocate_dynamic_resources(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for allocation in plan.allocations
                    for field in (
                        allocation.allocation_id,
                        allocation.source_resource_recommendation_id,
                        allocation.source_resource_utilization_id,
                        allocation.source_dynamic_strategy_id,
                        allocation.source_cost_quality_candidate_id,
                        allocation.source_latency_candidate_id,
                        allocation.source_budget_policy_id,
                        *allocation.source_dynamic_agent_allocation_ids,
                        allocation.fallback_summary,
                        *allocation.advisory_actions,
                        *allocation.evidence,
                        *allocation.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "allocate_resource(",
            "measure_cpu(",
            "measure_memory(",
            "change_concurrency_limit(",
            "manage_queue(",
            "autoscale(",
            "enforce_capacity(",
            "enforce_budget(",
            "route_provider(",
            "execute_provider(",
            "control_workflow(",
            "change_workflow_timing(",
            "mutate_workflow_graph(",
            "execute_workflow(",
            "invoke_agent(",
            "invoke_node_handler(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
