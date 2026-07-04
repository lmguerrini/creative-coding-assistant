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
    DynamicAgentAllocationPlan,
    allocate_dynamic_agents,
    dynamic_agent_allocation_by_agent_id,
    dynamic_agent_allocations_for_lane,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_CANDIDATE_FIELDS = {
    "allocation_id",
    "agent_id",
    "role_id",
    "route_name",
    "task_type",
    "execution_mode_id",
    "source_activation_candidate_id",
    "source_activation_status",
    "source_dynamic_strategy_id",
    "source_dynamic_strategy_kind",
    "source_scheduling_group_id",
    "source_dependency_node_id",
    "priority_band",
    "allocation_lane",
    "allocation_status",
    "activation_order",
    "allocation_order",
    "activation_score",
    "strategy_score",
    "scheduling_weight",
    "capability_weight",
    "hitl_penalty",
    "allocation_score",
    "scheduling_hint",
    "parallelizable",
    "blocking_group_ids",
    "downstream_group_ids",
    "source_capability_ids",
    "required_metadata_input_count",
    "produced_metadata_output_count",
    "upstream_dependency_count",
    "downstream_dependency_count",
    "hitl_required",
    "fallback_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "dynamic_agent_allocation_implemented",
    "agent_allocation_recommendation_implemented",
    "dynamic_execution_strategy_metadata_used",
    "activation_metadata_used",
    "scheduling_metadata_used",
    "dependency_metadata_used",
    "runtime_agent_allocation_implemented",
    "agent_activation_implemented",
    "agent_instantiation_implemented",
    "agent_invocation_implemented",
    "lifecycle_transition_execution_implemented",
    "scheduler_runtime_hook_implemented",
    "parallel_execution_implemented",
    "async_behavior_changed",
    "workflow_routing_implemented",
    "workflow_timing_changed",
    "workflow_control_implemented",
    "provider_model_routing_implemented",
    "runtime_selection_implemented",
    "hitl_request_emitted",
    "budget_enforcement_implemented",
    "retry_triggering_implemented",
    "memory_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class DynamicAgentAllocationTests(unittest.TestCase):
    def test_plan_combines_activation_strategy_scheduling_and_dependencies(
        self,
    ) -> None:
        plan = allocate_dynamic_agents(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "dynamic_agent_allocator")
        self.assertEqual(plan.serialization_version, "dynamic_agent_allocation_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_agent_activation_serialization_version,
            "agent_activation_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_dynamic_execution_strategy_serialization_version,
            "adaptive_execution_strategy_selection_plan.v1",
        )
        self.assertEqual(
            plan.source_parallel_scheduling_serialization_version,
            "parallel_scheduling_registry.v1",
        )
        self.assertEqual(
            plan.source_dependency_graph_serialization_version,
            "agent_dependency_graph.v1",
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.allocation_count, 12)
        self.assertEqual(plan.primary_allocation_count, 3)
        self.assertEqual(len(plan.standby_allocation_ids), 9)
        self.assertEqual(plan.hitl_required_allocation_count, 3)
        self.assertFalse(plan.applied_allocation_ids)
        self.assertIn("does not allocate agents at runtime", plan.authority_boundary)
        self.assertTrue(plan.dynamic_agent_allocation_implemented)
        self.assertTrue(plan.agent_allocation_recommendation_implemented)
        self.assertTrue(plan.dynamic_execution_strategy_metadata_used)
        self.assertTrue(plan.activation_metadata_used)
        self.assertTrue(plan.scheduling_metadata_used)
        self.assertTrue(plan.dependency_metadata_used)
        self.assertFalse(plan.runtime_agent_allocation_implemented)
        self.assertFalse(plan.agent_activation_implemented)
        self.assertFalse(plan.agent_instantiation_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.lifecycle_transition_execution_implemented)
        self.assertFalse(plan.scheduler_runtime_hook_implemented)
        self.assertFalse(plan.parallel_execution_implemented)
        self.assertFalse(plan.async_behavior_changed)
        self.assertFalse(plan.workflow_routing_implemented)
        self.assertFalse(plan.workflow_timing_changed)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.runtime_selection_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.memory_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_allocations_assign_agents_to_advisory_lanes(self) -> None:
        plan = allocate_dynamic_agents(route="generate")

        for allocation in plan.allocations:
            dumped = allocation.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CANDIDATE_FIELDS)
            self.assertEqual(
                allocation.serialization_version,
                "dynamic_agent_allocation_candidate.v1",
            )
            self.assertEqual(allocation.route_name, RouteName.GENERATE)
            self.assertEqual(allocation.allocation_order, allocation.activation_order)
            self.assertEqual(
                allocation.source_activation_candidate_id,
                f"agent_activation::{allocation.agent_id}",
            )
            self.assertEqual(
                allocation.source_dependency_node_id,
                f"agent::{allocation.agent_id}",
            )
            self.assertEqual(
                allocation.allocation_score,
                min(
                    320,
                    max(
                        0,
                        allocation.activation_score
                        + allocation.strategy_score // 5
                        + allocation.scheduling_weight
                        + allocation.capability_weight
                        - allocation.hitl_penalty,
                    ),
                ),
            )
            self.assertIn(
                "runtime_agent_allocation", allocation.blocked_runtime_behaviors
            )
            self.assertTrue(allocation.dynamic_agent_allocation_implemented)
            self.assertTrue(allocation.dynamic_execution_strategy_metadata_used)
            self.assertTrue(allocation.activation_metadata_used)
            self.assertTrue(allocation.scheduling_metadata_used)
            self.assertTrue(allocation.dependency_metadata_used)
            self.assertFalse(allocation.runtime_agent_allocation_implemented)
            self.assertFalse(allocation.agent_activation_implemented)
            self.assertFalse(allocation.agent_instantiation_implemented)
            self.assertFalse(allocation.agent_invocation_implemented)
            self.assertFalse(allocation.lifecycle_transition_execution_implemented)
            self.assertFalse(allocation.scheduler_runtime_hook_implemented)
            self.assertFalse(allocation.parallel_execution_implemented)
            self.assertFalse(allocation.async_behavior_changed)
            self.assertFalse(allocation.workflow_routing_implemented)
            self.assertFalse(allocation.workflow_timing_changed)
            self.assertFalse(allocation.provider_model_routing_implemented)
            self.assertFalse(allocation.runtime_selection_implemented)
            self.assertFalse(allocation.hitl_request_emitted)
            self.assertFalse(allocation.budget_enforcement_implemented)
            self.assertFalse(allocation.memory_mutation_implemented)
            self.assertFalse(allocation.generated_output_mutation_implemented)
            self.assertTrue(allocation.advisory_only)

        planner = dynamic_agent_allocation_by_agent_id("planner_agent", plan)
        missing = dynamic_agent_allocation_by_agent_id("missing_agent", plan)
        primary = dynamic_agent_allocations_for_lane("strategy_primary", plan)
        standby = dynamic_agent_allocations_for_lane("standby_pool", plan)
        self.assertIsNotNone(planner)
        self.assertIsNone(missing)
        assert planner is not None
        self.assertEqual(planner.allocation_lane, "strategy_primary")
        self.assertEqual(planner.allocation_status, "requires_hitl")
        self.assertTrue(planner.hitl_required)
        self.assertEqual(len(primary), 3)
        self.assertEqual(len(standby), 9)

    def test_preview_route_limits_allocations_to_preview_agents(self) -> None:
        plan = allocate_dynamic_agents(route="preview")
        agent_ids = tuple(allocation.agent_id for allocation in plan.allocations)

        self.assertIn("runtime_agent", agent_ids)
        self.assertIn("artifact_agent", agent_ids)
        self.assertIn("style_agent", agent_ids)
        self.assertNotIn("research_agent", agent_ids)
        self.assertLess(plan.allocation_count, 12)

    def test_plan_rejects_mismatched_allocation_metadata(self) -> None:
        plan = allocate_dynamic_agents()
        payload = plan.model_dump(mode="json")
        payload["allocation_ids"] = ("missing",) + tuple(payload["allocation_ids"][1:])

        with self.assertRaisesRegex(ValueError, "allocation_ids must match"):
            DynamicAgentAllocationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["highest_allocation_score"] += 1

        with self.assertRaisesRegex(ValueError, "highest_allocation_score must match"):
            DynamicAgentAllocationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_allocation_ids"] = (plan.allocation_ids[0],)

        with self.assertRaisesRegex(
            ValueError, "applied_allocation_ids must remain empty"
        ):
            DynamicAgentAllocationPlan(**payload)

    def test_allocator_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Allocate advisory agents for a shader workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = allocate_dynamic_agents(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_allocator_does_not_declare_runtime_application_terms(self) -> None:
        plan = allocate_dynamic_agents(route=RouteName.GENERATE)
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
                        allocation.agent_id,
                        allocation.role_id,
                        allocation.source_activation_candidate_id,
                        allocation.source_dynamic_strategy_id,
                        allocation.source_scheduling_group_id,
                        allocation.source_dependency_node_id,
                        *allocation.source_capability_ids,
                        allocation.fallback_summary,
                        *allocation.advisory_actions,
                        *allocation.evidence,
                        *allocation.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "allocate_agent(",
            "instantiate_agent(",
            "invoke_agent(",
            "activate_agent(",
            "run_lifecycle_transition(",
            "run_scheduler(",
            "execute_parallel_tasks(",
            "change_async_behavior(",
            "change_workflow_route(",
            "change_workflow_timing(",
            "control_workflow(",
            "route_provider(",
            "select_runtime(",
            "emit_hitl_request(",
            "enforce_budget(",
            "trigger_retry(",
            "mutate_memory(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
