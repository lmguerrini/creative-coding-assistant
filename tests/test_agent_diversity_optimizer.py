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
    AgentDiversityOptimizationPlan,
    agent_diversity_candidate_by_agent_id,
    agent_diversity_candidates_for_status,
    optimize_agent_diversity,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_CANDIDATE_FIELDS = {
    "candidate_id",
    "agent_id",
    "role_id",
    "route_name",
    "task_type",
    "execution_mode_id",
    "source_dynamic_allocation_id",
    "source_allocation_lane",
    "source_allocation_status",
    "source_allocation_score",
    "source_capability_alignment_agent_id",
    "source_role_registry_role_id",
    "role_family",
    "capability_family",
    "priority_band",
    "scheduling_hint",
    "parallelizable",
    "source_capability_ids",
    "aligned_capability_ids",
    "required_metadata_input_count",
    "produced_metadata_output_count",
    "upstream_dependency_count",
    "downstream_dependency_count",
    "role_family_weight",
    "capability_family_weight",
    "alignment_breadth_score",
    "allocation_signal_score",
    "parallelism_bonus",
    "hitl_penalty",
    "agent_diversity_score",
    "status",
    "recommended_diversity_path_count",
    "applied_diversity_path_count",
    "hitl_required",
    "diversity_summary",
    "fallback_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "agent_diversity_optimizer_implemented",
    "agent_diversity_metadata_implemented",
    "dynamic_agent_allocation_metadata_used",
    "agent_role_metadata_used",
    "capability_alignment_metadata_used",
    "scheduling_metadata_used",
    "dependency_metadata_used",
    "hitl_posture_metadata_used",
    "agent_diversity_behavior_application_implemented",
    "runtime_agent_selection_implemented",
    "agent_pool_rebalancing_implemented",
    "capability_activation_implemented",
    "runtime_work_routing_implemented",
    "dynamic_task_routing_implemented",
    "agent_activation_implemented",
    "agent_instantiation_implemented",
    "agent_invocation_implemented",
    "scheduler_runtime_hook_implemented",
    "parallel_execution_implemented",
    "workflow_routing_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "provider_model_routing_implemented",
    "runtime_selection_implemented",
    "hitl_request_emitted",
    "budget_enforcement_implemented",
    "retry_triggering_implemented",
    "memory_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class AgentDiversityOptimizerTests(unittest.TestCase):
    def test_plan_combines_allocation_role_and_alignment_sources(self) -> None:
        plan = optimize_agent_diversity(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "agent_diversity_optimizer")
        self.assertEqual(
            plan.serialization_version,
            "agent_diversity_optimization_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_dynamic_agent_allocation_serialization_version,
            "dynamic_agent_allocation_plan.v1",
        )
        self.assertEqual(
            plan.source_agent_capability_alignment_serialization_version,
            "agent_capability_alignment_registry.v1",
        )
        self.assertEqual(
            plan.source_agent_role_serialization_version,
            "agent_role_registry.v1",
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.role_family_count, 11)
        self.assertEqual(plan.capability_family_count, 12)
        self.assertEqual(plan.aligned_capability_count, 13)
        self.assertEqual(plan.candidate_count, 12)
        self.assertEqual(plan.recommended_candidate_count, 3)
        self.assertEqual(len(plan.standby_candidate_ids), 8)
        self.assertEqual(plan.guardrail_candidate_count, 1)
        self.assertEqual(plan.hitl_required_candidate_count, 3)
        self.assertFalse(plan.applied_diversity_candidate_ids)
        self.assertEqual(plan.total_recommended_diversity_path_count, 11)
        self.assertEqual(plan.total_applied_diversity_path_count, 0)
        self.assertEqual(plan.highest_agent_diversity_score, 330)
        self.assertIn(
            "does not apply agent diversity behavior",
            plan.authority_boundary,
        )
        self.assertTrue(plan.agent_diversity_optimizer_implemented)
        self.assertTrue(plan.agent_diversity_metadata_implemented)
        self.assertTrue(plan.dynamic_agent_allocation_metadata_used)
        self.assertTrue(plan.agent_role_metadata_used)
        self.assertTrue(plan.capability_alignment_metadata_used)
        self.assertTrue(plan.scheduling_metadata_used)
        self.assertTrue(plan.dependency_metadata_used)
        self.assertTrue(plan.hitl_posture_metadata_used)
        self.assertFalse(plan.agent_diversity_behavior_application_implemented)
        self.assertFalse(plan.runtime_agent_selection_implemented)
        self.assertFalse(plan.agent_pool_rebalancing_implemented)
        self.assertFalse(plan.capability_activation_implemented)
        self.assertFalse(plan.runtime_work_routing_implemented)
        self.assertFalse(plan.dynamic_task_routing_implemented)
        self.assertFalse(plan.agent_activation_implemented)
        self.assertFalse(plan.agent_instantiation_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.scheduler_runtime_hook_implemented)
        self.assertFalse(plan.parallel_execution_implemented)
        self.assertFalse(plan.workflow_routing_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.runtime_selection_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.memory_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_candidates_score_agent_diversity_without_selection(self) -> None:
        plan = optimize_agent_diversity(route="generate")

        for candidate in plan.candidates:
            dumped = candidate.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CANDIDATE_FIELDS)
            self.assertEqual(
                candidate.serialization_version,
                "agent_diversity_optimization_candidate.v1",
            )
            self.assertEqual(candidate.route_name, RouteName.GENERATE)
            self.assertEqual(
                candidate.candidate_id,
                f"agent_diversity_optimizer::{candidate.agent_id}",
            )
            self.assertEqual(
                candidate.source_dynamic_allocation_id,
                f"dynamic_agent_allocation::{candidate.agent_id}",
            )
            self.assertEqual(
                candidate.source_capability_alignment_agent_id,
                candidate.agent_id,
            )
            self.assertEqual(candidate.source_role_registry_role_id, candidate.role_id)
            self.assertEqual(candidate.role_family_weight, 70)
            self.assertEqual(candidate.capability_family_weight, 70)
            self.assertEqual(
                candidate.alignment_breadth_score,
                min(100, len(candidate.aligned_capability_ids) * 7),
            )
            self.assertEqual(
                candidate.allocation_signal_score,
                min(120, candidate.source_allocation_score // 3),
            )
            self.assertEqual(
                candidate.parallelism_bonus,
                20 if candidate.parallelizable else 6,
            )
            self.assertEqual(candidate.hitl_penalty, 28 if candidate.hitl_required else 0)
            self.assertEqual(
                candidate.agent_diversity_score,
                min(
                    360,
                    max(
                        0,
                        candidate.role_family_weight
                        + candidate.capability_family_weight
                        + candidate.alignment_breadth_score
                        + candidate.allocation_signal_score
                        + candidate.parallelism_bonus
                        - candidate.hitl_penalty,
                    ),
                ),
            )
            self.assertEqual(candidate.applied_diversity_path_count, 0)
            self.assertIn(
                "agent_diversity_behavior_application",
                candidate.blocked_runtime_behaviors,
            )
            self.assertTrue(candidate.agent_diversity_optimizer_implemented)
            self.assertTrue(candidate.dynamic_agent_allocation_metadata_used)
            self.assertTrue(candidate.agent_role_metadata_used)
            self.assertTrue(candidate.capability_alignment_metadata_used)
            self.assertFalse(candidate.agent_diversity_behavior_application_implemented)
            self.assertFalse(candidate.runtime_agent_selection_implemented)
            self.assertFalse(candidate.agent_pool_rebalancing_implemented)
            self.assertFalse(candidate.capability_activation_implemented)
            self.assertFalse(candidate.runtime_work_routing_implemented)
            self.assertFalse(candidate.dynamic_task_routing_implemented)
            self.assertFalse(candidate.agent_activation_implemented)
            self.assertFalse(candidate.agent_instantiation_implemented)
            self.assertFalse(candidate.agent_invocation_implemented)
            self.assertFalse(candidate.scheduler_runtime_hook_implemented)
            self.assertFalse(candidate.parallel_execution_implemented)
            self.assertFalse(candidate.workflow_routing_implemented)
            self.assertFalse(candidate.workflow_control_implemented)
            self.assertFalse(candidate.workflow_graph_mutation_implemented)
            self.assertFalse(candidate.workflow_execution_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.runtime_selection_implemented)
            self.assertFalse(candidate.hitl_request_emitted)
            self.assertFalse(candidate.budget_enforcement_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.memory_mutation_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.advisory_only)

        planner = agent_diversity_candidate_by_agent_id("planner_agent", plan)
        guardrails = agent_diversity_candidates_for_status("guardrail", plan)
        recommended = agent_diversity_candidates_for_status("recommended", plan)
        self.assertIsNotNone(planner)
        assert planner is not None
        self.assertEqual(planner.status, "recommended")
        self.assertTrue(planner.hitl_required)
        self.assertEqual(planner.recommended_diversity_path_count, 1)
        self.assertEqual(len(recommended), 3)
        self.assertEqual(len(guardrails), 1)
        self.assertTrue(
            all(
                candidate.recommended_diversity_path_count == 0
                for candidate in guardrails
            )
        )

    def test_plan_rejects_mismatched_agent_diversity_metadata(self) -> None:
        plan = optimize_agent_diversity()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            AgentDiversityOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["highest_agent_diversity_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "highest_agent_diversity_score must match",
        ):
            AgentDiversityOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_diversity_candidate_ids"] = (plan.candidate_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_diversity_candidate_ids must remain empty",
        ):
            AgentDiversityOptimizationPlan(**payload)

    def test_optimizer_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Optimize agent diversity for a shader workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = optimize_agent_diversity(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_optimizer_does_not_declare_runtime_agent_diversity_terms(self) -> None:
        plan = optimize_agent_diversity(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for candidate in plan.candidates
                    for field in (
                        candidate.candidate_id,
                        candidate.agent_id,
                        candidate.role_id,
                        candidate.source_dynamic_allocation_id,
                        candidate.source_capability_alignment_agent_id,
                        candidate.source_role_registry_role_id,
                        candidate.role_family,
                        candidate.capability_family,
                        candidate.priority_band,
                        candidate.scheduling_hint,
                        *candidate.source_capability_ids,
                        *candidate.aligned_capability_ids,
                        candidate.diversity_summary,
                        candidate.fallback_summary,
                        *candidate.advisory_actions,
                        *candidate.evidence,
                        *candidate.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "apply_agent_diversity(",
            "select_agent(",
            "rebalance_agent_pool(",
            "activate_capability(",
            "route_runtime_work(",
            "route_task(",
            "activate_agent(",
            "instantiate_agent(",
            "invoke_agent(",
            "run_scheduler(",
            "execute_parallel_task(",
            "route_workflow(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
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
