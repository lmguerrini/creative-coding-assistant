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
    AgentActivationOptimizationPlan,
    agent_activation_candidate_by_agent_id,
    agent_activation_candidates_for_status,
    optimize_agent_activation,
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
    "priority_band",
    "route_candidate_count",
    "required_metadata_input_count",
    "produced_metadata_output_count",
    "source_lifecycle_profile_id",
    "source_capability_ids",
    "estimated_cost_class",
    "estimated_latency_class",
    "activation_order",
    "activation_score",
    "status",
    "hitl_required",
    "escalation_posture",
    "activation_reason",
    "fallback_summary",
    "evidence",
    "blocked_runtime_behaviors",
    "agent_activation_optimizer_implemented",
    "agent_activation_recommendation_implemented",
    "agent_activation_implemented",
    "agent_instantiation_implemented",
    "agent_invocation_implemented",
    "lifecycle_transition_execution_implemented",
    "workflow_routing_implemented",
    "workflow_control_implemented",
    "runtime_selection_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "retry_triggering_implemented",
    "memory_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class AgentActivationOptimizerTests(unittest.TestCase):
    def test_plan_ranks_route_applicable_agent_activation_candidates(self) -> None:
        plan = optimize_agent_activation(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "agent_activation_optimizer")
        self.assertEqual(
            plan.serialization_version,
            "agent_activation_optimization_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_agent_routing_serialization_version,
            "agent_routing_registry.v1",
        )
        self.assertEqual(
            plan.source_agent_metadata_serialization_version,
            "agent_metadata_registry.v1",
        )
        self.assertEqual(
            plan.source_agent_capability_serialization_version,
            "agent_capability_registry.v1",
        )
        self.assertEqual(
            plan.source_agent_lifecycle_serialization_version,
            "agent_lifecycle_registry.v1",
        )
        self.assertEqual(
            plan.source_escalation_optimization_serialization_version,
            "adaptive_escalation_optimization_plan.v1",
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.candidate_count, 12)
        self.assertEqual(plan.candidates[0].agent_id, "planner_agent")
        self.assertEqual(plan.highest_activation_score, plan.candidates[0].activation_score)
        self.assertEqual(plan.activation_recommendation_count, 3)
        self.assertIn("does not instantiate agents", plan.authority_boundary)
        self.assertTrue(plan.agent_activation_optimizer_implemented)
        self.assertTrue(plan.agent_activation_recommendation_implemented)
        self.assertFalse(plan.agent_activation_implemented)
        self.assertFalse(plan.agent_instantiation_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.lifecycle_transition_execution_implemented)
        self.assertFalse(plan.workflow_routing_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.runtime_selection_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.memory_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_candidates_include_capability_lifecycle_and_escalation_posture(self) -> None:
        plan = optimize_agent_activation(route="generate")

        for candidate in plan.candidates:
            dumped = candidate.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CANDIDATE_FIELDS)
            self.assertEqual(
                candidate.serialization_version,
                "agent_activation_optimization_candidate.v1",
            )
            self.assertEqual(candidate.route_name, RouteName.GENERATE)
            self.assertTrue(candidate.source_capability_ids)
            self.assertEqual(
                candidate.source_lifecycle_profile_id,
                f"{candidate.agent_id}_lifecycle_profile",
            )
            self.assertGreaterEqual(candidate.activation_score, 0)
            self.assertIn("agent_invocation", candidate.blocked_runtime_behaviors)
            self.assertTrue(candidate.agent_activation_optimizer_implemented)
            self.assertTrue(candidate.agent_activation_recommendation_implemented)
            self.assertFalse(candidate.agent_activation_implemented)
            self.assertFalse(candidate.agent_instantiation_implemented)
            self.assertFalse(candidate.agent_invocation_implemented)
            self.assertFalse(candidate.lifecycle_transition_execution_implemented)
            self.assertFalse(candidate.workflow_routing_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.provider_execution_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.memory_mutation_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.advisory_only)

        planner = agent_activation_candidate_by_agent_id("planner_agent", plan)
        missing = agent_activation_candidate_by_agent_id("missing_agent", plan)
        gated = agent_activation_candidates_for_status("requires_hitl", plan)
        standby = agent_activation_candidates_for_status("standby", plan)
        self.assertIsNotNone(planner)
        self.assertIsNone(missing)
        assert planner is not None
        self.assertEqual(planner.activation_order, 1)
        self.assertEqual(planner.status, "requires_hitl")
        self.assertTrue(planner.hitl_required)
        self.assertEqual(len(gated), 3)
        self.assertEqual(len(standby), 9)

    def test_preview_route_limits_candidates_to_preview_capable_agents(self) -> None:
        plan = optimize_agent_activation(route="preview")
        agent_ids = tuple(candidate.agent_id for candidate in plan.candidates)

        self.assertIn("runtime_agent", agent_ids)
        self.assertIn("artifact_agent", agent_ids)
        self.assertIn("style_agent", agent_ids)
        self.assertNotIn("research_agent", agent_ids)
        self.assertLess(plan.candidate_count, 12)

    def test_plan_rejects_mismatched_candidate_totals(self) -> None:
        plan = optimize_agent_activation()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            AgentActivationOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["highest_activation_score"] += 1

        with self.assertRaisesRegex(ValueError, "highest_activation_score must match"):
            AgentActivationOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["activation_recommendation_count"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "activation_recommendation_count must match",
        ):
            AgentActivationOptimizationPlan(**payload)

    def test_optimizer_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Generate a p5.js sketch with an agentic plan.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = optimize_agent_activation(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_optimizer_does_not_declare_runtime_application_terms(self) -> None:
        plan = optimize_agent_activation(route=RouteName.GENERATE)
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
                        candidate.source_lifecycle_profile_id,
                        *candidate.source_capability_ids,
                        candidate.activation_reason,
                        candidate.fallback_summary,
                        *candidate.evidence,
                        *candidate.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "instantiate_agent(",
            "invoke_agent(",
            "activate_agent(",
            "run_lifecycle_transition(",
            "change_workflow_route(",
            "control_workflow(",
            "select_runtime(",
            "route_provider(",
            "execute_provider(",
            "trigger_retry(",
            "mutate_memory(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
