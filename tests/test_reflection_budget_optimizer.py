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
    ReflectionBudgetOptimizationPlan,
    optimize_reflection_budget,
    reflection_budget_candidate_by_id,
    reflection_budget_candidates_for_status,
    route_request,
)

REQUIRED_CANDIDATE_FIELDS = {
    "candidate_id",
    "budget_kind",
    "status",
    "source_reasoning_recommendation_id",
    "source_reasoning_budget_kind",
    "source_reasoning_budget_status",
    "source_workflow_risk_factor_id",
    "source_workflow_risk_severity",
    "reflection_priority",
    "reflection_depth",
    "reflection_hitl_recommendation",
    "expected_quality_gain",
    "expected_risk_reduction",
    "expected_cost",
    "expected_latency",
    "source_reasoning_tokens",
    "source_reserve_tokens",
    "source_reasoning_pressure_score",
    "source_workflow_risk_score",
    "advisory_reflection_tokens",
    "advisory_reflection_reserve_tokens",
    "depth_weight",
    "quality_gain_weight",
    "risk_reduction_weight",
    "workflow_risk_weight",
    "guardrail_penalty",
    "reflection_budget_score",
    "recommended_reflection_pass_count",
    "applied_reflection_pass_count",
    "hitl_required",
    "budget_summary",
    "fallback_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "reflection_budget_optimizer_implemented",
    "reflection_budget_metadata_implemented",
    "reasoning_budget_metadata_used",
    "reflection_loop_metadata_used",
    "workflow_risk_metadata_used",
    "hitl_posture_metadata_used",
    "reflection_budget_enforcement_implemented",
    "runtime_reflection_token_allocation_implemented",
    "runtime_reasoning_token_allocation_implemented",
    "reflection_loop_execution_implemented",
    "refinement_triggering_implemented",
    "retry_triggering_implemented",
    "budget_enforcement_implemented",
    "context_trimming_implemented",
    "prompt_compression_implemented",
    "memory_summarization_implemented",
    "hitl_request_emitted",
    "provider_model_routing_implemented",
    "model_selection_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class ReflectionBudgetOptimizerTests(unittest.TestCase):
    def test_plan_combines_reasoning_reflection_and_risk_sources(self) -> None:
        plan = optimize_reflection_budget()

        self.assertEqual(plan.role, "reflection_budget_optimizer")
        self.assertEqual(
            plan.serialization_version,
            "reflection_budget_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_reasoning_budget_serialization_version,
            "reasoning_budget_optimization_plan.v1",
        )
        self.assertEqual(plan.source_reflection_loop_serialization_version, "v1")
        self.assertEqual(
            plan.source_workflow_risk_serialization_version,
            "workflow_risk_plan.v1",
        )
        self.assertEqual(plan.reflection_priority, "high")
        self.assertEqual(plan.reflection_depth, "moderate")
        self.assertEqual(plan.reflection_hitl_recommendation, "recommended")
        self.assertEqual(plan.candidate_count, 4)
        self.assertEqual(plan.recommended_candidate_count, 2)
        self.assertEqual(plan.reserve_guardrail_candidate_count, 1)
        self.assertEqual(plan.review_guardrail_candidate_count, 1)
        self.assertEqual(plan.hitl_required_candidate_count, 4)
        self.assertFalse(plan.applied_reflection_budget_candidate_ids)
        self.assertEqual(plan.total_advisory_reflection_tokens, 2880)
        self.assertEqual(plan.total_advisory_reflection_reserve_tokens, 2533)
        self.assertEqual(plan.total_recommended_reflection_pass_count, 2)
        self.assertEqual(plan.total_applied_reflection_pass_count, 0)
        self.assertEqual(plan.highest_reflection_budget_score, 573)
        self.assertEqual(plan.reflection_budget_pressure, "guarded")
        self.assertIn("does not enforce reflection budgets", plan.authority_boundary)
        self.assertTrue(plan.reflection_budget_optimizer_implemented)
        self.assertTrue(plan.reflection_budget_metadata_implemented)
        self.assertTrue(plan.reasoning_budget_metadata_used)
        self.assertTrue(plan.reflection_loop_metadata_used)
        self.assertTrue(plan.workflow_risk_metadata_used)
        self.assertTrue(plan.hitl_posture_metadata_used)
        self.assertFalse(plan.reflection_budget_enforcement_implemented)
        self.assertFalse(plan.runtime_reflection_token_allocation_implemented)
        self.assertFalse(plan.runtime_reasoning_token_allocation_implemented)
        self.assertFalse(plan.reflection_loop_execution_implemented)
        self.assertFalse(plan.refinement_triggering_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.context_trimming_implemented)
        self.assertFalse(plan.prompt_compression_implemented)
        self.assertFalse(plan.memory_summarization_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.model_selection_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.node_handler_invocation_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_candidates_score_reflection_budget_without_applying_passes(self) -> None:
        plan = optimize_reflection_budget()

        for candidate in plan.candidates:
            dumped = candidate.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CANDIDATE_FIELDS)
            self.assertEqual(
                candidate.serialization_version,
                "reflection_budget_optimization_candidate.v1",
            )
            self.assertEqual(
                candidate.candidate_id,
                f"reflection_budget_optimizer::{candidate.budget_kind}",
            )
            expected_status = {
                "optimization_candidate": "recommended",
                "reserve_guardrail": "reserve_guardrail",
                "review_guardrail": "review_guardrail",
            }[candidate.source_reasoning_budget_status]
            self.assertEqual(candidate.status, expected_status)
            if candidate.status == "recommended":
                expected_tokens = (
                    candidate.source_reasoning_tokens // 2 + candidate.depth_weight * 5
                )
            else:
                expected_tokens = 0
            if candidate.status == "review_guardrail":
                expected_reserve = 0
            elif candidate.status == "reserve_guardrail":
                expected_reserve = (
                    candidate.source_reserve_tokens // 2 + candidate.depth_weight * 5
                )
            else:
                expected_reserve = candidate.source_reserve_tokens // 3
            self.assertEqual(candidate.advisory_reflection_tokens, expected_tokens)
            self.assertEqual(
                candidate.advisory_reflection_reserve_tokens,
                expected_reserve,
            )
            self.assertEqual(candidate.depth_weight, 80)
            self.assertEqual(candidate.quality_gain_weight, 50)
            self.assertEqual(candidate.risk_reduction_weight, 50)
            self.assertEqual(
                candidate.workflow_risk_weight,
                min(250, candidate.source_workflow_risk_score // 8),
            )
            expected_penalty = 0
            if candidate.status == "reserve_guardrail":
                expected_penalty += 80
            elif candidate.status == "review_guardrail":
                expected_penalty += 120
            if candidate.source_reasoning_budget_status in {
                "reserve_guardrail",
                "review_guardrail",
            }:
                expected_penalty += 40
            if candidate.source_workflow_risk_severity == "guarded":
                expected_penalty += 60
            elif candidate.source_workflow_risk_severity == "high":
                expected_penalty += 30
            if candidate.reflection_hitl_recommendation in {
                "recommended",
                "required",
            }:
                expected_penalty += 30
            self.assertEqual(candidate.guardrail_penalty, min(260, expected_penalty))
            self.assertEqual(
                candidate.reflection_budget_score,
                min(
                    600,
                    max(
                        0,
                        candidate.source_reasoning_pressure_score
                        + candidate.workflow_risk_weight
                        + candidate.depth_weight
                        + candidate.quality_gain_weight
                        + candidate.risk_reduction_weight
                        - candidate.guardrail_penalty,
                    ),
                ),
            )
            self.assertEqual(candidate.applied_reflection_pass_count, 0)
            self.assertIn(
                "reflection_budget_enforcement",
                candidate.blocked_runtime_behaviors,
            )
            self.assertTrue(candidate.reflection_budget_optimizer_implemented)
            self.assertTrue(candidate.reasoning_budget_metadata_used)
            self.assertTrue(candidate.reflection_loop_metadata_used)
            self.assertTrue(candidate.workflow_risk_metadata_used)
            self.assertFalse(candidate.reflection_budget_enforcement_implemented)
            self.assertFalse(candidate.runtime_reflection_token_allocation_implemented)
            self.assertFalse(candidate.runtime_reasoning_token_allocation_implemented)
            self.assertFalse(candidate.reflection_loop_execution_implemented)
            self.assertFalse(candidate.refinement_triggering_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.budget_enforcement_implemented)
            self.assertFalse(candidate.context_trimming_implemented)
            self.assertFalse(candidate.prompt_compression_implemented)
            self.assertFalse(candidate.memory_summarization_implemented)
            self.assertFalse(candidate.hitl_request_emitted)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.model_selection_implemented)
            self.assertFalse(candidate.workflow_control_implemented)
            self.assertFalse(candidate.workflow_graph_mutation_implemented)
            self.assertFalse(candidate.workflow_execution_implemented)
            self.assertFalse(candidate.agent_invocation_implemented)
            self.assertFalse(candidate.node_handler_invocation_implemented)
            self.assertFalse(candidate.prompt_mutation_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.hitl_required)
            self.assertTrue(candidate.advisory_only)

        loop_budget = reflection_budget_candidate_by_id(
            "reflection_budget_optimizer::loop_depth_budget",
            plan,
        )
        recommended = reflection_budget_candidates_for_status("recommended", plan)
        review = reflection_budget_candidates_for_status("review_guardrail", plan)
        self.assertIsNotNone(loop_budget)
        assert loop_budget is not None
        self.assertEqual(loop_budget.status, "recommended")
        self.assertEqual(loop_budget.recommended_reflection_pass_count, 1)
        self.assertEqual(len(recommended), 2)
        self.assertEqual(len(review), 1)
        self.assertTrue(
            all(
                candidate.recommended_reflection_pass_count == 0 for candidate in review
            )
        )

    def test_plan_rejects_mismatched_reflection_budget_metadata(self) -> None:
        plan = optimize_reflection_budget()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            ReflectionBudgetOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["total_advisory_reflection_tokens"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "total_advisory_reflection_tokens must match",
        ):
            ReflectionBudgetOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_reflection_budget_candidate_ids"] = (plan.candidate_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_reflection_budget_candidate_ids must remain empty",
        ):
            ReflectionBudgetOptimizationPlan(**payload)

    def test_optimizer_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Optimize reflection budget for a visual workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = optimize_reflection_budget()
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.role, "reflection_budget_optimizer")
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_optimizer_does_not_declare_runtime_reflection_budget_terms(self) -> None:
        plan = optimize_reflection_budget()
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
                        candidate.budget_kind,
                        candidate.source_reasoning_recommendation_id,
                        candidate.source_reasoning_budget_kind,
                        candidate.source_workflow_risk_factor_id,
                        candidate.source_workflow_risk_severity,
                        candidate.reflection_priority,
                        candidate.reflection_depth,
                        candidate.expected_quality_gain,
                        candidate.expected_risk_reduction,
                        candidate.expected_cost,
                        candidate.expected_latency,
                        candidate.budget_summary,
                        candidate.fallback_summary,
                        *candidate.advisory_actions,
                        *candidate.evidence,
                        *candidate.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "enforce_reflection_budget(",
            "allocate_reflection_tokens(",
            "allocate_reasoning_tokens(",
            "execute_reflection_loop(",
            "trigger_refinement(",
            "trigger_retry(",
            "enforce_budget(",
            "trim_context(",
            "compress_prompt(",
            "summarize_memory(",
            "emit_hitl_request(",
            "route_provider(",
            "select_model(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
            "invoke_agent(",
            "call_node_handler(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
