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
    EscalationOptimizationPlan,
    escalation_optimization_decision_by_id,
    escalation_optimization_decisions_for_posture,
    optimize_escalation_policy,
    route_request,
)

REQUIRED_DECISION_IDS = (
    "escalation_optimizer::policy_diagnostics",
    "escalation_optimizer::signal_thresholds",
    "escalation_optimizer::budget_review",
    "escalation_optimizer::hybrid_availability",
    "escalation_optimizer::execution_mode_review",
    "escalation_optimizer::adaptive_boundary",
)
REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_DECISION_FIELDS = {
    "decision_id",
    "decision_kind",
    "source_surface_id",
    "source_serialization_version",
    "task_type",
    "execution_mode_id",
    "posture",
    "priority_rank",
    "escalation_score",
    "guardrail_signal_count",
    "hitl_required",
    "budget_gate_status",
    "risk_band",
    "unavailable_reason_codes",
    "reason_summary",
    "suggested_action",
    "fallback_summary",
    "evidence",
    "blocked_runtime_behaviors",
    "escalation_optimizer_implemented",
    "escalation_recommendation_implemented",
    "policy_evaluation_implemented",
    "escalation_triggering_implemented",
    "escalation_execution_implemented",
    "human_review_request_implemented",
    "hitl_request_emitted",
    "execution_blocking_implemented",
    "budget_enforcement_implemented",
    "agent_invocation_implemented",
    "multi_agent_orchestration_implemented",
    "runtime_selection_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "trace_capture_implemented",
    "trace_emission_implemented",
    "memory_write_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class AdaptiveEscalationOptimizerTests(unittest.TestCase):
    def test_plan_links_escalation_budget_and_hybrid_sources(self) -> None:
        plan = optimize_escalation_policy(task_type="creative_coding")

        self.assertEqual(plan.role, "escalation_optimizer")
        self.assertEqual(
            plan.serialization_version,
            "adaptive_escalation_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_escalation_diagnostics_serialization_version,
            "escalation_diagnostics.v1",
        )
        self.assertEqual(
            plan.source_hitl_budget_gate_serialization_version,
            "hitl_budget_gate_plan.v1",
        )
        self.assertEqual(
            plan.source_hybrid_workflow_serialization_version,
            "adaptive_hybrid_workflow_optimization_plan.v1",
        )
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.decision_ids, REQUIRED_DECISION_IDS)
        self.assertEqual(plan.decision_count, 6)
        self.assertEqual(
            plan.highest_priority_decision_id,
            "escalation_optimizer::hybrid_availability",
        )
        self.assertEqual(plan.optimized_escalation_posture, "requires_hitl")
        self.assertGreaterEqual(plan.hitl_required_decision_count, 1)
        self.assertIn("does not evaluate policy", plan.authority_boundary)
        self.assertTrue(plan.escalation_optimizer_implemented)
        self.assertTrue(plan.adaptive_escalation_policy_metadata_implemented)
        self.assertFalse(plan.policy_evaluation_implemented)
        self.assertFalse(plan.escalation_triggering_implemented)
        self.assertFalse(plan.escalation_execution_implemented)
        self.assertFalse(plan.human_review_request_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.execution_blocking_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.trace_capture_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_decisions_rank_review_posture_without_triggering(self) -> None:
        plan = optimize_escalation_policy(task_type="creative_coding")

        for decision in plan.decisions:
            dumped = decision.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_DECISION_FIELDS)
            self.assertEqual(
                decision.serialization_version,
                "adaptive_escalation_optimization_decision.v1",
            )
            self.assertIn(decision.execution_mode_id, REQUIRED_EXECUTION_MODES)
            self.assertIn("policy_evaluation", decision.blocked_runtime_behaviors)
            self.assertTrue(decision.escalation_optimizer_implemented)
            self.assertTrue(decision.escalation_recommendation_implemented)
            self.assertFalse(decision.policy_evaluation_implemented)
            self.assertFalse(decision.escalation_triggering_implemented)
            self.assertFalse(decision.escalation_execution_implemented)
            self.assertFalse(decision.human_review_request_implemented)
            self.assertFalse(decision.hitl_request_emitted)
            self.assertFalse(decision.execution_blocking_implemented)
            self.assertFalse(decision.budget_enforcement_implemented)
            self.assertFalse(decision.agent_invocation_implemented)
            self.assertFalse(decision.multi_agent_orchestration_implemented)
            self.assertFalse(decision.runtime_selection_implemented)
            self.assertFalse(decision.provider_model_routing_implemented)
            self.assertFalse(decision.provider_execution_implemented)
            self.assertFalse(decision.workflow_control_implemented)
            self.assertFalse(decision.retry_triggering_implemented)
            self.assertFalse(decision.trace_capture_implemented)
            self.assertFalse(decision.trace_emission_implemented)
            self.assertFalse(decision.memory_write_implemented)
            self.assertFalse(decision.persistent_storage_write_implemented)
            self.assertFalse(decision.generated_output_mutation_implemented)
            self.assertTrue(decision.advisory_only)

        hybrid = escalation_optimization_decision_by_id(
            "escalation_optimizer::hybrid_availability",
            plan,
        )
        budget = escalation_optimization_decision_by_id(
            "escalation_optimizer::budget_review",
            plan,
        )
        required = escalation_optimization_decisions_for_posture(
            "requires_hitl",
            plan,
        )
        review = escalation_optimization_decisions_for_posture(
            "review_recommended",
            plan,
        )
        self.assertIsNotNone(hybrid)
        self.assertIsNotNone(budget)
        assert hybrid is not None
        assert budget is not None
        self.assertEqual(hybrid.posture, "requires_hitl")
        self.assertTrue(hybrid.hitl_required)
        self.assertIn("missing_api_key", hybrid.unavailable_reason_codes)
        self.assertIn("local_runtime_unavailable", hybrid.unavailable_reason_codes)
        self.assertEqual(budget.budget_gate_status, "review_recommended")
        self.assertTrue(required)
        self.assertTrue(review)

    def test_auto_mode_keeps_hitl_boundary_for_risky_candidates(self) -> None:
        plan = optimize_escalation_policy(
            task_type="fast_draft",
            execution_mode_id="auto_mode",
        )
        mode_decision = escalation_optimization_decision_by_id(
            "escalation_optimizer::execution_mode_review",
            plan,
        )

        self.assertIsNotNone(mode_decision)
        assert mode_decision is not None
        self.assertEqual(mode_decision.execution_mode_id, "auto_mode")
        self.assertEqual(mode_decision.posture, "requires_hitl")
        self.assertTrue(mode_decision.hitl_required)
        self.assertEqual(plan.optimized_escalation_posture, "requires_hitl")
        self.assertFalse(plan.hitl_request_emitted)

    def test_plan_rejects_mismatched_decisions_or_posture(self) -> None:
        plan = optimize_escalation_policy()
        payload = plan.model_dump(mode="json")
        payload["decision_ids"] = ("missing",) + tuple(payload["decision_ids"][1:])

        with self.assertRaisesRegex(ValueError, "decision_ids must match"):
            EscalationOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["highest_escalation_score"] += 1

        with self.assertRaisesRegex(ValueError, "highest_escalation_score must match"):
            EscalationOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["optimized_escalation_posture"] = "no_review_required"

        with self.assertRaisesRegex(
            ValueError,
            "optimized_escalation_posture must match",
        ):
            EscalationOptimizationPlan(**payload)

    def test_optimizer_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review the generated shader for visual quality.",
            mode=AssistantMode.REVIEW,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = optimize_escalation_policy(task_type="creative_coding")
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_optimizer_does_not_declare_runtime_application_terms(self) -> None:
        plan = optimize_escalation_policy(task_type="creative_coding")
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for decision in plan.decisions
                    for field in (
                        decision.decision_id,
                        decision.source_surface_id,
                        decision.reason_summary,
                        decision.suggested_action,
                        decision.fallback_summary,
                        *decision.evidence,
                        *decision.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "evaluate_policy(",
            "trigger_escalation(",
            "execute_escalation(",
            "emit_hitl_request(",
            "request_human_review(",
            "block_execution(",
            "enforce_budget(",
            "invoke_agent(",
            "orchestrate_agents(",
            "select_runtime(",
            "route_provider(",
            "execute_provider(",
            "control_workflow(",
            "trigger_retry(",
            "capture_trace(",
            "emit_trace(",
            "write_memory(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
