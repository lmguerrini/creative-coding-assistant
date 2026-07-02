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
    PromptEvolutionPlan,
    build_prompt_evolution,
    prompt_evolution_proposal_by_id,
    prompt_evolution_proposals_for_confidence,
    prompt_evolution_proposals_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.self_evolution_common import (
    BLOCKED_RUNTIME_BEHAVIORS,
    CROSS_CUTTING_CONTRACTS,
    UPSTREAM_CAPABILITIES,
    proposal_rank_score,
)

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Prompt Evolution",)
EXPECTED_SOURCE_ROLES = (
    "adaptive_learning_engine",
    "creative_memory_core_surface",
    "knowledge_evolution_core_surface",
    "research_core_surface",
)


class PromptEvolutionTests(unittest.TestCase):
    def test_plan_builds_prompt_evolution_proposal_metadata(self) -> None:
        plan = build_prompt_evolution(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "prompt_evolution")
        self.assertEqual(plan.serialization_version, "prompt_evolution_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.covered_roadmap_items, EXPECTED_ROADMAP_ITEMS)
        self.assertEqual(plan.covered_roadmap_item_count, 1)
        self.assertEqual(plan.upstream_capabilities, UPSTREAM_CAPABILITIES)
        self.assertEqual(plan.upstream_signal_source_count, 4)
        self.assertEqual(plan.source_plan_roles, EXPECTED_SOURCE_ROLES)
        self.assertEqual(len(plan.source_plan_serialization_versions), 4)
        self.assertEqual(plan.upstream_signal_id_count, 20)
        self.assertEqual(len(plan.upstream_signal_ids), 20)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.proposal_count, 5)
        self.assertEqual(plan.guarded_proposal_count, 2)
        self.assertEqual(plan.review_required_proposal_count, 3)
        self.assertEqual(plan.high_confidence_proposal_count, 3)
        self.assertEqual(plan.hitl_required_proposal_count, 5)
        self.assertEqual(plan.cross_cutting_contracts, CROSS_CUTTING_CONTRACTS)
        self.assertEqual(plan.blocked_runtime_behaviors, BLOCKED_RUNTIME_BEHAVIORS)
        self.assertFalse(plan.generated_evolution_report_ids)
        self.assertFalse(plan.applied_evolution_proposal_ids)
        self.assertFalse(plan.mutated_prompt_ids)
        self.assertFalse(plan.mutated_workflow_ids)
        self.assertFalse(plan.mutated_routing_ids)
        self.assertFalse(plan.mutated_memory_ids)
        self.assertFalse(plan.mutated_retrieval_ids)
        self.assertFalse(plan.written_storage_record_ids)
        self.assertFalse(plan.provider_execution_ids)
        self.assertFalse(plan.mutated_output_ids)
        self.assertEqual(plan.overall_evolution_posture, "guarded")
        self.assertIn("does not apply Runtime Evolution", plan.authority_boundary)
        self.assertIn("rewrite prompts", plan.authority_boundary)
        self.assertTrue(plan.self_evolution_capability_implemented)
        self.assertTrue(plan.evolution_proposal_contract_implemented)
        self.assertTrue(plan.evolution_graph_metadata_implemented)
        self.assertTrue(plan.evolution_explainability_report_implemented)
        self.assertTrue(plan.proposal_impact_model_implemented)
        self.assertTrue(plan.cost_benefit_model_implemented)
        self.assertTrue(plan.risk_model_implemented)
        self.assertTrue(plan.rollback_strategy_model_implemented)
        self.assertTrue(plan.capability_ownership_boundary_check_implemented)
        self.assertTrue(plan.cross_capability_governance_check_implemented)
        self.assertTrue(plan.all_v6_signal_sources_integrated)
        self.assertFalse(plan.autonomous_runtime_evolution_implemented)
        self.assertFalse(plan.prompt_rewriting_implemented)
        self.assertFalse(plan.workflow_mutation_implemented)
        self.assertFalse(plan.routing_mutation_implemented)
        self.assertFalse(plan.memory_mutation_implemented)
        self.assertFalse(plan.retrieval_mutation_implemented)
        self.assertFalse(plan.storage_mutation_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_proposals_score_and_explain_without_prompt_rewriting(self) -> None:
        plan = build_prompt_evolution(route="generate")
        source_ids = set(plan.upstream_signal_source_ids)

        for proposal in plan.proposals:
            self.assertEqual(
                proposal.serialization_version,
                "prompt_evolution_proposal.v1",
            )
            self.assertEqual(proposal.route_name, RouteName.GENERATE)
            self.assertEqual(proposal.surface_role, "prompt_evolution")
            self.assertEqual(proposal.roadmap_item, "Prompt Evolution")
            self.assertEqual(
                proposal.proposal_id,
                f"prompt_evolution::{proposal.proposal_kind}",
            )
            self.assertEqual(set(proposal.upstream_signal_source_ids), source_ids)
            self.assertEqual(proposal.upstream_capabilities, UPSTREAM_CAPABILITIES)
            self.assertIn("prompt_templates", proposal.downstream_systems)
            self.assertEqual(
                proposal.proposal_rank_score,
                proposal_rank_score(
                    impact_score=proposal.impact_score,
                    cost_score=proposal.cost_score,
                    risk_score=proposal.risk_score,
                    confidence_score=proposal.confidence_score,
                    dependency_score=proposal.dependency_score,
                    rollback_feasibility_score=proposal.rollback_feasibility_score,
                ),
            )
            self.assertIn("V6.1 learning", proposal.upstream_signal_explanation)
            self.assertIn(
                "Downstream impact is advisory",
                proposal.downstream_impact_explanation,
            )
            self.assertEqual(proposal.cross_cutting_contracts, CROSS_CUTTING_CONTRACTS)
            self.assertIn("prompt_rewriting", proposal.blocked_runtime_behaviors)
            self.assertIn("workflow_mutation", proposal.blocked_runtime_behaviors)
            self.assertIn("provider_execution", proposal.blocked_runtime_behaviors)
            self.assertTrue(proposal.hitl_required_before_application)
            self.assertTrue(proposal.evolution_proposal_contract_implemented)
            self.assertTrue(proposal.evolution_explainability_report_implemented)
            self.assertTrue(proposal.proposal_impact_model_implemented)
            self.assertTrue(proposal.cost_benefit_model_implemented)
            self.assertTrue(proposal.risk_model_implemented)
            self.assertTrue(proposal.rollback_strategy_model_implemented)
            self.assertTrue(proposal.capability_ownership_boundary_check_implemented)
            self.assertTrue(proposal.cross_capability_governance_check_implemented)
            self.assertFalse(proposal.autonomous_runtime_evolution_implemented)
            self.assertFalse(proposal.prompt_rewriting_implemented)
            self.assertFalse(proposal.workflow_mutation_implemented)
            self.assertFalse(proposal.routing_mutation_implemented)
            self.assertFalse(proposal.memory_mutation_implemented)
            self.assertFalse(proposal.retrieval_mutation_implemented)
            self.assertFalse(proposal.storage_mutation_implemented)
            self.assertFalse(proposal.provider_execution_implemented)
            self.assertFalse(proposal.generated_output_mutation_implemented)
            self.assertTrue(proposal.advisory_only)

        prompt_quality = prompt_evolution_proposal_by_id(
            "prompt_evolution::prompt_quality_drift",
            plan,
        )
        self.assertIsNotNone(prompt_quality)
        assert prompt_quality is not None
        self.assertEqual(prompt_quality.status, "guarded")
        self.assertEqual(prompt_quality.confidence, "guarded")
        self.assertEqual(
            len(prompt_evolution_proposals_for_status("guarded", plan)),
            2,
        )
        self.assertEqual(
            len(prompt_evolution_proposals_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_prompt_evolution_metadata(self) -> None:
        plan = build_prompt_evolution()
        payload = plan.model_dump(mode="json")
        payload["proposal_ids"] = ("missing",) + tuple(payload["proposal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "proposal_ids must match"):
            PromptEvolutionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_proposal_rank_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_proposal_rank_score must match",
        ):
            PromptEvolutionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["mutated_prompt_ids"] = (plan.proposal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "runtime mutation and generated report ids must be empty",
        ):
            PromptEvolutionPlan(**payload)

    def test_prompt_evolution_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review prompt evolution suggestions.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_prompt_evolution(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_prompt_evolution_does_not_declare_active_mutation_terms(self) -> None:
        plan = build_prompt_evolution(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for proposal in plan.proposals
                    for field in (
                        proposal.why_proposal_exists,
                        proposal.upstream_signal_explanation,
                        proposal.downstream_impact_explanation,
                        *proposal.advisory_actions,
                    )
                ),
            )
        ).lower()

        forbidden_terms = (
            "apply runtime evolution now",
            "rewrite prompt now",
            "mutate workflow now",
            "mutate routing now",
            "write memory now",
            "execute provider now",
            "modify generated output now",
        )
        for forbidden in forbidden_terms:
            self.assertNotIn(forbidden, combined_text)


if __name__ == "__main__":
    unittest.main()
