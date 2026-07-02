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
    RollbackStrategyGeneratorPlan,
    build_rollback_strategy_generator,
    rollback_strategy_generator_proposal_by_id,
    rollback_strategy_generator_proposals_for_confidence,
    rollback_strategy_generator_proposals_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.self_evolution_common import (
    CROSS_CUTTING_CONTRACTS,
    UPSTREAM_CAPABILITIES,
    proposal_rank_score,
)

EXPECTED_ROADMAP_ITEMS = ("Rollback Strategy Generator",)


class RollbackStrategyGeneratorTests(unittest.TestCase):
    def test_plan_builds_rollback_strategy_metadata(self) -> None:
        plan = build_rollback_strategy_generator(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "rollback_strategy_generator")
        self.assertEqual(
            plan.serialization_version,
            "rollback_strategy_generator_plan.v1",
        )
        self.assertEqual(plan.covered_roadmap_items, EXPECTED_ROADMAP_ITEMS)
        self.assertEqual(plan.upstream_capabilities, UPSTREAM_CAPABILITIES)
        self.assertEqual(plan.upstream_signal_source_count, 4)
        self.assertEqual(plan.upstream_signal_id_count, 20)
        self.assertEqual(plan.proposal_count, 5)
        self.assertEqual(plan.guarded_proposal_count, 2)
        self.assertEqual(plan.review_required_proposal_count, 3)
        self.assertEqual(plan.hitl_required_proposal_count, 5)
        self.assertEqual(plan.cross_cutting_contracts, CROSS_CUTTING_CONTRACTS)
        self.assertFalse(plan.generated_evolution_report_ids)
        self.assertFalse(plan.applied_evolution_proposal_ids)
        self.assertFalse(plan.mutated_prompt_ids)
        self.assertFalse(plan.mutated_workflow_ids)
        self.assertFalse(plan.mutated_routing_ids)
        self.assertFalse(plan.written_storage_record_ids)
        self.assertFalse(plan.provider_execution_ids)
        self.assertFalse(plan.mutated_output_ids)
        self.assertTrue(plan.evolution_graph_metadata_implemented)
        self.assertTrue(plan.evolution_explainability_report_implemented)
        self.assertTrue(plan.proposal_impact_model_implemented)
        self.assertTrue(plan.cost_benefit_model_implemented)
        self.assertTrue(plan.risk_model_implemented)
        self.assertTrue(plan.rollback_strategy_model_implemented)
        self.assertTrue(plan.capability_ownership_boundary_check_implemented)
        self.assertTrue(plan.cross_capability_governance_check_implemented)
        self.assertFalse(plan.prompt_rewriting_implemented)
        self.assertFalse(plan.workflow_mutation_implemented)
        self.assertFalse(plan.routing_mutation_implemented)
        self.assertFalse(plan.storage_mutation_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_proposals_score_rollback_without_rollback_execution(
        self,
    ) -> None:
        plan = build_rollback_strategy_generator(route="generate")

        for proposal in plan.proposals:
            self.assertEqual(
                proposal.serialization_version,
                "rollback_strategy_generator_proposal.v1",
            )
            self.assertEqual(proposal.surface_role, "rollback_strategy_generator")
            self.assertEqual(proposal.roadmap_item, "Rollback Strategy Generator")
            self.assertIn("knowledge_rollback", proposal.downstream_systems)
            self.assertIn("workflow_replay_engine", proposal.downstream_systems)
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
            self.assertIn(
                f"rollback_feasibility_score:{proposal.rollback_feasibility_score}",
                proposal.evidence,
            )
            self.assertIn(
                "rollback_feasibility",
                proposal.evolution_report_sections,
            )
            self.assertTrue(proposal.hitl_required_before_application)
            self.assertFalse(proposal.prompt_rewriting_implemented)
            self.assertFalse(proposal.workflow_mutation_implemented)
            self.assertFalse(proposal.routing_mutation_implemented)
            self.assertFalse(proposal.storage_mutation_implemented)
            self.assertFalse(proposal.provider_execution_implemented)
            self.assertFalse(proposal.generated_output_mutation_implemented)
            self.assertFalse(proposal.runtime_evolution_implemented)

        signal = rollback_strategy_generator_proposal_by_id(
            "rollback_strategy_generator::rollback_signal_alignment",
            plan,
        )
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.status, "guarded")
        self.assertEqual(
            len(
                rollback_strategy_generator_proposals_for_status(
                    "review_required",
                    plan,
                )
            ),
            3,
        )
        self.assertEqual(
            len(
                rollback_strategy_generator_proposals_for_confidence(
                    "guarded",
                    plan,
                )
            ),
            2,
        )

    def test_plan_rejects_mismatched_rollback_strategy_metadata(self) -> None:
        plan = build_rollback_strategy_generator()
        payload = plan.model_dump(mode="json")
        payload["proposal_ids"] = ("missing",) + tuple(payload["proposal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "proposal_ids must match"):
            RollbackStrategyGeneratorPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["generated_evolution_report_ids"] = (plan.proposal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "runtime mutation and generated report ids must be empty",
        ):
            RollbackStrategyGeneratorPlan(**payload)

    def test_rollback_strategy_does_not_change_routing_or_provider(self) -> None:
        request = AssistantRequest(
            query="Review rollback strategy proposals.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_rollback_strategy_generator(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")


if __name__ == "__main__":
    unittest.main()
