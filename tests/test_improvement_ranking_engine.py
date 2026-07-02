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
    ImprovementRankingEnginePlan,
    build_improvement_ranking_engine,
    improvement_ranking_engine_proposal_by_id,
    improvement_ranking_engine_proposals_for_confidence,
    improvement_ranking_engine_proposals_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.self_evolution_common import (
    CROSS_CUTTING_CONTRACTS,
    UPSTREAM_CAPABILITIES,
    overall_proposal_rank_score,
    proposal_rank_score,
)

EXPECTED_ROADMAP_ITEMS = ("Improvement Ranking Engine",)


class ImprovementRankingEngineTests(unittest.TestCase):
    def test_plan_builds_improvement_ranking_metadata(self) -> None:
        plan = build_improvement_ranking_engine(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "improvement_ranking_engine")
        self.assertEqual(
            plan.serialization_version,
            "improvement_ranking_engine_plan.v1",
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
        self.assertEqual(
            plan.highest_proposal_rank_score,
            max(proposal.proposal_rank_score for proposal in plan.proposals),
        )
        self.assertEqual(
            plan.overall_proposal_rank_score,
            overall_proposal_rank_score(plan.proposals),
        )
        self.assertEqual(plan.overall_evolution_posture, "guarded")
        self.assertFalse(plan.applied_evolution_proposal_ids)
        self.assertFalse(plan.mutated_prompt_ids)
        self.assertFalse(plan.mutated_workflow_ids)
        self.assertFalse(plan.mutated_routing_ids)
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
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_proposals_rank_improvements_without_proposal_application(
        self,
    ) -> None:
        plan = build_improvement_ranking_engine(route="generate")

        for proposal in plan.proposals:
            self.assertEqual(
                proposal.serialization_version,
                "improvement_ranking_engine_proposal.v1",
            )
            self.assertEqual(proposal.surface_role, "improvement_ranking_engine")
            self.assertEqual(proposal.roadmap_item, "Improvement Ranking Engine")
            self.assertIn("self_evolution_common", proposal.downstream_systems)
            self.assertIn("quality_trends", proposal.downstream_systems)
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
                "proposal_ranking",
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

        signal = improvement_ranking_engine_proposal_by_id(
            "improvement_ranking_engine::cross_capability_signal_ranking",
            plan,
        )
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.status, "guarded")
        self.assertEqual(
            len(
                improvement_ranking_engine_proposals_for_status(
                    "review_required",
                    plan,
                )
            ),
            3,
        )
        self.assertEqual(
            len(
                improvement_ranking_engine_proposals_for_confidence(
                    "guarded",
                    plan,
                )
            ),
            2,
        )

    def test_plan_rejects_mismatched_improvement_ranking_metadata(self) -> None:
        plan = build_improvement_ranking_engine()
        payload = plan.model_dump(mode="json")
        payload["overall_proposal_rank_score"] = 0

        with self.assertRaisesRegex(
            ValueError,
            "overall_proposal_rank_score must match",
        ):
            ImprovementRankingEnginePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_evolution_proposal_ids"] = (plan.proposal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "runtime mutation and generated report ids must be empty",
        ):
            ImprovementRankingEnginePlan(**payload)

    def test_improvement_ranking_does_not_change_routing_or_provider(self) -> None:
        request = AssistantRequest(
            query="Review improvement ranking proposals.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_improvement_ranking_engine(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")


if __name__ == "__main__":
    unittest.main()
