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
    RoutingEvolutionPoliciesPlan,
    build_routing_evolution_policies,
    route_request,
    routing_evolution_policies_proposal_by_id,
    routing_evolution_policies_proposals_for_confidence,
    routing_evolution_policies_proposals_for_status,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.self_evolution_common import (
    CROSS_CUTTING_CONTRACTS,
    UPSTREAM_CAPABILITIES,
    proposal_rank_score,
)

EXPECTED_ROADMAP_ITEMS = ("Routing Evolution Policies",)


class RoutingEvolutionPoliciesTests(unittest.TestCase):
    def test_plan_builds_routing_policy_metadata(self) -> None:
        plan = build_routing_evolution_policies(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "routing_evolution_policies")
        self.assertEqual(
            plan.serialization_version,
            "routing_evolution_policies_plan.v1",
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
        self.assertFalse(plan.applied_evolution_proposal_ids)
        self.assertFalse(plan.mutated_workflow_ids)
        self.assertFalse(plan.mutated_routing_ids)
        self.assertFalse(plan.provider_execution_ids)
        self.assertTrue(plan.evolution_graph_metadata_implemented)
        self.assertTrue(plan.evolution_explainability_report_implemented)
        self.assertTrue(plan.proposal_impact_model_implemented)
        self.assertTrue(plan.cost_benefit_model_implemented)
        self.assertTrue(plan.risk_model_implemented)
        self.assertTrue(plan.rollback_strategy_model_implemented)
        self.assertTrue(plan.capability_ownership_boundary_check_implemented)
        self.assertTrue(plan.cross_capability_governance_check_implemented)
        self.assertFalse(plan.routing_mutation_implemented)
        self.assertFalse(plan.workflow_mutation_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_proposals_score_routing_policies_without_routing_mutation(
        self,
    ) -> None:
        plan = build_routing_evolution_policies(route="generate")

        for proposal in plan.proposals:
            self.assertEqual(
                proposal.serialization_version,
                "routing_evolution_policies_proposal.v1",
            )
            self.assertEqual(proposal.surface_role, "routing_evolution_policies")
            self.assertEqual(proposal.roadmap_item, "Routing Evolution Policies")
            self.assertIn("routing_intelligence", proposal.downstream_systems)
            self.assertIn("routing_explainability", proposal.downstream_systems)
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
            self.assertIn("proposal_ranking", proposal.evolution_report_sections)
            self.assertTrue(proposal.hitl_required_before_application)
            self.assertFalse(proposal.routing_mutation_implemented)
            self.assertFalse(proposal.workflow_mutation_implemented)
            self.assertFalse(proposal.storage_mutation_implemented)
            self.assertFalse(proposal.provider_execution_implemented)
            self.assertFalse(proposal.generated_output_mutation_implemented)
            self.assertFalse(proposal.runtime_evolution_implemented)

        signal = routing_evolution_policies_proposal_by_id(
            "routing_evolution_policies::routing_policy_signal_alignment",
            plan,
        )
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.status, "guarded")
        self.assertEqual(
            len(
                routing_evolution_policies_proposals_for_status(
                    "review_required",
                    plan,
                )
            ),
            3,
        )
        self.assertEqual(
            len(
                routing_evolution_policies_proposals_for_confidence(
                    "guarded",
                    plan,
                )
            ),
            2,
        )

    def test_plan_rejects_mismatched_routing_policy_metadata(self) -> None:
        plan = build_routing_evolution_policies()
        payload = plan.model_dump(mode="json")
        payload["proposal_ids"] = ("missing",) + tuple(payload["proposal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "proposal_ids must match"):
            RoutingEvolutionPoliciesPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["mutated_routing_ids"] = (plan.proposal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "runtime mutation and generated report ids must be empty",
        ):
            RoutingEvolutionPoliciesPlan(**payload)

    def test_routing_policies_do_not_change_routing_or_provider(self) -> None:
        request = AssistantRequest(
            query="Review routing evolution policy proposals.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_routing_evolution_policies(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")


if __name__ == "__main__":
    unittest.main()
