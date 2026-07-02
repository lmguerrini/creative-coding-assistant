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
    SelfEvolutionCoreSurfacePlan,
    build_self_evolution_core_surface,
    route_request,
    self_evolution_core_surface_plan_by_roadmap_item,
    self_evolution_core_surface_proposal_by_id,
    self_evolution_core_surface_proposals_for_confidence,
    self_evolution_core_surface_proposals_for_status,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.self_evolution_common import (
    BLOCKED_RUNTIME_BEHAVIORS,
    CROSS_CUTTING_CONTRACTS,
    UPSTREAM_CAPABILITIES,
    proposal_rank_score,
)
from creative_coding_assistant.orchestration.self_evolution_core_surface import (
    CORE_ROADMAP_ITEMS,
)


class SelfEvolutionCoreSurfaceTests(unittest.TestCase):
    def test_core_surface_aggregates_all_roadmap_item_plans(self) -> None:
        surface = build_self_evolution_core_surface(route=RouteName.GENERATE)

        self.assertEqual(surface.role, "self_evolution_core_surface")
        self.assertEqual(
            surface.serialization_version,
            "self_evolution_core_surface.v1",
        )
        self.assertEqual(surface.covered_roadmap_items, CORE_ROADMAP_ITEMS)
        self.assertEqual(surface.covered_roadmap_item_count, 22)
        self.assertEqual(surface.plan_count, 22)
        self.assertEqual(surface.proposal_count, 110)
        self.assertEqual(surface.guarded_proposal_count, 44)
        self.assertEqual(surface.review_required_proposal_count, 66)
        self.assertEqual(surface.hitl_required_proposal_count, 110)
        self.assertEqual(surface.upstream_capabilities, UPSTREAM_CAPABILITIES)
        self.assertEqual(surface.upstream_signal_source_count, 4)
        self.assertEqual(surface.upstream_signal_id_count, 20)
        self.assertEqual(surface.cross_cutting_contracts, CROSS_CUTTING_CONTRACTS)
        self.assertEqual(surface.blocked_runtime_behaviors, BLOCKED_RUNTIME_BEHAVIORS)
        self.assertFalse(surface.generated_evolution_report_ids)
        self.assertFalse(surface.applied_evolution_proposal_ids)
        self.assertFalse(surface.mutated_prompt_ids)
        self.assertFalse(surface.mutated_workflow_ids)
        self.assertFalse(surface.mutated_routing_ids)
        self.assertFalse(surface.mutated_memory_ids)
        self.assertFalse(surface.mutated_retrieval_ids)
        self.assertFalse(surface.written_storage_record_ids)
        self.assertFalse(surface.provider_execution_ids)
        self.assertFalse(surface.mutated_output_ids)
        self.assertTrue(surface.all_v6_signal_sources_integrated)
        self.assertTrue(surface.roadmap_traceability_implemented)
        self.assertTrue(surface.evolution_graph_metadata_implemented)
        self.assertTrue(surface.evolution_explainability_report_implemented)
        self.assertTrue(surface.proposal_impact_model_implemented)
        self.assertTrue(surface.cost_benefit_model_implemented)
        self.assertTrue(surface.risk_model_implemented)
        self.assertTrue(surface.rollback_strategy_model_implemented)
        self.assertTrue(surface.capability_ownership_boundary_check_implemented)
        self.assertTrue(surface.cross_capability_governance_check_implemented)
        self.assertFalse(surface.prompt_rewriting_implemented)
        self.assertFalse(surface.workflow_mutation_implemented)
        self.assertFalse(surface.routing_mutation_implemented)
        self.assertFalse(surface.provider_execution_implemented)
        self.assertFalse(surface.generated_output_mutation_implemented)
        self.assertFalse(surface.runtime_evolution_implemented)
        self.assertTrue(surface.advisory_only)

    def test_core_surface_exposes_ranked_explainable_proposals(self) -> None:
        surface = build_self_evolution_core_surface(route="generate")

        for proposal in surface.proposals:
            self.assertIn(proposal.roadmap_item, CORE_ROADMAP_ITEMS)
            self.assertTrue(proposal.why_proposal_exists)
            self.assertTrue(proposal.upstream_signal_explanation)
            self.assertTrue(proposal.downstream_impact_explanation)
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
            self.assertIn(
                "ownership_and_governance",
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

        prompt_plan = self_evolution_core_surface_plan_by_roadmap_item(
            "Prompt Evolution",
            surface,
        )
        self.assertIsNotNone(prompt_plan)
        proposal = self_evolution_core_surface_proposal_by_id(
            "prompt_evolution::prompt_quality_drift",
            surface,
        )
        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal.status, "guarded")
        self.assertEqual(
            len(self_evolution_core_surface_proposals_for_status("guarded", surface)),
            44,
        )
        self.assertEqual(
            len(
                self_evolution_core_surface_proposals_for_confidence(
                    "guarded",
                    surface,
                )
            ),
            44,
        )

    def test_core_surface_rejects_mismatched_or_mutating_payloads(self) -> None:
        surface = build_self_evolution_core_surface()
        payload = surface.model_dump(mode="json")
        payload["proposal_ids"] = ("missing",) + tuple(payload["proposal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "proposal_ids must match"):
            SelfEvolutionCoreSurfacePlan(**payload)

        payload = surface.model_dump(mode="json")
        payload["applied_evolution_proposal_ids"] = (surface.proposal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "core surface mutation ids must be empty",
        ):
            SelfEvolutionCoreSurfacePlan(**payload)

    def test_core_surface_does_not_change_routing_or_provider(self) -> None:
        request = AssistantRequest(
            query="Review the self evolution core surface.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        surface = build_self_evolution_core_surface(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(surface.upstream_capabilities, UPSTREAM_CAPABILITIES)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")


if __name__ == "__main__":
    unittest.main()
