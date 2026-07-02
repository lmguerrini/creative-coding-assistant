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
    SelfEvolutionAdvisoryReportEntry,
    SelfEvolutionSecondarySurfacePlan,
    build_self_evolution_core_surface,
    build_self_evolution_secondary_surface,
    route_request,
    self_evolution_secondary_surface_report_by_id,
    self_evolution_secondary_surface_report_by_roadmap_item,
    self_evolution_secondary_surface_reports_for_confidence,
    self_evolution_secondary_surface_reports_for_status,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.self_evolution_common import (
    BLOCKED_RUNTIME_BEHAVIORS,
    CROSS_CUTTING_CONTRACTS,
    UPSTREAM_CAPABILITIES,
)
from creative_coding_assistant.orchestration.self_evolution_core_surface import (
    CORE_ROADMAP_ITEMS,
)
from creative_coding_assistant.orchestration.self_evolution_secondary_surface import (
    SECONDARY_REPORT_SECTIONS,
)


class SelfEvolutionSecondarySurfaceTests(unittest.TestCase):
    def test_secondary_surface_composes_core_surface_report_metadata(self) -> None:
        surface = build_self_evolution_secondary_surface(route=RouteName.GENERATE)

        self.assertEqual(surface.role, "self_evolution_secondary_surface")
        self.assertEqual(
            surface.serialization_version,
            "self_evolution_secondary_surface.v1",
        )
        self.assertEqual(
            surface.source_core_surface_role,
            "self_evolution_core_surface",
        )
        self.assertEqual(surface.source_core_surface_plan_count, 22)
        self.assertEqual(surface.source_core_surface_proposal_count, 110)
        self.assertEqual(surface.covered_roadmap_items, CORE_ROADMAP_ITEMS)
        self.assertEqual(surface.covered_roadmap_item_count, 22)
        self.assertEqual(surface.report_entry_count, 22)
        self.assertEqual(surface.proposal_count, 110)
        self.assertEqual(surface.guarded_proposal_count, 44)
        self.assertEqual(surface.review_required_proposal_count, 66)
        self.assertEqual(surface.hitl_required_proposal_count, 110)
        self.assertEqual(surface.high_confidence_proposal_count, 66)
        self.assertEqual(surface.top_proposal_rank_score, 966)
        self.assertEqual(surface.overall_proposal_rank_score, 1000)
        self.assertEqual(surface.overall_evolution_posture, "guarded")
        self.assertEqual(surface.upstream_capabilities, UPSTREAM_CAPABILITIES)
        self.assertEqual(surface.upstream_signal_source_count, 4)
        self.assertEqual(surface.upstream_signal_id_count, 20)
        self.assertEqual(surface.report_sections, SECONDARY_REPORT_SECTIONS)
        self.assertEqual(surface.cross_cutting_contracts, CROSS_CUTTING_CONTRACTS)
        self.assertEqual(surface.blocked_runtime_behaviors, BLOCKED_RUNTIME_BEHAVIORS)
        self.assertFalse(surface.generated_report_artifact_ids)
        self.assertFalse(surface.written_storage_record_ids)
        self.assertFalse(surface.applied_evolution_proposal_ids)
        self.assertFalse(surface.executed_rollback_ids)
        self.assertFalse(surface.mutated_prompt_ids)
        self.assertFalse(surface.mutated_workflow_ids)
        self.assertFalse(surface.mutated_routing_ids)
        self.assertFalse(surface.mutated_memory_ids)
        self.assertFalse(surface.mutated_retrieval_ids)
        self.assertFalse(surface.provider_execution_ids)
        self.assertFalse(surface.mutated_output_ids)
        self.assertTrue(surface.secondary_surface_implemented)
        self.assertTrue(surface.advisory_evolution_report_metadata_implemented)
        self.assertTrue(surface.roadmap_traceability_implemented)
        self.assertTrue(surface.all_v6_signal_sources_integrated)
        self.assertTrue(surface.evolution_proposal_contract_implemented)
        self.assertTrue(surface.evolution_graph_metadata_implemented)
        self.assertTrue(surface.evolution_explainability_report_implemented)
        self.assertTrue(surface.proposal_impact_model_implemented)
        self.assertTrue(surface.cost_benefit_model_implemented)
        self.assertTrue(surface.risk_model_implemented)
        self.assertTrue(surface.rollback_strategy_model_implemented)
        self.assertTrue(surface.capability_ownership_boundary_check_implemented)
        self.assertTrue(surface.cross_capability_governance_check_implemented)
        self.assertFalse(surface.report_artifact_generation_implemented)
        self.assertFalse(surface.storage_write_implemented)
        self.assertFalse(surface.proposal_application_implemented)
        self.assertFalse(surface.rollback_execution_implemented)
        self.assertFalse(surface.prompt_rewriting_implemented)
        self.assertFalse(surface.workflow_mutation_implemented)
        self.assertFalse(surface.routing_mutation_implemented)
        self.assertFalse(surface.memory_mutation_implemented)
        self.assertFalse(surface.retrieval_mutation_implemented)
        self.assertFalse(surface.provider_execution_implemented)
        self.assertFalse(surface.generated_output_mutation_implemented)
        self.assertFalse(surface.runtime_evolution_implemented)
        self.assertTrue(surface.advisory_only)

    def test_report_entries_remain_traceable_and_explainable(self) -> None:
        surface = build_self_evolution_secondary_surface(route="generate")

        for entry in surface.report_entries:
            self.assertIn(entry.roadmap_item, CORE_ROADMAP_ITEMS)
            self.assertEqual(
                entry.source_core_surface_role,
                "self_evolution_core_surface",
            )
            self.assertEqual(entry.proposal_count, 5)
            self.assertEqual(entry.guarded_proposal_count, 2)
            self.assertEqual(entry.review_required_proposal_count, 3)
            self.assertEqual(entry.hitl_required_proposal_count, 5)
            self.assertEqual(entry.top_proposal_rank_score, 966)
            self.assertEqual(entry.upstream_capabilities, UPSTREAM_CAPABILITIES)
            self.assertEqual(entry.report_sections, SECONDARY_REPORT_SECTIONS)
            self.assertTrue(entry.why_report_exists)
            self.assertIn("V6.1 learning", entry.upstream_signal_explanation)
            self.assertIn("without mutating", entry.downstream_impact_explanation)
            self.assertTrue(entry.ownership_boundary_checks)
            self.assertTrue(entry.governance_checks)
            self.assertFalse(entry.generated_report_artifact_ids)
            self.assertFalse(entry.written_storage_record_ids)
            self.assertFalse(entry.applied_evolution_proposal_ids)
            self.assertFalse(entry.executed_rollback_ids)
            self.assertFalse(entry.provider_execution_ids)
            self.assertFalse(entry.mutated_output_ids)
            self.assertFalse(entry.report_artifact_generation_implemented)
            self.assertFalse(entry.storage_write_implemented)
            self.assertFalse(entry.proposal_application_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        prompt_report = self_evolution_secondary_surface_report_by_roadmap_item(
            "Prompt Evolution",
            surface,
        )
        self.assertIsNotNone(prompt_report)
        assert prompt_report is not None
        self.assertEqual(prompt_report.plan_role, "prompt_evolution")
        self.assertEqual(
            prompt_report.top_proposal_id,
            "prompt_evolution::prompt_quality_drift",
        )
        self.assertIs(
            self_evolution_secondary_surface_report_by_id(
                "self_evolution_advisory_report::prompt_evolution",
                surface,
            ),
            prompt_report,
        )
        self.assertEqual(
            len(
                self_evolution_secondary_surface_reports_for_status(
                    "guarded",
                    surface,
                )
            ),
            22,
        )
        self.assertEqual(
            len(
                self_evolution_secondary_surface_reports_for_confidence(
                    "guarded",
                    surface,
                )
            ),
            22,
        )

    def test_secondary_surface_rejects_mismatched_or_mutating_payloads(self) -> None:
        surface = build_self_evolution_secondary_surface()
        payload = surface.model_dump(mode="json")
        payload["report_entry_ids"] = ("missing",) + tuple(
            payload["report_entry_ids"][1:]
        )

        with self.assertRaisesRegex(ValueError, "report_entry_ids must match"):
            SelfEvolutionSecondarySurfacePlan(**payload)

        payload = surface.model_dump(mode="json")
        payload["generated_report_artifact_ids"] = ("report_artifact",)

        with self.assertRaisesRegex(
            ValueError,
            "secondary surface mutation ids must be empty",
        ):
            SelfEvolutionSecondarySurfacePlan(**payload)

        entry_payload = surface.report_entries[0].model_dump(mode="json")
        entry_payload["written_storage_record_ids"] = ("storage_record",)

        with self.assertRaisesRegex(
            ValueError,
            "secondary report mutation ids must be empty",
        ):
            SelfEvolutionAdvisoryReportEntry(**entry_payload)

    def test_secondary_surface_does_not_change_routing_or_provider(self) -> None:
        request = AssistantRequest(
            query="Review the self evolution secondary surface.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        core_surface = build_self_evolution_core_surface(route=RouteName.GENERATE)
        surface = build_self_evolution_secondary_surface(core_surface)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(surface.source_core_surface_proposal_count, 110)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")


if __name__ == "__main__":
    unittest.main()
