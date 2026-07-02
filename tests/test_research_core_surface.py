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
    ResearchCoreSurfacePlan,
    build_research_core_surface,
    research_core_surface_entries_for_confidence,
    research_core_surface_entries_for_status,
    research_core_surface_entry_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_SOURCE_ROLES = (
    "research_planner",
    "research_decomposer",
    "paper_research",
    "web_research",
    "cross_source_comparison",
    "knowledge_distillation",
    "automatic_kb_enrichment",
    "research_reports",
    "research_memory",
    "source_validation_engine",
    "source_credibility_engine",
    "contradiction_detection",
    "research_confidence_engine",
    "research_gap_discovery",
    "research_recommendation_engine",
    "research_execution_policy",
    "research_hitl_policies",
    "creative_research_engine",
    "cross_domain_inspiration_discovery",
)
EXPECTED_ROADMAP_ITEMS = (
    "Research Planner",
    "Research Decomposer",
    "Paper Research",
    "Web Research",
    "Cross-source Comparison",
    "Knowledge Distillation",
    "Automatic KB Enrichment",
    "Research Reports",
    "Research Memory",
    "Source Validation Engine",
    "Source Credibility Engine",
    "Contradiction Detection",
    "Research Confidence Engine",
    "Research Gap Discovery",
    "Research Recommendation Engine",
    "Research Execution Policy",
    "Research HITL Policies",
    "Creative Research Engine",
    "Cross-domain Inspiration Discovery",
)


class ResearchCoreSurfaceTests(unittest.TestCase):
    def test_plan_builds_core_surface_metadata(self) -> None:
        plan = build_research_core_surface(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "research_core_surface")
        self.assertEqual(
            plan.serialization_version,
            "research_core_surface_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.source_plan_roles, EXPECTED_SOURCE_ROLES)
        self.assertEqual(len(plan.source_plan_serialization_versions), 19)
        self.assertEqual(plan.source_item_count, 95)
        self.assertEqual(len(plan.source_item_ids), 95)
        self.assertEqual(plan.covered_roadmap_items, EXPECTED_ROADMAP_ITEMS)
        self.assertEqual(plan.covered_roadmap_item_count, 19)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.entry_count, 5)
        self.assertEqual(plan.candidate_entry_count, 1)
        self.assertEqual(plan.review_required_entry_count, 2)
        self.assertEqual(plan.guarded_entry_count, 2)
        self.assertEqual(plan.high_confidence_entry_count, 3)
        self.assertEqual(plan.hitl_required_entry_count, 5)
        self.assertFalse(plan.activated_core_surface_ids)
        self.assertFalse(plan.executed_research_ids)
        self.assertFalse(plan.fetched_source_ids)
        self.assertFalse(plan.written_kb_record_ids)
        self.assertFalse(plan.mutated_workflow_ids)
        self.assertFalse(plan.mutated_output_ids)
        self.assertEqual(plan.highest_core_surface_score, 918)
        self.assertEqual(plan.overall_core_surface_score, 859)
        self.assertEqual(plan.overall_core_surface_posture, "guarded")
        self.assertIn("does not activate core surfaces", plan.authority_boundary)
        self.assertIn("write KB storage", plan.authority_boundary)
        self.assertTrue(plan.core_surface_implemented)
        self.assertTrue(plan.core_surface_metadata_implemented)
        self.assertTrue(plan.all_roadmap_items_traceable)
        self.assertTrue(plan.all_sources_metadata_only)
        self.assertFalse(plan.core_surface_activation_implemented)
        self.assertFalse(plan.research_execution_implemented)
        self.assertFalse(plan.research_plan_mutation_implemented)
        self.assertFalse(plan.research_task_creation_implemented)
        self.assertFalse(plan.paper_research_execution_implemented)
        self.assertFalse(plan.web_research_execution_implemented)
        self.assertFalse(plan.external_source_fetch_implemented)
        self.assertFalse(plan.web_browsing_implemented)
        self.assertFalse(plan.paper_download_implemented)
        self.assertFalse(plan.kb_enrichment_execution_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
        self.assertFalse(plan.research_report_generation_implemented)
        self.assertFalse(plan.research_memory_write_implemented)
        self.assertFalse(plan.research_recommendation_generation_implemented)
        self.assertFalse(plan.research_execution_policy_application_implemented)
        self.assertFalse(plan.hitl_request_emission_implemented)
        self.assertFalse(plan.inspiration_discovery_execution_implemented)
        self.assertFalse(plan.live_cross_domain_search_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_core_surface_without_activation(self) -> None:
        plan = build_research_core_surface(route="generate")
        source_items = set(plan.source_item_ids)
        source_roles = set(plan.source_plan_roles)

        for entry in plan.entries:
            self.assertEqual(
                entry.serialization_version,
                "research_core_surface_entry.v1",
            )
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.core_surface_id,
                f"research_core::{entry.surface_kind}",
            )
            self.assertEqual(entry.roadmap_item_count, len(entry.roadmap_items))
            self.assertEqual(entry.source_item_count, len(entry.source_item_ids))
            self.assertTrue(set(entry.source_item_ids).issubset(source_items))
            self.assertTrue(set(entry.source_plan_roles).issubset(source_roles))
            self.assertEqual(
                entry.core_surface_score,
                min(
                    1000,
                    max(
                        0,
                        entry.surface_coverage_score * 3
                        + entry.source_traceability_score * 3
                        + entry.governance_alignment_score * 2
                        + entry.activation_risk_score * 2
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("research_core_surface", entry.context_tags)
            self.assertIn("core_surface_activation", entry.blocked_runtime_behaviors)
            self.assertIn("research_execution", entry.blocked_runtime_behaviors)
            self.assertTrue(entry.hitl_required_before_core_surface_activation)
            self.assertTrue(entry.core_surface_implemented)
            self.assertTrue(entry.core_surface_metadata_implemented)
            self.assertTrue(entry.all_roadmap_items_traceable)
            self.assertTrue(entry.all_sources_metadata_only)
            self.assertFalse(entry.core_surface_activation_implemented)
            self.assertFalse(entry.research_execution_implemented)
            self.assertFalse(entry.external_source_fetch_implemented)
            self.assertFalse(entry.kb_enrichment_execution_implemented)
            self.assertFalse(entry.kb_storage_write_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.workflow_control_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        planning = research_core_surface_entry_by_id(
            "research_core::planning_acquisition_surface",
            plan,
        )
        self.assertIsNotNone(planning)
        assert planning is not None
        self.assertEqual(planning.status, "guarded")
        self.assertEqual(planning.confidence, "guarded")
        self.assertEqual(
            len(research_core_surface_entries_for_status("guarded", plan)),
            2,
        )
        self.assertEqual(
            len(research_core_surface_entries_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_core_surface_metadata(self) -> None:
        plan = build_research_core_surface()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            ResearchCoreSurfacePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_core_surface_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_core_surface_score must match",
        ):
            ResearchCoreSurfacePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["activated_core_surface_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "activated_core_surface_ids must remain empty",
        ):
            ResearchCoreSurfacePlan(**payload)

    def test_core_surface_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review the autonomous research core surface.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_research_core_surface(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_core_surface_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_research_core_surface(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *plan.covered_roadmap_items,
                *(
                    field
                    for entry in plan.entries
                    for field in (
                        entry.core_surface_id,
                        entry.surface_kind,
                        entry.status,
                        entry.confidence,
                        entry.surface_axis,
                        *entry.roadmap_items,
                        *entry.source_plan_roles,
                        *entry.source_serialization_versions,
                        *entry.source_item_ids,
                        entry.surface_summary,
                        *entry.context_tags,
                        *entry.explainability_notes,
                        *entry.advisory_actions,
                        *entry.evidence,
                        *entry.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "activate_core_surface(",
            "execute_research(",
            "mutate_research_plan(",
            "create_research_task(",
            "execute_paper_research(",
            "execute_web_research(",
            "fetch_external_source(",
            "browse_web(",
            "download_paper(",
            "execute_cross_source_comparison(",
            "execute_knowledge_distillation(",
            "execute_kb_enrichment(",
            "write_kb_storage(",
            "generate_research_report(",
            "write_research_memory(",
            "execute_source_validation(",
            "score_source_credibility(",
            "execute_contradiction_detection(",
            "score_research_confidence(",
            "discover_research_gap(",
            "generate_research_recommendation(",
            "apply_research_execution_policy(",
            "emit_hitl_request(",
            "apply_hitl_decision(",
            "execute_inspiration_discovery(",
            "perform_live_cross_domain_search(",
            "route_provider(",
            "execute_provider(",
            "invoke_agent(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
