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
    ResearchSecondarySurfacePlan,
    build_research_secondary_surface,
    research_secondary_surface_entries_for_confidence,
    research_secondary_surface_entries_for_status,
    research_secondary_surface_entry_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_SOURCE_ROLES = (
    "research_core_surface",
    "adaptive_learning_engine",
    "adaptive_execution_policy_engine",
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
REQUIRED_ENTRY_FIELDS = {
    "secondary_surface_id",
    "surface_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "surface_axis",
    "roadmap_items",
    "roadmap_item_count",
    "source_plan_roles",
    "source_serialization_versions",
    "source_item_ids",
    "source_item_count",
    "surface_summary",
    "roadmap_traceability_score",
    "source_composition_score",
    "governance_alignment_score",
    "v5_v6_foundation_score",
    "activation_risk_score",
    "governance_weight",
    "secondary_surface_score",
    "hitl_required_before_secondary_surface_activation",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "secondary_surface_implemented",
    "secondary_surface_metadata_implemented",
    "all_roadmap_items_traceable",
    "v5_policy_foundation_used",
    "v6_learning_foundation_used",
    "v6_research_core_foundation_used",
    "all_sources_metadata_only",
    "secondary_surface_activation_implemented",
    "adaptive_learning_application_implemented",
    "adaptive_execution_policy_application_implemented",
    "research_execution_implemented",
    "research_plan_mutation_implemented",
    "research_task_creation_implemented",
    "paper_research_execution_implemented",
    "web_research_execution_implemented",
    "external_source_fetch_implemented",
    "web_browsing_implemented",
    "paper_download_implemented",
    "cross_source_comparison_execution_implemented",
    "knowledge_distillation_execution_implemented",
    "kb_enrichment_execution_implemented",
    "kb_storage_write_implemented",
    "research_report_generation_implemented",
    "research_memory_write_implemented",
    "source_validation_execution_implemented",
    "source_credibility_scoring_execution_implemented",
    "contradiction_detection_execution_implemented",
    "research_confidence_scoring_execution_implemented",
    "research_gap_discovery_execution_implemented",
    "research_recommendation_generation_implemented",
    "research_execution_policy_application_implemented",
    "hitl_request_emission_implemented",
    "hitl_decision_application_implemented",
    "inspiration_discovery_execution_implemented",
    "live_cross_domain_search_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class ResearchSecondarySurfaceTests(unittest.TestCase):
    def test_plan_builds_secondary_surface_metadata(self) -> None:
        plan = build_research_secondary_surface(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "research_secondary_surface")
        self.assertEqual(
            plan.serialization_version,
            "research_secondary_surface_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.source_plan_roles, EXPECTED_SOURCE_ROLES)
        self.assertEqual(
            plan.source_plan_serialization_versions,
            (
                "research_core_surface_plan.v1",
                "adaptive_learning_plan.v1",
                "adaptive_execution_policy_plan.v1",
            ),
        )
        self.assertEqual(plan.source_item_count, 15)
        self.assertEqual(len(plan.source_item_ids), 15)
        self.assertEqual(plan.covered_roadmap_items, EXPECTED_ROADMAP_ITEMS)
        self.assertEqual(plan.covered_roadmap_item_count, 19)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.entry_count, 5)
        self.assertEqual(plan.candidate_entry_count, 1)
        self.assertEqual(plan.review_required_entry_count, 2)
        self.assertEqual(plan.guarded_entry_count, 2)
        self.assertEqual(plan.high_confidence_entry_count, 3)
        self.assertEqual(plan.hitl_required_entry_count, 5)
        self.assertFalse(plan.activated_secondary_surface_ids)
        self.assertFalse(plan.applied_learning_ids)
        self.assertFalse(plan.applied_policy_ids)
        self.assertFalse(plan.executed_research_ids)
        self.assertFalse(plan.fetched_source_ids)
        self.assertFalse(plan.written_kb_record_ids)
        self.assertFalse(plan.mutated_workflow_ids)
        self.assertFalse(plan.mutated_output_ids)
        self.assertEqual(plan.highest_secondary_surface_score, 867)
        self.assertEqual(plan.overall_secondary_surface_score, 830)
        self.assertEqual(plan.overall_secondary_surface_posture, "guarded")
        self.assertIn("V5 controlled execution policy", plan.authority_boundary)
        self.assertIn("does not activate secondary surfaces", plan.authority_boundary)
        self.assertTrue(plan.secondary_surface_implemented)
        self.assertTrue(plan.secondary_surface_metadata_implemented)
        self.assertTrue(plan.all_roadmap_items_traceable)
        self.assertTrue(plan.v5_policy_foundation_used)
        self.assertTrue(plan.v6_learning_foundation_used)
        self.assertTrue(plan.v6_research_core_foundation_used)
        self.assertTrue(plan.all_sources_metadata_only)
        self.assertFalse(plan.secondary_surface_activation_implemented)
        self.assertFalse(plan.adaptive_learning_application_implemented)
        self.assertFalse(plan.adaptive_execution_policy_application_implemented)
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
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_secondary_surface_without_activation(self) -> None:
        plan = build_research_secondary_surface(route="generate")
        source_items = set(plan.source_item_ids)
        source_roles = set(plan.source_plan_roles)

        for entry in plan.entries:
            dumped = entry.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_ENTRY_FIELDS)
            self.assertEqual(
                entry.serialization_version,
                "research_secondary_surface_entry.v1",
            )
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.secondary_surface_id,
                f"research_secondary::{entry.surface_kind}",
            )
            self.assertEqual(entry.roadmap_item_count, len(entry.roadmap_items))
            self.assertEqual(entry.source_item_count, len(entry.source_item_ids))
            self.assertTrue(set(entry.source_item_ids).issubset(source_items))
            self.assertTrue(set(entry.source_plan_roles).issubset(source_roles))
            self.assertEqual(
                entry.secondary_surface_score,
                min(
                    1000,
                    max(
                        0,
                        entry.roadmap_traceability_score * 2
                        + entry.source_composition_score * 2
                        + entry.governance_alignment_score * 2
                        + entry.v5_v6_foundation_score * 2
                        + entry.activation_risk_score
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("research_secondary_surface", entry.context_tags)
            self.assertIn(
                "secondary_surface_activation",
                entry.blocked_runtime_behaviors,
            )
            self.assertIn("external_source_fetch", entry.blocked_runtime_behaviors)
            self.assertIn("kb_storage_write", entry.blocked_runtime_behaviors)
            self.assertTrue(entry.hitl_required_before_secondary_surface_activation)
            self.assertTrue(entry.secondary_surface_implemented)
            self.assertTrue(entry.secondary_surface_metadata_implemented)
            self.assertTrue(entry.all_roadmap_items_traceable)
            self.assertTrue(entry.v5_policy_foundation_used)
            self.assertTrue(entry.v6_learning_foundation_used)
            self.assertTrue(entry.v6_research_core_foundation_used)
            self.assertTrue(entry.all_sources_metadata_only)
            self.assertFalse(entry.secondary_surface_activation_implemented)
            self.assertFalse(entry.adaptive_learning_application_implemented)
            self.assertFalse(entry.adaptive_execution_policy_application_implemented)
            self.assertFalse(entry.research_execution_implemented)
            self.assertFalse(entry.external_source_fetch_implemented)
            self.assertFalse(entry.web_browsing_implemented)
            self.assertFalse(entry.paper_download_implemented)
            self.assertFalse(entry.kb_enrichment_execution_implemented)
            self.assertFalse(entry.kb_storage_write_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.workflow_control_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        operational = research_secondary_surface_entry_by_id(
            "research_secondary::research_operational_support_surface",
            plan,
        )
        self.assertIsNotNone(operational)
        assert operational is not None
        self.assertEqual(operational.status, "guarded")
        self.assertEqual(operational.confidence, "guarded")
        self.assertEqual(
            len(research_secondary_surface_entries_for_status("guarded", plan)),
            2,
        )
        self.assertEqual(
            len(research_secondary_surface_entries_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_secondary_surface_metadata(self) -> None:
        plan = build_research_secondary_surface()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            ResearchSecondarySurfacePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_secondary_surface_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_secondary_surface_score must match",
        ):
            ResearchSecondarySurfacePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["activated_secondary_surface_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "activated_secondary_surface_ids must remain empty",
        ):
            ResearchSecondarySurfacePlan(**payload)

    def test_secondary_surface_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review the autonomous research secondary surface.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_research_secondary_surface(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_secondary_surface_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_research_secondary_surface(route=RouteName.GENERATE)
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
                        entry.secondary_surface_id,
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
            "activate_secondary_surface(",
            "apply_adaptive_learning(",
            "apply_execution_policy(",
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
            "mutate_prompt(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
