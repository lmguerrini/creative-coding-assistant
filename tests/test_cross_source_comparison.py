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
    CrossSourceComparisonPlan,
    build_cross_source_comparison,
    build_paper_research,
    build_web_research,
    cross_source_comparison_entries_for_confidence,
    cross_source_comparison_entries_for_status,
    cross_source_comparison_entry_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Cross-source Comparison",)


class CrossSourceComparisonTests(unittest.TestCase):
    def test_plan_builds_advisory_cross_source_comparison_metadata(self) -> None:
        plan = build_cross_source_comparison(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "cross_source_comparison")
        self.assertEqual(
            plan.serialization_version,
            "cross_source_comparison_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.paper_research_role, "paper_research")
        self.assertEqual(plan.web_research_role, "web_research")
        self.assertEqual(plan.paper_research_entry_count, 5)
        self.assertEqual(plan.web_research_entry_count, 5)
        self.assertEqual(plan.covered_roadmap_items, EXPECTED_ROADMAP_ITEMS)
        self.assertEqual(plan.covered_roadmap_item_count, 1)
        self.assertEqual(plan.source_count, 57)
        self.assertEqual(plan.domain_count, 43)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.entry_count, 5)
        self.assertEqual(plan.candidate_entry_count, 1)
        self.assertEqual(plan.review_required_entry_count, 3)
        self.assertEqual(plan.guarded_entry_count, 1)
        self.assertEqual(plan.high_confidence_entry_count, 3)
        self.assertEqual(plan.hitl_required_entry_count, 5)
        self.assertFalse(plan.executed_comparison_ids)
        self.assertFalse(plan.compared_live_claim_ids)
        self.assertFalse(plan.detected_contradiction_ids)
        self.assertFalse(plan.scored_source_credibility_ids)
        self.assertFalse(plan.written_storage_record_ids)
        self.assertEqual(plan.overall_comparison_posture, "guarded")
        self.assertIn(
            "does not execute cross-source comparison",
            plan.authority_boundary,
        )
        self.assertTrue(plan.cross_source_comparison_capability_implemented)
        self.assertTrue(plan.cross_source_comparison_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.paper_research_metadata_used)
        self.assertTrue(plan.web_research_metadata_used)
        self.assertFalse(plan.cross_source_comparison_execution_implemented)
        self.assertFalse(plan.live_claim_comparison_implemented)
        self.assertFalse(plan.contradiction_detection_implemented)
        self.assertFalse(plan.source_credibility_scoring_implemented)
        self.assertFalse(plan.research_confidence_scoring_implemented)
        self.assertFalse(plan.external_source_fetch_implemented)
        self.assertFalse(plan.web_browsing_implemented)
        self.assertFalse(plan.paper_download_implemented)
        self.assertFalse(plan.kb_enrichment_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
        self.assertFalse(plan.retrieval_configuration_mutation_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.provider_provisioning_implemented)
        self.assertFalse(plan.api_key_inference_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_cross_source_comparison_without_execution(self) -> None:
        plan = build_cross_source_comparison(route="generate")
        paper_ids = set(plan.paper_research_entry_ids)
        web_ids = set(plan.web_research_entry_ids)

        for entry in plan.entries:
            self.assertEqual(
                entry.serialization_version,
                "cross_source_comparison_entry.v1",
            )
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.entry_id,
                f"cross_source_comparison::{entry.comparison_kind}",
            )
            self.assertTrue(set(entry.paper_research_entry_ids).issubset(paper_ids))
            self.assertTrue(set(entry.web_research_entry_ids).issubset(web_ids))
            self.assertEqual(entry.source_count, plan.source_count)
            self.assertEqual(entry.domain_count, plan.domain_count)
            self.assertEqual(
                entry.comparison_score,
                min(
                    1000,
                    max(
                        0,
                        entry.alignment_score * 3
                        + entry.provenance_score * 2
                        + entry.governance_alignment_score * 3
                        + entry.mutation_risk_score * 2
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("cross_source_comparison", entry.context_tags)
            self.assertIn(
                "cross_source_comparison_execution",
                entry.blocked_runtime_behaviors,
            )
            self.assertIn(
                "contradiction_detection_execution",
                entry.blocked_runtime_behaviors,
            )
            self.assertTrue(entry.hitl_required_before_comparison)
            self.assertTrue(entry.cross_source_comparison_capability_implemented)
            self.assertTrue(entry.cross_source_comparison_metadata_implemented)
            self.assertTrue(entry.paper_research_metadata_used)
            self.assertTrue(entry.web_research_metadata_used)
            self.assertFalse(entry.cross_source_comparison_execution_implemented)
            self.assertFalse(entry.live_claim_comparison_implemented)
            self.assertFalse(entry.contradiction_detection_implemented)
            self.assertFalse(entry.source_credibility_scoring_implemented)
            self.assertFalse(entry.research_confidence_scoring_implemented)
            self.assertFalse(entry.external_source_fetch_implemented)
            self.assertFalse(entry.web_browsing_implemented)
            self.assertFalse(entry.paper_download_implemented)
            self.assertFalse(entry.kb_storage_write_implemented)
            self.assertFalse(entry.retrieval_execution_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        alignment = cross_source_comparison_entry_by_id(
            "cross_source_comparison::paper_web_alignment_review",
            plan,
        )
        self.assertIsNotNone(alignment)
        assert alignment is not None
        self.assertEqual(alignment.status, "guarded")
        self.assertEqual(alignment.confidence, "guarded")
        self.assertEqual(
            len(cross_source_comparison_entries_for_status("review_required", plan)),
            3,
        )
        self.assertEqual(
            len(cross_source_comparison_entries_for_confidence("high", plan)),
            2,
        )

    def test_plan_rejects_mismatched_comparison_metadata(self) -> None:
        plan = build_cross_source_comparison()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            CrossSourceComparisonPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_comparison_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_comparison_score must match",
        ):
            CrossSourceComparisonPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["detected_contradiction_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "detected_contradiction_ids must remain empty",
        ):
            CrossSourceComparisonPlan(**payload)

    def test_comparison_composes_with_paper_and_web_metadata(self) -> None:
        paper = build_paper_research(route=RouteName.REVIEW)
        web = build_web_research(route=RouteName.REVIEW)
        plan = build_cross_source_comparison(
            route=RouteName.REVIEW,
            paper_research=paper,
            web_research=web,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.checked_at, paper.checked_at)
        self.assertEqual(plan.paper_research_entry_ids, paper.entry_ids)
        self.assertEqual(plan.web_research_entry_ids, web.entry_ids)
        self.assertTrue(plan.paper_research_metadata_used)
        self.assertTrue(plan.web_research_metadata_used)
        self.assertFalse(plan.cross_source_comparison_execution_implemented)

    def test_comparison_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Plan cross-source comparison for a p5.js shader study.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_cross_source_comparison(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_comparison_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_cross_source_comparison(route=RouteName.GENERATE)
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
                        entry.entry_id,
                        entry.comparison_kind,
                        entry.status,
                        entry.confidence,
                        entry.comparison_axis,
                        *entry.paper_research_entry_ids,
                        *entry.web_research_entry_ids,
                        entry.comparison_summary,
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
            "execute_cross_source_comparison(",
            "compare_live_claim(",
            "detect_contradiction(",
            "score_source_credibility(",
            "score_research_confidence(",
            "fetch_external_source(",
            "browse_web(",
            "download_paper(",
            "enrich_kb(",
            "write_kb_storage(",
            "mutate_source_registry(",
            "mutate_retrieval_config(",
            "execute_retrieval(",
            "mutate_ranking(",
            "provision_provider(",
            "infer_api_key(",
            "route_provider(",
            "execute_provider(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "write_storage(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
