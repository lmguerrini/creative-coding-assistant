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
    PaperResearchPlan,
    build_paper_research,
    build_research_decomposer,
    paper_research_entries_for_confidence,
    paper_research_entries_for_status,
    paper_research_entry_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Paper Research",)


class PaperResearchTests(unittest.TestCase):
    def test_plan_builds_advisory_paper_research_metadata(self) -> None:
        plan = build_paper_research(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "paper_research")
        self.assertEqual(plan.serialization_version, "paper_research_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.research_decomposer_role, "research_decomposer")
        self.assertEqual(
            plan.research_decomposer_serialization_version,
            "research_decomposer_plan.v1",
        )
        self.assertEqual(plan.decomposition_entry_count, 5)
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
        self.assertFalse(plan.executed_paper_research_ids)
        self.assertFalse(plan.searched_external_paper_ids)
        self.assertFalse(plan.downloaded_paper_ids)
        self.assertFalse(plan.parsed_pdf_ids)
        self.assertFalse(plan.extracted_citation_ids)
        self.assertFalse(plan.written_storage_record_ids)
        self.assertEqual(plan.overall_paper_research_posture, "guarded")
        self.assertIn("does not execute paper research", plan.authority_boundary)
        self.assertIn("download papers", plan.authority_boundary)
        self.assertTrue(plan.paper_research_capability_implemented)
        self.assertTrue(plan.paper_research_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.research_decomposer_metadata_used)
        self.assertFalse(plan.paper_research_execution_implemented)
        self.assertFalse(plan.external_paper_search_implemented)
        self.assertFalse(plan.paper_download_implemented)
        self.assertFalse(plan.external_source_fetch_implemented)
        self.assertFalse(plan.pdf_parse_execution_implemented)
        self.assertFalse(plan.citation_extraction_execution_implemented)
        self.assertFalse(plan.live_source_validation_implemented)
        self.assertFalse(plan.source_credibility_scoring_implemented)
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

    def test_entries_score_paper_research_without_execution(self) -> None:
        plan = build_paper_research(route="generate")
        decomposition_ids = set(plan.decomposition_entry_ids)

        for entry in plan.entries:
            self.assertEqual(entry.serialization_version, "paper_research_entry.v1")
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.entry_id,
                f"paper_research::{entry.paper_research_kind}",
            )
            self.assertEqual(
                entry.decomposition_entry_count,
                len(entry.decomposition_entry_ids),
            )
            self.assertTrue(
                set(entry.decomposition_entry_ids).issubset(decomposition_ids)
            )
            self.assertEqual(entry.source_count, plan.source_count)
            self.assertEqual(entry.domain_count, plan.domain_count)
            self.assertEqual(
                entry.paper_research_score,
                min(
                    1000,
                    max(
                        0,
                        entry.literature_scope_score * 3
                        + entry.citation_strategy_score * 2
                        + entry.governance_alignment_score * 3
                        + entry.mutation_risk_score * 2
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("paper_research", entry.context_tags)
            self.assertIn("paper_research_execution", entry.blocked_runtime_behaviors)
            self.assertIn("paper_download", entry.blocked_runtime_behaviors)
            self.assertIn("kb_storage_write", entry.blocked_runtime_behaviors)
            self.assertTrue(entry.hitl_required_before_paper_research)
            self.assertTrue(entry.paper_research_capability_implemented)
            self.assertTrue(entry.paper_research_metadata_implemented)
            self.assertTrue(entry.research_decomposer_metadata_used)
            self.assertFalse(entry.paper_research_execution_implemented)
            self.assertFalse(entry.external_paper_search_implemented)
            self.assertFalse(entry.paper_download_implemented)
            self.assertFalse(entry.pdf_parse_execution_implemented)
            self.assertFalse(entry.citation_extraction_execution_implemented)
            self.assertFalse(entry.external_source_fetch_implemented)
            self.assertFalse(entry.live_source_validation_implemented)
            self.assertFalse(entry.source_credibility_scoring_implemented)
            self.assertFalse(entry.kb_enrichment_implemented)
            self.assertFalse(entry.kb_storage_write_implemented)
            self.assertFalse(entry.retrieval_execution_implemented)
            self.assertFalse(entry.provider_provisioning_implemented)
            self.assertFalse(entry.api_key_inference_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        scope = paper_research_entry_by_id(
            "paper_research::literature_scope_planning",
            plan,
        )
        self.assertIsNotNone(scope)
        assert scope is not None
        self.assertEqual(scope.status, "guarded")
        self.assertEqual(scope.confidence, "guarded")
        self.assertEqual(
            len(paper_research_entries_for_status("review_required", plan)),
            3,
        )
        self.assertEqual(len(paper_research_entries_for_confidence("high", plan)), 2)

    def test_plan_rejects_mismatched_paper_research_metadata(self) -> None:
        plan = build_paper_research()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            PaperResearchPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_paper_research_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_paper_research_score must match",
        ):
            PaperResearchPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["downloaded_paper_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "downloaded_paper_ids must remain empty",
        ):
            PaperResearchPlan(**payload)

    def test_paper_research_composes_with_decomposer_metadata(self) -> None:
        decomposer = build_research_decomposer(route=RouteName.REVIEW)
        plan = build_paper_research(
            route=RouteName.REVIEW,
            research_decomposer=decomposer,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.checked_at, decomposer.checked_at)
        self.assertEqual(plan.decomposition_entry_ids, decomposer.entry_ids)
        self.assertEqual(plan.source_count, decomposer.source_count)
        self.assertTrue(plan.research_decomposer_metadata_used)
        self.assertFalse(plan.paper_research_execution_implemented)

    def test_paper_research_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Plan paper research for a p5.js shader study.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_paper_research(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_paper_research_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_paper_research(route=RouteName.GENERATE)
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
                        entry.paper_research_kind,
                        entry.status,
                        entry.confidence,
                        entry.paper_axis,
                        *entry.decomposition_entry_ids,
                        entry.paper_research_summary,
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
            "execute_paper_research(",
            "search_external_papers(",
            "download_paper(",
            "fetch_external_source(",
            "parse_pdf(",
            "extract_citation(",
            "validate_source_live(",
            "score_source_credibility(",
            "enrich_kb(",
            "write_kb_storage(",
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
