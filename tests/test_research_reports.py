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
    ResearchReportPlan,
    build_automatic_kb_enrichment,
    build_research_reports,
    research_report_entries_for_confidence,
    research_report_entries_for_status,
    research_report_entry_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Research Reports",)


class ResearchReportTests(unittest.TestCase):
    def test_plan_builds_advisory_research_report_metadata(self) -> None:
        plan = build_research_reports(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "research_reports")
        self.assertEqual(plan.serialization_version, "research_report_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.automatic_kb_enrichment_role, "automatic_kb_enrichment")
        self.assertEqual(plan.enrichment_entry_count, 5)
        self.assertEqual(plan.covered_roadmap_items, EXPECTED_ROADMAP_ITEMS)
        self.assertEqual(plan.covered_roadmap_item_count, 1)
        self.assertEqual(plan.source_count, 57)
        self.assertEqual(plan.domain_count, 43)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.entry_count, 5)
        self.assertEqual(plan.candidate_entry_count, 1)
        self.assertEqual(plan.review_required_entry_count, 1)
        self.assertEqual(plan.guarded_entry_count, 3)
        self.assertEqual(plan.high_confidence_entry_count, 4)
        self.assertEqual(plan.hitl_required_entry_count, 5)
        self.assertFalse(plan.generated_report_ids)
        self.assertFalse(plan.generated_report_output_ids)
        self.assertFalse(plan.exported_report_file_ids)
        self.assertFalse(plan.written_report_storage_record_ids)
        self.assertFalse(plan.mutated_generated_output_ids)
        self.assertEqual(plan.overall_report_score, 820)
        self.assertEqual(plan.overall_report_posture, "guarded")
        self.assertIn("does not generate research reports", plan.authority_boundary)
        self.assertTrue(plan.research_reports_capability_implemented)
        self.assertTrue(plan.research_reports_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.automatic_kb_enrichment_metadata_used)
        self.assertFalse(plan.research_report_generation_implemented)
        self.assertFalse(plan.report_output_generation_implemented)
        self.assertFalse(plan.file_export_generation_implemented)
        self.assertFalse(plan.report_storage_write_implemented)
        self.assertFalse(plan.automatic_kb_enrichment_execution_implemented)
        self.assertFalse(plan.kb_enrichment_execution_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
        self.assertFalse(plan.provenance_record_write_implemented)
        self.assertFalse(plan.external_source_fetch_implemented)
        self.assertFalse(plan.web_browsing_implemented)
        self.assertFalse(plan.paper_download_implemented)
        self.assertFalse(plan.source_validation_execution_implemented)
        self.assertFalse(plan.source_credibility_scoring_implemented)
        self.assertFalse(plan.contradiction_detection_implemented)
        self.assertFalse(plan.research_confidence_scoring_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_reports_without_generation(self) -> None:
        plan = build_research_reports(route="generate")
        enrichment_ids = set(plan.enrichment_entry_ids)

        for entry in plan.entries:
            self.assertEqual(entry.serialization_version, "research_report_entry.v1")
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.entry_id,
                f"research_reports::{entry.report_kind}",
            )
            self.assertTrue(set(entry.enrichment_entry_ids).issubset(enrichment_ids))
            self.assertEqual(entry.source_count, plan.source_count)
            self.assertEqual(entry.domain_count, plan.domain_count)
            self.assertEqual(
                entry.report_score,
                min(
                    1000,
                    max(
                        0,
                        entry.scope_readiness_score * 2
                        + entry.provenance_disclosure_score * 3
                        + entry.confidence_disclosure_score * 2
                        + entry.governance_alignment_score * 2
                        + entry.mutation_risk_score
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("research_reports", entry.context_tags)
            self.assertIn(
                "research_report_generation",
                entry.blocked_runtime_behaviors,
            )
            self.assertIn("report_storage_write", entry.blocked_runtime_behaviors)
            self.assertTrue(entry.hitl_required_before_report_generation)
            self.assertTrue(entry.research_reports_capability_implemented)
            self.assertTrue(entry.research_reports_metadata_implemented)
            self.assertTrue(entry.automatic_kb_enrichment_metadata_used)
            self.assertFalse(entry.research_report_generation_implemented)
            self.assertFalse(entry.report_output_generation_implemented)
            self.assertFalse(entry.file_export_generation_implemented)
            self.assertFalse(entry.report_storage_write_implemented)
            self.assertFalse(entry.kb_storage_write_implemented)
            self.assertFalse(entry.retrieval_execution_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        report_scope = research_report_entry_by_id(
            "research_reports::report_scope_review",
            plan,
        )
        self.assertIsNotNone(report_scope)
        assert report_scope is not None
        self.assertEqual(report_scope.status, "guarded")
        self.assertEqual(report_scope.confidence, "guarded")
        self.assertEqual(
            len(research_report_entries_for_status("review_required", plan)),
            1,
        )
        self.assertEqual(
            len(research_report_entries_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_report_metadata(self) -> None:
        plan = build_research_reports()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            ResearchReportPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_report_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_report_score must match",
        ):
            ResearchReportPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["generated_report_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "generated_report_ids must remain empty",
        ):
            ResearchReportPlan(**payload)

    def test_reports_compose_with_enrichment_metadata(self) -> None:
        enrichment = build_automatic_kb_enrichment(route=RouteName.REVIEW)
        plan = build_research_reports(
            route=RouteName.REVIEW,
            enrichment=enrichment,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.checked_at, enrichment.checked_at)
        self.assertEqual(plan.enrichment_entry_ids, enrichment.entry_ids)
        self.assertTrue(plan.automatic_kb_enrichment_metadata_used)
        self.assertFalse(plan.research_report_generation_implemented)

    def test_reports_preserve_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Plan a research report for a p5.js shader study.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_research_reports(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_reports_do_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_research_reports(route=RouteName.GENERATE)
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
                        entry.report_kind,
                        entry.status,
                        entry.confidence,
                        entry.report_axis,
                        *entry.enrichment_entry_ids,
                        entry.report_summary,
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
            "generate_research_report(",
            "generate_report_output(",
            "modify_generated_output(",
            "export_report_file(",
            "write_report_storage(",
            "execute_automatic_kb_enrichment(",
            "write_kb_storage(",
            "write_provenance_record(",
            "fetch_external_source(",
            "browse_web(",
            "download_paper(",
            "validate_source(",
            "score_source_credibility(",
            "detect_contradiction(",
            "score_research_confidence(",
            "execute_retrieval(",
            "mutate_retrieval_config(",
            "provision_provider(",
            "infer_api_key(",
            "route_provider(",
            "execute_provider(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "write_storage(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
