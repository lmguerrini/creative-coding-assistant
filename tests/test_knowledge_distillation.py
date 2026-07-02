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
    KnowledgeDistillationPlan,
    build_cross_source_comparison,
    build_knowledge_distillation,
    knowledge_distillation_entries_for_confidence,
    knowledge_distillation_entries_for_status,
    knowledge_distillation_entry_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Knowledge Distillation",)


class KnowledgeDistillationTests(unittest.TestCase):
    def test_plan_builds_advisory_knowledge_distillation_metadata(self) -> None:
        plan = build_knowledge_distillation(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "knowledge_distillation")
        self.assertEqual(
            plan.serialization_version,
            "knowledge_distillation_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.cross_source_comparison_role,
            "cross_source_comparison",
        )
        self.assertEqual(plan.comparison_entry_count, 5)
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
        self.assertFalse(plan.executed_distillation_ids)
        self.assertFalse(plan.generated_distilled_output_ids)
        self.assertFalse(plan.synthesized_claim_ids)
        self.assertFalse(plan.summarized_evidence_ids)
        self.assertFalse(plan.written_provenance_record_ids)
        self.assertFalse(plan.written_kb_record_ids)
        self.assertEqual(plan.overall_distillation_score, 815)
        self.assertEqual(plan.overall_distillation_posture, "guarded")
        self.assertIn(
            "does not execute knowledge distillation",
            plan.authority_boundary,
        )
        self.assertTrue(plan.knowledge_distillation_capability_implemented)
        self.assertTrue(plan.knowledge_distillation_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.cross_source_comparison_metadata_used)
        self.assertFalse(plan.knowledge_distillation_execution_implemented)
        self.assertFalse(plan.distilled_output_generation_implemented)
        self.assertFalse(plan.claim_synthesis_execution_implemented)
        self.assertFalse(plan.evidence_summarization_execution_implemented)
        self.assertFalse(plan.provenance_record_write_implemented)
        self.assertFalse(plan.research_report_generation_implemented)
        self.assertFalse(plan.kb_enrichment_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
        self.assertFalse(plan.source_validation_execution_implemented)
        self.assertFalse(plan.source_credibility_scoring_implemented)
        self.assertFalse(plan.contradiction_detection_implemented)
        self.assertFalse(plan.research_confidence_scoring_implemented)
        self.assertFalse(plan.external_source_fetch_implemented)
        self.assertFalse(plan.web_browsing_implemented)
        self.assertFalse(plan.paper_download_implemented)
        self.assertFalse(plan.retrieval_configuration_mutation_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.provider_provisioning_implemented)
        self.assertFalse(plan.api_key_inference_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_distillation_without_execution(self) -> None:
        plan = build_knowledge_distillation(route="generate")
        comparison_ids = set(plan.comparison_entry_ids)

        for entry in plan.entries:
            self.assertEqual(
                entry.serialization_version,
                "knowledge_distillation_entry.v1",
            )
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.entry_id,
                f"knowledge_distillation::{entry.distillation_kind}",
            )
            self.assertTrue(set(entry.comparison_entry_ids).issubset(comparison_ids))
            self.assertEqual(entry.source_count, plan.source_count)
            self.assertEqual(entry.domain_count, plan.domain_count)
            self.assertEqual(
                entry.distillation_score,
                min(
                    1000,
                    max(
                        0,
                        entry.source_alignment_score * 2
                        + entry.provenance_preservation_score * 3
                        + entry.abstraction_quality_score * 2
                        + entry.governance_alignment_score * 2
                        + entry.mutation_risk_score
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("knowledge_distillation", entry.context_tags)
            self.assertIn(
                "knowledge_distillation_execution",
                entry.blocked_runtime_behaviors,
            )
            self.assertIn(
                "distilled_output_generation",
                entry.blocked_runtime_behaviors,
            )
            self.assertTrue(entry.hitl_required_before_distillation)
            self.assertTrue(entry.knowledge_distillation_capability_implemented)
            self.assertTrue(entry.knowledge_distillation_metadata_implemented)
            self.assertTrue(entry.cross_source_comparison_metadata_used)
            self.assertFalse(entry.knowledge_distillation_execution_implemented)
            self.assertFalse(entry.distilled_output_generation_implemented)
            self.assertFalse(entry.claim_synthesis_execution_implemented)
            self.assertFalse(entry.evidence_summarization_execution_implemented)
            self.assertFalse(entry.provenance_record_write_implemented)
            self.assertFalse(entry.research_report_generation_implemented)
            self.assertFalse(entry.kb_storage_write_implemented)
            self.assertFalse(entry.source_validation_execution_implemented)
            self.assertFalse(entry.retrieval_execution_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        readiness = knowledge_distillation_entry_by_id(
            "knowledge_distillation::source_claim_distillation_readiness",
            plan,
        )
        self.assertIsNotNone(readiness)
        assert readiness is not None
        self.assertEqual(readiness.status, "guarded")
        self.assertEqual(readiness.confidence, "guarded")
        self.assertEqual(
            len(knowledge_distillation_entries_for_status("review_required", plan)),
            1,
        )
        self.assertEqual(
            len(knowledge_distillation_entries_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_distillation_metadata(self) -> None:
        plan = build_knowledge_distillation()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            KnowledgeDistillationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_distillation_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_distillation_score must match",
        ):
            KnowledgeDistillationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["generated_distilled_output_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "generated_distilled_output_ids must remain empty",
        ):
            KnowledgeDistillationPlan(**payload)

    def test_distillation_composes_with_comparison_metadata(self) -> None:
        comparison = build_cross_source_comparison(route=RouteName.REVIEW)
        plan = build_knowledge_distillation(
            route=RouteName.REVIEW,
            comparison=comparison,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.checked_at, comparison.checked_at)
        self.assertEqual(plan.comparison_entry_ids, comparison.entry_ids)
        self.assertTrue(plan.cross_source_comparison_metadata_used)
        self.assertFalse(plan.knowledge_distillation_execution_implemented)

    def test_distillation_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Plan knowledge distillation for a p5.js shader study.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_knowledge_distillation(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_distillation_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_knowledge_distillation(route=RouteName.GENERATE)
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
                        entry.distillation_kind,
                        entry.status,
                        entry.confidence,
                        entry.distillation_axis,
                        *entry.comparison_entry_ids,
                        entry.distillation_summary,
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
            "execute_knowledge_distillation(",
            "generate_distilled_output(",
            "synthesize_claim(",
            "summarize_evidence(",
            "write_provenance_record(",
            "generate_research_report(",
            "enrich_kb(",
            "write_kb_storage(",
            "validate_source(",
            "score_source_credibility(",
            "detect_contradiction(",
            "score_research_confidence(",
            "fetch_external_source(",
            "browse_web(",
            "download_paper(",
            "mutate_source_registry(",
            "mutate_retrieval_config(",
            "execute_retrieval(",
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
