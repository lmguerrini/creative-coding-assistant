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
    AutomaticKbEnrichmentPlan,
    automatic_kb_enrichment_entries_for_confidence,
    automatic_kb_enrichment_entries_for_status,
    automatic_kb_enrichment_entry_by_id,
    build_automatic_kb_enrichment,
    build_knowledge_distillation,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Automatic KB Enrichment",)


class AutomaticKbEnrichmentTests(unittest.TestCase):
    def test_plan_builds_advisory_automatic_kb_enrichment_metadata(self) -> None:
        plan = build_automatic_kb_enrichment(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "automatic_kb_enrichment")
        self.assertEqual(
            plan.serialization_version,
            "automatic_kb_enrichment_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.knowledge_distillation_role, "knowledge_distillation")
        self.assertEqual(plan.distillation_entry_count, 5)
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
        self.assertFalse(plan.executed_enrichment_ids)
        self.assertFalse(plan.created_kb_record_ids)
        self.assertFalse(plan.updated_kb_record_ids)
        self.assertFalse(plan.deleted_kb_record_ids)
        self.assertFalse(plan.written_kb_storage_record_ids)
        self.assertFalse(plan.mutated_retrieval_config_ids)
        self.assertFalse(plan.mutated_vector_index_ids)
        self.assertEqual(plan.overall_enrichment_score, 817)
        self.assertEqual(plan.overall_enrichment_posture, "guarded")
        self.assertIn(
            "does not execute automatic KB enrichment",
            plan.authority_boundary,
        )
        self.assertTrue(plan.automatic_kb_enrichment_capability_implemented)
        self.assertTrue(plan.automatic_kb_enrichment_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.knowledge_distillation_metadata_used)
        self.assertFalse(plan.automatic_kb_enrichment_execution_implemented)
        self.assertFalse(plan.kb_enrichment_execution_implemented)
        self.assertFalse(plan.kb_record_creation_implemented)
        self.assertFalse(plan.kb_record_update_implemented)
        self.assertFalse(plan.kb_record_deletion_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
        self.assertFalse(plan.provenance_record_write_implemented)
        self.assertFalse(plan.source_registry_mutation_implemented)
        self.assertFalse(plan.retrieval_configuration_mutation_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.vector_index_mutation_implemented)
        self.assertFalse(plan.vector_upsert_implemented)
        self.assertFalse(plan.embedding_refresh_execution_implemented)
        self.assertFalse(plan.external_source_fetch_implemented)
        self.assertFalse(plan.provider_provisioning_implemented)
        self.assertFalse(plan.api_key_inference_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_enrichment_without_execution(self) -> None:
        plan = build_automatic_kb_enrichment(route="generate")
        distillation_ids = set(plan.distillation_entry_ids)

        for entry in plan.entries:
            self.assertEqual(
                entry.serialization_version,
                "automatic_kb_enrichment_entry.v1",
            )
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.entry_id,
                f"automatic_kb_enrichment::{entry.enrichment_kind}",
            )
            self.assertTrue(
                set(entry.distillation_entry_ids).issubset(distillation_ids)
            )
            self.assertEqual(entry.source_count, plan.source_count)
            self.assertEqual(entry.domain_count, plan.domain_count)
            self.assertEqual(
                entry.enrichment_score,
                min(
                    1000,
                    max(
                        0,
                        entry.candidate_quality_score * 2
                        + entry.provenance_readiness_score * 3
                        + entry.storage_safety_score * 2
                        + entry.governance_alignment_score * 2
                        + entry.mutation_risk_score
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("automatic_kb_enrichment", entry.context_tags)
            self.assertIn(
                "automatic_kb_enrichment_execution",
                entry.blocked_runtime_behaviors,
            )
            self.assertIn("kb_storage_write", entry.blocked_runtime_behaviors)
            self.assertTrue(entry.hitl_required_before_enrichment)
            self.assertTrue(entry.automatic_kb_enrichment_capability_implemented)
            self.assertTrue(entry.automatic_kb_enrichment_metadata_implemented)
            self.assertTrue(entry.knowledge_distillation_metadata_used)
            self.assertFalse(entry.automatic_kb_enrichment_execution_implemented)
            self.assertFalse(entry.kb_enrichment_execution_implemented)
            self.assertFalse(entry.kb_record_creation_implemented)
            self.assertFalse(entry.kb_record_update_implemented)
            self.assertFalse(entry.kb_record_deletion_implemented)
            self.assertFalse(entry.kb_storage_write_implemented)
            self.assertFalse(entry.provenance_record_write_implemented)
            self.assertFalse(entry.retrieval_execution_implemented)
            self.assertFalse(entry.vector_index_mutation_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        candidate = automatic_kb_enrichment_entry_by_id(
            "automatic_kb_enrichment::enrichment_candidate_review",
            plan,
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.status, "guarded")
        self.assertEqual(candidate.confidence, "guarded")
        self.assertEqual(
            len(automatic_kb_enrichment_entries_for_status("review_required", plan)),
            1,
        )
        self.assertEqual(
            len(automatic_kb_enrichment_entries_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_enrichment_metadata(self) -> None:
        plan = build_automatic_kb_enrichment()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            AutomaticKbEnrichmentPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_enrichment_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_enrichment_score must match",
        ):
            AutomaticKbEnrichmentPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["created_kb_record_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "created_kb_record_ids must remain empty",
        ):
            AutomaticKbEnrichmentPlan(**payload)

    def test_enrichment_composes_with_distillation_metadata(self) -> None:
        distillation = build_knowledge_distillation(route=RouteName.REVIEW)
        plan = build_automatic_kb_enrichment(
            route=RouteName.REVIEW,
            distillation=distillation,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.checked_at, distillation.checked_at)
        self.assertEqual(plan.distillation_entry_ids, distillation.entry_ids)
        self.assertTrue(plan.knowledge_distillation_metadata_used)
        self.assertFalse(plan.automatic_kb_enrichment_execution_implemented)

    def test_enrichment_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Plan KB enrichment for a p5.js shader study.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_automatic_kb_enrichment(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_enrichment_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_automatic_kb_enrichment(route=RouteName.GENERATE)
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
                        entry.enrichment_kind,
                        entry.status,
                        entry.confidence,
                        entry.enrichment_axis,
                        *entry.distillation_entry_ids,
                        entry.enrichment_summary,
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
            "execute_automatic_kb_enrichment(",
            "execute_kb_enrichment(",
            "create_kb_record(",
            "update_kb_record(",
            "delete_kb_record(",
            "write_kb_storage(",
            "write_provenance_record(",
            "mutate_source_registry(",
            "mutate_retrieval_config(",
            "execute_retrieval(",
            "mutate_vector_index(",
            "upsert_vector(",
            "refresh_embedding(",
            "fetch_external_source(",
            "browse_web(",
            "download_paper(",
            "validate_source(",
            "score_source_credibility(",
            "detect_contradiction(",
            "score_research_confidence(",
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
