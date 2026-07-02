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
    ResearchMemoryPlan,
    build_research_memory,
    build_research_reports,
    research_memory_entries_for_confidence,
    research_memory_entries_for_status,
    research_memory_entry_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Research Memory",)


class ResearchMemoryTests(unittest.TestCase):
    def test_plan_builds_advisory_research_memory_metadata(self) -> None:
        plan = build_research_memory(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "research_memory")
        self.assertEqual(plan.serialization_version, "research_memory_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.research_reports_role, "research_reports")
        self.assertEqual(plan.report_entry_count, 5)
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
        self.assertFalse(plan.created_memory_record_ids)
        self.assertFalse(plan.updated_memory_record_ids)
        self.assertFalse(plan.deleted_memory_record_ids)
        self.assertFalse(plan.retrieved_memory_record_ids)
        self.assertFalse(plan.written_memory_storage_record_ids)
        self.assertFalse(plan.mutated_memory_index_ids)
        self.assertEqual(plan.overall_memory_score, 820)
        self.assertEqual(plan.overall_memory_posture, "guarded")
        self.assertIn(
            "does not create research memory records",
            plan.authority_boundary,
        )
        self.assertTrue(plan.research_memory_capability_implemented)
        self.assertTrue(plan.research_memory_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.research_reports_metadata_used)
        self.assertFalse(plan.research_memory_record_creation_implemented)
        self.assertFalse(plan.research_memory_record_update_implemented)
        self.assertFalse(plan.research_memory_record_deletion_implemented)
        self.assertFalse(plan.research_memory_retrieval_execution_implemented)
        self.assertFalse(plan.research_memory_storage_write_implemented)
        self.assertFalse(plan.memory_index_mutation_implemented)
        self.assertFalse(plan.research_report_generation_implemented)
        self.assertFalse(plan.report_storage_write_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_memory_without_mutation(self) -> None:
        plan = build_research_memory(route="generate")
        report_ids = set(plan.report_entry_ids)

        for entry in plan.entries:
            self.assertEqual(entry.serialization_version, "research_memory_entry.v1")
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.entry_id,
                f"research_memory::{entry.memory_kind}",
            )
            self.assertTrue(set(entry.report_entry_ids).issubset(report_ids))
            self.assertEqual(entry.source_count, plan.source_count)
            self.assertEqual(entry.domain_count, plan.domain_count)
            self.assertEqual(
                entry.memory_score,
                min(
                    1000,
                    max(
                        0,
                        entry.memory_scope_score * 2
                        + entry.provenance_linkage_score * 3
                        + entry.retention_policy_score * 2
                        + entry.governance_alignment_score * 2
                        + entry.mutation_risk_score
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("research_memory", entry.context_tags)
            self.assertIn(
                "research_memory_record_creation",
                entry.blocked_runtime_behaviors,
            )
            self.assertIn(
                "research_memory_storage_write",
                entry.blocked_runtime_behaviors,
            )
            self.assertTrue(entry.hitl_required_before_memory_mutation)
            self.assertTrue(entry.research_memory_capability_implemented)
            self.assertTrue(entry.research_memory_metadata_implemented)
            self.assertTrue(entry.research_reports_metadata_used)
            self.assertFalse(entry.research_memory_record_creation_implemented)
            self.assertFalse(entry.research_memory_record_update_implemented)
            self.assertFalse(entry.research_memory_record_deletion_implemented)
            self.assertFalse(entry.research_memory_retrieval_execution_implemented)
            self.assertFalse(entry.research_memory_storage_write_implemented)
            self.assertFalse(entry.memory_index_mutation_implemented)
            self.assertFalse(entry.retrieval_execution_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        memory_scope = research_memory_entry_by_id(
            "research_memory::memory_scope_review",
            plan,
        )
        self.assertIsNotNone(memory_scope)
        assert memory_scope is not None
        self.assertEqual(memory_scope.status, "guarded")
        self.assertEqual(memory_scope.confidence, "guarded")
        self.assertEqual(
            len(research_memory_entries_for_status("review_required", plan)),
            1,
        )
        self.assertEqual(
            len(research_memory_entries_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_memory_metadata(self) -> None:
        plan = build_research_memory()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            ResearchMemoryPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_memory_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_memory_score must match",
        ):
            ResearchMemoryPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["created_memory_record_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "created_memory_record_ids must remain empty",
        ):
            ResearchMemoryPlan(**payload)

    def test_memory_composes_with_report_metadata(self) -> None:
        reports = build_research_reports(route=RouteName.REVIEW)
        plan = build_research_memory(route=RouteName.REVIEW, reports=reports)

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.checked_at, reports.checked_at)
        self.assertEqual(plan.report_entry_ids, reports.entry_ids)
        self.assertTrue(plan.research_reports_metadata_used)
        self.assertFalse(plan.research_memory_record_creation_implemented)

    def test_memory_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Plan research memory for a p5.js shader study.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_research_memory(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_memory_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_research_memory(route=RouteName.GENERATE)
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
                        entry.memory_kind,
                        entry.status,
                        entry.confidence,
                        entry.memory_axis,
                        *entry.report_entry_ids,
                        entry.memory_summary,
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
            "create_research_memory_record(",
            "update_research_memory_record(",
            "delete_research_memory_record(",
            "execute_research_memory_retrieval(",
            "write_research_memory_storage(",
            "mutate_memory_index(",
            "generate_research_report(",
            "write_report_storage(",
            "enrich_kb(",
            "write_kb_storage(",
            "write_provenance_record(",
            "execute_retrieval(",
            "mutate_retrieval_config(",
            "mutate_vector_index(",
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
