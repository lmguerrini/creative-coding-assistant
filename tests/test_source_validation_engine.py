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
    SourceValidationPlan,
    build_research_memory,
    build_source_validation_engine,
    route_request,
    source_validation_entries_for_confidence,
    source_validation_entries_for_status,
    source_validation_entry_by_id,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Source Validation Engine",)


class SourceValidationEngineTests(unittest.TestCase):
    def test_plan_builds_advisory_source_validation_metadata(self) -> None:
        plan = build_source_validation_engine(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "source_validation_engine")
        self.assertEqual(plan.serialization_version, "source_validation_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.research_memory_role, "research_memory")
        self.assertEqual(plan.memory_entry_count, 5)
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
        self.assertFalse(plan.executed_validation_ids)
        self.assertFalse(plan.live_checked_source_ids)
        self.assertFalse(plan.fetched_external_source_ids)
        self.assertFalse(plan.mutated_source_registry_ids)
        self.assertFalse(plan.written_validation_record_ids)
        self.assertEqual(plan.overall_validation_score, 820)
        self.assertEqual(plan.overall_validation_posture, "guarded")
        self.assertIn("does not execute source validation", plan.authority_boundary)
        self.assertTrue(plan.source_validation_engine_capability_implemented)
        self.assertTrue(plan.source_validation_engine_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.research_memory_metadata_used)
        self.assertFalse(plan.source_validation_execution_implemented)
        self.assertFalse(plan.live_source_validation_implemented)
        self.assertFalse(plan.source_health_check_execution_implemented)
        self.assertFalse(plan.external_source_fetch_implemented)
        self.assertFalse(plan.web_browsing_implemented)
        self.assertFalse(plan.paper_download_implemented)
        self.assertFalse(plan.source_registry_mutation_implemented)
        self.assertFalse(plan.validation_record_write_implemented)
        self.assertFalse(plan.source_credibility_scoring_implemented)
        self.assertFalse(plan.contradiction_detection_implemented)
        self.assertFalse(plan.research_confidence_scoring_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_validation_without_execution(self) -> None:
        plan = build_source_validation_engine(route="generate")
        memory_ids = set(plan.memory_entry_ids)

        for entry in plan.entries:
            self.assertEqual(entry.serialization_version, "source_validation_entry.v1")
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.entry_id,
                f"source_validation_engine::{entry.validation_kind}",
            )
            self.assertTrue(set(entry.memory_entry_ids).issubset(memory_ids))
            self.assertEqual(entry.source_count, plan.source_count)
            self.assertEqual(entry.domain_count, plan.domain_count)
            self.assertEqual(
                entry.validation_score,
                min(
                    1000,
                    max(
                        0,
                        entry.source_presence_score * 2
                        + entry.provenance_completeness_score * 3
                        + entry.freshness_policy_score * 2
                        + entry.governance_alignment_score * 2
                        + entry.mutation_risk_score
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("source_validation_engine", entry.context_tags)
            self.assertIn(
                "source_validation_execution",
                entry.blocked_runtime_behaviors,
            )
            self.assertIn("external_source_fetch", entry.blocked_runtime_behaviors)
            self.assertTrue(entry.hitl_required_before_source_validation)
            self.assertTrue(entry.source_validation_engine_capability_implemented)
            self.assertTrue(entry.source_validation_engine_metadata_implemented)
            self.assertTrue(entry.research_memory_metadata_used)
            self.assertFalse(entry.source_validation_execution_implemented)
            self.assertFalse(entry.live_source_validation_implemented)
            self.assertFalse(entry.source_health_check_execution_implemented)
            self.assertFalse(entry.external_source_fetch_implemented)
            self.assertFalse(entry.source_registry_mutation_implemented)
            self.assertFalse(entry.validation_record_write_implemented)
            self.assertFalse(entry.source_credibility_scoring_implemented)
            self.assertFalse(entry.retrieval_execution_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        presence = source_validation_entry_by_id(
            "source_validation_engine::source_presence_validation_review",
            plan,
        )
        self.assertIsNotNone(presence)
        assert presence is not None
        self.assertEqual(presence.status, "guarded")
        self.assertEqual(presence.confidence, "guarded")
        self.assertEqual(
            len(source_validation_entries_for_status("review_required", plan)),
            1,
        )
        self.assertEqual(
            len(source_validation_entries_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_validation_metadata(self) -> None:
        plan = build_source_validation_engine()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            SourceValidationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_validation_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_validation_score must match",
        ):
            SourceValidationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["executed_validation_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "executed_validation_ids must remain empty",
        ):
            SourceValidationPlan(**payload)

    def test_validation_composes_with_memory_metadata(self) -> None:
        memory = build_research_memory(route=RouteName.REVIEW)
        plan = build_source_validation_engine(
            route=RouteName.REVIEW,
            memory=memory,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.checked_at, memory.checked_at)
        self.assertEqual(plan.memory_entry_ids, memory.entry_ids)
        self.assertTrue(plan.research_memory_metadata_used)
        self.assertFalse(plan.source_validation_execution_implemented)

    def test_validation_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Plan source validation for a p5.js shader study.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_source_validation_engine(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_validation_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_source_validation_engine(route=RouteName.GENERATE)
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
                        entry.validation_kind,
                        entry.status,
                        entry.confidence,
                        entry.validation_axis,
                        *entry.memory_entry_ids,
                        entry.validation_summary,
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
            "execute_source_validation(",
            "validate_source_live(",
            "run_source_health_check(",
            "fetch_external_source(",
            "browse_web(",
            "download_paper(",
            "mutate_source_registry(",
            "write_validation_record(",
            "score_source_credibility(",
            "detect_contradiction(",
            "score_research_confidence(",
            "execute_retrieval(",
            "mutate_retrieval_config(",
            "mutate_vector_index(",
            "enrich_kb(",
            "write_kb_storage(",
            "write_storage(",
            "provision_provider(",
            "infer_api_key(",
            "route_provider(",
            "execute_provider(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
