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
    DocumentationIntelligencePlan,
    build_automatic_kb_updates,
    build_documentation_intelligence,
    documentation_intelligence_signal_by_id,
    documentation_intelligence_signals_for_confidence,
    documentation_intelligence_signals_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Documentation Intelligence",)
REQUIRED_SIGNAL_FIELDS = {
    "signal_id",
    "signal_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "intelligence_axis",
    "source_update_candidate_ids",
    "source_update_candidate_count",
    "source_count",
    "domain_count",
    "documentation_signal_summary",
    "documentation_mapping_score",
    "source_traceability_score",
    "governance_alignment_score",
    "mutation_risk_score",
    "governance_weight",
    "intelligence_score",
    "hitl_required_before_documentation_action",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "documentation_intelligence_implemented",
    "documentation_intelligence_metadata_implemented",
    "automatic_kb_update_metadata_used",
    "documentation_action_execution_implemented",
    "documentation_fetch_execution_implemented",
    "documentation_parse_execution_implemented",
    "documentation_summarization_implemented",
    "documentation_rewrite_implemented",
    "documentation_generation_implemented",
    "live_content_classification_implemented",
    "kb_enrichment_implemented",
    "source_record_update_implemented",
    "embedding_refresh_implemented",
    "retrieval_configuration_mutation_implemented",
    "retrieval_execution_implemented",
    "ranking_mutation_implemented",
    "kb_storage_write_implemented",
    "provider_provisioning_implemented",
    "api_key_inference_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class DocumentationIntelligenceTests(unittest.TestCase):
    def test_plan_builds_documentation_intelligence_metadata(self) -> None:
        plan = build_documentation_intelligence(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "documentation_intelligence")
        self.assertEqual(
            plan.serialization_version,
            "documentation_intelligence_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.source_update_role, "automatic_kb_updates")
        self.assertEqual(
            plan.source_update_serialization_version,
            "automatic_kb_update_plan.v1",
        )
        self.assertEqual(len(plan.source_update_candidate_ids), 5)
        self.assertEqual(plan.source_update_candidate_count, 5)
        self.assertEqual(plan.covered_roadmap_items, EXPECTED_ROADMAP_ITEMS)
        self.assertEqual(plan.covered_roadmap_item_count, 1)
        self.assertEqual(plan.source_count, 57)
        self.assertEqual(plan.domain_count, 43)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.signal_count, 5)
        self.assertEqual(plan.candidate_signal_count, 1)
        self.assertEqual(plan.review_required_signal_count, 3)
        self.assertEqual(plan.guarded_signal_count, 1)
        self.assertEqual(plan.high_confidence_signal_count, 3)
        self.assertEqual(plan.hitl_required_signal_count, 5)
        self.assertFalse(plan.executed_documentation_action_ids)
        self.assertFalse(plan.fetched_documentation_source_ids)
        self.assertFalse(plan.parsed_documentation_source_ids)
        self.assertFalse(plan.enriched_kb_record_ids)
        self.assertFalse(plan.mutated_retrieval_source_ids)
        self.assertEqual(plan.overall_intelligence_posture, "guarded")
        self.assertIn("does not fetch documentation", plan.authority_boundary)
        self.assertIn("write storage", plan.authority_boundary)
        self.assertTrue(plan.documentation_intelligence_implemented)
        self.assertTrue(plan.documentation_intelligence_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.automatic_kb_update_metadata_used)
        self.assertFalse(plan.documentation_action_execution_implemented)
        self.assertFalse(plan.documentation_fetch_execution_implemented)
        self.assertFalse(plan.documentation_parse_execution_implemented)
        self.assertFalse(plan.documentation_summarization_implemented)
        self.assertFalse(plan.documentation_rewrite_implemented)
        self.assertFalse(plan.documentation_generation_implemented)
        self.assertFalse(plan.kb_enrichment_implemented)
        self.assertFalse(plan.source_record_update_implemented)
        self.assertFalse(plan.embedding_refresh_implemented)
        self.assertFalse(plan.retrieval_configuration_mutation_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.ranking_mutation_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
        self.assertFalse(plan.provider_provisioning_implemented)
        self.assertFalse(plan.api_key_inference_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_signals_score_documentation_intelligence_without_execution(self) -> None:
        plan = build_documentation_intelligence(route="generate")
        source_candidate_ids = set(plan.source_update_candidate_ids)

        for signal in plan.signals:
            dumped = signal.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SIGNAL_FIELDS)
            self.assertEqual(
                signal.serialization_version,
                "documentation_intelligence_entry.v1",
            )
            self.assertEqual(signal.route_name, RouteName.GENERATE)
            self.assertEqual(
                signal.signal_id,
                f"documentation_intelligence::{signal.signal_kind}",
            )
            self.assertEqual(
                signal.source_update_candidate_count,
                len(signal.source_update_candidate_ids),
            )
            self.assertTrue(
                set(signal.source_update_candidate_ids).issubset(source_candidate_ids)
            )
            self.assertEqual(
                signal.intelligence_score,
                min(
                    1000,
                    max(
                        0,
                        signal.documentation_mapping_score * 3
                        + signal.source_traceability_score * 3
                        + signal.governance_alignment_score * 2
                        + signal.mutation_risk_score * 2
                        + signal.governance_weight,
                    ),
                ),
            )
            self.assertIn("documentation_intelligence", signal.context_tags)
            self.assertIn(
                "documentation_fetch_execution",
                signal.blocked_runtime_behaviors,
            )
            self.assertIn("kb_storage_write", signal.blocked_runtime_behaviors)
            self.assertIn("retrieval_execution", signal.blocked_runtime_behaviors)
            self.assertTrue(signal.hitl_required_before_documentation_action)
            self.assertTrue(signal.documentation_intelligence_implemented)
            self.assertTrue(signal.documentation_intelligence_metadata_implemented)
            self.assertTrue(signal.automatic_kb_update_metadata_used)
            self.assertFalse(signal.documentation_action_execution_implemented)
            self.assertFalse(signal.documentation_fetch_execution_implemented)
            self.assertFalse(signal.documentation_parse_execution_implemented)
            self.assertFalse(signal.documentation_summarization_implemented)
            self.assertFalse(signal.documentation_rewrite_implemented)
            self.assertFalse(signal.documentation_generation_implemented)
            self.assertFalse(signal.live_content_classification_implemented)
            self.assertFalse(signal.kb_enrichment_implemented)
            self.assertFalse(signal.source_record_update_implemented)
            self.assertFalse(signal.embedding_refresh_implemented)
            self.assertFalse(signal.retrieval_configuration_mutation_implemented)
            self.assertFalse(signal.retrieval_execution_implemented)
            self.assertFalse(signal.ranking_mutation_implemented)
            self.assertFalse(signal.kb_storage_write_implemented)
            self.assertFalse(signal.provider_provisioning_implemented)
            self.assertFalse(signal.api_key_inference_implemented)
            self.assertFalse(signal.provider_model_routing_implemented)
            self.assertFalse(signal.provider_execution_implemented)
            self.assertFalse(signal.workflow_control_implemented)
            self.assertFalse(signal.generated_output_mutation_implemented)
            self.assertFalse(signal.runtime_evolution_implemented)
            self.assertTrue(signal.advisory_only)

        source_map = documentation_intelligence_signal_by_id(
            "documentation_intelligence::documentation_source_map",
            plan,
        )
        self.assertIsNotNone(source_map)
        assert source_map is not None
        self.assertEqual(source_map.status, "guarded")
        self.assertEqual(source_map.confidence, "guarded")
        self.assertEqual(
            len(documentation_intelligence_signals_for_status("review_required", plan)),
            3,
        )
        self.assertEqual(
            len(documentation_intelligence_signals_for_confidence("high", plan)),
            2,
        )

    def test_plan_rejects_mismatched_documentation_metadata(self) -> None:
        plan = build_documentation_intelligence()
        payload = plan.model_dump(mode="json")
        payload["signal_ids"] = ("missing",) + tuple(payload["signal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "signal_ids must match"):
            DocumentationIntelligencePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_intelligence_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_intelligence_score must match",
        ):
            DocumentationIntelligencePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["executed_documentation_action_ids"] = (plan.signal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "executed_documentation_action_ids must remain empty",
        ):
            DocumentationIntelligencePlan(**payload)

    def test_documentation_intelligence_composes_task_2_metadata(self) -> None:
        update_plan = build_automatic_kb_updates(route=RouteName.REVIEW)
        plan = build_documentation_intelligence(
            route=RouteName.REVIEW,
            automatic_kb_updates=update_plan,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.source_update_candidate_ids, update_plan.candidate_ids)
        self.assertEqual(plan.source_count, update_plan.source_count)
        self.assertEqual(plan.domain_count, update_plan.domain_count)
        for signal in plan.signals:
            self.assertTrue(
                set(signal.source_update_candidate_ids).issubset(
                    set(update_plan.candidate_ids)
                )
            )
            self.assertFalse(signal.documentation_fetch_execution_implemented)
            self.assertFalse(signal.kb_enrichment_implemented)

    def test_documentation_intelligence_preserves_routing_and_provider_factory(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Review documentation intelligence posture for p5.js examples.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_documentation_intelligence(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_documentation_intelligence_omits_runtime_mutation_calls(self) -> None:
        plan = build_documentation_intelligence(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *plan.covered_roadmap_items,
                *(
                    field
                    for signal in plan.signals
                    for field in (
                        signal.signal_id,
                        signal.signal_kind,
                        signal.status,
                        signal.confidence,
                        signal.intelligence_axis,
                        *signal.source_update_candidate_ids,
                        signal.documentation_signal_summary,
                        *signal.context_tags,
                        *signal.explainability_notes,
                        *signal.advisory_actions,
                        *signal.evidence,
                        *signal.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "fetch_documentation(",
            "crawl_documentation(",
            "parse_documentation(",
            "summarize_documentation(",
            "rewrite_documentation(",
            "generate_documentation(",
            "classify_live_content(",
            "enrich_kb(",
            "update_source_record(",
            "refresh_embedding(",
            "mutate_retrieval_config(",
            "execute_retrieval(",
            "mutate_ranking(",
            "write_kb_storage(",
            "provision_provider(",
            "infer_api_key(",
            "route_provider(",
            "execute_provider(",
            "invoke_agent(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
