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
    KnowledgeConsolidationPlan,
    build_knowledge_consolidation,
    build_source_reliability_engine,
    knowledge_consolidation_signal_by_id,
    knowledge_consolidation_signals_for_confidence,
    knowledge_consolidation_signals_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Knowledge Consolidation",)
REQUIRED_SIGNAL_FIELDS = {
    "signal_id",
    "signal_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "consolidation_axis",
    "source_reliability_signal_ids",
    "source_reliability_signal_count",
    "source_count",
    "domain_count",
    "consolidation_signal_summary",
    "consolidation_signal_score",
    "source_alignment_score",
    "governance_alignment_score",
    "mutation_risk_score",
    "governance_weight",
    "consolidation_score",
    "hitl_required_before_consolidation",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "knowledge_consolidation_capability_implemented",
    "knowledge_consolidation_metadata_implemented",
    "source_reliability_metadata_used",
    "knowledge_consolidation_execution_implemented",
    "consolidation_candidate_generation_implemented",
    "knowledge_merge_execution_implemented",
    "knowledge_deduplication_execution_implemented",
    "canonical_record_write_implemented",
    "consolidation_record_write_implemented",
    "kb_storage_write_implemented",
    "source_record_update_implemented",
    "source_reliability_engine_execution_implemented",
    "source_reliability_scoring_execution_implemented",
    "source_health_check_execution_implemented",
    "source_trust_score_computation_implemented",
    "source_rank_mutation_implemented",
    "source_registry_mutation_implemented",
    "source_fetch_execution_implemented",
    "freshness_scan_execution_implemented",
    "knowledge_drift_detection_execution_implemented",
    "drift_detection_execution_implemented",
    "drift_scan_execution_implemented",
    "timeline_scan_execution_implemented",
    "snapshot_comparison_execution_implemented",
    "drift_record_write_implemented",
    "conflict_resolution_execution_implemented",
    "conflict_arbitration_execution_implemented",
    "gap_scan_execution_implemented",
    "gap_remediation_execution_implemented",
    "kb_enrichment_implemented",
    "quality_score_computation_implemented",
    "retrieval_query_execution_implemented",
    "retrieval_configuration_mutation_implemented",
    "ranking_mutation_implemented",
    "embedding_request_execution_implemented",
    "embedding_refresh_execution_implemented",
    "vector_indexing_implemented",
    "vector_upsert_implemented",
    "documentation_fetch_execution_implemented",
    "provider_provisioning_implemented",
    "api_key_inference_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class KnowledgeConsolidationTests(unittest.TestCase):
    def test_plan_builds_knowledge_consolidation_metadata(self) -> None:
        plan = build_knowledge_consolidation(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "knowledge_consolidation")
        self.assertEqual(plan.serialization_version, "knowledge_consolidation_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.source_reliability_role, "source_reliability_engine")
        self.assertEqual(
            plan.source_reliability_serialization_version,
            "source_reliability_plan.v1",
        )
        self.assertEqual(len(plan.source_reliability_signal_ids), 5)
        self.assertEqual(plan.source_reliability_signal_count, 5)
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
        self.assertFalse(plan.planned_consolidation_ids)
        self.assertFalse(plan.generated_consolidation_candidate_ids)
        self.assertFalse(plan.merged_knowledge_record_ids)
        self.assertFalse(plan.deduplicated_knowledge_record_ids)
        self.assertFalse(plan.written_canonical_record_ids)
        self.assertFalse(plan.written_consolidation_record_ids)
        self.assertFalse(plan.written_kb_record_ids)
        self.assertFalse(plan.updated_source_record_ids)
        self.assertEqual(plan.overall_consolidation_posture, "guarded")
        self.assertIn(
            "does not execute knowledge consolidation",
            plan.authority_boundary,
        )
        self.assertIn("write KB storage", plan.authority_boundary)
        self.assertTrue(plan.knowledge_consolidation_capability_implemented)
        self.assertTrue(plan.knowledge_consolidation_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.source_reliability_metadata_used)
        self.assertFalse(plan.knowledge_consolidation_execution_implemented)
        self.assertFalse(plan.consolidation_candidate_generation_implemented)
        self.assertFalse(plan.knowledge_merge_execution_implemented)
        self.assertFalse(plan.knowledge_deduplication_execution_implemented)
        self.assertFalse(plan.canonical_record_write_implemented)
        self.assertFalse(plan.consolidation_record_write_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
        self.assertFalse(plan.source_record_update_implemented)
        self.assertFalse(plan.source_reliability_engine_execution_implemented)
        self.assertFalse(plan.source_reliability_scoring_execution_implemented)
        self.assertFalse(plan.source_health_check_execution_implemented)
        self.assertFalse(plan.source_trust_score_computation_implemented)
        self.assertFalse(plan.source_rank_mutation_implemented)
        self.assertFalse(plan.source_registry_mutation_implemented)
        self.assertFalse(plan.source_fetch_execution_implemented)
        self.assertFalse(plan.freshness_scan_execution_implemented)
        self.assertFalse(plan.knowledge_drift_detection_execution_implemented)
        self.assertFalse(plan.drift_detection_execution_implemented)
        self.assertFalse(plan.drift_scan_execution_implemented)
        self.assertFalse(plan.timeline_scan_execution_implemented)
        self.assertFalse(plan.snapshot_comparison_execution_implemented)
        self.assertFalse(plan.drift_record_write_implemented)
        self.assertFalse(plan.conflict_resolution_execution_implemented)
        self.assertFalse(plan.conflict_arbitration_execution_implemented)
        self.assertFalse(plan.gap_scan_execution_implemented)
        self.assertFalse(plan.gap_remediation_execution_implemented)
        self.assertFalse(plan.kb_enrichment_implemented)
        self.assertFalse(plan.quality_score_computation_implemented)
        self.assertFalse(plan.retrieval_query_execution_implemented)
        self.assertFalse(plan.retrieval_configuration_mutation_implemented)
        self.assertFalse(plan.ranking_mutation_implemented)
        self.assertFalse(plan.embedding_request_execution_implemented)
        self.assertFalse(plan.embedding_refresh_execution_implemented)
        self.assertFalse(plan.vector_indexing_implemented)
        self.assertFalse(plan.vector_upsert_implemented)
        self.assertFalse(plan.documentation_fetch_execution_implemented)
        self.assertFalse(plan.provider_provisioning_implemented)
        self.assertFalse(plan.api_key_inference_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_signals_score_knowledge_consolidation_without_execution(self) -> None:
        plan = build_knowledge_consolidation(route="generate")
        reliability_signal_ids = set(plan.source_reliability_signal_ids)

        for signal in plan.signals:
            dumped = signal.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SIGNAL_FIELDS)
            self.assertEqual(
                signal.serialization_version,
                "knowledge_consolidation_entry.v1",
            )
            self.assertEqual(signal.route_name, RouteName.GENERATE)
            self.assertEqual(
                signal.signal_id,
                f"knowledge_consolidation::{signal.signal_kind}",
            )
            self.assertEqual(
                signal.source_reliability_signal_count,
                len(signal.source_reliability_signal_ids),
            )
            self.assertTrue(
                set(signal.source_reliability_signal_ids).issubset(
                    reliability_signal_ids
                )
            )
            self.assertEqual(
                signal.consolidation_score,
                min(
                    1000,
                    max(
                        0,
                        signal.consolidation_signal_score * 3
                        + signal.source_alignment_score * 3
                        + signal.governance_alignment_score * 2
                        + signal.mutation_risk_score * 2
                        + signal.governance_weight,
                    ),
                ),
            )
            self.assertIn("knowledge_consolidation", signal.context_tags)
            self.assertIn(
                "knowledge_consolidation_execution",
                signal.blocked_runtime_behaviors,
            )
            self.assertIn("knowledge_merge_execution", signal.blocked_runtime_behaviors)
            self.assertIn("kb_storage_write", signal.blocked_runtime_behaviors)
            self.assertTrue(signal.hitl_required_before_consolidation)
            self.assertTrue(signal.knowledge_consolidation_capability_implemented)
            self.assertTrue(signal.knowledge_consolidation_metadata_implemented)
            self.assertTrue(signal.source_reliability_metadata_used)
            self.assertFalse(signal.knowledge_consolidation_execution_implemented)
            self.assertFalse(signal.consolidation_candidate_generation_implemented)
            self.assertFalse(signal.knowledge_merge_execution_implemented)
            self.assertFalse(signal.knowledge_deduplication_execution_implemented)
            self.assertFalse(signal.canonical_record_write_implemented)
            self.assertFalse(signal.consolidation_record_write_implemented)
            self.assertFalse(signal.kb_storage_write_implemented)
            self.assertFalse(signal.source_record_update_implemented)
            self.assertFalse(signal.source_reliability_engine_execution_implemented)
            self.assertFalse(signal.source_reliability_scoring_execution_implemented)
            self.assertFalse(signal.source_health_check_execution_implemented)
            self.assertFalse(signal.source_trust_score_computation_implemented)
            self.assertFalse(signal.source_rank_mutation_implemented)
            self.assertFalse(signal.source_registry_mutation_implemented)
            self.assertFalse(signal.source_fetch_execution_implemented)
            self.assertFalse(signal.freshness_scan_execution_implemented)
            self.assertFalse(signal.knowledge_drift_detection_execution_implemented)
            self.assertFalse(signal.drift_detection_execution_implemented)
            self.assertFalse(signal.drift_scan_execution_implemented)
            self.assertFalse(signal.timeline_scan_execution_implemented)
            self.assertFalse(signal.snapshot_comparison_execution_implemented)
            self.assertFalse(signal.drift_record_write_implemented)
            self.assertFalse(signal.conflict_resolution_execution_implemented)
            self.assertFalse(signal.conflict_arbitration_execution_implemented)
            self.assertFalse(signal.gap_scan_execution_implemented)
            self.assertFalse(signal.gap_remediation_execution_implemented)
            self.assertFalse(signal.kb_enrichment_implemented)
            self.assertFalse(signal.quality_score_computation_implemented)
            self.assertFalse(signal.retrieval_query_execution_implemented)
            self.assertFalse(signal.retrieval_configuration_mutation_implemented)
            self.assertFalse(signal.ranking_mutation_implemented)
            self.assertFalse(signal.embedding_request_execution_implemented)
            self.assertFalse(signal.embedding_refresh_execution_implemented)
            self.assertFalse(signal.vector_indexing_implemented)
            self.assertFalse(signal.vector_upsert_implemented)
            self.assertFalse(signal.documentation_fetch_execution_implemented)
            self.assertFalse(signal.provider_provisioning_implemented)
            self.assertFalse(signal.api_key_inference_implemented)
            self.assertFalse(signal.provider_model_routing_implemented)
            self.assertFalse(signal.provider_execution_implemented)
            self.assertFalse(signal.workflow_control_implemented)
            self.assertFalse(signal.generated_output_mutation_implemented)
            self.assertFalse(signal.runtime_evolution_implemented)
            self.assertTrue(signal.advisory_only)

        inventory = knowledge_consolidation_signal_by_id(
            "knowledge_consolidation::knowledge_consolidation_inventory_review",
            plan,
        )
        self.assertIsNotNone(inventory)
        assert inventory is not None
        self.assertEqual(inventory.status, "guarded")
        self.assertEqual(inventory.confidence, "guarded")
        self.assertEqual(
            len(knowledge_consolidation_signals_for_status("review_required", plan)),
            3,
        )
        self.assertEqual(
            len(knowledge_consolidation_signals_for_confidence("high", plan)),
            2,
        )

    def test_plan_rejects_mismatched_knowledge_consolidation_metadata(self) -> None:
        plan = build_knowledge_consolidation()
        payload = plan.model_dump(mode="json")
        payload["signal_ids"] = ("missing",) + tuple(payload["signal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "signal_ids must match"):
            KnowledgeConsolidationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_consolidation_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_consolidation_score must match",
        ):
            KnowledgeConsolidationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["planned_consolidation_ids"] = (plan.signal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "planned_consolidation_ids must remain empty",
        ):
            KnowledgeConsolidationPlan(**payload)

    def test_knowledge_consolidation_composes_task_12_metadata(self) -> None:
        reliability_plan = build_source_reliability_engine(route=RouteName.REVIEW)
        plan = build_knowledge_consolidation(
            route=RouteName.REVIEW,
            source_reliability=reliability_plan,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(
            plan.source_reliability_signal_ids,
            reliability_plan.signal_ids,
        )
        self.assertEqual(plan.source_count, reliability_plan.source_count)
        self.assertEqual(plan.domain_count, reliability_plan.domain_count)
        for signal in plan.signals:
            self.assertTrue(
                set(signal.source_reliability_signal_ids).issubset(
                    set(reliability_plan.signal_ids)
                )
            )
            self.assertFalse(signal.knowledge_merge_execution_implemented)
            self.assertFalse(signal.knowledge_deduplication_execution_implemented)
            self.assertFalse(signal.canonical_record_write_implemented)

    def test_knowledge_consolidation_preserves_routing_and_provider_factory(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Review knowledge consolidation posture for p5.js examples.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_knowledge_consolidation(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_knowledge_consolidation_omits_runtime_mutation_calls(self) -> None:
        plan = build_knowledge_consolidation(route=RouteName.GENERATE)
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
                        signal.consolidation_axis,
                        *signal.source_reliability_signal_ids,
                        signal.consolidation_signal_summary,
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
            "execute_knowledge_consolidation(",
            "generate_consolidation_candidate(",
            "merge_knowledge(",
            "deduplicate_knowledge(",
            "write_canonical_record(",
            "write_consolidation_record(",
            "write_kb_storage(",
            "update_source_record(",
            "score_source_reliability(",
            "check_source_health(",
            "compute_source_trust_score(",
            "mutate_source_rank(",
            "mutate_source_registry(",
            "fetch_source(",
            "scan_source_freshness(",
            "execute_drift_detection(",
            "scan_timeline(",
            "compare_snapshot(",
            "write_drift_record(",
            "execute_conflict_resolution(",
            "arbitrate_source(",
            "scan_for_gaps(",
            "remediate_gap(",
            "enrich_kb(",
            "compute_quality_score(",
            "execute_retrieval_query(",
            "mutate_retrieval_config(",
            "mutate_ranking(",
            "request_embedding(",
            "refresh_embedding(",
            "index_vectors(",
            "upsert_vectors(",
            "fetch_documentation(",
            "provision_provider(",
            "infer_api_key(",
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
