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
    SourceReliabilityEnginePlan,
    build_knowledge_drift_detection,
    build_source_reliability_engine,
    route_request,
    source_reliability_signal_by_id,
    source_reliability_signals_for_confidence,
    source_reliability_signals_for_status,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Source Reliability Engine",)
REQUIRED_SIGNAL_FIELDS = {
    "signal_id",
    "signal_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "reliability_axis",
    "knowledge_drift_signal_ids",
    "knowledge_drift_signal_count",
    "source_count",
    "domain_count",
    "reliability_signal_summary",
    "reliability_signal_score",
    "health_signal_score",
    "governance_alignment_score",
    "mutation_risk_score",
    "governance_weight",
    "reliability_score",
    "hitl_required_before_source_reliability",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "source_reliability_engine_capability_implemented",
    "source_reliability_engine_metadata_implemented",
    "knowledge_drift_metadata_used",
    "source_reliability_engine_execution_implemented",
    "source_reliability_scoring_execution_implemented",
    "source_health_check_execution_implemented",
    "source_trust_score_computation_implemented",
    "source_rank_mutation_implemented",
    "source_registry_mutation_implemented",
    "source_fetch_execution_implemented",
    "source_record_update_implemented",
    "freshness_scan_execution_implemented",
    "knowledge_drift_detection_execution_implemented",
    "drift_detection_execution_implemented",
    "drift_scan_execution_implemented",
    "timeline_scan_execution_implemented",
    "snapshot_comparison_execution_implemented",
    "drift_score_computation_implemented",
    "drift_record_write_implemented",
    "knowledge_conflict_resolver_execution_implemented",
    "conflict_detection_execution_implemented",
    "conflict_resolution_execution_implemented",
    "conflict_arbitration_execution_implemented",
    "conflict_record_write_implemented",
    "source_precedence_mutation_implemented",
    "gap_scan_execution_implemented",
    "gap_remediation_execution_implemented",
    "kb_enrichment_implemented",
    "kb_storage_write_implemented",
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


class SourceReliabilityEngineTests(unittest.TestCase):
    def test_plan_builds_source_reliability_metadata(self) -> None:
        plan = build_source_reliability_engine(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "source_reliability_engine")
        self.assertEqual(plan.serialization_version, "source_reliability_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.knowledge_drift_role, "knowledge_drift_detection")
        self.assertEqual(
            plan.knowledge_drift_serialization_version,
            "knowledge_drift_plan.v1",
        )
        self.assertEqual(len(plan.knowledge_drift_signal_ids), 5)
        self.assertEqual(plan.knowledge_drift_signal_count, 5)
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
        self.assertFalse(plan.planned_source_reliability_ids)
        self.assertFalse(plan.computed_source_reliability_ids)
        self.assertFalse(plan.checked_source_health_ids)
        self.assertFalse(plan.computed_source_trust_score_ids)
        self.assertFalse(plan.mutated_source_rank_ids)
        self.assertFalse(plan.mutated_source_registry_ids)
        self.assertFalse(plan.fetched_source_ids)
        self.assertFalse(plan.updated_source_record_ids)
        self.assertFalse(plan.scanned_freshness_ids)
        self.assertFalse(plan.written_kb_record_ids)
        self.assertEqual(plan.overall_reliability_posture, "guarded")
        self.assertIn(
            "does not execute source reliability engine",
            plan.authority_boundary,
        )
        self.assertIn("write KB storage", plan.authority_boundary)
        self.assertTrue(plan.source_reliability_engine_capability_implemented)
        self.assertTrue(plan.source_reliability_engine_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.knowledge_drift_metadata_used)
        self.assertFalse(plan.source_reliability_engine_execution_implemented)
        self.assertFalse(plan.source_reliability_scoring_execution_implemented)
        self.assertFalse(plan.source_health_check_execution_implemented)
        self.assertFalse(plan.source_trust_score_computation_implemented)
        self.assertFalse(plan.source_rank_mutation_implemented)
        self.assertFalse(plan.source_registry_mutation_implemented)
        self.assertFalse(plan.source_fetch_execution_implemented)
        self.assertFalse(plan.source_record_update_implemented)
        self.assertFalse(plan.freshness_scan_execution_implemented)
        self.assertFalse(plan.knowledge_drift_detection_execution_implemented)
        self.assertFalse(plan.drift_detection_execution_implemented)
        self.assertFalse(plan.drift_scan_execution_implemented)
        self.assertFalse(plan.timeline_scan_execution_implemented)
        self.assertFalse(plan.snapshot_comparison_execution_implemented)
        self.assertFalse(plan.drift_score_computation_implemented)
        self.assertFalse(plan.drift_record_write_implemented)
        self.assertFalse(plan.knowledge_conflict_resolver_execution_implemented)
        self.assertFalse(plan.conflict_detection_execution_implemented)
        self.assertFalse(plan.conflict_resolution_execution_implemented)
        self.assertFalse(plan.conflict_arbitration_execution_implemented)
        self.assertFalse(plan.conflict_record_write_implemented)
        self.assertFalse(plan.source_precedence_mutation_implemented)
        self.assertFalse(plan.gap_scan_execution_implemented)
        self.assertFalse(plan.gap_remediation_execution_implemented)
        self.assertFalse(plan.kb_enrichment_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
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

    def test_signals_score_source_reliability_without_execution(self) -> None:
        plan = build_source_reliability_engine(route="generate")
        drift_signal_ids = set(plan.knowledge_drift_signal_ids)

        for signal in plan.signals:
            dumped = signal.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SIGNAL_FIELDS)
            self.assertEqual(
                signal.serialization_version,
                "source_reliability_entry.v1",
            )
            self.assertEqual(signal.route_name, RouteName.GENERATE)
            self.assertEqual(
                signal.signal_id,
                f"source_reliability_engine::{signal.signal_kind}",
            )
            self.assertEqual(
                signal.knowledge_drift_signal_count,
                len(signal.knowledge_drift_signal_ids),
            )
            self.assertTrue(
                set(signal.knowledge_drift_signal_ids).issubset(drift_signal_ids)
            )
            self.assertEqual(
                signal.reliability_score,
                min(
                    1000,
                    max(
                        0,
                        signal.reliability_signal_score * 3
                        + signal.health_signal_score * 3
                        + signal.governance_alignment_score * 2
                        + signal.mutation_risk_score * 2
                        + signal.governance_weight,
                    ),
                ),
            )
            self.assertIn("source_reliability_engine", signal.context_tags)
            self.assertIn(
                "source_reliability_engine_execution",
                signal.blocked_runtime_behaviors,
            )
            self.assertIn(
                "source_health_check_execution",
                signal.blocked_runtime_behaviors,
            )
            self.assertIn("kb_storage_write", signal.blocked_runtime_behaviors)
            self.assertTrue(signal.hitl_required_before_source_reliability)
            self.assertTrue(signal.source_reliability_engine_capability_implemented)
            self.assertTrue(signal.source_reliability_engine_metadata_implemented)
            self.assertTrue(signal.knowledge_drift_metadata_used)
            self.assertFalse(signal.source_reliability_engine_execution_implemented)
            self.assertFalse(signal.source_reliability_scoring_execution_implemented)
            self.assertFalse(signal.source_health_check_execution_implemented)
            self.assertFalse(signal.source_trust_score_computation_implemented)
            self.assertFalse(signal.source_rank_mutation_implemented)
            self.assertFalse(signal.source_registry_mutation_implemented)
            self.assertFalse(signal.source_fetch_execution_implemented)
            self.assertFalse(signal.source_record_update_implemented)
            self.assertFalse(signal.freshness_scan_execution_implemented)
            self.assertFalse(signal.knowledge_drift_detection_execution_implemented)
            self.assertFalse(signal.drift_detection_execution_implemented)
            self.assertFalse(signal.drift_scan_execution_implemented)
            self.assertFalse(signal.timeline_scan_execution_implemented)
            self.assertFalse(signal.snapshot_comparison_execution_implemented)
            self.assertFalse(signal.drift_score_computation_implemented)
            self.assertFalse(signal.drift_record_write_implemented)
            self.assertFalse(signal.knowledge_conflict_resolver_execution_implemented)
            self.assertFalse(signal.conflict_detection_execution_implemented)
            self.assertFalse(signal.conflict_resolution_execution_implemented)
            self.assertFalse(signal.conflict_arbitration_execution_implemented)
            self.assertFalse(signal.conflict_record_write_implemented)
            self.assertFalse(signal.source_precedence_mutation_implemented)
            self.assertFalse(signal.gap_scan_execution_implemented)
            self.assertFalse(signal.gap_remediation_execution_implemented)
            self.assertFalse(signal.kb_enrichment_implemented)
            self.assertFalse(signal.kb_storage_write_implemented)
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

        inventory = source_reliability_signal_by_id(
            "source_reliability_engine::source_reliability_inventory_review",
            plan,
        )
        self.assertIsNotNone(inventory)
        assert inventory is not None
        self.assertEqual(inventory.status, "guarded")
        self.assertEqual(inventory.confidence, "guarded")
        self.assertEqual(
            len(source_reliability_signals_for_status("review_required", plan)),
            3,
        )
        self.assertEqual(
            len(source_reliability_signals_for_confidence("high", plan)),
            2,
        )

    def test_plan_rejects_mismatched_source_reliability_metadata(self) -> None:
        plan = build_source_reliability_engine()
        payload = plan.model_dump(mode="json")
        payload["signal_ids"] = ("missing",) + tuple(payload["signal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "signal_ids must match"):
            SourceReliabilityEnginePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_reliability_score"] -= 1

        with self.assertRaisesRegex(ValueError, "overall_reliability_score must match"):
            SourceReliabilityEnginePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["planned_source_reliability_ids"] = (plan.signal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "planned_source_reliability_ids must remain empty",
        ):
            SourceReliabilityEnginePlan(**payload)

    def test_source_reliability_composes_task_11_metadata(self) -> None:
        drift_plan = build_knowledge_drift_detection(route=RouteName.REVIEW)
        plan = build_source_reliability_engine(
            route=RouteName.REVIEW,
            knowledge_drift=drift_plan,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.knowledge_drift_signal_ids, drift_plan.signal_ids)
        self.assertEqual(plan.source_count, drift_plan.source_count)
        self.assertEqual(plan.domain_count, drift_plan.domain_count)
        for signal in plan.signals:
            self.assertTrue(
                set(signal.knowledge_drift_signal_ids).issubset(
                    set(drift_plan.signal_ids)
                )
            )
            self.assertFalse(signal.source_reliability_scoring_execution_implemented)
            self.assertFalse(signal.source_health_check_execution_implemented)
            self.assertFalse(signal.source_fetch_execution_implemented)

    def test_source_reliability_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review source reliability posture for p5.js examples.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_source_reliability_engine(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_source_reliability_omits_runtime_mutation_calls(self) -> None:
        plan = build_source_reliability_engine(route=RouteName.GENERATE)
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
                        signal.reliability_axis,
                        *signal.knowledge_drift_signal_ids,
                        signal.reliability_signal_summary,
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
            "execute_source_reliability_engine(",
            "score_source_reliability(",
            "check_source_health(",
            "compute_source_trust_score(",
            "mutate_source_rank(",
            "mutate_source_registry(",
            "fetch_source(",
            "update_source_record(",
            "scan_source_freshness(",
            "execute_drift_detection(",
            "detect_drift(",
            "scan_timeline(",
            "compare_snapshot(",
            "compute_drift_score(",
            "write_drift_record(",
            "execute_conflict_resolution(",
            "detect_conflict(",
            "resolve_conflict(",
            "arbitrate_source(",
            "write_conflict_record(",
            "mutate_source_precedence(",
            "scan_for_gaps(",
            "remediate_gap(",
            "enrich_kb(",
            "write_kb_storage(",
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
