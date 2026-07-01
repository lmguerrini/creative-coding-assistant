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
    KnowledgeHealthMonitoringPlan,
    build_knowledge_health_monitoring,
    build_ranking_optimization,
    knowledge_health_signal_by_id,
    knowledge_health_signals_for_confidence,
    knowledge_health_signals_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Knowledge Health Monitoring",)
REQUIRED_SIGNAL_FIELDS = {
    "signal_id",
    "signal_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "health_axis",
    "ranking_optimization_signal_ids",
    "ranking_optimization_signal_count",
    "source_count",
    "domain_count",
    "health_signal_summary",
    "health_signal_score",
    "reliability_signal_score",
    "governance_alignment_score",
    "mutation_risk_score",
    "governance_weight",
    "health_score",
    "hitl_required_before_health_monitoring",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "knowledge_health_monitoring_capability_implemented",
    "knowledge_health_monitoring_metadata_implemented",
    "ranking_optimization_metadata_used",
    "knowledge_health_monitoring_execution_implemented",
    "health_metric_collection_implemented",
    "health_monitor_mutation_implemented",
    "health_alert_emission_implemented",
    "ranking_optimization_execution_implemented",
    "ranking_mutation_implemented",
    "retrieval_query_execution_implemented",
    "retrieval_configuration_mutation_implemented",
    "retrieval_reranking_implemented",
    "embedding_request_execution_implemented",
    "embedding_refresh_execution_implemented",
    "vector_indexing_implemented",
    "vector_upsert_implemented",
    "kb_storage_write_implemented",
    "documentation_fetch_execution_implemented",
    "kb_enrichment_implemented",
    "source_record_update_implemented",
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


class KnowledgeHealthMonitoringTests(unittest.TestCase):
    def test_plan_builds_knowledge_health_monitoring_metadata(self) -> None:
        plan = build_knowledge_health_monitoring(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "knowledge_health_monitoring")
        self.assertEqual(plan.serialization_version, "knowledge_health_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.ranking_optimization_role, "ranking_optimization")
        self.assertEqual(
            plan.ranking_optimization_serialization_version,
            "ranking_optimization_plan.v1",
        )
        self.assertEqual(len(plan.ranking_optimization_signal_ids), 5)
        self.assertEqual(plan.ranking_optimization_signal_count, 5)
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
        self.assertFalse(plan.planned_health_monitor_ids)
        self.assertFalse(plan.collected_health_metric_ids)
        self.assertFalse(plan.emitted_health_alert_ids)
        self.assertFalse(plan.mutated_health_monitor_ids)
        self.assertFalse(plan.mutated_ranking_profile_ids)
        self.assertFalse(plan.executed_retrieval_query_ids)
        self.assertFalse(plan.written_kb_record_ids)
        self.assertEqual(plan.overall_health_posture, "guarded")
        self.assertIn("does not execute health monitoring", plan.authority_boundary)
        self.assertIn("write KB storage", plan.authority_boundary)
        self.assertTrue(plan.knowledge_health_monitoring_capability_implemented)
        self.assertTrue(plan.knowledge_health_monitoring_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.ranking_optimization_metadata_used)
        self.assertFalse(plan.knowledge_health_monitoring_execution_implemented)
        self.assertFalse(plan.health_metric_collection_implemented)
        self.assertFalse(plan.health_monitor_mutation_implemented)
        self.assertFalse(plan.health_alert_emission_implemented)
        self.assertFalse(plan.ranking_optimization_execution_implemented)
        self.assertFalse(plan.ranking_mutation_implemented)
        self.assertFalse(plan.retrieval_query_execution_implemented)
        self.assertFalse(plan.retrieval_configuration_mutation_implemented)
        self.assertFalse(plan.retrieval_reranking_implemented)
        self.assertFalse(plan.embedding_request_execution_implemented)
        self.assertFalse(plan.embedding_refresh_execution_implemented)
        self.assertFalse(plan.vector_indexing_implemented)
        self.assertFalse(plan.vector_upsert_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
        self.assertFalse(plan.documentation_fetch_execution_implemented)
        self.assertFalse(plan.kb_enrichment_implemented)
        self.assertFalse(plan.source_record_update_implemented)
        self.assertFalse(plan.provider_provisioning_implemented)
        self.assertFalse(plan.api_key_inference_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_signals_score_knowledge_health_without_execution(self) -> None:
        plan = build_knowledge_health_monitoring(route="generate")
        ranking_signal_ids = set(plan.ranking_optimization_signal_ids)

        for signal in plan.signals:
            dumped = signal.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SIGNAL_FIELDS)
            self.assertEqual(signal.serialization_version, "knowledge_health_entry.v1")
            self.assertEqual(signal.route_name, RouteName.GENERATE)
            self.assertEqual(
                signal.signal_id,
                f"knowledge_health_monitoring::{signal.signal_kind}",
            )
            self.assertEqual(
                signal.ranking_optimization_signal_count,
                len(signal.ranking_optimization_signal_ids),
            )
            self.assertTrue(
                set(signal.ranking_optimization_signal_ids).issubset(
                    ranking_signal_ids
                )
            )
            self.assertEqual(
                signal.health_score,
                min(
                    1000,
                    max(
                        0,
                        signal.health_signal_score * 3
                        + signal.reliability_signal_score * 3
                        + signal.governance_alignment_score * 2
                        + signal.mutation_risk_score * 2
                        + signal.governance_weight,
                    ),
                ),
            )
            self.assertIn("knowledge_health_monitoring", signal.context_tags)
            self.assertIn(
                "knowledge_health_monitoring_execution",
                signal.blocked_runtime_behaviors,
            )
            self.assertIn(
                "health_metric_collection_execution",
                signal.blocked_runtime_behaviors,
            )
            self.assertIn("kb_storage_write", signal.blocked_runtime_behaviors)
            self.assertTrue(signal.hitl_required_before_health_monitoring)
            self.assertTrue(signal.knowledge_health_monitoring_capability_implemented)
            self.assertTrue(signal.knowledge_health_monitoring_metadata_implemented)
            self.assertTrue(signal.ranking_optimization_metadata_used)
            self.assertFalse(signal.knowledge_health_monitoring_execution_implemented)
            self.assertFalse(signal.health_metric_collection_implemented)
            self.assertFalse(signal.health_monitor_mutation_implemented)
            self.assertFalse(signal.health_alert_emission_implemented)
            self.assertFalse(signal.ranking_optimization_execution_implemented)
            self.assertFalse(signal.ranking_mutation_implemented)
            self.assertFalse(signal.retrieval_query_execution_implemented)
            self.assertFalse(signal.retrieval_configuration_mutation_implemented)
            self.assertFalse(signal.retrieval_reranking_implemented)
            self.assertFalse(signal.embedding_request_execution_implemented)
            self.assertFalse(signal.embedding_refresh_execution_implemented)
            self.assertFalse(signal.vector_indexing_implemented)
            self.assertFalse(signal.vector_upsert_implemented)
            self.assertFalse(signal.kb_storage_write_implemented)
            self.assertFalse(signal.documentation_fetch_execution_implemented)
            self.assertFalse(signal.kb_enrichment_implemented)
            self.assertFalse(signal.source_record_update_implemented)
            self.assertFalse(signal.provider_provisioning_implemented)
            self.assertFalse(signal.api_key_inference_implemented)
            self.assertFalse(signal.provider_model_routing_implemented)
            self.assertFalse(signal.provider_execution_implemented)
            self.assertFalse(signal.workflow_control_implemented)
            self.assertFalse(signal.generated_output_mutation_implemented)
            self.assertFalse(signal.runtime_evolution_implemented)
            self.assertTrue(signal.advisory_only)

        inventory = knowledge_health_signal_by_id(
            "knowledge_health_monitoring::knowledge_health_inventory_review",
            plan,
        )
        self.assertIsNotNone(inventory)
        assert inventory is not None
        self.assertEqual(inventory.status, "guarded")
        self.assertEqual(inventory.confidence, "guarded")
        self.assertEqual(
            len(knowledge_health_signals_for_status("review_required", plan)),
            3,
        )
        self.assertEqual(
            len(knowledge_health_signals_for_confidence("high", plan)),
            2,
        )

    def test_plan_rejects_mismatched_knowledge_health_metadata(self) -> None:
        plan = build_knowledge_health_monitoring()
        payload = plan.model_dump(mode="json")
        payload["signal_ids"] = ("missing",) + tuple(payload["signal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "signal_ids must match"):
            KnowledgeHealthMonitoringPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_health_score"] -= 1

        with self.assertRaisesRegex(ValueError, "overall_health_score must match"):
            KnowledgeHealthMonitoringPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["planned_health_monitor_ids"] = (plan.signal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "planned_health_monitor_ids must remain empty",
        ):
            KnowledgeHealthMonitoringPlan(**payload)

    def test_knowledge_health_composes_task_6_metadata(self) -> None:
        ranking_plan = build_ranking_optimization(route=RouteName.REVIEW)
        plan = build_knowledge_health_monitoring(
            route=RouteName.REVIEW,
            ranking_optimization=ranking_plan,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(
            plan.ranking_optimization_signal_ids,
            ranking_plan.signal_ids,
        )
        self.assertEqual(plan.source_count, ranking_plan.source_count)
        self.assertEqual(plan.domain_count, ranking_plan.domain_count)
        for signal in plan.signals:
            self.assertTrue(
                set(signal.ranking_optimization_signal_ids).issubset(
                    set(ranking_plan.signal_ids)
                )
            )
            self.assertFalse(signal.health_metric_collection_implemented)
            self.assertFalse(signal.health_alert_emission_implemented)

    def test_knowledge_health_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review knowledge health monitoring posture for p5.js examples.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_knowledge_health_monitoring(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_knowledge_health_omits_runtime_mutation_calls(self) -> None:
        plan = build_knowledge_health_monitoring(route=RouteName.GENERATE)
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
                        signal.health_axis,
                        *signal.ranking_optimization_signal_ids,
                        signal.health_signal_summary,
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
            "execute_health_monitoring(",
            "collect_health_metrics(",
            "mutate_health_monitor(",
            "emit_health_alert(",
            "execute_ranking_optimization(",
            "mutate_ranking(",
            "execute_retrieval_query(",
            "mutate_retrieval_config(",
            "rerank_retrieval_results(",
            "request_embedding(",
            "refresh_embedding(",
            "index_vectors(",
            "upsert_vectors(",
            "write_kb_storage(",
            "fetch_documentation(",
            "enrich_kb(",
            "update_source_record(",
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
