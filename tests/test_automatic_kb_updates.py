import unittest
from datetime import UTC, datetime, timedelta

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
    AutomaticKBUpdatePlan,
    automatic_kb_update_candidate_by_id,
    automatic_kb_update_candidates_for_confidence,
    automatic_kb_update_candidates_for_status,
    build_automatic_kb_updates,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.rag import (
    OfficialSourceSyncMetadata,
    SourceSyncStatus,
    get_official_source,
)

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Automatic KB Updates",)
REQUIRED_CANDIDATE_FIELDS = {
    "update_id",
    "update_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "update_axis",
    "source_ids",
    "source_count",
    "domain_count",
    "source_type_count",
    "unknown_health_count",
    "stale_health_count",
    "refresh_recommended_count",
    "update_summary",
    "source_registry_score",
    "health_metadata_score",
    "governance_alignment_score",
    "mutation_risk_score",
    "governance_weight",
    "update_score",
    "hitl_required_before_update_execution",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "automatic_kb_updates_implemented",
    "automatic_kb_update_metadata_implemented",
    "official_source_registry_used",
    "source_health_metadata_used",
    "sync_request_metadata_used",
    "automatic_update_execution_implemented",
    "source_fetch_execution_implemented",
    "source_normalization_execution_implemented",
    "source_chunking_execution_implemented",
    "embedding_request_execution_implemented",
    "embedding_refresh_execution_implemented",
    "vector_record_indexing_implemented",
    "vector_record_upsert_implemented",
    "kb_storage_write_implemented",
    "source_registry_mutation_implemented",
    "retrieval_configuration_mutation_implemented",
    "retrieval_execution_implemented",
    "ranking_mutation_implemented",
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


class AutomaticKBUpdateTests(unittest.TestCase):
    def test_plan_builds_advisory_update_metadata(self) -> None:
        plan = build_automatic_kb_updates(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "automatic_kb_updates")
        self.assertEqual(plan.serialization_version, "automatic_kb_update_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.source_registry_role, "approved_official_sources")
        self.assertEqual(
            plan.source_registry_serialization_version,
            "official_source_registry.v1",
        )
        self.assertEqual(
            plan.source_health_serialization_version,
            "official_source_health_snapshot.v1",
        )
        self.assertEqual(
            plan.sync_request_serialization_version,
            "official_source_sync_request.v1",
        )
        self.assertEqual(plan.covered_roadmap_items, EXPECTED_ROADMAP_ITEMS)
        self.assertEqual(plan.covered_roadmap_item_count, 1)
        self.assertEqual(plan.source_count, 57)
        self.assertEqual(len(plan.source_ids), 57)
        self.assertEqual(plan.domain_count, 43)
        self.assertEqual(plan.source_type_count, 4)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.candidate_count, 5)
        self.assertEqual(plan.candidate_entry_count, 5)
        self.assertEqual(plan.candidate_status_count, 1)
        self.assertEqual(plan.review_required_candidate_count, 3)
        self.assertEqual(plan.guarded_candidate_count, 1)
        self.assertEqual(plan.high_confidence_candidate_count, 2)
        self.assertEqual(plan.hitl_required_candidate_count, 5)
        self.assertFalse(plan.planned_update_execution_ids)
        self.assertFalse(plan.fetched_source_ids)
        self.assertFalse(plan.indexed_source_ids)
        self.assertFalse(plan.mutated_retrieval_source_ids)
        self.assertFalse(plan.written_storage_record_ids)
        self.assertEqual(plan.overall_update_posture, "guarded")
        self.assertIn("does not fetch sources", plan.authority_boundary)
        self.assertIn("write KB storage", plan.authority_boundary)
        self.assertTrue(plan.automatic_kb_updates_implemented)
        self.assertTrue(plan.automatic_kb_update_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.official_source_registry_used)
        self.assertTrue(plan.source_health_metadata_used)
        self.assertTrue(plan.sync_request_metadata_used)
        self.assertFalse(plan.automatic_update_execution_implemented)
        self.assertFalse(plan.source_fetch_execution_implemented)
        self.assertFalse(plan.embedding_request_execution_implemented)
        self.assertFalse(plan.embedding_refresh_execution_implemented)
        self.assertFalse(plan.vector_record_indexing_implemented)
        self.assertFalse(plan.vector_record_upsert_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
        self.assertFalse(plan.source_registry_mutation_implemented)
        self.assertFalse(plan.retrieval_configuration_mutation_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.ranking_mutation_implemented)
        self.assertFalse(plan.provider_provisioning_implemented)
        self.assertFalse(plan.api_key_inference_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_candidates_score_updates_without_execution(self) -> None:
        plan = build_automatic_kb_updates(route="generate")
        plan_source_ids = set(plan.source_ids)

        for candidate in plan.candidates:
            dumped = candidate.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CANDIDATE_FIELDS)
            self.assertEqual(
                candidate.serialization_version,
                "automatic_kb_update_entry.v1",
            )
            self.assertEqual(candidate.route_name, RouteName.GENERATE)
            self.assertEqual(
                candidate.update_id,
                f"automatic_kb_updates::{candidate.update_kind}",
            )
            self.assertEqual(candidate.source_count, len(candidate.source_ids))
            self.assertTrue(set(candidate.source_ids).issubset(plan_source_ids))
            self.assertEqual(
                candidate.update_score,
                min(
                    1000,
                    max(
                        0,
                        candidate.source_registry_score * 3
                        + candidate.health_metadata_score * 2
                        + candidate.governance_alignment_score * 3
                        + candidate.mutation_risk_score * 2
                        + candidate.governance_weight,
                    ),
                ),
            )
            self.assertIn("automatic_kb_updates", candidate.context_tags)
            self.assertIn(
                "automatic_kb_update_execution",
                candidate.blocked_runtime_behaviors,
            )
            self.assertIn("kb_storage_write", candidate.blocked_runtime_behaviors)
            self.assertIn("retrieval_execution", candidate.blocked_runtime_behaviors)
            self.assertTrue(candidate.hitl_required_before_update_execution)
            self.assertTrue(candidate.automatic_kb_updates_implemented)
            self.assertTrue(candidate.automatic_kb_update_metadata_implemented)
            self.assertTrue(candidate.official_source_registry_used)
            self.assertTrue(candidate.source_health_metadata_used)
            self.assertTrue(candidate.sync_request_metadata_used)
            self.assertFalse(candidate.automatic_update_execution_implemented)
            self.assertFalse(candidate.source_fetch_execution_implemented)
            self.assertFalse(candidate.source_normalization_execution_implemented)
            self.assertFalse(candidate.source_chunking_execution_implemented)
            self.assertFalse(candidate.embedding_request_execution_implemented)
            self.assertFalse(candidate.embedding_refresh_execution_implemented)
            self.assertFalse(candidate.vector_record_indexing_implemented)
            self.assertFalse(candidate.vector_record_upsert_implemented)
            self.assertFalse(candidate.kb_storage_write_implemented)
            self.assertFalse(candidate.source_registry_mutation_implemented)
            self.assertFalse(candidate.retrieval_configuration_mutation_implemented)
            self.assertFalse(candidate.retrieval_execution_implemented)
            self.assertFalse(candidate.ranking_mutation_implemented)
            self.assertFalse(candidate.provider_provisioning_implemented)
            self.assertFalse(candidate.api_key_inference_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.provider_execution_implemented)
            self.assertFalse(candidate.workflow_control_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertFalse(candidate.runtime_evolution_implemented)
            self.assertTrue(candidate.advisory_only)

        registry = automatic_kb_update_candidate_by_id(
            "automatic_kb_updates::approved_source_registry_monitor",
            plan,
        )
        self.assertIsNotNone(registry)
        assert registry is not None
        self.assertEqual(registry.status, "guarded")
        self.assertEqual(registry.confidence, "guarded")
        self.assertEqual(
            len(automatic_kb_update_candidates_for_status("review_required", plan)),
            3,
        )
        self.assertEqual(
            len(automatic_kb_update_candidates_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_update_metadata(self) -> None:
        plan = build_automatic_kb_updates()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            AutomaticKBUpdatePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_update_score"] -= 1

        with self.assertRaisesRegex(ValueError, "overall_update_score must match"):
            AutomaticKBUpdatePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["planned_update_execution_ids"] = (plan.candidate_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "planned_update_execution_ids must remain empty",
        ):
            AutomaticKBUpdatePlan(**payload)

    def test_source_health_metadata_marks_stale_without_sync(self) -> None:
        checked_at = _time() + timedelta(days=200)
        source = get_official_source("three_docs")
        stale_sync = OfficialSourceSyncMetadata(
            source_id=source.source_id,
            domain=source.domain,
            source_type=source.source_type,
            source_url=source.url,
            resolved_url=source.url,
            sync_status=SourceSyncStatus.SUCCEEDED,
            requested_at=_time(),
            last_synced_at=_time(),
            completed_at=_time(),
            content_hash="a" * 64,
            chunk_count=3,
            record_count=3,
        )

        plan = build_automatic_kb_updates(
            sync_metadata_by_source_id={source.source_id: stale_sync},
            checked_at=checked_at,
        )
        freshness = automatic_kb_update_candidate_by_id(
            "automatic_kb_updates::freshness_policy_monitor",
            plan,
        )

        self.assertIsNotNone(freshness)
        assert freshness is not None
        self.assertEqual(freshness.stale_health_count, 1)
        self.assertEqual(freshness.refresh_recommended_count, 1)
        self.assertEqual(freshness.unknown_health_count, freshness.source_count - 1)
        self.assertFalse(freshness.source_fetch_execution_implemented)
        self.assertFalse(freshness.kb_storage_write_implemented)

    def test_automatic_kb_updates_preserves_routing_and_provider_factory(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Review automatic KB update posture for p5.js examples.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_automatic_kb_updates(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_automatic_kb_updates_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_automatic_kb_updates(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *plan.covered_roadmap_items,
                *(
                    field
                    for candidate in plan.candidates
                    for field in (
                        candidate.update_id,
                        candidate.update_kind,
                        candidate.status,
                        candidate.confidence,
                        candidate.update_axis,
                        *candidate.source_ids,
                        candidate.update_summary,
                        *candidate.context_tags,
                        *candidate.explainability_notes,
                        *candidate.advisory_actions,
                        *candidate.evidence,
                        *candidate.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_automatic_kb_update(",
            "fetch_source(",
            "normalize_source(",
            "chunk_source(",
            "request_embedding(",
            "refresh_embedding(",
            "index_vector_record(",
            "upsert_vector_record(",
            "write_kb_storage(",
            "mutate_source_registry(",
            "mutate_retrieval_config(",
            "execute_retrieval(",
            "mutate_ranking(",
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


def _time() -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


if __name__ == "__main__":
    unittest.main()
