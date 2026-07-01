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
    ArtifactHistoryPlan,
    artifact_history_record_by_id,
    artifact_history_records_for_confidence,
    artifact_history_records_for_status,
    build_artifact_history,
    multimodal_workspace_history_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_RECORD_FIELDS = {
    "artifact_history_id",
    "history_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "history_scope",
    "source_session_evolution_signal_id",
    "source_project_memory_signal_id",
    "source_long_term_memory_record_id",
    "source_workspace_history_profile_id",
    "history_summary",
    "history_continuity_score",
    "provenance_strength_score",
    "session_alignment_score",
    "mutation_risk_score",
    "governance_weight",
    "artifact_history_score",
    "hitl_required_before_history_persistence",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "artifact_history_implemented",
    "artifact_history_metadata_implemented",
    "session_memory_evolution_source_used",
    "project_memory_source_used",
    "long_term_memory_source_used",
    "workspace_history_source_used",
    "artifact_history_storage_write_implemented",
    "artifact_history_record_creation_implemented",
    "artifact_history_record_update_implemented",
    "artifact_history_record_deletion_implemented",
    "artifact_history_reconstruction_implemented",
    "artifact_history_application_implemented",
    "workspace_history_recording_implemented",
    "workspace_history_persistence_implemented",
    "artifact_mutation_implemented",
    "session_recording_implemented",
    "session_replay_execution_implemented",
    "memory_retrieval_execution_implemented",
    "memory_storage_write_implemented",
    "memory_consolidation_implemented",
    "personalization_application_implemented",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
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


class ArtifactHistoryTests(unittest.TestCase):
    def test_plan_builds_artifact_history_metadata(self) -> None:
        workspace_history = multimodal_workspace_history_registry()
        plan = build_artifact_history(
            route=RouteName.GENERATE,
            workspace_history=workspace_history,
        )

        self.assertEqual(plan.role, "artifact_history")
        self.assertEqual(plan.serialization_version, "artifact_history_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_session_evolution_serialization_version,
            "session_memory_evolution_plan.v1",
        )
        self.assertEqual(
            plan.source_project_memory_serialization_version,
            "project_memory_plan.v1",
        )
        self.assertEqual(
            plan.source_long_term_memory_serialization_version,
            "long_term_creative_memory_plan.v1",
        )
        self.assertEqual(
            plan.source_workspace_history_serialization_version,
            "multimodal_workspace_history_registry.v1",
        )
        self.assertEqual(
            plan.source_workspace_history_profile_ids,
            workspace_history.profile_ids,
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.record_count, 5)
        self.assertEqual(plan.candidate_record_count, 1)
        self.assertEqual(plan.review_required_record_count, 2)
        self.assertEqual(plan.guarded_record_count, 2)
        self.assertEqual(plan.high_confidence_record_count, 3)
        self.assertEqual(plan.hitl_required_record_count, 5)
        self.assertFalse(plan.persisted_artifact_history_ids)
        self.assertFalse(plan.reconstructed_artifact_history_ids)
        self.assertFalse(plan.mutated_artifact_ids)
        self.assertFalse(plan.recorded_workspace_history_ids)
        self.assertEqual(plan.overall_artifact_history_posture, "guarded")
        self.assertIn(
            "does not write artifact history storage",
            plan.authority_boundary,
        )
        self.assertIn("reconstruct artifact timelines", plan.authority_boundary)
        self.assertTrue(plan.artifact_history_implemented)
        self.assertTrue(plan.artifact_history_metadata_implemented)
        self.assertTrue(plan.session_memory_evolution_source_used)
        self.assertTrue(plan.project_memory_source_used)
        self.assertTrue(plan.long_term_memory_source_used)
        self.assertTrue(plan.workspace_history_source_used)
        self.assertFalse(plan.artifact_history_storage_write_implemented)
        self.assertFalse(plan.artifact_history_record_creation_implemented)
        self.assertFalse(plan.artifact_history_record_update_implemented)
        self.assertFalse(plan.artifact_history_record_deletion_implemented)
        self.assertFalse(plan.artifact_history_reconstruction_implemented)
        self.assertFalse(plan.artifact_history_application_implemented)
        self.assertFalse(plan.workspace_history_recording_implemented)
        self.assertFalse(plan.workspace_history_persistence_implemented)
        self.assertFalse(plan.artifact_mutation_implemented)
        self.assertFalse(plan.session_recording_implemented)
        self.assertFalse(plan.session_replay_execution_implemented)
        self.assertFalse(plan.memory_retrieval_execution_implemented)
        self.assertFalse(plan.memory_storage_write_implemented)
        self.assertFalse(plan.memory_consolidation_implemented)
        self.assertFalse(plan.personalization_application_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_records_score_artifact_history_without_reconstruction(self) -> None:
        plan = build_artifact_history(route="generate")

        for record in plan.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "artifact_history_record.v1",
            )
            self.assertEqual(record.route_name, RouteName.GENERATE)
            self.assertEqual(
                record.artifact_history_id,
                f"artifact_history::{record.history_kind}",
            )
            self.assertEqual(
                record.artifact_history_score,
                min(
                    1000,
                    max(
                        0,
                        record.history_continuity_score * 3
                        + record.provenance_strength_score * 3
                        + record.session_alignment_score * 2
                        + record.mutation_risk_score * 3
                        + record.governance_weight,
                    ),
                ),
            )
            self.assertIn("artifact_history", record.context_tags)
            self.assertIn(
                "artifact_history_reconstruction",
                record.blocked_runtime_behaviors,
            )
            self.assertIn("artifact_mutation", record.blocked_runtime_behaviors)
            self.assertTrue(record.explainability_notes)
            self.assertTrue(record.advisory_actions)
            self.assertTrue(record.evidence)
            self.assertTrue(record.hitl_required_before_history_persistence)
            self.assertTrue(record.artifact_history_implemented)
            self.assertTrue(record.artifact_history_metadata_implemented)
            self.assertTrue(record.session_memory_evolution_source_used)
            self.assertTrue(record.project_memory_source_used)
            self.assertTrue(record.long_term_memory_source_used)
            self.assertTrue(record.workspace_history_source_used)
            self.assertFalse(record.artifact_history_storage_write_implemented)
            self.assertFalse(record.artifact_history_record_creation_implemented)
            self.assertFalse(record.artifact_history_record_update_implemented)
            self.assertFalse(record.artifact_history_record_deletion_implemented)
            self.assertFalse(record.artifact_history_reconstruction_implemented)
            self.assertFalse(record.artifact_history_application_implemented)
            self.assertFalse(record.workspace_history_recording_implemented)
            self.assertFalse(record.workspace_history_persistence_implemented)
            self.assertFalse(record.artifact_mutation_implemented)
            self.assertFalse(record.session_recording_implemented)
            self.assertFalse(record.session_replay_execution_implemented)
            self.assertFalse(record.memory_retrieval_execution_implemented)
            self.assertFalse(record.memory_storage_write_implemented)
            self.assertFalse(record.memory_consolidation_implemented)
            self.assertFalse(record.personalization_application_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.provider_execution_implemented)
            self.assertFalse(record.agent_invocation_implemented)
            self.assertFalse(record.workflow_control_implemented)
            self.assertFalse(record.workflow_graph_mutation_implemented)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.persistent_storage_write_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertTrue(record.advisory_only)

        iteration = artifact_history_record_by_id(
            "artifact_history::iteration_history",
            plan,
        )
        self.assertIsNotNone(iteration)
        assert iteration is not None
        self.assertEqual(iteration.status, "guarded")
        self.assertEqual(iteration.confidence, "guarded")
        self.assertEqual(len(artifact_history_records_for_status("guarded", plan)), 2)
        self.assertEqual(
            len(artifact_history_records_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_artifact_history_metadata(self) -> None:
        plan = build_artifact_history()
        payload = plan.model_dump(mode="json")
        payload["record_ids"] = ("missing",) + tuple(payload["record_ids"][1:])

        with self.assertRaisesRegex(ValueError, "record_ids must match"):
            ArtifactHistoryPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_artifact_history_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_artifact_history_score must match",
        ):
            ArtifactHistoryPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["mutated_artifact_ids"] = (plan.record_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "mutated_artifact_ids must remain empty",
        ):
            ArtifactHistoryPlan(**payload)

    def test_artifact_history_does_not_change_routing_or_provider_factory(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Review artifact history for a creative coding project.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_artifact_history(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_artifact_history_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_artifact_history(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for record in plan.records
                    for field in (
                        record.artifact_history_id,
                        record.history_kind,
                        record.status,
                        record.confidence,
                        record.history_scope,
                        record.source_session_evolution_signal_id,
                        record.source_project_memory_signal_id,
                        record.source_long_term_memory_record_id,
                        record.source_workspace_history_profile_id,
                        record.history_summary,
                        *record.context_tags,
                        *record.explainability_notes,
                        *record.advisory_actions,
                        *record.evidence,
                        *record.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "write_artifact_history(",
            "create_artifact_history(",
            "update_artifact_history(",
            "delete_artifact_history(",
            "reconstruct_artifact_history(",
            "apply_artifact_history(",
            "record_workspace_history(",
            "persist_workspace_history(",
            "mutate_artifact(",
            "record_session(",
            "execute_session_replay(",
            "retrieve_memory(",
            "write_memory(",
            "consolidate_memory(",
            "apply_personalization(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
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
