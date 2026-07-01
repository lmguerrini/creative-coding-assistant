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
    SessionMemoryEvolutionPlan,
    build_personalization_engine,
    build_session_memory_evolution,
    route_request,
    session_memory_evolution_signal_by_id,
    session_memory_evolution_signals_for_confidence,
    session_memory_evolution_signals_for_status,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_SIGNAL_FIELDS = {
    "evolution_id",
    "evolution_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "evolution_scope",
    "source_personalization_id",
    "source_creative_dna_id",
    "source_long_term_memory_record_id",
    "source_project_memory_signal_id",
    "evolution_summary",
    "session_continuity_score",
    "personalization_alignment_score",
    "memory_stability_score",
    "drift_risk_score",
    "governance_weight",
    "session_evolution_score",
    "hitl_required_before_session_memory_update",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "session_memory_evolution_implemented",
    "session_memory_evolution_metadata_implemented",
    "personalization_source_used",
    "creative_dna_source_used",
    "long_term_memory_source_used",
    "project_memory_source_used",
    "session_memory_storage_write_implemented",
    "session_memory_record_creation_implemented",
    "session_memory_record_update_implemented",
    "session_memory_record_deletion_implemented",
    "session_memory_evolution_application_implemented",
    "session_recording_implemented",
    "session_replay_execution_implemented",
    "memory_retrieval_execution_implemented",
    "memory_storage_write_implemented",
    "memory_consolidation_implemented",
    "personalization_application_implemented",
    "creative_dna_application_implemented",
    "preference_mutation_implemented",
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


class SessionMemoryEvolutionTests(unittest.TestCase):
    def test_plan_builds_session_memory_evolution_metadata(self) -> None:
        personalization = build_personalization_engine(route=RouteName.GENERATE)
        plan = build_session_memory_evolution(
            route=RouteName.GENERATE,
            personalization_engine=personalization,
        )

        self.assertEqual(plan.role, "session_memory_evolution")
        self.assertEqual(
            plan.serialization_version,
            "session_memory_evolution_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_personalization_engine_serialization_version,
            "personalization_engine_plan.v1",
        )
        self.assertEqual(
            plan.source_creative_dna_serialization_version,
            "creative_dna_plan.v1",
        )
        self.assertEqual(
            plan.source_long_term_memory_serialization_version,
            "long_term_creative_memory_plan.v1",
        )
        self.assertEqual(
            plan.source_project_memory_serialization_version,
            "project_memory_plan.v1",
        )
        self.assertEqual(
            plan.source_personalization_ids,
            personalization.recommendation_ids,
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.signal_count, 5)
        self.assertEqual(plan.candidate_signal_count, 1)
        self.assertEqual(plan.review_required_signal_count, 2)
        self.assertEqual(plan.guarded_signal_count, 2)
        self.assertEqual(plan.high_confidence_signal_count, 3)
        self.assertEqual(plan.hitl_required_signal_count, 5)
        self.assertFalse(plan.evolved_session_memory_ids)
        self.assertFalse(plan.persisted_session_memory_ids)
        self.assertFalse(plan.replayed_session_ids)
        self.assertFalse(plan.consolidated_memory_ids)
        self.assertEqual(plan.overall_session_evolution_posture, "guarded")
        self.assertIn(
            "does not write session memory storage",
            plan.authority_boundary,
        )
        self.assertIn("does not write session memory", plan.authority_boundary)
        self.assertTrue(plan.session_memory_evolution_implemented)
        self.assertTrue(plan.session_memory_evolution_metadata_implemented)
        self.assertTrue(plan.personalization_source_used)
        self.assertTrue(plan.creative_dna_source_used)
        self.assertTrue(plan.long_term_memory_source_used)
        self.assertTrue(plan.project_memory_source_used)
        self.assertFalse(plan.session_memory_storage_write_implemented)
        self.assertFalse(plan.session_memory_record_creation_implemented)
        self.assertFalse(plan.session_memory_record_update_implemented)
        self.assertFalse(plan.session_memory_record_deletion_implemented)
        self.assertFalse(plan.session_memory_evolution_application_implemented)
        self.assertFalse(plan.session_recording_implemented)
        self.assertFalse(plan.session_replay_execution_implemented)
        self.assertFalse(plan.memory_retrieval_execution_implemented)
        self.assertFalse(plan.memory_storage_write_implemented)
        self.assertFalse(plan.memory_consolidation_implemented)
        self.assertFalse(plan.personalization_application_implemented)
        self.assertFalse(plan.creative_dna_application_implemented)
        self.assertFalse(plan.preference_mutation_implemented)
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

    def test_signals_score_session_memory_evolution_without_writes(self) -> None:
        plan = build_session_memory_evolution(route="generate")

        for signal in plan.signals:
            dumped = signal.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SIGNAL_FIELDS)
            self.assertEqual(
                signal.serialization_version,
                "session_memory_evolution_signal.v1",
            )
            self.assertEqual(signal.route_name, RouteName.GENERATE)
            self.assertEqual(
                signal.evolution_id,
                f"session_memory_evolution::{signal.evolution_kind}",
            )
            self.assertEqual(
                signal.session_evolution_score,
                min(
                    1000,
                    max(
                        0,
                        signal.session_continuity_score * 3
                        + signal.personalization_alignment_score * 3
                        + signal.memory_stability_score * 2
                        + signal.drift_risk_score * 3
                        + signal.governance_weight,
                    ),
                ),
            )
            self.assertIn("session_memory_evolution", signal.context_tags)
            self.assertIn("session_recording", signal.blocked_runtime_behaviors)
            self.assertIn(
                "session_memory_storage_write",
                signal.blocked_runtime_behaviors,
            )
            self.assertTrue(signal.explainability_notes)
            self.assertTrue(signal.advisory_actions)
            self.assertTrue(signal.evidence)
            self.assertTrue(signal.hitl_required_before_session_memory_update)
            self.assertTrue(signal.session_memory_evolution_implemented)
            self.assertTrue(signal.session_memory_evolution_metadata_implemented)
            self.assertTrue(signal.personalization_source_used)
            self.assertTrue(signal.creative_dna_source_used)
            self.assertTrue(signal.long_term_memory_source_used)
            self.assertTrue(signal.project_memory_source_used)
            self.assertFalse(signal.session_memory_storage_write_implemented)
            self.assertFalse(signal.session_memory_record_creation_implemented)
            self.assertFalse(signal.session_memory_record_update_implemented)
            self.assertFalse(signal.session_memory_record_deletion_implemented)
            self.assertFalse(signal.session_memory_evolution_application_implemented)
            self.assertFalse(signal.session_recording_implemented)
            self.assertFalse(signal.session_replay_execution_implemented)
            self.assertFalse(signal.memory_retrieval_execution_implemented)
            self.assertFalse(signal.memory_storage_write_implemented)
            self.assertFalse(signal.memory_consolidation_implemented)
            self.assertFalse(signal.personalization_application_implemented)
            self.assertFalse(signal.creative_dna_application_implemented)
            self.assertFalse(signal.preference_mutation_implemented)
            self.assertFalse(signal.provider_model_routing_implemented)
            self.assertFalse(signal.provider_execution_implemented)
            self.assertFalse(signal.agent_invocation_implemented)
            self.assertFalse(signal.workflow_control_implemented)
            self.assertFalse(signal.workflow_graph_mutation_implemented)
            self.assertFalse(signal.workflow_execution_implemented)
            self.assertFalse(signal.persistent_storage_write_implemented)
            self.assertFalse(signal.generated_output_mutation_implemented)
            self.assertFalse(signal.runtime_evolution_implemented)
            self.assertTrue(signal.advisory_only)

        intent = session_memory_evolution_signal_by_id(
            "session_memory_evolution::intent_evolution",
            plan,
        )
        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.status, "guarded")
        self.assertEqual(intent.confidence, "guarded")
        self.assertEqual(
            len(session_memory_evolution_signals_for_status("guarded", plan)),
            2,
        )
        self.assertEqual(
            len(session_memory_evolution_signals_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_session_evolution_metadata(self) -> None:
        plan = build_session_memory_evolution()
        payload = plan.model_dump(mode="json")
        payload["signal_ids"] = ("missing",) + tuple(payload["signal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "signal_ids must match"):
            SessionMemoryEvolutionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_session_evolution_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_session_evolution_score must match",
        ):
            SessionMemoryEvolutionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["evolved_session_memory_ids"] = (plan.signal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "evolved_session_memory_ids must remain empty",
        ):
            SessionMemoryEvolutionPlan(**payload)

    def test_session_evolution_does_not_change_routing_or_provider_factory(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Review session memory evolution for a creative coding project.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_session_memory_evolution(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_session_evolution_does_not_declare_runtime_update_terms(self) -> None:
        plan = build_session_memory_evolution(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for signal in plan.signals
                    for field in (
                        signal.evolution_id,
                        signal.evolution_kind,
                        signal.status,
                        signal.confidence,
                        signal.evolution_scope,
                        signal.source_personalization_id,
                        signal.source_creative_dna_id,
                        signal.source_long_term_memory_record_id,
                        signal.source_project_memory_signal_id,
                        signal.evolution_summary,
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
            "write_session_memory(",
            "create_session_memory(",
            "update_session_memory(",
            "delete_session_memory(",
            "apply_session_memory_evolution(",
            "record_session(",
            "execute_session_replay(",
            "retrieve_memory(",
            "write_memory(",
            "consolidate_memory(",
            "apply_personalization(",
            "apply_creative_dna(",
            "mutate_preference(",
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
