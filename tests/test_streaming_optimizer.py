import unittest

from creative_coding_assistant.contracts import StreamEventType
from creative_coding_assistant.orchestration import (
    StreamingOptimizationPlan,
    optimize_streaming,
    plan_async_execution,
    streaming_optimization_candidate_by_id,
    streaming_optimization_candidates_for_status,
)

REQUIRED_STREAMING_CANDIDATE_FIELDS = {
    "candidate_id",
    "phase_id",
    "status",
    "optimization_focus",
    "stream_event_types",
    "source_async_candidate_ids",
    "event_order_required",
    "sequence_monotonic_required",
    "advisory_batch_window_ms",
    "advisory_stream_readiness_score",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "streaming_optimizer_implemented",
    "stream_event_emission_change_implemented",
    "stream_sequence_mutation_implemented",
    "token_delta_buffering_implemented",
    "chunk_batching_runtime_implemented",
    "stream_payload_mutation_implemented",
    "workflow_timing_change_implemented",
    "workflow_graph_mutation_implemented",
    "graph_compilation_implemented",
    "workflow_execution_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class StreamingOptimizerTests(unittest.TestCase):
    def test_default_plan_derives_advisory_streaming_candidates(self) -> None:
        async_plan = plan_async_execution()
        plan = optimize_streaming(async_execution=async_plan)

        self.assertEqual(plan.role, "streaming_optimizer")
        self.assertEqual(
            plan.serialization_version,
            "streaming_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_async_execution_serialization_version,
            async_plan.serialization_version,
        )
        self.assertEqual(plan.source_stream_event_contract, "StreamEventType")
        self.assertEqual(plan.candidate_count, 5)
        self.assertEqual(
            plan.candidate_ids,
            (
                "streaming_optimizer::workflow_lifecycle",
                "streaming_optimizer::generation_token_flow",
                "streaming_optimizer::review_retry_visibility",
                "streaming_optimizer::artifact_preview_visibility",
                "streaming_optimizer::terminal_integrity",
            ),
        )
        self.assertEqual(plan.optimization_candidate_count, 2)
        self.assertEqual(plan.contract_guardrail_count, 3)
        self.assertIn(StreamEventType.TOKEN_DELTA.value, plan.stream_event_types)
        self.assertIn(StreamEventType.FINAL.value, plan.stream_event_types)
        self.assertEqual(plan.stream_event_type_count, len(plan.stream_event_types))
        self.assertEqual(plan.highest_advisory_stream_readiness_score, 300)
        self.assertEqual(plan.total_advisory_stream_readiness_score, 500)
        self.assertEqual(plan.streaming_optimization_pressure, "high")
        self.assertTrue(plan.stream_contract_preserved)
        self.assertIn("does not emit events", plan.authority_boundary)
        self.assertTrue(plan.streaming_optimizer_implemented)
        self.assertFalse(plan.stream_event_emission_change_implemented)
        self.assertFalse(plan.stream_sequence_mutation_implemented)
        self.assertFalse(plan.token_delta_buffering_implemented)
        self.assertFalse(plan.chunk_batching_runtime_implemented)
        self.assertFalse(plan.stream_payload_mutation_implemented)
        self.assertFalse(plan.workflow_timing_change_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.graph_compilation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.node_handler_invocation_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_candidates_preserve_stream_event_contract_boundaries(self) -> None:
        plan = optimize_streaming()
        generation = streaming_optimization_candidate_by_id(
            "streaming_optimizer::generation_token_flow",
            plan,
        )
        terminal = streaming_optimization_candidate_by_id(
            "streaming_optimizer::terminal_integrity",
            plan,
        )

        self.assertIsNotNone(generation)
        self.assertIsNotNone(terminal)
        assert generation is not None
        assert terminal is not None
        self.assertEqual(generation.status, "optimization_candidate")
        self.assertEqual(generation.optimization_focus, "generation_token_flow")
        self.assertEqual(generation.advisory_batch_window_ms, 50)
        self.assertIn(StreamEventType.TOKEN_DELTA.value, generation.stream_event_types)
        self.assertTrue(generation.source_async_candidate_ids)
        self.assertEqual(terminal.status, "contract_guardrail")
        self.assertEqual(terminal.advisory_batch_window_ms, 0)
        self.assertIn(StreamEventType.FINAL.value, terminal.stream_event_types)

        for candidate in plan.candidates:
            self.assertEqual(
                set(candidate.model_dump(mode="json")),
                REQUIRED_STREAMING_CANDIDATE_FIELDS,
            )
            self.assertEqual(
                candidate.serialization_version,
                "streaming_optimization_candidate.v1",
            )
            self.assertTrue(candidate.event_order_required)
            self.assertTrue(candidate.sequence_monotonic_required)
            self.assertIn(
                "stream_event_emission_change",
                candidate.blocked_runtime_behaviors,
            )
            self.assertTrue(candidate.streaming_optimizer_implemented)
            self.assertFalse(candidate.stream_event_emission_change_implemented)
            self.assertFalse(candidate.stream_sequence_mutation_implemented)
            self.assertFalse(candidate.token_delta_buffering_implemented)
            self.assertFalse(candidate.chunk_batching_runtime_implemented)
            self.assertFalse(candidate.stream_payload_mutation_implemented)
            self.assertFalse(candidate.workflow_timing_change_implemented)
            self.assertFalse(candidate.workflow_graph_mutation_implemented)
            self.assertFalse(candidate.graph_compilation_implemented)
            self.assertFalse(candidate.workflow_execution_implemented)
            self.assertFalse(candidate.agent_invocation_implemented)
            self.assertFalse(candidate.node_handler_invocation_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.prompt_mutation_implemented)
            self.assertFalse(candidate.persistent_storage_write_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.advisory_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = optimize_streaming()
        optimization_candidates = streaming_optimization_candidates_for_status(
            "optimization_candidate",
            plan,
        )
        guardrails = streaming_optimization_candidates_for_status(
            "contract_guardrail",
            plan,
        )
        missing = streaming_optimization_candidate_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in optimization_candidates),
            plan.optimization_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in guardrails),
            plan.contract_guardrail_candidate_ids,
        )
        self.assertIs(
            optimization_candidates[0],
            streaming_optimization_candidate_by_id(
                optimization_candidates[0].candidate_id,
                plan,
            ),
        )

    def test_plan_rejects_mismatched_candidates_or_scores(self) -> None:
        plan = optimize_streaming()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            StreamingOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["total_advisory_stream_readiness_score"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "total_advisory_stream_readiness_score must match",
        ):
            StreamingOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["streaming_optimization_pressure"] = "low"

        with self.assertRaisesRegex(
            ValueError,
            "streaming_optimization_pressure must match",
        ):
            StreamingOptimizationPlan(**payload)

    def test_plan_does_not_declare_runtime_streaming_terms(self) -> None:
        plan = optimize_streaming()
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for candidate in plan.candidates
                    for field in (
                        candidate.candidate_id,
                        candidate.optimization_focus,
                        *candidate.stream_event_types,
                        *candidate.evidence,
                        *candidate.advisory_actions,
                        *candidate.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "emit_event(",
            "buffer_token_delta(",
            "batch_chunk(",
            "reorder_stream(",
            "mutate_payload(",
            "change_workflow_timing(",
            "compile_graph(",
            "execute_workflow(",
            "invoke_agent(",
            "invoke_node_handler(",
            "route_provider(",
            "trigger_retry(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
