import unittest
from datetime import UTC, datetime

from creative_coding_assistant.contracts import AssistantMode
from creative_coding_assistant.memory import ConversationRole, ProjectMemoryKind
from creative_coding_assistant.orchestration import (
    ConversationSummaryContext,
    MemoryContextRequest,
    MemoryContextResponse,
    MemorySummarizationResult,
    MemorySummarySegment,
    ProjectMemoryContext,
    RecentConversationTurn,
    RouteName,
    memory_summary_segment_by_id,
    memory_summary_segments_for_kind,
    summarize_memory_context,
)

REQUIRED_MEMORY_SUMMARY_SEGMENT_FIELDS = {
    "segment_id",
    "source_kind",
    "source_id",
    "turn_indices",
    "project_memory_kind",
    "original_text",
    "summary_text",
    "original_token_estimate",
    "summary_token_estimate",
    "saved_tokens",
    "summarization_status",
    "summarization_pressure",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "memory_summarization_implemented",
    "memory_storage_write_implemented",
    "running_summary_replacement_implemented",
    "conversation_turn_mutation_implemented",
    "project_memory_mutation_implemented",
    "memory_query_execution_implemented",
    "context_routing_implemented",
    "provider_model_routing_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "summarization_only",
}


class MemorySummarizationTests(unittest.TestCase):
    def test_short_memory_context_remains_unchanged_with_boundary_flags(self) -> None:
        result = summarize_memory_context(_memory_context(short=True))
        segment = result.segments[0]

        self.assertEqual(result.role, "memory_summarizer")
        self.assertEqual(
            result.serialization_version,
            "memory_summarization_result.v1",
        )
        self.assertEqual(result.segment_ids, ("memory_summary::recent_turns",))
        self.assertEqual(result.source_route, "explain")
        self.assertEqual(result.source_recent_turn_count, 1)
        self.assertEqual(result.source_project_memory_count, 0)
        self.assertFalse(result.source_has_running_summary)
        self.assertEqual(result.saved_total_tokens, 0)
        self.assertTrue(result.within_budget)
        self.assertEqual(result.summarization_pressure, "low")
        self.assertIn("does not write memory", result.authority_boundary)
        self.assertTrue(result.memory_summarization_implemented)
        self.assertFalse(result.memory_storage_write_implemented)
        self.assertFalse(result.running_summary_replacement_implemented)
        self.assertFalse(result.conversation_turn_mutation_implemented)
        self.assertFalse(result.project_memory_mutation_implemented)
        self.assertFalse(result.memory_query_execution_implemented)
        self.assertFalse(result.context_routing_implemented)
        self.assertFalse(result.provider_model_routing_implemented)
        self.assertFalse(result.persistent_storage_write_implemented)
        self.assertFalse(result.generated_output_mutation_implemented)
        self.assertTrue(result.summarization_only)
        self.assertEqual(segment.summarization_status, "unchanged")
        self.assertEqual(segment.original_text, segment.summary_text)

    def test_long_memory_context_is_summarized_without_storage_writes(self) -> None:
        result = summarize_memory_context(
            _memory_context(short=False),
            target_token_budget=120,
        )

        self.assertEqual(result.source_recent_turn_count, 2)
        self.assertEqual(result.source_project_memory_count, 1)
        self.assertTrue(result.source_has_running_summary)
        self.assertGreater(result.original_total_tokens, result.summary_total_tokens)
        self.assertGreater(result.saved_total_tokens, 0)
        self.assertLessEqual(result.summary_total_tokens, result.target_token_budget)
        self.assertTrue(result.within_budget)
        self.assertEqual(result.summarization_pressure, "medium")
        self.assertIn("[memory:recent_turns:memory_context.recent_turns]", result.summary_text)
        self.assertIn(
            "Use summary artifact only when explicitly selected.",
            result.advisory_actions,
        )

        for segment in result.segments:
            self.assertEqual(
                set(segment.model_dump(mode="json")),
                REQUIRED_MEMORY_SUMMARY_SEGMENT_FIELDS,
            )
            self.assertEqual(
                segment.serialization_version,
                "memory_summary_segment.v1",
            )
            self.assertLessEqual(
                segment.summary_token_estimate,
                segment.original_token_estimate,
            )
            self.assertEqual(
                segment.saved_tokens,
                segment.original_token_estimate - segment.summary_token_estimate,
            )
            self.assertIn("memory_storage_write", segment.blocked_runtime_behaviors)
            self.assertTrue(segment.memory_summarization_implemented)
            self.assertFalse(segment.memory_storage_write_implemented)
            self.assertFalse(segment.running_summary_replacement_implemented)
            self.assertFalse(segment.conversation_turn_mutation_implemented)
            self.assertFalse(segment.project_memory_mutation_implemented)
            self.assertFalse(segment.memory_query_execution_implemented)
            self.assertFalse(segment.context_routing_implemented)
            self.assertFalse(segment.provider_model_routing_implemented)
            self.assertFalse(segment.persistent_storage_write_implemented)
            self.assertFalse(segment.generated_output_mutation_implemented)
            self.assertTrue(segment.summarization_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        result = summarize_memory_context(_memory_context(short=False), target_token_budget=120)
        recent = memory_summary_segment_by_id("memory_summary::recent_turns", result)
        project = memory_summary_segments_for_kind("project_memory", result)
        missing = memory_summary_segment_by_id("missing", result)

        self.assertIsNone(missing)
        self.assertIsNotNone(recent)
        assert recent is not None
        self.assertEqual(recent.source_kind, "recent_turns")
        self.assertEqual(len(project), 1)
        self.assertEqual(project[0].project_memory_kind, ProjectMemoryKind.PREFERENCE.value)

    def test_result_rejects_mismatched_segments_or_totals(self) -> None:
        result = summarize_memory_context(_memory_context(short=False), target_token_budget=120)
        payload = result.model_dump(mode="json")
        payload["segment_ids"] = ("missing",) + tuple(payload["segment_ids"][1:])

        with self.assertRaisesRegex(ValueError, "segment_ids must match"):
            MemorySummarizationResult(**payload)

        payload = result.model_dump(mode="json")
        payload["summary_total_tokens"] += 1

        with self.assertRaisesRegex(ValueError, "summary_total_tokens must match"):
            MemorySummarizationResult(**payload)

        segment_payload = result.segments[0].model_dump(mode="json")
        segment_payload["saved_tokens"] += 1

        with self.assertRaisesRegex(ValueError, "saved_tokens must match"):
            MemorySummarySegment(**segment_payload)

    def test_result_does_not_declare_storage_or_provider_mutation_terms(self) -> None:
        result = summarize_memory_context(_memory_context(short=False), target_token_budget=120)
        combined_text = " ".join(
            (
                result.authority_boundary,
                *result.blocked_runtime_behaviors,
                *result.advisory_actions,
                *(
                    field
                    for segment in result.segments
                    for field in (
                        segment.segment_id,
                        segment.source_kind,
                        segment.source_id,
                        segment.project_memory_kind or "",
                        *segment.evidence,
                        *segment.advisory_actions,
                        *segment.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "write_memory(",
            "replace_running_summary(",
            "mutate_conversation_turn(",
            "mutate_project_memory(",
            "query_memory_storage(",
            "route_context(",
            "select_provider(",
            "route_provider(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


def _memory_context(*, short: bool) -> MemoryContextResponse:
    if short:
        turns = (
            RecentConversationTurn(
                turn_index=1,
                role=ConversationRole.USER,
                content="Keep the palette calm.",
                created_at=_now(),
                mode=AssistantMode.EXPLAIN,
            ),
        )
        return MemoryContextResponse(
            request=MemoryContextRequest(route=RouteName.EXPLAIN),
            recent_turns=turns,
        )

    turns = (
        RecentConversationTurn(
            turn_index=1,
            role=ConversationRole.USER,
            content=_repeated("The sketch should use gentle motion and blue light.", 80),
            created_at=_now(),
            mode=AssistantMode.GENERATE,
        ),
        RecentConversationTurn(
            turn_index=2,
            role=ConversationRole.ASSISTANT,
            content=_repeated("Use slow particles, layered gradients, and p5.js.", 90),
            created_at=_now(),
            mode=AssistantMode.GENERATE,
        ),
    )
    return MemoryContextResponse(
        request=MemoryContextRequest(
            route=RouteName.GENERATE,
            conversation_id="conversation-1",
            project_id="project-1",
        ),
        recent_turns=turns,
        running_summary=ConversationSummaryContext(
            content=_repeated("The project favors calm luminous geometry.", 80),
            created_at=_now(),
            covered_turn_count=4,
        ),
        project_memories=(
            ProjectMemoryContext(
                content=_repeated("Prefer p5.js and restrained color palettes.", 80),
                created_at=_now(),
                memory_kind=ProjectMemoryKind.PREFERENCE,
                source="manual",
            ),
        ),
    )


def _repeated(sentence: str, count: int) -> str:
    return " ".join([sentence] * count)


def _now() -> datetime:
    return datetime(2026, 6, 28, 12, 0, tzinfo=UTC)


if __name__ == "__main__":
    unittest.main()
