import unittest
from datetime import UTC, datetime

from creative_coding_assistant.contracts import (
    AssistantMode,
    CreativeCodingDomain,
)
from creative_coding_assistant.memory.schemas import (
    ConversationRole,
    ProjectMemoryKind,
)
from creative_coding_assistant.orchestration import (
    AssembledContextRequest,
    AssembledContextResponse,
    AssembledContextSummary,
    ContextBudgetPlan,
    ConversationSummaryContext,
    MemoryContextRequest,
    MemoryContextResponse,
    ProjectMemoryContext,
    RecentConversationTurn,
    RetrievalContextFilter,
    RetrievalContextRequest,
    RetrievalContextResponse,
    RetrievedKnowledgeChunk,
    RouteName,
    analyze_creative_complexity,
    analyze_workflow_cost,
    context_budget_allocation_by_id,
    context_budget_allocations_for_kind,
    plan_context_budget,
)
from creative_coding_assistant.rag.sources import OfficialSourceType

REQUIRED_ALLOCATION_FIELDS = {
    "allocation_id",
    "source_kind",
    "source_id",
    "priority",
    "requested_tokens",
    "allocated_tokens",
    "max_tokens",
    "pressure",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "context_trimming_implemented",
    "prompt_compression_implemented",
    "retrieval_compression_implemented",
    "memory_summarization_implemented",
    "context_routing_implemented",
    "provider_model_routing_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "planning_only",
}


class ContextBudgetPlannerTests(unittest.TestCase):
    def test_default_plan_reserves_context_and_response_budget(self) -> None:
        plan = plan_context_budget()

        self.assertEqual(plan.role, "context_budget_planner")
        self.assertEqual(plan.serialization_version, "context_budget_plan.v1")
        self.assertEqual(plan.total_budget_tokens, 16_000)
        self.assertEqual(plan.response_reserve_tokens, 3_000)
        self.assertEqual(plan.available_context_tokens, 13_000)
        self.assertGreater(plan.requested_context_tokens, 0)
        self.assertLessEqual(
            plan.allocated_context_tokens, plan.available_context_tokens
        )
        self.assertEqual(plan.over_budget_tokens, 0)
        self.assertEqual(plan.budget_pressure, "low")
        self.assertIn("does not trim context", plan.authority_boundary)
        self.assertTrue(plan.context_budget_planning_implemented)
        self.assertFalse(plan.context_trimming_implemented)
        self.assertFalse(plan.prompt_compression_implemented)
        self.assertFalse(plan.retrieval_compression_implemented)
        self.assertFalse(plan.memory_summarization_implemented)
        self.assertFalse(plan.context_routing_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.planning_only)

    def test_allocations_cover_context_sources_and_boundary_flags(self) -> None:
        plan = plan_context_budget(
            assembled_context=_assembled_context(),
            creative_complexity=analyze_creative_complexity(),
            workflow_cost=analyze_workflow_cost(),
        )

        self.assertEqual(plan.memory_recent_turn_count, 2)
        self.assertEqual(plan.project_memory_count, 1)
        self.assertEqual(plan.retrieval_chunk_count, 2)
        self.assertEqual(plan.workflow_cost_pressure, "medium")
        self.assertEqual(
            plan.allocation_ids,
            (
                "context::user_request",
                "context::memory_recent_turns",
                "context::memory_summary",
                "context::project_memory",
                "context::retrieval_chunks",
                "context::creative_metadata",
                "context::workflow_overhead",
                "context::response_reserve",
            ),
        )

        for allocation in plan.allocations:
            self.assertEqual(
                set(allocation.model_dump(mode="json")),
                REQUIRED_ALLOCATION_FIELDS,
            )
            self.assertEqual(
                allocation.serialization_version,
                "context_budget_allocation.v1",
            )
            self.assertLessEqual(
                allocation.allocated_tokens, allocation.requested_tokens
            )
            self.assertLessEqual(allocation.allocated_tokens, allocation.max_tokens)
            self.assertIn("context_trimming", allocation.blocked_runtime_behaviors)
            self.assertFalse(allocation.context_trimming_implemented)
            self.assertFalse(allocation.prompt_compression_implemented)
            self.assertFalse(allocation.retrieval_compression_implemented)
            self.assertFalse(allocation.memory_summarization_implemented)
            self.assertFalse(allocation.context_routing_implemented)
            self.assertFalse(allocation.provider_model_routing_implemented)
            self.assertFalse(allocation.prompt_mutation_implemented)
            self.assertFalse(allocation.generated_output_mutation_implemented)
            self.assertTrue(allocation.planning_only)

        retrieval = context_budget_allocation_by_id(
            "context::retrieval_chunks",
            plan,
        )
        response = context_budget_allocation_by_id(
            "context::response_reserve",
            plan,
        )
        self.assertIsNotNone(retrieval)
        self.assertIsNotNone(response)
        assert retrieval is not None
        assert response is not None
        self.assertEqual(retrieval.priority, "high")
        self.assertGreater(retrieval.requested_tokens, 0)
        self.assertEqual(response.source_kind, "response_reserve")
        self.assertEqual(response.allocated_tokens, plan.response_reserve_tokens)

    def test_small_budget_flags_overflow_without_mutating_context(self) -> None:
        plan = plan_context_budget(
            assembled_context=_large_assembled_context(),
            total_budget_tokens=1_200,
            response_reserve_tokens=400,
        )

        self.assertEqual(plan.available_context_tokens, 800)
        self.assertGreater(plan.over_budget_tokens, 0)
        self.assertEqual(plan.budget_pressure, "high")
        self.assertLessEqual(
            plan.allocated_context_tokens, plan.available_context_tokens
        )
        self.assertTrue(
            any(allocation.pressure == "high" for allocation in plan.allocations)
        )
        self.assertIn(
            "Flag overflow for later compression or routing tasks.",
            plan.advisory_actions,
        )

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = plan_context_budget(assembled_context=_assembled_context())
        recent = context_budget_allocations_for_kind("memory_recent_turns", plan)
        missing = context_budget_allocation_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0].allocation_id, "context::memory_recent_turns")
        self.assertIs(
            recent[0],
            context_budget_allocation_by_id("context::memory_recent_turns", plan),
        )

    def test_plan_rejects_mismatched_allocations_or_totals(self) -> None:
        plan = plan_context_budget(assembled_context=_assembled_context())
        payload = plan.model_dump(mode="json")
        payload["allocation_ids"] = ("missing",) + tuple(payload["allocation_ids"][1:])

        with self.assertRaisesRegex(ValueError, "allocation_ids must match"):
            ContextBudgetPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["allocated_context_tokens"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "allocated_context_tokens must match",
        ):
            ContextBudgetPlan(**payload)

    def test_plan_does_not_declare_context_mutation_terms(self) -> None:
        plan = plan_context_budget(assembled_context=_assembled_context())
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for allocation in plan.allocations
                    for field in (
                        allocation.allocation_id,
                        allocation.source_id,
                        *allocation.evidence,
                        *allocation.advisory_actions,
                        *allocation.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "trim_context(",
            "compress_prompt(",
            "compress_retrieval(",
            "summarize_memory(",
            "route_context(",
            "mutate_prompt(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


def _assembled_context() -> AssembledContextResponse:
    memory = MemoryContextResponse(
        request=MemoryContextRequest(
            route=RouteName.GENERATE,
            conversation_id="conversation-1",
            project_id="project-1",
        ),
        recent_turns=(
            RecentConversationTurn(
                turn_index=1,
                role=ConversationRole.USER,
                content="Make the visual language feel like luminous geometry.",
                created_at=_now(),
                mode=AssistantMode.GENERATE,
            ),
            RecentConversationTurn(
                turn_index=2,
                role=ConversationRole.ASSISTANT,
                content="Use radial symmetry, slow rotation, and layered color fields.",
                created_at=_now(),
                mode=AssistantMode.GENERATE,
            ),
        ),
        running_summary=ConversationSummaryContext(
            content="The project favors sacred geometry and audiovisual motion.",
            created_at=_now(),
            covered_turn_count=4,
        ),
        project_memories=(
            ProjectMemoryContext(
                content="Prefer p5.js sketches with exportable, readable structure.",
                created_at=_now(),
                memory_kind=ProjectMemoryKind.PREFERENCE,
                source="manual",
            ),
        ),
    )
    retrieval = RetrievalContextResponse(
        request=RetrievalContextRequest(
            query="p5.js audio reactive geometry",
            route=RouteName.GENERATE,
            filters=RetrievalContextFilter(domain=CreativeCodingDomain.P5_JS),
        ),
        chunks=(
            _chunk(1, "Use createCanvas and draw loops for animated sketches."),
            _chunk(2, "Amplitude and FFT analysis can drive visual parameters."),
        ),
    )
    return AssembledContextResponse(
        request=AssembledContextRequest(
            route=RouteName.GENERATE,
            memory_context=memory,
            retrieval_context=retrieval,
        ),
        summary=AssembledContextSummary(
            recent_turn_count=2,
            has_running_summary=True,
            project_memory_count=1,
            retrieval_chunk_count=2,
        ),
        memory_context=memory,
        retrieval_context=retrieval,
    )


def _large_assembled_context() -> AssembledContextResponse:
    base = _assembled_context()
    repeated = " ".join(["context detail"] * 800)
    memory = base.memory_context
    retrieval = base.retrieval_context
    assert memory is not None
    assert retrieval is not None
    memory = memory.model_copy(
        update={
            "recent_turns": tuple(
                turn.model_copy(update={"content": repeated})
                for turn in memory.recent_turns
            ),
            "project_memories": tuple(
                item.model_copy(update={"content": repeated})
                for item in memory.project_memories
            ),
        }
    )
    retrieval = retrieval.model_copy(
        update={
            "chunks": tuple(
                chunk.model_copy(update={"excerpt": repeated})
                for chunk in retrieval.chunks
            )
        }
    )
    return base.model_copy(
        update={
            "memory_context": memory,
            "retrieval_context": retrieval,
        }
    )


def _chunk(rank: int, excerpt: str) -> RetrievedKnowledgeChunk:
    return RetrievedKnowledgeChunk(
        source_id="p5-reference",
        domain=CreativeCodingDomain.P5_JS,
        source_type=OfficialSourceType.GUIDE,
        publisher="p5.js",
        registry_title="p5.js Reference",
        document_title=f"Guide {rank}",
        source_url="https://p5js.org/reference/",
        chunk_index=rank,
        excerpt=excerpt,
        score=0.8,
        rank=rank,
    )


def _now() -> datetime:
    return datetime(2026, 6, 28, 12, 0, tzinfo=UTC)


if __name__ == "__main__":
    unittest.main()
