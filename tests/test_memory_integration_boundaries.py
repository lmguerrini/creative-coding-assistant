import unittest
from datetime import UTC, datetime, timedelta

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    StreamEventType,
)
from creative_coding_assistant.memory import (
    ConversationRole,
    ConversationSummaryRecord,
    ConversationTurnRecord,
    ProjectMemoryKind,
    ProjectMemoryRecord,
)
from creative_coding_assistant.orchestration import (
    AssistantService,
    ChromaMemoryAdapter,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryContextSource,
    RecentConversationTurn,
    RouteCapability,
    RouteDecision,
    RouteName,
    build_memory_context_request,
)


class MemoryIntegrationBoundaryTests(unittest.TestCase):
    def test_build_memory_request_uses_conversation_and_project_ids(self) -> None:
        assistant_request = AssistantRequest(
            query="Continue the conversation.",
            conversation_id="conversation-1",
            project_id="project-1",
            mode=AssistantMode.EXPLAIN,
        )
        route_decision = RouteDecision(
            route=RouteName.EXPLAIN,
            mode=AssistantMode.EXPLAIN,
            capabilities=(RouteCapability.MEMORY_CONTEXT,),
        )

        memory_request = build_memory_context_request(
            assistant_request,
            route_decision,
        )

        self.assertIsNotNone(memory_request)
        assert memory_request is not None
        self.assertEqual(memory_request.route, RouteName.EXPLAIN)
        self.assertEqual(memory_request.conversation_id, "conversation-1")
        self.assertEqual(memory_request.project_id, "project-1")
        self.assertEqual(memory_request.recent_turn_limit, 6)
        self.assertEqual(memory_request.include_running_summary, True)
        self.assertEqual(memory_request.include_project_memory, True)

    def test_build_memory_request_skips_when_no_memory_identifiers(self) -> None:
        assistant_request = AssistantRequest(query="Continue the conversation.")
        route_decision = RouteDecision(
            route=RouteName.EXPLAIN,
            mode=AssistantMode.EXPLAIN,
            capabilities=(RouteCapability.MEMORY_CONTEXT,),
        )

        memory_request = build_memory_context_request(
            assistant_request,
            route_decision,
        )

        self.assertIsNone(memory_request)

    def test_memory_adapter_translates_repositories_into_context(self) -> None:
        turn_repository = _FakeTurnRepository(
            turns=(
                _turn(
                    turn_index=0,
                    role=ConversationRole.USER,
                    content="I want a calmer palette.",
                ),
                _turn(
                    turn_index=1,
                    role=ConversationRole.ASSISTANT,
                    content="We can keep the motion subtle and cool-toned.",
                ),
            )
        )
        summary_repository = _FakeSummaryRepository(
            summary=ConversationSummaryRecord(
                conversation_id="conversation-1",
                covered_turn_count=2,
                content="The user prefers restrained motion and palettes.",
                created_at=_time(2),
                project_id="project-1",
            )
        )
        project_memory_repository = _FakeProjectMemoryRepository(
            memories=(
                ProjectMemoryRecord(
                    project_id="project-1",
                    memory_kind=ProjectMemoryKind.STYLE,
                    content="Prefer restrained palettes.",
                    source="user",
                    created_at=_time(3),
                ),
                ProjectMemoryRecord(
                    project_id="project-1",
                    memory_kind=ProjectMemoryKind.GOAL,
                    content="Build atmospheric shader studies.",
                    source="user",
                    created_at=_time(4),
                ),
            )
        )
        adapter = ChromaMemoryAdapter(
            turn_repository=turn_repository,
            summary_repository=summary_repository,
            project_memory_repository=project_memory_repository,
        )
        request = MemoryContextRequest(
            route=RouteName.EXPLAIN,
            conversation_id="conversation-1",
            project_id="project-1",
        )

        context = adapter.retrieve_context(request)

        self.assertEqual(turn_repository.calls, [("conversation-1", 6)])
        self.assertEqual(summary_repository.calls, ["conversation-1"])
        self.assertEqual(project_memory_repository.calls, ["project-1"])
        self.assertEqual(context.source, MemoryContextSource.CHROMA_MEMORY)
        self.assertEqual(len(context.recent_turns), 2)
        self.assertIsNotNone(context.running_summary)
        self.assertEqual(len(context.project_memories), 2)
        self.assertEqual(
            context.project_memories[0].memory_kind,
            ProjectMemoryKind.STYLE,
        )

    def test_service_emits_memory_events_when_gateway_present(self) -> None:
        gateway = _FakeMemoryGateway(
            response=MemoryContextResponse(
                request=MemoryContextRequest(
                    route=RouteName.EXPLAIN,
                    conversation_id="conversation-1",
                    project_id="project-1",
                ),
                recent_turns=(
                    RecentConversationTurn(
                        turn_index=1,
                        role=ConversationRole.USER,
                        content="Keep it subtle.",
                        created_at=_time(1),
                    ),
                ),
            )
        )
        service = AssistantService(
            route_fn=_route_with_memory,
            memory_gateway=gateway,
        )
        request = AssistantRequest(
            query="Continue the sketch direction.",
            conversation_id="conversation-1",
            project_id="project-1",
            mode=AssistantMode.EXPLAIN,
        )

        events = tuple(service.stream(request))

        self.assertEqual(
            [event.event_type for event in events],
            [
                StreamEventType.STATUS,
                StreamEventType.STATUS,
                StreamEventType.MEMORY,
                StreamEventType.MEMORY,
                StreamEventType.FINAL,
            ],
        )
        self.assertEqual(events[2].payload["code"], "memory_requested")
        self.assertEqual(events[3].payload["code"], "memory_completed")
        self.assertEqual(len(gateway.requests), 1)
        self.assertEqual(gateway.requests[0].conversation_id, "conversation-1")

    def test_service_skips_memory_without_context_identifiers(self) -> None:
        gateway = _FakeMemoryGateway(
            response=MemoryContextResponse(
                request=MemoryContextRequest(route=RouteName.EXPLAIN),
            )
        )
        service = AssistantService(
            route_fn=_route_with_memory,
            memory_gateway=gateway,
        )
        request = AssistantRequest(query="Continue the sketch direction.")

        events = tuple(service.stream(request))

        self.assertEqual(
            [event.event_type for event in events],
            [
                StreamEventType.STATUS,
                StreamEventType.STATUS,
                StreamEventType.FINAL,
            ],
        )
        self.assertEqual(len(gateway.requests), 0)


def _route_with_memory(request: AssistantRequest) -> RouteDecision:
    return RouteDecision(
        route=RouteName.EXPLAIN,
        mode=request.mode,
        domain=request.domain,
        capabilities=(RouteCapability.MEMORY_CONTEXT,),
    )


def _turn(
    *,
    turn_index: int,
    role: ConversationRole,
    content: str,
) -> ConversationTurnRecord:
    return ConversationTurnRecord(
        conversation_id="conversation-1",
        turn_index=turn_index,
        role=role,
        content=content,
        created_at=_time(turn_index),
        project_id="project-1",
        mode=AssistantMode.EXPLAIN,
    )


def _time(offset_minutes: int) -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC) + timedelta(
        minutes=offset_minutes
    )


class _FakeTurnRepository:
    def __init__(self, *, turns: tuple[ConversationTurnRecord, ...]) -> None:
        self._turns = turns
        self.calls: list[tuple[str, int]] = []

    def list_recent(
        self,
        *,
        conversation_id: str,
        limit: int,
    ) -> tuple[ConversationTurnRecord, ...]:
        self.calls.append((conversation_id, limit))
        return self._turns


class _FakeSummaryRepository:
    def __init__(self, *, summary: ConversationSummaryRecord | None) -> None:
        self._summary = summary
        self.calls: list[str] = []

    def get_latest(
        self,
        *,
        conversation_id: str,
    ) -> ConversationSummaryRecord | None:
        self.calls.append(conversation_id)
        return self._summary


class _FakeProjectMemoryRepository:
    def __init__(self, *, memories: tuple[ProjectMemoryRecord, ...]) -> None:
        self._memories = memories
        self.calls: list[str] = []

    def list(
        self,
        *,
        project_id: str,
        memory_kind: ProjectMemoryKind | None = None,
    ) -> tuple[ProjectMemoryRecord, ...]:
        del memory_kind
        self.calls.append(project_id)
        return self._memories


class _FakeMemoryGateway:
    def __init__(self, *, response: MemoryContextResponse) -> None:
        self.response = response
        self.requests: list[MemoryContextRequest] = []

    def retrieve_context(
        self,
        request: MemoryContextRequest,
    ) -> MemoryContextResponse:
        self.requests.append(request)
        return self.response


if __name__ == "__main__":
    unittest.main()
