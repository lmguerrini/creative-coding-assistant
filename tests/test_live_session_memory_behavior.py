import tempfile
import unittest
from pathlib import Path

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEventType,
)
from creative_coding_assistant.llm.generation import (
    GeneratedOutput,
    GenerationEventType,
    GenerationFinishReason,
    GenerationResponse,
    GenerationStreamEvent,
)
from creative_coding_assistant.memory import (
    ConversationSummaryRepository,
    ConversationTurnRepository,
    ProjectMemoryRepository,
)
from creative_coding_assistant.orchestration.context import (
    OrchestrationContextAssembler,
)
from creative_coding_assistant.orchestration.generation import LlmGenerationAdapter
from creative_coding_assistant.orchestration.memory import ChromaMemoryAdapter
from creative_coding_assistant.orchestration.memory_recording import (
    ChromaConversationMemoryRecorder,
)
from creative_coding_assistant.orchestration.prompt_inputs import (
    StructuredPromptInputBuilder,
)
from creative_coding_assistant.orchestration.prompt_templates import (
    JinjaPromptRenderer,
)
from creative_coding_assistant.orchestration.routing import (
    RouteCapability,
    RouteDecision,
    RouteName,
)
from creative_coding_assistant.orchestration.service import AssistantService
from creative_coding_assistant.vectorstore import create_chroma_client


class LiveSessionMemoryBehaviorTests(unittest.TestCase):
    def test_service_persists_turns_and_reuses_memory_on_follow_up(self) -> None:
        with _memory_client() as client:
            turn_repository = ConversationTurnRepository(client=client)
            service = AssistantService(
                route_fn=_memory_route,
                memory_gateway=ChromaMemoryAdapter(
                    turn_repository=turn_repository,
                    summary_repository=ConversationSummaryRepository(client=client),
                    project_memory_repository=ProjectMemoryRepository(client=client),
                ),
                context_assembler=OrchestrationContextAssembler(),
                prompt_input_builder=StructuredPromptInputBuilder(),
                prompt_renderer=JinjaPromptRenderer(),
                generation_gateway=LlmGenerationAdapter(),
                generation_provider=_CompletedGenerationProvider(
                    answers=(
                        "Use a basic cube scene with requestAnimationFrame rotation.",
                        "Increase the rotation speed and switch to a blue material.",
                    )
                ),
                memory_recorder=ChromaConversationMemoryRecorder(
                    turn_repository=turn_repository,
                    embedder=_FakeTextEmbedder(),
                ),
            )
            first_request = AssistantRequest(
                query="Create a simple rotating cube in three.js",
                conversation_id="conversation-1",
                domain=CreativeCodingDomain.THREE_JS,
                mode=AssistantMode.GENERATE,
            )

            first_events = tuple(service.stream(first_request))

            self.assertEqual(first_events[-1].event_type, StreamEventType.FINAL)
            stored_turns = turn_repository.list_recent(
                conversation_id="conversation-1",
                limit=10,
            )
            self.assertEqual([turn.turn_index for turn in stored_turns], [0, 1])
            self.assertEqual(
                [turn.content for turn in stored_turns],
                [
                    "Create a simple rotating cube in three.js",
                    "Use a basic cube scene with requestAnimationFrame rotation.",
                ],
            )

            second_request = AssistantRequest(
                query="Now make it rotate faster and add a blue material",
                conversation_id="conversation-1",
                domain=CreativeCodingDomain.THREE_JS,
                mode=AssistantMode.GENERATE,
            )

            second_events = tuple(service.stream(second_request))

            memory_event = next(
                event
                for event in second_events
                if event.event_type is StreamEventType.MEMORY
                and event.payload.get("code") == "memory_completed"
            )
            recent_turns = memory_event.payload["context"]["recent_turns"]
            self.assertEqual(len(recent_turns), 2)
            self.assertEqual(recent_turns[0]["content"], first_request.query)
            self.assertEqual(
                recent_turns[1]["content"],
                "Use a basic cube scene with requestAnimationFrame rotation.",
            )

            prompt_input_event = next(
                event
                for event in second_events
                if event.event_type is StreamEventType.PROMPT_INPUT
            )
            self.assertEqual(
                len(prompt_input_event.payload["prompt_input"]["memory_input"]["recent_turns"]),
                2,
            )

            captured_request = service._generation_provider.requests[1]
            memory_messages = [
                message
                for message in captured_request.messages
                if message.name.value == "memory"
            ]
            self.assertEqual(len(memory_messages), 1)
            self.assertIn(first_request.query, memory_messages[0].content)
            self.assertIn(
                "requestAnimationFrame rotation",
                memory_messages[0].content,
            )


def _memory_route(request: AssistantRequest) -> RouteDecision:
    return RouteDecision(
        route=RouteName.GENERATE,
        mode=request.mode,
        domain=request.domain,
        capabilities=(RouteCapability.MEMORY_CONTEXT,),
    )


class _CompletedGenerationProvider:
    def __init__(self, *, answers: tuple[str, ...]) -> None:
        self._answers = list(answers)
        self.requests = []

    def stream(self, request):
        self.requests.append(request)
        answer = self._answers.pop(0)
        yield GenerationStreamEvent(
            event_type=GenerationEventType.COMPLETED,
            response=GenerationResponse(
                request=request,
                output=GeneratedOutput(
                    content=answer,
                    finish_reason=GenerationFinishReason.STOP,
                ),
            ),
        )


class _FakeTextEmbedder:
    def embed_texts(self, texts):
        return tuple(
            [float(index + 1), 0.0, 0.0]
            for index, _ in enumerate(texts)
        )


class _memory_client:
    def __enter__(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        return create_chroma_client(path=Path(self._temp_dir.name) / "chroma")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
