import os
import unittest
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.core import (
    GenerationProviderName,
    Settings,
    load_settings,
)
from creative_coding_assistant.llm import OpenAIGenerationProvider
from creative_coding_assistant.memory import ConversationRole, ProjectMemoryKind
from creative_coding_assistant.orchestration import (
    AssembledContextResponse,
    AssembledContextSummary,
    ConversationSummaryContext,
    JinjaPromptRenderer,
    LlmGenerationAdapter,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryContextSource,
    ProjectMemoryContext,
    RecentConversationTurn,
    RetrievalContextRequest,
    RetrievalContextResponse,
    RetrievalContextSource,
    RetrievedKnowledgeChunk,
    RouteCapability,
    RouteDecision,
    RouteName,
    StructuredPromptInputBuilder,
    build_assembled_context_request,
    build_prompt_input_request,
    build_provider_generation_request,
    build_rendered_prompt_request,
)
from creative_coding_assistant.rag.sources import OfficialSourceType


class ProviderConfigSettingsTests(unittest.TestCase):
    def test_settings_defaults_include_provider_and_openai_model(self) -> None:
        settings = Settings(_env_file=None)

        self.assertEqual(
            settings.default_generation_provider,
            GenerationProviderName.OPENAI,
        )
        self.assertEqual(settings.openai_model, "gpt-5-mini")
        self.assertFalse(settings.has_openai_api_key)
        self.assertIsNone(settings.get_openai_api_key())

    def test_settings_accept_environment_overrides(self) -> None:
        with _temporary_env(
            CCA_DEFAULT_GENERATION_PROVIDER="openai",
            CCA_OPENAI_MODEL="gpt-5",
            OPENAI_API_KEY="sk-test-secret",
        ):
            settings = load_settings()

        self.assertEqual(
            settings.default_generation_provider,
            GenerationProviderName.OPENAI,
        )
        self.assertEqual(settings.openai_model, "gpt-5")
        self.assertTrue(settings.has_openai_api_key)
        self.assertEqual(settings.get_openai_api_key(), "sk-test-secret")
        self.assertNotIn("sk-test-secret", repr(settings.openai_api_key))

    def test_settings_trim_blank_openai_api_key_to_none(self) -> None:
        with _temporary_env(OPENAI_API_KEY="   "):
            settings = load_settings()

        self.assertFalse(settings.has_openai_api_key)
        self.assertIsNone(settings.get_openai_api_key())

    def test_openai_provider_uses_settings_backed_model_default(self) -> None:
        response = SimpleNamespace(
            output_text="Use a calm camera drift.",
            status="completed",
        )
        client = _FakeOpenAIClient(response=response)
        settings = Settings(openai_model="gpt-5")
        provider = OpenAIGenerationProvider(settings=settings, client=client)

        events = tuple(provider.stream(_generation_input(stream=False)))

        self.assertEqual(len(events), 1)
        self.assertEqual(client.last_kwargs["model"], "gpt-5")
        self.assertFalse(client.last_kwargs["stream"])


def _generation_input(*, stream: bool):
    prompt_input_request = build_prompt_input_request(
        assistant_request=AssistantRequest(
            query="Explain the scene setup.",
            conversation_id="conversation-1",
            project_id="project-1",
            domain=CreativeCodingDomain.THREE_JS,
            mode=AssistantMode.EXPLAIN,
        ),
        route_decision=_route_decision(),
        assembled_context=_assembled_context(),
    )
    prompt_input = StructuredPromptInputBuilder().build(prompt_input_request)
    rendered_prompt = JinjaPromptRenderer().render(
        build_rendered_prompt_request(
            route_decision=_route_decision(),
            prompt_input=prompt_input,
        )
    )
    provider_request = build_provider_generation_request(
        route_decision=_route_decision(),
        rendered_prompt=rendered_prompt,
        stream=stream,
    )
    return LlmGenerationAdapter().prepare_generation(provider_request)


def _route_decision() -> RouteDecision:
    return RouteDecision(
        route=RouteName.EXPLAIN,
        mode=AssistantMode.EXPLAIN,
        capabilities=(
            RouteCapability.MEMORY_CONTEXT,
            RouteCapability.OFFICIAL_DOCS,
        ),
    )


def _assembled_context() -> AssembledContextResponse:
    request = build_assembled_context_request(
        route_decision=_route_decision(),
        memory_context=_memory_context(),
        retrieval_context=_retrieval_context(),
    )
    assert request is not None
    return AssembledContextResponse(
        request=request,
        summary=AssembledContextSummary(
            recent_turn_count=2,
            has_running_summary=True,
            project_memory_count=2,
            retrieval_chunk_count=1,
        ),
        memory_context=_memory_context(),
        retrieval_context=_retrieval_context(),
    )


def _memory_context() -> MemoryContextResponse:
    return MemoryContextResponse(
        request=MemoryContextRequest(
            route=RouteName.EXPLAIN,
            conversation_id="conversation-1",
            project_id="project-1",
        ),
        source=MemoryContextSource.CHROMA_MEMORY,
        recent_turns=(
            RecentConversationTurn(
                turn_index=0,
                role=ConversationRole.USER,
                content="Keep the motion restrained.",
                created_at=_time(),
                mode=AssistantMode.EXPLAIN,
            ),
            RecentConversationTurn(
                turn_index=1,
                role=ConversationRole.ASSISTANT,
                content="We can keep the camera drift subtle.",
                created_at=_time(),
                mode=AssistantMode.EXPLAIN,
            ),
        ),
        running_summary=ConversationSummaryContext(
            content="The user prefers restrained motion and calm palettes.",
            created_at=_time(),
            covered_turn_count=2,
        ),
        project_memories=(
            ProjectMemoryContext(
                content="Prefer restrained palettes.",
                created_at=_time(),
                memory_kind=ProjectMemoryKind.STYLE,
                source="user",
            ),
            ProjectMemoryContext(
                content="Build atmospheric shader studies.",
                created_at=_time(),
                memory_kind=ProjectMemoryKind.GOAL,
                source="user",
            ),
        ),
    )


def _retrieval_context() -> RetrievalContextResponse:
    return RetrievalContextResponse(
        request=RetrievalContextRequest(
            query="Explain the scene setup.",
            route=RouteName.EXPLAIN,
        ),
        source=RetrievalContextSource.OFFICIAL_KB,
        chunks=(
            RetrievedKnowledgeChunk(
                source_id="three_docs",
                domain=CreativeCodingDomain.THREE_JS,
                source_type=OfficialSourceType.API_REFERENCE,
                publisher="three.js",
                registry_title="three.js Documentation",
                document_title="PerspectiveCamera",
                source_url="https://threejs.org/docs/",
                resolved_url="https://threejs.org/docs/",
                chunk_index=0,
                excerpt="PerspectiveCamera controls field of view and aspect ratio.",
                score=0.83,
            ),
        ),
    )


def _time() -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


@contextmanager
def _temporary_env(**updates: str) -> object:
    original_values = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, original in original_values.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


class _FakeResponsesApi:
    def __init__(self, *, response: object) -> None:
        self.response = response
        self.last_kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> object:
        self.last_kwargs = dict(kwargs)
        return self.response


class _FakeOpenAIClient:
    def __init__(self, *, response: object) -> None:
        self.responses = _FakeResponsesApi(response=response)

    @property
    def last_kwargs(self) -> dict[str, object]:
        assert self.responses.last_kwargs is not None
        return self.responses.last_kwargs


if __name__ == "__main__":
    unittest.main()
