"""Top-level service composition helpers."""

from __future__ import annotations

from typing import Any

from loguru import logger

from creative_coding_assistant.core import Settings, load_settings
from creative_coding_assistant.eval import (
    LiveSessionRecorder,
    build_live_session_eval_recorder,
)
from creative_coding_assistant.llm.generation import GenerationProvider
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
    ConversationMemoryRecorder,
)
from creative_coding_assistant.orchestration.prompt_inputs import (
    StructuredPromptInputBuilder,
)
from creative_coding_assistant.orchestration.prompt_templates import (
    JinjaPromptRenderer,
)
from creative_coding_assistant.orchestration.retrieval import (
    KnowledgeBaseRetrievalAdapter,
)
from creative_coding_assistant.orchestration.service import AssistantService
from creative_coding_assistant.rag.embeddings import OpenAIEmbeddingClient
from creative_coding_assistant.rag.retrieval import (
    KnowledgeBaseRetriever,
    QueryEmbedder,
    build_query_embedder,
)
from creative_coding_assistant.vectorstore import (
    create_chroma_client,
    ensure_project_collections,
)


def build_assistant_service(
    *,
    settings: Settings | None = None,
    query_embedder: QueryEmbedder | None = None,
    generation_provider: GenerationProvider | None = None,
    eval_recorder: LiveSessionRecorder | None = None,
    memory_recorder: ConversationMemoryRecorder | None = None,
) -> AssistantService:
    """Compose the current assistant service stack from settings and local wiring."""

    resolved_settings = settings or load_settings()
    client = create_chroma_client(settings=resolved_settings)
    ensure_project_collections(client)

    resolved_query_embedder = (
        query_embedder
        if query_embedder is not None
        else build_query_embedder(resolved_settings)
    )
    resolved_eval_recorder = (
        eval_recorder
        if eval_recorder is not None
        else build_live_session_eval_recorder(resolved_settings)
    )
    resolved_memory_recorder = (
        memory_recorder
        if memory_recorder is not None
        else _build_memory_recorder(client=client, settings=resolved_settings)
    )
    memory_gateway = _build_memory_gateway(client=client)
    retrieval_gateway = _build_retrieval_gateway(client, resolved_query_embedder)
    service = AssistantService(
        settings=resolved_settings,
        memory_gateway=memory_gateway,
        retrieval_gateway=retrieval_gateway,
        context_assembler=OrchestrationContextAssembler(),
        prompt_input_builder=StructuredPromptInputBuilder(),
        prompt_renderer=JinjaPromptRenderer(),
        generation_gateway=LlmGenerationAdapter(),
        generation_provider=generation_provider,
        eval_recorder=resolved_eval_recorder,
        memory_recorder=resolved_memory_recorder,
    )
    logger.info(
        "Composed assistant service with retrieval={}, explicit_provider={}, "
        "eval_recorder={}, memory_recorder={}",
        retrieval_gateway is not None,
        generation_provider is not None,
        resolved_eval_recorder is not None,
        resolved_memory_recorder is not None,
    )
    return service


def _build_memory_gateway(*, client: Any) -> ChromaMemoryAdapter:
    return ChromaMemoryAdapter(
        turn_repository=ConversationTurnRepository(client=client),
        summary_repository=ConversationSummaryRepository(client=client),
        project_memory_repository=ProjectMemoryRepository(client=client),
    )


def _build_memory_recorder(
    *,
    client: Any,
    settings: Settings,
) -> ChromaConversationMemoryRecorder | None:
    if not settings.has_openai_embedding_config:
        return None

    return ChromaConversationMemoryRecorder(
        turn_repository=ConversationTurnRepository(client=client),
        embedder=OpenAIEmbeddingClient(settings=settings),
    )


def _build_retrieval_gateway(
    client: Any,
    query_embedder: QueryEmbedder | None,
) -> KnowledgeBaseRetrievalAdapter | None:
    if query_embedder is None:
        return None
    retriever = KnowledgeBaseRetriever(client=client, embedder=query_embedder)
    return KnowledgeBaseRetrievalAdapter(retriever=retriever)
