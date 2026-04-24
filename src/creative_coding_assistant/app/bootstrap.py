"""Top-level service composition helpers."""

from __future__ import annotations

from typing import Any

from loguru import logger

from creative_coding_assistant.core import Settings, load_settings
from creative_coding_assistant.llm import GenerationProvider
from creative_coding_assistant.memory import (
    ConversationSummaryRepository,
    ConversationTurnRepository,
    ProjectMemoryRepository,
)
from creative_coding_assistant.orchestration import (
    AssistantService,
    ChromaMemoryAdapter,
    JinjaPromptRenderer,
    KnowledgeBaseRetrievalAdapter,
    LlmGenerationAdapter,
    OrchestrationContextAssembler,
    StructuredPromptInputBuilder,
)
from creative_coding_assistant.rag.retrieval import (
    KnowledgeBaseRetriever,
    QueryEmbedder,
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
) -> AssistantService:
    """Compose the current assistant service stack from settings and local wiring."""

    resolved_settings = settings or load_settings()
    client = create_chroma_client(settings=resolved_settings)
    ensure_project_collections(client)

    memory_gateway = _build_memory_gateway(client=client)
    retrieval_gateway = _build_retrieval_gateway(client, query_embedder)
    service = AssistantService(
        settings=resolved_settings,
        memory_gateway=memory_gateway,
        retrieval_gateway=retrieval_gateway,
        context_assembler=OrchestrationContextAssembler(),
        prompt_input_builder=StructuredPromptInputBuilder(),
        prompt_renderer=JinjaPromptRenderer(),
        generation_gateway=LlmGenerationAdapter(),
        generation_provider=generation_provider,
    )
    logger.info(
        "Composed assistant service with retrieval={} and explicit_provider={}",
        retrieval_gateway is not None,
        generation_provider is not None,
    )
    return service


def _build_memory_gateway(*, client: Any) -> ChromaMemoryAdapter:
    return ChromaMemoryAdapter(
        turn_repository=ConversationTurnRepository(client=client),
        summary_repository=ConversationSummaryRepository(client=client),
        project_memory_repository=ProjectMemoryRepository(client=client),
    )


def _build_retrieval_gateway(
    client: Any,
    query_embedder: QueryEmbedder | None,
) -> KnowledgeBaseRetrievalAdapter | None:
    if query_embedder is None:
        return None
    retriever = KnowledgeBaseRetriever(client=client, embedder=query_embedder)
    return KnowledgeBaseRetrievalAdapter(retriever=retriever)
