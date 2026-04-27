"""Conversation-memory persistence boundaries for live assistant turns."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from loguru import logger

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.memory import (
    ConversationRole,
    ConversationTurnRepository,
    ConversationTurnWrite,
)


class TextEmbedder(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> tuple[list[float], ...]:
        """Return one embedding per input text."""


class ConversationMemoryRecorder(Protocol):
    def record_turns(
        self,
        *,
        request: AssistantRequest,
        answer: str,
        started_at: datetime,
        completed_at: datetime,
    ) -> None:
        """Persist user and assistant turns for future memory retrieval."""


class ChromaConversationMemoryRecorder:
    """Persist successful live turns into the conversation-turn store."""

    def __init__(
        self,
        *,
        turn_repository: ConversationTurnRepository,
        embedder: TextEmbedder,
    ) -> None:
        self._turn_repository = turn_repository
        self._embedder = embedder

    def record_turns(
        self,
        *,
        request: AssistantRequest,
        answer: str,
        started_at: datetime,
        completed_at: datetime,
    ) -> None:
        conversation_id = request.conversation_id
        if conversation_id is None:
            return

        user_text = request.query.strip()
        assistant_text = answer.strip()
        if not user_text or not assistant_text:
            return

        next_turn_index = self._next_turn_index(conversation_id)
        user_embedding, assistant_embedding = self._embedder.embed_texts(
            (user_text, assistant_text)
        )
        self._turn_repository.upsert(
            ConversationTurnWrite(
                conversation_id=conversation_id,
                turn_index=next_turn_index,
                role=ConversationRole.USER,
                content=user_text,
                created_at=started_at,
                project_id=request.project_id,
                domain=request.domain,
                mode=request.mode,
                embedding=user_embedding,
            )
        )
        self._turn_repository.upsert(
            ConversationTurnWrite(
                conversation_id=conversation_id,
                turn_index=next_turn_index + 1,
                role=ConversationRole.ASSISTANT,
                content=assistant_text,
                created_at=completed_at,
                project_id=request.project_id,
                domain=request.domain,
                mode=request.mode,
                embedding=assistant_embedding,
            )
        )
        logger.info(
            "Persisted conversation turns {} and {} for conversation '{}'",
            next_turn_index,
            next_turn_index + 1,
            conversation_id,
        )

    def _next_turn_index(self, conversation_id: str) -> int:
        turns = self._turn_repository.list_recent(
            conversation_id=conversation_id,
            limit=1,
        )
        if not turns:
            return 0
        return turns[-1].turn_index + 1
