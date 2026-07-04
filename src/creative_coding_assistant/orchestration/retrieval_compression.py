"""V5.1 retrieval compression contracts for retrieved knowledge chunks."""

from __future__ import annotations

import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.orchestration.retrieval import (
    RetrievalContextResponse,
    RetrievedKnowledgeChunk,
)
from creative_coding_assistant.rag.sources import OfficialSourceType

RetrievalCompressionStatus = Literal["unchanged", "compressed"]
RetrievalCompressionPressure = Literal["low", "medium", "high"]

RETRIEVAL_COMPRESSION_CHUNK_SERIALIZATION_VERSION = "retrieval_compression_chunk.v1"
RETRIEVAL_COMPRESSION_RESULT_SERIALIZATION_VERSION = "retrieval_compression_result.v1"
RETRIEVAL_COMPRESSION_AUTHORITY_BOUNDARY = (
    "Retrieval compression produces separate compressed excerpts from existing "
    "retrieved chunks only; it preserves source retrieval metadata, does not "
    "execute retrieval queries, rerank sources, change retrieval filters, mutate "
    "source chunks, route context, select providers or models, write storage, or "
    "modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "retrieval_query_execution",
    "retrieval_reranking",
    "retrieval_filter_mutation",
    "source_chunk_mutation",
    "context_routing",
    "prompt_compression",
    "provider_or_model_routing",
    "persistent_storage_write",
    "generated_output_modification",
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


class RetrievalCompressionChunk(BaseModel):
    """One compressed retrieval chunk with provenance preserved."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    chunk_id: str = Field(min_length=1, max_length=200)
    source_id: str = Field(min_length=1, max_length=160)
    domain: CreativeCodingDomain
    source_type: OfficialSourceType
    publisher: str = Field(min_length=1, max_length=160)
    registry_title: str = Field(min_length=1, max_length=200)
    document_title: str = Field(min_length=1, max_length=240)
    source_url: str = Field(min_length=1, max_length=1200)
    chunk_index: int = Field(ge=0)
    rank: int | None = Field(default=None, ge=1)
    score: float = Field(ge=0, le=1)
    original_excerpt: str = Field(min_length=1, max_length=120_000)
    compressed_excerpt: str = Field(min_length=1, max_length=120_000)
    original_token_estimate: int = Field(ge=1, le=240_000)
    compressed_token_estimate: int = Field(ge=1, le=240_000)
    saved_tokens: int = Field(ge=0, le=240_000)
    compression_status: RetrievalCompressionStatus
    compression_pressure: RetrievalCompressionPressure
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    retrieval_compression_implemented: Literal[True] = True
    retrieval_query_execution_implemented: Literal[False] = False
    retrieval_reranking_implemented: Literal[False] = False
    retrieval_filter_mutation_implemented: Literal[False] = False
    source_chunk_mutation_implemented: Literal[False] = False
    context_routing_implemented: Literal[False] = False
    prompt_compression_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["retrieval_compression_chunk.v1"] = (
        RETRIEVAL_COMPRESSION_CHUNK_SERIALIZATION_VERSION
    )
    compression_only: Literal[True] = True

    @model_validator(mode="after")
    def _chunk_matches_compression(self) -> Self:
        if self.compressed_token_estimate > self.original_token_estimate:
            raise ValueError(
                "compressed_token_estimate must not exceed original_token_estimate"
            )
        if self.saved_tokens != (
            self.original_token_estimate - self.compressed_token_estimate
        ):
            raise ValueError("saved_tokens must match token estimate delta")
        expected_status = "compressed" if self.saved_tokens > 0 else "unchanged"
        if self.compression_status != expected_status:
            raise ValueError("compression_status must match saved tokens")
        if self.compression_status == "unchanged" and (
            self.compressed_excerpt != self.original_excerpt
        ):
            raise ValueError("unchanged chunks must preserve original excerpt")
        return self


class RetrievalCompressionResult(BaseModel):
    """Bounded V5.1 retrieval compression result for existing chunks."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["retrieval_compressor"] = "retrieval_compressor"
    serialization_version: Literal["retrieval_compression_result.v1"] = (
        RETRIEVAL_COMPRESSION_RESULT_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RETRIEVAL_COMPRESSION_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    source_retrieval_query: str | None = Field(default=None, max_length=2000)
    source_chunk_count: int = Field(ge=1, le=40)
    chunks: tuple[RetrievalCompressionChunk, ...] = Field(
        min_length=1,
        max_length=40,
    )
    chunk_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    target_token_budget: int = Field(ge=1, le=240_000)
    original_total_tokens: int = Field(ge=1, le=240_000)
    compressed_total_tokens: int = Field(ge=1, le=240_000)
    saved_total_tokens: int = Field(ge=0, le=240_000)
    within_budget: bool
    compression_pressure: RetrievalCompressionPressure
    compressed_retrieval_text: str = Field(min_length=1, max_length=240_000)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    retrieval_compression_implemented: Literal[True] = True
    retrieval_query_execution_implemented: Literal[False] = False
    retrieval_reranking_implemented: Literal[False] = False
    retrieval_filter_mutation_implemented: Literal[False] = False
    source_chunk_mutation_implemented: Literal[False] = False
    context_routing_implemented: Literal[False] = False
    prompt_compression_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    compression_only: Literal[True] = True

    @model_validator(mode="after")
    def _result_matches_chunks(self) -> Self:
        derived_chunk_ids = tuple(chunk.chunk_id for chunk in self.chunks)
        if len(set(derived_chunk_ids)) != len(derived_chunk_ids):
            raise ValueError("chunk_ids must be unique")
        if self.chunk_ids != derived_chunk_ids:
            raise ValueError("chunk_ids must match chunks")
        if self.source_chunk_count != len(self.chunks):
            raise ValueError("source_chunk_count must match chunks")
        original_total = sum(chunk.original_token_estimate for chunk in self.chunks)
        compressed_total = sum(chunk.compressed_token_estimate for chunk in self.chunks)
        saved_total = sum(chunk.saved_tokens for chunk in self.chunks)
        if self.original_total_tokens != original_total:
            raise ValueError("original_total_tokens must match chunks")
        if self.compressed_total_tokens != compressed_total:
            raise ValueError("compressed_total_tokens must match chunks")
        if self.saved_total_tokens != saved_total:
            raise ValueError("saved_total_tokens must match chunks")
        if self.saved_total_tokens != (
            self.original_total_tokens - self.compressed_total_tokens
        ):
            raise ValueError("saved_total_tokens must match token delta")
        if self.within_budget != (
            self.compressed_total_tokens <= self.target_token_budget
        ):
            raise ValueError("within_budget must match compressed token total")
        if self.compressed_retrieval_text != _join_compressed_chunks(self.chunks):
            raise ValueError("compressed_retrieval_text must match chunks")
        return self


def compress_retrieval_context(
    retrieval_context: RetrievalContextResponse,
    *,
    target_token_budget: int = 2_000,
) -> RetrievalCompressionResult:
    """Compress retrieved chunks without executing a retrieval query."""

    return compress_retrieval_chunks(
        retrieval_context.chunks,
        target_token_budget=target_token_budget,
        source_retrieval_query=retrieval_context.request.query,
    )


def compress_retrieval_chunks(
    chunks: tuple[RetrievedKnowledgeChunk, ...],
    *,
    target_token_budget: int = 2_000,
    source_retrieval_query: str | None = None,
) -> RetrievalCompressionResult:
    """Compress existing retrieval chunks with deterministic text reduction."""

    if not chunks:
        raise ValueError("retrieval compression requires at least one chunk")
    if target_token_budget <= 0:
        raise ValueError("target_token_budget must be positive")

    original_total = sum(_estimate_tokens(chunk.excerpt) for chunk in chunks)
    chunk_budgets = _chunk_token_budgets(
        chunks,
        target_token_budget=target_token_budget,
        original_total=original_total,
    )
    compressed_chunks = tuple(
        _compress_chunk(chunk, chunk_budgets[_chunk_id(chunk)]) for chunk in chunks
    )
    compressed_total = sum(
        chunk.compressed_token_estimate for chunk in compressed_chunks
    )
    saved_total = sum(chunk.saved_tokens for chunk in compressed_chunks)

    return RetrievalCompressionResult(
        source_retrieval_query=source_retrieval_query,
        source_chunk_count=len(chunks),
        chunks=compressed_chunks,
        chunk_ids=tuple(chunk.chunk_id for chunk in compressed_chunks),
        target_token_budget=target_token_budget,
        original_total_tokens=original_total,
        compressed_total_tokens=compressed_total,
        saved_total_tokens=saved_total,
        within_budget=compressed_total <= target_token_budget,
        compression_pressure=_compression_pressure(
            original_total=original_total,
            compressed_total=compressed_total,
            target_token_budget=target_token_budget,
        ),
        compressed_retrieval_text=_join_compressed_chunks(compressed_chunks),
        advisory_actions=_result_actions(saved_total),
    )


def retrieval_compression_chunk_by_id(
    chunk_id: str,
    result: RetrievalCompressionResult | None = None,
) -> RetrievalCompressionChunk | None:
    """Return one compressed retrieval chunk without mutating source chunks."""

    source_result = result or compress_retrieval_chunks((_placeholder_chunk(),))
    for chunk in source_result.chunks:
        if chunk.chunk_id == chunk_id:
            return chunk
    return None


def retrieval_compression_chunks_for_status(
    status: RetrievalCompressionStatus,
    result: RetrievalCompressionResult | None = None,
) -> tuple[RetrievalCompressionChunk, ...]:
    """Return retrieval compression chunks by status without reranking."""

    source_result = result or compress_retrieval_chunks((_placeholder_chunk(),))
    return tuple(
        chunk for chunk in source_result.chunks if chunk.compression_status == status
    )


def _compress_chunk(
    chunk: RetrievedKnowledgeChunk,
    target_tokens: int,
) -> RetrievalCompressionChunk:
    original = chunk.excerpt
    original_tokens = _estimate_tokens(original)
    compressed = (
        original
        if original_tokens <= target_tokens
        else _compress_text_to_budget(original, target_tokens)
    )
    compressed_tokens = _estimate_tokens(compressed)
    if compressed_tokens >= original_tokens:
        compressed = original
        compressed_tokens = original_tokens
    saved_tokens = original_tokens - compressed_tokens

    return RetrievalCompressionChunk(
        chunk_id=_chunk_id(chunk),
        source_id=chunk.source_id,
        domain=chunk.domain,
        source_type=chunk.source_type,
        publisher=chunk.publisher,
        registry_title=chunk.registry_title,
        document_title=chunk.document_title,
        source_url=chunk.source_url,
        chunk_index=chunk.chunk_index,
        rank=chunk.rank,
        score=chunk.score,
        original_excerpt=original,
        compressed_excerpt=compressed,
        original_token_estimate=original_tokens,
        compressed_token_estimate=compressed_tokens,
        saved_tokens=saved_tokens,
        compression_status="compressed" if saved_tokens else "unchanged",
        compression_pressure=_chunk_pressure(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            target_tokens=target_tokens,
        ),
        evidence=(
            f"source:{chunk.source_id}",
            f"chunk_index:{chunk.chunk_index}",
            f"rank:{chunk.rank or 'unranked'}",
            f"score:{chunk.score:.3f}",
            f"target_tokens:{target_tokens}",
        ),
        advisory_actions=_chunk_actions(saved_tokens),
    )


def _compress_text_to_budget(text: str, target_tokens: int) -> str:
    normalized_lines = tuple(
        " ".join(line.strip().split()) for line in text.splitlines() if line.strip()
    )
    normalized = (
        "\n".join(normalized_lines) if normalized_lines else " ".join(text.split())
    )
    if _estimate_tokens(normalized) <= target_tokens:
        return normalized

    marker = "[compressed: retrieval detail omitted]"
    character_budget = max(24, target_tokens * 4 - len(marker) - 1)
    sentences = tuple(
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(normalized)
        if sentence.strip()
    )
    selected: list[str] = []
    used = 0
    for sentence in sentences:
        separator = "\n" if selected else ""
        next_length = used + len(separator) + len(sentence)
        if next_length > character_budget:
            break
        selected.append(sentence)
        used = next_length

    if not selected:
        selected_text = normalized[:character_budget].rstrip()
    else:
        selected_text = "\n".join(selected).rstrip()
    return f"{selected_text}\n{marker}".strip()


def _chunk_token_budgets(
    chunks: tuple[RetrievedKnowledgeChunk, ...],
    *,
    target_token_budget: int,
    original_total: int,
) -> dict[str, int]:
    if original_total <= target_token_budget:
        return {_chunk_id(chunk): _estimate_tokens(chunk.excerpt) for chunk in chunks}

    remaining = target_token_budget
    budgets: dict[str, int] = {}
    for index, chunk in enumerate(chunks):
        original_tokens = _estimate_tokens(chunk.excerpt)
        if index == len(chunks) - 1:
            budget = max(1, remaining)
        else:
            proportional = max(
                1, target_token_budget * original_tokens // original_total
            )
            budget = min(original_tokens, proportional)
        budgets[_chunk_id(chunk)] = max(1, budget)
        remaining = max(0, remaining - budgets[_chunk_id(chunk)])
    return budgets


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _chunk_pressure(
    *,
    original_tokens: int,
    compressed_tokens: int,
    target_tokens: int,
) -> RetrievalCompressionPressure:
    if compressed_tokens > target_tokens:
        return "high"
    if compressed_tokens < original_tokens:
        return "medium"
    return "low"


def _compression_pressure(
    *,
    original_total: int,
    compressed_total: int,
    target_token_budget: int,
) -> RetrievalCompressionPressure:
    if compressed_total > target_token_budget:
        return "high"
    if compressed_total < original_total:
        return "medium"
    return "low"


def _chunk_id(chunk: RetrievedKnowledgeChunk) -> str:
    return f"retrieval::{chunk.source_id}::{chunk.chunk_index}"


def _chunk_actions(saved_tokens: int) -> tuple[str, ...]:
    actions = [
        "Produce compressed retrieval excerpt as a separate artifact.",
        "Preserve source chunk metadata for citation traceability.",
    ]
    if saved_tokens:
        actions.append("Record token savings without reranking retrieval sources.")
    return tuple(actions)


def _result_actions(saved_total: int) -> tuple[str, ...]:
    actions = [
        "Expose compressed retrieval excerpts only through this result.",
        "Preserve retrieval query, ranking, filters, provider routing, and output boundaries.",
    ]
    if saved_total:
        actions.append("Use compressed excerpts only when explicitly selected.")
    else:
        actions.append("Keep original retrieval excerpts because they fit the budget.")
    return tuple(actions)


def _join_compressed_chunks(
    chunks: tuple[RetrievalCompressionChunk, ...],
) -> str:
    return "\n\n".join(
        (
            f"[retrieval:{chunk.source_id}:{chunk.chunk_index}] "
            f"{chunk.document_title}\n{chunk.compressed_excerpt}"
        )
        for chunk in chunks
    )


def _placeholder_chunk() -> RetrievedKnowledgeChunk:
    return RetrievedKnowledgeChunk(
        source_id="placeholder",
        domain=CreativeCodingDomain.P5_JS,
        source_type=OfficialSourceType.GUIDE,
        publisher="placeholder",
        registry_title="Placeholder",
        document_title="Placeholder",
        source_url="https://example.invalid/placeholder",
        chunk_index=0,
        excerpt="Retrieval compression placeholder.",
        score=1.0,
        rank=1,
    )
