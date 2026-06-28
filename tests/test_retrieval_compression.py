import unittest

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.orchestration import (
    RetrievalCompressionChunk,
    RetrievalCompressionResult,
    RetrievalContextFilter,
    RetrievalContextRequest,
    RetrievalContextResponse,
    RetrievedKnowledgeChunk,
    RouteName,
    compress_retrieval_chunks,
    compress_retrieval_context,
    retrieval_compression_chunk_by_id,
    retrieval_compression_chunks_for_status,
)
from creative_coding_assistant.rag.sources import OfficialSourceType

REQUIRED_RETRIEVAL_COMPRESSION_CHUNK_FIELDS = {
    "chunk_id",
    "source_id",
    "domain",
    "source_type",
    "publisher",
    "registry_title",
    "document_title",
    "source_url",
    "chunk_index",
    "rank",
    "score",
    "original_excerpt",
    "compressed_excerpt",
    "original_token_estimate",
    "compressed_token_estimate",
    "saved_tokens",
    "compression_status",
    "compression_pressure",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "retrieval_compression_implemented",
    "retrieval_query_execution_implemented",
    "retrieval_reranking_implemented",
    "retrieval_filter_mutation_implemented",
    "source_chunk_mutation_implemented",
    "context_routing_implemented",
    "prompt_compression_implemented",
    "provider_model_routing_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "compression_only",
}


class RetrievalCompressionTests(unittest.TestCase):
    def test_short_retrieval_chunk_remains_unchanged_with_boundary_flags(self) -> None:
        result = compress_retrieval_chunks((_chunk(1, "Use createCanvas in setup."),))
        chunk = result.chunks[0]

        self.assertEqual(result.role, "retrieval_compressor")
        self.assertEqual(
            result.serialization_version,
            "retrieval_compression_result.v1",
        )
        self.assertEqual(result.chunk_ids, ("retrieval::p5-reference::1",))
        self.assertEqual(result.source_chunk_count, 1)
        self.assertEqual(result.saved_total_tokens, 0)
        self.assertTrue(result.within_budget)
        self.assertEqual(result.compression_pressure, "low")
        self.assertIn("preserves source retrieval metadata", result.authority_boundary)
        self.assertTrue(result.retrieval_compression_implemented)
        self.assertFalse(result.retrieval_query_execution_implemented)
        self.assertFalse(result.retrieval_reranking_implemented)
        self.assertFalse(result.retrieval_filter_mutation_implemented)
        self.assertFalse(result.source_chunk_mutation_implemented)
        self.assertFalse(result.context_routing_implemented)
        self.assertFalse(result.prompt_compression_implemented)
        self.assertFalse(result.provider_model_routing_implemented)
        self.assertFalse(result.persistent_storage_write_implemented)
        self.assertFalse(result.generated_output_mutation_implemented)
        self.assertTrue(result.compression_only)
        self.assertEqual(chunk.compression_status, "unchanged")
        self.assertEqual(chunk.original_excerpt, chunk.compressed_excerpt)

    def test_long_retrieval_context_is_compressed_with_provenance(self) -> None:
        context = RetrievalContextResponse(
            request=RetrievalContextRequest(
                query="p5.js audio reactive geometry",
                route=RouteName.GENERATE,
                filters=RetrievalContextFilter(domain=CreativeCodingDomain.P5_JS),
            ),
            chunks=(
                _chunk(1, _repeated("Amplitude analysis maps sound to shape.", 120)),
                _chunk(2, _repeated("FFT bins can drive color and scale.", 140)),
            ),
        )
        result = compress_retrieval_context(context, target_token_budget=120)

        self.assertEqual(result.source_retrieval_query, context.request.query)
        self.assertGreater(result.original_total_tokens, result.compressed_total_tokens)
        self.assertGreater(result.saved_total_tokens, 0)
        self.assertLessEqual(result.compressed_total_tokens, result.target_token_budget)
        self.assertTrue(result.within_budget)
        self.assertEqual(result.compression_pressure, "medium")
        self.assertIn("[retrieval:p5-reference:1]", result.compressed_retrieval_text)
        self.assertIn(
            "Use compressed excerpts only when explicitly selected.",
            result.advisory_actions,
        )

        for chunk in result.chunks:
            self.assertEqual(
                set(chunk.model_dump(mode="json")),
                REQUIRED_RETRIEVAL_COMPRESSION_CHUNK_FIELDS,
            )
            self.assertEqual(
                chunk.serialization_version,
                "retrieval_compression_chunk.v1",
            )
            self.assertEqual(chunk.source_id, "p5-reference")
            self.assertEqual(chunk.domain, CreativeCodingDomain.P5_JS)
            self.assertEqual(chunk.source_type, OfficialSourceType.GUIDE)
            self.assertEqual(chunk.compression_status, "compressed")
            self.assertLess(
                chunk.compressed_token_estimate,
                chunk.original_token_estimate,
            )
            self.assertEqual(
                chunk.saved_tokens,
                chunk.original_token_estimate - chunk.compressed_token_estimate,
            )
            self.assertIn("retrieval_reranking", chunk.blocked_runtime_behaviors)
            self.assertTrue(chunk.retrieval_compression_implemented)
            self.assertFalse(chunk.retrieval_query_execution_implemented)
            self.assertFalse(chunk.retrieval_reranking_implemented)
            self.assertFalse(chunk.retrieval_filter_mutation_implemented)
            self.assertFalse(chunk.source_chunk_mutation_implemented)
            self.assertFalse(chunk.context_routing_implemented)
            self.assertFalse(chunk.prompt_compression_implemented)
            self.assertFalse(chunk.provider_model_routing_implemented)
            self.assertFalse(chunk.persistent_storage_write_implemented)
            self.assertFalse(chunk.generated_output_mutation_implemented)
            self.assertTrue(chunk.compression_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        result = compress_retrieval_chunks(
            (
                _chunk(1, _repeated("Amplitude analysis maps sound to shape.", 80)),
                _chunk(2, _repeated("FFT bins can drive color and scale.", 100)),
            ),
            target_token_budget=90,
        )
        first = retrieval_compression_chunk_by_id("retrieval::p5-reference::1", result)
        compressed = retrieval_compression_chunks_for_status("compressed", result)
        missing = retrieval_compression_chunk_by_id("missing", result)

        self.assertIsNone(missing)
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first.chunk_index, 1)
        self.assertEqual(len(compressed), 2)
        self.assertIs(first, compressed[0])

    def test_result_rejects_mismatched_chunks_or_totals(self) -> None:
        result = compress_retrieval_chunks(
            (_chunk(1, _repeated("Amplitude analysis maps sound to shape.", 100)),),
            target_token_budget=50,
        )
        payload = result.model_dump(mode="json")
        payload["chunk_ids"] = ("missing",)

        with self.assertRaisesRegex(ValueError, "chunk_ids must match"):
            RetrievalCompressionResult(**payload)

        payload = result.model_dump(mode="json")
        payload["compressed_total_tokens"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "compressed_total_tokens must match",
        ):
            RetrievalCompressionResult(**payload)

        chunk_payload = result.chunks[0].model_dump(mode="json")
        chunk_payload["saved_tokens"] += 1

        with self.assertRaisesRegex(ValueError, "saved_tokens must match"):
            RetrievalCompressionChunk(**chunk_payload)

    def test_result_does_not_declare_retrieval_or_provider_mutation_terms(self) -> None:
        result = compress_retrieval_chunks(
            (_chunk(1, _repeated("Amplitude analysis maps sound to shape.", 100)),),
            target_token_budget=50,
        )
        combined_text = " ".join(
            (
                result.authority_boundary,
                *result.blocked_runtime_behaviors,
                *result.advisory_actions,
                *(
                    field
                    for chunk in result.chunks
                    for field in (
                        chunk.chunk_id,
                        chunk.source_id,
                        chunk.publisher,
                        chunk.registry_title,
                        chunk.document_title,
                        *chunk.evidence,
                        *chunk.advisory_actions,
                        *chunk.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_retrieval_query(",
            "rerank_retrieval(",
            "mutate_retrieval_filter(",
            "mutate_source_chunk(",
            "route_context(",
            "compress_prompt(",
            "select_provider(",
            "route_provider(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


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


def _repeated(sentence: str, count: int) -> str:
    return " ".join([sentence] * count)


if __name__ == "__main__":
    unittest.main()
