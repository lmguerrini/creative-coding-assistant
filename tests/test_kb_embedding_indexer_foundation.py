import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.core import Settings
from creative_coding_assistant.rag.sources import OfficialSourceType
from creative_coding_assistant.rag.sync import (
    OfficialKnowledgeBaseIndexer,
    OfficialSourceChunk,
    OpenAIChunkEmbedder,
    build_chunk_embedder,
)
from creative_coding_assistant.vectorstore import create_chroma_client


class KnowledgeBaseEmbeddingIndexerFoundationTests(unittest.TestCase):
    def test_build_chunk_embedder_returns_none_without_embedding_config(self) -> None:
        settings = Settings(openai_api_key=None)

        embedder = build_chunk_embedder(settings)

        self.assertIsNone(embedder)

    def test_build_chunk_embedder_uses_openai_when_configured(self) -> None:
        settings = Settings(openai_api_key="sk-test-secret")

        embedder = build_chunk_embedder(settings)

        self.assertIsInstance(embedder, OpenAIChunkEmbedder)

    def test_openai_chunk_embedder_uses_settings_backed_model(self) -> None:
        settings = Settings(
            openai_api_key="sk-test-secret",
            openai_embedding_model="text-embedding-3-large",
        )
        chunks = (_chunk(index=0, text="Camera setup guidance."),)
        client = _FakeOpenAIClient(embeddings=([0.1, 0.2, 0.3],))
        embedder = OpenAIChunkEmbedder(settings=settings, client=client)

        embeddings = embedder.embed_chunks(chunks)

        self.assertEqual(embeddings, ([0.1, 0.2, 0.3],))
        self.assertEqual(
            client.calls,
            [
                {
                    "model": "text-embedding-3-large",
                    "input": ["Camera setup guidance."],
                }
            ],
        )

    def test_indexer_builds_vector_records_with_explicit_embeddings(self) -> None:
        with _kb_client() as client:
            indexer = OfficialKnowledgeBaseIndexer(client=client)
            chunk = _chunk(index=0, text="Official docs chunk for indexing.")

            records = indexer.build_vector_records([chunk], [[0.1, 0.2, 0.3]])

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].embedding, [0.1, 0.2, 0.3])
            self.assertEqual(records[0].document, chunk.text)
            self.assertEqual(records[0].metadata.source_id, chunk.source_id)
            self.assertEqual(records[0].metadata.domain, chunk.domain)

    def test_indexer_embeds_and_upserts_chunks(self) -> None:
        with _kb_client() as client:
            indexer = OfficialKnowledgeBaseIndexer(client=client)
            chunks = (
                _chunk(index=0, text="Camera setup guidance."),
                _chunk(index=1, text="Lighting setup guidance."),
            )
            embedder = _FakeChunkEmbedder(
                embeddings=(
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                )
            )

            record_ids = indexer.embed_and_upsert_chunks(chunks, embedder=embedder)
            stored_records = indexer.list_source_chunks(source_id="three_docs")

            self.assertEqual(len(record_ids), 2)
            self.assertEqual(len(stored_records), 2)
            self.assertEqual(embedder.calls, [tuple(chunk.text for chunk in chunks)])

    def test_indexer_embed_path_requires_embedding_alignment(self) -> None:
        with _kb_client() as client:
            indexer = OfficialKnowledgeBaseIndexer(client=client)
            chunks = (
                _chunk(index=0, text="Camera setup guidance."),
                _chunk(index=1, text="Lighting setup guidance."),
            )
            embedder = _FakeChunkEmbedder(embeddings=([0.1, 0.2, 0.3],))

            with self.assertRaisesRegex(ValueError, "align one-to-one"):
                indexer.embed_and_upsert_chunks(chunks, embedder=embedder)


def _chunk(*, index: int, text: str) -> OfficialSourceChunk:
    return OfficialSourceChunk(
        source_id="three_docs",
        domain=CreativeCodingDomain.THREE_JS,
        source_type=OfficialSourceType.API_REFERENCE,
        registry_title="three.js Documentation",
        publisher="three.js",
        source_url="https://threejs.org/docs/",
        resolved_url="https://threejs.org/docs/",
        fetched_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        document_title="three.js Documentation",
        content_hash="a" * 64,
        chunk_index=index,
        text=text,
        chunk_hash=f"{index + 1:b}".zfill(64),
        char_count=len(text),
    )


class _FakeEmbeddingsApi:
    def __init__(
        self,
        *,
        embeddings: tuple[list[float], ...],
        calls: list[dict[str, object]],
    ) -> None:
        self._embeddings = embeddings
        self._calls = calls

    def create(self, *, model: str, input: list[str]) -> dict[str, object]:
        self._calls.append({"model": model, "input": input})
        return {
            "data": [
                {"embedding": embedding}
                for embedding in self._embeddings
            ]
        }


class _FakeOpenAIClient:
    def __init__(self, *, embeddings: tuple[list[float], ...]) -> None:
        self.calls: list[dict[str, object]] = []
        self.embeddings = _FakeEmbeddingsApi(
            embeddings=embeddings,
            calls=self.calls,
        )


class _FakeChunkEmbedder:
    def __init__(self, *, embeddings: tuple[list[float], ...]) -> None:
        self._embeddings = embeddings
        self.calls: list[tuple[str, ...]] = []

    def embed_chunks(
        self,
        chunks: tuple[OfficialSourceChunk, ...] | list[OfficialSourceChunk],
    ) -> tuple[list[float], ...]:
        self.calls.append(tuple(chunk.text for chunk in chunks))
        return self._embeddings


class _kb_client:
    def __enter__(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        return create_chroma_client(path=Path(self._temp_dir.name) / "chroma")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
