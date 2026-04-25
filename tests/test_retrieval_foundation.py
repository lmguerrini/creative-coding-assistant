import hashlib
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.rag.retrieval import (
    KnowledgeBaseRetrievalFilter,
    KnowledgeBaseRetrievalRequest,
    KnowledgeBaseRetriever,
)
from creative_coding_assistant.rag.sources import OfficialSourceType
from creative_coding_assistant.rag.sync import OfficialKnowledgeBaseIndexer
from creative_coding_assistant.rag.sync.models import OfficialSourceChunk
from creative_coding_assistant.vectorstore import (
    ChromaCollection,
    ChromaRecordMetadata,
    ChromaRepository,
    VectorRecord,
    VectorRecordKind,
    create_chroma_client,
    get_collection_definition,
)


class RetrievalFoundationTests(unittest.TestCase):
    def test_retriever_returns_semantic_matches_from_indexed_kb_chunks(self) -> None:
        with _kb_client() as client:
            _seed_kb_records(client)
            retriever = KnowledgeBaseRetriever(
                client=client,
                embedder=_FakeQueryEmbedder({"camera guidance": [1.0, 0.0, 0.0]}),
            )

            response = retriever.search(
                KnowledgeBaseRetrievalRequest(query="camera guidance", limit=2)
            )

            self.assertEqual(len(response.results), 2)
            self.assertEqual(response.results[0].source_id, "three_docs")
            self.assertEqual(response.results[0].domain, CreativeCodingDomain.THREE_JS)
            self.assertEqual(
                response.results[0].source_type,
                OfficialSourceType.API_REFERENCE,
            )
            self.assertIn("camera", response.results[0].text.lower())
            self.assertGreaterEqual(
                response.results[0].score,
                response.results[1].score,
            )

    def test_retriever_applies_domain_filter(self) -> None:
        with _kb_client() as client:
            _seed_kb_records(client)
            retriever = KnowledgeBaseRetriever(
                client=client,
                embedder=_FakeQueryEmbedder({"creative code": [1.0, 0.0, 0.0]}),
            )

            response = retriever.search(
                KnowledgeBaseRetrievalRequest(
                    query="creative code",
                    filters=KnowledgeBaseRetrievalFilter(
                        domain=CreativeCodingDomain.P5_JS
                    ),
                )
            )

            self.assertEqual(len(response.results), 1)
            self.assertEqual(response.results[0].domain, CreativeCodingDomain.P5_JS)
            self.assertEqual(response.results[0].source_id, "p5_reference")

    def test_retriever_treats_empty_domains_as_unconstrained(self) -> None:
        with _kb_client() as client:
            _seed_kb_records(client)
            retriever = KnowledgeBaseRetriever(
                client=client,
                embedder=_FakeQueryEmbedder({"creative code": [1.0, 0.0, 0.0]}),
            )

            response = retriever.search(
                KnowledgeBaseRetrievalRequest(
                    query="creative code",
                    limit=3,
                    filters=KnowledgeBaseRetrievalFilter(domains=()),
                )
            )

            self.assertEqual(len(response.results), 3)
            self.assertEqual(
                {result.domain for result in response.results},
                {
                    CreativeCodingDomain.THREE_JS,
                    CreativeCodingDomain.P5_JS,
                    CreativeCodingDomain.GLSL,
                },
            )

    def test_retriever_applies_multi_domain_filter(self) -> None:
        with _kb_client() as client:
            _seed_kb_records(client)
            retriever = KnowledgeBaseRetriever(
                client=client,
                embedder=_FakeQueryEmbedder({"creative code": [1.0, 0.0, 0.0]}),
            )

            response = retriever.search(
                KnowledgeBaseRetrievalRequest(
                    query="creative code",
                    limit=3,
                    filters=KnowledgeBaseRetrievalFilter(
                        domains=(
                            CreativeCodingDomain.P5_JS,
                            CreativeCodingDomain.GLSL,
                        )
                    ),
                )
            )

            self.assertEqual(len(response.results), 2)
            self.assertEqual(
                {result.domain for result in response.results},
                {
                    CreativeCodingDomain.P5_JS,
                    CreativeCodingDomain.GLSL,
                },
            )

    def test_retriever_applies_source_filter(self) -> None:
        with _kb_client() as client:
            _seed_kb_records(client)
            retriever = KnowledgeBaseRetriever(
                client=client,
                embedder=_FakeQueryEmbedder({"shader docs": [1.0, 0.0, 0.0]}),
            )

            response = retriever.search(
                KnowledgeBaseRetrievalRequest(
                    query="shader docs",
                    filters=KnowledgeBaseRetrievalFilter(
                        source_id="glsl_language_spec_460"
                    ),
                )
            )

            self.assertEqual(len(response.results), 1)
            self.assertEqual(
                response.results[0].source_id,
                "glsl_language_spec_460",
            )
            self.assertEqual(response.results[0].publisher, "Khronos Group")

    def test_retriever_uses_only_kb_collection(self) -> None:
        with _kb_client() as client:
            _seed_kb_records(client)
            _seed_non_kb_record(client)
            retriever = KnowledgeBaseRetriever(
                client=client,
                embedder=_FakeQueryEmbedder({"camera guidance": [1.0, 0.0, 0.0]}),
            )

            response = retriever.search(
                KnowledgeBaseRetrievalRequest(query="camera guidance", limit=1)
            )

            self.assertEqual(len(response.results), 1)
            self.assertEqual(response.results[0].source_id, "three_docs")
            self.assertNotEqual(
                response.results[0].record_id,
                "project_memory:project_memory:v1:manual",
            )

    def test_retriever_rejects_empty_query_embedding(self) -> None:
        with _kb_client() as client:
            _seed_kb_records(client)
            retriever = KnowledgeBaseRetriever(
                client=client,
                embedder=_FakeQueryEmbedder({"bad query": []}),
            )

            with self.assertRaisesRegex(ValueError, "must not be empty"):
                retriever.search(KnowledgeBaseRetrievalRequest(query="bad query"))


def _seed_kb_records(client) -> None:
    indexer = OfficialKnowledgeBaseIndexer(client=client)
    chunks = (
        _chunk(
            source_id="three_docs",
            domain=CreativeCodingDomain.THREE_JS,
            source_type=OfficialSourceType.API_REFERENCE,
            publisher="three.js",
            registry_title="three.js Documentation",
            source_url="https://threejs.org/docs/",
            text="Camera setup covers perspective cameras and scene framing.",
        ),
        _chunk(
            source_id="p5_reference",
            domain=CreativeCodingDomain.P5_JS,
            source_type=OfficialSourceType.API_REFERENCE,
            publisher="p5.js",
            registry_title="p5.js Reference",
            source_url="https://p5js.org/reference/",
            text="noise() creates smooth motion for sketches and generative art.",
        ),
        _chunk(
            source_id="glsl_language_spec_460",
            domain=CreativeCodingDomain.GLSL,
            source_type=OfficialSourceType.SPECIFICATION,
            publisher="Khronos Group",
            registry_title="GLSL 4.60 Specification",
            source_url=(
                "https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html"
            ),
            text="Uniform variables provide external values to shader programs.",
        ),
    )
    embeddings = (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    )
    indexer.upsert_chunks(chunks, embeddings)


def _seed_non_kb_record(client) -> None:
    repository = ChromaRepository(
        client=client,
        definition=get_collection_definition(ChromaCollection.PROJECT_MEMORY),
    )
    repository.upsert(
        [
            VectorRecord(
                id="project_memory:project_memory:v1:manual",
                document="This project memory vector should never appear in KB search.",
                metadata=ChromaRecordMetadata(
                    collection=ChromaCollection.PROJECT_MEMORY,
                    record_kind=VectorRecordKind.PROJECT_MEMORY,
                    source_id="memory-note",
                    project_id="project-1",
                    domain=CreativeCodingDomain.THREE_JS,
                ),
                embedding=[1.0, 0.0, 0.0],
            )
        ]
    )


def _chunk(
    *,
    source_id: str,
    domain: CreativeCodingDomain,
    source_type: OfficialSourceType,
    publisher: str,
    registry_title: str,
    source_url: str,
    text: str,
) -> OfficialSourceChunk:
    content_hash = _digest(source_id + source_url)
    chunk_hash = _digest(text)
    return OfficialSourceChunk(
        source_id=source_id,
        domain=domain,
        source_type=source_type,
        registry_title=registry_title,
        publisher=publisher,
        source_url=source_url,
        resolved_url=source_url,
        fetched_at=_time(),
        document_title=registry_title,
        content_hash=content_hash,
        chunk_index=0,
        text=text,
        chunk_hash=chunk_hash,
        char_count=len(text),
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _time() -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


class _FakeQueryEmbedder:
    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self._embeddings = embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings[text]


class _kb_client:
    def __enter__(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        return create_chroma_client(path=Path(self._temp_dir.name) / "chroma")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
