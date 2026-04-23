import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.rag.sources import OfficialSourceType
from creative_coding_assistant.rag.sync import (
    ChunkingPolicy,
    FetchedSourceDocument,
    NormalizedSourceDocument,
    OfficialKnowledgeBaseIndexer,
    OfficialSourceChunker,
    OfficialSourceFetcher,
    OfficialSourceNormalizer,
    OfficialSourceSyncRequest,
    SourceContentFormat,
    TransportResponse,
)
from creative_coding_assistant.vectorstore import create_chroma_client


class OfficialKnowledgeBaseSyncFoundationTests(unittest.TestCase):
    def test_fetcher_requires_registered_source_id(self) -> None:
        fetcher = OfficialSourceFetcher(transport=_FakeTransport({}))
        request = OfficialSourceSyncRequest(
            source_id="unknown_source",
            requested_at=_time(),
        )

        with self.assertRaises(ValueError):
            fetcher.fetch(request)

    def test_fetcher_rejects_resolved_url_outside_approved_scope(self) -> None:
        fetcher = OfficialSourceFetcher(
            transport=_FakeTransport(
                {
                    "https://threejs.org/docs/": TransportResponse(
                        resolved_url="https://threejs.org/manual/",
                        status_code=200,
                        content_type="text/html; charset=utf-8",
                        content="<html><body>Manual</body></html>",
                    )
                }
            )
        )
        request = OfficialSourceSyncRequest(
            source_id="three_docs",
            requested_at=_time(),
        )

        with self.assertRaisesRegex(ValueError, "approved path scope"):
            fetcher.fetch(request)

    def test_fetcher_builds_fetched_document_from_approved_source(self) -> None:
        fetcher = OfficialSourceFetcher(
            transport=_FakeTransport(
                {
                    "https://threejs.org/docs/": TransportResponse(
                        resolved_url="https://threejs.org/docs/",
                        status_code=200,
                        content_type="text/html; charset=utf-8",
                        content="<html><title>Docs</title><body>Hello</body></html>",
                    )
                }
            )
        )
        request = OfficialSourceSyncRequest(
            source_id="three_docs",
            requested_at=_time(),
        )

        document = fetcher.fetch(request)

        self.assertEqual(document.domain, CreativeCodingDomain.THREE_JS)
        self.assertEqual(document.source_type, OfficialSourceType.API_REFERENCE)
        self.assertEqual(document.content_format, SourceContentFormat.HTML)
        self.assertEqual(document.resolved_url, "https://threejs.org/docs/")
        self.assertEqual(len(document.raw_content_hash), 64)

    def test_normalizer_extracts_html_text_and_title(self) -> None:
        document = _fetched_document(
            raw_content="""
            <html>
              <head><title>Three.js API</title><style>.x { color: red; }</style></head>
              <body>
                <h1>Raycaster</h1>
                <p>Intersects objects in 3D scenes.</p>
                <script>window.secret = true;</script>
              </body>
            </html>
            """,
        )

        normalized = OfficialSourceNormalizer().normalize(document)

        self.assertEqual(normalized.document_title, "Three.js API")
        self.assertIn("Raycaster", normalized.normalized_text)
        self.assertIn("Intersects objects in 3D scenes.", normalized.normalized_text)
        self.assertNotIn("window.secret", normalized.normalized_text)

    def test_chunker_respects_character_boundaries(self) -> None:
        document = _normalized_document(
            normalized_text=(
                "Paragraph one explains setup and renderer wiring for a basic scene. "
                "It includes camera setup, sizing, and render loop notes.\n\n"
                "Paragraph two explains cameras, lighting, and scene composition "
                "for a more detailed example with orbit controls.\n\n"
                "Paragraph three covers animation loops, resizing, and cleanup steps "
                "that keep the example maintainable over time."
            )
        )
        chunker = OfficialSourceChunker(
            policy=ChunkingPolicy(max_chars=200, min_chunk_chars=50)
        )

        chunks = chunker.chunk(document)

        self.assertEqual(
            [chunk.chunk_index for chunk in chunks],
            list(range(len(chunks))),
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.char_count <= 200 for chunk in chunks))
        self.assertIn("Paragraph one", chunks[0].text)
        self.assertIn("Paragraph three", chunks[-1].text)

    def test_chunker_splits_large_paragraph(self) -> None:
        long_paragraph = " ".join(f"token{index}" for index in range(80))
        document = _normalized_document(normalized_text=long_paragraph)
        chunker = OfficialSourceChunker(
            policy=ChunkingPolicy(max_chars=200, min_chunk_chars=50)
        )

        chunks = chunker.chunk(document)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.char_count <= 200 for chunk in chunks))

    def test_indexer_upserts_official_kb_chunks_into_chroma(self) -> None:
        with _kb_client() as client:
            indexer = OfficialKnowledgeBaseIndexer(client=client)
            chunks = OfficialSourceChunker().chunk(
                _normalized_document(
                    normalized_text="Official docs chunk for indexing boundaries."
                )
            )

            record_ids = indexer.upsert_chunks(chunks, embeddings=[[0.1, 0.2, 0.3]])
            stored_records = indexer.list_source_chunks(source_id="three_docs")

            self.assertEqual(len(record_ids), 1)
            self.assertEqual(len(stored_records), 1)
            self.assertEqual(
                stored_records[0].metadata["collection"],
                "kb_official_docs",
            )
            self.assertEqual(stored_records[0].metadata["source_type"], "api_reference")
            self.assertEqual(stored_records[0].metadata["chunk_index"], 0)

    def test_indexer_requires_chunk_embedding_alignment(self) -> None:
        with _kb_client() as client:
            indexer = OfficialKnowledgeBaseIndexer(client=client)
            chunks = OfficialSourceChunker(
                policy=ChunkingPolicy(max_chars=200, min_chunk_chars=50)
            ).chunk(
                _normalized_document(
                    normalized_text=(
                        "First long chunk contains setup details, rendering notes, "
                        "and animation loop guidance for a maintained example.\n\n"
                        "Second long chunk contains scene graph guidance, camera "
                        "positioning, and cleanup details for another example."
                    )
                )
            )

            with self.assertRaisesRegex(ValueError, "align one-to-one"):
                indexer.upsert_chunks(chunks, embeddings=[[0.1, 0.2, 0.3]])


def _fetched_document(
    *,
    raw_content: str,
    content_format: SourceContentFormat = SourceContentFormat.HTML,
) -> FetchedSourceDocument:
    return FetchedSourceDocument.from_content(
        source_id="three_docs",
        domain=CreativeCodingDomain.THREE_JS,
        source_type=OfficialSourceType.API_REFERENCE,
        registry_title="three.js Documentation",
        publisher="three.js",
        source_url="https://threejs.org/docs/",
        resolved_url="https://threejs.org/docs/",
        fetched_at=_time(),
        content_format=content_format,
        raw_content=raw_content,
    )


def _normalized_document(*, normalized_text: str) -> NormalizedSourceDocument:
    return NormalizedSourceDocument.from_text(
        fetched_document=_fetched_document(raw_content="<html><body>stub</body></html>"),
        document_title="three.js Documentation",
        normalized_text=normalized_text,
    )


def _time() -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


class _FakeTransport:
    def __init__(self, responses: dict[str, TransportResponse]) -> None:
        self._responses = responses

    def fetch(self, url: str) -> TransportResponse:
        return self._responses[url]


class _kb_client:
    def __enter__(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        return create_chroma_client(path=Path(self._temp_dir.name) / "chroma")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
