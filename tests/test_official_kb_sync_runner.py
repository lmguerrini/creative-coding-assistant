import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.rag.sync import (
    ChunkingPolicy,
    OfficialKnowledgeBaseIndexer,
    OfficialKnowledgeBaseSyncRunner,
    OfficialSourceChunker,
    OfficialSourceFetcher,
    OfficialSourceNormalizer,
    OfficialSourceSyncRequest,
    TransportResponse,
)
from creative_coding_assistant.vectorstore import create_chroma_client


class OfficialKnowledgeBaseSyncRunnerTests(unittest.TestCase):
    def test_runner_executes_full_sync_pipeline(self) -> None:
        with _kb_client() as client:
            runner = OfficialKnowledgeBaseSyncRunner(
                fetcher=OfficialSourceFetcher(transport=_FakeTransport()),
                normalizer=OfficialSourceNormalizer(),
                chunker=OfficialSourceChunker(
                    policy=ChunkingPolicy(max_chars=200, min_chunk_chars=50)
                ),
                embedder=_FakeChunkEmbedder(),
                indexer=OfficialKnowledgeBaseIndexer(client=client),
            )

            result = runner.run(_sync_request())

            self.assertEqual(result.request.source_id, "three_docs")
            self.assertEqual(
                result.fetched_document.domain,
                CreativeCodingDomain.THREE_JS,
            )
            self.assertEqual(result.normalized_document.document_title, "Three.js Docs")
            self.assertGreater(len(result.chunks), 1)
            self.assertEqual(len(result.embeddings), len(result.chunks))
            self.assertEqual(len(result.vector_records), len(result.chunks))
            self.assertEqual(len(result.record_ids), len(result.chunks))
            self.assertEqual(
                result.vector_records[0].embedding,
                [1.0, 0.0, 0.0],
            )
            self.assertEqual(
                result.vector_records[0].metadata.source_id,
                "three_docs",
            )
            stored_records = OfficialKnowledgeBaseIndexer(
                client=client
            ).list_source_chunks(
                source_id="three_docs",
            )
            self.assertEqual(len(stored_records), len(result.chunks))

    def test_runner_is_idempotent_for_repeated_source_syncs(self) -> None:
        with _kb_client() as client:
            indexer = OfficialKnowledgeBaseIndexer(client=client)
            runner = OfficialKnowledgeBaseSyncRunner(
                fetcher=OfficialSourceFetcher(transport=_FakeTransport()),
                chunker=OfficialSourceChunker(
                    policy=ChunkingPolicy(max_chars=200, min_chunk_chars=50)
                ),
                embedder=_FakeChunkEmbedder(),
                indexer=indexer,
            )
            request = _sync_request()

            first_result = runner.run(request)
            second_result = runner.run(request)
            stored_records = indexer.list_source_chunks(source_id="three_docs")

            self.assertEqual(first_result.record_ids, second_result.record_ids)
            self.assertEqual(len(stored_records), len(first_result.chunks))


def _sync_request() -> OfficialSourceSyncRequest:
    return OfficialSourceSyncRequest(
        source_id="three_docs",
        requested_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )


class _FakeTransport:
    def fetch(self, url: str) -> TransportResponse:
        if url != "https://threejs.org/docs/":
            raise KeyError(url)
        return TransportResponse(
            resolved_url="https://threejs.org/docs/",
            status_code=200,
            content_type="text/html",
            content="""
            <html>
              <head><title>Three.js Docs</title></head>
              <body>
                <p>Camera setup covers perspective cameras and scene framing.</p>
                <p>Lighting setup explains ambient, directional, and point lights.</p>
                <p>
                  Renderer setup covers canvas sizing, pixel ratio,
                  and animation loops.
                </p>
                <p>
                  Scene composition guidance covers fog, background color,
                  object grouping, and update flow for maintainable examples.
                </p>
                <p>
                  Interaction guidance covers raycasting, resize handling,
                  and cleanup work that keeps examples stable across reruns.
                </p>
              </body>
            </html>
            """,
        )


class _FakeChunkEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed_chunks(self, chunks) -> tuple[list[float], ...]:
        chunk_texts = tuple(chunk.text for chunk in chunks)
        self.calls.append(chunk_texts)
        return tuple(
            [float(index + 1), 0.0, 0.0]
            for index, _chunk in enumerate(chunks)
        )


class _kb_client:
    def __enter__(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        return create_chroma_client(path=Path(self._temp_dir.name) / "chroma")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
