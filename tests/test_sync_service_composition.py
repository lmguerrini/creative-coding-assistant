import unittest
from datetime import UTC, datetime
from unittest.mock import patch

from creative_coding_assistant.app import (
    OfficialKBSyncService,
    build_official_kb_sync_service,
)
from creative_coding_assistant.core import Settings
from creative_coding_assistant.rag.sources import get_official_source
from creative_coding_assistant.rag.sync import (
    FetchedSourceDocument,
    NormalizedSourceDocument,
    OfficialKnowledgeBaseSyncResult,
    OfficialSourceChunk,
    OfficialSourceSyncRequest,
    SourceContentFormat,
)
from creative_coding_assistant.vectorstore import (
    ChromaCollection,
    ChromaRecordMetadata,
    VectorRecord,
    VectorRecordKind,
)


class SyncServiceCompositionTests(unittest.TestCase):
    def test_build_sync_service_defers_runner_construction_until_sync(self) -> None:
        settings = Settings(openai_api_key="sk-test-secret")
        runner = _FakeRunner()

        with patch(
            "creative_coding_assistant.app.sync_bootstrap.build_official_kb_sync_runner",
            return_value=runner,
        ) as build_runner:
            service = build_official_kb_sync_service(settings=settings)
            build_runner.assert_not_called()

            result = service.sync_selected_sources(("three_docs",))

        build_runner.assert_called_once_with(
            settings=settings,
            transport=None,
            chunk_embedder=None,
            normalizer=None,
            chunker=None,
        )
        self.assertIsInstance(service, OfficialKBSyncService)
        self.assertEqual(result.source_ids, ("three_docs",))

    def test_build_sync_service_preserves_explicit_runner_override(self) -> None:
        runner = _FakeRunner()

        with patch(
            "creative_coding_assistant.app.sync_bootstrap.build_official_kb_sync_runner",
            side_effect=AssertionError("Runner builder should not be called."),
        ) as build_runner:
            service = build_official_kb_sync_service(runner=runner)
            result = service.sync_selected_sources(("three_docs",))

        build_runner.assert_not_called()
        self.assertEqual(result.source_ids, ("three_docs",))


class _FakeRunner:
    def __init__(self) -> None:
        self.requests: list[OfficialSourceSyncRequest] = []

    def run(
        self,
        request: OfficialSourceSyncRequest,
    ) -> OfficialKnowledgeBaseSyncResult:
        self.requests.append(request)
        return _sync_result(request)


def _sync_result(request: OfficialSourceSyncRequest) -> OfficialKnowledgeBaseSyncResult:
    source = get_official_source(request.source_id)
    fetched_document = FetchedSourceDocument.from_content(
        source_id=source.source_id,
        domain=source.domain,
        source_type=source.source_type,
        registry_title=source.title,
        publisher=source.publisher,
        source_url=source.url,
        resolved_url=source.url,
        fetched_at=_time(),
        content_format=SourceContentFormat.HTML,
        raw_content="<html><body>test</body></html>",
    )
    normalized_document = NormalizedSourceDocument.from_text(
        fetched_document=fetched_document,
        document_title=source.title,
        normalized_text="Normalized content",
    )
    chunk = OfficialSourceChunk.from_text(
        normalized_document=normalized_document,
        chunk_index=0,
        text="Normalized content",
    )
    vector_record = VectorRecord(
        id=f"{request.source_id}::chunk-0001",
        document=chunk.text,
        metadata=ChromaRecordMetadata(
            collection=ChromaCollection.KB_OFFICIAL_DOCS,
            record_kind=VectorRecordKind.OFFICIAL_DOC_CHUNK,
            source_id=chunk.source_id,
            domain=chunk.domain,
            extras={
                "source_url": chunk.source_url,
                "resolved_url": chunk.resolved_url,
                "source_type": chunk.source_type.value,
                "publisher": chunk.publisher,
                "registry_title": chunk.registry_title,
                "document_title": chunk.document_title,
                "chunk_index": chunk.chunk_index,
                "char_count": chunk.char_count,
                "content_hash": chunk.content_hash,
                "chunk_hash": chunk.chunk_hash,
                "fetched_at": chunk.fetched_at.isoformat(),
            },
        ),
        embedding=[0.1, 0.2, 0.3],
    )
    return OfficialKnowledgeBaseSyncResult(
        request=request,
        fetched_document=fetched_document,
        normalized_document=normalized_document,
        chunks=(chunk,),
        embeddings=((0.1, 0.2, 0.3),),
        vector_records=(vector_record,),
        record_ids=(vector_record.id,),
    )


def _time() -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
