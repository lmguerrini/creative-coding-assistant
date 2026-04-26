import unittest
from datetime import UTC, datetime
from unittest.mock import patch

from creative_coding_assistant.app import (
    OfficialKBSyncService,
    OfficialKnowledgeBaseBatchSyncResult,
    build_official_kb_sync_runner,
    resolve_sync_source_ids,
    sync_official_sources,
)
from creative_coding_assistant.app.sync import SyncFailureMode
from creative_coding_assistant.core import Settings
from creative_coding_assistant.rag.sources import (
    approved_official_sources,
    get_official_source,
)
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


class OfficialKBSyncServiceTests(unittest.TestCase):
    def test_resolve_sync_source_ids_defaults_to_approved_registry_order(self) -> None:
        self.assertEqual(
            resolve_sync_source_ids(None),
            tuple(source.source_id for source in approved_official_sources()),
        )

    def test_resolve_sync_source_ids_dedupes_selected_sources(self) -> None:
        resolved = resolve_sync_source_ids(
            ["three_docs", "p5_reference", "three_docs"]
        )

        self.assertEqual(resolved, ("three_docs", "p5_reference"))

    def test_resolve_sync_source_ids_rejects_unknown_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown official source"):
            resolve_sync_source_ids(["unknown_source"])

    def test_build_sync_runner_requires_embedding_config_before_chroma_setup(
        self,
    ) -> None:
        settings = Settings(openai_api_key=None)

        with patch(
            "creative_coding_assistant.app.sync.create_chroma_client",
            side_effect=AssertionError(
                "Chroma client should not be created when embedding config is missing."
            ),
        ) as create_client:
            with self.assertRaisesRegex(
                RuntimeError,
                "requires OpenAI embedding configuration",
            ):
                build_official_kb_sync_runner(settings=settings)

        create_client.assert_not_called()

    def test_service_sync_all_sources_uses_approved_registry_order(self) -> None:
        runner = _FakeRunner()
        service = OfficialKBSyncService(runner=runner)

        result = service.sync_all_sources()

        self.assertEqual(
            result.source_ids,
            tuple(source.source_id for source in approved_official_sources()),
        )
        self.assertEqual(
            tuple(request.source_id for request in runner.requests),
            result.source_ids,
        )

    def test_service_sync_selected_sources_aggregates_runner_results(self) -> None:
        runner = _FakeRunner()
        service = OfficialKBSyncService(runner=runner)

        result = service.sync_selected_sources(("three_docs", "p5_reference"))

        self.assertEqual(result.source_ids, ("three_docs", "p5_reference"))
        self.assertEqual(result.total_chunks, 2)
        self.assertEqual(result.total_records, 2)
        self.assertEqual(
            tuple(request.source_id for request in runner.requests),
            ("three_docs", "p5_reference"),
        )

    def test_service_continues_after_runner_failure(self) -> None:
        runner = _PartiallyFailingRunner(failing_source_id="p5_reference")
        service = OfficialKBSyncService(
            runner=runner,
            failure_mode=SyncFailureMode.CONTINUE,
        )

        result = service.sync_selected_sources(("three_docs", "p5_reference"))

        self.assertEqual(result.source_ids, ("three_docs", "p5_reference"))
        self.assertEqual(result.failed_source_ids, ("p5_reference",))
        self.assertEqual(result.succeeded_count, 1)
        self.assertEqual(result.failed_count, 1)

    def test_sync_official_sources_wrapper_aggregates_runner_results(self) -> None:
        runner = _FakeRunner()

        result = sync_official_sources(
            source_ids=("three_docs", "p5_reference"),
            runner=runner,
        )

        self.assertEqual(result.source_ids, ("three_docs", "p5_reference"))
        self.assertEqual(result.total_chunks, 2)
        self.assertEqual(result.total_records, 2)

    def test_sync_batch_result_exposes_structured_summary(self) -> None:
        result = OfficialKnowledgeBaseBatchSyncResult(
            source_ids=("three_docs", "p5_reference"),
            failed_source_ids=("p5_reference",),
            total_chunks=1,
            total_records=1,
        )

        self.assertEqual(
            result.summary_payload(),
            {
                "source_ids": ["three_docs", "p5_reference"],
                "succeeded_source_ids": [],
                "failed_source_ids": ["p5_reference"],
                "total_chunks": 1,
                "total_records": 1,
            },
        )


class _FakeRunner:
    def __init__(self) -> None:
        self.requests: list[OfficialSourceSyncRequest] = []

    def run(
        self,
        request: OfficialSourceSyncRequest,
    ) -> OfficialKnowledgeBaseSyncResult:
        self.requests.append(request)
        return _sync_result(request)


class _PartiallyFailingRunner(_FakeRunner):
    def __init__(self, *, failing_source_id: str) -> None:
        super().__init__()
        self._failing_source_id = failing_source_id

    def run(
        self,
        request: OfficialSourceSyncRequest,
    ) -> OfficialKnowledgeBaseSyncResult:
        self.requests.append(request)
        if request.source_id == self._failing_source_id:
            raise RuntimeError(f"Sync failed for {request.source_id}")
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
