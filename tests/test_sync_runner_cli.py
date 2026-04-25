import unittest
from datetime import UTC, datetime
from unittest.mock import patch

from creative_coding_assistant.app import (
    OfficialKnowledgeBaseBatchSyncResult,
    build_official_kb_sync_runner,
    resolve_sync_source_ids,
    sync_official_sources,
)
from creative_coding_assistant.app.sync_cli import main
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


class SyncRunnerCliTests(unittest.TestCase):
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

    def test_sync_official_sources_aggregates_runner_results(self) -> None:
        runner = _FakeRunner()

        result = sync_official_sources(
            source_ids=("three_docs", "p5_reference"),
            runner=runner,
        )

        self.assertEqual(result.source_ids, ("three_docs", "p5_reference"))
        self.assertEqual(result.total_chunks, 2)
        self.assertEqual(result.total_records, 2)
        self.assertEqual(
            tuple(request.source_id for request in runner.requests),
            ("three_docs", "p5_reference"),
        )

    def test_cli_main_passes_selected_sources(self) -> None:
        settings = Settings(log_level="DEBUG", openai_api_key="sk-test-secret")
        batch_result = OfficialKnowledgeBaseBatchSyncResult(
            source_ids=("three_docs",),
            total_chunks=1,
            total_records=1,
        )

        with patch(
            "creative_coding_assistant.app.sync_cli.load_settings",
            return_value=settings,
        ):
            with patch(
                "creative_coding_assistant.app.sync_cli.configure_logging",
            ) as configure_logging:
                with patch(
                    "creative_coding_assistant.app.sync_cli.sync_official_sources",
                    return_value=batch_result,
                ) as sync_sources:
                    exit_code = main(["--source-id", "three_docs"])

        self.assertEqual(exit_code, 0)
        configure_logging.assert_called_once_with("DEBUG")
        sync_sources.assert_called_once_with(
            source_ids=["three_docs"],
            settings=settings,
        )

    def test_cli_main_returns_config_error_code(self) -> None:
        settings = Settings(log_level="INFO", openai_api_key=None)

        with patch(
            "creative_coding_assistant.app.sync_cli.load_settings",
            return_value=settings,
        ):
            with patch(
                "creative_coding_assistant.app.sync_cli.configure_logging",
            ):
                with patch(
                    "creative_coding_assistant.app.sync_cli.sync_official_sources",
                    side_effect=RuntimeError(
                        "Official KB sync requires OpenAI embedding configuration."
                    ),
                ):
                    exit_code = main([])

        self.assertEqual(exit_code, 2)


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
        fetched_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        content_format=SourceContentFormat.HTML,
        raw_content="<html><body>stub</body></html>",
    )
    normalized_document = NormalizedSourceDocument.from_text(
        fetched_document=fetched_document,
        document_title=source.title,
        normalized_text=f"{source.title} normalized text",
    )
    chunk = OfficialSourceChunk.from_text(
        normalized_document=normalized_document,
        chunk_index=0,
        text=f"{source.title} chunk",
    )
    vector_record = VectorRecord(
        id=f"kb_official_docs:official_doc_chunk:v1:{source.source_id}",
        document=chunk.text,
        metadata=ChromaRecordMetadata(
            collection=ChromaCollection.KB_OFFICIAL_DOCS,
            record_kind=VectorRecordKind.OFFICIAL_DOC_CHUNK,
            source_id=source.source_id,
            domain=source.domain,
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


if __name__ == "__main__":
    unittest.main()
