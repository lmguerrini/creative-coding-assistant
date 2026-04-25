import unittest
from datetime import UTC, datetime
from io import StringIO
from unittest.mock import patch

from creative_coding_assistant.app import (
    OfficialKnowledgeBaseBatchSyncResult,
    build_official_kb_sync_runner,
    resolve_sync_source_ids,
    sync_official_sources,
)
from creative_coding_assistant.app.sync import SyncFailureMode
from creative_coding_assistant.app.sync_cli import build_sync_parser, main
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

    def test_sync_official_sources_continues_after_runner_failure(self) -> None:
        runner = _PartiallyFailingRunner(failing_source_id="p5_reference")

        result = sync_official_sources(
            source_ids=("three_docs", "p5_reference"),
            runner=runner,
            failure_mode=SyncFailureMode.CONTINUE,
        )

        self.assertEqual(result.source_ids, ("three_docs", "p5_reference"))
        self.assertEqual(result.failed_source_ids, ("p5_reference",))
        self.assertEqual(result.succeeded_count, 1)
        self.assertEqual(result.failed_count, 1)

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

    def test_sync_parser_accepts_all_and_option_flags(self) -> None:
        parser = build_sync_parser()

        args = parser.parse_args(
            ["--all", "--continue-on-error", "--summary-format", "json"]
        )

        self.assertTrue(args.all)
        self.assertTrue(args.continue_on_error)
        self.assertEqual(args.summary_format, "json")

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
            failure_mode=SyncFailureMode.FAIL_FAST,
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

    def test_cli_main_supports_all_and_continue_mode(self) -> None:
        settings = Settings(log_level="INFO", openai_api_key="sk-test-secret")
        batch_result = OfficialKnowledgeBaseBatchSyncResult(
            source_ids=("three_docs", "p5_reference"),
            failed_source_ids=("p5_reference",),
            total_chunks=1,
            total_records=1,
        )

        with patch(
            "creative_coding_assistant.app.sync_cli.load_settings",
            return_value=settings,
        ):
            with patch(
                "creative_coding_assistant.app.sync_cli.configure_logging",
            ):
                with patch(
                    "creative_coding_assistant.app.sync_cli.sync_official_sources",
                    return_value=batch_result,
                ) as sync_sources:
                    exit_code = main(["--all", "--continue-on-error"])

        self.assertEqual(exit_code, 1)
        sync_sources.assert_called_once_with(
            source_ids=None,
            settings=settings,
            failure_mode=SyncFailureMode.CONTINUE,
        )

    def test_cli_main_emits_json_summary_when_requested(self) -> None:
        settings = Settings(log_level="INFO", openai_api_key="sk-test-secret")
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
            ):
                with patch(
                    "creative_coding_assistant.app.sync_cli.sync_official_sources",
                    return_value=batch_result,
                ):
                    with patch("sys.stdout", new=StringIO()) as stdout:
                        exit_code = main(["--all", "--summary-format", "json"])

        self.assertEqual(exit_code, 0)
        self.assertIn('"source_ids": ["three_docs"]', stdout.getvalue())


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
