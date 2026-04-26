import unittest
from io import StringIO
from unittest.mock import patch

from creative_coding_assistant.app import OfficialKnowledgeBaseBatchSyncResult
from creative_coding_assistant.app.sync import SyncFailureMode
from creative_coding_assistant.app.sync_cli import build_sync_parser, main
from creative_coding_assistant.core import Settings


class SyncRunnerCliTests(unittest.TestCase):
    def test_sync_parser_accepts_all_and_option_flags(self) -> None:
        parser = build_sync_parser()

        args = parser.parse_args(
            ["--all", "--continue-on-error", "--summary-format", "json"]
        )

        self.assertTrue(args.all)
        self.assertTrue(args.continue_on_error)
        self.assertEqual(args.summary_format, "json")

    def test_cli_main_passes_selected_sources_to_service(self) -> None:
        settings = Settings(log_level="DEBUG", openai_api_key="sk-test-secret")
        batch_result = OfficialKnowledgeBaseBatchSyncResult(
            source_ids=("three_docs",),
            total_chunks=1,
            total_records=1,
        )
        service = _FakeSyncService(sync_selected_result=batch_result)

        with patch(
            "creative_coding_assistant.app.sync_cli.load_settings",
            return_value=settings,
        ):
            with patch(
                "creative_coding_assistant.app.sync_cli.configure_logging",
            ) as configure_logging:
                with patch(
                    "creative_coding_assistant.app.sync_cli.build_official_kb_sync_service",
                    return_value=service,
                ) as build_service:
                    exit_code = main(["--source-id", "three_docs"])

        self.assertEqual(exit_code, 0)
        configure_logging.assert_called_once_with("DEBUG")
        build_service.assert_called_once_with(
            settings=settings,
            failure_mode=SyncFailureMode.FAIL_FAST,
        )
        self.assertEqual(service.selected_source_ids, [["three_docs"]])

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
                    "creative_coding_assistant.app.sync_cli.build_official_kb_sync_service",
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
        service = _FakeSyncService(sync_all_result=batch_result)

        with patch(
            "creative_coding_assistant.app.sync_cli.load_settings",
            return_value=settings,
        ):
            with patch(
                "creative_coding_assistant.app.sync_cli.configure_logging",
            ):
                with patch(
                    "creative_coding_assistant.app.sync_cli.build_official_kb_sync_service",
                    return_value=service,
                ) as build_service:
                    exit_code = main(["--all", "--continue-on-error"])

        self.assertEqual(exit_code, 1)
        build_service.assert_called_once_with(
            settings=settings,
            failure_mode=SyncFailureMode.CONTINUE,
        )
        self.assertEqual(service.sync_all_call_count, 1)

    def test_cli_main_emits_json_summary_when_requested(self) -> None:
        settings = Settings(log_level="INFO", openai_api_key="sk-test-secret")
        batch_result = OfficialKnowledgeBaseBatchSyncResult(
            source_ids=("three_docs",),
            total_chunks=1,
            total_records=1,
        )
        service = _FakeSyncService(sync_all_result=batch_result)

        with patch(
            "creative_coding_assistant.app.sync_cli.load_settings",
            return_value=settings,
        ):
            with patch(
                "creative_coding_assistant.app.sync_cli.configure_logging",
            ):
                with patch(
                    "creative_coding_assistant.app.sync_cli.build_official_kb_sync_service",
                    return_value=service,
                ):
                    with patch("sys.stdout", new=StringIO()) as stdout:
                        exit_code = main(["--all", "--summary-format", "json"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(service.sync_all_call_count, 1)
        self.assertIn('"source_ids": ["three_docs"]', stdout.getvalue())


class _FakeSyncService:
    def __init__(
        self,
        *,
        sync_all_result: OfficialKnowledgeBaseBatchSyncResult | None = None,
        sync_selected_result: OfficialKnowledgeBaseBatchSyncResult | None = None,
    ) -> None:
        self._sync_all_result = sync_all_result
        self._sync_selected_result = sync_selected_result
        self.sync_all_call_count = 0
        self.selected_source_ids: list[list[str]] = []

    def sync_all_sources(self) -> OfficialKnowledgeBaseBatchSyncResult:
        self.sync_all_call_count += 1
        assert self._sync_all_result is not None
        return self._sync_all_result

    def sync_selected_sources(
        self,
        source_ids: list[str],
    ) -> OfficialKnowledgeBaseBatchSyncResult:
        self.selected_source_ids.append(list(source_ids))
        assert self._sync_selected_result is not None
        return self._sync_selected_result
