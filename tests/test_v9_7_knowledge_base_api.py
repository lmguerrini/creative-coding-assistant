"""Controlled Knowledge Base operation contracts."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from creative_coding_assistant.api.domain_experience import (
    build_domain_experience_payload,
)
from creative_coding_assistant.api.knowledge_base import (
    KnowledgeBaseApplication,
    _check_official_sources,
)
from creative_coding_assistant.core import Settings
from creative_coding_assistant.rag.sources import approved_official_sources
from creative_coding_assistant.rag.sync import TransportResponse


class V97KnowledgeBaseApiTests(unittest.TestCase):
    def test_inventory_exposes_each_approved_source_without_claiming_upstream_freshness(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            payload = build_domain_experience_payload(
                chroma_sqlite_path=Path(temporary_directory) / "chroma.sqlite3"
            )

        sources = payload["knowledgeBase"]["sources"]
        self.assertEqual(len(sources), len(approved_official_sources()))
        self.assertEqual(sources[0]["id"], approved_official_sources()[0].source_id)
        self.assertFalse(sources[0]["indexed"])
        self.assertEqual(sources[0]["health"], "registered_only")
        self.assertIn("do not confirm", sources[0]["freshnessLimitation"])

    def test_check_fetches_only_selected_official_sources_without_constructing_sync(
        self,
    ) -> None:
        calls: list[object] = []
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = Settings(
                _env_file=None,
                chroma_persist_dir=Path(temporary_directory) / "chroma",
            )
            app = KnowledgeBaseApplication(
                settings_factory=lambda: settings,
                sync_service_factory=lambda **kwargs: calls.append(kwargs),
                source_check_fn=lambda source_ids, local_fingerprints: [
                    {
                        "sourceId": source_ids[0],
                        "fingerprint": "a" * 64,
                        "localFingerprint": local_fingerprints.get(source_ids[0]),
                        "changeStatus": "new",
                    }
                ],
            )
            headers: dict[str, object] = {}
            response = _call(app, {"action": "check", "sourceIds": ["three_docs"]}, headers)

        self.assertEqual(headers["status"], "200 OK")
        self.assertEqual(response["status"], "review_ready")
        self.assertEqual(response["sourceIds"], ["three_docs"])
        self.assertEqual(calls, [])
        self.assertEqual(response["sourceChanges"][0]["changeStatus"], "new")
        self.assertIn("Official source content was checked", response["detail"])

    def test_check_reports_unavailable_sources_without_failing_the_selected_batch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = Settings(
                _env_file=None,
                chroma_persist_dir=Path(temporary_directory) / "chroma",
            )
            app = KnowledgeBaseApplication(
                settings_factory=lambda: settings,
                source_check_fn=lambda source_ids, _local_fingerprints: [
                    {"sourceId": source_ids[0], "changeStatus": "changed"},
                    {
                        "sourceId": source_ids[1],
                        "changeStatus": "unavailable",
                        "detail": "The official source could not be reached.",
                    },
                ],
            )
            headers: dict[str, object] = {}
            response = _call(
                app,
                {
                    "action": "check",
                    "sourceIds": ["three_docs", "three_manual"],
                },
                headers,
            )

        self.assertEqual(headers["status"], "200 OK")
        self.assertEqual(response["status"], "review_ready_with_unavailable_sources")
        self.assertEqual(response["unavailableSourceIds"], ["three_manual"])
        self.assertIn("could not be reached", response["detail"])

    def test_check_continues_after_an_individual_official_source_fetch_failure(self) -> None:
        with patch(
            "creative_coding_assistant.api.knowledge_base.UrllibSourceTransport",
            return_value=_PartiallyUnavailableTransport(),
        ):
            changes = _check_official_sources(("three_docs", "three_manual"), {})

        self.assertEqual(changes[0]["changeStatus"], "new")
        self.assertEqual(changes[1]["changeStatus"], "unavailable")
        self.assertIn("could not be reached", str(changes[1]["detail"]))

    def test_check_requires_source_selection_before_network_activity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = Settings(
                _env_file=None,
                chroma_persist_dir=Path(temporary_directory) / "chroma",
            )
            app = KnowledgeBaseApplication(settings_factory=lambda: settings)
            headers: dict[str, object] = {}
            response = _call(app, {"action": "check"}, headers)

        self.assertEqual(headers["status"], "400 Bad Request")
        self.assertEqual(response["error"], "knowledge_base_source_selection_required")

    def test_mutating_operation_requires_explicit_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = Settings(
                _env_file=None,
                chroma_persist_dir=Path(temporary_directory) / "chroma",
            )
            app = KnowledgeBaseApplication(settings_factory=lambda: settings)
            headers: dict[str, object] = {}
            response = _call(
                app,
                {"action": "update", "sourceIds": ["three_docs"]},
                headers,
            )

        self.assertEqual(headers["status"], "400 Bad Request")
        self.assertEqual(response["error"], "knowledge_base_confirmation_required")

    def test_mutating_operation_requires_selected_sources_before_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = Settings(
                _env_file=None,
                chroma_persist_dir=Path(temporary_directory) / "chroma",
            )
            app = KnowledgeBaseApplication(settings_factory=lambda: settings)
            headers: dict[str, object] = {}
            response = _call(
                app,
                {"action": "update", "confirmed": True},
                headers,
            )

        self.assertEqual(headers["status"], "400 Bad Request")
        self.assertEqual(response["error"], "knowledge_base_source_selection_required")

    def test_validation_reports_missing_local_records_without_fabricating_retrieval_quality(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = Settings(
                _env_file=None,
                chroma_persist_dir=Path(temporary_directory) / "chroma",
            )
            app = KnowledgeBaseApplication(settings_factory=lambda: settings)
            headers: dict[str, object] = {}
            response = _call(
                app,
                {"action": "validate", "sourceIds": ["three_docs"]},
                headers,
            )

        self.assertEqual(headers["status"], "200 OK")
        self.assertEqual(response["status"], "needs_indexing")
        self.assertEqual(response["missingSourceIds"], ["three_docs"])
        self.assertEqual(
            response["postBuildRetrievalValidation"],
            "deferred_to_explicit_evaluation",
        )

    def test_failed_selected_update_restores_the_previous_valid_local_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            persist_dir = Path(temporary_directory) / "chroma"
            persist_dir.mkdir()
            marker_path = persist_dir / "valid-index-marker.txt"
            marker_path.write_text("valid before update")
            settings = Settings(
                _env_file=None,
                chroma_persist_dir=persist_dir,
            )

            class FailingSyncService:
                def sync_selected_sources(self, source_ids):
                    assert source_ids == ("three_docs",)
                    marker_path.write_text("partial mutation")
                    return SimpleNamespace(failed_count=1)

            app = KnowledgeBaseApplication(
                settings_factory=lambda: settings,
                sync_service_factory=lambda **_kwargs: FailingSyncService(),
            )
            headers: dict[str, object] = {}
            response = _call(
                app,
                {
                    "action": "update",
                    "confirmed": True,
                    "sourceIds": ["three_docs"],
                },
                headers,
            )

            self.assertEqual(headers["status"], "503 Service Unavailable")
            self.assertEqual(response["error"], "knowledge_base_update_failed")
            self.assertIn("prior local index was restored", response["message"])
            self.assertEqual(marker_path.read_text(), "valid before update")


def _call(
    app: KnowledgeBaseApplication,
    payload: dict[str, object],
    headers: dict[str, object],
) -> dict[str, object]:
    body = json.dumps(payload).encode()
    response = b"".join(
        app(
            {
                "PATH_INFO": "/api/knowledge-base",
                "REQUEST_METHOD": "POST",
                "CONTENT_LENGTH": str(len(body)),
                "wsgi.input": io.BytesIO(body),
            },
            _capture_response(headers),
        )
    )
    return json.loads(response)


def _capture_response(target: dict[str, object]):
    def start_response(status, headers, exc_info=None):
        del exc_info
        target["status"] = status
        target["headers"] = headers

    return start_response


class _PartiallyUnavailableTransport:
    def fetch(self, url: str) -> TransportResponse:
        if url == "https://threejs.org/docs/":
            return TransportResponse(
                resolved_url=url,
                status_code=200,
                content_type="text/html",
                content="<html><body>Three.js documentation</body></html>",
            )
        raise RuntimeError("Official source is temporarily unavailable.")
