"""Controlled Knowledge Base operation contracts."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from creative_coding_assistant.api.domain_experience import (
    build_domain_experience_payload,
)
from creative_coding_assistant.api.knowledge_base import KnowledgeBaseApplication
from creative_coding_assistant.core import Settings
from creative_coding_assistant.rag.sources import approved_official_sources


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
