"""Explicit RAGAs dashboard action contracts."""

from __future__ import annotations

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from creative_coding_assistant.api.evaluation import EvaluationApplication
from creative_coding_assistant.core import Settings


class V97EvaluationApiTests(unittest.TestCase):
    def test_dry_run_is_an_explicit_local_action_and_reports_no_fabricated_score(self) -> None:
        app = EvaluationApplication(settings_factory=lambda: Settings(_env_file=None))
        headers: dict[str, object] = {}
        payload = json.dumps({"dryRun": True, "allowProviderCalls": False}).encode()
        result = SimpleNamespace(
            run_id="run-1",
            metrics=("context_precision",),
            result_rows=(),
            metric_failures=0,
            dry_run=True,
            provider_calls_allowed=False,
            cost_warning="Dry run: evaluator providers were not called.",
            manifest=SimpleNamespace(
                dataset=SimpleNamespace(dataset_id="dataset-1"),
                evaluated_at=__import__("datetime").datetime(2026, 7, 11),
            ),
        )

        with patch(
            "creative_coding_assistant.api.evaluation.run_ragas_live_eval",
            return_value=result,
        ) as run:
            body = b"".join(
                app(
                    {
                        "PATH_INFO": "/api/evaluation/run",
                        "REQUEST_METHOD": "POST",
                        "CONTENT_LENGTH": str(len(payload)),
                        "wsgi.input": io.BytesIO(payload),
                    },
                    _capture_response(headers),
                )
            )

        response = json.loads(body)
        self.assertEqual(headers["status"], "200 OK")
        self.assertTrue(response["dryRun"])
        self.assertFalse(response["providerCallsAllowed"])
        self.assertEqual(response["resultRows"], 0)
        self.assertNotIn("score", response)
        self.assertTrue(run.call_args.kwargs["dry_run"])
        self.assertFalse(run.call_args.kwargs["allow_provider_calls"])

    def test_live_evaluation_requires_explicit_provider_authorization(self) -> None:
        app = EvaluationApplication(settings_factory=lambda: Settings(_env_file=None))
        headers: dict[str, object] = {}
        payload = json.dumps({"dryRun": False, "allowProviderCalls": False}).encode()

        body = b"".join(
            app(
                {
                    "PATH_INFO": "/api/evaluation/run",
                    "REQUEST_METHOD": "POST",
                    "CONTENT_LENGTH": str(len(payload)),
                    "wsgi.input": io.BytesIO(payload),
                },
                _capture_response(headers),
            )
        )

        response = json.loads(body)
        self.assertEqual(headers["status"], "400 Bad Request")
        self.assertEqual(response["error"], "provider_evaluation_requires_opt_in")


def _capture_response(target: dict[str, object]):
    def start_response(status, headers, exc_info=None):
        del exc_info
        target["status"] = status
        target["headers"] = headers

    return start_response
