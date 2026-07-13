"""Explicit RAGAs dashboard action contracts."""

from __future__ import annotations

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from creative_coding_assistant.api.evaluation import EvaluationApplication
from creative_coding_assistant.core import Settings
from creative_coding_assistant.eval import RagasDependencyError


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
        self.assertEqual(
            run.call_args.kwargs["metric_names"],
            (
                "context_precision",
                "faithfulness",
                "answer_relevancy",
                "context_relevancy",
            ),
        )
        self.assertIn("sanitized_public", run.call_args.kwargs["output_path"].name)
        self.assertEqual(len(run.call_args.kwargs["run_id"]), 32)

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

    def test_approved_fixture_returns_safe_scores_and_case_evidence(self) -> None:
        app = EvaluationApplication(settings_factory=lambda: Settings(_env_file=None))
        headers: dict[str, object] = {}
        payload = json.dumps({
            "dryRun": True,
            "allowProviderCalls": False,
            "approvedDataset": "redacted_public",
        }).encode()
        row = SimpleNamespace(
            sample_id="public-sample-1",
            metrics={"context_precision": 0.8, "faithfulness": None},
            metric_errors={"faithfulness": "metric unavailable"},
            source_ids=("kb-source",),
            domains=("creative_coding",),
        )
        result = SimpleNamespace(
            run_id="run-safe",
            metrics=("context_precision", "faithfulness", "answer_relevancy"),
            result_rows=(row,),
            total_samples=4,
            eligible_samples=4,
            skipped_samples=0,
            metric_failures=1,
            dry_run=True,
            provider_calls_allowed=False,
            cost_warning="Dry run: evaluator providers were not called.",
            manifest=SimpleNamespace(
                dataset=SimpleNamespace(dataset_id="private-run-id"),
                evaluated_at=__import__("datetime").datetime(2026, 7, 13),
            ),
        )

        with patch(
            "creative_coding_assistant.api.evaluation.run_ragas_live_eval",
            return_value=result,
        ) as run:
            body = b"".join(app(_environ(payload), _capture_response(headers)))

        response = json.loads(body)
        self.assertEqual(headers["status"], "200 OK")
        self.assertEqual(response["datasetId"], "redacted_public")
        self.assertEqual(response["privacyClass"], "committed_redacted_public")
        self.assertEqual(response["metricScores"], {"context_precision": 0.8})
        self.assertEqual(response["caseResults"][0]["sampleId"], "public-sample-1")
        self.assertNotIn("prompt", json.dumps(response).lower())
        self.assertTrue(str(run.call_args.kwargs["input_path"]).endswith(
            "demo/evaluation/redacted_live_session_ragas_latest4.jsonl"
        ))

    def test_missing_live_provider_credentials_is_blocked_not_failed(self) -> None:
        app = EvaluationApplication(settings_factory=lambda: Settings(_env_file=None))
        headers: dict[str, object] = {}
        payload = json.dumps({
            "dryRun": False,
            "allowProviderCalls": True,
            "approvedDataset": "sanitized_public",
        }).encode()

        body = b"".join(app(_environ(payload), _capture_response(headers)))

        response = json.loads(body)
        self.assertEqual(headers["status"], "503 Service Unavailable")
        self.assertEqual(response["error"], "blocked_by_execution_environment")
        self.assertEqual(response["details"]["status"], "BLOCKED_BY_EXECUTION_ENVIRONMENT")

    def test_missing_ragas_dependency_is_blocked_not_failed(self) -> None:
        app = EvaluationApplication(settings_factory=lambda: Settings(_env_file=None))
        headers: dict[str, object] = {}
        payload = json.dumps({"dryRun": True, "allowProviderCalls": False}).encode()

        with patch(
            "creative_coding_assistant.api.evaluation.run_ragas_live_eval",
            side_effect=RagasDependencyError("missing"),
        ):
            body = b"".join(app(_environ(payload), _capture_response(headers)))

        response = json.loads(body)
        self.assertEqual(response["error"], "blocked_by_execution_environment")
        self.assertIn("BLOCKED_BY_EXECUTION_ENVIRONMENT", response["message"])

    def test_unexpected_evaluator_error_is_not_mislabeled_as_environment_block(self) -> None:
        app = EvaluationApplication(settings_factory=lambda: Settings(_env_file=None))
        headers: dict[str, object] = {}
        payload = json.dumps({"dryRun": True, "allowProviderCalls": False}).encode()

        with patch(
            "creative_coding_assistant.api.evaluation.run_ragas_live_eval",
            side_effect=RuntimeError("invalid evaluator result shape"),
        ):
            body = b"".join(app(_environ(payload), _capture_response(headers)))

        response = json.loads(body)
        self.assertEqual(headers["status"], "500 Internal Server Error")
        self.assertEqual(response["error"], "evaluation_failed")
        self.assertNotIn("BLOCKED_BY_EXECUTION_ENVIRONMENT", response["message"])


def _capture_response(target: dict[str, object]):
    def start_response(status, headers, exc_info=None):
        del exc_info
        target["status"] = status
        target["headers"] = headers

    return start_response


def _environ(payload: bytes) -> dict[str, object]:
    return {
        "PATH_INFO": "/api/evaluation/run",
        "REQUEST_METHOD": "POST",
        "CONTENT_LENGTH": str(len(payload)),
        "wsgi.input": io.BytesIO(payload),
    }
