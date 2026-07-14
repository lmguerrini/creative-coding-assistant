"""Asynchronous current-product evaluation API contracts."""

from __future__ import annotations

import io
import json
import unittest
from threading import Event
from time import monotonic, sleep

from creative_coding_assistant.api.evaluation import (
    EVALUATION_CONTRACT_VERSION,
    EvaluationApplication,
    EvaluationJobRegistry,
    EvaluationJobRegistryFullError,
    EvaluationRunRequest,
)
from creative_coding_assistant.core import Settings
from creative_coding_assistant.eval.current_product import (
    CurrentProductEvaluationBlockedError,
    CurrentProductEvaluationProgress,
)
from creative_coding_assistant.eval.retrieval_demo_pack import (
    build_capstone_retrieval_demo_pack,
)


class V97EvaluationApiTests(unittest.TestCase):
    def test_full_scope_queues_only_the_seven_current_product_cases(self) -> None:
        release = Event()
        captured: dict[str, object] = {}

        def run_job(request, settings, run_id, progress):
            del settings, progress
            captured["request"] = request
            release.wait(timeout=2)
            return {
                "runId": run_id,
                "status": "prepared",
                "benchmarkMode": request.benchmark_mode,
                "scoreOrigin": "unscored",
                "metricScores": {},
                "detail": "Full current-product preflight prepared seven cases.",
            }

        registry = EvaluationJobRegistry(run_job=run_job, max_workers=1)
        app = EvaluationApplication(
            settings_factory=lambda: Settings(_env_file=None),
            registry=registry,
        )

        accepted = _request(
            app,
            "POST",
            payload={
                "benchmarkMode": "current_product",
                "scope": "full",
                "caseIds": [],
                "allowProviderCalls": False,
                "dryRun": True,
                "approvedDataset": "sanitized_public",
            },
        )

        self.assertEqual(accepted["progress"]["totalCases"], 7)
        self.assertEqual(accepted["progress"]["remainingCases"], 7)
        release.set()
        terminal = _poll_terminal(app, accepted["runId"])
        registry.shutdown()

        request = captured["request"]
        self.assertEqual(request.scope, "full")
        self.assertEqual(request.case_ids, ())
        self.assertEqual(terminal["status"], "prepared")
        self.assertEqual(terminal["progress"]["totalCases"], 7)

    def test_post_returns_202_then_get_publishes_terminal_result(self) -> None:
        release = Event()
        captured: dict[str, object] = {}

        def run_job(request, settings, run_id, progress):
            del settings
            captured["request"] = request
            captured["runId"] = run_id
            progress(
                CurrentProductEvaluationProgress(
                    phase="generation",
                    lane="current_product_rag",
                    current_case_id=request.case_ids[0],
                    current_case_label="Selected case",
                    completed_cases=0,
                    total_cases=1,
                    remaining_cases=1,
                    percent=25,
                    execution_state="running",
                    detail="Running selected current case.",
                )
            )
            release.wait(timeout=2)
            return {
                "runId": run_id,
                "status": "completed",
                "benchmarkMode": request.benchmark_mode,
                "scoreOrigin": "unscored",
                "metricScores": {},
                "detail": "Diagnostic subset completed without a promoted score.",
            }

        registry = EvaluationJobRegistry(run_job=run_job, max_workers=1)
        app = EvaluationApplication(
            settings_factory=lambda: Settings(_env_file=None),
            registry=registry,
        )
        case_id = build_capstone_retrieval_demo_pack().scenarios[0].demo_id
        headers: dict[str, object] = {}
        payload = {
            "benchmarkMode": "current_product",
            "scope": "cases",
            "caseIds": [case_id],
            "allowProviderCalls": False,
            "dryRun": True,
            "approvedDataset": "sanitized_public",
        }

        accepted = _request(app, "POST", payload=payload, headers=headers)

        self.assertEqual(headers["status"], "202 Accepted")
        self.assertEqual(accepted["status"], "queued")
        self.assertEqual(accepted["progress"]["phase"], "queued")
        self.assertEqual(accepted["progress"]["totalCases"], 1)
        self.assertNotIn("result", accepted)
        release.set()
        terminal = _poll_terminal(app, accepted["runId"])
        registry.shutdown()

        self.assertEqual(terminal["status"], "completed")
        self.assertEqual(terminal["progress"]["phase"], "completed")
        self.assertEqual(terminal["progress"]["percent"], 100)
        self.assertEqual(terminal["result"]["benchmarkMode"], "current_product")
        request = captured["request"]
        self.assertEqual(request.scope, "cases")
        self.assertEqual(request.case_ids, (case_id,))
        self.assertEqual(captured["runId"], accepted["runId"])

    def test_runner_exception_becomes_terminal_failed_without_result(self) -> None:
        def fail(*args):
            del args
            raise RuntimeError("raw internal detail must not escape")

        registry = EvaluationJobRegistry(run_job=fail)
        app = EvaluationApplication(
            settings_factory=lambda: Settings(_env_file=None),
            registry=registry,
        )
        accepted = _request(
            app,
            "POST",
            payload={"scope": "rag", "dryRun": True},
        )
        terminal = _poll_terminal(app, accepted["runId"])
        registry.shutdown()

        self.assertEqual(terminal["status"], "failed")
        self.assertEqual(terminal["progress"]["executionState"], "failed")
        self.assertNotIn("result", terminal)
        self.assertIn("no score was fabricated", terminal["progress"]["detail"])
        self.assertNotIn("raw internal", json.dumps(terminal))

    def test_environment_boundary_becomes_terminal_blocked(self) -> None:
        def block(*args):
            del args
            raise CurrentProductEvaluationBlockedError(
                "Provider credentials are unavailable."
            )

        registry = EvaluationJobRegistry(run_job=block)
        app = EvaluationApplication(
            settings_factory=lambda: Settings(_env_file=None),
            registry=registry,
        )
        accepted = _request(
            app,
            "POST",
            payload={"scope": "rag", "dryRun": True},
        )
        terminal = _poll_terminal(app, accepted["runId"])
        registry.shutdown()

        self.assertEqual(terminal["status"], "blocked")
        self.assertNotIn("result", terminal)
        self.assertIn("credentials", terminal["progress"]["detail"])

    def test_default_live_job_without_credentials_is_accepted_then_blocked(self) -> None:
        registry = EvaluationJobRegistry()
        app = EvaluationApplication(
            settings_factory=lambda: Settings(_env_file=None, openai_api_key=None),
            registry=registry,
        )

        accepted = _request(
            app,
            "POST",
            payload={
                "benchmarkMode": "current_product",
                "scope": "rag",
                "dryRun": False,
                "allowProviderCalls": True,
            },
        )
        terminal = _poll_terminal(app, accepted["runId"])
        registry.shutdown()

        self.assertEqual(terminal["status"], "blocked")
        self.assertNotIn("result", terminal)
        self.assertIn("credentials", terminal["progress"]["detail"])

    def test_unknown_get_is_404_and_get_is_advertised(self) -> None:
        app = EvaluationApplication(settings_factory=lambda: Settings(_env_file=None))
        headers: dict[str, object] = {}

        response = _request(
            app,
            "GET",
            query="runId=missing",
            headers=headers,
        )

        self.assertEqual(headers["status"], "404 Not Found")
        self.assertEqual(response["error"], "evaluation_run_not_found")
        response_headers = dict(headers["headers"])
        self.assertIn("GET", response_headers["Access-Control-Allow-Methods"])
        self.assertEqual(
            response_headers["X-CCA-Evaluation-Contract-Version"],
            EVALUATION_CONTRACT_VERSION,
        )

    def test_live_run_requires_provider_authorization_before_job_creation(self) -> None:
        app = EvaluationApplication(settings_factory=lambda: Settings(_env_file=None))
        headers: dict[str, object] = {}

        response = _request(
            app,
            "POST",
            payload={
                "benchmarkMode": "current_product",
                "scope": "rag",
                "dryRun": False,
                "allowProviderCalls": False,
            },
            headers=headers,
        )

        self.assertEqual(headers["status"], "400 Bad Request")
        self.assertEqual(response["error"], "provider_evaluation_requires_opt_in")

    def test_current_case_scope_rejects_empty_and_unknown_ids(self) -> None:
        app = EvaluationApplication(settings_factory=lambda: Settings(_env_file=None))
        for case_ids in ([], ["unknown_case"]):
            with self.subTest(case_ids=case_ids):
                headers: dict[str, object] = {}
                response = _request(
                    app,
                    "POST",
                    payload={
                        "benchmarkMode": "current_product",
                        "scope": "cases",
                        "caseIds": case_ids,
                        "dryRun": True,
                    },
                    headers=headers,
                )
                self.assertEqual(headers["status"], "400 Bad Request")
                self.assertEqual(response["error"], "invalid_evaluation_request")

    def test_registry_never_exceeds_bound_when_all_jobs_are_active(self) -> None:
        release = Event()

        def run_job(request, settings, run_id, progress):
            del request, settings, progress
            release.wait(timeout=2)
            return {"runId": run_id, "status": "prepared", "detail": "done"}

        registry = EvaluationJobRegistry(run_job=run_job, max_jobs=1, max_workers=1)
        request = EvaluationRunRequest(scope="rag", dryRun=True)
        registry.submit(request, Settings(_env_file=None))

        with self.assertRaises(EvaluationJobRegistryFullError):
            registry.submit(request, Settings(_env_file=None))

        release.set()
        registry.shutdown()


def _request(
    app,
    method: str,
    *,
    payload: dict[str, object] | None = None,
    query: str = "",
    headers: dict[str, object] | None = None,
) -> dict[str, object]:
    target = headers if headers is not None else {}
    encoded = json.dumps(payload).encode() if payload is not None else b""
    environ = {
        "PATH_INFO": "/api/evaluation/run",
        "QUERY_STRING": query,
        "REQUEST_METHOD": method,
        "CONTENT_LENGTH": str(len(encoded)),
        "wsgi.input": io.BytesIO(encoded),
    }
    body = b"".join(app(environ, _capture_response(target)))
    return json.loads(body) if body else {}


def _poll_terminal(app, run_id: str) -> dict[str, object]:
    deadline = monotonic() + 3
    while monotonic() < deadline:
        snapshot = _request(app, "GET", query=f"runId={run_id}")
        if snapshot["status"] in {"completed", "prepared", "blocked", "failed"}:
            return snapshot
        sleep(0.01)
    raise AssertionError("evaluation job did not reach a terminal status")


def _capture_response(target: dict[str, object]):
    def start_response(status, headers, exc_info=None):
        del exc_info
        target["status"] = status
        target["headers"] = headers

    return start_response


if __name__ == "__main__":
    unittest.main()
