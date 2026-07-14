"""Asynchronous current-product and explicit historical evaluation API."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from http import HTTPStatus
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Literal, Protocol
from urllib.parse import parse_qs
from uuid import uuid4

from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from creative_coding_assistant.api.contracts import (
    ApiRequestBodyError,
    StartResponse,
    empty_response,
    error_response,
    json_response,
    read_json_body,
    request_id_from_environ,
)
from creative_coding_assistant.api.cors import resolve_cors_allow_origin
from creative_coding_assistant.core.config import Settings, load_settings
from creative_coding_assistant.eval.current_product import (
    CurrentProductEvaluationBlockedError,
    CurrentProductEvaluationProgress,
    CurrentProductEvaluationResult,
    CurrentProductEvaluationRunner,
    CurrentProductRunOptions,
    ProgressCallback,
    build_safe_current_product_evidence,
)
from creative_coding_assistant.eval.ragas_runner import (
    RagasDependencyError,
    RagasEvaluatorConfig,
    RagasProviderCostBoundaryError,
    run_ragas_live_eval,
)
from creative_coding_assistant.eval.retrieval_demo_pack import (
    build_capstone_retrieval_demo_pack,
)

EVALUATION_CONTRACT_VERSION = "evaluation.v3"
EVALUATION_CONTRACT_HEADER = "X-CCA-Evaluation-Contract-Version"
DEFAULT_EVALUATION_PATH = "/api/evaluation/run"
EVALUATION_METHODS = "GET, POST, OPTIONS"
MAX_EVALUATION_REQUEST_BYTES = 16 * 1024
DEFAULT_MAX_EVALUATION_JOBS = 32

EvaluationJobStatus = Literal[
    "queued",
    "running",
    "completed",
    "prepared",
    "blocked",
    "failed",
]


class EvaluationRunRequest(BaseModel):
    """Bounded current-product request; historical fixtures are opt-in only."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    benchmark_mode: Literal["current_product", "historical_fixture"] = Field(
        default="current_product",
        alias="benchmarkMode",
    )
    scope: Literal[
        "full",
        "rag",
        "cases",
        "creative_artifact",
        "workflow",
        "product_reliability",
    ] = "full"
    case_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        alias="caseIds",
        max_length=100,
    )
    allow_provider_calls: bool = Field(default=False, alias="allowProviderCalls")
    dry_run: bool = Field(default=True, alias="dryRun")
    approved_dataset: Literal["sanitized_public", "redacted_public"] = Field(
        default="sanitized_public",
        alias="approvedDataset",
    )

    @field_validator("case_ids")
    @classmethod
    def normalize_case_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.strip() for item in value if item.strip()))

    @model_validator(mode="after")
    def validate_current_case_selection(self) -> EvaluationRunRequest:
        if self.benchmark_mode != "current_product" or self.scope != "cases":
            return self
        if not self.case_ids:
            raise ValueError("Current-product case scope requires at least one case ID.")
        canonical_ids = {
            item.demo_id for item in build_capstone_retrieval_demo_pack().scenarios
        }
        if any(case_id not in canonical_ids for case_id in self.case_ids):
            raise ValueError("Current-product case scope contains an unknown case ID.")
        return self


APPROVED_EVALUATION_DATASETS = {
    "sanitized_public": {
        "path": Path("demo/evaluation/sanitized_ragas_live_sessions.jsonl"),
        "version": "sanitized-ragas.v1",
        "privacy_class": "committed_synthetic_public",
    },
    "redacted_public": {
        "path": Path("demo/evaluation/redacted_live_session_ragas_latest4.jsonl"),
        "version": "redacted-live-latest4.v1",
        "privacy_class": "committed_redacted_public",
    },
}


class EvaluationJobRunner(Protocol):
    def __call__(
        self,
        request: EvaluationRunRequest,
        settings: Settings,
        run_id: str,
        progress: ProgressCallback,
    ) -> Mapping[str, object] | CurrentProductEvaluationResult:
        """Execute one job and return a terminal result."""


class EvaluationJobRegistryFullError(RuntimeError):
    """Raised when every bounded registry slot is occupied by an active job."""


class EvaluationJobRegistry:
    """Thread-safe, finite async job registry for short local evaluation runs."""

    def __init__(
        self,
        *,
        run_job: EvaluationJobRunner | None = None,
        max_jobs: int = DEFAULT_MAX_EVALUATION_JOBS,
        max_workers: int = 1,
    ) -> None:
        if max_jobs < 1:
            raise ValueError("Evaluation registry max_jobs must be at least one.")
        self._run_job = run_job or _run_evaluation_job
        self._max_jobs = max_jobs
        self._jobs: OrderedDict[str, dict[str, object]] = OrderedDict()
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="cca-evaluation",
        )

    def submit(
        self,
        request: EvaluationRunRequest,
        settings: Settings,
    ) -> dict[str, object]:
        run_id = uuid4().hex
        selected_count = _selected_case_count(request)
        progress = CurrentProductEvaluationProgress(
            phase="queued",
            lane=_lane_for_request(request),
            current_case_id=None,
            current_case_label="",
            completed_cases=0,
            total_cases=selected_count,
            remaining_cases=selected_count,
            percent=0,
            execution_state="queued",
            detail="Evaluation job accepted and queued.",
        ).model_dump(mode="json", by_alias=True)
        initial = {
            "runId": run_id,
            "status": "queued",
            "progress": progress,
        }
        with self._lock:
            self._prune_terminal_locked(reserve=1)
            if len(self._jobs) >= self._max_jobs:
                raise EvaluationJobRegistryFullError(
                    "Evaluation registry is full of active jobs."
                )
            self._jobs[run_id] = deepcopy(initial)
        self._executor.submit(self._execute, run_id, request, settings)
        return initial

    def get(self, run_id: str) -> dict[str, object] | None:
        with self._lock:
            snapshot = self._jobs.get(run_id)
            return deepcopy(snapshot) if snapshot is not None else None

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _execute(
        self,
        run_id: str,
        request: EvaluationRunRequest,
        settings: Settings,
    ) -> None:
        self._set_status(run_id, "running", execution_state="running")

        def update(progress: CurrentProductEvaluationProgress) -> None:
            with self._lock:
                job = self._jobs.get(run_id)
                if job is not None:
                    job["progress"] = progress.model_dump(mode="json", by_alias=True)

        try:
            result = self._run_job(request, settings, run_id, update)
            payload = (
                build_safe_current_product_evidence(result)
                if isinstance(result, CurrentProductEvaluationResult)
                else dict(result)
            )
            status = str(payload.get("status", "failed"))
            if status not in {"completed", "prepared", "blocked", "failed"}:
                raise ValueError("Evaluation runner returned a non-terminal status.")
            with self._lock:
                job = self._jobs.get(run_id)
                if job is None:
                    return
                job["status"] = status
                job["result"] = payload
                job["progress"] = _terminal_progress(
                    job["progress"],
                    status=status,
                    detail=str(payload.get("detail", "Evaluation job finished.")),
                )
                self._prune_terminal_locked()
        except CurrentProductEvaluationBlockedError as exc:
            self._finish_without_result(run_id, status="blocked", detail=str(exc))
        except (
            RagasDependencyError,
            RagasProviderCostBoundaryError,
            APIConnectionError,
            APITimeoutError,
            AuthenticationError,
            RateLimitError,
            ConnectionError,
            TimeoutError,
            OSError,
        ):
            self._finish_without_result(
                run_id,
                status="blocked",
                detail=(
                    "BLOCKED_BY_EXECUTION_ENVIRONMENT: evaluation dependencies or "
                    "provider connectivity are unavailable."
                ),
            )
        except Exception:
            self._finish_without_result(
                run_id,
                status="failed",
                detail=(
                    "Evaluation failed before a complete metric contract was produced; "
                    "no score was fabricated."
                ),
            )

    def _finish_without_result(
        self,
        run_id: str,
        *,
        status: Literal["blocked", "failed"],
        detail: str,
    ) -> None:
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None:
                return
            job["status"] = status
            job.pop("result", None)
            job["progress"] = _terminal_progress(
                job["progress"],
                status=status,
                detail=detail,
            )
            self._prune_terminal_locked()

    def _set_status(
        self,
        run_id: str,
        status: EvaluationJobStatus,
        *,
        execution_state: str,
    ) -> None:
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None:
                return
            job["status"] = status
            raw_progress = job.get("progress")
            if isinstance(raw_progress, dict):
                raw_progress["phase"] = status
                raw_progress["executionState"] = execution_state

    def _prune_terminal_locked(self, *, reserve: int = 0) -> None:
        terminal = {"completed", "prepared", "blocked", "failed"}
        while len(self._jobs) + reserve > self._max_jobs:
            removable = next(
                (
                    job_id
                    for job_id, snapshot in self._jobs.items()
                    if snapshot.get("status") in terminal
                ),
                None,
            )
            if removable is None:
                return
            del self._jobs[removable]


class EvaluationApplication:
    """Accept jobs immediately and expose polling snapshots by run ID."""

    def __init__(
        self,
        *,
        settings_factory: Callable[[], Settings] = load_settings,
        registry: EvaluationJobRegistry | None = None,
        path: str = DEFAULT_EVALUATION_PATH,
    ) -> None:
        self._settings_factory = settings_factory
        self._registry = registry or EvaluationJobRegistry()
        self._path = path

    def __call__(
        self,
        environ: dict[str, Any],
        start_response: StartResponse,
    ) -> Iterable[bytes]:
        request_id = request_id_from_environ(environ)
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        settings = self._settings_factory()
        allow_origin = resolve_cors_allow_origin(environ, settings=settings)
        if str(environ.get("PATH_INFO", "")) != self._path:
            return self._error(
                start_response,
                HTTPStatus.NOT_FOUND,
                "not_found",
                "Evaluation route was not found.",
                request_id,
                allow_origin,
            )
        if method == "OPTIONS":
            return empty_response(
                start_response,
                HTTPStatus.NO_CONTENT,
                request_id=request_id,
                allow_methods=EVALUATION_METHODS,
                allow_origin=allow_origin,
                extra_headers=self._headers(),
            )
        if method == "GET":
            query = parse_qs(str(environ.get("QUERY_STRING", "")))
            run_id = (query.get("runId") or [""])[0].strip()
            if not run_id:
                return self._error(
                    start_response,
                    HTTPStatus.BAD_REQUEST,
                    "evaluation_run_id_required",
                    "A runId query parameter is required.",
                    request_id,
                    allow_origin,
                )
            snapshot = self._registry.get(run_id)
            if snapshot is None:
                return self._error(
                    start_response,
                    HTTPStatus.NOT_FOUND,
                    "evaluation_run_not_found",
                    "Evaluation run was not found or has expired.",
                    request_id,
                    allow_origin,
                )
            return json_response(
                start_response,
                HTTPStatus.OK,
                snapshot,
                request_id=request_id,
                allow_methods=EVALUATION_METHODS,
                allow_origin=allow_origin,
                extra_headers=self._headers(),
            )
        if method != "POST":
            return self._error(
                start_response,
                HTTPStatus.METHOD_NOT_ALLOWED,
                "method_not_allowed",
                "Evaluation accepts GET, POST, and OPTIONS.",
                request_id,
                allow_origin,
            )
        try:
            request = EvaluationRunRequest.model_validate(
                read_json_body(environ, max_bytes=MAX_EVALUATION_REQUEST_BYTES)
            )
        except ApiRequestBodyError as exc:
            return self._error(
                start_response,
                exc.status,
                exc.code,
                exc.message,
                request_id,
                allow_origin,
            )
        except ValidationError:
            return self._error(
                start_response,
                HTTPStatus.BAD_REQUEST,
                "invalid_evaluation_request",
                "Evaluation options were invalid.",
                request_id,
                allow_origin,
            )
        if not request.dry_run and not request.allow_provider_calls:
            return self._error(
                start_response,
                HTTPStatus.BAD_REQUEST,
                "provider_evaluation_requires_opt_in",
                "A live evaluation requires explicit provider-call authorization.",
                request_id,
                allow_origin,
            )
        try:
            snapshot = self._registry.submit(request, settings)
        except EvaluationJobRegistryFullError:
            return self._error(
                start_response,
                HTTPStatus.SERVICE_UNAVAILABLE,
                "evaluation_registry_full",
                "Evaluation capacity is temporarily full; retry after an active run finishes.",
                request_id,
                allow_origin,
            )
        return json_response(
            start_response,
            HTTPStatus.ACCEPTED,
            snapshot,
            request_id=request_id,
            allow_methods=EVALUATION_METHODS,
            allow_origin=allow_origin,
            extra_headers=self._headers(),
        )

    def _error(
        self,
        start_response: StartResponse,
        status: HTTPStatus,
        code: str,
        message: str,
        request_id: str,
        allow_origin: str | None,
    ) -> Iterable[bytes]:
        return error_response(
            start_response,
            status,
            error=code,
            message=message,
            request_id=request_id,
            allow_methods=EVALUATION_METHODS,
            allow_origin=allow_origin,
            extra_headers=[
                ("Allow", EVALUATION_METHODS),
                *self._headers(),
            ],
        )

    @staticmethod
    def _headers() -> list[tuple[str, str]]:
        return [(EVALUATION_CONTRACT_HEADER, EVALUATION_CONTRACT_VERSION)]


def _run_evaluation_job(
    request: EvaluationRunRequest,
    settings: Settings,
    run_id: str,
    progress: ProgressCallback,
) -> Mapping[str, object] | CurrentProductEvaluationResult:
    if request.benchmark_mode == "current_product":
        return CurrentProductEvaluationRunner(settings=settings).run(
            run_id=run_id,
            options=CurrentProductRunOptions(
                scope=request.scope,
                case_ids=request.case_ids,
                allow_provider_calls=request.allow_provider_calls,
                dry_run=request.dry_run,
            ),
            progress=progress,
        )
    return _run_historical_fixture(request, settings, run_id, progress)


def _run_historical_fixture(
    request: EvaluationRunRequest,
    settings: Settings,
    run_id: str,
    progress: ProgressCallback,
) -> Mapping[str, object]:
    if not request.dry_run and not settings.has_openai_api_key:
        raise CurrentProductEvaluationBlockedError(
            "Historical evaluator provider credentials are unavailable."
        )
    contract = APPROVED_EVALUATION_DATASETS[request.approved_dataset]
    repository_root = Path(__file__).resolve().parents[3]
    input_path = repository_root / contract["path"]
    metrics = (
        "context_precision",
        "faithfulness",
        "answer_relevancy",
        "context_relevancy",
    )
    progress(
        CurrentProductEvaluationProgress(
            phase="evaluation",
            lane="historical_fixture",
            current_case_id=None,
            current_case_label="",
            completed_cases=0,
            total_cases=4,
            remaining_cases=4,
            percent=10,
            execution_state="prepared" if request.dry_run else "running",
            detail="Preparing the explicitly selected historical public fixture.",
        )
    )
    output_path = settings.eval_ragas_results_path.with_name(
        f"dashboard-{request.approved_dataset}-{run_id}-ragas-results.jsonl"
    )
    started_at = perf_counter()
    result = run_ragas_live_eval(
        input_path=input_path,
        output_path=output_path,
        metric_names=metrics,
        evaluator_config=RagasEvaluatorConfig(
            model=settings.eval_ragas_model,
            embedding_model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
            timeout_seconds=settings.eval_ragas_timeout_seconds,
            max_retries=settings.eval_ragas_max_retries,
            max_workers=settings.eval_ragas_max_workers,
        ),
        dry_run=request.dry_run,
        allow_provider_calls=request.allow_provider_calls,
        run_id=run_id,
    )
    metric_scores = _aggregate_metric_scores(result.result_rows, result.metrics)
    complete = (
        not request.dry_run
        and result.result_rows
        and result.metric_failures == 0
        and all(
            row.metrics.get(metric) is not None
            for row in result.result_rows
            for metric in metrics
        )
    )
    return {
        "schemaVersion": "historical-ragas-evidence.v1",
        "benchmarkMode": "historical_fixture",
        "scoreOrigin": "historical_fixture" if complete else "unscored",
        "scope": request.scope,
        "runId": result.run_id,
        "status": "prepared" if result.dry_run else "completed",
        "datasetId": request.approved_dataset,
        "datasetVersion": contract["version"],
        "datasetFingerprint": _file_fingerprint(input_path),
        "privacyClass": contract["privacy_class"],
        "metrics": list(result.metrics),
        "metricScores": metric_scores,
        "retrievalScore": (
            sum(metric_scores.values()) / len(metrics) if complete else None
        ),
        "resultRows": len(result.result_rows),
        "totalSamples": result.total_samples,
        "eligibleSamples": result.eligible_samples,
        "skippedSamples": result.skipped_samples,
        "metricFailures": result.metric_failures,
        "provider": "OpenAI evaluator" if not result.dry_run else None,
        "evaluator": (
            f"RAGAS OpenAI evaluator ({settings.eval_ragas_model})"
            if not result.dry_run
            else None
        ),
        "model": settings.eval_ragas_model if not result.dry_run else None,
        "embeddingModel": (
            settings.openai_embedding_model if not result.dry_run else None
        ),
        "ragasVersion": _package_version("ragas"),
        "metricContract": "ragas-historical-fixture.v1",
        "durationMs": round((perf_counter() - started_at) * 1000),
        "evaluatedAt": result.manifest.evaluated_at.isoformat(),
        "timestamp": result.manifest.evaluated_at.isoformat(),
        "detail": result.cost_warning,
        "caseResults": [
            {
                "sampleId": row.sample_id,
                "metrics": dict(row.metrics),
                "metricErrors": dict(row.metric_errors),
                "sourceIds": list(row.source_ids),
                "domains": list(row.domains),
            }
            for row in result.result_rows
        ],
    }


def _terminal_progress(
    raw_progress: object,
    *,
    status: str,
    detail: str,
) -> dict[str, object]:
    progress = dict(raw_progress) if isinstance(raw_progress, dict) else {}
    total = int(progress.get("totalCases", 0) or 0)
    completed = int(progress.get("completedCases", 0) or 0)
    if status in {"completed", "prepared"}:
        completed = total if status == "completed" else completed
    progress.update(
        {
            "phase": status,
            "lane": str(progress.get("lane", "evaluation")),
            "currentCaseId": None,
            "currentCaseLabel": "",
            "completedCases": completed,
            "totalCases": total,
            "remainingCases": max(total - completed, 0),
            "percent": 100 if status == "completed" else progress.get("percent"),
            "executionState": status,
            "detail": detail,
        }
    )
    return progress


def _selected_case_count(request: EvaluationRunRequest) -> int:
    if request.scope in {"full", "rag"}:
        return 7
    if request.scope == "cases":
        return len(request.case_ids)
    return 0


def _lane_for_request(request: EvaluationRunRequest) -> str:
    if request.benchmark_mode == "historical_fixture":
        return "historical_fixture"
    return "current_product_rag"


def _aggregate_metric_scores(result_rows: Any, metrics: Sequence[str]) -> dict[str, float]:
    aggregates: dict[str, float] = {}
    for metric in metrics:
        values = [
            row.metrics.get(metric)
            for row in result_rows
            if isinstance(row.metrics.get(metric), (int, float))
        ]
        if values:
            aggregates[metric] = sum(values) / len(values)
    return aggregates


def _file_fingerprint(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def create_evaluation_app(
    *,
    settings_factory: Callable[[], Settings] = load_settings,
    registry: EvaluationJobRegistry | None = None,
) -> EvaluationApplication:
    return EvaluationApplication(
        settings_factory=settings_factory,
        registry=registry,
    )
