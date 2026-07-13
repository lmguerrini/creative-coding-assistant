"""Explicit, local RAGAs evaluation action for the workspace dashboard."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from http import HTTPStatus
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
from creative_coding_assistant.eval import (
    RagasDependencyError,
    RagasEvaluatorConfig,
    RagasProviderCostBoundaryError,
    run_ragas_live_eval,
)

EVALUATION_CONTRACT_VERSION = "evaluation.v1"
EVALUATION_CONTRACT_HEADER = "X-CCA-Evaluation-Contract-Version"
DEFAULT_EVALUATION_PATH = "/api/evaluation/run"
EVALUATION_METHODS = "POST, OPTIONS"
MAX_EVALUATION_REQUEST_BYTES = 8 * 1024


class EvaluationRunRequest(BaseModel):
    """An explicit, bounded request to evaluate local recorded sessions."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    dry_run: bool = Field(default=True, alias="dryRun")
    allow_provider_calls: bool = Field(default=False, alias="allowProviderCalls")
    approved_dataset: Literal["sanitized_public", "redacted_public"] = Field(
        default="sanitized_public", alias="approvedDataset"
    )


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


class EvaluationApplication:
    """Run only the local evaluation action selected by the operator."""

    def __init__(
        self,
        *,
        settings_factory: Callable[[], Settings] = load_settings,
        path: str = DEFAULT_EVALUATION_PATH,
    ) -> None:
        self._settings_factory = settings_factory
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
            return error_response(
                start_response,
                HTTPStatus.NOT_FOUND,
                error="not_found",
                message="Evaluation route was not found.",
                request_id=request_id,
                allow_methods=EVALUATION_METHODS,
                allow_origin=allow_origin,
                details={"available_paths": [self._path]},
            )
        if method == "OPTIONS":
            return empty_response(
                start_response,
                HTTPStatus.NO_CONTENT,
                request_id=request_id,
                allow_methods=EVALUATION_METHODS,
                allow_origin=allow_origin,
                extra_headers=[(EVALUATION_CONTRACT_HEADER, EVALUATION_CONTRACT_VERSION)],
            )
        if method != "POST":
            return error_response(
                start_response,
                HTTPStatus.METHOD_NOT_ALLOWED,
                error="method_not_allowed",
                message="Evaluation accepts POST and OPTIONS.",
                request_id=request_id,
                allow_methods=EVALUATION_METHODS,
                allow_origin=allow_origin,
                details={"allowed_methods": ["POST", "OPTIONS"]},
                extra_headers=[
                    ("Allow", EVALUATION_METHODS),
                    (EVALUATION_CONTRACT_HEADER, EVALUATION_CONTRACT_VERSION),
                ],
            )
        try:
            request = EvaluationRunRequest.model_validate(
                read_json_body(environ, max_bytes=MAX_EVALUATION_REQUEST_BYTES)
            )
        except ApiRequestBodyError as exc:
            return error_response(
                start_response,
                exc.status,
                error=exc.code,
                message=exc.message,
                request_id=request_id,
                allow_methods=EVALUATION_METHODS,
                allow_origin=allow_origin,
                extra_headers=[(EVALUATION_CONTRACT_HEADER, EVALUATION_CONTRACT_VERSION)],
            )
        except ValidationError:
            return error_response(
                start_response,
                HTTPStatus.BAD_REQUEST,
                error="invalid_evaluation_request",
                message="Evaluation options were invalid.",
                request_id=request_id,
                allow_methods=EVALUATION_METHODS,
                allow_origin=allow_origin,
                extra_headers=[(EVALUATION_CONTRACT_HEADER, EVALUATION_CONTRACT_VERSION)],
            )
        if not request.dry_run and not request.allow_provider_calls:
            return error_response(
                start_response,
                HTTPStatus.BAD_REQUEST,
                error="provider_evaluation_requires_opt_in",
                message="A live evaluation requires explicit provider-call authorization.",
                request_id=request_id,
                allow_methods=EVALUATION_METHODS,
                allow_origin=allow_origin,
                extra_headers=[(EVALUATION_CONTRACT_HEADER, EVALUATION_CONTRACT_VERSION)],
            )
        if not request.dry_run and not settings.has_openai_api_key:
            return _blocked_environment_response(
                start_response,
                request_id=request_id,
                allow_origin=allow_origin,
                message="BLOCKED_BY_EXECUTION_ENVIRONMENT: evaluator provider credentials are unavailable.",
            )
        dataset_contract = APPROVED_EVALUATION_DATASETS[request.approved_dataset]
        repository_root = Path(__file__).resolve().parents[3]
        input_path = repository_root / dataset_contract["path"]
        output_path = settings.eval_ragas_results_path.with_name(
            f"dashboard-{request.approved_dataset}-ragas-results.jsonl"
        )
        started_at = perf_counter()
        try:
            result = run_ragas_live_eval(
                input_path=input_path,
                output_path=output_path,
                metric_names=(
                    "context_precision",
                    "faithfulness",
                    "answer_relevancy",
                ),
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
            )
        except RagasProviderCostBoundaryError:
            return error_response(
                start_response,
                HTTPStatus.BAD_REQUEST,
                error="provider_evaluation_requires_opt_in",
                message="A live evaluation requires explicit provider-call authorization.",
                request_id=request_id,
                allow_methods=EVALUATION_METHODS,
                allow_origin=allow_origin,
                extra_headers=[(EVALUATION_CONTRACT_HEADER, EVALUATION_CONTRACT_VERSION)],
            )
        except RagasDependencyError:
            return _blocked_environment_response(
                start_response,
                request_id=request_id,
                allow_origin=allow_origin,
                message="BLOCKED_BY_EXECUTION_ENVIRONMENT: optional RAGAS dependencies are unavailable.",
            )
        except Exception:
            return _blocked_environment_response(
                start_response,
                request_id=request_id,
                allow_origin=allow_origin,
                message="BLOCKED_BY_EXECUTION_ENVIRONMENT: evaluator execution is unavailable.",
            )

        metric_scores = _aggregate_metric_scores(result.result_rows, result.metrics)
        case_results = [
            {
                "sampleId": row.sample_id,
                "metrics": dict(row.metrics),
                "metricErrors": dict(row.metric_errors),
                "sourceIds": list(row.source_ids),
                "domains": list(row.domains),
            }
            for row in result.result_rows
        ]

        return json_response(
            start_response,
            HTTPStatus.OK,
            {
                "runId": result.run_id,
                "datasetId": request.approved_dataset,
                "approvedDataset": request.approved_dataset,
                "datasetVersion": dataset_contract["version"],
                "privacyClass": dataset_contract["privacy_class"],
                "metrics": list(result.metrics),
                "resultRows": len(result.result_rows),
                "totalSamples": getattr(result, "total_samples", 0),
                "eligibleSamples": getattr(result, "eligible_samples", 0),
                "skippedSamples": getattr(result, "skipped_samples", 0),
                "metricFailures": result.metric_failures,
                "metricScores": metric_scores,
                "caseResults": case_results,
                "dryRun": result.dry_run,
                "providerCallsAllowed": result.provider_calls_allowed,
                "status": "evaluation_prepared"
                if result.dry_run
                else "evaluation_completed",
                "detail": result.cost_warning,
                "evaluationType": "RAGAs",
                "provider": "OpenAI evaluator" if not result.dry_run else None,
                "model": settings.eval_ragas_model if not result.dry_run else None,
                "embeddingModel": settings.openai_embedding_model if not result.dry_run else None,
                "durationMs": round((perf_counter() - started_at) * 1000),
                "evaluatedAt": result.manifest.evaluated_at.isoformat(),
            },
            request_id=request_id,
            allow_methods=EVALUATION_METHODS,
            allow_origin=allow_origin,
            extra_headers=[(EVALUATION_CONTRACT_HEADER, EVALUATION_CONTRACT_VERSION)],
        )


def _aggregate_metric_scores(result_rows: Any, metrics: Any) -> dict[str, float]:
    aggregates: dict[str, float] = {}
    for metric in metrics:
        values = [
            row.metrics.get(metric)
            for row in result_rows
            if isinstance(row.metrics.get(metric), (int, float))
        ]
        if values:
            aggregates[str(metric)] = sum(values) / len(values)
    return aggregates


def _blocked_environment_response(
    start_response: StartResponse,
    *,
    request_id: str,
    allow_origin: str | None,
    message: str,
) -> Iterable[bytes]:
    return error_response(
        start_response,
        HTTPStatus.SERVICE_UNAVAILABLE,
        error="blocked_by_execution_environment",
        message=message,
        request_id=request_id,
        allow_methods=EVALUATION_METHODS,
        allow_origin=allow_origin,
        details={"status": "BLOCKED_BY_EXECUTION_ENVIRONMENT"},
        extra_headers=[(EVALUATION_CONTRACT_HEADER, EVALUATION_CONTRACT_VERSION)],
    )


def create_evaluation_app(
    *,
    settings_factory: Callable[[], Settings] = load_settings,
) -> EvaluationApplication:
    return EvaluationApplication(settings_factory=settings_factory)
