"""Explicit, local RAGAs evaluation action for the workspace dashboard."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from http import HTTPStatus
from typing import Any

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
        try:
            result = run_ragas_live_eval(
                input_path=settings.eval_data_path,
                output_path=settings.eval_ragas_results_path,
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
        except Exception:
            return error_response(
                start_response,
                HTTPStatus.SERVICE_UNAVAILABLE,
                error="evaluation_unavailable",
                message="Evaluation could not run with the current local dataset.",
                request_id=request_id,
                allow_methods=EVALUATION_METHODS,
                allow_origin=allow_origin,
                extra_headers=[(EVALUATION_CONTRACT_HEADER, EVALUATION_CONTRACT_VERSION)],
            )

        return json_response(
            start_response,
            HTTPStatus.OK,
            {
                "runId": result.run_id,
                "datasetId": result.manifest.dataset.dataset_id,
                "metrics": list(result.metrics),
                "resultRows": len(result.result_rows),
                "metricFailures": result.metric_failures,
                "dryRun": result.dry_run,
                "providerCallsAllowed": result.provider_calls_allowed,
                "status": "evaluation_prepared"
                if result.dry_run
                else "evaluation_completed",
                "detail": result.cost_warning,
                "evaluationType": "RAGAs",
                "evaluatedAt": result.manifest.evaluated_at.isoformat(),
            },
            request_id=request_id,
            allow_methods=EVALUATION_METHODS,
            allow_origin=allow_origin,
            extra_headers=[(EVALUATION_CONTRACT_HEADER, EVALUATION_CONTRACT_VERSION)],
        )


def create_evaluation_app(
    *,
    settings_factory: Callable[[], Settings] = load_settings,
) -> EvaluationApplication:
    return EvaluationApplication(settings_factory=settings_factory)
