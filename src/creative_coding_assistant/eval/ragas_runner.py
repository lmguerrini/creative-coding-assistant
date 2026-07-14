"""Manual RAGAs runner for recorded live session samples."""

from __future__ import annotations

import asyncio
import math
import os
import sys
from collections.abc import Awaitable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from creative_coding_assistant.analytics import (
    LangSmithObservability,
    LangSmithRunMetadata,
)
from creative_coding_assistant.eval.ragas_models import (
    DEFAULT_RAGAS_METRICS,
    RagasLiveEvalDataset,
    RagasLiveEvalRow,
    RagasSkippedSample,
    load_live_session_samples,
    prepare_ragas_live_eval_dataset,
)


class RagasDependencyError(RuntimeError):
    """Raised when optional RAGAs runtime dependencies are unavailable."""


class RagasProviderCostBoundaryError(RuntimeError):
    """Raised when a run would call evaluator providers without explicit opt-in."""


class RagasProviderBoundaryError(RuntimeError):
    """Raised when every evaluator call fails at the provider boundary."""


class RagasEvaluatorConfig(BaseModel):
    """Configuration for the optional RAGAs evaluator runtime."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(default="gpt-4o-mini", min_length=1)
    embedding_model: str = Field(default="text-embedding-3-small", min_length=1)
    openai_api_key: SecretStr | None = Field(default=None, exclude=True)
    timeout_seconds: int = Field(default=180, ge=1)
    max_retries: int = Field(default=2, ge=0)
    max_workers: int = Field(default=2, ge=1)
    max_output_tokens: int = Field(default=8192, ge=256, le=16384)

    @field_validator("model", "embedding_model")
    @classmethod
    def normalize_model_name(cls, value: str) -> str:
        return value.strip()

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def normalize_openai_api_key(
        cls,
        value: SecretStr | str | None,
    ) -> str | None:
        if value is None:
            return None
        raw_value = (value.get_secret_value() if isinstance(value, SecretStr) else str(value)).strip()
        return raw_value or None

    def get_openai_api_key(self) -> str | None:
        if self.openai_api_key is None:
            return None
        secret_value = self.openai_api_key.get_secret_value().strip()
        return secret_value or None


class RagasLiveEvalResultRow(BaseModel):
    """One local JSONL result row produced by a RAGAs evaluation run."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    metric_errors: dict[str, str] = Field(default_factory=dict)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    domains: tuple[str, ...] = Field(default_factory=tuple)
    retrieval_scores: tuple[float, ...] = Field(default_factory=tuple)
    evaluated_at: datetime


class RagasLiveEvalRunManifest(BaseModel):
    """Persisted summary for one local RAGAs evaluation attempt."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    dataset: RagasLiveEvalDataset
    input_path: Path
    output_path: Path
    manifest_path: Path
    metrics: tuple[str, ...] = Field(default_factory=tuple)
    dry_run: bool = False
    provider_calls_allowed: bool = False
    cost_warning: str = Field(min_length=1)
    result_rows: int = Field(ge=0)
    metric_failures: int = Field(default=0, ge=0)
    langsmith: LangSmithRunMetadata | None = None
    evaluated_at: datetime

    def summary_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "run_id": self.run_id,
            "dataset": self.dataset.summary_payload(),
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "manifest_path": str(self.manifest_path),
            "metrics": list(self.metrics),
            "dry_run": self.dry_run,
            "provider_calls_allowed": self.provider_calls_allowed,
            "cost_warning": self.cost_warning,
            "result_rows": self.result_rows,
            "metric_failures": self.metric_failures,
            "evaluated_at": self.evaluated_at.isoformat(),
        }
        if self.langsmith is not None:
            payload["langsmith"] = self.langsmith.payload()
        return payload


class RagasLiveEvalRunResult(BaseModel):
    """Summary returned by the manual live-session RAGAs runner."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    input_path: Path
    output_path: Path
    manifest_path: Path
    total_samples: int = Field(ge=0)
    eligible_samples: int = Field(ge=0)
    skipped_samples: int = Field(ge=0)
    metrics: tuple[str, ...] = Field(default_factory=tuple)
    dry_run: bool = False
    provider_calls_allowed: bool = False
    cost_warning: str = Field(min_length=1)
    metric_failures: int = Field(default=0, ge=0)
    result_rows: tuple[RagasLiveEvalResultRow, ...] = Field(default_factory=tuple)
    skipped: tuple[RagasSkippedSample, ...] = Field(default_factory=tuple)
    manifest: RagasLiveEvalRunManifest

    def summary_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "manifest_path": str(self.manifest_path),
            "total_samples": self.total_samples,
            "eligible_samples": self.eligible_samples,
            "skipped_samples": self.skipped_samples,
            "metrics": list(self.metrics),
            "dry_run": self.dry_run,
            "provider_calls_allowed": self.provider_calls_allowed,
            "cost_warning": self.cost_warning,
            "metric_failures": self.metric_failures,
            "result_rows": len(self.result_rows),
        }


class RagasEvaluationBackend(Protocol):
    def evaluate(
        self,
        rows: Sequence[RagasLiveEvalRow],
        metric_names: Sequence[str],
    ) -> tuple[Mapping[str, float | None], ...]:
        """Evaluate eligible rows and return per-row metric scores."""


class DefaultRagasEvaluationBackend:
    """Runtime adapter around the supported RAGAS collections API."""

    def __init__(self, config: RagasEvaluatorConfig | None = None) -> None:
        self._config = config or RagasEvaluatorConfig()

    def evaluate(
        self,
        rows: Sequence[RagasLiveEvalRow],
        metric_names: Sequence[str],
    ) -> tuple[Mapping[str, float | None], ...]:
        _disable_ragas_analytics()
        try:
            from openai import AsyncOpenAI
            from ragas.embeddings.base import embedding_factory
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                AnswerRelevancy,
                ContextPrecisionWithoutReference,
                ContextPrecisionWithReference,
                ContextRecall,
                ContextRelevance,
                Faithfulness,
            )
        except ImportError as exc:
            raise RagasDependencyError(
                "RAGAs evaluation requires optional dependency 'ragas'. "
                "Install it in the active environment before running live eval."
            ) from exc

        config = self._config
        api_key = config.get_openai_api_key()

        async def evaluate_with_client() -> tuple[Mapping[str, float | None], ...]:
            async with AsyncOpenAI(
                api_key=api_key,
                timeout=config.timeout_seconds,
                max_retries=config.max_retries,
            ) as client:
                llm = llm_factory(
                    config.model,
                    client=client,
                    temperature=0,
                    max_tokens=config.max_output_tokens,
                )
                embeddings = embedding_factory(
                    "openai",
                    model=config.embedding_model,
                    client=client,
                    interface="modern",
                )
                metrics: dict[str, object] = {
                    "context_precision_with_reference": ContextPrecisionWithReference(llm=llm),
                    "context_precision_without_reference": (ContextPrecisionWithoutReference(llm=llm)),
                    "faithfulness": Faithfulness(llm=llm),
                    "answer_relevancy": AnswerRelevancy(
                        llm=llm,
                        embeddings=embeddings,
                    ),
                    "context_relevancy": ContextRelevance(
                        llm=llm,
                        max_retries=max(1, config.max_retries + 1),
                    ),
                    "context_recall": ContextRecall(llm=llm),
                }
                return await _evaluate_collection_rows(
                    rows,
                    metric_names,
                    metrics=metrics,
                    max_workers=config.max_workers,
                    timeout_seconds=config.timeout_seconds,
                )

        return _run_async(evaluate_with_client())


def _disable_ragas_analytics() -> None:
    """Keep evaluator usage metadata off RAGAS' secondary analytics endpoint."""

    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    analytics_module = sys.modules.get("ragas._analytics")
    cached_check = getattr(analytics_module, "do_not_track", None)
    clear_cache = getattr(cached_check, "cache_clear", None)
    if callable(clear_cache):
        clear_cache()


async def _evaluate_collection_rows(
    rows: Sequence[RagasLiveEvalRow],
    metric_names: Sequence[str],
    *,
    metrics: Mapping[str, object],
    max_workers: int,
    timeout_seconds: int,
) -> tuple[Mapping[str, float | None], ...]:
    """Score rows with bounded metric concurrency and isolated null failures."""

    unsupported = tuple(
        name
        for name in metric_names
        if name
        not in {
            "context_precision",
            "faithfulness",
            "answer_relevancy",
            "context_relevancy",
            "context_recall",
        }
    )
    if unsupported:
        raise ValueError(f"Unsupported RAGAS metric names: {', '.join(unsupported)}")

    semaphore = asyncio.Semaphore(max_workers)

    async def score_metric(
        row: RagasLiveEvalRow,
        metric_name: str,
    ) -> tuple[float | None, bool, bool]:
        if metric_name == "context_recall" and row.ground_truth is None:
            return None, False, False
        async with semaphore:
            try:
                metric_result = await asyncio.wait_for(
                    _collection_metric_call(
                        row,
                        metric_name,
                        metrics=metrics,
                    ),
                    timeout=timeout_seconds,
                )
            except Exception as exc:
                logger.warning(
                    "RAGAS metric {} failed for sample {} ({})",
                    metric_name,
                    row.sample_id,
                    type(exc).__name__,
                )
                return None, True, _is_provider_boundary_failure(exc)
        return _coerce_score(getattr(metric_result, "value", None)), True, False

    async def score_row(
        row: RagasLiveEvalRow,
    ) -> tuple[Mapping[str, float | None], tuple[tuple[float | None, bool, bool], ...]]:
        values = await asyncio.gather(*(score_metric(row, name) for name in metric_names))
        outcomes = tuple(values)
        return (
            {
                name: outcome[0]
                for name, outcome in zip(metric_names, outcomes, strict=True)
            },
            outcomes,
        )

    row_results = tuple(await asyncio.gather(*(score_row(row) for row in rows)))
    attempted = tuple(
        outcome
        for _, outcomes in row_results
        for outcome in outcomes
        if outcome[1]
    )
    if attempted and all(outcome[2] for outcome in attempted):
        raise RagasProviderBoundaryError(
            "Every evaluator metric call failed at the provider boundary."
        )
    return tuple(scores for scores, _ in row_results)


_PROVIDER_BOUNDARY_EXCEPTION_NAMES = frozenset(
    {
        "APIConnectionError",
        "APIStatusError",
        "APITimeoutError",
        "AuthenticationError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "HTTPStatusError",
        "InternalServerError",
        "NetworkError",
        "NotFoundError",
        "PermissionDeniedError",
        "RateLimitError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutError",
        "TimeoutException",
        "TransportError",
    }
)


def _is_provider_boundary_failure(exc: BaseException) -> bool:
    """Recognize network/provider rejection while leaving parse failures isolated."""

    pending: list[BaseException] = [exc]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        names = {base.__name__ for base in type(current).__mro__}
        if names & _PROVIDER_BOUNDARY_EXCEPTION_NAMES:
            return True
        for related in (current.__cause__, current.__context__):
            if isinstance(related, BaseException):
                pending.append(related)
    return False


async def _collection_metric_call(
    row: RagasLiveEvalRow,
    metric_name: str,
    *,
    metrics: Mapping[str, object],
) -> object:
    contexts = list(row.retrieved_contexts)
    if metric_name == "context_precision":
        if row.ground_truth is not None:
            metric = metrics["context_precision_with_reference"]
            return await metric.ascore(
                user_input=row.user_input,
                reference=row.ground_truth,
                retrieved_contexts=contexts,
            )
        metric = metrics["context_precision_without_reference"]
        return await metric.ascore(
            user_input=row.user_input,
            response=row.response,
            retrieved_contexts=contexts,
        )
    if metric_name == "faithfulness":
        return await metrics[metric_name].ascore(
            user_input=row.user_input,
            response=row.response,
            retrieved_contexts=contexts,
        )
    if metric_name == "answer_relevancy":
        return await metrics[metric_name].ascore(
            user_input=row.user_input,
            response=row.response,
        )
    if metric_name == "context_relevancy":
        return await metrics[metric_name].ascore(
            user_input=row.user_input,
            retrieved_contexts=contexts,
        )
    if metric_name == "context_recall":
        if row.ground_truth is None:
            raise ValueError("Context recall requires a reference answer.")
        return await metrics[metric_name].ascore(
            user_input=row.user_input,
            reference=row.ground_truth,
            retrieved_contexts=contexts,
        )
    raise ValueError(f"Unsupported RAGAS metric name: {metric_name}")


def _run_async(
    awaitable: Awaitable[tuple[Mapping[str, float | None], ...]],
) -> tuple[Mapping[str, float | None], ...]:
    """Run an evaluator coroutine from sync or already-async callers."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, awaitable).result()


def run_ragas_live_eval(
    *,
    input_path: Path,
    output_path: Path,
    metric_names: Sequence[str] | None = DEFAULT_RAGAS_METRICS,
    limit: int | None = None,
    latest: int | None = None,
    evaluator_config: RagasEvaluatorConfig | None = None,
    backend: RagasEvaluationBackend | None = None,
    langsmith_observability: LangSmithObservability | None = None,
    run_id: str | None = None,
    evaluated_at: datetime | None = None,
    dry_run: bool = False,
    allow_provider_calls: bool = False,
) -> RagasLiveEvalRunResult:
    """Run manual RAGAs evaluation over recorded live-session samples."""

    samples = load_live_session_samples(input_path)
    resolved_run_id = run_id or uuid4().hex
    resolved_evaluated_at = evaluated_at or datetime.now(UTC)
    dataset = prepare_ragas_live_eval_dataset(
        samples,
        dataset_id=resolved_run_id,
        source_path=input_path,
        metric_names=metric_names,
        limit=limit,
        latest=latest,
        created_at=resolved_evaluated_at,
    )
    resolved_metrics = dataset.metrics
    manifest_path = ragas_run_manifest_path(output_path)
    cost_warning = _cost_warning_for_run(
        dry_run=dry_run,
        provider_calls_allowed=allow_provider_calls,
    )
    langsmith_run = (
        langsmith_observability.evaluation_run_context(
            eval_run_id=resolved_run_id,
            dataset_id=dataset.dataset_id,
            metrics=resolved_metrics,
            eligible_samples=dataset.eligible_samples,
            skipped_samples=dataset.skipped_samples,
            dry_run=dry_run,
        )
        if langsmith_observability is not None
        else None
    )

    result_rows: tuple[RagasLiveEvalResultRow, ...] = ()
    with _optional_langsmith_trace(
        langsmith_observability=langsmith_observability,
        langsmith_run=langsmith_run,
        dataset=dataset,
    ):
        if dataset.rows and not dry_run:
            if backend is None and not allow_provider_calls:
                raise RagasProviderCostBoundaryError(
                    "RAGAs evaluation may call evaluator LLM and embedding providers. "
                    "Pass allow_provider_calls=True or run with --dry-run first."
                )
            evaluator = backend or DefaultRagasEvaluationBackend(config=evaluator_config)
            scores = evaluator.evaluate(dataset.rows, resolved_metrics)
            if len(scores) != len(dataset.rows):
                raise ValueError("RAGAs backend returned a mismatched score row count.")
            result_rows = tuple(
                _build_result_row(
                    row=row,
                    score=score,
                    metric_names=resolved_metrics,
                    run_id=resolved_run_id,
                    evaluated_at=resolved_evaluated_at,
                )
                for row, score in zip(dataset.rows, scores, strict=True)
            )
            _write_result_rows(output_path, result_rows)

    metric_failures = sum(len(row.metric_errors) for row in result_rows)
    manifest = RagasLiveEvalRunManifest(
        run_id=resolved_run_id,
        dataset=dataset,
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        metrics=resolved_metrics,
        dry_run=dry_run,
        provider_calls_allowed=allow_provider_calls,
        cost_warning=cost_warning,
        result_rows=len(result_rows),
        metric_failures=metric_failures,
        langsmith=_manifest_langsmith_metadata(
            langsmith_observability=langsmith_observability,
            langsmith_run=langsmith_run,
        ),
        evaluated_at=resolved_evaluated_at,
    )
    _write_run_manifest(manifest)

    return RagasLiveEvalRunResult(
        run_id=resolved_run_id,
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        total_samples=dataset.total_samples,
        eligible_samples=dataset.eligible_samples,
        skipped_samples=dataset.skipped_samples,
        metrics=resolved_metrics,
        dry_run=dry_run,
        provider_calls_allowed=allow_provider_calls,
        cost_warning=cost_warning,
        metric_failures=metric_failures,
        result_rows=result_rows,
        skipped=dataset.skipped,
        manifest=manifest,
    )


def _build_result_row(
    *,
    row: RagasLiveEvalRow,
    score: Mapping[str, float | None],
    metric_names: Sequence[str],
    run_id: str,
    evaluated_at: datetime,
) -> RagasLiveEvalResultRow:
    metrics = _normalize_metric_scores(
        score,
        metric_names,
        _RAGAS_RESULT_KEY_ALIASES,
    )
    return RagasLiveEvalResultRow(
        run_id=run_id,
        sample_id=row.sample_id,
        metrics=metrics,
        metric_errors=_metric_errors_for_scores(metrics),
        source_ids=row.source_ids,
        domains=row.domains,
        retrieval_scores=row.retrieval_scores,
        evaluated_at=evaluated_at,
    )


def _write_result_rows(
    output_path: Path,
    rows: Sequence[RagasLiveEvalResultRow],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json())
            handle.write("\n")


def _optional_langsmith_trace(
    *,
    langsmith_observability: LangSmithObservability | None,
    langsmith_run: LangSmithRunMetadata | None,
    dataset: RagasLiveEvalDataset,
) -> object:
    if langsmith_observability is None or langsmith_run is None:
        return nullcontext()
    return langsmith_observability.trace(
        langsmith_run,
        run_type="chain",
        inputs={
            "dataset_id": dataset.dataset_id,
            "eligible_samples": dataset.eligible_samples,
            "skipped_samples": dataset.skipped_samples,
            "metrics": list(dataset.metrics),
        },
    )


def _manifest_langsmith_metadata(
    *,
    langsmith_observability: LangSmithObservability | None,
    langsmith_run: LangSmithRunMetadata | None,
) -> LangSmithRunMetadata | None:
    if langsmith_observability is None or langsmith_run is None:
        return None
    if langsmith_observability.event_payload(langsmith_run) is None:
        return None
    return langsmith_run


def _write_run_manifest(manifest: RagasLiveEvalRunManifest) -> None:
    manifest.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.manifest_path.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )


def ragas_run_manifest_path(output_path: Path) -> Path:
    """Return the sidecar manifest path for a JSONL RAGAs result artifact."""

    return output_path.with_name(f"{output_path.name}.manifest.json")


def _cost_warning_for_run(
    *,
    dry_run: bool,
    provider_calls_allowed: bool,
) -> str:
    if dry_run:
        return "Dry run only: no evaluator LLM or embedding provider calls are made."
    if provider_calls_allowed:
        return "Provider calls explicitly enabled: RAGAs may call evaluator LLM and embedding APIs and incur cost."
    return (
        "Provider calls disabled: pass --allow-provider-calls only after "
        "reviewing the prepared dataset and expected cost."
    )


def _normalize_metric_scores(
    scores: Mapping[str, float | None],
    metric_names: Sequence[str],
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, float | None]:
    normalized: dict[str, float | None] = {}
    for name in metric_names:
        value = _lookup_score(scores, name, aliases or {})
        normalized[name] = _coerce_score(value)
    return normalized


def _metric_errors_for_scores(
    metrics: Mapping[str, float | None],
) -> dict[str, str]:
    return {metric_name: "metric_returned_null" for metric_name, value in metrics.items() if value is None}


def _lookup_score(
    scores: Mapping[str, object],
    metric_name: str,
    aliases: Mapping[str, Sequence[str]],
) -> object:
    for key in (metric_name, *aliases.get(metric_name, ())):
        if key in scores:
            return scores[key]
    return None


def _scores_from_ragas_result(
    result: Any,
    row_count: int,
    metric_names: Sequence[str],
) -> tuple[Mapping[str, float | None], ...]:
    if hasattr(result, "to_pandas"):
        records = result.to_pandas().to_dict(orient="records")
        return tuple(
            {name: _coerce_score(_lookup_score(record, name, _RAGAS_RESULT_KEY_ALIASES)) for name in metric_names}
            for record in records
        )

    if isinstance(result, Mapping):
        scores = {name: _coerce_score(_lookup_score(result, name, _RAGAS_RESULT_KEY_ALIASES)) for name in metric_names}
        return tuple(scores for _ in range(row_count))

    raise TypeError("Unsupported RAGAs evaluation result shape.")


def _coerce_score(value: object) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) and 0 <= score <= 1 else None


_RAGAS_RESULT_KEY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "answer_relevancy": ("response_relevancy",),
    "context_precision": (
        "llm_context_precision_with_reference",
        "llm_context_precision_without_reference",
    ),
    "context_relevancy": ("context_relevance", "nv_context_relevance"),
}
