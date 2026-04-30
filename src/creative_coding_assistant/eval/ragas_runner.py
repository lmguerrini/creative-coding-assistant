"""Manual RAGAs runner for recorded live session samples."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from creative_coding_assistant.eval.ragas_models import (
    DEFAULT_RAGAS_METRICS,
    RagasLiveEvalRow,
    RagasSkippedSample,
    load_live_session_samples,
    resolve_ragas_metric_names,
    select_ragas_live_eval_rows,
)


class RagasDependencyError(RuntimeError):
    """Raised when optional RAGAs runtime dependencies are unavailable."""


class RagasEvaluatorConfig(BaseModel):
    """Configuration for the optional RAGAs evaluator runtime."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(default="gpt-4o-mini", min_length=1)
    embedding_model: str = Field(default="text-embedding-3-small", min_length=1)
    openai_api_key: SecretStr | None = Field(default=None, exclude=True)
    timeout_seconds: int = Field(default=180, ge=1)
    max_retries: int = Field(default=2, ge=0)
    max_workers: int = Field(default=2, ge=1)

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
        raw_value = (
            value.get_secret_value()
            if isinstance(value, SecretStr)
            else str(value)
        ).strip()
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
    evaluated_at: datetime


class RagasLiveEvalRunResult(BaseModel):
    """Summary returned by the manual live-session RAGAs runner."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    input_path: Path
    output_path: Path
    total_samples: int = Field(ge=0)
    eligible_samples: int = Field(ge=0)
    skipped_samples: int = Field(ge=0)
    metrics: tuple[str, ...] = Field(default_factory=tuple)
    metric_failures: int = Field(default=0, ge=0)
    result_rows: tuple[RagasLiveEvalResultRow, ...] = Field(default_factory=tuple)
    skipped: tuple[RagasSkippedSample, ...] = Field(default_factory=tuple)

    def summary_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "total_samples": self.total_samples,
            "eligible_samples": self.eligible_samples,
            "skipped_samples": self.skipped_samples,
            "metrics": list(self.metrics),
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
    """Runtime adapter around optional RAGAs imports."""

    def __init__(self, config: RagasEvaluatorConfig | None = None) -> None:
        self._config = config or RagasEvaluatorConfig()

    def evaluate(
        self,
        rows: Sequence[RagasLiveEvalRow],
        metric_names: Sequence[str],
    ) -> tuple[Mapping[str, float | None], ...]:
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from ragas import EvaluationDataset, SingleTurnSample, evaluate
            from ragas.metrics import (
                Faithfulness,
                LLMContextPrecisionWithoutReference,
                ResponseRelevancy,
            )
            from ragas.run_config import RunConfig
        except ImportError as exc:
            raise RagasDependencyError(
                "RAGAs evaluation requires optional dependency 'ragas'. "
                "Install it in the active environment before running live eval."
            ) from exc

        config = self._config
        api_key = config.get_openai_api_key()
        llm_kwargs: dict[str, object] = {
            "model": config.model,
            "temperature": 0,
            "timeout": config.timeout_seconds,
            "max_retries": config.max_retries,
        }
        embedding_kwargs: dict[str, object] = {
            "model": config.embedding_model,
            "timeout": config.timeout_seconds,
            "max_retries": config.max_retries,
        }
        if api_key is not None:
            llm_kwargs["api_key"] = api_key
            embedding_kwargs["api_key"] = api_key

        metric_factories = {
            "faithfulness": Faithfulness,
            "answer_relevancy": ResponseRelevancy,
            "context_precision": LLMContextPrecisionWithoutReference,
        }
        metrics = [metric_factories[name]() for name in metric_names]
        samples = [SingleTurnSample(**row.ragas_payload()) for row in rows]
        dataset = EvaluationDataset(samples=samples)
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ChatOpenAI(**llm_kwargs),
            embeddings=OpenAIEmbeddings(**embedding_kwargs),
            run_config=RunConfig(
                timeout=config.timeout_seconds,
                max_retries=config.max_retries,
                max_workers=config.max_workers,
            ),
            raise_exceptions=False,
        )
        return _scores_from_ragas_result(result, len(rows), metric_names)


def run_ragas_live_eval(
    *,
    input_path: Path,
    output_path: Path,
    metric_names: Sequence[str] | None = DEFAULT_RAGAS_METRICS,
    limit: int | None = None,
    latest: int | None = None,
    evaluator_config: RagasEvaluatorConfig | None = None,
    backend: RagasEvaluationBackend | None = None,
    run_id: str | None = None,
    evaluated_at: datetime | None = None,
) -> RagasLiveEvalRunResult:
    """Run manual RAGAs evaluation over recorded live-session samples."""

    samples = load_live_session_samples(input_path)
    selection = select_ragas_live_eval_rows(samples, limit=limit, latest=latest)
    resolved_run_id = run_id or uuid4().hex
    resolved_evaluated_at = evaluated_at or datetime.now(UTC)
    resolved_metrics = resolve_ragas_metric_names(metric_names)

    result_rows: tuple[RagasLiveEvalResultRow, ...] = ()
    if selection.rows:
        evaluator = backend or DefaultRagasEvaluationBackend(config=evaluator_config)
        scores = evaluator.evaluate(selection.rows, resolved_metrics)
        if len(scores) != len(selection.rows):
            raise ValueError("RAGAs backend returned a mismatched score row count.")
        result_rows = tuple(
            _build_result_row(
                row=row,
                score=score,
                metric_names=resolved_metrics,
                run_id=resolved_run_id,
                evaluated_at=resolved_evaluated_at,
            )
            for row, score in zip(selection.rows, scores, strict=True)
        )
        _write_result_rows(output_path, result_rows)

    return RagasLiveEvalRunResult(
        run_id=resolved_run_id,
        input_path=input_path,
        output_path=output_path,
        total_samples=selection.total_samples,
        eligible_samples=selection.eligible_samples,
        skipped_samples=selection.skipped_samples,
        metrics=resolved_metrics,
        metric_failures=sum(len(row.metric_errors) for row in result_rows),
        result_rows=result_rows,
        skipped=selection.skipped,
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
    return {
        metric_name: "metric_returned_null"
        for metric_name, value in metrics.items()
        if value is None
    }


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
            {
                name: _coerce_score(
                    _lookup_score(record, name, _RAGAS_RESULT_KEY_ALIASES)
                )
                for name in metric_names
            }
            for record in records
        )

    if isinstance(result, Mapping):
        scores = {
            name: _coerce_score(
                _lookup_score(result, name, _RAGAS_RESULT_KEY_ALIASES)
            )
            for name in metric_names
        }
        return tuple(scores for _ in range(row_count))

    raise TypeError("Unsupported RAGAs evaluation result shape.")


def _coerce_score(value: object) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


_RAGAS_RESULT_KEY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "answer_relevancy": ("response_relevancy",),
    "context_precision": ("llm_context_precision_without_reference",),
}
