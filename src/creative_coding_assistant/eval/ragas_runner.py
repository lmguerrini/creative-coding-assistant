"""Manual RAGAs runner for recorded live session samples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.eval.ragas_models import (
    DEFAULT_RAGAS_METRICS,
    RagasLiveEvalRow,
    RagasSkippedSample,
    load_live_session_samples,
    select_ragas_live_eval_rows,
)


class RagasDependencyError(RuntimeError):
    """Raised when optional RAGAs runtime dependencies are unavailable."""


class RagasLiveEvalResultRow(BaseModel):
    """One local JSONL result row produced by a RAGAs evaluation run."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    metrics: dict[str, float | None] = Field(default_factory=dict)
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

    def evaluate(
        self,
        rows: Sequence[RagasLiveEvalRow],
        metric_names: Sequence[str],
    ) -> tuple[Mapping[str, float | None], ...]:
        try:
            from ragas import EvaluationDataset, SingleTurnSample, evaluate
            from ragas.metrics import (
                Faithfulness,
                LLMContextPrecisionWithoutReference,
                ResponseRelevancy,
            )
        except ImportError as exc:
            raise RagasDependencyError(
                "RAGAs evaluation requires optional dependency 'ragas'. "
                "Install it in the active environment before running live eval."
            ) from exc

        metric_factories = {
            "faithfulness": Faithfulness,
            "answer_relevancy": ResponseRelevancy,
            "context_precision": LLMContextPrecisionWithoutReference,
        }
        metrics = [metric_factories[name]() for name in metric_names]
        samples = [SingleTurnSample(**row.ragas_payload()) for row in rows]
        dataset = EvaluationDataset(samples=samples)
        result = evaluate(dataset=dataset, metrics=metrics)
        return _scores_from_ragas_result(result, len(rows), metric_names)


def run_ragas_live_eval(
    *,
    input_path: Path,
    output_path: Path,
    metric_names: Sequence[str] = DEFAULT_RAGAS_METRICS,
    backend: RagasEvaluationBackend | None = None,
    run_id: str | None = None,
    evaluated_at: datetime | None = None,
) -> RagasLiveEvalRunResult:
    """Run manual RAGAs evaluation over recorded live-session samples."""

    samples = load_live_session_samples(input_path)
    selection = select_ragas_live_eval_rows(samples)
    resolved_run_id = run_id or uuid4().hex
    resolved_evaluated_at = evaluated_at or datetime.now(UTC)
    resolved_metrics = tuple(dict.fromkeys(metric_names))

    result_rows: tuple[RagasLiveEvalResultRow, ...] = ()
    if selection.rows:
        evaluator = backend or DefaultRagasEvaluationBackend()
        scores = evaluator.evaluate(selection.rows, resolved_metrics)
        if len(scores) != len(selection.rows):
            raise ValueError("RAGAs backend returned a mismatched score row count.")
        result_rows = tuple(
            RagasLiveEvalResultRow(
                run_id=resolved_run_id,
                sample_id=row.sample_id,
                metrics=_normalize_metric_scores(
                    score,
                    resolved_metrics,
                    _RAGAS_RESULT_KEY_ALIASES,
                ),
                source_ids=row.source_ids,
                domains=row.domains,
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
        result_rows=result_rows,
        skipped=selection.skipped,
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
        normalized[name] = None if value is None else float(value)
    return normalized


def _lookup_score(
    scores: Mapping[str, float | None],
    metric_name: str,
    aliases: Mapping[str, Sequence[str]],
) -> float | None:
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
        scores = {name: _coerce_score(result.get(name)) for name in metric_names}
        return tuple(scores for _ in range(row_count))

    raise TypeError("Unsupported RAGAs evaluation result shape.")


def _coerce_score(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_RAGAS_RESULT_KEY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "answer_relevancy": ("response_relevancy",),
    "context_precision": ("llm_context_precision_without_reference",),
}
