"""Models and conversion helpers for live-session RAGAs evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from creative_coding_assistant.analytics import LangSmithRunMetadata
from creative_coding_assistant.eval.live_session import (
    LiveSessionEvalSample,
    LiveSessionProviderMetadata,
)

SUPPORTED_RAGAS_METRICS: tuple[str, ...] = (
    "context_precision",
    "faithfulness",
    "answer_relevancy",
    "context_relevancy",
    "context_recall",
)
DEFAULT_RAGAS_METRICS: tuple[str, ...] = ("context_precision",)


class RagasLiveEvalRow(BaseModel):
    """One live-session sample converted into RAGAs-compatible fields."""

    model_config = ConfigDict(frozen=True)

    sample_id: str = Field(min_length=1)
    user_input: str = Field(min_length=1)
    response: str = Field(min_length=1)
    retrieved_contexts: tuple[str, ...] = Field(min_length=1)
    ground_truth: str | None = None
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    domains: tuple[str, ...] = Field(default_factory=tuple)
    retrieval_scores: tuple[float, ...] = Field(default_factory=tuple)
    provider_metadata: LiveSessionProviderMetadata | None = None
    observability_metadata: LangSmithRunMetadata | None = None
    recorded_at: datetime | None = None

    @field_validator("user_input", "response")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("RAGAs eval text fields must not be blank.")
        return normalized

    @field_validator("retrieved_contexts")
    @classmethod
    def normalize_contexts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(context.strip() for context in value if context.strip())
        if not normalized:
            raise ValueError("RAGAs eval rows require at least one context.")
        return normalized

    @field_validator("ground_truth")
    @classmethod
    def normalize_optional_ground_truth(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    def ragas_payload(self) -> dict[str, object]:
        """Return only fields consumed by RAGAs metrics."""

        payload: dict[str, object] = {
            "user_input": self.user_input,
            "response": self.response,
            "retrieved_contexts": list(self.retrieved_contexts),
        }
        if self.ground_truth is not None:
            payload["reference"] = self.ground_truth
        return payload


class RagasSkippedSample(BaseModel):
    """A recorded live sample skipped before evaluator execution."""

    model_config = ConfigDict(frozen=True)

    sample_id: str | None = None
    reason: str = Field(min_length=1)
    detail: str | None = None


class RagasLiveEvalSelection(BaseModel):
    """Eligible and skipped samples selected from a live-session JSONL file."""

    model_config = ConfigDict(frozen=True)

    total_samples: int = Field(ge=0)
    rows: tuple[RagasLiveEvalRow, ...] = Field(default_factory=tuple)
    skipped: tuple[RagasSkippedSample, ...] = Field(default_factory=tuple)

    @computed_field
    @property
    def eligible_samples(self) -> int:
        return len(self.rows)

    @computed_field
    @property
    def skipped_samples(self) -> int:
        return len(self.skipped)


class RagasLiveEvalDataset(BaseModel):
    """Deterministic local dataset prepared from recorded live sessions."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(min_length=1)
    created_at: datetime
    source_path: Path | None = None
    total_samples: int = Field(ge=0)
    rows: tuple[RagasLiveEvalRow, ...] = Field(default_factory=tuple)
    skipped: tuple[RagasSkippedSample, ...] = Field(default_factory=tuple)
    metrics: tuple[str, ...] = Field(default_factory=tuple)

    @computed_field
    @property
    def eligible_samples(self) -> int:
        return len(self.rows)

    @computed_field
    @property
    def skipped_samples(self) -> int:
        return len(self.skipped)

    def ragas_payloads(self) -> tuple[dict[str, object], ...]:
        return tuple(row.ragas_payload() for row in self.rows)

    def summary_payload(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "created_at": self.created_at.isoformat(),
            "source_path": str(self.source_path) if self.source_path else None,
            "total_samples": self.total_samples,
            "eligible_samples": self.eligible_samples,
            "skipped_samples": self.skipped_samples,
            "metrics": list(self.metrics),
        }


def load_live_session_samples(path: Path) -> tuple[LiveSessionEvalSample, ...]:
    """Load recorded live-session samples from JSONL."""

    if not path.exists():
        return ()

    samples: list[LiveSessionEvalSample] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        samples.append(LiveSessionEvalSample.model_validate_json(raw_line))
    return tuple(samples)


def prepare_ragas_live_eval_dataset(
    samples: tuple[LiveSessionEvalSample, ...] | list[LiveSessionEvalSample],
    *,
    dataset_id: str = "live-session-ragas",
    source_path: Path | None = None,
    metric_names: Sequence[str] | None = DEFAULT_RAGAS_METRICS,
    limit: int | None = None,
    latest: int | None = None,
    created_at: datetime | None = None,
) -> RagasLiveEvalDataset:
    """Prepare a deterministic RAGAs dataset from recorded live sessions."""

    resolved_metrics = resolve_ragas_metric_names(metric_names)
    selection = select_ragas_live_eval_rows(samples, limit=limit, latest=latest)
    return RagasLiveEvalDataset(
        dataset_id=dataset_id,
        created_at=created_at or datetime.now(UTC),
        source_path=source_path,
        total_samples=selection.total_samples,
        rows=selection.rows,
        skipped=selection.skipped,
        metrics=resolved_metrics,
    )


def select_ragas_live_eval_rows(
    samples: tuple[LiveSessionEvalSample, ...] | list[LiveSessionEvalSample],
    *,
    limit: int | None = None,
    latest: int | None = None,
) -> RagasLiveEvalSelection:
    """Select live samples that have enough data for honest RAG evaluation."""

    if limit is not None and limit < 1:
        raise ValueError("RAGAs live eval limit must be at least 1.")
    if latest is not None and latest < 1:
        raise ValueError("RAGAs live eval latest count must be at least 1.")

    eligible_rows: list[RagasLiveEvalRow] = []
    skipped: list[RagasSkippedSample] = []
    for sample in samples:
        eligibility_reasons = ragas_sample_eligibility_reasons(sample)
        if eligibility_reasons:
            skipped.extend(
                RagasSkippedSample(
                    sample_id=sample.sample_id,
                    reason=reason,
                )
                for reason in eligibility_reasons
            )
            continue

        eligible_rows.append(
            RagasLiveEvalRow(
                sample_id=sample.sample_id,
                user_input=sample.question,
                response=sample.answer,
                retrieved_contexts=tuple(
                    context.excerpt for context in sample.retrieved_contexts
                ),
                ground_truth=sample.ground_truth,
                source_ids=tuple(
                    dict.fromkeys(
                        context.source_id for context in sample.retrieved_contexts
                    )
                ),
                domains=tuple(
                    dict.fromkeys(
                        context.domain.value for context in sample.retrieved_contexts
                    )
                ),
                retrieval_scores=tuple(
                    context.score for context in sample.retrieved_contexts
                ),
                provider_metadata=sample.provider_metadata,
                observability_metadata=sample.observability_metadata,
                recorded_at=sample.recorded_at,
            )
        )

    selection_limit = latest if latest is not None else limit
    if selection_limit is None:
        rows = eligible_rows
    elif latest is not None:
        rows = eligible_rows[-selection_limit:]
        skipped.extend(
            RagasSkippedSample(
                sample_id=row.sample_id,
                reason="limit_exceeded",
            )
            for row in eligible_rows[:-selection_limit]
        )
    else:
        rows = eligible_rows[:selection_limit]
        skipped.extend(
            RagasSkippedSample(
                sample_id=row.sample_id,
                reason="limit_exceeded",
            )
            for row in eligible_rows[selection_limit:]
        )

    return RagasLiveEvalSelection(
        total_samples=len(samples),
        rows=tuple(rows),
        skipped=tuple(skipped),
    )


def ragas_sample_eligibility_reasons(
    sample: LiveSessionEvalSample,
) -> tuple[str, ...]:
    """Return deterministic reasons a live sample cannot run through RAGAs."""

    reasons: list[str] = []
    if not sample.question.strip():
        reasons.append("missing_question")
    if not sample.answer.strip():
        reasons.append("missing_answer")
    if not sample.retrieved_contexts:
        reasons.append("missing_retrieved_contexts")

    for context in sample.retrieved_contexts:
        if not context.excerpt.strip():
            reasons.append("missing_context_excerpt")
        if not context.source_id.strip():
            reasons.append("missing_source_id")
        if not context.publisher.strip():
            reasons.append("missing_publisher")
        if not context.registry_title.strip():
            reasons.append("missing_registry_title")

    return tuple(dict.fromkeys(reasons))


def resolve_ragas_metric_names(
    metric_names: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve and validate requested RAGAs metric names."""

    requested = DEFAULT_RAGAS_METRICS if metric_names is None else tuple(metric_names)
    resolved: list[str] = []
    unsupported: list[str] = []
    for metric_name in requested:
        normalized = metric_name.strip()
        if normalized not in SUPPORTED_RAGAS_METRICS:
            unsupported.append(metric_name)
            continue
        if normalized not in resolved:
            resolved.append(normalized)

    if unsupported:
        supported = ", ".join(SUPPORTED_RAGAS_METRICS)
        unknown = ", ".join(unsupported)
        raise ValueError(
            f"Unsupported RAGAs metric(s): {unknown}. Supported: {supported}."
        )

    if not resolved:
        raise ValueError("At least one RAGAs metric must be selected.")

    return tuple(resolved)
