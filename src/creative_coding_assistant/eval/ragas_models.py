"""Models and conversion helpers for live-session RAGAs evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.eval.live_session import LiveSessionEvalSample

SUPPORTED_RAGAS_METRICS: tuple[str, ...] = (
    "context_precision",
    "faithfulness",
    "answer_relevancy",
)
DEFAULT_RAGAS_METRICS: tuple[str, ...] = ("context_precision",)


class RagasLiveEvalRow(BaseModel):
    """One live-session sample converted into RAGAs-compatible fields."""

    model_config = ConfigDict(frozen=True)

    sample_id: str = Field(min_length=1)
    user_input: str = Field(min_length=1)
    response: str = Field(min_length=1)
    retrieved_contexts: tuple[str, ...] = Field(min_length=1)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    domains: tuple[str, ...] = Field(default_factory=tuple)

    def ragas_payload(self) -> dict[str, object]:
        """Return only fields consumed by RAGAs metrics."""

        return {
            "user_input": self.user_input,
            "response": self.response,
            "retrieved_contexts": list(self.retrieved_contexts),
        }


class RagasSkippedSample(BaseModel):
    """A recorded live sample skipped before evaluator execution."""

    model_config = ConfigDict(frozen=True)

    sample_id: str | None = None
    reason: str = Field(min_length=1)


class RagasLiveEvalSelection(BaseModel):
    """Eligible and skipped samples selected from a live-session JSONL file."""

    model_config = ConfigDict(frozen=True)

    total_samples: int = Field(ge=0)
    rows: tuple[RagasLiveEvalRow, ...] = Field(default_factory=tuple)
    skipped: tuple[RagasSkippedSample, ...] = Field(default_factory=tuple)

    @property
    def eligible_samples(self) -> int:
        return len(self.rows)

    @property
    def skipped_samples(self) -> int:
        return len(self.skipped)


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
        if not sample.retrieved_contexts:
            skipped.append(
                RagasSkippedSample(
                    sample_id=sample.sample_id,
                    reason="missing_retrieved_contexts",
                )
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
