"""Bounded multi-pass refinement helpers for workflow-owned artifacts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.orchestration.artifacts import (
    RefinementPassRecord,
    RefinementPassStopReason,
    WorkflowArtifact,
)

DEFAULT_REFINEMENT_PASS_LIMIT = 2
MAX_REFINEMENT_PASS_LIMIT = 3
QUALITY_IMPROVEMENT_THRESHOLD = 0.04


class RefinementPassDecision(BaseModel):
    """Deterministic decision for the next controlled refinement pass."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    should_continue: bool
    next_pass_number: int = Field(ge=1)
    max_passes: int = Field(ge=1, le=MAX_REFINEMENT_PASS_LIMIT)
    source_artifact_id: str | None = None
    refinement_objective: str | None = None
    stop_reason: RefinementPassStopReason | None
    quality_before: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_after: float | None = Field(default=None, ge=0.0, le=1.0)
    opportunities: tuple[str, ...] = ()


def normalize_refinement_pass_limit(value: int | None = None) -> int:
    """Keep refinement loops intentionally low and never unbounded."""

    if value is None:
        return DEFAULT_REFINEMENT_PASS_LIMIT
    return min(max(value, 1), MAX_REFINEMENT_PASS_LIMIT)


def select_refinement_source(
    artifacts: tuple[WorkflowArtifact, ...],
) -> WorkflowArtifact | None:
    """Return the artifact that should carry the next refinement pass."""

    if not artifacts:
        return None
    return next(
        (
            artifact
            for artifact in artifacts
            if artifact.is_recommended or artifact.is_default
        ),
        artifacts[0],
    )


def plan_next_refinement_pass(
    *,
    source_artifact: WorkflowArtifact | None,
    pass_history: tuple[RefinementPassRecord, ...] = (),
    max_passes: int | None = None,
) -> RefinementPassDecision:
    """Plan the next pass from current artifact signals and prior pass outcomes."""

    pass_limit = normalize_refinement_pass_limit(max_passes)
    next_pass_number = len(pass_history) + 1
    if source_artifact is None:
        return RefinementPassDecision(
            should_continue=False,
            next_pass_number=next_pass_number,
            max_passes=pass_limit,
            stop_reason="no_useful_opportunities",
        )

    latest = pass_history[-1] if pass_history else None
    if latest and latest.stop_reason in {
        "quality_improved",
        "no_useful_opportunities",
        "runtime_preview_safety_failed",
        "max_passes_reached",
    }:
        return RefinementPassDecision(
            should_continue=False,
            next_pass_number=next_pass_number,
            max_passes=pass_limit,
            source_artifact_id=source_artifact.id,
            stop_reason=latest.stop_reason,
            quality_before=latest.quality_before,
            quality_after=latest.quality_after,
        )

    if next_pass_number > pass_limit:
        return RefinementPassDecision(
            should_continue=False,
            next_pass_number=next_pass_number,
            max_passes=pass_limit,
            source_artifact_id=source_artifact.id,
            stop_reason="max_passes_reached",
            quality_before=artifact_quality_score(source_artifact),
        )

    opportunities = refinement_opportunities(source_artifact)
    if not opportunities:
        return RefinementPassDecision(
            should_continue=False,
            next_pass_number=next_pass_number,
            max_passes=pass_limit,
            source_artifact_id=source_artifact.id,
            stop_reason="no_useful_opportunities",
            quality_before=artifact_quality_score(source_artifact),
        )

    if runtime_preview_safety_failed(source_artifact):
        return RefinementPassDecision(
            should_continue=False,
            next_pass_number=next_pass_number,
            max_passes=pass_limit,
            source_artifact_id=source_artifact.id,
            stop_reason="runtime_preview_safety_failed",
            quality_before=artifact_quality_score(source_artifact),
            opportunities=opportunities,
        )

    return RefinementPassDecision(
        should_continue=True,
        next_pass_number=next_pass_number,
        max_passes=pass_limit,
        source_artifact_id=source_artifact.id,
        refinement_objective=build_refinement_objective(
            source_artifact,
            opportunities=opportunities,
        ),
        stop_reason=None,
        quality_before=artifact_quality_score(source_artifact),
        opportunities=opportunities,
    )


def start_refinement_pass_record(
    *,
    source_artifact: WorkflowArtifact,
    decision: RefinementPassDecision,
) -> RefinementPassRecord:
    """Create the lineage record before the next generation pass runs."""

    objective = decision.refinement_objective or build_refinement_objective(
        source_artifact,
        opportunities=(
            decision.opportunities or refinement_opportunities(source_artifact)
        ),
    )
    objective = _truncate(objective, 720)
    return RefinementPassRecord(
        pass_number=decision.next_pass_number,
        source_artifact_id=source_artifact.id,
        source_artifact_title=source_artifact.title,
        refinement_objective=objective,
        quality_before=decision.quality_before,
        quality_after=None,
        stop_reason="continue_available",
        summary=_truncate(f"Pass {decision.next_pass_number}: {objective}", 360),
    )


def complete_latest_refinement_pass(
    *,
    pass_history: tuple[RefinementPassRecord, ...],
    result_artifact: WorkflowArtifact | None,
    max_passes: int | None = None,
) -> tuple[RefinementPassRecord, ...]:
    """Attach after-quality and terminal reason once a refinement pass completes."""

    if not pass_history:
        return ()
    latest = pass_history[-1]
    if latest.quality_after is not None:
        return pass_history

    pass_limit = normalize_refinement_pass_limit(max_passes)
    quality_after = artifact_quality_score(result_artifact)
    stop_reason = _completed_stop_reason(
        latest=latest,
        result_artifact=result_artifact,
        quality_after=quality_after,
        max_passes=pass_limit,
    )
    completed = latest.model_copy(
        update={
            "result_artifact_id": result_artifact.id if result_artifact else None,
            "result_artifact_title": (
                result_artifact.title if result_artifact else None
            ),
            "quality_after": quality_after,
            "stop_reason": stop_reason,
            "summary": _pass_summary(
                pass_number=latest.pass_number,
                stop_reason=stop_reason,
                quality_before=latest.quality_before,
                quality_after=quality_after,
            ),
        }
    )
    return (*pass_history[:-1], completed)


def attach_refinement_history(
    artifacts: tuple[WorkflowArtifact, ...],
    pass_history: tuple[RefinementPassRecord, ...],
) -> tuple[WorkflowArtifact, ...]:
    """Carry explicit pass lineage on the recommended artifact version."""

    if not artifacts or not pass_history:
        return artifacts
    target = select_refinement_source(artifacts)
    if target is None:
        return artifacts
    return tuple(
        artifact.model_copy(update={"refinement_passes": pass_history})
        if artifact.id == target.id
        else artifact
        for artifact in artifacts
    )


def artifact_quality_score(artifact: WorkflowArtifact | None) -> float | None:
    """Prefer calibrated quality, then legacy critique scores."""

    if artifact is None:
        return None
    critique = artifact.critique
    if critique and critique.calibrated_quality is not None:
        return critique.calibrated_quality.score
    if artifact.quality_score is not None:
        return artifact.quality_score
    if critique is not None:
        return critique.overall_score
    return None


def refinement_opportunities(
    artifact: WorkflowArtifact,
    *,
    limit: int = 6,
) -> tuple[str, ...]:
    """Collect existing critique/calibration/translation signals for refinement."""

    opportunities: list[str] = []
    critique = artifact.critique
    translation = artifact.creative_translation
    if critique is not None:
        _append_unique(opportunities, critique.refinement_guidance)
        if critique.creative_evaluation is not None:
            _extend_unique(
                opportunities,
                critique.creative_evaluation.refinement_opportunities,
            )
        if critique.sacred_consistency is not None:
            _extend_unique(
                opportunities,
                critique.sacred_consistency.refinement_opportunities,
            )
        if critique.calibrated_quality is not None:
            _extend_unique(opportunities, critique.calibrated_quality.adjustments)
        _extend_unique(opportunities, critique.reasons)
    _extend_unique(opportunities, translation.refinement_targets if translation else ())
    if translation and translation.audio_reactive is not None:
        for mapping in translation.audio_reactive.mappings:
            targets = ", ".join(target.value for target in mapping.targets)
            _append_unique(
                opportunities,
                f"Preserve audio-reactive {mapping.source.value} mapping to {targets}.",
            )
    return tuple(opportunities[:limit])


def build_refinement_objective(
    artifact: WorkflowArtifact,
    *,
    opportunities: tuple[str, ...] | None = None,
) -> str:
    """Create compact provider-facing guidance from existing quality signals."""

    focus = (
        opportunities
        if opportunities is not None
        else refinement_opportunities(artifact)
    )
    if not focus:
        return (
            f"Create an explicit refined version of {artifact.title} while "
            "preserving the original domain, runtime, and source artifact."
        )
    return _truncate(
        f"Refine {artifact.title} by addressing: "
        + " ".join(focus[:4])
        + (
            " Preserve the selected artifact lineage, runtime contract, and a "
            "complete runnable output."
        ),
        720,
    )


def runtime_preview_safety_failed(artifact: WorkflowArtifact) -> bool:
    critique = artifact.critique
    if critique is None:
        return False
    return (
        critique.runtime_suitability.score < 0.5
        and critique.preview_readiness.score < 0.5
    )


def _completed_stop_reason(
    *,
    latest: RefinementPassRecord,
    result_artifact: WorkflowArtifact | None,
    quality_after: float | None,
    max_passes: int,
) -> RefinementPassStopReason:
    if _quality_improved(latest.quality_before, quality_after):
        return "quality_improved"
    if result_artifact is not None and runtime_preview_safety_failed(result_artifact):
        return "runtime_preview_safety_failed"
    if result_artifact is not None and not refinement_opportunities(result_artifact):
        return "no_useful_opportunities"
    if latest.pass_number >= max_passes:
        return "max_passes_reached"
    return "continue_available"


def _quality_improved(before: float | None, after: float | None) -> bool:
    if before is None or after is None:
        return False
    return after - before >= QUALITY_IMPROVEMENT_THRESHOLD


def _pass_summary(
    *,
    pass_number: int,
    stop_reason: RefinementPassStopReason,
    quality_before: float | None,
    quality_after: float | None,
) -> str:
    quality = (
        f" Quality {quality_before:.2f} -> {quality_after:.2f}."
        if quality_before is not None and quality_after is not None
        else ""
    )
    return f"Pass {pass_number} completed: {stop_reason.replace('_', ' ')}.{quality}"


def _append_unique(items: list[str], value: str | None) -> None:
    normalized = (value or "").strip()
    if normalized and normalized not in items:
        items.append(normalized)


def _extend_unique(items: list[str], values: tuple[str, ...]) -> None:
    for value in values:
        _append_unique(items, value)


def _truncate(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."
