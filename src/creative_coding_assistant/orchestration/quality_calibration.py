"""Calibrate existing artifact-quality signals into bounded guidance."""

from __future__ import annotations

from statistics import mean

from creative_coding_assistant.orchestration.artifacts import (
    ArtifactCritiqueDimension,
    CalibratedQualityEvaluation,
    CalibratedQualitySignal,
    CreativeQualityEvaluation,
    SacredConsistencyEvaluation,
    WorkflowArtifact,
)


def calibrate_artifact_quality(
    artifact: WorkflowArtifact,
    *,
    dimensions: dict[str, ArtifactCritiqueDimension],
    legacy_score: float,
    creative_evaluation: CreativeQualityEvaluation,
    sacred_consistency: SacredConsistencyEvaluation | None,
    reasons: tuple[str, ...],
) -> CalibratedQualityEvaluation:
    """Compose existing deterministic quality signals into decision support."""

    signals = _quality_signals(
        artifact,
        dimensions=dimensions,
        legacy_score=legacy_score,
        creative_evaluation=creative_evaluation,
        sacred_consistency=sacred_consistency,
        reasons=reasons,
    )
    weighted = _weighted_score(signals)
    score, adjustments = _apply_conservative_caps(
        weighted,
        dimensions=dimensions,
        sacred_consistency=sacred_consistency,
        reasons=reasons,
    )
    decision_band = _decision_band(score, reasons=reasons)
    confidence = _confidence(signals, sacred_consistency=sacred_consistency)

    return CalibratedQualityEvaluation(
        score=score,
        legacy_score=round(legacy_score, 3),
        decision_band=decision_band,
        confidence=confidence,
        signals=signals,
        adjustments=adjustments,
        rationale=_rationale(
            score,
            legacy_score=legacy_score,
            decision_band=decision_band,
            signals=signals,
            adjustments=adjustments,
        ),
        summary=(
            f"Calibrated decision-support score {score:.2f} from "
            f"{len(signals)} available signal(s). This is bounded guidance, "
            "not an objective measure of artistic quality."
        ),
    )


def _quality_signals(
    artifact: WorkflowArtifact,
    *,
    dimensions: dict[str, ArtifactCritiqueDimension],
    legacy_score: float,
    creative_evaluation: CreativeQualityEvaluation,
    sacred_consistency: SacredConsistencyEvaluation | None,
    reasons: tuple[str, ...],
) -> tuple[CalibratedQualitySignal, ...]:
    signals = [
        _signal(
            "legacy_critique",
            "Legacy critique",
            legacy_score,
            0.34,
            "Existing weighted artifact critique score is preserved as a core input.",
        ),
        _signal(
            "creative_quality",
            "Creative quality",
            creative_evaluation.overall_score,
            0.22,
            "Creative Quality Critic score contributes artistic decision support.",
        ),
        _signal(
            "runtime_preview",
            "Runtime and preview",
            _runtime_preview_score(artifact, dimensions),
            0.18,
            "Runtime suitability and preview readiness are aligned conservatively.",
        ),
        _signal(
            "refinement_pressure",
            "Refinement pressure",
            _refinement_pressure_score(
                reasons,
                creative_evaluation=creative_evaluation,
                sacred_consistency=sacred_consistency,
            ),
            0.12,
            "Open critique reasons and refinement opportunities lower confidence.",
        ),
    ]
    if sacred_consistency is not None:
        signals.append(
            _signal(
                "sacred_consistency",
                "Sacred consistency",
                sacred_consistency.overall_score,
                0.14,
                (
                    "Sacred Consistency Evaluator score is included only when "
                    "symbolic or geometric metadata exists."
                ),
            )
        )
    grounding = _grounding_score(artifact)
    if grounding is not None:
        signals.append(
            _signal(
                "grounding",
                "Grounding",
                grounding,
                0.1,
                "Retrieval or grounding signal exists for this artifact.",
            )
        )
    return tuple(signals)


def _runtime_preview_score(
    artifact: WorkflowArtifact,
    dimensions: dict[str, ArtifactCritiqueDimension],
) -> float:
    runtime = dimensions["runtime_suitability"].score
    preview = dimensions["preview_readiness"].score
    if not artifact.preview_eligible and runtime >= 0.8:
        return round((runtime * 0.7) + (preview * 0.3), 3)
    return round(mean((runtime, preview)), 3)


def _refinement_pressure_score(
    reasons: tuple[str, ...],
    *,
    creative_evaluation: CreativeQualityEvaluation,
    sacred_consistency: SacredConsistencyEvaluation | None,
) -> float:
    opportunity_count = len(creative_evaluation.refinement_opportunities)
    if sacred_consistency is not None:
        opportunity_count += len(sacred_consistency.refinement_opportunities)
    score = 1.0 - min(opportunity_count, 5) * 0.08 - min(len(reasons), 4) * 0.12
    return round(min(max(score, 0.0), 1.0), 3)


def _grounding_score(artifact: WorkflowArtifact) -> float | None:
    translation = artifact.creative_translation
    if translation is None:
        return None
    # Placeholder for future artifact-level retrieval metadata; do not fabricate it.
    return None


def _apply_conservative_caps(
    score: float,
    *,
    dimensions: dict[str, ArtifactCritiqueDimension],
    sacred_consistency: SacredConsistencyEvaluation | None,
    reasons: tuple[str, ...],
) -> tuple[float, tuple[str, ...]]:
    capped = score
    adjustments: list[str] = []

    if sacred_consistency and sacred_consistency.claim_safety.level == "unsupported":
        capped = min(capped, 0.58)
        adjustments.append(
            "Capped because generated artifact signals contain unsupported "
            "symbolic claims."
        )
    if "overall_quality_below_threshold" in reasons:
        capped = min(capped, 0.66)
        adjustments.append("Capped because the legacy critique is below threshold.")
    if dimensions["code_quality"].score < 0.5:
        capped = min(capped, 0.64)
        adjustments.append("Capped because static code quality is weak.")
    if (
        dimensions["runtime_suitability"].score < 0.5
        and dimensions["preview_readiness"].score < 0.5
    ):
        capped = min(capped, 0.7)
        adjustments.append(
            "Capped because runtime and preview signals are both weak."
        )

    return round(min(max(capped, 0.0), 1.0), 3), tuple(adjustments[:5])


def _decision_band(
    score: float,
    *,
    reasons: tuple[str, ...],
) -> str:
    if score >= 0.82 and not reasons:
        return "strong_candidate"
    if score >= 0.68 and "sacred_claim_safety" not in reasons:
        return "usable_candidate"
    if score >= 0.5:
        return "needs_refinement"
    return "high_risk"


def _confidence(
    signals: tuple[CalibratedQualitySignal, ...],
    *,
    sacred_consistency: SacredConsistencyEvaluation | None,
) -> str:
    if len(signals) >= 5 and sacred_consistency is not None:
        return "high"
    if len(signals) >= 4:
        return "medium"
    return "low"


def _weighted_score(signals: tuple[CalibratedQualitySignal, ...]) -> float:
    total_weight = sum(signal.weight for signal in signals)
    if total_weight <= 0:
        return 0.0
    score = sum(signal.score * signal.weight for signal in signals) / total_weight
    return round(min(max(score, 0.0), 1.0), 3)


def _rationale(
    score: float,
    *,
    legacy_score: float,
    decision_band: str,
    signals: tuple[CalibratedQualitySignal, ...],
    adjustments: tuple[str, ...],
) -> str:
    strongest = max(signals, key=lambda signal: signal.score)
    weakest = min(signals, key=lambda signal: signal.score)
    adjustment_text = (
        f" {len(adjustments)} conservative adjustment(s) applied."
        if adjustments
        else " No conservative caps were required."
    )
    return (
        f"{decision_band.replace('_', ' ')} at {score:.2f}; legacy score "
        f"{legacy_score:.2f}. Strongest signal: {strongest.label} "
        f"({strongest.score:.2f}); weakest signal: {weakest.label} "
        f"({weakest.score:.2f}).{adjustment_text}"
    )


def _signal(
    key: str,
    label: str,
    score: float,
    weight: float,
    rationale: str,
) -> CalibratedQualitySignal:
    return CalibratedQualitySignal(
        key=key,  # type: ignore[arg-type]
        label=label,
        score=round(min(max(score, 0.0), 1.0), 3),
        weight=weight,
        rationale=rationale,
    )
