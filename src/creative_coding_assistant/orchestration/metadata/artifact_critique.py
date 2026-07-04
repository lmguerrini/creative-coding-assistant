"""Deterministic artifact critique and ranking for workflow outputs."""

from __future__ import annotations

from statistics import mean

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest, CreativeCodingDomain
from creative_coding_assistant.orchestration._metadata_utils import _token_set
from creative_coding_assistant.orchestration.artifacts import (
    ArtifactCritiqueDimension,
    CreativeQualityEvaluation,
    SacredConsistencyEvaluation,
    WorkflowArtifact,
    WorkflowArtifactCritique,
)
from creative_coding_assistant.orchestration.creative_quality import (
    evaluate_artifact_creative_quality,
)
from creative_coding_assistant.orchestration.domain_generation import (
    get_domain_runtime_support,
    is_previewable_generation_domain,
)
from creative_coding_assistant.orchestration.quality_calibration import (
    calibrate_artifact_quality,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.sacred_consistency import (
    evaluate_artifact_sacred_consistency,
)
from creative_coding_assistant.preview import PreviewResult

ARTIFACT_CRITIQUE_PASS_THRESHOLD = 0.68
CALIBRATED_RANK_STABILITY_BUCKET = 0.03

_CODE_MARKERS = frozenset(
    {
        "const",
        "export",
        "function",
        "import",
        "let",
        "return",
        "setup",
        "draw",
        "void",
    }
)


class ArtifactCritiqueSummary(BaseModel):
    """Run-level summary for artifact critique results."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    artifact_count: int = Field(ge=0)
    critiques: tuple[WorkflowArtifactCritique, ...] = ()
    recommended_artifact_id: str | None = None
    recommended_artifact_title: str | None = None
    average_score: float = Field(ge=0.0, le=1.0)
    average_calibrated_score: float | None = Field(default=None, ge=0.0, le=1.0)
    failed_artifact_count: int = Field(ge=0)
    refinement_required: bool = False
    refinement_reasons: tuple[str, ...] = ()
    refinement_guidance: str | None = None


def critique_workflow_artifacts(
    artifacts: tuple[WorkflowArtifact, ...],
    *,
    request: AssistantRequest,
    route_decision: RouteDecision,
    preview_results: tuple[PreviewResult, ...] = (),
) -> tuple[tuple[WorkflowArtifact, ...], ArtifactCritiqueSummary]:
    """Score, rank, and annotate generated artifacts deterministically."""

    if not artifacts:
        return (), ArtifactCritiqueSummary(
            artifact_count=0,
            average_score=0.0,
            failed_artifact_count=0,
        )

    preview_artifact_ids = {
        result.artifact_id
        for result in preview_results
        if result.status.value == "succeeded"
    }
    scored = [
        _score_artifact(
            artifact,
            request=request,
            route_decision=route_decision,
            preview_artifact_ids=preview_artifact_ids,
        )
        for artifact in artifacts
    ]
    legacy_ranked = sorted(
        scored,
        key=lambda critique: (
            critique.overall_score,
            int(_artifact_by_id(artifacts, critique.artifact_id).preview_eligible),
            -critique.source_order,
        ),
        reverse=True,
    )
    legacy_rank_by_id = {
        critique.artifact_id: index
        for index, critique in enumerate(legacy_ranked, start=1)
    }
    ranked = sorted(
        scored,
        key=lambda critique: (
            _calibrated_rank_bucket(critique),
            -legacy_rank_by_id[critique.artifact_id],
            int(_artifact_by_id(artifacts, critique.artifact_id).preview_eligible),
            -critique.source_order,
        ),
        reverse=True,
    )
    recommended_id = ranked[0].artifact_id if ranked else None
    critique_by_id = {
        critique.artifact_id: critique.model_copy(
            update={
                "rank": index,
                "legacy_rank": legacy_rank_by_id[critique.artifact_id],
                "recommended": critique.artifact_id == recommended_id,
            }
        )
        for index, critique in enumerate(ranked, start=1)
    }
    annotated = tuple(
        artifact.model_copy(
            update={
                "critique": critique_by_id[artifact.id],
                "quality_score": critique_by_id[artifact.id].overall_score,
                "quality_rank": critique_by_id[artifact.id].rank,
                "is_recommended": critique_by_id[artifact.id].recommended,
                "is_default": critique_by_id[artifact.id].recommended,
                "refinement_reason": critique_by_id[artifact.id].refinement_guidance,
            }
        )
        for artifact in artifacts
    )
    ordered_critiques = tuple(
        critique_by_id[artifact.id]
        for artifact in sorted(annotated, key=lambda item: item.quality_rank or 999)
    )
    failed_critiques = tuple(
        critique for critique in ordered_critiques if not critique.passed
    )
    recommended_artifact = (
        _artifact_by_id(annotated, recommended_id) if recommended_id else annotated[0]
    )
    refinement_required = (
        recommended_artifact.critique is not None
        and not recommended_artifact.critique.passed
    )
    refinement_reasons = (
        recommended_artifact.critique.reasons if refinement_required else ()
    )
    refinement_guidance = (
        recommended_artifact.critique.refinement_guidance
        if refinement_required and recommended_artifact.critique is not None
        else None
    )

    return annotated, ArtifactCritiqueSummary(
        artifact_count=len(annotated),
        critiques=ordered_critiques,
        recommended_artifact_id=recommended_artifact.id,
        recommended_artifact_title=recommended_artifact.title,
        average_score=round(
            mean(critique.overall_score for critique in ordered_critiques),
            3,
        ),
        average_calibrated_score=round(
            mean(
                critique.calibrated_quality.score
                if critique.calibrated_quality is not None
                else critique.overall_score
                for critique in ordered_critiques
            ),
            3,
        ),
        failed_artifact_count=len(failed_critiques),
        refinement_required=refinement_required,
        refinement_reasons=refinement_reasons,
        refinement_guidance=refinement_guidance,
    )


def _score_artifact(
    artifact: WorkflowArtifact,
    *,
    request: AssistantRequest,
    route_decision: RouteDecision,
    preview_artifact_ids: set[str],
) -> WorkflowArtifactCritique:
    creative_evaluation = evaluate_artifact_creative_quality(artifact)
    sacred_consistency = evaluate_artifact_sacred_consistency(artifact)
    dimensions = {
        "prompt_alignment": _score_prompt_alignment(artifact, request),
        "creative_quality": _dimension(
            creative_evaluation.overall_score,
            creative_evaluation.summary,
        ),
        "runtime_suitability": _score_runtime_suitability(artifact),
        "code_quality": _score_code_quality(artifact),
        "preview_readiness": _score_preview_readiness(artifact, preview_artifact_ids),
        "domain_appropriateness": _score_domain_appropriateness(
            artifact,
            route_decision.domains or request.domains,
        ),
    }
    overall = round(
        (
            dimensions["prompt_alignment"].score * 0.2
            + dimensions["creative_quality"].score * 0.2
            + dimensions["runtime_suitability"].score * 0.15
            + dimensions["code_quality"].score * 0.2
            + dimensions["preview_readiness"].score * 0.15
            + dimensions["domain_appropriateness"].score * 0.1
        ),
        3,
    )
    reasons = _critique_reasons(
        dimensions,
        overall,
        sacred_consistency=sacred_consistency,
    )
    calibrated_quality = calibrate_artifact_quality(
        artifact,
        dimensions=dimensions,
        legacy_score=overall,
        creative_evaluation=creative_evaluation,
        sacred_consistency=sacred_consistency,
        reasons=reasons,
    )
    passed = (
        overall >= ARTIFACT_CRITIQUE_PASS_THRESHOLD
        and calibrated_quality.score >= ARTIFACT_CRITIQUE_PASS_THRESHOLD
        and not reasons
    )

    return WorkflowArtifactCritique(
        artifact_id=artifact.id,
        artifact_title=artifact.title,
        source_order=artifact.source_order,
        overall_score=overall,
        rank=1,
        passed=passed,
        prompt_alignment=dimensions["prompt_alignment"],
        creative_quality=dimensions["creative_quality"],
        runtime_suitability=dimensions["runtime_suitability"],
        code_quality=dimensions["code_quality"],
        preview_readiness=dimensions["preview_readiness"],
        domain_appropriateness=dimensions["domain_appropriateness"],
        creative_evaluation=creative_evaluation,
        sacred_consistency=sacred_consistency,
        calibrated_quality=calibrated_quality,
        reasons=reasons,
        rationale=_critique_rationale(
            artifact,
            overall,
            reasons,
            dimensions["domain_appropriateness"].rationale,
            calibrated_quality_score=calibrated_quality.score,
        ),
        refinement_guidance=_refinement_guidance(
            artifact,
            reasons,
            creative_evaluation,
            sacred_consistency,
        ),
    )


def _score_prompt_alignment(
    artifact: WorkflowArtifact,
    request: AssistantRequest,
) -> ArtifactCritiqueDimension:
    query_tokens = _tokens(request.query)
    artifact_tokens = _tokens(
        " ".join(
            [
                artifact.title,
                artifact.language,
                artifact.summary,
                artifact.runtime or "",
                artifact.content,
            ]
        )
    )
    overlap = len(query_tokens.intersection(artifact_tokens))
    score = 0.65 if not query_tokens else min(1.0, 0.45 + min(overlap, 5) * 0.11)
    if artifact.runtime and artifact.runtime in query_tokens:
        score = min(1.0, score + 0.15)
    if _query_matches_runtime(query_tokens, artifact.runtime):
        score = min(1.0, score + 0.25)
    return _dimension(score, f"{overlap} prompt token(s) matched artifact signals.")


def _score_runtime_suitability(artifact: WorkflowArtifact) -> ArtifactCritiqueDimension:
    if artifact.runtime and artifact.renderer_id:
        return _dimension(
            1.0,
            f"{artifact.runtime} runtime maps to {artifact.renderer_id}.",
        )
    if artifact.preview_target:
        return _dimension(
            0.72,
            "Preview target metadata exists without an executable renderer match.",
        )
    return _dimension(
        0.45,
        "No preview runtime or target metadata matched this artifact.",
    )


def _score_code_quality(artifact: WorkflowArtifact) -> ArtifactCritiqueDimension:
    content = artifact.content.strip()
    tokens = _tokens(content)
    marker_count = len(tokens.intersection(_CODE_MARKERS))
    brace_penalty = 0.18 if content.count("{") != content.count("}") else 0.0
    todo_penalty = 0.12 if "todo" in content.lower() else 0.0
    score = 0.58 + min(marker_count, 4) * 0.08 - brace_penalty - todo_penalty
    if len(content.splitlines()) >= 4:
        score += 0.1
    return _dimension(score, f"{marker_count} code structure signal(s) detected.")


def _score_preview_readiness(
    artifact: WorkflowArtifact,
    preview_artifact_ids: set[str],
) -> ArtifactCritiqueDimension:
    if artifact.id in preview_artifact_ids:
        return _dimension(1.0, "Preview preparation succeeded for this artifact.")
    if artifact.preview_eligible:
        return _dimension(
            0.72,
            "Preview eligible, but no prepared preview result was observed.",
        )
    return _dimension(0.32, "Artifact is inspectable as code but not previewable.")


def _score_domain_appropriateness(
    artifact: WorkflowArtifact,
    domains: tuple[CreativeCodingDomain, ...],
) -> ArtifactCritiqueDimension:
    artifact_domain = _artifact_domain(artifact)
    if not domains:
        if artifact_domain is None:
            return _dimension(
                0.75,
                "No domain was selected, so artifact domain fit is neutral.",
            )
        return _dimension(0.78, f"Artifact declares domain {artifact_domain.value}.")

    if artifact_domain in domains:
        support = get_domain_runtime_support(artifact_domain)
        if support is None:
            if artifact.preview_eligible:
                return _dimension(
                    0.48,
                    (
                        f"{artifact_domain.value} is code-only here, but the "
                        "artifact claims preview readiness."
                    ),
                )
            return _dimension(
                0.88,
                (
                    f"{artifact_domain.value} matches the requested domain and "
                    "is correctly code-only."
                ),
            )
        if (
            artifact.runtime == support.runtime
            and artifact.renderer_id == support.renderer_id
        ):
            return _dimension(
                1.0,
                (
                    f"{artifact.runtime} runtime matches requested domain "
                    f"{artifact_domain.value}."
                ),
            )
        return _dimension(
            0.55,
            (
                f"{artifact_domain.value} matches the requested domain, but "
                "runtime metadata is incomplete."
            ),
        )

    for domain in domains:
        support = get_domain_runtime_support(domain)
        if support is not None and artifact.runtime == support.runtime:
            return _dimension(
                0.74,
                (
                    f"{artifact.runtime} runtime fits {domain.value}, but "
                    "artifact domain metadata differs."
                ),
            )

    if artifact.preview_eligible and any(
        is_previewable_generation_domain(domain) for domain in domains
    ):
        return _dimension(
            0.58,
            "Previewable artifact does not directly match the requested domain set.",
        )
    return _dimension(0.38, "Artifact does not match the requested domain set.")


def _critique_reasons(
    dimensions: dict[str, ArtifactCritiqueDimension],
    overall: float,
    *,
    sacred_consistency: SacredConsistencyEvaluation | None = None,
) -> tuple[str, ...]:
    reasons = [name for name, dimension in dimensions.items() if dimension.score < 0.5]
    if overall < ARTIFACT_CRITIQUE_PASS_THRESHOLD:
        reasons.append("overall_quality_below_threshold")
    if sacred_consistency is not None:
        if sacred_consistency.claim_safety.level == "unsupported":
            reasons.append("sacred_claim_safety")
        if (
            sacred_consistency.overall_score < 0.55
            or sacred_consistency.alignment.level == "unsupported"
            or sacred_consistency.motif_consistency.level == "unsupported"
            or sacred_consistency.modality_coherence.level == "unsupported"
        ):
            reasons.append("sacred_consistency")
    return tuple(reasons)


def _critique_rationale(
    artifact: WorkflowArtifact,
    score: float,
    reasons: tuple[str, ...],
    domain_rationale: str,
    *,
    calibrated_quality_score: float | None = None,
) -> str:
    calibration_text = (
        f" Calibrated decision-support score: {calibrated_quality_score:.2f}."
        if calibrated_quality_score is not None
        else ""
    )
    if not reasons:
        return (
            f"{artifact.title} is a strong candidate with score {score:.2f}. "
            f"{domain_rationale}{calibration_text}"
        )
    return (
        f"{artifact.title} needs refinement for {', '.join(reasons)}. "
        f"{domain_rationale}{calibration_text}"
    )


def _refinement_guidance(
    artifact: WorkflowArtifact,
    reasons: tuple[str, ...],
    creative_evaluation: CreativeQualityEvaluation,
    sacred_consistency: SacredConsistencyEvaluation | None,
) -> str | None:
    if not reasons:
        return None
    creative_focus = " ".join(creative_evaluation.refinement_opportunities[:3])
    creative_guidance = f" Creative focus: {creative_focus}" if creative_focus else ""
    sacred_focus = (
        " ".join(sacred_consistency.refinement_opportunities[:3])
        if sacred_consistency is not None
        else ""
    )
    sacred_guidance = f" Sacred focus: {sacred_focus}" if sacred_focus else ""
    return (
        f"Revise {artifact.title}: address {', '.join(reasons)}, preserve the "
        "original brief, and return a complete runnable artifact when applicable."
        f"{creative_guidance}{sacred_guidance}"
    )


def _dimension(score: float, rationale: str) -> ArtifactCritiqueDimension:
    return ArtifactCritiqueDimension(
        score=round(min(max(score, 0.0), 1.0), 3),
        rationale=rationale,
    )


def _calibrated_rank_bucket(critique: WorkflowArtifactCritique) -> int:
    score = (
        critique.calibrated_quality.score
        if critique.calibrated_quality is not None
        else critique.overall_score
    )
    return int(score / CALIBRATED_RANK_STABILITY_BUCKET)


def _query_matches_runtime(query_tokens: set[str], runtime: str | None) -> bool:
    normalized_tokens = {token.strip(".,:;!?") for token in query_tokens}
    if runtime == "p5":
        return bool(
            normalized_tokens.intersection({"p5", "p5.js", "sketch", "sketches"})
        )
    if runtime == "three":
        return bool(normalized_tokens.intersection({"3d", "scene", "scenes", "three"}))
    if runtime == "glsl":
        return bool(
            normalized_tokens.intersection({"fragment", "glsl", "shader", "shaders"})
        )
    return False


def _artifact_domain(artifact: WorkflowArtifact) -> CreativeCodingDomain | None:
    if artifact.domain is None:
        return None
    return CreativeCodingDomain(artifact.domain)


def _artifact_by_id(
    artifacts: tuple[WorkflowArtifact, ...],
    artifact_id: str | None,
) -> WorkflowArtifact:
    if artifact_id is not None:
        for artifact in artifacts:
            if artifact.id == artifact_id:
                return artifact
    return artifacts[0]


def _tokens(value: str) -> set[str]:
    return _token_set(value)
