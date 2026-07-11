"""Deterministic review checks for assistant workflow outputs."""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantMode, AssistantRequest
from creative_coding_assistant.orchestration._metadata_utils import _token_set
from creative_coding_assistant.orchestration.artifact_critique import (
    ArtifactCritiqueSummary,
)
from creative_coding_assistant.orchestration.artifacts import WorkflowArtifact
from creative_coding_assistant.orchestration.metadata.domain_generation import (
    is_previewable_generation_domain,
)
from creative_coding_assistant.orchestration.refinement_passes import (
    DEFAULT_REFINEMENT_PASS_LIMIT,
)
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName

MAX_WORKFLOW_REFINEMENT_COUNT = DEFAULT_REFINEMENT_PASS_LIMIT

_MIN_ANSWER_CHARS = 8

_CODE_REQUEST_MARKERS = {
    "code",
    "snippet",
    "implementation",
    "implement",
    "function",
    "component",
    "shader",
    "sketch",
    "glsl",
    "wgsl",
    "javascript",
    "typescript",
    "jsx",
    "tsx",
}
_EXPLANATION_MARKERS = {
    "because",
    "means",
    "mean",
    "works",
    "work",
    "use",
    "uses",
    "step",
    "steps",
    "explain",
    "reason",
    "when",
    "how",
}
_DEBUG_MARKERS = {
    "bug",
    "cause",
    "check",
    "debug",
    "error",
    "fix",
    "issue",
    "line",
    "problem",
    "trace",
}
_FAILURE_MARKERS = (
    "generation failed",
    "provider failed",
    "model failed",
)


class WorkflowReviewOutcome(StrEnum):
    PASS = "pass"
    NEEDS_REFINEMENT = "needs_refinement"


class WorkflowReviewResult(BaseModel):
    """Structured result for the workflow quality gate."""

    model_config = ConfigDict(frozen=True)

    outcome: WorkflowReviewOutcome
    reasons: tuple[str, ...] = ()
    refinement_count: int = Field(default=0, ge=0)
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)

    @property
    def passed(self) -> bool:
        return self.outcome is WorkflowReviewOutcome.PASS


def review_assistant_answer(
    *,
    request: AssistantRequest,
    answer: str | None,
    refinement_count: int,
    artifact_critique_summary: ArtifactCritiqueSummary | None = None,
    artifacts: Sequence[WorkflowArtifact] | None = None,
    route_decision: RouteDecision | None = None,
) -> WorkflowReviewResult:
    """Run conservative deterministic checks on a generated answer."""

    normalized_answer = (answer or "").strip()
    reasons: list[str] = []

    if not normalized_answer:
        reasons.append("missing_answer")
    else:
        _collect_answer_quality_reasons(
            request=request,
            answer=normalized_answer,
            reasons=reasons,
            artifacts=artifacts,
            route_decision=route_decision,
        )
    if (
        artifact_critique_summary
        and artifact_critique_summary.refinement_required
        and refinement_count < MAX_WORKFLOW_REFINEMENT_COUNT
    ):
        reasons.append("artifact_quality_below_threshold")

    outcome = (
        WorkflowReviewOutcome.NEEDS_REFINEMENT
        if reasons
        else WorkflowReviewOutcome.PASS
    )
    return WorkflowReviewResult(
        outcome=outcome,
        reasons=tuple(reasons),
        refinement_count=refinement_count,
        score=_score_review(reasons, artifact_critique_summary),
        rationale=_review_rationale(reasons, artifact_critique_summary),
    )


def _collect_answer_quality_reasons(
    *,
    request: AssistantRequest,
    answer: str,
    reasons: list[str],
    artifacts: Sequence[WorkflowArtifact] | None,
    route_decision: RouteDecision | None,
) -> None:
    answer_lower = answer.lower()
    answer_tokens = _tokens(answer)

    if any(marker in answer_lower for marker in _FAILURE_MARKERS):
        reasons.append("generation_failure")

    if len(answer) < _MIN_ANSWER_CHARS:
        reasons.append("answer_too_short")

    requires_deliverable = request_requires_deliverable(request, route_decision)
    if request_requires_code_block(request, route_decision):
        fence_count = answer.count("```")
        if fence_count == 0:
            reasons.append("missing_code_block")
        elif fence_count % 2:
            reasons.append("unterminated_code_block")

    if artifacts is not None and requires_deliverable:
        if not artifacts and not any(
            reason in {"missing_code_block", "unterminated_code_block"}
            for reason in reasons
        ):
            reasons.append("missing_requested_artifact")
        elif artifacts and request_requires_live_preview(request, route_decision) and not any(
            artifact.preview_eligible for artifact in artifacts
        ):
            reasons.append("missing_runnable_artifact")

    if request.mode is AssistantMode.EXPLAIN and not answer_tokens.intersection(
        _EXPLANATION_MARKERS
    ):
        reasons.append("missing_explanation")

    if request.mode is AssistantMode.DEBUG and not answer_tokens.intersection(
        _DEBUG_MARKERS
    ):
        reasons.append("missing_debug_guidance")


def _request_explicitly_asks_for_code(request: AssistantRequest) -> bool:
    query_tokens = _tokens(request.query)
    return bool(query_tokens.intersection(_CODE_REQUEST_MARKERS))


def _request_explicitly_asks_for_markdown_artifact(
    request: AssistantRequest,
) -> bool:
    return bool(
        "markdown" in _tokens(request.query)
        and re.search(r"\b[A-Za-z0-9][A-Za-z0-9._-]*\.md\b", request.query)
    )


def request_requires_deliverable(
    request: AssistantRequest,
    route_decision: RouteDecision | None,
) -> bool:
    """Return whether the current request needs an artifact to pass review.

    Explain-mode requests can discuss a sketch, shader, or component without
    asking the assistant to generate one. Treating those domain words as a
    deliverable requirement incorrectly sends a usable explanatory response
    through bounded code refinements and then a terminal product failure.
    """

    if request.mode is AssistantMode.EXPLAIN or (
        route_decision is not None and route_decision.route is RouteName.EXPLAIN
    ):
        return False

    return (
        _request_explicitly_asks_for_markdown_artifact(request)
        or _request_explicitly_asks_for_code(request)
        or request_requires_live_preview(
            request,
            route_decision,
        )
    )


def request_requires_code_block(
    request: AssistantRequest,
    route_decision: RouteDecision | None,
) -> bool:
    if _request_explicitly_asks_for_markdown_artifact(request):
        return False
    return request_requires_deliverable(request, route_decision)


def request_requires_live_preview(
    request: AssistantRequest,
    route_decision: RouteDecision | None,
) -> bool:
    domains = (
        route_decision.domains
        if route_decision is not None
        else (request.domain, *request.domains)
    )
    return (
        request_requests_preview(request)
        and any(
            domain is not None and is_previewable_generation_domain(domain)
            for domain in domains
        )
    )


def request_requests_preview(request: AssistantRequest) -> bool:
    """Return whether the user explicitly asked for browser or runnable output."""

    return bool(
        _tokens(request.query).intersection(
            {
                "browser",
                "browser-ready",
                "browser-safe",
                "browser_safe",
                "preview",
                "runnable",
            }
        )
    )


def _tokens(value: str) -> set[str]:
    return _token_set(value)


def _score_review(
    reasons: list[str],
    artifact_critique_summary: ArtifactCritiqueSummary | None,
) -> float:
    answer_score = 1.0 if not reasons else max(0.0, 1.0 - (0.25 * len(reasons)))
    if (
        artifact_critique_summary is None
        or artifact_critique_summary.artifact_count == 0
    ):
        return answer_score
    return round(min(answer_score, artifact_critique_summary.average_score), 3)


def _review_rationale(
    reasons: list[str],
    artifact_critique_summary: ArtifactCritiqueSummary | None,
) -> str:
    if not reasons:
        if (
            artifact_critique_summary
            and artifact_critique_summary.recommended_artifact_title
        ):
            return (
                "Deterministic review passed; recommended artifact is "
                f"{artifact_critique_summary.recommended_artifact_title}."
            )
        return "Deterministic review passed without quality gate findings."
    guidance = (
        f" {artifact_critique_summary.refinement_guidance}"
        if artifact_critique_summary and artifact_critique_summary.refinement_guidance
        else ""
    )
    return (
        "Deterministic review requested refinement: "
        + ", ".join(reasons)
        + "."
        + guidance
    )
