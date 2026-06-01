"""Deterministic review checks for assistant workflow outputs."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantMode, AssistantRequest

MAX_WORKFLOW_REFINEMENT_COUNT = 1

_MIN_ANSWER_CHARS = 8
_TOKEN_PATTERN = re.compile(r"[a-z0-9_.+#-]+")

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
        )

    outcome = (
        WorkflowReviewOutcome.NEEDS_REFINEMENT
        if reasons
        else WorkflowReviewOutcome.PASS
    )
    return WorkflowReviewResult(
        outcome=outcome,
        reasons=tuple(reasons),
        refinement_count=refinement_count,
        score=_score_review(reasons),
        rationale=_review_rationale(reasons),
    )


def _collect_answer_quality_reasons(
    *,
    request: AssistantRequest,
    answer: str,
    reasons: list[str],
) -> None:
    answer_lower = answer.lower()
    answer_tokens = _tokens(answer)

    if any(marker in answer_lower for marker in _FAILURE_MARKERS):
        reasons.append("generation_failure")

    if len(answer) < _MIN_ANSWER_CHARS:
        reasons.append("answer_too_short")

    if _request_explicitly_asks_for_code(request) and "```" not in answer:
        reasons.append("missing_code_block")

    if (
        request.mode is AssistantMode.EXPLAIN
        and not answer_tokens.intersection(_EXPLANATION_MARKERS)
    ):
        reasons.append("missing_explanation")

    if (
        request.mode is AssistantMode.DEBUG
        and not answer_tokens.intersection(_DEBUG_MARKERS)
    ):
        reasons.append("missing_debug_guidance")


def _request_explicitly_asks_for_code(request: AssistantRequest) -> bool:
    query_tokens = _tokens(request.query)
    return bool(query_tokens.intersection(_CODE_REQUEST_MARKERS))


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(value.lower()))


def _score_review(reasons: list[str]) -> float:
    if not reasons:
        return 1.0
    return max(0.0, 1.0 - (0.25 * len(reasons)))


def _review_rationale(reasons: list[str]) -> str:
    if not reasons:
        return "Deterministic review passed without quality gate findings."
    return "Deterministic review requested refinement: " + ", ".join(reasons) + "."
