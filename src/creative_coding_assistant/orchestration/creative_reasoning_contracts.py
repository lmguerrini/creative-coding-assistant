"""Contracts for the bounded Creative Reasoning Engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ReasoningStage = Literal[
    "strategy",
    "technique",
    "runtime",
    "tradeoff",
    "recommendation",
]
ReasoningEvidenceSource = Literal[
    "request",
    "translation",
    "creative_intent",
    "creative_hierarchy",
    "planning",
    "director",
    "constraint_solver",
    "constraint_prioritizer",
    "creative_strategy",
    "creative_technique",
    "runtime_capability",
    "tradeoff_explorer",
    "quality_predictor",
    "symbolic_narrative",
    "creative_composition",
    "procedural_structure",
    "generative_structure",
    "semantic_motif",
    "emotional_consistency",
    "cross_modality",
    "audio_visual_scene",
    "artifact_plan",
    "artifact_dependency_graph",
    "runtime_compatibility",
    "artifact_capability_matrix",
    "multi_artifact_strategy",
    "artifact_critic",
    "future_knowledge",
]

CREATIVE_REASONING_AUTHORITY_BOUNDARY = (
    "The Creative Reasoning Engine synthesizes inspectable guidance only; it "
    "does not generate code, select artifacts, auto-select runtimes, route "
    "providers or models, run autonomous loops, change preview or runtime "
    "behavior, or implement HoloMind."
)


class CreativeReasoningStep(BaseModel):
    """One explicit link in the creative reasoning chain."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    stage: ReasoningStage
    claim: str = Field(min_length=1, max_length=520)
    because: str = Field(min_length=1, max_length=360)
    implications: tuple[str, ...] = Field(min_length=1, max_length=4)


class CreativeReasoningEvidence(BaseModel):
    """Interpreted evidence used by the reasoning engine."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source: ReasoningEvidenceSource
    signal: str = Field(min_length=1, max_length=240)
    interpretation: str = Field(min_length=1, max_length=360)


class CreativeRejectedAlternative(BaseModel):
    """A direction intentionally not leading the recommendation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    alternative: str = Field(min_length=1, max_length=180)
    reason: str = Field(min_length=1, max_length=320)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=4)


class CreativeReasoningResult(BaseModel):
    """Synthesis result that turns metadata into one explainable direction."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_reasoning_engine"] = "creative_reasoning_engine"
    recommended_creative_direction: str = Field(min_length=1, max_length=520)
    reasoning_path: tuple[CreativeReasoningStep, ...] = Field(
        min_length=5,
        max_length=5,
    )
    evidence_chain: tuple[CreativeReasoningEvidence, ...] = Field(
        min_length=3,
        max_length=29,
    )
    strongest_supporting_signals: tuple[str, ...] = Field(min_length=1, max_length=8)
    rejected_alternatives: tuple[CreativeRejectedAlternative, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    unresolved_decisions: tuple[str, ...] = Field(min_length=1, max_length=6)
    implementation_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    future_knowledge_context: dict[str, object] = Field(default_factory=dict)
    authority_boundary: str = Field(
        default=CREATIVE_REASONING_AUTHORITY_BOUNDARY,
        max_length=520,
    )
