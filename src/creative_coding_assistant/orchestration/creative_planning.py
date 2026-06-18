"""Deterministic creative execution planning for assistant workflows."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import (
    AssistantArtifactRefinement,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeOutputModality,
    CreativeTranslation,
)
from creative_coding_assistant.orchestration.domain_generation import (
    DomainRuntimeSupport,
    get_domain_runtime_support,
)
from creative_coding_assistant.orchestration.routing import RouteDecision


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(item.capitalize() for item in tail)


ExpectedComplexity = Literal["low", "medium", "high"]
ExportReadiness = Literal["ready", "partial", "blocked"]


class CreativeExecutionPlan(BaseModel):
    """Structured deterministic plan produced before provider generation."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    output_modality: CreativeOutputModality
    generation_strategy: str = Field(min_length=1, max_length=360)
    recommended_runtime: str | None = None
    recommended_renderer_id: str | None = None
    recommended_preview_target: str | None = None
    recommended_shader_style: str | None = None
    candidate_count: int = Field(ge=1, le=3)
    refinement_budget: int = Field(ge=0, le=3)
    expected_complexity: ExpectedComplexity
    estimated_token_cost: int = Field(ge=500, le=12000)
    export_readiness: ExportReadiness
    runtime_available: bool
    runtime_support_summary: str = Field(min_length=1, max_length=240)
    plan_steps: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    constraints: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=10)


def derive_creative_execution_plan(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    retrieval_chunk_count: int = 0,
) -> CreativeExecutionPlan:
    """Build a deterministic execution plan from existing metadata only."""

    domains = _effective_domains(request, route_decision)
    modality = _resolve_modality(
        request=request,
        creative_translation=creative_translation,
        domains=domains,
    )
    support = _resolve_runtime_support(
        request=request,
        creative_translation=creative_translation,
        domains=domains,
    )
    shader_style = _recommended_shader_style(creative_translation)
    candidate_count = _candidate_count(request=request, domains=domains)
    complexity = _expected_complexity(
        request=request,
        creative_translation=creative_translation,
        candidate_count=candidate_count,
        retrieval_chunk_count=retrieval_chunk_count,
    )
    refinement_budget = _refinement_budget(
        request=request,
        creative_translation=creative_translation,
        complexity=complexity,
        candidate_count=candidate_count,
    )
    export_readiness = _export_readiness(
        modality=modality,
        runtime_support=support,
        route_decision=route_decision,
    )

    return CreativeExecutionPlan(
        output_modality=modality,
        generation_strategy=_generation_strategy(
            request=request,
            modality=modality,
            support=support,
            candidate_count=candidate_count,
            creative_translation=creative_translation,
        ),
        recommended_runtime=support.runtime if support is not None else None,
        recommended_renderer_id=support.renderer_id if support is not None else None,
        recommended_preview_target=(
            support.preview_target if support is not None else None
        ),
        recommended_shader_style=shader_style,
        candidate_count=candidate_count,
        refinement_budget=refinement_budget,
        expected_complexity=complexity,
        estimated_token_cost=_estimated_token_cost(
            creative_translation=creative_translation,
            candidate_count=candidate_count,
            refinement_budget=refinement_budget,
            retrieval_chunk_count=retrieval_chunk_count,
            has_attachments=bool(request.attachments),
        ),
        export_readiness=export_readiness,
        runtime_available=support is not None,
        runtime_support_summary=_runtime_support_summary(support, domains),
        plan_steps=_plan_steps(
            request=request,
            modality=modality,
            support=support,
            candidate_count=candidate_count,
            creative_translation=creative_translation,
        ),
        constraints=_constraints(
            request=request,
            creative_translation=creative_translation,
            support=support,
        ),
        evidence=_evidence(
            request=request,
            route_decision=route_decision,
            creative_translation=creative_translation,
            support=support,
            retrieval_chunk_count=retrieval_chunk_count,
        ),
    )


def creative_execution_plan_prompt_lines(
    plan: CreativeExecutionPlan,
) -> tuple[str, ...]:
    """Render a compact deterministic plan into provider prompt guidance."""

    lines = [
        f"Output modality: {plan.output_modality.value}.",
        f"Generation strategy: {plan.generation_strategy}",
        f"Candidate count: {plan.candidate_count}.",
        f"Refinement budget: {plan.refinement_budget}.",
        f"Expected complexity: {plan.expected_complexity}.",
        f"Estimated token cost: {plan.estimated_token_cost}.",
        f"Export readiness: {plan.export_readiness}.",
        f"Runtime availability: {plan.runtime_support_summary}",
    ]
    if plan.recommended_runtime is not None:
        lines.append(f"Recommended runtime: {plan.recommended_runtime}.")
    if plan.recommended_renderer_id is not None:
        lines.append(f"Recommended renderer: {plan.recommended_renderer_id}.")
    if plan.recommended_shader_style is not None:
        lines.append(f"Recommended shader/style: {plan.recommended_shader_style}.")
    lines.extend(f"Plan step: {step}" for step in plan.plan_steps)
    lines.extend(
        f"Planning constraint: {constraint}" for constraint in plan.constraints
    )
    return tuple(lines)


def _effective_domains(
    request: AssistantRequest,
    route_decision: RouteDecision | None,
) -> tuple[CreativeCodingDomain, ...]:
    domains = request.domains
    if domains:
        return domains
    if request.domain is not None:
        return (request.domain,)
    if route_decision is not None and route_decision.domains:
        return route_decision.domains
    if request.artifact_refinement and request.artifact_refinement.domain:
        return (request.artifact_refinement.domain,)
    return ()


def _resolve_modality(
    *,
    request: AssistantRequest,
    creative_translation: CreativeTranslation | None,
    domains: tuple[CreativeCodingDomain, ...],
) -> CreativeOutputModality:
    clarified = _modality_from_clarification(request.clarification_response)
    if clarified is not None:
        return clarified
    if creative_translation and creative_translation.output_modality is not None:
        return creative_translation.output_modality
    if _contains_audio_domain(domains):
        if _contains_visual_domain(domains):
            return CreativeOutputModality.AUDIOVISUAL
        return CreativeOutputModality.AUDIO
    return CreativeOutputModality.VISUAL


def _modality_from_clarification(value: str | None) -> CreativeOutputModality | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if "audiovisual" in normalized or (
        "audio" in normalized and "visual" in normalized
    ):
        return CreativeOutputModality.AUDIOVISUAL
    if "audio" in normalized or "sound" in normalized:
        return CreativeOutputModality.AUDIO
    if "visual" in normalized or "sketch" in normalized or "image" in normalized:
        return CreativeOutputModality.VISUAL
    return None


def _resolve_runtime_support(
    *,
    request: AssistantRequest,
    creative_translation: CreativeTranslation | None,
    domains: tuple[CreativeCodingDomain, ...],
) -> DomainRuntimeSupport | None:
    refinement = request.artifact_refinement
    if refinement is not None:
        support = _support_for_refinement(refinement)
        if support is not None:
            return support
    for domain in domains:
        support = get_domain_runtime_support(domain)
        if support is not None:
            return support
    if creative_translation is not None:
        for label in creative_translation.runtime_recommendations:
            support = _support_for_runtime_label(label)
            if support is not None:
                return support
    return None


def _support_for_refinement(
    refinement: AssistantArtifactRefinement,
) -> DomainRuntimeSupport | None:
    if refinement.domain is not None:
        support = get_domain_runtime_support(refinement.domain)
        if support is not None and support.runtime == refinement.runtime:
            return support
    return _support_for_runtime_label(refinement.runtime)


def _support_for_runtime_label(value: str | None) -> DomainRuntimeSupport | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    for domain in (
        CreativeCodingDomain.P5_JS,
        CreativeCodingDomain.GLSL,
        CreativeCodingDomain.THREE_JS,
        CreativeCodingDomain.REACT_THREE_FIBER,
    ):
        support = get_domain_runtime_support(domain)
        if support is None:
            continue
        if support.runtime in normalized or domain.value in normalized:
            return support
    return None


def _recommended_shader_style(
    creative_translation: CreativeTranslation | None,
) -> str | None:
    if creative_translation is None:
        return None
    if creative_translation.shader_presets is not None:
        presets = creative_translation.shader_presets.presets
        if presets:
            return _value_label(presets[0])
    if creative_translation.visual_style is not None:
        styles = creative_translation.visual_style.styles
        if styles:
            return _value_label(styles[0])
    if creative_translation.sacred_geometry is not None:
        concepts = creative_translation.sacred_geometry.concepts
        if concepts:
            return _value_label(concepts[0])
    return None


def _candidate_count(
    *,
    request: AssistantRequest,
    domains: tuple[CreativeCodingDomain, ...],
) -> int:
    if request.artifact_refinement is not None:
        return 1
    normalized = request.query.lower()
    if len(domains) >= 3:
        return 3
    if len(domains) == 2 or re.search(
        r"\b(compare|comparison|alternatives?|variations?|candidates?)\b",
        normalized,
    ):
        return 2
    return 1


def _expected_complexity(
    *,
    request: AssistantRequest,
    creative_translation: CreativeTranslation | None,
    candidate_count: int,
    retrieval_chunk_count: int,
) -> ExpectedComplexity:
    score = 0
    if candidate_count > 1:
        score += 2
    if request.attachments:
        score += 1
    if retrieval_chunk_count >= 3:
        score += 1
    if creative_translation is not None:
        score += int(creative_translation.reference_fusion is not None)
        score += int(creative_translation.sacred_geometry is not None)
        score += int(creative_translation.audio_reactive is not None)
        score += int(creative_translation.shader_presets is not None)
    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def _refinement_budget(
    *,
    request: AssistantRequest,
    creative_translation: CreativeTranslation | None,
    complexity: ExpectedComplexity,
    candidate_count: int,
) -> int:
    if request.artifact_refinement is not None:
        return min(request.artifact_refinement.max_passes or 1, 3)
    if creative_translation and len(creative_translation.refinement_targets) >= 3:
        return 2
    if complexity == "high" or candidate_count > 1:
        return 2
    if complexity == "medium":
        return 1
    return 1


def _export_readiness(
    *,
    modality: CreativeOutputModality,
    runtime_support: DomainRuntimeSupport | None,
    route_decision: RouteDecision | None,
) -> ExportReadiness:
    if route_decision is None:
        return "partial"
    if modality is CreativeOutputModality.AUDIO and runtime_support is None:
        return "partial"
    if runtime_support is not None:
        return "ready"
    return "partial"


def _generation_strategy(
    *,
    request: AssistantRequest,
    modality: CreativeOutputModality,
    support: DomainRuntimeSupport | None,
    candidate_count: int,
    creative_translation: CreativeTranslation | None,
) -> str:
    action = (
        "Refine the selected artifact" if request.artifact_refinement else "Generate"
    )
    target = (
        f"{candidate_count} candidate"
        if candidate_count == 1
        else f"{candidate_count} candidates"
    )
    runtime = support.label if support is not None else "a code-only creative surface"
    intent = (
        creative_translation.creative_intent
        if creative_translation is not None
        else "the user request"
    )
    return (
        f"{action} {target} for a {modality.value} output using {runtime}; "
        f"optimize for {intent}."
    )


def _estimated_token_cost(
    *,
    creative_translation: CreativeTranslation | None,
    candidate_count: int,
    refinement_budget: int,
    retrieval_chunk_count: int,
    has_attachments: bool,
) -> int:
    cost = 1200 + candidate_count * 850 + refinement_budget * 450
    cost += retrieval_chunk_count * 180
    if has_attachments:
        cost += 500
    if creative_translation is not None:
        cost += len(creative_translation.generation_constraints) * 80
        cost += len(creative_translation.refinement_targets) * 80
        cost += 350 if creative_translation.reference_fusion is not None else 0
        cost += 250 if creative_translation.sacred_geometry is not None else 0
        cost += 250 if creative_translation.audio_reactive is not None else 0
    return min(max(cost, 500), 12000)


def _runtime_support_summary(
    support: DomainRuntimeSupport | None,
    domains: tuple[CreativeCodingDomain, ...],
) -> str:
    if support is not None:
        return f"{support.label} is available for {support.domain.value}."
    if domains:
        joined = ", ".join(domain.value for domain in domains)
        return f"No live preview runtime is available for {joined}; plan as code-only."
    return "No explicit runtime support selected; plan conservatively."


def _plan_steps(
    *,
    request: AssistantRequest,
    modality: CreativeOutputModality,
    support: DomainRuntimeSupport | None,
    candidate_count: int,
    creative_translation: CreativeTranslation | None,
) -> tuple[str, ...]:
    steps = [
        "Use the translated creative intent as the source of truth.",
        f"Produce {candidate_count} {modality.value} candidate"
        f"{'' if candidate_count == 1 else 's'} without planning-time code.",
    ]
    if request.clarification_response:
        steps.append(f"Honor clarification answer: {request.clarification_response}.")
    if creative_translation and creative_translation.reference_fusion is not None:
        steps.append("Preserve reference fusion palette, composition, and safety cues.")
    if creative_translation and creative_translation.sacred_geometry is not None:
        steps.append("Carry sacred geometry structure into the generated composition.")
    if creative_translation and creative_translation.audio_reactive is not None:
        steps.append("Map audio-reactive signals only where runtime support allows it.")
    if support is not None:
        steps.append(
            f"Target {support.runtime} output compatible with {support.renderer_id}."
        )
    return tuple(steps[:6])


def _constraints(
    *,
    request: AssistantRequest,
    creative_translation: CreativeTranslation | None,
    support: DomainRuntimeSupport | None,
) -> tuple[str, ...]:
    constraints: list[str] = []
    if creative_translation is not None:
        constraints.extend(creative_translation.generation_constraints)
        if creative_translation.reference_fusion is not None:
            constraints.extend(creative_translation.reference_fusion.safety_constraints)
    if request.artifact_refinement is not None:
        constraints.append("Preserve selected artifact lineage during refinement.")
    if support is not None:
        constraints.append(
            f"Keep generated code compatible with {support.renderer_id}."
        )
    else:
        constraints.append(
            "Do not claim live preview support for unsupported runtime output."
        )
    return _dedupe_text(constraints)[:8]


def _evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    support: DomainRuntimeSupport | None,
    retrieval_chunk_count: int,
) -> tuple[str, ...]:
    evidence: list[str] = []
    if route_decision is not None:
        evidence.append(f"Route selected: {route_decision.route.value}.")
    if request.clarification_response:
        evidence.append(f"Clarification answer: {request.clarification_response}.")
    if creative_translation is not None:
        evidence.append(f"Creative intent: {creative_translation.creative_intent}.")
        if creative_translation.reference_fusion is not None:
            evidence.append(
                "Reference fusion: "
                f"{creative_translation.reference_fusion.source_count} source(s)."
            )
        if creative_translation.sacred_geometry is not None:
            evidence.append("Sacred geometry guidance present.")
        if creative_translation.audio_reactive is not None:
            evidence.append("Audio-reactive mapping guidance present.")
    if support is not None:
        evidence.append(f"Runtime support: {support.label}.")
    if retrieval_chunk_count:
        evidence.append(f"Retrieval context: {retrieval_chunk_count} chunk(s).")
    return tuple(evidence[:10])


def _contains_audio_domain(domains: tuple[CreativeCodingDomain, ...]) -> bool:
    return any(
        "audio" in domain.value or domain in _AUDIO_DOMAINS for domain in domains
    )


def _contains_visual_domain(domains: tuple[CreativeCodingDomain, ...]) -> bool:
    return any(domain not in _AUDIO_DOMAINS for domain in domains)


def _dedupe_text(values: list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        cleaned = " ".join(value.strip().split())
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)


def _value_label(value: object) -> str:
    return str(getattr(value, "value", value))


_AUDIO_DOMAINS = frozenset(
    {
        CreativeCodingDomain.ABLETON_LIVE,
        CreativeCodingDomain.MAX_MSP,
        CreativeCodingDomain.P5_SOUND,
        CreativeCodingDomain.PURE_DATA,
        CreativeCodingDomain.SONIC_PI,
        CreativeCodingDomain.SUPERCOLLIDER,
        CreativeCodingDomain.TIDALCYCLES,
        CreativeCodingDomain.TONE_JS,
        CreativeCodingDomain.VCV_RACK,
        CreativeCodingDomain.WEB_AUDIO_API,
    }
)
