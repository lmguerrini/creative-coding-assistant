"""Delivery and bundle helpers for V8.7 HoloGenesis reports."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.knowledge.hologenesis_core_contracts import (
    HoloGenesisProjectBundle,
    HoloGenesisValidationFinding,
    HoloGenesisValidationSeverity,
)
from creative_coding_assistant.knowledge.immersive_audiovisual_composer_contracts import (
    ImmersiveAudiovisualComposerReport,
)
from creative_coding_assistant.knowledge.mythopoetic_narrative_contracts import (
    MythopoeticNarrativeReport,
)
from creative_coding_assistant.knowledge.sacred_architecture_contracts import (
    SacredArchitectureReport,
)
from creative_coding_assistant.knowledge.sacred_geometry_contracts import SacredGeometryReport
from creative_coding_assistant.orchestration.creative_planning import CreativeExecutionPlan
from creative_coding_assistant.orchestration.creative_translation import CreativeTranslation


def build_hologenesis_project_bundle(
    *,
    source_query: str,
    domains: Sequence[CreativeCodingDomain],
    translation: CreativeTranslation,
    execution_plan: CreativeExecutionPlan,
    geometry: SacredGeometryReport,
    architecture: SacredArchitectureReport,
    narrative: MythopoeticNarrativeReport,
    composer: ImmersiveAudiovisualComposerReport,
) -> HoloGenesisProjectBundle:
    """Build an in-memory project bundle outline without writing files."""

    _ = (geometry, architecture, narrative, composer)
    runtime = execution_plan.recommended_runtime or "browser-internal preview runtime"
    domain_names = tuple(domain.value for domain in domains) or ("three_js", "p5_js", "glsl")
    title = _project_title(source_query)
    return HoloGenesisProjectBundle(
        project_title=title,
        project_summary=_clip(
            f"{title} unifies symbolic translation, sacred geometry, spatial architecture, "
            f"mythopoetic narrative, and immersive audiovisual composition into a bounded "
            "installation planning bundle.",
            820,
        ),
        architecture_outline=(
            "System boundary: deterministic report/planning layer over V8.1-V8.6 outputs.",
            f"Primary runtime plan: {runtime}; external tools remain handoff planning only.",
            "Graph layer: symbolic, sacred knowledge, geometry, narrative, and installation projections.",
            "Validation layer: curatorial reasoning, aesthetic evaluation, readiness scores, and HITL questions.",
        ),
        portfolio_outline=(
            f"Lead with the installation thesis: {translation.creative_intent}",
            "Show graph unification, scene flow, and audience journey as the project story.",
            "Separate implemented browser/report behavior from future external production handoff.",
        ),
        readme_outline=(
            "Overview and authority boundary.",
            "Inputs and reused V8.1-V8.6 engines.",
            "Generated graph, planner, curatorial, readiness, and bundle sections.",
            "External integration audit and deferred execution boundaries.",
            "Focused validation evidence.",
        ),
        capstone_outputs=(
            "Typed HoloGenesis report model.",
            "Deterministic V8.7 builder and prompt lines.",
            "Roadmap coverage and external integration audit.",
            "Focused unit tests and architecture documentation.",
        ),
        pipeline_steps=(
            "Gather V8.1-V8.6 reports.",
            "Build unified graphs and blackboard entries.",
            "Schedule symbolic composition and creative planning.",
            "Evaluate curatorial consistency and readiness.",
            "Generate bundle outlines and export-planning notes.",
        ),
        workflow_recommendations=(
            "Use browser-internal preview foundations before manual external production.",
            "Run human curatorial review before public museum or exhibition claims.",
            "Keep research and reference discovery as planned follow-up unless live browsing is explicitly approved.",
        ),
        tool_recommendations=_dedupe(
            (
                *domain_names[:5],
                "browser preview sandbox",
                "manual Unity/Unreal/TouchDesigner/Blender/Houdini handoff planning",
            )
        ),
        tech_stack_recommendations=_dedupe(
            (
                runtime,
                *(execution_plan.recommended_preview_target or "preview target").split(", ")[:2],
                "typed Python/Pydantic backend contracts",
                "Next.js preview foundations for implemented browser validation",
            )
        ),
        tradeoff_analysis=(
            "Higher external production ambition increases manual handoff cost "
            "because V8.7 does not execute DCC tools.",
            "Stronger symbolic/mystical framing requires stricter non-authoritative claim boundaries.",
            "Museum and international readiness need human production, legal, access, and institutional review.",
        ),
        research_mode_plan=(
            "Frame follow-up research questions around source provenance, cultural "
            "context, venue constraints, and materials.",
            "Identify reference categories without fetching or storing sources.",
            "Defer live source discovery to explicit user-approved research tooling.",
        ),
        reference_discovery_queries=(
            f"{title} interactive installation reference",
            f"{title} sacred geometry audiovisual installation",
            f"{title} museum readiness accessibility production checklist",
            "browser-based immersive audiovisual preview case studies",
        ),
    )


def build_hologenesis_validation_findings(
    unsupported_claim_risks: Sequence[str],
) -> tuple[HoloGenesisValidationFinding, ...]:
    """Build deterministic validation findings for V8.7 reports."""

    findings = [
        HoloGenesisValidationFinding(
            finding_id="validation::report_boundary",
            severity=HoloGenesisValidationSeverity.INFO,
            summary="V8.7 is implemented as bounded report/planning behavior.",
            action=(
                "Keep final and public claims scoped to typed reports, "
                "recommendations, readiness scoring, and bundle outlines."
            ),
        ),
        HoloGenesisValidationFinding(
            finding_id="validation::external_integrations",
            severity=HoloGenesisValidationSeverity.WARNING,
            summary="External DCC and MCP layers are export-planning-only.",
            action="Do not claim live Unity, Unreal, TouchDesigner, Blender, Houdini, or MCP execution.",
        ),
    ]
    if unsupported_claim_risks:
        findings.append(
            HoloGenesisValidationFinding(
                finding_id="validation::unsupported_claims",
                severity=HoloGenesisValidationSeverity.HITL_REQUIRED,
                summary=(
                    "The request contains unsupported live-integration, HoloMind, "
                    "HOLOiVERSE, or authority claim language."
                ),
                action="Review and reframe unsupported claims before public use.",
            )
        )
    return tuple(findings)


def hologenesis_composition_audit_summary() -> tuple[str, ...]:
    """Return the V8.1-V8.6 composition audit summary for V8.7."""

    return (
        "V8.1 is reused for creative knowledge, provenance, confidence, and KB reality.",
        "V8.2 is reused for symbolic graph signals, symbolic explainability, and interpretation boundaries.",
        "V8.3 is reused for geometry graph signals, harmonic guidance, and geometry-to-media mappings.",
        "V8.4 is reused for architecture, spatial topology, installation flow, and audience path planning.",
        "V8.5 is reused for narrative graph signals, scene sequence, creative brief, and audience communication.",
        "V8.6 is reused for immersive scene graph, visual language, spatial audio, "
        "preview audit, and artistic decisions.",
        "V8.7 adds unification, curatorial reasoning, readiness scoring, recommendations, and bundle outlines only.",
        "External DCC and MCP layers are audited as handoff/export-planning behavior, not live integrations.",
    )


def _project_title(query: str) -> str:
    tokens = [token.strip(".,:;!?()[]{}").capitalize() for token in query.split() if token.strip()]
    selected = [token for token in tokens[:8] if token.lower() not in {"build", "create", "compose", "an", "a", "the"}]
    return " ".join(selected[:5]) or "HoloGenesis Project"


def _clip(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "."


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return tuple(result)
