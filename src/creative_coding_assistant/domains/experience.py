"""Canonical, public-safe capability contracts for creative-coding domains.

The routing registry owns names and prompt guidance. This module owns the
additional product signal needed for a safe decision: whether an
artifact is generated as browser-preview code, code to export, or an external
tool handoff.  It intentionally does not infer support from a filename or a
client-side adapter.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.domains.registry import SUPPORTED_DOMAINS, DomainCategory
from creative_coding_assistant.rag import approved_sources_for_domain


class DomainDeliveryKind(StrEnum):
    """The product-level delivery boundary for a domain."""

    BROWSER_PREVIEW = "browser_preview"
    CODE_EXPORT = "code_export"
    EXTERNAL_HANDOFF = "external_handoff"


class DomainValidationStatus(StrEnum):
    """A bounded statement about the currently supported product contract."""

    VALIDATED_BROWSER_CONTRACT = "validated_browser_contract"
    VALIDATED_CODE_EXPORT = "validated_code_export"
    HANDOFF_PACKAGE = "handoff_package"


class DomainExperienceRecord(BaseModel):
    """One canonical public product contract for a registered domain."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    domain: CreativeCodingDomain
    display_name: str = Field(min_length=1)
    aliases: tuple[str, ...] = Field(min_length=1)
    intent_triggers: tuple[str, ...] = Field(min_length=1)
    knowledge_source_ids: tuple[str, ...] = Field(default_factory=tuple)
    workflow_compatibility: tuple[str, ...] = Field(min_length=1)
    artifact_types: tuple[str, ...] = Field(min_length=1)
    filename_extensions: tuple[str, ...] = Field(min_length=1)
    delivery_kind: DomainDeliveryKind
    live_preview: bool
    runtime_requirements: tuple[str, ...] = Field(min_length=1)
    validation_status: DomainValidationStatus
    fallback: str = Field(min_length=1)
    public_claim_boundary: str = Field(min_length=1)
    demo_eligible: bool


_BROWSER_CONTRACTS: dict[CreativeCodingDomain, dict[str, object]] = {
    CreativeCodingDomain.P5_JS: {
        "artifact_types": ("global-mode JavaScript sketch",),
        "extensions": (".p5.js",),
        "requirements": (
            "One self-contained global-mode sketch with setup() and draw().",
            "Only the bounded p5 browser API is accepted by the preview validator.",
        ),
        "fallback": (
            "Open or download the code artifact when the bounded preview rejects "
            "the source."
        ),
        "boundary": (
            "p5.js is a live browser-preview domain only when the generated source "
            "passes the controlled global-mode contract."
        ),
        "demo_eligible": True,
    },
    CreativeCodingDomain.THREE_JS: {
        "artifact_types": ("browser-oriented JavaScript scene",),
        "extensions": (".three.js",),
        "requirements": (
            "One compact JavaScript scene with no HTML document, CDN import, or "
            "React wrapper.",
            "The controlled Three.js runtime supplies the canvas and animation surface.",
        ),
        "fallback": (
            "Open or download the scene source if it does not pass the controlled "
            "renderer contract."
        ),
        "boundary": (
            "Three.js is live-previewed only through the controlled JavaScript scene "
            "runtime, not as standalone HTML or a React component."
        ),
        "demo_eligible": True,
    },
    CreativeCodingDomain.GLSL: {
        "artifact_types": ("bounded WebGL fragment shader",),
        "extensions": (".frag", ".glsl"),
        "requirements": (
            "A fragment shader using the bounded WebGL source subset.",
            "Textures, samplers, #version directives, discard, and while loops are "
            "outside the current browser contract.",
        ),
        "fallback": (
            "Open or download the shader source when its features exceed the bounded "
            "WebGL runtime."
        ),
        "boundary": (
            "GLSL live preview is limited to the validated fragment-shader subset; it "
            "is not a general GPU or Shadertoy runtime."
        ),
        "demo_eligible": True,
    },
    CreativeCodingDomain.TONE_JS: {
        "artifact_types": ("bounded Tone.js program",),
        "extensions": (".tone.js", ".tone.ts"),
        "requirements": (
            "One self-contained Tone.js program with a supported synth, oscillator, or noise voice.",
            "The controlled audio runtime remains muted until the operator explicitly starts playback.",
        ),
        "fallback": (
            "Inspect or download the source when it falls outside the controlled Tone.js parser contract."
        ),
        "boundary": (
            "Tone.js is a live browser-preview domain only for parsed bounded "
            "programs; no microphone access or autoplay is claimed."
        ),
        "demo_eligible": True,
    },
}

_EXTERNAL_HANDOFF_DOMAINS = frozenset(
    {
        CreativeCodingDomain.TOUCHDESIGNER,
        CreativeCodingDomain.HOUDINI,
        CreativeCodingDomain.BLENDER_GEOMETRY_NODES,
        CreativeCodingDomain.UNITY,
        CreativeCodingDomain.UNREAL,
        CreativeCodingDomain.MAX_MSP,
        CreativeCodingDomain.NOTCH,
        CreativeCodingDomain.VVVV,
        CreativeCodingDomain.OPENFRAMEWORKS,
        CreativeCodingDomain.OPENRNDR,
        CreativeCodingDomain.SUPERCOLLIDER,
        CreativeCodingDomain.SONIC_PI,
        CreativeCodingDomain.TIDALCYCLES,
        CreativeCodingDomain.COMFYUI,
        CreativeCodingDomain.STABLE_DIFFUSION_WORKFLOWS,
        CreativeCodingDomain.RUNWAY,
        CreativeCodingDomain.BLENDER_PYTHON_API,
        CreativeCodingDomain.UNREAL_BLUEPRINTS,
        CreativeCodingDomain.ABLETON_LIVE,
        CreativeCodingDomain.VCV_RACK,
        CreativeCodingDomain.GODOT,
        CreativeCodingDomain.RESOLUME,
        CreativeCodingDomain.MADMAPPER,
        CreativeCodingDomain.CABLES_GL,
        CreativeCodingDomain.PURE_DATA,
    }
)

_EXTENSIONS: dict[CreativeCodingDomain, tuple[str, ...]] = {
    CreativeCodingDomain.REACT_THREE_FIBER: (".r3f.tsx", ".tsx"),
    CreativeCodingDomain.HYDRA: (".hydra.js",),
    CreativeCodingDomain.PROCESSING: (".pde",),
    CreativeCodingDomain.CANVAS_2D: (".canvas.js",),
    CreativeCodingDomain.WEBGPU_WGSL: (".wgsl", ".webgpu.js"),
    CreativeCodingDomain.GSAP: (".gsap.js",),
    CreativeCodingDomain.TONE_JS: (".tone.js",),
    CreativeCodingDomain.PIXI_JS: (".js",),
    CreativeCodingDomain.MATTER_JS: (".js",),
    CreativeCodingDomain.RAPIER: (".js", ".ts"),
    CreativeCodingDomain.SHADERTOY: (".frag",),
    CreativeCodingDomain.TOUCHDESIGNER: (".toe", ".tox", ".md", ".json"),
    CreativeCodingDomain.HOUDINI: (".hip", ".hda", ".vfl", ".md", ".json"),
    CreativeCodingDomain.BLENDER_GEOMETRY_NODES: (".blend", ".py", ".md", ".json"),
    CreativeCodingDomain.UNITY: (".unity", ".cs", ".md", ".json"),
    CreativeCodingDomain.UNREAL: (".uproject", ".uasset", ".md", ".json"),
    CreativeCodingDomain.MAX_MSP: (".maxpat", ".md", ".json"),
    CreativeCodingDomain.NOTCH: (".dfx", ".md", ".json"),
    CreativeCodingDomain.VVVV: (".v4p", ".vl", ".md", ".json"),
    CreativeCodingDomain.OPENFRAMEWORKS: (".cpp", ".h", ".md", ".json"),
    CreativeCodingDomain.OPENRNDR: (".kt", ".md", ".json"),
    CreativeCodingDomain.SUPERCOLLIDER: (".scd", ".md", ".json"),
    CreativeCodingDomain.SONIC_PI: (".rb", ".md", ".json"),
    CreativeCodingDomain.TIDALCYCLES: (".tidal", ".md", ".json"),
    CreativeCodingDomain.WEB_AUDIO_API: (".js",),
    CreativeCodingDomain.P5_SOUND: (".p5.js",),
    CreativeCodingDomain.ML5_JS: (".js",),
    CreativeCodingDomain.TENSORFLOW_JS: (".js",),
    CreativeCodingDomain.COMFYUI: (".json", ".md"),
    CreativeCodingDomain.STABLE_DIFFUSION_WORKFLOWS: (".json", ".md"),
    CreativeCodingDomain.RUNWAY: (".json", ".md"),
    CreativeCodingDomain.BLENDER_PYTHON_API: (".py", ".md", ".json"),
    CreativeCodingDomain.UNREAL_BLUEPRINTS: (".md", ".json"),
    CreativeCodingDomain.ABLETON_LIVE: (".als", ".md", ".json"),
    CreativeCodingDomain.VCV_RACK: (".vcv", ".md", ".json"),
    CreativeCodingDomain.GODOT: (".gd", ".tscn", ".md", ".json"),
    CreativeCodingDomain.RESOLUME: (".avc", ".md", ".json"),
    CreativeCodingDomain.MADMAPPER: (".mad", ".md", ".json"),
    CreativeCodingDomain.CABLES_GL: (".json", ".js", ".md"),
    CreativeCodingDomain.PURE_DATA: (".pd", ".md", ".json"),
}

_PRIMARY_TRIGGERS: dict[CreativeCodingDomain, tuple[str, ...]] = {
    CreativeCodingDomain.P5_JS: ("p5", "sketch", "draw loop", "flow field"),
    CreativeCodingDomain.THREE_JS: ("three.js", "3d scene", "webgl scene", "camera"),
    CreativeCodingDomain.REACT_THREE_FIBER: ("react three fiber", "r3f", "useFrame", "react scene"),
    CreativeCodingDomain.GLSL: ("glsl", "fragment shader", "shader", "raymarch"),
    CreativeCodingDomain.HYDRA: ("hydra", "live coding", "feedback synth", "osc()"),
    CreativeCodingDomain.TOUCHDESIGNER: ("touchdesigner", "tops", "chops", "node network"),
    CreativeCodingDomain.UNREAL: ("unreal", "niagara", "blueprint", "lumen"),
    CreativeCodingDomain.HOUDINI: ("houdini", "sop", "vex", "hda"),
    CreativeCodingDomain.BLENDER_GEOMETRY_NODES: ("blender", "geometry nodes", "node tree"),
}


def domain_experience_records() -> tuple[DomainExperienceRecord, ...]:
    """Return one complete, ordered contract record for every registered domain."""

    return tuple(_build_record(info.value) for info in SUPPORTED_DOMAINS)


def get_domain_experience(
    domain: CreativeCodingDomain | str,
) -> DomainExperienceRecord:
    """Resolve a domain contract without guessing an unregistered capability."""

    resolved_domain = (
        domain if isinstance(domain, CreativeCodingDomain) else CreativeCodingDomain(domain)
    )
    for record in domain_experience_records():
        if record.domain is resolved_domain:
            return record
    raise ValueError(f"No domain experience record is registered for {resolved_domain.value}.")


def _build_record(domain: CreativeCodingDomain) -> DomainExperienceRecord:
    info = next(item for item in SUPPORTED_DOMAINS if item.value is domain)
    source_ids = tuple(source.source_id for source in approved_sources_for_domain(domain))
    aliases = _aliases(info.label, info.slug, domain)
    triggers = _PRIMARY_TRIGGERS.get(domain, aliases)

    if domain in _BROWSER_CONTRACTS:
        contract = _BROWSER_CONTRACTS[domain]
        return DomainExperienceRecord(
            domain=domain,
            display_name=info.label,
            aliases=aliases,
            intent_triggers=triggers,
            knowledge_source_ids=source_ids,
            workflow_compatibility=("retrieve", "generate", "refine", "preview", "export"),
            artifact_types=contract["artifact_types"],  # type: ignore[arg-type]
            filename_extensions=contract["extensions"],  # type: ignore[arg-type]
            delivery_kind=DomainDeliveryKind.BROWSER_PREVIEW,
            live_preview=True,
            runtime_requirements=contract["requirements"],  # type: ignore[arg-type]
            validation_status=DomainValidationStatus.VALIDATED_BROWSER_CONTRACT,
            fallback=contract["fallback"],  # type: ignore[arg-type]
            public_claim_boundary=contract["boundary"],  # type: ignore[arg-type]
            demo_eligible=contract["demo_eligible"],  # type: ignore[arg-type]
        )

    if domain in _EXTERNAL_HANDOFF_DOMAINS:
        return DomainExperienceRecord(
            domain=domain,
            display_name=info.label,
            aliases=aliases,
            intent_triggers=triggers,
            knowledge_source_ids=source_ids,
            workflow_compatibility=(
                "retrieve",
                "generate",
                "refine",
                "export",
                "external_handoff",
            ),
            artifact_types=(
                "external-tool handoff package",
                "implementation notes",
                "parameter schema",
            ),
            filename_extensions=_EXTENSIONS.get(domain, (".md", ".json")),
            delivery_kind=DomainDeliveryKind.EXTERNAL_HANDOFF,
            live_preview=False,
            runtime_requirements=(
                f"{info.label} must be opened and completed in its own installed "
                "runtime.",
                "The workstation exports a brief, manifest, assumptions, and "
                "validation checklist; it does not run the external tool.",
            ),
            validation_status=DomainValidationStatus.HANDOFF_PACKAGE,
            fallback=(
                "Use the exported brief and manifest to continue in the named "
                "external tool."
            ),
            public_claim_boundary=(
                f"{info.label} is an external-tool handoff, not an internal live "
                "preview or remote execution claim."
            ),
            demo_eligible=False,
        )

    return DomainExperienceRecord(
        domain=domain,
        display_name=info.label,
        aliases=aliases,
        intent_triggers=triggers,
        knowledge_source_ids=source_ids,
        workflow_compatibility=("retrieve", "generate", "refine", "export"),
        artifact_types=("source artifact", "implementation notes"),
        filename_extensions=_EXTENSIONS.get(
            domain, _default_extensions(info.category)
        ),
        delivery_kind=DomainDeliveryKind.CODE_EXPORT,
        live_preview=False,
        runtime_requirements=_code_export_requirements(domain, info.label),
        validation_status=DomainValidationStatus.VALIDATED_CODE_EXPORT,
        fallback=(
            "Inspect, copy, or download the generated source for use in its compatible "
            "runtime."
        ),
        public_claim_boundary=_code_export_boundary(domain, info.label),
        demo_eligible=False,
    )


def _aliases(label: str, slug: str, domain: CreativeCodingDomain) -> tuple[str, ...]:
    values = [label.lower(), slug.replace("_", " "), domain.value.replace("_", " ")]
    return tuple(dict.fromkeys(value for value in values if value))


def _default_extensions(category: DomainCategory) -> tuple[str, ...]:
    if category is DomainCategory.SHADERS_GPU:
        return (".glsl", ".txt")
    if category is DomainCategory.AUDIO_LIVE_CODING:
        return (".js", ".txt")
    return (".js", ".md")


def _code_export_requirements(
    domain: CreativeCodingDomain,
    label: str,
) -> tuple[str, ...]:
    if domain is CreativeCodingDomain.REACT_THREE_FIBER:
        return (
            "A consumer React project with React Three Fiber and its renderer "
            "dependencies is required.",
            "The controlled Three.js preview does not execute React components.",
        )
    if domain is CreativeCodingDomain.HYDRA:
        return (
            "Use a compatible Hydra environment for full synth execution.",
            "A bounded client adapter exists for isolated source analysis, but the "
            "generation route does not claim an active Hydra live-preview contract.",
        )
    return (
        f"{label} source must run in a compatible consumer runtime or editor.",
        "The workstation preserves the artifact and its retrieval provenance without "
        "claiming an internal live preview.",
    )


def _code_export_boundary(domain: CreativeCodingDomain, label: str) -> str:
    if domain is CreativeCodingDomain.REACT_THREE_FIBER:
        return (
            "React Three Fiber is generated as a code export and requires a consumer "
            "React bundle; it is not an internal preview surface."
        )
    if domain is CreativeCodingDomain.HYDRA:
        return (
            "Hydra source may be exported, but it is not advertised as an active "
            "product-generation live preview until the complete artifact-to-runtime "
            "contract is validated."
        )
    return (
        f"{label} is available as source/export only in the current workstation; no "
        "internal live-preview runtime is claimed."
    )
