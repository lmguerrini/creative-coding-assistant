"""Domain-aware generation and runtime support helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.preview import PreviewTarget
from creative_coding_assistant.rag.retrieval.domain_intent import (
    resolve_effective_query_domains,
)


@dataclass(frozen=True)
class DomainRuntimeSupport:
    """Runtime metadata for domains with current live preview support."""

    domain: CreativeCodingDomain
    runtime: str
    renderer_id: str
    preview_target: str
    label: str


PREVIEWABLE_GENERATION_DOMAINS: tuple[CreativeCodingDomain, ...] = (
    CreativeCodingDomain.P5_JS,
    CreativeCodingDomain.GLSL,
    CreativeCodingDomain.THREE_JS,
    CreativeCodingDomain.TONE_JS,
)

SUPPORTED_GENERATION_RUNTIMES = frozenset({"p5", "glsl", "three", "tone"})

_RUNTIME_SUPPORT_BY_DOMAIN: dict[CreativeCodingDomain, DomainRuntimeSupport] = {
    CreativeCodingDomain.P5_JS: DomainRuntimeSupport(
        domain=CreativeCodingDomain.P5_JS,
        runtime="p5",
        renderer_id="surface.p5",
        preview_target=PreviewTarget.BROWSER_SANDBOX.value,
        label="p5.js browser preview",
    ),
    CreativeCodingDomain.GLSL: DomainRuntimeSupport(
        domain=CreativeCodingDomain.GLSL,
        runtime="glsl",
        renderer_id="surface.glsl",
        preview_target=PreviewTarget.BROWSER_SANDBOX.value,
        label="GLSL fragment shader preview",
    ),
    CreativeCodingDomain.THREE_JS: DomainRuntimeSupport(
        domain=CreativeCodingDomain.THREE_JS,
        runtime="three",
        renderer_id="surface.three",
        preview_target=PreviewTarget.BROWSER_SANDBOX.value,
        label="Three.js browser preview",
    ),
    CreativeCodingDomain.TONE_JS: DomainRuntimeSupport(
        domain=CreativeCodingDomain.TONE_JS,
        runtime="tone",
        renderer_id="surface.tone",
        preview_target=PreviewTarget.BROWSER_SANDBOX.value,
        label="Tone.js muted browser preview",
    ),
}

_DOMAINS_BY_RUNTIME: dict[str, tuple[CreativeCodingDomain, ...]] = {
    "p5": (CreativeCodingDomain.P5_JS,),
    "glsl": (CreativeCodingDomain.GLSL,),
    "three": (CreativeCodingDomain.THREE_JS,),
    "tone": (CreativeCodingDomain.TONE_JS,),
}

_MULTI_CANDIDATE_PATTERN = re.compile(
    r"\b(?:multiple|several|compare|comparison|alternatives?|options?|"
    r"variations?|candidates?)\b"
)
_VISUAL_OUTPUT_PATTERN = re.compile(
    r"\b(?:animated?|animation|visual|generative|sketch|particles?|field|"
    r"canvas|interactive|motion|scene|shader|runtime|preview|render)\b"
)
_R3F_INFERENCE_PATTERN = re.compile(
    r"\b(?:react\s+component|jsx|tsx|hooks?|useframe|fiber)\b"
)
_THREE_INFERENCE_PATTERN = re.compile(
    r"\b(?:3d|webgl|scene|camera|mesh|cube|sphere|orbit|material|geometry|"
    r"lighting|lights?)\b"
)
_GLSL_INFERENCE_PATTERN = re.compile(
    r"\b(?:shader|fragment|raymarch|sdf|uniform|fragcoord|gl_fragcolor|"
    r"uv|noise field)\b"
)
_P5_INFERENCE_PATTERN = re.compile(
    r"\b(?:sketch|p5|draw loop|2d|particles?|boids?|flow field|"
    r"generative|canvas|interactive drawing)\b"
)

_DOMAIN_GENERATION_GUIDANCE: dict[CreativeCodingDomain, tuple[str, ...]] = {
    CreativeCodingDomain.P5_JS: (
        "For p5.js generation, return one self-contained global-mode sketch with "
        "function setup() and function draw().",
        "Return plain JavaScript p5 source only: prefer a .p5.js artifact name; "
        "do not use TypeScript, imports, instance-mode new p5 wrappers, HTML, or Markdown fences.",
        "Keep p5 calls inside setup(), draw(), or helpers called by them. The browser "
        "preview supports createCanvas, colorMode, background, fill, stroke, basic "
        "shapes, push/pop transforms, beginShape/vertex/endShape, noise/random, and "
        "mouseX/mouseY/mouseIsPressed pointer interaction. Build translucent trails "
        "directly with background alpha. Do not use createGraphics, image, createVector, "
        "p5.Vector, millis, noiseSeed, constrain, DOM controls, assets, keyboard callbacks, "
        "text/HUD APIs, or other unsupported p5 APIs. Keep the sketch compact enough for "
        "the browser preview rather than returning a full p5 application.",
    ),
    CreativeCodingDomain.GLSL: (
        "For GLSL generation, return a fragment shader with uniforms such as "
        "u_time and u_resolution when animation is useful.",
        "Prefer a .frag artifact name and avoid host-framework boilerplate "
        "unless requested. Keep it within the bounded WebGL fragment preview: use "
        "void main() or mainImage(), and do not use #version declarations, textures, "
        "sampler declarations, discard, or while loops.",
    ),
    CreativeCodingDomain.THREE_JS: (
        "For Three.js generation, return exactly one compact, fully closed fenced "
        "javascript artifact using a filename=...three.js fence attribute. The filename "
        "must briefly describe the requested scene (for example, "
        "kinetic-light-sculpture.three.js).",
        "Use self-contained browser-oriented Three.js scene JavaScript with "
        "THREE.Scene, PerspectiveCamera, WebGLRenderer, geometry, material, lights, "
        "and an animation-loop concept. Keep the source below 7,500 characters so it "
        "fits the bounded preview runtime.",
        "Do not return HTML, <!doctype>, <script>, CSS, CDN URLs, imports, React "
        "wrappers, TypeScript syntax, Markdown outside the one fence, or browser DOM "
        "operations such as document or window. The controlled preview supplies the "
        "canvas and animation surface.",
    ),
    CreativeCodingDomain.REACT_THREE_FIBER: (
        "For React Three Fiber generation, return React component code using "
        "Canvas-friendly components and hooks such as useFrame.",
        "Prefer a .r3f.tsx artifact name and keep Three.js imperative escapes minimal. "
        "React Three Fiber is code-only in the current workstation: it needs its own "
        "React bundle runtime and must not be presented as a live controlled Three.js preview.",
    ),
    CreativeCodingDomain.TONE_JS: (
        "For Tone.js generation, return exactly one self-contained JavaScript "
        "program in a filename=...tone.js fenced block.",
        "Use a supported Tone synth, oscillator, or noise voice and optional "
        "Tone.Transport or Tone.Sequence timing.",
        "The controlled preview is muted until the operator explicitly starts "
        "audio. Do not request microphone access, autoplay, imports, HTML, or "
        "external assets.",
    ),
}


def resolve_generation_domains(
    *,
    query: str,
    selected_domains: Sequence[CreativeCodingDomain],
) -> tuple[CreativeCodingDomain, ...]:
    """Resolve effective generation domains without one hardcoded fallback."""

    effective_domains = resolve_effective_query_domains(
        query=query,
        selected_domains=selected_domains,
    )
    if effective_domains:
        return effective_domains

    return infer_likely_generation_domains(query)


def infer_likely_generation_domains(query: str) -> tuple[CreativeCodingDomain, ...]:
    """Infer likely runtime-oriented domains for underspecified visual generation."""

    normalized = " ".join(query.strip().lower().split())
    if not normalized:
        return ()

    inferred: list[CreativeCodingDomain] = []

    if _GLSL_INFERENCE_PATTERN.search(normalized):
        inferred.append(CreativeCodingDomain.GLSL)
    if _R3F_INFERENCE_PATTERN.search(normalized) and _THREE_INFERENCE_PATTERN.search(
        normalized
    ):
        inferred.append(CreativeCodingDomain.REACT_THREE_FIBER)
    elif _THREE_INFERENCE_PATTERN.search(normalized):
        inferred.append(CreativeCodingDomain.THREE_JS)
    if _P5_INFERENCE_PATTERN.search(normalized):
        inferred.append(CreativeCodingDomain.P5_JS)

    if (
        not inferred
        and _VISUAL_OUTPUT_PATTERN.search(normalized)
        and not _THREE_INFERENCE_PATTERN.search(normalized)
        and not _GLSL_INFERENCE_PATTERN.search(normalized)
    ):
        inferred.append(CreativeCodingDomain.P5_JS)

    if (
        len(inferred) == 1
        and inferred[0] in PREVIEWABLE_GENERATION_DOMAINS
        and _MULTI_CANDIDATE_PATTERN.search(normalized)
        and _VISUAL_OUTPUT_PATTERN.search(normalized)
    ):
        inferred = [
            CreativeCodingDomain.P5_JS,
            CreativeCodingDomain.GLSL,
            CreativeCodingDomain.THREE_JS,
        ]

    return _dedupe_domains(inferred)


def get_domain_runtime_support(
    domain: CreativeCodingDomain | None,
) -> DomainRuntimeSupport | None:
    if domain is None:
        return None
    return _RUNTIME_SUPPORT_BY_DOMAIN.get(domain)


def is_previewable_generation_domain(domain: CreativeCodingDomain | None) -> bool:
    return domain in _RUNTIME_SUPPORT_BY_DOMAIN


def domains_for_runtime(runtime: str | None) -> tuple[CreativeCodingDomain, ...]:
    if runtime is None:
        return ()
    return _DOMAINS_BY_RUNTIME.get(runtime, ())


def domain_generation_guidance_lines(
    domains: Sequence[CreativeCodingDomain],
) -> tuple[str, ...]:
    lines: list[str] = []
    for domain in _dedupe_domains(domains):
        support = get_domain_runtime_support(domain)
        if support is not None:
            lines.append(
                f"{domain.value} has current live preview support through "
                f"{support.label}."
            )
        else:
            lines.append(
                f"{domain.value} is code-only in the current workstation; do "
                "not claim live preview readiness for it."
            )
        lines.extend(_DOMAIN_GENERATION_GUIDANCE.get(domain, ()))
    return tuple(lines)


def runtime_is_supported(runtime: str | None) -> bool:
    return runtime in SUPPORTED_GENERATION_RUNTIMES


def _dedupe_domains(
    domains: Sequence[CreativeCodingDomain],
) -> tuple[CreativeCodingDomain, ...]:
    normalized: list[CreativeCodingDomain] = []
    for domain in domains:
        if domain not in normalized:
            normalized.append(domain)
    return tuple(normalized)
