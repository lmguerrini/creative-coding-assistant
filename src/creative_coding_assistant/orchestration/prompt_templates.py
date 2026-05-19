"""Prompt template contracts and rendering boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Self

from jinja2 import Environment, StrictUndefined
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.orchestration.prompt_inputs import (
    PromptInputResponse,
    PromptUserInput,
)
from creative_coding_assistant.orchestration.routing import (
    DomainSelectionShape,
    RouteDecision,
    RouteName,
)

_SYSTEM_TEMPLATE = """
Route: {{ route.value }}
Mode: {{ prompt_input.user_input.mode.value }}
Domain Scope: {{ effective_domain_scope_label(prompt_input.user_input) }}
{% if prompt_input.user_input.effective_domains -%}
Effective Domains:
{% for domain in prompt_input.user_input.effective_domains -%}
- {{ domain.value }}
{% endfor %}
{% endif %}
{% if prompt_input.user_input.detected_domains -%}
Detected Query Domains:
{% for domain in prompt_input.user_input.detected_domains -%}
- {{ domain.value }}
{% endfor %}
{% endif %}
{% if show_ui_selected_domains(prompt_input.user_input) -%}
UI Selected Domains:
{% for domain in prompt_input.user_input.ui_selected_domains -%}
- {{ domain.value }}
{% endfor %}
{% endif %}
{% if prompt_input.user_input.is_follow_up -%}
Follow-Up Request:
- Treat the current request as a continuation or modification of the immediately
  previous working turn.
- Use the compact prior turn pair only as short-term working context.
- Let the current request and effective domains override stale details from
  earlier context.
{% endif %}
Use the provided context sections as working context. Keep responses grounded in
the structured inputs that follow.
Global Guardrails:
{% for instruction in global_guardrail_lines() -%}
- {{ instruction }}
{% endfor %}
Route Guidance:
{% for instruction in route_guidance_lines(route) -%}
- {{ instruction }}
{% endfor %}
Domain Discipline:
{% for instruction in domain_guidance_lines(prompt_input.user_input) -%}
- {{ instruction }}
{% endfor %}
When you provide code, place each runnable snippet in a fenced code block with
an explicit language tag such as ```html, ```javascript, ```jsx, ```glsl, or
```python.
Do not leave runnable code unfenced.
Keep explanation, notes, and setup guidance outside code fences.
""".strip()

_USER_TEMPLATE = """
User Request:
{{ prompt_input.user_input.query }}
""".strip()

_MEMORY_TEMPLATE = """
{% if prompt_input.memory_input.running_summary is not none -%}
Running Summary:
{{ prompt_input.memory_input.running_summary.content }}

{% endif -%}
{% if prompt_input.memory_input.recent_turns -%}
{% if prompt_input.user_input.is_follow_up -%}
Immediate Prior Turn Pair:
{% else -%}
Recent Turns:
{% endif %}
{% for turn in prompt_input.memory_input.recent_turns -%}
- {{ turn.role.value }}[{{ turn.turn_index }}]:
{{ turn.content }}
{% endfor %}

{% endif -%}
{% if not prompt_input.user_input.is_follow_up
   and prompt_input.memory_input.session_summaries -%}
Session Memory:
{% for item in prompt_input.memory_input.session_summaries -%}
- {{ item.summary }}
{% endfor %}

{% endif -%}
{% if prompt_input.memory_input.project_memories -%}
Project Memory:
{% for memory in prompt_input.memory_input.project_memories -%}
- {{ memory.memory_kind.value }} ({{ memory.source }}): {{ memory.content }}
{% endfor %}
{% endif -%}
""".strip()

_RETRIEVAL_TEMPLATE = """
Official Knowledge Base:
{% for chunk in prompt_input.retrieval_input.chunks -%}
- {{ chunk.registry_title }} / {{ chunk.document_title }} ({{ chunk.source_id }})
  Source: {{ chunk.source_url }}
  Score: {{ '%.4f'|format(chunk.score) }}
  Excerpt: {{ chunk.excerpt }}
{% endfor %}
""".strip()


class RenderedPromptRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    CONTEXT = "context"


class RenderedPromptSectionName(StrEnum):
    SYSTEM = "system"
    USER = "user"
    MEMORY = "memory"
    RETRIEVAL = "retrieval"


@dataclass(frozen=True)
class _PromptSectionSpec:
    role: RenderedPromptRole
    name: RenderedPromptSectionName
    template: str


class RenderedPromptSection(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: RenderedPromptRole
    name: RenderedPromptSectionName
    content: str = Field(min_length=1)


class RenderedPromptRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    route: RouteName
    prompt_input: PromptInputResponse

    @model_validator(mode="after")
    def validate_route_alignment(self) -> Self:
        if self.prompt_input.request.route != self.route:
            raise ValueError("Prompt input route must match the rendered route.")
        return self


class RenderedPromptResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: RenderedPromptRequest
    sections: tuple[RenderedPromptSection, ...] = Field(default_factory=tuple)


class PromptRenderer(Protocol):
    def render(
        self,
        request: RenderedPromptRequest,
    ) -> RenderedPromptResponse:
        """Render provider-independent prompt sections from structured inputs."""


class JinjaPromptRenderer:
    """Render prompt-ready sections with Jinja2 and no provider assumptions."""

    def __init__(self) -> None:
        self._environment = Environment(
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )
        self._environment.globals.update(
            global_guardrail_lines=_global_guardrail_lines,
            route_guidance_lines=_route_guidance_lines,
            domain_guidance_lines=_domain_guidance_lines,
            effective_domain_scope_label=_effective_domain_scope_label,
            show_ui_selected_domains=_show_ui_selected_domains,
        )

    def render(
        self,
        request: RenderedPromptRequest,
    ) -> RenderedPromptResponse:
        sections = tuple(
            section
            for section in (
                self._render_section(spec=spec, request=request)
                for spec in _section_specs_for_request(request)
            )
            if section is not None
        )

        rendered = RenderedPromptResponse(
            request=request,
            sections=sections,
        )
        logger.info(
            "Rendered prompt with {} section(s) for route '{}'",
            len(rendered.sections),
            request.route.value,
        )
        return rendered

    def _render_section(
        self,
        *,
        spec: _PromptSectionSpec,
        request: RenderedPromptRequest,
    ) -> RenderedPromptSection | None:
        content = self._environment.from_string(spec.template).render(
            route=request.route,
            prompt_input=request.prompt_input,
        )
        normalized = "\n".join(
            line.rstrip() for line in content.splitlines() if line.strip()
        ).strip()
        if not normalized:
            logger.info(
                "Skipped empty rendered prompt section '{}' for route '{}'",
                spec.name.value,
                request.route.value,
            )
            return None
        return RenderedPromptSection(
            role=spec.role,
            name=spec.name,
            content=normalized,
        )


def _section_specs_for_request(
    request: RenderedPromptRequest,
) -> tuple[_PromptSectionSpec, ...]:
    specs = [
        _PromptSectionSpec(
            role=RenderedPromptRole.SYSTEM,
            name=RenderedPromptSectionName.SYSTEM,
            template=_SYSTEM_TEMPLATE,
        ),
        _PromptSectionSpec(
            role=RenderedPromptRole.USER,
            name=RenderedPromptSectionName.USER,
            template=_USER_TEMPLATE,
        ),
    ]
    if request.prompt_input.memory_input is not None:
        specs.append(
            _PromptSectionSpec(
                role=RenderedPromptRole.CONTEXT,
                name=RenderedPromptSectionName.MEMORY,
                template=_MEMORY_TEMPLATE,
            )
        )
    if request.prompt_input.retrieval_input is not None:
        specs.append(
            _PromptSectionSpec(
                role=RenderedPromptRole.CONTEXT,
                name=RenderedPromptSectionName.RETRIEVAL,
                template=_RETRIEVAL_TEMPLATE,
            )
        )
    return tuple(specs)


def build_rendered_prompt_request(
    *,
    route_decision: RouteDecision | RouteName,
    prompt_input: PromptInputResponse,
) -> RenderedPromptRequest:
    route = (
        route_decision
        if isinstance(route_decision, RouteName)
        else route_decision.route
    )
    return RenderedPromptRequest(route=route, prompt_input=prompt_input)


def _route_guidance_lines(route: RouteName) -> tuple[str, ...]:
    if route is RouteName.GENERATE:
        return (
            "Lead with runnable code first when the request calls for implementation.",
            (
                "Keep explanation short and add setup or run notes only when "
                "they are useful."
            ),
            "Avoid long conceptual sections unless the user explicitly asks for them.",
        )
    if route is RouteName.EXPLAIN:
        return (
            "Lead with conceptual clarity and explain the cause-and-effect first.",
            "Use concise code snippets only when they sharpen the explanation.",
            "Avoid full runnable projects unless the user explicitly asks for them.",
        )
    if route is RouteName.DEBUG:
        return (
            "Lead with the most likely issue before proposing changes.",
            "Structure the response as Issue, Fix, and Why it works.",
            (
                "Provide corrected code or patch-style guidance, and briefly ask "
                "for the missing code or error if the user did not supply enough "
                "context."
            ),
        )
    if route is RouteName.DESIGN:
        return (
            "Focus on structure, tradeoffs, and the recommended approach first.",
            "Keep implementation details scoped to the design choices that matter.",
        )
    if route is RouteName.REVIEW:
        return (
            "Review the request directly and list concrete findings first.",
            "Call out bugs, risks, regressions, and missing tests before suggestions.",
        )
    return (
        "Describe the intended artifact and the implementation path that supports it.",
        "Do not invent rendered output that has not actually been produced.",
    )


def _global_guardrail_lines() -> tuple[str, ...]:
    return (
        (
            "Keep the answer focused on the user's request and avoid "
            "unnecessary verbosity."
        ),
        "Prefer practical creative-coding examples over abstract discussion.",
        "Keep code blocks clean and runnable when code is requested.",
        (
            "Do not mix frameworks unless the user asks for it or the effective "
            "domains require it."
        ),
    )


def _domain_guidance_lines(user_input: PromptUserInput) -> tuple[str, ...]:
    if user_input.domain_selection is DomainSelectionShape.NONE:
        return (
            "Infer the relevant domain from the request and provided context only.",
            "Do not drift into unrelated frameworks or libraries without a clear need.",
        )

    guidance: list[str] = []

    if (
        user_input.detected_domains
        and user_input.detected_domains != user_input.ui_selected_domains
    ):
        guidance.append(
            "Prioritize the explicitly detected query domains over any broader "
            "selected UI scope."
        )

    guidance.append(
        "Stay within the effective domain set and avoid introducing unrelated "
        "ecosystems."
    )

    if user_input.domain_selection is DomainSelectionShape.SINGLE:
        guidance.append(
            "Prefer the effective ecosystem's APIs, terminology, and examples."
        )
    else:
        guidance.append(
            "Bridge domains only when the request actually spans them, and name "
            "which domain each major API belongs to when that reduces ambiguity."
        )

    for domain in user_input.effective_domains:
        guidance.append(_domain_preference_line(domain))

    return tuple(guidance)


def _effective_domain_scope_label(user_input: PromptUserInput) -> str:
    if user_input.domain_selection is DomainSelectionShape.NONE:
        return "inferred from request"
    if user_input.domain_selection is DomainSelectionShape.SINGLE:
        assert user_input.domain is not None
        return user_input.domain.value
    return "multi-domain selection"


def _show_ui_selected_domains(user_input: PromptUserInput) -> bool:
    return bool(
        user_input.ui_selected_domains
        and user_input.ui_selected_domains != user_input.effective_domains
    )


def _domain_preference_line(domain: CreativeCodingDomain) -> str:
    if domain is CreativeCodingDomain.THREE_JS:
        return (
            "Prefer plain Three.js patterns over React wrappers unless the user "
            "explicitly asks for React Three Fiber."
        )
    if domain is CreativeCodingDomain.REACT_THREE_FIBER:
        return (
            "Prefer React Three Fiber components and hooks; only drop to raw "
            "Three.js APIs when the low-level concept matters."
        )
    if domain is CreativeCodingDomain.P5_JS:
        return (
            "Prefer p5.js sketch structure such as setup(), draw(), and concise "
            "runnable examples."
        )
    if domain is CreativeCodingDomain.GLSL:
        return (
            "Prefer concrete shader snippets and shader terminology over "
            "host-framework setup details."
        )
    if domain is CreativeCodingDomain.PROCESSING:
        return (
            "Prefer Processing sketch structure such as setup(), draw(), size(), "
            "and concise PDE-style examples."
        )
    if domain is CreativeCodingDomain.CANVAS_2D:
        return (
            "Prefer standard CanvasRenderingContext2D APIs, clear canvas setup, "
            "and requestAnimationFrame for browser animation."
        )
    if domain is CreativeCodingDomain.WEBGPU_WGSL:
        return (
            "Prefer WebGPU host setup and WGSL shader syntax; do not substitute "
            "GLSL unless the user explicitly asks for it."
        )
    if domain is CreativeCodingDomain.GSAP:
        return (
            "Prefer GSAP tweens and timelines such as gsap.to() and "
            "gsap.timeline() for browser animation sequencing."
        )
    if domain is CreativeCodingDomain.TONE_JS:
        return (
            "Prefer Tone.js Transport, synth, sampler, and signal APIs, and "
            "mention browser audio start requirements when relevant."
        )
    if domain is CreativeCodingDomain.PIXI_JS:
        return (
            "Prefer PixiJS Application, stage, ticker, Graphics, Sprite, and "
            "renderer terminology for 2D WebGL/WebGPU work."
        )
    if domain is CreativeCodingDomain.MATTER_JS:
        return (
            "Prefer Matter.js Engine, World, Bodies, Runner, constraints, and "
            "clear physics-step structure."
        )
    if domain is CreativeCodingDomain.RAPIER:
        return (
            "Prefer Rapier rigid bodies, colliders, joints, and world stepping; "
            "do not substitute Matter.js unless the user asks for it."
        )
    if domain is CreativeCodingDomain.HYDRA:
        return (
            "Prefer Hydra live-coding chains using sources, oscillators, "
            "modulation, and concise video-synth examples."
        )
    if domain is CreativeCodingDomain.SHADERTOY:
        return (
            "Prefer Shadertoy GLSL structure with mainImage(), fragCoord, iTime, "
            "and iResolution."
        )
    if domain is CreativeCodingDomain.TOUCHDESIGNER:
        return (
            "Treat TouchDesigner as an external workflow domain; prefer operator "
            "families such as TOPs, CHOPs, DATs, COMPs, and node-network guidance."
        )
    if domain is CreativeCodingDomain.HOUDINI:
        return (
            "Treat Houdini as an external procedural workflow domain; prefer SOP, "
            "VOP, DOP, LOP, VEX, HDA, and node graph terminology."
        )
    if domain is CreativeCodingDomain.BLENDER_GEOMETRY_NODES:
        return (
            "Treat Blender Geometry Nodes as an external DCC workflow domain; "
            "prefer modifiers, node trees, fields, attributes, and Blender manual "
            "terminology."
        )
    if domain is CreativeCodingDomain.UNITY:
        return (
            "Treat Unity as an external engine workflow domain; prefer scenes, "
            "GameObjects, components, prefabs, C# scripts, URP, and HDRP guidance."
        )
    if domain is CreativeCodingDomain.UNREAL:
        return (
            "Treat Unreal as an external engine workflow domain; prefer Unreal "
            "Editor, Blueprint, C++, Niagara, PCG, Nanite, and Lumen terminology."
        )
    if domain is CreativeCodingDomain.MAX_MSP:
        return (
            "Treat Max/MSP as an external visual patching domain; prefer patchers, "
            "objects, MSP signal flow, Jitter matrices, Gen, and Max terminology."
        )
    if domain is CreativeCodingDomain.NOTCH:
        return (
            "Treat Notch as an external realtime VFX workflow domain; prefer "
            "Builder, nodegraph, blocks, exposed parameters, and media-server "
            "pipeline guidance."
        )
    if domain is CreativeCodingDomain.VVVV:
        return (
            "Treat vvvv gamma as an external visual programming domain; prefer VL "
            "patches, nodes, pins, links, Skia, Stride, and gamma terminology."
        )
    if domain is CreativeCodingDomain.OPENFRAMEWORKS:
        return (
            "Treat openFrameworks as an external native C++ creative-coding "
            "framework; prefer ofApp, setup(), update(), draw(), ofx addons, and "
            "graphics/audio/video pipeline terminology."
        )
    if domain is CreativeCodingDomain.OPENRNDR:
        return (
            "Treat OPENRNDR as an external Kotlin creative-coding framework; "
            "prefer Program, Drawer, extensions, shade styles, and render-target "
            "terminology."
        )
    if domain is CreativeCodingDomain.SUPERCOLLIDER:
        return (
            "Treat SuperCollider as an external audio live-coding domain; prefer "
            "sclang, SynthDef, UGens, patterns, buses, and server/client "
            "terminology."
        )
    if domain is CreativeCodingDomain.SONIC_PI:
        return (
            "Treat Sonic Pi as an external live-coding music domain; prefer "
            "live_loop, synths, samples, FX, cues, and concise Ruby-like examples."
        )
    if domain is CreativeCodingDomain.TIDALCYCLES:
        return (
            "Treat TidalCycles as an external pattern live-coding domain; prefer "
            "mini-notation, patterns, cycles, Haskell syntax, and SuperDirt "
            "terminology."
        )
    if domain is CreativeCodingDomain.WEB_AUDIO_API:
        return (
            "Prefer standard Web Audio API graph terminology such as AudioContext, "
            "AudioNode, GainNode, OscillatorNode, AudioWorklet, and scheduling."
        )
    if domain is CreativeCodingDomain.P5_SOUND:
        return (
            "Prefer p5.sound APIs such as loadSound(), p5.SoundFile, oscillators, "
            "envelopes, FFT, and p5 sketch preload/setup/draw structure."
        )
    if domain is CreativeCodingDomain.ML5_JS:
        return (
            "Prefer ml5.js browser ML APIs and model names such as BodyPose, "
            "HandPose, FaceMesh, ImageClassifier, SoundClassifier, and "
            "NeuralNetwork."
        )
    if domain is CreativeCodingDomain.TENSORFLOW_JS:
        return (
            "Prefer TensorFlow.js APIs such as tf.tensor(), Layers, model loading, "
            "training, browser execution, and Node.js execution when relevant."
        )
    if domain is CreativeCodingDomain.COMFYUI:
        return (
            "Treat ComfyUI as an external node-based generative AI workflow "
            "domain; prefer nodes, models, samplers, latents, conditioning, and "
            "workflow JSON terminology."
        )
    if domain is CreativeCodingDomain.STABLE_DIFFUSION_WORKFLOWS:
        return (
            "Treat Stable Diffusion as an external generative AI workflow domain; "
            "prefer checkpoints, prompts, samplers, schedulers, LoRA, ControlNet, "
            "and Diffusers pipeline terminology."
        )
    if domain is CreativeCodingDomain.RUNWAY:
        return (
            "Treat Runway as an external creative AI platform/API domain; prefer "
            "generation tasks, assets, prompts, model versions, and API workflow "
            "terminology."
        )
    if domain is CreativeCodingDomain.BLENDER_PYTHON_API:
        return (
            "Treat Blender Python as an external DCC scripting domain; prefer bpy, "
            "operators, data blocks, context, properties, and addon structure."
        )
    if domain is CreativeCodingDomain.UNREAL_BLUEPRINTS:
        return (
            "Treat Unreal Blueprints as an external visual scripting domain; "
            "prefer Blueprint classes, graphs, nodes, events, pins, variables, and "
            "Unreal Editor workflow terminology."
        )
    return (
        "Prefer the selected domain's official APIs, terminology, and examples."
    )
