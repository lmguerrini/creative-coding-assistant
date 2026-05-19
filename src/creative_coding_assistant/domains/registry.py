"""Domain metadata used by routing, RAG, prompts, memory, and UI layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from creative_coding_assistant.contracts.requests import CreativeCodingDomain


class DomainCategory(StrEnum):
    WEB_CREATIVE_CODING = "web_creative_coding"
    SHADERS_GPU = "shaders_gpu"
    ANIMATION = "animation"
    PHYSICS = "physics"
    AUDIO_LIVE_CODING = "audio_live_coding"
    AV_VJ = "av_vj"
    DCC_PROCEDURAL = "dcc_procedural"
    GAME_ENGINES = "game_engines"
    CREATIVE_AI = "creative_ai"
    VISUAL_PATCHING = "visual_patching"
    MODULAR_SYNTHESIS = "modular_synthesis"
    PROJECTION_MAPPING = "projection_mapping"


@dataclass(frozen=True)
class DomainInfo:
    value: CreativeCodingDomain
    label: str
    slug: str
    category: DomainCategory
    prompt_guidance: str
    memory_label: str
    default_generated_topic: str

    @property
    def official_source_key(self) -> str:
        """Backward-compatible alias for older source-key naming."""

        return self.slug


def _info(
    value: CreativeCodingDomain,
    label: str,
    slug: str,
    category: DomainCategory,
    prompt_guidance: str,
    default_generated_topic: str,
    *,
    memory_label: str | None = None,
) -> DomainInfo:
    return DomainInfo(
        value=value,
        label=label,
        slug=slug,
        category=category,
        prompt_guidance=prompt_guidance,
        memory_label=memory_label or label,
        default_generated_topic=default_generated_topic,
    )


SUPPORTED_DOMAINS: tuple[DomainInfo, ...] = (
    _info(
        CreativeCodingDomain.THREE_JS,
        "Three.js",
        "three_js",
        DomainCategory.WEB_CREATIVE_CODING,
        (
            "Prefer plain Three.js patterns over React wrappers unless the user "
            "explicitly asks for React Three Fiber."
        ),
        "scene code",
    ),
    _info(
        CreativeCodingDomain.REACT_THREE_FIBER,
        "React Three Fiber",
        "r3f",
        DomainCategory.WEB_CREATIVE_CODING,
        (
            "Prefer React Three Fiber components and hooks; only drop to raw "
            "Three.js APIs when the low-level concept matters."
        ),
        "component code",
    ),
    _info(
        CreativeCodingDomain.P5_JS,
        "p5.js",
        "p5_js",
        DomainCategory.WEB_CREATIVE_CODING,
        (
            "Prefer p5.js sketch structure such as setup(), draw(), and concise "
            "runnable examples."
        ),
        "sketch",
    ),
    _info(
        CreativeCodingDomain.GLSL,
        "GLSL",
        "glsl",
        DomainCategory.SHADERS_GPU,
        (
            "Prefer concrete shader snippets and shader terminology over "
            "host-framework setup details."
        ),
        "shader code",
    ),
    _info(
        CreativeCodingDomain.PROCESSING,
        "Processing",
        "processing",
        DomainCategory.WEB_CREATIVE_CODING,
        (
            "Prefer Processing sketch structure such as setup(), draw(), size(), "
            "and concise PDE-style examples."
        ),
        "sketch",
    ),
    _info(
        CreativeCodingDomain.CANVAS_2D,
        "Canvas 2D",
        "canvas_2d",
        DomainCategory.WEB_CREATIVE_CODING,
        (
            "Prefer standard CanvasRenderingContext2D APIs, clear canvas setup, "
            "and requestAnimationFrame for browser animation."
        ),
        "canvas sketch",
    ),
    _info(
        CreativeCodingDomain.WEBGPU_WGSL,
        "WebGPU/WGSL",
        "webgpu_wgsl",
        DomainCategory.SHADERS_GPU,
        (
            "Prefer WebGPU host setup and WGSL shader syntax; do not substitute "
            "GLSL unless the user explicitly asks for it."
        ),
        "WebGPU/WGSL code",
    ),
    _info(
        CreativeCodingDomain.GSAP,
        "GSAP",
        "gsap",
        DomainCategory.ANIMATION,
        (
            "Prefer GSAP tweens and timelines such as gsap.to() and "
            "gsap.timeline() for browser animation sequencing."
        ),
        "animation code",
    ),
    _info(
        CreativeCodingDomain.TONE_JS,
        "Tone.js",
        "tone_js",
        DomainCategory.AUDIO_LIVE_CODING,
        (
            "Prefer Tone.js Transport, synth, sampler, and signal APIs, and "
            "mention browser audio start requirements when relevant."
        ),
        "audio code",
    ),
    _info(
        CreativeCodingDomain.PIXI_JS,
        "PixiJS",
        "pixi_js",
        DomainCategory.WEB_CREATIVE_CODING,
        (
            "Prefer PixiJS Application, stage, ticker, Graphics, Sprite, and "
            "renderer terminology for 2D WebGL/WebGPU work."
        ),
        "rendering code",
    ),
    _info(
        CreativeCodingDomain.MATTER_JS,
        "Matter.js",
        "matter_js",
        DomainCategory.PHYSICS,
        (
            "Prefer Matter.js Engine, World, Bodies, Runner, constraints, and "
            "clear physics-step structure."
        ),
        "physics code",
    ),
    _info(
        CreativeCodingDomain.RAPIER,
        "Rapier",
        "rapier",
        DomainCategory.PHYSICS,
        (
            "Prefer Rapier rigid bodies, colliders, joints, and world stepping; "
            "do not substitute Matter.js unless the user asks for it."
        ),
        "physics code",
    ),
    _info(
        CreativeCodingDomain.HYDRA,
        "Hydra",
        "hydra",
        DomainCategory.AV_VJ,
        (
            "Prefer Hydra live-coding chains using sources, oscillators, "
            "modulation, and concise video-synth examples."
        ),
        "live-coding sketch",
    ),
    _info(
        CreativeCodingDomain.SHADERTOY,
        "Shadertoy",
        "shadertoy",
        DomainCategory.SHADERS_GPU,
        (
            "Prefer Shadertoy GLSL structure with mainImage(), fragCoord, iTime, "
            "and iResolution."
        ),
        "shader code",
    ),
    _info(
        CreativeCodingDomain.TOUCHDESIGNER,
        "TouchDesigner",
        "touchdesigner",
        DomainCategory.VISUAL_PATCHING,
        (
            "Treat TouchDesigner as an external workflow domain; prefer operator "
            "families such as TOPs, CHOPs, DATs, COMPs, and node-network guidance."
        ),
        "operator network guidance",
    ),
    _info(
        CreativeCodingDomain.HOUDINI,
        "Houdini",
        "houdini",
        DomainCategory.DCC_PROCEDURAL,
        (
            "Treat Houdini as an external procedural workflow domain; prefer SOP, "
            "VOP, DOP, LOP, VEX, HDA, and node graph terminology."
        ),
        "procedural workflow guidance",
    ),
    _info(
        CreativeCodingDomain.BLENDER_GEOMETRY_NODES,
        "Blender / Geometry Nodes",
        "blender_geometry_nodes",
        DomainCategory.DCC_PROCEDURAL,
        (
            "Treat Blender Geometry Nodes as an external DCC workflow domain; "
            "prefer modifiers, node trees, fields, attributes, and Blender manual "
            "terminology."
        ),
        "geometry nodes guidance",
    ),
    _info(
        CreativeCodingDomain.UNITY,
        "Unity",
        "unity",
        DomainCategory.GAME_ENGINES,
        (
            "Treat Unity as an external engine workflow domain; prefer scenes, "
            "GameObjects, components, prefabs, C# scripts, URP, and HDRP guidance."
        ),
        "engine workflow guidance",
    ),
    _info(
        CreativeCodingDomain.UNREAL,
        "Unreal",
        "unreal",
        DomainCategory.GAME_ENGINES,
        (
            "Treat Unreal as an external engine workflow domain; prefer Unreal "
            "Editor, Blueprint, C++, Niagara, PCG, Nanite, and Lumen terminology."
        ),
        "engine workflow guidance",
    ),
    _info(
        CreativeCodingDomain.MAX_MSP,
        "Max/MSP",
        "max_msp",
        DomainCategory.VISUAL_PATCHING,
        (
            "Treat Max/MSP as an external visual patching domain; prefer patchers, "
            "objects, MSP signal flow, Jitter matrices, Gen, and Max terminology."
        ),
        "patching guidance",
    ),
    _info(
        CreativeCodingDomain.NOTCH,
        "Notch",
        "notch",
        DomainCategory.AV_VJ,
        (
            "Treat Notch as an external realtime VFX workflow domain; prefer "
            "Builder, nodegraph, blocks, exposed parameters, and media-server "
            "pipeline guidance."
        ),
        "realtime VFX workflow guidance",
    ),
    _info(
        CreativeCodingDomain.VVVV,
        "VVVV",
        "vvvv",
        DomainCategory.VISUAL_PATCHING,
        (
            "Treat vvvv gamma as an external visual programming domain; prefer VL "
            "patches, nodes, pins, links, Skia, Stride, and gamma terminology."
        ),
        "visual programming guidance",
    ),
    _info(
        CreativeCodingDomain.OPENFRAMEWORKS,
        "openFrameworks",
        "openframeworks",
        DomainCategory.WEB_CREATIVE_CODING,
        (
            "Treat openFrameworks as an external native C++ creative-coding "
            "framework; prefer ofApp, setup(), update(), draw(), ofx addons, and "
            "graphics/audio/video pipeline terminology."
        ),
        "creative coding framework guidance",
    ),
    _info(
        CreativeCodingDomain.OPENRNDR,
        "OPENRNDR",
        "openrndr",
        DomainCategory.WEB_CREATIVE_CODING,
        (
            "Treat OPENRNDR as an external Kotlin creative-coding framework; "
            "prefer Program, Drawer, extensions, shade styles, and render-target "
            "terminology."
        ),
        "creative coding framework guidance",
    ),
    _info(
        CreativeCodingDomain.SUPERCOLLIDER,
        "SuperCollider",
        "supercollider",
        DomainCategory.AUDIO_LIVE_CODING,
        (
            "Treat SuperCollider as an external audio live-coding domain; prefer "
            "sclang, SynthDef, UGens, patterns, buses, and server/client "
            "terminology."
        ),
        "audio live-coding guidance",
    ),
    _info(
        CreativeCodingDomain.SONIC_PI,
        "Sonic Pi",
        "sonic_pi",
        DomainCategory.AUDIO_LIVE_CODING,
        (
            "Treat Sonic Pi as an external live-coding music domain; prefer "
            "live_loop, synths, samples, FX, cues, and concise Ruby-like examples."
        ),
        "audio live-coding guidance",
    ),
    _info(
        CreativeCodingDomain.TIDALCYCLES,
        "TidalCycles",
        "tidalcycles",
        DomainCategory.AUDIO_LIVE_CODING,
        (
            "Treat TidalCycles as an external pattern live-coding domain; prefer "
            "mini-notation, patterns, cycles, Haskell syntax, and SuperDirt "
            "terminology."
        ),
        "pattern live-coding guidance",
    ),
    _info(
        CreativeCodingDomain.WEB_AUDIO_API,
        "Web Audio API",
        "web_audio_api",
        DomainCategory.AUDIO_LIVE_CODING,
        (
            "Prefer standard Web Audio API graph terminology such as AudioContext, "
            "AudioNode, GainNode, OscillatorNode, AudioWorklet, and scheduling."
        ),
        "browser audio code",
    ),
    _info(
        CreativeCodingDomain.P5_SOUND,
        "p5.sound",
        "p5_sound",
        DomainCategory.AUDIO_LIVE_CODING,
        (
            "Prefer p5.sound APIs such as loadSound(), p5.SoundFile, oscillators, "
            "envelopes, FFT, and p5 sketch preload/setup/draw structure."
        ),
        "p5.sound audio code",
    ),
    _info(
        CreativeCodingDomain.ML5_JS,
        "ml5.js",
        "ml5_js",
        DomainCategory.CREATIVE_AI,
        (
            "Prefer ml5.js browser ML APIs and model names such as BodyPose, "
            "HandPose, FaceMesh, ImageClassifier, SoundClassifier, and "
            "NeuralNetwork."
        ),
        "browser ML code",
    ),
    _info(
        CreativeCodingDomain.TENSORFLOW_JS,
        "TensorFlow.js",
        "tensorflow_js",
        DomainCategory.CREATIVE_AI,
        (
            "Prefer TensorFlow.js APIs such as tf.tensor(), Layers, model loading, "
            "training, browser execution, and Node.js execution when relevant."
        ),
        "browser ML code",
    ),
    _info(
        CreativeCodingDomain.COMFYUI,
        "ComfyUI",
        "comfyui",
        DomainCategory.CREATIVE_AI,
        (
            "Treat ComfyUI as an external node-based generative AI workflow "
            "domain; prefer nodes, models, samplers, latents, conditioning, and "
            "workflow JSON terminology."
        ),
        "node workflow guidance",
    ),
    _info(
        CreativeCodingDomain.STABLE_DIFFUSION_WORKFLOWS,
        "Stable Diffusion workflows",
        "stable_diffusion_workflows",
        DomainCategory.CREATIVE_AI,
        (
            "Treat Stable Diffusion as an external generative AI workflow domain; "
            "prefer checkpoints, prompts, samplers, schedulers, LoRA, ControlNet, "
            "and Diffusers pipeline terminology."
        ),
        "diffusion workflow guidance",
    ),
    _info(
        CreativeCodingDomain.RUNWAY,
        "Runway",
        "runway",
        DomainCategory.CREATIVE_AI,
        (
            "Treat Runway as an external creative AI platform/API domain; prefer "
            "generation tasks, assets, prompts, model versions, and API workflow "
            "terminology."
        ),
        "creative AI workflow guidance",
    ),
    _info(
        CreativeCodingDomain.BLENDER_PYTHON_API,
        "Blender Python API",
        "blender_python_api",
        DomainCategory.DCC_PROCEDURAL,
        (
            "Treat Blender Python as an external DCC scripting domain; prefer bpy, "
            "operators, data blocks, context, properties, and addon structure."
        ),
        "Blender scripting guidance",
    ),
    _info(
        CreativeCodingDomain.UNREAL_BLUEPRINTS,
        "Unreal Blueprints",
        "unreal_blueprints",
        DomainCategory.GAME_ENGINES,
        (
            "Treat Unreal Blueprints as an external visual scripting domain; "
            "prefer Blueprint classes, graphs, nodes, events, pins, variables, and "
            "Unreal Editor workflow terminology."
        ),
        "Blueprint workflow guidance",
    ),
    _info(
        CreativeCodingDomain.ABLETON_LIVE,
        "Ableton Live",
        "ableton_live",
        DomainCategory.AUDIO_LIVE_CODING,
        (
            "Treat Ableton Live as an external DAW workflow domain; prefer clips, "
            "Session View, Arrangement View, devices, racks, automation, and Max "
            "for Live terminology."
        ),
        "DAW workflow guidance",
    ),
    _info(
        CreativeCodingDomain.VCV_RACK,
        "VCV Rack",
        "vcv_rack",
        DomainCategory.MODULAR_SYNTHESIS,
        (
            "Treat VCV Rack as an external modular synthesis domain; prefer "
            "modules, patch cables, CV/Gate, oscillators, filters, sequencers, "
            "and Eurorack signal-flow terminology."
        ),
        "modular synthesis guidance",
    ),
    _info(
        CreativeCodingDomain.GODOT,
        "Godot",
        "godot",
        DomainCategory.GAME_ENGINES,
        (
            "Treat Godot as an external game-engine workflow domain; prefer "
            "scenes, nodes, resources, GDScript, signals, scripts, and editor "
            "workflow terminology."
        ),
        "engine workflow guidance",
    ),
    _info(
        CreativeCodingDomain.RESOLUME,
        "Resolume",
        "resolume",
        DomainCategory.AV_VJ,
        (
            "Treat Resolume as an external AV/VJ workflow domain; prefer "
            "compositions, decks, layers, clips, effects, BPM sync, output "
            "routing, slices, and Advanced Output terminology."
        ),
        "AV workflow guidance",
    ),
    _info(
        CreativeCodingDomain.MADMAPPER,
        "MadMapper",
        "madmapper",
        DomainCategory.PROJECTION_MAPPING,
        (
            "Treat MadMapper as an external projection-mapping workflow domain; "
            "prefer surfaces, quads, masks, materials, media, DMX/Art-Net, "
            "fixtures, and calibration terminology."
        ),
        "projection mapping guidance",
    ),
    _info(
        CreativeCodingDomain.CABLES_GL,
        "Cables.gl",
        "cables_gl",
        DomainCategory.VISUAL_PATCHING,
        (
            "Treat Cables.gl as an external realtime visual patching domain; "
            "prefer patches, operators, ports, op graphs, variables, timelines, "
            "WebGL, and export terminology."
        ),
        "visual patching guidance",
    ),
    _info(
        CreativeCodingDomain.PURE_DATA,
        "Pure Data",
        "pure_data",
        DomainCategory.VISUAL_PATCHING,
        (
            "Treat Pure Data as an external visual patching/audio domain; prefer "
            "patches, objects, messages, inlets, outlets, DSP graphs, abstractions, "
            "and signal/message-flow terminology."
        ),
        "patching guidance",
    ),
)

_DOMAIN_INFO_BY_VALUE = {domain.value: domain for domain in SUPPORTED_DOMAINS}


def get_domain_info(domain: CreativeCodingDomain) -> DomainInfo:
    """Return metadata for a supported creative-coding domain."""

    return _DOMAIN_INFO_BY_VALUE[domain]


def get_domain_label(domain: CreativeCodingDomain) -> str:
    return get_domain_info(domain).label


def get_domain_slug(domain: CreativeCodingDomain) -> str:
    return get_domain_info(domain).slug


def get_domain_category(domain: CreativeCodingDomain) -> DomainCategory:
    return get_domain_info(domain).category


def get_domain_prompt_guidance(domain: CreativeCodingDomain) -> str:
    return get_domain_info(domain).prompt_guidance


def get_domain_memory_label(domain: CreativeCodingDomain | None) -> str | None:
    if domain is None:
        return None
    return get_domain_info(domain).memory_label


def get_domain_default_topic(domain: CreativeCodingDomain | None) -> str:
    if domain is None:
        return "code"
    return get_domain_info(domain).default_generated_topic


def get_supported_domain_values() -> tuple[str, ...]:
    """Return stable domain values for clients and validators."""

    return tuple(domain.value.value for domain in SUPPORTED_DOMAINS)
