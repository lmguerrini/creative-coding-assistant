"""Domain metadata used by routing, RAG, and UI layers."""

from __future__ import annotations

from dataclasses import dataclass

from creative_coding_assistant.contracts.requests import CreativeCodingDomain


@dataclass(frozen=True)
class DomainInfo:
    value: CreativeCodingDomain
    label: str
    official_source_key: str


SUPPORTED_DOMAINS: tuple[DomainInfo, ...] = (
    DomainInfo(CreativeCodingDomain.THREE_JS, "Three.js", "three_js"),
    DomainInfo(CreativeCodingDomain.REACT_THREE_FIBER, "React Three Fiber", "r3f"),
    DomainInfo(CreativeCodingDomain.P5_JS, "p5.js", "p5_js"),
    DomainInfo(CreativeCodingDomain.GLSL, "GLSL", "glsl"),
    DomainInfo(CreativeCodingDomain.PROCESSING, "Processing", "processing"),
    DomainInfo(CreativeCodingDomain.CANVAS_2D, "Canvas 2D", "canvas_2d"),
    DomainInfo(CreativeCodingDomain.WEBGPU_WGSL, "WebGPU/WGSL", "webgpu_wgsl"),
    DomainInfo(CreativeCodingDomain.GSAP, "GSAP", "gsap"),
    DomainInfo(CreativeCodingDomain.TONE_JS, "Tone.js", "tone_js"),
    DomainInfo(CreativeCodingDomain.PIXI_JS, "PixiJS", "pixi_js"),
    DomainInfo(CreativeCodingDomain.MATTER_JS, "Matter.js", "matter_js"),
    DomainInfo(CreativeCodingDomain.RAPIER, "Rapier", "rapier"),
    DomainInfo(CreativeCodingDomain.HYDRA, "Hydra", "hydra"),
    DomainInfo(CreativeCodingDomain.SHADERTOY, "Shadertoy", "shadertoy"),
    DomainInfo(CreativeCodingDomain.TOUCHDESIGNER, "TouchDesigner", "touchdesigner"),
    DomainInfo(CreativeCodingDomain.HOUDINI, "Houdini", "houdini"),
    DomainInfo(
        CreativeCodingDomain.BLENDER_GEOMETRY_NODES,
        "Blender / Geometry Nodes",
        "blender_geometry_nodes",
    ),
    DomainInfo(CreativeCodingDomain.UNITY, "Unity", "unity"),
    DomainInfo(CreativeCodingDomain.UNREAL, "Unreal", "unreal"),
    DomainInfo(CreativeCodingDomain.MAX_MSP, "Max/MSP", "max_msp"),
    DomainInfo(CreativeCodingDomain.NOTCH, "Notch", "notch"),
    DomainInfo(CreativeCodingDomain.VVVV, "VVVV", "vvvv"),
    DomainInfo(CreativeCodingDomain.OPENFRAMEWORKS, "openFrameworks", "openframeworks"),
    DomainInfo(CreativeCodingDomain.OPENRNDR, "OPENRNDR", "openrndr"),
    DomainInfo(CreativeCodingDomain.SUPERCOLLIDER, "SuperCollider", "supercollider"),
    DomainInfo(CreativeCodingDomain.SONIC_PI, "Sonic Pi", "sonic_pi"),
    DomainInfo(CreativeCodingDomain.TIDALCYCLES, "TidalCycles", "tidalcycles"),
    DomainInfo(CreativeCodingDomain.WEB_AUDIO_API, "Web Audio API", "web_audio_api"),
    DomainInfo(CreativeCodingDomain.P5_SOUND, "p5.sound", "p5_sound"),
    DomainInfo(CreativeCodingDomain.ML5_JS, "ml5.js", "ml5_js"),
    DomainInfo(CreativeCodingDomain.TENSORFLOW_JS, "TensorFlow.js", "tensorflow_js"),
    DomainInfo(CreativeCodingDomain.COMFYUI, "ComfyUI", "comfyui"),
    DomainInfo(
        CreativeCodingDomain.STABLE_DIFFUSION_WORKFLOWS,
        "Stable Diffusion workflows",
        "stable_diffusion_workflows",
    ),
    DomainInfo(CreativeCodingDomain.RUNWAY, "Runway", "runway"),
    DomainInfo(
        CreativeCodingDomain.BLENDER_PYTHON_API,
        "Blender Python API",
        "blender_python_api",
    ),
    DomainInfo(
        CreativeCodingDomain.UNREAL_BLUEPRINTS,
        "Unreal Blueprints",
        "unreal_blueprints",
    ),
    DomainInfo(CreativeCodingDomain.ABLETON_LIVE, "Ableton Live", "ableton_live"),
    DomainInfo(CreativeCodingDomain.VCV_RACK, "VCV Rack", "vcv_rack"),
    DomainInfo(CreativeCodingDomain.GODOT, "Godot", "godot"),
    DomainInfo(CreativeCodingDomain.RESOLUME, "Resolume", "resolume"),
    DomainInfo(CreativeCodingDomain.MADMAPPER, "MadMapper", "madmapper"),
    DomainInfo(CreativeCodingDomain.CABLES_GL, "Cables.gl", "cables_gl"),
    DomainInfo(CreativeCodingDomain.PURE_DATA, "Pure Data", "pure_data"),
)


def get_supported_domain_values() -> tuple[str, ...]:
    """Return stable domain values for clients and validators."""

    return tuple(domain.value.value for domain in SUPPORTED_DOMAINS)
