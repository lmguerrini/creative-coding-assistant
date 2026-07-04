"""Approved official source definitions for the knowledge base."""

from __future__ import annotations

from enum import StrEnum
from typing import Self
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.contracts import CreativeCodingDomain


class OfficialSourceType(StrEnum):
    API_REFERENCE = "api_reference"
    GUIDE = "guide"
    EXAMPLES = "examples"
    SPECIFICATION = "specification"


class SourceApprovalStatus(StrEnum):
    APPROVED = "approved"


OFFICIAL_HOSTS_BY_DOMAIN: dict[CreativeCodingDomain, tuple[str, ...]] = {
    CreativeCodingDomain.THREE_JS: ("threejs.org",),
    CreativeCodingDomain.REACT_THREE_FIBER: ("r3f.docs.pmnd.rs",),
    CreativeCodingDomain.P5_JS: ("p5js.org",),
    CreativeCodingDomain.GLSL: ("registry.khronos.org", "developer.mozilla.org"),
    CreativeCodingDomain.PROCESSING: ("processing.org",),
    CreativeCodingDomain.CANVAS_2D: ("developer.mozilla.org",),
    CreativeCodingDomain.WEBGPU_WGSL: ("developer.mozilla.org", "www.w3.org"),
    CreativeCodingDomain.GSAP: ("gsap.com",),
    CreativeCodingDomain.TONE_JS: ("tonejs.github.io",),
    CreativeCodingDomain.PIXI_JS: ("pixijs.com",),
    CreativeCodingDomain.MATTER_JS: ("www.brm.io",),
    CreativeCodingDomain.RAPIER: ("rapier.rs",),
    CreativeCodingDomain.HYDRA: ("hydra.ojack.xyz",),
    CreativeCodingDomain.SHADERTOY: ("www.shadertoy.com",),
    CreativeCodingDomain.TOUCHDESIGNER: ("derivative.ca",),
    CreativeCodingDomain.HOUDINI: ("www.sidefx.com",),
    CreativeCodingDomain.BLENDER_GEOMETRY_NODES: ("docs.blender.org",),
    CreativeCodingDomain.UNITY: ("docs.unity3d.com",),
    CreativeCodingDomain.UNREAL: ("dev.epicgames.com",),
    CreativeCodingDomain.MAX_MSP: ("docs.cycling74.com",),
    CreativeCodingDomain.NOTCH: ("manual.notch.one",),
    CreativeCodingDomain.VVVV: ("thegraybook.vvvv.org",),
    CreativeCodingDomain.OPENFRAMEWORKS: ("openframeworks.cc",),
    CreativeCodingDomain.OPENRNDR: ("guide.openrndr.org",),
    CreativeCodingDomain.SUPERCOLLIDER: ("docs.supercollider.online",),
    CreativeCodingDomain.SONIC_PI: ("sonic-pi.net",),
    CreativeCodingDomain.TIDALCYCLES: ("tidalcycles.org",),
    CreativeCodingDomain.WEB_AUDIO_API: ("developer.mozilla.org",),
    CreativeCodingDomain.P5_SOUND: ("p5js.org",),
    CreativeCodingDomain.ML5_JS: ("ml5js.org",),
    CreativeCodingDomain.TENSORFLOW_JS: ("www.tensorflow.org", "js.tensorflow.org"),
    CreativeCodingDomain.COMFYUI: ("docs.comfy.org",),
    CreativeCodingDomain.STABLE_DIFFUSION_WORKFLOWS: ("huggingface.co",),
    CreativeCodingDomain.RUNWAY: ("docs.dev.runwayml.com",),
    CreativeCodingDomain.BLENDER_PYTHON_API: ("docs.blender.org",),
    CreativeCodingDomain.UNREAL_BLUEPRINTS: ("dev.epicgames.com",),
    CreativeCodingDomain.ABLETON_LIVE: ("www.ableton.com",),
    CreativeCodingDomain.VCV_RACK: ("vcvrack.com",),
    CreativeCodingDomain.GODOT: ("docs.godotengine.org",),
    CreativeCodingDomain.RESOLUME: ("www.resolume.com",),
    CreativeCodingDomain.MADMAPPER: ("docs.madmapper.com",),
    CreativeCodingDomain.CABLES_GL: ("cables.gl",),
    CreativeCodingDomain.PURE_DATA: ("puredata.info",),
}


class OfficialSource(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    domain: CreativeCodingDomain
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    url: str = Field(min_length=1)
    source_type: OfficialSourceType
    approval_status: SourceApprovalStatus = SourceApprovalStatus.APPROVED
    priority: int = Field(ge=1)
    allowed_path_prefixes: tuple[str, ...] = Field(min_length=1)
    additional_urls: tuple[str, ...] = Field(default_factory=tuple)
    tags: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("url")
    @classmethod
    def require_https_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Official source URLs must be absolute HTTPS URLs.")
        return value

    @field_validator("allowed_path_prefixes")
    @classmethod
    def validate_path_prefixes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for prefix in value:
            if not prefix.startswith("/"):
                raise ValueError("Allowed path prefixes must start with '/'.")
        return value

    @field_validator("additional_urls")
    @classmethod
    def validate_additional_urls(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for url in value:
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError(
                    "Additional official source URLs must be absolute HTTPS URLs."
                )
        return value

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for tag in value:
            if not tag or tag != tag.lower() or " " in tag:
                raise ValueError("Source tags must be lowercase tokens.")
        return value

    @model_validator(mode="after")
    def validate_official_scope(self) -> Self:
        allowed_hosts = OFFICIAL_HOSTS_BY_DOMAIN[self.domain]
        for url in (self.url, *self.additional_urls):
            parsed = urlparse(url)
            if parsed.hostname not in allowed_hosts:
                raise ValueError(
                    "Official source host is not approved for this domain."
                )

            path_is_allowed = any(
                parsed.path.startswith(prefix) for prefix in self.allowed_path_prefixes
            )
            if not path_is_allowed:
                raise ValueError(
                    "Official source URL is outside its approved path scope."
                )

        return self


def _validate_source_registry(
    sources: tuple[OfficialSource, ...],
) -> tuple[OfficialSource, ...]:
    source_ids = [source.source_id for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Official source IDs must be unique.")

    registered_domains = {source.domain for source in sources}
    missing_domains = tuple(
        domain for domain in CreativeCodingDomain if domain not in registered_domains
    )
    if missing_domains:
        values = ", ".join(domain.value for domain in missing_domains)
        raise ValueError(f"Official sources missing domains: {values}")

    return sources


APPROVED_OFFICIAL_SOURCES: tuple[OfficialSource, ...] = _validate_source_registry(
    (
        OfficialSource(
            source_id="three_docs",
            domain=CreativeCodingDomain.THREE_JS,
            title="three.js Documentation",
            publisher="three.js",
            url="https://threejs.org/docs/",
            source_type=OfficialSourceType.API_REFERENCE,
            priority=10,
            allowed_path_prefixes=("/docs/",),
            tags=("reference", "api", "webgl"),
        ),
        OfficialSource(
            source_id="three_manual",
            domain=CreativeCodingDomain.THREE_JS,
            title="three.js Manual",
            publisher="three.js",
            url="https://threejs.org/manual/",
            source_type=OfficialSourceType.GUIDE,
            priority=20,
            allowed_path_prefixes=("/manual/",),
            additional_urls=(
                "https://threejs.org/manual/en/fundamentals.html",
                "https://threejs.org/manual/en/responsive.html",
                "https://threejs.org/manual/en/cameras.html",
                "https://threejs.org/manual/en/lights.html",
            ),
            tags=("guide", "fundamentals", "webgl"),
        ),
        OfficialSource(
            source_id="three_manual_effects",
            domain=CreativeCodingDomain.THREE_JS,
            title="three.js Effects and Render Pipeline Manual",
            publisher="three.js",
            url="https://threejs.org/manual/en/post-processing.html",
            source_type=OfficialSourceType.GUIDE,
            priority=25,
            allowed_path_prefixes=("/manual/",),
            additional_urls=(
                "https://threejs.org/manual/en/shadows.html",
                "https://threejs.org/manual/en/rendertargets.html",
            ),
            tags=("guide", "postprocessing", "debugging"),
        ),
        OfficialSource(
            source_id="three_examples",
            domain=CreativeCodingDomain.THREE_JS,
            title="three.js Examples",
            publisher="three.js",
            url="https://threejs.org/examples/",
            source_type=OfficialSourceType.EXAMPLES,
            priority=30,
            allowed_path_prefixes=("/examples/",),
            tags=("examples", "patterns", "webgl"),
        ),
        OfficialSource(
            source_id="r3f_introduction",
            domain=CreativeCodingDomain.REACT_THREE_FIBER,
            title="React Three Fiber Introduction",
            publisher="pmndrs",
            url="https://r3f.docs.pmnd.rs/getting-started/introduction",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/getting-started/",),
            tags=("guide", "react", "three"),
        ),
        OfficialSource(
            source_id="r3f_canvas_api",
            domain=CreativeCodingDomain.REACT_THREE_FIBER,
            title="React Three Fiber Canvas API",
            publisher="pmndrs",
            url="https://r3f.docs.pmnd.rs/api/canvas",
            source_type=OfficialSourceType.API_REFERENCE,
            priority=20,
            allowed_path_prefixes=("/api/",),
            tags=("reference", "canvas", "react"),
        ),
        OfficialSource(
            source_id="r3f_hooks_api",
            domain=CreativeCodingDomain.REACT_THREE_FIBER,
            title="React Three Fiber Hooks API",
            publisher="pmndrs",
            url="https://r3f.docs.pmnd.rs/api/hooks",
            source_type=OfficialSourceType.API_REFERENCE,
            priority=30,
            allowed_path_prefixes=("/api/",),
            tags=("reference", "hooks", "react"),
        ),
        OfficialSource(
            source_id="p5_reference",
            domain=CreativeCodingDomain.P5_JS,
            title="p5.js Core Sketch Reference",
            publisher="p5.js",
            url="https://p5js.org/reference/p5/setup/",
            source_type=OfficialSourceType.API_REFERENCE,
            priority=10,
            allowed_path_prefixes=("/reference/p5/",),
            additional_urls=(
                "https://p5js.org/reference/p5/draw/",
                "https://p5js.org/reference/p5/createCanvas/",
                "https://p5js.org/reference/p5/background/",
                "https://p5js.org/reference/p5/circle/",
                "https://p5js.org/reference/p5/ellipse/",
                "https://p5js.org/reference/p5/fill/",
                "https://p5js.org/reference/p5/frameCount/",
                "https://p5js.org/reference/p5/mouseX/",
                "https://p5js.org/reference/p5/mouseY/",
                "https://p5js.org/reference/p5/random/",
                "https://p5js.org/reference/p5/userStartAudio/",
            ),
            tags=("reference", "api", "sketch"),
        ),
        OfficialSource(
            source_id="p5_tutorials",
            domain=CreativeCodingDomain.P5_JS,
            title="p5.js Motion Tutorials",
            publisher="p5.js",
            url="https://p5js.org/tutorials/get-started/",
            source_type=OfficialSourceType.GUIDE,
            priority=20,
            allowed_path_prefixes=("/tutorials/",),
            additional_urls=(
                "https://p5js.org/tutorials/variables-and-change",
                "https://p5js.org/tutorials/conditionals-and-interactivity/",
            ),
            tags=("guide", "animation", "creative-coding"),
        ),
        OfficialSource(
            source_id="p5_examples",
            domain=CreativeCodingDomain.P5_JS,
            title="p5.js Runnable Examples",
            publisher="p5.js",
            url="https://p5js.org/examples/calculating-values-constrain/",
            source_type=OfficialSourceType.EXAMPLES,
            priority=30,
            allowed_path_prefixes=("/examples/",),
            additional_urls=(
                "https://p5js.org/examples/animation-and-variables-drawing-lines/",
                "https://p5js.org/examples/angles-and-motion-sine-cosine/",
                "https://p5js.org/examples/games-circle-clicker/",
            ),
            tags=("examples", "patterns", "animation"),
        ),
        OfficialSource(
            source_id="glsl_language_spec_460",
            domain=CreativeCodingDomain.GLSL,
            title="OpenGL Shading Language 4.60 Specification",
            publisher="Khronos Group",
            url="https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html",
            source_type=OfficialSourceType.SPECIFICATION,
            priority=10,
            allowed_path_prefixes=("/OpenGL/specs/gl/",),
            tags=("specification", "opengl", "glsl"),
        ),
        OfficialSource(
            source_id="glsl_mdn_webgl_examples",
            domain=CreativeCodingDomain.GLSL,
            title="MDN WebGL GLSL Examples",
            publisher="MDN",
            url="https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/By_example/Hello_GLSL",
            source_type=OfficialSourceType.EXAMPLES,
            priority=15,
            allowed_path_prefixes=(
                "/en-US/docs/Web/API/WebGL_API/By_example/",
                "/en-US/docs/Web/API/WebGL_API/Tutorial/",
            ),
            additional_urls=(
                "https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/By_example/Textures_from_code",
                "https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial/Using_shaders_to_apply_color_in_WebGL",
            ),
            tags=("examples", "fragment-shader", "webgl"),
        ),
        OfficialSource(
            source_id="glsl_es_language_spec_320",
            domain=CreativeCodingDomain.GLSL,
            title="OpenGL ES Shading Language 3.20 Specification",
            publisher="Khronos Group",
            url=(
                "https://registry.khronos.org/OpenGL/specs/es/3.2/"
                "GLSL_ES_Specification_3.20.html"
            ),
            source_type=OfficialSourceType.SPECIFICATION,
            priority=20,
            allowed_path_prefixes=("/OpenGL/specs/es/3.2/",),
            tags=("specification", "opengl-es", "glsl-es"),
        ),
        OfficialSource(
            source_id="processing_reference",
            domain=CreativeCodingDomain.PROCESSING,
            title="Processing Reference",
            publisher="Processing Foundation",
            url="https://processing.org/reference/",
            source_type=OfficialSourceType.API_REFERENCE,
            priority=10,
            allowed_path_prefixes=("/reference/",),
            tags=("reference", "api", "sketch"),
        ),
        OfficialSource(
            source_id="canvas2d_context_api",
            domain=CreativeCodingDomain.CANVAS_2D,
            title="CanvasRenderingContext2D API",
            publisher="MDN",
            url="https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D",
            source_type=OfficialSourceType.API_REFERENCE,
            priority=10,
            allowed_path_prefixes=("/en-US/docs/Web/API/CanvasRenderingContext2D",),
            tags=("reference", "api", "canvas"),
        ),
        OfficialSource(
            source_id="webgpu_mdn_api",
            domain=CreativeCodingDomain.WEBGPU_WGSL,
            title="WebGPU API",
            publisher="MDN",
            url="https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/en-US/docs/Web/API/WebGPU_API",),
            tags=("guide", "api", "webgpu"),
        ),
        OfficialSource(
            source_id="wgsl_spec",
            domain=CreativeCodingDomain.WEBGPU_WGSL,
            title="WebGPU Shading Language Specification",
            publisher="W3C",
            url="https://www.w3.org/TR/WGSL/",
            source_type=OfficialSourceType.SPECIFICATION,
            priority=20,
            allowed_path_prefixes=("/TR/WGSL/",),
            tags=("specification", "wgsl", "shader"),
        ),
        OfficialSource(
            source_id="gsap_docs",
            domain=CreativeCodingDomain.GSAP,
            title="GSAP Documentation",
            publisher="GSAP",
            url="https://gsap.com/docs/v3/",
            source_type=OfficialSourceType.API_REFERENCE,
            priority=10,
            allowed_path_prefixes=("/docs/v3/",),
            tags=("reference", "animation", "timeline"),
        ),
        OfficialSource(
            source_id="tone_js_docs",
            domain=CreativeCodingDomain.TONE_JS,
            title="Tone.js Documentation",
            publisher="Tone.js",
            url="https://tonejs.github.io/docs/15.1.22/index.html",
            source_type=OfficialSourceType.API_REFERENCE,
            priority=10,
            allowed_path_prefixes=("/docs/",),
            tags=("reference", "audio", "synthesis"),
        ),
        OfficialSource(
            source_id="tone_js_analysis_reference",
            domain=CreativeCodingDomain.TONE_JS,
            title="Tone.js Analysis and Playback Reference",
            publisher="Tone.js",
            url="https://tonejs.github.io/docs/15.1.22/classes/FFT.html",
            source_type=OfficialSourceType.API_REFERENCE,
            priority=20,
            allowed_path_prefixes=("/docs/15.1.22/",),
            additional_urls=(
                "https://tonejs.github.io/docs/15.1.22/classes/Meter.html",
                "https://tonejs.github.io/docs/15.1.22/classes/Loop.html",
                "https://tonejs.github.io/docs/15.1.22/classes/Player.html",
            ),
            tags=("reference", "audio-reactive", "playback"),
        ),
        OfficialSource(
            source_id="pixi_js_guides",
            domain=CreativeCodingDomain.PIXI_JS,
            title="PixiJS Guides",
            publisher="PixiJS",
            url="https://pixijs.com/8.x/guides/getting-started/intro",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/8.x/guides/",),
            tags=("guide", "renderer", "webgl"),
        ),
        OfficialSource(
            source_id="matter_js_docs",
            domain=CreativeCodingDomain.MATTER_JS,
            title="Matter.js Documentation",
            publisher="Matter.js",
            url="https://www.brm.io/matter-js/docs/",
            source_type=OfficialSourceType.API_REFERENCE,
            priority=10,
            allowed_path_prefixes=("/matter-js/docs/",),
            tags=("reference", "physics", "engine"),
        ),
        OfficialSource(
            source_id="rapier_js_getting_started",
            domain=CreativeCodingDomain.RAPIER,
            title="Rapier JavaScript Getting Started",
            publisher="Dimforge",
            url="https://rapier.rs/docs/user_guides/templates/getting_started_js/",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/docs/",),
            tags=("guide", "physics", "javascript"),
        ),
        OfficialSource(
            source_id="hydra_docs",
            domain=CreativeCodingDomain.HYDRA,
            title="Hydra Documentation",
            publisher="Hydra",
            url="https://hydra.ojack.xyz/docs",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/docs",),
            tags=("guide", "live-coding", "video"),
        ),
        OfficialSource(
            source_id="shadertoy_howto",
            domain=CreativeCodingDomain.SHADERTOY,
            title="Shadertoy How To",
            publisher="Shadertoy",
            url="https://www.shadertoy.com/howto",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/howto",),
            tags=("guide", "shader", "glsl"),
        ),
        OfficialSource(
            source_id="touchdesigner_user_guide",
            domain=CreativeCodingDomain.TOUCHDESIGNER,
            title="TouchDesigner User Guide",
            publisher="Derivative",
            url="https://derivative.ca/UserGuide/Getting_started",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/UserGuide/",),
            tags=("guide", "operators", "realtime"),
        ),
        OfficialSource(
            source_id="houdini_docs",
            domain=CreativeCodingDomain.HOUDINI,
            title="Houdini Documentation",
            publisher="SideFX",
            url="https://www.sidefx.com/docs/houdini/",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/docs/houdini/",),
            tags=("guide", "procedural", "vfx"),
        ),
        OfficialSource(
            source_id="blender_geometry_nodes_manual",
            domain=CreativeCodingDomain.BLENDER_GEOMETRY_NODES,
            title="Blender Geometry Nodes Manual",
            publisher="Blender Foundation",
            url=(
                "https://docs.blender.org/manual/en/latest/modeling/"
                "geometry_nodes/index.html"
            ),
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/manual/en/latest/modeling/geometry_nodes/",),
            tags=("guide", "nodes", "procedural"),
        ),
        OfficialSource(
            source_id="unity_manual",
            domain=CreativeCodingDomain.UNITY,
            title="Unity Manual",
            publisher="Unity",
            url="https://docs.unity3d.com/Manual/index.html",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/Manual/", "/ScriptReference/"),
            additional_urls=("https://docs.unity3d.com/ScriptReference/index.html",),
            tags=("guide", "engine", "csharp"),
        ),
        OfficialSource(
            source_id="unreal_engine_docs",
            domain=CreativeCodingDomain.UNREAL,
            title="Unreal Engine Documentation",
            publisher="Epic Games",
            url="https://dev.epicgames.com/documentation/en-us/unreal-engine/get-started",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/documentation/en-us/unreal-engine/",),
            tags=("guide", "engine", "blueprints"),
        ),
        OfficialSource(
            source_id="max_msp_docs",
            domain=CreativeCodingDomain.MAX_MSP,
            title="Max Documentation",
            publisher="Cycling '74",
            url="https://docs.cycling74.com/",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/",),
            tags=("guide", "audio", "patching"),
        ),
        OfficialSource(
            source_id="notch_manual",
            domain=CreativeCodingDomain.NOTCH,
            title="Notch Manual",
            publisher="Notch",
            url="https://manual.notch.one/1.0/en/docs/",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/1.0/en/docs/",),
            tags=("guide", "realtime", "vfx"),
        ),
        OfficialSource(
            source_id="vvvv_gamma_docs",
            domain=CreativeCodingDomain.VVVV,
            title="vvvv gamma Documentation",
            publisher="vvvv",
            url="https://thegraybook.vvvv.org/",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/",),
            tags=("guide", "visual-programming", "vl"),
        ),
        OfficialSource(
            source_id="openframeworks_docs",
            domain=CreativeCodingDomain.OPENFRAMEWORKS,
            title="openFrameworks Documentation",
            publisher="openFrameworks",
            url="https://openframeworks.cc/documentation/",
            source_type=OfficialSourceType.API_REFERENCE,
            priority=10,
            allowed_path_prefixes=("/documentation/",),
            tags=("reference", "cpp", "creative-coding"),
        ),
        OfficialSource(
            source_id="openrndr_guide",
            domain=CreativeCodingDomain.OPENRNDR,
            title="OPENRNDR Guide",
            publisher="OPENRNDR",
            url="https://guide.openrndr.org/",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/",),
            tags=("guide", "kotlin", "creative-coding"),
        ),
        OfficialSource(
            source_id="supercollider_help",
            domain=CreativeCodingDomain.SUPERCOLLIDER,
            title="SuperCollider Help",
            publisher="SuperCollider",
            url="https://docs.supercollider.online/",
            source_type=OfficialSourceType.API_REFERENCE,
            priority=10,
            allowed_path_prefixes=("/",),
            tags=("reference", "audio", "live-coding"),
        ),
        OfficialSource(
            source_id="sonic_pi_tutorial",
            domain=CreativeCodingDomain.SONIC_PI,
            title="Sonic Pi Tutorial",
            publisher="Sonic Pi",
            url="https://sonic-pi.net/tutorial.html",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/tutorial.html",),
            tags=("guide", "audio", "live-coding"),
        ),
        OfficialSource(
            source_id="tidalcycles_docs",
            domain=CreativeCodingDomain.TIDALCYCLES,
            title="TidalCycles Documentation",
            publisher="TidalCycles",
            url="https://tidalcycles.org/docs/",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/docs/",),
            tags=("guide", "patterns", "live-coding"),
        ),
        OfficialSource(
            source_id="web_audio_mdn_api",
            domain=CreativeCodingDomain.WEB_AUDIO_API,
            title="Web Audio API",
            publisher="MDN",
            url="https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API",
            source_type=OfficialSourceType.API_REFERENCE,
            priority=10,
            allowed_path_prefixes=("/en-US/docs/Web/API/Web_Audio_API",),
            tags=("reference", "audio", "browser"),
        ),
        OfficialSource(
            source_id="web_audio_analyser_node",
            domain=CreativeCodingDomain.WEB_AUDIO_API,
            title="AnalyserNode",
            publisher="MDN",
            url="https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode",
            source_type=OfficialSourceType.API_REFERENCE,
            priority=15,
            allowed_path_prefixes=("/en-US/docs/Web/API/AnalyserNode",),
            tags=("reference", "audio-reactive", "analysis"),
        ),
        OfficialSource(
            source_id="web_audio_visualization_guide",
            domain=CreativeCodingDomain.WEB_AUDIO_API,
            title="Web Audio Visualization and Best Practices",
            publisher="MDN",
            url=(
                "https://developer.mozilla.org/en-US/docs/Web/API/"
                "Web_Audio_API/Visualizations_with_Web_Audio_API"
            ),
            source_type=OfficialSourceType.GUIDE,
            priority=20,
            allowed_path_prefixes=("/en-US/docs/Web/API/Web_Audio_API",),
            additional_urls=(
                "https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Using_Web_Audio_API",
                "https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Best_practices",
            ),
            tags=("guide", "audio-reactive", "debugging"),
        ),
        OfficialSource(
            source_id="p5_sound_reference",
            domain=CreativeCodingDomain.P5_SOUND,
            title="p5.sound Reference",
            publisher="p5.js",
            url="https://p5js.org/reference/p5.sound/",
            source_type=OfficialSourceType.API_REFERENCE,
            priority=10,
            allowed_path_prefixes=("/reference/p5.sound/",),
            tags=("reference", "audio", "p5"),
        ),
        OfficialSource(
            source_id="p5_sound_analysis_reference",
            domain=CreativeCodingDomain.P5_SOUND,
            title="p5.sound Analysis Reference",
            publisher="p5.js",
            url="https://p5js.org/reference/p5.sound/p5.Amplitude/",
            source_type=OfficialSourceType.API_REFERENCE,
            priority=20,
            allowed_path_prefixes=("/reference/p5.sound/",),
            additional_urls=("https://p5js.org/reference/p5.sound/p5.FFT/",),
            tags=("reference", "audio-reactive", "analysis"),
        ),
        OfficialSource(
            source_id="ml5_js_learn",
            domain=CreativeCodingDomain.ML5_JS,
            title="ml5.js Learning Resources",
            publisher="ml5.js",
            url="https://ml5js.org/learn",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/learn",),
            tags=("guide", "machine-learning", "browser"),
        ),
        OfficialSource(
            source_id="tensorflow_js_guide",
            domain=CreativeCodingDomain.TENSORFLOW_JS,
            title="TensorFlow.js Guide",
            publisher="TensorFlow",
            url="https://www.tensorflow.org/js/guide",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/js/guide", "/api/latest/"),
            additional_urls=("https://js.tensorflow.org/api/latest/",),
            tags=("guide", "machine-learning", "browser"),
        ),
        OfficialSource(
            source_id="comfyui_docs",
            domain=CreativeCodingDomain.COMFYUI,
            title="ComfyUI Official Documentation",
            publisher="ComfyUI",
            url="https://docs.comfy.org/index",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/index",),
            tags=("guide", "node-workflow", "generative-ai"),
        ),
        OfficialSource(
            source_id="stable_diffusion_diffusers",
            domain=CreativeCodingDomain.STABLE_DIFFUSION_WORKFLOWS,
            title="Stable Diffusion Pipelines",
            publisher="Hugging Face Diffusers",
            url=(
                "https://huggingface.co/docs/diffusers/api/pipelines/"
                "stable_diffusion/overview"
            ),
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/docs/diffusers/api/pipelines/stable_diffusion/",),
            tags=("guide", "diffusion", "generative-ai"),
        ),
        OfficialSource(
            source_id="runway_api_docs",
            domain=CreativeCodingDomain.RUNWAY,
            title="Runway API Documentation",
            publisher="Runway",
            url="https://docs.dev.runwayml.com/",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/",),
            tags=("guide", "generative-ai", "video"),
        ),
        OfficialSource(
            source_id="blender_python_api",
            domain=CreativeCodingDomain.BLENDER_PYTHON_API,
            title="Blender Python API",
            publisher="Blender Foundation",
            url="https://docs.blender.org/api/current/",
            source_type=OfficialSourceType.API_REFERENCE,
            priority=10,
            allowed_path_prefixes=("/api/current/",),
            tags=("reference", "python", "dcc"),
        ),
        OfficialSource(
            source_id="unreal_blueprints_docs",
            domain=CreativeCodingDomain.UNREAL_BLUEPRINTS,
            title="Unreal Engine Blueprint Documentation",
            publisher="Epic Games",
            url=(
                "https://dev.epicgames.com/documentation/en-us/unreal-engine/"
                "introduction-to-blueprints"
            ),
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/documentation/en-us/unreal-engine/",),
            tags=("guide", "blueprints", "engine"),
        ),
        OfficialSource(
            source_id="ableton_live_manual",
            domain=CreativeCodingDomain.ABLETON_LIVE,
            title="Ableton Live Manual",
            publisher="Ableton",
            url="https://www.ableton.com/en/live-manual/12/welcome-to-live/",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/en/live-manual/12/",),
            tags=("guide", "daw", "audio"),
        ),
        OfficialSource(
            source_id="vcv_rack_manual",
            domain=CreativeCodingDomain.VCV_RACK,
            title="VCV Rack Manual",
            publisher="VCV Rack",
            url="https://vcvrack.com/manual/index",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/manual/",),
            tags=("guide", "modular", "synthesis"),
        ),
        OfficialSource(
            source_id="godot_docs",
            domain=CreativeCodingDomain.GODOT,
            title="Godot Documentation",
            publisher="Godot Engine",
            url="https://docs.godotengine.org/en/stable/",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/en/stable/",),
            tags=("guide", "engine", "gdscript"),
        ),
        OfficialSource(
            source_id="resolume_arena_manual",
            domain=CreativeCodingDomain.RESOLUME,
            title="Resolume Arena Manual",
            publisher="Resolume",
            url="https://www.resolume.com/support/en/arena/manual",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/support/en/",),
            tags=("guide", "vj", "av"),
        ),
        OfficialSource(
            source_id="madmapper_docs",
            domain=CreativeCodingDomain.MADMAPPER,
            title="MadMapper Documentation",
            publisher="MadMapper",
            url="https://docs.madmapper.com/",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/",),
            tags=("guide", "projection-mapping", "av"),
        ),
        OfficialSource(
            source_id="cables_gl_docs",
            domain=CreativeCodingDomain.CABLES_GL,
            title="Cables.gl Documentation",
            publisher="Cables.gl",
            url="https://cables.gl/docs/docs",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/docs",),
            tags=("guide", "patching", "realtime"),
        ),
        OfficialSource(
            source_id="pure_data_manual",
            domain=CreativeCodingDomain.PURE_DATA,
            title="Pure Data HTML Manual",
            publisher="Pd Community Site",
            url="https://puredata.info/docs/manuals/pd",
            source_type=OfficialSourceType.GUIDE,
            priority=10,
            allowed_path_prefixes=("/docs/",),
            tags=("guide", "patching", "audio"),
        ),
    )
)


def approved_official_sources() -> tuple[OfficialSource, ...]:
    return APPROVED_OFFICIAL_SOURCES


def approved_sources_for_domain(
    domain: CreativeCodingDomain,
) -> tuple[OfficialSource, ...]:
    sources = (
        source for source in APPROVED_OFFICIAL_SOURCES if source.domain == domain
    )
    return tuple(
        sorted(sources, key=lambda source: (source.priority, source.source_id))
    )


def official_source_domains() -> tuple[CreativeCodingDomain, ...]:
    domains = {source.domain for source in APPROVED_OFFICIAL_SOURCES}
    return tuple(domain for domain in CreativeCodingDomain if domain in domains)


def get_official_source(source_id: str) -> OfficialSource:
    for source in APPROVED_OFFICIAL_SOURCES:
        if source.source_id == source_id:
            return source
    raise ValueError(f"Unknown official source: {source_id}")
