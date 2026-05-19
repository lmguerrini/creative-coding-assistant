import unittest

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.domains import (
    SUPPORTED_DOMAINS,
    DomainCategory,
    get_domain_categories,
    get_domain_category,
    get_domain_category_label,
    get_domain_default_topic,
    get_domain_info,
    get_domain_label,
    get_domain_memory_label,
    get_domain_prompt_guidance,
    get_domain_slug,
    get_domains_for_category,
    get_supported_domain_values,
)
from creative_coding_assistant.rag.sources import official_source_domains

_EXPECTED_DOMAIN_METADATA = (
    (CreativeCodingDomain.THREE_JS, "Three.js", "three_js", "scene code"),
    (
        CreativeCodingDomain.REACT_THREE_FIBER,
        "React Three Fiber",
        "r3f",
        "component code",
    ),
    (CreativeCodingDomain.P5_JS, "p5.js", "p5_js", "sketch"),
    (CreativeCodingDomain.GLSL, "GLSL", "glsl", "shader code"),
    (CreativeCodingDomain.PROCESSING, "Processing", "processing", "sketch"),
    (
        CreativeCodingDomain.CANVAS_2D,
        "Canvas 2D",
        "canvas_2d",
        "canvas sketch",
    ),
    (
        CreativeCodingDomain.WEBGPU_WGSL,
        "WebGPU/WGSL",
        "webgpu_wgsl",
        "WebGPU/WGSL code",
    ),
    (CreativeCodingDomain.GSAP, "GSAP", "gsap", "animation code"),
    (CreativeCodingDomain.TONE_JS, "Tone.js", "tone_js", "audio code"),
    (CreativeCodingDomain.PIXI_JS, "PixiJS", "pixi_js", "rendering code"),
    (CreativeCodingDomain.MATTER_JS, "Matter.js", "matter_js", "physics code"),
    (CreativeCodingDomain.RAPIER, "Rapier", "rapier", "physics code"),
    (CreativeCodingDomain.HYDRA, "Hydra", "hydra", "live-coding sketch"),
    (CreativeCodingDomain.SHADERTOY, "Shadertoy", "shadertoy", "shader code"),
    (
        CreativeCodingDomain.TOUCHDESIGNER,
        "TouchDesigner",
        "touchdesigner",
        "operator network guidance",
    ),
    (
        CreativeCodingDomain.HOUDINI,
        "Houdini",
        "houdini",
        "procedural workflow guidance",
    ),
    (
        CreativeCodingDomain.BLENDER_GEOMETRY_NODES,
        "Blender / Geometry Nodes",
        "blender_geometry_nodes",
        "geometry nodes guidance",
    ),
    (CreativeCodingDomain.UNITY, "Unity", "unity", "engine workflow guidance"),
    (CreativeCodingDomain.UNREAL, "Unreal", "unreal", "engine workflow guidance"),
    (CreativeCodingDomain.MAX_MSP, "Max/MSP", "max_msp", "patching guidance"),
    (
        CreativeCodingDomain.NOTCH,
        "Notch",
        "notch",
        "realtime VFX workflow guidance",
    ),
    (CreativeCodingDomain.VVVV, "VVVV", "vvvv", "visual programming guidance"),
    (
        CreativeCodingDomain.OPENFRAMEWORKS,
        "openFrameworks",
        "openframeworks",
        "creative coding framework guidance",
    ),
    (
        CreativeCodingDomain.OPENRNDR,
        "OPENRNDR",
        "openrndr",
        "creative coding framework guidance",
    ),
    (
        CreativeCodingDomain.SUPERCOLLIDER,
        "SuperCollider",
        "supercollider",
        "audio live-coding guidance",
    ),
    (
        CreativeCodingDomain.SONIC_PI,
        "Sonic Pi",
        "sonic_pi",
        "audio live-coding guidance",
    ),
    (
        CreativeCodingDomain.TIDALCYCLES,
        "TidalCycles",
        "tidalcycles",
        "pattern live-coding guidance",
    ),
    (
        CreativeCodingDomain.WEB_AUDIO_API,
        "Web Audio API",
        "web_audio_api",
        "browser audio code",
    ),
    (
        CreativeCodingDomain.P5_SOUND,
        "p5.sound",
        "p5_sound",
        "p5.sound audio code",
    ),
    (CreativeCodingDomain.ML5_JS, "ml5.js", "ml5_js", "browser ML code"),
    (
        CreativeCodingDomain.TENSORFLOW_JS,
        "TensorFlow.js",
        "tensorflow_js",
        "browser ML code",
    ),
    (
        CreativeCodingDomain.COMFYUI,
        "ComfyUI",
        "comfyui",
        "node workflow guidance",
    ),
    (
        CreativeCodingDomain.STABLE_DIFFUSION_WORKFLOWS,
        "Stable Diffusion workflows",
        "stable_diffusion_workflows",
        "diffusion workflow guidance",
    ),
    (
        CreativeCodingDomain.RUNWAY,
        "Runway",
        "runway",
        "creative AI workflow guidance",
    ),
    (
        CreativeCodingDomain.BLENDER_PYTHON_API,
        "Blender Python API",
        "blender_python_api",
        "Blender scripting guidance",
    ),
    (
        CreativeCodingDomain.UNREAL_BLUEPRINTS,
        "Unreal Blueprints",
        "unreal_blueprints",
        "Blueprint workflow guidance",
    ),
    (
        CreativeCodingDomain.ABLETON_LIVE,
        "Ableton Live",
        "ableton_live",
        "DAW workflow guidance",
    ),
    (
        CreativeCodingDomain.VCV_RACK,
        "VCV Rack",
        "vcv_rack",
        "modular synthesis guidance",
    ),
    (CreativeCodingDomain.GODOT, "Godot", "godot", "engine workflow guidance"),
    (CreativeCodingDomain.RESOLUME, "Resolume", "resolume", "AV workflow guidance"),
    (
        CreativeCodingDomain.MADMAPPER,
        "MadMapper",
        "madmapper",
        "projection mapping guidance",
    ),
    (
        CreativeCodingDomain.CABLES_GL,
        "Cables.gl",
        "cables_gl",
        "visual patching guidance",
    ),
    (CreativeCodingDomain.PURE_DATA, "Pure Data", "pure_data", "patching guidance"),
)

_EXPECTED_PROMPT_GUIDANCE_SNIPPETS = (
    (CreativeCodingDomain.THREE_JS, "Prefer plain Three.js patterns"),
    (CreativeCodingDomain.REACT_THREE_FIBER, "Prefer React Three Fiber components"),
    (CreativeCodingDomain.P5_JS, "Prefer p5.js sketch structure"),
    (CreativeCodingDomain.GLSL, "Prefer concrete shader snippets"),
    (CreativeCodingDomain.PROCESSING, "Prefer Processing sketch structure"),
    (CreativeCodingDomain.CANVAS_2D, "Prefer standard CanvasRenderingContext2D APIs"),
    (CreativeCodingDomain.WEBGPU_WGSL, "Prefer WebGPU host setup"),
    (CreativeCodingDomain.GSAP, "Prefer GSAP tweens and timelines"),
    (CreativeCodingDomain.TONE_JS, "Prefer Tone.js Transport"),
    (CreativeCodingDomain.PIXI_JS, "Prefer PixiJS Application"),
    (CreativeCodingDomain.MATTER_JS, "Prefer Matter.js Engine"),
    (CreativeCodingDomain.RAPIER, "Prefer Rapier rigid bodies"),
    (CreativeCodingDomain.HYDRA, "Prefer Hydra live-coding chains"),
    (CreativeCodingDomain.SHADERTOY, "Prefer Shadertoy GLSL structure"),
    (CreativeCodingDomain.TOUCHDESIGNER, "Treat TouchDesigner as an external"),
    (CreativeCodingDomain.HOUDINI, "Treat Houdini as an external"),
    (CreativeCodingDomain.BLENDER_GEOMETRY_NODES, "Treat Blender Geometry Nodes"),
    (CreativeCodingDomain.UNITY, "Treat Unity as an external"),
    (CreativeCodingDomain.UNREAL, "Treat Unreal as an external"),
    (CreativeCodingDomain.MAX_MSP, "Treat Max/MSP as an external"),
    (CreativeCodingDomain.NOTCH, "Treat Notch as an external"),
    (CreativeCodingDomain.VVVV, "Treat vvvv gamma as an external"),
    (CreativeCodingDomain.OPENFRAMEWORKS, "Treat openFrameworks as an external"),
    (CreativeCodingDomain.OPENRNDR, "Treat OPENRNDR as an external"),
    (CreativeCodingDomain.SUPERCOLLIDER, "Treat SuperCollider as an external"),
    (CreativeCodingDomain.SONIC_PI, "Treat Sonic Pi as an external"),
    (CreativeCodingDomain.TIDALCYCLES, "Treat TidalCycles as an external"),
    (CreativeCodingDomain.WEB_AUDIO_API, "Prefer standard Web Audio API"),
    (CreativeCodingDomain.P5_SOUND, "Prefer p5.sound APIs"),
    (CreativeCodingDomain.ML5_JS, "Prefer ml5.js browser ML APIs"),
    (CreativeCodingDomain.TENSORFLOW_JS, "Prefer TensorFlow.js APIs"),
    (CreativeCodingDomain.COMFYUI, "Treat ComfyUI as an external"),
    (CreativeCodingDomain.STABLE_DIFFUSION_WORKFLOWS, "Treat Stable Diffusion"),
    (CreativeCodingDomain.RUNWAY, "Treat Runway as an external"),
    (CreativeCodingDomain.BLENDER_PYTHON_API, "Treat Blender Python"),
    (CreativeCodingDomain.UNREAL_BLUEPRINTS, "Treat Unreal Blueprints"),
    (CreativeCodingDomain.ABLETON_LIVE, "Treat Ableton Live"),
    (CreativeCodingDomain.VCV_RACK, "Treat VCV Rack"),
    (CreativeCodingDomain.GODOT, "Treat Godot as an external"),
    (CreativeCodingDomain.RESOLUME, "Treat Resolume as an external"),
    (CreativeCodingDomain.MADMAPPER, "Treat MadMapper as an external"),
    (CreativeCodingDomain.CABLES_GL, "Treat Cables.gl as an external"),
    (CreativeCodingDomain.PURE_DATA, "Treat Pure Data as an external"),
)


class DomainMetadataRegistryTests(unittest.TestCase):
    def test_registry_covers_current_enum_without_reordering(self) -> None:
        self.assertEqual(
            tuple(info.value for info in SUPPORTED_DOMAINS),
            tuple(CreativeCodingDomain),
        )
        self.assertEqual(
            get_supported_domain_values(),
            tuple(domain.value for domain in CreativeCodingDomain),
        )

    def test_labels_slugs_and_memory_topics_are_stable(self) -> None:
        for domain, label, slug, default_topic in _EXPECTED_DOMAIN_METADATA:
            with self.subTest(domain=domain):
                self.assertEqual(get_domain_label(domain), label)
                self.assertEqual(get_domain_memory_label(domain), label)
                self.assertEqual(get_domain_slug(domain), slug)
                self.assertEqual(get_domain_info(domain).official_source_key, slug)
                self.assertEqual(get_domain_default_topic(domain), default_topic)

        self.assertIsNone(get_domain_memory_label(None))
        self.assertEqual(get_domain_default_topic(None), "code")

    def test_categories_are_present_for_every_domain(self) -> None:
        for info in SUPPORTED_DOMAINS:
            with self.subTest(domain=info.value):
                self.assertIsInstance(info.category, DomainCategory)
                self.assertEqual(get_domain_category(info.value), info.category)

    def test_category_labels_and_groups_are_registry_backed(self) -> None:
        self.assertEqual(
            get_domain_categories(),
            (
                DomainCategory.WEB_CREATIVE_CODING,
                DomainCategory.SHADERS_GPU,
                DomainCategory.ANIMATION,
                DomainCategory.AUDIO_LIVE_CODING,
                DomainCategory.PHYSICS,
                DomainCategory.AV_VJ,
                DomainCategory.VISUAL_PATCHING,
                DomainCategory.DCC_PROCEDURAL,
                DomainCategory.GAME_ENGINES,
                DomainCategory.CREATIVE_AI,
                DomainCategory.MODULAR_SYNTHESIS,
                DomainCategory.PROJECTION_MAPPING,
            ),
        )
        self.assertEqual(
            get_domain_category_label(DomainCategory.WEB_CREATIVE_CODING),
            "Web Creative Coding",
        )
        self.assertEqual(
            get_domain_category_label(DomainCategory.SHADERS_GPU),
            "Shader / GPU",
        )
        self.assertEqual(
            get_domains_for_category(DomainCategory.SHADERS_GPU),
            (
                CreativeCodingDomain.GLSL,
                CreativeCodingDomain.WEBGPU_WGSL,
                CreativeCodingDomain.SHADERTOY,
            ),
        )
        grouped_domains = tuple(
            domain
            for category in get_domain_categories()
            for domain in get_domains_for_category(category)
        )
        self.assertEqual(len(grouped_domains), len(CreativeCodingDomain))
        self.assertEqual(set(grouped_domains), set(CreativeCodingDomain))

    def test_prompt_guidance_is_registered_for_every_domain(self) -> None:
        for domain, expected_snippet in _EXPECTED_PROMPT_GUIDANCE_SNIPPETS:
            with self.subTest(domain=domain):
                guidance = get_domain_prompt_guidance(domain)
                self.assertIn(expected_snippet, guidance)
                self.assertGreater(len(guidance), len(expected_snippet))

    def test_source_registry_still_covers_metadata_domains(self) -> None:
        self.assertEqual(
            official_source_domains(),
            tuple(info.value for info in SUPPORTED_DOMAINS),
        )


if __name__ == "__main__":
    unittest.main()
