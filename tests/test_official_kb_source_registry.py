import unittest

from pydantic import ValidationError

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.rag import (
    OFFICIAL_HOSTS_BY_DOMAIN,
    OfficialSource,
    OfficialSourceType,
    SourceApprovalStatus,
    approved_official_sources,
    approved_sources_for_domain,
    get_official_source,
    official_source_domains,
)


class OfficialKnowledgeBaseSourceRegistryTests(unittest.TestCase):
    def test_registry_covers_all_supported_domains(self) -> None:
        self.assertEqual(
            official_source_domains(),
            (
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.P5_JS,
                CreativeCodingDomain.GLSL,
                CreativeCodingDomain.PROCESSING,
                CreativeCodingDomain.CANVAS_2D,
                CreativeCodingDomain.WEBGPU_WGSL,
                CreativeCodingDomain.GSAP,
                CreativeCodingDomain.TONE_JS,
                CreativeCodingDomain.PIXI_JS,
                CreativeCodingDomain.MATTER_JS,
                CreativeCodingDomain.RAPIER,
                CreativeCodingDomain.HYDRA,
                CreativeCodingDomain.SHADERTOY,
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
                CreativeCodingDomain.WEB_AUDIO_API,
                CreativeCodingDomain.P5_SOUND,
                CreativeCodingDomain.ML5_JS,
                CreativeCodingDomain.TENSORFLOW_JS,
                CreativeCodingDomain.COMFYUI,
                CreativeCodingDomain.STABLE_DIFFUSION_WORKFLOWS,
                CreativeCodingDomain.RUNWAY,
                CreativeCodingDomain.BLENDER_PYTHON_API,
                CreativeCodingDomain.UNREAL_BLUEPRINTS,
            ),
        )

    def test_sources_are_explicitly_approved(self) -> None:
        sources = approved_official_sources()

        self.assertGreater(len(sources), 0)
        for source in sources:
            self.assertEqual(source.approval_status, SourceApprovalStatus.APPROVED)
            self.assertEqual(source.url.startswith("https://"), True)
            self.assertGreaterEqual(source.priority, 1)
            self.assertGreater(len(source.allowed_path_prefixes), 0)

    def test_source_ids_are_unique(self) -> None:
        source_ids = [source.source_id for source in approved_official_sources()]

        self.assertEqual(len(source_ids), len(set(source_ids)))

    def test_sources_for_domain_are_priority_sorted(self) -> None:
        sources = approved_sources_for_domain(CreativeCodingDomain.THREE_JS)

        self.assertEqual(
            [source.source_id for source in sources],
            ["three_docs", "three_manual", "three_examples"],
        )

    def test_source_lookup_returns_registered_source(self) -> None:
        source = get_official_source("p5_reference")

        self.assertEqual(source.domain, CreativeCodingDomain.P5_JS)
        self.assertEqual(source.source_type, OfficialSourceType.API_REFERENCE)

    def test_three_manual_includes_practical_scene_guides(self) -> None:
        source = get_official_source("three_manual")

        self.assertEqual(source.domain, CreativeCodingDomain.THREE_JS)
        self.assertEqual(source.source_type, OfficialSourceType.GUIDE)
        self.assertEqual(
            source.additional_urls,
            (
                "https://threejs.org/manual/en/fundamentals.html",
                "https://threejs.org/manual/en/responsive.html",
                "https://threejs.org/manual/en/cameras.html",
                "https://threejs.org/manual/en/lights.html",
            ),
        )

    def test_glsl_mdn_examples_source_is_registered_with_expected_metadata(
        self,
    ) -> None:
        source = get_official_source("glsl_mdn_webgl_examples")

        self.assertEqual(source.domain, CreativeCodingDomain.GLSL)
        self.assertEqual(source.publisher, "MDN")
        self.assertEqual(source.source_type, OfficialSourceType.EXAMPLES)
        self.assertEqual(
            source.url,
            (
                "https://developer.mozilla.org/en-US/docs/Web/API/"
                "WebGL_API/By_example/Hello_GLSL"
            ),
        )
        self.assertEqual(
            source.additional_urls,
            (
                "https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/By_example/Textures_from_code",
                "https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial/Using_shaders_to_apply_color_in_WebGL",
            ),
        )

    def test_source_lookup_rejects_unknown_source(self) -> None:
        with self.assertRaises(ValueError):
            get_official_source("unknown_source")

    def test_registry_restricts_hosts_by_domain(self) -> None:
        for source in approved_official_sources():
            hosts = OFFICIAL_HOSTS_BY_DOMAIN[source.domain]

            self.assertIn(source.url.split("/")[2], hosts)

    def test_source_model_rejects_unapproved_host(self) -> None:
        with self.assertRaises(ValidationError):
            OfficialSource(
                source_id="bad_three_source",
                domain=CreativeCodingDomain.THREE_JS,
                title="Bad Three Source",
                publisher="Not Official",
                url="https://example.com/docs/",
                source_type=OfficialSourceType.API_REFERENCE,
                priority=1,
                allowed_path_prefixes=("/docs/",),
            )

    def test_source_model_rejects_out_of_scope_path(self) -> None:
        with self.assertRaises(ValidationError):
            OfficialSource(
                source_id="bad_three_path",
                domain=CreativeCodingDomain.THREE_JS,
                title="Bad Three Path",
                publisher="three.js",
                url="https://threejs.org/examples/",
                source_type=OfficialSourceType.EXAMPLES,
                priority=1,
                allowed_path_prefixes=("/docs/",),
            )

    def test_glsl_sources_use_approved_registry_and_mdn_hosts(self) -> None:
        sources = approved_sources_for_domain(CreativeCodingDomain.GLSL)

        self.assertEqual(
            {source.url.split("/")[2] for source in sources},
            {"developer.mozilla.org", "registry.khronos.org"},
        )
        self.assertTrue(
            all(
                extra_url.split("/")[2]
                in {"developer.mozilla.org", "registry.khronos.org"}
                for source in sources
                for extra_url in source.additional_urls
            )
        )

    def test_first_v2_domain_sources_are_registered_with_expected_metadata(
        self,
    ) -> None:
        processing = get_official_source("processing_reference")
        canvas = get_official_source("canvas2d_context_api")
        webgpu = get_official_source("webgpu_mdn_api")
        wgsl = get_official_source("wgsl_spec")

        self.assertEqual(processing.domain, CreativeCodingDomain.PROCESSING)
        self.assertEqual(processing.url, "https://processing.org/reference/")
        self.assertEqual(processing.source_type, OfficialSourceType.API_REFERENCE)

        self.assertEqual(canvas.domain, CreativeCodingDomain.CANVAS_2D)
        self.assertEqual(canvas.publisher, "MDN")
        self.assertEqual(
            canvas.url,
            "https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D",
        )

        self.assertEqual(webgpu.domain, CreativeCodingDomain.WEBGPU_WGSL)
        self.assertEqual(webgpu.publisher, "MDN")
        self.assertEqual(
            webgpu.url,
            "https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API",
        )

        self.assertEqual(wgsl.domain, CreativeCodingDomain.WEBGPU_WGSL)
        self.assertEqual(wgsl.publisher, "W3C")
        self.assertEqual(wgsl.source_type, OfficialSourceType.SPECIFICATION)
        self.assertEqual(wgsl.url, "https://www.w3.org/TR/WGSL/")

    def test_first_v2_domain_sources_use_approved_hosts(self) -> None:
        expected_hosts = {
            CreativeCodingDomain.PROCESSING: {"processing.org"},
            CreativeCodingDomain.CANVAS_2D: {"developer.mozilla.org"},
            CreativeCodingDomain.WEBGPU_WGSL: {
                "developer.mozilla.org",
                "www.w3.org",
            },
        }

        for domain, hosts in expected_hosts.items():
            with self.subTest(domain=domain):
                sources = approved_sources_for_domain(domain)
                self.assertGreater(len(sources), 0)
                self.assertEqual(
                    {source.url.split("/")[2] for source in sources},
                    hosts,
                )

    def test_second_v2_domain_sources_are_registered_with_expected_metadata(
        self,
    ) -> None:
        cases = (
            (
                "gsap_docs",
                CreativeCodingDomain.GSAP,
                "GSAP",
                "https://gsap.com/docs/v3/",
                OfficialSourceType.API_REFERENCE,
            ),
            (
                "tone_js_docs",
                CreativeCodingDomain.TONE_JS,
                "Tone.js",
                "https://tonejs.github.io/docs/15.1.22/index.html",
                OfficialSourceType.API_REFERENCE,
            ),
            (
                "pixi_js_guides",
                CreativeCodingDomain.PIXI_JS,
                "PixiJS",
                "https://pixijs.com/8.x/guides/getting-started/intro",
                OfficialSourceType.GUIDE,
            ),
            (
                "matter_js_docs",
                CreativeCodingDomain.MATTER_JS,
                "Matter.js",
                "https://www.brm.io/matter-js/docs/",
                OfficialSourceType.API_REFERENCE,
            ),
            (
                "rapier_js_getting_started",
                CreativeCodingDomain.RAPIER,
                "Dimforge",
                "https://rapier.rs/docs/user_guides/templates/getting_started_js/",
                OfficialSourceType.GUIDE,
            ),
            (
                "hydra_docs",
                CreativeCodingDomain.HYDRA,
                "Hydra",
                "https://hydra.ojack.xyz/docs",
                OfficialSourceType.GUIDE,
            ),
            (
                "shadertoy_howto",
                CreativeCodingDomain.SHADERTOY,
                "Shadertoy",
                "https://www.shadertoy.com/howto",
                OfficialSourceType.GUIDE,
            ),
        )

        for source_id, domain, publisher, url, source_type in cases:
            with self.subTest(source_id=source_id):
                source = get_official_source(source_id)
                self.assertEqual(source.domain, domain)
                self.assertEqual(source.publisher, publisher)
                self.assertEqual(source.url, url)
                self.assertEqual(source.source_type, source_type)

    def test_second_v2_domain_sources_use_approved_hosts(self) -> None:
        expected_hosts = {
            CreativeCodingDomain.GSAP: {"gsap.com"},
            CreativeCodingDomain.TONE_JS: {"tonejs.github.io"},
            CreativeCodingDomain.PIXI_JS: {"pixijs.com"},
            CreativeCodingDomain.MATTER_JS: {"www.brm.io"},
            CreativeCodingDomain.RAPIER: {"rapier.rs"},
            CreativeCodingDomain.HYDRA: {"hydra.ojack.xyz"},
            CreativeCodingDomain.SHADERTOY: {"www.shadertoy.com"},
        }

        for domain, hosts in expected_hosts.items():
            with self.subTest(domain=domain):
                sources = approved_sources_for_domain(domain)
                self.assertGreater(len(sources), 0)
                self.assertEqual(
                    {source.url.split("/")[2] for source in sources},
                    hosts,
                )

    def test_third_v2_domain_sources_are_registered_with_expected_metadata(
        self,
    ) -> None:
        cases = (
            (
                "touchdesigner_user_guide",
                CreativeCodingDomain.TOUCHDESIGNER,
                "Derivative",
                "https://derivative.ca/UserGuide/Getting_started",
            ),
            (
                "houdini_docs",
                CreativeCodingDomain.HOUDINI,
                "SideFX",
                "https://www.sidefx.com/docs/houdini/",
            ),
            (
                "blender_geometry_nodes_manual",
                CreativeCodingDomain.BLENDER_GEOMETRY_NODES,
                "Blender Foundation",
                "https://docs.blender.org/manual/en/latest/modeling/geometry_nodes/index.html",
            ),
            (
                "unity_manual",
                CreativeCodingDomain.UNITY,
                "Unity",
                "https://docs.unity3d.com/Manual/index.html",
            ),
            (
                "unreal_engine_docs",
                CreativeCodingDomain.UNREAL,
                "Epic Games",
                "https://dev.epicgames.com/documentation/en-us/unreal-engine/get-started",
            ),
            (
                "max_msp_docs",
                CreativeCodingDomain.MAX_MSP,
                "Cycling '74",
                "https://docs.cycling74.com/",
            ),
            (
                "notch_manual",
                CreativeCodingDomain.NOTCH,
                "Notch",
                "https://manual.notch.one/1.0/en/docs/",
            ),
            (
                "vvvv_gamma_docs",
                CreativeCodingDomain.VVVV,
                "vvvv",
                "https://thegraybook.vvvv.org/",
            ),
        )

        for source_id, domain, publisher, url in cases:
            with self.subTest(source_id=source_id):
                source = get_official_source(source_id)
                self.assertEqual(source.domain, domain)
                self.assertEqual(source.publisher, publisher)
                self.assertEqual(source.url, url)
                self.assertEqual(source.source_type, OfficialSourceType.GUIDE)

    def test_third_v2_domain_sources_use_approved_hosts(self) -> None:
        expected_hosts = {
            CreativeCodingDomain.TOUCHDESIGNER: {"derivative.ca"},
            CreativeCodingDomain.HOUDINI: {"www.sidefx.com"},
            CreativeCodingDomain.BLENDER_GEOMETRY_NODES: {"docs.blender.org"},
            CreativeCodingDomain.UNITY: {"docs.unity3d.com"},
            CreativeCodingDomain.UNREAL: {"dev.epicgames.com"},
            CreativeCodingDomain.MAX_MSP: {"docs.cycling74.com"},
            CreativeCodingDomain.NOTCH: {"manual.notch.one"},
            CreativeCodingDomain.VVVV: {"thegraybook.vvvv.org"},
        }

        for domain, hosts in expected_hosts.items():
            with self.subTest(domain=domain):
                sources = approved_sources_for_domain(domain)
                self.assertGreater(len(sources), 0)
                self.assertEqual(
                    {source.url.split("/")[2] for source in sources},
                    hosts,
                )

    def test_fourth_v2_domain_sources_are_registered_with_expected_metadata(
        self,
    ) -> None:
        cases = (
            (
                "openframeworks_docs",
                CreativeCodingDomain.OPENFRAMEWORKS,
                "openFrameworks",
                "https://openframeworks.cc/documentation/",
            ),
            (
                "openrndr_guide",
                CreativeCodingDomain.OPENRNDR,
                "OPENRNDR",
                "https://guide.openrndr.org/",
            ),
            (
                "supercollider_help",
                CreativeCodingDomain.SUPERCOLLIDER,
                "SuperCollider",
                "https://docs.supercollider.online/",
            ),
            (
                "sonic_pi_tutorial",
                CreativeCodingDomain.SONIC_PI,
                "Sonic Pi",
                "https://sonic-pi.net/tutorial.html",
            ),
            (
                "tidalcycles_docs",
                CreativeCodingDomain.TIDALCYCLES,
                "TidalCycles",
                "https://tidalcycles.org/docs/",
            ),
            (
                "web_audio_mdn_api",
                CreativeCodingDomain.WEB_AUDIO_API,
                "MDN",
                "https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API",
            ),
            (
                "p5_sound_reference",
                CreativeCodingDomain.P5_SOUND,
                "p5.js",
                "https://p5js.org/reference/p5.sound/",
            ),
            (
                "ml5_js_learn",
                CreativeCodingDomain.ML5_JS,
                "ml5.js",
                "https://ml5js.org/learn",
            ),
            (
                "tensorflow_js_guide",
                CreativeCodingDomain.TENSORFLOW_JS,
                "TensorFlow",
                "https://www.tensorflow.org/js/guide",
            ),
            (
                "comfyui_docs",
                CreativeCodingDomain.COMFYUI,
                "ComfyUI",
                "https://docs.comfy.org/index",
            ),
            (
                "stable_diffusion_diffusers",
                CreativeCodingDomain.STABLE_DIFFUSION_WORKFLOWS,
                "Hugging Face Diffusers",
                (
                    "https://huggingface.co/docs/diffusers/api/pipelines/"
                    "stable_diffusion/overview"
                ),
            ),
            (
                "runway_api_docs",
                CreativeCodingDomain.RUNWAY,
                "Runway",
                "https://docs.dev.runwayml.com/",
            ),
            (
                "blender_python_api",
                CreativeCodingDomain.BLENDER_PYTHON_API,
                "Blender Foundation",
                "https://docs.blender.org/api/current/",
            ),
            (
                "unreal_blueprints_docs",
                CreativeCodingDomain.UNREAL_BLUEPRINTS,
                "Epic Games",
                (
                    "https://dev.epicgames.com/documentation/en-us/"
                    "unreal-engine/introduction-to-blueprints"
                ),
            ),
        )

        for source_id, domain, publisher, url in cases:
            with self.subTest(source_id=source_id):
                source = get_official_source(source_id)
                self.assertEqual(source.domain, domain)
                self.assertEqual(source.publisher, publisher)
                self.assertEqual(source.url, url)
                self.assertGreaterEqual(source.priority, 1)

    def test_fourth_v2_domain_sources_use_approved_hosts(self) -> None:
        expected_hosts = {
            CreativeCodingDomain.OPENFRAMEWORKS: {"openframeworks.cc"},
            CreativeCodingDomain.OPENRNDR: {"guide.openrndr.org"},
            CreativeCodingDomain.SUPERCOLLIDER: {"docs.supercollider.online"},
            CreativeCodingDomain.SONIC_PI: {"sonic-pi.net"},
            CreativeCodingDomain.TIDALCYCLES: {"tidalcycles.org"},
            CreativeCodingDomain.WEB_AUDIO_API: {"developer.mozilla.org"},
            CreativeCodingDomain.P5_SOUND: {"p5js.org"},
            CreativeCodingDomain.ML5_JS: {"ml5js.org"},
            CreativeCodingDomain.TENSORFLOW_JS: {
                "www.tensorflow.org",
                "js.tensorflow.org",
            },
            CreativeCodingDomain.COMFYUI: {"docs.comfy.org"},
            CreativeCodingDomain.STABLE_DIFFUSION_WORKFLOWS: {"huggingface.co"},
            CreativeCodingDomain.RUNWAY: {"docs.dev.runwayml.com"},
            CreativeCodingDomain.BLENDER_PYTHON_API: {"docs.blender.org"},
            CreativeCodingDomain.UNREAL_BLUEPRINTS: {"dev.epicgames.com"},
        }

        for domain, hosts in expected_hosts.items():
            with self.subTest(domain=domain):
                sources = approved_sources_for_domain(domain)
                self.assertGreater(len(sources), 0)
                self.assertEqual(
                    {
                        url.split("/")[2]
                        for source in sources
                        for url in (source.url, *source.additional_urls)
                    },
                    hosts,
                )


if __name__ == "__main__":
    unittest.main()
