import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration.artifacts import (
    extract_workflow_artifacts,
    prepare_workflow_preview_results,
)
from creative_coding_assistant.orchestration.creative_translation import (
    derive_creative_translation,
)
from creative_coding_assistant.orchestration.prompt_inputs import (
    StructuredPromptInputBuilder,
    build_prompt_input_request,
)
from creative_coding_assistant.orchestration.prompt_templates import (
    JinjaPromptRenderer,
    build_rendered_prompt_request,
)
from creative_coding_assistant.orchestration.routing import route_request


class DomainGenerationTests(unittest.TestCase):
    def test_route_infers_previewable_domain_from_visual_3d_prompt(self) -> None:
        decision = route_request(
            AssistantRequest(
                query="Create a rotating 3D cube scene with an orbiting camera.",
                mode=AssistantMode.GENERATE,
            )
        )

        self.assertEqual(decision.domain, CreativeCodingDomain.THREE_JS)
        self.assertEqual(decision.domains, (CreativeCodingDomain.THREE_JS,))

    def test_route_expands_generic_multi_candidate_visual_prompt(self) -> None:
        decision = route_request(
            AssistantRequest(
                query="Create multiple visual candidates for an animated particle field.",
                mode=AssistantMode.GENERATE,
            )
        )

        self.assertEqual(
            decision.domains,
            (
                CreativeCodingDomain.P5_JS,
                CreativeCodingDomain.GLSL,
                CreativeCodingDomain.THREE_JS,
            ),
        )
        self.assertIsNone(decision.domain)

    def test_selected_unsupported_domain_stays_code_only(self) -> None:
        request = AssistantRequest(
            query="Create a Processing sketch with soft orbital trails.",
            domains=(CreativeCodingDomain.PROCESSING,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)

        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```js",
                    "function setup() { createCanvas(640, 360); }",
                    "function draw() { background(12); ellipse(80, 80, 24); }",
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
        )
        preview_results = prepare_workflow_preview_results(
            artifacts,
            request=request,
            route_decision=decision,
        )

        self.assertEqual(decision.domains, (CreativeCodingDomain.PROCESSING,))
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].domain, CreativeCodingDomain.PROCESSING.value)
        self.assertFalse(artifacts[0].preview_eligible)
        self.assertIsNone(artifacts[0].runtime)
        self.assertIsNone(artifacts[0].renderer_id)
        self.assertEqual(preview_results, ())
        self.assertIn("code-only", artifacts[0].summary)

    def test_multi_domain_artifacts_keep_distinct_runtime_metadata(self) -> None:
        request = AssistantRequest(
            query="Create multiple visual candidates for an animated field.",
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)

        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```js p5-field.p5.js",
                    "function setup() { createCanvas(640, 360); }",
                    "function draw() { background(0); circle(120, 120, 24); }",
                    "```",
                    "```glsl shader-field.frag",
                    "void main() { gl_FragColor = vec4(1.0); }",
                    "```",
                    "```ts three-field.three.ts",
                    "const scene = new THREE.Scene();",
                    "const camera = new THREE.PerspectiveCamera();",
                    "const renderer = new THREE.WebGLRenderer();",
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
        )

        self.assertEqual([artifact.domain for artifact in artifacts], [
            CreativeCodingDomain.P5_JS.value,
            CreativeCodingDomain.GLSL.value,
            CreativeCodingDomain.THREE_JS.value,
        ])
        self.assertEqual([artifact.runtime for artifact in artifacts], ["p5", "glsl", "three"])
        self.assertTrue(all(artifact.preview_eligible for artifact in artifacts))

    def test_artifacts_preserve_creative_translation_metadata(self) -> None:
        request = AssistantRequest(
            query="Create a meditative spiral with drifting cyan particles.",
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)
        translation = derive_creative_translation(
            request.query,
            domains=decision.domains,
        )

        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```js spiral-field.p5.js",
                    "function setup() { createCanvas(640, 360); }",
                    "function draw() { background(0); circle(120, 120, 24); }",
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
            creative_translation=translation,
        )

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].creative_translation, translation)
        self.assertEqual(
            artifacts[0].creative_translation.geometric_references,
            ("spiral",),
        )
        self.assertEqual(
            artifacts[0].creative_translation.movement_language,
            ("drift",),
        )
        preview_results = prepare_workflow_preview_results(
            artifacts,
            request=request,
            route_decision=decision,
        )
        self.assertEqual(
            preview_results[0].details["artifact"]["creative_translation"][
                "output_modality"
            ],
            "visual",
        )

    def test_prompt_renderer_adds_runtime_support_guidance(self) -> None:
        request = AssistantRequest(
            query="Create a React Three Fiber installation study.",
            domains=(CreativeCodingDomain.REACT_THREE_FIBER,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)
        prompt_input = StructuredPromptInputBuilder().build(
            build_prompt_input_request(
                assistant_request=request,
                route_decision=decision,
                assembled_context=None,
            )
        )
        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=decision,
                prompt_input=prompt_input,
            )
        )

        system_section = rendered.sections[0].content
        self.assertIn("Generation Runtime Guidance:", system_section)
        self.assertIn("react_three_fiber has current live preview support", system_section)
        self.assertIn("Prefer a .r3f.tsx artifact name", system_section)
