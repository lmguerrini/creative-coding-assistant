import unittest
from dataclasses import dataclass

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
from creative_coding_assistant.orchestration.domain_generation import (
    resolve_generation_domains,
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
    def test_generation_domain_resolution_preserves_explicit_bridge_scope(self) -> None:
        selected_domains = (
            CreativeCodingDomain.TONE_JS,
            CreativeCodingDomain.P5_JS,
            CreativeCodingDomain.THREE_JS,
        )

        resolved_domains = resolve_generation_domains(
            query=(
                "Explain how Tone.js coordinates timing across the browser "
                "visual runtime."
            ),
            selected_domains=selected_domains,
        )

        self.assertEqual(resolved_domains, selected_domains)

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
                query=(
                    "Create multiple visual candidates for an animated particle field."
                ),
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

    def test_react_three_fiber_artifact_stays_code_only_without_a_bundle_runtime(self) -> None:
        request = AssistantRequest(
            query="Create a React Three Fiber installation study.",
            domains=(CreativeCodingDomain.REACT_THREE_FIBER,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)

        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```tsx",
                    'import { Canvas, useFrame } from "@react-three/fiber";',
                    "function Orb() { useFrame(() => {}); return <mesh />; }",
                    "export default function Study() { return <Canvas><Orb /></Canvas>; }",
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

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].domain, CreativeCodingDomain.REACT_THREE_FIBER.value)
        self.assertEqual(artifacts[0].title, "generated-study-1.r3f.tsx")
        self.assertEqual(artifacts[0].language, "TypeScript + React Three Fiber")
        self.assertFalse(artifacts[0].preview_eligible)
        self.assertIsNone(artifacts[0].runtime)
        self.assertIsNone(artifacts[0].renderer_id)
        self.assertEqual(artifacts[0].status, "Generated")
        self.assertEqual(preview_results, ())
        self.assertIn("code-only", artifacts[0].summary)

    def test_unsupported_glsl_source_stays_code_only_before_preview_preparation(self) -> None:
        request = AssistantRequest(
            query="Create a GLSL fragment shader with a texture sampler.",
            domains=(CreativeCodingDomain.GLSL,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)
        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```glsl sampled-field.frag",
                    "uniform sampler2D sourceTexture;",
                    "void main() {",
                    "  gl_FragColor = texture2D(sourceTexture, vec2(0.5));",
                    "}",
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
        )

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].domain, CreativeCodingDomain.GLSL.value)
        self.assertFalse(artifacts[0].preview_eligible)
        self.assertIsNone(artifacts[0].runtime)
        self.assertIsNone(artifacts[0].renderer_id)
        self.assertEqual(artifacts[0].status, "Preview unavailable")
        self.assertEqual(
            prepare_workflow_preview_results(
                artifacts,
                request=request,
                route_decision=decision,
            ),
            (),
        )
        self.assertIn("outside the current bounded runtime subset", artifacts[0].summary)

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

        self.assertEqual(
            [artifact.domain for artifact in artifacts],
            [
                CreativeCodingDomain.P5_JS.value,
                CreativeCodingDomain.GLSL.value,
                CreativeCodingDomain.THREE_JS.value,
            ],
        )
        self.assertEqual(
            [artifact.runtime for artifact in artifacts],
            ["p5", "glsl", "three"],
        )
        self.assertTrue(all(artifact.preview_eligible for artifact in artifacts))

    def test_tone_artifact_is_previewable_only_with_a_bounded_voice_program(self) -> None:
        request = AssistantRequest(
            query="Create a muted Tone.js cymatic study.",
            domains=(CreativeCodingDomain.TONE_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)

        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```javascript filename=cymatic-study.tone.js",
                    "const synth = new Tone.FMSynth().toDestination();",
                    "new Tone.Sequence((time, note) => "
                    "synth.triggerAttackRelease(note, '8n', time), ['C3', 'G3'], "
                    "'8n').start(0);",
                    "Tone.Transport.bpm.value = 96;",
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
        )

        self.assertEqual(len(artifacts), 1)
        artifact = artifacts[0]
        self.assertEqual(artifact.domain, CreativeCodingDomain.TONE_JS.value)
        self.assertEqual(artifact.title, "cymatic-study.tone.js")
        self.assertEqual(artifact.runtime, "tone")
        self.assertEqual(artifact.renderer_id, "surface.tone")
        self.assertTrue(artifact.preview_eligible)
        self.assertEqual(artifact.language, "JavaScript + Tone.js")
        self.assertEqual(
            prepare_workflow_preview_results(
                artifacts,
                request=request,
                route_decision=decision,
            )[0].preview_artifact_id,
            artifact.id,
        )

        microphone_artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```javascript filename=unsafe-input.tone.js",
                    "const synth = new Tone.Synth().toDestination();",
                    "navigator.mediaDevices.getUserMedia({ audio: true });",
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
        )
        self.assertEqual(len(microphone_artifacts), 1)
        self.assertFalse(microphone_artifacts[0].preview_eligible)
        self.assertEqual(microphone_artifacts[0].status, "Preview unavailable")
        self.assertIn("Microphone", microphone_artifacts[0].summary)

    def test_p5_typescript_fence_is_normalized_to_previewable_javascript(self) -> None:
        request = AssistantRequest(
            query=(
                "Create a p5.js flow-field particle system with soft trails and "
                "interaction controls."
            ),
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)

        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```ts",
                    "import p5 from 'p5';",
                    "type Particle = { x: number; y: number };",
                    "const particles: Particle[] = [];",
                    "function setup(): void {",
                    "  createCanvas(640, 360);",
                    "  pixelDensity(1);",
                    "  colorMode(HSL, 360, 100, 100, 1);",
                    "  noiseDetail(3, 0.5);",
                    "}",
                    "function draw(): void {",
                    "  background(8, 12, 18);",
                    "  push(); translate(12, 12); beginShape(); vertex(0, 0); vertex(8, 4); endShape(CLOSE); pop();",
                    "  particles.forEach((p: Particle) => circle(p.x, p.y, 4));",
                    "}",
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

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].title, "generated-sketch-1.p5.js")
        self.assertEqual(artifacts[0].source_language, "javascript")
        self.assertEqual(artifacts[0].language, "JavaScript + p5.js")
        self.assertEqual(artifacts[0].runtime, "p5")
        self.assertEqual(artifacts[0].renderer_id, "surface.p5")
        self.assertTrue(artifacts[0].preview_eligible)
        self.assertNotIn("import p5", artifacts[0].content)
        self.assertNotIn("type Particle", artifacts[0].content)
        self.assertNotIn(": number", artifacts[0].content)
        self.assertIn("colorMode(HSL, 360, 100, 100, 1)", artifacts[0].content)
        self.assertIn("endShape(CLOSE)", artifacts[0].content)
        self.assertEqual(preview_results[0].preview_artifact_id, artifacts[0].id)

    def test_p5_unsupported_runtime_api_is_not_marked_preview_ready(self) -> None:
        request = AssistantRequest(
            query="Create a p5.js flow-field particle system.",
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)

        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```js generated-sketch-1.p5.js",
                    "function setup() { createCanvas(640, 360); createGraphics(640, 360); }",
                    "function draw() { background(0); circle(20, 20, 10); }",
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
        )

        self.assertEqual(len(artifacts), 1)
        self.assertFalse(artifacts[0].preview_eligible)
        self.assertEqual(artifacts[0].status, "Runnable artifact extraction failed")
        self.assertIn("Runnable artifact extraction failed", artifacts[0].summary)

    def test_p5_constrain_is_marked_preview_ready(self) -> None:
        request = AssistantRequest(
            query="Create a p5.js morphogenesis sketch.",
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)

        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```js generated-sketch-1.p5.js",
                    "const retainParticle = function (particle) { return particle.life > 0; };",
                    "function setup() { createCanvas(640, 360); }",
                    "function draw() {",
                    "  const x = constrain(int(noise(frameCount * 0.01) * width), 0, width);",
                    "  const particles = [{ life: 1 }].filter(retainParticle);",
                    "  strokeCap(ROUND);",
                    "  background(8); circle(x, height / 2, 18);",
                    "}",
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
        )

        self.assertEqual(len(artifacts), 1)
        self.assertTrue(artifacts[0].preview_eligible)
        self.assertEqual(artifacts[0].runtime, "p5")
        self.assertEqual(artifacts[0].renderer_id, "surface.p5")
        self.assertEqual(artifacts[0].status, "Generated")

    def test_p5_parenthesized_return_expression_is_not_treated_as_a_call(self) -> None:
        request = AssistantRequest(
            query="Create a p5.js aurora garden.",
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)

        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```javascript filename=recursive-aurora-garden.p5.js",
                    "function setup() { createCanvas(640, 360); }",
                    "function boundedMax(a, b) { return (a > b) ? a : b; }",
                    "function draw() {",
                    "  background(8);",
                    "  circle(width / 2, height / 2, boundedMax(24, 32));",
                    "}",
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
        )

        self.assertEqual(len(artifacts), 1)
        self.assertTrue(artifacts[0].preview_eligible)
        self.assertEqual(artifacts[0].runtime, "p5")
        self.assertEqual(artifacts[0].renderer_id, "surface.p5")
        self.assertEqual(artifacts[0].status, "Generated")

    def test_p5_standard_degrees_helper_is_marked_preview_ready(self) -> None:
        request = AssistantRequest(
            query="Create a p5.js flow-field sketch with a colour gradient.",
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)

        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```js generated-sketch-1.p5.js",
                    "function setup() { createCanvas(640, 360); colorMode(HSL, 360, 100, 100); }",
                    "function draw() {",
                    "  const hue = degrees(noise(frameCount * 0.01) * TWO_PI) % 360;",
                    "  background(hue, 48, 12); circle(width / 2, height / 2, 40);",
                    "}",
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
        )

        self.assertEqual(len(artifacts), 1)
        self.assertTrue(artifacts[0].preview_eligible)
        self.assertEqual(artifacts[0].runtime, "p5")
        self.assertEqual(artifacts[0].status, "Generated")

    def test_p5_fenced_javascript_labels_produce_previewable_javascript_artifacts(self) -> None:
        request = AssistantRequest(
            query="Create a p5.js flow-field particle system.",
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)
        source = "\n".join(
            [
                "function setup() { createCanvas(640, 360); rectMode(CENTER); smooth(); }",
                "function draw() { const swatch = color(20, 40, 60); background(12); "
                "fill(red(swatch), green(swatch), blue(swatch)); "
                "rect(width / 2, height / 2, 32, 18); }",
            ]
        )

        for label in ("javascript", "js", "p5", "p5.js"):
            with self.subTest(label=label):
                artifacts = extract_workflow_artifacts(
                    f"Intro prose.\n```{label}\n{source}\n```\nClosing prose.",
                    request=request,
                    route_decision=decision,
                )
                preview_results = prepare_workflow_preview_results(
                    artifacts,
                    request=request,
                    route_decision=decision,
                )

                self.assertEqual(len(artifacts), 1)
                self.assertTrue(artifacts[0].title.endswith(".p5.js"))
                self.assertEqual(artifacts[0].type, "code")
                self.assertEqual(artifacts[0].source_language, "javascript")
                self.assertTrue(artifacts[0].preview_eligible)
                self.assertEqual(artifacts[0].runtime, "p5")
                self.assertEqual(artifacts[0].renderer_id, "surface.p5")
                self.assertNotIn("```", artifacts[0].content)
                self.assertEqual(artifacts[0].content, source)
                self.assertNotIn("Intro prose", artifacts[0].content)
                self.assertNotIn("Closing prose", artifacts[0].content)
                self.assertEqual(len(preview_results), 1)

    def test_p5_fence_filename_metadata_is_preserved_without_the_prefix(self) -> None:
        request = AssistantRequest(
            query="Create a p5.js flow-field particle system.",
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)
        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```javascript filename=named-field.p5.js",
                    "function setup() { createCanvas(640, 360); }",
                    "function draw() { background(12); circle(80, 80, 24); }",
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
        )

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].title, "named-field.p5.js")
        self.assertTrue(artifacts[0].preview_eligible)

    def test_unclosed_provider_code_fence_stays_inspectable_and_fails_preview_truthfully(self) -> None:
        request = AssistantRequest(
            query="Create a p5.js flow-field particle system.",
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)
        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "The attached image uses gold, olive, and a radial rhythm.",
                    "```javascript filename=interrupted-field.p5.js",
                    "function setup() { createCanvas(640, 360); }",
                    "function draw() { background(12); circle(80, 80, 24); }",
                ]
            ),
            request=request,
            route_decision=decision,
        )

        self.assertEqual(len(artifacts), 1)
        artifact = artifacts[0]
        self.assertEqual(artifact.title, "interrupted-field.p5.js")
        self.assertEqual(artifact.status, "Runnable artifact extraction failed")
        self.assertFalse(artifact.preview_eligible)
        self.assertIsNone(artifact.runtime)
        self.assertIn("code fence was not closed", artifact.summary)
        self.assertNotIn("attached image", artifact.content)
        self.assertNotIn("```", artifact.content)
        self.assertEqual(
            artifact.content,
            "function setup() { createCanvas(640, 360); }\n"
            "function draw() { background(12); circle(80, 80, 24); }",
        )

    def test_p5_closed_fence_cannot_hide_a_trailing_unclosed_attempt(self) -> None:
        request = AssistantRequest(
            query="Create one p5.js artwork.",
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)
        closed_source = (
            "function setup() { createCanvas(640, 360); }\n"
            "function draw() { background(12); circle(80, 80, 24); }"
        )
        interrupted_source = (
            "function setup() { createCanvas(640, 360); }\n"
            "function draw() { background(24); rect(60, 60, 48, 32); }"
        )
        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "A complete attempt follows.",
                    "```javascript filename=first-study.p5.js",
                    closed_source,
                    "```",
                    "The provider then started a replacement.",
                    "```p5.js filename=interrupted-study.p5.js",
                    interrupted_source,
                ]
            ),
            request=request,
            route_decision=decision,
        )

        self.assertEqual(len(artifacts), 1)
        artifact = artifacts[0]
        self.assertEqual(artifact.content, closed_source)
        self.assertEqual(artifact.status, "Runnable artifact extraction failed")
        self.assertFalse(artifact.preview_eligible)
        self.assertIsNone(artifact.runtime)
        self.assertIn("code fence was not closed", artifact.summary)
        self.assertNotIn("```", artifact.content)
        self.assertNotIn("complete attempt", artifact.content)
        self.assertEqual(
            prepare_workflow_preview_results(
                artifacts,
                request=request,
                route_decision=decision,
            ),
            (),
        )

    def test_explicit_multi_p5_request_keeps_valid_candidate_when_second_is_unclosed(self) -> None:
        request = AssistantRequest(
            query="Create two p5.js candidate sketches.",
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)
        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```javascript filename=first-study.p5.js",
                    "function setup() { createCanvas(640, 360); }",
                    "function draw() { background(12); circle(80, 80, 24); }",
                    "```",
                    "```p5.js filename=interrupted-study.p5.js",
                    "function setup() { createCanvas(640, 360); }",
                    "function draw() { background(24); rect(60, 60, 48, 32); }",
                ]
            ),
            request=request,
            route_decision=decision,
        )

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].title, "first-study.p5.js")
        self.assertTrue(artifacts[0].preview_eligible)
        self.assertEqual(artifacts[0].runtime, "p5")
        self.assertEqual(
            len(
                prepare_workflow_preview_results(
                    artifacts,
                    request=request,
                    route_decision=decision,
                )
            ),
            1,
        )

    def test_p5_multiple_incompatible_fences_fail_without_selecting_a_preview(self) -> None:
        request = AssistantRequest(
            query="Create one p5.js artwork from the attached image.",
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)
        p5_source = (
            "function setup() { createCanvas(640, 360); }\n"
            "function draw() { background(12); circle(80, 80, 24); }"
        )
        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "Two incompatible implementations follow.",
                    "```javascript filename=image-study.p5.js",
                    p5_source,
                    "```",
                    "```glsl filename=image-study.frag",
                    "void main() { gl_FragColor = vec4(1.0); }",
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
        )

        self.assertEqual(len(artifacts), 1)
        artifact = artifacts[0]
        self.assertEqual(artifact.content, p5_source)
        self.assertEqual(artifact.status, "Runnable artifact extraction failed")
        self.assertFalse(artifact.preview_eligible)
        self.assertIsNone(artifact.runtime)
        self.assertIn("multiple distinct or incompatible code blocks", artifact.summary)
        self.assertEqual(
            prepare_workflow_preview_results(
                artifacts,
                request=request,
                route_decision=decision,
            ),
            (),
        )

    def test_inferred_p5_route_rejects_incompatible_fences(self) -> None:
        request = AssistantRequest(
            query="Create an animated particle field.",
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)
        self.assertEqual(decision.domains, (CreativeCodingDomain.P5_JS,))
        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```javascript filename=particle-field.p5.js",
                    "function setup() { createCanvas(640, 360); }",
                    "function draw() { background(12); circle(80, 80, 24); }",
                    "```",
                    "```glsl filename=particle-field.frag",
                    "void main() { gl_FragColor = vec4(1.0); }",
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
        )

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].status, "Runnable artifact extraction failed")
        self.assertFalse(artifacts[0].preview_eligible)
        self.assertIsNone(artifacts[0].runtime)
        self.assertIn(
            "multiple distinct or incompatible code blocks",
            artifacts[0].summary,
        )

    def test_p5_two_distinct_complete_fences_fail_single_artifact_contract(self) -> None:
        request = AssistantRequest(
            query="Create a p5.js artwork that draws two spirals.",
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)
        first_source = (
            "function setup() { createCanvas(640, 360); }\n"
            "function draw() { background(12); circle(80, 80, 24); }"
        )
        second_source = (
            "function setup() { createCanvas(640, 360); }\n"
            "function draw() { background(24); rect(60, 60, 48, 32); }"
        )
        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```javascript filename=first-study.p5.js",
                    first_source,
                    "```",
                    "```p5.js filename=second-study.p5.js",
                    second_source,
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
        )

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].content, first_source)
        self.assertEqual(
            artifacts[0].status,
            "Runnable artifact extraction failed",
        )
        self.assertFalse(artifacts[0].preview_eligible)
        self.assertIsNone(artifacts[0].runtime)
        self.assertIn(
            "multiple distinct or incompatible code blocks",
            artifacts[0].summary,
        )

    def test_single_p5_artwork_with_two_color_options_stays_single_artifact(self) -> None:
        request = AssistantRequest(
            query="Create one p5.js artwork with two color options.",
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)
        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```javascript filename=gold-study.p5.js",
                    "function setup() { createCanvas(640, 360); }",
                    "function draw() { background(12); circle(80, 80, 24); }",
                    "```",
                    "```p5.js filename=olive-study.p5.js",
                    "function setup() { createCanvas(640, 360); }",
                    "function draw() { background(24); rect(60, 60, 48, 32); }",
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
        )

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].status, "Runnable artifact extraction failed")
        self.assertFalse(artifacts[0].preview_eligible)
        self.assertIsNone(artifacts[0].runtime)
        self.assertIn(
            "multiple distinct or incompatible code blocks",
            artifacts[0].summary,
        )

    def test_p5_explicit_multi_candidate_request_keeps_distinct_previews(self) -> None:
        request = AssistantRequest(
            query="Create two p5.js candidate sketches.",
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)
        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```javascript filename=first-study.p5.js",
                    "function setup() { createCanvas(640, 360); }",
                    "function draw() { background(12); circle(80, 80, 24); }",
                    "```",
                    "```p5.js filename=second-study.p5.js",
                    "function setup() { createCanvas(640, 360); }",
                    "function draw() { background(24); rect(60, 60, 48, 32); }",
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
        )

        self.assertEqual(len(artifacts), 2)
        self.assertEqual(
            [artifact.title for artifact in artifacts],
            ["first-study.p5.js", "second-study.p5.js"],
        )
        self.assertTrue(all(artifact.preview_eligible for artifact in artifacts))
        self.assertTrue(all(artifact.runtime == "p5" for artifact in artifacts))

    def test_p5_extraction_prefers_lifecycle_source_and_marks_invalid_source_honestly(self) -> None:
        request = AssistantRequest(
            query="Create a p5.js flow-field particle system.",
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)
        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```javascript",
                    "console.log('not a p5 sketch');",
                    "```",
                    "```javascript",
                    "function setup() { createCanvas(640, 360); }",
                    "function draw() { background(12); circle(80, 80, 24); }",
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
        )

        self.assertEqual(len(artifacts), 1)
        self.assertTrue(artifacts[0].title.endswith(".p5.js"))
        self.assertTrue(artifacts[0].preview_eligible)
        self.assertNotIn("console.log", artifacts[0].content)

        invalid = extract_workflow_artifacts(
            "```javascript\nconsole.log('not a p5 sketch');\n```",
            request=request,
            route_decision=decision,
        )
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0].status, "Runnable artifact extraction failed")
        self.assertFalse(invalid[0].preview_eligible)
        self.assertIsNone(invalid[0].runtime)
        self.assertIn("Runnable artifact extraction failed", invalid[0].summary)
        self.assertEqual(
            extract_workflow_artifacts(
                "A prose-only response without executable source.",
                request=request,
                route_decision=decision,
            ),
            (),
        )

    def test_p5_html_document_is_not_marked_preview_ready(self) -> None:
        request = AssistantRequest(
            query="Create a p5.js sketch.",
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)

        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```html generated-sketch-1.p5.ts",
                    "<!doctype html>",
                    "<html><body><script>",
                    "function setup() { createCanvas(640, 360); }",
                    "function draw() { circle(20, 20, 10); }",
                    "</script></body></html>",
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

        self.assertEqual(len(artifacts), 1)
        self.assertFalse(artifacts[0].preview_eligible)
        self.assertIsNone(artifacts[0].runtime)
        self.assertIsNone(artifacts[0].renderer_id)
        self.assertEqual(preview_results, ())

    def test_three_html_document_is_not_marked_preview_ready(self) -> None:
        request = AssistantRequest(
            query="Create a Three.js scene.",
            domains=(CreativeCodingDomain.THREE_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)

        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```html exported-scene.three.ts",
                    "<!doctype html>",
                    "<html><body><script>",
                    "const scene = new THREE.Scene();",
                    "</script></body></html>",
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
        )

        self.assertEqual(len(artifacts), 1)
        self.assertFalse(artifacts[0].preview_eligible)
        self.assertIsNone(artifacts[0].runtime)
        self.assertIsNone(artifacts[0].renderer_id)
        self.assertEqual(
            prepare_workflow_preview_results(
                artifacts,
                request=request,
                route_decision=decision,
            ),
            (),
        )

    def test_artifacts_preserve_creative_translation_metadata(self) -> None:
        request = AssistantRequest(
            query=("Create a meditative glowing spiral with drifting cyan particles."),
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        decision = route_request(request)
        translation = derive_creative_translation(
            request.query,
            domains=decision.domains,
            image_references=(
                _ImageReference(
                    id="image-reference-1",
                    name="warm-neon-grid-glass-drift.png",
                    mime_type="image/png",
                    size_bytes=128,
                ),
            ),
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
        self.assertIsNotNone(artifacts[0].creative_translation.sacred_geometry)
        assert artifacts[0].creative_translation.sacred_geometry is not None
        self.assertEqual(
            artifacts[0].creative_translation.sacred_geometry.concepts,
            ("spiral",),
        )
        self.assertIsNotNone(artifacts[0].creative_translation.shader_presets)
        shader_presets = artifacts[0].creative_translation.shader_presets
        assert shader_presets is not None
        self.assertEqual(
            [preset.value for preset in shader_presets.presets],
            ["glow"],
        )
        self.assertIsNotNone(artifacts[0].creative_translation.visual_style)
        visual_style = artifacts[0].creative_translation.visual_style
        assert visual_style is not None
        self.assertEqual(
            [style.value for style in visual_style.styles],
            ["sacred geometry"],
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
        self.assertEqual(
            preview_results[0].details["artifact"]["creative_translation"][
                "sacred_geometry"
            ]["concepts"],
            ["spiral"],
        )
        self.assertEqual(
            preview_results[0].details["artifact"]["creative_translation"][
                "shader_presets"
            ]["presets"],
            ["glow"],
        )
        self.assertEqual(
            preview_results[0].details["artifact"]["creative_translation"][
                "visual_style"
            ]["styles"],
            ["sacred geometry"],
        )
        self.assertEqual(
            preview_results[0].details["artifact"]["creative_translation"][
                "reference_fusion"
            ]["source_count"],
            1,
        )
        self.assertEqual(
            preview_results[0].details["artifact"]["creative_translation"][
                "reference_fusion"
            ]["palette_direction"],
            [],
        )

    def test_artifacts_preserve_audio_reactive_mapping_metadata(self) -> None:
        request = AssistantRequest(
            query=(
                "Create an audio-reactive p5.js field where bass drives pulse "
                "and rhythm controls rotation."
            ),
            domains=(
                CreativeCodingDomain.P5_JS,
                CreativeCodingDomain.TONE_JS,
            ),
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
                    "```js reactive-field.p5.js",
                    "function setup() { createCanvas(640, 360); }",
                    "function draw() { background(0); circle(120, 120, 24); }",
                    "```",
                ]
            ),
            request=request,
            route_decision=decision,
            creative_translation=translation,
        )

        self.assertIsNotNone(artifacts[0].creative_translation)
        assert artifacts[0].creative_translation is not None
        mapping = artifacts[0].creative_translation.audio_reactive
        self.assertIsNotNone(mapping)
        assert mapping is not None
        self.assertEqual(
            tuple(item.source.value for item in mapping.mappings),
            ("bass", "rhythm"),
        )

        preview_results = prepare_workflow_preview_results(
            artifacts,
            request=request,
            route_decision=decision,
        )
        mapping_payload = preview_results[0].details["artifact"][
            "creative_translation"
        ]["audio_reactive"]
        self.assertEqual(mapping_payload["activation"], "explicit_user_gesture")
        self.assertEqual(mapping_payload["mappings"][0]["source"], "bass")

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
        self.assertIn("react_three_fiber is code-only", system_section)
        self.assertIn("do not claim live preview readiness", system_section)
        self.assertIn("Prefer a .r3f.tsx artifact name", system_section)
        self.assertIn("needs its own React bundle runtime", system_section)

    def test_prompt_renderer_keeps_p5_generation_inside_the_preview_contract(self) -> None:
        request = AssistantRequest(
            query="Create a p5.js flow-field particle system.",
            domains=(CreativeCodingDomain.P5_JS,),
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
        self.assertIn("Build translucent trails directly with background alpha", system_section)
        self.assertIn("Do not use createGraphics", system_section)
        self.assertIn("p5.Vector", system_section)

    def test_prompt_renderer_keeps_three_generation_inside_the_preview_contract(self) -> None:
        request = AssistantRequest(
            query="Design a Three.js kinetic sculpture with camera motion.",
            domains=(CreativeCodingDomain.THREE_JS,),
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
        self.assertIn("fully closed fenced javascript artifact", system_section)
        self.assertIn("filename=...three.js", system_section)
        self.assertIn("Do not return HTML", system_section)
        self.assertIn("below 7,500 characters", system_section)


@dataclass(frozen=True)
class _ImageReference:
    id: str
    name: str
    mime_type: str
    size_bytes: int
