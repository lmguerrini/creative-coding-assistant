import { describe, expect, it } from "vitest";
import { getLocalWorkspaceSnapshot } from "./assistant-client";
import { buildArtifactDocument } from "./artifact-inspector";
import {
  hydrateWorkspaceFromArtifactExtractedEvent,
  hydrateWorkspaceFromFinalEvent
} from "./live-artifact-hydration";
import { buildPreviewRendererRoute } from "./preview-renderers";
import type { AssistantStreamEvent } from "./assistant-stream";

describe("live artifact hydration", () => {
  it("hydrates final stream code into a previewable Three.js artifact", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromFinalEvent(
      snapshot,
      finalEvent({
        answer: [
          "Here is the scene:",
          "```ts",
          "import * as THREE from 'three';",
          "const scene = new THREE.Scene();",
          "const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 100);",
          "const renderer = new THREE.WebGLRenderer({ antialias: true });",
          "scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial()));",
          "```"
        ].join("\n")
      })
    );

    expect(result.artifact).toMatchObject({
      id: "live-generated-artifact",
      title: "generated-scene.three.ts",
      type: "code",
      language: "TypeScript + Three.js",
      status: "Generated",
      actions: ["Open", "Preview", "Copy", "Download"]
    });
    expect(result.previewArtifactId).toBe("live-generated-artifact");
    expect(result.previewAvailable).toBe(true);
    expect(result.artifact?.content).not.toContain("import * as THREE");
    expect(result.snapshot.code.title).toBe("generated-scene.three.ts");
    expect(result.snapshot.code.excerpt).toContain(
      "const renderer = new THREE.WebGLRenderer({ antialias: true });"
    );
    expect(result.snapshot.preview).toMatchObject({
      available: true,
      artifactName: "generated-scene.three.ts",
      outputArtifactName: "generated-scene.three.ts",
      state: "ready",
      status: "Ready when opened",
      sourceArtifactId: "live-generated-artifact",
      targetId: "browser_sandbox"
    });

    expect(
      buildPreviewRendererRoute({
        artifacts: result.snapshot.artifacts,
        preview: result.snapshot.preview,
        previewArtifactId: result.previewArtifactId
      })
    ).toMatchObject({
      rendererId: "surface.three",
      supportState: "supported",
      surfaceKind: "three"
    });
  });

  it("retains a final partial product outcome for session persistence", () => {
    const result = hydrateWorkspaceFromFinalEvent(
      getLocalWorkspaceSnapshot(),
      finalEvent({
        answer: "The artifact is available as code.",
        workflow: {
          phase: "completed",
          status: "completed",
          product_outcome: {
            orchestration_status: "COMPLETED",
            provider_status: "COMPLETED",
            generation_status: "COMPLETED",
            deliverable_status: "USABLE",
            artifact_extraction_status: "EXTRACTED",
            artifact_runnability: "UNSUPPORTED",
            preview_status: "UNAVAILABLE",
            runtime_health: "NOT_AVAILABLE",
            product_outcome: "PARTIAL",
            summary: "A usable artifact was produced, but live preview is unavailable.",
            recovery_action: "Open Code to use the artifact."
          }
        }
      })
    );

    expect(result.snapshot.workflow.productOutcome).toMatchObject({
      product_outcome: "PARTIAL",
      artifact_runnability: "UNSUPPORTED",
      preview_status: "UNAVAILABLE"
    });
    expect(result.snapshot.workflow).toMatchObject({
      status: "completed",
      currentNode: "finalization",
      currentStep: "A usable artifact was produced, but live preview is unavailable."
    });
    expect(
      result.snapshot.workflow.steps.find((step) => step.nodeId === "finalization")
    ).toMatchObject({ state: "complete" });
  });

  it("settles a provider fallback as partial instead of claiming preview success", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromFinalEvent(
      {
        ...snapshot,
        workflow: {
          ...snapshot.workflow,
          productOutcome: {
            orchestration_status: "FALLBACK",
            provider_status: "FALLBACK",
            generation_status: "PENDING",
            deliverable_status: "UNKNOWN",
            artifact_extraction_status: "UNKNOWN",
            artifact_runnability: "UNKNOWN",
            preview_status: "UNKNOWN",
            runtime_health: "UNKNOWN",
            product_outcome: "IN_PROGRESS",
            summary: "A local fallback is being prepared after the provider became unavailable.",
            recovery_action: ""
          }
        }
      },
      finalEvent({
        answer: "Provider fallback completed with a local draft while preserving the workspace session."
      })
    );

    expect(result.previewAvailable).toBe(false);
    expect(result.snapshot.workflow.productOutcome).toMatchObject({
      provider_status: "FALLBACK",
      preview_status: "UNAVAILABLE",
      product_outcome: "PARTIAL"
    });
  });

  it("downgrades a claimed success when hydration cannot prepare a preview", () => {
    const result = hydrateWorkspaceFromFinalEvent(
      getLocalWorkspaceSnapshot(),
      finalEvent({
        answer: "The artifact is available as code.",
        workflow: {
          phase: "completed",
          status: "completed",
          product_outcome: {
            orchestration_status: "COMPLETED",
            provider_status: "COMPLETED",
            generation_status: "COMPLETED",
            deliverable_status: "USABLE",
            artifact_extraction_status: "EXTRACTED",
            artifact_runnability: "RUNNABLE",
            preview_status: "PREPARED",
            runtime_health: "PENDING_BROWSER_VALIDATION",
            product_outcome: "SUCCESS",
            summary: "The requested deliverable is ready.",
            recovery_action: ""
          }
        }
      })
    );

    expect(result.previewAvailable).toBe(false);
    expect(result.snapshot.workflow.productOutcome).toMatchObject({
      product_outcome: "PARTIAL",
      artifact_runnability: "UNSUPPORTED",
      preview_status: "UNAVAILABLE",
      runtime_health: "NOT_AVAILABLE"
    });
  });

  it("keeps standalone Three.js HTML inspectable without falsely opening the controlled runtime", () => {
    const result = hydrateWorkspaceFromFinalEvent(
      getLocalWorkspaceSnapshot(),
      finalEvent({
        artifacts: [
          {
            id: "standalone-three-document",
            title: "generated-scene-1.three.ts",
            type: "code",
            language: "typescript",
            runtime: "three",
            content: [
              "<!-- standalone Three.js document -->",
              "<!doctype html>",
              "<html><body><canvas id=\"c\"></canvas>",
              "<script type=\"module\">",
              "import * as THREE from 'https://example.test/three.js';",
              "const scene = new THREE.Scene();",
              "</script></body></html>"
            ].join("\n")
          }
        ]
      })
    );

    expect(result.artifact).toMatchObject({
      id: "standalone-three-document",
      runtime: null,
      previewEligible: false,
      actions: ["Open", "Copy", "Download"]
    });
    expect(result.artifact?.title).not.toMatch(/^generated-/);
    expect(result.artifact?.title).toMatch(/\.three\.ts$/);
    expect(result.previewAvailable).toBe(false);
    expect(result.previewArtifactId).toBe("");
    expect(result.snapshot.preview).toMatchObject({
      available: false,
      state: "unavailable",
      title: "Preview unavailable"
    });
    expect(result.snapshot.preview.summary).toContain(
      "Standalone HTML documents cannot run"
    );
  });

  it("keeps React Three Fiber source code-only even when stale metadata claims a Three.js preview", () => {
    const result = hydrateWorkspaceFromFinalEvent(
      getLocalWorkspaceSnapshot(),
      finalEvent({
        artifacts: [
          {
            id: "react-three-fiber-component",
            title: "generated-scene-1.three.ts",
            type: "code",
            language: "typescript",
            runtime: "three",
            renderer_id: "surface.three",
            preview_eligible: true,
            preview_target: "browser_sandbox",
            summary: "Extracted from the generation result; matched Three.js creative runtime; for react three fiber.",
            content: [
              'import { Canvas, useFrame } from "@react-three/fiber";',
              "function Orb() { useFrame(() => {}); return <mesh />; }",
              "export default function Study() { return <Canvas><Orb /></Canvas>; }"
            ].join("\n")
          }
        ]
      })
    );

    expect(result.artifact).toMatchObject({
      id: "react-three-fiber-component",
      language: "TypeScript + React Three Fiber",
      runtime: null,
      previewEligible: false,
      actions: ["Open", "Copy", "Download"]
    });
    expect(result.artifact?.title).not.toMatch(/^generated-/);
    expect(result.artifact?.title).toMatch(/\.r3f\.tsx$/);
    expect(result.artifact?.summary).toContain(
      "React Three Fiber components need their own bundle runtime"
    );
    expect(result.artifact?.summary).not.toContain("matched Three.js creative runtime");
    expect(result.previewAvailable).toBe(false);
    expect(result.snapshot.preview).toMatchObject({
      available: false,
      state: "unavailable",
      title: "Preview unavailable"
    });
    expect(result.snapshot.preview.summary).toContain(
      "React Three Fiber components need their own bundle runtime"
    );
  });

  it("hydrates final stream code into a previewable GSAP artifact", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromFinalEvent(
      snapshot,
      finalEvent({
        answer: [
          "Here is the motion study:",
          "```ts",
          "const tl = gsap.timeline({ repeat: -1, yoyo: true });",
          "tl.to('.particle', { x: 140, rotation: 90, opacity: 0.3, stagger: 0.08 });",
          "tl.to('.ring', { scale: 1.2, duration: 1.4 }, 0);",
          "```"
        ].join("\n")
      })
    );

    expect(result.artifact).toMatchObject({
      id: "live-generated-artifact",
      title: "generated-motion.gsap.ts",
      type: "code",
      language: "TypeScript + GSAP",
      status: "Generated",
      actions: ["Open", "Preview", "Copy", "Download"]
    });
    expect(result.previewArtifactId).toBe("live-generated-artifact");
    expect(result.previewAvailable).toBe(true);
    expect(result.snapshot.preview).toMatchObject({
      available: true,
      artifactName: "generated-motion.gsap.ts",
      outputArtifactName: "generated-motion.gsap.ts",
      renderer: "surface.gsap",
      state: "ready",
      target: "Browser preview / GSAP",
      targetId: "browser_sandbox"
    });
    expect(
      buildPreviewRendererRoute({
        artifacts: result.snapshot.artifacts,
        preview: result.snapshot.preview,
        previewArtifactId: result.previewArtifactId
      })
    ).toMatchObject({
      rendererId: "surface.gsap",
      supportState: "supported",
      surfaceKind: "gsap"
    });
  });

  it("hydrates final stream SVG into a previewable vector artifact", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromFinalEvent(
      snapshot,
      finalEvent({
        answer: [
          "```svg",
          '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">',
          '  <circle cx="60" cy="60" r="24" fill="#4cd7c8">',
          '    <animate attributeName="r" values="22;30;22" dur="2s" repeatCount="indefinite" />',
          "  </circle>",
          "</svg>",
          "```"
        ].join("\n")
      })
    );

    expect(result.artifact).toMatchObject({
      title: "generated-vector.svg",
      type: "code",
      language: "SVG",
      runtime: "svg"
    });
    expect(result.snapshot.preview).toMatchObject({
      artifactName: "generated-vector.svg",
      renderer: "surface.svg",
      target: "Browser preview / SVG",
      targetId: "browser_sandbox"
    });
    expect(
      buildPreviewRendererRoute({
        artifacts: result.snapshot.artifacts,
        preview: result.snapshot.preview,
        previewArtifactId: result.previewArtifactId
      })
    ).toMatchObject({
      rendererId: "surface.svg",
      supportState: "supported",
      surfaceKind: "svg"
    });
  });

  it("hydrates final stream Canvas code into a previewable runtime artifact", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromFinalEvent(
      snapshot,
      finalEvent({
        answer: [
          "```js",
          "const canvas = document.querySelector('canvas');",
          "const ctx = canvas.getContext('2d');",
          "function draw(time) {",
          "  ctx.clearRect(0, 0, canvas.width, canvas.height);",
          "  ctx.fillStyle = '#4cd7c8';",
          "  ctx.fillRect(24 + Math.sin(time * 0.002) * 32, 24, 96, 96);",
          "  requestAnimationFrame(draw);",
          "}",
          "requestAnimationFrame(draw);",
          "```"
        ].join("\n")
      })
    );

    expect(result.artifact).toMatchObject({
      title: "generated-canvas.canvas.js",
      type: "code",
      language: "JavaScript + Canvas",
      runtime: "canvas"
    });
    expect(result.snapshot.preview).toMatchObject({
      artifactName: "generated-canvas.canvas.js",
      renderer: "surface.canvas",
      target: "Browser preview / Canvas",
      targetId: "browser_sandbox"
    });
    expect(
      buildPreviewRendererRoute({
        artifacts: result.snapshot.artifacts,
        preview: result.snapshot.preview,
        previewArtifactId: result.previewArtifactId
      })
    ).toMatchObject({
      rendererId: "surface.canvas",
      supportState: "supported",
      surfaceKind: "canvas"
    });
  });

  it("normalizes the homepage p5 suggestion into a previewable JavaScript artifact", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromFinalEvent(
      snapshot,
      finalEvent({
        answer: [
          "```ts",
          "import p5 from 'p5';",
          "type Particle = { x: number; y: number };",
          "const particles: Particle[] = [];",
          "function setup(): void {",
          "  createCanvas(640, 360);",
          "  for (let i = 0; i < 24; i += 1) particles.push({ x: random(width), y: random(height) });",
          "}",
          "function draw(): void {",
          "  background(8, 12, 18);",
          "  particles.forEach((p: Particle) => circle(p.x, p.y, 4));",
          "}",
          "```"
        ].join("\n")
      })
    );

    expect(result.artifact).toMatchObject({
      title: "generated-sketch.p5.js",
      type: "code",
      language: "JavaScript + p5.js",
      runtime: "p5",
      rendererId: "surface.p5"
    });
    expect(result.artifact?.content).not.toContain("import p5");
    expect(result.artifact?.content).not.toContain("type Particle");
    expect(result.artifact?.content).not.toContain(": number");
    expect(result.snapshot.preview).toMatchObject({
      available: true,
      artifactName: "generated-sketch.p5.js",
      renderer: "surface.p5",
      state: "ready",
      targetId: "browser_sandbox"
    });

    expect(
      buildPreviewRendererRoute({
        artifacts: result.snapshot.artifacts,
        preview: result.snapshot.preview,
        previewArtifactId: result.previewArtifactId
      })
    ).toMatchObject({
      rendererId: "surface.p5",
      supportState: "supported",
      surfaceKind: "p5",
      supportLabel: "Runtime ready"
    });
  });

  it("promotes fenced p5 source from a structured Markdown response before selecting an artifact", () => {
    const source = [
      "function setup() { createCanvas(640, 360); }",
      "function draw() { background(12); circle(width / 2, height / 2, 32); }"
    ].join("\n");
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromFinalEvent(
      snapshot,
      finalEvent({
        answer: "A runnable sketch was generated.",
        artifacts: [
          {
            id: "assistant-response",
            title: "assistant-response.md",
            type: "export",
            language: "markdown",
            is_default: true,
            content: [
              "Use the following source:",
              "```javascript",
              source,
              "```",
              "Adjust the density after previewing."
            ].join("\n")
          }
        ]
      })
    );

    expect(result.artifact).toMatchObject({
      runtime: "p5",
      previewEligible: true
    });
    expect(result.artifact?.content).toBe(source);
    expect(result.artifact?.title).not.toMatch(/^generated-/);
    expect(result.artifact?.title).toMatch(/\.p5\.js$/);
    expect(result.artifact?.content).not.toContain("```");
    expect(result.snapshot.artifacts).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ title: expect.stringMatching(/\.p5\.js$/) }),
        expect.objectContaining({
          title: expect.stringMatching(/^(?!assistant-response).+\.md$/),
          type: "export"
        })
      ])
    );
    expect(result.previewArtifactId).toBe(result.artifact?.id);
    expect(result.snapshot.code.title).toBe(result.artifact?.title);
  });

  it("accepts every supported fenced p5 JavaScript label", () => {
    const source = [
      "function setup() { createCanvas(640, 360); }",
      "function draw() { background(12); circle(80, 80, 24); }"
    ].join("\n");

    for (const label of ["javascript", "js", "p5", "p5.js"]) {
      const result = hydrateWorkspaceFromFinalEvent(
        getLocalWorkspaceSnapshot(),
        finalEvent({ answer: `Intro\n\`\`\`${label}\n${source}\n\`\`\`\nOutro` })
      );

      expect(result.artifact?.title).toMatch(/\.p5\.js$/);
      expect(result.previewAvailable).toBe(true);
      expect(result.artifact?.content).not.toContain("```");
    }
  });

  it("uses filename fence metadata without retaining the metadata prefix", () => {
    const source = [
      "function setup() { createCanvas(640, 360); }",
      "function draw() { background(12); circle(80, 80, 24); }"
    ].join("\n");
    const result = hydrateWorkspaceFromFinalEvent(
      getLocalWorkspaceSnapshot(),
      finalEvent({
        answer: `\`\`\`javascript filename=named-field.p5.js\n${source}\n\`\`\``
      })
    );

    expect(result.artifact?.title).toBe("named-field.p5.js");
  });

  it("keeps a constrained p5 sketch eligible when streamed as an artifact", () => {
    const result = hydrateWorkspaceFromFinalEvent(
      getLocalWorkspaceSnapshot(),
      finalEvent({
        artifacts: [
          {
            id: "constrained-p5",
            title: "generated-sketch-1.p5.js",
            language: "javascript",
            runtime: "p5",
            content: [
              "function setup() { createCanvas(640, 360); }",
              "function draw() {",
              "  const x = constrain(noise(frameCount * 0.01) * width, 0, width);",
              "  background(8); circle(x, height / 2, 18);",
              "}"
            ].join("\n")
          }
        ]
      })
    );

    expect(result.artifact).toMatchObject({
      id: "constrained-p5",
      runtime: "p5",
      rendererId: "surface.p5",
      previewEligible: true,
      status: "Generated"
    });
    expect(result.previewAvailable).toBe(true);
    expect(result.previewArtifactId).toBe("constrained-p5");
  });

  it("selects a lifecycle-ready p5 block over unrelated JavaScript", () => {
    const result = hydrateWorkspaceFromFinalEvent(
      getLocalWorkspaceSnapshot(),
      finalEvent({
        answer: [
          "```js",
          "console.log('helper only');",
          "```",
          "```javascript",
          "function setup() { createCanvas(640, 360); }",
          "function draw() { background(12); circle(80, 80, 24); }",
          "```"
        ].join("\n")
      })
    );

    expect(result.artifact?.title).toMatch(/\.p5\.js$/);
    expect(result.artifact?.content).not.toContain("helper only");
    expect(result.previewAvailable).toBe(true);
  });

  it("keeps a failed p5 extraction explicit and unavailable", () => {
    const result = hydrateWorkspaceFromFinalEvent(
      getLocalWorkspaceSnapshot(),
      finalEvent({
        answer: "The generated source could not be previewed.",
        artifacts: [
          {
            id: "failed-p5",
            title: "generated-sketch-1.p5.js",
            language: "javascript",
            status: "Runnable artifact extraction failed",
            content: "console.log('not a p5 lifecycle sketch');"
          }
        ]
      })
    );

    expect(result.artifact).toMatchObject({
      status: "Runnable artifact extraction failed",
    });
    expect(result.artifact?.title).not.toMatch(/^generated-/);
    expect(result.artifact?.title).toMatch(/\.p5\.js$/);
    expect(result.previewAvailable).toBe(false);
    expect(result.snapshot.preview).toMatchObject({
      status: "Completed without runnable artifact",
      title: "Preview unavailable"
    });
  });

  it("does not route HTML p5-looking output to the live p5 renderer", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromFinalEvent(
      snapshot,
      finalEvent({
        answer: [
          "```html generated-sketch-1.p5.ts",
          "<!doctype html>",
          "<html><body><script>",
          "function setup() { createCanvas(640, 360); }",
          "function draw() { circle(20, 20, 10); }",
          "</script></body></html>",
          "```"
        ].join("\n")
      })
    );

    expect(result.artifact).toMatchObject({
      title: "generated-sketch-1.p5.ts",
      runtime: null,
      rendererId: null
    });
    expect(result.previewAvailable).toBe(false);
    expect(result.snapshot.preview).toMatchObject({
      available: false,
      state: "unavailable",
      title: "Preview unavailable"
    });
  });

  it("uses structured artifact payloads before parsing chat prose", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromFinalEvent(
      snapshot,
      finalEvent({
        answer: "Generated shader attached.",
        artifacts: [
          {
            id: "shader-output",
            filename: "aurora.frag",
            language: "glsl",
            content: "void main() {\n  gl_FragColor = vec4(1.0);\n}"
          }
        ]
      })
    );

    expect(result.artifact).toMatchObject({
      id: "shader-output",
      title: "aurora.frag",
      language: "GLSL",
      actions: ["Open", "Preview", "Copy", "Download"]
    });
    expect(result.snapshot.preview).toMatchObject({
      artifactName: "aurora.frag",
      renderer: "surface.glsl",
      targetId: "browser_sandbox"
    });
  });

  it("replaces a generic structured provider filename with a prompt-derived runtime name", () => {
    const baseSnapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromFinalEvent(
      {
        ...baseSnapshot,
        messages: [
          ...baseSnapshot.messages,
          {
            role: "user",
            time: "15:14",
            content: "Create a p5.js flow-field particle system with soft trails."
          }
        ]
      },
      finalEvent({
        artifacts: [
          {
            id: "flow-field-provider-output",
            title: "generated-sketch.p5.js",
            language: "javascript",
            runtime: "p5",
            content:
              "function setup() { createCanvas(640, 360); }\nfunction draw() { background(12); }"
          }
        ]
      })
    );

    expect(result.artifact?.title).toBe("flow-field-particle-system-soft.p5.js");
    expect(result.snapshot.preview).toMatchObject({
      artifactName: "flow-field-particle-system-soft.p5.js",
      sourceArtifactName: "flow-field-particle-system-soft.p5.js"
    });
  });

  it("keeps unsupported GLSL source code-only before the WebGL runtime mounts", () => {
    const result = hydrateWorkspaceFromFinalEvent(
      getLocalWorkspaceSnapshot(),
      finalEvent({
        artifacts: [
          {
            id: "sampled-shader",
            filename: "sampled-field.frag",
            language: "glsl",
            runtime: "glsl",
            renderer_id: "surface.glsl",
            preview_eligible: true,
            content:
              "uniform sampler2D sourceTexture; void main() { gl_FragColor = texture2D(sourceTexture, vec2(0.5)); }"
          }
        ]
      })
    );

    expect(result.artifact).toMatchObject({
      id: "sampled-shader",
      runtime: null,
      rendererId: null,
      previewEligible: false,
      actions: ["Open", "Copy", "Download"]
    });
    expect(result.previewAvailable).toBe(false);
    expect(result.snapshot.preview.summary).toContain(
      "outside the current bounded runtime subset"
    );
  });

  it("hydrates graph-owned artifact extraction events before finalization", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromArtifactExtractedEvent(
      snapshot,
      {
        event_type: "artifact_extracted",
        payload: {
          artifacts: [
            {
              id: "graph-sketch",
              title: "graph-sketch.p5.js",
              language: "JavaScript + p5.js",
              source_language: "javascript",
              creative_translation: {
                output_modality: "visual",
                creative_intent:
                  "Create a meditative spiral with drifting cyan particles.",
                symbolic_references: [],
                geometric_references: ["spiral"],
                musical_references: [],
                mood_atmosphere: ["meditative"],
                movement_language: ["drift"],
                color_material_direction: ["cyan"],
                runtime_recommendations: ["p5.js"],
                structure_direction: [
                  "Build visual structure from the requested geometry: spiral."
                ],
                generation_constraints: [],
                refinement_targets: [
                  "Preserve atmosphere: meditative",
                  "Tune motion character: drift"
                ],
                sacred_geometry: {
                  concepts: ["spiral"],
                  geometric_structure: [
                    "Build from a continuous polar spiral path."
                  ],
                  symmetry_type: ["Use rotational progression."],
                  movement_behavior: ["Move points along the curve."],
                  visual_composition: ["Protect the spiral origin."],
                  color_material_direction: [
                    "Use a directional hue progression."
                  ],
                  runtime_recommendations: ["p5.js", "GLSL"],
                  audio_implications: [],
                  generation_constraints: [
                    "Do not add unsupported symbolic claims."
                  ]
                },
                shader_presets: {
                  presets: ["glow"],
                  color_behavior: ["Use a bright core color."],
                  light_material_behavior: ["Use bounded emission layers."],
                  motion_behavior: ["Pulse intensity slowly."],
                  shader_structure: ["Separate an emission mask."],
                  runtime_suitability: [
                    "Use the selected compatible runtime: p5.js."
                  ],
                  performance_constraints: [
                    "Use a bounded number of glow layers."
                  ]
                },
                visual_style: {
                  styles: ["sacred geometry"],
                  palette_behavior: ["Use controlled luminous accents."],
                  contrast_behavior: ["Separate primary geometry."],
                  composition_tendencies: [
                    "Build from explicit symmetry."
                  ],
                  motion_tendencies: ["Preserve proportional relationships."],
                  texture_tendencies: ["Keep texture subordinate to line."],
                  spatial_organization: ["Maintain a clear center."],
                  runtime_suitability: [
                    "Use the selected compatible runtime: p5.js."
                  ]
                },
                audio_reactive: {
                  mappings: [
                    {
                      source: "amplitude",
                      targets: ["scale", "glow"],
                      intensity: "subtle",
                      behavior: "Smooth short peaks.",
                      evidence: ["user prompt", "shader presets"]
                    }
                  ],
                  audio_runtime: "Tone.js",
                  visual_runtime: "p5.js",
                  activation: "explicit_user_gesture",
                  summary: "amplitude -> scale / glow"
                }
              },
              content:
                "function setup() {\n  createCanvas(640, 360);\n}\nfunction draw() {\n  background(12);\n}"
            }
          ]
        },
        sequence: 3
      }
    );

    expect(result.artifact).toMatchObject({
      id: "graph-sketch",
      title: "graph-sketch.p5.js",
      language: "JavaScript + p5.js",
      creativeTranslation: {
        outputModality: "visual",
        geometricReferences: ["spiral"],
        movementLanguage: ["drift"],
        runtimeRecommendations: ["p5.js"],
        sacredGeometry: {
          concepts: ["spiral"],
          symmetryType: ["Use rotational progression."],
          runtimeRecommendations: ["p5.js", "GLSL"]
        },
        shaderPresets: {
          presets: ["glow"],
          runtimeSuitability: [
            "Use the selected compatible runtime: p5.js."
          ]
        },
        visualStyle: {
          styles: ["sacred geometry"],
          runtimeSuitability: [
            "Use the selected compatible runtime: p5.js."
          ]
        },
        audioReactive: {
          activation: "explicit_user_gesture",
          audioRuntime: "Tone.js",
          mappings: [
            {
              source: "amplitude",
              targets: ["scale", "glow"]
            }
          ],
          visualRuntime: "p5.js"
        }
      },
      actions: ["Open", "Preview", "Copy", "Download"]
    });
    expect(result.previewArtifactId).toBe("graph-sketch");
    expect(result.snapshot.preview).toMatchObject({
      available: true,
      renderer: "surface.p5",
      sourceArtifactId: "graph-sketch",
      trigger: "Artifact extraction"
    });
  });

  it("hydrates multiple graph-owned artifacts and selects the default preview candidate", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromArtifactExtractedEvent(snapshot, {
      event_type: "artifact_extracted",
      payload: {
        artifacts: [
          {
            id: "palette-notes",
            title: "palette-notes.py",
            language: "Python",
            source_language: "python",
            content: "palette = ['#0bf', '#111']",
            preview_eligible: false,
            source_order: 1,
            is_default: false
          },
          {
            id: "orbit-sketch",
            title: "orbit-sketch.p5.js",
            language: "JavaScript + p5.js",
            source_language: "javascript",
            content:
              "function setup() {\n  createCanvas(640, 360);\n}\nfunction draw() {\n  background(12);\n}",
            preview_eligible: true,
            preview_target: "browser_sandbox",
            runtime: "p5",
            renderer_id: "surface.p5",
            source_order: 2,
            is_default: true,
            is_recommended: true,
            quality_score: 0.91,
            quality_rank: 1,
            critique: {
              artifact_id: "orbit-sketch",
              artifact_title: "orbit-sketch.p5.js",
              source_order: 2,
              overall_score: 0.91,
              rank: 1,
              passed: true,
              recommended: true,
              prompt_alignment: {
                score: 0.88,
                rationale: "p5 sketch matches the brief."
              },
              creative_quality: {
                score: 0.92,
                rationale: "Strong motion and visual structure."
              },
              runtime_suitability: {
                score: 1,
                rationale: "p5 runtime is supported."
              },
              code_quality: {
                score: 0.9,
                rationale: "Setup and draw loop are complete."
              },
              preview_readiness: {
                score: 1,
                rationale: "Preview metadata is ready."
              },
              domain_appropriateness: {
                score: 0.86,
                rationale: "Domain matches p5."
              },
              creative_evaluation: {
                overall_score: 0.87,
                composition: {
                  score: 0.88,
                  level: "strong",
                  observation: "Clear focal hierarchy detected.",
                  evidence: ["marker: center"]
                },
                originality: {
                  score: 0.82,
                  level: "strong",
                  observation: "Generative variation is present.",
                  evidence: ["marker: noise"]
                },
                coherence: {
                  score: 0.91,
                  level: "strong",
                  observation: "Runtime structure is coherent.",
                  evidence: ["balanced blocks"]
                },
                aesthetic_consistency: {
                  score: 0.86,
                  level: "strong",
                  observation: "Palette signals are consistent.",
                  evidence: ["marker: color"]
                },
                expressiveness: {
                  score: 0.88,
                  level: "strong",
                  observation: "Motion develops over time.",
                  evidence: ["marker: framecount"]
                },
                strengths: ["Composition: Clear focal hierarchy detected."],
                refinement_opportunities: [],
                summary: "5 of 5 creative dimensions are strong."
              },
              sacred_consistency: {
                overall_score: 0.84,
                alignment: {
                  score: 0.86,
                  level: "aligned",
                  observation: "Matched mandala metadata cues.",
                  evidence: ["marker: mandala"]
                },
                motif_consistency: {
                  score: 0.83,
                  level: "aligned",
                  observation: "Detected radial geometry signals.",
                  evidence: ["marker: radial"]
                },
                modality_coherence: {
                  score: 0.78,
                  level: "aligned",
                  observation: "Detected visual and motion signals.",
                  evidence: ["marker: canvas"]
                },
                claim_safety: {
                  score: 0.9,
                  level: "aligned",
                  observation:
                    "No unsupported symbolic authority markers were detected.",
                  evidence: ["bounded design-motif language"]
                },
                strengths: [
                  "Claim safety: No unsupported symbolic authority markers."
                ],
                refinement_opportunities: [],
                summary: "Checked 2 symbolic/geometric metadata cues."
              },
              calibrated_quality: {
                score: 0.86,
                legacy_score: 0.91,
                decision_band: "strong_candidate",
                confidence: "medium",
                signals: [
                  {
                    key: "legacy_critique",
                    label: "Legacy critique",
                    score: 0.91,
                    weight: 0.34,
                    rationale:
                      "Existing weighted artifact critique score is preserved."
                  },
                  {
                    key: "runtime_preview",
                    label: "Runtime and preview",
                    score: 1,
                    weight: 0.18,
                    rationale:
                      "Runtime suitability and preview readiness are aligned."
                  }
                ],
                adjustments: [],
                rationale:
                  "strong candidate at 0.86; legacy score 0.91. No conservative caps were required.",
                summary:
                  "Calibrated decision-support score 0.86 from 2 available signal(s)."
              },
              legacy_rank: 1,
              reasons: [],
              rationale: "orbit-sketch.p5.js is the recommended candidate.",
              refinement_guidance: null
            }
          }
        ]
      },
      sequence: 3
    });

    expect(result.activeArtifactId).toBe("orbit-sketch");
    expect(result.artifact).toMatchObject({
      id: "orbit-sketch",
      isDefault: true,
      isRecommended: true,
      previewEligible: true,
      qualityRank: 1,
      qualityScore: 0.91,
      sourceOrder: 2
    });
    expect(result.artifact?.critique).toMatchObject({
      artifactId: "orbit-sketch",
      creativeEvaluation: {
        overallScore: 0.87,
        aestheticConsistency: {
          level: "strong",
          score: 0.86
        },
        refinementOpportunities: []
      },
      sacredConsistency: {
        overallScore: 0.84,
        motifConsistency: {
          level: "aligned",
          score: 0.83
        },
        claimSafety: {
          level: "aligned",
          score: 0.9
        },
        refinementOpportunities: []
      },
      calibratedQuality: {
        score: 0.86,
        legacyScore: 0.91,
        decisionBand: "strong_candidate",
        signals: [
          {
            key: "legacy_critique",
            score: 0.91
          },
          {
            key: "runtime_preview",
            score: 1
          }
        ]
      },
      legacyRank: 1,
      overallScore: 0.91,
      rank: 1,
      recommended: true,
      rationale: "orbit-sketch.p5.js is the recommended candidate."
    });
    expect(result.snapshot.artifacts.slice(0, 2).map((artifact) => artifact.id)).toEqual([
      "palette-notes",
      "orbit-sketch"
    ]);
    expect(result.snapshot.artifacts[0]).toMatchObject({
      actions: ["Open", "Copy", "Download"],
      previewEligible: false,
      sourceOrder: 1
    });
    expect(result.snapshot.code).toMatchObject({
      title: "orbit-sketch.p5.js",
      language: "JavaScript + p5.js",
      status: "Generated"
    });
    expect(result.snapshot.code.excerpt).toContain("  createCanvas(640, 360);");
    expect(result.previewArtifactId).toBe("orbit-sketch");
    expect(result.snapshot.preview).toMatchObject({
      available: true,
      renderer: "surface.p5",
      sourceArtifactId: "orbit-sketch",
      targetId: "browser_sandbox"
    });
  });

  it("creates a readable artifact and disables preview for non-runnable answers", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromFinalEvent(
      snapshot,
      finalEvent({
        answer:
          "Use a slower modulation curve and keep the palette contrast high for projection."
      })
    );
    const document = buildArtifactDocument(result.snapshot, result.artifact!);

    expect(result.artifact).toMatchObject({
      id: "live-response-artifact",
      title: "assistant-response.md",
      type: "export",
      language: "Markdown",
      actions: ["Open", "Copy", "Download"]
    });
    expect(document.content).toContain("Use a slower modulation curve");
    expect(result.previewArtifactId).toBe("");
    expect(result.previewAvailable).toBe(false);
    expect(result.snapshot.preview).toMatchObject({
      available: false,
      state: "unavailable",
      status: "Unavailable",
      title: "Preview unavailable",
      targetId: ""
    });
  });

  it("hydrates Hydra code into a previewable sandbox artifact", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromFinalEvent(
      snapshot,
      finalEvent({
        answer: [
          "Here is the patch:",
          "```js feedback-lattice.hydra.js",
          "osc(10, 0.1, 1.2).modulate(shape(4)).out();",
          "```"
        ].join("\n")
      })
    );

    expect(result.artifact).toMatchObject({
      title: "feedback-lattice.hydra.js",
      previewEligible: true,
      rendererId: "surface.hydra",
      runtime: "hydra",
      actions: ["Open", "Preview", "Copy", "Download"]
    });
    expect(result.previewArtifactId).toBe("live-generated-artifact");
    expect(result.previewAvailable).toBe(true);
    expect(result.snapshot.preview).toMatchObject({
      available: true,
      renderer: "surface.hydra",
      state: "ready",
      target: "Browser preview / Hydra",
      targetId: "browser_sandbox"
    });
  });

  it("hydrates Tone.js code into an explicitly controlled audio preview", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromFinalEvent(
      snapshot,
      finalEvent({
        answer: [
          "Here is the generative audio patch:",
          "```js generative-pulse.tone.js",
          "const synth = new Tone.Synth().toDestination();",
          "new Tone.Sequence((time, note) => synth.triggerAttackRelease(note, '8n', time), ['C4', 'E4', 'G4'], '8n').start(0);",
          "Tone.Transport.start();",
          "```"
        ].join("\n")
      })
    );

    expect(result.artifact).toMatchObject({
      actions: ["Open", "Preview", "Copy", "Download"],
      previewEligible: true,
      rendererId: "surface.tone",
      runtime: "tone",
      title: "generative-pulse.tone.js"
    });
    expect(result.previewAvailable).toBe(true);
    expect(result.snapshot.preview).toMatchObject({
      available: true,
      renderer: "surface.tone",
      state: "ready",
      target: "Browser preview / Tone.js",
      targetId: "browser_sandbox"
    });
  });

  it("hydrates structured refinement pass history without requiring legacy fields", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromFinalEvent(
      snapshot,
      finalEvent({
        answer: "Refined artifact attached.",
        artifacts: [
          {
            id: "refined-sketch",
            title: "refined-sketch.p5.js",
            language: "javascript",
            content: "function draw() { circle(width / 2, height / 2, 120); }",
            refinement_passes: [
              {
                pass_number: 1,
                source_artifact_id: "source-sketch",
                source_artifact_title: "source-sketch.p5.js",
                result_artifact_id: "refined-sketch",
                result_artifact_title: "refined-sketch.p5.js",
                refinement_objective: "Clarify focal hierarchy.",
                quality_before: 0.55,
                quality_after: 0.72,
                stop_reason: "quality_improved",
                summary: "Pass 1: Quality improved. Quality 0.55 -> 0.72."
              }
            ]
          }
        ]
      })
    );

    expect(result.artifact?.refinementPasses).toEqual([
      expect.objectContaining({
        passNumber: 1,
        qualityBefore: 0.55,
        qualityAfter: 0.72,
        sourceArtifactId: "source-sketch",
        stopReason: "quality_improved"
      })
    ]);
  });
});

function finalEvent(payload: Record<string, unknown>): AssistantStreamEvent {
  return {
    event_type: "final",
    payload,
    sequence: 7
  };
}
