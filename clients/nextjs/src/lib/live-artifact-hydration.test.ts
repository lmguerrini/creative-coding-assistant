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
});

function finalEvent(payload: Record<string, unknown>): AssistantStreamEvent {
  return {
    event_type: "final",
    payload,
    sequence: 7
  };
}
