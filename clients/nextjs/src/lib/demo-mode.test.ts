import { describe, expect, it } from "vitest";
import {
  demoModeScenarioCount,
  demoModeScenarios,
  getDefaultDemoModeScenario
} from "./demo-mode";

describe("demo mode scenarios", () => {
  it("defines the final eight curated Capstone scenarios", () => {
    expect(demoModeScenarioCount).toBe(8);
    expect(demoModeScenarios.map((scenario) => scenario.id)).toEqual([
      "three-audio-reactive-visual-system",
      "p5-generative-morphogenesis-sketch",
      "glsl-shader-post-processing-visual",
      "hydra-feedback-pattern-demo",
      "retrieval-grounded-creative-coding-answer",
      "concept-to-visual-translation",
      "geometry-morphogenesis-visual-system",
      "installation-immersive-scene-planning"
    ]);
  });

  it("keeps app-facing labels within the public Capstone naming boundary", () => {
    const appFacingText = demoModeScenarios
      .flatMap((scenario) => [
        scenario.title,
        scenario.category,
        scenario.prompt,
        scenario.expectedBehavior,
        scenario.fallback,
        scenario.outputGuidance,
        scenario.sourceBoundary,
        scenario.validationPath,
        ...scenario.evidence
      ])
      .join("\n");

    expect(appFacingText).not.toMatch(/HoloGenesis/i);
    expect(appFacingText).not.toMatch(/\bsacred\b/i);
    expect(appFacingText).not.toMatch(/\bsymbolic\b/i);
  });

  it("defaults to the Three.js scenario for presenter startup", () => {
    expect(getDefaultDemoModeScenario().id).toBe(
      "three-audio-reactive-visual-system"
    );
  });
});
