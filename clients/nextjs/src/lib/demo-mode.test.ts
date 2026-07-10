import { describe, expect, it } from "vitest";
import {
  demoModeRecommendedLiveSequence,
  demoModeScenarioCatalog,
  demoModeScenarioCount,
  demoModeScenarios,
  getDefaultDemoModeScenario
} from "./demo-mode";

describe("demo mode scenarios", () => {
  it("exposes only the currently validated p5 browser-preview prompt", () => {
    expect(demoModeScenarioCount).toBe(1);
    expect(demoModeScenarios.map((scenario) => scenario.id)).toEqual([
      "p5-generative-morphogenesis-sketch"
    ]);
    expect(demoModeScenarios[0]?.prompt).toContain(
      "Optimize for browser preview at 60 fps"
    );
    expect(demoModeScenarios[0]?.prompt).toContain("strokeCap(ROUND)");
  });

  it("retains the wider Capstone catalog without presenting unvalidated prompts", () => {
    expect(demoModeScenarioCatalog).toHaveLength(8);
    expect(demoModeScenarios).not.toContainEqual(
      expect.objectContaining({ id: "three-audio-reactive-visual-system" })
    );
    expect(demoModeScenarios).not.toContainEqual(
      expect.objectContaining({ id: "glsl-shader-post-processing-visual" })
    );
    expect(demoModeScenarios).not.toContainEqual(
      expect.objectContaining({ id: "hydra-feedback-pattern-demo" })
    );
  });

  it("keeps the visible scenario presenter-ready and within public naming boundaries", () => {
    const scenario = getDefaultDemoModeScenario();
    const appFacingText = [
      scenario.title,
      scenario.description,
      scenario.category,
      scenario.prompt,
      scenario.expectedBehavior,
      scenario.fallback,
      scenario.outputGuidance,
      scenario.sourceBoundary,
      scenario.validationPath,
      ...scenario.evidence
    ].join("\n");

    expect(scenario.description).toBeTruthy();
    expect(scenario.estimatedGenerationTime).toBeTruthy();
    expect(scenario.expectedOutput).toBeTruthy();
    expect(scenario.validationPath).toContain("Chromium");
    expect(appFacingText).not.toMatch(/HoloGenesis/i);
    expect(appFacingText).not.toMatch(/\bsacred\b/i);
    expect(appFacingText).not.toMatch(/\bsymbolic\b/i);
  });

  it("keeps featured paths limited to the visible scenario", () => {
    expect(demoModeRecommendedLiveSequence).toEqual([
      expect.objectContaining({
        role: "Verified browser preview",
        scenarioId: "p5-generative-morphogenesis-sketch"
      })
    ]);
  });
});
