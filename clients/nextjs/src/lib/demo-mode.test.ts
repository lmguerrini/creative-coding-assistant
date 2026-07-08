import { describe, expect, it } from "vitest";
import {
  demoModeRecommendedLiveSequence,
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
        scenario.description,
        scenario.category,
        scenario.prompt,
        scenario.estimatedGenerationTime,
        scenario.estimatedTokenUsage,
        scenario.workflowType,
        scenario.providerRequirement,
        scenario.retrievalRequirement,
        scenario.previewAvailability,
        scenario.fallbackAvailability,
        scenario.expectedOutput,
        scenario.complexity,
        scenario.recommendedForDemo,
        scenario.presentationTime,
        scenario.talkingPoint,
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

  it("exposes reviewer-ready metadata for every scenario", () => {
    for (const scenario of demoModeScenarios) {
      expect(scenario.description).toBeTruthy();
      expect(scenario.estimatedGenerationTime).toBeTruthy();
      expect(scenario.estimatedTokenUsage).toBeTruthy();
      expect(scenario.workflowType).toBeTruthy();
      expect(scenario.providerRequirement).toBeTruthy();
      expect(scenario.retrievalRequirement).toBeTruthy();
      expect(scenario.previewAvailability).toBeTruthy();
      expect(scenario.fallbackAvailability).toBeTruthy();
      expect(scenario.expectedOutput).toBeTruthy();
      expect(scenario.complexity).toBeTruthy();
      expect(scenario.recommendedForDemo).toBeTruthy();
      expect(scenario.presentationTime).toBeTruthy();
      expect(scenario.talkingPoint).toBeTruthy();
    }

    expect(
      demoModeScenarios.find(
        (scenario) => scenario.id === "three-audio-reactive-visual-system"
      )?.estimatedGenerationTime
    ).toBe("68.8s optimized live smoke");
    expect(
      demoModeScenarios.find(
        (scenario) => scenario.id === "hydra-feedback-pattern-demo"
      )?.estimatedGenerationTime
    ).toBe("0.4s optimized bounded route; no provider call");
  });

  it("defines the presenter recommended live sequence", () => {
    expect(demoModeRecommendedLiveSequence.map((item) => item.role)).toEqual([
      "Fastest reliable demo",
      "Most visually impressive demo",
      "Safest fallback demo",
      "Best RAG demo",
      "Best Q&A demo"
    ]);
    expect(demoModeRecommendedLiveSequence[0].scenarioId).toBe(
      "retrieval-grounded-creative-coding-answer"
    );
  });

  it("defaults to the Three.js scenario for presenter startup", () => {
    expect(getDefaultDemoModeScenario().id).toBe(
      "three-audio-reactive-visual-system"
    );
  });
});
