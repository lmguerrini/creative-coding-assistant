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
      "Source grounding",
      "3D audiovisual",
      "Shader preview",
      "Retrieval evidence",
      "Installation scope"
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

  it("keeps the final demo pack mapped to Capstone capabilities and evidence", () => {
    const capabilities = new Set(
      demoModeScenarios.map((scenario) => scenario.recommendedForDemo)
    );
    expect(capabilities).toEqual(
      new Set([
        "3D visual system",
        "Generative growth system",
        "Shader validation",
        "Feedback-pattern runtime",
        "Source-grounded answer",
        "Concept translation",
        "Multi-runtime morphogenesis",
        "Installation planning"
      ])
    );

    expect(
      demoModeScenarios.some((scenario) =>
        scenario.workflowType.toLowerCase().includes("single-domain")
      )
    ).toBe(true);
    expect(
      demoModeScenarios.some((scenario) =>
        scenario.workflowType.toLowerCase().includes("hybrid")
      )
    ).toBe(true);
    expect(
      demoModeScenarios.some((scenario) =>
        scenario.workflowType.toLowerCase().includes("multi-domain")
      )
    ).toBe(true);
    expect(
      demoModeScenarios.some((scenario) =>
        scenario.workflowType.toLowerCase().includes("planning")
      )
    ).toBe(true);

    for (const scenario of demoModeScenarios) {
      expect(scenario.retrievalRequirement).toMatch(
        /retrieved contexts|ragas/i
      );
      expect(scenario.previewAvailability).toBeTruthy();
      expect(scenario.fallbackAvailability).toBeTruthy();
      expect(scenario.expectedOutput).toBeTruthy();
      expect(scenario.presentationTime).toMatch(/\d/);
      expect(scenario.talkingPoint).toBeTruthy();
      expect(scenario.evidence.length).toBeGreaterThanOrEqual(2);
      expect(scenario.sourceBoundary).toBeTruthy();
      expect(scenario.validationPath).toBeTruthy();
    }
  });

  it("does not present multi-domain planning as live multi-agent execution", () => {
    const appFacingText = demoModeScenarios
      .flatMap((scenario) => [
        scenario.title,
        scenario.description,
        scenario.category,
        scenario.runtime,
        scenario.prompt,
        scenario.workflowType,
        scenario.expectedOutput,
        scenario.expectedBehavior,
        scenario.outputGuidance,
        scenario.sourceBoundary,
        scenario.validationPath,
        ...scenario.evidence
      ])
      .join("\n");

    expect(appFacingText).not.toMatch(/live multi-agent/i);
    expect(appFacingText).not.toMatch(/critic[- ]refinement/i);
    expect(appFacingText).not.toMatch(/studio mode/i);
    expect(appFacingText).not.toMatch(/multi-agent execution/i);
  });
});
