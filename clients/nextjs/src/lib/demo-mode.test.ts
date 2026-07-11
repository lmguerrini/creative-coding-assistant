import { describe, expect, it } from "vitest";
import {
  demoModeRecommendedLiveSequence,
  demoModeScenarioCatalog,
  demoModeScenarioCount,
  demoModeScenarios,
  getDefaultDemoModeScenario
} from "./demo-mode";

describe("demo mode scenarios", () => {
  it("keeps a curated set of presenter-ready, supported workflows visible", () => {
    expect(demoModeScenarioCount).toBe(10);
    expect(demoModeScenarios.map((scenario) => scenario.id)).toEqual([
      "cymatic-chladni-audiovisual",
      "physarum-p5-hero",
      "kinetic-three-hero",
      "chladni-glsl-hero",
      "retrieval-grounded-design-brief",
      "multi-agent-production-plan",
      "single-agent-line-study",
      "export-handoff-package",
      "multimodal-reference-study",
      "failure-recovery-rehearsal"
    ]);
    expect(demoModeScenarioCatalog).toHaveLength(10);
    expect(demoModeScenarios).not.toContainEqual(
      expect.objectContaining({ runtime: "Hydra" })
    );
  });

  it("gives every visible scenario its complete demo contract", () => {
    for (const scenario of demoModeScenarios) {
      expect(scenario.title).toBeTruthy();
      expect(scenario.concept).toBeTruthy();
      expect(scenario.purpose).toBeTruthy();
      expect(scenario.runtime).toBeTruthy();
      expect(scenario.workflow).toBeTruthy();
      expect(scenario.inputRequirement).toBeTruthy();
      expect(scenario.prompt).toBeTruthy();
      expect(scenario.expectedArtifact).toBeTruthy();
      expect(scenario.expectedPreview).toBeTruthy();
      expect(scenario.expectedInteraction).toBeTruthy();
      expect(scenario.expectedValidation).toBeTruthy();
      expect(scenario.fallback).toBeTruthy();
    }
  });

  it("keeps the Cymatics opener deterministic and silent until the user opts in", () => {
    const cymatic = getDefaultDemoModeScenario();

    expect(cymatic.id).toBe("cymatic-chladni-audiovisual");
    expect(cymatic.workflowMode).toBe("single_agent");
    expect(cymatic.prompt).toContain("// CCA_VISUAL: cymatics");
    expect(cymatic.prompt).toContain("Tone.Transport.bpm.value = 96");
    expect(cymatic.expectedInteraction).toContain("Start audio");
    expect(cymatic.sourceBoundary).toContain("microphone");
  });

  it("keeps the featured sequence within the visible, live browser scenarios", () => {
    expect(demoModeRecommendedLiveSequence.map((item) => item.scenarioId)).toEqual([
      "cymatic-chladni-audiovisual",
      "physarum-p5-hero",
      "kinetic-three-hero",
      "chladni-glsl-hero"
    ]);
    expect(
      demoModeRecommendedLiveSequence.every((item) =>
        demoModeScenarios.some((scenario) => scenario.id === item.scenarioId)
      )
    ).toBe(true);
  });

  it("makes multimodal input and controlled failure boundaries explicit", () => {
    const multimodal = demoModeScenarios.find(
      (scenario) => scenario.id === "multimodal-reference-study"
    );
    const recovery = demoModeScenarios.find(
      (scenario) => scenario.id === "failure-recovery-rehearsal"
    );

    expect(multimodal?.inputRequirement).toContain("Attach one PNG");
    expect(multimodal?.expectedPreview).toContain("self-contained p5.js canvas");
    expect(recovery?.providerRequirement).toContain("Controlled failure fixture");
    expect(recovery?.expectedPreview).toContain("No live preview");
  });

  it("keeps workflow prompts inside the bounded planning-summary range", () => {
    for (const id of [
      "retrieval-grounded-design-brief",
      "multi-agent-production-plan",
      "export-handoff-package"
    ]) {
      const scenario = demoModeScenarios.find((item) => item.id === id);

      expect(scenario?.prompt.length).toBeLessThanOrEqual(280);
    }
  });

  it("keeps the export case anchored to a named Markdown handoff artifact", () => {
    const exportHandoff = demoModeScenarios.find(
      (scenario) => scenario.id === "export-handoff-package"
    );

    expect(exportHandoff?.prompt).toContain(
      "chladni-touchdesigner-handoff.md"
    );
    expect(exportHandoff?.prompt).toContain("fenced markdown block");
    expect(exportHandoff?.expectedPreview).toContain("No internal TouchDesigner");
  });
});
