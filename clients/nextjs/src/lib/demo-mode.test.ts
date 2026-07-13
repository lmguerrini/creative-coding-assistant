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
    expect(
      new Set(demoModeScenarios.slice(0, 4).map((scenario) => scenario.runtime)).size
    ).toBe(4);
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

  it("keeps the original audio opener distinct and silent until the user opts in", () => {
    const audio = getDefaultDemoModeScenario();

    expect(audio.id).toBe("cymatic-chladni-audiovisual");
    expect(audio.title).toBe("Polyrhythmic constellation");
    expect(audio.workflowMode).toBe("single_agent");
    expect(audio.prompt).toContain("Tone.FMSynth");
    expect(audio.prompt).toContain("Tone.MembraneSynth");
    expect(audio.prompt).toContain("Tone.Transport.bpm.value = 108");
    expect(audio.expectedInteraction).toContain("Start audio");
    expect(audio.sourceBoundary).toContain("microphone");
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
    expect(multimodal?.requiresImageAttachment).toBe(true);
    expect(multimodal?.expectedPreview).toContain("self-contained p5.js canvas");
    expect(multimodal?.expectedArtifact).toContain("no attachment record is persisted");
    expect(multimodal?.expectedValidation).toContain("cleared after submission");
    expect(multimodal?.sourceBoundary).toContain("before Send");
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
