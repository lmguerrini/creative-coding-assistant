import { describe, expect, it } from "vitest";
import { demoModeScenarios } from "./demo-mode";
import {
  auditDemoScenarioMetadata,
  demoClarificationFixtures,
  demoDurationBudgets,
  goldenDemoFixtures,
  summarizeDemoOutputQuality,
  summarizeDemoReliability,
  totalDemoDurationSeconds,
  validateDemoPromptContracts
} from "./demo-engine";

describe("demo engine contracts", () => {
  it("keeps golden fixtures, duration budgets, and clarification choices aligned to Demo Mode", () => {
    const scenarioIds = demoModeScenarios.map((scenario) => scenario.id);

    expect(goldenDemoFixtures.map((fixture) => fixture.scenarioId)).toEqual(scenarioIds);
    expect(demoDurationBudgets.map((budget) => budget.scenarioId)).toEqual(scenarioIds);
    expect(totalDemoDurationSeconds()).toBeLessThanOrEqual(20 * 60);
    expect(demoClarificationFixtures).toHaveLength(2);
  });

  it("audits complete metadata and exact-prompt boundaries", () => {
    expect(auditDemoScenarioMetadata()).toEqual([]);
    expect(validateDemoPromptContracts()).toEqual([]);
  });

  it("summarizes reliability and quality without inventing an evaluation run", () => {
    const reliability = summarizeDemoReliability([
      { scenarioId: "physarum-p5-hero", layer: "automated", passed: true },
      { scenarioId: "failure-recovery-rehearsal", layer: "visible_output", passed: false }
    ]);
    const quality = summarizeDemoOutputQuality([
      {
        scenarioId: "cymatic-chladni-audiovisual",
        craft: 0.9,
        clarity: 0.8,
        safety: 1,
        truthfulness: 1
      }
    ]);

    expect(reliability).toMatchObject({
      failedScenarioIds: ["failure-recovery-rehearsal"],
      passed: 1,
      total: 2,
      passRate: 0.5
    });
    expect(quality.overall).toBeCloseTo(0.925);
  });
});
