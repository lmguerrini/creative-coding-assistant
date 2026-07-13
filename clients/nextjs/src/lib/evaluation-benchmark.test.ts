import { describe, expect, it } from "vitest";
import type { ProductIntelligenceModel } from "./product-intelligence";
import {
  buildEvaluationBenchmarkRun,
  buildGoldenEvaluationDataset,
  createEvaluationCandidate,
  emptyRagasEvidence,
  selectEvaluationCases,
  type EvaluationCaseResult
} from "./evaluation-benchmark";

const emptyModel = {
  artifactRegistry: [],
  details: null
} as unknown as ProductIntelligenceModel;

describe("canonical evaluation benchmark", () => {
  it("deduplicates the product-authored sources into 31 stable cases", () => {
    const first = buildGoldenEvaluationDataset();
    const second = buildGoldenEvaluationDataset();

    expect(first.cases).toHaveLength(31);
    expect(first.rawSourceCount).toBe(42);
    expect(first.duplicateCount).toBe(11);
    expect(first.fingerprint).toBe(second.fingerprint);
    expect(new Set(first.cases.map((item) => item.prompt)).size).toBe(31);
    expect(first.cases.every((item) => item.applicableMetricIds.length > 0)).toBe(true);
  });

  it("keeps RAG, creative, workflow, and reliability scopes distinct", () => {
    const dataset = buildGoldenEvaluationDataset();
    const ragCases = selectEvaluationCases(dataset, { scope: "rag", caseIds: [] });
    const creativeCases = selectEvaluationCases(dataset, { scope: "creative_artifact", caseIds: [] });

    expect(ragCases.length).toBeGreaterThan(0);
    expect(creativeCases.length).toBeGreaterThan(0);
    expect(ragCases.every((item) => item.categories.includes("rag"))).toBe(true);
    expect(creativeCases.every((item) => item.categories.includes("creative_artifact"))).toBe(true);
  });

  it("withholds an aggregate and preserves unavailable evidence as blocked or missing", () => {
    const run = buildEvaluationBenchmarkRun({
      model: emptyModel,
      now: new Date("2026-07-13T12:00:00Z"),
      ragas: emptyRagasEvidence(),
      request: {
        scope: "full",
        caseIds: [],
        allowProviderCalls: false,
        approvedRagasDataset: "sanitized_public"
      }
    });

    expect(run.categoryResults.map((item) => item.category)).toEqual([
      "rag",
      "creative_artifact",
      "workflow",
      "product_reliability"
    ]);
    expect(run.measuredScore).toBeNull();
    expect(run.counts.blocked + run.counts.missing + run.counts.notRun).toBeGreaterThan(0);
    expect(run.caseResults.flatMap((item) => item.metrics).every((metric) =>
      metric.status !== "blocked" && metric.status !== "missing_evidence" || metric.score === null
    )).toBe(true);
  });

  it("creates a separate candidate without mutating the canonical prompt", () => {
    const caseResult = {
      caseId: "demo/example",
      title: "Example",
      domain: "p5.js",
      origins: ["demo"],
      categories: ["creative_artifact"],
      status: "partial",
      score: .65,
      prompt: "Draw a quiet field.",
      expectedArtifact: "example.p5.js",
      previewContract: "Live preview",
      metrics: [],
      recommendation: {
        id: "recommendation-1",
        caseId: "demo/example",
        category: "creative_artifact",
        title: "Strengthen the constraint",
        detail: "Add a measurable constraint.",
        candidateConstraint: "Keep motion below 0.5 px per frame."
      }
    } satisfies EvaluationCaseResult;

    const candidate = createEvaluationCandidate({ caseResult, createdAt: new Date("2026-07-13T12:00:00Z") });

    expect(candidate?.originalPrompt).toBe("Draw a quiet field.");
    expect(candidate?.candidatePrompt).toContain("Keep motion below 0.5 px per frame.");
    expect(candidate?.candidateScore).toBeNull();
    expect(candidate?.delta).toBeNull();
    expect(caseResult.prompt).toBe("Draw a quiet field.");
  });
});
