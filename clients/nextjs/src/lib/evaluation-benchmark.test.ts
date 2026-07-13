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
  it("deduplicates the product-authored sources into 35 stable cases", () => {
    const first = buildGoldenEvaluationDataset();
    const second = buildGoldenEvaluationDataset();

    expect(first.cases).toHaveLength(35);
    expect(first.rawSourceCount).toBe(42);
    expect(first.duplicateCount).toBe(7);
    expect(first.fingerprint).toBe(second.fingerprint);
    expect(new Set(first.cases.map((item) => item.prompt)).size).toBe(35);
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

  it("never creates a cross-category score and compares RAG trends only under one evaluator contract", () => {
    const request = {
      scope: "rag" as const,
      caseIds: [],
      allowProviderCalls: true,
      approvedRagasDataset: "sanitized_public" as const
    };
    const first = buildEvaluationBenchmarkRun({
      model: emptyModel,
      now: new Date("2026-07-13T12:00:00Z"),
      ragas: {
        ...emptyRagasEvidence(),
        state: "completed",
        datasetId: "sanitized_public",
        datasetVersion: "sanitized-ragas.v1",
        privacyClass: "committed_synthetic_public",
        metrics: ["context_precision", "faithfulness", "answer_relevancy", "context_relevancy"],
        metricScores: { context_precision: .9, faithfulness: .8, answer_relevancy: .7, context_relevancy: .8 },
        resultRows: 4,
        totalSamples: 4,
        eligibleSamples: 4,
        provider: "OpenAI evaluator",
        model: "gpt-4o-mini",
        embeddingModel: "text-embedding-3-small",
        ragasVersion: "0.4.3",
        metricContract: "ragas-supported.v2"
      },
      request
    });
    const second = buildEvaluationBenchmarkRun({
      model: emptyModel,
      now: new Date("2026-07-13T13:00:00Z"),
      previousRun: first,
      ragas: {
        ...first.ragas,
        datasetId: "redacted_public",
        datasetVersion: "redacted-live-latest4.v1",
        privacyClass: "committed_redacted_public"
      },
      request: { ...request, approvedRagasDataset: "redacted_public" }
    });

    expect(first.categoryResults[0]?.score).not.toBeNull();
    expect(first.measuredScore).toBeNull();
    expect(second.categoryResults[0]?.previousScore).toBeNull();
    expect(second.categoryResults[0]?.delta).toBeNull();

    const changedProvider = buildEvaluationBenchmarkRun({
      model: emptyModel,
      now: new Date("2026-07-13T14:00:00Z"),
      previousRun: first,
      ragas: { ...first.ragas, provider: "Another evaluator" },
      request
    });
    const changedContract = buildEvaluationBenchmarkRun({
      model: emptyModel,
      now: new Date("2026-07-13T15:00:00Z"),
      previousRun: first,
      ragas: { ...first.ragas, metricContract: "ragas-supported.v3" },
      request
    });

    expect(changedProvider.categoryResults[0]?.delta).toBeNull();
    expect(changedContract.categoryResults[0]?.delta).toBeNull();
  });
});
