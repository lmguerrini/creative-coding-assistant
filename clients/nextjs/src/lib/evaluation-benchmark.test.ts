import { describe, expect, it } from "vitest";
import type { ProductIntelligenceModel } from "./product-intelligence";
import {
  buildEvaluationBenchmarkRun,
  buildGoldenEvaluationDataset,
  CURRENT_PRODUCT_RETRIEVAL_CASE_IDS,
  CURRENT_PRODUCT_RETRIEVAL_DATASET_FINGERPRINT,
  CURRENT_PRODUCT_RETRIEVAL_GOLDEN_CASE_IDS,
  createEvaluationCandidate,
  currentProductRetrievalScoreFromEvidence,
  emptyRagasEvidence,
  matchesCurrentProductEvidenceIdentity,
  normalizeEvaluationBenchmarkMode,
  selectEvaluationCases,
  type EvaluationCaseResult,
  type RagasExecutionEvidence
} from "./evaluation-benchmark";

const emptyModel = {
  artifactRegistry: [],
  details: null
} as unknown as ProductIntelligenceModel;

const backendMetricOrder = [
  "context_precision",
  "faithfulness",
  "answer_relevancy",
  "context_relevancy",
  "context_recall"
];
const hash = (character: string) => `sha256:${character.repeat(64)}`;

function canonicalCurrentProductEvidence(
  overrides: Partial<RagasExecutionEvidence> = {}
): RagasExecutionEvidence {
  const metricScores = Object.fromEntries(backendMetricOrder.map((metricId) => [metricId, .8]));
  return {
    ...emptyRagasEvidence(),
    schemaVersion: "current-product-ragas-evidence.v1",
    scope: "rag",
    state: "completed",
    runId: "canonical-current-product-run",
    evaluatedAt: "2026-07-14T09:00:00.000Z",
    datasetId: "creative_coding_retrieval_benchmark",
    datasetVersion: "current-product-retrieval.v1",
    privacyClass: "public_official_contexts_with_authored_references",
    metrics: backendMetricOrder,
    metricScores,
    retrievalScore: .8,
    resultRows: 7,
    totalSamples: 7,
    eligibleSamples: 7,
    skippedSamples: 0,
    metricFailures: 0,
    provider: "OpenAI",
    model: "gpt-5-mini",
    embeddingModel: "text-embedding-3-small",
    ragasVersion: "0.4.3",
    metricContract: "ragas-current-product-reference.v2",
    durationMs: 1_200,
    detail: "Canonical current-product evaluation completed.",
    caseRows: CURRENT_PRODUCT_RETRIEVAL_CASE_IDS.map((sampleId, index) => ({
      sampleId,
      metrics: { ...metricScores },
      metricErrors: {},
      sourceIds: [`official-source-${index}`],
      domains: [`domain-${index}`],
      promptFingerprint: hash(String((index + 1) % 10)),
      generationFingerprint: hash(String((index + 2) % 10))
    })),
    benchmarkMode: "current_product",
    scoreOrigin: "current_product",
    benchmarkVersion: "current-product-retrieval.v1",
    selectedCaseIds: [...CURRENT_PRODUCT_RETRIEVAL_CASE_IDS],
    datasetFingerprint: CURRENT_PRODUCT_RETRIEVAL_DATASET_FINGERPRINT,
    retrievalFingerprint: hash("a"),
    promptFingerprint: hash("b"),
    generationFingerprint: hash("c"),
    outputFingerprint: hash("d"),
    selectionFingerprint: hash("e"),
    kbFingerprint: hash("f"),
    generationModel: "gpt-5-mini",
    evaluator: "OpenAI RAGAS",
    evaluatorModel: "gpt-5-mini",
    timestamp: "2026-07-14T09:00:00.000Z",
    ...overrides
  };
}

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

  it("exports exactly the seven canonical current-product retrieval case IDs", () => {
    const dataset = buildGoldenEvaluationDataset();
    const ragCases = selectEvaluationCases(dataset, { scope: "rag", caseIds: [] });

    expect(CURRENT_PRODUCT_RETRIEVAL_CASE_IDS).toHaveLength(7);
    expect(CURRENT_PRODUCT_RETRIEVAL_GOLDEN_CASE_IDS).toEqual(
      CURRENT_PRODUCT_RETRIEVAL_CASE_IDS.map((caseId) => `retrieval/${caseId}`)
    );
    expect(ragCases.map((item) => item.id)).toEqual(CURRENT_PRODUCT_RETRIEVAL_GOLDEN_CASE_IDS);
    expect(CURRENT_PRODUCT_RETRIEVAL_CASE_IDS).not.toContain("demo/retrieval-grounded-design-brief");
    expect(new Set(CURRENT_PRODUCT_RETRIEVAL_CASE_IDS).size).toBe(7);
  });

  it("accepts the exact backend five-metric contract without depending on list order", () => {
    const evidence = canonicalCurrentProductEvidence();

    expect(currentProductRetrievalScoreFromEvidence(evidence)).toBe(.8);
    expect(currentProductRetrievalScoreFromEvidence({
      ...evidence,
      metrics: [...backendMetricOrder].reverse()
    })).toBe(.8);
    expect(currentProductRetrievalScoreFromEvidence({
      ...evidence,
      metrics: [...backendMetricOrder.slice(0, 4), backendMetricOrder[0]]
    })).toBeNull();
  });

  it("recognizes only explicit current-product and historical fixture modes", () => {
    expect(normalizeEvaluationBenchmarkMode("current_product")).toBe("current_product");
    expect(normalizeEvaluationBenchmarkMode("historical_fixture")).toBe("historical_fixture");
    expect(normalizeEvaluationBenchmarkMode("approved_fixture")).toBe("not_selected");
    expect(normalizeEvaluationBenchmarkMode("future_mode")).toBe("not_selected");
    expect(normalizeEvaluationBenchmarkMode(undefined)).toBe("not_selected");
  });

  it.each([
    ["schema", { schemaVersion: "current-product-ragas-evidence.v2" }],
    ["diagnostic case scope", { scope: "cases" }],
    ["privacy class", { privacyClass: "public_official_evidence" }],
    ["immutable dataset digest", { datasetFingerprint: hash("9") }],
    ["run identifier", { runId: "" }],
    ["evaluated timestamp", { evaluatedAt: "not-a-timestamp" }],
    ["publication timestamp", { timestamp: null }]
  ])("rejects current-product evidence with invalid %s metadata", (_label, overrides) => {
    expect(currentProductRetrievalScoreFromEvidence(
      canonicalCurrentProductEvidence(overrides)
    )).toBeNull();
  });

  it("rejects non-finite, out-of-range, or aggregate-inconsistent scores", () => {
    const evidence = canonicalCurrentProductEvidence();

    expect(currentProductRetrievalScoreFromEvidence({
      ...evidence,
      metricScores: { ...evidence.metricScores, faithfulness: Number.NaN }
    })).toBeNull();
    expect(currentProductRetrievalScoreFromEvidence({
      ...evidence,
      caseRows: evidence.caseRows.map((row, index) => index === 0
        ? { ...row, metrics: { ...row.metrics, context_recall: 1.01 } }
        : row)
    })).toBeNull();
    expect(currentProductRetrievalScoreFromEvidence({
      ...evidence,
      retrievalScore: .81
    })).toBeNull();
  });

  it("uses retrieval and KB identity only to decide whether current-product trends compare", () => {
    const anchor = canonicalCurrentProductEvidence();
    const newer = canonicalCurrentProductEvidence({
      runId: "newer-current-product-run",
      evaluatedAt: "2026-07-14T10:00:00.000Z",
      timestamp: "2026-07-14T10:00:00.000Z"
    });

    expect(matchesCurrentProductEvidenceIdentity(newer, anchor)).toBe(true);
    expect(matchesCurrentProductEvidenceIdentity({
      ...newer,
      retrievalFingerprint: hash("1")
    }, anchor)).toBe(false);
    expect(matchesCurrentProductEvidenceIdentity({
      ...newer,
      kbFingerprint: hash("2")
    }, anchor)).toBe(false);

    const request = {
      scope: "rag" as const,
      caseIds: [],
      allowProviderCalls: true,
      approvedRagasDataset: "sanitized_public" as const
    };
    const anchorRun = buildEvaluationBenchmarkRun({
      model: emptyModel,
      now: new Date("2026-07-14T09:00:00.000Z"),
      ragas: anchor,
      request
    });
    const changedPipelineRun = buildEvaluationBenchmarkRun({
      model: emptyModel,
      now: new Date("2026-07-14T10:00:00.000Z"),
      previousRun: anchorRun,
      ragas: {
        ...newer,
        retrievalFingerprint: hash("1"),
        kbFingerprint: hash("2")
      },
      request
    });

    expect(changedPipelineRun.categoryResults[0]?.previousScore).toBeNull();
    expect(changedPipelineRun.categoryResults[0]?.delta).toBeNull();
  });

  it("keeps Full execution to seven RAG cases plus three explicit workspace snapshot lanes", () => {
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
    expect(run.selectedCaseIds).toEqual(CURRENT_PRODUCT_RETRIEVAL_GOLDEN_CASE_IDS);
    expect(run.selectedCases).toBe(7);
    expect(run.executedCases).toBe(0);
    expect(run.caseCoverage).toBe(0);
    expect(run.caseResults.map((item) => item.caseId)).toEqual([
      "ragas/not_selected",
      "system/current-workspace-creative",
      "system/current-workspace-workflow",
      "system/current-workspace-reliability"
    ]);
    expect(run.caseResults).toHaveLength(4);
    expect(run.caseResults.every((item) => item.status !== "not_run")).toBe(true);
    expect(run.counts.notRun).toBe(0);
    expect(run.counts.blocked + run.counts.missing).toBeGreaterThan(0);
    expect(run.caseResults.flatMap((item) => item.metrics).every((metric) =>
      metric.status !== "blocked" && metric.status !== "missing_evidence" || metric.score === null
    )).toBe(true);
  });

  it("reports a prepared current-product preflight as missing evidence rather than blocked", () => {
    const run = buildEvaluationBenchmarkRun({
      model: emptyModel,
      now: new Date("2026-07-14T08:00:00.000Z"),
      ragas: {
        ...emptyRagasEvidence(),
        state: "prepared",
        datasetId: "creative_coding_retrieval_benchmark",
        totalSamples: 7,
        skippedSamples: 7,
        detail: "Dry-run evidence only."
      },
      request: {
        scope: "rag",
        caseIds: [],
        allowProviderCalls: false,
        approvedRagasDataset: "sanitized_public"
      }
    });

    expect(run.caseResults).toHaveLength(1);
    expect(run.caseResults[0]?.status).toBe("missing_evidence");
    expect(run.caseResults[0]?.metrics.every((metric) => metric.status === "missing_evidence")).toBe(true);
    expect(run.counts.blocked).toBe(0);
    expect(run.counts.missing).toBe(1);
    expect(run.environmentStatus).toBe("partially_available");
  });

  it("records a completed Full run as seven executed RAG cases without catalog placeholders", () => {
    const evidence = canonicalCurrentProductEvidence({ scope: "full" });
    const run = buildEvaluationBenchmarkRun({
      model: emptyModel,
      now: new Date("2026-07-14T09:00:00.000Z"),
      ragas: evidence,
      request: {
        scope: "full",
        caseIds: [],
        allowProviderCalls: true,
        approvedRagasDataset: "sanitized_public"
      }
    });

    expect(run.selectedCases).toBe(7);
    expect(run.executedCases).toBe(7);
    expect(run.caseCoverage).toBe(1);
    expect(run.ragas.caseRows).toHaveLength(7);
    expect(run.caseResults).toHaveLength(4);
    expect(run.caseResults.every((item) => item.status !== "not_run")).toBe(true);
    expect(run.caseResults.some((item) => item.caseId.startsWith("demo/"))).toBe(false);
    expect(run.caseResults.some((item) => item.caseId.startsWith("prompt/"))).toBe(false);
    expect(run.counts.notRun).toBe(0);
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
        datasetId: "sanitized_public_variant",
        datasetVersion: "sanitized-ragas.v2"
      },
      request
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
