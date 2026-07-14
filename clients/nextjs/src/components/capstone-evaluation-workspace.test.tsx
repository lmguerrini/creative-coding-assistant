import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  buildEvaluationBenchmarkRun,
  CURRENT_PRODUCT_RETRIEVAL_CASE_IDS,
  CURRENT_PRODUCT_RETRIEVAL_DATASET_FINGERPRINT,
  emptyRagasEvidence,
  type EvaluationBenchmarkRun,
  type EvaluationExecutionProgress,
  type RagasExecutionEvidence
} from "@/lib/evaluation-benchmark";
import type { EvaluationHistoryRecord } from "@/lib/product-controls";
import type { ProductIntelligenceModel } from "@/lib/product-intelligence";
import { CapstoneEvaluationWorkspace } from "./capstone-evaluation-workspace";

const model = { artifactRegistry: [], details: null } as unknown as ProductIntelligenceModel;

const fiveMetricScores = {
  context_precision: .9,
  faithfulness: .9,
  answer_relevancy: .9,
  context_relevancy: .9,
  context_recall: .9
};
const backendMetricOrder = [
  "context_precision",
  "faithfulness",
  "answer_relevancy",
  "context_relevancy",
  "context_recall"
];
const hash = (character: string) => `sha256:${character.repeat(64)}`;

function currentProductEvidence({
  metricScores = fiveMetricScores,
  runId = "current-product-run-1",
  sampleCount = 7,
  state = "completed",
  timestamp = "2026-07-13T12:00:00.000Z"
}: {
  metricScores?: Record<string, number>;
  runId?: string;
  sampleCount?: number;
  state?: RagasExecutionEvidence["state"];
  timestamp?: string;
} = {}): RagasExecutionEvidence {
  const selectedCaseIds = CURRENT_PRODUCT_RETRIEVAL_CASE_IDS.slice(0, sampleCount);
  const retrievalScore = Object.values(metricScores).reduce((sum, score) => sum + score, 0) /
    Object.values(metricScores).length;
  return {
    ...emptyRagasEvidence(),
    schemaVersion: "current-product-ragas-evidence.v1",
    scope: "rag",
    state,
    runId,
    evaluatedAt: timestamp,
    datasetId: "capstone_kb_expansion_retrieval_demo_pack",
    datasetVersion: "current-product-retrieval.v1",
    privacyClass: "public_official_contexts_with_authored_references",
    metrics: backendMetricOrder,
    metricScores,
    retrievalScore,
    resultRows: state === "completed" ? sampleCount : 0,
    totalSamples: sampleCount,
    eligibleSamples: sampleCount,
    skippedSamples: 0,
    metricFailures: 0,
    provider: "OpenAI",
    model: "gpt-5-mini",
    embeddingModel: "text-embedding-3-small",
    ragasVersion: "0.3.9",
    metricContract: "ragas-current-product-reference.v2",
    durationMs: 1_200,
    detail: state === "completed"
      ? "Current-product evaluation completed."
      : "Current-product evaluation stopped before scores were published.",
    caseRows: state === "completed" ? selectedCaseIds.map((sampleId, index) => ({
      sampleId,
      metrics: { ...metricScores },
      metricErrors: {},
      sourceIds: [`official-source-${index}`],
      domains: [`domain-${index}`],
      promptFingerprint: hash(String((index + 1) % 10)),
      generationFingerprint: hash(String((index + 2) % 10))
    })) : [],
    benchmarkMode: "current_product",
    scoreOrigin: "current_product",
    benchmarkVersion: "current-product-retrieval.v1",
    selectedCaseIds: [...selectedCaseIds],
    datasetFingerprint: CURRENT_PRODUCT_RETRIEVAL_DATASET_FINGERPRINT,
    retrievalFingerprint: hash("a"),
    promptFingerprint: hash("b"),
    generationFingerprint: hash("c"),
    outputFingerprint: hash("d"),
    selectionFingerprint: hash("e"),
    kbFingerprint: hash("f"),
    generationModel: "gpt-5-mini",
    evaluator: "OpenAI / gpt-5-mini",
    evaluatorModel: "gpt-5-mini",
    timestamp
  };
}

function benchmarkRun(
  evidence: RagasExecutionEvidence,
  previousRun: EvaluationBenchmarkRun | null = null
): EvaluationBenchmarkRun {
  return buildEvaluationBenchmarkRun({
    model,
    now: new Date(evidence.timestamp ?? evidence.evaluatedAt ?? "2026-07-13T12:00:00.000Z"),
    previousRun,
    ragas: evidence,
    request: {
      scope: "rag",
      caseIds: [],
      allowProviderCalls: true,
      approvedRagasDataset: "sanitized_public"
    }
  });
}

function historyRecord(run: EvaluationBenchmarkRun): EvaluationHistoryRecord {
  return {
    benchmark: run,
    datasetId: run.ragas.datasetId,
    detail: run.ragas.detail,
    dryRun: false,
    evaluatedAt: run.timestamp,
    id: `history-${run.runId}`,
    metricFailures: run.ragas.metricFailures,
    metrics: run.ragas.metrics,
    providerCallsAllowed: true,
    resultRows: run.ragas.resultRows,
    runId: run.runId,
    status: run.ragas.state
  };
}

describe("Capstone Evaluation workspace", () => {
  it("keeps the primary score current-product-only and the historical 61.44% fixture inside History", () => {
    render(<CapstoneEvaluationWorkspace currentProductEvidence={null} history={[]} model={model} onRun={vi.fn()} running={false} />);

    expect(screen.getByText("AI Engineering Lab")).toBeVisible();
    expect(screen.getByRole("heading", { name: "Measure retrieval. Diagnose weaknesses. Improve the real system." })).toBeVisible();
    expect(screen.getByText(/unique cases/)).toHaveTextContent("35 unique cases");
    expect(screen.getAllByRole("button", { name: "Run Evaluation" })).toHaveLength(1);
    expect(screen.queryByRole("button", { name: /Run \d+ cases/ })).not.toBeInTheDocument();
    expect(screen.getByLabelText("Evaluation benchmark summary")).toHaveTextContent(
      "Frozen contract coverage only; Full does not generate all 35 prompts."
    );

    const retrieval = screen.getByLabelText("RAGAS retrieval evaluation");
    const overall = within(retrieval).getByText("Current-product Overall Retrieval Score").closest("article");
    expect(overall).not.toBeNull();
    expect(within(overall as HTMLElement).getByText("—")).toBeVisible();
    expect(retrieval).toHaveTextContent("No partial metric set is promoted to the primary score");
    expect(screen.getByLabelText("Evaluation benchmark summary")).toHaveTextContent("target 85% · stretch 90%");
    expect(screen.queryByText("Approved-fixture Overall Retrieval Score")).not.toBeInTheDocument();

    const historySummary = screen.getByText("Comparable stored runs", { selector: "summary" });
    const historyDisclosure = historySummary.closest("details");
    expect(historyDisclosure).not.toBeNull();
    expect(within(historyDisclosure as HTMLElement).getByText("61.44%")).not.toBeVisible();
    fireEvent.click(historySummary);
    expect(within(historyDisclosure as HTMLElement).getByText("61.44%")).toBeVisible();
    expect(historyDisclosure).toHaveTextContent("Historical approved fixture · not current product");
    expect(historyDisclosure).toHaveTextContent("four-metric limitation");
  });

  it("uses the single above-fold action and renders backend progress verbatim", async () => {
    const reportedProgress: EvaluationExecutionProgress = {
      runId: "evaluation-run-async-1",
      status: "running",
      phase: "ragas_scoring",
      lane: "RAG / Retrieval",
      currentCaseId: "runtime_selection_hydra_vs_p5",
      currentCaseLabel: "Runtime selection for fast live visuals",
      completedCases: 3,
      totalCases: 7,
      remainingCases: 4,
      percent: 43,
      executionState: "local_preflight",
      detail: "Scoring current retrieved contexts against the reference answer."
    };
    const onRun = vi.fn(async (_request, onProgress) => {
      onProgress(reportedProgress);
    });
    render(<CapstoneEvaluationWorkspace currentProductEvidence={null} history={[]} model={model} onRun={onRun} running={false} />);

    fireEvent.click(screen.getByRole("button", { name: "Run Evaluation" }));

    await waitFor(() => expect(onRun).toHaveBeenCalledWith({
      scope: "full",
      caseIds: [],
      allowProviderCalls: false,
      approvedRagasDataset: "sanitized_public"
    }, expect.any(Function)));
    const progress = screen.getByLabelText("Live evaluation progress");
    expect(progress).toHaveTextContent("43% complete");
    expect(progress).toHaveTextContent("golden_eval.v1 · evaluation-run-async-1");
    expect(progress).toHaveTextContent("RAG / Retrieval");
    expect(progress).toHaveTextContent("Runtime selection for fast live visuals · runtime_selection_hydra_vs_p5");
    expect(progress).toHaveTextContent("3 completed / 4 remaining of 7");
    expect(progress).toHaveTextContent("ragas scoring · running");
    expect(progress).toHaveTextContent("local preflight");
    expect(within(progress).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "43");
  });

  it("configures provider consent without adding a second submit action", async () => {
    const onRun = vi.fn().mockResolvedValue(undefined);
    render(<CapstoneEvaluationWorkspace currentProductEvidence={null} history={[]} model={model} onRun={onRun} running={false} />);

    fireEvent.click(screen.getByRole("button", { name: "Configure run" }));
    const authorization = screen.getByRole("checkbox", { name: /current-product public benchmark/i });
    expect(authorization).not.toBeChecked();
    expect(screen.queryByLabelText("Current-product benchmark evidence policy")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Evaluation preflight")).toHaveTextContent("Local preflight / workspace snapshot");
    expect(screen.getByLabelText("Evaluation preflight")).toHaveTextContent("Zero retrieval, generation, or evaluator provider calls; no new Retrieval Quality score");
    expect(screen.getByLabelText("Evaluation preflight")).toHaveTextContent("7 canonical RAG cases + workspace snapshots");
    expect(screen.getByLabelText("Evaluation preflight")).toHaveTextContent("local creative, workflow, and reliability evidence is snapshotted separately");

    fireEvent.click(authorization);
    expect(screen.getAllByRole("button", { name: "Run Evaluation" })).toHaveLength(1);
    expect(screen.getByLabelText("Evaluation preflight")).toHaveTextContent("Public official KB excerpts only; explicitly authorized");
    fireEvent.click(screen.getByRole("button", { name: "Run Evaluation" }));

    await waitFor(() => expect(onRun).toHaveBeenCalledWith(expect.objectContaining({
      allowProviderCalls: true,
      approvedRagasDataset: "sanitized_public"
    }), expect.any(Function)));
  });

  it("withholds a five-metric run whose current-product provenance is incomplete", () => {
    const evidence = currentProductEvidence({ runId: "missing-provenance-run" });
    const run = benchmarkRun({ ...evidence, generationFingerprint: null });

    render(
      <CapstoneEvaluationWorkspace
        currentProductEvidence={null}
        history={[historyRecord(run)]}
        model={model}
        onRun={vi.fn()}
        running={false}
      />
    );

    const retrieval = screen.getByLabelText("RAGAS retrieval evaluation");
    const overall = within(retrieval).getByText("Current-product Overall Retrieval Score").closest("article");
    expect(within(overall as HTMLElement).getByText("—")).toBeVisible();
    expect(screen.queryByLabelText("Current Retrieval Quality provenance")).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("Comparable stored runs", { selector: "summary" }));
    expect(screen.getByLabelText("Evaluation history and trends")).toHaveTextContent("missing-provenance-run");
  });

  it("keeps a fully scored diagnostic subset in History instead of promoting it", () => {
    const subset = benchmarkRun(currentProductEvidence({
      runId: "diagnostic-subset-run",
      sampleCount: 2
    }));

    render(
      <CapstoneEvaluationWorkspace
        currentProductEvidence={null}
        history={[historyRecord(subset)]}
        model={model}
        onRun={vi.fn()}
        running={false}
      />
    );

    const retrieval = screen.getByLabelText("RAGAS retrieval evaluation");
    const overall = within(retrieval).getByText("Current-product Overall Retrieval Score").closest("article");
    expect(within(overall as HTMLElement).getByText("—")).toBeVisible();
    expect(screen.queryByLabelText("Current Retrieval Quality provenance")).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("Comparable stored runs", { selector: "summary" }));
    const history = screen.getByLabelText("Evaluation history and trends");
    expect(history).toHaveTextContent("diagnostic-subset-run");
    expect(history).toHaveTextContent("PRIOR CURRENT-PRODUCT RUN");
  });

  it("promotes only a complete five-metric current-product run, shows provenance, and clears stale history selection", async () => {
    const anchor = currentProductEvidence({
      runId: "committed-current-product-anchor",
      timestamp: "2026-07-13T11:30:00.000Z"
    });
    const partial = benchmarkRun(currentProductEvidence({
      metricScores: {
        faithfulness: .99,
        answer_relevancy: .99,
        context_precision: .99,
        context_relevancy: .99
      },
      runId: "partial-run",
      timestamp: "2026-07-13T11:00:00.000Z"
    }));
    const completed = benchmarkRun(currentProductEvidence(), partial);
    const blocked = benchmarkRun(currentProductEvidence({
      runId: "blocked-run",
      state: "blocked",
      timestamp: "2026-07-13T13:00:00.000Z"
    }), completed);

    const { rerender } = render(
      <CapstoneEvaluationWorkspace
        currentProductEvidence={anchor}
        history={[historyRecord(partial), historyRecord(completed), historyRecord(blocked)]}
        model={model}
        onRun={vi.fn()}
        running={false}
      />
    );

    const retrieval = screen.getByLabelText("RAGAS retrieval evaluation");
    const overall = within(retrieval).getByText("Current-product Overall Retrieval Score").closest("article");
    expect(within(overall as HTMLElement).getByText("90.00%")).toBeVisible();
    expect(retrieval).toHaveTextContent("5/5 justified RAGAS dimensions measured");

    const provenance = screen.getByLabelText("Current Retrieval Quality provenance");
    expect(provenance).toHaveTextContent("current product");
    expect(provenance).toHaveTextContent("current-product-retrieval.v1");
    expect(provenance).toHaveTextContent("b5fbc0e7cc9a");
    expect(provenance).toHaveTextContent("aaaaaaaaaaaa");
    expect(provenance).toHaveTextContent("bbbbbbbbbbbb");
    expect(provenance).toHaveTextContent("cccccccccccc");
    expect(provenance).toHaveTextContent("gpt-5-mini");
    expect(provenance).toHaveTextContent("OpenAI / gpt-5-mini");
    expect(provenance).toHaveTextContent("text-embedding-3-small");
    expect(provenance).toHaveTextContent("current-product-run-1");

    fireEvent.click(screen.getByText("Comparable stored runs", { selector: "summary" }));
    expect(screen.getByLabelText("Evaluation history and trends")).toHaveTextContent("blocked-run");
    expect(screen.getByLabelText("Evaluation history and trends")).toHaveTextContent("current-product-run-1");
    expect(screen.getByLabelText("Evaluation history and trends")).toHaveTextContent("partial-run");

    fireEvent.click(screen.getByRole("button", { name: /partial-run/ }));
    expect(screen.getByText("Historical run selected")).toBeVisible();
    const failed = benchmarkRun(currentProductEvidence({
      runId: "new-failed-current-run",
      state: "failed",
      timestamp: "2026-07-13T14:00:00.000Z"
    }), completed);
    rerender(
      <CapstoneEvaluationWorkspace
        currentProductEvidence={anchor}
        history={[historyRecord(partial), historyRecord(completed), historyRecord(blocked), historyRecord(failed)]}
        model={model}
        onRun={vi.fn()}
        running={false}
      />
    );
    await waitFor(() => expect(screen.queryByText("Historical run selected")).not.toBeInTheDocument());
  });

  it("promotes the newest fully validated run when retrieval and KB fingerprints legitimately change", () => {
    const anchor = currentProductEvidence({
      runId: "committed-anchor",
      timestamp: "2026-07-13T10:00:00.000Z"
    });
    const newerScores = Object.fromEntries(backendMetricOrder.map((metricId) => [metricId, .95]));
    const staleRetrieval = benchmarkRun({
      ...currentProductEvidence({
        metricScores: newerScores,
        runId: "stale-retrieval-run",
        timestamp: "2026-07-13T11:00:00.000Z"
      }),
      retrievalFingerprint: hash("1")
    });
    const staleKb = benchmarkRun({
      ...currentProductEvidence({
        metricScores: newerScores,
        runId: "stale-kb-run",
        timestamp: "2026-07-13T12:00:00.000Z"
      }),
      retrievalFingerprint: hash("2"),
      kbFingerprint: hash("2")
    });

    render(
      <CapstoneEvaluationWorkspace
        currentProductEvidence={anchor}
        history={[historyRecord(staleRetrieval), historyRecord(staleKb)]}
        model={model}
        onRun={vi.fn()}
        running={false}
      />
    );

    const retrieval = screen.getByLabelText("RAGAS retrieval evaluation");
    const overall = within(retrieval).getByText("Current-product Overall Retrieval Score").closest("article");
    expect(within(overall as HTMLElement).getByText("95.00%")).toBeVisible();
    expect(within(overall as HTMLElement).queryByText("90.00%")).not.toBeInTheDocument();
    const provenance = screen.getByLabelText("Current Retrieval Quality provenance");
    expect(provenance).toHaveTextContent("stale-kb-run");
    expect(provenance).toHaveTextContent("222222222222");

    fireEvent.click(screen.getByText("Comparable stored runs", { selector: "summary" }));
    const history = screen.getByLabelText("Evaluation history and trends");
    expect(history).toHaveTextContent("stale-retrieval-run");
    expect(history).toHaveTextContent("stale-kb-run");
    expect(history).toHaveTextContent("PRIOR CURRENT-PRODUCT RUN");
    expect(history).toHaveTextContent("CURRENT PRODUCT");
  });

  it("promotes a fully validated persisted run without a static evidence anchor", () => {
    const persisted = benchmarkRun(currentProductEvidence({
      runId: "persisted-without-anchor",
      timestamp: "2026-07-13T15:00:00.000Z"
    }));

    render(
      <CapstoneEvaluationWorkspace
        currentProductEvidence={null}
        history={[historyRecord(persisted)]}
        model={model}
        onRun={vi.fn()}
        running={false}
      />
    );

    const retrieval = screen.getByLabelText("RAGAS retrieval evaluation");
    const overall = within(retrieval).getByText("Current-product Overall Retrieval Score").closest("article");
    expect(within(overall as HTMLElement).getByText("90.00%")).toBeVisible();
    expect(screen.getByLabelText("Current Retrieval Quality provenance")).toHaveTextContent(
      "persisted-without-anchor"
    );
  });
});
