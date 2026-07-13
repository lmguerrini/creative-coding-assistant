import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { buildEvaluationBenchmarkRun, emptyRagasEvidence } from "@/lib/evaluation-benchmark";
import type { EvaluationHistoryRecord } from "@/lib/product-controls";
import type { ProductIntelligenceModel } from "@/lib/product-intelligence";
import { CapstoneEvaluationWorkspace } from "./capstone-evaluation-workspace";

const model = { artifactRegistry: [], details: null } as unknown as ProductIntelligenceModel;

function retrievalHistory(): EvaluationHistoryRecord[] {
  const benchmark = buildEvaluationBenchmarkRun({
    model,
    now: new Date("2026-07-13T12:00:00.000Z"),
    ragas: {
      ...emptyRagasEvidence(),
      state: "completed",
      datasetId: "sanitized_public",
      privacyClass: "committed_sanitized_public",
      metrics: ["faithfulness", "answer_relevancy", "context_precision", "context_relevancy"],
      metricScores: { faithfulness: .9, answer_relevancy: .8, context_precision: .7, context_relevancy: .8 },
      resultRows: 4,
      totalSamples: 4,
      eligibleSamples: 4,
      provider: "OpenAI",
      detail: "Real RAGAS evaluation completed on the approved fixture."
    },
    request: { scope: "rag", caseIds: [], allowProviderCalls: true, approvedRagasDataset: "sanitized_public" }
  });
  return [{
    benchmark,
    datasetId: "sanitized_public",
    detail: "Completed",
    dryRun: false,
    evaluatedAt: benchmark.completedAt,
    id: "history-1",
    metricFailures: 0,
    metrics: ["faithfulness", "answer_relevancy", "context_precision", "context_relevancy"],
    providerCallsAllowed: true,
    resultRows: 4,
    runId: benchmark.id,
    status: "completed"
  }];
}

describe("Capstone Evaluation workspace", () => {
  it("presents four separate systems and a provider-safe local preflight", async () => {
    const onRun = vi.fn().mockResolvedValue(undefined);
    render(<CapstoneEvaluationWorkspace history={[]} model={model} onRun={onRun} running={false} />);

    expect(screen.getByText("AI Engineering Lab")).toBeVisible();
    expect(screen.getByRole("heading", { name: "Measure retrieval. Diagnose weaknesses. Improve the real system." })).toBeVisible();
    expect(screen.getByText(/unique cases/)).toHaveTextContent("35 unique cases");
    expect(screen.getAllByText("RAG / Retrieval").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Creative Artifacts").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Agents & Workflow").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Product Reliability").length).toBeGreaterThan(0);
    expect(screen.queryByText(/overall quality score/i)).not.toBeInTheDocument();
    expect(screen.queryByText("Overall product quality")).not.toBeInTheDocument();
    const boundaries = screen.getByLabelText("Evaluation score boundaries");
    expect(boundaries).toHaveTextContent("Six independent signals. No global score.");
    for (const label of ["Retrieval Quality", "Creative Quality", "Workflow Quality", "Product Reliability", "Benchmark Coverage", "Evidence Coverage"]) {
      expect(within(boundaries).getByText(label)).toBeVisible();
    }
    const categoryCards = screen.getByRole("region", { name: "Evaluation categories" });
    expect(within(categoryCards).getAllByText("Not measured")).toHaveLength(4);
    expect(within(categoryCards).queryByText(/target 80%/i)).not.toBeInTheDocument();
    expect(within(categoryCards).queryByLabelText(/measured score/i)).not.toBeInTheDocument();
    expect(boundaries).toHaveTextContent(/coverage is not quality/i);
    expect(screen.getByText("Approved-fixture Overall Retrieval Score")).toBeVisible();
    for (const label of ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall", "Context Relevancy"]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
    expect(screen.getByText(/not an overall product or project score/i)).toBeVisible();
    expect(screen.getByText(/latest scored approved baseline/i)).toBeVisible();
    const baselineOverall = screen.getByText("Approved-fixture Overall Retrieval Score").closest("article");
    expect(within(baselineOverall as HTMLElement).getByText("61.44%")).toBeVisible();
    expect(screen.getByText("Benchmark coverage").closest("article")).toHaveTextContent("7/7 retrieval queries7/7 canonical retrieval-pack queries");
    expect(screen.getByText("Evidence coverage").closest("article")).toHaveTextContent("80%4/5 requested RAG metric dimensions measured");
    const evolution = screen.getByLabelText("Retrieval engineering evolution");
    expect(evolution).toHaveTextContent("Current verified");
    expect(evolution).toHaveTextContent("9/23");
    expect(evolution).toHaveTextContent("17/23");
    expect(evolution).toHaveTextContent("19/23");
    expect(evolution).toHaveTextContent("15/23");
    expect(evolution).toHaveTextContent("16/23");
    expect(evolution).toHaveTextContent("7/19");
    expect(evolution).toHaveTextContent("18/19");
    expect(evolution).toHaveTextContent("BLOCKED_BY_EXECUTION_ENVIRONMENT");
    expect(evolution).toHaveTextContent("canonical_retrieval_report.json");
    expect(evolution).toHaveTextContent("1,445 chunks");
    expect(evolution).toHaveTextContent("b64323bf1424");
    expect(evolution).toHaveTextContent("74acf5d62f66");
    expect(evolution).toHaveTextContent(/not answer quality, evidence completeness, or an overall product score/i);
    expect(screen.getByText("Evidence lineage")).toBeVisible();
    expect(screen.getByLabelText("Retrieval evaluation execution timeline")).toHaveTextContent("MISSING_EVIDENCE");
    expect(screen.getByLabelText("Retrieval evaluation execution timeline")).toHaveTextContent("19/23 RAW ANCHORS → 15/23 SUBSTANTIVE → 16/23 FINAL");
    const scoreContract = screen.getByText("Retrieval score contract").closest("details");
    expect(scoreContract).not.toBeNull();
    expect(within(scoreContract as HTMLElement).getByLabelText("Included retrieval metrics")).toHaveTextContent("Equal weight inside Retrieval Quality only");
    expect(within(scoreContract as HTMLElement).getByText("MISSING_EVIDENCE")).toBeVisible();
    expect(within(scoreContract as HTMLElement).getByText("BLOCKED_BY_EXECUTION_ENVIRONMENT")).toBeVisible();
    expect(within(scoreContract as HTMLElement).getAllByText("NOT_COMPARABLE")).toHaveLength(2);
    expect(within(scoreContract as HTMLElement).getByText("SUBJECTIVE")).toBeVisible();
    expect(scoreContract).toHaveTextContent(/global score is not calculated/i);
    const diagnostics = screen.getByLabelText("Metric engineering diagnostics");
    for (const metricName of ["Faithfulness", "Answer Relevancy", "Context Relevancy"]) {
      const weakMetric = within(diagnostics).getByText(metricName).closest("details");
      expect(weakMetric).not.toBeNull();
      for (const field of ["Current approved score", "Target", "Root cause", "Product improvement", "Comparable benchmark delta", "Remaining limitation", "Recommended next engineering step"]) {
        expect(within(weakMetric as HTMLElement).getByText(field)).toBeInTheDocument();
      }
      expect(weakMetric).toHaveTextContent("At least 80%");
      expect(weakMetric).toHaveTextContent("NOT_COMPARABLE");
    }
    const recallDiagnostic = within(diagnostics).getByText("Context Recall").closest("details");
    expect(recallDiagnostic).toHaveTextContent("MISSING_EVIDENCE");
    expect(recallDiagnostic).toHaveTextContent("Recommended next engineering step");

    fireEvent.click(screen.getByRole("button", { name: "Run Evaluation" }));
    expect(screen.getByLabelText("Evaluation preflight")).toHaveTextContent(/no provider call/i);
    await waitFor(() => expect(onRun).toHaveBeenCalledWith({
      scope: "full",
      caseIds: [],
      allowProviderCalls: false,
      approvedRagasDataset: "sanitized_public"
    }));
    expect(screen.getByLabelText("Live evaluation progress")).toHaveTextContent(
      "Current workspace snapshot"
    );
    const authorization = screen.getByRole("checkbox", { name: /explicitly authorize evaluator provider calls/i });
    expect(authorization).not.toBeChecked();
    expect(screen.getByLabelText("Approved RAGAS dataset")).toBeDisabled();

    onRun.mockClear();
    fireEvent.click(screen.getByRole("button", { name: "Run 35 cases" }));
    await waitFor(() => expect(onRun).toHaveBeenCalledWith({
      scope: "full",
      caseIds: [],
      allowProviderCalls: false,
      approvedRagasDataset: "sanitized_public"
    }));
    const progress = screen.getByLabelText("Live evaluation progress");
    expect(progress).toHaveTextContent("Estimated progress: indeterminate");
    expect(progress).toHaveTextContent("golden_eval.v1 · 35 contracts enumerated");
    expect(progress).toHaveTextContent("Current workspace snapshot");
    expect(progress).toHaveTextContent("0 confirmed / 1 unresolved snapshot");
    expect(within(progress).getByRole("progressbar")).not.toHaveAttribute("aria-valuenow");
  });

  it("requires explicit authorization before sending an approved provider request", async () => {
    const onRun = vi.fn(() => new Promise<void>(() => undefined));
    render(<CapstoneEvaluationWorkspace history={[]} model={model} onRun={onRun} running={false} />);

    fireEvent.click(screen.getByRole("button", { name: "Configure run" }));
    fireEvent.click(screen.getByRole("checkbox", { name: /explicitly authorize evaluator provider calls/i }));
    fireEvent.change(screen.getByLabelText("Approved RAGAS dataset"), { target: { value: "redacted_public" } });
    fireEvent.click(screen.getByRole("button", { name: "Run 35 cases" }));

    await waitFor(() => expect(onRun).toHaveBeenCalledWith(expect.objectContaining({
      allowProviderCalls: true,
      approvedRagasDataset: "redacted_public"
    })));
    expect(screen.getByLabelText("Live evaluation progress")).toHaveTextContent("Authorized provider batch in progress");
    expect(screen.getByLabelText("Live evaluation progress")).toHaveTextContent("0 confirmed / 4 unresolved fixture rows");
  });

  it("makes the approved-fixture RAGAS macro prominent without treating missing evidence as zero", () => {
    render(<CapstoneEvaluationWorkspace history={retrievalHistory()} model={model} onRun={vi.fn()} running={false} />);

    const retrieval = screen.getByLabelText("RAGAS retrieval evaluation");
    const overall = within(retrieval).getByText("Approved-fixture Overall Retrieval Score").closest("article");
    expect(overall).not.toBeNull();
    expect(within(overall as HTMLElement).getByText("80.00%")).toBeVisible();
    expect(within(retrieval).getByText("4 evaluated rows from 4/4 eligible fixture samples", { exact: false })).toBeVisible();

    const metricGrid = within(retrieval).getByLabelText("RAGAS metric scores");
    const recall = within(metricGrid).getByText("Context Recall").closest("article");
    const relevancy = within(metricGrid).getByText("Context Relevancy").closest("article");
    expect(within(recall as HTMLElement).getByText("MISSING EVIDENCE")).toBeVisible();
    expect(within(relevancy as HTMLElement).getByText("80.00%")).toBeVisible();
    const reliability = within(retrieval).getByText("Product Reliability").closest("article");
    expect(reliability).not.toBeNull();
    expect(reliability).toHaveTextContent("Not measured");
  });
});
