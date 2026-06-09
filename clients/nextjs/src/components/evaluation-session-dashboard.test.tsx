import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { EvaluationSessionModel } from "@/lib/evaluation-session";
import { EvaluationSessionDashboard } from "./evaluation-session-dashboard";

describe("EvaluationSessionDashboard", () => {
  it("renders the latest score, status, metadata, and quality signals", () => {
    render(<EvaluationSessionDashboard evaluation={scoredEvaluation()} />);

    const dashboard = screen.getByRole("group", {
      name: "Evaluation session dashboard"
    });
    expect(within(dashboard).getByText("Evaluation complete")).toBeVisible();
    expect(
      within(dashboard).getByLabelText("Evaluation status Pass")
    ).toBeVisible();
    expect(
      within(dashboard).getByRole("region", { name: "Latest evaluation score" })
    ).toHaveTextContent("84%");
    expect(within(dashboard).getByText("RAGAs live")).toBeVisible();
    expect(within(dashboard).getByText(/May 24, 2026/)).toBeVisible();

    const signals = within(dashboard).getByRole("region", {
      name: "Evaluation quality signals"
    });
    expect(
      within(signals).getByRole("group", {
        name: "Answer quality evaluation signal"
      })
    ).toHaveTextContent("91%");
    expect(
      within(signals).getByRole("group", {
        name: "Grounding quality evaluation signal"
      })
    ).toHaveTextContent("Warn");
    expect(
      within(signals).getByRole("group", {
        name: "Provider / runtime evaluation signal"
      })
    ).toHaveTextContent("67%");
  });

  it("renders a clean empty state for sessions without evaluation data", () => {
    render(
      <EvaluationSessionDashboard
        evaluation={{
          ...scoredEvaluation(),
          state: "unavailable",
          statusLabel: "No evaluation run",
          detail:
            "Session evaluation appears here when eval_update events are available.",
          evaluationType: "Not evaluated",
          latestAt: null,
          score: null,
          outcome: "unscored"
        }}
      />
    );

    expect(screen.getByText("No session evaluation results yet")).toBeVisible();
    expect(
      screen.getByText(/Answer, retrieval, grounding, artifact/)
    ).toBeVisible();
    expect(
      screen.queryByRole("region", { name: "Evaluation quality signals" })
    ).not.toBeInTheDocument();
  });

  it("renders legacy lineage without fabricating a score", () => {
    render(
      <EvaluationSessionDashboard
        evaluation={{
          ...scoredEvaluation(),
          score: null,
          outcome: "unscored",
          statusLabel: "Evaluation manifest ready",
          signals: scoredEvaluation().signals.map((signal) => ({
            ...signal,
            score: null,
            outcome: "unscored",
            metrics: [],
            detail: `No ${signal.label.toLowerCase()} metric in the latest evaluation.`
          }))
        }}
      />
    );

    const dashboard = screen.getByRole("group", {
      name: "Evaluation session dashboard"
    });
    expect(
      within(dashboard).getByLabelText("Evaluation status Unscored")
    ).toBeVisible();
    expect(within(dashboard).getByText("Score unavailable")).toBeVisible();
    expect(
      within(dashboard).getByRole("region", {
        name: "RAGAs evaluation lineage"
      })
    ).toHaveTextContent("Context Precision");
  });
});

function scoredEvaluation(): EvaluationSessionModel {
  return {
    state: "available",
    runId: "eval-run-1",
    datasetId: "dataset-live-1",
    metrics: [
      "answer_relevancy",
      "context_precision",
      "faithfulness",
      "artifact_quality",
      "runtime_quality"
    ],
    resultRows: 2,
    metricFailures: 0,
    dryRun: false,
    providerCallsAllowed: true,
    statusLabel: "Evaluation complete",
    detail: "Latest session quality run completed.",
    latestAt: "2026-05-24T10:00:00Z",
    evaluationType: "RAGAs live",
    score: 0.84,
    outcome: "pass",
    signals: [
      signal("answer", "Answer quality", 0.91, "pass", "answer_relevancy"),
      signal("retrieval", "Retrieval quality", 0.82, "pass", "context_precision"),
      signal("grounding", "Grounding quality", 0.74, "warn", "faithfulness"),
      signal("artifact", "Artifact quality", 0.84, "pass", "artifact_quality"),
      signal(
        "provider-runtime",
        "Provider / runtime",
        0.67,
        "warn",
        "runtime_quality"
      )
    ]
  };
}

function signal(
  id: EvaluationSessionModel["signals"][number]["id"],
  label: string,
  score: number,
  outcome: EvaluationSessionModel["signals"][number]["outcome"],
  metric: string
): EvaluationSessionModel["signals"][number] {
  return {
    id,
    label,
    score,
    outcome,
    metrics: [metric],
    detail: metric.replace(/_/g, " ")
  };
}
