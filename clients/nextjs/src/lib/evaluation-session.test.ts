import { describe, expect, it } from "vitest";
import {
  buildEvaluationSessionModel,
  type EvaluationSessionSignal
} from "./evaluation-session";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

describe("evaluation session model", () => {
  it("summarizes the latest score, status, type, timestamp, and quality signals", () => {
    const model = buildEvaluationSessionModel(
      [
        traceEvent({
          at: "2026-05-24T10:00:00Z",
          evaluation: {
            dataset_id: "session-dataset",
            evaluated_at: "2026-05-24T10:00:04Z",
            evaluation_type: "RAGAs live",
            metric_scores: {
              answer_relevancy: 0.91,
              artifact_quality: 0.84,
              context_precision: 0.82,
              faithfulness: 0.74,
              runtime_quality: 0.67
            },
            result_rows: 3,
            run_id: "eval-session-7",
            status: "Evaluation complete"
          }
        })
      ],
      noObservability
    );

    expect(model).toMatchObject({
      state: "available",
      runId: "eval-session-7",
      datasetId: "session-dataset",
      evaluationType: "RAGAs live",
      latestAt: "2026-05-24T10:00:04Z",
      statusLabel: "Evaluation complete",
      outcome: "warn"
    });
    expect(model.score).toBeCloseTo(0.796);
    expect(signal(model.signals, "answer")).toMatchObject({
      score: 0.91,
      outcome: "pass"
    });
    expect(signal(model.signals, "retrieval")).toMatchObject({
      score: 0.82,
      outcome: "pass"
    });
    expect(signal(model.signals, "grounding")).toMatchObject({
      score: 0.74,
      outcome: "warn"
    });
    expect(signal(model.signals, "artifact")).toMatchObject({
      score: 0.84,
      outcome: "pass"
    });
    expect(signal(model.signals, "provider-runtime")).toMatchObject({
      score: 0.67,
      outcome: "warn"
    });
  });

  it("honors explicit fail status and percentage scores", () => {
    const model = buildEvaluationSessionModel(
      [
        traceEvent({
          at: "2026-05-24T10:00:00Z",
          evaluation: {
            overall_score: 42,
            outcome: "failed",
            scores: {
              answer_quality: 0.42
            },
            status: "Quality gate failed"
          }
        })
      ],
      noObservability
    );

    expect(model.score).toBe(0.42);
    expect(model.outcome).toBe("fail");
    expect(signal(model.signals, "answer").outcome).toBe("fail");
  });

  it("keeps legacy lineage events valid without inventing scores", () => {
    const model = buildEvaluationSessionModel(
      [
        traceEvent({
          at: "2026-05-24T10:00:00Z",
          code: "ragas_eval_completed",
          evaluation: {
            dataset_id: "dataset-legacy",
            dry_run: true,
            metric_failures: 0,
            metrics: ["context_precision"],
            provider_calls_allowed: false,
            result_rows: 2,
            run_id: "eval-legacy",
            status: "Evaluation manifest ready"
          }
        })
      ],
      noObservability
    );

    expect(model).toMatchObject({
      state: "available",
      evaluationType: "RAGAs",
      score: null,
      outcome: "unscored",
      runId: "eval-legacy"
    });
    expect(model.signals.every((item) => item.score == null)).toBe(true);
    expect(signal(model.signals, "retrieval")).toMatchObject({
      metrics: ["context_precision"],
      detail: "Context Precision recorded without a score."
    });
  });

  it("renders a clean empty or pending model when evaluation data is absent", () => {
    expect(buildEvaluationSessionModel([], noObservability)).toMatchObject({
      state: "unavailable",
      score: null,
      statusLabel: "No evaluation run"
    });
    expect(
      buildEvaluationSessionModel(
        [
          {
            event: {
              event_type: "status",
              payload: { emitted_at: "2026-05-24T10:00:00Z" },
              sequence: 0
            },
            receivedAt: "2026-05-24T10:00:00.050Z",
            receivedAtMs: Date.parse("2026-05-24T10:00:00.050Z")
          }
        ],
        noObservability
      )
    ).toMatchObject({
      state: "pending",
      statusLabel: "No evaluation event in stream"
    });
  });
});

const noObservability = {
  latestAt: null,
  traceKind: null
};

function signal(
  signals: EvaluationSessionSignal[],
  id: EvaluationSessionSignal["id"]
) {
  const result = signals.find((item) => item.id === id);
  if (!result) {
    throw new Error(`Missing ${id} signal.`);
  }
  return result;
}

function traceEvent({
  at,
  code,
  evaluation
}: {
  at: string;
  code?: string;
  evaluation: Record<string, unknown>;
}): WorkflowRuntimeTraceEvent {
  return {
    event: {
      event_type: "eval_update",
      payload: {
        ...(code ? { code } : {}),
        emitted_at: at,
        evaluation
      },
      sequence: 0
    },
    receivedAt: new Date(Date.parse(at) + 50).toISOString(),
    receivedAtMs: Date.parse(at) + 50
  };
}
