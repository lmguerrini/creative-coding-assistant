import { describe, expect, it } from "vitest";
import {
  getLocalWorkspaceSnapshot,
  type AssistantWorkspaceSnapshot
} from "./assistant-client";
import { buildProviderTelemetryModel } from "./provider-telemetry";
import { buildRetrievalRuntimeModel } from "./retrieval-runtime";
import { buildTelemetryDashboardModel } from "./telemetry-dashboard";
import { buildWorkflowRuntimeModel } from "./workflow-runtime";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

describe("telemetry dashboard model", () => {
  it("aggregates runtime, provider, retrieval, observability, and eval lineage", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const traceEvents = [
      traceEvent({
        at: "2026-05-24T10:00:00Z",
        code: "request_received",
        event_type: "status",
        observability: {
          provider: "langsmith",
          trace_kind: "assistant_workflow",
          trace_id: "trace-langsmith-123456",
          requested: true,
          enabled: true,
          project_name: "creative-prod",
          status: "enabled",
          tags: ["assistant", "workflow"]
        },
        sequence: 0
      }),
      traceEvent({
        at: "2026-05-24T10:00:01Z",
        event_type: "token_delta",
        sequence: 1,
        telemetry: {
          provider: {
            name: "openai",
            model: "gpt-5-mini"
          }
        },
        text: "Draft "
      }),
      traceEvent({
        at: "2026-05-24T10:00:02Z",
        code: "ragas_eval_completed",
        event_type: "eval_update",
        evaluation: {
          dataset_id: "dataset-1",
          dry_run: true,
          metric_failures: 0,
          metric_scores: {
            answer_relevancy: 0.9,
            context_precision: 0.84,
            faithfulness: 0.79
          },
          provider_calls_allowed: false,
          result_rows: 2,
          run_id: "eval-run-1",
          status: "Evaluation manifest ready"
        },
        sequence: 2
      }),
      traceEvent({
        answer: "Done.",
        at: "2026-05-24T10:00:03Z",
        event_type: "final",
        sequence: 3,
        telemetry: {
          provider: {
            name: "openai",
            model: "gpt-5-mini",
            response_id: "resp_123"
          },
          token_usage: {
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: 150
          }
        }
      })
    ];

    const model = buildDashboard(snapshot, traceEvents);

    expect(model.status).toBe("complete");
    expect(model.stream).toMatchObject({
      eventCount: 4,
      evalEventCount: 1
    });
    expect(model.provider.tokenUsage.totalTokens).toBe(150);
    expect(model.observability).toMatchObject({
      state: "linked",
      traceId: "trace-langsmith-123456",
      projectName: "creative-prod"
    });
    expect(model.langsmithTrace).toMatchObject({
      state: "linked",
      status: "complete",
      traceId: "trace-langsmith-123456",
      projectName: "creative-prod"
    });
    expect(model.evaluation).toMatchObject({
      state: "available",
      runId: "eval-run-1",
      datasetId: "dataset-1",
      resultRows: 2,
      metricFailures: 0,
      evaluationType: "RAGAs",
      outcome: "pass"
    });
    expect(model.evaluation.score).toBeCloseTo(0.843);
    expect(model.evaluation.signals.find((signal) => signal.id === "answer")).toMatchObject({
      score: 0.9,
      outcome: "pass"
    });
    expect(model.signals.map((signal) => signal.label)).toContain("LangSmith");
    expect(model.summary.coverageLabel).toMatch(/telemetry domains populated/);
  });

  it("degrades gracefully when optional telemetry is missing", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const model = buildDashboard(snapshot, []);

    expect(model.status).toBe("idle");
    expect(model.stream.latestEventLabel).toBe("No stream events");
    expect(model.observability).toMatchObject({
      state: "unavailable",
      traceId: null
    });
    expect(model.langsmithTrace).toMatchObject({
      state: "unavailable",
      traceId: null,
      spans: []
    });
    expect(model.evaluation).toMatchObject({
      state: "unavailable",
      statusLabel: "No evaluation run",
      score: null,
      outcome: "unscored"
    });
    expect(model.provider.summary.costLabel).toBe("Cost pending");
  });
});

function buildDashboard(
  snapshot: AssistantWorkspaceSnapshot,
  traceEvents: WorkflowRuntimeTraceEvent[]
) {
  const workflowRuntime = buildWorkflowRuntimeModel(snapshot.workflow, traceEvents);
  const providerTelemetry = buildProviderTelemetryModel(traceEvents);
  const retrievalRuntime = buildRetrievalRuntimeModel(
    snapshot.retrieval,
    traceEvents
  );

  return buildTelemetryDashboardModel({
    activeArtifact: snapshot.artifacts[0],
    creativeCostHistory: [],
    draftPrompt: "",
    providerTelemetry,
    retrievalRuntime,
    snapshot,
    traceEvents,
    workflowRuntime
  });
}

function traceEvent({
  answer,
  at,
  code,
  evaluation,
  event_type,
  observability,
  sequence,
  telemetry,
  text
}: {
  answer?: string;
  at: string;
  code?: string;
  evaluation?: Record<string, unknown>;
  event_type: WorkflowRuntimeTraceEvent["event"]["event_type"];
  observability?: Record<string, unknown>;
  sequence: number;
  telemetry?: Record<string, unknown>;
  text?: string;
}): WorkflowRuntimeTraceEvent {
  const receivedAtMs = Date.parse(at) + 50;

  return {
    event: {
      event_type,
      sequence,
      payload: {
        ...(answer ? { answer } : {}),
        ...(code ? { code } : {}),
        ...(evaluation ? { evaluation } : {}),
        ...(observability ? { observability } : {}),
        ...(telemetry ? { telemetry } : {}),
        ...(text ? { text } : {}),
        emitted_at: at
      }
    },
    receivedAt: new Date(receivedAtMs).toISOString(),
    receivedAtMs
  };
}
