import { describe, expect, it } from "vitest";
import type { AssistantStreamEvent } from "./assistant-stream";
import { buildWorkflowTimelineModel } from "./workflow-timeline";

describe("workflow timeline model", () => {
  it("returns a clean empty state without workflow events", () => {
    expect(buildWorkflowTimelineModel([])).toEqual({
      state: "empty",
      events: [],
      summary: {
        eventCount: 0,
        warningCount: 0,
        errorCount: 0,
        startedAt: null,
        completedAt: null,
        totalDurationMs: null
      }
    });
  });

  it("orders high-signal events by stream sequence and removes token noise", () => {
    const timeline = buildWorkflowTimelineModel([
      traceEvent({
        at: "2026-06-01T10:00:03Z",
        event_type: "final",
        sequence: 3
      }),
      traceEvent({
        at: "2026-06-01T10:00:00Z",
        code: "request_received",
        event_type: "status",
        message: "Request accepted.",
        sequence: 0
      }),
      traceEvent({
        at: "2026-06-01T10:00:02Z",
        code: "route_selected",
        event_type: "status",
        message: "Route selected.",
        sequence: 2
      }),
      traceEvent({
        at: "2026-06-01T10:00:01Z",
        event_type: "token_delta",
        sequence: 1
      })
    ]);

    expect(timeline.events.map((event) => event.label)).toEqual([
      "Request received",
      "Route selected",
      "Final response"
    ]);
    expect(timeline.summary).toMatchObject({
      eventCount: 3,
      totalDurationMs: 3000
    });
  });

  it("derives node duration and transition metadata from lifecycle events", () => {
    const timeline = buildWorkflowTimelineModel([
      traceEvent({
        at: "2026-06-01T10:00:00Z",
        event_type: "node_started",
        node: "generation",
        nodeLabel: "Generation",
        sequence: 0,
        workflow: workflow("generation", "running")
      }),
      traceEvent({
        at: "2026-06-01T10:00:01.500Z",
        decisionReason: "generation_completed",
        event_type: "node_completed",
        node: "generation",
        nodeLabel: "Generation",
        sequence: 1,
        workflow: workflow("generation", "completed")
      })
    ]);

    expect(timeline.events[1]).toMatchObject({
      label: "Provider generation completed",
      stageLabel: "Generation",
      phase: "completed",
      status: "complete",
      durationMs: 1500,
      transitionReason: "generation_completed"
    });
  });

  it("surfaces warning and error details without losing event order", () => {
    const timeline = buildWorkflowTimelineModel([
      traceEvent({
        at: "2026-06-01T10:00:00Z",
        event_type: "review_failed",
        message: "Generated answer is missing a code block.",
        retryReason: "missing_code_block",
        sequence: 0,
        workflow: workflow("review", "running")
      }),
      traceEvent({
        at: "2026-06-01T10:00:01Z",
        errorMessage: "Provider request timed out.",
        event_type: "node_failed",
        message: "Generation failed.",
        node: "generation",
        sequence: 1,
        workflow: workflow("generation", "failed")
      })
    ]);

    expect(timeline.summary).toMatchObject({
      warningCount: 1,
      errorCount: 1
    });
    expect(timeline.events[0]).toMatchObject({
      status: "warning",
      warning: "Generated answer is missing a code block.",
      transitionReason: "missing_code_block"
    });
    expect(timeline.events[1]).toMatchObject({
      status: "error",
      error: "Provider request timed out."
    });
  });

  it("renders legacy events with received-time and inferred-stage fallbacks", () => {
    const timeline = buildWorkflowTimelineModel([
      traceEvent({
        event_type: "retrieval",
        code: "retrieval_completed",
        message: "Retrieved three context chunks.",
        sequence: 0,
        receivedAt: "2026-06-01T10:00:04Z"
      })
    ]);

    expect(timeline.events[0]).toMatchObject({
      label: "Retrieval completed",
      detail: "Retrieved three context chunks.",
      at: "2026-06-01T10:00:04Z",
      phase: "completed",
      nodeId: "retrieval",
      stageLabel: "Retrieval",
      status: "complete"
    });
  });
});

function workflow(step: string, phase: string) {
  return {
    step,
    phase,
    status: phase === "failed" ? "failed" : "running",
    current_step: phase === "completed" ? null : step,
    completed_steps: [],
    skipped_steps: [],
    refinement_count: 0,
    review_outcome: null,
    review_reasons: []
  };
}

function traceEvent({
  at,
  code,
  decisionReason,
  errorMessage,
  event_type,
  message,
  node,
  nodeLabel,
  receivedAt,
  retryReason,
  sequence,
  workflow: workflowMetadata
}: {
  at?: string;
  code?: string;
  decisionReason?: string;
  errorMessage?: string;
  event_type: AssistantStreamEvent["event_type"];
  message?: string;
  node?: string;
  nodeLabel?: string;
  receivedAt?: string;
  retryReason?: string;
  sequence: number;
  workflow?: Record<string, unknown>;
}) {
  const resolvedReceivedAt = receivedAt ?? at ?? "2026-06-01T10:00:00Z";
  return {
    event: {
      event_type,
      sequence,
      payload: {
        ...(at ? { emitted_at: at } : {}),
        ...(code ? { code } : {}),
        ...(decisionReason ? { decision_reason: decisionReason } : {}),
        ...(errorMessage ? { error_message: errorMessage } : {}),
        ...(message ? { message } : {}),
        ...(node ? { node } : {}),
        ...(nodeLabel ? { node_label: nodeLabel } : {}),
        ...(retryReason ? { retry_reason: retryReason } : {}),
        ...(workflowMetadata ? { workflow: workflowMetadata } : {})
      }
    },
    receivedAt: resolvedReceivedAt,
    receivedAtMs: Date.parse(resolvedReceivedAt)
  };
}
