import { describe, expect, it } from "vitest";
import {
  buildLangSmithTraceModel,
  type LangSmithTraceMetadataGroup
} from "./langsmith-trace";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

describe("LangSmith trace model", () => {
  it("returns a clean empty state when no observability records exist", () => {
    expect(buildLangSmithTraceModel([])).toMatchObject({
      state: "unavailable",
      status: "idle",
      availabilityLabel: "Trace unavailable",
      traceId: null,
      spans: [],
      summary: {
        spanCount: 0,
        nestedSpanCount: 0,
        transitionCount: 0,
        metadataCount: 0
      }
    });
  });

  it("builds trace overview timing and execution identity", () => {
    const model = buildLangSmithTraceModel([
      traceEvent({
        at: "2026-06-09T10:00:00Z",
        code: "request_received",
        eventType: "status",
        observability: observability({
          created_at: "2026-06-09T10:00:00Z",
          metadata: {
            environment: "test",
            mode: "generate"
          },
          run_id: "run-root-1",
          run_name: "assistant.workflow"
        }),
        sequence: 0
      }),
      traceEvent({
        answer: "Complete.",
        at: "2026-06-09T10:00:03Z",
        eventType: "final",
        observability: observability({
          ended_at: "2026-06-09T10:00:03Z",
          run_id: "run-root-1",
          run_name: "assistant.workflow"
        }),
        sequence: 1
      })
    ]);

    expect(model).toMatchObject({
      state: "linked",
      status: "complete",
      traceId: "trace-123",
      runId: "run-root-1",
      runName: "assistant.workflow",
      traceKind: "assistant_workflow",
      projectName: "creative-test",
      startedAt: "2026-06-09T10:00:00Z",
      endedAt: "2026-06-09T10:00:03Z",
      durationMs: 3000,
      tags: ["assistant", "workflow"]
    });
  });

  it("renders workflow transitions and nested span hierarchy in order", () => {
    const model = buildLangSmithTraceModel([
      traceEvent({
        at: "2026-06-09T10:00:00Z",
        code: "request_received",
        eventType: "status",
        observability: observability({
          lineage: { stage: "intake" },
          spans: [
            {
              end_time: "2026-06-09T10:00:02Z",
              name: "assistant.workflow",
              run_id: "root-run",
              run_type: "chain",
              start_time: "2026-06-09T10:00:00Z",
              status: "completed",
              children: [
                {
                  end_time: "2026-06-09T10:00:01Z",
                  name: "route.request",
                  run_id: "route-run",
                  run_type: "tool",
                  start_time: "2026-06-09T10:00:00.250Z",
                  status: "completed"
                }
              ]
            }
          ]
        }),
        sequence: 0
      }),
      traceEvent({
        at: "2026-06-09T10:00:01Z",
        code: "retrieval_completed",
        eventType: "retrieval",
        message: "Official context selected.",
        observability: observability({
          lineage: {
            chunk_count: 2,
            source: "official_kb",
            stage: "retrieval"
          }
        }),
        sequence: 1
      }),
      traceEvent({
        answer: "Done.",
        at: "2026-06-09T10:00:02Z",
        eventType: "final",
        observability: observability({
          lineage: { stage: "finalization" }
        }),
        sequence: 2
      })
    ]);

    expect(model.spans.map((span) => span.name)).toEqual([
      "assistant.workflow",
      "Intake",
      "route.request",
      "Retrieval",
      "Finalization"
    ]);
    expect(model.spans.find((span) => span.id === "route-run")).toMatchObject({
      parentId: "root-run",
      depth: 1,
      durationMs: 750
    });
    expect(model.spans.find((span) => span.stage === "retrieval")).toMatchObject({
      transitionFrom: "intake",
      transitionReason: "Official context selected."
    });
    expect(model.summary).toMatchObject({
      spanCount: 5,
      nestedSpanCount: 1,
      transitionCount: 2
    });
  });

  it("categorizes provider, retrieval, evaluation, and execution metadata", () => {
    const model = buildLangSmithTraceModel([
      traceEvent({
        at: "2026-06-09T10:00:00Z",
        code: "request_received",
        eventType: "status",
        observability: observability({
          metadata: {
            conversation_id: "conversation-1",
            domain: "three_js"
          },
          run_id: "run-root-1"
        }),
        sequence: 0,
        telemetry: {
          execution: {
            request_duration_ms: 940
          },
          provider: {
            model: "gpt-5-mini",
            name: "openai",
            request_id: "req-1"
          }
        }
      }),
      traceEvent({
        at: "2026-06-09T10:00:01Z",
        code: "retrieval_completed",
        eventType: "retrieval",
        observability: observability({
          lineage: {
            chunk_count: 3,
            source: "official_kb",
            source_ids: ["three-docs"],
            stage: "retrieval"
          }
        }),
        sequence: 1
      }),
      traceEvent({
        at: "2026-06-09T10:00:02Z",
        code: "ragas_eval_completed",
        evaluation: {
          dataset_id: "dataset-1",
          overall_score: 0.87,
          run_id: "eval-run-1"
        },
        eventType: "eval_update",
        observability: observability({
          lineage: { stage: "evaluation" }
        }),
        sequence: 2
      })
    ]);

    expect(entries(model.metadataGroups, "provider")).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ key: "name", value: "openai" }),
        expect.objectContaining({ key: "model", value: "gpt-5-mini" }),
        expect.objectContaining({
          key: "request_duration_ms",
          value: "940"
        })
      ])
    );
    expect(entries(model.metadataGroups, "retrieval")).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ key: "source", value: "official_kb" }),
        expect.objectContaining({ key: "chunk_count", value: "3" })
      ])
    );
    expect(entries(model.metadataGroups, "evaluation")).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ key: "run_id", value: "eval-run-1" }),
        expect.objectContaining({ key: "overall_score", value: "0.87" })
      ])
    );
    expect(entries(model.metadataGroups, "execution")).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          key: "conversation_id",
          value: "conversation-1"
        }),
        expect.objectContaining({ key: "trace_id", value: "trace-123" }),
        expect.objectContaining({ key: "run_id", value: "run-root-1" })
      ])
    );
  });

  it("supports optional camelCase legacy fields without fabricating spans", () => {
    const model = buildLangSmithTraceModel([
      traceEvent({
        at: "2026-06-09T10:00:00Z",
        eventType: "status",
        observability: {
          enabled: false,
          parentRunId: "parent-legacy",
          projectName: "legacy-project",
          reason: "missing_api_key",
          requested: true,
          runId: "legacy-run",
          runName: "legacy.workflow",
          status: "disabled",
          traceId: "legacy-trace",
          traceKind: "assistant_workflow"
        },
        sequence: 0
      })
    ]);

    expect(model).toMatchObject({
      state: "local",
      status: "disabled",
      statusLabel: "Missing Api Key",
      traceId: "legacy-trace",
      runId: "legacy-run",
      parentRunId: "parent-legacy",
      projectName: "legacy-project",
      spans: []
    });
  });
});

function entries(
  groups: LangSmithTraceMetadataGroup[],
  id: LangSmithTraceMetadataGroup["id"]
) {
  return groups.find((group) => group.id === id)?.entries ?? [];
}

function observability(
  overrides: Record<string, unknown> = {}
): Record<string, unknown> {
  return {
    enabled: true,
    project_name: "creative-test",
    provider: "langsmith",
    requested: true,
    status: "enabled",
    tags: ["assistant", "workflow"],
    trace_id: "trace-123",
    trace_kind: "assistant_workflow",
    ...overrides
  };
}

function traceEvent({
  answer,
  at,
  code,
  evaluation,
  eventType,
  message,
  observability,
  sequence,
  telemetry
}: {
  answer?: string;
  at: string;
  code?: string;
  evaluation?: Record<string, unknown>;
  eventType: WorkflowRuntimeTraceEvent["event"]["event_type"];
  message?: string;
  observability?: Record<string, unknown>;
  sequence: number;
  telemetry?: Record<string, unknown>;
}): WorkflowRuntimeTraceEvent {
  const receivedAtMs = Date.parse(at) + 20;
  return {
    event: {
      event_type: eventType,
      sequence,
      payload: {
        ...(answer ? { answer } : {}),
        ...(code ? { code } : {}),
        ...(evaluation ? { evaluation } : {}),
        ...(message ? { message } : {}),
        ...(observability ? { observability } : {}),
        ...(telemetry ? { telemetry } : {}),
        emitted_at: at
      }
    },
    receivedAt: new Date(receivedAtMs).toISOString(),
    receivedAtMs
  };
}
