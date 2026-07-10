import { describe, expect, it } from "vitest";
import { buildConversationContextModel } from "./conversation-context";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

describe("conversation context diagnostics", () => {
  it("reports published context counts without retaining private content", () => {
    const model = buildConversationContextModel({
      traceEvents: [memoryTraceEvent(), contextTraceEvent()],
      visibleEntryCount: 8
    });

    expect(model).toMatchObject({ source: "stream" });
    expect(model.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "visible_history",
          value: "8 entries",
          state: "observed"
        }),
        expect.objectContaining({
          id: "model_context",
          value: "3 turns",
          state: "observed"
        }),
        expect.objectContaining({
          id: "memory",
          value: "2 items",
          state: "observed"
        }),
        expect.objectContaining({
          id: "summary",
          value: "9 covered turns",
          state: "observed"
        }),
        expect.objectContaining({
          id: "retrieval",
          value: "4 chunks",
          state: "observed"
        }),
        expect.objectContaining({
          id: "truncation",
          value: "Up to 6 turns",
          state: "observed"
        }),
        expect.objectContaining({
          id: "token_allocation",
          value: "Not reported",
          state: "unavailable"
        })
      ])
    );
    expect(JSON.stringify(model)).not.toContain("private-memory-content");
    expect(JSON.stringify(model)).not.toContain("private-retrieval-content");
  });

  it("does not infer absent context evidence", () => {
    const model = buildConversationContextModel({
      traceEvents: [],
      visibleEntryCount: 0
    });

    expect(model).toMatchObject({ source: "none" });
    expect(model.diagnostics.find((item) => item.id === "model_context")).toMatchObject({
      value: "Not published",
      state: "unavailable"
    });
  });
});

function contextTraceEvent(): WorkflowRuntimeTraceEvent {
  const at = "2026-07-11T10:00:00Z";
  return {
    event: {
      event_type: "context",
      sequence: 1,
      payload: {
        context: {
          summary: {
            recent_turn_count: 3,
            has_running_summary: true,
            project_memory_count: 2,
            retrieval_chunk_count: 4
          }
        }
      }
    },
    receivedAt: at,
    receivedAtMs: Date.parse(at)
  };
}

function memoryTraceEvent(): WorkflowRuntimeTraceEvent {
  const at = "2026-07-11T09:59:59Z";
  return {
    event: {
      event_type: "memory",
      sequence: 0,
      payload: {
        request: { recent_turn_limit: 6 },
        context: {
          recent_turns: [
            { content: "private-memory-content" },
            { content: "private-memory-content" },
            { content: "private-memory-content" }
          ],
          running_summary: {
            content: "private-memory-content",
            covered_turn_count: 9
          },
          project_memories: [
            { content: "private-memory-content" },
            { content: "private-memory-content" }
          ]
        }
      }
    },
    receivedAt: at,
    receivedAtMs: Date.parse(at)
  };
}
