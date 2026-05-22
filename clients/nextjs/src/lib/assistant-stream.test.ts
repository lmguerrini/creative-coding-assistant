import { describe, expect, it, vi } from "vitest";
import {
  AssistantStreamError,
  decodeAssistantStream,
  parseAssistantStreamLine,
  readEventTimestamp,
  readPreviewArtifactUpdate,
  readWorkflowMetadata,
  streamAssistantEvents,
  workflowNodeFromAssistantStreamEvent,
  type AssistantStreamEvent
} from "./assistant-stream";

describe("assistant stream client", () => {
  it("parses backend NDJSON lines into typed events", () => {
    const event = parseAssistantStreamLine(
      '{"event_type":"status","sequence":0,"payload":{"code":"request_received"}}'
    );

    expect(event).toEqual({
      event_type: "status",
      sequence: 0,
      payload: {
        code: "request_received"
      }
    });
  });

  it("rejects invalid stream event shapes", () => {
    expect(() =>
      parseAssistantStreamLine('{"event_type":"unknown","sequence":0,"payload":{}}')
    ).toThrow(AssistantStreamError);
  });

  it("decodes chunked response bodies", async () => {
    const response = new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(
            new TextEncoder().encode(
              '{"event_type":"status","sequence":0,"payload":{"code":"request'
            )
          );
          controller.enqueue(
            new TextEncoder().encode(
              '_received"}}\n{"event_type":"final","sequence":1,"payload":{"answer":"Done."}}\n'
            )
          );
          controller.close();
        }
      })
    );

    const events = [];
    for await (const event of decodeAssistantStream(response)) {
      events.push(event);
    }

    expect(events.map((event) => event.event_type)).toEqual(["status", "final"]);
    expect(events[1].payload.answer).toBe("Done.");
  });

  it("posts frontend stream requests to the configured endpoint", async () => {
    const fetchImpl = vi.fn(
      async (_url: Parameters<typeof fetch>[0], _init?: Parameters<typeof fetch>[1]) => {
        return new Response(
          '{"event_type":"final","sequence":0,"payload":{"answer":"Done."}}\n'
        );
      }
    );

    const events = [];
    for await (const event of streamAssistantEvents(
      {
        query: "Generate particles.",
        conversationId: "browser-session",
        domain: "webgpu_wgsl",
        mode: "generate"
      },
      {
        endpoint: "http://backend.test/api/assistant/stream",
        fetchImpl
      }
    )) {
      events.push(event);
    }

    expect(fetchImpl).toHaveBeenCalledWith(
      "http://backend.test/api/assistant/stream",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          Accept: "application/x-ndjson",
          "Content-Type": "application/json"
        })
      })
    );
    expect(JSON.parse(String(fetchImpl.mock.calls[0][1]?.body))).toMatchObject({
      conversationId: "browser-session",
      domain: "webgpu_wgsl",
      mode: "generate",
      query: "Generate particles."
    });
    expect(events).toHaveLength(1);
  });

  it("throws a stream error for failed HTTP responses", async () => {
    const events = decodeAssistantStream(new Response("Nope.", { status: 503 }));

    await expect(events.next()).rejects.toThrow(AssistantStreamError);
  });

  it("maps backend events to workflow nodes", () => {
    const events: AssistantStreamEvent[] = [
      {
        event_type: "status",
        sequence: 0,
        payload: { code: "request_received" }
      },
      {
        event_type: "status",
        sequence: 1,
        payload: { code: "route_selected" }
      },
      {
        event_type: "prompt_rendered",
        sequence: 2,
        payload: { code: "prompt_rendered" }
      },
      {
        event_type: "token_delta",
        sequence: 3,
        payload: { text: "Hello" }
      },
      {
        event_type: "final",
        sequence: 4,
        payload: { answer: "Done." }
      }
    ];

    expect(events.map(workflowNodeFromAssistantStreamEvent)).toEqual([
      "intake",
      "routing",
      "prompt_rendering",
      "generation",
      "finalization"
    ]);
  });

  it("reads workflow runtime metadata from enriched stream events", () => {
    const event: AssistantStreamEvent = {
      event_type: "generation_input",
      sequence: 2,
      payload: {
        code: "generation_input_prepared",
        emitted_at: "2026-05-22T10:20:30Z",
        workflow: {
          step: "generation",
          phase: "running",
          status: "running",
          current_step: "generation",
          completed_steps: ["intake", "routing"],
          skipped_steps: ["memory"],
          refinement_count: 0,
          review_outcome: null,
          review_reasons: []
        }
      }
    };

    expect(readEventTimestamp(event)).toBe("2026-05-22T10:20:30Z");
    expect(readWorkflowMetadata(event)).toEqual({
      step: "generation",
      phase: "running",
      status: "running",
      current_step: "generation",
      completed_steps: ["intake", "routing"],
      skipped_steps: ["memory"],
      refinement_count: 0,
      review_outcome: null,
      review_reasons: []
    });
    expect(workflowNodeFromAssistantStreamEvent(event)).toBe("generation");
  });

  it("reads preview runtime updates from preview artifact events", () => {
    const event: AssistantStreamEvent = {
      event_type: "preview_artifact",
      sequence: 4,
      payload: {
        artifact_id: "source-sketch",
        status: "skipped",
        emitted_at: "2026-05-22T10:25:00Z",
        result: {
          summary: "Preview pipeline foundation only; renderer execution is deferred.",
          completed_at: "2026-05-22T10:25:00Z",
          preview_artifact_id: "preview-manifest",
          request: {
            target: "browser_sandbox"
          },
          provenance: {
            renderer_id: "preview.noop"
          }
        }
      }
    };

    expect(readPreviewArtifactUpdate(event)).toEqual({
      status: "skipped",
      artifactId: "source-sketch",
      previewArtifactId: "preview-manifest",
      rendererId: "preview.noop",
      target: "browser_sandbox",
      summary: "Preview pipeline foundation only; renderer execution is deferred.",
      errorMessage: null,
      emittedAt: "2026-05-22T10:25:00Z",
      completedAt: "2026-05-22T10:25:00Z"
    });
  });
});
