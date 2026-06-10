import { describe, expect, it, vi } from "vitest";
import {
  AssistantStreamError,
  decodeAssistantStream,
  parseAssistantStreamLine,
  readEventTimestamp,
  readPreviewArtifactUpdate,
  readStreamEventError,
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
        attachments: [
          {
            type: "image",
            id: "image-reference-1",
            name: "palette.png",
            mimeType: "image/png",
            sizeBytes: 128,
            dataUrl: "data:image/png;base64,cGFsZXR0ZQ=="
          }
        ],
        conversationId: "browser-session",
        domain: "webgpu_wgsl",
        mode: "generate",
        artifactRefinement: {
          artifactId: "source-sketch",
          title: "aurora-field.p5.js",
          language: "p5.js",
          content: "function draw() { background(0); }",
          instruction: "Make this more organic.",
          domain: "p5_js",
          runtime: "p5",
          rendererId: "surface.p5",
          previewEligible: true,
          qualityScore: 0.91,
          qualityRank: 1,
          critiqueRationale: "Strong visual candidate.",
          refinementGuidance: "Soften particle motion.",
          creativeTranslation: {
            outputModality: "visual",
            creativeIntent: "Create a calm particle field.",
            symbolicReferences: [],
            geometricReferences: [],
            musicalReferences: [],
            moodAtmosphere: ["calm"],
            movementLanguage: ["drift"],
            colorMaterialDirection: [],
            runtimeRecommendations: ["p5.js"],
            structureDirection: [],
            generationConstraints: [],
            refinementTargets: ["Preserve atmosphere: calm"],
            sacredGeometry: {
              concepts: ["mandala"],
              geometricStructure: ["Build nested rings."],
              symmetryType: ["Use radial symmetry."],
              movementBehavior: [],
              visualComposition: [],
              colorMaterialDirection: [],
              runtimeRecommendations: ["p5.js"],
              audioImplications: [],
              generationConstraints: [
                "Do not add unsupported symbolic claims."
              ]
            }
          }
        }
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
      query: "Generate particles.",
      artifactRefinement: expect.objectContaining({
        artifactId: "source-sketch",
        title: "aurora-field.p5.js",
        content: "function draw() { background(0); }",
        instruction: "Make this more organic.",
        domain: "p5_js",
        runtime: "p5",
        rendererId: "surface.p5",
        previewEligible: true,
        creativeTranslation: expect.objectContaining({
          creativeIntent: "Create a calm particle field.",
          moodAtmosphere: ["calm"],
          sacredGeometry: expect.objectContaining({
            concepts: ["mandala"],
            runtimeRecommendations: ["p5.js"]
          })
        })
      }),
      attachments: [
        expect.objectContaining({
          type: "image",
          name: "palette.png",
          mimeType: "image/png",
          dataUrl: "data:image/png;base64,cGFsZXR0ZQ=="
        })
      ]
    });
    expect(events).toHaveLength(1);
  });

  it("throws a stream error for failed HTTP responses", async () => {
    const events = decodeAssistantStream(new Response("Nope.", { status: 503 }));

    await expect(events.next()).rejects.toThrow(AssistantStreamError);
  });

  it("builds a structured stream error from error events", () => {
    const event: AssistantStreamEvent = {
      event_type: "error",
      sequence: 3,
      payload: {
        code: "provider_unavailable",
        message: "Provider unavailable."
      }
    };

    expect(readStreamEventError(event)).toMatchObject({
      category: "stream",
      subsystem: "generation_provider",
      type: "provider_unavailable",
      userMessage: "The model provider is unavailable for this live response.",
      retryLabel: "Send prompt again",
      resetLabel: "Clear workspace session"
    });
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
        event_type: "node_started",
        sequence: 4,
        payload: { code: "node_started", node: "review" }
      },
      {
        event_type: "review_failed",
        sequence: 5,
        payload: { code: "review_failed" }
      },
      {
        event_type: "refinement_requested",
        sequence: 6,
        payload: { code: "refinement_requested" }
      },
      {
        event_type: "retry_started",
        sequence: 7,
        payload: { code: "retry_started" }
      },
      {
        event_type: "refinement_completed",
        sequence: 8,
        payload: { code: "refinement_completed" }
      },
      {
        event_type: "node_completed",
        sequence: 9,
        payload: { code: "node_completed", node: "refinement" }
      },
      {
        event_type: "retry_completed",
        sequence: 10,
        payload: { code: "retry_completed" }
      },
      {
        event_type: "review_passed",
        sequence: 11,
        payload: { code: "review_passed" }
      },
      {
        event_type: "node_failed",
        sequence: 12,
        payload: { code: "node_failed", node: "generation" }
      },
      {
        event_type: "artifact_extracted",
        sequence: 13,
        payload: { code: "artifact_extracted" }
      },
      {
        event_type: "preview_artifact",
        sequence: 14,
        payload: { code: "preview_artifact_prepared", status: "succeeded" }
      },
      {
        event_type: "final",
        sequence: 15,
        payload: { answer: "Done." }
      }
    ];

    expect(events.map(workflowNodeFromAssistantStreamEvent)).toEqual([
      "intake",
      "routing",
      "prompt_rendering",
      "generation",
      "review",
      "review",
      "review",
      "review",
      "refinement",
      "refinement",
      "review",
      "review",
      "generation",
      "artifact_extraction",
      "preview_preparation",
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
          review_reasons: [],
          artifact_count: 1,
          preview_artifact_count: 0,
          image_reference_count: 1,
          image_references: [
            {
              id: "image-reference-1",
              name: "palette.png",
              mime_type: "image/png",
              size_bytes: 128
            }
          ]
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
      review_reasons: [],
      artifact_count: 1,
      artifact_critique_count: 0,
      recommended_artifact_id: null,
      preview_artifact_count: 0,
      image_reference_count: 1,
      image_references: [
        {
          id: "image-reference-1",
          name: "palette.png",
          mime_type: "image/png",
          size_bytes: 128
        }
      ]
    });
    expect(workflowNodeFromAssistantStreamEvent(event)).toBe("generation");
  });

  it("reads preview runtime updates from preview artifact events", () => {
    const event: AssistantStreamEvent = {
      event_type: "preview_artifact",
      sequence: 4,
      payload: {
        artifact_id: "source-sketch",
        status: "succeeded",
        emitted_at: "2026-05-22T10:25:00Z",
        result: {
          summary: "p5.js runtime ready for sandbox execution.",
          completed_at: "2026-05-22T10:25:00Z",
          preview_artifact_id: "source-sketch",
          request: {
            target: "browser_sandbox"
          },
          provenance: {
            renderer_id: "surface.p5"
          }
        }
      }
    };

    expect(readPreviewArtifactUpdate(event)).toEqual({
      status: "succeeded",
      artifactId: "source-sketch",
      previewArtifactId: "source-sketch",
      rendererId: "surface.p5",
      target: "browser_sandbox",
      summary: "p5.js runtime ready for sandbox execution.",
      errorMessage: null,
      error: null,
      emittedAt: "2026-05-22T10:25:00Z",
      completedAt: "2026-05-22T10:25:00Z"
    });
  });
});
