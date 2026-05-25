import { describe, expect, it } from "vitest";
import { getLocalWorkspaceSnapshot } from "./assistant-client";
import type { AssistantStreamEvent } from "./assistant-stream";
import { buildRetrievalRuntimeModel } from "./retrieval-runtime";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

describe("retrieval runtime", () => {
  it("preserves the richer fallback retrieval model from the local snapshot", () => {
    const snapshot = getLocalWorkspaceSnapshot();

    expect(buildRetrievalRuntimeModel(snapshot.retrieval, [])).toMatchObject({
      request: {
        query:
          "Stable WebGPU particle field for a projection wall with low-frequency audio response",
        domainLabels: ["WebGPU / WGSL", "GLSL"]
      },
      summary: {
        state: "available",
        status: "Grounded",
        sourceCount: 2,
        chunkCount: 3,
        freshnessLabel: "1 stale source",
        warning:
          "1 source is older than the preferred refresh window for shader guidance."
      }
    });
  });

  it("hydrates retrieval sources and chunks from streamed retrieval completion events", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const runtime = buildRetrievalRuntimeModel(snapshot.retrieval, [
      retrievalTraceEvent({
        code: "retrieval_requested",
        payload: {
          emitted_at: "2026-05-23T10:05:00Z",
          request: {
            query: "Build a projection-safe WebGPU particle wall with shader guidance.",
            limit: 5,
            filters: {
              domains: ["webgpu_wgsl", "glsl"]
            }
          }
        },
        sequence: 2
      }),
      retrievalTraceEvent({
        code: "retrieval_completed",
        payload: {
          context: {
            source: "official_kb",
            request: {
              query:
                "Build a projection-safe WebGPU particle wall with shader guidance.",
              limit: 5,
              filters: {
                domains: ["webgpu_wgsl", "glsl"]
              }
            },
            chunks: [
              {
                source_id: "webgpu_mdn_api",
                domain: "webgpu_wgsl",
                source_type: "api_reference",
                publisher: "MDN",
                registry_title: "WebGPU API",
                document_title: "WebGPU API",
                source_url:
                  "https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API",
                resolved_url:
                  "https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API",
                chunk_index: 0,
                excerpt:
                  "The WebGPU API separates device setup and queue submission so compute and render passes can remain isolated.",
                score: 0.94
              },
              {
                source_id: "webgpu_mdn_api",
                domain: "webgpu_wgsl",
                source_type: "api_reference",
                publisher: "MDN",
                registry_title: "WebGPU API",
                document_title: "GPUCanvasContext",
                source_url:
                  "https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API",
                resolved_url:
                  "https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API",
                chunk_index: 1,
                excerpt:
                  "Canvas configuration should happen once for each presentation surface to keep preview output stable.",
                score: 0.86
              },
              {
                source_id: "glsl_language_spec_460",
                domain: "glsl",
                source_type: "specification",
                publisher: "Khronos Group",
                registry_title: "OpenGL Shading Language 4.60 Specification",
                document_title: "Type Qualifiers",
                source_url:
                  "https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html",
                resolved_url:
                  "https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html",
                chunk_index: 7,
                excerpt:
                  "Explicit types and layout-compatible data flow keep shader state deterministic across stages.",
                score: 0.72
              }
            ]
          },
          emitted_at: "2026-05-23T10:05:01Z"
        },
        sequence: 3
      })
    ]);

    expect(runtime.request.domainLabels).toEqual(["WebGPU / WGSL", "GLSL"]);
    expect(runtime.summary).toMatchObject({
      state: "available",
      status: "Grounded",
      providerLabel: "Official knowledge base",
      sourceCount: 2,
      chunkCount: 3,
      qualityLabel: "Top score 94%",
      freshnessLabel: "1 stale source",
      coverageLabel: "2 domains",
      warning: "1 source may be stale."
    });
    expect(runtime.sources[0]).toMatchObject({
      sourceId: "webgpu_mdn_api",
      title: "WebGPU API",
      freshness: "fresh",
      quality: "high",
      qualityLabel: "94% match"
    });
    expect(runtime.sources[0].chunks).toHaveLength(2);
    expect(runtime.sources[0].chunks[0]).toMatchObject({
      chunkIndex: 0,
      relevanceLabel: "Best match"
    });
    expect(runtime.sources[1]).toMatchObject({
      sourceId: "glsl_language_spec_460",
      domainLabel: "GLSL",
      freshness: "stale",
      sourceTypeLabel: "Specification"
    });
  });

  it("shows a pending retrieval state when a newer request has not completed yet", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const runtime = buildRetrievalRuntimeModel(snapshot.retrieval, [
      retrievalTraceEvent({
        code: "retrieval_requested",
        payload: {
          emitted_at: "2026-05-23T10:07:00Z",
          request: {
            query: "Find p5.js references for low-frequency motion.",
            limit: 5,
            filters: {
              domains: ["p5_js"]
            }
          }
        },
        sequence: 6
      })
    ]);

    expect(runtime.summary).toMatchObject({
      state: "pending",
      status: "Searching"
    });
    expect(runtime.summary.headline).toContain("Searching for");
    expect(runtime.request.filterLabels).toEqual(["p5.js"]);
    expect(runtime.sources).toEqual([]);
  });

  it("surfaces an empty retrieval state when no chunks are returned", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const runtime = buildRetrievalRuntimeModel(snapshot.retrieval, [
      retrievalTraceEvent({
        code: "retrieval_completed",
        payload: {
          context: {
            source: "official_kb",
            request: {
              query: "Find TouchDesigner references for this projection loop.",
              limit: 5,
              filters: {
                domains: ["touchdesigner"]
              }
            },
            chunks: []
          },
          emitted_at: "2026-05-23T10:08:00Z"
        },
        sequence: 7
      })
    ]);

    expect(runtime.summary).toMatchObject({
      state: "empty",
      status: "No matches",
      warning: "No retrieved chunks for TouchDesigner."
    });
    expect(runtime.sources).toEqual([]);
  });

  it("surfaces structured retrieval failures from streamed retrieval events", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const runtime = buildRetrievalRuntimeModel(snapshot.retrieval, [
      retrievalTraceEvent({
        code: "retrieval_completed",
        payload: {
          context: {
            source: "official_kb",
            request: {
              query: "Find TouchDesigner references for this projection loop.",
              limit: 5,
              filters: {
                domains: ["touchdesigner"]
              }
            },
            chunks: []
          },
          error: {
            type: "retrieval_gateway_failed",
            message: "Retrieval references are unavailable for this request.",
            recoverable: true,
            retry_label: "Retry retrieval",
            subsystem: "retrieval_gateway"
          },
          emitted_at: "2026-05-23T10:09:00Z"
        },
        sequence: 8
      })
    ]);

    expect(runtime.summary).toMatchObject({
      state: "error",
      status: "Retrieval failed",
      headline: "Retrieval failed"
    });
    expect(runtime.summary.error).toMatchObject({
      category: "retrieval",
      subsystem: "retrieval_gateway",
      type: "retrieval_gateway_failed",
      retryLabel: "Retry retrieval"
    });
    expect(runtime.sources).toEqual([]);
  });
});

function retrievalTraceEvent({
  code,
  payload,
  sequence
}: {
  code: "retrieval_requested" | "retrieval_completed";
  payload: Record<string, unknown>;
  sequence: number;
}): WorkflowRuntimeTraceEvent {
  const event: AssistantStreamEvent = {
    event_type: "retrieval",
    sequence,
    payload: {
      code,
      ...payload
    }
  };

  return {
    event,
    receivedAt:
      typeof payload.emitted_at === "string"
        ? payload.emitted_at
        : "2026-05-23T10:05:00Z",
    receivedAtMs: Date.parse(
      typeof payload.emitted_at === "string"
        ? payload.emitted_at
        : "2026-05-23T10:05:00Z"
    )
  };
}
