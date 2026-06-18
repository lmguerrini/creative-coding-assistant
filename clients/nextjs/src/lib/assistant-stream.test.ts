import { describe, expect, it, vi } from "vitest";
import {
  AssistantStreamError,
  decodeAssistantStream,
  parseAssistantStreamLine,
  readClarificationSummary,
  readCreativeExecutionPlanSummary,
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
            },
            shaderPresets: {
              presets: ["glow"],
              colorBehavior: ["Use a bright core color."],
              lightMaterialBehavior: ["Use bounded emission layers."],
              motionBehavior: ["Pulse intensity slowly."],
              shaderStructure: ["Separate an emission mask."],
              runtimeSuitability: [
                "Use the selected compatible runtime: p5.js."
              ],
              performanceConstraints: [
                "Use a bounded number of glow layers."
              ]
            },
            visualStyle: {
              styles: ["minimal"],
              paletteBehavior: ["Use one dominant tone."],
              contrastBehavior: ["Use clear value hierarchy."],
              compositionTendencies: ["Use deliberate negative space."],
              motionTendencies: ["Use slow readable transitions."],
              textureTendencies: ["Keep surfaces clean."],
              spatialOrganization: ["Favor a stable focal point."],
              runtimeSuitability: [
                "Use the selected compatible runtime: p5.js."
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
          }),
          shaderPresets: expect.objectContaining({
            presets: ["glow"],
            runtimeSuitability: [
              "Use the selected compatible runtime: p5.js."
            ]
          }),
          visualStyle: expect.objectContaining({
            styles: ["minimal"],
            runtimeSuitability: [
              "Use the selected compatible runtime: p5.js."
            ]
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
        event_type: "planning",
        sequence: 2,
        payload: { code: "creative_plan_prepared" }
      },
      {
        event_type: "prompt_rendered",
        sequence: 3,
        payload: { code: "prompt_rendered" }
      },
      {
        event_type: "token_delta",
        sequence: 4,
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
      "planning",
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
    const creativePlan = {
      output_modality: "visual",
      generation_strategy: "Generate one p5 candidate.",
      recommended_runtime: "p5",
      recommended_renderer_id: "surface.p5",
      recommended_preview_target: "browser_sandbox",
      recommended_shader_style: "glow",
      candidate_count: 1,
      refinement_budget: 1,
      expected_complexity: "medium",
      estimated_token_cost: 2800,
      export_readiness: "ready",
      runtime_available: true,
      runtime_support_summary: "p5.js browser preview is available.",
      plan_steps: ["Target p5 output."],
      constraints: ["Keep code browser-safe."],
      evidence: ["Route selected: generate."]
    };
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
          planning_available: true,
          creative_plan: creativePlan,
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
      creative_plan: {
        outputModality: "visual",
        generationStrategy: "Generate one p5 candidate.",
        recommendedRuntime: "p5",
        recommendedRendererId: "surface.p5",
        recommendedPreviewTarget: "browser_sandbox",
        recommendedShaderStyle: "glow",
        candidateCount: 1,
        refinementBudget: 1,
        expectedComplexity: "medium",
        estimatedTokenCost: 2800,
        exportReadiness: "ready",
        runtimeAvailable: true,
        runtimeSupportSummary: "p5.js browser preview is available.",
        planSteps: ["Target p5 output."],
        constraints: ["Keep code browser-safe."],
        evidence: ["Route selected: generate."]
      },
      planning_available: true,
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

  it("reads creative execution plans from planning events", () => {
    const plan = readCreativeExecutionPlanSummary({
      outputModality: "visual",
      generationStrategy: "Generate one p5 candidate.",
      recommendedRuntime: "p5",
      recommendedRendererId: "surface.p5",
      recommendedPreviewTarget: "browser_sandbox",
      candidateCount: 1,
      refinementBudget: 1,
      expectedComplexity: "medium",
      estimatedTokenCost: 2800,
      exportReadiness: "ready",
      runtimeAvailable: true,
      runtimeSupportSummary: "p5.js browser preview is available.",
      planSteps: ["Target p5 output."],
      constraints: ["Keep code browser-safe."],
      evidence: ["Route selected: generate."]
    });

    expect(plan).toMatchObject({
      outputModality: "visual",
      recommendedRuntime: "p5",
      candidateCount: 1,
      exportReadiness: "ready"
    });
  });

  it("reads clarification metadata from prompt input workflow events", () => {
    const clarification = {
      reason: "ambiguous_modality",
      confidence: 0.44,
      summary: "The output modality is unclear.",
      original_query: "Make something evocative about rain.",
      suggested_options: ["Visual sketch", "Audio piece", "Audiovisual piece"],
      default_recommendation: "Visual sketch",
      signal_summary: ["route=generate", "modality=unspecified"],
      questions: [
        {
          id: "output_modality",
          prompt: "What should the assistant generate first?",
          kind: "single_choice",
          suggested_options: ["Visual sketch", "Audio piece", "Audiovisual piece"],
          default_recommendation: "Visual sketch"
        }
      ]
    };
    const event: AssistantStreamEvent = {
      event_type: "prompt_input",
      sequence: 4,
      payload: {
        code: "clarification_required",
        clarification,
        workflow: {
          step: "prompt_input",
          phase: "running",
          status: "running",
          current_step: "prompt_input",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_outcome: null,
          review_reasons: [],
          artifact_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          clarification_required: true,
          clarification_reason: "ambiguous_modality",
          clarification_question_count: 1,
          clarification
        }
      }
    };

    expect(workflowNodeFromAssistantStreamEvent(event)).toBe("prompt_input");
    expect(readClarificationSummary(event.payload.clarification)).toEqual({
      reason: "ambiguous_modality",
      confidence: 0.44,
      summary: "The output modality is unclear.",
      originalQuery: "Make something evocative about rain.",
      suggestedOptions: ["Visual sketch", "Audio piece", "Audiovisual piece"],
      defaultRecommendation: "Visual sketch",
      signalSummary: ["route=generate", "modality=unspecified"],
      questions: [
        {
          id: "output_modality",
          prompt: "What should the assistant generate first?",
          kind: "single_choice",
          suggestedOptions: ["Visual sketch", "Audio piece", "Audiovisual piece"],
          defaultRecommendation: "Visual sketch"
        }
      ]
    });
    expect(readWorkflowMetadata(event)).toEqual(
      expect.objectContaining({
        clarification: expect.objectContaining({
          reason: "ambiguous_modality",
          questions: expect.arrayContaining([
            expect.objectContaining({ id: "output_modality" })
          ])
        }),
        clarification_question_count: 1,
        clarification_reason: "ambiguous_modality",
        clarification_required: true
      })
    );
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
