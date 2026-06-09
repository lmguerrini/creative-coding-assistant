import { describe, expect, it } from "vitest";
import { buildProviderTelemetryModel } from "./provider-telemetry";
import {
  buildCreativeCostIntelligenceModel,
  buildCreativeCostRunRecord,
  type CreativeCostRunRecord
} from "./creative-cost-intelligence";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

describe("creative cost intelligence model", () => {
  it("keeps the first-run estimate and session summary clean", () => {
    const model = buildCreativeCostIntelligenceModel({
      draftPrompt: "",
      providerTelemetry: buildProviderTelemetryModel([]),
      retrievalChunkCount: 0,
      runHistory: [],
      traceEvents: []
    });

    expect(model.estimate).toMatchObject({
      state: "empty",
      costRange: null,
      inputTokenRange: null,
      outputTokenRange: null
    });
    expect(model.current).toMatchObject({
      state: "idle",
      cost: null,
      totalTokens: null
    });
    expect(model.session).toMatchObject({
      runCount: 0,
      totalCost: null,
      coverage: "none"
    });
  });

  it("shows token scope without inventing cost when pricing is unavailable", () => {
    const model = buildCreativeCostIntelligenceModel({
      draftPrompt: "Create three alternatives and compare the strongest result.",
      providerTelemetry: buildProviderTelemetryModel([]),
      retrievalChunkCount: 2,
      runHistory: [],
      traceEvents: []
    });

    expect(model.estimate).toMatchObject({
      state: "pricing_unavailable",
      requestedArtifactCount: 3,
      includesReview: true,
      costRange: null
    });
    expect(model.estimate.promptTokens).toBeGreaterThan(0);
    expect(model.estimate.contextTokens).toBe(360);
    expect(model.estimate.inputTokenRange?.[1]).toBeGreaterThan(
      model.estimate.inputTokenRange?.[0] ?? 0
    );
    expect(model.estimate.outputTokenRange?.[1]).toBeGreaterThan(
      model.estimate.outputTokenRange?.[0] ?? 0
    );
  });

  it("uses known provider pricing for a bounded pre-generation estimate", () => {
    const model = buildCreativeCostIntelligenceModel({
      draftPrompt: "Build two visual concepts and review the best option.",
      providerTelemetry: buildProviderTelemetryModel([]),
      retrievalChunkCount: 1,
      runHistory: [runRecord({ id: "prior-run", cost: 0.004 })],
      traceEvents: []
    });

    expect(model.estimate).toMatchObject({
      state: "ready",
      providerName: "openai",
      modelName: "gpt-5-mini",
      requestedArtifactCount: 2,
      includesReview: true
    });
    expect(model.estimate.costRange?.[0]).toBeGreaterThan(0);
    expect(model.estimate.costRange?.[1]).toBeGreaterThan(
      model.estimate.costRange?.[0] ?? 0
    );
  });

  it("aggregates completed generations and refinements across the session", () => {
    const traceEvents = refinementTrace();
    const providerTelemetry = buildProviderTelemetryModel(traceEvents);
    const model = buildCreativeCostIntelligenceModel({
      draftPrompt: "",
      providerTelemetry,
      retrievalChunkCount: 0,
      runHistory: [
        runRecord({
          id: "generation-1",
          artifactCount: 1,
          cost: 0.003,
          critiqueCount: 1,
          reviewCount: 1
        })
      ],
      traceEvents
    });

    expect(model.current).toMatchObject({
      state: "complete",
      inputTokens: 1200,
      outputTokens: 500,
      totalTokens: 1700,
      cost: 0.006,
      durationMs: 1450,
      retryCount: 1,
      fallbackCount: 1,
      retryCost: 0.001,
      fallbackCost: 0.0004,
      artifactCount: 2,
      refinementCount: 1,
      critiqueCount: 2,
      reviewCount: 1
    });
    expect(model.session).toMatchObject({
      runCount: 2,
      generationCount: 2,
      refinementCount: 1,
      artifactCount: 3,
      critiqueCount: 3,
      reviewCount: 2,
      costedRunCount: 2,
      coverage: "complete"
    });
    expect(model.session.totalCost).toBeCloseTo(0.009);
    expect(model.session.averagePerGeneration).toBeCloseTo(0.0045);
    expect(model.session.averagePerArtifact).toBeCloseTo(0.003);
  });

  it("extracts truthful artifact and refinement context from workflow events", () => {
    const traceEvents = refinementTrace();
    const record = buildCreativeCostRunRecord({
      providerTelemetry: buildProviderTelemetryModel(traceEvents),
      traceEvents
    });

    expect(record).toMatchObject({
      kind: "artifact_refinement",
      artifactCount: 2,
      refinementCount: 1,
      critiqueCount: 2,
      reviewCount: 1,
      retryCost: 0.001,
      fallbackCost: 0.0004
    });
  });

  it("preserves legacy completed traces without fake usage or cost", () => {
    const traceEvents = [
      traceEvent({
        event_type: "final",
        payload: {
          answer: "Legacy response."
        },
        sequence: 0
      })
    ];
    const model = buildCreativeCostIntelligenceModel({
      draftPrompt: "",
      providerTelemetry: buildProviderTelemetryModel(traceEvents),
      retrievalChunkCount: 0,
      runHistory: [],
      traceEvents
    });

    expect(model.current).toMatchObject({
      state: "complete",
      totalTokens: null,
      cost: null
    });
    expect(model.session).toMatchObject({
      runCount: 1,
      totalCost: null,
      costedRunCount: 0,
      coverage: "none"
    });
  });
});

function refinementTrace(): WorkflowRuntimeTraceEvent[] {
  return [
    traceEvent({
      event_type: "generation_input",
      payload: {
        generation_input: {
          request: {
            stream: true
          }
        }
      },
      sequence: 0
    }),
    traceEvent({
      event_type: "prompt_input",
      payload: {
        code: "prompt_inputs_prepared",
        prompt_input: {
          user_input: {
            artifact_refinement: {
              artifact_id: "artifact-1",
              instruction: "Strengthen the motion."
            }
          }
        }
      },
      sequence: 1
    }),
    traceEvent({
      event_type: "artifact_critique",
      payload: {
        code: "artifact_scored"
      },
      sequence: 2
    }),
    traceEvent({
      event_type: "review_passed",
      payload: {
        code: "review_passed"
      },
      sequence: 3
    }),
    traceEvent({
      event_type: "final",
      payload: {
        answer: "Refinement complete.",
        telemetry: {
          cost: {
            currency: "USD",
            total_usd: 0.006
          },
          execution: {
            fallback_cost_usd: 0.0004,
            fallback_paths: [
              {
                label: "Regional fallback",
                reason: "Primary unavailable"
              }
            ],
            generation_mode: "streaming",
            request_duration_ms: 1450,
            retry_cost_usd: 0.001,
            retry_count: 1,
            streaming: true,
            streaming_status: "completed"
          },
          pricing: {
            input_usd_per_million_tokens: 1,
            output_usd_per_million_tokens: 4
          },
          provider: {
            model: "gpt-5-mini",
            name: "openai",
            request_id: "request-refine"
          },
          token_usage: {
            input_tokens: 1200,
            output_tokens: 500,
            total_tokens: 1700
          }
        },
        workflow: {
          artifact_count: 2,
          artifact_critique_count: 2,
          completed_steps: ["generation", "artifact_critique", "review"],
          current_step: "finalization",
          image_reference_count: 0,
          image_references: [],
          phase: "completed",
          preview_artifact_count: 1,
          recommended_artifact_id: "artifact-2",
          refinement_count: 1,
          review_outcome: "passed",
          review_reasons: [],
          skipped_steps: [],
          status: "completed",
          step: "finalization"
        }
      },
      sequence: 4
    })
  ];
}

function runRecord(
  overrides: Partial<CreativeCostRunRecord> & { id: string }
): CreativeCostRunRecord {
  const { id, ...rest } = overrides;

  return {
    id,
    status: "complete",
    completedAt: "2026-06-09T12:00:00Z",
    kind: "generation",
    providerName: "openai",
    modelName: "gpt-5-mini",
    generationMode: "streaming",
    pricing: {
      currency: "USD",
      inputCostPerMillionTokens: 1,
      outputCostPerMillionTokens: 4
    },
    inputTokens: 800,
    outputTokens: 300,
    totalTokens: 1100,
    cost: 0.004,
    currency: "USD",
    costSource: "pricing_metadata",
    durationMs: 900,
    retryCount: 0,
    fallbackCount: 0,
    retryCost: null,
    fallbackCost: null,
    artifactCount: 1,
    refinementCount: 0,
    critiqueCount: 0,
    reviewCount: 0,
    ...rest
  };
}

function traceEvent({
  event_type,
  payload,
  sequence
}: {
  event_type: WorkflowRuntimeTraceEvent["event"]["event_type"];
  payload: Record<string, unknown>;
  sequence: number;
}): WorkflowRuntimeTraceEvent {
  const receivedAtMs = Date.parse("2026-06-09T10:00:00Z") + sequence * 500;

  return {
    event: {
      event_type,
      payload: {
        ...payload,
        emitted_at: new Date(receivedAtMs).toISOString()
      },
      sequence
    },
    receivedAt: new Date(receivedAtMs + 25).toISOString(),
    receivedAtMs: receivedAtMs + 25
  };
}
