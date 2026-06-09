import { describe, expect, it } from "vitest";
import {
  buildProviderTelemetryModel,
  type ProviderTelemetryModel
} from "./provider-telemetry";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

describe("provider telemetry model", () => {
  it("aggregates provider usage, cost, and generation timing from stream traces", () => {
    const telemetry = buildProviderTelemetryModel([
      traceEvent({
        at: "2026-05-24T10:00:00Z",
        code: "generation_input_prepared",
        event_type: "generation_input",
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
        answer: "Draft complete.",
        at: "2026-05-24T10:00:03Z",
        event_type: "final",
        sequence: 2,
        telemetry: {
          provider: {
            name: "openai",
            model: "gpt-5-mini",
            response_id: "resp_123"
          },
          token_usage: {
            input_tokens: 1200,
            output_tokens: 300,
            total_tokens: 1500,
            reasoning_tokens: 12
          },
          pricing: {
            input_usd_per_million_tokens: 0.25,
            output_usd_per_million_tokens: 2
          }
        }
      })
    ]);

    expect(telemetry.status).toBe("complete");
    expect(telemetry.provider).toMatchObject({
      name: "openai",
      model: "gpt-5-mini",
      responseId: "resp_123"
    });
    expect(telemetry.tokenUsage).toMatchObject({
      inputTokens: 1200,
      outputTokens: 300,
      totalTokens: 1500,
      reasoningTokens: 12,
      source: "provider"
    });
    expect(telemetry.cost.source).toBe("pricing_metadata");
    expect(telemetry.cost.totalCost).toBeCloseTo(0.0009);
    expect(telemetry.timing.timeToFirstTokenMs).toBe(1000);
    expect(telemetry.timing.generationDurationMs).toBe(3000);
    expect(telemetry.timing.firstEventLatencyMs).toBe(50);
    expect(telemetry.stream).toMatchObject({
      eventCount: 3,
      tokenDeltaCount: 1,
      streamedCharacterCount: 6
    });
    expect(completeLifecycleLabels(telemetry)).toEqual([
      "Request",
      "Generation input",
      "First token",
      "Completion"
    ]);
  });

  it("keeps cost unavailable when provider usage or pricing is absent", () => {
    const telemetry = buildProviderTelemetryModel([
      traceEvent({
        at: "2026-05-24T10:00:00Z",
        event_type: "status",
        sequence: 0
      }),
      traceEvent({
        at: "2026-05-24T10:00:01Z",
        event_type: "token_delta",
        sequence: 1,
        text: "Unmetered stream"
      })
    ]);

    expect(telemetry.status).toBe("streaming");
    expect(telemetry.tokenUsage.source).toBe("unavailable");
    expect(telemetry.tokenUsage.totalTokens).toBeNull();
    expect(telemetry.cost.source).toBe("unavailable");
    expect(telemetry.cost.totalCost).toBeNull();
    expect(telemetry.stream.streamedCharacterCount).toBe(16);
    expect(telemetry.summary.tokenLabel).toBe("Usage pending");
    expect(telemetry.summary.costLabel).toBe("Cost pending");
  });

  it("models provider execution mode, retries, warnings, and fallback paths", () => {
    const telemetry = buildProviderTelemetryModel([
      traceEvent({
        at: "2026-05-24T10:00:00Z",
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
        at: "2026-05-24T10:00:00.200Z",
        event_type: "token_delta",
        sequence: 1,
        telemetry: {
          execution: {
            generation_mode: "streaming",
            request_duration_ms: 200,
            request_started_at: "2026-05-24T10:00:00Z",
            retry_count: 1,
            retry_events: [
              {
                attempt: 1,
                reason: "upstream timeout",
                status: "recovered"
              }
            ],
            streaming: true,
            streaming_status: "active"
          },
          provider: {
            name: "openai",
            model: "gpt-5-mini"
          }
        },
        text: "Draft "
      }),
      traceEvent({
        answer: "Draft complete.",
        at: "2026-05-24T10:00:01.280Z",
        event_type: "final",
        sequence: 2,
        telemetry: {
          execution: {
            fallback_paths: [
              {
                label: "Regional fallback",
                reason: "Primary region saturated",
                source: "us-east",
                target: "eu-west"
              }
            ],
            generation_mode: "streaming",
            request_completed_at: "2026-05-24T10:00:01.280Z",
            request_duration_ms: 1280,
            request_started_at: "2026-05-24T10:00:00Z",
            retry_count: 1,
            streaming: true,
            streaming_status: "completed",
            warnings: [
              {
                code: "rate_limit_near",
                message: "Provider capacity is near its configured limit."
              }
            ]
          },
          finish_reason: "length",
          provider: {
            name: "openai",
            model: "gpt-5-mini",
            request_id: "req_123",
            response_id: "resp_123"
          }
        }
      })
    ]);

    expect(telemetry.execution).toMatchObject({
      generationMode: "streaming",
      streamingEnabled: true,
      streamingState: "completed",
      requestDurationMs: 1280,
      retryCount: 1,
      finishReason: "length"
    });
    expect(telemetry.execution.retryEvents).toEqual([
      expect.objectContaining({
        attempt: 1,
        reason: "upstream timeout",
        status: "recovered"
      })
    ]);
    expect(telemetry.execution.fallbackPaths).toEqual([
      expect.objectContaining({
        label: "Regional fallback",
        source: "us-east",
        target: "eu-west"
      })
    ]);
    expect(telemetry.execution.warnings.map((warning) => warning.code)).toEqual([
      "rate_limit_near",
      "finish_reason_length"
    ]);
    expect(telemetry.summary).toMatchObject({
      generationModeLabel: "Streaming generation",
      requestDurationLabel: "1.3 s",
      retryLabel: "1 provider retry",
      streamingStatusLabel: "Stream completed"
    });
  });

  it("surfaces structured provider errors without treating unrelated errors as provider failures", () => {
    const telemetry = buildProviderTelemetryModel([
      traceEvent({
        at: "2026-05-24T10:00:00Z",
        event_type: "generation_input",
        payload: {
          generation_input: {
            request: {
              stream: false
            }
          }
        },
        sequence: 0
      }),
      traceEvent({
        at: "2026-05-24T10:00:00.050Z",
        event_type: "error",
        payload: {
          category: "stream",
          code: "provider_unavailable",
          message: "Provider unavailable.",
          recoverable: true,
          subsystem: "generation_provider",
          telemetry: {
            execution: {
              errors: [
                {
                  code: "provider_unavailable",
                  message: "Provider unavailable."
                }
              ],
              generation_mode: "non_streaming",
              request_duration_ms: 50,
              retry_count: 0,
              streaming: false,
              streaming_status: "failed"
            }
          }
        },
        sequence: 1
      })
    ]);

    expect(telemetry.status).toBe("error");
    expect(telemetry.execution).toMatchObject({
      generationMode: "non_streaming",
      streamingEnabled: false,
      streamingState: "failed",
      requestDurationMs: 50,
      retryCount: 0
    });
    expect(telemetry.execution.errors).toEqual([
      expect.objectContaining({
        code: "provider_unavailable",
        recoverable: true,
        source: "stream"
      })
    ]);
  });

  it("recognizes real generation fallback transitions and keeps legacy telemetry safe", () => {
    const telemetry = buildProviderTelemetryModel([
      traceEvent({
        at: "2026-05-24T10:00:00Z",
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
        at: "2026-05-24T10:00:00.100Z",
        event_type: "node_completed",
        payload: {
          decision_reason: "generation_unavailable",
          transition_source: "generation",
          transition_target: "artifact_extraction"
        },
        sequence: 1
      }),
      traceEvent({
        answer: "Local fallback response.",
        at: "2026-05-24T10:00:00.200Z",
        event_type: "final",
        sequence: 2
      })
    ]);

    expect(telemetry.status).toBe("complete");
    expect(telemetry.execution.generationMode).toBe("streaming");
    expect(telemetry.execution.requestDurationMs).toBeNull();
    expect(telemetry.execution.retryCount).toBeNull();
    expect(telemetry.execution.fallbackPaths).toEqual([
      expect.objectContaining({
        label: "Generation fallback",
        source: "generation",
        target: "artifact_extraction"
      })
    ]);
    expect(telemetry.summary.requestDurationLabel).toBe("Duration unavailable");
    expect(telemetry.summary.retryLabel).toBe("Retry metadata unavailable");
  });
});

function completeLifecycleLabels(telemetry: ProviderTelemetryModel) {
  return telemetry.lifecycle
    .filter((step) => step.state === "complete")
    .map((step) => step.label);
}

function traceEvent({
  answer,
  at,
  code,
  event_type,
  payload,
  sequence,
  telemetry,
  text
}: {
  answer?: string;
  at: string;
  code?: string;
  event_type: WorkflowRuntimeTraceEvent["event"]["event_type"];
  payload?: Record<string, unknown>;
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
        ...payload,
        ...(answer ? { answer } : {}),
        ...(code ? { code } : {}),
        ...(telemetry ? { telemetry } : {}),
        ...(text ? { text } : {}),
        emitted_at: at
      }
    },
    receivedAt: new Date(receivedAtMs).toISOString(),
    receivedAtMs
  };
}
