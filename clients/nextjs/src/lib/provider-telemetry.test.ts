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
  sequence,
  telemetry,
  text
}: {
  answer?: string;
  at: string;
  code?: string;
  event_type: WorkflowRuntimeTraceEvent["event"]["event_type"];
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
        ...(telemetry ? { telemetry } : {}),
        ...(text ? { text } : {}),
        emitted_at: at
      }
    },
    receivedAt: new Date(receivedAtMs).toISOString(),
    receivedAtMs
  };
}
