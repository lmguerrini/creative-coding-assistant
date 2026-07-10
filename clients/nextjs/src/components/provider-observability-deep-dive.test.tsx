import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { buildProviderTelemetryModel } from "@/lib/provider-telemetry";
import type { WorkflowRuntimeTraceEvent } from "@/lib/workflow-runtime";
import { ProviderObservabilityDeepDive } from "./provider-observability-deep-dive";

describe("ProviderObservabilityDeepDive", () => {
  it("renders provider identity, tokens, cost, execution state, and recovery details", () => {
    render(
      <ProviderObservabilityDeepDive
        telemetry={buildProviderTelemetryModel([
          traceEvent({
            event_type: "generation_input",
            payload: {
              generation_input: {
                request: {
                  route: "general",
                  stream: true
                },
                messages: [{ role: "user" }, { role: "system" }]
              }
            },
            sequence: 0
          }),
          traceEvent({
            event_type: "final",
            payload: {
              answer: "Complete.",
              telemetry: {
                cost: {
                  currency: "USD",
                  total_usd: 0.0042
                },
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
                  request_duration_ms: 920,
                  request_started_at: "2026-05-24T10:00:00Z",
                  retry_count: 1,
                  retry_events: [
                    {
                      attempt: 1,
                      reason: "Transient timeout",
                      status: "recovered"
                    }
                  ],
                  streaming: true,
                  streaming_status: "completed",
                  warnings: [
                    {
                      code: "capacity_warning",
                      message: "Provider capacity is constrained."
                    }
                  ]
                },
                finish_reason: "stop",
                provider: {
                  model: "gpt-5-mini",
                  name: "openai",
                  request_id: "req_123",
                  response_id: "resp_123"
                },
                parameters: {
                  max_output_tokens: 512,
                  temperature: 0.35,
                  top_p: 0.9
                },
                token_usage: {
                  input_tokens: 1200,
                  output_tokens: 300,
                  total_tokens: 1500
                }
              }
            },
            sequence: 1
          })
        ])}
      />
    );

    const deepDive = screen.getByRole("group", {
      name: "Provider observability deep dive"
    });
    expect(within(deepDive).getByText("openai / gpt-5-mini")).toBeVisible();
    expect(within(deepDive).getByText("Streaming generation")).toBeVisible();
    expect(within(deepDive).getByText("920 ms")).toBeVisible();
    expect(within(deepDive).getByText("1 retry")).toBeVisible();
    expect(within(deepDive).getByText("Stream completed")).toBeVisible();

    const tokens = within(deepDive).getByRole("region", {
      name: "Provider token transparency"
    });
    expect(within(tokens).getByText("1,200")).toBeVisible();
    expect(within(tokens).getByText("300")).toBeVisible();
    expect(within(tokens).getByText("1,500")).toBeVisible();
    expect(within(tokens).getByText("$0.0042")).toBeVisible();

    expect(
      within(deepDive).getByText("Provider capacity is constrained.")
    ).toBeVisible();
    expect(within(deepDive).getByText("Transient timeout")).toBeVisible();
    expect(within(deepDive).getByText("Primary region saturated")).toBeVisible();

    const configuration = within(deepDive).getByRole("region", {
      name: "Agent configuration"
    });
    expect(within(configuration).getByText("general")).toBeVisible();
    expect(within(configuration).getByText("2")).toBeVisible();
    expect(within(configuration).getByText("0.35")).toBeVisible();
    expect(within(configuration).getByText("0.9")).toBeVisible();
    expect(within(configuration).getByText("512")).toBeVisible();
    expect(
      within(configuration).getByText(
        "Sampling parameters are provider-reported for this request."
      )
    ).toBeVisible();
  });

  it("renders legacy sessions without inventing unavailable provider metadata", () => {
    render(
      <ProviderObservabilityDeepDive
        telemetry={buildProviderTelemetryModel([])}
      />
    );

    const deepDive = screen.getByRole("group", {
      name: "Provider observability deep dive"
    });
    expect(
      within(deepDive).getByText("Provider pending / Model pending")
    ).toBeVisible();
    expect(within(deepDive).getByText("Generation mode unavailable")).toBeVisible();
    expect(within(deepDive).getByText("Duration unavailable")).toBeVisible();
    expect(
      within(deepDive).getByText("Provider retry metadata unavailable.")
    ).toBeVisible();
    expect(within(deepDive).getByText("No provider fallback path observed.")).toBeVisible();
    expect(within(deepDive).getAllByText("Not published")).toHaveLength(2);
  });
});

function traceEvent({
  event_type,
  payload,
  sequence
}: {
  event_type: WorkflowRuntimeTraceEvent["event"]["event_type"];
  payload: Record<string, unknown>;
  sequence: number;
}): WorkflowRuntimeTraceEvent {
  const receivedAtMs = Date.parse("2026-05-24T10:00:00Z") + sequence * 1000;

  return {
    event: {
      event_type,
      payload: {
        ...payload,
        emitted_at: new Date(receivedAtMs).toISOString()
      },
      sequence
    },
    receivedAt: new Date(receivedAtMs + 50).toISOString(),
    receivedAtMs: receivedAtMs + 50
  };
}
