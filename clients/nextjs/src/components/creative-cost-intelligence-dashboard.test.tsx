import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { CreativeCostIntelligenceModel } from "@/lib/creative-cost-intelligence";
import { CreativeCostIntelligenceDashboard } from "./creative-cost-intelligence-dashboard";

describe("CreativeCostIntelligenceDashboard", () => {
  it("renders a clean first-run state without fake cost", () => {
    render(<CreativeCostIntelligenceDashboard intelligence={emptyModel()} />);

    const dashboard = screen.getByRole("group", {
      name: "Creative cost intelligence dashboard"
    });
    expect(within(dashboard).getByText("Session cost unavailable")).toBeVisible();
    expect(within(dashboard).getByText("Awaiting draft")).toBeVisible();
    expect(within(dashboard).getByText("No completed run")).toBeVisible();
    expect(within(dashboard).getByText("No cost coverage")).toBeVisible();
    expect(within(dashboard).queryByText("$0.0000")).not.toBeInTheDocument();
  });

  it("renders estimate, latest run, session averages, and creative context", () => {
    render(
      <CreativeCostIntelligenceDashboard intelligence={populatedModel()} />
    );

    const dashboard = screen.getByRole("group", {
      name: "Creative cost intelligence dashboard"
    });
    const estimate = within(dashboard).getByRole("region", {
      name: "Pre-generation cost estimate"
    });
    const latest = within(dashboard).getByRole("region", {
      name: "Latest generation cost"
    });
    const session = within(dashboard).getByRole("region", {
      name: "Session cost summary"
    });

    expect(within(estimate).getByText("$0.0010 - $0.0040")).toBeVisible();
    expect(within(estimate).getByText("2 requested artifacts")).toBeVisible();
    expect(within(latest).getByText("$0.0060")).toBeVisible();
    expect(within(latest).getByText("1,700 tokens")).toBeVisible();
    expect(within(latest).getByText("1 retry / $0.0010")).toBeVisible();
    expect(within(latest).getByText("1 fallback / $0.0004")).toBeVisible();
    expect(within(latest).getByText("2 artifacts")).toBeVisible();
    expect(within(latest).getByText("1 refinement")).toBeVisible();
    expect(within(latest).getByText("2 critiques / 1 review")).toBeVisible();
    expect(within(session).getByText("2 completed runs")).toBeVisible();
    expect(within(session).getByText("$0.0045")).toBeVisible();
    expect(within(session).getByText("$0.0030")).toBeVisible();
    expect(within(session).getByText("3 critiques / 2 reviews")).toBeVisible();
  });
});

function emptyModel(): CreativeCostIntelligenceModel {
  return {
    estimate: {
      state: "empty",
      providerName: null,
      modelName: null,
      generationMode: "unknown",
      promptTokens: null,
      contextTokens: null,
      requestedArtifactCount: 1,
      includesReview: false,
      includesRefinement: false,
      inputTokenRange: null,
      outputTokenRange: null,
      costRange: null,
      currency: "USD",
      detail: "Add a prompt to preview the likely generation scope.",
      assumptions: []
    },
    current: {
      state: "idle",
      providerName: null,
      modelName: null,
      generationMode: "unknown",
      inputTokens: null,
      outputTokens: null,
      totalTokens: null,
      cost: null,
      currency: "USD",
      costSource: "unavailable",
      durationMs: null,
      retryCount: null,
      fallbackCount: 0,
      retryCost: null,
      fallbackCost: null,
      artifactCount: 0,
      refinementCount: 0,
      critiqueCount: 0,
      reviewCount: 0
    },
    session: {
      runCount: 0,
      generationCount: 0,
      refinementCount: 0,
      inputTokens: null,
      outputTokens: null,
      totalTokens: null,
      tokenedRunCount: 0,
      artifactCount: 0,
      critiqueCount: 0,
      reviewCount: 0,
      totalCost: null,
      currency: "USD",
      costedRunCount: 0,
      coverage: "none",
      averagePerGeneration: null,
      averagePerArtifact: null
    }
  };
}

function populatedModel(): CreativeCostIntelligenceModel {
  return {
    estimate: {
      state: "ready",
      providerName: "openai",
      modelName: "gpt-5-mini",
      generationMode: "streaming",
      promptTokens: 40,
      contextTokens: 180,
      requestedArtifactCount: 2,
      includesReview: true,
      includesRefinement: false,
      inputTokenRange: [175, 445],
      outputTokenRange: [1140, 3040],
      costRange: [0.001, 0.004],
      currency: "USD",
      detail: "Bounded estimate from known pricing.",
      assumptions: [
        "Token counts are approximate.",
        "2 requested artifacts",
        "Review overhead included."
      ]
    },
    current: {
      state: "complete",
      providerName: "openai",
      modelName: "gpt-5-mini",
      generationMode: "streaming",
      inputTokens: 1200,
      outputTokens: 500,
      totalTokens: 1700,
      cost: 0.006,
      currency: "USD",
      costSource: "provider_reported",
      durationMs: 1450,
      retryCount: 1,
      fallbackCount: 1,
      retryCost: 0.001,
      fallbackCost: 0.0004,
      artifactCount: 2,
      refinementCount: 1,
      critiqueCount: 2,
      reviewCount: 1
    },
    session: {
      runCount: 2,
      generationCount: 2,
      refinementCount: 1,
      inputTokens: 2100,
      outputTokens: 900,
      totalTokens: 3000,
      tokenedRunCount: 2,
      artifactCount: 3,
      critiqueCount: 3,
      reviewCount: 2,
      totalCost: 0.009,
      currency: "USD",
      costedRunCount: 2,
      coverage: "complete",
      averagePerGeneration: 0.0045,
      averagePerArtifact: 0.003
    }
  };
}
