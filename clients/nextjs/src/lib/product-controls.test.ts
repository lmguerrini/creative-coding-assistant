import { describe, expect, it } from "vitest";

import {
  buildGenerationControls,
  createFeedbackSignal,
  privacyContract,
  selectPersonalizationContext
} from "./product-controls";

describe("V9.7 product controls", () => {
  it("maps the compact creativity choices to bounded provider requests", () => {
    expect(buildGenerationControls("controlled").requestedTemperature).toBe(0.35);
    expect(buildGenerationControls("balanced").requestedTemperature).toBe(0.7);
    expect(buildGenerationControls("exploratory").requestedTemperature).toBe(1);
    expect(buildGenerationControls("balanced").providerParameterState).toBe(
      "pending_provider_confirmation"
    );
  });

  it("selects only bounded, relevant preference signals", () => {
    const palette = createFeedbackSignal({
      artifact: null,
      comment: "Prefer a soft blue palette with gentle motion.",
      creativity: "balanced",
      id: "palette",
      sentiment: "positive",
      sessionId: "session-1",
      workflowMode: "auto",
      createdAt: "2026-07-11T10:00:00.000Z"
    });
    const exportSignal = createFeedbackSignal({
      artifact: null,
      comment: "I need an export handoff.",
      creativity: "controlled",
      id: "export",
      sentiment: "positive",
      sessionId: "session-1",
      workflowMode: "single_agent",
      createdAt: "2026-07-11T09:00:00.000Z"
    });

    const selected = selectPersonalizationContext({
      enabled: true,
      prompt: "Create a blue animated particle field.",
      signals: [exportSignal, palette]
    });

    expect(selected.selectedSignalIds).toEqual(["palette"]);
    expect(selected.categories).toContain("palette");
    expect(selected.categories).toContain("motion");
  });

  it("records feedback provenance without claiming provider parameter application", () => {
    const signal = createFeedbackSignal({
      artifact: null,
      comment: "Keep this palette direction.",
      creativity: "exploratory",
      id: "provenance",
      promptExcerpt: "Create a long prompt that should be retained only as a bounded excerpt.",
      providerModel: "gpt-5",
      providerName: "openai",
      sentiment: "positive",
      sessionId: "session-1",
      workflowMode: "auto"
    });

    expect(signal.requestedTemperature).toBe(1);
    expect(signal.parameterApplication).toBe("requested_not_confirmed");
    expect(signal.providerName).toBe("openai");
    expect(signal.promptExcerpt).toContain("Create a long prompt");
  });

  it("does not include signals while personalization is disabled", () => {
    const selected = selectPersonalizationContext({
      enabled: false,
      prompt: "Create an interactive field.",
      signals: []
    });

    expect(selected.signalCount).toBe(0);
    expect(selected.detail).toContain("off");
  });

  it("states local and opt-in privacy boundaries without a hidden network action", () => {
    expect(privacyContract.map((item) => item.label)).toEqual(
      expect.arrayContaining([
        "Workspace sessions and artifacts",
        "Knowledge Base updates",
        "LangSmith traces and evaluation artifacts"
      ])
    );
    expect(privacyContract.map((item) => item.boundary).join(" ").toLowerCase()).toContain(
      "never silently"
    );
  });
});
