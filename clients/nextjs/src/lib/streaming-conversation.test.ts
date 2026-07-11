import { describe, expect, it } from "vitest";
import {
  buildConversationEntries,
  getComposerStatusLabel,
  getConversationPhaseBadge,
  getConversationPhasePlaceholder,
  toPersistedConversation
} from "./streaming-conversation";
import { createWorkstationError } from "./workstation-errors";

describe("streaming conversation helpers", () => {
  it("hydrates persisted messages into complete conversation entries", () => {
    const entries = buildConversationEntries(
      [
        {
          role: "assistant",
          time: "10:15",
          content: "Ready."
        }
      ],
      () => "message-1"
    );

    expect(entries).toEqual([
      {
        role: "assistant",
        time: "10:15",
        content: "Ready.",
        activity: null,
        id: "message-1",
        pending: false,
        phase: "complete"
      }
    ]);
  });

  it("restores the terminal assistant card from the persisted product outcome", () => {
    const entries = buildConversationEntries(
      [
        { role: "assistant", time: "10:15", content: "Earlier result." },
        { role: "user", time: "10:16", content: "Create a p5 study." },
        { role: "assistant", time: "10:17", content: "Latest result." }
      ],
      () => "message-id",
      {
        status: "Completed",
        currentNode: "finalization",
        currentStep: "Finalization",
        steps: [],
        productOutcome: {
          orchestration_status: "COMPLETED",
          provider_status: "COMPLETED",
          generation_status: "COMPLETED",
          deliverable_status: "USABLE",
          artifact_extraction_status: "EXTRACTED",
          artifact_runnability: "UNSUPPORTED",
          preview_status: "UNAVAILABLE",
          runtime_health: "NOT_AVAILABLE",
          product_outcome: "PARTIAL",
          summary: "A usable artifact was produced, but live preview is unavailable.",
          recovery_action: "Open Code to use the artifact."
        }
      }
    );

    expect(entries[0]).toMatchObject({ activity: null, phase: "complete" });
    expect(entries[2]).toMatchObject({
      activity: "A usable artifact was produced, but live preview is unavailable.",
      phase: "partial"
    });
  });

  it("keeps a restored in-progress card aligned with its workflow phase", () => {
    const entries = buildConversationEntries(
      [{ role: "assistant", time: "10:15", content: "Working." }],
      () => "message-1",
      {
        status: "Running",
        currentNode: "retrieval",
        currentStep: "Retrieval",
        steps: [],
        productOutcome: {
          orchestration_status: "RUNNING",
          provider_status: "PENDING",
          generation_status: "PENDING",
          deliverable_status: "UNKNOWN",
          artifact_extraction_status: "UNKNOWN",
          artifact_runnability: "UNKNOWN",
          preview_status: "UNKNOWN",
          runtime_health: "UNKNOWN",
          product_outcome: "IN_PROGRESS",
          summary: "Retrieving sources for the requested output.",
          recovery_action: ""
        }
      }
    );

    expect(entries[0]).toMatchObject({
      activity: "Retrieving sources for the requested output.",
      phase: "retrieving"
    });
  });

  it("filters pending entries before persistence", () => {
    expect(
      toPersistedConversation([
        {
          role: "user",
          time: "10:15",
          content: "Prompt",
          activity: null,
          id: "message-1",
          pending: false,
          phase: "complete"
        },
        {
          role: "assistant",
          time: "10:16",
          content: "Thinking through the request...",
          activity: "Route selected.",
          id: "message-2",
          pending: true,
          phase: "thinking"
        }
      ])
    ).toEqual([
      {
        role: "user",
        time: "10:15",
        content: "Prompt"
      }
    ]);
  });

  it("returns polished phase badges, placeholders, and composer labels", () => {
    expect(getConversationPhaseBadge("streaming")).toBe("Generating");
    expect(getConversationPhasePlaceholder("planning")).toBe(
      "Planning the requested work..."
    );
    expect(
      getComposerStatusLabel({
        isReady: false,
        isStreaming: true,
        phase: "reviewing",
        streamError: null
      })
    ).toBe("Reviewing");
    expect(
      getComposerStatusLabel({
        isReady: false,
        isStreaming: false,
        phase: null,
        streamError: createWorkstationError({
          type: "assistant_stream_unavailable",
          category: "stream",
          subsystem: "assistant_stream",
          userMessage: "The backend stream is unavailable.",
          recoverable: true,
          suggestedAction: "Retry the request from the composer.",
          retryLabel: "Send prompt again"
        })
      })
    ).toBe("Stream interrupted");
  });
});
