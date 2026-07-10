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
