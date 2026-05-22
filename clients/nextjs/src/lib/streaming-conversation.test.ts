import { describe, expect, it } from "vitest";
import {
  buildConversationEntries,
  getComposerStatusLabel,
  getConversationPhaseBadge,
  getConversationPhasePlaceholder,
  toPersistedConversation
} from "./streaming-conversation";

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
    expect(getConversationPhaseBadge("streaming")).toBe("Live");
    expect(getConversationPhasePlaceholder("thinking")).toBe(
      "Thinking through the request..."
    );
    expect(
      getComposerStatusLabel({
        isReady: false,
        isStreaming: true,
        phase: "streaming",
        streamError: null
      })
    ).toBe("Generating response");
    expect(
      getComposerStatusLabel({
        isReady: false,
        isStreaming: false,
        phase: null,
        streamError: "offline"
      })
    ).toBe("Stream interrupted");
  });
});
