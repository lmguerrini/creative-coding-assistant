import type { AssistantMessage } from "./assistant-client";

export type ConversationEntryPhase =
  | "complete"
  | "connecting"
  | "thinking"
  | "streaming"
  | "error"
  | "fallback";

export type ConversationEntry = AssistantMessage & {
  activity: string | null;
  id: string;
  pending: boolean;
  phase: ConversationEntryPhase;
};

export function buildConversationEntries(
  messages: AssistantMessage[],
  createId: () => string
): ConversationEntry[] {
  return messages.map((message) => ({
    ...message,
    activity: null,
    id: createId(),
    pending: false,
    phase: "complete"
  }));
}

export function toPersistedConversation(
  entries: ConversationEntry[]
): AssistantMessage[] {
  return entries
    .filter((entry) => !entry.pending)
    .map(({ content, role, time }) => ({ content, role, time }));
}

export function getConversationPhaseBadge(phase: ConversationEntryPhase) {
  switch (phase) {
    case "connecting":
      return "Connecting";
    case "thinking":
      return "Thinking";
    case "streaming":
      return "Live";
    case "error":
      return "Error";
    case "fallback":
      return "Fallback";
    default:
      return "Complete";
  }
}

export function getConversationPhasePlaceholder(phase: ConversationEntryPhase) {
  switch (phase) {
    case "connecting":
      return "Opening the live response...";
    case "thinking":
      return "Thinking through the request...";
    case "streaming":
      return "Generating the first response tokens...";
    case "error":
      return "The live response stopped before completion.";
    case "fallback":
      return "Switching to the local fallback response.";
    default:
      return "";
  }
}

export function getComposerStatusLabel({
  isStreaming,
  isReady,
  phase,
  streamError
}: {
  isReady: boolean;
  isStreaming: boolean;
  phase: ConversationEntryPhase | null;
  streamError: string | null;
}) {
  if (isStreaming) {
    if (phase === "streaming") {
      return "Generating response";
    }

    if (phase === "connecting") {
      return "Opening live response";
    }

    return "Thinking through the request";
  }

  if (streamError) {
    return "Stream interrupted";
  }

  if (isReady) {
    return "Ready to generate";
  }

  return "Type a prompt to begin";
}
