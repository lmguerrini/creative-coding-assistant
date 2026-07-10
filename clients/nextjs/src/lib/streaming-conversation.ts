import type { AssistantMessage } from "./assistant-client";
import type { WorkstationError } from "./workstation-errors";

export type ConversationEntryPhase =
  | "complete"
  | "connecting"
  | "thinking"
  | "streaming"
  | "planning"
  | "retrieving"
  | "generating"
  | "reviewing"
  | "refining"
  | "finalizing"
  | "completed"
  | "partial"
  | "failed"
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
    case "thinking":
    case "planning":
      return "Planning";
    case "retrieving":
      return "Retrieving";
    case "generating":
    case "streaming":
      return "Generating";
    case "reviewing":
      return "Reviewing";
    case "refining":
      return "Refining";
    case "finalizing":
      return "Finalizing";
    case "partial":
    case "fallback":
      return "Partial";
    case "failed":
    case "error":
      return "Failed";
    case "completed":
    default:
      return "Completed";
  }
}

export function getConversationPhasePlaceholder(phase: ConversationEntryPhase) {
  switch (phase) {
    case "connecting":
    case "thinking":
    case "planning":
      return "Planning the requested work...";
    case "retrieving":
      return "Retrieving relevant context...";
    case "generating":
    case "streaming":
      return "Generating the requested artifact...";
    case "reviewing":
      return "Reviewing the generated output...";
    case "refining":
      return "Refining the generated output...";
    case "finalizing":
      return "Finalizing the product result...";
    case "partial":
      return "A partial result is available.";
    case "failed":
      return "The requested output could not be completed.";
    case "error":
      return "The live response stopped before completion.";
    case "fallback":
      return "Switching to a local draft.";
    default:
      return "";
  }
}

export function getComposerStatusLabel({
  activityLabel,
  isStreaming,
  isReady,
  phase,
  streamError
}: {
  activityLabel?: string | null;
  isReady: boolean;
  isStreaming: boolean;
  phase: ConversationEntryPhase | null;
  streamError: WorkstationError | null;
}) {
  if (isStreaming) {
    if (activityLabel) {
      return activityLabel;
    }

    return getConversationPhaseBadge(phase ?? "planning");
  }

  if (streamError) {
    return "Stream interrupted";
  }

  if (isReady) {
    return "Ready to generate";
  }

  return "Type a prompt to begin";
}
