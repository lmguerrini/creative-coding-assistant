import type {
  AssistantMessage,
  AssistantWorkspaceSnapshot
} from "./assistant-client";
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
  createId: () => string,
  workflow?: AssistantWorkspaceSnapshot["workflow"]
): ConversationEntry[] {
  const terminalAssistantIndex = findTerminalAssistantIndex(messages);
  const productOutcome = workflow?.productOutcome;

  return messages.map((message, index) => {
    const restoresOutcome =
      index === terminalAssistantIndex ? productOutcome ?? null : null;
    const restoresTerminalOutcome =
      restoresOutcome?.product_outcome !== "IN_PROGRESS" ? restoresOutcome : null;
    const restoresInProgressOutcome =
      restoresOutcome?.product_outcome === "IN_PROGRESS" ? restoresOutcome : null;

    return {
      ...message,
      activity: restoresOutcome?.summary ?? null,
      id: createId(),
      pending: false,
      phase: restoresTerminalOutcome
        ? conversationPhaseForProductOutcome(restoresTerminalOutcome.product_outcome)
        : restoresInProgressOutcome
          ? conversationPhaseForWorkflowNode(workflow?.currentNode)
        : "complete"
    };
  });
}

function findTerminalAssistantIndex(messages: AssistantMessage[]) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "assistant") {
      return index;
    }
  }
  return -1;
}

function conversationPhaseForProductOutcome(
  productOutcome: NonNullable<
    AssistantWorkspaceSnapshot["workflow"]["productOutcome"]
  >["product_outcome"]
): ConversationEntryPhase {
  switch (productOutcome) {
    case "PARTIAL":
      return "partial";
    case "FAILURE":
      return "failed";
    case "SUCCESS":
      return "completed";
    case "IN_PROGRESS":
    default:
      return "complete";
  }
}

function conversationPhaseForWorkflowNode(
  currentNode: AssistantWorkspaceSnapshot["workflow"]["currentNode"] | undefined
): ConversationEntryPhase {
  switch (currentNode) {
    case "memory":
    case "retrieval":
    case "context_assembly":
      return "retrieving";
    case "generation":
      return "generating";
    case "artifact_extraction":
    case "preview_preparation":
    case "artifact_critique":
    case "review":
      return "reviewing";
    case "refinement":
      return "refining";
    case "finalization":
      return "finalizing";
    default:
      return "planning";
  }
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
      return "Failure";
    case "completed":
    default:
      return "Success";
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
