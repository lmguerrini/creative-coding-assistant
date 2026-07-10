import type { AssistantStreamEvent } from "./assistant-stream";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

export type ConversationContextDiagnosticState =
  | "observed"
  | "not_injected"
  | "unavailable";

export type ConversationContextDiagnostic = {
  id:
    | "visible_history"
    | "model_context"
    | "memory"
    | "summary"
    | "retrieval"
    | "truncation"
    | "token_allocation";
  label: string;
  value: string;
  detail: string;
  state: ConversationContextDiagnosticState;
};

export type ConversationContextModel = {
  source: "stream" | "none";
  diagnostics: ConversationContextDiagnostic[];
  summary: string;
};

type ContextEvidence = {
  recentTurnCount: number | null;
  recentTurnLimit: number | null;
  runningSummaryCoveredTurnCount: number | null;
  projectMemoryCount: number | null;
  retrievalChunkCount: number | null;
};

const emptyContextEvidence: ContextEvidence = {
  recentTurnCount: null,
  recentTurnLimit: null,
  runningSummaryCoveredTurnCount: null,
  projectMemoryCount: null,
  retrievalChunkCount: null
};

/**
 * Exposes context boundaries without rendering private conversation, memory, or
 * retrieved content. Values are derived only from the stream's published counts.
 */
export function buildConversationContextModel({
  traceEvents,
  visibleEntryCount
}: {
  traceEvents: readonly WorkflowRuntimeTraceEvent[];
  visibleEntryCount: number;
}): ConversationContextModel {
  const evidence = readContextEvidence(traceEvents);
  const hasStreamEvidence = Object.values(evidence).some((value) => value !== null);
  const visibleHistoryState = visibleEntryCount > 0 ? "observed" : "not_injected";
  const recentTurnCount = evidence.recentTurnCount;
  const summaryCovered = evidence.runningSummaryCoveredTurnCount;
  const projectMemoryCount = evidence.projectMemoryCount;
  const retrievalChunkCount = evidence.retrievalChunkCount;
  const recentTurnLimit = evidence.recentTurnLimit;

  return {
    source: hasStreamEvidence ? "stream" : "none",
    summary: hasStreamEvidence
      ? "Counts reflect the context published for this run; private content remains hidden."
      : "No context-assembly evidence has been published for this run yet.",
    diagnostics: [
      {
        id: "visible_history",
        label: "Visible history",
        value: formatCount(visibleEntryCount, "entry"),
        detail: "Conversation entries currently visible in this workspace.",
        state: visibleHistoryState
      },
      {
        id: "model_context",
        label: "Model-visible recent turns",
        value:
          recentTurnCount == null
            ? "Not published"
            : formatCount(recentTurnCount, "turn"),
        detail:
          recentTurnCount == null
            ? "The runtime did not publish a context-assembly count."
            : "Recent persisted turns included in prompt preparation.",
        state: stateForCount(recentTurnCount)
      },
      {
        id: "memory",
        label: "Project memory",
        value:
          projectMemoryCount == null
            ? "Not published"
            : formatCount(projectMemoryCount, "item"),
        detail:
          projectMemoryCount == null
            ? "The runtime did not publish memory-injection counts."
            : "Project-memory items available to the assembled context.",
        state: stateForCount(projectMemoryCount)
      },
      {
        id: "summary",
        label: "Running summary",
        value:
          summaryCovered == null
            ? "Not injected"
            : `${formatCount(summaryCovered, "covered turn")}`,
        detail:
          summaryCovered == null
            ? "No carried session summary was published for this run."
            : "A bounded session summary preserves decisions outside the recent-turn window.",
        state:
          summaryCovered == null
            ? hasStreamEvidence
              ? "not_injected"
              : "unavailable"
            : "observed"
      },
      {
        id: "retrieval",
        label: "Retrieval context",
        value:
          retrievalChunkCount == null
            ? "Not published"
            : formatCount(retrievalChunkCount, "chunk"),
        detail:
          retrievalChunkCount == null
            ? "The runtime did not publish retrieval-context counts."
            : "Retrieved knowledge chunks available to prompt preparation.",
        state: stateForCount(retrievalChunkCount)
      },
      {
        id: "truncation",
        label: "Recent-turn window",
        value:
          recentTurnLimit == null
            ? "Not published"
            : `Up to ${formatCount(recentTurnLimit, "turn")}`,
        detail:
          recentTurnLimit == null
            ? "No context-window limit was published by the runtime."
            : "This is a bounded recent-history window, not a claim that history was truncated.",
        state: recentTurnLimit == null ? "unavailable" : "observed"
      },
      {
        id: "token_allocation",
        label: "Context token allocation",
        value: "Not reported",
        detail:
          "Provider token usage is shown separately; per-source context allocation was not published.",
        state: "unavailable"
      }
    ]
  };
}

function readContextEvidence(
  traceEvents: readonly WorkflowRuntimeTraceEvent[]
): ContextEvidence {
  let evidence = emptyContextEvidence;

  for (const traceEvent of traceEvents) {
    const payload = traceEvent.event.payload;
    evidence = mergeContextEvidence(evidence, readMemoryEvidence(traceEvent.event));
    evidence = mergeContextEvidence(evidence, readAssembledContextEvidence(payload));
    evidence = mergeContextEvidence(evidence, readPromptInputEvidence(payload));
  }

  return evidence;
}

function readMemoryEvidence(event: AssistantStreamEvent): ContextEvidence {
  if (event.event_type !== "memory") {
    return emptyContextEvidence;
  }

  const request = asRecord(event.payload.request);
  const context = asRecord(event.payload.context);
  const recentTurns = asList(context?.recent_turns);
  const projectMemories = asList(context?.project_memories);
  const runningSummary = asRecord(context?.running_summary);

  return {
    recentTurnCount: recentTurns?.length ?? null,
    recentTurnLimit: asNumber(request?.recent_turn_limit),
    runningSummaryCoveredTurnCount: asNumber(runningSummary?.covered_turn_count),
    projectMemoryCount: projectMemories?.length ?? null,
    retrievalChunkCount: null
  };
}

function readAssembledContextEvidence(payload: Record<string, unknown>): ContextEvidence {
  const context = asRecord(payload.context);
  const summary = asRecord(context?.summary);
  if (!summary) {
    return emptyContextEvidence;
  }

  return {
    recentTurnCount: asNumber(summary.recent_turn_count),
    recentTurnLimit: null,
    // The assembled summary confirms presence but deliberately omits the covered
    // turn count. That count is sourced from the earlier memory event when it is
    // published; never manufacture it from a boolean.
    runningSummaryCoveredTurnCount: null,
    projectMemoryCount: asNumber(summary.project_memory_count),
    retrievalChunkCount: asNumber(summary.retrieval_chunk_count)
  };
}

function readPromptInputEvidence(payload: Record<string, unknown>): ContextEvidence {
  const promptInput = asRecord(payload.prompt_input);
  const memoryInput = asRecord(promptInput?.memory_input);
  const retrievalInput = asRecord(promptInput?.retrieval_input);
  const recentTurns = asList(memoryInput?.recent_turns);
  const projectMemories = asList(memoryInput?.project_memories);
  const runningSummary = asRecord(memoryInput?.running_summary);
  const retrievalChunks = asList(retrievalInput?.chunks);

  return {
    recentTurnCount: recentTurns?.length ?? null,
    recentTurnLimit: null,
    runningSummaryCoveredTurnCount: asNumber(runningSummary?.covered_turn_count),
    projectMemoryCount: projectMemories?.length ?? null,
    retrievalChunkCount: retrievalChunks?.length ?? null
  };
}

function mergeContextEvidence(
  current: ContextEvidence,
  candidate: ContextEvidence
): ContextEvidence {
  return {
    recentTurnCount: candidate.recentTurnCount ?? current.recentTurnCount,
    recentTurnLimit: candidate.recentTurnLimit ?? current.recentTurnLimit,
    runningSummaryCoveredTurnCount:
      candidate.runningSummaryCoveredTurnCount ?? current.runningSummaryCoveredTurnCount,
    projectMemoryCount: candidate.projectMemoryCount ?? current.projectMemoryCount,
    retrievalChunkCount: candidate.retrievalChunkCount ?? current.retrievalChunkCount
  };
}

function stateForCount(value: number | null): ConversationContextDiagnosticState {
  if (value == null) {
    return "unavailable";
  }
  return value > 0 ? "observed" : "not_injected";
}

function formatCount(value: number, singular: string) {
  const plural = singular.endsWith("y")
    ? `${singular.slice(0, -1)}ies`
    : `${singular}s`;
  return `${value} ${value === 1 ? singular : plural}`;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function asList(value: unknown): unknown[] | null {
  return Array.isArray(value) ? value : null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
