export type WorkstationErrorCategory =
  | "stream"
  | "retrieval"
  | "preview_runtime"
  | "renderer"
  | "artifact_export"
  | "persistence"
  | "multimodal"
  | "workflow_runtime"
  | "hitl_approval";

export type WorkstationError = {
  id: string;
  type: string;
  category: WorkstationErrorCategory;
  subsystem: string;
  userMessage: string;
  debugMessage: string | null;
  recoverable: boolean;
  suggestedAction: string;
  retryLabel: string | null;
  resetLabel: string | null;
};

export type ParsedSubsystemErrorPayload = {
  type: string | null;
  category: string | null;
  subsystem: string | null;
  message: string | null;
  debugMessage: string | null;
  recoverable: boolean | null;
  suggestedAction: string | null;
  retryLabel: string | null;
  resetLabel: string | null;
};

type CreateWorkstationErrorInput = {
  type: string;
  category: WorkstationErrorCategory;
  subsystem: string;
  userMessage: string;
  debugMessage?: string | null;
  recoverable?: boolean;
  suggestedAction: string;
  retryLabel?: string | null;
  resetLabel?: string | null;
};

export function createWorkstationError({
  type,
  category,
  subsystem,
  userMessage,
  debugMessage = null,
  recoverable = false,
  suggestedAction,
  retryLabel = null,
  resetLabel = null
}: CreateWorkstationErrorInput): WorkstationError {
  return {
    id: `${category}:${subsystem}:${type}:${userMessage}`.toLowerCase(),
    type,
    category,
    subsystem,
    userMessage,
    debugMessage,
    recoverable,
    suggestedAction,
    retryLabel,
    resetLabel
  };
}

export function parseSubsystemErrorPayload(
  value: unknown
): ParsedSubsystemErrorPayload | null {
  if (!isRecord(value)) {
    return null;
  }

  const details = isRecord(value.details) ? value.details : null;

  return {
    type:
      readText(value.type) ??
      readText(value.code) ??
      readText(value.error) ??
      readText(details?.type) ??
      readText(details?.code),
    category: readText(value.category) ?? readText(details?.category),
    subsystem: readText(value.subsystem) ?? readText(details?.subsystem),
    message: readText(value.message) ?? readText(details?.message),
    debugMessage:
      readText(value.debug_message) ??
      readText(value.debugMessage) ??
      readText(details?.debug_message) ??
      readText(details?.debugMessage),
    recoverable:
      readBoolean(value.recoverable) ??
      readBoolean(value.retryable) ??
      readBoolean(details?.recoverable) ??
      readBoolean(details?.retryable),
    suggestedAction:
      readText(value.suggested_action) ??
      readText(value.suggestedAction) ??
      readText(details?.suggested_action) ??
      readText(details?.suggestedAction),
    retryLabel:
      readText(value.retry_label) ??
      readText(value.retryLabel) ??
      readText(details?.retry_label) ??
      readText(details?.retryLabel),
    resetLabel:
      readText(value.reset_label) ??
      readText(value.resetLabel) ??
      readText(details?.reset_label) ??
      readText(details?.resetLabel)
  };
}

function readBoolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function readText(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
