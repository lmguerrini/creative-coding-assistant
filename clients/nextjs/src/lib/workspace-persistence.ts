import type {
  ArtifactSummary,
  AssistantMessage,
  AssistantWorkspaceSnapshot,
  InspectorTabName,
  MultimodalSummary,
  PreviewSummary,
  WorkflowNodeId,
  WorkflowStepState
} from "./assistant-client";
import {
  buildMultimodalSummary,
  normalizeImageAttachments
} from "./multimodal-attachments";
import {
  createWorkstationError,
  parseSubsystemErrorPayload,
  type WorkstationError
} from "./workstation-errors";

export const defaultLocalUserId = "local-user";
export const defaultLocalSessionId = "local-nextjs-session";
export const defaultLocalProjectId = "local-nextjs-workspace";
export const workspaceLayoutBounds = {
  defaultInspectorWidth: 420,
  minInspectorWidth: 320,
  maxInspectorWidth: 560,
  compactPreviewHeight: 280,
  defaultPreviewHeight: 320,
  minPreviewHeight: 220,
  maxPreviewHeight: 520
} as const;

export type WorkspaceDensity = "cozy" | "compact";
export type WorkspaceThemePreset =
  | "aqua"
  | "codex"
  | "light"
  | "matrix"
  | "terminal"
  | "horizon"
  | "zen"
  | "blueprint";

export type WorkspaceLayoutState = {
  density: WorkspaceDensity;
  inspectorCollapsed: boolean;
  inspectorWidth: number;
  previewHeight: number;
};

export type WorkspacePreferences = {
  theme: WorkspaceThemePreset;
  autoOpenPreview: boolean;
  showDebugPanels: boolean;
};

export const defaultWorkspaceLayoutState: WorkspaceLayoutState = {
  density: "cozy",
  inspectorCollapsed: true,
  inspectorWidth: workspaceLayoutBounds.defaultInspectorWidth,
  previewHeight: workspaceLayoutBounds.defaultPreviewHeight
};

export const defaultWorkspacePreferences: WorkspacePreferences = {
  theme: "codex",
  autoOpenPreview: true,
  showDebugPanels: false
};

export type WorkspaceSessionRecord = {
  schemaVersion: 1 | 2 | 3 | 4;
  userId: string;
  sessionId: string;
  projectId: string;
  title: string;
  activeArtifactId: string;
  activeInspectorTab: InspectorTabName;
  previewOpen: boolean;
  previewArtifactId: string;
  layout?: WorkspaceLayoutState;
  preferences?: WorkspacePreferences;
  workspace: AssistantWorkspaceSnapshot["workspace"];
  messages: AssistantMessage[];
  workflow: AssistantWorkspaceSnapshot["workflow"];
  artifacts: ArtifactSummary[];
  multimodal?: MultimodalSummary;
  preview: PreviewSummary;
  snapshot: AssistantWorkspaceSnapshot;
  createdAt?: string;
  updatedAt?: string;
};

export type WorkspacePersistenceSaveResult = {
  target: "remote" | "local" | "none";
  error: WorkstationError | null;
};

export type WorkspacePersistenceLoadResult = {
  record: WorkspaceSessionRecord | null;
  source: "remote" | "local" | "none";
  error: WorkstationError | null;
};

export type WorkspacePersistenceClient = {
  load: () => Promise<WorkspacePersistenceLoadResult>;
  save: (
    record: WorkspaceSessionRecord
  ) => Promise<WorkspacePersistenceSaveResult>;
};

export type WorkspacePersistenceClientOptions = {
  endpoint?: string;
  fetchImpl?: typeof fetch;
  storage?: Storage | null;
  timeoutMs?: number;
  userId?: string;
  sessionId?: string;
};

export type WorkspaceSessionRecordInput = {
  activeArtifactId: string;
  activeInspectorTab: InspectorTabName;
  layout?: Partial<WorkspaceLayoutState>;
  preferences?: Partial<WorkspacePreferences>;
  previewArtifactId: string;
  previewOpen: boolean;
  snapshot: AssistantWorkspaceSnapshot;
};

const defaultPersistenceEndpoint =
  process.env.NEXT_PUBLIC_WORKSPACE_SESSION_URL ??
  "http://localhost:8000/api/workspace/session";
const maxPersistedMessages = 12;
const maxPersistedArtifacts = 12;
const maxPersistedRetrievalSources = 5;
const maxPersistedRetrievalChunks = 3;
const maxPersistedDebugEvents = 40;
const maxPersistedUserMessageChars = 2_000;
const maxPersistedAssistantMessageChars = 12_000;
const maxPersistedArtifactContentChars = 12_000;
const maxPersistedCodeLineChars = 2_000;
const maxPersistedRetrievalTextChars = 1_200;
const maxPersistedDebugDetailChars = 800;

export function createWorkspacePersistenceClient(
  options: WorkspacePersistenceClientOptions = {}
): WorkspacePersistenceClient {
  const endpoint = options.endpoint ?? defaultPersistenceEndpoint;
  const timeoutMs = options.timeoutMs ?? 1200;
  const userId = options.userId ?? defaultLocalUserId;
  const sessionId = options.sessionId ?? defaultLocalSessionId;

  return {
    async load() {
      const remoteResult = await loadRemoteSession({
        endpoint,
        fetchImpl: options.fetchImpl,
        sessionId,
        timeoutMs,
        userId
      });
      if (remoteResult.record) {
        writeLocalSession(remoteResult.record, options.storage);
        return {
          error: remoteResult.error,
          record: remoteResult.record,
          source: "remote"
        };
      }

      const localRecord = readLocalSession(options.storage, userId, sessionId);
      if (localRecord) {
        return {
          error: remoteResult.error,
          record: localRecord,
          source: "local"
        };
      }

      return {
        error: remoteResult.error,
        record: null,
        source: "none"
      };
    },
    async save(record) {
      const storedLocally = writeLocalSession(record, options.storage);
      const remoteResult = await saveRemoteSession({
        endpoint,
        fetchImpl: options.fetchImpl,
        record,
        timeoutMs
      });

      if (remoteResult.ok) {
        return { target: "remote", error: null };
      }

      return {
        target: storedLocally ? "local" : "none",
        error: remoteResult.error
      };
    }
  };
}

export function createWorkspaceSessionRecord({
  activeArtifactId,
  activeInspectorTab,
  layout,
  preferences,
  previewArtifactId,
  previewOpen,
  snapshot
}: WorkspaceSessionRecordInput): WorkspaceSessionRecord {
  const updatedAt = new Date().toISOString();
  const messages = compactWorkspaceMessages(snapshot.messages);
  const artifacts = compactWorkspaceArtifacts(snapshot.artifacts);
  const compactSnapshot = compactWorkspaceSnapshot({
    artifacts,
    messages,
    snapshot,
    updatedAt
  });

  return {
    schemaVersion: 4,
    userId: snapshot.session.userId,
    sessionId: snapshot.session.sessionId,
    projectId: snapshot.session.projectId,
    title: snapshot.session.title || snapshot.workspace.name,
    activeArtifactId,
    activeInspectorTab,
    previewOpen,
    previewArtifactId,
    layout: normalizeWorkspaceLayoutState(layout),
    preferences: normalizeWorkspacePreferences(preferences),
    workspace: snapshot.workspace,
    messages,
    workflow: snapshot.workflow,
    artifacts,
    multimodal: snapshot.multimodal,
    preview: snapshot.preview,
    snapshot: compactSnapshot,
    updatedAt
  };
}

export function snapshotFromWorkspaceSessionRecord(
  fallback: AssistantWorkspaceSnapshot,
  record: WorkspaceSessionRecord
): AssistantWorkspaceSnapshot {
  const restoredSnapshot = record.snapshot ?? fallback;
  const artifacts = record.artifacts.length
    ? record.artifacts
    : restoredSnapshot.artifacts;
  const multimodal = normalizeWorkspaceMultimodal(
    fallback.multimodal,
    record.multimodal ?? restoredSnapshot.multimodal
  );
  const preview = {
    ...restoredSnapshot.preview,
    ...record.preview,
    active: record.previewOpen,
    collapsed: !record.previewOpen
  };

  return {
    ...fallback,
    ...restoredSnapshot,
    session: {
      ...fallback.session,
      ...restoredSnapshot.session,
      userId: record.userId,
      sessionId: record.sessionId,
      projectId: record.projectId,
      title: record.title,
      updatedAt: record.updatedAt
    },
    workspace: {
      ...restoredSnapshot.workspace,
      ...record.workspace
    },
    inspectorTabs: fallback.inspectorTabs.map((tab) => ({
      ...tab,
      active: tab.label === record.activeInspectorTab
    })),
    messages: record.messages.length ? record.messages : restoredSnapshot.messages,
    workflow: normalizeWorkspaceWorkflow(
      fallback.workflow,
      record.workflow ?? restoredSnapshot.workflow
    ),
    artifacts,
    multimodal,
    preview
  };
}

export function fingerprintWorkspaceSessionRecord(
  record: WorkspaceSessionRecord
): string {
  return JSON.stringify({
    ...record,
    createdAt: undefined,
    updatedAt: undefined,
    snapshot: {
      ...record.snapshot,
      session: {
        ...record.snapshot.session,
        updatedAt: undefined
      }
    }
  });
}

export function normalizeWorkspaceLayoutState(
  layout?: Partial<WorkspaceLayoutState> | null
): WorkspaceLayoutState {
  return {
    density: layout?.density === "compact" ? "compact" : "cozy",
    inspectorCollapsed:
      layout?.inspectorCollapsed ?? defaultWorkspaceLayoutState.inspectorCollapsed,
    inspectorWidth: clampLayoutValue(
      layout?.inspectorWidth,
      workspaceLayoutBounds.minInspectorWidth,
      workspaceLayoutBounds.maxInspectorWidth,
      workspaceLayoutBounds.defaultInspectorWidth
    ),
    previewHeight: clampLayoutValue(
      layout?.previewHeight,
      workspaceLayoutBounds.minPreviewHeight,
      workspaceLayoutBounds.maxPreviewHeight,
      workspaceLayoutBounds.defaultPreviewHeight
    )
  };
}

export function normalizeWorkspacePreferences(
  preferences?: Partial<WorkspacePreferences> | null
): WorkspacePreferences {
  return {
    theme: isWorkspaceThemePreset(preferences?.theme)
      ? preferences.theme
      : defaultWorkspacePreferences.theme,
    autoOpenPreview:
      typeof preferences?.autoOpenPreview === "boolean"
        ? preferences.autoOpenPreview
        : defaultWorkspacePreferences.autoOpenPreview,
    showDebugPanels:
      typeof preferences?.showDebugPanels === "boolean"
        ? preferences.showDebugPanels
        : defaultWorkspacePreferences.showDebugPanels
  };
}

export function isWorkspaceSessionRecord(
  value: unknown
): value is WorkspaceSessionRecord {
  if (!isRecord(value)) {
    return false;
  }

  return (
    (value.schemaVersion === 1 ||
      value.schemaVersion === 2 ||
      value.schemaVersion === 3 ||
      value.schemaVersion === 4) &&
    typeof value.userId === "string" &&
    typeof value.sessionId === "string" &&
    typeof value.projectId === "string" &&
    typeof value.title === "string" &&
    typeof value.activeArtifactId === "string" &&
    isInspectorTabName(value.activeInspectorTab) &&
    typeof value.previewOpen === "boolean" &&
    typeof value.previewArtifactId === "string" &&
    (value.layout === undefined || isWorkspaceLayoutState(value.layout)) &&
    (value.preferences === undefined ||
      isWorkspacePreferences(value.preferences)) &&
    isRecord(value.workspace) &&
    Array.isArray(value.messages) &&
    isRecord(value.workflow) &&
    Array.isArray(value.artifacts) &&
    (value.multimodal === undefined || isRecord(value.multimodal)) &&
    isRecord(value.preview) &&
    isRecord(value.snapshot)
  );
}

function normalizeWorkspaceMultimodal(
  fallback: MultimodalSummary,
  value: unknown
): MultimodalSummary {
  if (!isRecord(value)) {
    return fallback;
  }

  return buildMultimodalSummary({
    baseMultimodal: {
      ...fallback,
      state:
        value.state === "ready" || value.state === "error" || value.state === "empty"
          ? value.state
          : fallback.state,
      status: typeof value.status === "string" ? value.status : fallback.status,
      detail: typeof value.detail === "string" ? value.detail : fallback.detail,
      imageAttachments: [],
      error: null
    },
    imageAttachments: normalizeImageAttachments(value.imageAttachments),
    uploadError: null
  });
}

function normalizeWorkspaceWorkflow(
  fallback: AssistantWorkspaceSnapshot["workflow"],
  value: unknown
): AssistantWorkspaceSnapshot["workflow"] {
  if (!isRecord(value)) {
    return fallback;
  }

  const rawSteps = Array.isArray(value.steps) ? value.steps : [];
  const fallbackNodeIds = new Set(
    fallback.steps.map((step) => step.nodeId as string)
  );
  const restoredSteps = new Map<string, WorkflowStepState>();

  for (const rawStep of rawSteps) {
    if (!isRecord(rawStep)) {
      continue;
    }
    if (
      typeof rawStep.nodeId !== "string" ||
      !fallbackNodeIds.has(rawStep.nodeId) ||
      typeof rawStep.displayLabel !== "string" ||
      !isWorkflowStepState(rawStep.state) ||
      typeof rawStep.detail !== "string"
    ) {
      continue;
    }
    restoredSteps.set(rawStep.nodeId, {
      nodeId: rawStep.nodeId as WorkflowStepState["nodeId"],
      displayLabel: rawStep.displayLabel,
      state: rawStep.state,
      detail: rawStep.detail
    });
  }

  const mergedSteps = fallback.steps.map((fallbackStep, index) => {
    const restoredStep = restoredSteps.get(fallbackStep.nodeId);
    if (restoredStep) {
      return restoredStep;
    }
    return inferInsertedWorkflowStepState(
      fallbackStep,
      index,
      fallback.steps,
      restoredSteps
    );
  });

  return {
    ...fallback,
    status: typeof value.status === "string" ? value.status : fallback.status,
    currentNode:
      typeof value.currentNode === "string" &&
      fallbackNodeIds.has(value.currentNode)
        ? (value.currentNode as WorkflowNodeId)
        : fallback.currentNode,
    currentStep:
      typeof value.currentStep === "string" ? value.currentStep : fallback.currentStep,
    steps: mergedSteps
  };
}

function compactWorkspaceSnapshot({
  artifacts,
  messages,
  snapshot,
  updatedAt
}: {
  artifacts: ArtifactSummary[];
  messages: AssistantMessage[];
  snapshot: AssistantWorkspaceSnapshot;
  updatedAt: string;
}): AssistantWorkspaceSnapshot {
  return {
    ...snapshot,
    session: {
      ...snapshot.session,
      updatedAt
    },
    messages,
    artifacts,
    code: {
      ...snapshot.code,
      excerpt: snapshot.code.excerpt
        .slice(0, 20)
        .map((line) => truncatePersistedText(line, maxPersistedCodeLineChars))
    },
    retrieval: compactWorkspaceRetrieval(snapshot.retrieval),
    debug: {
      ...snapshot.debug,
      events: snapshot.debug.events.slice(-maxPersistedDebugEvents).map((event) => ({
        ...event,
        detail: truncatePersistedText(event.detail, maxPersistedDebugDetailChars)
      }))
    }
  };
}

function compactWorkspaceMessages(
  messages: AssistantMessage[]
): AssistantMessage[] {
  return messages.slice(-maxPersistedMessages).map((message) => ({
    ...message,
    content: truncatePersistedText(
      message.content,
      message.role === "assistant"
        ? maxPersistedAssistantMessageChars
        : maxPersistedUserMessageChars
    )
  }));
}

function compactWorkspaceArtifacts(
  artifacts: ArtifactSummary[]
): ArtifactSummary[] {
  return artifacts.slice(-maxPersistedArtifacts).map((artifact) => ({
    ...artifact,
    content:
      typeof artifact.content === "string"
        ? truncatePersistedText(
            artifact.content,
            maxPersistedArtifactContentChars
          )
        : artifact.content
  }));
}

function compactWorkspaceRetrieval(
  retrieval: AssistantWorkspaceSnapshot["retrieval"]
): AssistantWorkspaceSnapshot["retrieval"] {
  return {
    ...retrieval,
    detail: truncatePersistedText(retrieval.detail, maxPersistedRetrievalTextChars),
    warning:
      retrieval.warning === null
        ? null
        : truncatePersistedText(retrieval.warning, maxPersistedRetrievalTextChars),
    sources: retrieval.sources
      .slice(0, maxPersistedRetrievalSources)
      .map((source) => ({
        ...source,
        detail: truncatePersistedText(
          source.detail,
          maxPersistedRetrievalTextChars
        ),
        whyUsed: truncatePersistedText(
          source.whyUsed,
          maxPersistedRetrievalTextChars
        ),
        chunks: source.chunks
          .slice(0, maxPersistedRetrievalChunks)
          .map((chunk) => ({
            ...chunk,
            snippet: truncatePersistedText(
              chunk.snippet,
              maxPersistedRetrievalTextChars
            )
          }))
      }))
  };
}

function truncatePersistedText(value: string, limit: number): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (normalized.length <= limit) {
    return normalized;
  }
  return `${normalized.slice(0, limit - 14).trimEnd()}... [truncated]`;
}

function inferInsertedWorkflowStepState(
  fallbackStep: WorkflowStepState,
  index: number,
  canonicalSteps: WorkflowStepState[],
  restoredSteps: Map<string, WorkflowStepState>
): WorkflowStepState {
  const laterStepReached = canonicalSteps
    .slice(index + 1)
    .some((step) => isReachedWorkflowState(restoredSteps.get(step.nodeId)?.state));
  if (!laterStepReached) {
    return fallbackStep;
  }
  return {
    ...fallbackStep,
    state: "complete",
    detail: `${fallbackStep.displayLabel} completed in restored session.`
  };
}

function isWorkflowStepState(value: unknown): value is WorkflowStepState["state"] {
  return (
    value === "complete" ||
    value === "active" ||
    value === "queued" ||
    value === "skipped" ||
    value === "branch"
  );
}

function isReachedWorkflowState(
  value: WorkflowStepState["state"] | undefined
): boolean {
  return value === "complete" || value === "active" || value === "skipped";
}

async function loadRemoteSession({
  endpoint,
  fetchImpl,
  sessionId,
  timeoutMs,
  userId
}: {
  endpoint: string;
  fetchImpl?: typeof fetch;
  sessionId: string;
  timeoutMs: number;
  userId: string;
}): Promise<{
  record: WorkspaceSessionRecord | null;
  error: WorkstationError | null;
}> {
  const resolvedFetch = fetchImpl ?? globalThis.fetch;
  if (!resolvedFetch) {
    return {
      record: null,
      error: createPersistenceError({
        defaultMessage: "Remote session restore is unavailable in this environment.",
        fallbackType: "persistence_unavailable",
        operation: "load",
        payload: null
      })
    };
  }

  try {
    const url = new URL(endpoint);
    url.searchParams.set("userId", userId);
    url.searchParams.set("sessionId", sessionId);
    const abort = createAbortSignal();
    const response = await withTimeout(
      resolvedFetch(url.toString(), {
        headers: {
          Accept: "application/json"
        },
        method: "GET",
        signal: abort.signal
      }),
      timeoutMs,
      abort.abort
    );

    if (!response.ok) {
      const payload = (await tryReadJson(response)) ?? (await tryReadText(response));
      const errorCode = readPersistenceResponseCode(payload);
      if (response.status === 404 && errorCode === "session_not_found") {
        return { record: null, error: null };
      }

      return {
        record: null,
        error: createPersistenceError({
          defaultMessage: "Remote session restore failed.",
          fallbackType: errorCode ?? `http_${response.status}`,
          operation: "load",
          payload
        })
      };
    }

    const payload: unknown = await response.json();
    return {
      record: isWorkspaceSessionRecord(payload) ? payload : null,
      error: isWorkspaceSessionRecord(payload)
        ? null
        : createPersistenceError({
            defaultMessage: "The saved session response had an unexpected shape.",
            fallbackType: "invalid_session_payload",
            operation: "load",
            payload
          })
    };
  } catch {
    return {
      record: null,
      error: createPersistenceError({
        defaultMessage: "Remote session restore timed out or could not be reached.",
        fallbackType: "session_restore_unavailable",
        operation: "load",
        payload: null
      })
    };
  }
}

async function saveRemoteSession({
  endpoint,
  fetchImpl,
  record,
  timeoutMs
}: {
  endpoint: string;
  fetchImpl?: typeof fetch;
  record: WorkspaceSessionRecord;
  timeoutMs: number;
}): Promise<{ ok: boolean; error: WorkstationError | null }> {
  const resolvedFetch = fetchImpl ?? globalThis.fetch;
  if (!resolvedFetch) {
    return {
      ok: false,
      error: createPersistenceError({
        defaultMessage: "Remote session save is unavailable in this environment.",
        fallbackType: "persistence_unavailable",
        operation: "save",
        payload: null
      })
    };
  }

  try {
    const abort = createAbortSignal();
    const response = await withTimeout(
      resolvedFetch(endpoint, {
        body: JSON.stringify(record),
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json"
        },
        method: "POST",
        signal: abort.signal
      }),
      timeoutMs,
      abort.abort
    );
    if (response.ok) {
      return { ok: true, error: null };
    }

    const payload = (await tryReadJson(response)) ?? (await tryReadText(response));
    return {
      ok: false,
      error: createPersistenceError({
        defaultMessage: "Remote session save failed.",
        fallbackType: readPersistenceResponseCode(payload) ?? `http_${response.status}`,
        operation: "save",
        payload
      })
    };
  } catch {
    return {
      ok: false,
      error: createPersistenceError({
        defaultMessage: "Remote session save timed out or could not be reached.",
        fallbackType: "session_save_unavailable",
        operation: "save",
        payload: null
      })
    };
  }
}

async function tryReadJson(response: Response) {
  try {
    return (await response.clone().json()) as unknown;
  } catch {
    return null;
  }
}

async function tryReadText(response: Response) {
  try {
    const text = await response.clone().text();
    return text.trim() ? text : null;
  } catch {
    return null;
  }
}

function createPersistenceError({
  defaultMessage,
  fallbackType,
  operation,
  payload
}: {
  defaultMessage: string;
  fallbackType: string;
  operation: "load" | "save";
  payload: unknown;
}) {
  const parsed = parseSubsystemErrorPayload(payload);
  const type = parsed?.type ?? fallbackType;
  const recoverable = parsed?.recoverable ?? true;

  return createWorkstationError({
    type,
    category: "persistence",
    subsystem: parsed?.subsystem ?? "workspace_session_store",
    userMessage: parsed?.message ?? defaultMessage,
    debugMessage: parsed?.debugMessage ?? readPersistenceDebugMessage(payload),
    recoverable,
    suggestedAction:
      parsed?.suggestedAction ??
      (operation === "load"
        ? "Continue from the local session copy or reset the workspace session."
        : "Keep editing locally; the workspace can save again when the connection is available."),
    retryLabel:
      parsed?.retryLabel ?? (operation === "save" && recoverable ? "Retry save" : null),
    resetLabel:
      parsed?.resetLabel ?? (operation === "load" ? "Clear workspace session" : null)
  });
}

function readPersistenceResponseCode(payload: unknown) {
  const parsed = parseSubsystemErrorPayload(payload);
  return parsed?.type ?? null;
}

function readPersistenceDebugMessage(payload: unknown) {
  if (typeof payload === "string" && payload.trim()) {
    return payload;
  }

  if (!isRecord(payload)) {
    return null;
  }

  return typeof payload.message === "string" ? payload.message : null;
}

function createAbortSignal(): {
  abort: () => void;
  signal?: AbortSignal;
} {
  if (typeof AbortController === "undefined") {
    return {
      abort() {
        return undefined;
      }
    };
  }

  const controller = new AbortController();

  return {
    abort() {
      controller.abort();
    },
    signal: controller.signal
  };
}

function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  onTimeout: () => void
): Promise<T> {
  return new Promise((resolve, reject) => {
    const timeoutId = globalThis.setTimeout(() => {
      onTimeout();
      reject(new Error("Workspace persistence request timed out."));
    }, timeoutMs);

    promise
      .then(resolve, reject)
      .finally(() => globalThis.clearTimeout(timeoutId));
  });
}

function readLocalSession(
  storage: Storage | null | undefined,
  userId: string,
  sessionId: string
): WorkspaceSessionRecord | null {
  const resolvedStorage = resolveStorage(storage);
  if (!resolvedStorage) {
    return null;
  }

  try {
    const rawRecord = resolvedStorage.getItem(
      workspaceSessionStorageKey(userId, sessionId)
    );
    if (!rawRecord) {
      return null;
    }
    const parsed: unknown = JSON.parse(rawRecord);
    return isWorkspaceSessionRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function writeLocalSession(
  record: WorkspaceSessionRecord,
  storage: Storage | null | undefined
): boolean {
  const resolvedStorage = resolveStorage(storage);
  if (!resolvedStorage) {
    return false;
  }

  try {
    resolvedStorage.setItem(
      workspaceSessionStorageKey(record.userId, record.sessionId),
      JSON.stringify(record)
    );
    return true;
  } catch {
    return false;
  }
}

function workspaceSessionStorageKey(userId: string, sessionId: string) {
  return `cca.workspace.${userId}.${sessionId}`;
}

function resolveStorage(storage: Storage | null | undefined): Storage | null {
  if (storage !== undefined) {
    return storage;
  }

  try {
    return globalThis.window?.localStorage ?? null;
  } catch {
    return null;
  }
}

function isInspectorTabName(value: unknown): value is InspectorTabName {
  return (
    value === "Overview" ||
    value === "Preview" ||
    value === "Runtime" ||
    value === "Code" ||
    value === "Workflow" ||
    value === "Telemetry" ||
    value === "Artifacts" ||
    value === "Retrieval"
  );
}

function isWorkspaceThemePreset(value: unknown): value is WorkspaceThemePreset {
  return (
    value === "aqua" ||
    value === "codex" ||
    value === "light" ||
    value === "matrix" ||
    value === "terminal" ||
    value === "horizon" ||
    value === "zen" ||
    value === "blueprint"
  );
}

function isWorkspaceLayoutState(value: unknown): value is WorkspaceLayoutState {
  if (!isRecord(value)) {
    return false;
  }

  return (
    (value.density === "cozy" || value.density === "compact") &&
    typeof value.inspectorCollapsed === "boolean" &&
    typeof value.inspectorWidth === "number" &&
    Number.isFinite(value.inspectorWidth) &&
    typeof value.previewHeight === "number" &&
    Number.isFinite(value.previewHeight)
  );
}

function isWorkspacePreferences(value: unknown): value is WorkspacePreferences {
  if (!isRecord(value)) {
    return false;
  }

  return (
    isWorkspaceThemePreset(value.theme) &&
    typeof value.autoOpenPreview === "boolean" &&
    typeof value.showDebugPanels === "boolean"
  );
}

function clampLayoutValue(
  value: number | undefined,
  minimum: number,
  maximum: number,
  fallback: number
) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return fallback;
  }

  return Math.min(Math.max(Math.round(value), minimum), maximum);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
