import type {
  ArtifactSummary,
  AssistantMessage,
  AssistantWorkspaceSnapshot,
  InspectorTabName,
  PreviewSummary
} from "./assistant-client";

export const defaultLocalUserId = "local-user";
export const defaultLocalSessionId = "local-nextjs-session";
export const defaultLocalProjectId = "local-nextjs-workspace";
export const workspaceLayoutBounds = {
  defaultInspectorWidth: 420,
  minInspectorWidth: 320,
  maxInspectorWidth: 560,
  defaultPreviewHeight: 220,
  minPreviewHeight: 160,
  maxPreviewHeight: 360
} as const;

export type WorkspaceDensity = "cozy" | "compact";
export type WorkspaceThemePreset = "aqua" | "codex" | "matrix";

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
  inspectorCollapsed: false,
  inspectorWidth: workspaceLayoutBounds.defaultInspectorWidth,
  previewHeight: workspaceLayoutBounds.defaultPreviewHeight
};

export const defaultWorkspacePreferences: WorkspacePreferences = {
  theme: "aqua",
  autoOpenPreview: true,
  showDebugPanels: true
};

export type WorkspaceSessionRecord = {
  schemaVersion: 1 | 2 | 3;
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
  preview: PreviewSummary;
  snapshot: AssistantWorkspaceSnapshot;
  createdAt?: string;
  updatedAt?: string;
};

export type WorkspacePersistenceSaveResult = {
  target: "remote" | "local" | "none";
};

export type WorkspacePersistenceClient = {
  load: () => Promise<WorkspaceSessionRecord | null>;
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

export function createWorkspacePersistenceClient(
  options: WorkspacePersistenceClientOptions = {}
): WorkspacePersistenceClient {
  const endpoint = options.endpoint ?? defaultPersistenceEndpoint;
  const timeoutMs = options.timeoutMs ?? 1200;
  const userId = options.userId ?? defaultLocalUserId;
  const sessionId = options.sessionId ?? defaultLocalSessionId;

  return {
    async load() {
      const remoteRecord = await loadRemoteSession({
        endpoint,
        fetchImpl: options.fetchImpl,
        sessionId,
        timeoutMs,
        userId
      });
      if (remoteRecord) {
        writeLocalSession(remoteRecord, options.storage);
        return remoteRecord;
      }

      return readLocalSession(options.storage, userId, sessionId);
    },
    async save(record) {
      const storedLocally = writeLocalSession(record, options.storage);
      const savedRemotely = await saveRemoteSession({
        endpoint,
        fetchImpl: options.fetchImpl,
        record,
        timeoutMs
      });

      if (savedRemotely) {
        return { target: "remote" };
      }

      return { target: storedLocally ? "local" : "none" };
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

  return {
    schemaVersion: 3,
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
    messages: snapshot.messages,
    workflow: snapshot.workflow,
    artifacts: snapshot.artifacts,
    preview: snapshot.preview,
    snapshot: {
      ...snapshot,
      session: {
        ...snapshot.session,
        updatedAt
      }
    },
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
    workflow: record.workflow ?? restoredSnapshot.workflow,
    artifacts,
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
    inspectorCollapsed: Boolean(layout?.inspectorCollapsed),
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
      value.schemaVersion === 3) &&
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
    isRecord(value.preview) &&
    isRecord(value.snapshot)
  );
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
}): Promise<WorkspaceSessionRecord | null> {
  const resolvedFetch = fetchImpl ?? globalThis.fetch;
  if (!resolvedFetch) {
    return null;
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
      return null;
    }

    const payload: unknown = await response.json();
    return isWorkspaceSessionRecord(payload) ? payload : null;
  } catch {
    return null;
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
}): Promise<boolean> {
  const resolvedFetch = fetchImpl ?? globalThis.fetch;
  if (!resolvedFetch) {
    return false;
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
    return response.ok;
  } catch {
    return false;
  }
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
    value === "Code" ||
    value === "Workflow" ||
    value === "Artifacts" ||
    value === "Retrieval"
  );
}

function isWorkspaceThemePreset(value: unknown): value is WorkspaceThemePreset {
  return value === "aqua" || value === "codex" || value === "matrix";
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
