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
import { readProductOutcome } from "./assistant-stream";
import {
  buildMultimodalSummary
} from "./multimodal-attachments";
import {
  createWorkstationError,
  parseSubsystemErrorPayload,
  type WorkstationError
} from "./workstation-errors";
import {
  getArtifactPreviewSupportIssue,
  normalizeStoredArtifactRuntimeBoundary
} from "./live-artifact-hydration";
import { isArtifactPreviewable } from "./preview-runtime";
import type { WorkflowExecutionMode } from "./workflow-execution";
import {
  buildGenerationControls,
  type CreativityProfile,
  type EvaluationHistoryRecord,
  type FeedbackSignal,
  type FontScale
} from "./product-controls";

export const defaultLocalUserId = "local-user";
export const defaultLocalSessionId = "local-nextjs-session";
export const defaultLocalProjectId = "local-nextjs-workspace";
const workspaceProfileIdentityStorageKey = "cca.workspace.profile-identity.v1";

export type WorkspaceIdentity = {
  userId: string;
  sessionId: string;
  projectId: string;
};

export type WorkspaceSessionSummary = {
  sessionId: string;
  projectId: string;
  title: string;
  updatedAt: string | null;
  artifactCount: number;
};

export const defaultWorkspaceIdentity: WorkspaceIdentity = {
  userId: defaultLocalUserId,
  sessionId: defaultLocalSessionId,
  projectId: defaultLocalProjectId
};
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
  | "codex_white"
  | "light"
  | "matrix"
  | "terminal"
  | "horizon"
  | "zen"
  | "blueprint";

export type WorkspaceLayoutState = {
  density: WorkspaceDensity;
  sidebarCollapsed: boolean;
  inspectorCollapsed: boolean;
  inspectorWidth: number;
  previewHeight: number;
};

export type WorkspacePreferences = {
  theme: WorkspaceThemePreset;
  autoOpenPreview: boolean;
  showDebugPanels: boolean;
  workflowMode: WorkflowExecutionMode;
  creativity: CreativityProfile;
  personalizationEnabled: boolean;
  headingFontSize: FontScale;
  uiFontSize: FontScale;
  labelFontSize: FontScale;
  codeFontSize: FontScale;
  feedbackSignals: FeedbackSignal[];
  evaluationHistory: EvaluationHistoryRecord[];
};

export const defaultWorkspaceLayoutState: WorkspaceLayoutState = {
  density: "cozy",
  sidebarCollapsed: false,
  inspectorCollapsed: true,
  inspectorWidth: workspaceLayoutBounds.defaultInspectorWidth,
  previewHeight: workspaceLayoutBounds.defaultPreviewHeight
};

export const defaultWorkspacePreferences: WorkspacePreferences = {
  theme: "codex",
  autoOpenPreview: true,
  showDebugPanels: true,
  workflowMode: "auto",
  creativity: "balanced",
  personalizationEnabled: true,
  headingFontSize: "medium",
  uiFontSize: "medium",
  labelFontSize: "medium",
  codeFontSize: "medium",
  feedbackSignals: [],
  evaluationHistory: []
};

export type WorkspaceSessionRecord = {
  schemaVersion: 1 | 2 | 3 | 4 | 5;
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
  identity?: WorkspaceIdentity;
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
  projectId?: string;
  useProfileIdentity?: boolean;
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
const maxPersistedCodeChars = 96_000;
const maxPersistedRetrievalTextChars = 1_200;
const maxPersistedDebugDetailChars = 800;

export function createWorkspacePersistenceClient(
  options: WorkspacePersistenceClientOptions = {}
): WorkspacePersistenceClient {
  const endpoint = options.endpoint ?? defaultPersistenceEndpoint;
  const timeoutMs = options.timeoutMs ?? 1200;
  const profileIdentity =
    options.useProfileIdentity ?? options.storage === undefined
      ? resolveWorkspaceIdentity(options.storage)
      : defaultWorkspaceIdentity;
  const identity: WorkspaceIdentity = {
    userId: options.userId ?? profileIdentity.userId,
    sessionId: options.sessionId ?? profileIdentity.sessionId,
    projectId: options.projectId ?? profileIdentity.projectId
  };
  const { projectId, sessionId, userId } = identity;

  return {
    identity,
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

export function resolveWorkspaceIdentity(
  storage?: Storage | null
): WorkspaceIdentity {
  const resolvedStorage = resolveStorage(storage);
  if (!resolvedStorage) {
    return createEphemeralWorkspaceIdentity();
  }

  try {
    const storedIdentity = resolvedStorage.getItem(workspaceProfileIdentityStorageKey);
    if (storedIdentity) {
      const parsedIdentity: unknown = JSON.parse(storedIdentity);
      if (isWorkspaceIdentity(parsedIdentity)) {
        return parsedIdentity;
      }
    }

    const identity = createEphemeralWorkspaceIdentity();
    resolvedStorage.setItem(
      workspaceProfileIdentityStorageKey,
      JSON.stringify(identity)
    );
    return identity;
  } catch {
    return createEphemeralWorkspaceIdentity();
  }
}

export function withWorkspaceIdentity(
  snapshot: AssistantWorkspaceSnapshot,
  identity: WorkspaceIdentity
): AssistantWorkspaceSnapshot {
  return {
    ...snapshot,
    session: {
      ...snapshot.session,
      ...identity
    }
  };
}

export function listLocalWorkspaceSessions(
  userId: string,
  storage?: Storage | null
): WorkspaceSessionSummary[] {
  const resolvedStorage = resolveStorage(storage);
  if (!resolvedStorage) {
    return [];
  }
  try {
    const rawIndex = resolvedStorage.getItem(workspaceSessionIndexStorageKey(userId));
    const parsed: unknown = rawIndex ? JSON.parse(rawIndex) : [];
    if (!Array.isArray(parsed)) {
      resolvedStorage.removeItem(workspaceSessionIndexStorageKey(userId));
      return [];
    }
    const records = parsed
      .map((item) => normalizeSessionSummary(item))
      .filter((item): item is WorkspaceSessionSummary => item !== null)
      .filter((summary) =>
        Boolean(readLocalSession(resolvedStorage, userId, summary.sessionId))
      )
      .sort((left, right) => {
        const rightTime = right.updatedAt ?? "";
        const leftTime = left.updatedAt ?? "";
        return rightTime.localeCompare(leftTime) || left.sessionId.localeCompare(right.sessionId);
      });
    writeLocalWorkspaceSessionIndex(records, resolvedStorage, userId);
    return records;
  } catch {
    return [];
  }
}

export function removeLocalWorkspaceSession(
  userId: string,
  sessionId: string,
  storage?: Storage | null
): boolean {
  const resolvedStorage = resolveStorage(storage);
  if (!resolvedStorage || !sessionId.trim()) {
    return false;
  }
  try {
    resolvedStorage.removeItem(workspaceSessionStorageKey(userId, sessionId));
    const remaining = listLocalWorkspaceSessions(userId, resolvedStorage).filter(
      (summary) => summary.sessionId !== sessionId
    );
    writeLocalWorkspaceSessionIndex(remaining, resolvedStorage, userId);
    return true;
  } catch {
    return false;
  }
}

export async function deletePersistedWorkspaceSession({
  fetchImpl,
  identity,
  storage
}: {
  fetchImpl?: typeof fetch;
  identity: WorkspaceIdentity;
  storage?: Storage | null;
}): Promise<"remote" | "local"> {
  removeLocalWorkspaceSession(
    identity.userId,
    identity.sessionId,
    storage
  );
  const resolvedFetch = fetchImpl ?? globalThis.fetch;
  if (!resolvedFetch) {
    return "local";
  }
  try {
    const endpoint = new URL(defaultPersistenceEndpoint);
    endpoint.searchParams.set("userId", identity.userId);
    endpoint.searchParams.set("sessionId", identity.sessionId);
    const response = await resolvedFetch(endpoint.toString(), {
      headers: { Accept: "application/json" },
      method: "DELETE"
    });
    if (response.ok || response.status === 404) {
      return "remote";
    }
  } catch {
    // The local delete is the safe fallback; a later explicit save cannot revive it.
  }
  return "local";
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
  const persistedMultimodal = requestScopedPersistenceBoundary(snapshot.multimodal);
  const persistedSnapshot = {
    ...snapshot,
    multimodal: persistedMultimodal
  };
  const messages = compactWorkspaceMessages(persistedSnapshot.messages);
  const artifacts = compactWorkspaceArtifacts(persistedSnapshot.artifacts);
  const compactSnapshot = compactWorkspaceSnapshot({
    artifacts,
    messages,
    snapshot: persistedSnapshot,
    updatedAt
  });

  return {
    schemaVersion: 5,
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
    multimodal: persistedMultimodal,
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
  const artifacts = (record.artifacts.length
    ? record.artifacts
    : restoredSnapshot.artifacts
  ).map(normalizeStoredArtifactRuntimeBoundary);
  const multimodal = normalizeWorkspaceMultimodal(
    fallback.multimodal,
    record.multimodal ?? restoredSnapshot.multimodal
  );
  const restoredPreview = {
    ...restoredSnapshot.preview,
    ...record.preview,
    active: record.previewOpen,
    collapsed: !record.previewOpen
  };
  const previewArtifact = findRestoredPreviewArtifact(
    artifacts,
    record,
    restoredPreview
  );
  const recoveredPreview = recoverInterruptedPreviewSession(
    restoredPreview,
    previewArtifact
  );
  const previewSupportIssue = previewArtifact
    ? getArtifactPreviewSupportIssue(previewArtifact)
    : null;
  const previewRecoveryIssue = getPersistedPreviewSourceRecoveryIssue(
    restoredSnapshot,
    record,
    previewArtifact
  );
  const previewIssue = previewRecoveryIssue ?? previewSupportIssue;
  const preview = previewIssue
    ? {
        ...recoveredPreview,
        active: false,
        artifactName: previewArtifact?.title ?? recoveredPreview.artifactName,
        available: false,
        collapsed: true,
        outputArtifactName: "",
        renderer: "",
        sourceArtifactId: previewArtifact?.id ?? "",
        sourceArtifactName: previewArtifact?.title ?? "",
        state: "unavailable" as const,
        status: "Unavailable",
        summary: `${previewArtifact?.title ?? "This artifact"} is saved as code, but it will not be started in the live preview. ${previewIssue}`,
        target: "",
        targetId: "" as const,
        title: "Preview unavailable"
      }
    : recoveredPreview;

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

function recoverInterruptedPreviewSession(
  preview: PreviewSummary,
  previewArtifact: ArtifactSummary | null
): PreviewSummary {
  const isInterruptedRecovery =
    preview.state === "generating" &&
    (preview.status === "Reloading" || preview.status === "Restarting");

  if (
    !isInterruptedRecovery ||
    !preview.available ||
    !previewArtifact ||
    !isArtifactPreviewable(previewArtifact)
  ) {
    return preview;
  }

  return {
    ...preview,
    state: "ready",
    status: "Ready when opened",
    restoredFromInterruptedSession: true,
    summary: `Restored ${previewArtifact.title} for a fresh live preview.`,
    trigger: `Preview restored ${previewArtifact.title}`
  };
}

function getPersistedPreviewSourceRecoveryIssue(
  snapshot: AssistantWorkspaceSnapshot,
  record: WorkspaceSessionRecord,
  previewArtifact: ArtifactSummary | null
): string | null {
  if (!previewArtifact || !isCompactedCodeRecoveryNotice(snapshot.code.excerpt)) {
    return null;
  }

  const previewOwnsRestoredCode =
    previewArtifact.id === record.previewArtifactId ||
    previewArtifact.id === record.activeArtifactId ||
    previewArtifact.title === snapshot.code.title;
  if (!previewOwnsRestoredCode) {
    return null;
  }

  return "Its executable source was not restored because it exceeds the local session limit. Open the saved artifact or rerun the workflow before starting a live preview.";
}

function isCompactedCodeRecoveryNotice(excerpt: string[]): boolean {
  return (
    excerpt[0] === "// The generated source exceeds the local session restore limit." &&
    excerpt[1] ===
      "// Open the saved artifact or rerun the workflow instead of previewing a partial file."
  );
}

function findRestoredPreviewArtifact(
  artifacts: ArtifactSummary[],
  record: WorkspaceSessionRecord,
  preview: PreviewSummary
): ArtifactSummary | null {
  const candidateIds = [
    record.previewArtifactId,
    record.activeArtifactId,
    preview.sourceArtifactId,
    preview.artifactName
  ].filter(Boolean);
  return (
    candidateIds
      .map((candidateId) =>
        artifacts.find(
          (artifact) => artifact.id === candidateId || artifact.title === candidateId
        )
      )
      .find((artifact): artifact is ArtifactSummary => artifact !== undefined) ?? null
  );
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
    sidebarCollapsed: layout?.sidebarCollapsed ?? defaultWorkspaceLayoutState.sidebarCollapsed,
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
        : defaultWorkspacePreferences.showDebugPanels,
    workflowMode: isWorkflowExecutionMode(preferences?.workflowMode)
      ? preferences.workflowMode
      : defaultWorkspacePreferences.workflowMode,
    creativity: isCreativityProfile(preferences?.creativity)
      ? preferences.creativity
      : defaultWorkspacePreferences.creativity,
    personalizationEnabled:
      typeof preferences?.personalizationEnabled === "boolean"
        ? preferences.personalizationEnabled
        : defaultWorkspacePreferences.personalizationEnabled,
    headingFontSize: isFontScale(preferences?.headingFontSize)
      ? preferences.headingFontSize
      : defaultWorkspacePreferences.headingFontSize,
    uiFontSize: isFontScale(preferences?.uiFontSize)
      ? preferences.uiFontSize
      : defaultWorkspacePreferences.uiFontSize,
    labelFontSize: isFontScale(preferences?.labelFontSize)
      ? preferences.labelFontSize
      : defaultWorkspacePreferences.labelFontSize,
    codeFontSize: isFontScale(preferences?.codeFontSize)
      ? preferences.codeFontSize
      : defaultWorkspacePreferences.codeFontSize,
    feedbackSignals: normalizeFeedbackSignals(preferences?.feedbackSignals),
    evaluationHistory: normalizeEvaluationHistory(preferences?.evaluationHistory)
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
      value.schemaVersion === 4 ||
      value.schemaVersion === 5) &&
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
    // Image bytes are request-scoped. Older records may contain attachments from
    // the previous session-scoped contract, but they must never be rehydrated.
    imageAttachments: [],
    uploadError: null
  });
}

function requestScopedPersistenceBoundary(
  multimodal: MultimodalSummary
): MultimodalSummary {
  return {
    ...multimodal,
    state: "empty",
    status: "No image references",
    detail:
      "Image inputs are request-scoped and are not stored with workspace sessions.",
    imageAttachments: [],
    error: null
  };
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
    steps: mergedSteps,
    productOutcome: readProductOutcome(value.productOutcome)
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
      excerpt: compactPersistedCodeExcerpt(snapshot.code.excerpt)
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
        ? compactPersistedArtifactContent(artifact.content)
        : artifact.content
  }));
}

function compactPersistedCodeExcerpt(excerpt: string[]): string[] {
  const source = excerpt.join("\n").replace(/\r\n/g, "\n");
  if (source.length <= maxPersistedCodeChars) {
    return source.split("\n");
  }

  // Never persist a partial executable source: it can look previewable but fail at runtime.
  return [
    "// The generated source exceeds the local session restore limit.",
    "// Open the saved artifact or rerun the workflow instead of previewing a partial file."
  ];
}

function compactPersistedArtifactContent(value: string): string | undefined {
  const source = value.replace(/\r\n/g, "\n");
  return source.length <= maxPersistedArtifactContentChars ? source : undefined;
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
    // A first-run profile has no saved record yet. Ask the supported endpoint
    // for an explicit empty result so the expected bootstrap path does not
    // surface as a browser-console resource failure.
    url.searchParams.set("missingSession", "empty");
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

    if (response.status === 204) {
      return { record: null, error: null };
    }

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
    const retained = listLocalWorkspaceSessions(record.userId, resolvedStorage).filter(
      (summary) => summary.sessionId !== record.sessionId
    );
    writeLocalWorkspaceSessionIndex(
      [
        {
          sessionId: record.sessionId,
          projectId: record.projectId,
          title: record.title,
          updatedAt: record.updatedAt ?? null,
          artifactCount: record.artifacts.length
        },
        ...retained
      ],
      resolvedStorage,
      record.userId
    );
    return true;
  } catch {
    return false;
  }
}

function workspaceSessionStorageKey(userId: string, sessionId: string) {
  return `cca.workspace.${userId}.${sessionId}`;
}

function workspaceSessionIndexStorageKey(userId: string) {
  return `cca.workspace.${userId}.session-index.v1`;
}

function writeLocalWorkspaceSessionIndex(
  summaries: WorkspaceSessionSummary[],
  storage: Storage,
  userId: string
) {
  storage.setItem(
    workspaceSessionIndexStorageKey(userId),
    JSON.stringify(summaries.slice(0, 80))
  );
}

function normalizeSessionSummary(value: unknown): WorkspaceSessionSummary | null {
  if (!isRecord(value)) {
    return null;
  }
  if (
    typeof value.sessionId !== "string" ||
    typeof value.projectId !== "string" ||
    typeof value.title !== "string" ||
    (typeof value.updatedAt !== "string" && value.updatedAt !== null) ||
    typeof value.artifactCount !== "number" ||
    value.artifactCount < 0
  ) {
    return null;
  }
  return {
    sessionId: value.sessionId,
    projectId: value.projectId,
    title: value.title,
    updatedAt: value.updatedAt,
    artifactCount: Math.floor(value.artifactCount)
  };
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

function createWorkspaceIdentitySuffix() {
  const generated = globalThis.crypto?.randomUUID?.().replaceAll("-", "");
  if (generated) {
    return generated;
  }
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 12)}`;
}

function createEphemeralWorkspaceIdentity(): WorkspaceIdentity {
  if (typeof window === "undefined") {
    return defaultWorkspaceIdentity;
  }
  const suffix = createWorkspaceIdentitySuffix();
  return {
    userId: `browser-user-${suffix}`,
    sessionId: `browser-session-${suffix}`,
    projectId: `browser-workspace-${suffix}`
  };
}

function isWorkspaceIdentity(value: unknown): value is WorkspaceIdentity {
  return (
    isRecord(value) &&
    typeof value.userId === "string" &&
    value.userId.length > 0 &&
    typeof value.sessionId === "string" &&
    value.sessionId.length > 0 &&
    typeof value.projectId === "string" &&
    value.projectId.length > 0
  );
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
    value === "codex_white" ||
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
    (value.sidebarCollapsed === undefined || typeof value.sidebarCollapsed === "boolean") &&
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
    typeof value.showDebugPanels === "boolean" &&
    (value.workflowMode === undefined || isWorkflowExecutionMode(value.workflowMode)) &&
    (value.creativity === undefined || isCreativityProfile(value.creativity)) &&
    (value.personalizationEnabled === undefined ||
      typeof value.personalizationEnabled === "boolean") &&
    (value.headingFontSize === undefined || isFontScale(value.headingFontSize)) &&
    (value.uiFontSize === undefined || isFontScale(value.uiFontSize)) &&
    (value.labelFontSize === undefined || isFontScale(value.labelFontSize)) &&
    (value.codeFontSize === undefined || isFontScale(value.codeFontSize)) &&
    (value.feedbackSignals === undefined || Array.isArray(value.feedbackSignals)) &&
    (value.evaluationHistory === undefined || Array.isArray(value.evaluationHistory))
  );
}

function isWorkflowExecutionMode(value: unknown): value is WorkflowExecutionMode {
  return value === "auto" || value === "single_agent" || value === "multi_agent";
}

function isCreativityProfile(value: unknown): value is CreativityProfile {
  return value === "controlled" || value === "balanced" || value === "exploratory";
}

function isFontScale(value: unknown): value is FontScale {
  return value === "small" || value === "medium" || value === "large";
}

function normalizeFeedbackSignals(value: unknown): FeedbackSignal[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map(normalizeFeedbackSignal)
    .filter((signal): signal is FeedbackSignal => signal !== null)
    .slice(-120);
}

function isFeedbackSignal(value: unknown): value is FeedbackSignal {
  if (!isRecord(value)) {
    return false;
  }
  return (
    typeof value.id === "string" &&
    (value.sentiment === "positive" || value.sentiment === "negative") &&
    (typeof value.comment === "string" || value.comment === null) &&
    typeof value.sessionId === "string" &&
    (typeof value.artifactId === "string" || value.artifactId === null) &&
    (typeof value.artifactTitle === "string" || value.artifactTitle === null) &&
    (typeof value.domain === "string" || value.domain === null) &&
    typeof value.workflowMode === "string" &&
    isCreativityProfile(value.creativity) &&
    Array.isArray(value.categories) &&
    typeof value.createdAt === "string" &&
    (value.promptExcerpt === undefined || value.promptExcerpt === null || typeof value.promptExcerpt === "string") &&
    (value.providerName === undefined || value.providerName === null || typeof value.providerName === "string") &&
    (value.providerModel === undefined || value.providerModel === null || typeof value.providerModel === "string") &&
    (value.requestedTemperature === undefined || typeof value.requestedTemperature === "number") &&
    (value.effectiveTemperature === undefined || value.effectiveTemperature === null || typeof value.effectiveTemperature === "number") &&
    (value.parameterApplication === undefined || value.parameterApplication === "requested_not_confirmed" || value.parameterApplication === "provider_reported") &&
    (value.productOutcome === undefined || value.productOutcome === null || typeof value.productOutcome === "string")
  );
}

function normalizeFeedbackSignal(value: unknown): FeedbackSignal | null {
  if (!isFeedbackSignal(value)) {
    return null;
  }
  return {
    ...value,
    promptExcerpt:
      typeof value.promptExcerpt === "string" ? value.promptExcerpt : null,
    providerName: typeof value.providerName === "string" ? value.providerName : null,
    providerModel: typeof value.providerModel === "string" ? value.providerModel : null,
    requestedTemperature:
      typeof value.requestedTemperature === "number"
        ? value.requestedTemperature
        : buildGenerationControls(value.creativity).requestedTemperature,
    effectiveTemperature:
      typeof value.effectiveTemperature === "number"
        ? value.effectiveTemperature
        : null,
    parameterApplication:
      value.parameterApplication === "provider_reported"
        ? "provider_reported"
        : "requested_not_confirmed",
    productOutcome:
      typeof value.productOutcome === "string" ? value.productOutcome : null
  };
}

function normalizeEvaluationHistory(value: unknown): EvaluationHistoryRecord[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map(normalizeEvaluationHistoryRecord)
    .filter((record): record is EvaluationHistoryRecord => record !== null)
    .slice(-24);
}

function normalizeEvaluationHistoryRecord(value: unknown): EvaluationHistoryRecord | null {
  if (
    !isRecord(value) ||
    typeof value.id !== "string" ||
    typeof value.status !== "string" ||
    typeof value.detail !== "string" ||
    typeof value.evaluatedAt !== "string"
  ) {
    return null;
  }
  return {
    id: value.id,
    runId: typeof value.runId === "string" ? value.runId : null,
    datasetId: typeof value.datasetId === "string" ? value.datasetId : null,
    metrics: Array.isArray(value.metrics)
      ? value.metrics.filter((metric): metric is string => typeof metric === "string")
      : [],
    status: value.status,
    detail: value.detail,
    evaluatedAt: value.evaluatedAt,
    resultRows: typeof value.resultRows === "number" ? value.resultRows : null,
    metricFailures: typeof value.metricFailures === "number" ? value.metricFailures : null,
    dryRun: typeof value.dryRun === "boolean" ? value.dryRun : null,
    providerCallsAllowed:
      typeof value.providerCallsAllowed === "boolean" ? value.providerCallsAllowed : null,
    benchmark: normalizeEvaluationBenchmark(value.benchmark)
  };
}

function normalizeEvaluationBenchmark(
  value: unknown
): NonNullable<EvaluationHistoryRecord["benchmark"]> | null {
  if (
    !isRecord(value) ||
    (value.schemaVersion !== 2 && value.schemaVersion !== 3) ||
    typeof value.id !== "string" ||
    typeof value.datasetVersion !== "string" ||
    typeof value.datasetFingerprint !== "string" ||
    typeof value.promptVersion !== "string" ||
    typeof value.scope !== "string" ||
    !Array.isArray(value.selectedCaseIds) ||
    !value.selectedCaseIds.every((item) => typeof item === "string") ||
    typeof value.completedAt !== "string" ||
    !isRecord(value.counts) ||
    !Array.isArray(value.categoryResults) ||
    !value.categoryResults.every((item) =>
      isRecord(item) && typeof item.category === "string" && typeof item.status === "string"
    ) ||
    !Array.isArray(value.caseResults) ||
    !value.caseResults.every((item) =>
      isRecord(item) &&
      typeof item.caseId === "string" &&
      typeof item.title === "string" &&
      typeof item.status === "string" &&
      Array.isArray(item.categories) &&
      Array.isArray(item.metrics) &&
      item.metrics.every((metric) =>
        isRecord(metric) && typeof metric.id === "string" && typeof metric.status === "string"
      )
    ) ||
    !Array.isArray(value.recommendations) ||
    !isRecord(value.ragas)
  ) {
    return null;
  }
  return value as unknown as NonNullable<EvaluationHistoryRecord["benchmark"]>;
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
