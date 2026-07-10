import type {
  ArtifactSummary,
  AssistantWorkspaceSnapshot,
  PreviewSummary,
  WorkflowNodeId
} from "./assistant-client";
import {
  createWorkstationError,
  type WorkstationError
} from "./workstation-errors";
import type { PreviewRuntimeSessionOverride } from "./preview-controller";
import { readPreviewArtifactUpdate } from "./assistant-stream";
import {
  derivePreviewTargetIdFromArtifact,
  formatPreviewTargetLabel,
  normalizePreviewTargetId
} from "./preview-targets";
import {
  getGlslRuntimeSourceSupportIssue,
  getP5RuntimeSourceSupportIssue,
  getThreeRuntimeSourceSupportIssue
} from "./preview-source-classification";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

type BuildPreviewRuntimeSummaryInput = {
  artifacts: ArtifactSummary[];
  basePreview: PreviewSummary;
  isOpen: boolean;
  previewArtifactId: string;
  sessionOverride?: PreviewRuntimeSessionOverride | null;
  streamError: WorkstationError | null;
  traceEvents: WorkflowRuntimeTraceEvent[];
  workflow: AssistantWorkspaceSnapshot["workflow"];
};

const previewGeneratingNodes = new Set<WorkflowNodeId>([
  "generation",
  "artifact_extraction",
  "preview_preparation",
  "review",
  "refinement",
  "finalization"
]);

export function buildPreviewRuntimeSummary({
  artifacts,
  basePreview,
  isOpen,
  previewArtifactId,
  sessionOverride = null,
  streamError,
  traceEvents,
  workflow
}: BuildPreviewRuntimeSummaryInput): PreviewSummary {
  const contextArtifact =
    artifacts.find((artifact) => artifact.id === previewArtifactId) ??
    artifacts.find((artifact) => artifact.title === basePreview.artifactName) ??
    artifacts.find((artifact) => artifact.title === basePreview.sourceArtifactName) ??
    artifacts.find((artifact) => isArtifactPreviewable(artifact)) ??
    artifacts[0] ??
    null;
  const contextMatchesBasePreview = matchesBasePreviewContext(
    contextArtifact,
    basePreview
  );
  const traceHasPreview = traceEvents.some(
    ({ event }) => readPreviewArtifactUpdate(event) !== null
  );
  const workspaceHasPreview =
    (basePreview.available &&
      contextMatchesBasePreview &&
      (contextArtifact ? isArtifactPreviewable(contextArtifact) : true)) ||
    (contextArtifact ? isArtifactPreviewable(contextArtifact) : false) ||
    traceHasPreview;
  const activeSessionOverride = matchesPreviewSessionOverride(
    sessionOverride,
    contextArtifact,
    basePreview
  )
    ? sessionOverride
    : null;
  const previewUpdate = activeSessionOverride
    ? null
    : findLatestPreviewUpdate(traceEvents, contextArtifact?.id ?? previewArtifactId);
  const sourceArtifact =
    artifacts.find((artifact) => artifact.id === (previewUpdate?.artifactId ?? "")) ??
    (contextMatchesBasePreview
      ? artifacts.find((artifact) => artifact.id === basePreview.sourceArtifactId) ??
        artifacts.find((artifact) => artifact.title === basePreview.sourceArtifactName)
      : null) ??
    contextArtifact;
  const outputArtifact =
    artifacts.find(
      (artifact) => artifact.id === (previewUpdate?.previewArtifactId ?? "")
    ) ??
    (contextMatchesBasePreview
      ? artifacts.find(
          (artifact) =>
            artifact.type === "preview" &&
            artifact.title === basePreview.outputArtifactName
        )
      : null) ??
    (contextArtifact?.type === "preview" ? contextArtifact : null);
  const contextIsPreviewable =
    (contextArtifact ? isArtifactPreviewable(contextArtifact) : false) ||
    previewUpdate !== null;
  const hasConcretePreviewOutput =
    outputArtifact !== null ||
    (contextMatchesBasePreview &&
      (basePreview.state === "ready" || basePreview.state === "error") &&
      Boolean(basePreview.outputArtifactName));
  const state = resolvePreviewRuntimeState({
    activeSessionOverride,
    basePreview,
    contextIsPreviewable,
    hasConcretePreviewOutput,
    previewUpdate,
    streamError,
    workflow,
    workspaceHasPreview
  });
  const artifactName =
    outputArtifact?.title ?? contextArtifact?.title ?? basePreview.artifactName;
  const sourceArtifactName =
    sourceArtifact?.title ??
    contextArtifact?.title ??
    basePreview.sourceArtifactName ??
    artifactName;
  const outputArtifactName =
    activeSessionOverride
      ? ""
      : outputArtifact?.title ??
        previewUpdate?.previewArtifactId ??
        basePreview.outputArtifactName;
  const previewError = buildPreviewRuntimeError({
    artifactName: sourceArtifactName || artifactName,
    basePreview,
    previewUpdate,
    state
  });
  const targetId =
    normalizePreviewTargetId(previewUpdate?.target ?? null) ??
    derivePreviewTargetIdFromArtifact(outputArtifact ?? contextArtifact) ??
    normalizePreviewTargetId(contextMatchesBasePreview ? basePreview.targetId : null) ??
    null;

  return {
    ...basePreview,
    active: workspaceHasPreview ? isOpen : false,
    artifactName: artifactName || "No preview target",
    available: workspaceHasPreview,
    collapsed: workspaceHasPreview ? !isOpen : true,
    outputArtifactName,
    renderer: previewUpdate?.rendererId ?? basePreview.renderer,
    sourceArtifactId:
      sourceArtifact?.id ?? previewUpdate?.artifactId ?? basePreview.sourceArtifactId,
    sourceArtifactName: sourceArtifactName || "",
    state,
    status: formatPreviewRuntimeStatus({
      activeSessionOverride,
      basePreview,
      isOpen,
      previewUpdate,
      state
    }),
    summary: buildPreviewRuntimeSummaryCopy({
      activeSessionOverride,
      artifactName: sourceArtifactName || artifactName,
      basePreview,
      hasConcretePreviewOutput,
      outputArtifactName,
      previewError,
      previewUpdate,
      state,
      streamError,
      workflow
    }),
    target: formatPreviewTargetCopy(targetId, contextMatchesBasePreview ? basePreview.target : ""),
    targetId: targetId ?? "",
    title: workspaceHasPreview ? basePreview.title : "Preview unavailable",
    trigger: buildPreviewRuntimeTrigger({
      activeSessionOverride,
      artifactName,
      basePreview,
      previewUpdate,
      state,
      workflow
    }),
    version: basePreview.version,
    error: previewError
  };
}

export function isArtifactPreviewable(artifact: ArtifactSummary): boolean {
  if (!artifact.actions.includes("Preview") && artifact.type !== "preview") {
    return false;
  }

  const source = artifact.content?.trim();
  if (!source) {
    return true;
  }

  const title = artifact.title.toLowerCase();
  const isP5Artifact =
    artifact.runtime === "p5" ||
    artifact.rendererId === "surface.p5" ||
    title.endsWith(".p5.js") ||
    title.endsWith(".p5.ts");
  if (isP5Artifact && getP5RuntimeSourceSupportIssue(source)) {
    return false;
  }

  const isGlslArtifact =
    artifact.runtime === "glsl" ||
    artifact.rendererId === "surface.glsl" ||
    /\.(?:glsl|frag|fs)$/i.test(title);
  if (isGlslArtifact && getGlslRuntimeSourceSupportIssue(source)) {
    return false;
  }

  const isThreeArtifact =
    artifact.runtime === "three" ||
    artifact.rendererId === "surface.three" ||
    title.endsWith(".three.js") ||
    title.endsWith(".three.ts") ||
    title.endsWith(".r3f.tsx");

  return !(isThreeArtifact && getThreeRuntimeSourceSupportIssue(source));
}

function resolvePreviewRuntimeState({
  activeSessionOverride,
  basePreview,
  contextIsPreviewable,
  hasConcretePreviewOutput,
  previewUpdate,
  streamError,
  workflow,
  workspaceHasPreview
}: {
  activeSessionOverride: PreviewRuntimeSessionOverride | null;
  basePreview: PreviewSummary;
  contextIsPreviewable: boolean;
  hasConcretePreviewOutput: boolean;
  previewUpdate: ReturnType<typeof findLatestPreviewUpdate>;
  streamError: WorkstationError | null;
  workflow: AssistantWorkspaceSnapshot["workflow"];
  workspaceHasPreview: boolean;
}): PreviewSummary["state"] {
  if (!workspaceHasPreview) {
    return "unavailable";
  }

  if (
    activeSessionOverride?.mode === "restarting" ||
    activeSessionOverride?.mode === "reloading" ||
    activeSessionOverride?.mode === "settled"
  ) {
    return "generating";
  }

  if (activeSessionOverride?.mode === "cleared") {
    return isPreviewRuntimeActive(workflow) && contextIsPreviewable
      ? "generating"
      : "unavailable";
  }

  if (previewUpdate?.status === "failed") {
    return "error";
  }

  if (!contextIsPreviewable && !previewUpdate) {
    return "unavailable";
  }

  if (previewUpdate?.status === "succeeded" || previewUpdate?.status === "skipped") {
    return "ready";
  }

  if (hasConcretePreviewOutput) {
    return basePreview.state === "error" ? "error" : "ready";
  }

  if (basePreview.restoredFromInterruptedSession && contextIsPreviewable) {
    return "ready";
  }

  if (isPreviewRuntimeActive(workflow)) {
    return "generating";
  }

  if (streamError) {
    return "unavailable";
  }

  return "unavailable";
}

function buildPreviewRuntimeSummaryCopy({
  activeSessionOverride,
  artifactName,
  basePreview,
  hasConcretePreviewOutput,
  outputArtifactName,
  previewError,
  previewUpdate,
  state,
  streamError,
  workflow
}: {
  activeSessionOverride: PreviewRuntimeSessionOverride | null;
  artifactName: string;
  basePreview: PreviewSummary;
  hasConcretePreviewOutput: boolean;
  outputArtifactName: string;
  previewError: WorkstationError | null;
  previewUpdate: ReturnType<typeof findLatestPreviewUpdate>;
  state: PreviewSummary["state"];
  streamError: WorkstationError | null;
  workflow: AssistantWorkspaceSnapshot["workflow"];
}) {
  if (activeSessionOverride?.mode === "restarting") {
    return `Restart requested for ${artifactName}. Preview output will update when the next runtime event arrives.`;
  }

  if (activeSessionOverride?.mode === "reloading") {
    return `Reload requested for ${artifactName}. Restoring the latest preview context when runtime metadata becomes available.`;
  }

  if (activeSessionOverride?.mode === "cleared") {
    return `Preview state cleared for ${artifactName}. Reload or reset the session to restore the latest runtime context.`;
  }

  if (state === "error") {
    return previewError?.userMessage ?? `Preview runtime failed for ${artifactName}.`;
  }

  if (state === "generating") {
    return `Preview context for ${artifactName} is following ${workflow.currentStep.toLowerCase()} while output metadata is still arriving.`;
  }

  if (state === "unavailable") {
    if (streamError) {
      return `The current run stopped before a preview result was emitted for ${artifactName}.`;
    }

    return `${artifactName} does not expose a preview target in this workspace yet. Choose a previewable artifact to restore the shelf context.`;
  }

  if (previewUpdate?.status === "skipped") {
    return (
      previewUpdate.summary ??
      "Preview target metadata is ready; live execution starts when a supported preview runtime opens."
    );
  }

  if (previewUpdate?.status === "succeeded") {
    return (
      previewUpdate.summary ??
      `Preview output is ready${outputArtifactName ? ` as ${outputArtifactName}` : ""}.`
    );
  }

  if (hasConcretePreviewOutput && basePreview.state === "ready") {
    return basePreview.summary;
  }

  return basePreview.summary;
}

function buildPreviewRuntimeError({
  artifactName,
  basePreview,
  previewUpdate,
  state
}: {
  artifactName: string;
  basePreview: PreviewSummary;
  previewUpdate: ReturnType<typeof findLatestPreviewUpdate>;
  state: PreviewSummary["state"];
}) {
  if (state !== "error") {
    return null;
  }

  if (previewUpdate?.error) {
    return previewUpdate.error;
  }

  if (basePreview.error) {
    return basePreview.error;
  }

  return createWorkstationError({
    type: "preview_runtime_failed",
    category: "preview_runtime",
    subsystem: basePreview.renderer || "preview_runtime",
    userMessage: `Preview output failed for ${artifactName}.`,
    recoverable: true,
    suggestedAction:
      "Reload the preview state or reset the preview session before trying again.",
    retryLabel: "Reload preview state",
    resetLabel: "Reset preview session"
  });
}

function buildPreviewRuntimeTrigger({
  activeSessionOverride,
  artifactName,
  basePreview,
  previewUpdate,
  state,
  workflow
}: {
  activeSessionOverride: PreviewRuntimeSessionOverride | null;
  artifactName: string;
  basePreview: PreviewSummary;
  previewUpdate: ReturnType<typeof findLatestPreviewUpdate>;
  state: PreviewSummary["state"];
  workflow: AssistantWorkspaceSnapshot["workflow"];
}) {
  if (activeSessionOverride?.mode === "restarting") {
    return `Preview restart ${artifactName}`;
  }

  if (activeSessionOverride?.mode === "reloading") {
    return `Preview reload ${artifactName}`;
  }

  if (activeSessionOverride?.mode === "cleared") {
    return `Preview cleared ${artifactName}`;
  }

  if (state === "generating") {
    return `Workflow ${workflow.currentStep}`;
  }

  if (state === "ready") {
    return `Preview ${artifactName}`;
  }

  if (previewUpdate?.artifactId) {
    return `Preview ${artifactName}`;
  }

  return basePreview.trigger;
}

function formatPreviewRuntimeStatus({
  activeSessionOverride,
  basePreview,
  isOpen,
  previewUpdate,
  state
}: {
  activeSessionOverride: PreviewRuntimeSessionOverride | null;
  basePreview: PreviewSummary;
  isOpen: boolean;
  previewUpdate: ReturnType<typeof findLatestPreviewUpdate>;
  state: PreviewSummary["state"];
}) {
  if (activeSessionOverride?.mode === "restarting") {
    return "Restarting";
  }

  if (activeSessionOverride?.mode === "reloading") {
    return "Reloading";
  }

  if (activeSessionOverride?.mode === "cleared") {
    return "Cleared";
  }

  switch (state) {
    case "generating":
      return "Generating";
    case "ready":
      if (isOpen) {
        return "Preview open";
      }
      if (previewUpdate === null && basePreview.state === "ready") {
        return basePreview.status === "Deferred renderer"
          ? basePreview.status
          : "Ready when opened";
      }
      return previewUpdate?.status === "skipped"
        ? "Deferred renderer"
        : "Ready when opened";
    case "error":
      return "Preview failed";
    case "unavailable":
      return "Unavailable";
    default:
      return "Ready when opened";
  }
}

function matchesPreviewSessionOverride(
  sessionOverride: PreviewRuntimeSessionOverride | null,
  contextArtifact: ArtifactSummary | null,
  basePreview: PreviewSummary
) {
  if (!sessionOverride) {
    return false;
  }

  return (
    sessionOverride.artifactId === contextArtifact?.id ||
    sessionOverride.artifactId === basePreview.sourceArtifactId
  );
}

function findLatestPreviewUpdate(
  traceEvents: WorkflowRuntimeTraceEvent[],
  artifactId: string
) {
  for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
    const previewUpdate = readPreviewArtifactUpdate(traceEvents[index].event);
    if (!previewUpdate) {
      continue;
    }

    if (
      previewUpdate.artifactId === artifactId ||
      previewUpdate.previewArtifactId === artifactId
    ) {
      return previewUpdate;
    }
  }

  return null;
}

function isPreviewRuntimeActive(workflow: AssistantWorkspaceSnapshot["workflow"]) {
  const normalizedStatus = workflow.status.toLowerCase();
  return (
    normalizedStatus === "running" &&
    previewGeneratingNodes.has(workflow.currentNode)
  );
}

function formatPreviewTargetCopy(targetId: string | null, fallbackTarget: string) {
  return formatPreviewTargetLabel(normalizePreviewTargetId(targetId)) ?? fallbackTarget;
}

function matchesBasePreviewContext(
  contextArtifact: ArtifactSummary | null,
  basePreview: PreviewSummary
) {
  if (!contextArtifact) {
    return basePreview.available;
  }

  return (
    contextArtifact.id === basePreview.sourceArtifactId ||
    contextArtifact.title === basePreview.sourceArtifactName ||
    contextArtifact.title === basePreview.outputArtifactName ||
    contextArtifact.title === basePreview.artifactName
  );
}
