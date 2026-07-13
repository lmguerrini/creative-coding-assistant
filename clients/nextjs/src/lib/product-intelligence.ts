import type { ArtifactSummary, AssistantWorkspaceSnapshot } from "./assistant-client";
import type { ConversationContextModel } from "./conversation-context";
import {
  getDomainExperienceRecord,
  type DomainExperienceCatalog
} from "./domain-experience";
import type { ProviderTelemetryModel } from "./provider-telemetry";
import type { RetrievalRuntimeModel } from "./retrieval-runtime";
import type { RuntimeConsoleModel } from "./runtime-console";
import type { SessionIntelligenceModel } from "./session-intelligence";
import type { TelemetryDashboardModel } from "./telemetry-dashboard";
import type { V3InspectorPanelsModel } from "./v3-inspector-panels";
import type { WorkflowExecutionModel } from "./workflow-execution";
import type { WorkflowRuntimeModel } from "./workflow-runtime";
import type { WorkstationDashboardModel } from "./workstation-dashboard";

export const productIntelligenceCategories = [
  "Overview",
  "Architecture",
  "Workflow",
  "Agents",
  "Runtime",
  "Preview",
  "Code",
  "Artifacts",
  "Domains",
  "Retrieval",
  "Knowledge Base",
  "Memory",
  "Sessions",
  "Providers",
  "Telemetry",
  "Metrics",
  "Validation",
  "Product Bugs",
  "LangSmith",
  "Settings"
] as const;

export type ProductIntelligenceCategory =
  (typeof productIntelligenceCategories)[number];

export type ProductIntelligenceTone =
  | "ready"
  | "active"
  | "attention"
  | "empty";

export type ProductIntelligenceMetric = {
  label: string;
  value: string;
};

export type ProductIntelligenceSection = {
  category: ProductIntelligenceCategory;
  tone: ProductIntelligenceTone;
  summary: string;
  detail: string;
  metrics: ProductIntelligenceMetric[];
  notes: string[];
};

export type ProductIntelligenceModel = {
  activeArtifactId?: string | null;
  activeDomainId: string | null;
  domainExperience: DomainExperienceCatalog;
  artifactRegistry: ArtifactSummary[];
  session: {
    id: string;
    title: string;
  };
  sections: ProductIntelligenceSection[];
  summary: {
    activeCount: number;
    attentionCount: number;
    readyCount: number;
  };
  /**
   * Published source models for richer Dashboard views. These are deliberately
   * presentation data only: the Dashboard must not infer hidden provider
   * reasoning or manufacture runtime state from them.
   */
  details?: {
    conversationContext: ConversationContextModel;
    providerTelemetry: ProviderTelemetryModel;
    retrievalRuntime: RetrievalRuntimeModel;
    runtimeConsole: RuntimeConsoleModel;
    sessionIntelligence: SessionIntelligenceModel;
    snapshot: AssistantWorkspaceSnapshot;
    telemetryDashboard: TelemetryDashboardModel;
    v3InspectorPanels: V3InspectorPanelsModel;
    workflowExecution: WorkflowExecutionModel;
    workflowRuntime: WorkflowRuntimeModel;
    workstationDashboard: WorkstationDashboardModel;
  };
};

type BuildProductIntelligenceInput = {
  activeArtifactId?: string | null;
  conversationContext: ConversationContextModel;
  domainExperience: DomainExperienceCatalog;
  providerTelemetry: ProviderTelemetryModel;
  retrievalRuntime: RetrievalRuntimeModel;
  runtimeConsole: RuntimeConsoleModel;
  sessionIntelligence: SessionIntelligenceModel;
  snapshot: AssistantWorkspaceSnapshot;
  telemetryDashboard: TelemetryDashboardModel;
  v3InspectorPanels: V3InspectorPanelsModel;
  workflowExecution: WorkflowExecutionModel;
  workflowRuntime: WorkflowRuntimeModel;
  workstationDashboard: WorkstationDashboardModel;
};

/**
 * Shared, presentation-neutral information architecture for the full Dashboard
 * and the compact Inspector. It only rephrases existing product models; it does
 * not infer provider, workflow, or runtime state.
 */
export function buildProductIntelligenceModel({
  activeArtifactId = null,
  conversationContext,
  domainExperience,
  providerTelemetry,
  retrievalRuntime,
  runtimeConsole,
  sessionIntelligence,
  snapshot,
  telemetryDashboard,
  v3InspectorPanels,
  workflowExecution,
  workflowRuntime,
  workstationDashboard
}: BuildProductIntelligenceInput): ProductIntelligenceModel {
  const activeArtifact =
    snapshot.artifacts.find((artifact) => artifact.id === activeArtifactId) ??
    snapshot.artifacts[0] ??
    null;
  const activeDomainId = activeArtifact?.domain ?? null;
  const activeDomain = getDomainExperienceRecord(domainExperience, activeDomainId);
  const workflowTone = toneForWorkflow(workflowRuntime.summary.activity.state);
  const previewTone = toneForPreview(snapshot.preview.state, snapshot.preview.available);
  const retrievalTone = toneForRetrieval(retrievalRuntime.summary.state);
  const knowledgeBaseTone = toneForKnowledgeBase(domainExperience.knowledgeBase.status);
  const providerTone = toneForProvider(providerTelemetry.status);
  const runtimeTone = toneForRuntime(runtimeConsole.health.signal);
  const validationAttention = workstationDashboard.summary.errorCount > 0;
  const v3Available = v3InspectorPanels.summary.availableItemCount;
  const sections: ProductIntelligenceSection[] = [
    section({
      category: "Overview",
      tone: workflowTone,
      summary: workflowRuntime.summary.activity.label,
      detail: workflowRuntime.summary.activity.detail,
      metrics: [
        metric("Outcome", workflowRuntime.summary.productOutcome.product_outcome),
        metric("Artifact", activeArtifact?.title ?? "No artifact yet"),
        metric("Preview", previewLabel(snapshot.preview.state, snapshot.preview.available))
      ],
      notes: [
        sessionIntelligence.metadata.session_summary,
        sessionIntelligence.metadata.active_request_summary
      ]
    }),
    section({
      category: "Architecture",
      tone: workflowExecution.state === "available" ? "ready" : "empty",
      summary:
        workflowExecution.state === "available"
          ? "Execution architecture is published for this run."
          : "Architecture details appear when a run publishes its execution plan.",
      detail: workflowExecution.rationale,
      metrics: [
        metric("Workflow nodes", `${workflowRuntime.summary.reached}/${workflowRuntime.summary.total}`),
        metric("Route", formatExecutionMode(workflowExecution.resolvedMode)),
        metric("Source", workflowExecution.source)
      ],
      notes: workflowExecution.agentRoles.length
        ? [`Active roles: ${workflowExecution.agentRoles.join(", ")}`]
        : ["Run a request to inspect the published execution route."]
    }),
    section({
      category: "Workflow",
      tone: workflowTone,
      summary: workflowRuntime.summary.activity.label,
      detail: workflowRuntime.summary.activity.detail,
      metrics: [
        metric("Current step", workflowRuntime.summary.currentStep),
        metric("Transitions", String(workflowRuntime.summary.transitionCount)),
        metric("Retries", String(workflowRuntime.summary.retryCount))
      ],
      notes: workflowRuntime.error
        ? [workflowRuntime.error.userMessage]
        : workflowRuntime.events.length > 0
          ? workflowRuntime.events
              .slice(-6)
              .reverse()
              .map((event) => `${event.label}: ${event.detail}`)
          : ["No workflow events have been published for this run yet."]
    }),
    section({
      category: "Agents",
      tone: workflowExecution.state === "available" ? "ready" : "empty",
      summary:
        workflowExecution.resolvedMode
          ? `${formatExecutionMode(workflowExecution.resolvedMode)} route`
          : "No agent route published yet",
      detail: workflowExecution.rationale,
      metrics: [
        metric("Requested", formatExecutionMode(workflowExecution.requestedMode)),
        metric("Executed roles", String(workflowExecution.agentRoles.length)),
        metric(
          "Research",
          workflowExecution.researcherRequired == null
            ? "Not published"
            : workflowExecution.researcherRequired
              ? "Required"
              : "Skipped"
        )
      ],
      notes: [workflowExecution.researcherReason ?? "No research decision published."]
    }),
    section({
      category: "Runtime",
      tone: runtimeTone,
      summary: runtimeConsole.health.label,
      detail: runtimeConsole.health.explanation,
      metrics: [
        metric("Renderer", runtimeConsole.context.rendererLabel),
        metric("Runtime", runtimeConsole.context.runtimeTypeLabel),
        metric("Reloads", String(runtimeConsole.reloadHistory.length))
      ],
      notes: runtimeConsole.latestError
        ? [runtimeConsole.latestError]
        : runtimeConsole.diagnostics.length
          ? [...runtimeConsole.diagnostics]
          : ["No runtime diagnostics are currently reported."]
    }),
    section({
      category: "Preview",
      tone: previewTone,
      summary: previewLabel(snapshot.preview.state, snapshot.preview.available),
      detail: snapshot.preview.summary,
      metrics: [
        metric("Target", snapshot.preview.target),
        metric("Renderer", snapshot.preview.renderer || "Pending"),
        metric("Artifact", snapshot.preview.artifactName || "No artifact yet")
      ],
      notes: [
        snapshot.preview.available
          ? "Open the preview shelf to inspect the live artwork."
          : "Generate a browser-compatible artifact to make a preview available."
      ]
    }),
    section({
      category: "Code",
      tone: activeArtifact ? "ready" : "empty",
      summary: activeArtifact?.language ?? "No generated source yet",
      detail:
        activeArtifact?.title
          ? `${activeArtifact.title} is available in the Code inspector.`
          : "Generate an artifact to inspect its source.",
      metrics: [
        metric("Language", activeArtifact?.language ?? "Pending"),
        metric("Artifact", activeArtifact?.title ?? "No artifact yet"),
        metric("Status", activeArtifact?.status ?? "Awaiting generation")
      ],
      notes: ["Code remains available as source, while execution status is reported separately."]
    }),
    section({
      category: "Artifacts",
      tone: activeArtifact ? "ready" : "empty",
      summary: activeArtifact?.title ?? "No artifact selected",
      detail: activeArtifact?.summary ?? "Generated artifacts appear here after a successful run.",
      metrics: [
        metric("Available", String(snapshot.artifacts.length)),
        metric("Type", activeArtifact?.language ?? "Pending"),
        metric("Status", activeArtifact?.status ?? "Awaiting generation")
      ],
      notes: [
        activeArtifact?.previewEligible
          ? "The selected artifact is preview-capable."
          : "Artifact availability and preview readiness are reported separately."
      ]
    }),
    section({
      category: "Domains",
      tone:
        domainExperience.state === "unavailable"
          ? "attention"
          : domainExperience.state === "loading"
            ? "empty"
            : "ready",
      summary: activeDomain
        ? `${activeDomain.displayName} / ${formatDomainDelivery(activeDomain.deliveryKind)}`
        : `${domainExperience.domains.length} registered domain contracts`,
      detail: activeDomain
        ? activeDomain.publicClaimBoundary
        : "Domain contracts distinguish browser preview, code/export, and external-tool handoff.",
      metrics: [
        metric("Live preview", String(domainExperience.domains.filter((domain) => domain.livePreview).length)),
        metric("Code/export", String(domainExperience.domains.filter((domain) => domain.deliveryKind === "code_export").length)),
        metric("External handoff", String(domainExperience.domains.filter((domain) => domain.deliveryKind === "external_handoff").length))
      ],
      notes: activeDomain
        ? [
            `Fallback: ${activeDomain.fallback}`,
            `Knowledge: ${activeDomain.knowledge.indexedSourceCount}/${activeDomain.knowledge.registeredSourceCount} sources indexed.`
          ]
        : ["Choose an artifact or open the Dashboard to inspect the complete registry."]
    }),
    section({
      category: "Retrieval",
      tone: retrievalTone,
      summary: retrievalRuntime.summary.status,
      detail: retrievalRuntime.summary.detail,
      metrics: [
        metric("Sources", String(retrievalRuntime.summary.sourceCount)),
        metric("Chunks", String(retrievalRuntime.summary.chunkCount)),
        metric("Quality", retrievalRuntime.summary.qualityLabel)
      ],
      notes: [retrievalRuntime.request.query ?? "No retrieval query has been published."]
    }),
    section({
      category: "Knowledge Base",
      tone: knowledgeBaseTone,
      summary: domainExperience.knowledgeBase.status.replace(/_/g, " "),
      detail: domainExperience.knowledgeBase.detail,
      metrics: [
        metric("Registered", String(domainExperience.knowledgeBase.registeredSourceCount)),
        metric("Indexed", String(domainExperience.knowledgeBase.indexedSourceCount)),
        metric("Chunks", String(domainExperience.knowledgeBase.indexedChunkCount))
      ],
      notes: [
        domainExperience.knowledgeBase.updateHint,
        domainExperience.knowledgeBase.provenanceBoundary
      ]
    }),
    section({
      category: "Memory",
      tone: conversationContext.source === "stream" ? "ready" : "empty",
      summary: conversationContext.source === "stream" ? "Published context counts" : "No context evidence yet",
      detail: conversationContext.summary,
      metrics: conversationContext.diagnostics.slice(0, 3).map((diagnostic) =>
        metric(diagnostic.label, diagnostic.value)
      ),
      notes: conversationContext.diagnostics.slice(3).map((diagnostic) => diagnostic.detail)
    }),
    section({
      category: "Sessions",
      tone: toneForSession(sessionIntelligence.metadata.completion_status),
      summary: sessionIntelligence.statusLabel,
      detail: sessionIntelligence.metadata.session_summary,
      metrics: [
        metric("Metadata", String(sessionIntelligence.availableMetadataCount)),
        metric("Warnings", String(sessionIntelligence.warningCount)),
        metric("Recommended actions", String(sessionIntelligence.recommendedActionCount))
      ],
      notes: sessionIntelligence.metadata.recommended_next_user_actions.length
        ? sessionIntelligence.metadata.recommended_next_user_actions
        : ["No follow-up action is recommended for this session."]
    }),
    section({
      category: "Providers",
      tone: providerTone,
      summary: `${providerTelemetry.summary.providerLabel} / ${providerTelemetry.summary.modelLabel}`,
      detail: providerTelemetry.summary.generationModeLabel,
      metrics: [
        metric("Streaming", providerTelemetry.summary.streamingStatusLabel),
        metric("Tokens", providerTelemetry.summary.tokenLabel),
        metric("Latency", providerTelemetry.summary.latencyLabel)
      ],
      notes: [
        providerTelemetry.configuration.parameterSource === "not_published"
          ? "Parameters not published by the selected provider."
          : `Parameters sourced from ${providerTelemetry.configuration.parameterSource.replace(/_/g, " ")}.`
      ]
    }),
    section({
      category: "Telemetry",
      tone: toneForTelemetry(telemetryDashboard.status),
      summary: telemetryDashboard.summary.operatorStatus,
      detail: telemetryDashboard.summary.signalLabel,
      metrics: [
        metric("Run events", String(telemetryDashboard.stream.eventCount)),
        metric("Session runs", String(telemetryDashboard.creativeCost.session.runCount)),
        metric("Session tokens", formatTokenCount(telemetryDashboard.creativeCost.session.totalTokens)),
        metric("Tokens this run", providerTelemetry.summary.tokenLabel),
        metric("Estimated cost", providerTelemetry.summary.costLabel),
        metric("Errors", String(telemetryDashboard.stream.errorCount))
      ],
      notes: telemetryDashboard.signals.slice(0, 3).map(
        (signal) => `${signal.label}: ${signal.value} — ${signal.detail}`
      )
    }),
    section({
      category: "Metrics",
      tone: toneForTelemetry(telemetryDashboard.status),
      summary: telemetryDashboard.summary.runtimeLabel,
      detail: telemetryDashboard.runtime.activity.detail,
      metrics: [
        metric("Input tokens", formatTokenCount(providerTelemetry.tokenUsage.inputTokens)),
        metric("Output tokens", formatTokenCount(providerTelemetry.tokenUsage.outputTokens)),
        metric("Total tokens", formatTokenCount(providerTelemetry.tokenUsage.totalTokens)),
        metric("Cost", providerTelemetry.summary.costLabel),
        metric("Run time", telemetryDashboard.summary.runtimeLabel)
      ],
      notes: [providerTelemetry.summary.streamLabel, providerTelemetry.summary.retryLabel]
    }),
    section({
      category: "Validation",
      tone: validationAttention ? "attention" : workstationDashboard.state === "available" ? "ready" : "empty",
      summary: validationAttention
        ? "Validation needs attention"
        : "Validation signals are clear",
      detail: `${workstationDashboard.summary.goodCount} ready, ${workstationDashboard.summary.watchCount} watch, ${workstationDashboard.summary.missingCount} pending.`,
      metrics: [
        metric("Ready", String(workstationDashboard.summary.goodCount)),
        metric("Watch", String(workstationDashboard.summary.watchCount)),
        metric("Errors", String(workstationDashboard.summary.errorCount))
      ],
      notes: workstationDashboard.cards.slice(0, 4).map(
        (card) => `${card.label}: ${card.value}`
      )
    }),
    section({
      category: "Product Bugs",
      tone:
        telemetryDashboard.stream.errorCount > 0 || runtimeConsole.health.signal === "failed"
          ? "attention"
          : "ready",
      summary:
        telemetryDashboard.stream.errorCount > 0 || runtimeConsole.health.signal === "failed"
          ? "A product signal needs attention"
          : "No active product signal",
      detail:
        runtimeConsole.latestError ??
        (telemetryDashboard.stream.errorCount > 0
          ? `${telemetryDashboard.stream.errorCount} stream error${telemetryDashboard.stream.errorCount === 1 ? "" : "s"} reported.`
          : "The current workspace has no reported runtime or stream error."),
      metrics: [
        metric("Stream errors", String(telemetryDashboard.stream.errorCount)),
        metric("Runtime health", runtimeConsole.health.label),
        metric("Retries", String(telemetryDashboard.runtime.retryCount))
      ],
      notes: ["This surface reports current product signals; it does not mask a limitation as success."]
    }),
    section({
      category: "LangSmith",
      tone: telemetryDashboard.observability.enabled ? "ready" : "empty",
      summary: telemetryDashboard.observability.status ?? "Local trace state",
      detail:
        telemetryDashboard.observability.reason ??
        "Trace export remains bounded to the configured observability path.",
      metrics: [
        metric("Enabled", telemetryDashboard.observability.enabled ? "Yes" : "No"),
        metric("Requested", telemetryDashboard.observability.requested ? "Yes" : "No"),
        metric("Trace", telemetryDashboard.observability.traceId ?? "Not published")
      ],
      notes: telemetryDashboard.observability.tags.length
        ? [`Tags: ${telemetryDashboard.observability.tags.join(", ")}`]
        : ["No trace tags are published for this run."]
    }),
    section({
      category: "Settings",
      tone: "ready",
      summary: "Workspace preferences",
      detail: "Theme, display mode, density, and preview behavior are available from the top-right settings control.",
      metrics: [
        metric("Theme", "Session preference"),
        metric("Display", "User or Developer"),
        metric("Density", "Cozy or Compact")
      ],
      notes: ["Settings affect presentation only; they do not alter workflow truthfulness."]
    })
  ];

  return {
    activeArtifactId: activeArtifact?.id ?? null,
    activeDomainId,
    domainExperience,
    artifactRegistry: snapshot.artifacts,
    session: {
      id: snapshot.session.sessionId,
      title: snapshot.session.title || snapshot.workspace.name
    },
    sections,
    summary: {
      activeCount: sections.filter((item) => item.tone === "active").length,
      attentionCount: sections.filter((item) => item.tone === "attention").length,
      readyCount: sections.filter((item) => item.tone === "ready").length
    },
    details: {
      conversationContext,
      providerTelemetry,
      retrievalRuntime,
      runtimeConsole,
      sessionIntelligence,
      snapshot,
      telemetryDashboard,
      v3InspectorPanels,
      workflowExecution,
      workflowRuntime,
      workstationDashboard
    }
  };
}

export function getProductIntelligenceSection(
  model: ProductIntelligenceModel,
  category: ProductIntelligenceCategory
) {
  return model.sections.find((section) => section.category === category) ?? model.sections[0]!;
}

function section(value: ProductIntelligenceSection): ProductIntelligenceSection {
  return { ...value, notes: value.notes.filter(Boolean) };
}

function metric(label: string, value: string): ProductIntelligenceMetric {
  return { label, value };
}

function formatTokenCount(value: number | null) {
  return value == null ? "Not reported" : new Intl.NumberFormat().format(value);
}

function toneForWorkflow(
  state: WorkflowRuntimeModel["summary"]["activity"]["state"]
): ProductIntelligenceTone {
  if (state === "failed" || state === "partial") return "attention";
  if (state === "planning" || state === "retrieving" || state === "generating" || state === "reviewing" || state === "refining" || state === "finalizing") return "active";
  return "ready";
}

function toneForPreview(
  state: AssistantWorkspaceSnapshot["preview"]["state"],
  available: boolean
): ProductIntelligenceTone {
  if (state === "error") return "attention";
  if (state === "generating") return "active";
  return available ? "ready" : "empty";
}

function toneForRetrieval(
  state: RetrievalRuntimeModel["summary"]["state"]
): ProductIntelligenceTone {
  if (state === "error") return "attention";
  if (state === "pending") return "active";
  return state === "available" ? "ready" : "empty";
}

function toneForKnowledgeBase(
  status: DomainExperienceCatalog["knowledgeBase"]["status"]
): ProductIntelligenceTone {
  if (status === "unavailable") return "attention";
  return status === "available" ? "ready" : "empty";
}

function formatDomainDelivery(
  deliveryKind: DomainExperienceCatalog["domains"][number]["deliveryKind"]
) {
  if (deliveryKind === "browser_preview") return "Live browser preview";
  if (deliveryKind === "external_handoff") return "External handoff";
  return "Code/export";
}

function toneForProvider(
  state: ProviderTelemetryModel["status"]
): ProductIntelligenceTone {
  if (state === "error") return "attention";
  if (state === "streaming") return "active";
  return state === "complete" ? "ready" : "empty";
}

function toneForRuntime(
  state: RuntimeConsoleModel["health"]["signal"]
): ProductIntelligenceTone {
  if (state === "failed") return "attention";
  return state === "healthy" ? "ready" : "empty";
}

function toneForSession(
  state: SessionIntelligenceModel["metadata"]["completion_status"]
): ProductIntelligenceTone {
  if (state === "needs_attention") return "attention";
  if (state === "running") return "active";
  return state === "completed" ? "ready" : "empty";
}

function toneForTelemetry(
  state: TelemetryDashboardModel["status"]
): ProductIntelligenceTone {
  if (state === "error" || state === "degraded") return "attention";
  if (state === "running") return "active";
  return state === "complete" ? "ready" : "empty";
}

function formatExecutionMode(mode: WorkflowExecutionModel["resolvedMode"] | WorkflowExecutionModel["requestedMode"]) {
  if (mode === "single_agent") return "Single Agent";
  if (mode === "multi_agent") return "Multi Agent";
  return "Auto";
}

function previewLabel(
  state: AssistantWorkspaceSnapshot["preview"]["state"],
  available: boolean
) {
  if (state === "error") return "Failure";
  if (state === "generating") return "Generating";
  return available ? "Success" : "Unavailable";
}
