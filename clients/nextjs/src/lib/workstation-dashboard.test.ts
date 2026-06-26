import { describe, expect, it } from "vitest";
import {
  getInitialWorkspaceSnapshot,
  getLocalWorkspaceSnapshot,
  type AssistantWorkspaceSnapshot,
  type WorkflowNodeId
} from "./assistant-client";
import type { AssistantStreamEvent, AssistantStreamEventType } from "./assistant-stream";
import { buildProvenanceEngineModel } from "./provenance-engine";
import { buildV3InspectorPanelsModel } from "./v3-inspector-panels";
import {
  buildWorkstationDashboardModel,
  type WorkstationDashboardModel
} from "./workstation-dashboard";
import {
  buildWorkflowRuntimeModel,
  type WorkflowRuntimeTraceEvent
} from "./workflow-runtime";
import { buildWorkstationState } from "./workstation-state";

describe("workstation dashboard model", () => {
  it("shows missing quality metadata while preserving workflow health", () => {
    const dashboard = buildDashboard(getInitialWorkspaceSnapshot());

    expect(dashboard.cards).toHaveLength(8);
    expect(card(dashboard, "creative_quality")).toMatchObject({
      tone: "missing",
      value: "Missing"
    });
    expect(card(dashboard, "workflow_health")).toMatchObject({
      tone: "good",
      value: "Idle"
    });
    expect(card(dashboard, "hitl_recommendation")).toMatchObject({
      tone: "missing",
      value: "Unknown"
    });
  });

  it("uses workspace artifact and preview metadata for readiness fallbacks", () => {
    const dashboard = buildDashboard(getLocalWorkspaceSnapshot());

    expect(card(dashboard, "artifact_readiness")).toMatchObject({
      tone: "good",
      value: "Ready"
    });
    expect(card(dashboard, "runtime_fit")).toMatchObject({
      tone: "good",
      source: "Preview metadata"
    });
  });

  it("surfaces partial V3 confidence, evaluation, and HITL signals", () => {
    const dashboard = buildDashboard(getInitialWorkspaceSnapshot(), [
      traceEvent({
        eventType: "final",
        nodeId: "finalization",
        sequence: 1,
        workflow: {
          creative_confidence: {
            confidenceSummary: "Confidence is medium pending human review."
          },
          creative_score: {
            scoreSummary: "Quality score is directionally strong."
          },
          evaluation_report: {
            executiveSummary: "Evaluation report is available.",
            recommendations: ["Human review recommended before final use."]
          }
        }
      })
    ]);

    expect(card(dashboard, "creative_quality")).toMatchObject({
      tone: "watch",
      summary: "Quality score is directionally strong."
    });
    expect(card(dashboard, "confidence")).toMatchObject({
      tone: "watch",
      summary: "Confidence is medium pending human review."
    });
    expect(card(dashboard, "hitl_recommendation")).toMatchObject({
      tone: "watch",
      value: "Review"
    });
  });
});

function buildDashboard(
  snapshot: AssistantWorkspaceSnapshot,
  traceEvents: WorkflowRuntimeTraceEvent[] = []
): WorkstationDashboardModel {
  const runtime = buildWorkflowRuntimeModel(snapshot.workflow, traceEvents);
  const workstationState = buildWorkstationState({
    activeWorkflowNodeId: traceEvents.length > 0 ? "finalization" : undefined,
    snapshot,
    traceEvents
  });
  const provenance = buildProvenanceEngineModel({
    snapshot,
    traceEvents,
    workstationState
  });
  const v3InspectorPanels = buildV3InspectorPanelsModel({
    provenance,
    traceEvents,
    workstationState
  });

  return buildWorkstationDashboardModel({
    runtime,
    snapshot,
    v3InspectorPanels,
    workstationState
  });
}

function card(dashboard: WorkstationDashboardModel, id: string) {
  const match = dashboard.cards.find((candidate) => candidate.id === id);
  expect(match).toBeDefined();
  return match!;
}

function traceEvent({
  eventType,
  nodeId,
  sequence,
  workflow
}: {
  eventType: AssistantStreamEventType;
  nodeId: WorkflowNodeId;
  sequence: number;
  workflow: Record<string, unknown>;
}): WorkflowRuntimeTraceEvent {
  const event: AssistantStreamEvent = {
    event_type: eventType,
    sequence,
    payload: {
      emitted_at: `2026-06-26T09:40:0${sequence}.000Z`,
      workflow: {
        ...workflow,
        current_step: nodeId,
        phase: eventType === "final" ? "completed" : "running",
        status: eventType === "final" ? "completed" : "running",
        step: nodeId
      }
    }
  };

  return {
    event,
    receivedAt: `2026-06-26T09:40:0${sequence}.000Z`,
    receivedAtMs: Date.parse(`2026-06-26T09:40:0${sequence}.000Z`)
  };
}
