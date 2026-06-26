import { describe, expect, it } from "vitest";
import {
  getInitialWorkspaceSnapshot,
  getLocalWorkspaceSnapshot,
  type AssistantWorkspaceSnapshot,
  type WorkflowNodeId
} from "./assistant-client";
import type { AssistantStreamEvent, AssistantStreamEventType } from "./assistant-stream";
import { buildProvenanceEngineModel } from "./provenance-engine";
import {
  buildV3InspectorPanelsModel,
  type V3InspectorPanel,
  type V3InspectorPanelsModel
} from "./v3-inspector-panels";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";
import { buildWorkstationState } from "./workstation-state";

describe("V3 inspector panels model", () => {
  it("renders all bounded panels as missing for a first-run workspace", () => {
    const model = buildModel(getInitialWorkspaceSnapshot());

    expect(model.state).toBe("empty");
    expect(model.panels.map((panel) => panel.label)).toEqual([
      "Creative Intelligence",
      "Generative Design",
      "Artifact Intelligence",
      "Creative Evaluation",
      "Confidence",
      "Reflection",
      "Improvement Plan",
      "Evaluation Trace",
      "Provenance"
    ]);
    expect(panel(model, "creative_intelligence")).toMatchObject({
      missingItemCount: 8,
      status: "missing"
    });
    expect(item(panel(model, "creative_intelligence"), "creative_intent")).toMatchObject(
      {
        source: "missing",
        status: "missing",
        summary: "Intent metadata has not been captured yet."
      }
    );
    expect(panel(model, "provenance")).toMatchObject({
      status: "missing"
    });
  });

  it("keeps partial V3 payloads inspectable without inventing hydrated metadata", () => {
    const traceEvents = [
      traceEvent({
        eventType: "final",
        nodeId: "finalization",
        sequence: 1,
        workflow: {
          artifact_plan: {
            primaryArtifactIntent: "Deliver one runnable p5 sketch."
          },
          creative_confidence: {
            confidenceScore: 0.72,
            confidenceSummary: "Confidence is constrained by partial evaluation."
          },
          creative_improvement_planner: {
            improvementSummary: "Increase motion contrast before a later pass."
          },
          creative_intent: {
            normalizedIntent: "Kinetic ritual field.",
            primaryExpression: "Layered light and orbiting motion."
          },
          evaluation_report: {
            evaluation_trace: [
              {
                contribution: "Checked coherence against the prompt.",
                source: "creative_critic"
              }
            ],
            executiveSummary: "Evaluation trace captured."
          },
          procedural_structure: {
            spatialStructurePlan: "Radial lattice.",
            temporalStructurePlan: "Slow orbiting pulse."
          },
          reflection_loop: {
            reflectionSummary: "Reflection may improve pacing."
          }
        }
      })
    ];
    const model = buildModel(getInitialWorkspaceSnapshot(), traceEvents);

    expect(model.state).toBe("available");
    expect(item(panel(model, "creative_intelligence"), "creative_intent")).toMatchObject(
      {
        source: "partial",
        status: "partial",
        summary: "Kinetic ritual field."
      }
    );
    expect(item(panel(model, "generative_design"), "procedural_structure")).toMatchObject(
      {
        status: "partial",
        summary: "Radial lattice."
      }
    );
    expect(item(panel(model, "confidence"), "creative_confidence")).toMatchObject({
      status: "partial",
      summary: "Confidence is constrained by partial evaluation."
    });
    expect(item(panel(model, "evaluation_trace"), "evaluation_trace").details).toEqual(
      expect.arrayContaining([
        "Evaluation Trace: Checked coherence against the prompt."
      ])
    );
  });

  it("summarizes existing provenance alongside missing V3 metadata", () => {
    const model = buildModel(getLocalWorkspaceSnapshot());
    const provenance = panel(model, "provenance");

    expect(model.state).toBe("available");
    expect(provenance).toMatchObject({
      availableItemCount: 2,
      status: "partial"
    });
    expect(item(provenance, "evidence")).toMatchObject({
      status: "available",
      summary: "2 evidence sources"
    });
    expect(item(provenance, "artifact")).toMatchObject({
      status: "available",
      summary: "3 artifact sources"
    });
  });
});

function buildModel(
  snapshot: AssistantWorkspaceSnapshot,
  traceEvents: WorkflowRuntimeTraceEvent[] = []
): V3InspectorPanelsModel {
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

  return buildV3InspectorPanelsModel({
    provenance,
    traceEvents,
    workstationState
  });
}

function panel(model: V3InspectorPanelsModel, id: string): V3InspectorPanel {
  const match = model.panels.find((candidate) => candidate.id === id);
  expect(match).toBeDefined();
  return match!;
}

function item(panel: V3InspectorPanel, id: string) {
  const match = panel.items.find((candidate) => candidate.id === id);
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
      emitted_at: `2026-06-26T09:30:0${sequence}.000Z`,
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
    receivedAt: `2026-06-26T09:30:0${sequence}.000Z`,
    receivedAtMs: Date.parse(`2026-06-26T09:30:0${sequence}.000Z`)
  };
}
