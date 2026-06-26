import { describe, expect, it } from "vitest";
import {
  getInitialWorkspaceSnapshot,
  getLocalWorkspaceSnapshot,
  type WorkflowNodeId
} from "./assistant-client";
import type { AssistantStreamEvent, AssistantStreamEventType } from "./assistant-stream";
import {
  buildWorkflowExplorerModel,
  type WorkflowExplorerStageId
} from "./workflow-explorer";
import {
  buildWorkflowRuntimeModel,
  type WorkflowRuntimeTraceEvent
} from "./workflow-runtime";
import { buildWorkstationState } from "./workstation-state";

describe("workflow explorer model", () => {
  it("handles missing metadata for a first-run workspace", () => {
    const snapshot = getInitialWorkspaceSnapshot();
    const runtime = buildWorkflowRuntimeModel(snapshot.workflow, []);
    const workstationState = buildWorkstationState({ snapshot });
    const explorer = buildWorkflowExplorerModel({
      runtime,
      snapshot,
      traceEvents: [],
      workstationState
    });

    expect(explorer.state).toBe("empty");
    expect(explorer.summary.availableMetadataGroupCount).toBe(0);
    expect(stage(explorer.stages, "planning")).toMatchObject({
      status: "missing",
      summary: "No metadata groups are available for this stage yet."
    });
  });

  it("uses workspace snapshot metadata for retrieval, artifacts, and final response", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const runtime = buildWorkflowRuntimeModel(snapshot.workflow, []);
    const workstationState = buildWorkstationState({ snapshot });
    const explorer = buildWorkflowExplorerModel({
      runtime,
      snapshot,
      traceEvents: [],
      workstationState
    });

    expect(explorer.state).toBe("available");
    expect(stage(explorer.stages, "retrieval").metadataGroups[0]).toMatchObject({
      label: "Retrieval context",
      status: "available",
      summary: "2 sources available."
    });
    expect(stage(explorer.stages, "artifact_intelligence").metadataGroups[0]).toMatchObject({
      label: "Artifact plan",
      status: "available",
      summary: "3 workspace artifacts."
    });
    expect(stage(explorer.stages, "final_response").metadataGroups[0].status).toBe(
      "available"
    );
  });

  it("groups stream payload metadata into creative and artifact stages", () => {
    const snapshot = getInitialWorkspaceSnapshot();
    const traceEvents = [
      traceEvent({
        code: "creative_plan_prepared",
        eventType: "planning",
        nodeId: "planning",
        payload: {
          creative_intent: { role: "creative_intent_decomposer" },
          creative_strategy: {
            role: "creative_strategy_engine",
            rationale: "Use a restrained recursive field."
          },
          procedural_structure: {
            role: "procedural_structure_planner",
            summary: "Layered particle structure."
          }
        },
        sequence: 1
      }),
      traceEvent({
        code: "critique_completed",
        eventType: "artifact_critique",
        nodeId: "artifact_critique",
        payload: {
          artifact_plan: { role: "artifact_planner" },
          artifact_critic: {
            role: "artifact_critic",
            rationale: "Candidate is previewable."
          },
          self_evaluation: {
            role: "self_evaluation",
            evaluation_summary: "Quality metadata is complete."
          }
        },
        sequence: 2
      }),
      traceEvent({
        eventType: "final",
        nodeId: "finalization",
        payload: {
          answer: "Final response",
          session_intelligence: {
            session_summary: "Completed session."
          }
        },
        sequence: 3
      })
    ];
    const runtime = buildWorkflowRuntimeModel(snapshot.workflow, traceEvents);
    const workstationState = buildWorkstationState({
      activeWorkflowNodeId: "finalization",
      snapshot,
      traceEvents
    });
    const explorer = buildWorkflowExplorerModel({
      runtime,
      snapshot,
      traceEvents,
      workstationState
    });

    expect(stage(explorer.stages, "creative_intelligence").status).toBe("partial");
    expect(stage(explorer.stages, "generative_design").metadataGroups[0]).toMatchObject({
      label: "Structure engines",
      status: "available",
      summary: "Layered particle structure."
    });
    expect(stage(explorer.stages, "artifact_intelligence").metadataGroups[2]).toMatchObject({
      label: "Critique and refinement",
      status: "available",
      summary: "Candidate is previewable."
    });
    expect(stage(explorer.stages, "creative_evaluation").metadataGroups[0]).toMatchObject({
      label: "Self evaluation",
      status: "available",
      summary: "Quality metadata is complete."
    });
    expect(stage(explorer.stages, "final_response").metadataGroups[1]).toMatchObject({
      label: "Session intelligence",
      status: "available"
    });
  });
});

function stage(
  stages: ReturnType<typeof buildWorkflowExplorerModel>["stages"],
  id: WorkflowExplorerStageId
) {
  const match = stages.find((candidate) => candidate.id === id);
  expect(match).toBeDefined();
  return match!;
}

function traceEvent({
  code,
  eventType,
  nodeId,
  payload,
  sequence
}: {
  code?: string;
  eventType: AssistantStreamEventType;
  nodeId: WorkflowNodeId;
  payload: Record<string, unknown>;
  sequence: number;
}): WorkflowRuntimeTraceEvent {
  const event: AssistantStreamEvent = {
    event_type: eventType,
    sequence,
    payload: {
      ...payload,
      ...(code ? { code } : {}),
      emitted_at: `2026-06-26T09:00:0${sequence}.000Z`,
      workflow: {
        current_step: nodeId,
        phase: eventType === "final" ? "completed" : "running",
        status: eventType === "final" ? "completed" : "running",
        step: nodeId
      }
    }
  };

  return {
    event,
    receivedAt: `2026-06-26T09:00:0${sequence}.000Z`,
    receivedAtMs: Date.parse(`2026-06-26T09:00:0${sequence}.000Z`)
  };
}
