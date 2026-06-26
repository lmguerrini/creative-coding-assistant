import { describe, expect, it } from "vitest";
import {
  getInitialWorkspaceSnapshot,
  getLocalWorkspaceSnapshot,
  type WorkflowNodeId
} from "./assistant-client";
import {
  buildWorkstationState,
  type WorkstationReadinessState
} from "./workstation-state";
import { buildEvaluationSessionModel } from "./evaluation-session";
import type { AssistantStreamEvent, AssistantStreamEventType } from "./assistant-stream";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

describe("workstation state", () => {
  it("models first-run readiness with explicit missing metadata", () => {
    const snapshot = getInitialWorkspaceSnapshot();
    const state = buildWorkstationState({ snapshot });

    expect(state.readiness.state).toBe("empty");
    expect(state.status).toMatchObject({
      label: "Idle",
      detail: "Ready to start",
      artifactLabel: "No artifact selected",
      evaluationLabel: "No evaluation selected"
    });
    expect(state.metadata.selected_artifact.status).toBe("missing");
    expect(state.metadata.preview.status).toBe("missing");
    expect(state.metadata.selected_evaluation.status).toBe("missing");
    expect(state.readiness.missingMetadata.map((metadata) => metadata.key)).toEqual(
      expect.arrayContaining([
        "assistant_run",
        "selected_artifact",
        "selected_evaluation"
      ])
    );
  });

  it("resolves selected artifact, workflow, session, and panel state", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const activeArtifact = snapshot.artifacts[0];
    const state = buildWorkstationState({
      activeArtifactId: activeArtifact.id,
      activeInspectorTab: "Workflow",
      activeWorkflowNodeId: "generation",
      inspectorCollapsed: true,
      previewArtifactId: activeArtifact.id,
      previewOpen: true,
      snapshot
    });

    expect(state.readiness.state).toBe("ready" satisfies WorkstationReadinessState);
    expect(state.selection.activeArtifact?.id).toBe(activeArtifact.id);
    expect(state.selection.activeWorkflowStep?.nodeId).toBe("generation");
    expect(state.selection.activeInspectorTab).toBe("Workflow");
    expect(state.panels).toMatchObject({
      activeTab: "Workflow",
      inspectorCollapsed: true,
      previewOpen: true
    });
    expect(state.session).toMatchObject({
      userId: "local-user",
      sessionId: "local-nextjs-session",
      projectId: "local-nextjs-workspace"
    });
    expect(state.metadata.selected_artifact.status).toBe("available");
    expect(state.metadata.selected_workflow.status).toBe("available");
  });

  it("summarizes the active assistant run from stream trace events", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const traceEvents = [
      traceEvent({
        at: "2026-06-26T09:00:00.000Z",
        eventType: "node_started",
        nodeId: "generation",
        sequence: 4,
        workflowStatus: "running"
      })
    ];
    const state = buildWorkstationState({
      activeWorkflowNodeId: "generation",
      isStreaming: true,
      snapshot,
      traceEvents
    });

    expect(state.currentRun).toMatchObject({
      state: "streaming",
      streamEventCount: 1,
      traceEventCount: 1,
      latestEventSequence: 4,
      latestEventType: "node_started"
    });
    expect(state.readiness.state).toBe("active");
    expect(state.status.label).toBe("Streaming");
    expect(state.metadata.assistant_run.status).toBe("pending");
  });

  it("keeps selected evaluation metadata inspectable when eval events exist", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const traceEvents = [
      traceEvent({
        at: "2026-06-26T09:05:00.000Z",
        code: "ragas_eval_completed",
        eventType: "eval_update",
        payload: {
          evaluation: {
            dataset_id: "creative-live",
            evaluation_type: "RAGAs live",
            metric_scores: {
              answer_relevancy: 0.82
            },
            metrics: ["answer_relevancy"],
            run_id: "eval-run-1",
            status: "Evaluation complete"
          }
        },
        sequence: 9,
        workflowStatus: "completed"
      })
    ];
    const evaluation = buildEvaluationSessionModel(traceEvents, {
      latestAt: null,
      traceKind: null
    });
    const state = buildWorkstationState({
      selectedEvaluation: evaluation,
      snapshot,
      traceEvents
    });

    expect(state.selection.selectedEvaluation?.runId).toBe("eval-run-1");
    expect(state.metadata.selected_evaluation).toMatchObject({
      status: "available",
      detail: "Evaluation complete: eval-run-1"
    });
    expect(state.status.evaluationLabel).toBe("Evaluation complete");
  });
});

function traceEvent({
  at,
  code,
  eventType,
  nodeId = "generation",
  payload = {},
  sequence,
  workflowStatus
}: {
  at: string;
  code?: string;
  eventType: AssistantStreamEventType;
  nodeId?: WorkflowNodeId;
  payload?: Record<string, unknown>;
  sequence: number;
  workflowStatus: "running" | "completed";
}): WorkflowRuntimeTraceEvent {
  const event: AssistantStreamEvent = {
    event_type: eventType,
    sequence,
    payload: {
      ...payload,
      ...(code ? { code } : {}),
      emitted_at: at,
      workflow: {
        current_step: nodeId,
        phase: workflowStatus === "completed" ? "completed" : "running",
        status: workflowStatus,
        step: nodeId
      }
    }
  };

  return {
    event,
    receivedAt: at,
    receivedAtMs: Date.parse(at)
  };
}
