import { describe, expect, it } from "vitest";
import { getLocalWorkspaceSnapshot } from "./assistant-client";
import {
  buildWorkflowRuntimeModel,
  type WorkflowRuntimeTraceEvent
} from "./workflow-runtime";

describe("workflow runtime model", () => {
  it("falls back to the workspace workflow state when no trace events exist", () => {
    const workflow = getLocalWorkspaceSnapshot().workflow;
    const runtime = buildWorkflowRuntimeModel(workflow, []);

    expect(runtime.summary.status).toBe("running");
    expect(runtime.summary.currentNode).toBe("generation");
    expect(runtime.summary.traceEventCount).toBe(0);
    expect(runtime.steps.find((step) => step.nodeId === "generation")).toMatchObject({
      state: "active",
      attemptCount: 1
    });
  });

  it("derives retries, timing, and completed review steps from workflow traces", () => {
    const workflow = getLocalWorkspaceSnapshot().workflow;
    const traceEvents: WorkflowRuntimeTraceEvent[] = [
      traceEvent({
        at: "2026-05-22T10:00:00Z",
        code: "request_received",
        currentStep: "intake",
        event_type: "status",
        sequence: 0,
        step: "intake"
      }),
      traceEvent({
        at: "2026-05-22T10:00:01Z",
        code: "route_selected",
        completedSteps: ["intake"],
        currentStep: "routing",
        event_type: "status",
        sequence: 1,
        step: "routing"
      }),
      traceEvent({
        at: "2026-05-22T10:00:02Z",
        code: "generation_input_prepared",
        completedSteps: ["intake", "routing"],
        currentStep: "generation",
        event_type: "generation_input",
        sequence: 2,
        skippedSteps: [
          "memory",
          "retrieval",
          "context_assembly",
          "prompt_input",
          "prompt_rendering"
        ],
        step: "generation"
      }),
      traceEvent({
        at: "2026-05-22T10:00:03Z",
        currentStep: "generation",
        event_type: "token_delta",
        sequence: 3,
        step: "generation",
        text: "draft"
      }),
      traceEvent({
        at: "2026-05-22T10:00:04Z",
        code: "generation_input_prepared",
        completedSteps: ["intake", "routing"],
        currentStep: "generation",
        event_type: "generation_input",
        refinementCount: 1,
        sequence: 4,
        skippedSteps: [
          "memory",
          "retrieval",
          "context_assembly",
          "prompt_input",
          "prompt_rendering"
        ],
        step: "generation"
      }),
      traceEvent({
        at: "2026-05-22T10:00:05Z",
        answer: "```ts\nconsole.log('refined');\n```",
        completedSteps: [
          "intake",
          "routing",
          "generation",
          "review",
          "refinement",
          "finalization"
        ],
        currentStep: null,
        event_type: "final",
        phase: "completed",
        refinementCount: 1,
        reviewOutcome: "pass",
        sequence: 5,
        skippedSteps: [
          "memory",
          "retrieval",
          "context_assembly",
          "prompt_input",
          "prompt_rendering"
        ],
        status: "completed",
        step: "finalization"
      })
    ];

    const runtime = buildWorkflowRuntimeModel(workflow, traceEvents);
    const generationStep = runtime.steps.find((step) => step.nodeId === "generation");
    const refinementStep = runtime.steps.find((step) => step.nodeId === "refinement");
    const finalizationStep = runtime.steps.find(
      (step) => step.nodeId === "finalization"
    );

    expect(runtime.summary.status).toBe("completed");
    expect(runtime.summary.retryCount).toBe(1);
    expect(runtime.summary.transitionCount).toBe(4);
    expect(runtime.summary.totalRuntimeMs).toBe(5000);
    expect(runtime.summary.currentNode).toBe("finalization");
    expect(generationStep).toMatchObject({
      attemptCount: 2,
      state: "complete"
    });
    expect(refinementStep).toMatchObject({
      attemptCount: 1,
      state: "complete"
    });
    expect(finalizationStep).toMatchObject({
      state: "complete"
    });
    expect(runtime.transitions.map((transition) => transition.label)).toEqual([
      "Intake -> Routing",
      "Routing -> Generation",
      "Generation retry",
      "Generation -> Finalization"
    ]);
  });
});

function traceEvent({
  answer,
  at,
  code,
  completedSteps = [],
  currentStep,
  event_type,
  phase = "running",
  refinementCount = 0,
  reviewOutcome = null,
  sequence,
  skippedSteps = [],
  status = "running",
  step,
  text
}: {
  answer?: string;
  at: string;
  code?: string;
  completedSteps?: string[];
  currentStep: string | null;
  event_type: WorkflowRuntimeTraceEvent["event"]["event_type"];
  phase?: string;
  refinementCount?: number;
  reviewOutcome?: string | null;
  sequence: number;
  skippedSteps?: string[];
  status?: string;
  step: string | null;
  text?: string;
}): WorkflowRuntimeTraceEvent {
  return {
    event: {
      event_type,
      sequence,
      payload: {
        ...(answer ? { answer } : {}),
        ...(code ? { code } : {}),
        ...(text ? { text } : {}),
        emitted_at: at,
        workflow: {
          step,
          phase,
          status,
          current_step: currentStep,
          completed_steps: completedSteps,
          skipped_steps: skippedSteps,
          refinement_count: refinementCount,
          review_outcome: reviewOutcome,
          review_reasons: []
        }
      }
    },
    receivedAt: at,
    receivedAtMs: Date.parse(at)
  };
}
