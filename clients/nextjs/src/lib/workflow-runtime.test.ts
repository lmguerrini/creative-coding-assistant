import { describe, expect, it } from "vitest";
import { getLocalWorkspaceSnapshot } from "./assistant-client";
import {
  buildWorkflowRuntimeModel,
  deriveWorkflowRuntimeActivity,
  type WorkflowRuntimeTraceEvent
} from "./workflow-runtime";

describe("workflow runtime model", () => {
  it("falls back to the workspace workflow state when no trace events exist", () => {
    const workflow = getLocalWorkspaceSnapshot().workflow;
    const runtime = buildWorkflowRuntimeModel(workflow, []);

    expect(runtime.summary.status).toBe("running");
    expect(runtime.summary.currentNode).toBe("generation");
    expect(runtime.summary.activity).toMatchObject({
      state: "generating",
      label: "Generating"
    });
    expect(runtime.summary.traceEventCount).toBe(0);
    expect(runtime.steps.find((step) => step.nodeId === "generation")).toMatchObject({
      state: "active",
      attemptCount: 1
    });
  });

  it("maps every user-facing live stage from the workflow node contract", () => {
    const stages = [
      ["planning", "Planning"],
      ["retrieval", "Retrieving"],
      ["generation", "Generating"],
      ["review", "Reviewing"],
      ["refinement", "Refining"],
      ["finalization", "Finalizing"]
    ] as const;

    expect(
      stages.map(([currentNode]) =>
        deriveWorkflowRuntimeActivity({
          currentNode,
          productOutcome: null,
          workflowStatus: "running"
        }).label
      )
    ).toEqual(stages.map(([, label]) => label));
    expect(
      deriveWorkflowRuntimeActivity({
        currentNode: "failure",
        productOutcome: null,
        workflowStatus: "failed"
      })
    ).toMatchObject({ state: "failed", label: "Failed", terminal: true });
    expect(
      deriveWorkflowRuntimeActivity({
        currentNode: "finalization",
        productOutcome: {
          orchestration_status: "COMPLETED",
          provider_status: "COMPLETED",
          generation_status: "COMPLETED",
          deliverable_status: "USABLE",
          artifact_extraction_status: "EXTRACTED",
          artifact_runnability: "RUNNABLE",
          preview_status: "READY",
          runtime_health: "PENDING_BROWSER_VALIDATION",
          product_outcome: "IN_PROGRESS",
          summary: "Product validation is still in progress.",
          recovery_action: ""
        },
        workflowStatus: "completed"
      })
    ).toMatchObject({ state: "finalizing", label: "Finalizing", terminal: false });
  });

  it("downgrades product success when the preview runtime fails", () => {
    const workflow = getLocalWorkspaceSnapshot().workflow;
    const runtime = buildWorkflowRuntimeModel(workflow, [
      {
        event: {
          event_type: "status",
          sequence: 1,
          payload: {
            code: "preview_runtime_error",
            message: "colorMode is not defined",
            preview_runtime: { error: "colorMode is not defined" },
            workflow: {
              current_step: "finalization",
              phase: "completed",
              status: "completed"
            }
          }
        },
        receivedAt: "2026-07-10T10:00:00Z",
        receivedAtMs: Date.parse("2026-07-10T10:00:00Z")
      }
    ]);

    expect(runtime.summary).toMatchObject({
      currentStep: "A usable artifact was produced, but the live preview failed.",
      status: "partial",
      activity: { state: "partial", label: "Partial", terminal: true },
      productOutcome: {
        preview_status: "FAILED",
        runtime_health: "FAILED",
        product_outcome: "PARTIAL"
      }
    });
    expect(runtime.error?.userMessage).toContain("colorMode is not defined");
  });

  it("preserves a backend partial outcome when an artifact cannot open in preview", () => {
    const workflow = getLocalWorkspaceSnapshot().workflow;
    const runtime = buildWorkflowRuntimeModel(workflow, [
      {
        event: {
          event_type: "final",
          sequence: 1,
          payload: {
            workflow: {
              current_step: "finalization",
              phase: "completed",
              status: "completed",
              product_outcome: {
                orchestration_status: "COMPLETED",
                provider_status: "COMPLETED",
                generation_status: "COMPLETED",
                deliverable_status: "USABLE",
                artifact_extraction_status: "EXTRACTED",
                artifact_runnability: "UNSUPPORTED",
                preview_status: "UNAVAILABLE",
                runtime_health: "NOT_AVAILABLE",
                product_outcome: "PARTIAL",
                summary: "A usable artifact was produced, but live preview is unavailable.",
                recovery_action: "Open Code to use the artifact, then regenerate the preview."
              }
            }
          }
        },
        receivedAt: "2026-07-10T10:00:00Z",
        receivedAtMs: Date.parse("2026-07-10T10:00:00Z")
      }
    ]);

    expect(runtime.summary).toMatchObject({
      currentStep: "A usable artifact was produced, but live preview is unavailable.",
      status: "partial",
      activity: { state: "partial", label: "Partial", terminal: true },
      productOutcome: {
        deliverable_status: "USABLE",
        artifact_runnability: "UNSUPPORTED",
        product_outcome: "PARTIAL"
      }
    });
    expect(runtime.error?.suggestedAction).toContain("Open Code");
  });

  it("preserves explicit planning and director transition metadata", () => {
    const workflow = getLocalWorkspaceSnapshot().workflow;
    const traceEvents: WorkflowRuntimeTraceEvent[] = [
      traceEvent({
        at: "2026-05-22T09:59:00Z",
        currentStep: "planning",
        event_type: "node_started",
        sequence: 0,
        step: "planning"
      }),
      traceEvent({
        at: "2026-05-22T09:59:01Z",
        code: "creative_plan_prepared",
        currentStep: "planning",
        event_type: "planning",
        sequence: 1,
        step: "planning"
      }),
      traceEvent({
        at: "2026-05-22T09:59:02Z",
        completedSteps: ["planning"],
        currentStep: null,
        decisionReason: "creative_plan_prepared",
        event_type: "node_completed",
        phase: "completed",
        sequence: 2,
        step: "planning",
        transitionSource: "planning",
        transitionTarget: "director"
      }),
      traceEvent({
        at: "2026-05-22T09:59:03Z",
        completedSteps: ["planning"],
        currentStep: "director",
        event_type: "node_started",
        sequence: 3,
        step: "director"
      }),
      traceEvent({
        at: "2026-05-22T09:59:04Z",
        code: "creative_director_prepared",
        completedSteps: ["planning"],
        currentStep: "director",
        event_type: "planning",
        sequence: 4,
        step: "director"
      }),
      traceEvent({
        at: "2026-05-22T09:59:05Z",
        completedSteps: ["planning", "director"],
        currentStep: null,
        decisionReason: "creative_director_prepared",
        event_type: "node_completed",
        phase: "completed",
        sequence: 5,
        step: "director",
        transitionSource: "director",
        transitionTarget: "reasoning"
      }),
      traceEvent({
        at: "2026-05-22T09:59:06Z",
        completedSteps: ["planning", "director"],
        currentStep: "reasoning",
        event_type: "node_started",
        sequence: 6,
        step: "reasoning"
      }),
      traceEvent({
        at: "2026-05-22T09:59:07Z",
        code: "creative_reasoning_prepared",
        completedSteps: ["planning", "director"],
        currentStep: "reasoning",
        event_type: "planning",
        sequence: 7,
        step: "reasoning"
      }),
      traceEvent({
        at: "2026-05-22T09:59:08Z",
        completedSteps: ["planning", "director", "reasoning"],
        currentStep: null,
        decisionReason: "creative_reasoning_prepared",
        event_type: "node_completed",
        phase: "completed",
        sequence: 8,
        step: "reasoning",
        transitionSource: "reasoning",
        transitionTarget: "prompt_rendering"
      }),
      traceEvent({
        at: "2026-05-22T09:59:09Z",
        completedSteps: ["planning", "director", "reasoning"],
        currentStep: "prompt_rendering",
        event_type: "node_started",
        sequence: 9,
        step: "prompt_rendering"
      })
    ];

    const runtime = buildWorkflowRuntimeModel(workflow, traceEvents);
    const planningTransition = runtime.transitions.find(
      (transition) =>
        transition.fromNodeId === "planning" &&
        transition.toNodeId === "director"
    );
    const directorTransition = runtime.transitions.find(
      (transition) =>
        transition.fromNodeId === "director" &&
        transition.toNodeId === "reasoning"
    );
    const reasoningTransition = runtime.transitions.find(
      (transition) =>
        transition.fromNodeId === "reasoning" &&
        transition.toNodeId === "prompt_rendering"
    );

    expect(planningTransition).toMatchObject({
      kind: "advance",
      label: "Planning -> Director",
      reason: "creative_plan_prepared"
    });
    expect(directorTransition).toMatchObject({
      kind: "advance",
      label: "Director -> Reasoning",
      reason: "creative_director_prepared"
    });
    expect(reasoningTransition).toMatchObject({
      kind: "advance",
      label: "Reasoning -> Prompt rendering",
      reason: "creative_reasoning_prepared"
    });
  });

  it("derives retries, timing, and completed review steps from explicit graph events", () => {
    const workflow = getLocalWorkspaceSnapshot().workflow;
    const skippedSteps = [
      "memory",
      "retrieval",
      "context_assembly",
      "prompt_input",
      "planning",
      "director",
      "reasoning",
      "prompt_rendering"
    ];
    const traceEvents: WorkflowRuntimeTraceEvent[] = [
      traceEvent({
        at: "2026-05-22T10:00:00Z",
        currentStep: "generation",
        event_type: "node_started",
        sequence: 0,
        step: "generation"
      }),
      traceEvent({
        at: "2026-05-22T10:00:01Z",
        completedSteps: ["intake", "routing", "generation"],
        currentStep: null,
        decisionReason: "generation_completed",
        event_type: "node_completed",
        phase: "completed",
        sequence: 1,
        skippedSteps,
        step: "generation",
        transitionSource: "generation",
        transitionTarget: "artifact_extraction"
      }),
      traceEvent({
        at: "2026-05-22T10:00:02Z",
        completedSteps: ["intake", "routing", "generation"],
        currentStep: "artifact_extraction",
        event_type: "node_started",
        sequence: 2,
        skippedSteps,
        step: "artifact_extraction"
      }),
      traceEvent({
        at: "2026-05-22T10:00:03Z",
        completedSteps: ["intake", "routing", "generation"],
        currentStep: null,
        decisionReason: "no_generated_artifacts",
        event_type: "node_completed",
        phase: "completed",
        sequence: 3,
        skippedSteps: [...skippedSteps, "artifact_extraction"],
        step: "artifact_extraction",
        transitionSource: "artifact_extraction",
        transitionTarget: "preview_preparation"
      }),
      traceEvent({
        at: "2026-05-22T10:00:04Z",
        completedSteps: ["intake", "routing", "generation"],
        currentStep: "preview_preparation",
        event_type: "node_started",
        sequence: 4,
        skippedSteps: [...skippedSteps, "artifact_extraction"],
        step: "preview_preparation"
      }),
      traceEvent({
        at: "2026-05-22T10:00:05Z",
        completedSteps: ["intake", "routing", "generation"],
        currentStep: null,
        decisionReason: "no_artifacts_for_preview",
        event_type: "node_completed",
        phase: "completed",
        sequence: 5,
        skippedSteps: [
          ...skippedSteps,
          "artifact_extraction",
          "preview_preparation"
        ],
        step: "preview_preparation",
        transitionSource: "preview_preparation",
        transitionTarget: "review"
      }),
      traceEvent({
        at: "2026-05-22T10:00:06Z",
        completedSteps: ["intake", "routing", "generation"],
        currentStep: "review",
        event_type: "node_started",
        sequence: 6,
        skippedSteps: [
          ...skippedSteps,
          "artifact_extraction",
          "preview_preparation"
        ],
        step: "review"
      }),
      traceEvent({
        at: "2026-05-22T10:00:07Z",
        completedSteps: ["intake", "routing", "generation"],
        currentStep: "review",
        event_type: "retry_started",
        refinementCount: 1,
        retryCount: 1,
        retryReason: "missing_code_block",
        sequence: 7,
        skippedSteps: [
          ...skippedSteps,
          "artifact_extraction",
          "preview_preparation"
        ],
        step: "review"
      }),
      traceEvent({
        at: "2026-05-22T10:00:08Z",
        completedSteps: ["intake", "routing", "generation", "review"],
        currentStep: null,
        decisionReason: "review_failed_retry_available",
        event_type: "node_completed",
        phase: "completed",
        refinementCount: 1,
        sequence: 8,
        skippedSteps: [
          ...skippedSteps,
          "artifact_extraction",
          "preview_preparation"
        ],
        step: "review",
        transitionSource: "review",
        transitionTarget: "refinement"
      }),
      traceEvent({
        at: "2026-05-22T10:00:09Z",
        completedSteps: ["intake", "routing", "generation", "review"],
        currentStep: "refinement",
        event_type: "node_started",
        refinementCount: 1,
        sequence: 9,
        skippedSteps: [
          ...skippedSteps,
          "artifact_extraction",
          "preview_preparation"
        ],
        step: "refinement"
      }),
      traceEvent({
        at: "2026-05-22T10:00:10Z",
        completedSteps: ["intake", "routing", "generation", "review", "refinement"],
        currentStep: null,
        decisionReason: "refinement_completed",
        event_type: "node_completed",
        phase: "completed",
        refinementCount: 1,
        sequence: 10,
        skippedSteps: [
          ...skippedSteps,
          "artifact_extraction",
          "preview_preparation"
        ],
        step: "refinement",
        transitionSource: "refinement",
        transitionTarget: "generation"
      }),
      traceEvent({
        at: "2026-05-22T10:00:11Z",
        completedSteps: ["intake", "routing", "review", "refinement"],
        currentStep: "generation",
        event_type: "node_started",
        refinementCount: 1,
        sequence: 11,
        skippedSteps,
        step: "generation"
      }),
      traceEvent({
        at: "2026-05-22T10:00:12Z",
        code: "generation_input_prepared",
        currentStep: "generation",
        event_type: "generation_input",
        refinementCount: 1,
        sequence: 12,
        skippedSteps,
        step: "generation"
      }),
      traceEvent({
        at: "2026-05-22T10:00:13Z",
        currentStep: "generation",
        event_type: "token_delta",
        refinementCount: 1,
        sequence: 13,
        skippedSteps,
        step: "generation",
        text: "refined"
      }),
      traceEvent({
        at: "2026-05-22T10:00:14Z",
        completedSteps: ["intake", "routing", "generation", "review", "refinement"],
        currentStep: null,
        decisionReason: "generation_completed",
        event_type: "node_completed",
        phase: "completed",
        refinementCount: 1,
        sequence: 14,
        skippedSteps,
        step: "generation",
        transitionSource: "generation",
        transitionTarget: "artifact_extraction"
      }),
      traceEvent({
        at: "2026-05-22T10:00:15Z",
        completedSteps: ["intake", "routing", "generation", "review", "refinement"],
        currentStep: "artifact_extraction",
        event_type: "node_started",
        refinementCount: 1,
        sequence: 15,
        skippedSteps,
        step: "artifact_extraction"
      }),
      traceEvent({
        at: "2026-05-22T10:00:16Z",
        code: "artifact_extracted",
        completedSteps: ["intake", "routing", "generation", "review", "refinement"],
        currentStep: "artifact_extraction",
        event_type: "artifact_extracted",
        refinementCount: 1,
        sequence: 16,
        skippedSteps,
        step: "artifact_extraction"
      }),
      traceEvent({
        at: "2026-05-22T10:00:17Z",
        completedSteps: [
          "intake",
          "routing",
          "generation",
          "artifact_extraction",
          "review",
          "refinement"
        ],
        currentStep: null,
        decisionReason: "artifacts_extracted",
        event_type: "node_completed",
        phase: "completed",
        refinementCount: 1,
        sequence: 17,
        skippedSteps,
        step: "artifact_extraction",
        transitionSource: "artifact_extraction",
        transitionTarget: "preview_preparation"
      }),
      traceEvent({
        at: "2026-05-22T10:00:18Z",
        completedSteps: [
          "intake",
          "routing",
          "generation",
          "artifact_extraction",
          "review",
          "refinement"
        ],
        currentStep: "preview_preparation",
        event_type: "node_started",
        refinementCount: 1,
        sequence: 18,
        skippedSteps,
        step: "preview_preparation"
      }),
      traceEvent({
        at: "2026-05-22T10:00:19Z",
        code: "preview_artifact_prepared",
        completedSteps: [
          "intake",
          "routing",
          "generation",
          "artifact_extraction",
          "review",
          "refinement"
        ],
        currentStep: "preview_preparation",
        event_type: "preview_artifact",
        refinementCount: 1,
        sequence: 19,
        skippedSteps,
        step: "preview_preparation"
      }),
      traceEvent({
        at: "2026-05-22T10:00:20Z",
        completedSteps: [
          "intake",
          "routing",
          "generation",
          "artifact_extraction",
          "preview_preparation",
          "review",
          "refinement"
        ],
        currentStep: null,
        decisionReason: "preview_metadata_prepared",
        event_type: "node_completed",
        phase: "completed",
        refinementCount: 1,
        sequence: 20,
        skippedSteps,
        step: "preview_preparation",
        transitionSource: "preview_preparation",
        transitionTarget: "review"
      }),
      traceEvent({
        at: "2026-05-22T10:00:21Z",
        completedSteps: [
          "intake",
          "routing",
          "generation",
          "artifact_extraction",
          "preview_preparation",
          "review",
          "refinement"
        ],
        currentStep: "review",
        event_type: "node_started",
        refinementCount: 1,
        sequence: 21,
        skippedSteps,
        step: "review"
      }),
      traceEvent({
        at: "2026-05-22T10:00:22Z",
        completedSteps: [
          "intake",
          "routing",
          "generation",
          "artifact_extraction",
          "preview_preparation",
          "review",
          "refinement"
        ],
        currentStep: "review",
        event_type: "review_passed",
        refinementCount: 1,
        reviewOutcome: "pass",
        sequence: 22,
        skippedSteps,
        step: "review"
      }),
      traceEvent({
        at: "2026-05-22T10:00:23Z",
        completedSteps: [
          "intake",
          "routing",
          "generation",
          "artifact_extraction",
          "preview_preparation",
          "review",
          "refinement"
        ],
        currentStep: "review",
        event_type: "retry_completed",
        refinementCount: 1,
        reviewOutcome: "pass",
        retryCount: 1,
        retryReason: "missing_code_block",
        sequence: 23,
        skippedSteps,
        step: "review"
      }),
      traceEvent({
        at: "2026-05-22T10:00:24Z",
        completedSteps: [
          "intake",
          "routing",
          "generation",
          "artifact_extraction",
          "preview_preparation",
          "review",
          "refinement"
        ],
        currentStep: null,
        decisionReason: "review_passed",
        event_type: "node_completed",
        phase: "completed",
        refinementCount: 1,
        reviewOutcome: "pass",
        sequence: 24,
        skippedSteps,
        step: "review",
        transitionSource: "review",
        transitionTarget: "finalization"
      }),
      traceEvent({
        at: "2026-05-22T10:00:25Z",
        completedSteps: [
          "intake",
          "routing",
          "generation",
          "artifact_extraction",
          "preview_preparation",
          "review",
          "refinement"
        ],
        currentStep: "finalization",
        event_type: "node_started",
        refinementCount: 1,
        reviewOutcome: "pass",
        sequence: 25,
        skippedSteps,
        step: "finalization"
      }),
      traceEvent({
        at: "2026-05-22T10:00:26Z",
        completedSteps: [
          "intake",
          "routing",
          "generation",
          "artifact_extraction",
          "preview_preparation",
          "review",
          "refinement",
          "finalization"
        ],
        currentStep: null,
        decisionReason: "final_answer_emitted",
        event_type: "node_completed",
        phase: "completed",
        refinementCount: 1,
        reviewOutcome: "pass",
        sequence: 26,
        skippedSteps,
        status: "completed",
        step: "finalization",
        transitionSource: "finalization",
        transitionTarget: "end"
      }),
      traceEvent({
        answer: "```ts\nconsole.log('refined');\n```",
        at: "2026-05-22T10:00:27Z",
        completedSteps: [
          "intake",
          "routing",
          "generation",
          "artifact_extraction",
          "preview_preparation",
          "review",
          "refinement",
          "finalization"
        ],
        currentStep: null,
        event_type: "final",
        phase: "completed",
        refinementCount: 1,
        reviewOutcome: "pass",
        sequence: 27,
        skippedSteps,
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
    expect(runtime.summary.transitionCount).toBe(10);
    expect(runtime.summary.totalRuntimeMs).toBe(27000);
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
      "Generation -> Artifact extraction",
      "Artifact extraction -> Preview preparation",
      "Preview preparation -> Review",
      "Review -> Refinement",
      "Refinement -> Generation",
      "Generation -> Artifact extraction",
      "Artifact extraction -> Preview preparation",
      "Preview preparation -> Review",
      "Review -> Finalization",
      "Finalization -> End"
    ]);
    expect(runtime.transitions.map((transition) => transition.reason)).toEqual([
      "generation_completed",
      "no_generated_artifacts",
      "no_artifacts_for_preview",
      "review_failed_retry_available",
      "refinement_completed",
      "generation_completed",
      "artifacts_extracted",
      "preview_metadata_prepared",
      "review_passed",
      "final_answer_emitted"
    ]);
  });
});

function traceEvent({
  answer,
  at,
  code,
  completedSteps = [],
  currentStep,
  decisionReason,
  event_type,
  phase = "running",
  refinementCount = 0,
  reviewOutcome = null,
  retryCount,
  retryReason,
  sequence,
  skippedSteps = [],
  status = "running",
  step,
  transitionSource,
  transitionTarget,
  text
}: {
  answer?: string;
  at: string;
  code?: string;
  completedSteps?: string[];
  currentStep: string | null;
  decisionReason?: string;
  event_type: WorkflowRuntimeTraceEvent["event"]["event_type"];
  phase?: string;
  refinementCount?: number;
  reviewOutcome?: string | null;
  retryCount?: number;
  retryReason?: string;
  sequence: number;
  skippedSteps?: string[];
  status?: string;
  step: string | null;
  transitionSource?: string;
  transitionTarget?: string;
  text?: string;
}): WorkflowRuntimeTraceEvent {
  return {
    event: {
      event_type,
      sequence,
      payload: {
        ...(answer ? { answer } : {}),
        ...(code ? { code } : {}),
        ...(step ? { node: step } : {}),
        ...(text ? { text } : {}),
        ...(retryCount != null ? { retry_count: retryCount } : {}),
        ...(retryReason ? { retry_reason: retryReason } : {}),
        ...(transitionSource ? { transition_source: transitionSource } : {}),
        ...(transitionTarget ? { transition_target: transitionTarget } : {}),
        ...(decisionReason ? { decision_reason: decisionReason } : {}),
        ...(transitionSource && transitionTarget && decisionReason
          ? {
              edge: {
                source: transitionSource,
                target: transitionTarget,
                decision_reason: decisionReason
              }
            }
          : {}),
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
