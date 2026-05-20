export type AssistantModeState = {
  label: "Generate" | "Preview" | "Review";
  active: boolean;
};

export type WorkflowStepState = {
  name: string;
  state: "complete" | "active" | "queued";
};

export type AssistantMessage = {
  role: "user" | "assistant";
  time: string;
  content: string;
};

export type ArtifactSummary = {
  id: string;
  title: string;
  type: "code" | "preview" | "export";
  summary: string;
};

export type PreviewSummary = {
  title: string;
  target: string;
  status: string;
  artifactName: string;
  summary: string;
  renderer: string;
  latency: string;
  version: string;
};

export type DebugEventSummary = {
  code: string;
  label: string;
  detail: string;
};

export type AssistantWorkspaceSnapshot = {
  workspace: {
    name: string;
  };
  modes: AssistantModeState[];
  messages: AssistantMessage[];
  workflow: {
    status: string;
    currentStep: string;
    steps: WorkflowStepState[];
  };
  artifacts: ArtifactSummary[];
  preview: PreviewSummary;
  debug: {
    traceId: string;
    status: string;
    events: DebugEventSummary[];
  };
};

export type AssistantFrontendClient = {
  getWorkspaceSnapshot: () => Promise<AssistantWorkspaceSnapshot>;
};

export function createAssistantClient(): AssistantFrontendClient {
  return {
    async getWorkspaceSnapshot() {
      return getLocalWorkspaceSnapshot();
    }
  };
}

export function getLocalWorkspaceSnapshot(): AssistantWorkspaceSnapshot {
  return {
    workspace: {
      name: "Session workspace / WebGPU kinetic field"
    },
    modes: [
      { label: "Generate", active: true },
      { label: "Preview", active: false },
      { label: "Review", active: false }
    ],
    messages: [
      {
        role: "user",
        time: "09:24",
        content:
          "Build a luminous particle field that reacts to low-frequency audio and keeps motion legible on a projection wall."
      },
      {
        role: "assistant",
        time: "09:25",
        content:
          "Drafting a WebGPU sketch with a stable simulation pass, palette controls, and a previewable browser target."
      }
    ],
    workflow: {
      status: "Running",
      currentStep: "Generation pipeline",
      steps: [
        { name: "Routing", state: "complete" },
        { name: "Retrieval", state: "complete" },
        { name: "Generation", state: "active" },
        { name: "Preview request", state: "queued" },
        { name: "Review", state: "queued" }
      ]
    },
    artifacts: [
      {
        id: "source-sketch",
        title: "webgpu-particle-field.ts",
        type: "code",
        summary: "Primary generated sketch artifact with browser preview target."
      },
      {
        id: "preview-manifest",
        title: "preview-request.json",
        type: "preview",
        summary: "Browser target, renderer identity, and artifact v1 linkage."
      },
      {
        id: "session-notes",
        title: "projection-notes.md",
        type: "export",
        summary: "Design constraints for projection scale, motion density, and palette."
      }
    ],
    preview: {
      title: "Preview",
      target: "Browser sandbox / WebGPU WGSL",
      status: "Pending",
      artifactName: "webgpu-particle-field.ts",
      summary:
        "Staged frame for particle density, palette balance, and projection-scale motion.",
      renderer: "preview.noop",
      latency: "--",
      version: "v1"
    },
    debug: {
      traceId: "trace.local.nextjs-foundation",
      status: "Live",
      events: [
        {
          code: "route_selected",
          label: "Route",
          detail: "generate route with tool and preview artifact capability"
        },
        {
          code: "artifact_linked",
          label: "Artifact",
          detail: "source-sketch is linked to the current workspace session"
        },
        {
          code: "preview_queued",
          label: "Preview",
          detail: "browser_sandbox target resolved from WebGPU artifact metadata"
        }
      ]
    }
  };
}
