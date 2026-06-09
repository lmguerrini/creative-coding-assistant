import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type {
  RuntimeConsoleHealthSignal,
  RuntimeConsoleModel,
  RuntimeConsoleTone
} from "@/lib/runtime-console";
import { RuntimeConsoleInspector } from "./runtime-console-inspector";

describe("RuntimeConsoleInspector", () => {
  it.each([
    ["healthy", "Healthy", "Renderer is within budget."],
    ["degraded", "Degraded", "Frame delivery is unstable."],
    ["failed", "Failed", "Renderer execution stopped."]
  ] as const)(
    "renders the %s runtime health signal with an explanation",
    (signal, label, explanation) => {
      render(
        <RuntimeConsoleInspector
          console={createConsoleModel(signal, label, explanation)}
        />
      );

      const health = screen.getByRole("group", { name: "Runtime health" });

      expect(health).toHaveTextContent(label);
      expect(health).toHaveTextContent(explanation);
      expect(health).toHaveTextContent("Execution state");
      expect(health).toHaveTextContent("Running");
    }
  );

  it("renders warnings, errors, reload history, and event history chronologically", () => {
    const model = createConsoleModel(
      "failed",
      "Failed",
      "Renderer execution stopped."
    );

    render(<RuntimeConsoleInspector console={model} />);

    expect(screen.getByRole("list", { name: "Runtime warnings" })).toHaveTextContent(
      "Frame budget exceeded."
    );
    expect(screen.getByRole("list", { name: "Runtime errors" })).toHaveTextContent(
      "Shader compilation failed."
    );
    expect(
      screen.getByRole("group", { name: "Runtime reload history" })
    ).toHaveTextContent("1 reload");

    const eventHistory = screen.getByRole("group", {
      name: "Runtime event history"
    });
    const events = eventHistory.querySelectorAll("[data-event-kind]");

    expect(events).toHaveLength(3);
    expect(events[0]).toHaveAttribute("data-event-kind", "start");
    expect(events[1]).toHaveAttribute("data-event-kind", "reload");
    expect(events[2]).toHaveAttribute("data-event-kind", "error");
    expect(within(eventHistory).getByText("3 chronological events")).toBeVisible();
  });
});

function createConsoleModel(
  signal: RuntimeConsoleHealthSignal,
  label: string,
  explanation: string
): RuntimeConsoleModel {
  const toneBySignal: Record<RuntimeConsoleHealthSignal, RuntimeConsoleTone> = {
    degraded: "warning",
    failed: "danger",
    healthy: "success"
  };
  const events: RuntimeConsoleModel["events"] = [
    {
      artifactName: "diagnostic.frag",
      at: "2026-06-09T10:00:00.000Z",
      atLabel: "10:00:00",
      detail: "Renderer started.",
      id: "1-start",
      kind: "start",
      label: "Start",
      runtimeId: "runtime-1",
      runtimeTypeLabel: "GLSL",
      stateLabel: "Running",
      tone: "active"
    },
    {
      artifactName: "diagnostic.frag",
      at: "2026-06-09T10:00:03.000Z",
      atLabel: "10:00:03",
      detail: "Reload requested.",
      id: "2-reload",
      kind: "reload",
      label: "Reload",
      runtimeId: "runtime-1",
      runtimeTypeLabel: "GLSL",
      stateLabel: "Reloading",
      tone: "warning"
    },
    {
      artifactName: "diagnostic.frag",
      at: "2026-06-09T10:00:05.000Z",
      atLabel: "10:00:05",
      detail: "Shader compilation failed.",
      id: "3-error",
      kind: "error",
      label: "Error",
      runtimeId: "runtime-2",
      runtimeTypeLabel: "GLSL",
      stateLabel: "Error",
      tone: "danger"
    }
  ];

  return {
    badge: label,
    context: {
      artifactName: "diagnostic.frag",
      fingerprint: "abc123",
      lineCountLabel: "42 lines",
      rendererLabel: "GLSL shader surface",
      runtimeTypeLabel: "GLSL",
      sourceName: "diagnostic.frag",
      supportLabel: "Live runtime",
      targetLabel: "Browser preview"
    },
    diagnostics: [],
    emptyDetail: null,
    emptyTitle: null,
    errors: ["Shader compilation failed."],
    events,
    hasRuntimeActivity: true,
    health: {
      explanation,
      label,
      signal,
      tone: toneBySignal[signal]
    },
    hero: {
      detail: "Runtime diagnostics are available.",
      eyebrow: "Runtime console",
      sessionLabel: "runtime-2",
      title: "Live runtime",
      tone: toneBySignal[signal]
    },
    latestError: signal === "failed" ? "Shader compilation failed." : null,
    metrics: [
      {
        id: "status",
        label: "Status",
        tone: "active",
        value: "Running"
      },
      {
        id: "fps",
        label: "FPS",
        tone: toneBySignal[signal],
        value: "60 fps"
      },
      {
        id: "frameTime",
        label: "Frame time",
        tone: toneBySignal[signal],
        value: "16.7 ms"
      },
      {
        id: "uptime",
        label: "Uptime",
        tone: "active",
        value: "5s"
      },
      {
        id: "reloadCount",
        label: "Reloads",
        tone: "warning",
        value: "1"
      },
      {
        id: "executionDuration",
        label: "Execution",
        tone: "active",
        value: "5s"
      }
    ],
    reloadHistory: [events[1]!],
    summary: "Runtime diagnostics available",
    warnings: ["Frame budget exceeded."]
  };
}
