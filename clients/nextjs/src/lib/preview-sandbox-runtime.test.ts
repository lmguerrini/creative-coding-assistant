import { describe, expect, it } from "vitest";
import {
  buildPreviewSandboxDocument,
  mountPreviewSandboxRuntime,
  preparePreviewExecutableSource,
  readPreviewSandboxRuntimeMessage
} from "./preview-sandbox-runtime";

describe("preview sandbox runtime", () => {
  it("prepares TypeScript-flavored p5 source for browser execution", () => {
    const prepared = preparePreviewExecutableSource(
      [
        "import p5 from 'p5';",
        "type Palette = { accent: string };",
        "export function drawParticle(x: number, y: number): void {",
        "  circle(x, y, 12);",
        "}",
        "export default function draw() {",
        "  const palette: Palette = { accent: '#4cd7c8' };",
        "  fill(palette.accent);",
        "  drawParticle(width * 0.5, height * 0.5);",
        "}"
      ].join("\n"),
      "p5"
    );

    expect(prepared).not.toContain("import p5");
    expect(prepared).not.toContain("type Palette");
    expect(prepared).not.toContain(": number");
    expect(prepared).toContain("function drawParticle(x, y)");
    expect(prepared).toContain("function draw()");
    expect(prepared).toContain("const palette = { accent: '#4cd7c8' };");
  });

  it("builds an escaped sandbox document with the selected runtime payload", () => {
    const document = buildPreviewSandboxDocument({
      kind: "glsl",
      runtimeId: "runtime-1",
      source: {
        fingerprint: "abc123",
        lineCount: 1,
        source: "void main() { gl_FragColor = vec4(1.0); }</script>",
        title: "field.frag"
      }
    });

    expect(document).toContain('<canvas id="preview-canvas">');
    expect(document).toContain("runtime-1");
    expect(document).toContain("field.frag");
    expect(document).toContain("\\u003c/script\\u003e");
    expect(document).not.toContain("</script>\"");
  });

  it("accepts only matching sandbox runtime messages", () => {
    expect(
      readPreviewSandboxRuntimeMessage(
        {
          source: "cca-preview-runtime",
          runtimeId: "runtime-1",
          type: "status",
          status: {
            detail: "Executing sketch.",
            label: "p5 runtime running",
            state: "running"
          }
        },
        "runtime-1"
      )
    ).toMatchObject({
      type: "status",
      status: {
        detail: "Executing sketch.",
        label: "p5 runtime running",
        state: "running"
      }
    });

    expect(
      readPreviewSandboxRuntimeMessage(
        {
          renderedAtMs: 16,
          runtimeId: "runtime-1",
          source: "cca-preview-runtime",
          type: "frame"
        },
        "runtime-1"
      )
    ).toMatchObject({
      renderedAtMs: 16,
      type: "frame"
    });

    expect(
      readPreviewSandboxRuntimeMessage(
        {
          source: "cca-preview-runtime",
          runtimeId: "runtime-2",
          type: "frame",
          renderedAtMs: 16
        },
        "runtime-1"
      )
    ).toBeNull();
  });

  it("mounts the same-origin sandbox host and receives runtime status", () => {
    const iframe = document.createElement("iframe");
    const statuses: string[] = [];
    document.body.appendChild(iframe);

    const runtime = mountPreviewSandboxRuntime({
      iframe,
      kind: "p5",
      onStatus: (status) => statuses.push(status.label),
      runtimeId: "runtime-1",
      source: {
        fingerprint: "abc123",
        lineCount: 1,
        source: "function draw() { circle(20, 20, 10); }",
        title: "sketch.p5.js"
      }
    });

    expect(iframe.dataset.runtimeId).toBe("runtime-1");
    expect(iframe.getAttribute("src")).toBe("/preview-sandbox.html");

    window.dispatchEvent(
      new MessageEvent("message", {
        data: {
          source: "cca-preview-runtime",
          runtimeId: "runtime-1",
          type: "status",
          status: {
            detail: "Executing sketch.",
            label: "p5 runtime running",
            state: "running"
          }
        }
      })
    );

    expect(statuses).toEqual([
      "Preview runtime starting",
      "p5 runtime running"
    ]);

    runtime.dispose();
    expect(iframe.dataset.runtimeId).toBeUndefined();
    expect(iframe.getAttribute("src")).toBe("about:blank");
    iframe.remove();
  });

  it("ignores stale messages after a sandbox remount", () => {
    const iframe = document.createElement("iframe");
    const statuses: string[] = [];
    document.body.appendChild(iframe);

    const firstRuntime = mountPreviewSandboxRuntime({
      iframe,
      kind: "p5",
      onStatus: (status) => statuses.push(status.label),
      runtimeId: "runtime-1",
      source: {
        fingerprint: "abc123",
        lineCount: 1,
        source: "function draw() { circle(20, 20, 10); }",
        title: "sketch.p5.js"
      }
    });

    firstRuntime.dispose();
    const secondRuntime = mountPreviewSandboxRuntime({
      iframe,
      kind: "p5",
      onStatus: (status) => statuses.push(status.label),
      runtimeId: "runtime-2",
      source: {
        fingerprint: "def456",
        lineCount: 1,
        source: "function draw() { circle(40, 40, 20); }",
        title: "sketch-v2.p5.js"
      }
    });

    window.dispatchEvent(
      new MessageEvent("message", {
        data: {
          source: "cca-preview-runtime",
          runtimeId: "runtime-1",
          type: "status",
          status: {
            detail: "Stale runtime failed.",
            error: {
              message: "Stale runtime failed.",
              type: "preview_sandbox_runtime_failed"
            },
            label: "p5 runtime failed",
            state: "error"
          }
        }
      })
    );
    window.dispatchEvent(
      new MessageEvent("message", {
        data: {
          source: "cca-preview-runtime",
          runtimeId: "runtime-2",
          type: "status",
          status: {
            detail: "Executing the remounted sketch.",
            label: "p5 runtime running",
            state: "running"
          }
        }
      })
    );

    expect(statuses).toEqual([
      "Preview runtime starting",
      "Preview runtime starting",
      "p5 runtime running"
    ]);

    secondRuntime.dispose();
    iframe.remove();
  });
});
