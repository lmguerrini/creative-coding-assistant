import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it, vi } from "vitest";
import type { PreviewRuntimeStatus } from "./preview-runtime-adapters";
import {
  buildPreviewSandboxDocument,
  getPreviewRuntimeSourceMismatch,
  mountPreviewSandboxRuntime,
  preparePreviewExecutableSource,
  readPreviewSandboxRuntimeMessage
} from "./preview-sandbox-runtime";

describe("preview sandbox runtime", () => {
  it("keeps standard p5 angle constants aligned with the sandbox globals", () => {
    const sandboxDocument = readFileSync(
      resolve(process.cwd(), "public/preview-sandbox.html"),
      "utf8"
    );

    expect(sandboxDocument).toContain("PI: Math.PI,");
    expect(sandboxDocument).toContain("TAU: Math.PI * 2,");
    expect(sandboxDocument).toContain("exp: Math.exp,");
    expect(sandboxDocument).toContain("constrain: function (value, minimum, maximum)");
    expect(sandboxDocument).toContain("degrees: function (radians) { return (Number(radians) || 0) * 180 / Math.PI; },");
    expect(sandboxDocument).toContain("int: function (value)");
    expect(sandboxDocument).toContain("TWO_PI: Math.PI * 2,");
    expect(sandboxDocument).toContain("HALF_PI: Math.PI / 2,");
    expect(sandboxDocument).toContain("ADD: \"lighter\",");
    expect(sandboxDocument).toContain("ROUND: \"round\",");
    expect(sandboxDocument).toContain("globals.blendMode = function (mode)");
    expect(sandboxDocument).toContain("globals.curveVertex = globals.vertex");
    expect(sandboxDocument).toContain("if (Array.isArray(min)) return min[Math.floor(value * min.length)];");
    expect(sandboxDocument).toContain("globals.rectMode = function (mode)");
    expect(sandboxDocument).toContain("globals.red = function (value)");
    expect(sandboxDocument).toContain("smooth: function () {}");
    expect(sandboxDocument).toContain("globals.strokeCap = function (cap)");
    expect(sandboxDocument).toContain("globals.strokeJoin = function (join)");
  });

  it("loads the same licensed Three.js r176 runtime in both sandbox documents", () => {
    const staticSandboxDocument = readFileSync(
      resolve(process.cwd(), "public/preview-sandbox.html"),
      "utf8"
    );
    const generatedSandboxDocument = buildPreviewSandboxDocument({
      kind: "three",
      runtimeId: "three-runtime-contract",
      source: {
        fingerprint: "three-runtime-contract",
        lineCount: 1,
        source: "new THREE.Scene();",
        title: "runtime-contract.three.js"
      }
    });
    const license = readFileSync(
      resolve(process.cwd(), "public/vendor/three.LICENSE.txt"),
      "utf8"
    );

    for (const sandboxDocument of [staticSandboxDocument, generatedSandboxDocument]) {
      expect(sandboxDocument).toContain('src="/vendor/three-r176.min.js"');
      expect(sandboxDocument).toContain('bundledThree.REVISION !== "176"');
      expect(sandboxDocument).toContain("class SandboxWebGLRenderer extends bundledThree.WebGLRenderer");
      expect(sandboxDocument).toContain("this.__ccaRender(scene, camera)");
      expect(sandboxDocument).toContain("gl.readPixels");
      expect(sandboxDocument).toContain("dataset.threeRuntimeRevision");
    }
    expect(license).toContain("Copyright © 2010-2025 three.js authors");
    expect(license).toContain("The MIT License");
  });

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

  it("detects HTML documents as incompatible with the p5 JavaScript runtime", () => {
    const htmlSource = [
      "<!doctype html>",
      "<html>",
      "<body><script>function draw() { circle(20, 20, 10); }</script></body>",
      "</html>"
    ].join("\n");

    expect(
      getPreviewRuntimeSourceMismatch({
        kind: "p5",
        source: {
          fingerprint: "html123",
          lineCount: 4,
          source: htmlSource,
          title: "generated-sketch-1.p5.ts"
        }
      })
    ).toContain("HTML documents cannot run");
  });

  it("detects leading-comment HTML documents as incompatible with the p5 JavaScript runtime", () => {
    const commentedHtmlSource = [
      "<!-- index.html -->",
      "<!doctype html>",
      "<html>",
      "<body><script>function draw() { circle(20, 20, 10); }</script></body>",
      "</html>"
    ].join("\n");
    const spacedCommentedHtmlSource = [
      "",
      "  <!-- index.html -->",
      "  <!doctype html>",
      "<html>",
      "<body><script>function setup() { createCanvas(200, 200); }</script></body>",
      "</html>"
    ].join("\n");

    expect(
      getPreviewRuntimeSourceMismatch({
        kind: "p5",
        source: {
          fingerprint: "html-comment123",
          lineCount: 5,
          source: commentedHtmlSource,
          title: "generated-sketch-1.p5.ts"
        }
      })
    ).toContain("HTML documents cannot run");

    expect(
      getPreviewRuntimeSourceMismatch({
        kind: "p5",
        source: {
          fingerprint: "html-comment-spaced123",
          lineCount: 6,
          source: spacedCommentedHtmlSource,
          title: "generated-sketch-1.p5.ts"
        }
      })
    ).toContain("HTML documents cannot run");
  });

  it("accepts valid p5 JavaScript source with normal comments", () => {
    expect(
      getPreviewRuntimeSourceMismatch({
        kind: "p5",
        source: {
          fingerprint: "p5js123",
          lineCount: 6,
          source: [
            "// index.js",
            "/* p5 sketch source */",
            "function setup() { createCanvas(200, 200); }",
            "function draw() {",
            "  circle(20, 20, 10);",
            "}"
          ].join("\n"),
          title: "generated-sketch-1.p5.ts"
        }
      })
    ).toBeNull();
  });

  it("accepts the global-mode p5 contract used by the flow-field suggestion", () => {
    const source = [
      "const particles = [];",
      "function setup() {",
      "  createCanvas(windowWidth, windowHeight);",
      "  pixelDensity(1);",
      "  colorMode(HSL, 360, 100, 100, 1);",
      "  noiseDetail(3, 0.5);",
      "  noStroke();",
      "}",
      "function draw() {",
      "  background(220, 38, 8, 0.14);",
      "  for (let i = 0; i < 24; i += 1) {",
      "    const x = map(i, 0, 23, 24, width - 24);",
      "    const angle = noise(i * 0.1, frameCount * 0.01) * 6.28;",
      "    push();",
      "    translate(x, height * 0.5);",
      "    rotate(angle);",
      "    beginShape();",
      "    vertex(-8, -4);",
      "    vertex(10, 0);",
      "    vertex(-8, 4);",
      "    endShape(CLOSE);",
      "    pop();",
      "  }",
      "}"
    ].join("\n");

    expect(
      getPreviewRuntimeSourceMismatch({
        kind: "p5",
        source: {
          fingerprint: "flow-field-global-mode",
          lineCount: source.split("\n").length,
          source,
          title: "generated-sketch-1.p5.js"
        }
      })
    ).toBeNull();
  });

  it("rejects unsupported p5 APIs before mounting a preview", () => {
    expect(
      getPreviewRuntimeSourceMismatch({
        kind: "p5",
        source: {
          fingerprint: "unsupported-p5-api",
          lineCount: 7,
          source: [
            "function setup() { createCanvas(200, 200); createGraphics(200, 200); }",
            "function draw() { background(0); circle(100, 100, 40); }"
          ].join("\n"),
          title: "unsafe-controls.p5.js"
        }
      })
    ).toContain("createGraphics() is not part of the supported browser p5 preview contract");
  });

  it("rejects unsupported GLSL source again at the sandbox mount boundary", () => {
    expect(
      getPreviewRuntimeSourceMismatch({
        kind: "glsl",
        source: {
          fingerprint: "sampled-fragment",
          lineCount: 1,
          source:
            "uniform sampler2D sourceTexture; void main() { gl_FragColor = texture2D(sourceTexture, vec2(0.5)); }",
          title: "sampled-field.frag"
        }
      })
    ).toContain("outside the current bounded runtime subset");
  });

  it("rejects unsupported Three.js source again at the sandbox mount boundary", () => {
    expect(
      getPreviewRuntimeSourceMismatch({
        kind: "three",
        source: {
          fingerprint: "three-html-document",
          lineCount: 1,
          source: "<!doctype html><html><body><script>new THREE.Scene()</script></body></html>",
          title: "exported-scene.three.ts"
        }
      })
    ).toContain("Standalone HTML documents cannot run");
  });

  it("reapplies every other bounded runtime source contract at the sandbox mount boundary", () => {
    const rejectedSources = [
      {
        kind: "hydra" as const,
        source: "osc(12, 0.2, 0.4)",
        expected: "end with out()"
      },
      {
        kind: "tone" as const,
        source: "const value = 1;",
        expected: "must reference the Tone namespace"
      },
      {
        kind: "gsap" as const,
        source: "fetch('https://example.test'); gsap.to('.particle', { x: 20 });",
        expected: "Remote network access"
      },
      {
        kind: "svg" as const,
        source: "<svg onclick=\"alert(1)\"></svg>",
        expected: "without event handlers"
      },
      {
        kind: "canvas" as const,
        source: "fetch('https://example.test'); ctx.fillRect(0, 0, 1, 1);",
        expected: "Remote network access"
      }
    ];

    for (const rejected of rejectedSources) {
      expect(
        getPreviewRuntimeSourceMismatch({
          kind: rejected.kind,
          source: {
            fingerprint: `rejected-${rejected.kind}`,
            lineCount: 1,
            source: rejected.source,
            title: `rejected.${rejected.kind}`
          }
        })
      ).toContain(rejected.expected);
    }
  });

  it("prepares Hydra source as a bounded execution plan", () => {
    const source =
      "osc(10, 0.1, 1.2).modulate(shape(4, 0.3, 0.02), 0.12).out(o1); render(o1);";
    const prepared = preparePreviewExecutableSource(source, "hydra");
    const program = JSON.parse(prepared);

    expect(program).toMatchObject({
      renderTarget: "o1",
      speed: 1,
      version: 1
    });
    expect(program.outputs.o1.source.name).toBe("osc");
    expect(program.outputs.o1.operators[0].name).toBe("modulate");
    expect(prepared).not.toContain(source);
  });

  it("prepares TypeScript-flavored GSAP source for sandbox execution", () => {
    const prepared = preparePreviewExecutableSource(
      [
        "import { gsap } from 'gsap';",
        "interface MotionStep { x: number; }",
        "const step: MotionStep = { x: 120 };",
        "export function animate(): void {",
        "  gsap.to('.particle', { x: step.x, stagger: 0.08, repeat: -1, yoyo: true });",
        "}"
      ].join("\n"),
      "gsap"
    );

    expect(prepared).not.toContain("import { gsap }");
    expect(prepared).not.toContain("interface MotionStep");
    expect(prepared).not.toContain(": number");
    expect(prepared).toContain("const step = { x: 120 };");
    expect(prepared).toContain("function animate()");
  });

  it("preserves inline SVG markup for sandbox execution", () => {
    const prepared = preparePreviewExecutableSource(
      [
        '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">',
        '  <circle cx="60" cy="60" r="24" fill="#4cd7c8">',
        '    <animate attributeName="r" values="22;30;22" dur="2s" repeatCount="indefinite" />',
        "  </circle>",
        "</svg>"
      ].join("\n"),
      "svg"
    );

    expect(prepared).toContain("<svg");
    expect(prepared).toContain("<animate");
    expect(prepared).not.toContain("function");
  });

  it("prepares Tone.js source as a silent bounded audio plan", () => {
    const source = [
      "const synth = new Tone.Synth().toDestination();",
      "new Tone.Sequence((time, note) => synth.triggerAttackRelease(note, '8n', time), ['C4', 'E4', 'G4'], '8n').start(0);",
      "Tone.Transport.start();"
    ].join("\n");
    const prepared = preparePreviewExecutableSource(source, "tone");
    const program = JSON.parse(prepared);

    expect(program).toMatchObject({
      version: 1,
      voices: [{ kind: "synth" }],
      patterns: [
        {
          notes: ["C4", "E4", "G4"],
          subdivision: "8n"
        }
      ]
    });
    expect(prepared).not.toContain("Transport.start");
  });

  it("carries the explicit Cymatics marker into a muted visual runtime plan", () => {
    const source = [
      "// CCA_VISUAL: cymatics",
      "const synth = new Tone.Synth().toDestination();",
      "new Tone.Sequence((time, note) => synth.triggerAttackRelease(note, '8n', time), ['C3', 'G3'], '8n').start(0);",
      "Tone.Transport.bpm.value = 96;",
      "Tone.Transport.start();"
    ].join("\n");
    const prepared = JSON.parse(preparePreviewExecutableSource(source, "tone"));
    const sandboxDocument = readFileSync(
      resolve(process.cwd(), "public/preview-sandbox.html"),
      "utf8"
    );

    expect(prepared.visualization).toBe("cymatics");
    expect(sandboxDocument).toContain('toneProgram.visualization === "cymatics"');
    expect(sandboxDocument).toContain("function drawCymaticsPreview");
    expect(sandboxDocument).toContain("Audio remains silent until Start audio is selected.");
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

  it("rejects HTML p5 payloads before mounting the sandbox host", () => {
    const iframe = document.createElement("iframe");
    const statuses: PreviewRuntimeStatus[] = [];
    document.body.appendChild(iframe);

    const runtime = mountPreviewSandboxRuntime({
      iframe,
      kind: "p5",
      onStatus: (status) => statuses.push(status),
      runtimeId: "runtime-html",
      source: {
        fingerprint: "html123",
        lineCount: 5,
        source: [
          "<!-- index.html -->",
          "<!doctype html>",
          "<html>",
          "<body><script>function draw() { circle(20, 20, 10); }</script></body>",
          "</html>"
        ].join("\n"),
        title: "generated-sketch-1.p5.ts"
      }
    });

    expect(iframe.getAttribute("src")).toBeNull();
    expect(iframe.dataset.runtimeId).toBeUndefined();
    expect(statuses).toHaveLength(1);
    expect(statuses[0]).toMatchObject({
      diagnostics: [expect.stringContaining("HTML documents cannot run")],
      error: {
        type: "preview_runtime_source_mismatch"
      },
      label: "p5 runtime rejected source",
      state: "error"
    });

    runtime.control("start");
    runtime.dispose();
    iframe.remove();
  });

  it("mounts GSAP previews on the shared sandbox host", () => {
    const iframe = document.createElement("iframe");
    const statuses: string[] = [];
    document.body.appendChild(iframe);

    const runtime = mountPreviewSandboxRuntime({
      iframe,
      kind: "gsap",
      onStatus: (status) => statuses.push(status.label),
      runtimeId: "gsap-runtime-1",
      source: {
        fingerprint: "gsap123",
        lineCount: 2,
        source: "gsap.to('.particle', { x: 80, stagger: 0.08, repeat: -1, yoyo: true });",
        title: "signal-bloom.gsap.js"
      }
    });

    expect(iframe.getAttribute("src")).toBe("/preview-sandbox.html");
    expect(statuses).toEqual(["Preview runtime starting"]);

    runtime.dispose();
    iframe.remove();
  });

  it("mounts SVG and Canvas previews on the shared sandbox host", () => {
    const svgIframe = document.createElement("iframe");
    const canvasIframe = document.createElement("iframe");
    const statuses: string[] = [];
    document.body.appendChild(svgIframe);
    document.body.appendChild(canvasIframe);

    const svgRuntime = mountPreviewSandboxRuntime({
      iframe: svgIframe,
      kind: "svg",
      onStatus: (status) => statuses.push(status.label),
      runtimeId: "svg-runtime-1",
      source: {
        fingerprint: "svg123",
        lineCount: 1,
        source:
          '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><circle cx="60" cy="60" r="24" fill="#4cd7c8" /></svg>',
        title: "signal-markup.svg"
      }
    });
    const canvasRuntime = mountPreviewSandboxRuntime({
      iframe: canvasIframe,
      kind: "canvas",
      onStatus: (status) => statuses.push(status.label),
      runtimeId: "canvas-runtime-1",
      source: {
        fingerprint: "canvas123",
        lineCount: 2,
        source:
          "const canvas = document.querySelector('canvas'); const ctx = canvas.getContext('2d'); ctx.fillRect(0, 0, 10, 10);",
        title: "signal-grid.canvas.js"
      }
    });

    expect(svgIframe.getAttribute("src")).toBe("/preview-sandbox.html");
    expect(canvasIframe.getAttribute("src")).toBe("/preview-sandbox.html");
    expect(statuses).toEqual([
      "Preview runtime starting",
      "Preview runtime starting"
    ]);

    svgRuntime.dispose();
    canvasRuntime.dispose();
    svgIframe.remove();
    canvasIframe.remove();
  });

  it("delivers explicit Tone.js start, stop, and mute controls to the sandbox", () => {
    const iframe = document.createElement("iframe");
    document.body.appendChild(iframe);
    const runtime = mountPreviewSandboxRuntime({
      iframe,
      kind: "tone",
      onStatus: () => undefined,
      runtimeId: "tone-runtime-1",
      source: {
        fingerprint: "tone123",
        lineCount: 1,
        source: "const synth = new Tone.Synth().toDestination();",
        title: "pulse.tone.js"
      }
    });
    const postMessage = vi.spyOn(iframe.contentWindow as Window, "postMessage");

    runtime.control("start");
    runtime.control("mute");
    runtime.control("unmute");
    runtime.control("stop");

    const controlActions = postMessage.mock.calls
      .map(([message]) => message)
      .filter(
        (message): message is { action: string; type: string } =>
          typeof message === "object" &&
          message !== null &&
          "type" in message &&
          message.type === "control"
      )
      .map((message) => message.action);
    expect(controlActions).toEqual(["start", "mute", "unmute", "stop"]);

    runtime.dispose();
    iframe.remove();
  });

  it("accepts Tone.js ready and stopped lifecycle status", () => {
    expect(
      readPreviewSandboxRuntimeMessage(
        {
          source: "cca-preview-runtime",
          runtimeId: "tone-runtime-1",
          type: "status",
          status: {
            detail: "Audio is armed.",
            label: "Tone.js runtime ready",
            state: "ready"
          }
        },
        "tone-runtime-1"
      )
    ).toMatchObject({
      status: { state: "ready" }
    });
    expect(
      readPreviewSandboxRuntimeMessage(
        {
          source: "cca-preview-runtime",
          runtimeId: "tone-runtime-1",
          type: "status",
          status: {
            detail: "Audio is silent.",
            label: "Tone.js runtime stopped",
            state: "stopped"
          }
        },
        "tone-runtime-1"
      )
    ).toMatchObject({
      status: { state: "stopped" }
    });
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
