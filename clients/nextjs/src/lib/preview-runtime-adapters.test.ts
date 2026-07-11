import { describe, expect, it } from "vitest";
import { getLocalWorkspaceSnapshot } from "./assistant-client";
import {
  buildPreviewRuntimeSource,
  canRunPreviewRuntime,
  getExecutablePreviewRuntimeKind
} from "./preview-runtime-adapters";
import { buildPreviewRendererRoute } from "./preview-renderers";

describe("preview runtime adapters", () => {
  it("extracts stable runtime source for routed code artifacts", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const route = buildPreviewRendererRoute({
      artifacts: [
        {
          ...snapshot.artifacts[0],
          summary: "Reactive p5 loop with createCanvas() and draw().",
          title: snapshot.code.title
        }
      ],
      preview: {
        ...snapshot.preview,
        artifactName: snapshot.code.title,
        sourceArtifactName: snapshot.code.title
      },
      previewArtifactId: "source-sketch"
    });
    const source = buildPreviewRuntimeSource({
      code: snapshot.code,
      route
    });

    expect(source).toMatchObject({
      lineCount: snapshot.code.excerpt.length,
      source: snapshot.code.excerpt.join("\n"),
      title: snapshot.code.title
    });
    expect(source.fingerprint).toMatch(/^[a-f0-9]+$/);
  });

  it("marks supported creative routes as executable runtimes", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const p5Artifact = {
      ...snapshot.artifacts[0],
      summary: "Reactive p5 loop with createCanvas() and draw().",
      title: "signal-orbit.p5.ts"
    };
    const threeArtifact = {
      ...snapshot.artifacts[0],
      summary: "Three scene with WebGLRenderer, lights, and camera motion.",
      title: "projection-scene.three.ts"
    };
    const glslArtifact = {
      ...snapshot.artifacts[0],
      language: "GLSL",
      summary: "Fragment shader with gl_FragColor and uniforms.",
      title: "chromatic-field.frag"
    };
    const hydraArtifact = {
      ...snapshot.artifacts[0],
      domain: "hydra",
      runtime: "hydra",
      summary: "Hydra patch with oscillators, modulation, and output routing.",
      title: "feedback-lattice.hydra.js"
    };
    const toneArtifact = {
      ...snapshot.artifacts[0],
      content: [
        "const synth = new Tone.Synth().toDestination();",
        "new Tone.Sequence((time, note) => synth.triggerAttackRelease(note, '8n', time), ['C4', 'E4', 'G4', 'B4'], '8n').start(0);",
        "Tone.Transport.bpm.value = 104;",
        "Tone.Transport.start();"
      ].join("\\n"),
      domain: "tone_js",
      runtime: "tone",
      summary: "Tone.js synth sequence with transport and delay.",
      title: "generative-pulse.tone.js"
    };
    const gsapArtifact = {
      ...snapshot.artifacts[0],
      domain: "gsap",
      runtime: "gsap",
      summary: "GSAP timeline with stagger, repeat, and yoyo transforms.",
      title: "signal-bloom.gsap.ts"
    };
    const svgArtifact = {
      ...snapshot.artifacts[0],
      content:
        '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><circle cx="60" cy="60" r="24" fill="#4cd7c8" /></svg>',
      language: "SVG",
      runtime: "svg",
      summary: "Inline SVG composition with vector paths and animate timing.",
      title: "signal-markup.svg"
    };
    const canvasArtifact = {
      ...snapshot.artifacts[0],
      content:
        "const canvas = document.querySelector('canvas'); const ctx = canvas.getContext('2d'); requestAnimationFrame(function draw(){ ctx.clearRect(0,0,canvas.width,canvas.height); requestAnimationFrame(draw); });",
      domain: "canvas_2d",
      runtime: "canvas",
      summary: "Canvas 2D sketch with requestAnimationFrame and fillRect motion.",
      title: "signal-grid.canvas.js"
    };
    const p5Route = buildPreviewRendererRoute({
      artifacts: [p5Artifact],
      preview: {
        ...snapshot.preview,
        active: true,
        artifactName: p5Artifact.title,
        sourceArtifactName: p5Artifact.title
      },
      previewArtifactId: p5Artifact.id
    });
    const threeRoute = buildPreviewRendererRoute({
      artifacts: [threeArtifact],
      preview: {
        ...snapshot.preview,
        active: true,
        artifactName: threeArtifact.title,
        sourceArtifactName: threeArtifact.title
      },
      previewArtifactId: threeArtifact.id
    });
    const glslRoute = buildPreviewRendererRoute({
      artifacts: [glslArtifact],
      preview: {
        ...snapshot.preview,
        active: true,
        artifactName: glslArtifact.title,
        sourceArtifactName: glslArtifact.title
      },
      previewArtifactId: glslArtifact.id
    });
    const hydraRoute = buildPreviewRendererRoute({
      artifacts: [hydraArtifact],
      preview: {
        ...snapshot.preview,
        active: true,
        artifactName: hydraArtifact.title,
        sourceArtifactName: hydraArtifact.title
      },
      previewArtifactId: hydraArtifact.id
    });
    const toneRoute = buildPreviewRendererRoute({
      artifacts: [toneArtifact],
      preview: {
        ...snapshot.preview,
        active: true,
        artifactName: toneArtifact.title,
        sourceArtifactName: toneArtifact.title
      },
      previewArtifactId: toneArtifact.id
    });
    const gsapRoute = buildPreviewRendererRoute({
      artifacts: [gsapArtifact],
      preview: {
        ...snapshot.preview,
        active: true,
        artifactName: gsapArtifact.title,
        sourceArtifactName: gsapArtifact.title
      },
      previewArtifactId: gsapArtifact.id
    });
    const svgRoute = buildPreviewRendererRoute({
      artifacts: [svgArtifact],
      preview: {
        ...snapshot.preview,
        active: true,
        artifactName: svgArtifact.title,
        sourceArtifactName: svgArtifact.title
      },
      previewArtifactId: svgArtifact.id
    });
    const canvasRoute = buildPreviewRendererRoute({
      artifacts: [canvasArtifact],
      preview: {
        ...snapshot.preview,
        active: true,
        artifactName: canvasArtifact.title,
        sourceArtifactName: canvasArtifact.title
      },
      previewArtifactId: canvasArtifact.id
    });

    expect(getExecutablePreviewRuntimeKind(p5Route)).toBe("p5");
    expect(getExecutablePreviewRuntimeKind(threeRoute)).toBe("three");
    expect(getExecutablePreviewRuntimeKind(glslRoute)).toBe("glsl");
    expect(getExecutablePreviewRuntimeKind(hydraRoute)).toBe("hydra");
    expect(getExecutablePreviewRuntimeKind(toneRoute)).toBe("tone");
    expect(getExecutablePreviewRuntimeKind(gsapRoute)).toBe("gsap");
    expect(getExecutablePreviewRuntimeKind(svgRoute)).toBe("svg");
    expect(getExecutablePreviewRuntimeKind(canvasRoute)).toBe("canvas");
    expect(
      canRunPreviewRuntime({
        preview: { ...snapshot.preview, active: true, state: "ready" },
        route: p5Route
      })
    ).toBe(true);
    expect(
      canRunPreviewRuntime({
        preview: { ...snapshot.preview, active: true, state: "unavailable" },
        route: p5Route
      })
    ).toBe(false);
  });
});
