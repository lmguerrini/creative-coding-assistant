import { describe, expect, it } from "vitest";
import {
  getLocalWorkspaceSnapshot,
  type ArtifactSummary,
  type PreviewSummary
} from "./assistant-client";
import {
  buildPreviewRendererRoute,
  matchCreativePreviewRenderer
} from "./preview-renderers";

describe("preview renderers", () => {
  it("routes the default p5 sketch into the sandbox browser runtime", () => {
    const snapshot = getLocalWorkspaceSnapshot();

    expect(
      buildPreviewRendererRoute({
        artifacts: snapshot.artifacts,
        preview: snapshot.preview,
        previewArtifactId: "source-sketch"
      })
    ).toMatchObject({
      targetId: "browser_sandbox",
      targetLabel: "Browser preview",
      rendererId: "surface.p5",
      rendererLabel: "p5.js",
      supportState: "supported",
      surfaceKind: "p5"
    });
  });

  it("routes preview manifests into the json panel surface", () => {
    const snapshot = getLocalWorkspaceSnapshot();

    expect(
      buildPreviewRendererRoute({
        artifacts: snapshot.artifacts,
        preview: {
          ...snapshot.preview,
          active: true,
          artifactName: "preview-request.json",
          outputArtifactName: "preview-request.json",
          sourceArtifactId: "source-sketch",
          sourceArtifactName: "aurora-field.p5.js",
          state: "ready",
          status: "Preview open"
        },
        previewArtifactId: "preview-manifest"
      })
    ).toMatchObject({
      targetId: "json_panel",
      targetLabel: "JSON panel",
      rendererId: "surface.json_panel",
      rendererLabel: "JSON panel surface",
      supportState: "supported",
      surfaceKind: "json_panel"
    });
  });

  it("executes the explicitly selected code artifact instead of stale GLSL preview metadata", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const threeArtifact = creativeArtifact({
      content: [
        "const scene = new THREE.Scene();",
        "const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);",
        "const renderer = new THREE.WebGLRenderer();",
        "scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial()));"
      ].join("\n"),
      domain: "three_js",
      runtime: "three",
      title: "concentric-audio-glow.three.js"
    });
    const staleShader = creativeArtifact({
      content: "void main() { gl_FragColor = vec4(1.0); }",
      domain: "glsl",
      language: "GLSL",
      runtime: "glsl",
      title: "stale-preview.frag"
    });

    expect(
      buildPreviewRendererRoute({
        artifacts: [threeArtifact, staleShader],
        preview: {
          ...snapshot.preview,
          active: true,
          available: true,
          artifactName: staleShader.title,
          outputArtifactName: staleShader.title,
          renderer: "surface.glsl",
          sourceArtifactId: staleShader.id,
          sourceArtifactName: staleShader.title,
          state: "ready",
          status: "Preview open",
          target: "Browser preview",
          targetId: "browser_sandbox"
        },
        previewArtifactId: threeArtifact.id
      })
    ).toMatchObject({
      rendererId: "surface.three",
      selectedArtifactId: threeArtifact.id,
      sourceArtifactId: threeArtifact.id,
      sourceArtifactName: threeArtifact.title,
      supportState: "supported",
      surfaceKind: "three"
    });
  });

  it.each([
    {
      id: "surface.p5",
      kind: "p5",
      artifact: creativeArtifact({
        summary: "Reactive p5 loop with createCanvas() and draw().",
        title: "signal-orbit.p5.ts"
      })
    },
    {
      id: "surface.three",
      kind: "three",
      artifact: creativeArtifact({
        summary: "Three scene with WebGLRenderer, lights, and camera motion.",
        title: "projection-scene.three.ts"
      })
    },
    {
      id: "surface.glsl",
      kind: "glsl",
      artifact: creativeArtifact({
        language: "GLSL",
        summary: "Fragment shader with gl_FragColor and uniforms.",
        title: "chromatic-field.frag"
      })
    },
    {
      id: "surface.hydra",
      kind: "hydra",
      artifact: creativeArtifact({
        domain: "hydra",
        runtime: "hydra",
        summary: "Hydra patch built from osc(), shape(), modulation, and out().",
        title: "feedback-lattice.hydra.js"
      })
    },
    {
      id: "surface.tone",
      kind: "tone",
      artifact: creativeArtifact({
        domain: "tone_js",
        runtime: "tone",
        summary: "Tone.js synth sequence with transport and delay.",
        title: "generative-pulse.tone.js"
      })
    },
    {
      id: "surface.gsap",
      kind: "gsap",
      artifact: creativeArtifact({
        content:
          "const tl = gsap.timeline({ repeat: -1, yoyo: true });\n" +
          "tl.to('.particle', { x: 120, rotation: 90, stagger: 0.08 });",
        domain: "gsap",
        runtime: "gsap",
        summary: "GSAP timeline with staggered transform motion and yoyo repeats.",
        title: "signal-bloom.gsap.ts"
      })
    },
    {
      id: "surface.svg",
      kind: "svg",
      artifact: creativeArtifact({
        content:
          '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">' +
          '<circle cx="60" cy="60" r="28" fill="#4cd7c8">' +
          '<animate attributeName="r" values="24;32;24" dur="2s" repeatCount="indefinite" />' +
          "</circle></svg>",
        language: "SVG",
        runtime: "svg",
        summary: "Inline SVG composition with circle geometry and native animate timing.",
        title: "signal-markup.svg"
      })
    },
    {
      id: "surface.canvas",
      kind: "canvas",
      artifact: creativeArtifact({
        content: [
          "const canvas = document.querySelector('canvas');",
          "const ctx = canvas.getContext('2d');",
          "function draw(time) {",
          "  ctx.clearRect(0, 0, canvas.width, canvas.height);",
          "  ctx.fillStyle = '#4cd7c8';",
          "  ctx.fillRect(24 + Math.sin(time * 0.002) * 40, 24, 88, 88);",
          "  requestAnimationFrame(draw);",
          "}",
          "requestAnimationFrame(draw);"
        ].join("\n"),
        domain: "canvas_2d",
        runtime: "canvas",
        summary: "Canvas 2D sketch with requestAnimationFrame and fillRect motion.",
        title: "signal-grid.canvas.js"
      })
    }
  ])(
    "matches $kind renderer signals and routes them into a supported surface",
    ({ artifact, id, kind }) => {
      const snapshot = getLocalWorkspaceSnapshot();
      const preview = creativePreviewSummary(artifact, snapshot.preview);

      expect(matchCreativePreviewRenderer(artifact)).toMatchObject({
        id,
        kind
      });
      expect(
        buildPreviewRendererRoute({
          artifacts: [artifact],
          preview,
          previewArtifactId: artifact.id
        })
      ).toMatchObject({
        targetId: "browser_sandbox",
        rendererId: id,
        supportState: "supported",
        surfaceKind: kind
      });
    }
  );

  it("keeps unsupported WebGPU code out of live renderer matching", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const artifact = creativeArtifact({
      previewEligible: true,
      summary: "WebGPU compute and render pipeline.",
      title: "feedback-field.webgpu.ts"
    });
    const preview = creativePreviewSummary(artifact, snapshot.preview);

    expect(matchCreativePreviewRenderer(artifact)).toBeNull();
    expect(
      buildPreviewRendererRoute({
        artifacts: [artifact],
        preview,
        previewArtifactId: artifact.id
      })
    ).toMatchObject({
      supportState: "unsupported",
      supportReason:
        "Current browser preview foundations cover p5.js, Three.js, GLSL, Hydra, Tone.js, GSAP, SVG, and Canvas only.",
      surfaceKind: "unsupported"
    });
  });

  it("keeps React Three Fiber components as explicit code exports", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const artifact: ArtifactSummary = {
      ...creativeArtifact({
        content: [
          'import { Canvas, useFrame } from "@react-three/fiber";',
          "export default function Study() { return <Canvas><mesh /></Canvas>; }"
        ].join("\n"),
        domain: "react_three_fiber",
        language: "TypeScript + React Three Fiber",
        title: "kinetic-study.r3f.tsx"
      }),
      actions: ["Open", "Copy", "Download"],
      previewEligible: false,
      previewTarget: "",
      rendererId: null,
      runtime: null
    };
    const preview = {
      ...snapshot.preview,
      active: false,
      artifactName: artifact.title,
      available: false,
      sourceArtifactId: artifact.id,
      sourceArtifactName: artifact.title,
      state: "unavailable" as const,
      status: "Preview unavailable"
    };

    expect(matchCreativePreviewRenderer(artifact)).toBeNull();
    expect(
      buildPreviewRendererRoute({
        artifacts: [artifact],
        preview,
        previewArtifactId: artifact.id
      })
    ).toMatchObject({
      rendererId: null,
      supportLabel: "Code/export-only",
      supportState: "unavailable",
      surfaceTitle: "React Three Fiber export"
    });
  });

  it("keeps HTML documents out of the p5 live renderer", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const artifact = creativeArtifact({
      content: [
        "<!doctype html>",
        "<html>",
        "<head><script src=\"https://cdn.jsdelivr.net/npm/p5/lib/p5.min.js\"></script></head>",
        "<body><script>function setup() { createCanvas(200, 200); }</script></body>",
        "</html>"
      ].join("\n"),
      domain: "p5_js",
      previewEligible: true,
      previewTarget: "browser_sandbox",
      runtime: "p5",
      summary: "A browser document that embeds a p5 sketch.",
      title: "generated-sketch-1.p5.ts"
    });
    const preview = creativePreviewSummary(artifact, snapshot.preview);

    expect(matchCreativePreviewRenderer(artifact)).toBeNull();
    expect(
      buildPreviewRendererRoute({
        artifacts: [artifact],
        preview,
        previewArtifactId: artifact.id
      })
    ).toMatchObject({
      supportState: "unsupported",
      supportReason:
        "HTML documents cannot run in the p5 JavaScript preview runtime. Use JavaScript p5 source with setup() or draw(), or route the artifact to a compatible preview surface.",
      surfaceKind: "unsupported"
    });
  });

  it("keeps leading-comment HTML documents out of the p5 live renderer", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const artifact = creativeArtifact({
      content: [
        "  <!-- index.html -->",
        "<!doctype html>",
        "<html>",
        "<head><script src=\"https://cdn.jsdelivr.net/npm/p5/lib/p5.min.js\"></script></head>",
        "<body><script>function setup() { createCanvas(200, 200); }</script></body>",
        "</html>"
      ].join("\n"),
      domain: "p5_js",
      previewEligible: true,
      previewTarget: "browser_sandbox",
      runtime: "p5",
      summary: "A browser document that embeds a p5 sketch.",
      title: "generated-sketch-1.p5.ts"
    });
    const preview = creativePreviewSummary(artifact, snapshot.preview);

    expect(matchCreativePreviewRenderer(artifact)).toBeNull();
    expect(
      buildPreviewRendererRoute({
        artifacts: [artifact],
        preview,
        previewArtifactId: artifact.id
      })
    ).toMatchObject({
      supportState: "unsupported",
      supportReason:
        "HTML documents cannot run in the p5 JavaScript preview runtime. Use JavaScript p5 source with setup() or draw(), or route the artifact to a compatible preview surface.",
      surfaceKind: "unsupported"
    });
  });

  it("falls back safely when GSAP source exceeds the bounded sandbox rules", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const artifact = creativeArtifact({
      content:
        "gsap.registerPlugin(ScrollTrigger);\n" +
        "gsap.to(document.body, { opacity: 0.5, scrollTrigger: '.hero' });",
      domain: "gsap",
      previewEligible: true,
      previewTarget: "browser_sandbox",
      runtime: "gsap",
      title: "unsafe-scroll.gsap.js"
    });
    const preview = creativePreviewSummary(artifact, snapshot.preview);

    expect(matchCreativePreviewRenderer(artifact)).toBeNull();
    expect(
      buildPreviewRendererRoute({
        artifacts: [artifact],
        preview,
        previewArtifactId: artifact.id
      })
    ).toMatchObject({
      supportState: "unsupported",
      supportReason: "GSAP previews can only target the bounded sandbox stage.",
      surfaceKind: "unsupported"
    });
  });

  it("falls back safely when SVG source exceeds the bounded sandbox rules", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const artifact = creativeArtifact({
      content:
        '<svg viewBox="0 0 100 100"><foreignObject width="100" height="100"><div>unsafe</div></foreignObject></svg>',
      language: "SVG",
      previewEligible: true,
      previewTarget: "browser_sandbox",
      runtime: "svg",
      title: "unsafe-markup.svg"
    });
    const preview = creativePreviewSummary(artifact, snapshot.preview);

    expect(matchCreativePreviewRenderer(artifact)).toBeNull();
    expect(
      buildPreviewRendererRoute({
        artifacts: [artifact],
        preview,
        previewArtifactId: artifact.id
      })
    ).toMatchObject({
      supportState: "unsupported",
      supportReason:
        "SVG preview support is limited to sanitized inline SVG markup without scriptable DOM containers.",
      surfaceKind: "unsupported"
    });
  });

  it("falls back safely when Canvas source exceeds the bounded sandbox rules", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const artifact = creativeArtifact({
      content: [
        "const canvas = document.querySelector('canvas');",
        "const ctx = canvas.getContext('2d');",
        "const image = new Image();",
        "image.src = 'https://example.com/remote.png';",
        "ctx.drawImage(image, 0, 0);"
      ].join("\n"),
      domain: "canvas_2d",
      previewEligible: true,
      previewTarget: "browser_sandbox",
      runtime: "canvas",
      title: "unsafe-grid.canvas.js"
    });
    const preview = creativePreviewSummary(artifact, snapshot.preview);

    expect(matchCreativePreviewRenderer(artifact)).toBeNull();
    expect(
      buildPreviewRendererRoute({
        artifacts: [artifact],
        preview,
        previewArtifactId: artifact.id
      })
    ).toMatchObject({
      supportState: "unsupported",
      supportReason:
        "Canvas preview support is limited to direct 2D drawing without image assets or auxiliary canvases.",
      surfaceKind: "unsupported"
    });
  });
});

function creativeArtifact(
  overrides: Partial<ArtifactSummary>
): ArtifactSummary {
  const snapshot = getLocalWorkspaceSnapshot();

  return {
    ...snapshot.artifacts[0],
    ...overrides,
    actions: ["Open", "Preview", "Copy", "Download"]
  };
}

function creativePreviewSummary(
  artifact: ArtifactSummary,
  fallback: PreviewSummary
): PreviewSummary {
  return {
    ...fallback,
    active: true,
    artifactName: artifact.title,
    sourceArtifactId: artifact.id,
    sourceArtifactName: artifact.title,
    state: "ready",
    status: "Ready when opened",
    target: "Browser preview",
    targetId: "browser_sandbox"
  };
}
