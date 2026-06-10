import type { CodeSummary, PreviewSummary } from "./assistant-client";
import type { PreviewRendererRoute } from "./preview-renderers";
import { parseHydraRuntimeSource } from "./hydra-runtime";
import {
  createWorkstationError,
  type WorkstationError
} from "./workstation-errors";

export type PreviewExecutableRuntimeKind = "p5" | "three" | "glsl" | "hydra";

export type PreviewRuntimeLifecycleState =
  | "idle"
  | "starting"
  | "running"
  | "error";

export type PreviewRuntimeStatus = {
  state: PreviewRuntimeLifecycleState;
  label: string;
  detail: string;
  error: WorkstationError | null;
  diagnostics?: readonly string[];
};

export type PreviewRuntimeSource = {
  title: string;
  source: string;
  lineCount: number;
  fingerprint: string;
};

export type PreviewRuntimeMount = {
  dispose: () => void;
};

export type PreviewRuntimeFrameSample = {
  renderedAtMs: number;
};

type MountPreviewRuntimeInput = {
  canvas: HTMLCanvasElement;
  kind: PreviewExecutableRuntimeKind;
  onFrame?: ((sample: PreviewRuntimeFrameSample) => void) | undefined;
  onStatus: (status: PreviewRuntimeStatus) => void;
  source: PreviewRuntimeSource;
};

type P5SketchSignals = {
  background: [number, number, number];
  fill: [number, number, number];
  radius: number;
  signal: number;
};

type ThreeScenePrimitive = "box" | "sphere" | "torus";

type ThreeSceneSignals = {
  background: [number, number, number];
  color: [number, number, number];
  accent: [number, number, number];
  primitive: ThreeScenePrimitive;
  spin: number;
};

const defaultRuntimeSource: PreviewRuntimeSource = {
  fingerprint: "empty",
  lineCount: 0,
  source: "",
  title: "Untitled preview source"
};

const vertexShaderSource = `
attribute vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const threeVertexShaderSource = `
attribute vec3 a_position;
attribute vec3 a_normal;
uniform float u_aspect;
uniform float u_time;
varying vec3 v_normal;
varying float v_depth;
void main() {
  float spin = u_time * 0.55;
  float tilt = 0.62 + sin(u_time * 0.24) * 0.08;
  mat3 rotateY = mat3(
    cos(spin), 0.0, -sin(spin),
    0.0, 1.0, 0.0,
    sin(spin), 0.0, cos(spin)
  );
  mat3 rotateX = mat3(
    1.0, 0.0, 0.0,
    0.0, cos(tilt), sin(tilt),
    0.0, -sin(tilt), cos(tilt)
  );
  vec3 position = rotateY * rotateX * a_position;
  vec3 normal = normalize(rotateY * rotateX * a_normal);
  float cameraDepth = position.z + 3.6;
  vec2 projected = position.xy / max(cameraDepth, 0.1);
  projected.x /= max(u_aspect, 0.1);
  gl_Position = vec4(projected * 2.45, (cameraDepth - 3.6) / 4.0, 1.0);
  v_normal = normal;
  v_depth = cameraDepth;
}
`;

const threeFragmentShaderSource = `
precision mediump float;
uniform vec3 u_accent;
uniform vec3 u_color;
varying vec3 v_normal;
varying float v_depth;
void main() {
  vec3 normal = normalize(v_normal);
  float key = max(dot(normal, normalize(vec3(-0.35, 0.72, 0.58))), 0.0);
  float fill = max(dot(normal, normalize(vec3(0.55, -0.2, 0.7))), 0.0) * 0.32;
  float rim = pow(1.0 - max(normal.z, 0.0), 2.0) * 0.44;
  float depthShade = smoothstep(2.2, 4.7, v_depth);
  vec3 color = u_color * (0.2 + key * 0.72 + fill) + u_accent * rim;
  gl_FragColor = vec4(mix(color, color * 0.58, depthShade * 0.34), 1.0);
}
`;

export function getExecutablePreviewRuntimeKind(
  route: PreviewRendererRoute
): PreviewExecutableRuntimeKind | null {
  if (route.supportState !== "supported") {
    return null;
  }

  switch (route.surfaceKind) {
    case "p5":
    case "three":
    case "glsl":
    case "hydra":
      return route.surfaceKind;
    default:
      return null;
  }
}

export function canRunPreviewRuntime({
  preview,
  route
}: {
  preview: PreviewSummary;
  route: PreviewRendererRoute;
}) {
  return (
    preview.active &&
    preview.state !== "error" &&
    preview.state !== "unavailable" &&
    getExecutablePreviewRuntimeKind(route) !== null
  );
}

export function buildPreviewRuntimeSource({
  code,
  route
}: {
  code: CodeSummary;
  route: PreviewRendererRoute;
}): PreviewRuntimeSource {
  if (
    route.sourceArtifactName !== code.title &&
    route.selectedArtifactName !== code.title
  ) {
    return defaultRuntimeSource;
  }

  const source = code.excerpt.join("\n");

  return {
    fingerprint: hashRuntimeSource(source),
    lineCount: code.excerpt.length,
    source,
    title: code.title
  };
}

export function getInitialPreviewRuntimeStatus({
  kind,
  preview
}: {
  kind: PreviewExecutableRuntimeKind;
  preview: PreviewSummary;
}): PreviewRuntimeStatus {
  if (preview.state === "error") {
    return {
      detail: "The preview runtime is waiting for the next successful preview state.",
      label: "Runtime stopped",
      state: "error",
      error:
        preview.error ??
        createRendererRuntimeError({
          kind,
          message: "The preview runtime is waiting for the next successful preview state.",
          type: "preview_runtime_stopped"
        })
    };
  }

  if (preview.state === "unavailable") {
    return {
      detail: "This preview route is not currently executable.",
      label: "Runtime unavailable",
      state: "idle",
      error: null
    };
  }

  return {
    detail:
      kind === "glsl"
        ? "Preparing a controlled WebGL fragment shader runtime."
        : kind === "three"
          ? "Preparing a controlled Three.js-compatible browser runtime."
          : kind === "hydra"
            ? "Preparing a controlled Hydra-compatible browser runtime."
            : "Preparing a controlled p5.js-compatible browser runtime.",
    label: "Runtime starting",
    state: "starting",
    error: null
  };
}

export function mountPreviewRuntime({
  canvas,
  kind,
  onFrame,
  onStatus,
  source
}: MountPreviewRuntimeInput): PreviewRuntimeMount {
  switch (kind) {
    case "p5":
      return mountP5Runtime({ canvas, onFrame, onStatus, source });
    case "three":
      return mountThreeRuntime({ canvas, onFrame, onStatus, source });
    case "glsl":
      return mountGlslRuntime({ canvas, onFrame, onStatus, source });
    case "hydra":
      return mountHydraRuntime({ canvas, onFrame, onStatus, source });
  }
}

function mountHydraRuntime({
  canvas,
  onFrame,
  onStatus,
  source
}: Omit<MountPreviewRuntimeInput, "kind">): PreviewRuntimeMount {
  const context = getCanvas2DContext(canvas);
  const parsed = parseHydraRuntimeSource(source.source);

  if (!context || !parsed.ok) {
    const message = !context
      ? "Canvas 2D is unavailable, so the Hydra runtime cannot mount here."
      : !parsed.ok
        ? parsed.message
        : "Hydra runtime could not mount.";
    onStatus({
      detail: message,
      label: "Hydra runtime unavailable",
      state: "error",
      error: createRendererRuntimeError({
        kind: "hydra",
        message,
        type: context ? "hydra_source_rejected" : "canvas_2d_unavailable"
      })
    });
    return { dispose: () => undefined };
  }

  const runtimeWindow = canvas.ownerDocument.defaultView;
  const context2d = context;
  const program = parsed.program;
  const activeChain = program.outputs[program.renderTarget];
  const sourceCall = activeChain?.source;
  const frequency = readHydraNumber(sourceCall?.args[0], 8);
  const sync = readHydraNumber(sourceCall?.args[1], 0.1);
  const offset = readHydraNumber(sourceCall?.args[2], 1.2);
  let animationFrame = 0;
  let disposed = false;

  onStatus({
    detail: `Rendering ${source.title} through a bounded Hydra synth adapter.`,
    label: "Hydra runtime running",
    state: "running",
    error: null
  });

  function drawFrame(time: number) {
    if (disposed) {
      return;
    }
    const { height, pixelRatio, width } = resizeCanvasToDisplaySize(canvas);
    const columns = Math.min(96, Math.max(32, Math.round(width / 8)));
    const rows = Math.min(64, Math.max(20, Math.round(height / 8)));
    const cellWidth = width / columns;
    const cellHeight = height / rows;
    const runtimeTime = time * 0.001 * program.speed;

    context2d.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context2d.clearRect(0, 0, width, height);
    for (let y = 0; y < rows; y += 1) {
      for (let x = 0; x < columns; x += 1) {
        const u = x / columns;
        const v = y / rows;
        const wave =
          0.5 +
          0.5 *
            Math.sin(
              (u * frequency + v * frequency * 0.58 + runtimeTime * sync) *
                Math.PI *
                2
            );
        const fold = 0.5 + 0.5 * Math.cos((u - v + runtimeTime * 0.08) * Math.PI * 6);
        const red = clampColor((wave * 0.72 + fold * 0.28) * 255);
        const green = clampColor((fold * 0.54 + offset * 0.08) * 255);
        const blue = clampColor((1 - wave * 0.68 + fold * 0.18) * 255);
        context2d.fillStyle = `rgb(${red}, ${green}, ${blue})`;
        context2d.fillRect(
          x * cellWidth,
          y * cellHeight,
          cellWidth + 1,
          cellHeight + 1
        );
      }
    }

    onFrame?.({ renderedAtMs: time });
    animationFrame = requestRuntimeFrame(runtimeWindow, drawFrame);
  }

  animationFrame = requestRuntimeFrame(runtimeWindow, drawFrame);

  return {
    dispose: () => {
      disposed = true;
      cancelRuntimeFrame(runtimeWindow, animationFrame);
    }
  };
}

function mountP5Runtime({
  canvas,
  onFrame,
  onStatus,
  source
}: Omit<MountPreviewRuntimeInput, "kind">): PreviewRuntimeMount {
  const context = getCanvas2DContext(canvas);

  if (!context) {
    onStatus({
      detail: "Canvas 2D is unavailable, so the p5 runtime cannot mount here.",
      label: "p5 runtime unavailable",
      state: "error",
      error: createRendererRuntimeError({
        kind: "p5",
        message: "Canvas 2D is unavailable, so the p5 runtime cannot mount here.",
        type: "canvas_2d_unavailable"
      })
    });
    return { dispose: () => undefined };
  }

  const signals = parseP5SketchSignals(source);
  const context2d = context;
  const runtimeWindow = canvas.ownerDocument.defaultView;
  let animationFrame = 0;
  let disposed = false;

  onStatus({
    detail: `Rendering ${source.title} through a constrained p5-style canvas adapter.`,
    label: "p5 runtime running",
    state: "running",
    error: null
  });

  function drawFrame(time: number) {
    if (disposed) {
      return;
    }

    const { height, pixelRatio, width } = resizeCanvasToDisplaySize(canvas);
    const centerX = width / 2;
    const centerY = height / 2;
    const pulse = (Math.sin(time * 0.002 + signals.signal) + 1) / 2;
    const orbit = Math.min(width, height) * (0.18 + pulse * 0.04);

    context2d.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context2d.clearRect(0, 0, width, height);
    context2d.fillStyle = toRgb(signals.background);
    context2d.fillRect(0, 0, width, height);

    for (let index = 0; index < 18; index += 1) {
      const angle = time * 0.0007 + signals.signal + index * 0.72;
      const x = centerX + Math.cos(angle) * orbit * (0.55 + (index % 4) * 0.13);
      const y = centerY + Math.sin(angle * 1.17) * orbit;
      const radius = signals.radius * (0.16 + (index % 5) * 0.025 + pulse * 0.04);

      context2d.globalAlpha = 0.18 + (index % 4) * 0.055;
      context2d.fillStyle = toRgba(signals.fill, 0.42 + pulse * 0.28);
      context2d.beginPath();
      context2d.arc(x, y, radius, 0, Math.PI * 2);
      context2d.fill();
    }

    context2d.globalAlpha = 0.9;
    context2d.strokeStyle = toRgba(signals.fill, 0.64);
    context2d.lineWidth = 1.4;
    context2d.beginPath();
    for (let index = 0; index <= 120; index += 1) {
      const phase = index / 120;
      const wave = Math.sin(phase * Math.PI * 6 + time * 0.0014 + signals.signal);
      const x = phase * width;
      const y = centerY + wave * orbit * 0.28;

      if (index === 0) {
        context2d.moveTo(x, y);
      } else {
        context2d.lineTo(x, y);
      }
    }
    context2d.stroke();
    context2d.globalAlpha = 1;

    onFrame?.({ renderedAtMs: time });
    animationFrame = requestRuntimeFrame(runtimeWindow, drawFrame);
  }

  animationFrame = requestRuntimeFrame(runtimeWindow, drawFrame);

  return {
    dispose: () => {
      disposed = true;
      cancelRuntimeFrame(runtimeWindow, animationFrame);
    }
  };
}

function mountThreeRuntime({
  canvas,
  onFrame,
  onStatus,
  source
}: Omit<MountPreviewRuntimeInput, "kind">): PreviewRuntimeMount {
  const gl = getWebGlContext(canvas, { depth: true });

  if (!gl) {
    onStatus({
      detail: "WebGL is unavailable, so the Three.js-style runtime cannot mount here.",
      label: "Three.js runtime unavailable",
      state: "error",
      error: createRendererRuntimeError({
        kind: "three",
        message: "WebGL is unavailable, so the Three.js-style runtime cannot mount here.",
        type: "webgl_unavailable"
      })
    });
    return { dispose: () => undefined };
  }

  const signals = parseThreeSceneSignals(source);
  if (!signals.allowed) {
    onStatus({
      detail: signals.reason,
      diagnostics: [signals.reason],
      label: "Three.js runtime rejected source",
      state: "error",
      error: createRendererRuntimeError({
        kind: "three",
        message: signals.reason,
        type: "three_scene_source_rejected"
      })
    });
    return { dispose: () => undefined };
  }

  const sceneSignals = signals.signals;
  const webgl = gl;
  const program = createShaderProgram(
    webgl,
    threeVertexShaderSource,
    threeFragmentShaderSource
  );
  if (!program.ok) {
    onStatus({
      detail: program.message,
      diagnostics: [program.message],
      label: "Three.js runtime failed",
      state: "error",
      error: createRendererRuntimeError({
        kind: "three",
        message: "The controlled Three.js-style runtime could not compile its WebGL program.",
        type: "three_scene_program_failed",
        debugMessage: program.message
      })
    });
    return { dispose: () => undefined };
  }

  const scene = createThreeSceneGeometry(sceneSignals.primitive);
  const programObject = program.program;
  const positionLocation = webgl.getAttribLocation(programObject, "a_position");
  const normalLocation = webgl.getAttribLocation(programObject, "a_normal");
  const aspectLocation = webgl.getUniformLocation(programObject, "u_aspect");
  const timeLocation = webgl.getUniformLocation(programObject, "u_time");
  const colorLocation = webgl.getUniformLocation(programObject, "u_color");
  const accentLocation = webgl.getUniformLocation(programObject, "u_accent");
  const positionBuffer = webgl.createBuffer();
  const normalBuffer = webgl.createBuffer();
  const runtimeWindow = canvas.ownerDocument.defaultView;
  let animationFrame = 0;
  let disposed = false;

  if (
    !positionBuffer ||
    !normalBuffer ||
    positionLocation < 0 ||
    normalLocation < 0
  ) {
    onStatus({
      detail: "The Three.js-style runtime could not allocate its scene buffers.",
      label: "Three.js runtime failed",
      state: "error",
      error: createRendererRuntimeError({
        kind: "three",
        message: "The controlled Three.js-style runtime could not allocate its scene buffers.",
        type: "three_scene_buffer_unavailable"
      })
    });
    return { dispose: () => undefined };
  }

  webgl.bindBuffer(webgl.ARRAY_BUFFER, positionBuffer);
  webgl.bufferData(webgl.ARRAY_BUFFER, scene.positions, webgl.STATIC_DRAW);
  webgl.bindBuffer(webgl.ARRAY_BUFFER, normalBuffer);
  webgl.bufferData(webgl.ARRAY_BUFFER, scene.normals, webgl.STATIC_DRAW);
  webgl.enable(webgl.DEPTH_TEST);
  webgl.enable(webgl.CULL_FACE);

  onStatus({
    detail: `Rendering ${source.title} as a controlled ${sceneSignals.primitive} scene without evaluating generated JavaScript.`,
    label: "Three.js runtime running",
    state: "running",
    error: null
  });

  function drawFrame(time: number) {
    if (disposed) {
      return;
    }

    const { height, width } = resizeCanvasToDisplaySize(canvas);
    const [red, green, blue] = sceneSignals.background.map(
      (channel) => channel / 255
    );

    webgl.viewport(0, 0, webgl.drawingBufferWidth, webgl.drawingBufferHeight);
    webgl.clearColor(red, green, blue, 1);
    webgl.clear(webgl.COLOR_BUFFER_BIT | webgl.DEPTH_BUFFER_BIT);
    webgl.useProgram(programObject);
    webgl.uniform1f(aspectLocation, width / Math.max(height, 1));
    webgl.uniform1f(timeLocation, time * 0.001 * sceneSignals.spin);
    webgl.uniform3f(colorLocation, ...toUnitRgb(sceneSignals.color));
    webgl.uniform3f(accentLocation, ...toUnitRgb(sceneSignals.accent));

    webgl.enableVertexAttribArray(positionLocation);
    webgl.bindBuffer(webgl.ARRAY_BUFFER, positionBuffer);
    webgl.vertexAttribPointer(positionLocation, 3, webgl.FLOAT, false, 0, 0);

    webgl.enableVertexAttribArray(normalLocation);
    webgl.bindBuffer(webgl.ARRAY_BUFFER, normalBuffer);
    webgl.vertexAttribPointer(normalLocation, 3, webgl.FLOAT, false, 0, 0);

    webgl.drawArrays(webgl.TRIANGLES, 0, scene.vertexCount);
    onFrame?.({ renderedAtMs: time });
    animationFrame = requestRuntimeFrame(runtimeWindow, drawFrame);
  }

  animationFrame = requestRuntimeFrame(runtimeWindow, drawFrame);

  return {
    dispose: () => {
      disposed = true;
      cancelRuntimeFrame(runtimeWindow, animationFrame);
      webgl.deleteBuffer(positionBuffer);
      webgl.deleteBuffer(normalBuffer);
      webgl.deleteProgram(programObject);
    }
  };
}

function mountGlslRuntime({
  canvas,
  onFrame,
  onStatus,
  source
}: Omit<MountPreviewRuntimeInput, "kind">): PreviewRuntimeMount {
  const gl = getWebGlContext(canvas);

  if (!gl) {
    onStatus({
      detail: "WebGL is unavailable, so the fragment shader runtime cannot mount here.",
      label: "GLSL runtime unavailable",
      state: "error",
      error: createRendererRuntimeError({
        kind: "glsl",
        message: "WebGL is unavailable, so the fragment shader runtime cannot mount here.",
        type: "webgl_unavailable"
      })
    });
    return { dispose: () => undefined };
  }

  const normalizedFragment = buildFragmentShaderSource(source);
  if (!normalizedFragment.allowed) {
    onStatus({
      detail: normalizedFragment.reason,
      diagnostics: [normalizedFragment.reason],
      label: "GLSL runtime rejected source",
      state: "error",
      error: createRendererRuntimeError({
        kind: "glsl",
        message: normalizedFragment.reason,
        type: "shader_source_rejected"
      })
    });
    return { dispose: () => undefined };
  }

  const webgl = gl;
  const program = createShaderProgram(webgl, vertexShaderSource, normalizedFragment.source);
  if (!program.ok) {
    onStatus({
      detail: program.message,
      diagnostics: [program.message],
      label: "GLSL runtime failed",
      state: "error",
      error: createRendererRuntimeError({
        kind: "glsl",
        message: "The bounded GLSL runtime could not compile the current shader.",
        type: "shader_program_failed",
        debugMessage: program.message
      })
    });
    return { dispose: () => undefined };
  }

  const programObject = program.program;
  const positionLocation = webgl.getAttribLocation(programObject, "a_position");
  const resolutionLocation = webgl.getUniformLocation(programObject, "u_resolution");
  const timeLocation = webgl.getUniformLocation(programObject, "u_time");
  const buffer = webgl.createBuffer();
  const runtimeWindow = canvas.ownerDocument.defaultView;
  let animationFrame = 0;
  let disposed = false;

  if (!buffer || positionLocation < 0) {
    onStatus({
      detail: "The shader runtime could not allocate its fullscreen triangle.",
      label: "GLSL runtime failed",
      state: "error",
      error: createRendererRuntimeError({
        kind: "glsl",
        message: "The bounded GLSL runtime could not allocate its fullscreen surface.",
        type: "fullscreen_triangle_unavailable"
      })
    });
    return { dispose: () => undefined };
  }

  webgl.bindBuffer(webgl.ARRAY_BUFFER, buffer);
  webgl.bufferData(
    webgl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 3, -1, -1, 3]),
    webgl.STATIC_DRAW
  );

  onStatus({
    detail: `Rendering ${source.title} through a bounded WebGL fragment adapter.`,
    label: "GLSL runtime running",
    state: "running",
    error: null
  });

  function drawFrame(time: number) {
    if (disposed) {
      return;
    }

    const { height, width } = resizeCanvasToDisplaySize(canvas);
    webgl.viewport(0, 0, webgl.drawingBufferWidth, webgl.drawingBufferHeight);
    webgl.useProgram(programObject);
    webgl.enableVertexAttribArray(positionLocation);
    webgl.bindBuffer(webgl.ARRAY_BUFFER, buffer);
    webgl.vertexAttribPointer(positionLocation, 2, webgl.FLOAT, false, 0, 0);
    webgl.uniform2f(resolutionLocation, width, height);
    webgl.uniform1f(timeLocation, time * 0.001);
    webgl.drawArrays(webgl.TRIANGLES, 0, 3);

    onFrame?.({ renderedAtMs: time });
    animationFrame = requestRuntimeFrame(runtimeWindow, drawFrame);
  }

  animationFrame = requestRuntimeFrame(runtimeWindow, drawFrame);

  return {
    dispose: () => {
      disposed = true;
      cancelRuntimeFrame(runtimeWindow, animationFrame);
      webgl.deleteBuffer(buffer);
      webgl.deleteProgram(programObject);
    }
  };
}

function parseP5SketchSignals(source: PreviewRuntimeSource): P5SketchSignals {
  return {
    background: parseRgbCall(source.source, "background") ?? [8, 12, 18],
    fill: parseRgbCall(source.source, "fill") ?? [76, 215, 200],
    radius: parseFirstNumber(source.source, /circle\s*\([^)]*,\s*([0-9.]+)\s*\)/i) ?? 96,
    signal: source.fingerprint
      .split("")
      .reduce((total, character) => total + character.charCodeAt(0), 0)
  };
}

function parseThreeSceneSignals(
  source: PreviewRuntimeSource
): { allowed: true; signals: ThreeSceneSignals } | { allowed: false; reason: string } {
  const sourceText = source.source.trim();

  if (sourceText.length > 9000) {
    return {
      allowed: false,
      reason: "The Three.js scene source is too large for this lightweight runtime."
    };
  }

  if (
    /\b(?:eval|Function|document|window|fetch|XMLHttpRequest|WebSocket|Worker|localStorage|sessionStorage)\b/i.test(
      sourceText
    ) ||
    /\bimport\s*\(/i.test(sourceText)
  ) {
    return {
      allowed: false,
      reason:
        "The Three.js scene uses browser or dynamic execution features outside the controlled preview subset."
    };
  }

  return {
    allowed: true,
    signals: {
      background:
        parseThreeHexColor(sourceText, /setClearColor\s*\(\s*(0x[0-9a-f]{3,6}|#[0-9a-f]{3,6})/i) ??
        parseThreeHexColor(sourceText, /background\s*=\s*new\s+THREE\.Color\s*\(\s*(0x[0-9a-f]{3,6}|#[0-9a-f]{3,6})/i) ??
        [5, 8, 11],
      color:
        parseThreeHexColor(sourceText, /color\s*:\s*(0x[0-9a-f]{3,6}|#[0-9a-f]{3,6})/i) ??
        parseThreeHexColor(sourceText, /Mesh(?:Standard|Phong|Basic)Material\s*\([^)]*(0x[0-9a-f]{3,6}|#[0-9a-f]{3,6})/i) ??
        [76, 215, 200],
      accent:
        parseThreeHexColor(sourceText, /emissive\s*:\s*(0x[0-9a-f]{3,6}|#[0-9a-f]{3,6})/i) ??
        [124, 167, 255],
      primitive: parseThreePrimitive(sourceText),
      spin: Math.max(
        0.35,
        Math.min(
          parseFirstNumber(sourceText, /rotation\.[xyza-z]+\s*\+=\s*([0-9.]+)/i) ??
            parseFirstNumber(sourceText, /rotate[XYZ]\s*\(\s*([0-9.]+)/i) ??
            1,
          2.4
        )
      )
    }
  };
}

function parseThreePrimitive(source: string): ThreeScenePrimitive {
  if (/Torus(?:Knot)?Geometry/i.test(source)) {
    return "torus";
  }

  if (/SphereGeometry|IcosahedronGeometry|DodecahedronGeometry/i.test(source)) {
    return "sphere";
  }

  return "box";
}

function parseThreeHexColor(source: string, pattern: RegExp) {
  const match = source.match(pattern);
  if (!match) {
    return null;
  }

  const rawValue = match[1].replace(/^#/, "0x");
  const parsed = Number.parseInt(rawValue.replace(/^0x/i, ""), 16);
  if (!Number.isFinite(parsed)) {
    return null;
  }

  if (rawValue.length === 5) {
    const red = (parsed >> 8) & 0xf;
    const green = (parsed >> 4) & 0xf;
    const blue = parsed & 0xf;
    return [red * 17, green * 17, blue * 17] as [number, number, number];
  }

  return [
    (parsed >> 16) & 255,
    (parsed >> 8) & 255,
    parsed & 255
  ] as [number, number, number];
}

function createThreeSceneGeometry(primitive: ThreeScenePrimitive) {
  switch (primitive) {
    case "sphere":
      return createOctahedronGeometry();
    case "torus":
      return createTorusLikeGeometry();
    case "box":
    default:
      return createCubeGeometry();
  }
}

function createCubeGeometry() {
  const faces = [
    { normal: [0, 0, 1], corners: [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]] },
    { normal: [0, 0, -1], corners: [[1, -1, -1], [-1, -1, -1], [-1, 1, -1], [1, 1, -1]] },
    { normal: [1, 0, 0], corners: [[1, -1, 1], [1, -1, -1], [1, 1, -1], [1, 1, 1]] },
    { normal: [-1, 0, 0], corners: [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]] },
    { normal: [0, 1, 0], corners: [[-1, 1, 1], [1, 1, 1], [1, 1, -1], [-1, 1, -1]] },
    { normal: [0, -1, 0], corners: [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]] }
  ] as const;
  const positions: number[] = [];
  const normals: number[] = [];

  for (const face of faces) {
    const [a, b, c, d] = face.corners;
    for (const vertex of [a, b, c, a, c, d]) {
      positions.push(...vertex.map((value) => value * 0.74));
      normals.push(...face.normal);
    }
  }

  return toThreeGeometry(positions, normals);
}

function createOctahedronGeometry() {
  const triangles = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
    [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
    [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
    [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
  ] as const;
  const positions: number[] = [];
  const normals: number[] = [];

  for (const triangle of triangles) {
    for (const vertex of triangle) {
      positions.push(...vertex.map((value) => value * 0.92));
      normals.push(...normalizeVec3(vertex));
    }
  }

  return toThreeGeometry(positions, normals);
}

function createTorusLikeGeometry() {
  const segments = 18;
  const positions: number[] = [];
  const normals: number[] = [];

  for (let index = 0; index < segments; index += 1) {
    const a0 = (index / segments) * Math.PI * 2;
    const a1 = ((index + 1) / segments) * Math.PI * 2;
    const inner0 = [Math.cos(a0) * 0.46, Math.sin(a0) * 0.46, -0.22];
    const outer0 = [Math.cos(a0) * 0.96, Math.sin(a0) * 0.96, 0.0];
    const inner1 = [Math.cos(a1) * 0.46, Math.sin(a1) * 0.46, -0.22];
    const outer1 = [Math.cos(a1) * 0.96, Math.sin(a1) * 0.96, 0.0];
    const cap0 = [Math.cos((a0 + a1) / 2) * 0.72, Math.sin((a0 + a1) / 2) * 0.72, 0.3];

    for (const vertex of [inner0, outer0, outer1, inner0, outer1, inner1, outer0, cap0, outer1]) {
      positions.push(...vertex);
      normals.push(...normalizeVec3(vertex));
    }
  }

  return toThreeGeometry(positions, normals);
}

function toThreeGeometry(positions: number[], normals: number[]) {
  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    vertexCount: positions.length / 3
  };
}

function buildFragmentShaderSource(
  source: PreviewRuntimeSource
): { allowed: true; source: string } | { allowed: false; reason: string } {
  const fragmentSource = source.source.trim();

  if (!fragmentSource) {
    return {
      allowed: true,
      source: createFallbackFragmentShader(source.fingerprint)
    };
  }

  if (fragmentSource.length > 6000) {
    return {
      allowed: false,
      reason: "The fragment shader is too large for this lightweight runtime."
    };
  }

  if (/\b(?:while|sampler2D|samplerCube|texture2D|textureCube|discard)\b/i.test(fragmentSource)) {
    return {
      allowed: false,
      reason:
        "The fragment shader uses features outside the current bounded runtime subset."
    };
  }

  if (/void\s+mainImage\s*\(/i.test(fragmentSource)) {
    return {
      allowed: true,
      source: `
precision mediump float;
uniform vec2 u_resolution;
uniform float u_time;
${fragmentSource}
void main() {
  vec4 previewColor = vec4(0.0);
  mainImage(previewColor, gl_FragCoord.xy);
  gl_FragColor = previewColor;
}
`
    };
  }

  if (/void\s+main\s*\(/i.test(fragmentSource)) {
    return {
      allowed: true,
      source: `
precision mediump float;
uniform vec2 u_resolution;
uniform float u_time;
${fragmentSource}
`
    };
  }

  return {
    allowed: true,
    source: createFallbackFragmentShader(source.fingerprint)
  };
}

function createFallbackFragmentShader(fingerprint: string) {
  const seed = fingerprint
    .split("")
    .reduce((total, character) => total + character.charCodeAt(0), 0);
  const hue = (seed % 360) / 360;

  return `
precision mediump float;
uniform vec2 u_resolution;
uniform float u_time;
void main() {
  vec2 uv = gl_FragCoord.xy / max(u_resolution, vec2(1.0));
  vec2 center = uv - 0.5;
  float field = sin((center.x * 10.0) + u_time * 1.4) * cos((center.y * 8.0) - u_time);
  float ring = smoothstep(0.42, 0.05, abs(length(center) - 0.22 - field * 0.035));
  vec3 base = 0.5 + 0.5 * cos(6.28318 * (vec3(${hue.toFixed(4)}, ${(
    hue + 0.23
  ).toFixed(4)}, ${(hue + 0.48).toFixed(4)}) + field + u_time * 0.18));
  gl_FragColor = vec4(base * (0.2 + ring), 1.0);
}
`;
}

function createRendererRuntimeError({
  debugMessage = null,
  kind,
  message,
  type
}: {
  debugMessage?: string | null;
  kind: PreviewExecutableRuntimeKind;
  message: string;
  type: string;
}) {
  const subsystem =
    kind === "glsl"
      ? "glsl_renderer"
      : kind === "three"
        ? "three_renderer"
        : kind === "hydra"
          ? "hydra_renderer"
          : "p5_renderer";

  return createWorkstationError({
    type,
    category: "renderer",
    subsystem,
    userMessage: message,
    debugMessage,
    recoverable: true,
    suggestedAction:
      "Reload the preview state or reset the preview session before trying again.",
    retryLabel: "Reload preview state",
    resetLabel: "Reset preview session"
  });
}

function createShaderProgram(
  gl: WebGLRenderingContext,
  vertexSource: string,
  fragmentSource: string
):
  | { ok: true; program: WebGLProgram }
  | { ok: false; message: string } {
  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  if (!vertexShader.ok) {
    return vertexShader;
  }

  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  if (!fragmentShader.ok) {
    gl.deleteShader(vertexShader.shader);
    return fragmentShader;
  }

  const program = gl.createProgram();
  if (!program) {
    gl.deleteShader(vertexShader.shader);
    gl.deleteShader(fragmentShader.shader);
    return { message: "The shader runtime could not create a WebGL program.", ok: false };
  }

  gl.attachShader(program, vertexShader.shader);
  gl.attachShader(program, fragmentShader.shader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader.shader);
  gl.deleteShader(fragmentShader.shader);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program) ?? "The shader program did not link.";
    gl.deleteProgram(program);
    return { message, ok: false };
  }

  return { ok: true, program };
}

function compileShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
):
  | { ok: true; shader: WebGLShader }
  | { ok: false; message: string } {
  const shader = gl.createShader(type);

  if (!shader) {
    return { message: "The shader runtime could not create a shader.", ok: false };
  }

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) ?? "The shader did not compile.";
    gl.deleteShader(shader);
    return { message, ok: false };
  }

  return { ok: true, shader };
}

function parseRgbCall(source: string, functionName: string): [number, number, number] | null {
  const match = source.match(new RegExp(`${functionName}\\\\s*\\\\(([^)]*)\\\\)`, "i"));
  if (!match) {
    return null;
  }

  const values = match[1]
    .split(",")
    .map((value) => Number.parseFloat(value.trim()))
    .filter((value) => Number.isFinite(value));

  if (values.length === 1) {
    const grey = clampColor(values[0]);
    return [grey, grey, grey];
  }

  if (values.length >= 3) {
    return [clampColor(values[0]), clampColor(values[1]), clampColor(values[2])];
  }

  return null;
}

function parseFirstNumber(source: string, pattern: RegExp) {
  const match = source.match(pattern);
  if (!match) {
    return null;
  }

  const value = Number.parseFloat(match[1]);
  return Number.isFinite(value) ? value : null;
}

function readHydraNumber(value: unknown, fallback: number) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function resizeCanvasToDisplaySize(canvas: HTMLCanvasElement) {
  const pixelRatio = Math.min(canvas.ownerDocument.defaultView?.devicePixelRatio ?? 1, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth || canvas.width || 640));
  const height = Math.max(1, Math.floor(canvas.clientHeight || canvas.height || 360));
  const nextWidth = Math.floor(width * pixelRatio);
  const nextHeight = Math.floor(height * pixelRatio);

  if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
    canvas.width = nextWidth;
    canvas.height = nextHeight;
  }

  return { height, pixelRatio, width };
}

function requestRuntimeFrame(
  runtimeWindow: Window | null,
  callback: FrameRequestCallback
) {
  if (runtimeWindow?.requestAnimationFrame) {
    return runtimeWindow.requestAnimationFrame(callback);
  }

  return globalThis.setTimeout(() => callback(Date.now()), 16) as unknown as number;
}

function cancelRuntimeFrame(runtimeWindow: Window | null, frame: number) {
  if (!frame) {
    return;
  }

  if (runtimeWindow?.cancelAnimationFrame) {
    runtimeWindow.cancelAnimationFrame(frame);
    return;
  }

  globalThis.clearTimeout(frame);
}

function getCanvas2DContext(canvas: HTMLCanvasElement) {
  try {
    return canvas.getContext("2d");
  } catch {
    return null;
  }
}

function getWebGlContext(
  canvas: HTMLCanvasElement,
  options: WebGLContextAttributes = {}
) {
  try {
    return (
      canvas.getContext("webgl", {
        alpha: false,
        antialias: false,
        depth: options.depth ?? false,
        preserveDrawingBuffer: false,
        stencil: false,
        ...options
      }) ?? null
    );
  } catch {
    return null;
  }
}

function hashRuntimeSource(source: string) {
  let hash = 2166136261;

  for (const character of source) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }

  return (hash >>> 0).toString(16);
}

function clampColor(value: number) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function normalizeVec3(vector: readonly number[]) {
  const length = Math.hypot(vector[0] ?? 0, vector[1] ?? 0, vector[2] ?? 0) || 1;

  return [
    (vector[0] ?? 0) / length,
    (vector[1] ?? 0) / length,
    (vector[2] ?? 0) / length
  ];
}

function toUnitRgb([red, green, blue]: [number, number, number]) {
  return [red / 255, green / 255, blue / 255] as const;
}

function toRgb([red, green, blue]: [number, number, number]) {
  return `rgb(${red}, ${green}, ${blue})`;
}

function toRgba([red, green, blue]: [number, number, number], alpha: number) {
  return `rgba(${red}, ${green}, ${blue}, ${alpha.toFixed(3)})`;
}
