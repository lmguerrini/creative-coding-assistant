import type { CodeSummary, PreviewSummary } from "./assistant-client";
import type { PreviewRendererRoute } from "./preview-renderers";
import {
  createWorkstationError,
  type WorkstationError
} from "./workstation-errors";

export type PreviewExecutableRuntimeKind = "p5" | "glsl";

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

type MountPreviewRuntimeInput = {
  canvas: HTMLCanvasElement;
  kind: PreviewExecutableRuntimeKind;
  onStatus: (status: PreviewRuntimeStatus) => void;
  source: PreviewRuntimeSource;
};

type P5SketchSignals = {
  background: [number, number, number];
  fill: [number, number, number];
  radius: number;
  signal: number;
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

export function getExecutablePreviewRuntimeKind(
  route: PreviewRendererRoute
): PreviewExecutableRuntimeKind | null {
  if (route.supportState !== "supported") {
    return null;
  }

  switch (route.surfaceKind) {
    case "p5":
    case "glsl":
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
        ? "Preparing a bounded WebGL fragment shader runtime."
        : "Preparing a constrained canvas sketch runtime.",
    label: "Runtime starting",
    state: "starting",
    error: null
  };
}

export function mountPreviewRuntime({
  canvas,
  kind,
  onStatus,
  source
}: MountPreviewRuntimeInput): PreviewRuntimeMount {
  switch (kind) {
    case "p5":
      return mountP5Runtime({ canvas, onStatus, source });
    case "glsl":
      return mountGlslRuntime({ canvas, onStatus, source });
  }
}

function mountP5Runtime({
  canvas,
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

function mountGlslRuntime({
  canvas,
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
  const subsystem = kind === "glsl" ? "glsl_renderer" : "p5_renderer";

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

function getWebGlContext(canvas: HTMLCanvasElement) {
  try {
    return (
      canvas.getContext("webgl", {
        alpha: false,
        antialias: false,
        depth: false,
        preserveDrawingBuffer: false,
        stencil: false
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

function toRgb([red, green, blue]: [number, number, number]) {
  return `rgb(${red}, ${green}, ${blue})`;
}

function toRgba([red, green, blue]: [number, number, number], alpha: number) {
  return `rgba(${red}, ${green}, ${blue}, ${alpha.toFixed(3)})`;
}
