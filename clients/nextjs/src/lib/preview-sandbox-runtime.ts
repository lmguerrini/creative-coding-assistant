import type {
  PreviewExecutableRuntimeKind,
  PreviewRuntimeFrameSample,
  PreviewRuntimeSource,
  PreviewRuntimeStatus
} from "./preview-runtime-adapters";
import {
  parseHydraRuntimeSource,
  prepareHydraRuntimeSource
} from "./hydra-runtime";
import { getGsapRuntimeSupportIssue } from "./gsap-runtime";
import {
  getCanvasRuntimeSupportIssue,
  getSvgRuntimeSupportIssue
} from "./svg-canvas-runtime";
import {
  parseToneRuntimeSource,
  prepareToneRuntimeSource
} from "./tone-runtime";
import {
  createWorkstationError,
  type WorkstationError
} from "./workstation-errors";
import {
  getGlslRuntimeSourceSupportIssue,
  getP5RuntimeSourceSupportIssue,
  getThreeRuntimeSourceSupportIssue,
  prepareP5JavaScriptSource
} from "./preview-source-classification";

export type PreviewSandboxRuntimeMessage =
  | {
      source: "cca-preview-runtime";
      runtimeId: string;
      type: "status";
      status: PreviewSandboxRuntimeStatusPayload;
    }
  | {
      source: "cca-preview-runtime";
      runtimeId: string;
      type: "frame";
      renderedAtMs: number;
    }
  | {
      source: "cca-preview-runtime";
      runtimeId: string;
      type: "keyboard-boundary";
      key: "Escape" | "Tab";
      shiftKey: boolean;
    };

export type PreviewSandboxKeyboardBoundaryEvent = Pick<
  Extract<PreviewSandboxRuntimeMessage, { type: "keyboard-boundary" }>,
  "key" | "shiftKey"
>;

export type PreviewSandboxRuntimeStatusPayload = {
  state: PreviewRuntimeStatus["state"];
  label: string;
  detail: string;
  diagnostics?: string[];
  error?: {
    message: string;
    debugMessage?: string;
    type?: string;
  } | null;
};

export type MountPreviewSandboxRuntimeInput = {
  captureHostKeyboard?: boolean;
  iframe: HTMLIFrameElement;
  kind: PreviewExecutableRuntimeKind;
  onFrame?: ((sample: PreviewRuntimeFrameSample) => void) | undefined;
  onKeyboardBoundary?:
    | ((event: PreviewSandboxKeyboardBoundaryEvent) => void)
    | undefined;
  onStatus: (status: PreviewRuntimeStatus) => void;
  runtimeId: string;
  showStatusOverlay?: boolean;
  source: PreviewRuntimeSource;
};

export type PreviewSandboxRuntimeMount = {
  control: (action: PreviewSandboxRuntimeControlAction) => void;
  dispose: () => void;
};

export type PreviewSandboxRuntimeControlAction =
  | "start"
  | "stop"
  | "mute"
  | "unmute";

const sandboxMessageSource = "cca-preview-runtime";

export function createPreviewSandboxRuntimeId() {
  return `preview-runtime-${Math.random().toString(36).slice(2, 10)}`;
}

export function mountPreviewSandboxRuntime({
  captureHostKeyboard = false,
  iframe,
  kind,
  onFrame,
  onKeyboardBoundary,
  onStatus,
  runtimeId,
  showStatusOverlay = true,
  source
}: MountPreviewSandboxRuntimeInput): PreviewSandboxRuntimeMount {
  const sourceMismatch = getPreviewRuntimeSourceMismatch({ kind, source });

  if (sourceMismatch) {
    onStatus(
      getSandboxSourceRejectedStatus({
        issue: sourceMismatch,
        kind,
        source
      })
    );
    return {
      control: () => undefined,
      dispose: () => undefined
    };
  }

  let disposed = false;
  let retryTimer = 0;
  const mountMessage = {
    source: sandboxMessageSource,
    runtimeId,
    type: "mount",
    runtime: {
      captureHostKeyboard,
      kind,
      runtimeId,
      showStatusOverlay,
      source: {
        ...source,
        source: preparePreviewExecutableSource(source.source, kind)
      }
    }
  };
  const disposeMessage = {
    source: sandboxMessageSource,
    runtimeId,
    type: "dispose"
  };

  function handleMessage(event: MessageEvent) {
    if (disposed) {
      return;
    }

    if (
      event.source &&
      iframe.contentWindow &&
      event.source !== iframe.contentWindow
    ) {
      return;
    }

    const message = readPreviewSandboxRuntimeMessage(event.data, runtimeId);
    if (!message) {
      return;
    }

    if (message.type === "frame") {
      onFrame?.({ renderedAtMs: message.renderedAtMs });
      return;
    }

    if (message.type === "keyboard-boundary") {
      onKeyboardBoundary?.({ key: message.key, shiftKey: message.shiftKey });
      return;
    }

    onStatus(toPreviewRuntimeStatus(kind, message.status));
  }

  function postMountMessage() {
    if (disposed) {
      return;
    }
    iframe.contentWindow?.postMessage(mountMessage, "*");
  }

  function postControlMessage(action: PreviewSandboxRuntimeControlAction) {
    if (disposed) {
      return;
    }
    iframe.contentWindow?.postMessage(
      {
        action,
        runtimeId,
        source: sandboxMessageSource,
        type: "control"
      },
      "*"
    );
  }

  iframe.dataset.runtimeId = runtimeId;
  window.addEventListener("message", handleMessage);
  iframe.addEventListener("load", postMountMessage);
  onStatus(getSandboxStartingStatus(kind));
  iframe.src = "/preview-sandbox.html";
  retryTimer = window.setTimeout(postMountMessage, 150);

  return {
    control: postControlMessage,
    dispose() {
      disposed = true;
      window.clearTimeout(retryTimer);
      window.removeEventListener("message", handleMessage);
      iframe.removeEventListener("load", postMountMessage);
      iframe.contentWindow?.postMessage(disposeMessage, "*");
      delete iframe.dataset.runtimeId;
      iframe.src = "about:blank";
      iframe.removeAttribute("srcdoc");
    }
  };
}

export function readPreviewSandboxRuntimeMessage(
  value: unknown,
  runtimeId: string
): PreviewSandboxRuntimeMessage | null {
  if (!isRecord(value) || value.source !== sandboxMessageSource) {
    return null;
  }

  if (value.runtimeId !== runtimeId) {
    return null;
  }

  if (value.type === "frame") {
    return typeof value.renderedAtMs === "number" &&
      Number.isFinite(value.renderedAtMs)
      ? {
          source: sandboxMessageSource,
          runtimeId,
          type: "frame",
          renderedAtMs: value.renderedAtMs
        }
      : null;
  }

  if (value.type === "keyboard-boundary") {
    return (value.key === "Escape" || value.key === "Tab") &&
      typeof value.shiftKey === "boolean"
      ? {
          source: sandboxMessageSource,
          runtimeId,
          type: "keyboard-boundary",
          key: value.key,
          shiftKey: value.shiftKey
        }
      : null;
  }

  if (value.type !== "status" || !isRecord(value.status)) {
    return null;
  }

  const status = value.status;
  const state = readRuntimeState(status.state);
  const label = readString(status.label);
  const detail = readString(status.detail);
  const diagnostics = Array.isArray(status.diagnostics)
    ? status.diagnostics.filter((entry): entry is string => typeof entry === "string")
    : undefined;
  const rawError = isRecord(status.error) ? status.error : null;
  const errorMessage = readString(rawError?.message);

  if (!state || !label || !detail) {
    return null;
  }

  return {
    source: sandboxMessageSource,
    runtimeId,
    type: "status",
    status: {
      state,
      label,
      detail,
      diagnostics,
      error: errorMessage
        ? {
            message: errorMessage,
            debugMessage: readString(rawError?.debugMessage) ?? undefined,
            type: readString(rawError?.type) ?? undefined
          }
        : null
    }
  };
}

export function buildPreviewSandboxDocument({
  captureHostKeyboard = false,
  kind,
  runtimeId,
  source
}: {
  captureHostKeyboard?: boolean;
  kind: PreviewExecutableRuntimeKind;
  runtimeId: string;
  source: PreviewRuntimeSource;
}) {
  const preparedSource = preparePreviewExecutableSource(source.source, kind);
  const payload = serializeForInlineScript({
    captureHostKeyboard,
    kind,
    runtimeId,
    source: {
      ...source,
      source: preparedSource
    }
  });

  return `<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; media-src 'none'; font-src 'none'" />
<style>
html,body{width:100%;height:100%;margin:0;overflow:hidden;background:#05080b;color:#edf3f2;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
canvas{display:block;width:100%;height:100%;}
#preview-root{position:fixed;inset:0;overflow:hidden;}
#preview-root:focus{outline:none;}
#preview-root:focus-visible{outline:2px solid rgba(103,185,255,.82);outline-offset:-3px;}
</style>
</head>
<body>
<div id="preview-root" tabindex="0"><canvas id="preview-canvas"></canvas></div>
<!-- Three.js r176, MIT; vendored from the official package. See /vendor/three.LICENSE.txt. -->
<script src="/vendor/three-r176.min.js"></script>
<script>
(${sandboxRuntimeScriptSource})(${payload});
</script>
</body>
</html>`;
}

export function preparePreviewExecutableSource(
  source: string,
  kind: PreviewExecutableRuntimeKind
) {
  if (kind === "glsl") {
    return source.replace(/\r\n/g, "\n").trim();
  }

  if (kind === "svg") {
    return source.replace(/\r\n/g, "\n").trim();
  }

  if (kind === "hydra") {
    return prepareHydraRuntimeSource(source);
  }

  if (kind === "tone") {
    return prepareToneRuntimeSource(source);
  }

  return prepareP5JavaScriptSource(source);
}

export function getPreviewRuntimeSourceMismatch({
  kind,
  source
}: {
  kind: PreviewExecutableRuntimeKind;
  source: PreviewRuntimeSource;
}) {
  switch (kind) {
    case "p5":
      return getP5RuntimeSourceSupportIssue(source.source);
    case "glsl":
      return getGlslRuntimeSourceSupportIssue(source.source);
    case "three":
      return getThreeRuntimeSourceSupportIssue(source.source);
    case "hydra": {
      const parsed = parseHydraRuntimeSource(source.source);
      return parsed.ok ? null : parsed.message;
    }
    case "tone": {
      const parsed = parseToneRuntimeSource(source.source);
      return parsed.ok ? null : parsed.message;
    }
    case "gsap":
      return getGsapRuntimeSupportIssue(source.source);
    case "svg":
      return getSvgRuntimeSupportIssue(source.source);
    case "canvas":
      return getCanvasRuntimeSupportIssue(source.source);
    default:
      return null;
  }
}

function getSandboxStartingStatus(
  kind: PreviewExecutableRuntimeKind
): PreviewRuntimeStatus {
  return {
    detail: describeSandboxRuntimeStart(kind),
    label: "Preview runtime starting",
    state: "starting",
    error: null
  };
}

function getSandboxSourceRejectedStatus({
  issue,
  kind,
  source
}: {
  issue: string;
  kind: PreviewExecutableRuntimeKind;
  source: PreviewRuntimeSource;
}): PreviewRuntimeStatus {
  const label = `${formatSandboxRuntimeLabel(kind)} runtime rejected source`;

  return {
    detail: issue,
    diagnostics: [issue],
    label,
    state: "error",
    error: createSandboxRuntimeSourceMismatchError({
      debugMessage: `${source.title}: ${issue}`,
      kind,
      message: issue
    })
  };
}

function toPreviewRuntimeStatus(
  kind: PreviewExecutableRuntimeKind,
  status: PreviewSandboxRuntimeStatusPayload
): PreviewRuntimeStatus {
  return {
    detail: status.detail,
    diagnostics: status.diagnostics,
    label: status.label,
    state: status.state,
    error: status.error
      ? createSandboxRuntimeError({
          debugMessage: status.error.debugMessage ?? null,
          kind,
          message: status.error.message,
          type: status.error.type ?? "preview_sandbox_runtime_failed"
        })
      : null
  };
}

function createSandboxRuntimeError({
  debugMessage,
  kind,
  message,
  type
}: {
  debugMessage: string | null;
  kind: PreviewExecutableRuntimeKind;
  message: string;
  type: string;
}): WorkstationError {
  return createWorkstationError({
    type,
    category: "renderer",
    subsystem: `${kind}_sandbox_runtime`,
    userMessage: message,
    debugMessage,
    recoverable: true,
    suggestedAction:
      "Reload the preview state or reset the preview session before trying again.",
    retryLabel: "Reload preview state",
    resetLabel: "Reset preview session"
  });
}

function createSandboxRuntimeSourceMismatchError({
  debugMessage,
  kind,
  message
}: {
  debugMessage: string | null;
  kind: PreviewExecutableRuntimeKind;
  message: string;
}): WorkstationError {
  return createWorkstationError({
    type: "preview_runtime_source_mismatch",
    category: "renderer",
    subsystem: `${kind}_sandbox_runtime`,
    userMessage: message,
    debugMessage,
    recoverable: true,
    suggestedAction:
      "Use a source artifact compatible with the selected preview runtime, or select a supported preview surface.",
    retryLabel: "Reload preview state",
    resetLabel: "Reset preview session"
  });
}

function describeSandboxRuntimeStart(kind: PreviewExecutableRuntimeKind) {
  switch (kind) {
    case "glsl":
      return "Mounting a controlled WebGL shader document.";
    case "three":
      return "Mounting a controlled Three.js-compatible browser document.";
    case "hydra":
      return "Mounting a controlled Hydra-compatible browser document.";
    case "tone":
      return "Mounting a controlled Tone.js-compatible audio document.";
    case "gsap":
      return "Mounting a controlled GSAP-compatible motion document.";
    case "svg":
      return "Mounting a controlled SVG vector document.";
    case "canvas":
      return "Mounting a controlled Canvas 2D document.";
    case "p5":
    default:
      return "Mounting a controlled p5.js-compatible browser document.";
  }
}

function formatSandboxRuntimeLabel(kind: PreviewExecutableRuntimeKind) {
  switch (kind) {
    case "three":
      return "Three.js";
    case "glsl":
      return "GLSL";
    case "hydra":
      return "Hydra";
    case "tone":
      return "Tone.js";
    case "gsap":
      return "GSAP";
    case "svg":
      return "SVG";
    case "canvas":
      return "Canvas";
    case "p5":
    default:
      return "p5";
  }
}

function serializeForInlineScript(value: unknown) {
  return JSON.stringify(value)
    .replace(/</g, "\\u003c")
    .replace(/>/g, "\\u003e")
    .replace(/&/g, "\\u0026")
    .replace(/\u2028/g, "\\u2028")
    .replace(/\u2029/g, "\\u2029");
}

function readRuntimeState(value: unknown): PreviewRuntimeStatus["state"] | null {
  return value === "idle" ||
    value === "starting" ||
    value === "ready" ||
    value === "running" ||
    value === "stopped" ||
    value === "error"
    ? value
    : null;
}

function readString(value: unknown) {
  return typeof value === "string" ? value : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

const sandboxRuntimeScriptSource = String.raw`function sandboxRuntimeScript(runtime) {
  const canvas = document.getElementById("preview-canvas");
  const root = document.getElementById("preview-root");
  let disposed = false;
  let frameCount = 0;
  let runningLabel = "";

  function post(message) {
    parent.postMessage(
      Object.assign({ source: "cca-preview-runtime", runtimeId: runtime.runtimeId }, message),
      "*"
    );
  }

  window.addEventListener("keydown", function (event) {
    if (runtime.captureHostKeyboard !== true) return;
    if (event.key === "Escape") {
      post({ type: "keyboard-boundary", key: "Escape", shiftKey: false });
      return;
    }
    if (event.key !== "Tab") return;
    event.preventDefault();
    post({ type: "keyboard-boundary", key: "Tab", shiftKey: event.shiftKey });
  }, true);

  function status(state, label, detail, extra) {
    runningLabel = label;
    post({
      type: "status",
      status: Object.assign(
        { state, label, detail },
        extra || {}
      )
    });
  }

  function fail(error, label) {
    const message = error && error.message ? error.message : String(error);
    status("error", label || "Runtime failed", message, {
      diagnostics: [message],
      error: {
        debugMessage: error && error.stack ? error.stack : message,
        message,
        type: "preview_sandbox_runtime_failed"
      }
    });
  }

  function rejectSource(message, label) {
    status("error", label || "Runtime rejected source", message, {
      diagnostics: [message],
      error: {
        debugMessage: message,
        message,
        type: "preview_runtime_source_mismatch"
      }
    });
  }

  function looksLikeHtmlSource(source) {
    const text = String(source || "").trim();
    return (
      /^<!doctype\s+html\b/i.test(text) ||
      /^<html(?:\s|>)/i.test(text) ||
      /^<head(?:\s|>)/i.test(text) ||
      /^<body(?:\s|>)/i.test(text) ||
      /^<(?:script|style|canvas|main|section|div|meta|link)(?:\s|>)/i.test(text)
    );
  }

  function getP5RuntimeSourceSupportIssue(source) {
    if (!looksLikeHtmlSource(source)) return null;
    return "HTML documents cannot run in the p5 JavaScript preview runtime. Use JavaScript p5 source with setup() or draw(), or route the artifact to a compatible preview surface.";
  }

  function frame(time) {
    frameCount += 1;
    post({ type: "frame", renderedAtMs: Number.isFinite(time) ? time : performance.now() });
  }

  function resizeCanvas() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const rect = root.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width || innerWidth || 640));
    const height = Math.max(1, Math.floor(rect.height || innerHeight || 360));
    const nextWidth = Math.floor(width * dpr);
    const nextHeight = Math.floor(height * dpr);
    if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
      canvas.width = nextWidth;
      canvas.height = nextHeight;
    }
    return { dpr, height, width };
  }

  function runUserScript(source, names, values) {
    return new Function(...names, source)(...values);
  }

  function normalizeColor(value, fallback) {
    if (typeof value === "number") {
      const next = Math.max(0, Math.min(255, Math.round(value)));
      return "rgb(" + next + "," + next + "," + next + ")";
    }
    if (typeof value === "string") return value;
    if (Array.isArray(value)) return "rgb(" + value.slice(0, 3).join(",") + ")";
    return fallback;
  }

  function startP5() {
    const sourceIssue = getP5RuntimeSourceSupportIssue(runtime.source.source);
    if (sourceIssue) {
      rejectSource(sourceIssue, "p5 runtime rejected source");
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) throw new Error("Canvas 2D is unavailable in the preview frame.");
    const paint = { fill: "#4cd7c8", stroke: "#edf3f2", useFill: true, useStroke: false, weight: 1 };
    const globals = {
      window,
      document,
      console,
      Math,
      PI: Math.PI,
      TWO_PI: Math.PI * 2,
      HALF_PI: Math.PI / 2,
      abs: Math.abs,
      atan2: Math.atan2,
      background,
      ceil: Math.ceil,
      circle,
      clear: () => context.clearRect(0, 0, canvas.width, canvas.height),
      color: (...values) => normalizeColor(values.length === 1 ? values[0] : values, "#fff"),
      cos: Math.cos,
      createCanvas,
      dist: Math.hypot,
      ellipse,
      fill,
      floor: Math.floor,
      frameCount: 0,
      frameRate: () => undefined,
      height: 0,
      lerp: (a, b, amount) => a + (b - a) * amount,
      line,
      map: (value, start1, stop1, start2, stop2) => start2 + ((value - start1) / (stop1 - start1 || 1)) * (stop2 - start2),
      max: Math.max,
      min: Math.min,
      mouseX: 0,
      mouseY: 0,
      noFill: () => { paint.useFill = false; },
      noise: (value) => (Math.sin(value * 12.9898) * 43758.5453) % 1,
      noStroke: () => { paint.useStroke = false; },
      pow: Math.pow,
      random: (min, max) => {
        const value = Math.random();
        if (typeof min !== "number") return value;
        if (typeof max !== "number") return value * min;
        return min + value * (max - min);
      },
      rect,
      resizeCanvas: createCanvas,
      sin: Math.sin,
      sqrt: Math.sqrt,
      stroke,
      strokeWeight: (weight) => { paint.weight = Number(weight) || 1; },
      width: 0,
      windowHeight: 0,
      windowWidth: 0
    };

    function syncDimensions() {
      const size = resizeCanvas();
      globals.width = size.width;
      globals.height = size.height;
      globals.windowWidth = size.width;
      globals.windowHeight = size.height;
      context.setTransform(size.dpr, 0, 0, size.dpr, 0, 0);
      return size;
    }

    function createCanvas(width, height) {
      const size = syncDimensions();
      globals.width = Number(width) || size.width;
      globals.height = Number(height) || size.height;
      return canvas;
    }

    function background(...values) {
      syncDimensions();
      context.save();
      context.fillStyle = normalizeColor(values.length === 1 ? values[0] : values, "#05080b");
      context.fillRect(0, 0, globals.width, globals.height);
      context.restore();
    }

    function fill(...values) {
      paint.fill = normalizeColor(values.length === 1 ? values[0] : values, paint.fill);
      paint.useFill = true;
    }

    function stroke(...values) {
      paint.stroke = normalizeColor(values.length === 1 ? values[0] : values, paint.stroke);
      paint.useStroke = true;
    }

    function drawShape(drawPath) {
      context.save();
      context.lineWidth = paint.weight;
      drawPath();
      if (paint.useFill) {
        context.fillStyle = paint.fill;
        context.fill();
      }
      if (paint.useStroke) {
        context.strokeStyle = paint.stroke;
        context.stroke();
      }
      context.restore();
    }

    function circle(x, y, diameter) {
      ellipse(x, y, diameter, diameter);
    }

    function ellipse(x, y, width, height) {
      drawShape(() => {
        context.beginPath();
        context.ellipse(Number(x) || 0, Number(y) || 0, Math.max(1, (Number(width) || 1) / 2), Math.max(1, (Number(height) || Number(width) || 1) / 2), 0, 0, Math.PI * 2);
      });
    }

    function rect(x, y, width, height) {
      drawShape(() => {
        context.beginPath();
        context.rect(Number(x) || 0, Number(y) || 0, Number(width) || 0, Number(height) || 0);
      });
    }

    function line(x1, y1, x2, y2) {
      context.save();
      context.lineWidth = paint.weight;
      context.strokeStyle = paint.stroke;
      context.beginPath();
      context.moveTo(Number(x1) || 0, Number(y1) || 0);
      context.lineTo(Number(x2) || 0, Number(y2) || 0);
      context.stroke();
      context.restore();
    }

    const result = new Function("globals", "with (globals) {" + runtime.source.source + "\\nreturn { setup: typeof setup === 'function' ? setup : null, draw: typeof draw === 'function' ? draw : null }; }")(globals);
    const setup = result.setup || window.setup;
    const draw = result.draw || window.draw;
    if (!setup && !draw) throw new Error("p5 source did not define setup() or draw().");
    syncDimensions();
    if (setup) setup.call(globals);
    status("running", "p5 runtime running", "Rendering " + runtime.source.title + " inside an isolated p5-compatible preview frame.");
    function loop(time) {
      if (disposed) return;
      syncDimensions();
      globals.frameCount += 1;
      if (draw) draw.call(globals);
      frame(time);
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  }

  function makeThree() {
    const bundledThree = window.CCAThree;
    if (!bundledThree || bundledThree.REVISION !== "176") {
      throw new Error("The locally bundled Three.js r176 runtime is unavailable.");
    }

    const activeRenderers = new Set();
    class SandboxWebGLRenderer extends bundledThree.WebGLRenderer {
      constructor(options) {
        const rendererOptions = Object.assign({}, options || {});
        if (!rendererOptions.canvas) rendererOptions.canvas = canvas;
        super(rendererOptions);
        this.__ccaAnimationFrame = 0;
        this.__ccaFrameIndex = 0;
        this.__ccaRender = this.render.bind(this);
        this.__ccaDispose = this.dispose.bind(this);
        this.render = SandboxWebGLRenderer.prototype.render.bind(this);
        this.setAnimationLoop = SandboxWebGLRenderer.prototype.setAnimationLoop.bind(this);
        this.dispose = SandboxWebGLRenderer.prototype.dispose.bind(this);
        activeRenderers.add(this);
      }
      render(scene, camera) {
        if (disposed) return;
        const size = resizeCanvas();
        const expectedWidth = Math.floor(size.width * size.dpr);
        const expectedHeight = Math.floor(size.height * size.dpr);
        if (this.domElement.width !== expectedWidth || this.domElement.height !== expectedHeight) {
          this.setPixelRatio(size.dpr);
          this.setSize(size.width, size.height, false);
        }
        if (camera && camera.isPerspectiveCamera) {
          camera.aspect = size.width / size.height;
          camera.updateProjectionMatrix();
        }
        this.__ccaRender(scene, camera);
        this.__ccaFrameIndex += 1;
        if (this.__ccaFrameIndex <= 2 || this.__ccaFrameIndex % 15 === 0) {
          const gl = this.getContext();
          const sampleWidth = Math.min(48, gl.drawingBufferWidth);
          const sampleHeight = Math.min(48, gl.drawingBufferHeight);
          const sampleX = Math.max(0, Math.floor((gl.drawingBufferWidth - sampleWidth) / 2));
          const sampleY = Math.max(0, Math.floor((gl.drawingBufferHeight - sampleHeight) / 2));
          const pixels = new Uint8Array(sampleWidth * sampleHeight * 4);
          gl.readPixels(sampleX, sampleY, sampleWidth, sampleHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
          const minimum = [255, 255, 255];
          const maximum = [0, 0, 0];
          let signature = 2166136261;
          for (let index = 0; index < pixels.length; index += 4) {
            for (let channel = 0; channel < 3; channel += 1) {
              const value = pixels[index + channel];
              minimum[channel] = Math.min(minimum[channel], value);
              maximum[channel] = Math.max(maximum[channel], value);
              signature ^= value;
              signature = Math.imul(signature, 16777619);
            }
          }
          document.body.dataset.threeFrameEnergy = String(
            maximum.reduce(function (sum, value, channel) {
              return sum + value - minimum[channel];
            }, 0)
          );
          document.body.dataset.threeFrameSignature = String(signature >>> 0);
          document.body.dataset.threeRuntimeRevision = bundledThree.REVISION;
        }
        frame(performance.now());
      }
      setAnimationLoop(callback) {
        if (this.__ccaAnimationFrame) {
          cancelAnimationFrame(this.__ccaAnimationFrame);
          this.__ccaAnimationFrame = 0;
        }
        if (typeof callback !== "function") return;
        const renderer = this;
        function loop(time) {
          if (disposed) return;
          try {
            callback(time);
            renderer.__ccaAnimationFrame = requestAnimationFrame(loop);
          } catch (error) {
            fail(error, "Three.js runtime failed");
          }
        }
        this.__ccaAnimationFrame = requestAnimationFrame(loop);
      }
      dispose() {
        if (this.__ccaAnimationFrame) {
          cancelAnimationFrame(this.__ccaAnimationFrame);
          this.__ccaAnimationFrame = 0;
        }
        activeRenderers.delete(this);
        this.__ccaDispose();
      }
    }

    return Object.assign({}, bundledThree, {
      WebGLRenderer: SandboxWebGLRenderer
    });

    /* Legacy facade retained below only for old serialized srcdoc snapshots. */
    const renderers = [];
    const scenes = [];
    const cameras = [];
    class Color {
      constructor(value) {
        this.value = value == null ? 0xffffff : value;
      }
      setHSL(h, s, l) {
        const hue = ((Number(h) % 1) + 1) % 1;
        const saturation = Math.max(0, Math.min(1, Number(s) || 0));
        const lightness = Math.max(0, Math.min(1, Number(l) || 0));
        const chroma = (1 - Math.abs(2 * lightness - 1)) * saturation;
        const sector = hue * 6;
        const match = chroma * (1 - Math.abs((sector % 2) - 1));
        const [red, green, blue] =
          sector < 1
            ? [chroma, match, 0]
            : sector < 2
              ? [match, chroma, 0]
              : sector < 3
                ? [0, chroma, match]
                : sector < 4
                  ? [0, match, chroma]
                  : sector < 5
                    ? [match, 0, chroma]
                    : [chroma, 0, match];
        const offset = lightness - chroma / 2;
        this.value =
          (Math.round((red + offset) * 255) << 16) |
          (Math.round((green + offset) * 255) << 8) |
          Math.round((blue + offset) * 255);
        return this;
      }
      toStyle() {
        if (typeof this.value === "string") return this.value;
        const value = Number(this.value) || 0xffffff;
        return "#" + value.toString(16).padStart(6, "0").slice(-6);
      }
    }
    class Object3D {
      constructor() {
        this.children = [];
        this.position = {
          x: 0,
          y: 0,
          z: 0,
          set: (x, y, z) => {
            this.position.x = x || 0;
            this.position.y = y || 0;
            this.position.z = z || 0;
            return this.position;
          },
          copy: (vector) => {
            const next = vector || {};
            return this.position.set(next.x, next.y, next.z);
          }
        };
        this.rotation = { x: 0, y: 0, z: 0 };
        this.scale = { x: 1, y: 1, z: 1, set: (x, y, z) => { this.scale.x = x || 1; this.scale.y = y || x || 1; this.scale.z = z || x || 1; } };
      }
      add(...objects) {
        this.children.push(...objects);
      }
    }
    class Scene extends Object3D {
      constructor() {
        super();
        this.background = new Color(0x05080b);
        scenes.push(this);
      }
    }
    class PerspectiveCamera extends Object3D {
      constructor() {
        super();
        this.position.z = 4;
        cameras.push(this);
      }
      lookAt() {}
    }
    class Group extends Object3D {}
    class Geometry {
      constructor(type) {
        this.type = type;
      }
    }
    class BoxGeometry extends Geometry { constructor() { super("box"); } }
    class SphereGeometry extends Geometry { constructor() { super("sphere"); } }
    class IcosahedronGeometry extends Geometry { constructor() { super("sphere"); } }
    class DodecahedronGeometry extends Geometry { constructor() { super("sphere"); } }
    class TorusGeometry extends Geometry { constructor() { super("torus"); } }
    class TorusKnotGeometry extends Geometry { constructor() { super("torus"); } }
    class PlaneGeometry extends Geometry { constructor() { super("plane"); } }
    class MeshStandardMaterial {
      constructor(options) {
        this.color = new Color(options && options.color != null ? options.color : 0x4cd7c8);
        this.emissive = new Color(options && options.emissive != null ? options.emissive : 0x7ca7ff);
      }
    }
    class MeshBasicMaterial extends MeshStandardMaterial {}
    class MeshPhongMaterial extends MeshStandardMaterial {}
    class Mesh extends Object3D {
      constructor(geometry, material) {
        super();
        this.geometry = geometry || new BoxGeometry();
        this.material = material || new MeshStandardMaterial();
      }
    }
    class Light extends Object3D {}
    class AmbientLight extends Light {}
    class DirectionalLight extends Light {}
    class HemisphereLight extends Light {}
    class PointLight extends Light {}
    class Clock {
      constructor() { this.started = performance.now(); }
      getElapsedTime() { return (performance.now() - this.started) / 1000; }
    }
    class WebGLRenderer {
      constructor(options) {
        this.domElement = options && options.canvas ? options.canvas : canvas;
        this.context = this.domElement.getContext("2d");
        this.clearColor = new Color(0x05080b);
        this.lastScene = null;
        this.lastCamera = null;
        this.outputEncoding = null;
        this.physicallyCorrectLights = false;
        this.shadowMap = { enabled: false, type: null };
        renderers.push(this);
      }
      setClearColor(value) { this.clearColor = new Color(value); }
      setPixelRatio() {}
      setSize() { resizeCanvas(); }
      render(scene, camera) {
        this.lastScene = scene;
        this.lastCamera = camera;
        drawThreeScene(this.context, scene, this.clearColor);
        frame(performance.now());
      }
      setAnimationLoop(callback) {
        const loop = (time) => {
          if (disposed) return;
          callback(time);
          requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
      }
    }
    function drawThreeScene(context, scene, clearColor) {
      if (!context) throw new Error("Canvas 2D is unavailable for the Three.js preview frame.");
      const size = resizeCanvas();
      context.setTransform(size.dpr, 0, 0, size.dpr, 0, 0);
      context.fillStyle = scene && scene.background ? scene.background.toStyle() : clearColor.toStyle();
      context.fillRect(0, 0, size.width, size.height);
      const meshes = [];
      const collect = (object) => {
        if (!object) return;
        if (object.geometry && object.material) meshes.push(object);
        if (object.children) object.children.forEach(collect);
      };
      collect(scene);
      meshes.forEach((mesh, index) => drawMesh(context, mesh, size, index));
    }
    function drawMesh(context, mesh, size, index) {
      const x = size.width * 0.5 + (mesh.position.x || 0) * size.width * 0.12;
      const y = size.height * 0.5 - (mesh.position.y || 0) * size.height * 0.12;
      const radius = Math.min(size.width, size.height) * 0.18 * (mesh.scale.x || 1);
      const spin = (mesh.rotation.y || 0) + performance.now() * 0.00035 + index;
      context.save();
      context.translate(x, y);
      context.rotate(spin);
      context.fillStyle = mesh.material.color.toStyle();
      context.strokeStyle = mesh.material.emissive.toStyle();
      context.lineWidth = 2;
      context.globalAlpha = 0.92;
      context.beginPath();
      if (mesh.geometry.type === "sphere") {
        context.arc(0, 0, radius, 0, Math.PI * 2);
      } else if (mesh.geometry.type === "torus") {
        context.arc(0, 0, radius, 0, Math.PI * 2);
        context.moveTo(radius * 0.48, 0);
        context.arc(0, 0, radius * 0.48, 0, Math.PI * 2, true);
      } else {
        context.rect(-radius * 0.72, -radius * 0.72, radius * 1.44, radius * 1.44);
      }
      context.fill();
      context.stroke();
      context.restore();
    }
    function runDefaultLoop() {
      const loop = () => {
        if (disposed) return;
        for (const renderer of renderers) {
          const scene = renderer.lastScene || scenes[0];
          const camera = renderer.lastCamera || cameras[0];
          if (scene) renderer.render(scene, camera);
        }
        requestAnimationFrame(loop);
      };
      requestAnimationFrame(loop);
    }
    return {
      AmbientLight,
      BoxGeometry,
      Clock,
      Color,
      DoubleSide: "double",
      DirectionalLight,
      DodecahedronGeometry,
      Group,
      HemisphereLight,
      IcosahedronGeometry,
      Mesh,
      MeshBasicMaterial,
      MeshPhongMaterial,
      MeshStandardMaterial,
      Object3D,
      PlaneGeometry,
      PerspectiveCamera,
      PCFSoftShadowMap: "pcf-soft",
      PointLight,
      Scene,
      SphereGeometry,
      TorusGeometry,
      TorusKnotGeometry,
      WebGLRenderer,
      sRGBEncoding: "srgb",
      MathUtils: { degToRad: (value) => value * Math.PI / 180 },
      __runDefaultLoop: runDefaultLoop
    };
  }

  function startThree() {
    const THREE = makeThree();
    const requestThreeAnimationFrame = function (callback) {
      return window.requestAnimationFrame(function (time) {
        if (disposed) return;
        try {
          callback(time);
        } catch (error) {
          fail(error, "Three.js runtime failed");
        }
      });
    };
    runUserScript(runtime.source.source, ["THREE", "window", "document", "requestAnimationFrame", "cancelAnimationFrame", "performance"], [THREE, window, document, requestThreeAnimationFrame, cancelAnimationFrame.bind(window), performance]);
    status("running", "Three.js runtime running", "Rendering " + runtime.source.title + " with the locally bundled Three.js r176 WebGL runtime.");
  }

  function fragmentSource(source) {
    const precision = /precision\\s+(lowp|mediump|highp)\\s+float/i.test(source) ? "" : "precision mediump float;\\n";
    const resolution = /uniform\\s+vec2\\s+u_resolution\\s*;/i.test(source) ? "" : "uniform vec2 u_resolution;\\n";
    const time = /uniform\\s+float\\s+u_time\\s*;/i.test(source) ? "" : "uniform float u_time;\\n";
    const header = precision + resolution + time;
    if (/void\\s+mainImage\\s*\\(/i.test(source)) {
      return header + source + "\\nvoid main(){vec4 color=vec4(0.0);mainImage(color,gl_FragCoord.xy);gl_FragColor=color;}";
    }
    if (/void\\s+main\\s*\\(/i.test(source)) {
      return header + source;
    }
    throw new Error("GLSL source did not define main() or mainImage().");
  }

  function startGlsl() {
    const gl = canvas.getContext("webgl", { alpha: false, antialias: false });
    if (!gl) throw new Error("WebGL is unavailable in the preview frame.");
    const vertex = "attribute vec2 a_position;void main(){gl_Position=vec4(a_position,0.0,1.0);}";
    const fragment = fragmentSource(runtime.source.source);
    const program = createProgram(gl, vertex, fragment);
    const buffer = gl.createBuffer();
    const position = gl.getAttribLocation(program, "a_position");
    const resolution = gl.getUniformLocation(program, "u_resolution");
    const timeUniform = gl.getUniformLocation(program, "u_time");
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    status("running", "GLSL runtime running", "Rendering " + runtime.source.title + " as an isolated WebGL fragment shader.");
    function loop(time) {
      if (disposed) return;
      const size = resizeCanvas();
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
      gl.useProgram(program);
      gl.enableVertexAttribArray(position);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
      gl.uniform2f(resolution, size.width, size.height);
      gl.uniform1f(timeUniform, time * 0.001);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      frame(time);
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  }

  function setupTone() {
    const program = JSON.parse(runtime.source.source);
    if (program.error) throw new Error(program.error);
    if (!Array.isArray(program.voices) || program.voices.length === 0) {
      throw new Error("Tone.js program did not define a supported audio voice.");
    }
    status(
      "ready",
      "Tone.js runtime ready",
      runtime.source.title + " is armed. Audio remains silent until Start audio is selected.",
      { diagnostics: ["Explicit operator interaction is required before audio starts."] }
    );
  }

  function startHydra() {
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Canvas 2D is unavailable in the preview frame.");
    const program = JSON.parse(runtime.source.source);
    if (program.error) throw new Error(program.error);
    const buffer = document.createElement("canvas");
    buffer.width = 128;
    buffer.height = 72;
    const bufferContext = buffer.getContext("2d");
    if (!bufferContext) throw new Error("Hydra frame buffer is unavailable.");
    let previous = {};

    function number(value, fallback) {
      return typeof value === "number" && Number.isFinite(value) ? value : fallback;
    }
    function clamp(value) {
      return Math.max(0, Math.min(1, value));
    }
    function fract(value) {
      return value - Math.floor(value);
    }
    function samplePrevious(name, u, v) {
      const pixels = previous[name];
      if (!pixels) return [0, 0, 0];
      const x = Math.max(0, Math.min(buffer.width - 1, Math.floor(fract(u) * buffer.width)));
      const y = Math.max(0, Math.min(buffer.height - 1, Math.floor(fract(v) * buffer.height)));
      const index = (y * buffer.width + x) * 3;
      return [pixels[index] || 0, pixels[index + 1] || 0, pixels[index + 2] || 0];
    }
    function sampleValue(value, u, v, time, depth) {
      if (typeof value === "string") return samplePrevious(value, u, v);
      if (value && value.source) return sampleChain(value, u, v, time, depth + 1);
      const scalar = number(value, 0);
      return [scalar, scalar, scalar];
    }
    function sampleChain(chain, inputU, inputV, time, depth) {
      if (!chain || !chain.source || depth > 8) return [0, 0, 0];
      let u = inputU;
      let v = inputV;
      const operators = chain.operators || [];
      operators.forEach(function (operator) {
        const args = operator.args || [];
        const modulation = sampleValue(args[0], u, v, time, depth);
        const amount = number(args[1], number(args[0], 0.1));
        if (operator.name === "rotate" || operator.name === "modulateRotate") {
          const angle = number(args[0], 0) + (operator.name === "modulateRotate" ? modulation[0] * amount : 0);
          const x = u - 0.5;
          const y = v - 0.5;
          u = x * Math.cos(angle) - y * Math.sin(angle) + 0.5;
          v = x * Math.sin(angle) + y * Math.cos(angle) + 0.5;
        } else if (operator.name === "scale" || operator.name === "modulateScale") {
          const scale = Math.max(0.05, number(args[0], 1) + (operator.name === "modulateScale" ? modulation[0] * amount : 0));
          u = (u - 0.5) / scale + 0.5;
          v = (v - 0.5) / scale + 0.5;
        } else if (operator.name === "scroll" || operator.name === "scrollX" || operator.name === "modulateScrollX") {
          u += number(args[0], 0) + (operator.name === "modulateScrollX" ? modulation[0] * amount : 0);
          if (operator.name === "scroll") v += number(args[1], 0);
        } else if (operator.name === "scrollY" || operator.name === "modulateScrollY") {
          v += number(args[0], 0) + (operator.name === "modulateScrollY" ? modulation[0] * amount : 0);
        } else if (operator.name.indexOf("repeat") !== -1 || operator.name.indexOf("modulateRepeat") === 0) {
          const repeatX = Math.max(1, number(args[0], 3));
          const repeatY = Math.max(1, number(args[1], repeatX));
          if (operator.name !== "repeatY" && operator.name !== "modulateRepeatY") u = fract(u * repeatX + modulation[0] * amount);
          if (operator.name !== "repeatX" && operator.name !== "modulateRepeatX") v = fract(v * repeatY + modulation[1] * amount);
        } else if (operator.name === "kaleid" || operator.name === "modulateKaleid") {
          const sides = Math.max(2, Math.round(number(args[0], 6) + modulation[0] * amount));
          const radius = Math.hypot(u - 0.5, v - 0.5);
          const segment = Math.PI * 2 / sides;
          const angle = Math.abs(((Math.atan2(v - 0.5, u - 0.5) % segment) + segment) % segment - segment / 2);
          u = 0.5 + Math.cos(angle) * radius;
          v = 0.5 + Math.sin(angle) * radius;
        } else if (operator.name === "pixelate" || operator.name === "modulatePixelate") {
          const cellsX = Math.max(1, number(args[0], 20) + modulation[0] * amount * 20);
          const cellsY = Math.max(1, number(args[1], cellsX));
          u = Math.floor(u * cellsX) / cellsX;
          v = Math.floor(v * cellsY) / cellsY;
        } else if (operator.name === "modulate" || operator.name === "modulateHue") {
          u += (modulation[0] - 0.5) * amount;
          v += (modulation[1] - 0.5) * amount;
        }
      });

      const source = chain.source;
      const args = source.args || [];
      let color;
      if (source.name === "solid") {
        color = [number(args[0], 0), number(args[1], number(args[0], 0)), number(args[2], number(args[0], 0))];
      } else if (source.name === "gradient") {
        const speed = number(args[0], 0);
        color = [fract(u + time * speed), fract(v + u * 0.5), fract(1 - u + time * speed * 0.5)];
      } else if (source.name === "noise") {
        const scale = number(args[0], 10);
        const offset = number(args[1], 0.1);
        const value = fract(Math.sin((u * scale + v * scale * 1.73 + time * offset) * 12.9898) * 43758.5453);
        color = [value * 0.72, value, 1 - value * 0.54];
      } else if (source.name === "voronoi") {
        const scale = number(args[0], 5);
        const speed = number(args[1], 0.3);
        const x = fract(u * scale) - 0.5;
        const y = fract(v * scale) - 0.5;
        const value = clamp(Math.hypot(x, y) * 2 + Math.sin(time * speed + Math.floor(u * scale)) * 0.15);
        color = [value, 1 - value * 0.65, 0.7 + value * 0.3];
      } else if (source.name === "shape") {
        const sides = Math.max(2, number(args[0], 3));
        const radius = number(args[1], 0.3);
        const smoothing = Math.max(0.001, number(args[2], 0.01));
        const angle = Math.atan2(v - 0.5, u - 0.5);
        const distance = Math.hypot(u - 0.5, v - 0.5);
        const edge = Math.cos(Math.floor(0.5 + angle / (Math.PI * 2 / sides)) * (Math.PI * 2 / sides) - angle) * distance;
        const value = clamp((radius - edge) / smoothing);
        color = [value, value * 0.72, 1 - value * 0.35];
      } else if (source.name === "src") {
        color = sampleValue(args[0] || "o0", u, v, time, depth);
      } else {
        const frequency = number(args[0], 60);
        const sync = number(args[1], 0.1);
        const offset = number(args[2], 0);
        const phase = (u * frequency + time * sync) * Math.PI * 2;
        color = [0.5 + 0.5 * Math.sin(phase), 0.5 + 0.5 * Math.sin(phase + offset), 0.5 + 0.5 * Math.sin(phase + offset * 2)];
      }

      operators.forEach(function (operator) {
        const args = operator.args || [];
        const other = sampleValue(args[0], u, v, time, depth);
        const amount = number(args[1], 0.5);
        if (operator.name === "color") color = color.map(function (channel, index) { return channel * number(args[index], 1); });
        else if (operator.name === "brightness") color = color.map(function (channel) { return channel + number(args[0], 0.4); });
        else if (operator.name === "contrast") color = color.map(function (channel) { return (channel - 0.5) * number(args[0], 1.6) + 0.5; });
        else if (operator.name === "invert") color = color.map(function (channel) { return 1 - channel * number(args[0], 1); });
        else if (operator.name === "thresh" || operator.name === "luma") color = color.map(function (channel) { return channel >= number(args[0], 0.5) ? 1 : 0; });
        else if (operator.name === "posterize") {
          const bins = Math.max(2, number(args[0], 3));
          color = color.map(function (channel) { return Math.floor(channel * bins) / (bins - 1); });
        } else if (operator.name === "saturate") {
          const grey = (color[0] + color[1] + color[2]) / 3;
          color = color.map(function (channel) { return grey + (channel - grey) * number(args[0], 2); });
        } else if (operator.name === "hue" || operator.name === "colorama") {
          const shift = Math.round(number(args[0], 0.4) * 3) % 3;
          color = [color[shift], color[(shift + 1) % 3], color[(shift + 2) % 3]];
        } else if (operator.name === "blend") color = color.map(function (channel, index) { return channel * (1 - amount) + other[index] * amount; });
        else if (operator.name === "add" || operator.name === "layer") color = color.map(function (channel, index) { return channel + other[index] * amount; });
        else if (operator.name === "mult" || operator.name === "mask") color = color.map(function (channel, index) { return channel * other[index]; });
        else if (operator.name === "diff") color = color.map(function (channel, index) { return Math.abs(channel - other[index]); });
      });
      return color.map(clamp);
    }

    status("running", "Hydra runtime running", "Rendering " + runtime.source.title + " as a bounded Hydra-compatible synth.");
    function loop(time) {
      if (disposed) return;
      const size = resizeCanvas();
      const next = {};
      Object.keys(program.outputs || {}).forEach(function (outputName) {
        const chain = program.outputs[outputName];
        const pixels = new Float32Array(buffer.width * buffer.height * 3);
        for (let y = 0; y < buffer.height; y += 1) {
          for (let x = 0; x < buffer.width; x += 1) {
            const color = sampleChain(chain, x / buffer.width, y / buffer.height, time * 0.001 * number(program.speed, 1), 0);
            const index = (y * buffer.width + x) * 3;
            pixels[index] = color[0];
            pixels[index + 1] = color[1];
            pixels[index + 2] = color[2];
          }
        }
        next[outputName] = pixels;
      });
      previous = next;
      const selected = next[program.renderTarget] || next.o0;
      if (!selected) throw new Error("Hydra program did not produce a renderable output.");
      const image = bufferContext.createImageData(buffer.width, buffer.height);
      for (let index = 0; index < selected.length / 3; index += 1) {
        image.data[index * 4] = Math.round(selected[index * 3] * 255);
        image.data[index * 4 + 1] = Math.round(selected[index * 3 + 1] * 255);
        image.data[index * 4 + 2] = Math.round(selected[index * 3 + 2] * 255);
        image.data[index * 4 + 3] = 255;
      }
      bufferContext.putImageData(image, 0, 0);
      context.setTransform(size.dpr, 0, 0, size.dpr, 0, 0);
      context.imageSmoothingEnabled = true;
      context.drawImage(buffer, 0, 0, size.width, size.height);
      frame(time);
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  }

  function createProgram(gl, vertexSource, fragmentSource) {
    const vertex = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
    const fragment = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
    const program = gl.createProgram();
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program) || "Shader program did not link.");
    }
    return program;
  }

  function compileShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader) || "Shader did not compile.");
    }
    return shader;
  }

  window.addEventListener("error", (event) => fail(event.error || event.message, runningLabel || "Runtime error"));
  window.addEventListener("unhandledrejection", (event) => fail(event.reason || "Unhandled runtime rejection", runningLabel || "Runtime error"));
  window.addEventListener("beforeunload", () => { disposed = true; });

  try {
    status("starting", "Preview runtime mounted", "Preview document loaded for " + runtime.source.title + ".");
    if (runtime.kind === "p5") startP5();
    else if (runtime.kind === "three") startThree();
    else if (runtime.kind === "hydra") startHydra();
    else if (runtime.kind === "tone") setupTone();
    else startGlsl();
  } catch (error) {
    fail(error, runtime.kind === "glsl" ? "GLSL runtime failed" : runtime.kind === "three" ? "Three.js runtime failed" : runtime.kind === "hydra" ? "Hydra runtime failed" : runtime.kind === "tone" ? "Tone.js runtime failed" : "p5 runtime failed");
  }
}`;
