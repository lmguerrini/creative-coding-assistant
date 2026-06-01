import type {
  PreviewExecutableRuntimeKind,
  PreviewRuntimeFrameSample,
  PreviewRuntimeSource,
  PreviewRuntimeStatus
} from "./preview-runtime-adapters";
import {
  createWorkstationError,
  type WorkstationError
} from "./workstation-errors";

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
    };

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
  iframe: HTMLIFrameElement;
  kind: PreviewExecutableRuntimeKind;
  onFrame?: ((sample: PreviewRuntimeFrameSample) => void) | undefined;
  onStatus: (status: PreviewRuntimeStatus) => void;
  runtimeId: string;
  source: PreviewRuntimeSource;
};

export type PreviewSandboxRuntimeMount = {
  dispose: () => void;
};

const sandboxMessageSource = "cca-preview-runtime";

export function createPreviewSandboxRuntimeId() {
  return `preview-runtime-${Math.random().toString(36).slice(2, 10)}`;
}

export function mountPreviewSandboxRuntime({
  iframe,
  kind,
  onFrame,
  onStatus,
  runtimeId,
  source
}: MountPreviewSandboxRuntimeInput): PreviewSandboxRuntimeMount {
  let disposed = false;
  let retryTimer = 0;
  const mountMessage = {
    source: sandboxMessageSource,
    runtimeId,
    type: "mount",
    runtime: {
      kind,
      runtimeId,
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

    onStatus(toPreviewRuntimeStatus(kind, message.status));
  }

  function postMountMessage() {
    if (disposed) {
      return;
    }
    iframe.contentWindow?.postMessage(mountMessage, "*");
  }

  iframe.dataset.runtimeId = runtimeId;
  window.addEventListener("message", handleMessage);
  iframe.addEventListener("load", postMountMessage);
  onStatus(getSandboxStartingStatus(kind));
  iframe.src = "/preview-sandbox.html";
  retryTimer = window.setTimeout(postMountMessage, 150);

  return {
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
  kind,
  runtimeId,
  source
}: {
  kind: PreviewExecutableRuntimeKind;
  runtimeId: string;
  source: PreviewRuntimeSource;
}) {
  const preparedSource = preparePreviewExecutableSource(source.source, kind);
  const payload = serializeForInlineScript({
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
<style>
html,body{width:100%;height:100%;margin:0;overflow:hidden;background:#05080b;color:#edf3f2;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
canvas{display:block;width:100%;height:100%;}
#preview-root{position:fixed;inset:0;overflow:hidden;}
</style>
</head>
<body>
<div id="preview-root"><canvas id="preview-canvas"></canvas></div>
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

  return source
    .replace(/\r\n/g, "\n")
    .replace(/^\s*import\s+[^;\n]+;?\s*$/gm, "")
    .replace(/^\s*export\s+default\s+/gm, "")
    .replace(/^\s*export\s+(?=(?:async\s+)?function|class|const|let|var)/gm, "")
    .replace(/^\s*(?:type|interface)\s+[^{=]+(?:=\s*[^;]+;|{[\s\S]*?}\s*)/gm, "")
    .replace(/\b(const|let|var)\s+([A-Za-z_$][\w$]*)\s*:\s*[^=;]+=/g, "$1 $2 =")
    .replace(/([,(]\s*[A-Za-z_$][\w$]*)\s*:\s*[^,)]+(?=[,)])/g, "$1")
    .replace(/\)\s*:\s*[A-Za-z_$][\w$<>,\s[\]|]*(?=\s*[{=])/g, ")")
    .replace(/\s+as\s+const\b/g, "")
    .trim();
}

function getSandboxStartingStatus(
  kind: PreviewExecutableRuntimeKind
): PreviewRuntimeStatus {
  return {
    detail:
      kind === "glsl"
        ? "Mounting a sandboxed WebGL shader document."
        : kind === "three"
          ? "Mounting a sandboxed Three.js-compatible browser document."
          : "Mounting a sandboxed p5.js-compatible browser document.",
    label: "Runtime sandbox starting",
    state: "starting",
    error: null
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
    value === "running" ||
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
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Canvas 2D is unavailable in the sandbox.");
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
    status("running", "p5 runtime running", "Executing " + runtime.source.title + " inside a sandboxed p5-compatible iframe.");
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
    const renderers = [];
    const scenes = [];
    const cameras = [];
    class Color {
      constructor(value) {
        this.value = value == null ? 0xffffff : value;
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
        this.position = { x: 0, y: 0, z: 0, set: (x, y, z) => { this.position.x = x || 0; this.position.y = y || 0; this.position.z = z || 0; } };
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
      if (!context) throw new Error("Canvas 2D is unavailable for the Three.js sandbox.");
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
      DirectionalLight,
      DodecahedronGeometry,
      IcosahedronGeometry,
      Mesh,
      MeshBasicMaterial,
      MeshPhongMaterial,
      MeshStandardMaterial,
      PerspectiveCamera,
      PointLight,
      Scene,
      SphereGeometry,
      TorusGeometry,
      TorusKnotGeometry,
      WebGLRenderer,
      MathUtils: { degToRad: (value) => value * Math.PI / 180 },
      __runDefaultLoop: runDefaultLoop
    };
  }

  function startThree() {
    const THREE = makeThree();
    runUserScript(runtime.source.source, ["THREE", "window", "document", "requestAnimationFrame", "cancelAnimationFrame", "performance"], [THREE, window, document, requestAnimationFrame.bind(window), cancelAnimationFrame.bind(window), performance]);
    THREE.__runDefaultLoop();
    status("running", "Three.js runtime running", "Executing " + runtime.source.title + " inside a sandboxed Three.js-compatible iframe.");
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
    if (!gl) throw new Error("WebGL is unavailable in the sandbox.");
    const vertex = "attribute vec2 a_position;void main(){gl_Position=vec4(a_position,0.0,1.0);}";
    const fragment = fragmentSource(runtime.source.source);
    const program = createProgram(gl, vertex, fragment);
    const buffer = gl.createBuffer();
    const position = gl.getAttribLocation(program, "a_position");
    const resolution = gl.getUniformLocation(program, "u_resolution");
    const timeUniform = gl.getUniformLocation(program, "u_time");
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    status("running", "GLSL runtime running", "Executing " + runtime.source.title + " as a sandboxed WebGL fragment shader.");
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
    status("starting", "Runtime sandbox mounted", "Sandbox document loaded for " + runtime.source.title + ".");
    if (runtime.kind === "p5") startP5();
    else if (runtime.kind === "three") startThree();
    else startGlsl();
  } catch (error) {
    fail(error, runtime.kind === "glsl" ? "GLSL runtime failed" : runtime.kind === "three" ? "Three.js runtime failed" : "p5 runtime failed");
  }
}`;
