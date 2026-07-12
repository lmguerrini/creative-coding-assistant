const htmlSourcePatterns: readonly RegExp[] = [
  /^\s*<!doctype\s+html\b/i,
  /^\s*<html(?:\s|>)/i,
  /^\s*<head(?:\s|>)/i,
  /^\s*<body(?:\s|>)/i,
  /^\s*<(?:script|style|canvas|main|section|div|meta|link)(?:\s|>)/i
];

export const p5HtmlSourceMismatchMessage =
  "HTML documents cannot run in the p5 JavaScript preview runtime. Use JavaScript p5 source with setup() or draw(), or route the artifact to a compatible preview surface.";

export const p5GlobalModeContractMessage =
  "Use a plain JavaScript global-mode sketch with function setup() and function draw(). Keep p5 calls inside those lifecycle functions or helpers they call.";

export const threeHtmlSourceMismatchMessage =
  "Standalone HTML documents cannot run in the controlled Three.js JavaScript preview runtime. Keep the document as an export, or provide self-contained Three.js scene JavaScript without HTML markup.";

export const threeRuntimeContractMessage =
  "Use self-contained JavaScript that creates a THREE.Scene, camera, renderer, and scene content. Module imports are normalized when possible, but HTML documents, React components, and TypeScript syntax are not executable in this bounded runtime.";

export const reactThreeFiberBundleRuntimeMessage =
  "React Three Fiber components need their own bundle runtime and cannot run in the controlled Three.js JavaScript preview.";

export const glslRuntimeContractMessage =
  "Use a compact WebGL 1 fragment shader with void main() or mainImage(), u_time and u_resolution uniforms, and no texture sampling, sampler declarations, discard, or while loops.";

export function isReactThreeFiberSource(source: string | null | undefined) {
  const rawSource = source?.trim() ?? "";
  return (
    /@react-three\/fiber|\breact-three-fiber\b/i.test(rawSource) ||
    /<Canvas(?:\s|>)/.test(rawSource)
  );
}

export function getGlslRuntimeSourceSupportIssue(
  source: string | null | undefined
) {
  const rawSource = source?.trim() ?? "";
  if (!rawSource) {
    return "The GLSL preview source is empty. Add a compact fragment shader.";
  }
  if (rawSource.length > 6000) {
    return "The fragment shader is too large for this lightweight runtime.";
  }
  const executableSource = stripCommentsAndStrings(rawSource);
  if (/^\s*#version\b/m.test(executableSource)) {
    return "GLSL #version declarations cannot run in the controlled WebGL 1 fragment preview.";
  }
  if (/\b(?:while|sampler2D|samplerCube|texture|texture2D|textureCube|discard)\b/i.test(executableSource)) {
    return "The fragment shader uses features outside the current bounded runtime subset.";
  }
  if (!/void\s+(?:main|mainImage)\s*\(/i.test(executableSource)) {
    return `The fragment shader needs void main() or mainImage(). ${glslRuntimeContractMessage}`;
  }
  return null;
}

const supportedP5GlobalFunctions = new Set([
  "abs",
  "atan2",
  "background",
  "beginShape",
  "blendMode",
  "blue",
  "ceil",
  "circle",
  "clear",
  "color",
  "colorMode",
  "constrain",
  "cos",
  "createCanvas",
  "curveVertex",
  "degrees",
  "dist",
  "ellipse",
  "endShape",
  "exp",
  "fill",
  "floor",
  "frameRate",
  "green",
  "int",
  "lerp",
  "line",
  "map",
  "max",
  "min",
  "noise",
  "noiseDetail",
  "noFill",
  "noStroke",
  "pixelDensity",
  "point",
  "pop",
  "pow",
  "push",
  "random",
  "rect",
  "rectMode",
  "red",
  "resizeCanvas",
  "rotate",
  "scale",
  "sin",
  "smooth",
  "sqrt",
  "stroke",
  "strokeCap",
  "strokeJoin",
  "strokeWeight",
  "translate",
  "vertex"
]);

const supportedJavaScriptGlobalFunctions = new Set([
  "Array",
  "Boolean",
  "Number",
  "Object",
  "String",
  "isFinite",
  "isNaN",
  "parseFloat",
  "parseInt"
]);

const controlFlowKeywords = new Set([
  "catch",
  "for",
  "function",
  "if",
  "switch",
  "while"
]);

export function getP5RuntimeSourceSupportIssue(
  source: string | null | undefined
) {
  const rawSource = source?.trim() ?? "";
  if (!rawSource) {
    return "The p5 preview source is empty. Add a self-contained JavaScript sketch.";
  }
  if (looksLikeHtmlSource(rawSource)) {
    return p5HtmlSourceMismatchMessage;
  }
  if (/```/.test(rawSource)) {
    return "Markdown fences cannot run in the p5 preview. Return the executable JavaScript source only.";
  }
  if (/\bnew\s+p5\s*\(/.test(rawSource)) {
    return `${p5GlobalModeContractMessage} Instance-mode p5 wrappers are not supported here.`;
  }

  const preparedSource = prepareP5JavaScriptSource(rawSource);
  if (!/\bfunction\s+setup\s*\(/.test(preparedSource) || !/\bfunction\s+draw\s*\(/.test(preparedSource)) {
    return `${p5GlobalModeContractMessage} Both setup() and draw() are required.`;
  }

  const topLevelP5Call = findTopLevelP5Call(preparedSource);
  if (topLevelP5Call) {
    return `${topLevelP5Call}() must run inside setup(), draw(), or a helper invoked by them.`;
  }

  const unsupportedCall = findUnsupportedBareCall(preparedSource);
  if (unsupportedCall) {
    return `${unsupportedCall}() is not part of the supported browser p5 preview contract. ${p5GlobalModeContractMessage}`;
  }

  return null;
}

export function prepareP5JavaScriptSource(source: string) {
  const withoutDeclarations = source
    .replace(/\r\n/g, "\n")
    .replace(/^\s*import\s+[^;\n]+;?\s*$/gm, "")
    .replace(/^\s*export\s+default\s+/gm, "")
    .replace(/^\s*export\s+(?=(?:async\s+)?function|class|const|let|var)/gm, "")
    .replace(/^\s*type\s+[A-Za-z_$][\w$]*(?:<[^>]+>)?\s*=\s*.*;?\s*$/gm, "")
    .replace(/^\s*(?:type|interface)\s+[^{=]+(?:=\s*[^;]+;|{[\s\S]*?}\s*)/gm, "")
    .replace(/\b(const|let|var)\s+([A-Za-z_$][\w$]*)\s*:\s*[^=;]+=/g, "$1 $2 =")
    .replace(/\)\s*:\s*[A-Za-z_$][\w$<>,\s[\]|]*(?=\s*[{=])/g, ")")
    .replace(/\s+as\s+const\b/g, "");

  const normalized = stripTypeScriptParameterAnnotations(withoutDeclarations).trim();
  if (/\bfunction\s+draw\s*\(/.test(normalized) && !/\bfunction\s+setup\s*\(/.test(normalized)) {
    return `${normalized}\n\nfunction setup() {}`;
  }
  if (/\bfunction\s+setup\s*\(/.test(normalized) && !/\bfunction\s+draw\s*\(/.test(normalized)) {
    return `${normalized}\n\nfunction draw() {}`;
  }
  return normalized;
}

export function getThreeRuntimeSourceSupportIssue(
  source: string | null | undefined
) {
  const rawSource = source?.trim() ?? "";
  if (!rawSource) {
    return "The Three.js preview source is empty. Add a self-contained scene script.";
  }
  if (looksLikeHtmlSource(rawSource)) {
    return threeHtmlSourceMismatchMessage;
  }
  if (/```/.test(rawSource)) {
    return "Markdown fences cannot run in the controlled Three.js preview. Return the executable JavaScript source only.";
  }
  if (isReactThreeFiberSource(rawSource)) {
    return reactThreeFiberBundleRuntimeMessage;
  }
  if (!/\bTHREE\s*\./.test(rawSource)) {
    return null;
  }
  if (
    /^\s*(?:interface|type)\s+[A-Za-z_$][\w$]*/m.test(rawSource) ||
    /\b(?:const|let|var)\s+[A-Za-z_$][\w$]*\s*:\s*[A-Za-z_$][\w$<>{}\[\]|, ]*(?==)/.test(rawSource) ||
    /\bfunction\s*[A-Za-z_$]*\s*\([^)]*:\s*[A-Za-z_$]/.test(rawSource) ||
    /\)\s*:\s*[A-Za-z_$][\w$<>{}\[\]|, ]*(?=\s*[{=])/.test(rawSource)
  ) {
    return `TypeScript syntax is not executable in this controlled preview. ${threeRuntimeContractMessage}`;
  }
  return null;
}

export function prepareThreeJavaScriptSource(source: string) {
  return source
    .replace(/\r\n/g, "\n")
    .replace(/^\s*import\s+[^;\n]+;?\s*$/gm, "")
    .replace(/^\s*export\s+default\s+/gm, "")
    .replace(/^\s*export\s+(?=(?:async\s+)?function|class|const|let|var)/gm, "")
    .trim();
}

function stripTypeScriptParameterAnnotations(source: string) {
  return source
    .replace(
      /function(\s+[A-Za-z_$][\w$]*\s*)\(([^)]*)\)/g,
      (_match, prefix: string, params: string) =>
        `function${prefix}(${stripParameterListTypes(params)})`
    )
    .replace(/\(([^)]*:[^)]*)\)\s*=>/g, (_match, params: string) => {
      return `(${stripParameterListTypes(params)}) =>`;
    });
}

function stripParameterListTypes(params: string) {
  return params
    .split(",")
    .map((param) => param.replace(/\s*:\s*[^=]+(?=$|=)/, ""))
    .join(",");
}

function findTopLevelP5Call(source: string) {
  const code = stripCommentsAndStrings(source);
  for (const call of findBareCalls(code)) {
    if (
      supportedP5GlobalFunctions.has(call.name) &&
      braceDepthAt(code, call.index) === 0
    ) {
      return call.name;
    }
  }
  return null;
}

function findUnsupportedBareCall(source: string) {
  const code = stripCommentsAndStrings(source);
  const declaredNames = new Set<string>();
  const declarationPattern = /\b(?:function|class)\s+([A-Za-z_$][\w$]*)\b|\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>/g;

  for (const match of code.matchAll(declarationPattern)) {
    const name = match[1] ?? match[2];
    if (name) {
      declaredNames.add(name);
    }
  }

  for (const call of findBareCalls(code)) {
    if (
      supportedP5GlobalFunctions.has(call.name) ||
      supportedJavaScriptGlobalFunctions.has(call.name) ||
      controlFlowKeywords.has(call.name) ||
      declaredNames.has(call.name)
    ) {
      continue;
    }
    return call.name;
  }

  return null;
}

function findBareCalls(source: string) {
  const calls: Array<{ index: number; name: string }> = [];
  const pattern = /(^|[^.$\w])([A-Za-z_$][\w$]*)\s*\(/g;

  for (const match of source.matchAll(pattern)) {
    const prefixLength = match[1]?.length ?? 0;
    const index = (match.index ?? 0) + prefixLength;
    const name = match[2];
    const before = source.slice(Math.max(0, index - 16), index);
    if (/\bfunction\s*$/.test(before)) {
      continue;
    }
    calls.push({ index, name });
  }

  return calls;
}

function braceDepthAt(source: string, index: number) {
  let depth = 0;
  for (let cursor = 0; cursor < index; cursor += 1) {
    if (source[cursor] === "{") {
      depth += 1;
    } else if (source[cursor] === "}") {
      depth = Math.max(0, depth - 1);
    }
  }
  return depth;
}

function stripCommentsAndStrings(source: string) {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/\/\/.*$/gm, " ")
    .replace(/'(?:\\.|[^'\\])*'/g, "''")
    .replace(/"(?:\\.|[^"\\])*"/g, '""')
    .replace(/`(?:\\.|[^`\\])*`/g, "``");
}

export function looksLikeHtmlSource(source: string | null | undefined) {
  const text = stripLeadingHtmlComments(source?.trimStart() ?? "");

  if (!text) {
    return false;
  }

  return htmlSourcePatterns.some((pattern) => pattern.test(text));
}

function stripLeadingHtmlComments(source: string) {
  let text = source;
  let match = text.match(/^<!--[\s\S]*?-->\s*/);

  while (match) {
    text = text.slice(match[0].length).trimStart();
    match = text.match(/^<!--[\s\S]*?-->\s*/);
  }

  return text.trim();
}
