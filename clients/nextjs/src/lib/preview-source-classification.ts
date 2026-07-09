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

const supportedP5GlobalFunctions = new Set([
  "abs",
  "atan2",
  "background",
  "beginShape",
  "ceil",
  "circle",
  "clear",
  "color",
  "colorMode",
  "cos",
  "createCanvas",
  "dist",
  "ellipse",
  "endShape",
  "fill",
  "floor",
  "frameRate",
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
  "resizeCanvas",
  "rotate",
  "scale",
  "sin",
  "sqrt",
  "stroke",
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
  if (/\bnew\s+p5\s*\(/.test(rawSource) || /\b(?:p|sketch)\s*=>/.test(rawSource)) {
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
