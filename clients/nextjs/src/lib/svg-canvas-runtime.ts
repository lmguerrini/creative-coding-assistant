const canvasSignalPatterns: readonly RegExp[] = [
  /\bgetcontext\s*\(\s*['"]2d['"]\s*\)/i,
  /\b(?:ctx|context)\.(?:fillrect|clearrect|beginpath|arc|lineto|moveto|stroke|fill|filltext|quadraticcurveto|beziercurveto|save|restore)\b/i,
  /\brequestanimationframe\s*\(/i,
  /\bcanvasrenderingcontext2d\b/i,
  /\bhtml5\s+canvas\b/i,
  /\bcanvas\s+2d\b/i
];

const canvasUnsupportedPatterns: readonly {
  pattern: RegExp;
  reason: string;
}[] = [
  {
    pattern: /\b(?:fetch|XMLHttpRequest|WebSocket|EventSource|Worker|SharedWorker|navigator\.serviceWorker)\b/i,
    reason: "Remote network access and worker execution are not allowed in the Canvas preview sandbox."
  },
  {
    pattern: /\b(?:eval|Function)\s*\(/i,
    reason: "Dynamic code execution is not allowed in the Canvas preview sandbox."
  },
  {
    pattern: /\b(?:document|window)\.(?:body|documentElement|head|write|open|parent|top|location)\b/i,
    reason: "Canvas previews can only use the bounded preview surface APIs."
  },
  {
    pattern:
      /\b(?:createElement|appendChild|prepend|insertAdjacentHTML|innerHTML|outerHTML)\b/i,
    reason: "Canvas previews must draw into the provided preview surface instead of mutating the DOM."
  },
  {
    pattern:
      /\b(?:new\s+Image|createImageBitmap|OffscreenCanvas|drawImage|captureStream|toBlob|toDataURL)\b/i,
    reason:
      "Canvas preview support is limited to direct 2D drawing without image assets or auxiliary canvases."
  },
  {
    pattern:
      /\baddEventListener\s*\(\s*['"](?:mouse|pointer|touch|key|click|wheel|input)/i,
    reason:
      "Canvas preview support is limited to deterministic animation without interactive input handlers."
  },
  {
    pattern: /\bon(?:mouse|pointer|touch|key|click|wheel|input)[a-z]*\s*=/i,
    reason:
      "Canvas preview support is limited to deterministic animation without interactive input handlers."
  },
  {
    pattern: /https?:\/\//i,
    reason: "Remote assets are not allowed in the Canvas preview sandbox."
  }
];

const svgSignalPatterns: readonly RegExp[] = [
  /<svg[\s>]/i,
  /\bxmlns\s*=\s*['"]http:\/\/www\.w3\.org\/2000\/svg['"]/i,
  /<(?:path|circle|rect|ellipse|line|polyline|polygon|text|linearGradient|radialGradient|animate|animateTransform)\b/i
];

const svgUnsupportedPatterns: readonly {
  pattern: RegExp;
  reason: string;
}[] = [
  {
    pattern: /<(?:script|foreignObject|iframe|object|embed)\b/i,
    reason:
      "SVG preview support is limited to sanitized inline SVG markup without scriptable DOM containers."
  },
  {
    pattern: /\bon[a-z]+\s*=/i,
    reason:
      "SVG preview support is limited to sanitized inline SVG markup without event handlers."
  },
  {
    pattern: /\b(?:javascript:|data:text\/html)/i,
    reason: "SVG previews cannot embed executable URLs."
  },
  {
    pattern: /<(?:image|audio|video)\b/i,
    reason: "SVG preview support is limited to self-contained vector markup without external media assets."
  },
  {
    pattern: /https?:\/\/(?!www\.w3\.org\/2000\/svg\b)/i,
    reason: "Remote assets are not allowed in the SVG preview sandbox."
  }
];

const canvasDomains = new Set(["canvas", "canvas_2d"]);
const svgDomains = new Set(["svg", "svg_markup"]);

export function hasCanvasPreviewSignal({
  content,
  domain,
  language,
  runtime,
  summary,
  title
}: {
  content?: string | null;
  domain?: string | null;
  language?: string | null;
  runtime?: string | null;
  summary?: string | null;
  title?: string | null;
}) {
  const normalizedTitle = (title ?? "").trim().toLowerCase();
  const normalizedDomain = (domain ?? "").trim().toLowerCase();
  const normalizedRuntime = (runtime ?? "").trim().toLowerCase();
  const normalizedLanguage = (language ?? "").trim().toLowerCase();
  const haystack = [title, language, summary, content].join(" ").toLowerCase();
  const signalCount = canvasSignalPatterns.reduce(
    (count, pattern) => count + Number(pattern.test(haystack)),
    0
  );

  return (
    canvasDomains.has(normalizedDomain) ||
    normalizedRuntime === "canvas" ||
    normalizedRuntime === "surface.canvas" ||
    normalizedLanguage === "canvas" ||
    normalizedLanguage === "canvas 2d" ||
    normalizedLanguage === "canvas_2d" ||
    normalizedTitle.endsWith(".canvas.js") ||
    normalizedTitle.endsWith(".canvas.ts") ||
    signalCount >= 2
  );
}

export function getCanvasRuntimeSupportIssue(source: string | null | undefined) {
  if (!source?.trim()) {
    return null;
  }

  if (source.length > 20_000) {
    return "The Canvas source is too large for the bounded preview runtime.";
  }

  for (const { pattern, reason } of canvasUnsupportedPatterns) {
    if (pattern.test(source)) {
      return reason;
    }
  }

  return null;
}

export function hasSvgPreviewSignal({
  content,
  domain,
  language,
  runtime,
  summary,
  title
}: {
  content?: string | null;
  domain?: string | null;
  language?: string | null;
  runtime?: string | null;
  summary?: string | null;
  title?: string | null;
}) {
  const normalizedTitle = (title ?? "").trim().toLowerCase();
  const normalizedDomain = (domain ?? "").trim().toLowerCase();
  const normalizedRuntime = (runtime ?? "").trim().toLowerCase();
  const normalizedLanguage = (language ?? "").trim().toLowerCase();
  const haystack = [title, language, summary, content].join(" ").toLowerCase();

  return (
    svgDomains.has(normalizedDomain) ||
    normalizedRuntime === "svg" ||
    normalizedRuntime === "surface.svg" ||
    normalizedLanguage === "svg" ||
    normalizedTitle.endsWith(".svg") ||
    svgSignalPatterns.some((pattern) => pattern.test(haystack))
  );
}

export function getSvgRuntimeSupportIssue(source: string | null | undefined) {
  if (!source?.trim()) {
    return null;
  }

  if (source.length > 40_000) {
    return "The SVG source is too large for the bounded preview runtime.";
  }

  if (!/<svg[\s>]/i.test(source)) {
    return "SVG preview source must contain an <svg> root element.";
  }

  for (const { pattern, reason } of svgUnsupportedPatterns) {
    if (pattern.test(source)) {
      return reason;
    }
  }

  return null;
}
