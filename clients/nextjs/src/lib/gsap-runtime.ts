const gsapSignalPatterns: readonly RegExp[] = [
  /\bgsap\.(?:to|from|fromTo|set|timeline)\b/i,
  /\bgsap\.timeline\s*\(/i,
  /\bstagger\s*:/i,
  /\byoyo\s*:/i
];

const gsapUnsupportedPatterns: readonly {
  pattern: RegExp;
  reason: string;
}[] = [
  {
    pattern: /\b(?:fetch|XMLHttpRequest|WebSocket)\b/i,
    reason: "Remote network access is not allowed in the GSAP preview sandbox."
  },
  {
    pattern: /\b(?:eval|Function)\s*\(/i,
    reason: "Dynamic code execution is not allowed in the GSAP preview sandbox."
  },
  {
    pattern: /\b(?:document|window)\.(?:body|documentElement|head)\b/i,
    reason: "GSAP previews can only target the bounded sandbox stage."
  },
  {
    pattern:
      /\b(?:ScrollTrigger|MotionPathPlugin|Draggable|Flip|MorphSVGPlugin|DrawSVGPlugin)\b/i,
    reason: "GSAP preview support is limited to core tweens and timelines without plugins."
  },
  {
    pattern: /\bgsap\.registerPlugin\b/i,
    reason: "GSAP preview support is limited to core tweens and timelines without plugins."
  },
  {
    pattern:
      /\b(?:createElement|appendChild|prepend|insertAdjacentHTML|innerHTML|outerHTML)\b/i,
    reason: "GSAP previews must animate the provided sandbox nodes instead of injecting new DOM."
  },
  {
    pattern: /https?:\/\//i,
    reason: "Remote assets are not allowed in the GSAP preview sandbox."
  }
];

export function hasGsapPreviewSignal({
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
    normalizedDomain === "gsap" ||
    normalizedRuntime === "gsap" ||
    normalizedRuntime === "surface.gsap" ||
    normalizedLanguage === "gsap" ||
    normalizedTitle.endsWith(".gsap.js") ||
    normalizedTitle.endsWith(".gsap.ts") ||
    gsapSignalPatterns.some((pattern) => pattern.test(haystack))
  );
}

export function getGsapRuntimeSupportIssue(source: string | null | undefined) {
  if (!source?.trim()) {
    return null;
  }

  if (source.length > 16_000) {
    return "The GSAP motion source is too large for the bounded preview runtime.";
  }

  for (const { pattern, reason } of gsapUnsupportedPatterns) {
    if (pattern.test(source)) {
      return reason;
    }
  }

  return null;
}
