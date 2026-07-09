const htmlSourcePatterns: readonly RegExp[] = [
  /^\s*<!doctype\s+html\b/i,
  /^\s*<html(?:\s|>)/i,
  /^\s*<head(?:\s|>)/i,
  /^\s*<body(?:\s|>)/i,
  /^\s*<(?:script|style|canvas|main|section|div|meta|link)(?:\s|>)/i
];

export const p5HtmlSourceMismatchMessage =
  "HTML documents cannot run in the p5 JavaScript preview runtime. Use JavaScript p5 source with setup() or draw(), or route the artifact to a compatible preview surface.";

export function getP5RuntimeSourceSupportIssue(
  source: string | null | undefined
) {
  return looksLikeHtmlSource(source) ? p5HtmlSourceMismatchMessage : null;
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

  return stripTypeScriptParameterAnnotations(withoutDeclarations).trim();
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
