import type { ReactNode } from "react";

type MarkdownBlock =
  | { kind: "code"; content: string; language: string }
  | { kind: "heading"; content: string; level: 2 | 3 | 4 }
  | { kind: "list"; items: string[]; ordered: boolean }
  | { kind: "paragraph"; content: string }
  | { kind: "quote"; content: string };

export function ConversationMarkdown({ content }: { content: string }) {
  const blocks = parseMarkdownBlocks(content);

  return (
    <div className="conversationMarkdown">
      {blocks.map((block, index) => {
        const key = `${block.kind}-${index}`;

        switch (block.kind) {
          case "code": {
            const languageLabel = formatLanguageLabel(block.language);
            return (
              <figure className="conversationCodeBlock" key={key}>
                <figcaption>{languageLabel}</figcaption>
                <pre aria-label={`${languageLabel} code example`}>
                  <code>{block.content}</code>
                </pre>
              </figure>
            );
          }
          case "heading": {
            const Heading = `h${block.level}` as "h2" | "h3" | "h4";
            return <Heading key={key}>{renderInlineCode(block.content)}</Heading>;
          }
          case "list": {
            const List = block.ordered ? "ol" : "ul";
            return (
              <List key={key}>
                {block.items.map((item, itemIndex) => (
                  <li key={`${key}-${itemIndex}`}>{renderInlineCode(item)}</li>
                ))}
              </List>
            );
          }
          case "quote":
            return <blockquote key={key}>{renderInlineCode(block.content)}</blockquote>;
          case "paragraph":
            return <p key={key}>{renderInlineCode(block.content)}</p>;
        }
      })}
    </div>
  );
}

function parseMarkdownBlocks(content: string): MarkdownBlock[] {
  const lines = content.replace(/\r\n?/g, "\n").split("\n");
  const blocks: MarkdownBlock[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index] ?? "";
    if (!line.trim()) {
      index += 1;
      continue;
    }

    const fence = line.match(/^\s*```\s*([^\s`]*)?.*$/);
    if (fence) {
      const codeLines: string[] = [];
      index += 1;
      while (index < lines.length && !/^\s*```\s*$/.test(lines[index] ?? "")) {
        codeLines.push(lines[index] ?? "");
        index += 1;
      }
      if (index < lines.length) {
        index += 1;
      }
      blocks.push({
        kind: "code",
        content: codeLines.join("\n").replace(/\s+$/, ""),
        language: fence[1] ?? ""
      });
      continue;
    }

    const heading = line.match(/^\s*(#{1,4})\s+(.+)$/);
    if (heading) {
      blocks.push({
        kind: "heading",
        content: heading[2]?.trim() ?? "",
        level: Math.min(4, Math.max(2, heading[1]?.length ?? 2)) as 2 | 3 | 4
      });
      index += 1;
      continue;
    }

    const listItem = matchListItem(line);
    if (listItem) {
      const items: string[] = [];
      const ordered = listItem.ordered;
      while (index < lines.length) {
        const nextItem = matchListItem(lines[index] ?? "");
        if (!nextItem || nextItem.ordered !== ordered) {
          break;
        }
        items.push(nextItem.content);
        index += 1;
      }
      blocks.push({ kind: "list", items, ordered });
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const quoteLines: string[] = [];
      while (index < lines.length && /^\s*>\s?/.test(lines[index] ?? "")) {
        quoteLines.push((lines[index] ?? "").replace(/^\s*>\s?/, ""));
        index += 1;
      }
      blocks.push({ kind: "quote", content: quoteLines.join(" ") });
      continue;
    }

    const paragraphLines = [line.trim()];
    index += 1;
    while (index < lines.length && !startsMarkdownBlock(lines[index] ?? "")) {
      paragraphLines.push((lines[index] ?? "").trim());
      index += 1;
    }
    blocks.push({
      kind: "paragraph",
      content: paragraphLines.filter(Boolean).join(" ")
    });
  }

  return blocks;
}

function startsMarkdownBlock(line: string) {
  return (
    !line.trim() ||
    /^\s*```/.test(line) ||
    /^\s*#{1,4}\s+/.test(line) ||
    /^\s*>\s?/.test(line) ||
    Boolean(matchListItem(line))
  );
}

function matchListItem(line: string) {
  const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
  if (unordered) {
    return { content: unordered[1]?.trim() ?? "", ordered: false };
  }
  const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
  return ordered
    ? { content: ordered[1]?.trim() ?? "", ordered: true }
    : null;
}

function renderInlineCode(content: string): ReactNode[] {
  return content.split(/(`[^`\n]+`)/g).map((part, index) =>
    part.startsWith("`") && part.endsWith("`") ? (
      <code key={`code-${index}`}>{part.slice(1, -1)}</code>
    ) : (
      part
    )
  );
}

function formatLanguageLabel(language: string) {
  const normalized = language.trim().toLowerCase();
  const labels: Record<string, string> = {
    css: "CSS",
    frag: "GLSL",
    glsl: "GLSL",
    html: "HTML",
    js: "JavaScript",
    javascript: "JavaScript",
    json: "JSON",
    jsx: "JSX",
    py: "Python",
    python: "Python",
    ts: "TypeScript",
    tsx: "TSX"
  };
  return labels[normalized] ?? (normalized ? normalized.toUpperCase() : "Code");
}
