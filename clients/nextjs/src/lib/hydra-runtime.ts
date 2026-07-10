export type HydraOutputName = "o0" | "o1" | "o2" | "o3";

export type HydraRuntimeCall = {
  name: string;
  args: HydraRuntimeValue[];
};

export type HydraRuntimeChain = {
  source: HydraRuntimeCall;
  operators: HydraRuntimeCall[];
};

export type HydraRuntimeValue =
  | number
  | HydraOutputName
  | HydraRuntimeChain;

export type HydraRuntimeProgram = {
  version: 1;
  speed: number;
  renderTarget: HydraOutputName;
  outputs: Partial<Record<HydraOutputName, HydraRuntimeChain>>;
};

export type HydraRuntimeParseResult =
  | { ok: true; program: HydraRuntimeProgram }
  | { ok: false; message: string };

type Token =
  | { type: "identifier"; value: string }
  | { type: "number"; value: number }
  | { type: "punctuation"; value: "." | "," | "(" | ")" };

const hydraSourceNames = new Set([
  "gradient",
  "noise",
  "osc",
  "shape",
  "solid",
  "src",
  "voronoi"
]);

const hydraOperatorNames = new Set([
  "add",
  "blend",
  "brightness",
  "color",
  "colorama",
  "contrast",
  "diff",
  "hue",
  "invert",
  "kaleid",
  "layer",
  "luma",
  "mask",
  "modulate",
  "modulateHue",
  "modulateKaleid",
  "modulatePixelate",
  "modulateRepeat",
  "modulateRepeatX",
  "modulateRepeatY",
  "modulateRotate",
  "modulateScale",
  "modulateScrollX",
  "modulateScrollY",
  "mult",
  "pixelate",
  "posterize",
  "repeat",
  "repeatX",
  "repeatY",
  "rotate",
  "saturate",
  "scale",
  "scroll",
  "scrollX",
  "scrollY",
  "thresh"
]);

const hydraOutputs = new Set<HydraOutputName>(["o0", "o1", "o2", "o3"]);
const maxHydraSourceLength = 12_000;
const maxHydraCalls = 80;
const maxHydraDepth = 8;

export function parseHydraRuntimeSource(source: string): HydraRuntimeParseResult {
  if (source.length > maxHydraSourceLength) {
    return {
      ok: false,
      message: `Hydra source exceeds the ${maxHydraSourceLength}-character sandbox limit.`
    };
  }

  const program: HydraRuntimeProgram = {
    version: 1,
    outputs: {},
    renderTarget: "o0",
    speed: 1
  };
  let callCount = 0;

  try {
    for (const statement of splitHydraStatements(source)) {
      if (/^(?:speed|bpm)\s*=/.test(statement)) {
        const [, rawValue = ""] = statement.split("=", 2);
        const value = Number(rawValue.trim());
        if (!Number.isFinite(value)) {
          throw new Error("Hydra speed must be a finite number.");
        }
        program.speed = clamp(value, 0.05, 8);
        continue;
      }

      if (/^hush\s*\(\s*\)$/.test(statement)) {
        program.outputs = {};
        continue;
      }

      const tokens = tokenizeHydraStatement(statement);
      if (tokens.length === 0) {
        continue;
      }

      if (isIdentifier(tokens[0], "render")) {
        const renderCall = parseCall(tokens, 0, 0);
        ensureConsumed(tokens, renderCall.nextIndex);
        const target = renderCall.call.args[0] ?? "o0";
        if (!isHydraOutputName(target)) {
          throw new Error("render() accepts only o0, o1, o2, or o3.");
        }
        program.renderTarget = target;
        callCount += 1;
        continue;
      }

      const parsed = parseChain(tokens, 0, 0);
      ensureConsumed(tokens, parsed.nextIndex);
      callCount += 1 + parsed.chain.operators.length;
      const operators = [...parsed.chain.operators];
      const outCall = operators.at(-1);
      if (!outCall || outCall.name !== "out") {
        throw new Error("Each Hydra source chain must end with out().");
      }
      operators.pop();
      const target = outCall.args[0] ?? "o0";
      if (!isHydraOutputName(target)) {
        throw new Error("out() accepts only o0, o1, o2, or o3.");
      }
      program.outputs[target] = {
        source: parsed.chain.source,
        operators
      };
    }

    if (callCount > maxHydraCalls) {
      return {
        ok: false,
        message: `Hydra source exceeds the ${maxHydraCalls}-call sandbox limit.`
      };
    }
    if (Object.keys(program.outputs).length === 0) {
      return {
        ok: false,
        message: "Hydra source must define at least one chain ending in out()."
      };
    }
    if (!program.outputs[program.renderTarget]) {
      program.renderTarget = firstHydraOutput(program.outputs) ?? "o0";
    }
    return { ok: true, program };
  } catch (error) {
    return {
      ok: false,
      message: error instanceof Error ? error.message : "Hydra source could not be parsed."
    };
  }
}

export function prepareHydraRuntimeSource(source: string) {
  const result = parseHydraRuntimeSource(source);
  return JSON.stringify(
    result.ok
      ? result.program
      : {
          version: 1,
          error: result.message
        }
  );
}

function splitHydraStatements(source: string) {
  const normalized = source
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\/\/.*$/gm, "");
  const foldedContinuations = normalized
    .split("\n")
    .reduce<string[]>((lines, line) => {
      const trimmed = line.trim();
      if (trimmed.startsWith(".") && lines.length > 0) {
        const lastIndex = lines.length - 1;
        lines[lastIndex] = `${lines[lastIndex]} ${trimmed}`;
        return lines;
      }

      lines.push(line);
      return lines;
    }, [])
    .join("\n");
  const statements: string[] = [];
  let current = "";
  let depth = 0;

  for (const character of foldedContinuations) {
    if (character === "(") {
      depth += 1;
    } else if (character === ")") {
      depth -= 1;
      if (depth < 0) {
        throw new Error("Hydra source contains an unmatched closing parenthesis.");
      }
    }

    if ((character === ";" || character === "\n") && depth === 0) {
      if (current.trim()) {
        statements.push(current.trim());
      }
      current = "";
    } else {
      current += character;
    }
  }

  if (depth !== 0) {
    throw new Error("Hydra source contains an unmatched opening parenthesis.");
  }
  if (current.trim()) {
    statements.push(current.trim());
  }
  return statements;
}

function tokenizeHydraStatement(statement: string): Token[] {
  const tokens: Token[] = [];
  let index = 0;

  while (index < statement.length) {
    const remaining = statement.slice(index);
    const whitespace = remaining.match(/^\s+/);
    if (whitespace) {
      index += whitespace[0].length;
      continue;
    }

    const number = remaining.match(/^-?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?/i);
    if (number) {
      tokens.push({ type: "number", value: Number(number[0]) });
      index += number[0].length;
      continue;
    }

    const identifier = remaining.match(/^[A-Za-z_$][\w$]*/);
    if (identifier) {
      tokens.push({ type: "identifier", value: identifier[0] });
      index += identifier[0].length;
      continue;
    }

    const punctuation = statement[index];
    if (
      punctuation === "." ||
      punctuation === "," ||
      punctuation === "(" ||
      punctuation === ")"
    ) {
      tokens.push({ type: "punctuation", value: punctuation });
      index += 1;
      continue;
    }

    throw new Error("Unsupported Hydra syntax was rejected by the bounded parser.");
  }

  return tokens;
}

function parseChain(tokens: Token[], startIndex: number, depth: number) {
  if (depth > maxHydraDepth) {
    throw new Error(`Hydra expressions exceed the ${maxHydraDepth}-level nesting limit.`);
  }

  const sourceResult = parseCall(tokens, startIndex, depth);
  if (!hydraSourceNames.has(sourceResult.call.name)) {
    throw new Error(`${sourceResult.call.name}() is not a supported Hydra source.`);
  }
  const operators: HydraRuntimeCall[] = [];
  let index = sourceResult.nextIndex;

  while (isPunctuation(tokens[index], ".")) {
    const operatorResult = parseCall(tokens, index + 1, depth);
    if (
      operatorResult.call.name !== "out" &&
      !hydraOperatorNames.has(operatorResult.call.name)
    ) {
      throw new Error(`${operatorResult.call.name}() is not a supported Hydra operator.`);
    }
    operators.push(operatorResult.call);
    index = operatorResult.nextIndex;
  }

  return {
    chain: {
      source: sourceResult.call,
      operators
    },
    nextIndex: index
  };
}

function parseCall(tokens: Token[], startIndex: number, depth: number) {
  const nameToken = tokens[startIndex];
  if (!nameToken || nameToken.type !== "identifier") {
    throw new Error("Expected a Hydra function name.");
  }
  if (!isPunctuation(tokens[startIndex + 1], "(")) {
    throw new Error(`Expected "(" after ${nameToken.value}.`);
  }

  const args: HydraRuntimeValue[] = [];
  let index = startIndex + 2;
  while (!isPunctuation(tokens[index], ")")) {
    if (index >= tokens.length) {
      throw new Error(`Expected ")" after ${nameToken.value}().`);
    }
    const parsedValue = parseValue(tokens, index, depth + 1);
    args.push(parsedValue.value);
    index = parsedValue.nextIndex;
    if (isPunctuation(tokens[index], ",")) {
      index += 1;
    } else if (!isPunctuation(tokens[index], ")")) {
      throw new Error(`Expected "," or ")" inside ${nameToken.value}().`);
    }
  }

  return {
    call: {
      args,
      name: nameToken.value
    },
    nextIndex: index + 1
  };
}

function parseValue(tokens: Token[], startIndex: number, depth: number) {
  const token = tokens[startIndex];
  if (!token) {
    throw new Error("Expected a Hydra argument.");
  }
  if (token.type === "number") {
    return { nextIndex: startIndex + 1, value: token.value };
  }
  if (token.type !== "identifier") {
    throw new Error("Hydra arguments must be numbers, output buffers, or source chains.");
  }
  if (isHydraOutputName(token.value)) {
    return { nextIndex: startIndex + 1, value: token.value };
  }
  if (isPunctuation(tokens[startIndex + 1], "(")) {
    const parsed = parseChain(tokens, startIndex, depth);
    return {
      nextIndex: parsed.nextIndex,
      value: parsed.chain
    };
  }
  throw new Error(`Unsupported Hydra argument "${token.value}".`);
}

function ensureConsumed(tokens: Token[], index: number) {
  if (index !== tokens.length) {
    throw new Error("Hydra source contains unsupported trailing syntax.");
  }
}

function isHydraOutputName(value: unknown): value is HydraOutputName {
  return typeof value === "string" && hydraOutputs.has(value as HydraOutputName);
}

function isIdentifier(token: Token | undefined, value: string) {
  return token?.type === "identifier" && token.value === value;
}

function isPunctuation(
  token: Token | undefined,
  value: Extract<Token, { type: "punctuation" }>["value"]
) {
  return token?.type === "punctuation" && token.value === value;
}

function firstHydraOutput(
  outputs: HydraRuntimeProgram["outputs"]
): HydraOutputName | null {
  for (const output of hydraOutputs) {
    if (outputs[output]) {
      return output;
    }
  }
  return null;
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}
