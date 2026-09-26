import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { aggregate } from "../src/aggregation.js";
import { computeStateBudget } from "../src/chunking.js";
import { decide, resolveTransport } from "../src/decide.js";
import { ARGMAX_TIE_TOLERANCE, DEFAULTS, MIN_STATE_TOKENS, QUESTION_KEY } from "../src/defaults.js";
import {
  JevAbortError,
  JevChunkFailedError,
  JevContextBudgetError,
  JevProviderError,
  JevResponseError,
  JevValidationError,
} from "../src/errors.js";
import { createTokenizer, defaultTokenizer } from "../src/tokenizer.js";
import { DirectJevTransport } from "../src/transports/direct.js";
import { OpenRouterJevTransport } from "../src/transports/openrouter.js";
import { resolveOptions } from "../src/validation.js";
import type {
  AggregationMethod,
  ChoiceQuestion,
  ChunkResult,
  JevInfiniteCTXEvent,
  JevInfiniteCTXRequest,
  JevQuestion,
  JevTransport,
  NativeChoiceAnswer,
  NativeJevResponse,
  NoulQuestion,
  RechunkEvent,
  ScoreQuestion,
} from "../src/types.js";
import {
  MockTransport,
  choiceAnswer,
  countWord,
  makeText,
  noulAnswer,
  providerError,
  scoreAnswer,
} from "./helpers/mock-transport.js";
import { buildInto, freshCheckout, isPublished, packedFiles, repoRoot } from "./helpers/packaging.js";
import { typeErrors } from "./helpers/typecheck.js";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const CHOICE: ChoiceQuestion = {
  type: "choice",
  instructions: "What is the text mostly about?",
  criteria: { tree: "Trees or forests.", rock: "Rocks or geology.", other: "Neither." },
};

const SCORE: ScoreQuestion = {
  type: "score",
  instructions: "How much of the text is about rocks?",
  criteria: ["None", "Some", "Most", "All"],
};

const NOUL: NoulQuestion = {
  type: "noul",
  instructions: "Is the text mainly about trees?",
  criteria: { true: "Mostly trees.", false: "Mostly something else." },
};

/** Tree-heavy prose followed by rock-heavy prose; ~9 chunks at a 256-token budget. */
const TOPIC_INPUT = `${makeText(1000, 11, ["oak"])}\n\n${makeText(500, 12, ["granite"])}`;

/** Per-chunk "evidence" read from the chunk's own text, so answers depend on the exact state sent. */
function evidence(state: string): { oak: number; granite: number } {
  return { oak: countWord(state, "oak"), granite: countWord(state, "granite") };
}

function topicDistribution(state: string): [number, number, number] {
  const { oak, granite } = evidence(state);
  const total = oak + granite + 3;
  return [(oak + 1) / total, (granite + 1) / total, 1 / total];
}

function topicAnswer(state: string): NativeChoiceAnswer {
  const [tree, rock, other] = topicDistribution(state);
  return choiceAnswer({ tree, rock, other }, Math.max(tree, rock, other));
}

function rockShare(state: string): number {
  const { oak, granite } = evidence(state);
  return (granite + 1) / (oak + granite + 2);
}

function scoreDistribution(state: string): number[] {
  const r = rockShare(state);
  return [(1 - r) ** 2, r * (1 - r), r * (1 - r), r ** 2];
}

/** A request with a small budget (many chunks) and instant retries. */
function request<Q extends JevQuestion>(
  question: Q,
  input: string,
  transport: JevTransport,
  overrides: Omit<Partial<JevInfiniteCTXRequest<Q>>, "question" | "input"> = {},
): JevInfiniteCTXRequest<Q> {
  return {
    input,
    question,
    provider: { transport },
    chunking: { maxStateTokens: MIN_STATE_TOKENS },
    execution: { retryBaseDelayMs: 0 },
    ...overrides,
  };
}

function eventsOf<T extends JevInfiniteCTXEvent["type"]>(
  events: readonly JevInfiniteCTXEvent[],
  type: T,
): Array<Extract<JevInfiniteCTXEvent, { type: T }>> {
  return events.filter((e): e is Extract<JevInfiniteCTXEvent, { type: T }> => e.type === type);
}

function stateOf(input: string, chunk: Pick<ChunkResult, "start" | "end">): string {
  return input.slice(chunk.start, chunk.end);
}

/** Rebuilds the input from chunk states by dropping each chunk's overlap with its predecessor. */
function reassemble(chunks: ReadonlyArray<{ start: number; end: number; state: string }>): string {
  let text = "";
  chunks.forEach((chunk, i) => {
    const previousEnd = i === 0 ? chunk.start : (chunks[i - 1] as { end: number }).end;
    text += chunk.state.slice(previousEnd - chunk.start);
  });
  return text;
}

function weightedMean(vectors: readonly number[][], weights: readonly number[]): number[] {
  const width = (vectors[0] as number[]).length;
  const sums = Array.from({ length: width }, (_, j) =>
    vectors.reduce((sum, v, i) => sum + (weights[i] as number) * (v[j] as number), 0),
  );
  const total = sums.reduce((a, b) => a + b, 0);
  return sums.map((s) => s / total);
}

function tv(p: readonly number[], q: readonly number[]): number {
  return 0.5 * p.reduce((sum, pj, j) => sum + Math.abs(pj - (q[j] as number)), 0);
}

/** spec 11.3–11.6 with the default parameters, computed independently of src/c3.ts. */
function c3ByHand(base: number, agreement: number, nEff: number): number {
  const { cap, lambda, agreementFloor, agreementExponent } = DEFAULTS.confidence;
  const saturation = 1 - Math.exp(-lambda * (nEff - 1));
  const gate = Math.min(1, Math.max(0, (agreement - agreementFloor) / (1 - agreementFloor))) ** agreementExponent;
  return base + Math.max(0, cap - base) * saturation * gate;
}

/** spec 11.3: new content in units of the largest chunk, clamped to [1, N]. */
function nEffByHand(results: readonly ChunkResult[]): number {
  const unique = results.reduce((sum, r) => sum + r.uniqueTokens, 0);
  const largest = Math.max(1, ...results.map((r) => r.estimatedTokens));
  return Math.min(results.length, Math.max(1, unique / largest));
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

// ---------------------------------------------------------------------------
// Package entry point
// ---------------------------------------------------------------------------

describe("package entry point", () => {
  it("exposes decide and serializeResult on the default JevInfiniteCTX object", async () => {
    const entry = await import("../src/index.js");
    expect(entry.default).toBe(entry.JevInfiniteCTX);
    expect(entry.JevInfiniteCTX.decide).toBe(decide);
    expect(Object.keys(entry.JevInfiniteCTX)).toEqual(["decide", "serializeResult"]);
    const mock = new MockTransport({ script: [noulAnswer(0.25)] });
    const result = await entry.default.decide({ input: "Text.", question: NOUL, provider: { transport: mock } });
    expect(entry.default.serializeResult(result)).toMatchObject({ type: "noul", noul: 0.25 });
  });

  it("re-exports the documented building blocks, errors, and defaults", async () => {
    const entry: Record<string, unknown> = await import("../src/index.js");
    const names = [
      "decide",
      "resolveTransport",
      "serializeResult",
      "OpenRouterJevTransport",
      "DirectJevTransport",
      "OPENROUTER_DEFAULT_MODEL",
      "DIRECT_DEFAULT_MODEL",
      "HeuristicTokenizer",
      "defaultTokenizer",
      "createTokenizer",
      "planChunks",
      "computeStateBudget",
      "aggregate",
      "computeAgreement",
      "totalVariation",
      "effectiveChunkCount",
      "baseConfidence",
      "applyC3",
      "DEFAULTS",
      "JevInfiniteCTXError",
      "JevValidationError",
      "JevProviderError",
      "JevResponseError",
      "JevChunkFailedError",
      "JevContextBudgetError",
      "JevAbortError",
      "isJevProviderError",
    ];
    for (const name of names) expect(entry[name], name).toBeDefined();
    expect(entry["DEFAULTS"]).toBe(DEFAULTS);
    // Internal helpers stay internal.
    for (const name of ["normalizeWeights", "clamp01", "postDecision", "mapWithConcurrency", "withRetry"]) {
      expect(entry[name], name).toBeUndefined();
    }
  });

  it("exports DEFAULTS deeply frozen, so a consumer cannot change the defaults of later calls", () => {
    // Checked first: mutating an unfrozen DEFAULTS below would leak into every later test.
    for (const section of [DEFAULTS, DEFAULTS.chunking, DEFAULTS.execution, DEFAULTS.confidence]) {
      expect(Object.isFrozen(section)).toBe(true);
    }
    // ESM is strict mode, so writing to a frozen object throws.
    expect(() => Object.assign(DEFAULTS.confidence, { method: "none" })).toThrow(TypeError);
    expect(() => Object.assign(DEFAULTS.execution, { retries: 0, maxConcurrency: 1 })).toThrow(TypeError);
    const resolved = resolveOptions({ input: "Text.", question: NOUL });
    expect(resolved.confidence.method).toBe("c3");
    expect(resolved.execution.retries).toBe(3);
    expect(resolved.execution.maxConcurrency).toBe(4);
  });
});

// ---------------------------------------------------------------------------
// Public types (vitest does not typecheck, so these run the compiler)
// ---------------------------------------------------------------------------

describe("public types", () => {
  const PRELUDE = `
import { decide } from "../src/index.js";
import type { JevCriterion, NoulQuestion, ScoreQuestion } from "../src/index.js";
declare const input: string;
`;
  const SNIPPETS = {
    mutableScore: `${PRELUDE}
void decide({ input, question: { type: "score", instructions: "Rate", criteria: ["Low", "High"] } });
`,
    asConstScore: `${PRELUDE}
const RISK = { type: "score", instructions: "Rate", criteria: ["Low", "High"] } as const;
void decide({ input, question: RISK });
`,
    readonlyRubric: `${PRELUDE}
const RUBRIC = ["Low", "Medium", "High"] as const;
const LEVELS: readonly string[] = ["Low", "High"];
export const a: ScoreQuestion = { type: "score", instructions: "Rate", criteria: RUBRIC };
export const b: ScoreQuestion = { type: "score", instructions: "Rate", criteria: LEVELS };
`,
    asConstStructured: `${PRELUDE}
const CHOICE = {
  type: "choice",
  instructions: "Pick",
  criteria: { a: { what: "A", examples: ["x", "y"] }, b: "B." },
} as const;
void decide({ input, question: CHOICE }).then((result) => {
  const choice: "a" | "b" = result.choice;
  return choice;
});
const INSTR = { task: "Classify", notes: ["one", "two"] } as const;
export const n: NoulQuestion = { type: "noul", instructions: INSTR };
`,
    notJson: `${PRELUDE}
export const bad: JevCriterion = [() => 1];
`,
  };
  let errors: Record<string, string[]> = {};
  // One program for every snippet, compiled only when this block runs.
  beforeAll(() => {
    errors = typeErrors(SNIPPETS);
  }, 60_000);

  it.each(["mutableScore", "asConstScore", "readonlyRubric", "asConstStructured"])(
    "accepts the %s question shape",
    (name) => {
      expect(errors[name]).toEqual([]);
    },
  );

  it("still rejects criteria that are not JSON values (so the checks above are not vacuous)", () => {
    expect(errors["notJson"]?.length).toBeGreaterThan(0);
  });
});

describe("public types under exactOptionalPropertyTypes", () => {
  // A consumer with the flag on must be able to pass possibly-unset values
  // (README: `apiKey: process.env.OPENROUTER_API_KEY`); runtime reads an
  // explicit undefined as omitted.
  const SNIPPETS = {
    readmeExample: `
import { decide, DirectJevTransport, OpenRouterJevTransport } from "../src/index.js";

const transport = new OpenRouterJevTransport({
  apiKey: process.env.OPENROUTER_API_KEY,
  appName: "contract-review",
  appUrl: "https://example.com",
  sessionId: "batch-2026-09-25",
  timeoutMs: 30_000,
});
export const direct = new DirectJevTransport({ apiKey: process.env.TYPESAFE_API_KEY });
void decide({
  input: "x",
  question: { type: "noul", instructions: "q" },
  provider: { transport, apiKey: process.env.OPENROUTER_API_KEY },
});
`,
    everyInputOption: `
import { decide, JevProviderError, serializeResult } from "../src/index.js";
import type { JevInfiniteCTXEvent, NativeJevRequest, NativeJevUsage, Tokenizer } from "../src/index.js";

declare function maybe<T>(): T | undefined;

void decide({
  input: "x",
  question: { type: "noul", instructions: "q", criteria: maybe<{ true: string; false: string }>() },
  aggregation: maybe<"mean">(),
  chunking: {
    overlap: maybe<number>(),
    maxStateTokens: maybe<number>(),
    contextSafetyReserve: maybe<number>(),
    protocolReserve: maybe<number>(),
    preferNaturalBoundaries: maybe<boolean>(),
    maxChunks: maybe<number>(),
  },
  confidence: {
    method: maybe<"none">(),
    cap: maybe<number>(),
    lambda: maybe<number>(),
    agreementFloor: maybe<number>(),
    agreementExponent: maybe<number>(),
  },
  execution: {
    maxConcurrency: maybe<number>(),
    retries: maybe<number>(),
    retryBaseDelayMs: maybe<number>(),
    retryMaxDelayMs: maybe<number>(),
    maxRechunks: maybe<number>(),
    rechunkShrinkFactor: maybe<number>(),
  },
  provider: { transport: maybe<"direct">(), model: maybe<string>(), apiKey: maybe<string>() },
  tokenizer: maybe<Tokenizer>(),
  signal: maybe<AbortSignal>(),
  onEvent: maybe<(event: JevInfiniteCTXEvent) => void>(),
}).then((result) => serializeResult(result, { includeChunkResults: maybe<boolean>(), includeRawAnswers: maybe<boolean>() }));

export const request: NativeJevRequest = { model: "m", state: "s", questions: {}, signal: maybe<AbortSignal>() };
export const usage: NativeJevUsage = {
  inputTokens: maybe<number>(),
  outputTokens: maybe<number>(),
  costUsd: maybe<number>(),
};
export const error = new JevProviderError("failed", {
  kind: "server",
  status: maybe<number>(),
  retryAfterMs: maybe<number>(),
  retryable: maybe<boolean>(),
  body: maybe<string>(),
});
`,
  };
  let errors: Record<string, string[]> = {};
  beforeAll(() => {
    errors = typeErrors(SNIPPETS, { exactOptionalPropertyTypes: true });
  }, 60_000);

  it.each(Object.keys(SNIPPETS))("accepts an explicit undefined in the %s snippet", (name) => {
    expect(errors[name]).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Package build and tarball
// ---------------------------------------------------------------------------

describe("package build and tarball", () => {
  it("packs a fresh checkout (no dist/) with the built library, by building first", () => {
    const dir = freshCheckout();
    try {
      const files = packedFiles(dir);
      expect(files).toContain("dist/index.js");
      expect(files).toContain("dist/index.d.ts");
      expect(files).toContain("LICENSE");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 120_000);

  it("ships the MIT license text that package.json declares", () => {
    const pkg = JSON.parse(readFileSync(path.join(repoRoot, "package.json"), "utf8")) as { license?: string };
    expect(pkg.license).toBe("MIT");
    const text = readFileSync(path.join(repoRoot, "LICENSE"), "utf8");
    expect(text).toMatch(/^MIT License\n/);
    expect(text).toMatch(/Permission is hereby granted, free of charge/);
  });

  it("ships the source every emitted .map points at, or embeds it", () => {
    const outDir = mkdtempSync(path.join(tmpdir(), "jev-build-"));
    try {
      const broken: string[] = [];
      for (const mapFile of buildInto(outDir).filter((file) => file.endsWith(".map"))) {
        const map = JSON.parse(readFileSync(mapFile, "utf8")) as {
          sources: string[];
          sourceRoot?: string;
          sourcesContent?: Array<string | null>;
        };
        map.sources.forEach((source, i) => {
          if (typeof map.sourcesContent?.[i] === "string") return;
          // Relative to the map, a source resolves to the real file it came from (e.g. src/index.ts).
          const absolute = path.resolve(path.dirname(mapFile), map.sourceRoot ?? "", source);
          const target = path.relative(repoRoot, absolute);
          if (!isPublished(target)) broken.push(`dist/${path.relative(outDir, mapFile)} -> ${target} (not published)`);
        });
      }
      expect(broken).toEqual([]);
    } finally {
      rmSync(outDir, { recursive: true, force: true });
    }
  }, 120_000);
});

// ---------------------------------------------------------------------------
// resolveTransport
// ---------------------------------------------------------------------------

describe("resolveTransport", () => {
  it("builds the OpenRouter transport by default and for 'openrouter'", () => {
    vi.stubEnv("OPENROUTER_API_KEY", "env-key");
    expect(resolveTransport(undefined)).toBeInstanceOf(OpenRouterJevTransport);
    expect(resolveTransport({})).toBeInstanceOf(OpenRouterJevTransport);
    expect(resolveTransport({ transport: "openrouter", apiKey: "sk-1" })).toBeInstanceOf(OpenRouterJevTransport);
  });

  it("builds the direct transport for 'direct'", () => {
    const transport = resolveTransport({ transport: "direct", apiKey: "ts-1" });
    expect(transport).toBeInstanceOf(DirectJevTransport);
    expect(transport.name).toBe("direct");
  });

  it("returns a custom transport object unchanged", () => {
    const mock = new MockTransport();
    expect(resolveTransport({ transport: mock })).toBe(mock);
  });

  it("throws JevValidationError when a built-in transport has no API key", () => {
    vi.stubEnv("OPENROUTER_API_KEY", "");
    vi.stubEnv("TYPESAFE_API_KEY", "");
    expect(() => resolveTransport({ transport: "openrouter" })).toThrow(JevValidationError);
    expect(() => resolveTransport({ transport: "direct" })).toThrow(JevValidationError);
  });

  it("rejects unknown transport names and objects that are not transports", () => {
    expect(() => resolveTransport({ transport: "chat" as never })).toThrow(JevValidationError);
    expect(() => resolveTransport({ transport: { name: "x" } as never })).toThrow(JevValidationError);
  });
});

// ---------------------------------------------------------------------------
// End to end, multi-chunk
// ---------------------------------------------------------------------------

describe("decide: choice across many chunks", () => {
  it("sends every chunk's exact state with the question and aggregates by unique-token weight", async () => {
    const mock = new MockTransport({ answer: (state) => topicAnswer(state), responseModel: "mock-jev-20260917" });
    const result = await decide(request(CHOICE, TOPIC_INPUT, mock));

    const chunks = result.chunks.results;
    expect(result.type).toBe("choice");
    expect(result.chunks.count).toBeGreaterThan(5);
    expect(chunks).toHaveLength(result.chunks.count);
    expect(mock.requests).toHaveLength(result.chunks.count);

    // Requests: the configured model, the question under QUESTION_KEY, the chunk's exact text.
    for (const req of mock.requests) {
      expect(req.model).toBe("mock-jev");
      expect(Object.keys(req.questions)).toEqual([QUESTION_KEY]);
      expect(req.questions[QUESTION_KEY]).toBe(CHOICE);
      expect(req.signal).toBeInstanceOf(AbortSignal);
    }
    expect(new Set(mock.states)).toEqual(new Set(chunks.map((c) => stateOf(TOPIC_INPUT, c))));

    // spec 6.4 weights and spec 9.2 weighted mean, recomputed from each chunk's own state.
    const uniqueSum = chunks.reduce((sum, c) => sum + c.uniqueTokens, 0);
    chunks.forEach((c) => expect(c.weight).toBeCloseTo(c.uniqueTokens / uniqueSum, 12));
    const perChunk = chunks.map((c) => topicDistribution(stateOf(TOPIC_INPUT, c)));
    const weights = chunks.map((c) => c.weight);
    const expected = weightedMean(perChunk, weights);
    expect(Object.keys(result.probabilities)).toEqual(["tree", "rock", "other"]);
    expect(result.probabilities.tree).toBeCloseTo(expected[0] as number, 12);
    expect(result.probabilities.rock).toBeCloseTo(expected[1] as number, 12);
    expect(result.probabilities.other).toBeCloseTo(expected[2] as number, 12);
    expect(result.choice).toBe((expected[0] as number) >= (expected[1] as number) ? "tree" : "rock");

    // spec 10 agreement, spec 11 base confidence and C3.
    const tvs = perChunk.map((p) => tv(p, expected));
    const agreement = 1 - tvs.reduce((sum, t, i) => sum + (weights[i] as number) * t, 0);
    const base = perChunk.reduce((sum, p, i) => sum + (weights[i] as number) * Math.max(...p), 0);
    chunks.forEach((c, i) => expect(c.totalVariation).toBeCloseTo(tvs[i] as number, 12));
    expect(result.confidence.agreement).toBeCloseTo(agreement, 12);
    expect(result.confidence.source).toBe("jev");
    expect(result.confidence.base).toBeCloseTo(base, 12);
    expect(result.chunks.effectiveCount).toBeCloseTo(nEffByHand(chunks), 12);
    expect(result.confidence.adjusted).toBeCloseTo(c3ByHand(base, agreement, nEffByHand(chunks)), 12);
    expect(result.confidence.adjustment).toBeCloseTo(result.confidence.adjusted - result.confidence.base, 15);
    expect(result.confidence.method).toBe("c3-v1");

    expect(result.model).toBe("mock-jev-20260917");
    expect(result.provider).toBe("mock");
    expect(result.aggregation).toEqual({ method: "weighted_mean", degenerate: false });
    expect(result.chunks.overlap).toBe(0.05);
    expect(result.chunks.stateTokenBudget).toBe(MIN_STATE_TOKENS);
    expect(result.chunks.overlapTokens).toBe(Math.floor(MIN_STATE_TOKENS * 0.05));
  });

  it("returns chunk results in chunk order with raw answers, confidences, and response metadata", async () => {
    const mock = new MockTransport({ answer: (state) => topicAnswer(state) });
    const result = await decide(request(CHOICE, TOPIC_INPUT, mock));
    result.chunks.results.forEach((chunk, i) => {
      expect(chunk.index).toBe(i);
      const state = stateOf(TOPIC_INPUT, chunk);
      expect(chunk.answer).toEqual(topicAnswer(state));
      expect(chunk.jevConfidence).toBe(Math.max(...topicDistribution(state)));
      expect(chunk.attempts).toBe(1);
      expect(chunk.usage).toEqual({ inputTokens: state.length, outputTokens: 1 });
      expect(chunk.model).toBe("mock-jev");
      expect(chunk.responseId).toMatch(/^mock-\d+$/);
      expect(chunk.estimatedTokens).toBe(defaultTokenizer.count(state));
      if (i > 0) {
        const previous = result.chunks.results[i - 1] as ChunkResult;
        expect(chunk.start).toBeGreaterThan(previous.start);
        expect(chunk.start).toBeLessThanOrEqual(previous.end);
      }
    });
  });

  it("falls back to a derived base confidence when any chunk omits Jev's confidence", async () => {
    const mock = new MockTransport({
      answer: (state, _q, callIndex) => {
        const [tree, rock, other] = topicDistribution(state);
        return choiceAnswer({ tree, rock, other }, callIndex === 0 ? undefined : 0.9);
      },
    });
    const result = await decide(request(CHOICE, TOPIC_INPUT, mock));
    const p = Object.values(result.probabilities);
    const entropy = -p.reduce((h, x) => (x > 0 ? h + x * Math.log(x) : h), 0);
    expect(result.confidence.source).toBe("derived");
    expect(result.confidence.base).toBeCloseTo(1 - entropy / Math.log(3), 12);
    expect(result.chunks.results[0]?.jevConfidence).toBeUndefined();
  });

  it.each(["mean", "median", "min", "max"] as const)("applies the %s aggregation", async (method) => {
    const mock = new MockTransport({ answer: (state) => topicAnswer(state) });
    const result = await decide(request(CHOICE, TOPIC_INPUT, mock, { aggregation: method }));
    const distributions = result.chunks.results.map((c) => ({
      labels: ["tree", "rock", "other"],
      probs: topicDistribution(stateOf(TOPIC_INPUT, c)),
    }));
    const expected = aggregate(distributions, result.chunks.results.map((c) => c.weight), method);
    expect(result.aggregation).toEqual({ method, degenerate: false });
    expect(Object.values(result.probabilities)).toEqual(expected.distribution.probs);
  });

  it("reports a degenerate aggregate (uniform, earliest label wins the tie) instead of hiding it", async () => {
    const mock = new MockTransport({
      answer: (_s, _q, callIndex) =>
        choiceAnswer(callIndex % 2 === 0 ? { tree: 1, rock: 0, other: 0 } : { tree: 0, rock: 1, other: 0 }, 1),
    });
    const result = await decide(request(CHOICE, TOPIC_INPUT, mock, { aggregation: "min" }));
    expect(result.aggregation.degenerate).toBe(true);
    expect(result.probabilities).toEqual({ tree: 1 / 3, rock: 1 / 3, other: 1 / 3 });
    expect(result.choice).toBe("tree");
  });

  it("reports adjusted = base when confidence.method is 'none', with components still computed", async () => {
    const mock = new MockTransport({ answer: () => choiceAnswer({ tree: 0.9, rock: 0.1, other: 0 }, 0.8) });
    const result = await decide(request(CHOICE, TOPIC_INPUT, mock, { confidence: { method: "none" } }));
    expect(result.confidence.method).toBe("none");
    expect(result.confidence.adjusted).toBe(result.confidence.base);
    expect(result.confidence.adjustment).toBe(0);
    expect(result.confidence.components.saturation).toBeGreaterThan(0);
    expect(result.confidence.components.gate).toBeCloseTo(1, 12);
  });
});

describe("decide: choice ties", () => {
  const TIE: ChoiceQuestion<"a" | "b" | "c"> = {
    type: "choice",
    instructions: "Which one?",
    criteria: { a: "A.", b: "B.", c: "C." },
  };
  type Answer = Readonly<Record<"a" | "b" | "c", number>>;
  // a and b tie at exactly 0.4 in decimal. In floating point the column sums
  // depend on chunk order, and b comes out 1 ulp ahead in one of the orders.
  const MEAN_TIE: readonly Answer[] = [
    { a: 0.1, b: 0.7, c: 0.2 },
    { a: 0.4, b: 0.4, c: 0.2 },
    { a: 0.7, b: 0.1, c: 0.2 },
  ];
  const WEIGHTED_TIE: readonly Answer[] = [
    { a: 0.2, b: 0.6, c: 0.2 },
    { a: 0.4, b: 0.4, c: 0.2 },
    { a: 0.6, b: 0.2, c: 0.2 },
  ];

  /** Three equal-weight chunks (one token per character, no overlap, hard cuts); chunk i gets perChunk[i]. */
  async function decideTie(perChunk: readonly Answer[], aggregation: AggregationMethod) {
    const mock = new MockTransport({ answer: (_s, _q, callIndex) => choiceAnswer({ ...perChunk[callIndex] }) });
    const result = await decide(
      request(TIE, "x".repeat(3 * MIN_STATE_TOKENS), mock, {
        aggregation,
        tokenizer: createTokenizer((text) => text.length, "chars"),
        chunking: { maxStateTokens: MIN_STATE_TOKENS, overlap: 0, preferNaturalBoundaries: false },
        // Sequential, so the call index is the chunk index.
        execution: { maxConcurrency: 1, retryBaseDelayMs: 0 },
      }),
    );
    expect(result.chunks.results.map((chunk) => chunk.weight)).toEqual([1 / 3, 1 / 3, 1 / 3]);
    return result;
  }

  it.each([
    ["mean", MEAN_TIE],
    ["weighted_mean", WEIGHTED_TIE],
  ] as const)("gives a %s tie to the earliest label in either chunk order", async (aggregation, perChunk) => {
    const results = [await decideTie(perChunk, aggregation), await decideTie([...perChunk].reverse(), aggregation)];
    for (const result of results) {
      expect(Math.abs(result.probabilities.a - result.probabilities.b)).toBeLessThan(ARGMAX_TIE_TOLERANCE);
      expect(result.probabilities.c).toBeCloseTo(0.2, 12);
    }
    // Float noise favors b in one order, so a strict comparison would return b there.
    expect(results.some((result) => result.probabilities.b > result.probabilities.a)).toBe(true);
    expect(results.map((result) => result.choice)).toEqual(["a", "a"]);
  });

  it("still picks a later label that leads by more than the tie tolerance", async () => {
    const lead = 1e-9;
    const result = await decideTie(Array.from({ length: 3 }, () => ({ a: 0.4, b: 0.4 + lead, c: 0.2 - lead })), "mean");
    expect(result.probabilities.b - result.probabilities.a).toBeGreaterThan(ARGMAX_TIE_TOLERANCE);
    expect(result.choice).toBe("b");
  });
});

describe("decide: C3 and a short final chunk", () => {
  it("does not become more confident when a tiny contradicting tail is appended", async () => {
    // Chunks containing "y" disagree completely with the rest.
    const answer = (state: string): NativeChoiceAnswer =>
      state.includes("y")
        ? choiceAnswer({ tree: 0, rock: 1, other: 0 }, 0.6)
        : choiceAnswer({ tree: 0.9, rock: 0.1, other: 0 }, 0.6);
    const run = (input: string) =>
      decide(
        request(CHOICE, input, new MockTransport({ answer }), {
          tokenizer: createTokenizer((text) => text.length, "chars"),
          chunking: { maxStateTokens: 1000, overlap: 0, preferNaturalBoundaries: false },
        }),
      );

    const alone = await run("x".repeat(1000));
    const withTail = await run("x".repeat(1000) + "y".repeat(10));

    expect(alone.chunks.count).toBe(1);
    expect(withTail.chunks.count).toBe(2);
    // The 10-token tail adds 1% of a chunk to N_eff, not a whole chunk.
    expect(withTail.chunks.effectiveCount).toBeCloseTo(1.01, 12);
    expect(withTail.confidence.adjusted - alone.confidence.adjusted).toBeLessThan(0.01);
  });
});

describe("decide: score across many chunks", () => {
  it("returns Σ j·P_agg[j], its normalized form, level probabilities, and the caller's legend", async () => {
    const mock = new MockTransport({ answer: (state) => scoreAnswer(scoreDistribution(state), 0.6) });
    const result = await decide(request(SCORE, TOPIC_INPUT, mock));
    const chunks = result.chunks.results;
    const expected = weightedMean(
      chunks.map((c) => scoreDistribution(stateOf(TOPIC_INPUT, c))),
      chunks.map((c) => c.weight),
    );
    const score = expected.reduce((sum, p, level) => sum + level * p, 0);

    expect(result.type).toBe("score");
    expect(chunks.length).toBeGreaterThan(1);
    expect(Object.keys(result.probabilities)).toEqual(["0", "1", "2", "3"]);
    expected.forEach((p, level) => expect(result.probabilities[String(level)]).toBeCloseTo(p, 12));
    expect(result.score).toBeCloseTo(score, 12);
    expect(result.normalizedScore).toBeCloseTo(score / 3, 12);
    expect(result.legend).toEqual({ "0": "None", "1": "Some", "2": "Most", "3": "All" });
    expect(result.confidence.source).toBe("jev");
    expect(result.confidence.base).toBeCloseTo(0.6, 12);
  });
});

describe("decide: noul across many chunks", () => {
  it("returns the weighted yes probability and a derived certainty |2·noul − 1|", async () => {
    const mock = new MockTransport({ answer: (state) => noulAnswer(1 - rockShare(state)) });
    const result = await decide(request(NOUL, TOPIC_INPUT, mock));
    const chunks = result.chunks.results;
    const noul = chunks.reduce((sum, c) => sum + c.weight * (1 - rockShare(stateOf(TOPIC_INPUT, c))), 0);

    expect(result.type).toBe("noul");
    expect(chunks.length).toBeGreaterThan(1);
    expect(result.noul).toBeCloseTo(noul, 12);
    expect(result.confidence.source).toBe("derived");
    expect(result.confidence.base).toBeCloseTo(Math.abs(2 * noul - 1), 12);
    for (const chunk of chunks) {
      expect(chunk.jevConfidence).toBeUndefined();
      expect(Object.keys(chunk.probabilities)).toEqual(["no", "yes"]);
    }
    expect("probabilities" in result).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Single chunk (spec 15)
// ---------------------------------------------------------------------------

describe("decide: input that fits in one chunk", () => {
  const SHORT = "The old oak stands beside a granite wall.";

  it("returns Jev's native choice and confidence untouched", async () => {
    const mock = new MockTransport({ script: [choiceAnswer({ tree: 0.5, rock: 0.5, other: 0 }, 0.61, "rock")] });
    const result = await decide(request(CHOICE, SHORT, mock));
    expect(mock.requests).toHaveLength(1);
    expect(mock.states).toEqual([SHORT]);
    // Native pick, even though argmax of the probabilities would break the tie towards "tree".
    expect(result.choice).toBe("rock");
    expect(result.probabilities).toEqual({ tree: 0.5, rock: 0.5, other: 0 });
    expect(result.confidence).toMatchObject({
      base: 0.61,
      adjusted: 0.61,
      adjustment: 0,
      source: "jev",
      agreement: 1,
    });
    expect(result.chunks).toMatchObject({ count: 1, effectiveCount: 1 });
    expect(result.chunks.results[0]).toMatchObject({
      index: 0,
      start: 0,
      end: SHORT.length,
      weight: 1,
      boundary: "end",
      totalVariation: 0,
    });
  });

  it("returns Jev's native score, not the recomputed expectation", async () => {
    const mock = new MockTransport({ script: [scoreAnswer([0.1, 0.4, 0.5, 0], 0.7, 1.37)] });
    const result = await decide(request(SCORE, SHORT, mock));
    expect(result.score).toBe(1.37);
    expect(result.normalizedScore).toBeCloseTo(1.37 / 3, 15);
    expect(result.probabilities).toEqual({ "0": 0.1, "1": 0.4, "2": 0.5, "3": 0 });
    expect(result.confidence).toMatchObject({ base: 0.7, adjusted: 0.7, adjustment: 0 });
  });

  it("clamps a native score that is a hair outside the level range", async () => {
    const mock = new MockTransport({ script: [scoreAnswer([0, 0, 0, 1], 0.9, 3 + 5e-7)] });
    const result = await decide(request(SCORE, SHORT, mock));
    expect(result.score).toBe(3);
    expect(result.normalizedScore).toBe(1);
  });

  it("returns Jev's native noul with a derived, unadjusted certainty", async () => {
    const mock = new MockTransport({ script: [noulAnswer(0.83)] });
    const result = await decide(request(NOUL, SHORT, mock));
    expect(result.noul).toBe(0.83);
    expect(result.confidence.source).toBe("derived");
    expect(result.confidence.base).toBeCloseTo(0.66, 12);
    expect(result.confidence.adjusted).toBe(result.confidence.base);
    expect(result.confidence.adjustment).toBe(0);
    expect(result.confidence.agreement).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Retries and permanent failures (spec 7)
// ---------------------------------------------------------------------------

describe("decide: retries", () => {
  it("retries a 429 and then succeeds, reporting the retry", async () => {
    const events: JevInfiniteCTXEvent[] = [];
    const mock = new MockTransport({
      script: [providerError("rate_limit", 429, { retryAfterMs: 0 })],
      answer: (state) => topicAnswer(state),
    });
    const result = await decide(request(CHOICE, TOPIC_INPUT, mock, { onEvent: (e) => events.push(e) }));

    const count = result.chunks.count;
    expect(result.chunks.results.map((c) => c.attempts)).toEqual([2, ...Array<number>(count - 1).fill(1)]);
    expect(result.usage).toMatchObject({ requests: count + 1, retries: 1, rechunks: 0 });
    expect(eventsOf(events, "chunk.retry")).toEqual([
      { type: "chunk.retry", chunkIndex: 0, attempt: 1, delayMs: 0, errorKind: "rate_limit", status: 429 },
    ]);
    expect(eventsOf(events, "decision.completed")[0]?.retryCount).toBe(1);
  });

  it("retries every retryable kind until it succeeds", async () => {
    const mock = new MockTransport({
      script: [
        providerError("server", 500),
        providerError("overloaded", 529),
        providerError("timeout"),
        providerError("network"),
      ],
      answer: (state) => topicAnswer(state),
    });
    const result = await decide(
      request(CHOICE, "A short text about an oak.", mock, { execution: { retries: 4, retryBaseDelayMs: 0 } }),
    );
    expect(result.chunks.results[0]?.attempts).toBe(5);
    expect(result.usage).toMatchObject({ requests: 5, retries: 4 });
  });

  it("fails with JevChunkFailedError once retries are exhausted", async () => {
    const mock = new MockTransport({ answer: () => providerError("server", 500) });
    const error = await decide(
      request(CHOICE, "A short text.", mock, { execution: { retries: 2, retryBaseDelayMs: 0 } }),
    ).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(JevChunkFailedError);
    expect(error).toMatchObject({ chunkIndex: 0, attempts: 3 });
    expect((error as JevChunkFailedError).cause).toMatchObject({ kind: "server", status: 500 });
    expect(mock.requests).toHaveLength(3);
  });
});

describe("decide: permanent failures fail closed", () => {
  it("fails the whole call on a 401 without retrying, and returns no result", async () => {
    const events: JevInfiniteCTXEvent[] = [];
    const mock = new MockTransport({
      latencyMs: (_s, callIndex) => (callIndex === 2 ? 0 : 20),
      answer: (state, _q, callIndex) => (callIndex === 2 ? providerError("auth", 401) : topicAnswer(state)),
    });
    const error = await decide(request(CHOICE, TOPIC_INPUT, mock, { onEvent: (e) => events.push(e) })).catch(
      (e: unknown) => e,
    );

    expect(error).toBeInstanceOf(JevChunkFailedError);
    expect(error).toMatchObject({ chunkIndex: 2, attempts: 1 });
    expect((error as JevChunkFailedError).cause).toMatchObject({ kind: "auth", status: 401, retryable: false });
    // In-flight siblings were cancelled and awaited; nothing new was scheduled.
    expect(mock.requests).toHaveLength(DEFAULTS.execution.maxConcurrency);
    expect(mock.requests.every((r) => r.signal?.aborted)).toBe(true);
    expect(mock.inFlight).toBe(0);
    expect(eventsOf(events, "decision.completed")).toHaveLength(0);
    expect(eventsOf(events, "decision.failed")).toEqual([
      expect.objectContaining({
        errorName: "JevChunkFailedError",
        errorKind: "auth",
        chunkIndex: 2,
        provider: "mock",
        model: "mock-jev",
        retryCount: 0,
        rechunkCount: 0,
      }),
    ]);
  });

  it.each([
    ["an unknown choice key", choiceAnswer({ tree: 1, rock: 0, other: 0 }, 0.9, "banana")],
    ["probabilities that do not sum to 1", choiceAnswer({ tree: 1, rock: 1, other: 0 }, 0.9, "tree")],
    ["a probability for an unknown key", choiceAnswer({ tree: 0.5, rock: 0.25, other: 0.25, lava: 0 }, 0.9)],
    ["missing probabilities", { type: "choice", choice: "tree", confidence: 0.9 } as NativeChoiceAnswer],
    ["a confidence above 1", choiceAnswer({ tree: 1, rock: 0, other: 0 }, 1.5)],
    ["the wrong answer type", noulAnswer(0.5)],
  ])("fails closed on a malformed answer: %s", async (_label, badAnswer) => {
    const mock = new MockTransport({
      answer: (state, _q, callIndex) => (callIndex === 1 ? badAnswer : topicAnswer(state)),
    });
    const error = await decide(request(CHOICE, TOPIC_INPUT, mock)).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(JevChunkFailedError);
    expect(error).toMatchObject({ chunkIndex: 1, attempts: 1 });
    expect((error as JevChunkFailedError).cause).toBeInstanceOf(JevResponseError);
  });

  it.each([
    ["a response without the question's answer", { model: "m", answers: {} }],
    ["a response whose answers is not an object", { model: "m", answers: null }],
    ["a response that is not an object", null],
  ])("fails closed on %s", async (_label, response) => {
    const transport: JevTransport = {
      name: "inline",
      defaultModel: "m",
      contextWindow: () => 32_000,
      decide: async () => response as unknown as NativeJevResponse,
    };
    const error = await decide(request(CHOICE, "Short.", transport)).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(JevChunkFailedError);
    expect((error as JevChunkFailedError).cause).toBeInstanceOf(JevResponseError);
  });

  it("does not retry an invalid_response provider error", async () => {
    const mock = new MockTransport({ answer: () => providerError("invalid_response", 200) });
    const error = await decide(request(CHOICE, "Short.", mock)).catch((e: unknown) => e);
    expect(error).toMatchObject({ name: "JevChunkFailedError", attempts: 1 });
    expect(mock.requests).toHaveLength(1);
  });

  it("treats any other thrown error as a permanent chunk failure without quoting its message", async () => {
    const failure = new TypeError("custom transport exploded while reading: the secret oak text");
    const mock = new MockTransport({ answer: () => failure });
    const error = await decide(request(CHOICE, "Short.", mock)).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(JevChunkFailedError);
    expect((error as JevChunkFailedError).cause).toBe(failure);
    expect((error as Error).message).toContain("TypeError");
    expect((error as Error).message).not.toContain("secret");
    expect(mock.requests).toHaveLength(1);
  });

  it("does not report a transport's spontaneous AbortError as a caller abort", async () => {
    const mock = new MockTransport({ answer: () => new JevAbortError("the transport gave up on its own") });
    const error = await decide(request(CHOICE, "Short.", mock)).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(JevChunkFailedError);
  });
});

// ---------------------------------------------------------------------------
// Context-limit shrink and re-chunk (spec 7, 15)
// ---------------------------------------------------------------------------

describe("decide: context-limit errors re-chunk the entire document", () => {
  const INPUT = makeText(1500, 21);

  it("shrinks the budget, re-plans, and re-runs every chunk until the provider accepts them", async () => {
    const events: JevInfiniteCTXEvent[] = [];
    const passStarts: number[] = [0];
    const mock = new MockTransport({ maxStateChars: 2000, answer: (state) => topicAnswer(state) });
    const result = await decide(
      request(CHOICE, INPUT, mock, {
        chunking: { maxStateTokens: 1000 },
        onEvent: (event) => {
          events.push(event);
          if (event.type === "rechunk") passStarts.push(mock.requests.length);
        },
      }),
    );

    const rechunks = eventsOf(events, "rechunk");
    expect(rechunks.length).toBeGreaterThanOrEqual(2);
    expect(result.usage.rechunks).toBe(rechunks.length);
    expect(rechunks[0]?.previousBudget).toBe(1000);
    rechunks.forEach((event, i) => {
      expect(event.newBudget).toBeLessThan(event.previousBudget);
      expect(event.newBudget).toBeLessThanOrEqual(Math.floor(event.previousBudget * 0.75));
      expect(event.status).toBe(413);
      if (i > 0) expect(event.previousBudget).toBe(rechunks[i - 1]?.newBudget);
    });
    expect(result.chunks.stateTokenBudget).toBe(rechunks.at(-1)?.newBudget);

    // The final pass covered the whole document again, from the first character.
    const finalPass = mock.requests.slice(passStarts.at(-1));
    expect(finalPass).toHaveLength(result.chunks.count);
    expect(finalPass.every((r) => r.state.length <= 2000)).toBe(true);
    const chunks = result.chunks.results.map((c) => ({ ...c, state: stateOf(INPUT, c) }));
    expect(new Set(finalPass.map((r) => r.state))).toEqual(new Set(chunks.map((c) => c.state)));
    expect(reassemble(chunks)).toBe(INPUT);
    expect(chunks[0]?.start).toBe(0);
    expect(chunks.at(-1)?.end).toBe(INPUT.length);
    for (const chunk of result.chunks.results) {
      expect(chunk.estimatedTokens).toBeLessThanOrEqual(result.chunks.stateTokenBudget);
    }
  });

  it("starts the shrink from the failed chunk's size, so a short input is split at once", async () => {
    const events: JevInfiniteCTXEvent[] = [];
    const input = makeText(600, 5);
    const tokens = defaultTokenizer.count(input);
    const mock = new MockTransport({ maxStateChars: 3000, answer: (state) => topicAnswer(state) });
    const result = await decide({
      input,
      question: CHOICE,
      provider: { transport: mock },
      onEvent: (event) => events.push(event),
    });
    const [rechunk] = eventsOf(events, "rechunk");
    expect(input.length).toBeGreaterThan(3000);
    expect(rechunk?.previousBudget).toBeGreaterThan(20_000);
    expect(rechunk?.newBudget).toBe(Math.floor(tokens * 0.75));
    expect(result.usage.rechunks).toBe(1);
    expect(result.chunks.count).toBeGreaterThan(1);
    expect(mock.requests).toHaveLength(1 + result.chunks.count);
  });

  it("throws JevContextBudgetError after maxRechunks passes", async () => {
    const events: JevInfiniteCTXEvent[] = [];
    const mock = new MockTransport({ maxStateChars: 100 });
    const error = await decide(
      request(CHOICE, INPUT, mock, {
        chunking: { maxStateTokens: 1000 },
        execution: { maxRechunks: 2 },
        onEvent: (event) => events.push(event),
      }),
    ).catch((e: unknown) => e);

    const rechunks = eventsOf(events, "rechunk");
    expect(error).toBeInstanceOf(JevContextBudgetError);
    expect(error).toMatchObject({ rechunks: 2, lastBudget: rechunks[1]?.newBudget });
    expect((error as JevContextBudgetError).cause).toMatchObject({ kind: "context_limit" });
    expect(rechunks).toHaveLength(2);
    expect(eventsOf(events, "decision.failed")).toEqual([
      expect.objectContaining({ errorName: "JevContextBudgetError", errorKind: "context_limit", rechunkCount: 2 }),
    ]);
  });

  it("stops before the budget would fall below MIN_STATE_TOKENS", async () => {
    const events: RechunkEvent[] = [];
    const mock = new MockTransport({ maxStateChars: 100 });
    const error = await decide(
      request(CHOICE, INPUT, mock, {
        chunking: { maxStateTokens: 1000 },
        execution: { maxRechunks: 50 },
        onEvent: (event) => {
          if (event.type === "rechunk") events.push(event);
        },
      }),
    ).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(JevContextBudgetError);
    const { lastBudget, rechunks } = error as JevContextBudgetError;
    expect(rechunks).toBe(events.length);
    expect(rechunks).toBeLessThan(50);
    expect(lastBudget).toBe(events.at(-1)?.newBudget);
    expect(lastBudget).toBeGreaterThanOrEqual(MIN_STATE_TOKENS);
    expect((error as Error).message).toContain(`cannot shrink below ${MIN_STATE_TOKENS} tokens`);
  });

  it("with maxRechunks 0 fails on the first context-limit error without re-chunking", async () => {
    const events: JevInfiniteCTXEvent[] = [];
    const mock = new MockTransport({ maxStateChars: 100 });
    const error = await decide(
      request(CHOICE, INPUT, mock, { execution: { maxRechunks: 0 }, onEvent: (event) => events.push(event) }),
    ).catch((e: unknown) => e);
    expect(error).toMatchObject({ name: "JevContextBudgetError", rechunks: 0, lastBudget: MIN_STATE_TOKENS });
    expect(eventsOf(events, "rechunk")).toHaveLength(0);
  });

  it("fails with JevContextBudgetError when a re-chunk pass would exceed maxChunks", async () => {
    const mock = new MockTransport({ maxStateChars: 2000 });
    const error = await decide(
      request(CHOICE, INPUT, mock, { chunking: { maxStateTokens: 1000, maxChunks: 3 } }),
    ).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(JevContextBudgetError);
    expect((error as Error).message).toContain("maxChunks");
    expect(error).toMatchObject({ rechunks: 1 });
  });

  it("counts usage from every response received, including passes that were discarded", async () => {
    const mock = new MockTransport({
      answer: (state, _q, callIndex) =>
        callIndex === 1 ? providerError("context_limit", 400) : topicAnswer(state),
      usage: (state) => ({ inputTokens: state.length, outputTokens: 2, costUsd: 0.001 }),
    });
    const result = await decide(
      request(CHOICE, INPUT, mock, {
        chunking: { maxStateTokens: 1000 },
        execution: { maxConcurrency: 1 },
      }),
    );
    const answered = mock.requests.filter((_r, i) => i !== 1);
    expect(result.usage.rechunks).toBe(1);
    expect(result.usage.requests).toBe(mock.requests.length);
    // Pass 1 answered chunk 0 before chunk 1 hit the limit; that response still counts.
    expect(answered).toHaveLength(result.chunks.count + 1);
    expect(result.usage.inputTokens).toBe(answered.reduce((sum, r) => sum + r.state.length, 0));
    expect(result.usage.outputTokens).toBe(2 * answered.length);
    expect(result.usage.costUsd).toBeCloseTo(0.001 * answered.length, 12);
  });
});

// ---------------------------------------------------------------------------
// Abort, concurrency, ordering
// ---------------------------------------------------------------------------

describe("decide: caller abort", () => {
  it("cancels in-flight requests and rejects with JevAbortError", async () => {
    const events: JevInfiniteCTXEvent[] = [];
    const controller = new AbortController();
    const mock = new MockTransport({ latencyMs: 5_000 });
    const started = performance.now();
    setTimeout(() => controller.abort(new Error("user cancelled")), 20);
    const error = await decide(
      request(CHOICE, TOPIC_INPUT, mock, { signal: controller.signal, onEvent: (event) => events.push(event) }),
    ).catch((e: unknown) => e);

    expect(error).toBeInstanceOf(JevAbortError);
    expect((error as Error).name).toBe("AbortError");
    expect(performance.now() - started).toBeLessThan(2_000);
    expect(mock.requests.length).toBeGreaterThan(0);
    expect(mock.requests.every((r) => r.signal?.aborted)).toBe(true);
    expect(mock.inFlight).toBe(0);
    expect(eventsOf(events, "decision.failed")).toEqual([expect.objectContaining({ errorName: "AbortError" })]);
  });

  it("rejects an already-aborted call without touching the transport", async () => {
    const mock = new MockTransport();
    const error = await decide(request(CHOICE, TOPIC_INPUT, mock, { signal: AbortSignal.abort() })).catch(
      (e: unknown) => e,
    );
    expect(error).toBeInstanceOf(JevAbortError);
    expect(mock.contextWindowCalls).toHaveLength(0);
    expect(mock.requests).toHaveLength(0);
  });

  it("does not wait for a slow context-window lookup once aborted", async () => {
    const controller = new AbortController();
    const mock = new MockTransport({ contextWindow: () => new Promise((resolve) => setTimeout(resolve, 5_000)) });
    const started = performance.now();
    setTimeout(() => controller.abort(), 10);
    await expect(decide(request(CHOICE, "Short.", mock, { signal: controller.signal }))).rejects.toBeInstanceOf(
      JevAbortError,
    );
    expect(performance.now() - started).toBeLessThan(2_000);
    expect(mock.requests).toHaveLength(0);
  });

  it("cancels a retry backoff", async () => {
    const controller = new AbortController();
    const mock = new MockTransport({ answer: () => providerError("rate_limit", 429) });
    const started = performance.now();
    setTimeout(() => controller.abort(), 20);
    await expect(
      decide(
        request(CHOICE, "Short.", mock, {
          signal: controller.signal,
          execution: { retryBaseDelayMs: 60_000, retryMaxDelayMs: 60_000 },
        }),
      ),
    ).rejects.toBeInstanceOf(JevAbortError);
    expect(performance.now() - started).toBeLessThan(2_000);
    expect(mock.requests).toHaveLength(1);
  });
});

describe("decide: concurrency and ordering", () => {
  it.each([1, 2, 4])("never has more than maxConcurrency = %i requests in flight", async (maxConcurrency) => {
    const mock = new MockTransport({ latencyMs: 3, answer: (state) => topicAnswer(state) });
    const result = await decide(request(CHOICE, TOPIC_INPUT, mock, { execution: { maxConcurrency } }));
    expect(result.chunks.count).toBeGreaterThan(maxConcurrency);
    expect(mock.maxInFlight).toBe(maxConcurrency);
  });

  it("keeps chunk results in document order when responses arrive in reverse", async () => {
    const yes = (state: string): number => 0.05 + (state.length % 89) / 100;
    // Replies are gated rather than timed: staggered timer latencies decide
    // the order by wall-clock timer creation, which CPU contention can invert.
    // Every request starts before any reply (concurrency exceeds the chunk
    // count), so all answers are pending when the first macrotask runs; they
    // are then released last-to-first, one macrotask apart.
    const release: Array<() => void> = [];
    const releaseInReverse = async (): Promise<void> => {
      for (let i = release.length - 1; i >= 0; i--) {
        release[i]?.();
        await new Promise((resolve) => setImmediate(resolve));
      }
    };
    const mock = new MockTransport({
      answer: (state, _question, callIndex) => {
        if (callIndex === 0) setTimeout(() => void releaseInReverse(), 0);
        return new Promise((resolve) => {
          release[callIndex] = () => resolve(noulAnswer(yes(state)));
        });
      },
    });
    const result = await decide(request(NOUL, TOPIC_INPUT, mock, { execution: { maxConcurrency: 64 } }));
    const count = result.chunks.count;
    expect(count).toBeGreaterThan(2);
    expect(mock.completionOrder).toEqual(Array.from({ length: count }, (_, i) => count - 1 - i));
    result.chunks.results.forEach((chunk, i) => {
      expect(chunk.index).toBe(i);
      expect(chunk.probabilities.yes).toBe(yes(stateOf(TOPIC_INPUT, chunk)));
    });
  });
});

// ---------------------------------------------------------------------------
// Usage, events, context window, model
// ---------------------------------------------------------------------------

describe("decide: usage", () => {
  it("sums provider usage across chunks and leaves cost undefined when none is reported", async () => {
    const mock = new MockTransport({ answer: (state) => topicAnswer(state) });
    const result = await decide(request(CHOICE, TOPIC_INPUT, mock));
    expect(result.usage.inputTokens).toBe(mock.states.reduce((sum, s) => sum + s.length, 0));
    expect(result.usage.outputTokens).toBe(result.chunks.count);
    expect(result.usage.costUsd).toBeUndefined();
    expect(result.usage.inputTokensEstimated).toBe(defaultTokenizer.count(TOPIC_INPUT));
    expect(result.usage.requests).toBe(result.chunks.count);
    expect(result.usage.elapsedMs).toBeGreaterThanOrEqual(0);
  });

  it("ignores malformed usage fields from a custom transport", async () => {
    const mock = new MockTransport({
      usage: () => ({ inputTokens: Number.NaN, outputTokens: -1, costUsd: 0.5 }),
    });
    const result = await decide(request(CHOICE, "Short.", mock));
    expect(result.usage).toMatchObject({ inputTokens: 0, outputTokens: 0, costUsd: 0.5 });
    expect(result.chunks.results[0]?.usage).toEqual({ costUsd: 0.5 });
  });

  it("omits per-chunk usage when the provider reports none", async () => {
    const mock = new MockTransport({ usage: null });
    const result = await decide(request(CHOICE, "Short.", mock));
    expect(result.chunks.results[0]?.usage).toBeUndefined();
    expect(result.usage).toMatchObject({ inputTokens: 0, outputTokens: 0, costUsd: undefined });
  });
});

describe("decide: events", () => {
  const SENTINEL = "zqxsentinelwordzqx";
  const INPUT = makeText(1500, 31, [SENTINEL]);

  it("never carries source text, on success or failure", async () => {
    const events: JevInfiniteCTXEvent[] = [];
    const onEvent = (event: JevInfiniteCTXEvent): void => {
      events.push(event);
    };
    expect(countWord(INPUT, SENTINEL)).toBeGreaterThan(20);

    // The first request that fits the provider's limit is rate-limited once.
    let rateLimited = false;
    const retrying = new MockTransport({
      maxStateChars: 3000,
      answer: (state) => {
        if (rateLimited) return topicAnswer(state);
        rateLimited = true;
        return providerError("rate_limit", 429, { retryAfterMs: 0 });
      },
    });
    const result = await decide(request(CHOICE, INPUT, retrying, { chunking: { maxStateTokens: 1000 }, onEvent }));
    // A custom transport that puts the state in its own error message.
    const leaky = new MockTransport({ answer: (state) => new TypeError(`cannot process ${state}`) });
    const failure = await decide(request(CHOICE, INPUT, leaky, { onEvent })).catch((e: unknown) => e);

    expect(new Set(events.map((e) => e.type))).toEqual(
      new Set(["chunk.retry", "rechunk", "decision.completed", "decision.failed"]),
    );
    expect(JSON.stringify(events)).not.toContain(SENTINEL);
    expect(JSON.stringify(result)).not.toContain(SENTINEL);
    expect((failure as Error).message).not.toContain(SENTINEL);
  });

  it("emits a completed event that mirrors the result", async () => {
    const events: JevInfiniteCTXEvent[] = [];
    const mock = new MockTransport({
      answer: (state) => topicAnswer(state),
      usage: (state) => ({ inputTokens: state.length, outputTokens: 1, costUsd: 0.25 }),
    });
    const result = await decide(request(CHOICE, TOPIC_INPUT, mock, { onEvent: (e) => events.push(e) }));
    expect(events).toHaveLength(1);
    expect(events[0]).toEqual({
      type: "decision.completed",
      model: result.model,
      provider: "mock",
      questionType: "choice",
      inputTokensEstimated: result.usage.inputTokensEstimated,
      inputTokensActual: result.usage.inputTokens,
      chunkCount: result.chunks.count,
      effectiveChunkCount: result.chunks.effectiveCount,
      overlap: 0.05,
      stateTokenBudget: MIN_STATE_TOKENS,
      aggregationMethod: "weighted_mean",
      agreement: result.confidence.agreement,
      baseConfidence: result.confidence.base,
      adjustedConfidence: result.confidence.adjusted,
      confidenceSource: "jev",
      finalAnswer: result.choice,
      perChunkProbabilities: result.chunks.results.map((c) => c.probabilities),
      costUsd: result.usage.costUsd,
      latencyMs: result.usage.elapsedMs,
      retryCount: 0,
      rechunkCount: 0,
    });
  });

  it("reports the final answer for score and noul", async () => {
    const answers: Array<string | number> = [];
    const onEvent = (e: JevInfiniteCTXEvent): void => {
      if (e.type === "decision.completed") answers.push(e.finalAnswer);
    };
    const score = await decide(
      request(SCORE, "Short.", new MockTransport({ script: [scoreAnswer([0, 1, 0, 0], 0.9)] }), { onEvent }),
    );
    const noulMock = new MockTransport({ script: [noulAnswer(0.3)] });
    const noul = await decide(request(NOUL, "Short.", noulMock, { onEvent }));
    expect(answers).toEqual([score.score, noul.noul]);
  });

  it("keeps working when onEvent throws, and still surfaces the real failure", async () => {
    const throwing = (): void => {
      throw new Error("hook failure");
    };
    const ok = await decide(
      request(CHOICE, TOPIC_INPUT, new MockTransport({ script: [providerError("rate_limit", 429)] }), {
        onEvent: throwing,
      }),
    );
    expect(ok.type).toBe("choice");
    const failing = new MockTransport({ answer: () => providerError("auth", 401) });
    await expect(decide(request(CHOICE, "Short.", failing, { onEvent: throwing }))).rejects.toBeInstanceOf(
      JevChunkFailedError,
    );
  });

  it("swallows rejections from an async onEvent", async () => {
    const onEvent = async (): Promise<void> => {
      throw new Error("async hook failure");
    };
    const result = await decide(request(NOUL, "Short.", new MockTransport(), { onEvent }));
    expect(result.noul).toBe(0.9);
  });

  it("gives hooks copies, so mutating an event cannot change the result", async () => {
    const mock = new MockTransport({ answer: (state) => topicAnswer(state) });
    const result = await decide(
      request(CHOICE, TOPIC_INPUT, mock, {
        onEvent: (e) => {
          if (e.type === "decision.completed") {
            for (const p of e.perChunkProbabilities) p.tree = -1;
          }
        },
      }),
    );
    expect(result.chunks.results.every((c) => (c.probabilities.tree ?? -1) >= 0)).toBe(true);
  });
});

describe("decide: validation and budgeting", () => {
  it("rejects an invalid request before any provider call or event", async () => {
    const onEvent = vi.fn();
    const mock = new MockTransport();
    await expect(decide(request(CHOICE, "   \n\t ", mock, { onEvent }))).rejects.toBeInstanceOf(
      JevValidationError,
    );
    await expect(
      decide(request(CHOICE, "Text.", mock, { onEvent, chunking: { overlap: 0.9 } })),
    ).rejects.toThrow(/chunking\.overlap/);
    await expect(decide({ input: "Text.", question: { type: "choice" } } as never)).rejects.toBeInstanceOf(
      JevValidationError,
    );
    expect(mock.contextWindowCalls).toHaveLength(0);
    expect(mock.requests).toHaveLength(0);
    expect(onEvent).not.toHaveBeenCalled();
  });

  it("rejects a plan that exceeds maxChunks before calling Jev", async () => {
    const mock = new MockTransport();
    const error = await decide(
      request(CHOICE, TOPIC_INPUT, mock, { chunking: { maxStateTokens: 256, maxChunks: 2 } }),
    ).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(JevValidationError);
    expect((error as Error).message).toContain("maxChunks");
    expect(mock.requests).toHaveLength(0);
  });

  it("rejects a question too large for the context window, before calling Jev", async () => {
    const events: JevInfiniteCTXEvent[] = [];
    const mock = new MockTransport({ contextWindow: 2_000 });
    const huge: NoulQuestion = { type: "noul", instructions: makeText(2_000, 3) };
    await expect(
      decide({ input: "Text.", question: huge, provider: { transport: mock }, onEvent: (e) => events.push(e) }),
    ).rejects.toThrow(/too large for the context window/);
    expect(mock.requests).toHaveLength(0);
    expect(eventsOf(events, "decision.failed")).toEqual([
      expect.objectContaining({ errorName: "JevValidationError" }),
    ]);
  });

  const chunking = resolveOptions({ input: "x", question: CHOICE }).chunking;
  const budgetFor = (contextWindow: number | undefined): number =>
    computeStateBudget({ question: CHOICE, contextWindow, tokenizer: defaultTokenizer, chunking })
      .usableStateTokens;

  it.each([
    ["a reported window", 4_000, 4_000],
    ["an unknown window", undefined, undefined],
    ["a non-finite window", Number.NaN, undefined],
    ["a zero window", 0, undefined],
  ])("budgets from %s", async (_label, reported, used) => {
    const mock = new MockTransport({ contextWindow: reported });
    const result = await decide({ input: "Text.", question: CHOICE, provider: { transport: mock } });
    expect(result.chunks.stateTokenBudget).toBe(budgetFor(used));
  });

  it("falls back to the conservative budget when the context-window lookup throws", async () => {
    const sync = new MockTransport({
      contextWindow: () => {
        throw new Error("metadata down");
      },
    });
    const rejecting = new MockTransport({ contextWindow: () => Promise.reject(new Error("metadata down")) });
    for (const transport of [sync, rejecting]) {
      const result = await decide({ input: "Text.", question: CHOICE, provider: { transport } });
      expect(result.chunks.stateTokenBudget).toBe(budgetFor(undefined));
    }
  });

  it("uses provider.model over the transport default, for requests and the window lookup", async () => {
    const mock = new MockTransport();
    const result = await decide(request(CHOICE, "Text.", mock, { provider: { transport: mock, model: "jev-x" } }));
    expect(mock.contextWindowCalls).toEqual(["jev-x"]);
    expect(mock.requests[0]?.model).toBe("jev-x");
    expect(result.model).toBe("jev-x");
  });

  it("falls back to the requested model when a response reports none", async () => {
    const transport: JevTransport = {
      name: "inline",
      defaultModel: "inline-model",
      contextWindow: () => undefined,
      decide: async () => ({ model: "", answers: { [QUESTION_KEY]: noulAnswer(0.4) } }),
    };
    const result = await decide({ input: "Text.", question: NOUL, provider: { transport } });
    expect(result.model).toBe("inline-model");
    expect(result.chunks.results[0]?.model).toBe("inline-model");
    expect(result.chunks.results[0]?.responseId).toBeUndefined();
  });

  it("uses a custom tokenizer for budgeting and planning", async () => {
    const perChar = { name: "per-char", count: (text: string): number => text.length };
    const mock = new MockTransport();
    const input = "x".repeat(1_000);
    const result = await decide(request(NOUL, input, mock, { tokenizer: perChar }));
    expect(result.usage.inputTokensEstimated).toBe(1_000);
    expect(mock.states.every((s) => s.length <= MIN_STATE_TOKENS)).toBe(true);
    expect(result.chunks.count).toBeGreaterThan(3);
  });

  it("validates and resolves the options once, reading each option section and value exactly once", async () => {
    const mock = new MockTransport({ script: [noulAnswer(0.25)] });
    let sectionReads = 0;
    let overlapReads = 0;
    const chunking = {};
    Object.defineProperty(chunking, "overlap", {
      enumerable: true,
      get() {
        overlapReads++;
        return 0.1;
      },
    });
    const req = { input: "some text about trees", question: NOUL, provider: { transport: mock } };
    Object.defineProperty(req, "chunking", {
      enumerable: true,
      get() {
        sectionReads++;
        return chunking;
      },
    });

    const result = await decide(req);

    expect(result.chunks.overlap).toBe(0.1);
    expect(sectionReads).toBe(1);
    expect(overlapReads).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Built-in transports end to end (fetch stubbed)
// ---------------------------------------------------------------------------

describe("decide: built-in transports", () => {
  it("runs through OpenRouter's Decisions API with the catalog context window", async () => {
    const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit): Promise<Response> => {
      if (String(url).includes("/api/v1/models")) {
        return Response.json({
          data: [
            { id: "typesafe/jev-1.13", canonical_slug: "typesafe/jev-1.13-20260917", context_length: 16_000 },
          ],
        });
      }
      const body = JSON.parse(String(init?.body)) as { state: string };
      return Response.json({
        id: "gen-dec-1",
        model: "typesafe/jev-1.13-20260917",
        provider: "TypeSafe",
        answers: { decision: { type: "noul", noul: body.state.includes("oak") ? 0.8 : 0.2 } },
        usage: { input_tokens: 40, output_tokens: 3, cost: 0.0002 },
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await decide({
      input: "An oak grows here.",
      question: NOUL,
      provider: { transport: "openrouter", apiKey: "sk-or-test" },
    });

    expect(result).toMatchObject({
      type: "noul",
      noul: 0.8,
      provider: "openrouter",
      model: "typesafe/jev-1.13-20260917",
      usage: { inputTokens: 40, outputTokens: 3, costUsd: 0.0002, requests: 1 },
    });
    expect(result.chunks.stateTokenBudget).toBe(
      computeStateBudget({
        question: NOUL,
        contextWindow: 16_000,
        tokenizer: defaultTokenizer,
        chunking: resolveOptions({ input: "x", question: NOUL }).chunking,
      }).usableStateTokens,
    );
    const post = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/api/alpha/decisions"));
    const init = post?.[1] as RequestInit;
    expect((init.headers as Record<string, string>)["authorization"]).toBe("Bearer sk-or-test");
    expect(JSON.parse(String(init.body))).toEqual({
      model: "typesafe/jev-1.13",
      state: "An oak grows here.",
      questions: { decision: NOUL },
    });
  });

  it("runs through the direct TypeSafe transport", async () => {
    const fetchMock = vi.fn(
      async (_url: string | URL | Request, _init?: RequestInit): Promise<Response> =>
        Response.json({
          model: "jev-latest-20260917",
          answers: {
            decision: {
              type: "choice",
              choice: "rock",
              probabilities: { tree: 0.1, rock: 0.8, other: 0.1 },
              confidence: 0.7,
            },
          },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const result = await decide({
      input: "Granite everywhere.",
      question: CHOICE,
      provider: { transport: "direct", apiKey: "ts-test" },
    });
    expect(result).toMatchObject({
      type: "choice",
      choice: "rock",
      provider: "direct",
      model: "jev-latest-20260917",
    });
    expect(String(fetchMock.mock.calls[0]?.[0])).toBe("https://api.typesafe.ai/v1/systemone");
  });

  it("fails before any call when the built-in transport has no API key", async () => {
    vi.stubEnv("OPENROUTER_API_KEY", "");
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const events: JevInfiniteCTXEvent[] = [];
    await expect(
      decide({ input: "Text.", question: NOUL, onEvent: (e) => events.push(e) }),
    ).rejects.toBeInstanceOf(JevValidationError);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(eventsOf(events, "decision.failed")).toEqual([
      expect.objectContaining({ provider: "openrouter", model: "", errorName: "JevValidationError" }),
    ]);
  });
});
