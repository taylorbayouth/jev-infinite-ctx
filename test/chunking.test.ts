import { afterEach, describe, expect, it, vi } from "vitest";
import fc from "fast-check";
import { effectiveChunkCount } from "../src/c3.js";
import { computeStateBudget, planChunks } from "../src/chunking.js";
import { decide } from "../src/decide.js";
import {
  DEFAULTS,
  FALLBACK_CONTEXT_WINDOW,
  FALLBACK_MAX_STATE_TOKENS,
  MAX_OVERLAP,
  MIN_STATE_TOKENS,
} from "../src/defaults.js";
import { JevAbortError, JevInfiniteCTXError, JevValidationError } from "../src/errors.js";
import { createTokenizer, defaultTokenizer } from "../src/tokenizer.js";
import type { ChunkPlan, JevQuestion, NoulQuestion, ResolvedOptions, Tokenizer } from "../src/types.js";
import { MockTransport } from "./helpers/mock-transport.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** One token per UTF-16 code unit: makes chunk ends exactly predictable. */
const charTokenizer = createTokenizer((text) => text.length, "chars");

/** Deliberately non-monotone: length plus a hash-derived 0..2. */
const noisyTokenizer = createTokenizer((text) => {
  if (text.length === 0) return 0;
  let hash = 5381;
  for (let i = 0; i < text.length; i++) hash = (Math.imul(hash, 33) ^ text.charCodeAt(i)) >>> 0;
  return text.length + (hash % 3);
}, "noisy");

/** Charges a fixed overhead on every call, so even "" counts as 2 (like BOS/EOS tokens). */
const overheadTokenizer = createTokenizer((text) => defaultTokenizer.count(text) + 2, "overhead");

const BREAKING_SPACE = /[\t\n\v\f\r \u0085\u1680\u2000-\u2006\u2008-\u200a\u2028\u2029\u205f\u3000]/;

function splitsSurrogatePair(input: string, pos: number): boolean {
  if (pos <= 0 || pos >= input.length) return false;
  const before = input.charCodeAt(pos - 1);
  const after = input.charCodeAt(pos);
  return before >= 0xd800 && before <= 0xdbff && after >= 0xdc00 && after <= 0xdfff;
}

function plan(
  input: string,
  stateTokenBudget: number,
  options: { overlap?: number; tokenizer?: Tokenizer; preferNaturalBoundaries?: boolean } = {},
): ChunkPlan {
  return planChunks(input, {
    stateTokenBudget,
    overlap: options.overlap ?? 0,
    tokenizer: options.tokenizer ?? charTokenizer,
    preferNaturalBoundaries: options.preferNaturalBoundaries ?? true,
  });
}

/** Asserts every structural invariant of a plan (spec 6.2–6.4, 11.3). */
function expectValidPlan(input: string, result: ChunkPlan, budget: number, overlap: number, tokenizer: Tokenizer): void {
  const { chunks } = result;
  expect(chunks.length).toBeGreaterThan(0);
  expect(result.stateTokenBudget).toBe(budget);
  expect(result.overlapRatio).toBe(overlap);
  expect(result.overlapTokens).toBe(Math.floor(budget * overlap));
  expect(result.totalTokens).toBe(tokenizer.count(input));
  expect(chunks.length === 1).toBe(result.totalTokens <= budget);

  expect(chunks[0]!.start).toBe(0);
  expect(chunks[chunks.length - 1]!.end).toBe(input.length);

  let rebuilt = "";
  let previousEnd = 0;
  let weightSum = 0;
  let uniqueSum = 0;
  chunks.forEach((chunk, i) => {
    expect(chunk.index).toBe(i);
    expect(chunk.boundary === "end").toBe(i === chunks.length - 1);
    expect(splitsSurrogatePair(input, chunk.start)).toBe(false);
    expect(splitsSurrogatePair(input, chunk.end)).toBe(false);
    expect(chunk.tokens).toBe(tokenizer.count(input.slice(chunk.start, chunk.end)));
    expect(chunk.tokens).toBeLessThanOrEqual(budget);
    if (i === 0) {
      expect(chunk.overlapTokens).toBe(0);
    } else {
      const previous = chunks[i - 1]!;
      expect(chunk.start).toBeGreaterThan(previous.start);
      expect(chunk.end).toBeGreaterThan(previous.end);
      expect(chunk.start).toBeLessThanOrEqual(previous.end);
      if (overlap === 0) expect(chunk.start).toBe(previous.end);
      const shared = chunk.start === previous.end ? 0 : tokenizer.count(input.slice(chunk.start, previous.end));
      expect(chunk.overlapTokens).toBe(shared);
    }
    expect(chunk.uniqueTokens).toBe(Math.max(1, chunk.tokens - chunk.overlapTokens));
    expect(chunk.weight).toBeGreaterThan(0);
    weightSum += chunk.weight;
    uniqueSum += chunk.uniqueTokens;
    // The unique region [previous end, end) of every chunk tiles the input.
    rebuilt += input.slice(previousEnd, chunk.end);
    previousEnd = chunk.end;
  });
  expect(rebuilt).toBe(input);
  expect(Math.abs(weightSum - 1)).toBeLessThanOrEqual(1e-9);
  for (const chunk of chunks) expect(chunk.weight).toBeCloseTo(chunk.uniqueTokens / uniqueSum, 12);
  // One N_eff implementation (spec 11.3): the plan reports exactly what C3 computes.
  expect(result.effectiveCount).toBe(effectiveChunkCount(chunks));
  expect(result.effectiveCount).toBeGreaterThanOrEqual(1);
  expect(result.effectiveCount).toBeLessThanOrEqual(chunks.length);
}

/** The error `fn` throws; fails the test if it returns. */
function caught(fn: () => unknown): unknown {
  try {
    fn();
  } catch (error) {
    return error;
  }
  return expect.unreachable("expected an error");
}

/** Deterministic PRNG (mulberry32). */
function random(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const WORDS = (
  "the a of and to in decision model reads every chunk returns probability for each option while overlap " +
  "preserves context between neighbouring regions long documents without losing any source text is it " +
  "internationalization responsibilities characteristics we they report evidence confidence agreement"
).split(" ");

/** Pseudo-English prose: sentences with varied punctuation, paragraphs separated by blank lines. */
function englishText(targetLength: number, seed = 1): string {
  const next = random(seed);
  const pick = <T>(items: readonly T[]): T => items[Math.floor(next() * items.length)]!;
  const paragraphs: string[] = [];
  let length = 0;
  while (length < targetLength) {
    const sentences: string[] = [];
    const sentenceCount = 2 + Math.floor(next() * 6);
    for (let s = 0; s < sentenceCount; s++) {
      const words = Array.from({ length: 5 + Math.floor(next() * 16) }, () => pick(WORDS));
      const body = words.join(next() < 0.1 ? ", " : " ");
      sentences.push(body.charAt(0).toUpperCase() + body.slice(1) + pick([".", ".", ".", "?", "!", "…", '."']));
    }
    const paragraph = sentences.join(" ");
    paragraphs.push(paragraph);
    length += paragraph.length + 2;
  }
  return paragraphs.join("\n\n");
}

/** Arbitrary text including lone surrogates, CJK, emoji, CRLF and boundary punctuation. */
const textUnit = fc.oneof(
  { weight: 4, arbitrary: fc.string({ unit: "binary", minLength: 1, maxLength: 1 }) },
  { weight: 1, arbitrary: fc.integer({ min: 0xd800, max: 0xdfff }).map((c) => String.fromCharCode(c)) },
  { weight: 2, arbitrary: fc.string({ unit: "grapheme", minLength: 1, maxLength: 1 }) },
  {
    weight: 8,
    arbitrary: fc.constantFrom(
      "a", "b", "word", "Word", " ", " ", "  ", "\n", "\n\n", "\r\n", "\r\n\r\n", "\t", ". ", "! ", "? ",
      '." ', "…", "3.14", "中", "文字", "。", "」", "😀", "👍🏽", "é", "\u00a0", "\u2029",
    ),
  },
);
const arbitraryText = fc.string({ unit: textUnit, maxLength: 400 });
const overlapRatio = fc.oneof(
  fc.constantFrom(0, 0.05, 0.1, 0.25, MAX_OVERLAP),
  fc.double({ min: 0, max: MAX_OVERLAP, noNaN: true }),
);

// ---------------------------------------------------------------------------
// computeStateBudget (spec 6.1)
// ---------------------------------------------------------------------------

describe("computeStateBudget", () => {
  const chunking: ResolvedOptions["chunking"] = { ...DEFAULTS.chunking };
  const question: JevQuestion = {
    type: "choice",
    instructions: "What is the primary subject of this material?",
    criteria: {
      tree: "Primarily about trees or forests.",
      rock: "Primarily about rocks or geology.",
      other: "Neither is the primary subject.",
    },
  };
  const questionTokens = defaultTokenizer.count(
    JSON.stringify({ type: question.type, instructions: question.instructions, criteria: question.criteria }),
  );

  it("derives the budget from a known context window", () => {
    const budget = computeStateBudget({ question, contextWindow: 32_000, tokenizer: defaultTokenizer, chunking });
    const remaining = 32_000 - questionTokens - 1024;
    const safetyReserve = Math.ceil(remaining * 0.08);
    expect(budget).toEqual({
      contextWindow: 32_000,
      questionTokens,
      protocolReserve: 1024,
      safetyReserve,
      usableStateTokens: Math.floor(remaining - safetyReserve),
      source: "metadata",
    });
  });

  it("does not apply the fallback cap to a known large window", () => {
    const budget = computeStateBudget({ question, contextWindow: 128_000, tokenizer: defaultTokenizer, chunking });
    expect(budget.source).toBe("metadata");
    expect(budget.usableStateTokens).toBeGreaterThan(FALLBACK_MAX_STATE_TOKENS);
  });

  it.each([undefined, Number.NaN, Number.POSITIVE_INFINITY, 0, -5, Number.MAX_VALUE, 1e20, 2 ** 53 + 2])(
    "falls back to a conservative ~28K budget when the window is %s",
    (contextWindow) => {
      const budget = computeStateBudget({ question, contextWindow, tokenizer: defaultTokenizer, chunking });
      const remaining = FALLBACK_CONTEXT_WINDOW - questionTokens - 1024;
      expect(budget.contextWindow).toBeUndefined();
      expect(budget.source).toBe("fallback");
      expect(budget.safetyReserve).toBe(Math.ceil(remaining * 0.08));
      expect(budget.usableStateTokens).toBe(FALLBACK_MAX_STATE_TOKENS);
    },
  );

  it("returns a safe-integer budget for the largest window it trusts", () => {
    const budget = computeStateBudget({
      question,
      contextWindow: Number.MAX_SAFE_INTEGER,
      tokenizer: defaultTokenizer,
      chunking,
    });
    expect(budget.source).toBe("metadata");
    expect(budget.contextWindow).toBe(Number.MAX_SAFE_INTEGER);
    expect(Number.isSafeInteger(budget.usableStateTokens)).toBe(true);
    expect(budget.usableStateTokens).toBeGreaterThan(FALLBACK_MAX_STATE_TOKENS);
  });

  it("keeps the smaller computed budget when the fallback window leaves less than the cap", () => {
    const zero = createTokenizer(() => 0);
    const budget = computeStateBudget({
      question,
      contextWindow: undefined,
      tokenizer: zero,
      chunking: { ...chunking, protocolReserve: 4000, contextSafetyReserve: 0.1 },
    });
    // 32000 − 4000 = 28000; safety 2800 → 25200 < 28000.
    expect(budget).toMatchObject({ usableStateTokens: 25_200, safetyReserve: 2800, source: "fallback" });
  });

  it("caps the budget at a smaller caller maxStateTokens", () => {
    const budget = computeStateBudget({
      question,
      contextWindow: 32_000,
      tokenizer: defaultTokenizer,
      chunking: { ...chunking, maxStateTokens: 1000 },
    });
    expect(budget.usableStateTokens).toBe(1000);
    expect(budget.source).toBe("caller");
  });

  it("ignores a caller maxStateTokens above the computed budget", () => {
    const budget = computeStateBudget({
      question,
      contextWindow: 32_000,
      tokenizer: defaultTokenizer,
      chunking: { ...chunking, maxStateTokens: 1_000_000 },
    });
    expect(budget.source).toBe("metadata");
    expect(budget.usableStateTokens).toBeLessThan(32_000);
  });

  it("counts question tokens with the supplied tokenizer and omits absent noul criteria", () => {
    const noul: JevQuestion = { type: "noul", instructions: "Is the author in favour?" };
    const budget = computeStateBudget({ question: noul, contextWindow: 32_000, tokenizer: charTokenizer, chunking });
    expect(budget.questionTokens).toBe('{"type":"noul","instructions":"Is the author in favour?"}'.length);
  });

  it("accepts a budget of exactly MIN_STATE_TOKENS and rejects one below it", () => {
    const zero = createTokenizer(() => 0);
    const exact = { ...chunking, protocolReserve: 0, contextSafetyReserve: 0 };
    expect(
      computeStateBudget({ question, contextWindow: MIN_STATE_TOKENS, tokenizer: zero, chunking: exact })
        .usableStateTokens,
    ).toBe(MIN_STATE_TOKENS);
    expect(() =>
      computeStateBudget({ question, contextWindow: MIN_STATE_TOKENS - 1, tokenizer: zero, chunking: exact }),
    ).toThrow(JevValidationError);
  });

  it("explains that the question is too large for the window", () => {
    const huge: JevQuestion = { type: "noul", instructions: "Is this relevant? ".repeat(2000) };
    expect(() => computeStateBudget({ question: huge, contextWindow: 8000, tokenizer: defaultTokenizer, chunking })).toThrow(
      /question is too large for the context window/,
    );
  });

  it("never lets a negative remainder produce a negative safety reserve", () => {
    expect(() =>
      computeStateBudget({ question, contextWindow: 500, tokenizer: defaultTokenizer, chunking }),
    ).toThrow(/leaving 0 state tokens in a 500-token window after a 1024-token protocol reserve and a 0-token safety reserve/);
  });

  it("rejects a caller cap below the minimum", () => {
    expect(() =>
      computeStateBudget({
        question,
        contextWindow: 32_000,
        tokenizer: defaultTokenizer,
        chunking: { ...chunking, maxStateTokens: 100 },
      }),
    ).toThrow(/chunking\.maxStateTokens \(100\) is below the minimum/);
  });

  it("rejects questions that cannot be serialized, and invalid tokenizers", () => {
    const bad = { type: "noul", instructions: 1n } as unknown as JevQuestion;
    expect(() => computeStateBudget({ question: bad, contextWindow: 32_000, tokenizer: defaultTokenizer, chunking })).toThrow(
      JevValidationError,
    );
    expect(() =>
      computeStateBudget({ question, contextWindow: 32_000, tokenizer: {} as Tokenizer, chunking }),
    ).toThrow(JevValidationError);
    const nan = { name: "nan", count: () => Number.NaN };
    expect(() => computeStateBudget({ question, contextWindow: 32_000, tokenizer: nan, chunking })).toThrow(
      /Tokenizer "nan" returned NaN/,
    );
  });

  it("wraps an exception from the tokenizer in a JevValidationError", () => {
    const failure = new TypeError("boom");
    const tokenizer: Tokenizer = {
      name: "throws",
      count: () => {
        throw failure;
      },
    };
    const error = caught(() => computeStateBudget({ question, contextWindow: 32_000, tokenizer, chunking }));
    expect(error).toBeInstanceOf(JevValidationError);
    expect((error as Error).message).toBe('Tokenizer "throws" threw while counting tokens');
    expect((error as Error).cause).toBe(failure);
  });
});

// ---------------------------------------------------------------------------
// planChunks: basic shapes
// ---------------------------------------------------------------------------

describe("planChunks", () => {
  it("returns one chunk for input that fits, including empty input", () => {
    for (const input of ["", "short text", "x".repeat(50)]) {
      const result = plan(input, 50);
      expect(result.chunks).toEqual([
        {
          index: 0,
          start: 0,
          end: input.length,
          tokens: input.length,
          overlapTokens: 0,
          uniqueTokens: Math.max(1, input.length),
          weight: 1,
          boundary: "end",
        },
      ]);
      expect(result.effectiveCount).toBe(1);
      expectValidPlan(input, result, 50, 0, charTokenizer);
    }
  });

  it("splits as soon as the input exceeds the budget by one token", () => {
    const result = plan("x".repeat(51), 50);
    expect(result.chunks.map((c) => [c.start, c.end])).toEqual([
      [0, 50],
      [50, 51],
    ]);
  });

  it("produces adjacent hard chunks without overlap or natural boundaries", () => {
    const result = plan("a".repeat(35), 10, { preferNaturalBoundaries: false });
    expect(result.chunks.map((c) => [c.start, c.end, c.boundary])).toEqual([
      [0, 10, "hard"],
      [10, 20, "hard"],
      [20, 30, "hard"],
      [30, 35, "end"],
    ]);
    expect(result.chunks.map((c) => c.weight)).toEqual([10 / 35, 10 / 35, 10 / 35, 5 / 35]);
    // The N_eff formula itself is pinned in test/c3.test.ts.
    expect(result.effectiveCount).toBe(effectiveChunkCount(result.chunks));
    expect(result.effectiveCount).toBeGreaterThan(3);
    expect(result.effectiveCount).toBeLessThanOrEqual(4);
  });

  it("slides the window back by the overlap target (spec 6.3)", () => {
    const result = plan("a".repeat(30), 10, { overlap: 0.2, preferNaturalBoundaries: false });
    expect(result.overlapTokens).toBe(2);
    expect(result.chunks.map((c) => [c.start, c.end, c.overlapTokens, c.uniqueTokens])).toEqual([
      [0, 10, 0, 10],
      [8, 18, 2, 8],
      [16, 26, 2, 8],
      [24, 30, 2, 4],
    ]);
    // Unique tokens tile the input exactly, so overlap earns no extra weight (spec 6.4).
    expect(result.chunks.map((c) => c.weight)).toEqual([10 / 30, 8 / 30, 8 / 30, 4 / 30]);
    // Overlap is discounted: N_eff < N. The formula itself is pinned in test/c3.test.ts.
    expect(result.effectiveCount).toBe(effectiveChunkCount(result.chunks));
    expect(result.effectiveCount).toBeGreaterThan(1);
    expect(result.effectiveCount).toBeLessThan(4);
  });

  it("gives N_eff = 1 + (N − 1)(1 − r) for uniform chunks (spec 11.3)", () => {
    const result = plan("a".repeat(460), 100, { overlap: 0.1, preferNaturalBoundaries: false });
    expect(result.chunks.map((c) => [c.start, c.end])).toEqual([
      [0, 100],
      [90, 190],
      [180, 280],
      [270, 370],
      [360, 460],
    ]);
    expect(result.effectiveCount).toBeCloseTo(1 + 4 * 0.9, 12);
  });

  it("counts a tiny final chunk in N_eff by its share of new content, not as a whole chunk", () => {
    const result = plan("x".repeat(1000) + "y".repeat(10), 1000, { preferNaturalBoundaries: false });
    expect(result.chunks.map((c) => c.tokens)).toEqual([1000, 10]);
    // The tail carries ~1% of the weight, and adds ~1% of a chunk to N_eff.
    expect(result.chunks[1]?.weight).toBeCloseTo(10 / 1010, 12);
    expect(result.effectiveCount).toBeCloseTo(1.01, 12);
  });

  it("uses no overlap when floor(budget × overlap) is zero", () => {
    const result = plan("a".repeat(250), 100, { overlap: 0.005 });
    expect(result.overlapTokens).toBe(0);
    expect(result.chunks.map((c) => c.start)).toEqual([0, 100, 200]);
  });

  it("never splits a surrogate pair", () => {
    const emoji = "😀".repeat(10);
    const adjacent = plan(emoji, 5);
    expect(adjacent.chunks.map((c) => [c.start, c.end])).toEqual([
      [0, 4],
      [4, 8],
      [8, 12],
      [12, 16],
      [16, 20],
    ]);
    const overlapping = plan(emoji, 5, { overlap: 0.2 });
    expectValidPlan(emoji, overlapping, 5, 0.2, charTokenizer);
    expect(overlapping.chunks[1]!.start).toBe(2);
  });

  describe("natural boundaries (spec 6.2)", () => {
    // With one token per character and a budget of 100, the first chunk's maximal end
    // is 100 and natural boundaries are searched for in ends [80, 100].
    const tail = "w".repeat(200);
    const firstChunk = (text: string, preferNaturalBoundaries = true) =>
      plan(text, 100, { preferNaturalBoundaries }).chunks[0]!;
    const layout = (paragraph: string, sentence: string, space: string): string =>
      "x".repeat(82) + paragraph + "y".repeat(8) + sentence + "zz" + space + tail;

    it("prefers a paragraph break over closer sentence and word breaks", () => {
      const text = layout("\n\n", ". ", " ");
      expect(firstChunk(text)).toMatchObject({ end: 84, boundary: "paragraph" });
    });

    it("prefers a sentence break over a closer word break", () => {
      const text = layout("xx", ". ", " ");
      expect(firstChunk(text)).toMatchObject({ end: 94, boundary: "sentence" });
    });

    it("falls back to a whitespace break", () => {
      const text = layout("xx", "yy", " ");
      expect(firstChunk(text)).toMatchObject({ end: 97, boundary: "whitespace" });
    });

    it("cuts hard when there is no break in the window", () => {
      expect(firstChunk(layout("xx", "yy", "z"))).toMatchObject({ end: 100, boundary: "hard" });
      expect(firstChunk("a".repeat(300))).toMatchObject({ end: 100, boundary: "hard" });
    });

    it("always cuts hard when natural boundaries are disabled", () => {
      const text = layout("\n\n", ". ", " ");
      expect(firstChunk(text, false)).toMatchObject({ end: 100, boundary: "hard" });
      const result = plan(englishText(2000), 100, { preferNaturalBoundaries: false, overlap: 0.1 });
      for (const chunk of result.chunks.slice(0, -1)) expect(chunk.boundary).toBe("hard");
    });

    it("picks the break of the chosen class closest to the maximal end", () => {
      const text = "x".repeat(84) + "\n\n" + "y".repeat(6) + "\n\n" + "z".repeat(4) + tail;
      expect(firstChunk(text)).toMatchObject({ end: 94, boundary: "paragraph" });
      const sentences = "x".repeat(84) + ". " + "y".repeat(6) + "? " + "z".repeat(4) + tail;
      expect(firstChunk(sentences)).toMatchObject({ end: 94, boundary: "sentence" });
    });

    it("ignores breaks before the last 20% of the chunk", () => {
      const text = "x".repeat(70) + "\n\n" + "y".repeat(20) + " " + tail;
      expect(firstChunk(text)).toMatchObject({ end: 93, boundary: "whitespace" });
    });

    it("keeps closing quotes and brackets with their sentence", () => {
      const quoted = "x".repeat(85) + 'said "Stop." Then' + tail;
      expect(firstChunk(quoted)).toMatchObject({ end: 98, boundary: "sentence" });
      const cjk = "中".repeat(90) + "。」" + "文".repeat(200);
      expect(firstChunk(cjk)).toMatchObject({ end: 92, boundary: "sentence" });
    });

    it("does not split a terminator from a closing bracket that lies past the maximal end", () => {
      // "。」" straddles the maximal end (100), so the earlier sentence end at 86 wins.
      const text = "中".repeat(85) + "。" + "中".repeat(13) + "。」" + "文".repeat(200);
      expect(firstChunk(text)).toMatchObject({ end: 86, boundary: "sentence" });
    });

    it("ends CJK sentences without whitespace", () => {
      const text = "中".repeat(90) + "。" + "文".repeat(200);
      expect(firstChunk(text)).toMatchObject({ end: 91, boundary: "sentence" });
    });

    it("accepts a sentence end whose trailing whitespace does not fit", () => {
      const text = "a".repeat(99) + ". " + tail;
      expect(firstChunk(text)).toMatchObject({ end: 100, boundary: "sentence" });
    });

    it("does not treat a decimal point as a sentence end", () => {
      const text = "a".repeat(85) + " 3.14" + tail;
      expect(firstChunk(text)).toMatchObject({ end: 86, boundary: "whitespace" });
    });

    it("recognises CRLF blank lines and U+2029 as paragraph breaks", () => {
      expect(firstChunk("a".repeat(85) + "\r\n\r\n" + tail)).toMatchObject({ end: 89, boundary: "paragraph" });
      expect(firstChunk("a".repeat(85) + "\n \t\n" + tail)).toMatchObject({ end: 89, boundary: "paragraph" });
      expect(firstChunk("a".repeat(85) + "\u2029" + tail)).toMatchObject({ end: 86, boundary: "paragraph" });
    });

    it("never ends a chunk at or before the previous chunk's end", () => {
      // 400 spaces (100 tokens) then emoji: the token-sparse overlap region spans most of
      // chunk 1's characters, so its 20% window would otherwise reach back to the break at 400.
      const text = " ".repeat(400) + "😀".repeat(100);
      const result = plan(text, 100, { overlap: 0.5, tokenizer: defaultTokenizer });
      expectValidPlan(text, result, 100, 0.5, defaultTokenizer);
      expect(result.chunks[0]).toMatchObject({ start: 0, end: 400, boundary: "whitespace" });
      expect(result.chunks[1]).toMatchObject({ start: 203, end: 432, boundary: "hard" });
    });

    it("does not break at a no-break space", () => {
      expect(firstChunk("a".repeat(85) + "\u00a0" + tail)).toMatchObject({ end: 100, boundary: "hard" });
    });

    it("starts overlapping chunks at word starts", () => {
      const text = englishText(5000, 7);
      const result = plan(text, 200, { overlap: 0.2 });
      expectValidPlan(text, result, 200, 0.2, charTokenizer);
      for (const chunk of result.chunks.slice(1)) {
        expect(text.charAt(chunk.start - 1)).toMatch(BREAKING_SPACE);
        expect(chunk.overlapTokens).toBeGreaterThanOrEqual(40);
        expect(chunk.overlapTokens).toBeLessThanOrEqual(80);
      }
    });

    it("starts at the exact overlap target without natural boundaries", () => {
      const text = englishText(5000, 8);
      const result = plan(text, 200, { overlap: 0.2, preferNaturalBoundaries: false });
      for (const chunk of result.chunks.slice(1)) expect(chunk.overlapTokens).toBe(40);
    });
  });

  describe("overlap (spec 6.3)", () => {
    it("gives an overlap region of about 5% of the budget by default", () => {
      const text = englishText(60_000, 3);
      const budget = 2000;
      const result = plan(text, budget, { overlap: DEFAULTS.chunking.overlap, tokenizer: defaultTokenizer });
      expectValidPlan(text, result, budget, DEFAULTS.chunking.overlap, defaultTokenizer);
      expect(result.overlapTokens).toBe(100);
      expect(result.chunks.length).toBeGreaterThan(5);
      for (const chunk of result.chunks.slice(1)) {
        expect(chunk.overlapTokens).toBeGreaterThanOrEqual(100);
        expect(chunk.overlapTokens).toBeLessThanOrEqual(110);
      }
    });

    it("makes interior chunks overlap both neighbours", () => {
      const text = englishText(20_000, 4);
      const result = plan(text, MIN_STATE_TOKENS, { overlap: 0.05, tokenizer: defaultTokenizer });
      expect(result.chunks.length).toBeGreaterThan(10);
      for (let i = 1; i < result.chunks.length - 1; i++) {
        const [previous, chunk, next] = [result.chunks[i - 1]!, result.chunks[i]!, result.chunks[i + 1]!];
        expect(chunk.start).toBeLessThan(previous.end);
        expect(next.start).toBeLessThan(chunk.end);
      }
    });
  });

  describe("hard inputs", () => {
    const cases: Array<[string, string]> = [
      ["only newlines", "\n".repeat(5000)],
      ["only spaces", " ".repeat(5000)],
      ["one long word", "x".repeat(20_000)],
      ["CJK prose", ("长文档很少能放进单个请求中，因此该库会将其拆分为相互重叠的片段。").repeat(200)],
      ["emoji", "😀🎉👍🏽".repeat(1000)],
      ["CRLF prose", englishText(20_000, 5).replace(/\n/g, "\r\n")],
      ["lone surrogates", "\ud800a\udc00b".repeat(2000)],
    ];

    it.each(cases)("plans %s at MIN_STATE_TOKENS without losing text", (_name, text) => {
      for (const preferNaturalBoundaries of [true, false]) {
        for (const overlap of [0, 0.05, MAX_OVERLAP]) {
          const result = planChunks(text, {
            stateTokenBudget: MIN_STATE_TOKENS,
            overlap,
            tokenizer: defaultTokenizer,
            preferNaturalBoundaries,
          });
          expect(result.chunks.length).toBeGreaterThan(1);
          expectValidPlan(text, result, MIN_STATE_TOKENS, overlap, defaultTokenizer);
        }
      }
    });

    it("finds paragraph breaks in newline-only input", () => {
      const result = plan("\n".repeat(5000), MIN_STATE_TOKENS, { tokenizer: defaultTokenizer });
      for (const chunk of result.chunks.slice(0, -1)) expect(chunk.boundary).toBe("paragraph");
    });

    it("stays correct for a tokenizer that is not monotone", () => {
      const text = englishText(3000, 6);
      const result = plan(text, 40, { overlap: 0.25, tokenizer: noisyTokenizer });
      expectValidPlan(text, result, 40, 0.25, noisyTokenizer);
    });
  });

  describe("validation", () => {
    const valid = { stateTokenBudget: 100, overlap: 0.05, tokenizer: charTokenizer, preferNaturalBoundaries: true };

    it("rejects non-string input", () => {
      expect(() => planChunks(42 as unknown as string, valid)).toThrow(JevValidationError);
    });

    it.each([0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY])("rejects a stateTokenBudget of %s", (stateTokenBudget) => {
      expect(() => planChunks("text", { ...valid, stateTokenBudget })).toThrow(/stateTokenBudget/);
    });

    it.each([-0.01, 0.51, Number.NaN])("rejects an overlap of %s", (overlap) => {
      expect(() => planChunks("text", { ...valid, overlap })).toThrow(/overlap/);
    });

    it("accepts the overlap bounds 0 and MAX_OVERLAP", () => {
      expect(() => planChunks("text", { ...valid, overlap: 0 })).not.toThrow();
      expect(() => planChunks("text", { ...valid, overlap: MAX_OVERLAP })).not.toThrow();
    });

    it("rejects invalid tokenizers and flags", () => {
      expect(() => planChunks("text", { ...valid, tokenizer: null as unknown as Tokenizer })).toThrow(/tokenizer/);
      expect(() => planChunks("text", { ...valid, tokenizer: { name: "x" } as unknown as Tokenizer })).toThrow(
        /tokenizer/,
      );
      expect(() =>
        planChunks("text", { ...valid, preferNaturalBoundaries: "yes" as unknown as boolean }),
      ).toThrow(/preferNaturalBoundaries/);
    });

    it("rejects invalid token counts from a raw tokenizer object", () => {
      const tokenizer: Tokenizer = { name: "raw", count: () => -1 };
      expect(() => planChunks("text", { ...valid, tokenizer })).toThrow(/Tokenizer "raw" returned -1/);
    });

    it("wraps an exception from the tokenizer, keeping it only as the cause", () => {
      const failure = new TypeError("boom");
      const tokenizer: Tokenizer = {
        name: "raw",
        count: () => {
          throw failure;
        },
      };
      const error = caught(() => planChunks("secret source text", { ...valid, tokenizer }));
      expect(error).toBeInstanceOf(JevValidationError);
      expect((error as Error).cause).toBe(failure);
      expect((error as Error).message).toBe('Tokenizer "raw" threw while counting tokens');
      expect((error as Error).message).not.toContain("secret");
    });

    it("wraps a createTokenizer tokenizer that throws partway through planning", () => {
      // Mimics js-tiktoken's default encode, which throws on special tokens.
      const tokenizer = createTokenizer((text) => {
        if (text.includes("<|endoftext|>")) throw new Error("disallowed special token");
        return text.length;
      }, "o200k_base");
      const input = "a".repeat(300) + "<|endoftext|>" + "b".repeat(300);
      const error = caught(() => planChunks(input, { ...valid, tokenizer }));
      expect(error).toBeInstanceOf(JevValidationError);
      expect((error as Error).message).toMatch(/Tokenizer "o200k_base" threw/);
    });

    it("rethrows a JevInfiniteCTXError from the tokenizer unchanged", () => {
      const abort = new JevAbortError("aborted");
      const tokenizer: Tokenizer = {
        name: "aborts",
        count: () => {
          throw abort;
        },
      };
      expect(caught(() => planChunks("text", { ...valid, tokenizer }))).toBe(abort);
    });

    it("reports a character that alone exceeds the budget, without leaking text", () => {
      expect(() => plan("😀", 1)).toThrow(/cannot hold the single character at offset 0/);
      try {
        plan("ab😀secret", 1);
        expect.unreachable();
      } catch (error) {
        expect(error).toBeInstanceOf(JevValidationError);
        expect((error as Error).message).toMatch(/offset 2/);
        expect((error as Error).message).not.toContain("secret");
      }
    });
  });

  // -------------------------------------------------------------------------
  // Properties
  // -------------------------------------------------------------------------

  describe("properties", () => {
    it("covers every string with a valid plan (one token per code unit)", () => {
      fc.assert(
        fc.property(
          arbitraryText,
          fc.integer({ min: 2, max: 40 }),
          overlapRatio,
          fc.boolean(),
          (text, budget, overlap, preferNaturalBoundaries) => {
            const result = plan(text, budget, { overlap, preferNaturalBoundaries });
            expectValidPlan(text, result, budget, overlap, charTokenizer);
          },
        ),
        { numRuns: 400 },
      );
    });

    it("covers every string with a valid plan (heuristic tokenizer)", () => {
      fc.assert(
        fc.property(
          arbitraryText,
          fc.integer({ min: 3, max: 60 }),
          overlapRatio,
          fc.boolean(),
          (text, budget, overlap, preferNaturalBoundaries) => {
            const result = plan(text, budget, { overlap, preferNaturalBoundaries, tokenizer: defaultTokenizer });
            expectValidPlan(text, result, budget, overlap, defaultTokenizer);
          },
        ),
        { numRuns: 400 },
      );
    });

    it("stays valid and terminates for non-monotone or overhead-charging tokenizers", () => {
      fc.assert(
        fc.property(
          arbitraryText,
          fc.constantFrom(noisyTokenizer, overheadTokenizer),
          fc.integer({ min: 6, max: 40 }),
          overlapRatio,
          fc.boolean(),
          (text, tokenizer, budget, overlap, preferNaturalBoundaries) => {
            const result = plan(text, budget, { overlap, preferNaturalBoundaries, tokenizer });
            expectValidPlan(text, result, budget, overlap, tokenizer);
          },
        ),
        { numRuns: 300 },
      );
    });

    it("overlaps every pair of neighbours whenever the overlap target is at least one token", () => {
      fc.assert(
        fc.property(
          arbitraryText,
          fc.oneof(
            fc.record({ tokenizer: fc.constant(charTokenizer), budget: fc.integer({ min: 6, max: 40 }) }),
            fc.record({ tokenizer: fc.constant(defaultTokenizer), budget: fc.integer({ min: 16, max: 60 }) }),
          ),
          fc.double({ min: 0.05, max: MAX_OVERLAP, noNaN: true }),
          fc.boolean(),
          (text, { tokenizer, budget }, overlap, preferNaturalBoundaries) => {
            fc.pre(Math.floor(budget * overlap) >= 1);
            const { chunks } = plan(text, budget, { overlap, preferNaturalBoundaries, tokenizer });
            for (let i = 1; i < chunks.length; i++) expect(chunks[i]!.start).toBeLessThan(chunks[i - 1]!.end);
          },
        ),
        { numRuns: 400 },
      );
    });

    it("reports exactly the N_eff that C3 computes, for any tokenizer", () => {
      // Non-monotone-ish: whitespace-only text counts as 0 tokens.
      const zeroWhitespace = createTokenizer((text) => (text.trim() === "" ? 0 : Math.ceil(text.length / 3)), "zero-ws");
      const words = createTokenizer((text) => (text.match(/\S+/g) ?? []).length, "words");
      fc.assert(
        fc.property(
          fc.string({ maxLength: 600, unit: fc.constantFrom("a", "b", " ", ".", "\n", "é", "😀") }),
          fc.integer({ min: 4, max: 60 }),
          overlapRatio,
          fc.constantFrom(words, charTokenizer, zeroWhitespace),
          fc.boolean(),
          (text, budget, overlap, tokenizer, preferNaturalBoundaries) => {
            let result: ChunkPlan;
            try {
              result = plan(text, budget, { overlap, preferNaturalBoundaries, tokenizer });
            } catch (error) {
              // A single character larger than the budget is the only expected failure.
              expect(error).toBeInstanceOf(JevValidationError);
              return;
            }
            expect(result.effectiveCount).toBe(effectiveChunkCount(result.chunks));
          },
        ),
        { numRuns: 500 },
      );
    });

    it("labels boundaries truthfully and cuts hard only when the window has no break", () => {
      fc.assert(
        fc.property(arbitraryText, fc.integer({ min: 4, max: 40 }), overlapRatio, (text, budget, overlap) => {
          const { chunks } = plan(text, budget, { overlap });
          chunks.forEach((chunk, i) => {
            const before = text.slice(0, chunk.end);
            switch (chunk.boundary) {
              case "paragraph":
                expect(before).toMatch(/(\n[ \t\r\v\f]*\n|\u2029)$/);
                break;
              case "sentence":
                expect(before).toMatch(/[.!?…。！？]["')\]}*_»’”›〉》」』】〕）］｝]*[\s\u0085]*$/u);
                break;
              case "whitespace":
                expect(before.charAt(before.length - 1)).toMatch(BREAKING_SPACE);
                break;
              case "hard": {
                // One token per code unit: the maximal end is start + budget (aligned).
                const maxEnd = chunk.start + budget - (splitsSurrogatePair(text, chunk.start + budget) ? 1 : 0);
                expect(chunk.end).toBe(maxEnd);
                const previousEnd = i === 0 ? 0 : chunks[i - 1]!.end;
                const lowest = Math.max(chunk.end - Math.floor((chunk.end - chunk.start) * 0.2), previousEnd + 1, chunk.start + 1);
                for (let e = lowest; e <= chunk.end; e++) expect(text.charAt(e - 1)).not.toMatch(BREAKING_SPACE);
                break;
              }
              case "end":
                expect(chunk.end).toBe(text.length);
                break;
            }
          });
        }),
        { numRuns: 400 },
      );
    });
  });

  // -------------------------------------------------------------------------
  // Scale (spec 15, 18)
  // -------------------------------------------------------------------------

  describe("scale", () => {
    const twoMillion = englishText(2_000_000, 11);

    it.each([27_000, MIN_STATE_TOKENS])(
      "plans ~2,000,000 characters in well under 5 s (budget %i)",
      (budget) => {
        const started = performance.now();
        const result = plan(twoMillion, budget, { overlap: 0.05, tokenizer: defaultTokenizer });
        const elapsedMs = performance.now() - started;
        expect(elapsedMs).toBeLessThan(2500);
        expect(result.chunks.length).toBeGreaterThan(1);
        expect(result.chunks[0]!.start).toBe(0);
        expect(result.chunks[result.chunks.length - 1]!.end).toBe(twoMillion.length);
        for (let i = 1; i < result.chunks.length; i++) {
          expect(result.chunks[i]!.start).toBeLessThan(result.chunks[i - 1]!.end);
          expect(result.chunks[i]!.tokens).toBeLessThanOrEqual(budget);
        }
      },
    );

    it("does work linear in the input size (never recounts the whole remainder)", () => {
      const countedPerInputChar = (text: string): number => {
        let counted = 0;
        const tokenizer: Tokenizer = {
          name: "metered",
          count: (slice) => {
            counted += slice.length;
            return defaultTokenizer.count(slice);
          },
        };
        plan(text, 4000, { overlap: 0.05, tokenizer });
        return counted / text.length;
      };
      const small = countedPerInputChar(twoMillion.slice(0, 250_000));
      const large = countedPerInputChar(twoMillion);
      expect(large).toBeLessThan(30);
      expect(large).toBeLessThan(small * 1.5);
    });

    const timedPlan = (text: string, preferNaturalBoundaries: boolean): { ms: number; result: ChunkPlan } => {
      const started = performance.now();
      const result = plan(text, MIN_STATE_TOKENS, { overlap: 0.05, tokenizer: defaultTokenizer, preferNaturalBoundaries });
      return { ms: performance.now() - started, result };
    };

    it("plans 2,000,000 spaces in well under 5 s: natural-boundary search stays linear", () => {
      const spaces = " ".repeat(2_000_000);
      const { ms, result } = timedPlan(spaces, true);
      expect(result.chunks[result.chunks.length - 1]!.end).toBe(spaces.length);
      expect(ms).toBeLessThan(2500);
    }, 60_000);

    it.each([
      ["tabs", "\t"],
      ["carriage returns", "\r"],
      ["ideographic spaces", "\u3000"],
    ])("adds only a constant factor for natural boundaries on 1,000,000 %s", (_label, ch) => {
      const text = ch.repeat(1_000_000);
      const off = timedPlan(text, false).ms;
      const on = timedPlan(text, true).ms;
      // Prose costs ~2x with natural boundaries on; a quadratic walk cost 50-90x here.
      expect(on).toBeLessThan(Math.max(off, 25) * 10);
    }, 60_000);

    it("labels a chunk a sentence end only when its terminator lies inside the chunk", () => {
      const text = "End." + " ".repeat(200_000) + "tail";
      const { result } = timedPlan(text, true);
      expect(result.chunks[0]).toMatchObject({ start: 0, boundary: "sentence" });
      const deep = result.chunks.filter((chunk) => chunk.start > 10_000 && chunk.end < text.length - 10);
      expect(deep.length).toBeGreaterThan(0);
      for (const chunk of deep) expect(chunk.boundary).toBe("whitespace");
    }, 60_000);

    it("processes a 100K+ token input into multiple chunks without truncation (spec 18)", () => {
      const text = englishText(450_000, 12);
      const totalTokens = defaultTokenizer.count(text);
      expect(totalTokens).toBeGreaterThan(100_000);
      const result = plan(text, FALLBACK_MAX_STATE_TOKENS, { overlap: 0.05, tokenizer: defaultTokenizer });
      expect(result.totalTokens).toBe(totalTokens);
      expect(result.chunks.length).toBeGreaterThanOrEqual(4);
      expectValidPlan(text, result, FALLBACK_MAX_STATE_TOKENS, 0.05, defaultTokenizer);
      // Overlap is discounted: 1 < N_eff < N.
      expect(result.effectiveCount).toBeGreaterThan(1);
      expect(result.effectiveCount).toBeLessThan(result.chunks.length);
    });
  });
});

// ---------------------------------------------------------------------------
// decide() integration: budgeting and planning errors reach the caller typed
// ---------------------------------------------------------------------------

describe("decide() with chunking edge cases", () => {
  const NOUL: NoulQuestion = { type: "noul", instructions: "Does the document mention a data breach?" };

  /** Mimics js-tiktoken's default `encode`, which throws on disallowed special tokens. */
  const tiktokenLike = (): Tokenizer =>
    createTokenizer((text) => {
      if (text.includes("<|endoftext|>")) {
        throw new Error("The text contains a special token that is not allowed: <|endoftext|>");
      }
      return defaultTokenizer.count(text);
    }, "o200k_base");

  async function rejectionOf(promise: Promise<unknown>): Promise<unknown> {
    try {
      await promise;
    } catch (error) {
      return error;
    }
    return expect.unreachable("expected a rejection");
  }

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("rejects with a JevValidationError when the tokenizer throws while planning chunks", async () => {
    const mock = new MockTransport();
    const error = await rejectionOf(
      decide({
        input: "Quarterly report.\n\n<|endoftext|>\n\nNo incidents.",
        question: NOUL,
        tokenizer: tiktokenLike(),
        provider: { transport: mock },
      }),
    );
    expect(mock.requests.length).toBe(0);
    expect(error).toBeInstanceOf(JevInfiniteCTXError);
    expect(error).toBeInstanceOf(JevValidationError);
    expect((error as Error).cause).toBeInstanceOf(Error);
  });

  it("rejects with a JevValidationError when the tokenizer throws while budgeting the question", async () => {
    const mock = new MockTransport();
    const error = await rejectionOf(
      decide({
        input: "Quarterly report.",
        question: { type: "noul", instructions: "Treat <|endoftext|> as a breach marker." },
        tokenizer: tiktokenLike(),
        provider: { transport: mock },
      }),
    );
    expect(mock.requests.length).toBe(0);
    expect(error).toBeInstanceOf(JevValidationError);
  });

  it.each([
    ["Number.MAX_VALUE", Number.MAX_VALUE],
    ["1e20", 1e20],
  ])("succeeds when a custom transport reports a %s context window", async (_label, window) => {
    const mock = new MockTransport({ contextWindow: window });
    const result = await decide({ input: "An oak grows here.", question: NOUL, provider: { transport: mock } });
    expect(result.type).toBe("noul");
    expect(Number.isSafeInteger(result.chunks.stateTokenBudget)).toBe(true);
  });

  it("succeeds when the OpenRouter catalog reports context_length 1e20", async () => {
    const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit): Promise<Response> => {
      if (String(url).includes("/api/v1/models")) {
        return Response.json({
          data: [{ id: "typesafe/jev-1.13", canonical_slug: "typesafe/jev-1.13-20260917", context_length: 1e20 }],
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
    expect(result).toMatchObject({ type: "noul", noul: 0.8 });
  });
});
