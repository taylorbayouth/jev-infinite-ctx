import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { estimateTokens, planChunks, type Span } from "../src/chunks.js";
import { prose } from "./helpers.js";

const bytes = (text: string) => new TextEncoder().encode(text).length;
const isHigh = (code: number) => code >= 0xd800 && code <= 0xdbff;
const isLow = (code: number) => code >= 0xdc00 && code <= 0xdfff;

/** Every guarantee a plan must meet, for any text and budget. */
function expectValidPlan(text: string, maxTokens: number, plan: Span[]): void {
  expect(plan[0]!.start).toBe(0);
  expect(plan.at(-1)!.end).toBe(text.length);
  let weights = 0;
  plan.forEach((chunk, i) => {
    expect(estimateTokens(text.slice(chunk.start, chunk.end))).toBeLessThanOrEqual(maxTokens);
    expect(chunk.weight).toBeGreaterThan(0);
    weights += chunk.weight;
    for (const k of [chunk.start, chunk.end]) {
      if (k > 0 && k < text.length) expect(isHigh(text.charCodeAt(k - 1)) && isLow(text.charCodeAt(k))).toBe(false);
    }
    if (i > 0) {
      const previous = plan[i - 1]!;
      expect(chunk.start).toBeGreaterThan(previous.start);
      expect(chunk.end).toBeGreaterThan(previous.end);
      expect(chunk.start).toBeLessThanOrEqual(previous.end); // no gap: nothing is skipped
    }
  });
  expect(weights).toBeCloseTo(1, 9);
}

describe("estimateTokens", () => {
  it("counts 3 bytes of UTF-8 per token, rounded up", () => {
    expect(estimateTokens("")).toBe(0);
    expect(estimateTokens("abc")).toBe(1);
    expect(estimateTokens("abcd")).toBe(2);
    expect(estimateTokens("é")).toBe(1); // 2 bytes
    expect(estimateTokens("日本語")).toBe(3); // 9 bytes
    expect(estimateTokens("😀😀😀")).toBe(4); // 12 bytes
  });

  it("matches TextEncoder for well-formed text", () => {
    fc.assert(fc.property(fc.string({ unit: "binary" }), (text) => estimateTokens(text) === Math.ceil(bytes(text) / 3)));
  });
});

describe("planChunks", () => {
  it("keeps input that fits as one chunk", () => {
    expect(planChunks("A short note.", 1000)).toEqual([{ start: 0, end: 13, weight: 1 }]);
  });

  it("covers every character within budget, in order, for any text", () => {
    const unit = fc.oneof(
      fc.constantFrom(" ", "\n", "\r\n", ". ", "! ", "。", "\t", "日", "😀", "é", "\ud800", "\udc00"),
      fc.string({ unit: "grapheme-ascii", minLength: 1, maxLength: 12 }),
    );
    fc.assert(
      fc.property(fc.array(unit, { minLength: 1, maxLength: 300 }), fc.integer({ min: 2, max: 80 }), (parts, maxTokens) => {
        const text = parts.join("");
        expectValidPlan(text, maxTokens, planChunks(text, maxTokens));
      }),
      { numRuns: 400 },
    );
  });

  it("weights each chunk by the text only it covers", () => {
    const text = prose(400);
    const plan = planChunks(text, 1500);
    expect(plan.length).toBeGreaterThan(2);
    plan.forEach((chunk, i) => {
      const fresh = text.slice(i === 0 ? 0 : plan[i - 1]!.end, chunk.end);
      expect(chunk.weight).toBeCloseTo(bytes(fresh) / bytes(text), 12);
    });
  });

  it("overlaps neighbours by about 5% and breaks at sentence ends", () => {
    const text = prose(2000);
    const maxTokens = 5000;
    const plan = planChunks(text, maxTokens);
    expect(plan.length).toBeGreaterThan(3);
    for (let i = 1; i < plan.length; i++) {
      const shared = bytes(text.slice(plan[i]!.start, plan[i - 1]!.end));
      expect(shared).toBeGreaterThanOrEqual(0.05 * maxTokens * 3);
      expect(shared).toBeLessThan(0.1 * maxTokens * 3);
    }
    for (const chunk of plan.slice(0, -1)) expect(text[chunk.end - 1]).toMatch(/\s/);
  });

  it("evens out chunk sizes instead of leaving a scrap at the end", () => {
    const text = prose(3000);
    const sizes = planChunks(text, 8000).map((chunk) => bytes(text.slice(chunk.start, chunk.end)));
    expect(sizes.length).toBeGreaterThan(3);
    expect(Math.min(...sizes) / Math.max(...sizes)).toBeGreaterThan(0.85);
  });

  it("splits text with no breaks at all", () => {
    const text = "x".repeat(50_000);
    expectValidPlan(text, 2000, planChunks(text, 2000));
  });

  it("plans a 1 MB input quickly", () => {
    const text = prose(13_000);
    expect(bytes(text)).toBeGreaterThan(1_000_000);
    const started = performance.now();
    const plan = planChunks(text, 28_000);
    expect(performance.now() - started).toBeLessThan(1000);
    expectValidPlan(text, 28_000, plan);
  });
});
