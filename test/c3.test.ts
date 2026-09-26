import { describe, it, expect } from "vitest";
import fc from "fast-check";
import { applyC3, baseConfidence, effectiveChunkCount } from "../src/c3.js";
import { aggregate } from "../src/aggregation.js";
import { computeAgreement } from "../src/agreement.js";
import { toDistribution } from "../src/probability.js";
import { DEFAULTS } from "../src/defaults.js";
import { JevInfiniteCTXError, JevValidationError } from "../src/errors.js";
import type { Distribution, ResolvedOptions } from "../src/types.js";

type ConfidenceOptions = ResolvedOptions["confidence"];

const OPTS: ConfidenceOptions = { ...DEFAULTS.confidence };
const dist = (labels: readonly string[], probs: number[]): Distribution => ({ labels, probs });
const noulAgg = (p: number) => dist(["no", "yes"], [1 - p, p]);

/** Direct transcription of spec 11.4–11.6 for the defaults, used as an oracle. */
function specAdjusted(base: number, agreement: number, nEff: number, o = OPTS): number {
  const s = 1 - Math.exp(-o.lambda * (nEff - 1));
  const g = Math.min(1, Math.max(0, (agreement - o.agreementFloor) / (1 - o.agreementFloor))) ** o.agreementExponent;
  return base + (o.cap - base) * s * g;
}

// ---------------------------------------------------------------------------
// effectiveChunkCount (spec 11.3)
// ---------------------------------------------------------------------------

describe("effectiveChunkCount", () => {
  it("is 1 for a single chunk", () => {
    expect(effectiveChunkCount([{ tokens: 5000, uniqueTokens: 5000 }])).toBe(1);
  });

  it("is 2.9 for three equal chunks with 5% overlap (spec 12 example)", () => {
    const chunks = [
      { tokens: 1000, uniqueTokens: 1000 },
      { tokens: 1000, uniqueTokens: 950 },
      { tokens: 1000, uniqueTokens: 950 },
    ];
    expect(effectiveChunkCount(chunks)).toBeCloseTo(2.9, 12);
  });

  it("is 3.85 for four equal chunks with 5% overlap (spec 12 noul example)", () => {
    const chunks = [
      { tokens: 2000, uniqueTokens: 2000 },
      { tokens: 2000, uniqueTokens: 1900 },
      { tokens: 2000, uniqueTokens: 1900 },
      { tokens: 2000, uniqueTokens: 1900 },
    ];
    expect(effectiveChunkCount(chunks)).toBeCloseTo(3.85, 12);
  });

  it("equals N without overlap", () => {
    const chunks = Array.from({ length: 7 }, () => ({ tokens: 300, uniqueTokens: 300 }));
    expect(effectiveChunkCount(chunks)).toBe(7);
  });

  it("measures new content in units of the largest chunk (short final chunk)", () => {
    const chunks = [
      { tokens: 1000, uniqueTokens: 1000 },
      { tokens: 1000, uniqueTokens: 950 },
      { tokens: 200, uniqueTokens: 150 },
    ];
    // (1000 + 950 + 150) / 1000
    expect(effectiveChunkCount(chunks)).toBeCloseTo(2.1, 12);
  });

  it("counts a tiny chunk as its share of a full chunk, not as a whole one (overlap 0)", () => {
    const full = { tokens: 1000, uniqueTokens: 1000 };
    const tiny = { tokens: 10, uniqueTokens: 10 };
    expect(effectiveChunkCount([full, tiny])).toBeCloseTo(1.01, 12);
    expect(effectiveChunkCount([tiny, full])).toBeCloseTo(1.01, 12);
    // README: a short final chunk adds less than a full one.
    expect(effectiveChunkCount([full, tiny])).toBeLessThan(effectiveChunkCount([full, full]));
  });

  it("treats tokens < 1 as 1", () => {
    expect(
      effectiveChunkCount([
        { tokens: 0, uniqueTokens: 1 },
        { tokens: 0, uniqueTokens: 1 },
      ]),
    ).toBe(2);
  });

  it("never exceeds N, even for inconsistent input (unique > tokens)", () => {
    expect(
      effectiveChunkCount([
        { tokens: 10, uniqueTokens: 10 },
        { tokens: 10, uniqueTokens: 50 },
      ]),
    ).toBe(2);
  });

  it("rejects an empty plan", () => {
    expect(() => effectiveChunkCount([])).toThrow(JevInfiniteCTXError);
  });

  it.each([
    [{ tokens: Number.NaN, uniqueTokens: 1 }],
    [{ tokens: 10, uniqueTokens: -1 }],
    [{ tokens: Number.POSITIVE_INFINITY, uniqueTokens: 1 }],
  ])("rejects invalid token counts %j, in any chunk", (bad) => {
    expect(() => effectiveChunkCount([{ tokens: 10, uniqueTokens: 10 }, bad])).toThrow(
      JevInfiniteCTXError,
    );
    expect(() => effectiveChunkCount([bad, { tokens: 10, uniqueTokens: 10 }])).toThrow(
      JevInfiniteCTXError,
    );
  });

  it("lies in [1, N] and matches 1 + (N − 1)(1 − r) for uniform chunks (property)", () => {
    const chunkArb = fc
      .integer({ min: 1, max: 100_000 })
      .chain((tokens) =>
        fc.record({ tokens: fc.constant(tokens), uniqueTokens: fc.integer({ min: 1, max: tokens }) }),
      );
    fc.assert(
      fc.property(fc.array(chunkArb, { minLength: 1, maxLength: 40 }), (chunks) => {
        const nEff = effectiveChunkCount(chunks);
        expect(nEff).toBeGreaterThanOrEqual(1);
        expect(nEff).toBeLessThanOrEqual(chunks.length);
      }),
    );
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 50 }),
        fc.integer({ min: 100, max: 30_000 }),
        fc.double({ min: 0, max: 0.5, noNaN: true }),
        (n, tokens, r) => {
          const overlap = Math.floor(tokens * r);
          const chunks = Array.from({ length: n }, (_, i) => ({
            tokens,
            uniqueTokens: i === 0 ? tokens : tokens - overlap,
          }));
          expect(effectiveChunkCount(chunks)).toBeCloseTo(1 + (n - 1) * (1 - overlap / tokens), 9);
        },
      ),
    );
  });
});

// ---------------------------------------------------------------------------
// baseConfidence (spec 11.2)
// ---------------------------------------------------------------------------

describe("baseConfidence", () => {
  describe("noul (derived)", () => {
    it.each([
      [0.5, 0],
      [0.75, 0.5],
      [1, 1],
      [0, 1],
      [0.84, 0.68],
      [0.25, 0.5],
    ])("P_agg[yes] = %s gives base %s", (p, expected) => {
      const result = baseConfidence({
        type: "noul",
        aggregate: noulAgg(p),
        chunkConfidences: [undefined, undefined],
        weights: [0.5, 0.5],
      });
      expect(result.source).toBe("derived");
      expect(result.base).toBeCloseTo(expected, 12);
    });

    it("finds the yes label by name", () => {
      const result = baseConfidence({
        type: "noul",
        aggregate: dist(["yes", "no"], [0.9, 0.1]),
        chunkConfidences: [undefined],
        weights: [1],
      });
      expect(result.base).toBeCloseTo(0.8, 12);
    });

    it("rejects an aggregate without a yes label", () => {
      expect(() =>
        baseConfidence({
          type: "noul",
          aggregate: dist(["a", "b"], [0.5, 0.5]),
          chunkConfidences: [undefined],
          weights: [1],
        }),
      ).toThrow(/"yes"/);
    });
  });

  describe("choice and score", () => {
    const agg = dist(["tree", "rock", "other"], [0.7, 0.2, 0.1]);

    it("is the weighted mean of Jev confidences, labeled jev", () => {
      // 0.75·0.8 + 0.25·0.6 = 0.75
      const result = baseConfidence({
        type: "choice",
        aggregate: agg,
        chunkConfidences: [0.8, 0.6],
        weights: [0.75, 0.25],
      });
      expect(result).toEqual({ base: expect.closeTo(0.75, 12), source: "jev" });
    });

    it("renormalizes raw weights", () => {
      const result = baseConfidence({
        type: "score",
        aggregate: dist(["0", "1"], [0.5, 0.5]),
        chunkConfidences: [0.8, 0.6],
        weights: [3000, 1000],
      });
      expect(result.base).toBeCloseTo(0.75, 12);
      expect(result.source).toBe("jev");
    });

    it("is Jev's own confidence for a single chunk", () => {
      const result = baseConfidence({
        type: "choice",
        aggregate: agg,
        chunkConfidences: [0.67],
        weights: [1],
      });
      expect(result).toEqual({ base: 0.67, source: "jev" });
    });

    it("falls back to 1 − H/ln K, labeled derived, when any confidence is missing", () => {
      const result = baseConfidence({
        type: "choice",
        aggregate: agg,
        chunkConfidences: [0.9, undefined],
        weights: [0.5, 0.5],
      });
      const h = -(0.7 * Math.log(0.7) + 0.2 * Math.log(0.2) + 0.1 * Math.log(0.1));
      expect(result.source).toBe("derived");
      expect(result.base).toBeCloseTo(1 - h / Math.log(3), 12);
      expect(result.base).toBeCloseTo(0.27015330083790245, 12);
    });

    it("derived fallback: one-hot → 1, uniform → 0, 0·ln 0 = 0", () => {
      const derive = (probs: number[]) =>
        baseConfidence({
          type: "score",
          aggregate: dist(probs.map((_, j) => String(j)), probs),
          chunkConfidences: [undefined],
          weights: [1],
        }).base;
      expect(derive([0, 1, 0, 0])).toBe(1);
      expect(derive([0.25, 0.25, 0.25, 0.25])).toBeCloseTo(0, 12);
      expect(derive([0.5, 0.5, 0, 0])).toBeCloseTo(0.5, 12);
    });

    it("clamps to [0, 1]", () => {
      const result = baseConfidence({
        type: "choice",
        aggregate: agg,
        chunkConfidences: [1, 1, 1],
        weights: [0.1, 0.2, 0.7],
      });
      expect(result.base).toBeLessThanOrEqual(1);
      expect(result.base).toBeCloseTo(1, 12);
    });
  });

  describe("validation", () => {
    const agg = dist(["a", "b"], [0.5, 0.5]);

    it("requires at least one chunk", () => {
      expect(() =>
        baseConfidence({ type: "choice", aggregate: agg, chunkConfidences: [], weights: [] }),
      ).toThrow(JevInfiniteCTXError);
    });

    it("requires one weight per chunk", () => {
      expect(() =>
        baseConfidence({ type: "choice", aggregate: agg, chunkConfidences: [0.5, 0.5], weights: [1] }),
      ).toThrow(JevInfiniteCTXError);
      expect(() =>
        baseConfidence({ type: "noul", aggregate: noulAgg(0.5), chunkConfidences: [undefined], weights: [1, 1] }),
      ).toThrow(JevInfiniteCTXError);
    });

    it("rejects non-finite chunk confidences", () => {
      expect(() =>
        baseConfidence({ type: "choice", aggregate: agg, chunkConfidences: [Number.NaN], weights: [1] }),
      ).toThrow(JevInfiniteCTXError);
    });

    it("rejects an invalid aggregate", () => {
      expect(() =>
        baseConfidence({
          type: "choice",
          aggregate: dist(["a", "b"], [Number.NaN, 1]),
          chunkConfidences: [undefined],
          weights: [1],
        }),
      ).toThrow(JevInfiniteCTXError);
    });
  });

  it("always lies in [0, 1] and is labeled correctly (property)", () => {
    const caseArb = fc
      .record({ k: fc.integer({ min: 2, max: 8 }), n: fc.integer({ min: 1, max: 8 }) })
      .chain(({ k, n }) =>
        fc.record({
          type: fc.constantFrom("choice" as const, "score" as const),
          raw: fc.array(fc.nat({ max: 100 }), { minLength: k, maxLength: k }).filter((xs) => xs.some((x) => x > 0)),
          confidences: fc.array(fc.option(fc.double({ min: 0, max: 1, noNaN: true }), { nil: undefined }), {
            minLength: n,
            maxLength: n,
          }),
          weights: fc.array(fc.double({ min: 1e-3, max: 1e3, noNaN: true }), { minLength: n, maxLength: n }),
        }),
      );
    fc.assert(
      fc.property(caseArb, ({ type, raw, confidences, weights }) => {
        const total = raw.reduce((a, b) => a + b, 0);
        const agg = dist(raw.map((_, j) => String(j)), raw.map((x) => x / total));
        const { base, source } = baseConfidence({ type, aggregate: agg, chunkConfidences: confidences, weights });
        expect(base).toBeGreaterThanOrEqual(0);
        expect(base).toBeLessThanOrEqual(1);
        expect(source).toBe(confidences.every((c) => c !== undefined) ? "jev" : "derived");
        if (source === "jev") {
          const defined = confidences as number[];
          expect(base).toBeGreaterThanOrEqual(Math.min(...defined) - 1e-12);
          expect(base).toBeLessThanOrEqual(Math.max(...defined) + 1e-12);
        }
      }),
    );
    fc.assert(
      fc.property(fc.double({ min: 0, max: 1, noNaN: true }), (p) => {
        const { base, source } = baseConfidence({
          type: "noul",
          aggregate: noulAgg(p),
          chunkConfidences: [undefined],
          weights: [1],
        });
        expect(source).toBe("derived");
        expect(base).toBeGreaterThanOrEqual(0);
        expect(base).toBeLessThanOrEqual(1);
        expect(base).toBeCloseTo(Math.abs(2 * p - 1), 12);
      }),
    );
  });
});

// ---------------------------------------------------------------------------
// applyC3 (spec 11.4–11.6)
// ---------------------------------------------------------------------------

describe("applyC3", () => {
  it("matches the spec 12 choice example: base 0.81, A 0.93, N_eff 2.9 → ≈ 0.86", () => {
    // S = 1 − e^(−0.25·1.9) = 0.378115; G = (0.43/0.5)² = 0.7396;
    // adjustment = 0.17 · S · G = 0.047541; adjusted = 0.857541.
    const r = applyC3({ base: 0.81, agreement: 0.93, effectiveCount: 2.9, chunkCount: 3, options: OPTS });
    expect(r.components.saturation).toBeCloseTo(0.3781149435349799, 12);
    expect(r.components.gate).toBeCloseTo(0.7396, 12);
    expect(r.adjustment).toBeCloseTo(0.04754114808054008, 12);
    expect(r.adjusted).toBeCloseTo(0.8575411480805402, 12);
    expect(r.adjusted).toBeCloseTo(0.86, 2);
    expect(r.adjusted).toBeCloseTo(specAdjusted(0.81, 0.93, 2.9), 12);
    expect(r.method).toBe("c3-v1");
  });

  it("matches the formula for the spec 12 noul example: base 0.68, A 0.91, N_eff 3.85", () => {
    // S = 1 − e^(−0.25·2.85) = 0.509583; G = 0.82² = 0.6724; adjusted = 0.68 + 0.30·S·G.
    const r = applyC3({ base: 0.68, agreement: 0.91, effectiveCount: 3.85, chunkCount: 4, options: OPTS });
    expect(r.components.saturation).toBeCloseTo(0.5095833779240939, 12);
    expect(r.components.gate).toBeCloseTo(0.6724, 12);
    expect(r.adjusted).toBeCloseTo(0.7827931589948482, 12);
  });

  it("reports every component", () => {
    const r = applyC3({ base: 0.5, agreement: 1, effectiveCount: 5, chunkCount: 5, options: OPTS });
    expect(r.components).toEqual({
      effectiveChunkCount: 5,
      saturation: expect.closeTo(1 - Math.exp(-1), 12),
      gate: 1,
      cap: 0.98,
      lambda: 0.25,
      agreementFloor: 0.5,
      agreementExponent: 2,
    });
    expect(r.adjustment).toBeCloseTo(0.48 * (1 - Math.exp(-1)), 12);
    expect(r.adjusted).toBeCloseTo(r.components.cap - 0.48 * Math.exp(-1), 12);
  });

  it("makes no adjustment for one chunk, even with an inconsistent N_eff", () => {
    const one = applyC3({ base: 0.6, agreement: 1, effectiveCount: 1, chunkCount: 1, options: OPTS });
    expect(one).toMatchObject({ adjusted: 0.6, adjustment: 0, method: "c3-v1" });
    expect(one.components.saturation).toBe(0);

    const inconsistent = applyC3({ base: 0.6, agreement: 1, effectiveCount: 3, chunkCount: 1, options: OPTS });
    expect(inconsistent.adjusted).toBe(0.6);
    expect(inconsistent.adjustment).toBe(0);
  });

  it("makes no adjustment when agreement is at or below the floor", () => {
    for (const agreement of [0.5, 0.3, 0]) {
      const r = applyC3({ base: 0.6, agreement, effectiveCount: 10, chunkCount: 10, options: OPTS });
      expect(r.components.gate).toBe(0);
      expect(r.adjustment).toBe(0);
      expect(r.adjusted).toBe(0.6);
    }
  });

  it("gives a small boost for moderate agreement and most of it for near-perfect agreement", () => {
    const at = (agreement: number) =>
      applyC3({ base: 0.5, agreement, effectiveCount: 20, chunkCount: 20, options: OPTS }).adjustment;
    expect(at(0.6)).toBeGreaterThan(0);
    expect(at(0.6)).toBeLessThan(at(0.99));
    expect(at(0.6)).toBeCloseTo(0.48 * (1 - Math.exp(-0.25 * 19)) * 0.04, 12);
  });

  it("never lowers a base at or above the cap", () => {
    for (const base of [0.98, 0.99, 1]) {
      const r = applyC3({ base, agreement: 1, effectiveCount: 50, chunkCount: 50, options: OPTS });
      expect(r.adjusted).toBe(base);
      expect(r.adjustment).toBe(0);
    }
  });

  it("approaches but never exceeds the cap", () => {
    const r = applyC3({ base: 0.1, agreement: 1, effectiveCount: 1e6, chunkCount: 1000, options: OPTS });
    expect(r.components.saturation).toBe(1);
    expect(r.adjusted).toBe(0.98);
    const withCapOne = applyC3({
      base: 0.1,
      agreement: 1,
      effectiveCount: 1e6,
      chunkCount: 1000,
      options: { ...OPTS, cap: 1 },
    });
    expect(withCapOne.adjusted).toBe(1);
  });

  it("treats N_eff below 1 as 1", () => {
    const r = applyC3({ base: 0.4, agreement: 1, effectiveCount: 0.2, chunkCount: 2, options: OPTS });
    expect(r.components.effectiveChunkCount).toBe(1);
    expect(r.components.saturation).toBe(0);
    expect(r.adjustment).toBe(0);
  });

  it('with method "none" reports adjusted = base but still computes components', () => {
    const none = applyC3({
      base: 0.81,
      agreement: 0.93,
      effectiveCount: 2.9,
      chunkCount: 3,
      options: { ...OPTS, method: "none" },
    });
    const c3 = applyC3({ base: 0.81, agreement: 0.93, effectiveCount: 2.9, chunkCount: 3, options: OPTS });
    expect(none.method).toBe("none");
    expect(none.adjusted).toBe(0.81);
    expect(none.adjustment).toBe(0);
    expect(none.components).toEqual(c3.components);
    expect(none.components.saturation).toBeGreaterThan(0);
  });

  describe("validation", () => {
    const valid = { base: 0.5, agreement: 0.9, effectiveCount: 2, chunkCount: 2 };

    it.each([
      ["cap", { cap: 0 }],
      ["cap", { cap: 1.5 }],
      ["cap", { cap: Number.NaN }],
      ["lambda", { lambda: 0 }],
      ["lambda", { lambda: Number.POSITIVE_INFINITY }],
      ["agreementFloor", { agreementFloor: 1 }],
      ["agreementFloor", { agreementFloor: -0.1 }],
      ["agreementExponent", { agreementExponent: 0 }],
      ["agreementExponent", { agreementExponent: Number.NaN }],
      ["method", { method: "c4" }],
    ])("rejects invalid confidence.%s", (field, patch) => {
      const options = { ...OPTS, ...patch } as ConfidenceOptions;
      expect(() => applyC3({ ...valid, options })).toThrow(JevValidationError);
      expect(() => applyC3({ ...valid, options })).toThrow(`confidence.${field}`);
    });

    it.each([
      ["base", { base: Number.NaN }],
      ["base", { base: 1.1 }],
      ["base", { base: -0.1 }],
      ["agreement", { agreement: 2 }],
      ["effectiveCount", { effectiveCount: Number.NaN }],
      ["chunkCount", { chunkCount: 0 }],
      ["chunkCount", { chunkCount: 2.5 }],
    ])("rejects invalid %s", (_, patch) => {
      expect(() => applyC3({ ...valid, ...patch, options: OPTS })).toThrow(JevInfiniteCTXError);
    });
  });

  describe("properties", () => {
    const unit = fc.double({ min: 0, max: 1, noNaN: true });
    const optionsArb: fc.Arbitrary<ConfidenceOptions> = fc.record({
      method: fc.constant("c3" as const),
      cap: fc.double({ min: 0.01, max: 1, noNaN: true }),
      lambda: fc.double({ min: 1e-3, max: 10, noNaN: true }),
      agreementFloor: fc.double({ min: 0, max: 0.99, noNaN: true }),
      agreementExponent: fc.double({ min: 0.1, max: 8, noNaN: true }),
    });
    const nEffArb = fc.double({ min: 0, max: 100, noNaN: true });
    const chunkCountArb = fc.integer({ min: 1, max: 100 });
    // libm exp/pow are not guaranteed to be monotone at the last ulp.
    const ULP_SLACK = 1e-12;

    it("keeps adjusted within [base, max(base, cap)]", () => {
      fc.assert(
        fc.property(unit, unit, nEffArb, chunkCountArb, optionsArb, (base, agreement, effectiveCount, chunkCount, options) => {
          const r = applyC3({ base, agreement, effectiveCount, chunkCount, options });
          expect(r.adjusted).toBeGreaterThanOrEqual(base);
          expect(r.adjusted).toBeLessThanOrEqual(Math.max(base, options.cap));
          expect(r.adjusted).toBeLessThanOrEqual(1);
          expect(r.adjustment).toBe(r.adjusted - base);
          expect(r.adjustment).toBeGreaterThanOrEqual(0);
          expect(r.components.saturation).toBeGreaterThanOrEqual(0);
          expect(r.components.saturation).toBeLessThanOrEqual(1);
          expect(r.components.gate).toBeGreaterThanOrEqual(0);
          expect(r.components.gate).toBeLessThanOrEqual(1);
          expect(r.method).toBe("c3-v1");
        }),
      );
    });

    it("matches the spec formula", () => {
      fc.assert(
        fc.property(unit, unit, fc.double({ min: 1, max: 100, noNaN: true }), optionsArb, (base, agreement, nEff, options) => {
          const r = applyC3({ base, agreement, effectiveCount: nEff, chunkCount: 2, options });
          const expected = base + Math.max(0, options.cap - base) * r.components.saturation * r.components.gate;
          expect(r.adjusted).toBeCloseTo(Math.min(Math.max(base, options.cap), expected), 12);
          if (base <= options.cap) {
            expect(r.adjusted).toBeCloseTo(specAdjusted(base, agreement, nEff, options), 9);
          }
        }),
      );
    });

    it("makes zero adjustment for one chunk", () => {
      fc.assert(
        fc.property(unit, unit, nEffArb, optionsArb, (base, agreement, effectiveCount, options) => {
          const r = applyC3({ base, agreement, effectiveCount, chunkCount: 1, options });
          expect(r.adjustment).toBe(0);
          expect(r.adjusted).toBe(base);
        }),
      );
    });

    it("makes zero adjustment whenever agreement <= agreementFloor", () => {
      fc.assert(
        fc.property(unit, unit, nEffArb, chunkCountArb, optionsArb, (base, t, effectiveCount, chunkCount, options) => {
          const agreement = options.agreementFloor * t;
          const r = applyC3({ base, agreement, effectiveCount, chunkCount, options });
          expect(r.components.gate).toBe(0);
          expect(r.adjustment).toBe(0);
          expect(r.adjusted).toBe(base);
        }),
      );
    });

    it("is monotone non-decreasing in N_eff", () => {
      fc.assert(
        fc.property(unit, unit, nEffArb, nEffArb, optionsArb, (base, agreement, n1, n2, options) => {
          const [lo, hi] = n1 <= n2 ? [n1, n2] : [n2, n1];
          const a = applyC3({ base, agreement, effectiveCount: lo, chunkCount: 5, options });
          const b = applyC3({ base, agreement, effectiveCount: hi, chunkCount: 5, options });
          expect(b.adjusted).toBeGreaterThanOrEqual(a.adjusted - ULP_SLACK);
        }),
      );
    });

    it("is monotone non-decreasing in agreement", () => {
      fc.assert(
        fc.property(unit, unit, unit, nEffArb, optionsArb, (base, a1, a2, effectiveCount, options) => {
          const [lo, hi] = a1 <= a2 ? [a1, a2] : [a2, a1];
          const a = applyC3({ base, agreement: lo, effectiveCount, chunkCount: 5, options });
          const b = applyC3({ base, agreement: hi, effectiveCount, chunkCount: 5, options });
          expect(b.adjusted).toBeGreaterThanOrEqual(a.adjusted - ULP_SLACK);
        }),
      );
    });

    it('with method "none" always reports adjusted = base', () => {
      fc.assert(
        fc.property(unit, unit, nEffArb, chunkCountArb, optionsArb, (base, agreement, effectiveCount, chunkCount, options) => {
          const r = applyC3({ base, agreement, effectiveCount, chunkCount, options: { ...options, method: "none" } });
          expect(r).toMatchObject({ adjusted: base, adjustment: 0, method: "none" });
        }),
      );
    });
  });
});

// ---------------------------------------------------------------------------
// End to end: probability → aggregation → agreement → C3
// ---------------------------------------------------------------------------

describe("C3 pipeline", () => {
  const chunks = [
    { tokens: 1000, uniqueTokens: 1000 },
    { tokens: 1000, uniqueTokens: 950 },
    { tokens: 1000, uniqueTokens: 950 },
  ];
  const weights = chunks.map((c) => c.uniqueTokens / 2900);
  const question = {
    type: "choice" as const,
    instructions: "What is the primary subject?",
    criteria: { tree: "Trees.", rock: "Rocks.", other: "Neither." },
  };

  it("boosts confidence when chunks agree", () => {
    const answers = [
      { type: "choice" as const, choice: "tree", probabilities: { tree: 0.8, rock: 0.15, other: 0.05 }, confidence: 0.8 },
      { type: "choice" as const, choice: "tree", probabilities: { tree: 0.75, rock: 0.2, other: 0.05 }, confidence: 0.82 },
      { type: "choice" as const, choice: "tree", probabilities: { tree: 0.78, rock: 0.14, other: 0.08 }, confidence: 0.81 },
    ];
    const distributions = answers.map((a) => toDistribution(question, a));
    const agg = aggregate(distributions, weights, "weighted_mean").distribution;
    const { agreement } = computeAgreement(distributions, agg, weights);
    const { base, source } = baseConfidence({
      type: "choice",
      aggregate: agg,
      chunkConfidences: answers.map((a) => a.confidence),
      weights,
    });
    const nEff = effectiveChunkCount(chunks);
    const r = applyC3({ base, agreement, effectiveCount: nEff, chunkCount: 3, options: OPTS });

    expect(nEff).toBeCloseTo(2.9, 12);
    expect(source).toBe("jev");
    expect(base).toBeCloseTo((1000 * 0.8 + 950 * 0.82 + 950 * 0.81) / 2900, 12);
    expect(agreement).toBeGreaterThan(0.9);
    expect(r.adjustment).toBeGreaterThan(0);
    expect(r.adjusted).toBeCloseTo(specAdjusted(base, agreement, nEff), 12);
    expect(r.adjusted).toBeLessThanOrEqual(OPTS.cap);
  });

  const noul = { type: "noul" as const, instructions: "Is it about trees?" };

  function runNoul(yes: readonly number[], w: readonly number[], nEff: number) {
    const distributions = yes.map((p) => toDistribution(noul, { type: "noul", noul: p }));
    const agg = aggregate(distributions, [...w], "weighted_mean").distribution;
    const { agreement } = computeAgreement(distributions, agg, [...w]);
    const { base, source } = baseConfidence({
      type: "noul",
      aggregate: agg,
      chunkConfidences: yes.map(() => undefined),
      weights: w,
    });
    expect(source).toBe("derived");
    return { agreement, ...applyC3({ base, agreement, effectiveCount: nEff, chunkCount: yes.length, options: OPTS }) };
  }

  it("suppresses the boost when chunks conflict", () => {
    const nEff = effectiveChunkCount(chunks);
    const agreeing = runNoul([0.9, 0.92, 0.88], weights, nEff);
    const conflicting = runNoul([1, 0, 1], weights, nEff);

    expect(agreeing.agreement).toBeGreaterThan(0.95);
    expect(conflicting.agreement).toBeLessThan(0.6);
    expect(conflicting.components.gate).toBeLessThan(0.02);
    expect(conflicting.adjustment).toBeLessThan(agreeing.adjustment / 10);
  });

  it("gives exactly zero boost to an even split (A = 0.5 = A_floor)", () => {
    const evenSplit = runNoul([1, 0], [0.5, 0.5], 2);
    expect(evenSplit.agreement).toBe(0.5);
    expect(evenSplit.components.gate).toBe(0);
    expect(evenSplit.adjustment).toBe(0);
  });
});
