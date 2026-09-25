import { describe, it, expect } from "vitest";
import fc from "fast-check";
import { computeAgreement, totalVariation } from "../src/agreement.js";
import { aggregate } from "../src/aggregation.js";
import { JevInfiniteCTXError } from "../src/errors.js";
import type { AggregationMethod, Distribution } from "../src/types.js";

const METHODS: readonly AggregationMethod[] = ["weighted_mean", "mean", "median", "min", "max"];
const AB = ["a", "b"] as const;
const dist = (labels: readonly string[], probs: number[]): Distribution => ({ labels, probs });
const sum = (xs: readonly number[]) => xs.reduce((a, b) => a + b, 0);

function probsArb(k: number): fc.Arbitrary<number[]> {
  const normalized = fc
    .array(fc.nat({ max: 1000 }), { minLength: k, maxLength: k })
    .filter((xs) => sum(xs) > 0)
    .map((xs) => {
      const s = sum(xs);
      return xs.map((x) => x / s);
    });
  const oneHot = fc
    .nat({ max: k - 1 })
    .map((hot) => Array.from({ length: k }, (_, j) => (j === hot ? 1 : 0)));
  return fc.oneof(normalized, oneHot);
}

// ---------------------------------------------------------------------------
// totalVariation
// ---------------------------------------------------------------------------

describe("totalVariation", () => {
  it("is 0.5 · Σ|p − q| (spec 10)", () => {
    expect(totalVariation([0.8, 0.2], [0.5, 0.5])).toBeCloseTo(0.3, 12);
    expect(totalVariation([0.7, 0.2, 0.1], [0.1, 0.2, 0.7])).toBeCloseTo(0.6, 12);
  });

  it("is 0 for identical vectors and 1 for disjoint one-hots", () => {
    expect(totalVariation([0.3, 0.7], [0.3, 0.7])).toBe(0);
    expect(totalVariation([1, 0], [0, 1])).toBe(1);
    expect(totalVariation([1, 0, 0], [0, 0, 1])).toBe(1);
  });

  it("clamps float noise above 1", () => {
    expect(totalVariation([1.0000000000000002, 0], [0, 1])).toBe(1);
  });

  it("handles empty vectors", () => {
    expect(totalVariation([], [])).toBe(0);
  });

  it("rejects vectors of different length", () => {
    expect(() => totalVariation([1, 0], [1, 0, 0])).toThrow(/equal length/);
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY])("rejects a %s entry", (x) => {
    expect(() => totalVariation([x, 0], [0.5, 0.5])).toThrow(JevInfiniteCTXError);
  });

  const pairArb = fc
    .integer({ min: 1, max: 10 })
    .chain((k) => fc.tuple(probsArb(k), probsArb(k), probsArb(k)));

  it("is bounded in [0, 1] and exactly symmetric (property)", () => {
    fc.assert(
      fc.property(pairArb, ([p, q]) => {
        const pq = totalVariation(p, q);
        expect(pq).toBeGreaterThanOrEqual(0);
        expect(pq).toBeLessThanOrEqual(1);
        expect(Object.is(pq, totalVariation(q, p))).toBe(true);
        expect(totalVariation(p, p)).toBe(0);
      }),
    );
  });

  it("satisfies the triangle inequality (property)", () => {
    fc.assert(
      fc.property(pairArb, ([p, q, r]) => {
        expect(totalVariation(p, r)).toBeLessThanOrEqual(
          totalVariation(p, q) + totalVariation(q, r) + 1e-12,
        );
      }),
    );
  });
});

// ---------------------------------------------------------------------------
// computeAgreement
// ---------------------------------------------------------------------------

describe("computeAgreement", () => {
  it("matches a hand-computed example", () => {
    // Aggregate [0.7, 0.3]; TV_1 = 0.1, TV_2 = 0.3; D = 0.75·0.1 + 0.25·0.3 = 0.15; A = 0.85.
    const chunks = [dist(AB, [0.8, 0.2]), dist(AB, [0.4, 0.6])];
    const result = computeAgreement(chunks, dist(AB, [0.7, 0.3]), [0.75, 0.25]);
    expect(result.perChunk).toHaveLength(2);
    expect(result.perChunk[0]).toBeCloseTo(0.1, 12);
    expect(result.perChunk[1]).toBeCloseTo(0.3, 12);
    expect(result.disagreement).toBeCloseTo(0.15, 12);
    expect(result.agreement).toBeCloseTo(0.85, 12);
  });

  it("agrees with aggregate() on the same example", () => {
    const chunks = [dist(AB, [0.8, 0.2]), dist(AB, [0.4, 0.6])];
    const agg = aggregate(chunks, [3, 1], "weighted_mean").distribution;
    expect(computeAgreement(chunks, agg, [3, 1]).agreement).toBeCloseTo(0.85, 12);
  });

  it("renormalizes raw weights", () => {
    const chunks = [dist(AB, [0.8, 0.2]), dist(AB, [0.4, 0.6])];
    const result = computeAgreement(chunks, dist(AB, [0.7, 0.3]), [3000, 1000]);
    expect(result.agreement).toBeCloseTo(0.85, 12);
  });

  it("two sharply conflicting chunks give A = 0.5", () => {
    const chunks = [dist(AB, [1, 0]), dist(AB, [0, 1])];
    const result = computeAgreement(chunks, dist(AB, [0.5, 0.5]), [0.5, 0.5]);
    expect(result.perChunk).toEqual([0.5, 0.5]);
    expect(result.agreement).toBe(0.5);
    expect(result.disagreement).toBe(0.5);
  });

  it("identical chunks equal to the aggregate give exactly A = 1", () => {
    const p = dist(AB, [0.18, 0.82]);
    const result = computeAgreement([p, p, p], p, [0.4, 0.3, 0.3]);
    expect(result).toEqual({ agreement: 1, disagreement: 0, perChunk: [0, 0, 0] });
  });

  it("identical one-hot chunks aggregate and agree exactly", () => {
    const p = dist(AB, [0, 1]);
    const weights = [1000, 950, 950];
    const agg = aggregate([p, p, p], weights, "weighted_mean").distribution;
    expect(computeAgreement([p, p, p], agg, weights).agreement).toBe(1);
  });

  it("a single chunk agrees with itself (spec 15)", () => {
    const p = dist(AB, [0.3, 0.7]);
    expect(computeAgreement([p], p, [1])).toEqual({ agreement: 1, disagreement: 0, perChunk: [0] });
    // Even against a different aggregate (e.g. a renormalized min/max), one chunk is agreement 1.
    expect(computeAgreement([p], dist(AB, [0.5, 0.5]), [1]).agreement).toBe(1);
  });

  it("returns per-chunk TV in the original chunk order", () => {
    const chunks = [dist(AB, [1, 0]), dist(AB, [0.5, 0.5]), dist(AB, [0.9, 0.1])];
    const agg = dist(AB, [0.8, 0.2]);
    const { perChunk } = computeAgreement(chunks, agg, [1, 1, 1]);
    expect(perChunk[0]).toBeCloseTo(0.2, 12);
    expect(perChunk[1]).toBeCloseTo(0.3, 12);
    expect(perChunk[2]).toBeCloseTo(0.1, 12);
  });

  it("weights low-content chunks less", () => {
    const chunks = [dist(AB, [1, 0]), dist(AB, [0, 1])];
    const agg = dist(AB, [0.9, 0.1]);
    // TV = [0.1, 0.9]; D = 0.9·0.1 + 0.1·0.9 = 0.18.
    expect(computeAgreement(chunks, agg, [9, 1]).disagreement).toBeCloseTo(0.18, 12);
  });

  describe("validation", () => {
    const p = dist(AB, [0.5, 0.5]);

    it("requires at least one distribution", () => {
      expect(() => computeAgreement([], p, [])).toThrow(/at least one distribution/);
    });

    it("requires one weight per distribution", () => {
      expect(() => computeAgreement([p, p], p, [1])).toThrow(JevInfiniteCTXError);
      expect(() => computeAgreement([p], p, [1, 1])).toThrow(JevInfiniteCTXError);
    });

    it("rejects invalid weights", () => {
      expect(() => computeAgreement([p, p], p, [0, 0])).toThrow(JevInfiniteCTXError);
      expect(() => computeAgreement([p, p], p, [1, Number.NaN])).toThrow(JevInfiniteCTXError);
    });

    it("requires labels matching the aggregate", () => {
      expect(() => computeAgreement([p, dist(["b", "a"], [0.5, 0.5])], p, [1, 1])).toThrow(
        /different labels/,
      );
      expect(() => computeAgreement([p], dist(["x", "y"], [0.5, 0.5]), [1])).toThrow(
        /different labels/,
      );
    });

    it("rejects probability vectors of the wrong length", () => {
      expect(() => computeAgreement([p, dist(AB, [1])], p, [1, 1])).toThrow(JevInfiniteCTXError);
    });
  });

  describe("properties", () => {
    const caseArb = fc
      .record({ k: fc.integer({ min: 2, max: 7 }), n: fc.integer({ min: 1, max: 9 }) })
      .chain(({ k, n }) =>
        fc.record({
          labels: fc.constant(Array.from({ length: k }, (_, j) => `L${j}`)),
          probs: fc.array(probsArb(k), { minLength: n, maxLength: n }),
          weights: fc.array(fc.double({ min: 1e-3, max: 1e3, noNaN: true }), {
            minLength: n,
            maxLength: n,
          }),
          method: fc.constantFrom(...METHODS),
        }),
      );

    it("agreement and disagreement are complementary and within [0, 1]", () => {
      fc.assert(
        fc.property(caseArb, ({ labels, probs, weights, method }) => {
          const distributions = probs.map((p) => dist(labels, p));
          const agg = aggregate(distributions, weights, method).distribution;
          const { agreement, disagreement, perChunk } = computeAgreement(distributions, agg, weights);

          expect(agreement).toBeGreaterThanOrEqual(0);
          expect(agreement).toBeLessThanOrEqual(1);
          expect(disagreement).toBeGreaterThanOrEqual(0);
          expect(disagreement).toBeLessThanOrEqual(1);
          expect(agreement).toBe(1 - disagreement);
          expect(perChunk).toHaveLength(distributions.length);
          for (const tv of perChunk) {
            expect(tv).toBeGreaterThanOrEqual(0);
            expect(tv).toBeLessThanOrEqual(1);
          }
          if (distributions.length > 1) {
            perChunk.forEach((tv, i) => expect(tv).toBe(totalVariation(probs[i]!, agg.probs)));
            // Weighted disagreement is a convex combination of the per-chunk distances.
            expect(disagreement).toBeGreaterThanOrEqual(Math.min(...perChunk) - 1e-12);
            expect(disagreement).toBeLessThanOrEqual(Math.max(...perChunk) + 1e-12);
          }
        }),
      );
    });

    it("identical distributions agree exactly (aggregate equal to them)", () => {
      fc.assert(
        fc.property(caseArb, ({ labels, probs, weights }) => {
          const p = dist(labels, probs[0]!);
          const same = probs.map(() => p);
          const result = computeAgreement(same, p, weights);
          expect(result.agreement).toBe(1);
          expect(result.disagreement).toBe(0);
          expect(result.perChunk.every((tv) => tv === 0)).toBe(true);
        }),
      );
    });

    it("identical distributions agree to within float error through aggregate()", () => {
      fc.assert(
        fc.property(caseArb, ({ labels, probs, weights, method }) => {
          const same = probs.map(() => dist(labels, probs[0]!));
          const agg = aggregate(same, weights, method).distribution;
          expect(computeAgreement(same, agg, weights).agreement).toBeCloseTo(1, 12);
        }),
      );
    });
  });
});
