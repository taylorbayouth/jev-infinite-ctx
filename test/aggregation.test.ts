import { describe, it, expect } from "vitest";
import fc from "fast-check";
import { aggregate } from "../src/aggregation.js";
import { JevInfiniteCTXError } from "../src/errors.js";
import type { AggregationMethod, Distribution } from "../src/types.js";

const METHODS: readonly AggregationMethod[] = ["weighted_mean", "mean", "median", "min", "max"];
const AB = ["a", "b"] as const;
const ABC = ["a", "b", "c"] as const;

const dist = (labels: readonly string[], probs: number[]): Distribution => ({ labels, probs });
const sum = (xs: readonly number[]) => xs.reduce((a, b) => a + b, 0);

function expectProbs(actual: readonly number[], expected: readonly number[]): void {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((p, j) => expect(actual[j]).toBeCloseTo(p, 12));
}

// ---------------------------------------------------------------------------
// Hand-computed cases (spec 9.2)
// ---------------------------------------------------------------------------

describe("aggregate: hand-computed values", () => {
  const p1 = dist(AB, [0.8, 0.2]);
  const p2 = dist(AB, [0.4, 0.6]);

  it("weighted_mean: Σ w_i · P_i[j]", () => {
    // 0.75·0.8 + 0.25·0.4 = 0.7; 0.75·0.2 + 0.25·0.6 = 0.3
    const { distribution, degenerate } = aggregate([p1, p2], [0.75, 0.25], "weighted_mean");
    expect(distribution.labels).toEqual(["a", "b"]);
    expectProbs(distribution.probs, [0.7, 0.3]);
    expect(degenerate).toBe(false);
  });

  it("weighted_mean renormalizes raw weights (e.g. unique-token counts)", () => {
    const { distribution } = aggregate([p1, p2], [3000, 1000], "weighted_mean");
    expectProbs(distribution.probs, [0.7, 0.3]);
  });

  it("weighted_mean ignores a zero-weight chunk", () => {
    const { distribution } = aggregate([p1, p2], [1, 0], "weighted_mean");
    expectProbs(distribution.probs, [0.8, 0.2]);
  });

  it("weighted_mean does not double-count overlap: three chunks with 5% overlap", () => {
    // Unique tokens 1000, 950, 950 (spec 6.4) → weights 1000/2900, 950/2900, 950/2900.
    const chunks = [dist(AB, [1, 0]), dist(AB, [0, 1]), dist(AB, [0, 1])];
    const { distribution } = aggregate(chunks, [1000, 950, 950], "weighted_mean");
    expectProbs(distribution.probs, [1000 / 2900, 1900 / 2900]);
  });

  it("mean: arithmetic mean, ignoring weights", () => {
    const { distribution } = aggregate([p1, p2], [100, 1], "mean");
    expectProbs(distribution.probs, [0.6, 0.4]);
  });

  it("median: odd count takes the middle value, then renormalizes", () => {
    const chunks = [
      dist(ABC, [0.7, 0.2, 0.1]),
      dist(ABC, [0.5, 0.3, 0.2]),
      dist(ABC, [0.1, 0.1, 0.8]),
    ];
    // Per-dimension medians [0.5, 0.2, 0.2], sum 0.9.
    const { distribution, degenerate } = aggregate(chunks, [1, 1, 1], "median");
    expectProbs(distribution.probs, [5 / 9, 2 / 9, 2 / 9]);
    expect(degenerate).toBe(false);
  });

  it("median: even count averages the two middle values", () => {
    const chunks = [
      dist(AB, [0.8, 0.2]),
      dist(AB, [0.6, 0.4]),
      dist(AB, [0.2, 0.8]),
      dist(AB, [0.4, 0.6]),
    ];
    // Sorted dim a: 0.2 0.4 0.6 0.8 → 0.5; dim b likewise → 0.5.
    const { distribution } = aggregate(chunks, [1, 1, 1, 1], "median");
    expectProbs(distribution.probs, [0.5, 0.5]);
  });

  it("median: even count with unequal middles", () => {
    const chunks = [dist(AB, [0.9, 0.1]), dist(AB, [0.3, 0.7])];
    // medians [0.6, 0.4]
    const { distribution } = aggregate(chunks, [1, 1], "median");
    expectProbs(distribution.probs, [0.6, 0.4]);
  });

  it("min: per-dimension minimum, renormalized", () => {
    const chunks = [dist(AB, [0.7, 0.3]), dist(AB, [0.4, 0.6])];
    // [0.4, 0.3] / 0.7
    const { distribution } = aggregate(chunks, [1, 1], "min");
    expectProbs(distribution.probs, [4 / 7, 3 / 7]);
  });

  it("max: per-dimension maximum, renormalized", () => {
    const chunks = [dist(AB, [0.7, 0.3]), dist(AB, [0.4, 0.6])];
    // [0.7, 0.6] / 1.3
    const { distribution } = aggregate(chunks, [1, 1], "max");
    expectProbs(distribution.probs, [7 / 13, 6 / 13]);
  });

  it("score-style distributions aggregate per level", () => {
    const levels = ["0", "1", "2", "3"];
    const chunks = [dist(levels, [0.05, 0.2, 0.7, 0.05]), dist(levels, [0.25, 0.4, 0.3, 0.05])];
    const { distribution } = aggregate(chunks, [0.5, 0.5], "weighted_mean");
    expectProbs(distribution.probs, [0.15, 0.3, 0.5, 0.05]);
  });

  it.each(METHODS)("%s over a single distribution returns it", (method) => {
    const { distribution, degenerate } = aggregate([dist(ABC, [0.2, 0.3, 0.5])], [1], method);
    expectProbs(distribution.probs, [0.2, 0.3, 0.5]);
    expect(degenerate).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Degenerate fallback
// ---------------------------------------------------------------------------

describe("aggregate: degenerate fallback", () => {
  it("min over disjoint one-hot chunks falls back to uniform", () => {
    const { distribution, degenerate } = aggregate(
      [dist(AB, [1, 0]), dist(AB, [0, 1])],
      [0.5, 0.5],
      "min",
    );
    expect(degenerate).toBe(true);
    expect(distribution.probs).toEqual([0.5, 0.5]);
    expect(distribution.labels).toEqual(["a", "b"]);
  });

  it("median of three disjoint one-hot chunks falls back to uniform", () => {
    const chunks = [dist(ABC, [1, 0, 0]), dist(ABC, [0, 1, 0]), dist(ABC, [0, 0, 1])];
    const { distribution, degenerate } = aggregate(chunks, [1, 1, 1], "median");
    expect(degenerate).toBe(true);
    expect(distribution.probs).toEqual([1 / 3, 1 / 3, 1 / 3]);
  });

  it("a non-finite reduced sum falls back to uniform", () => {
    // Unnormalized input whose per-dimension max overflows when summed.
    const big = dist(AB, [Number.MAX_VALUE, Number.MAX_VALUE]);
    const { distribution, degenerate } = aggregate([big], [1], "max");
    expect(degenerate).toBe(true);
    expect(distribution.probs).toEqual([0.5, 0.5]);
  });
});

// ---------------------------------------------------------------------------
// Validation and purity
// ---------------------------------------------------------------------------

describe("aggregate: validation", () => {
  const ok = dist(AB, [0.5, 0.5]);

  it("requires at least one distribution", () => {
    expect(() => aggregate([], [], "mean")).toThrow(JevInfiniteCTXError);
  });

  it("requires at least one label", () => {
    expect(() => aggregate([dist([], [])], [1], "mean")).toThrow(/at least one label/);
  });

  it("requires identical label arrays", () => {
    expect(() => aggregate([ok, dist(["b", "a"], [0.5, 0.5])], [1, 1], "mean")).toThrow(
      /different labels/,
    );
    expect(() => aggregate([ok, dist(ABC, [0.2, 0.3, 0.5])], [1, 1], "mean")).toThrow(
      /different labels/,
    );
  });

  it("requires one probability per label", () => {
    expect(() => aggregate([ok, dist(AB, [1])], [1, 1], "mean")).toThrow(JevInfiniteCTXError);
    // Sparse arrays have the right length but a hole.
    const sparse = dist(AB, new Array<number>(2));
    expect(() => aggregate([sparse], [1], "mean")).toThrow(JevInfiniteCTXError);
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, -0.1])("rejects probability %s", (p) => {
    expect(() => aggregate([ok, dist(AB, [p, 0.5])], [1, 1], "max")).toThrow(JevInfiniteCTXError);
  });

  it("requires one weight per distribution", () => {
    expect(() => aggregate([ok, ok], [1], "weighted_mean")).toThrow(/Expected 2 weights/);
    // Weights are validated for every method, not only weighted_mean.
    expect(() => aggregate([ok, ok], [1, 1, 1], "median")).toThrow(JevInfiniteCTXError);
  });

  it.each([
    [[1, -1]],
    [[1, Number.NaN]],
    [[0, 0]],
  ])("rejects weights %j", (weights) => {
    expect(() => aggregate([ok, ok], weights, "weighted_mean")).toThrow(JevInfiniteCTXError);
  });

  it("rejects an unknown method", () => {
    expect(() => aggregate([ok], [1], "mode" as AggregationMethod)).toThrow(
      /Unknown aggregation method/,
    );
  });

  it.each(METHODS)("%s never mutates its inputs", (method) => {
    const labels = Object.freeze(["a", "b", "c"]);
    const distributions = Object.freeze([
      Object.freeze({ labels, probs: Object.freeze([0.9, 0.05, 0.05]) as number[] }),
      Object.freeze({ labels, probs: Object.freeze([0.1, 0.3, 0.6]) as number[] }),
      Object.freeze({ labels, probs: Object.freeze([0.3, 0.3, 0.4]) as number[] }),
    ]) as Distribution[];
    const weights = Object.freeze([0.5, 0.25, 0.25]) as number[];

    const { distribution } = aggregate(distributions, weights, method);
    expect(distribution.labels).toEqual(labels);
    expect(distribution.labels).not.toBe(labels);
    expect(distributions[0]!.probs).toEqual([0.9, 0.05, 0.05]);
  });
});

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

/** A probability vector of length k: normalized integers, or one-hot (exercises degenerate cases). */
function distributionProbsArb(k: number): fc.Arbitrary<number[]> {
  const normalized = fc
    .array(fc.nat({ max: 1000 }), { minLength: k, maxLength: k })
    .filter((xs) => sum(xs) > 0)
    .map((xs) => {
      const s = sum(xs);
      return xs.map((x) => x / s);
    });
  const oneHot = fc.nat({ max: k - 1 }).map((hot) => Array.from({ length: k }, (_, j) => (j === hot ? 1 : 0)));
  return fc.oneof(normalized, oneHot);
}

const caseArb = fc
  .record({ k: fc.integer({ min: 2, max: 7 }), n: fc.integer({ min: 1, max: 9 }) })
  .chain(({ k, n }) =>
    fc.record({
      labels: fc.constant(Array.from({ length: k }, (_, j) => `L${j}`)),
      probs: fc.array(distributionProbsArb(k), { minLength: n, maxLength: n }),
      weights: fc.array(fc.double({ min: 1e-3, max: 1e3, noNaN: true }), {
        minLength: n,
        maxLength: n,
      }),
      method: fc.constantFrom(...METHODS),
    }),
  );

describe("aggregate: properties", () => {
  it("returns a valid distribution over the same labels for every method", () => {
    fc.assert(
      fc.property(caseArb, ({ labels, probs, weights, method }) => {
        const distributions = probs.map((p) => dist(labels, p));
        const { distribution, degenerate } = aggregate(distributions, weights, method);
        expect(distribution.labels).toEqual(labels);
        expect(distribution.probs).toHaveLength(labels.length);
        for (const p of distribution.probs) {
          expect(Number.isFinite(p)).toBe(true);
          expect(p).toBeGreaterThanOrEqual(0);
        }
        expect(Math.abs(sum(distribution.probs) - 1)).toBeLessThan(1e-9);
        if (degenerate) {
          expect(distribution.probs.every((p) => p === 1 / labels.length)).toBe(true);
        }
        // Mean-type and max reductions keep all the mass, so they are never degenerate.
        if (method === "weighted_mean" || method === "mean" || method === "max") {
          expect(degenerate).toBe(false);
        }
      }),
    );
  });

  it("is idempotent on identical distributions for every method", () => {
    fc.assert(
      fc.property(caseArb, ({ labels, probs, weights, method }) => {
        const same = probs.map(() => dist(labels, probs[0]!));
        const { distribution, degenerate } = aggregate(same, weights, method);
        expect(degenerate).toBe(false);
        distribution.probs.forEach((p, j) => expect(p).toBeCloseTo(probs[0]![j]!, 12));
      }),
    );
  });

  it("weighted_mean is invariant to scaling the weights", () => {
    fc.assert(
      fc.property(caseArb, fc.double({ min: 1e-3, max: 1e3, noNaN: true }), (c, factor) => {
        const distributions = c.probs.map((p) => dist(c.labels, p));
        const a = aggregate(distributions, c.weights, "weighted_mean").distribution.probs;
        const b = aggregate(
          distributions,
          c.weights.map((w) => w * factor),
          "weighted_mean",
        ).distribution.probs;
        a.forEach((p, j) => expect(p).toBeCloseTo(b[j]!, 12));
      }),
    );
  });

  it("unweighted methods are invariant to chunk order", () => {
    fc.assert(
      fc.property(caseArb, ({ labels, probs, weights, method }) => {
        const distributions = probs.map((p) => dist(labels, p));
        const forward = aggregate(distributions, weights, method);
        const reversed = aggregate([...distributions].reverse(), [...weights].reverse(), method);
        expect(reversed.degenerate).toBe(forward.degenerate);
        forward.distribution.probs.forEach((p, j) =>
          expect(p).toBeCloseTo(reversed.distribution.probs[j]!, 12),
        );
      }),
    );
  });

  it("weighted_mean stays within the per-dimension min and max of the chunks", () => {
    fc.assert(
      fc.property(caseArb, ({ labels, probs, weights }) => {
        const distributions = probs.map((p) => dist(labels, p));
        const agg = aggregate(distributions, weights, "weighted_mean").distribution.probs;
        agg.forEach((p, j) => {
          const column = probs.map((row) => row[j]!);
          expect(p).toBeGreaterThanOrEqual(Math.min(...column) - 1e-12);
          expect(p).toBeLessThanOrEqual(Math.max(...column) + 1e-12);
        });
      }),
    );
  });
});
