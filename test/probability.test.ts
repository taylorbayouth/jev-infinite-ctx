import { describe, it, expect } from "vitest";
import fc from "fast-check";
import {
  clamp01,
  distributionToRecord,
  extractConfidence,
  labelsFor,
  normalizeWeights,
  sameLabels,
  toDistribution,
  weightedSum,
} from "../src/probability.js";
import { JevInfiniteCTXError, JevResponseError } from "../src/errors.js";
import type {
  ChoiceQuestion,
  NativeJevAnswer,
  NoulQuestion,
  ScoreQuestion,
} from "../src/types.js";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function choiceQuestion(keys: readonly string[]): ChoiceQuestion {
  return {
    type: "choice",
    instructions: "Which option fits best?",
    criteria: Object.fromEntries(keys.map((k) => [k, `description of ${k}`])),
  };
}

function scoreQuestion(levels: number): ScoreQuestion {
  return {
    type: "score",
    instructions: "Rate it.",
    criteria: Array.from({ length: levels }, (_, i) => `level ${i}`),
  };
}

const NOUL: NoulQuestion = { type: "noul", instructions: "Is it so?" };
const TREE_ROCK = choiceQuestion(["tree", "rock", "other"]);

/** Builds an answer from untyped data, as a misbehaving custom transport might. */
function rawAnswer(value: unknown): NativeJevAnswer {
  return value as NativeJevAnswer;
}

function choiceAnswer(choice: string, probabilities?: Record<string, number>): NativeJevAnswer {
  return probabilities === undefined
    ? { type: "choice", choice }
    : { type: "choice", choice, probabilities };
}

/** Keys that break naive `obj[key] = v` / `key in obj` code, plus unicode. */
const SPECIAL_KEYS = [
  "__proto__",
  "constructor",
  "",
  "0",
  "toString",
  "hasOwnProperty",
  "日本語",
  "🌲 tree",
  "é",
];

const sum = (xs: readonly number[]) => xs.reduce((a, b) => a + b, 0);

// ---------------------------------------------------------------------------
// labelsFor
// ---------------------------------------------------------------------------

describe("labelsFor", () => {
  it("uses the criteria key order for choice", () => {
    expect(labelsFor(TREE_ROCK)).toEqual(["tree", "rock", "other"]);
  });

  it("uses level indices for score", () => {
    expect(labelsFor(scoreQuestion(4))).toEqual(["0", "1", "2", "3"]);
    expect(labelsFor(scoreQuestion(2))).toEqual(["0", "1"]);
    expect(labelsFor(scoreQuestion(10))).toHaveLength(10);
  });

  it("uses [no, yes] for noul, with or without criteria", () => {
    expect(labelsFor(NOUL)).toEqual(["no", "yes"]);
    expect(
      labelsFor({ type: "noul", instructions: "x", criteria: { true: "t", false: "f" } }),
    ).toEqual(["no", "yes"]);
  });

  it("returns a fresh array each call", () => {
    const a = labelsFor(NOUL);
    a.push("mutated");
    expect(labelsFor(NOUL)).toEqual(["no", "yes"]);
  });

  it("preserves special and unicode choice keys exactly", () => {
    const q = choiceQuestion(SPECIAL_KEYS);
    const labels = labelsFor(q);
    expect(new Set(labels)).toEqual(new Set(SPECIAL_KEYS));
    expect(labels).toEqual(Object.keys(q.criteria));
    // JS enumerates integer-like keys first; labels follow the object's own order.
    expect(labels[0]).toBe("0");
  });

  it("rejects an unknown question type", () => {
    expect(() => labelsFor({ type: "rank" } as unknown as ChoiceQuestion)).toThrow(
      JevInfiniteCTXError,
    );
  });
});

// ---------------------------------------------------------------------------
// toDistribution: choice
// ---------------------------------------------------------------------------

describe("toDistribution (choice)", () => {
  it("converts the probability map in label order (spec 8 example)", () => {
    const d = toDistribution(
      TREE_ROCK,
      choiceAnswer("tree", { other: 0.05, tree: 0.8, rock: 0.15 }),
    );
    expect(d.labels).toEqual(["tree", "rock", "other"]);
    expect(d.probs[0]).toBeCloseTo(0.8, 12);
    expect(d.probs[1]).toBeCloseTo(0.15, 12);
    expect(d.probs[2]).toBeCloseTo(0.05, 12);
    expect(sum(d.probs)).toBeCloseTo(1, 12);
  });

  it("treats missing labels as 0", () => {
    const d = toDistribution(TREE_ROCK, choiceAnswer("rock", { rock: 1 }));
    expect(d.probs).toEqual([0, 1, 0]);
  });

  it("renormalizes sums within tolerance", () => {
    const high = toDistribution(TREE_ROCK, choiceAnswer("tree", { tree: 0.63, rock: 0.42 }));
    expect(high.probs[0]).toBeCloseTo(0.6, 12);
    expect(high.probs[1]).toBeCloseTo(0.4, 12);
    const low = toDistribution(TREE_ROCK, choiceAnswer("tree", { tree: 0.57, rock: 0.38 }));
    expect(low.probs[0]).toBeCloseTo(0.6, 12);
    expect(low.probs[1]).toBeCloseTo(0.4, 12);
  });

  it("accepts sums exactly at the tolerance boundary", () => {
    expect(() => toDistribution(TREE_ROCK, choiceAnswer("tree", { tree: 0.5, rock: 0.5, other: 0.1 }))).not.toThrow();
    expect(() => toDistribution(TREE_ROCK, choiceAnswer("tree", { tree: 0.5, rock: 0.4 }))).not.toThrow();
  });

  it("rejects sums outside 1 ± 0.1", () => {
    expect(() => toDistribution(TREE_ROCK, choiceAnswer("tree", { tree: 0.5, rock: 0.3 }))).toThrow(
      JevResponseError,
    );
    expect(() =>
      toDistribution(TREE_ROCK, choiceAnswer("tree", { tree: 0.7, rock: 0.3, other: 0.2 })),
    ).toThrow(/sum/);
  });

  it("rejects a zero-sum map", () => {
    expect(() =>
      toDistribution(TREE_ROCK, choiceAnswer("tree", { tree: 0, rock: 0, other: 0 })),
    ).toThrow(/sum to 0/);
    expect(() => toDistribution(TREE_ROCK, choiceAnswer("tree", {}))).toThrow(JevResponseError);
  });

  it("clamps tiny negative probabilities to 0", () => {
    const d = toDistribution(TREE_ROCK, choiceAnswer("tree", { tree: 1, rock: -1e-10 }));
    expect(d.probs).toEqual([1, 0, 0]);
    expect(Object.is(d.probs[1], 0)).toBe(true);
  });

  it("rejects material negative probabilities", () => {
    expect(() =>
      toDistribution(TREE_ROCK, choiceAnswer("tree", { tree: 1.01, rock: -0.01 })),
    ).toThrow(/out of range/);
  });

  it("rejects a single probability above 1 + tolerance", () => {
    expect(() => toDistribution(TREE_ROCK, choiceAnswer("tree", { tree: 1.2 }))).toThrow(
      /out of range/,
    );
  });

  it.each([
    ["NaN", Number.NaN],
    ["Infinity", Number.POSITIVE_INFINITY],
    ["string", "0.5"],
    ["null", null],
  ])("rejects a %s probability value", (_, value) => {
    const probabilities = { tree: 0.5, rock: value } as unknown as Record<string, number>;
    expect(() => toDistribution(TREE_ROCK, choiceAnswer("tree", probabilities))).toThrow(
      JevResponseError,
    );
  });

  it("requires probabilities", () => {
    expect(() => toDistribution(TREE_ROCK, choiceAnswer("tree"))).toThrow(/missing probabilities/);
    expect(() =>
      toDistribution(TREE_ROCK, rawAnswer({ type: "choice", choice: "tree", probabilities: null })),
    ).toThrow(/missing probabilities/);
  });

  it("rejects non-object probabilities", () => {
    for (const probabilities of [[0.5, 0.5], 1, "tree"]) {
      expect(() =>
        toDistribution(TREE_ROCK, rawAnswer({ type: "choice", choice: "tree", probabilities })),
      ).toThrow(JevResponseError);
    }
  });

  it("rejects unknown probability keys, including prototype names, naming them by length only", () => {
    // Jev may copy an unknown key from the chunk text, so it is never quoted.
    expect(() =>
      toDistribution(TREE_ROCK, choiceAnswer("tree", { tree: 0.9, bush: 0.1 })),
    ).toThrow("Jev choice probabilities contain an unknown label, a string of length 4.");
    const protoKey = Object.fromEntries([
      ["tree", 0.9],
      ["__proto__", 0.1],
    ]);
    expect(() => toDistribution(TREE_ROCK, choiceAnswer("tree", protoKey))).toThrow(
      /unknown label, a string of length 9\./,
    );
    expect(() =>
      toDistribution(TREE_ROCK, choiceAnswer("tree", { tree: 0.9, toString: 0.1 })),
    ).toThrow(/unknown label, a string of length 8\./);
  });

  it("rejects a choice that is not a criteria key", () => {
    expect(() => toDistribution(TREE_ROCK, choiceAnswer("bush", { tree: 1 }))).toThrow(
      /not one of the question's criteria keys/,
    );
    // Prototype names must not pass a membership check.
    expect(() => toDistribution(TREE_ROCK, choiceAnswer("constructor", { tree: 1 }))).toThrow(
      JevResponseError,
    );
    expect(() => toDistribution(TREE_ROCK, choiceAnswer("__proto__", { tree: 1 }))).toThrow(
      JevResponseError,
    );
    expect(() =>
      toDistribution(TREE_ROCK, rawAnswer({ type: "choice", choice: 0, probabilities: { tree: 1 } })),
    ).toThrow(JevResponseError);
  });

  it("rejects an answer of the wrong type", () => {
    expect(() => toDistribution(TREE_ROCK, { type: "noul", noul: 0.5 })).toThrow(
      /answer type "noul" for a "choice" question/,
    );
    expect(() => toDistribution(TREE_ROCK, rawAnswer({ type: "rank", choice: "tree" }))).toThrow(
      JevResponseError,
    );
    expect(() => toDistribution(TREE_ROCK, rawAnswer({ choice: "tree" }))).toThrow(
      JevResponseError,
    );
  });

  it.each([
    ["null", null],
    ["undefined", undefined],
    ["an array", []],
    ["a string", "tree"],
    ["a number", 1],
  ])("rejects an answer that is %s", (_, value) => {
    expect(() => toDistribution(TREE_ROCK, rawAnswer(value))).toThrow(JevResponseError);
  });

  it("does not echo string answers or long labels into error messages", () => {
    const secret = "SECRET STATE TEXT ".repeat(20);
    try {
      toDistribution(TREE_ROCK, rawAnswer(secret));
      expect.unreachable();
    } catch (error) {
      expect((error as Error).message).not.toContain("SECRET");
    }

    const longKey = "k".repeat(10_000);
    try {
      toDistribution(TREE_ROCK, choiceAnswer("tree", Object.fromEntries([[longKey, 1]])));
      expect.unreachable();
    } catch (error) {
      expect(error).toBeInstanceOf(JevResponseError);
      expect((error as Error).message.length).toBeLessThan(200);
    }
  });

  it("round-trips special and unicode keys through labels, distribution, and record", () => {
    const q = choiceQuestion(SPECIAL_KEYS);
    const labels = labelsFor(q);
    const raw = labels.map((_, j) => j + 1);
    const total = sum(raw);
    const probabilities = Object.fromEntries(labels.map((l, j) => [l, raw[j]! / total]));

    const d = toDistribution(q, choiceAnswer("__proto__", probabilities));
    expect(d.labels).toEqual(labels);
    d.probs.forEach((p, j) => expect(p).toBeCloseTo(raw[j]! / total, 12));

    const record = distributionToRecord(d);
    expect(Object.getPrototypeOf(record)).toBe(Object.prototype);
    expect(Object.keys(record)).toEqual(labels);
    for (const [j, label] of labels.entries()) {
      expect(Object.hasOwn(record, label)).toBe(true);
      expect(record[label]).toBe(d.probs[j]);
    }
    const reparsed: unknown = JSON.parse(JSON.stringify(record));
    expect(Object.entries(reparsed as object)).toEqual(Object.entries(record));
  });

  it("round-trips arbitrary unique keys (property)", () => {
    const keyArb = fc.oneof(
      fc.constantFrom(...SPECIAL_KEYS),
      fc.string({ unit: "grapheme", maxLength: 6 }),
      fc.string({ unit: "binary", maxLength: 4 }),
    );
    const caseArb = fc
      .uniqueArray(keyArb, { minLength: 2, maxLength: 8 })
      .chain((keys) =>
        fc.record({
          keys: fc.constant(keys),
          raw: fc
            .array(fc.nat({ max: 1000 }), { minLength: keys.length, maxLength: keys.length })
            .filter((xs) => sum(xs) > 0),
          chosen: fc.nat({ max: keys.length - 1 }),
        }),
      );

    fc.assert(
      fc.property(caseArb, ({ keys, raw, chosen }) => {
        const q = choiceQuestion(keys);
        const labels = labelsFor(q);
        expect(new Set(labels)).toEqual(new Set(keys));

        const rawByKey = new Map(keys.map((k, i) => [k, raw[i]!]));
        const total = sum(raw);
        const probabilities = Object.fromEntries(keys.map((k, i) => [k, raw[i]! / total]));
        const d = toDistribution(q, choiceAnswer(keys[chosen]!, probabilities));

        expect(d.labels).toEqual(labels);
        expect(sum(d.probs)).toBeCloseTo(1, 12);
        labels.forEach((label, j) => {
          expect(d.probs[j]).toBeCloseTo(rawByKey.get(label)! / total, 12);
        });

        const record = distributionToRecord(d);
        expect(Object.getPrototypeOf(record)).toBe(Object.prototype);
        expect(Object.keys(record)).toEqual(labels);
        labels.forEach((label, j) => {
          expect(Object.hasOwn(record, label)).toBe(true);
          expect(record[label]).toBe(d.probs[j]);
        });
        const reparsed = JSON.parse(JSON.stringify(record)) as object;
        expect(Object.entries(reparsed)).toEqual(Object.entries(record));
      }),
    );
  });

  it("yields a valid distribution for any map within tolerance, and rejects beyond it (property)", () => {
    const caseArb = fc.integer({ min: 2, max: 10 }).chain((k) =>
      fc.record({
        k: fc.constant(k),
        raw: fc
          .array(fc.nat({ max: 1000 }), { minLength: k, maxLength: k })
          .filter((xs) => sum(xs) > 0),
        scale: fc.double({ min: 0.905, max: 1.095, noNaN: true }),
        badScale: fc.oneof(
          fc.double({ min: 0, max: 0.895, noNaN: true }),
          fc.double({ min: 1.105, max: 5, noNaN: true }),
        ),
      }),
    );
    fc.assert(
      fc.property(caseArb, ({ k, raw, scale, badScale }) => {
        const q = choiceQuestion(Array.from({ length: k }, (_, i) => `opt${i}`));
        const total = sum(raw);
        const build = (factor: number) =>
          Object.fromEntries(raw.map((x, i) => [`opt${i}`, (x / total) * factor]));

        const d = toDistribution(q, choiceAnswer("opt0", build(scale)));
        expect(d.probs).toHaveLength(k);
        expect(d.probs.every((p) => Number.isFinite(p) && p >= 0)).toBe(true);
        expect(Math.abs(sum(d.probs) - 1)).toBeLessThan(1e-12);
        d.probs.forEach((p, i) => expect(p).toBeCloseTo(raw[i]! / total, 9));

        expect(() => toDistribution(q, choiceAnswer("opt0", build(badScale)))).toThrow(
          JevResponseError,
        );
      }),
    );
  });
});

// ---------------------------------------------------------------------------
// toDistribution: score
// ---------------------------------------------------------------------------

describe("toDistribution (score)", () => {
  const q = scoreQuestion(4);

  it("converts level probabilities (spec 8 example)", () => {
    const d = toDistribution(q, {
      type: "score",
      score: 1.75,
      probabilities: { "0": 0.05, "1": 0.2, "2": 0.7, "3": 0.05 },
      confidence: 0.9,
    });
    expect(d.labels).toEqual(["0", "1", "2", "3"]);
    [0.05, 0.2, 0.7, 0.05].forEach((p, j) => expect(d.probs[j]).toBeCloseTo(p, 12));
  });

  it("accepts scores at and marginally beyond the ends of the range", () => {
    const probabilities = { "0": 1 };
    expect(() => toDistribution(q, { type: "score", score: 0, probabilities })).not.toThrow();
    expect(() => toDistribution(q, { type: "score", score: 3, probabilities })).not.toThrow();
    expect(() => toDistribution(q, { type: "score", score: 3 + 1e-7, probabilities })).not.toThrow();
    expect(() => toDistribution(q, { type: "score", score: -1e-7, probabilities })).not.toThrow();
  });

  it.each([-0.1, 3.1, 4, Number.NaN, Number.POSITIVE_INFINITY])(
    "rejects score %s for a 4-level rubric",
    (score) => {
      expect(() => toDistribution(q, { type: "score", score, probabilities: { "0": 1 } })).toThrow(
        JevResponseError,
      );
    },
  );

  it("rejects a non-numeric score", () => {
    expect(() =>
      toDistribution(q, rawAnswer({ type: "score", score: "2", probabilities: { "2": 1 } })),
    ).toThrow(/finite number/);
  });

  it("rejects level keys outside the rubric", () => {
    expect(() =>
      toDistribution(q, { type: "score", score: 1, probabilities: { "0": 0.5, "4": 0.5 } }),
    ).toThrow(/unknown label, a string of length 1\./);
    expect(() =>
      toDistribution(q, { type: "score", score: 1, probabilities: { "01": 1 } }),
    ).toThrow(JevResponseError);
  });

  it("requires probabilities", () => {
    expect(() => toDistribution(q, { type: "score", score: 1 })).toThrow(/missing probabilities/);
  });

  it("rejects a choice answer for a score question", () => {
    expect(() => toDistribution(q, choiceAnswer("0", { "0": 1 }))).toThrow(JevResponseError);
  });
});

// ---------------------------------------------------------------------------
// toDistribution: noul
// ---------------------------------------------------------------------------

describe("toDistribution (noul)", () => {
  it("maps p to [1 - p, p] (spec 8 example)", () => {
    const d = toDistribution(NOUL, { type: "noul", noul: 0.82 });
    expect(d.labels).toEqual(["no", "yes"]);
    expect(d.probs[0]).toBeCloseTo(0.18, 12);
    expect(d.probs[1]).toBe(0.82);
  });

  it("handles the endpoints exactly", () => {
    expect(toDistribution(NOUL, { type: "noul", noul: 0 }).probs).toEqual([1, 0]);
    expect(toDistribution(NOUL, { type: "noul", noul: 1 }).probs).toEqual([0, 1]);
  });

  it("clamps values marginally outside [0, 1]", () => {
    expect(toDistribution(NOUL, { type: "noul", noul: 1 + 1e-7 }).probs).toEqual([0, 1]);
    expect(toDistribution(NOUL, { type: "noul", noul: -1e-7 }).probs).toEqual([1, 0]);
  });

  it.each([1.01, -0.01, Number.NaN, Number.NEGATIVE_INFINITY])("rejects noul %s", (noul) => {
    expect(() => toDistribution(NOUL, { type: "noul", noul })).toThrow(JevResponseError);
  });

  it("rejects a missing or non-numeric noul", () => {
    expect(() => toDistribution(NOUL, rawAnswer({ type: "noul" }))).toThrow(JevResponseError);
    expect(() => toDistribution(NOUL, rawAnswer({ type: "noul", noul: "0.5" }))).toThrow(
      JevResponseError,
    );
  });

  it("rejects a choice answer for a noul question", () => {
    expect(() => toDistribution(NOUL, choiceAnswer("yes", { yes: 1 }))).toThrow(JevResponseError);
  });

  it("always sums to 1 within float error (property)", () => {
    fc.assert(
      fc.property(fc.double({ min: 0, max: 1, noNaN: true }), (p) => {
        const d = toDistribution(NOUL, { type: "noul", noul: p });
        expect(d.probs[1]).toBe(p);
        expect(d.probs[0]).toBeGreaterThanOrEqual(0);
        expect(Math.abs(sum(d.probs) - 1)).toBeLessThanOrEqual(Number.EPSILON);
      }),
    );
  });
});

// ---------------------------------------------------------------------------
// distributionToRecord
// ---------------------------------------------------------------------------

describe("distributionToRecord", () => {
  it("keys probabilities by label", () => {
    expect(distributionToRecord({ labels: ["no", "yes"], probs: [0.25, 0.75] })).toEqual({
      no: 0.25,
      yes: 0.75,
    });
  });

  it("creates an own __proto__ property instead of setting the prototype", () => {
    const record = distributionToRecord({ labels: ["__proto__", "a"], probs: [0.4, 0.6] });
    expect(Object.getPrototypeOf(record)).toBe(Object.prototype);
    expect(Object.hasOwn(record, "__proto__")).toBe(true);
    expect(record["__proto__"]).toBe(0.4);
    expect(JSON.stringify(record)).toBe('{"__proto__":0.4,"a":0.6}');
  });

  it("rejects mismatched lengths", () => {
    expect(() => distributionToRecord({ labels: ["a", "b"], probs: [1] })).toThrow(
      JevInfiniteCTXError,
    );
  });
});

// ---------------------------------------------------------------------------
// extractConfidence
// ---------------------------------------------------------------------------

describe("extractConfidence", () => {
  it("returns Jev's confidence for choice and score", () => {
    expect(extractConfidence({ type: "choice", choice: "a", confidence: 0.67 })).toBe(0.67);
    expect(extractConfidence({ type: "score", score: 1, confidence: 0.99 })).toBe(0.99);
    expect(extractConfidence({ type: "choice", choice: "a", confidence: 0 })).toBe(0);
    expect(extractConfidence({ type: "choice", choice: "a", confidence: 1 })).toBe(1);
  });

  it("returns undefined for noul, which has no native confidence", () => {
    expect(extractConfidence({ type: "noul", noul: 0.9 })).toBeUndefined();
    expect(extractConfidence(rawAnswer({ type: "noul", noul: 0.9, confidence: 0.5 }))).toBeUndefined();
  });

  it("returns undefined when confidence is omitted or null", () => {
    expect(extractConfidence({ type: "choice", choice: "a" })).toBeUndefined();
    expect(extractConfidence(rawAnswer({ type: "score", score: 1, confidence: null }))).toBeUndefined();
  });

  it("clamps values marginally outside [0, 1]", () => {
    expect(extractConfidence({ type: "choice", choice: "a", confidence: 1 + 1e-7 })).toBe(1);
    expect(extractConfidence({ type: "score", score: 0, confidence: -1e-7 })).toBe(0);
  });

  it.each([1.5, -0.1, Number.NaN, Number.POSITIVE_INFINITY, "0.5", true])(
    "rejects confidence %s",
    (confidence) => {
      expect(() =>
        extractConfidence(rawAnswer({ type: "choice", choice: "a", confidence })),
      ).toThrow(JevResponseError);
    },
  );

  it("rejects an unknown answer type", () => {
    expect(() => extractConfidence(rawAnswer({ type: "rank", confidence: 0.5 }))).toThrow(
      JevResponseError,
    );
  });

  it.each([null, undefined, [], "choice"])("rejects a non-object answer (%j)", (value) => {
    expect(() => extractConfidence(rawAnswer(value))).toThrow(JevResponseError);
  });
});

// ---------------------------------------------------------------------------
// Shared numeric helpers
// ---------------------------------------------------------------------------

describe("numeric helpers", () => {
  it("sameLabels compares contents and order", () => {
    expect(sameLabels(["a", "b"], ["a", "b"])).toBe(true);
    expect(sameLabels(["a", "b"], ["b", "a"])).toBe(false);
    expect(sameLabels(["a"], ["a", "b"])).toBe(false);
    expect(sameLabels([], [])).toBe(true);
  });

  it("normalizeWeights rescales to sum 1", () => {
    expect(normalizeWeights([3, 1], 2)).toEqual([0.75, 0.25]);
    expect(normalizeWeights([0, 2], 2)).toEqual([0, 1]);
    expect(normalizeWeights([0.25, 0.75], 2)).toEqual([0.25, 0.75]);
  });

  it("normalizeWeights does not mutate its input", () => {
    const weights = Object.freeze([2, 2]);
    expect(normalizeWeights(weights, 2)).toEqual([0.5, 0.5]);
  });

  it.each([
    ["wrong length", [1, 1], 3],
    ["negative", [1, -1], 2],
    ["NaN", [1, Number.NaN], 2],
    ["Infinity", [1, Number.POSITIVE_INFINITY], 2],
    ["all zero", [0, 0], 2],
    ["empty", [], 0],
    ["overflowing sum", [Number.MAX_VALUE, Number.MAX_VALUE], 2],
  ])("normalizeWeights rejects %s weights", (_, weights, n) => {
    expect(() => normalizeWeights(weights, n)).toThrow(JevInfiniteCTXError);
  });

  it("weightedSum computes a dot product", () => {
    expect(weightedSum([1, 2], [0.5, 0.5])).toBe(1.5);
    expect(weightedSum([], [])).toBe(0);
    expect(() => weightedSum([1], [0.5, 0.5])).toThrow(JevInfiniteCTXError);
  });

  it("clamp01 bounds values to [0, 1]", () => {
    expect(clamp01(-1)).toBe(0);
    expect(clamp01(2)).toBe(1);
    expect(clamp01(0.5)).toBe(0.5);
  });
});
