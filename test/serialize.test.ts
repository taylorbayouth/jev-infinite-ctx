import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { JevInfiniteCTXError } from "../src/errors.js";
import { serializeResult } from "../src/serialize.js";
import type {
  ChoiceResult,
  ChunkResult,
  JevInfiniteCTXResult,
  JsonObject,
  JsonValue,
  NoulResult,
  ScoreResult,
} from "../src/types.js";

const PROTO = "__proto__";

/** Probability map built without `obj[key] = v`, so "__proto__" stays an own key. */
function probs(entries: Array<[string, number]>): Record<string, number> {
  return Object.fromEntries(entries);
}

const baseMeta = {
  confidence: {
    base: 0.81,
    adjusted: 0.86,
    adjustment: 0.05,
    source: "jev" as const,
    agreement: 0.93,
    method: "c3-v1" as const,
    components: {
      effectiveChunkCount: 1.95,
      saturation: 0.21,
      gate: 0.74,
      cap: 0.98,
      lambda: 0.25,
      agreementFloor: 0.5,
      agreementExponent: 2,
    },
  },
  aggregation: { method: "weighted_mean" as const, degenerate: false },
  usage: {
    inputTokens: 74219,
    outputTokens: 102,
    costUsd: 0.0031,
    elapsedMs: 612,
    inputTokensEstimated: 70000,
    requests: 2,
    retries: 0,
    rechunks: 0,
  },
  model: "typesafe/jev-1.13",
  provider: "openrouter",
};

function chunkResult(index: number, overrides: Partial<ChunkResult> = {}): ChunkResult {
  return {
    index,
    start: index * 100,
    end: index * 100 + 120,
    estimatedTokens: 30,
    uniqueTokens: index === 0 ? 30 : 28,
    weight: index === 0 ? 30 / 58 : 28 / 58,
    boundary: index === 0 ? "paragraph" : "end",
    answer: {
      type: "choice",
      choice: PROTO,
      probabilities: probs([[PROTO, 0.7], ["constructor", 0.2], ["other", 0.1]]),
      confidence: 0.8,
    },
    probabilities: probs([[PROTO, 0.7], ["constructor", 0.2], ["other", 0.1]]),
    jevConfidence: 0.8,
    totalVariation: 0.05,
    usage: { inputTokens: 37000, outputTokens: 51, costUsd: 0.00155 },
    attempts: 1,
    elapsedMs: 300,
    model: "typesafe/jev-1.13",
    responseId: `resp-${index}`,
    ...overrides,
  };
}

/** A hand-built choice result whose option keys include "__proto__" and "constructor". */
function choiceResult(): ChoiceResult {
  return {
    type: "choice",
    choice: PROTO,
    probabilities: probs([[PROTO, 0.7], ["constructor", 0.2], ["other", 0.1]]),
    ...baseMeta,
    chunks: {
      count: 2,
      effectiveCount: 1.95,
      overlap: 0.05,
      overlapTokens: 1,
      stateTokenBudget: 30,
      results: [chunkResult(0), chunkResult(1)],
    },
  };
}

function scoreResult(): ScoreResult {
  return {
    type: "score",
    score: 1.6,
    normalizedScore: 0.8,
    probabilities: probs([["0", 0.1], ["1", 0.2], ["2", 0.7]]),
    legend: {
      "0": "Poor",
      "1": { what: "Fair", not_for: ["spam"], examples: [{ text: "ok", weight: 1 }] },
      "2": ["Good", "Excellent"],
    },
    ...baseMeta,
    chunks: {
      count: 1,
      effectiveCount: 1,
      overlap: 0.05,
      overlapTokens: 1,
      stateTokenBudget: 30,
      results: [
        chunkResult(0, {
          boundary: "end",
          answer: {
            type: "score",
            score: 1.6,
            probabilities: probs([["0", 0.1], ["1", 0.2], ["2", 0.7]]),
            legend: { "0": "Poor", "1": { what: "Fair" }, "2": ["Good"] },
            confidence: 0.77,
          },
          probabilities: probs([["0", 0.1], ["1", 0.2], ["2", 0.7]]),
          jevConfidence: 0.77,
          totalVariation: 0,
        }),
      ],
    },
  };
}

function noulResult(): NoulResult {
  return {
    type: "noul",
    noul: 0.84,
    ...baseMeta,
    confidence: { ...baseMeta.confidence, base: 0.68, adjusted: 0.76, adjustment: 0.08, source: "derived" },
    chunks: {
      count: 1,
      effectiveCount: 1,
      overlap: 0.05,
      overlapTokens: 1,
      stateTokenBudget: 30,
      results: [
        chunkResult(0, {
          answer: { type: "noul", noul: 0.84 },
          probabilities: probs([["no", 0.16], ["yes", 0.84]]),
          jevConfidence: undefined,
          usage: { inputTokens: 100, outputTokens: 5 }, // no cost reported
        }),
      ],
    },
  };
}

const asObject = (value: JsonValue | undefined): JsonObject => value as JsonObject;
const asArray = (value: JsonValue | undefined): JsonValue[] => value as JsonValue[];
const jsonRoundTrip = (value: unknown): unknown => JSON.parse(JSON.stringify(value));

describe("serializeResult: defaults", () => {
  it.each<[string, () => JevInfiniteCTXResult]>([
    ["choice", choiceResult],
    ["score", scoreResult],
    ["noul", noulResult],
  ])("includes chunk results and raw answers by default (%s)", (_type, make) => {
    const result = make();
    const out = serializeResult(result);
    expect(out).toEqual(result);
    const results = asArray(asObject(out["chunks"])["results"]);
    expect(results).toHaveLength(result.chunks.results.length);
    results.forEach((r, i) => expect(asObject(r)["answer"]).toEqual(result.chunks.results[i]?.answer));
  });

  it("returns a deep copy that shares no references and leaves the input untouched", () => {
    const result = choiceResult();
    const before = JSON.stringify(result);
    const out = serializeResult(result);

    expect(out).not.toBe(result);
    expect(out["probabilities"]).not.toBe(result.probabilities);
    expect(out["chunks"]).not.toBe(result.chunks);
    const outResults = asArray(asObject(out["chunks"])["results"]);
    expect(outResults[0]).not.toBe(result.chunks.results[0]);
    expect(asObject(outResults[0])["answer"]).not.toBe(result.chunks.results[0]?.answer);

    asObject(out["probabilities"])["other"] = 999;
    expect(JSON.stringify(result)).toBe(before);
  });

  it("keeps the score legend (structured criteria included)", () => {
    const out = serializeResult(scoreResult());
    expect(out["legend"]).toEqual(scoreResult().legend);
    expect(out["score"]).toBe(1.6);
    expect(out["normalizedScore"]).toBe(0.8);
  });
});

describe("serializeResult: options", () => {
  it("omits chunks.results but keeps the other chunk metadata when includeChunkResults is false", () => {
    const result = choiceResult();
    const out = serializeResult(result, { includeChunkResults: false });
    const chunks = asObject(out["chunks"]);
    expect(Object.hasOwn(chunks, "results")).toBe(false);
    expect(chunks).toEqual({
      count: 2,
      effectiveCount: 1.95,
      overlap: 0.05,
      overlapTokens: 1,
      stateTokenBudget: 30,
    });
    const { chunks: _chunks, ...rest } = result;
    expect(out).toMatchObject(jsonRoundTrip(rest) as JsonObject);
    // The runtime result is unaffected.
    expect(result.chunks.results).toHaveLength(2);
  });

  it("omits each chunk result's raw answer but keeps everything else when includeRawAnswers is false", () => {
    const result = choiceResult();
    const out = serializeResult(result, { includeRawAnswers: false });
    const results = asArray(asObject(out["chunks"])["results"]);
    expect(results).toHaveLength(2);
    results.forEach((r, i) => {
      const chunk = asObject(r);
      expect(Object.hasOwn(chunk, "answer")).toBe(false);
      const { answer: _answer, ...rest } = result.chunks.results[i] as ChunkResult;
      expect(chunk).toEqual(jsonRoundTrip(rest));
    });
    expect(result.chunks.results[0]?.answer).toBeDefined();
  });

  it("drops results entirely when both options are false", () => {
    const out = serializeResult(choiceResult(), { includeChunkResults: false, includeRawAnswers: false });
    expect(Object.hasOwn(asObject(out["chunks"]), "results")).toBe(false);
  });

  it("treats explicitly true options like the defaults", () => {
    const result = choiceResult();
    expect(serializeResult(result, { includeChunkResults: true, includeRawAnswers: true })).toEqual(
      serializeResult(result),
    );
  });

  it("preserves the original key order", () => {
    const result = choiceResult();
    const out = serializeResult(result, { includeRawAnswers: false });
    expect(Object.keys(out)).toEqual(Object.keys(result));
    expect(Object.keys(asObject(out["chunks"]))).toEqual(Object.keys(result.chunks));
  });
});

describe("serializeResult: JSON safety", () => {
  it.each([{}, { includeChunkResults: false }, { includeRawAnswers: false }])(
    "round-trips through JSON.stringify/parse unchanged (%j)",
    (options) => {
      for (const result of [choiceResult(), scoreResult(), noulResult()]) {
        const out = serializeResult(result, options);
        const text = JSON.stringify(out);
        expect(JSON.parse(text)).toEqual(out);
        expect(JSON.stringify(JSON.parse(text))).toBe(text);
      }
    },
  );

  it('preserves "__proto__" keys as own properties without touching any prototype', () => {
    const out = serializeResult(choiceResult());
    const checkMap = (value: JsonValue | undefined): void => {
      const map = asObject(value);
      expect(Object.hasOwn(map, PROTO)).toBe(true);
      expect(Object.keys(map)).toEqual([PROTO, "constructor", "other"]);
      expect(Object.getOwnPropertyDescriptor(map, PROTO)?.value).toBe(0.7);
      expect(Object.getPrototypeOf(map)).toBe(Object.prototype);
    };

    expect(out["choice"]).toBe(PROTO);
    checkMap(out["probabilities"]);
    for (const r of asArray(asObject(out["chunks"])["results"])) {
      checkMap(asObject(r)["probabilities"]);
      checkMap(asObject(asObject(r)["answer"])["probabilities"]);
    }
    expect(JSON.stringify(out)).toContain('"__proto__":0.7');
    expect(Object.hasOwn(Object.prototype, "other")).toBe(false);
    expect(({} as Record<string, unknown>)["other"]).toBeUndefined();
  });

  it("drops undefined optional fields (costUsd, jevConfidence, responseId)", () => {
    const result = noulResult();
    result.usage = { ...result.usage, costUsd: undefined };
    const out = serializeResult(result);
    expect(Object.hasOwn(asObject(out["usage"]), "costUsd")).toBe(false);
    const chunk = asObject(asArray(asObject(out["chunks"])["results"])[0]);
    expect(Object.hasOwn(chunk, "jevConfidence")).toBe(false);
    expect(Object.hasOwn(asObject(chunk["usage"]), "costUsd")).toBe(false);

    const noId = choiceResult();
    noId.chunks.results[0] = chunkResult(0, { responseId: undefined });
    const noIdChunk = asObject(asArray(asObject(serializeResult(noId)["chunks"])["results"])[0]);
    expect(Object.hasOwn(noIdChunk, "responseId")).toBe(false);
  });

  it("normalizes -0 to 0 and non-finite numbers to null, as JSON does", () => {
    const result = choiceResult();
    result.chunks.results[0] = chunkResult(0, { totalVariation: -0, elapsedMs: Number.NaN });
    const chunk = asObject(asArray(asObject(serializeResult(result)["chunks"])["results"])[0]);
    expect(Object.is(chunk["totalVariation"], 0)).toBe(true);
    expect(chunk["elapsedMs"]).toBeNull();
  });

  it("throws JevInfiniteCTXError for a hand-built result that JSON cannot represent", () => {
    const cyclic = choiceResult();
    const legendLike = { self: undefined as unknown };
    legendLike.self = legendLike;
    (cyclic as unknown as Record<string, unknown>)["extra"] = legendLike;
    expect(() => serializeResult(cyclic)).toThrow(JevInfiniteCTXError);

    const bigint = choiceResult();
    bigint.chunks.results[0] = chunkResult(0, { attempts: 1n as unknown as number });
    expect(() => serializeResult(bigint)).toThrow(JevInfiniteCTXError);
  });

  it("property: arbitrary option keys and probabilities survive serialization exactly", () => {
    const keyArb = fc.oneof(
      fc.constantFrom(PROTO, "constructor", "toString", "hasOwnProperty", "", "0", "10", "__defineGetter__"),
      fc.string(),
    );
    fc.assert(
      fc.property(
        fc.uniqueArray(fc.tuple(keyArb, fc.double({ min: 0, max: 1, noNaN: true })), {
          minLength: 2,
          maxLength: 12,
          selector: ([key]) => key,
        }),
        fc.boolean(),
        fc.boolean(),
        (entries, includeChunkResults, includeRawAnswers) => {
          const map = probs(entries);
          const choice = entries[0]?.[0] ?? PROTO;
          const result: ChoiceResult = {
            ...choiceResult(),
            choice,
            probabilities: map,
            chunks: {
              ...choiceResult().chunks,
              results: [
                chunkResult(0, {
                  answer: { type: "choice", choice, probabilities: probs(entries), confidence: 0.5 },
                  probabilities: probs(entries),
                }),
              ],
            },
          };

          const out = serializeResult(result, { includeChunkResults, includeRawAnswers });
          const expected = Object.fromEntries(
            entries.map(([k, v]): [string, number] => [k, Object.is(v, -0) ? 0 : v]),
          );
          const outMap = asObject(out["probabilities"]);
          expect(Object.keys(outMap)).toEqual(Object.keys(map));
          for (const [key, value] of Object.entries(expected)) {
            expect(Object.getOwnPropertyDescriptor(outMap, key)?.value).toBe(value);
          }
          expect(out["choice"]).toBe(choice);

          const chunks = asObject(out["chunks"]);
          expect(Object.hasOwn(chunks, "results")).toBe(includeChunkResults);
          if (includeChunkResults) {
            const chunk = asObject(asArray(chunks["results"])[0]);
            expect(Object.hasOwn(chunk, "answer")).toBe(includeRawAnswers);
            expect(Object.keys(asObject(chunk["probabilities"]))).toEqual(Object.keys(map));
          }

          const text = JSON.stringify(out);
          expect(JSON.stringify(JSON.parse(text))).toBe(text);
          expect(JSON.parse(text)).toEqual(out);
        },
      ),
    );
  });
});
