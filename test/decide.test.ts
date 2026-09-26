import { afterEach, describe, expect, it, vi } from "vitest";
import { DecideError, decide, estimateTokens, type Question } from "../src/index.js";
import { choice, fakeJev, httpError, noul, prose, score } from "./helpers.js";

const TOPIC = {
  type: "choice",
  instructions: "What is this text mainly about?",
  criteria: { rock: "Rocks or geology", tree: "Trees or forests", other: "Something else" },
} as const;
const SEVERITY: Question = { type: "score", instructions: "How severe?", criteria: ["none", "minor", "major"] };
const FORESTS: Question = { type: "noul", instructions: "Does this text mention forests?" };

/** Answers by counting topic words in the chunk, like a caricature of Jev. */
const byTopic = (state: string) => {
  const trees = (state.match(/oak|maple|forest|conifer/gi) ?? []).length + 1;
  const rocks = (state.match(/granite|basalt|magma|quartz/gi) ?? []).length + 1;
  return choice({ rock: rocks / (rocks + trees), tree: trees / (rocks + trees), other: 0 }, 0.9);
};

/** About 100K estimated tokens: several chunks. */
const LONG = prose(2500) + prose(1000, "tree");

async function failure(promise: Promise<unknown>): Promise<DecideError> {
  const error = await promise.then(
    () => undefined,
    (e: unknown) => e,
  );
  expect(error).toBeInstanceOf(DecideError);
  return error as DecideError;
}

afterEach(() => vi.unstubAllEnvs());

describe("decide", () => {
  it("returns Jev's own answer when the input fits in one chunk", async () => {
    const jev = fakeJev(() => choice({ rock: 0.8, tree: 0.15, other: 0.05 }, 0.77));
    const result = await decide({ input: "Basalt forms from fast-cooling lava.", question: TOPIC, transport: jev.transport });
    expect(result).toEqual({
      type: "choice",
      choice: "rock",
      probabilities: { rock: 0.8, tree: 0.15, other: 0.05 },
      confidence: 0.77,
      agreement: 1,
      chunks: [{ start: 0, end: 36, weight: 1, choice: "rock", probabilities: { rock: 0.8, tree: 0.15, other: 0.05 }, confidence: 0.77 }],
      model: "typesafe/jev-1.13-test",
      usage: { requests: 1, inputTokens: 100, costUsd: 0.001 },
    });
  });

  it("asks about every part of a long input and averages by chunk weight", async () => {
    const jev = fakeJev(byTopic);
    const result = await decide({ input: LONG, question: TOPIC, transport: jev.transport });
    expect(result.chunks.length).toBeGreaterThan(3);
    // Nothing skipped: chunk states, minus overlap, rebuild the input exactly.
    let rebuilt = "";
    result.chunks.forEach((chunk, i) => {
      expect(jev.requests[i]!.state).toBe(LONG.slice(chunk.start, chunk.end));
      rebuilt += LONG.slice(i === 0 ? 0 : result.chunks[i - 1]!.end, chunk.end);
    });
    expect(rebuilt).toBe(LONG);
    const expected = result.chunks.reduce((sum, chunk) => sum + chunk.weight * chunk.probabilities.tree, 0);
    expect(result.probabilities.tree).toBeCloseTo(expected, 12);
    expect(result.choice).toBe("rock");
    expect(result.agreement).toBeLessThan(1);
    expect(jev.peak()).toBeLessThanOrEqual(4);
  });

  it("averages scores and noul probabilities", async () => {
    const severity = await decide({ input: LONG, question: SEVERITY, transport: fakeJev((_, __, call) => score(call === 0 ? [0, 0, 1] : [1, 0, 0])).transport });
    const first = severity.chunks[0]!;
    expect(severity.score).toBeCloseTo(2 * first.weight, 12);
    const forests = await decide({ input: LONG, question: FORESTS, transport: fakeJev((state) => noul(/forest/.test(state) ? 1 : 0)).transport });
    const expected = forests.chunks.reduce((sum, chunk) => sum + chunk.weight * chunk.noul, 0);
    expect(forests.noul).toBeCloseTo(expected, 12);
  });

  it('answers "appears anywhere" questions with combine: "max"', async () => {
    const jev = fakeJev((state) => noul(/forest/.test(state) ? 0.97 : 0.02));
    const result = await decide({ input: LONG, question: FORESTS, combine: "max", transport: jev.transport });
    expect(result.noul).toBe(0.97);
    await expect(decide({ input: LONG, question: TOPIC, combine: "max", transport: jev.transport })).rejects.toMatchObject({ code: "invalid_request" });
  });

  it("is deterministic", async () => {
    const run = () => decide({ input: LONG, question: TOPIC, transport: fakeJev(byTopic).transport });
    expect(JSON.stringify(await run())).toBe(JSON.stringify(await run()));
  });

  it("rejects an input over maxInputTokens before sending anything", async () => {
    const jev = fakeJev(byTopic);
    const error = await failure(decide({ input: LONG, question: TOPIC, transport: jev.transport, maxInputTokens: 50_000 }));
    expect(error.code).toBe("input_too_large");
    expect(error.message).toContain(estimateTokens(LONG).toLocaleString("en-US"));
    expect(jev.requests).toHaveLength(0);
  });

  it("defaults to a 250,000-token limit", async () => {
    const jev = fakeJev(byTopic);
    const error = await failure(decide({ input: "x".repeat(750_003), question: TOPIC, transport: jev.transport }));
    expect(error.code).toBe("input_too_large");
    await expect(decide({ input: "x".repeat(750_000), question: TOPIC, transport: jev.transport })).resolves.toBeDefined();
  });

  it("fails the whole call when a chunk fails, naming the chunk", async () => {
    const jev = fakeJev((state, _, call) => {
      if (call === 2) throw httpError(401);
      return byTopic(state);
    });
    const error = await failure(decide({ input: LONG, question: TOPIC, transport: jev.transport }));
    expect(error).toMatchObject({ code: "request_failed", chunk: 2, status: 401 });
    expect(error.message).toMatch(/^Chunk 3 of \d+ failed: HTTP 401$/);
  });

  it("fails on a malformed answer instead of guessing", async () => {
    const jev = fakeJev(() => ({ type: "choice", choice: "lava", probabilities: { rock: 1 } }));
    const error = await failure(decide({ input: "Some text.", question: TOPIC, transport: jev.transport }));
    expect(error).toMatchObject({ code: "request_failed", chunk: 0 });
    expect(error.message).toContain("malformed answer");
  });

  it("retries rate limits and server errors, then succeeds", async () => {
    const jev = fakeJev((state, _, call) => {
      if (call < 2) throw httpError(call === 0 ? 429 : 503, 0);
      return byTopic(state);
    });
    const result = await decide({ input: "Granite and basalt.", question: TOPIC, transport: jev.transport });
    expect(result.choice).toBe("rock");
    expect(result.usage.requests).toBe(3);
  });

  it("retries network errors", async () => {
    const jev = fakeJev((state, _, call) => {
      if (call === 0) throw new TypeError("fetch failed");
      return byTopic(state);
    });
    await expect(decide({ input: "Granite.", question: TOPIC, transport: jev.transport })).resolves.toMatchObject({ choice: "rock" });
  });

  it("gives up after three retries", async () => {
    const jev = fakeJev(() => {
      throw httpError(503, 0);
    });
    const error = await failure(decide({ input: "Granite.", question: TOPIC, transport: jev.transport }));
    expect(error).toMatchObject({ code: "request_failed", status: 503 });
    expect(jev.requests).toHaveLength(4);
  });

  it("shrinks the chunks and replans the whole input when Jev says one is too long", async () => {
    const limit = 60_000; // characters Jev will accept in this test
    const jev = fakeJev((state) => {
      if (state.length > limit) throw httpError(413);
      return byTopic(state);
    });
    const result = await decide({ input: LONG, question: TOPIC, transport: jev.transport });
    const final = result.chunks.map((chunk) => LONG.slice(chunk.start, chunk.end));
    expect(final.every((state) => state.length <= limit)).toBe(true);
    expect(final.join("").length).toBeGreaterThanOrEqual(LONG.length);
    expect(result.chunks.at(-1)!.end).toBe(LONG.length);
    expect(result.usage.requests).toBeGreaterThan(result.chunks.length);
  });

  it("fails if Jev keeps saying chunks are too long", async () => {
    const jev = fakeJev(() => {
      throw httpError(413);
    });
    const error = await failure(decide({ input: LONG, question: TOPIC, transport: jev.transport }));
    expect(error).toMatchObject({ code: "request_failed", status: 413 });
    expect(error.message).toContain("too long");
  });

  it("stops when the caller aborts", async () => {
    const controller = new AbortController();
    const jev = fakeJev((state, _, call) => {
      if (call === 1) controller.abort();
      return byTopic(state);
    });
    const error = await failure(decide({ input: LONG, question: TOPIC, transport: jev.transport, signal: controller.signal }));
    expect(error.code).toBe("aborted");
    const before = await failure(decide({ input: LONG, question: TOPIC, transport: jev.transport, signal: AbortSignal.abort() }));
    expect(before.code).toBe("aborted");
  });

  it.each([
    ["an unknown option", { overlap: 0.1 }],
    ["an empty input", { input: " \n " }],
    ["a bad limit", { maxInputTokens: 0 }],
    ["a bad combine", { combine: "median" }],
    ["a question too long for Jev", { question: { ...TOPIC, instructions: "x".repeat(100_000) } }],
  ])("rejects %s before sending anything", async (_, override) => {
    const jev = fakeJev(byTopic);
    const options = { input: "Granite.", question: TOPIC, transport: jev.transport, ...override };
    const error = await failure(decide(options as Parameters<typeof decide>[0]));
    expect(error.code).toBe("invalid_request");
    expect(jev.requests).toHaveLength(0);
  });

  it("needs an API key unless a transport is given", async () => {
    vi.stubEnv("OPENROUTER_API_KEY", "");
    const error = await failure(decide({ input: "Granite.", question: TOPIC }));
    expect(error).toMatchObject({ code: "invalid_request", message: expect.stringContaining("OPENROUTER_API_KEY") });
  });
});
