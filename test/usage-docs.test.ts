import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { decide } from "../src/decide.js";
import type { ChoiceQuestion } from "../src/types.js";
import { MockTransport, makeText, providerError } from "./helpers/mock-transport.js";

// Usage is recorded only from responses that come back (decide.ts
// interpretResponse), while `requests` is counted before the transport call.
// Requests aborted in flight, timed out, or failed therefore count in
// `usage.requests` but not in tokens or cost. The README must say so rather
// than promise that the reported cost is exact.

const CHOICE: ChoiceQuestion = {
  type: "choice",
  instructions: "What is this text mostly about?",
  criteria: { nature: "Nature", business: "Business" },
};

const COST_PER_RESPONSE = 0.01;

describe("usage accounting for requests without a response", () => {
  it("counts siblings aborted in flight in usage.requests but not in tokens or cost", async () => {
    let pass = 1;
    let pass1Requests = 0;
    const mock = new MockTransport({
      // Call 0 (pass 1, first chunk) fails at once with a context-limit error;
      // the other pass-1 chunks are still in flight and get aborted.
      script: [providerError("context_limit", 413)],
      latencyMs: (_state, callIndex) => (callIndex === 0 || pass > 1 ? 0 : 500),
      usage: { inputTokens: 100, outputTokens: 1, costUsd: COST_PER_RESPONSE },
    });
    const result = await decide({
      input: makeText(1500, 21),
      question: CHOICE,
      provider: { transport: mock },
      chunking: { maxStateTokens: 1000 },
      onEvent: (event) => {
        if (event.type === "rechunk") {
          pass1Requests = mock.requests.length;
          pass += 1;
        }
      },
    });

    expect(result.usage.rechunks).toBe(1);
    const abortedInFlight = pass1Requests - 1; // minus the 413 itself
    expect(abortedInFlight).toBeGreaterThan(0);

    // Every request sent is counted...
    expect(result.usage.requests).toBe(mock.requests.length);
    const answered = result.chunks.count;
    expect(result.usage.requests).toBe(pass1Requests + answered);
    // ...but only responses received contribute cost and tokens.
    expect(result.usage.costUsd).toBeCloseTo(answered * COST_PER_RESPONSE, 10);
    expect(result.usage.inputTokens).toBe(answered * 100);
    expect(result.usage.outputTokens).toBe(answered);
  });

  it("README qualifies the usage accuracy claim for requests that return no usage", () => {
    const readme = readFileSync(fileURLToPath(new URL("../README.md", import.meta.url)), "utf8");
    const line = readme.split("\n").find((l) => l.includes("**Usage from discarded passes**"));
    expect(line).toBeDefined();
    const text = line ?? "";
    expect(text).not.toMatch(/cost stays accurate/i);
    expect(text).toMatch(/aborted in flight/i);
    expect(text).toMatch(/times? out|timed out/i);
    expect(text).toMatch(/not in tokens or cost/i);
  });
});
