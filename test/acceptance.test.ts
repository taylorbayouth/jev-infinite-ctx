/**
 * Spec section 18: one test per acceptance checkbox, named after it.
 */

import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { planChunks } from "../src/chunking.js";
import { DEFAULTS, MIN_STATE_TOKENS } from "../src/defaults.js";
import { JevChunkFailedError } from "../src/errors.js";
import {
  DirectJevTransport,
  OpenRouterJevTransport,
  decide,
  defaultTokenizer,
  resolveTransport,
  serializeResult,
} from "../src/index.js";
import type {
  ChoiceQuestion,
  ChunkResult,
  JevInfiniteCTXEvent,
  JevInfiniteCTXResult,
  JevQuestion,
  JevTransport,
  NativeJevRequest,
  NoulQuestion,
  NoulResult,
  ScoreQuestion,
} from "../src/types.js";
import {
  MockTransport,
  choiceAnswer,
  countWord,
  makeText,
  noulAnswer,
  scoreAnswer,
  seededRandom,
} from "./helpers/mock-transport.js";

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

const NOUL: NoulQuestion = { type: "noul", instructions: "Is the text mainly about trees?" };

const SHORT = "A single oak grows beside a granite boulder.";

/** ~9 chunks at a MIN_STATE_TOKENS budget. */
const MEDIUM = `${makeText(1000, 41, ["oak"])}\n\n${makeText(500, 42, ["granite"])}`;

/** Over 100K estimated tokens; ~4 chunks at the default budget. */
const LARGE = makeText(72_000, 7);

/** Chunk states rebuilt from their offsets. */
function withStates(input: string, results: readonly ChunkResult[]): Array<ChunkResult & { state: string }> {
  return results.map((chunk) => ({ ...chunk, state: input.slice(chunk.start, chunk.end) }));
}

/** Concatenates chunk states after dropping each chunk's overlap with its predecessor. */
function reassemble(chunks: ReadonlyArray<{ start: number; end: number; state: string }>): string {
  return chunks
    .map((chunk, i) => chunk.state.slice(i === 0 ? 0 : (chunks[i - 1] as { end: number }).end - chunk.start))
    .join("");
}

function rockShare(state: string): number {
  const oak = countWord(state, "oak");
  const granite = countWord(state, "granite");
  return (granite + 1) / (oak + granite + 2);
}

function small<Q extends JevQuestion>(question: Q, input: string, transport: JevTransport) {
  return { input, question, provider: { transport }, chunking: { maxStateTokens: MIN_STATE_TOKENS } };
}

/** The ~100K-token run with every default, shared by the two criteria that inspect it. */
let largeRun: Promise<{ result: NoulResult; mock: MockTransport }> | undefined;
function runLarge(): Promise<{ result: NoulResult; mock: MockTransport }> {
  largeRun ??= (async () => {
    const mock = new MockTransport({ answer: (state) => noulAnswer(1 - rockShare(state)) });
    const result = await decide({ input: LARGE, question: NOUL, provider: { transport: mock } });
    return { result, mock };
  })();
  return largeRun;
}

describe("spec 18 acceptance criteria", () => {
  it("A 100K+ token string can be processed without truncation.", async () => {
    const { result, mock } = await runLarge();
    const chunks = withStates(LARGE, result.chunks.results);

    expect(result.usage.inputTokensEstimated).toBeGreaterThanOrEqual(100_000);
    expect(result.chunks.count).toBeGreaterThan(1);
    expect(mock.requests).toHaveLength(result.chunks.count);

    // Every character is inside some chunk's [start, end).
    const covered = new Uint8Array(LARGE.length);
    for (const chunk of chunks) covered.fill(1, chunk.start, chunk.end);
    expect(covered.indexOf(0)).toBe(-1);

    // Each chunk's exact text was sent, within the state budget, and the states
    // sent reconstruct the input exactly once overlaps are removed.
    const sent = new Set(mock.states);
    for (const chunk of chunks) {
      expect(sent.has(chunk.state)).toBe(true);
      expect(defaultTokenizer.count(chunk.state)).toBeLessThanOrEqual(result.chunks.stateTokenBudget);
    }
    expect(reassemble(chunks) === LARGE).toBe(true);
  });

  it("Input fitting in one chunk produces no confidence adjustment.", async () => {
    // Aggressive C3 parameters that would move any multi-chunk result.
    const confidence = { lambda: 10, agreementFloor: 0, agreementExponent: 0.5, cap: 1 };
    const answers = [
      [CHOICE, choiceAnswer({ tree: 0.7, rock: 0.2, other: 0.1 }, 0.55)],
      [SCORE, scoreAnswer([0.1, 0.2, 0.3, 0.4], 0.35)],
      [NOUL, noulAnswer(0.6)],
    ] as const;
    for (const [question, answer] of answers) {
      const mock = new MockTransport({ script: [answer] });
      const result: JevInfiniteCTXResult = await decide({
        input: SHORT,
        question,
        provider: { transport: mock },
        confidence,
      });
      expect(result.chunks.count).toBe(1);
      expect(result.chunks.effectiveCount).toBe(1);
      expect(result.confidence.agreement).toBe(1);
      expect(result.confidence.components.saturation).toBe(0);
      expect(result.confidence.adjustment).toBe(0);
      expect(result.confidence.adjusted).toBe(result.confidence.base);
    }
  });

  it("Default overlap is 5% and no source text is lost.", async () => {
    const { result } = await runLarge();
    const chunks = withStates(LARGE, result.chunks.results);
    const budget = result.chunks.stateTokenBudget;
    const target = Math.floor(budget * 0.05);

    expect(DEFAULTS.chunking.overlap).toBe(0.05);
    expect(result.chunks.overlap).toBe(0.05);
    expect(result.chunks.overlapTokens).toBe(target);
    chunks.forEach((chunk, i) => {
      if (i === 0) {
        expect(chunk.start).toBe(0);
        return;
      }
      const previous = chunks[i - 1] as ChunkResult;
      // Sliding window: each chunk re-reads the end of the previous one (spec 6.3).
      expect(chunk.start).toBeLessThan(previous.end);
      const shared = defaultTokenizer.count(LARGE.slice(chunk.start, previous.end));
      expect(shared).toBeGreaterThanOrEqual(target);
      expect(shared / budget).toBeLessThan(0.052); // 5% plus word-start snapping
    });
    expect(chunks.at(-1)?.end).toBe(LARGE.length);
    expect(reassemble(chunks) === LARGE).toBe(true);
  });

  it("Overlap is not double-counted in `weighted_mean`.", async () => {
    // At 50% overlap every later chunk re-reads half of its predecessor, so
    // counting whole chunks would give the first chunk half its real share.
    const input = makeText(3000, 43);
    const mock = new MockTransport({
      answer: (state) =>
        choiceAnswer(input.startsWith(state) ? { tree: 1, rock: 0, other: 0 } : { tree: 0, rock: 1, other: 0 }, 1),
    });
    const result = await decide({
      input,
      question: CHOICE,
      provider: { transport: mock },
      chunking: { maxStateTokens: 512, overlap: 0.5 },
    });
    const chunks = result.chunks.results;
    const [first] = chunks as [ChunkResult];
    const uniqueSum = chunks.reduce((sum, c) => sum + c.uniqueTokens, 0);
    const tokenSum = chunks.reduce((sum, c) => sum + c.estimatedTokens, 0);

    expect(chunks.length).toBeGreaterThan(3);
    expect(result.probabilities.tree).toBeCloseTo(first.uniqueTokens / uniqueSum, 12);
    expect(result.probabilities.tree).toBeGreaterThan(1.5 * (first.estimatedTokens / tokenSum));
    // Each source token is counted once: unique tokens add up to the document, whole chunks do not.
    expect(Math.abs(uniqueSum - result.usage.inputTokensEstimated)).toBeLessThanOrEqual(chunks.length);
    expect(tokenSum).toBeGreaterThan(1.3 * result.usage.inputTokensEstimated);
  });

  it("Choice works with arbitrary caller-defined string keys.", async () => {
    const keys = ["__proto__", "constructor", "toString", "🌲 tree", "has space", "", "0", "ключ", "a.b[c]"];
    const criteria = Object.fromEntries(keys.map((key) => [key, `Option ${JSON.stringify(key)}`]));
    const question: ChoiceQuestion = { type: "choice", instructions: "Pick one.", criteria };
    const order = Object.keys(criteria);
    // Rock-leaning chunks favour "🌲 tree", the rest "__proto__"; every other key gets 0.05.
    const answerFor = (state: string) => {
      const favourite = rockShare(state) > 0.5 ? "🌲 tree" : "__proto__";
      return choiceAnswer(Object.fromEntries(order.map((key) => [key, key === favourite ? 0.6 : 0.05])), 0.6);
    };

    for (const input of [SHORT, MEDIUM]) {
      const mock = new MockTransport({ answer: (state) => answerFor(state) });
      const result = await decide(small(question, input, mock));
      expect(mock.requests.every((r) => r.questions["decision"] === question)).toBe(true);
      expect(Object.keys(result.probabilities)).toEqual(order);
      expect(Object.getPrototypeOf(result.probabilities)).toBe(Object.prototype);
      expect(Object.hasOwn(result.probabilities, "__proto__")).toBe(true);
      expect(order).toContain(result.choice);
      for (const chunk of result.chunks.results) expect(Object.keys(chunk.probabilities)).toEqual(order);

      const json = JSON.parse(JSON.stringify(serializeResult(result))) as { probabilities: object };
      expect(Object.keys(json.probabilities)).toEqual(order);
    }
  });

  it("Score returns native weighted level index plus probabilities.", async () => {
    const levels = (state: string): number[] => {
      const r = rockShare(state);
      return [(1 - r) ** 2, r * (1 - r), r * (1 - r), r ** 2];
    };
    const multiMock = new MockTransport({ answer: (state) => scoreAnswer(levels(state), 0.8) });
    const multi = await decide(small(SCORE, MEDIUM, multiMock));
    const probs = ["0", "1", "2", "3"].map((level) => multi.probabilities[level] as number);
    expect(multi.chunks.count).toBeGreaterThan(1);
    expect(Object.keys(multi.probabilities)).toEqual(["0", "1", "2", "3"]);
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    // Native semantics: a level index in 0..(levels − 1), not a 0–1 score (spec 4, 9.4).
    expect(multi.score).toBeCloseTo(probs.reduce((sum, p, level) => sum + level * p, 0), 12);
    expect(multi.score).toBeGreaterThanOrEqual(0);
    expect(multi.score).toBeLessThanOrEqual(3);
    expect(multi.normalizedScore).toBeCloseTo(multi.score / 3, 12);
    expect(multi.legend).toEqual({ "0": "None", "1": "Some", "2": "Most", "3": "All" });

    const single = await decide(
      small(SCORE, SHORT, new MockTransport({ script: [scoreAnswer([0, 0.3, 0.7, 0], 0.9, 1.7)] })),
    );
    expect(single.score).toBe(1.7);
    expect(single.probabilities).toEqual({ "0": 0, "1": 0.3, "2": 0.7, "3": 0 });
  });

  it("Noul returns aggregated yes probability.", async () => {
    const mock = new MockTransport({ answer: (state) => noulAnswer(1 - rockShare(state)) });
    const result = await decide(small(NOUL, MEDIUM, mock));
    const chunks = result.chunks.results;
    const yes = (chunk: ChunkResult): number => (chunk.answer.type === "noul" ? chunk.answer.noul : NaN);
    expect(chunks.length).toBeGreaterThan(1);
    expect(result.noul).toBeCloseTo(chunks.reduce((sum, c) => sum + c.weight * yes(c), 0), 12);
    chunks.forEach((c) => expect(c.probabilities.yes).toBe(yes(c)));
  });

  it("Choice and Score preserve Jev's raw chunk confidence values.", async () => {
    const confidenceFor = (state: string): number => 0.3 + (state.length % 61) / 100;
    const choice = await decide(
      small(
        CHOICE,
        MEDIUM,
        new MockTransport({ answer: (s) => choiceAnswer({ tree: 0.6, rock: 0.3, other: 0.1 }, confidenceFor(s)) }),
      ),
    );
    const score = await decide(
      small(
        SCORE,
        MEDIUM,
        new MockTransport({ answer: (s) => scoreAnswer([0.1, 0.2, 0.3, 0.4], confidenceFor(s)) }),
      ),
    );
    for (const result of [choice, score]) {
      const chunks = withStates(MEDIUM, result.chunks.results);
      expect(new Set(chunks.map((c) => c.jevConfidence)).size).toBeGreaterThan(1);
      for (const chunk of chunks) {
        expect(chunk.jevConfidence).toBe(confidenceFor(chunk.state));
        expect(chunk.answer.type !== "noul" && chunk.answer.confidence).toBe(confidenceFor(chunk.state));
      }
      // The package's numbers are reported beside, never instead of, Jev's.
      expect(result.confidence.source).toBe("jev");
      expect(result.confidence.base).toBeCloseTo(
        chunks.reduce((sum, c) => sum + c.weight * confidenceFor(c.state), 0),
        12,
      );
    }
  });

  it("Noul clearly labels package-derived certainty as derived, not Jev confidence.", async () => {
    for (const input of [SHORT, MEDIUM]) {
      const mock = new MockTransport({ answer: (state) => noulAnswer(1 - rockShare(state)) });
      const result = await decide(small(NOUL, input, mock));
      expect(result.confidence.source).toBe("derived");
      expect(result.confidence.base).toBeCloseTo(Math.abs(2 * result.noul - 1), 12);
      for (const chunk of result.chunks.results) {
        expect(chunk.jevConfidence).toBeUndefined();
        expect("confidence" in chunk.answer).toBe(false);
      }
    }
  });

  it("Agreement is calculated from all chunk probability distributions.", async () => {
    const run = async (lastDisagrees: boolean) => {
      const mock = new MockTransport({
        answer: (state) =>
          choiceAnswer(
            lastDisagrees && MEDIUM.endsWith(state)
              ? { tree: 0.05, rock: 0.9, other: 0.05 }
              : { tree: 0.9, rock: 0.05, other: 0.05 },
            0.9,
          ),
      });
      return decide(small(CHOICE, MEDIUM, mock));
    };
    const agreeing = await run(false);
    const split = await run(true);

    const chunks = split.chunks.results;
    const aggregate = Object.values(split.probabilities);
    const tvs = chunks.map(
      (c) => 0.5 * Object.values(c.probabilities).reduce((s, p, j) => s + Math.abs(p - (aggregate[j] as number)), 0),
    );
    const disagreement = chunks.reduce((s, c, i) => s + c.weight * (tvs[i] as number), 0);
    expect(chunks.length).toBeGreaterThan(2);
    chunks.forEach((c, i) => expect(c.totalVariation).toBeCloseTo(tvs[i] as number, 12));
    expect(split.confidence.agreement).toBeCloseTo(1 - disagreement, 12);
    // Changing only the final chunk moves the agreement, so every chunk is included.
    expect(agreeing.confidence.agreement).toBeCloseTo(1, 12);
    expect(split.confidence.agreement).toBeLessThan(agreeing.confidence.agreement - 0.01);
  });

  it("C3 adjustment is bounded and equals zero for one chunk.", async () => {
    const unit = (min: number) => fc.double({ min, max: 1, noNaN: true });
    const distribution = fc
      .tuple(unit(0.001), unit(0), unit(0))
      .map(([a, b, c]) => [a / (a + b + c), b / (a + b + c), c / (a + b + c)] as const);
    const chunkAnswer = fc.record({ distribution, confidence: unit(0) });
    const options = fc.record({
      cap: fc.double({ min: 0.01, max: 1, noNaN: true }),
      lambda: fc.double({ min: 0.01, max: 10, noNaN: true }),
      agreementFloor: fc.double({ min: 0, max: 0.99, noNaN: true }),
      agreementExponent: fc.double({ min: 0.1, max: 5, noNaN: true }),
    });

    await fc.assert(
      fc.asyncProperty(fc.array(chunkAnswer, { minLength: 1, maxLength: 12 }), options, async (answers, confidence) => {
        for (const input of [SHORT, MEDIUM]) {
          const mock = new MockTransport({
            answer: (_state, _q, callIndex) => {
              const answer = answers[callIndex % answers.length] as (typeof answers)[number];
              const [tree, rock, other] = answer.distribution;
              return choiceAnswer({ tree, rock, other }, answer.confidence);
            },
          });
          const result = await decide({ ...small(CHOICE, input, mock), confidence });
          const { base, adjusted, adjustment } = result.confidence;
          expect(adjustment).toBeGreaterThanOrEqual(0);
          expect(adjustment).toBeCloseTo(adjusted - base, 15);
          expect(adjusted).toBeLessThanOrEqual(Math.max(base, confidence.cap));
          expect(adjusted).toBeLessThanOrEqual(1);
          if (result.chunks.count === 1) expect(adjustment).toBe(0);
        }
      }),
      { numRuns: 40 },
    );
  });

  it("Low agreement suppresses the adjustment.", async () => {
    const run = async (pattern: ReadonlyArray<readonly [number, number, number]>) => {
      const mock = new MockTransport({
        answer: (_state, _q, callIndex) => {
          const [tree, rock, other] = pattern[callIndex % pattern.length] as readonly [number, number, number];
          return choiceAnswer({ tree, rock, other }, 0.9);
        },
      });
      return decide(small(CHOICE, MEDIUM, mock));
    };
    const agree = await run([[0.9, 0.05, 0.05]]);
    const mixed = await run([[0.9, 0.05, 0.05], [0.5, 0.45, 0.05]]);
    const conflict = await run([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);

    // Same chunks and the same base confidence; only agreement differs.
    expect(agree.confidence.base).toBeCloseTo(0.9, 12);
    expect(conflict.confidence.base).toBeCloseTo(0.9, 12);
    expect(agree.confidence.agreement).toBeGreaterThan(mixed.confidence.agreement);
    expect(mixed.confidence.agreement).toBeGreaterThan(conflict.confidence.agreement);
    expect(agree.confidence.adjustment).toBeGreaterThan(mixed.confidence.adjustment);
    expect(mixed.confidence.adjustment).toBeGreaterThan(0);
    // At or below the agreement floor the gate closes completely.
    expect(conflict.confidence.agreement).toBeLessThanOrEqual(DEFAULTS.confidence.agreementFloor);
    expect(conflict.confidence.components.gate).toBe(0);
    expect(conflict.confidence.adjustment).toBe(0);
    expect(conflict.confidence.adjusted).toBe(conflict.confidence.base);
  });

  it("A permanently failed chunk fails the whole call by default.", async () => {
    const poison = "poisonedpassage";
    const input = `${makeText(700, 44)} ${poison} ${makeText(700, 45)}`;
    const events: JevInfiniteCTXEvent[] = [];
    const mock = new MockTransport({
      answer: (state) => (state.includes(poison) ? new Error("provider keeps failing") : noulAnswer(0.7)),
    });
    const error = await decide({
      ...small(NOUL, input, mock),
      execution: { retries: 2, retryBaseDelayMs: 0 },
      onEvent: (event) => events.push(event),
    }).catch((e: unknown) => e);

    expect(error).toBeInstanceOf(JevChunkFailedError);
    const failed = planChunks(input, {
      stateTokenBudget: MIN_STATE_TOKENS,
      overlap: DEFAULTS.chunking.overlap,
      tokenizer: defaultTokenizer,
      preferNaturalBoundaries: true,
    }).chunks[(error as JevChunkFailedError).chunkIndex];
    expect(input.slice(failed?.start, failed?.end)).toContain(poison);
    expect(events.map((e) => e.type)).not.toContain("decision.completed");
    expect(events.map((e) => e.type)).toContain("decision.failed");
  });

  it("Raw chunk results are inspectable.", async () => {
    const mock = new MockTransport({
      answer: (state) => choiceAnswer({ tree: 1 - rockShare(state), rock: rockShare(state), other: 0 }, 0.8),
      usage: (state) => ({ inputTokens: state.length, outputTokens: 4, costUsd: 0.0001 }),
    });
    const result = await decide(small(CHOICE, MEDIUM, mock));
    const chunks = withStates(MEDIUM, result.chunks.results);

    expect(chunks).toHaveLength(result.chunks.count);
    chunks.forEach((chunk, i) => {
      expect(chunk.index).toBe(i);
      expect(chunk.answer).toEqual(
        choiceAnswer({ tree: 1 - rockShare(chunk.state), rock: rockShare(chunk.state), other: 0 }, 0.8),
      );
      expect(Object.keys(chunk.probabilities)).toEqual(["tree", "rock", "other"]);
      expect(chunk).toMatchObject({
        attempts: 1,
        model: "mock-jev",
        usage: { inputTokens: chunk.state.length, outputTokens: 4, costUsd: 0.0001 },
      });
      expect(chunk.responseId).toMatch(/^mock-/);
      expect(chunk.totalVariation).toBeGreaterThanOrEqual(0);
      expect(chunk.estimatedTokens).toBe(defaultTokenizer.count(chunk.state));
    });

    // Serialized by default; optional for payload size (spec 12).
    const full = serializeResult(result) as { chunks: { results: Array<Record<string, unknown>> } };
    expect(full.chunks.results).toHaveLength(result.chunks.count);
    expect(full.chunks.results[0]?.["answer"]).toEqual(chunks[0]?.answer);
    const lean = serializeResult(result, { includeChunkResults: false }) as { chunks: Record<string, unknown> };
    expect(lean.chunks["results"]).toBeUndefined();
    expect(lean.chunks["count"]).toBe(result.chunks.count);
    const noAnswers = serializeResult(result, { includeRawAnswers: false }) as typeof full;
    expect(noAnswers.chunks.results.every((c) => !("answer" in c) && "probabilities" in c)).toBe(true);
  });

  it("Provider implementation is replaceable through `JevTransport`.", async () => {
    const seen: NativeJevRequest[] = [];
    const transport: JevTransport = {
      name: "in-house",
      defaultModel: "house-jev",
      contextWindow: () => 2_000,
      async decide(request) {
        seen.push(request);
        return {
          model: "house-jev-build-7",
          answers: { decision: noulAnswer(request.state.includes("oak") ? 0.8 : 0.3) },
          usage: { inputTokens: 10, outputTokens: 1 },
        };
      },
    };
    const result = await decide({ input: MEDIUM, question: NOUL, provider: { transport } });

    expect(result.provider).toBe("in-house");
    expect(result.model).toBe("house-jev-build-7");
    expect(result.chunks.count).toBeGreaterThan(1);
    expect(result.chunks.stateTokenBudget).toBeLessThan(2_000 - DEFAULTS.chunking.protocolReserve);
    expect(seen).toHaveLength(result.chunks.count);
    for (const request of seen) {
      expect(Object.keys(request).sort()).toEqual(["model", "questions", "signal", "state"]);
      expect(request.model).toBe("house-jev");
      expect(request.questions).toEqual({ decision: NOUL });
    }
    // The built-in names resolve to the two bundled adapters of the same interface.
    expect(resolveTransport({ transport: "openrouter", apiKey: "k" })).toBeInstanceOf(OpenRouterJevTransport);
    expect(resolveTransport({ transport: "direct", apiKey: "k" })).toBeInstanceOf(DirectJevTransport);
  });

  it("No business threshold is imposed by the library.", async () => {
    // Probabilities next to 0.5 and very low confidences come back as they are:
    // no snapping, no "undecided" outcome, no rejection.
    for (const p of [0.4999, 0.5, 0.5001, 0.01, 0.99]) {
      const result = await decide(small(NOUL, SHORT, new MockTransport({ script: [noulAnswer(p)] })));
      expect(result.noul).toBe(p);
      expect(Object.keys(result)).toEqual([
        "type",
        "noul",
        "confidence",
        "aggregation",
        "chunks",
        "usage",
        "model",
        "provider",
      ]);
    }
    const random = seededRandom(9);
    const nearTie = await decide(
      small(
        CHOICE,
        MEDIUM,
        new MockTransport({
          answer: () => {
            const jitter = (random() - 0.5) / 100;
            return choiceAnswer({ tree: 0.34 + jitter, rock: 0.33 - jitter, other: 0.33 }, 0.02);
          },
        }),
      ),
    );
    expect(Object.keys(nearTie)).toEqual([
      "type",
      "choice",
      "probabilities",
      "confidence",
      "aggregation",
      "chunks",
      "usage",
      "model",
      "provider",
    ]);
    const probs = Object.values(nearTie.probabilities);
    expect(nearTie.probabilities[nearTie.choice]).toBe(Math.max(...probs));
    expect(nearTie.confidence.base).toBeLessThan(0.05);
    expect(Object.keys(nearTie.confidence)).toEqual([
      "base",
      "adjusted",
      "adjustment",
      "source",
      "agreement",
      "method",
      "components",
    ]);
  });
});
