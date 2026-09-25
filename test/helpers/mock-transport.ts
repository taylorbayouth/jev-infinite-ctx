/**
 * Deterministic in-memory JevTransport for orchestrator tests, plus answer
 * builders and a seeded text generator.
 */

import { QUESTION_KEY } from "../../src/defaults.js";
import { JevAbortError, JevProviderError, type ProviderErrorKind } from "../../src/errors.js";
import type {
  JevQuestion,
  JevTransport,
  NativeChoiceAnswer,
  NativeJevAnswer,
  NativeJevRequest,
  NativeJevResponse,
  NativeJevUsage,
  NativeNoulAnswer,
  NativeScoreAnswer,
} from "../../src/types.js";

/** What the mock does for one call: answer, or throw the error. */
export type MockReply = NativeJevAnswer | Error;

export type MockAnswerFn = (
  state: string,
  question: JevQuestion,
  callIndex: number,
) => MockReply | Promise<MockReply>;

export interface MockTransportOptions {
  /** Default "mock". */
  name?: string;
  /** Default "mock-jev". */
  defaultModel?: string;
  /**
   * Context window reported for every model. Default 32_000. Pass
   * `undefined` explicitly for "unknown", or a function (which may throw or
   * be async).
   */
  contextWindow?: number | undefined | ((model: string) => number | undefined | Promise<number | undefined>);
  /** Replies by call index (0-based, across every decide call). Later calls use `answer`. */
  script?: readonly MockReply[];
  /** Reply for calls beyond the script. Default: `defaultAnswer(question)`. */
  answer?: MockAnswerFn;
  /** Delay before replying, in ms. Honors the request's signal. Default 0. */
  latencyMs?: number | ((state: string, callIndex: number) => number);
  /** A state longer than this fails with a context_limit JevProviderError (HTTP 413). */
  maxStateChars?: number;
  /**
   * Usage reported with each response. Default `{ inputTokens: state.length,
   * outputTokens: 1 }`. `null` reports none.
   */
  usage?: NativeJevUsage | null | ((state: string, callIndex: number) => NativeJevUsage | undefined);
  /** Model id reported in responses. Default: the requested model. */
  responseModel?: string;
}

export class MockTransport implements JevTransport {
  readonly name: string;
  readonly defaultModel: string;
  /** Every request received, in call order. */
  readonly requests: NativeJevRequest[] = [];
  /** Models passed to contextWindow(), in call order. */
  readonly contextWindowCalls: string[] = [];
  /** Call indices in the order their replies settled (resolved or rejected). */
  readonly completionOrder: number[] = [];
  inFlight = 0;
  maxInFlight = 0;

  readonly #options: MockTransportOptions;

  constructor(options: MockTransportOptions = {}) {
    this.#options = options;
    this.name = options.name ?? "mock";
    this.defaultModel = options.defaultModel ?? "mock-jev";
  }

  /** Request states in call order. */
  get states(): string[] {
    return this.requests.map((request) => request.state);
  }

  contextWindow(model: string): Promise<number | undefined> | number | undefined {
    this.contextWindowCalls.push(model);
    const option = this.#options.contextWindow;
    if (typeof option === "function") return option(model);
    return Object.hasOwn(this.#options, "contextWindow") ? option : 32_000;
  }

  async decide(request: NativeJevRequest): Promise<NativeJevResponse> {
    const callIndex = this.requests.length;
    this.requests.push(request);
    this.inFlight += 1;
    this.maxInFlight = Math.max(this.maxInFlight, this.inFlight);
    try {
      return await this.#reply(request, callIndex);
    } finally {
      this.inFlight -= 1;
      this.completionOrder.push(callIndex);
    }
  }

  async #reply(request: NativeJevRequest, callIndex: number): Promise<NativeJevResponse> {
    const { script = [], maxStateChars, responseModel } = this.#options;
    const { state } = request;
    const latency = this.#options.latencyMs;
    await delay(typeof latency === "function" ? latency(state, callIndex) : (latency ?? 0), request.signal);

    if (maxStateChars !== undefined && state.length > maxStateChars) {
      throw providerError("context_limit", 413, { message: "Mock: state exceeds the model's context window" });
    }
    const question = request.questions[QUESTION_KEY];
    if (question === undefined) throw new Error(`Mock: request has no "${QUESTION_KEY}" question`);

    const reply =
      callIndex < script.length
        ? (script[callIndex] as MockReply)
        : await (this.#options.answer ?? ((_state, q) => defaultAnswer(q)))(state, question, callIndex);
    if (reply instanceof Error) throw reply;

    const usage = this.#usage(state, callIndex);
    return {
      id: `mock-${callIndex}`,
      model: responseModel ?? request.model,
      provider: "MockProvider",
      answers: Object.fromEntries([[QUESTION_KEY, reply]]),
      ...(usage === undefined ? {} : { usage }),
    };
  }

  #usage(state: string, callIndex: number): NativeJevUsage | undefined {
    const option = this.#options.usage;
    if (option === null) return undefined;
    if (option === undefined) return { inputTokens: state.length, outputTokens: 1 };
    return typeof option === "function" ? option(state, callIndex) : option;
  }
}

/** Resolves after `ms`, or rejects with JevAbortError when `signal` aborts (like a real transport). */
function delay(ms: number, signal: AbortSignal | undefined): Promise<void> {
  return new Promise((resolve, reject) => {
    const abort = (): void => reject(new JevAbortError("Mock request aborted", { cause: signal?.reason }));
    if (signal?.aborted) {
      abort();
      return;
    }
    if (ms <= 0) {
      resolve();
      return;
    }
    const onAbort = (): void => {
      clearTimeout(timer);
      abort();
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

// ---------------------------------------------------------------------------
// Answer and error builders
// ---------------------------------------------------------------------------

/** A choice answer; `choice` defaults to the most probable key (first on ties). */
export function choiceAnswer(
  probabilities: Record<string, number>,
  confidence?: number,
  choice?: string,
): NativeChoiceAnswer {
  const entries = Object.entries(probabilities);
  const best = entries.reduce((a, b) => (b[1] > a[1] ? b : a), entries[0] ?? ["", 0]);
  return {
    type: "choice",
    choice: choice ?? best[0],
    probabilities: Object.fromEntries(entries),
    ...(confidence === undefined ? {} : { confidence }),
  };
}

/** A score answer over levels 0..n−1; `score` defaults to Σ j · p_j. */
export function scoreAnswer(probs: readonly number[], confidence?: number, score?: number): NativeScoreAnswer {
  return {
    type: "score",
    score: score ?? probs.reduce((sum, p, level) => sum + level * p, 0),
    probabilities: Object.fromEntries(probs.map((p, level) => [String(level), p])),
    ...(confidence === undefined ? {} : { confidence }),
  };
}

export function noulAnswer(noul: number): NativeNoulAnswer {
  return { type: "noul", noul };
}

/** A confident answer on the first option / lowest level; noul 0.9. */
export function defaultAnswer(question: JevQuestion): NativeJevAnswer {
  switch (question.type) {
    case "choice": {
      const keys = Object.keys(question.criteria);
      return choiceAnswer(Object.fromEntries(keys.map((key, i) => [key, i === 0 ? 1 : 0])), 0.9);
    }
    case "score":
      return scoreAnswer(question.criteria.map((_, level) => (level === 0 ? 1 : 0)), 0.9);
    case "noul":
      return noulAnswer(0.9);
  }
}

export function providerError(
  kind: ProviderErrorKind,
  status?: number,
  extra: { retryAfterMs?: number; message?: string } = {},
): JevProviderError {
  return new JevProviderError(extra.message ?? `Mock provider error (${kind})`, {
    kind,
    ...(status === undefined ? {} : { status }),
    ...(extra.retryAfterMs === undefined ? {} : { retryAfterMs: extra.retryAfterMs }),
  });
}

// ---------------------------------------------------------------------------
// Text generation
// ---------------------------------------------------------------------------

const WORDS = [
  "the", "forest", "river", "stone", "light", "morning", "quiet", "valley", "under", "above",
  "people", "walked", "along", "narrow", "path", "between", "tall", "trees", "and", "old",
  "rocks", "that", "were", "covered", "with", "moss", "while", "birds", "sang", "softly",
  "report", "shows", "growth", "in", "several", "regions", "during", "last", "quarter", "as",
  "analysts", "expected", "however", "costs", "rose", "faster", "than", "revenue", "which", "led",
];

/** mulberry32: small, fast, seeded PRNG so generated inputs are reproducible. */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * English-like prose of `wordCount` words: sentences of 6–15 words ending in
 * a period, paragraphs of 2–5 sentences separated by blank lines.
 * `extraWords` are mixed in at a fixed rate (e.g. a sentinel or topic word).
 */
export function makeText(wordCount: number, seed = 1, extraWords: readonly string[] = []): string {
  const random = seededRandom(seed);
  const pick = (): string => {
    if (extraWords.length > 0 && random() < 0.05) {
      return extraWords[Math.floor(random() * extraWords.length)] as string;
    }
    return WORDS[Math.floor(random() * WORDS.length)] as string;
  };
  const paragraphs: string[] = [];
  let remaining = wordCount;
  while (remaining > 0) {
    const sentences: string[] = [];
    const sentenceCount = 2 + Math.floor(random() * 4);
    for (let s = 0; s < sentenceCount && remaining > 0; s++) {
      const length = Math.min(remaining, 6 + Math.floor(random() * 10));
      const words = Array.from({ length }, pick);
      remaining -= length;
      const first = words[0] as string;
      words[0] = first.charAt(0).toUpperCase() + first.slice(1);
      sentences.push(`${words.join(" ")}.`);
    }
    paragraphs.push(sentences.join(" "));
  }
  return paragraphs.join("\n\n");
}

/** Occurrences of `word` in `text`. */
export function countWord(text: string, word: string): number {
  let count = 0;
  for (let at = text.indexOf(word); at !== -1; at = text.indexOf(word, at + word.length)) count++;
  return count;
}
