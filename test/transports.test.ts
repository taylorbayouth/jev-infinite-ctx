import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import fc from "fast-check";
import {
  classifyHttpError,
  isContextLimitMessage,
  parseNativeResponse,
  parseRetryAfter,
  postDecision,
  type HttpDecisionConfig,
} from "../src/transports/http.js";
import { OPENROUTER_DEFAULT_MODEL, OpenRouterJevTransport } from "../src/transports/openrouter.js";
import { DIRECT_DEFAULT_MODEL, DirectJevTransport } from "../src/transports/direct.js";
import { JevAbortError, JevProviderError, JevValidationError, type ProviderErrorKind } from "../src/errors.js";
import type { ChoiceQuestion, JsonObject, NativeJevRequest } from "../src/types.js";

// ---------------------------------------------------------------------------
// Fixtures and helpers
// ---------------------------------------------------------------------------

type FetchMock = Mock<typeof fetch>;

/** Distinctive, multi-line, quote-bearing state so JSON escaping differs from the raw text. */
const STATE = [
  'CONFIDENTIAL: Project Nightingale "Q3 forecast" shows revenue of $4.2M across 17 regions.',
  "",
  "Line two mentions the context window and too many tokens, to tempt the classifier.",
  "\tTabbed line with a backslash \\ and unicode: café, naïve, 東京, 🚀.",
].join("\n");

/** The body a provider would contain if it quoted STATE inside a JSON string. */
const STATE_JSON_ESCAPED = JSON.stringify(STATE).slice(1, -1);

const WIRE_RESPONSE = {
  id: "gen-dec-1790265859-EaXKST7hul1Wcqots1ZK",
  model: "typesafe/jev-1.13-20260917",
  provider: "TypeSafe",
  answers: {
    is_bug: { type: "noul", noul: 0.96 },
    team: {
      type: "choice",
      choice: "payments",
      probabilities: { payments: 0.78, frontend: 0.22, account: 0, none: 0 },
      confidence: 0.67,
    },
    urgency: {
      type: "score",
      score: 1.99,
      legend: { "0": "Can wait for the next release", "1": "Should be fixed this week", "2": "Blocking revenue right now" },
      probabilities: { "0": 0, "1": 0, "2": 1 },
      confidence: 0.99,
    },
  },
  usage: { input_tokens: 476, output_tokens: 70, cost: 0.000019992 },
};

const NORMALIZED_WIRE_RESPONSE = {
  id: "gen-dec-1790265859-EaXKST7hul1Wcqots1ZK",
  model: "typesafe/jev-1.13-20260917",
  provider: "TypeSafe",
  answers: WIRE_RESPONSE.answers,
  usage: { inputTokens: 476, outputTokens: 70, costUsd: 0.000019992 },
};

const QUESTION: ChoiceQuestion = {
  type: "choice",
  instructions: "Which team should own this ticket?",
  criteria: { payments: "Checkout issues.", frontend: "Rendering issues.", none: "None of the above." },
};

function request(overrides: Partial<NativeJevRequest> = {}): NativeJevRequest {
  return { model: "typesafe/jev-1.13", state: STATE, questions: { decision: QUESTION }, ...overrides };
}

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  const headers = new Headers(init.headers);
  headers.set("content-type", "application/json");
  return new Response(JSON.stringify(body), { status: 200, ...init, headers });
}

function textResponse(text: string, init: ResponseInit = {}): Response {
  return new Response(text, { status: 200, ...init });
}

/** Responses are single-use, so the mock builds a fresh one per call. */
function fetchAlways(factory: () => Response): FetchMock {
  return vi.fn<typeof fetch>(async () => factory());
}

function fetchSequence(...factories: Array<() => Response>): FetchMock {
  const fn = vi.fn<typeof fetch>();
  for (const factory of factories) fn.mockImplementationOnce(async () => factory());
  return fn;
}

/** Never settles on its own; rejects when its signal aborts, as a real fetch does. */
function fetchHonoringSignal(): FetchMock {
  return vi.fn<typeof fetch>(
    (_input, init) =>
      new Promise<Response>((_resolve, reject) => {
        const signal = init?.signal;
        if (!signal) return;
        if (signal.aborted) reject(signal.reason);
        signal.addEventListener("abort", () => reject(signal.reason), { once: true });
      }),
  );
}

/** Never settles and ignores its signal: the deadline must still hold. */
function fetchIgnoringSignal(): FetchMock {
  return vi.fn<typeof fetch>(() => new Promise<Response>(() => undefined));
}

function callAt(fetchMock: FetchMock, index = 0): { url: string; init: RequestInit; headers: Headers } {
  const call = fetchMock.mock.calls[index];
  if (call === undefined) throw new Error(`fetch call ${index} was not made`);
  const [input, init = {}] = call;
  return { url: String(input), init, headers: new Headers(init.headers) };
}

function sentBody(fetchMock: FetchMock, index = 0): Record<string, unknown> {
  const body: unknown = JSON.parse(String(callAt(fetchMock, index).init.body));
  if (typeof body !== "object" || body === null) throw new Error("body is not an object");
  return body as Record<string, unknown>;
}

async function rejectionOf(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected the promise to reject");
}

async function providerErrorOf(promise: Promise<unknown>): Promise<JevProviderError> {
  const error = await rejectionOf(promise);
  expect(error).toBeInstanceOf(JevProviderError);
  return error as JevProviderError;
}

function httpConfig(fetchFn: typeof fetch, overrides: Partial<HttpDecisionConfig> = {}): HttpDecisionConfig {
  return {
    url: "https://jev.example.test/decide",
    headers: { "content-type": "application/json", authorization: "Bearer test" },
    fetch: fetchFn,
    timeoutMs: 1_000,
    providerName: "TestProvider",
    ...overrides,
  };
}

function wireBody(state: string = STATE): JsonObject {
  return { model: "typesafe/jev-1.13", state, questions: { decision: { type: "noul", instructions: "Is it?" } } };
}

/** Length of the shortest state echo redaction guarantees to remove (2 x 16-char probe - 1). */
const GUARANTEED_RUN = 31;

/** True when `text` still contains a run of `minRun` characters that also occurs in `secret`. */
function sharesRun(text: string, secret: string, minRun = GUARANTEED_RUN): boolean {
  for (let i = 0; i + minRun <= text.length; i += 1) {
    if (secret.includes(text.slice(i, i + minRun))) return true;
  }
  return false;
}

function expectNoState(error: JevProviderError, state: string = STATE): void {
  const escaped = JSON.stringify(state).slice(1, -1);
  for (const text of [error.message, error.body ?? ""]) {
    expect(text.includes(state)).toBe(false);
    expect(text.includes(escaped)).toBe(false);
    expect(sharesRun(text, state)).toBe(false);
    expect(sharesRun(text, escaped)).toBe(false);
  }
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});

// ---------------------------------------------------------------------------
// parseRetryAfter
// ---------------------------------------------------------------------------

describe("parseRetryAfter", () => {
  const NOW = Date.parse("Sun, 06 Nov 1994 08:49:07 GMT");

  it.each([
    ["0", 0],
    ["1", 1_000],
    ["120", 120_000],
    [" 7 ", 7_000],
    ["1.5", 1_500],
  ])("reads delay-seconds %j as %d ms", (value, expected) => {
    expect(parseRetryAfter(value, NOW)).toBe(expected);
  });

  it.each([
    ["IMF-fixdate", "Sun, 06 Nov 1994 08:49:37 GMT"],
    ["RFC 850", "Sunday, 06-Nov-94 08:49:37 GMT"],
    ["asctime (implicitly GMT)", "Sun Nov  6 08:49:37 1994"],
  ])("reads an %s HTTP-date as the delay until that instant", (_format, value) => {
    expect(parseRetryAfter(value, NOW)).toBe(30_000);
  });

  it("clamps an HTTP-date in the past to 0", () => {
    expect(parseRetryAfter("Sun, 06 Nov 1994 08:00:00 GMT", NOW)).toBe(0);
  });

  it("defaults the reference time to Date.now()", () => {
    const value = new Date(Date.now() + 60_000).toUTCString();
    const delay = parseRetryAfter(value);
    expect(delay).toBeGreaterThanOrEqual(58_000);
    expect(delay).toBeLessThanOrEqual(60_000);
  });

  it.each([undefined, null, "", "   ", "soon", "-5", "5s", "1e3", "0x10", "Thu, 99 Foo 2020 99:99:99 GMT"])(
    "returns undefined for %j",
    (value) => {
      expect(parseRetryAfter(value, NOW)).toBeUndefined();
    },
  );

  it("property: delay-seconds n maps to exactly n * 1000 ms", () => {
    fc.assert(
      fc.property(fc.nat({ max: 10_000_000 }), (seconds) => {
        expect(parseRetryAfter(String(seconds), NOW)).toBe(seconds * 1000);
      }),
    );
  });

  it("property: an IMF-fixdate s seconds ahead maps to s * 1000 ms", () => {
    fc.assert(
      fc.property(fc.nat({ max: 10_000_000 }), (seconds) => {
        const value = new Date(NOW + seconds * 1000).toUTCString();
        expect(parseRetryAfter(value, NOW)).toBe(seconds * 1000);
      }),
    );
  });
});

// ---------------------------------------------------------------------------
// isContextLimitMessage
// ---------------------------------------------------------------------------

describe("isContextLimitMessage", () => {
  it.each([
    "This endpoint's maximum context length is 32768 tokens. However, you requested about 40000 tokens.",
    "context_length_exceeded",
    "Input exceeds the model's context window",
    "Request exceeds context limit",
    "CONTEXT WINDOW EXCEEDED",
    "Too many tokens in state",
    "too many input tokens",
    "Token limit exceeded",
    "The state exceeds the maximum of 32000 tokens",
    "Your request exceeds the maximum allowed number of tokens",
    "Input exceeds the maximum length",
    "prompt is too long: 40000 tokens > 32000 maximum",
    "Input is too long for the requested model.",
    "Payload Too Large",
    "State too large",
    "Request Entity Too Large",
    "input tokens exceed the configured limit",
    "state size exceeded",
    "Please reduce the length of the messages or completion.",
    "exceeds maximum context",
  ])("detects %j", (message) => {
    expect(isContextLimitMessage(message)).toBe(true);
  });

  it.each([
    "Invalid token",
    "invalid_token",
    "Token has expired",
    "The provided API token is invalid",
    "Missing authentication token",
    "Invalid token: exceeded retry limit",
    "Unexpected token < in JSON at position 0",
    "max_tokens must be at least 1",
    "Rate limit exceeded",
    "Model not found",
    "temperature exceeds the maximum allowed value",
    "Number of questions exceeds the maximum of 16 entries",
    "Insufficient credits",
    "Internal server error",
    "Provider returned error",
    "",
  ])("does not flag %j", (message) => {
    expect(isContextLimitMessage(message)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// classifyHttpError
// ---------------------------------------------------------------------------

describe("classifyHttpError", () => {
  const openRouterError = (message: string, code = 400): string => JSON.stringify({ error: { code, message } });

  const CASES: Array<[number, string, ProviderErrorKind, boolean]> = [
    [413, "anything at all", "context_limit", false],
    [400, openRouterError("This endpoint's maximum context length is 32768 tokens."), "context_limit", false],
    [400, openRouterError("questions.decision.type is invalid"), "bad_request", false],
    [400, openRouterError("Invalid token"), "bad_request", false],
    [422, "prompt is too long", "context_limit", false],
    [422, "Unprocessable entity", "bad_request", false],
    [401, "Invalid token", "auth", false],
    [401, "context length exceeded", "auth", false],
    [403, "Forbidden", "auth", false],
    [402, "Insufficient credits", "payment", false],
    [404, "No endpoints found for model", "not_found", false],
    [408, "Request timeout", "timeout", true],
    [429, "Rate limited", "rate_limit", true],
    [429, "Upstream overloaded", "rate_limit", true],
    [500, "Internal server error", "server", true],
    [502, "Bad gateway", "server", true],
    [504, "Gateway timeout", "server", true],
    [503, "Service unavailable", "overloaded", true],
    [529, "", "overloaded", true],
    [500, openRouterError("Model is overloaded, try again", 500), "overloaded", true],
    [418, "I'm a teapot", "unknown", false],
    [409, "Conflict", "unknown", false],
  ];

  it.each(CASES)("HTTP %d with body %j -> %s (retryable %s)", (status, body, kind, retryable) => {
    const error = classifyHttpError(status, body, undefined, "OpenRouter");
    expect(error).toBeInstanceOf(JevProviderError);
    expect(error.kind).toBe(kind);
    expect(error.status).toBe(status);
    expect(error.retryable).toBe(retryable);
    expect(error.message).toContain("OpenRouter");
    expect(error.message).toContain(`HTTP ${status}`);
    expect(error.message).toContain(`(${kind})`);
  });

  it.each([
    ["context length exceeded", "context_limit", false],
    ["The upstream model is overloaded", "overloaded", true],
    ["Something odd happened", "unknown", false],
  ] as const)("classifies a code-less error %j by message -> %s", (message, kind, retryable) => {
    const error = classifyHttpError(undefined, JSON.stringify({ error: { message } }), undefined, "OpenRouter");
    expect(error.kind).toBe(kind);
    expect(error.status).toBeUndefined();
    expect(error.retryable).toBe(retryable);
    expect(error.message).toBe(`OpenRouter returned an error (${kind}): ${message}`);
  });

  it("reads Retry-After in seconds and as an HTTP-date", () => {
    const seconds = classifyHttpError(429, "slow down", new Headers({ "retry-after": "12" }), "P");
    expect(seconds.retryAfterMs).toBe(12_000);

    const date = new Date(Date.now() + 30_000).toUTCString();
    const dated = classifyHttpError(503, "busy", new Headers({ "Retry-After": date }), "P");
    expect(dated.retryAfterMs).toBeGreaterThanOrEqual(28_000);
    expect(dated.retryAfterMs).toBeLessThanOrEqual(30_000);

    expect(classifyHttpError(429, "", new Headers(), "P").retryAfterMs).toBeUndefined();
    expect(classifyHttpError(429, "", undefined, "P").retryAfterMs).toBeUndefined();
    expect(classifyHttpError(429, "", new Headers({ "retry-after": "later" }), "P").retryAfterMs).toBeUndefined();
  });

  it("quotes the provider's message from the common body shapes", () => {
    const detail = (body: string): string => classifyHttpError(400, body, undefined, "P").message;
    expect(detail(JSON.stringify({ error: { message: "bad field" } }))).toBe("P request failed with HTTP 400 (bad_request): bad field");
    expect(detail(JSON.stringify({ error: "plain error string" }))).toMatch(/: plain error string$/);
    expect(detail(JSON.stringify({ message: "top-level message" }))).toMatch(/: top-level message$/);
    expect(detail(JSON.stringify({ detail: "fastapi detail" }))).toMatch(/: fastapi detail$/);
    expect(detail("  plain\n\ttext   body ")).toMatch(/: plain text body$/);
    expect(detail("")).toBe("P request failed with HTTP 400 (bad_request)");
  });

  it("keeps the raw body, truncated to 2000 characters, and a short message", () => {
    const long = "x".repeat(10_000);
    const error = classifyHttpError(500, long, undefined, "P");
    expect(error.body).toBeDefined();
    expect(error.body!.length).toBeLessThanOrEqual(2000);
    expect(error.body!.endsWith("…")).toBe(true);
    expect(error.message.length).toBeLessThan(400);

    const short = classifyHttpError(500, '{"error":{"message":"boom"}}', undefined, "P");
    expect(short.body).toBe('{"error":{"message":"boom"}}');
    expect(classifyHttpError(500, "", undefined, "P").body).toBeUndefined();
  });

  it("does not split a surrogate pair when truncating", () => {
    const error = classifyHttpError(500, `${"a".repeat(1998)}🚀🚀🚀`, undefined, "P");
    const body = error.body!;
    const beforeEllipsis = body.charCodeAt(body.length - 2);
    expect(beforeEllipsis >= 0xd800 && beforeEllipsis <= 0xdbff).toBe(false);
  });

  it("redacts every echo of the state passed as `redact`", () => {
    const body = JSON.stringify({ error: { message: `Invalid state: ${STATE}`, metadata: { raw: STATE } } });
    const error = classifyHttpError(400, body, undefined, "P", STATE);
    expectNoState(error);
    expect(error.message).toContain("[redacted]");
    expect(error.body).toContain("[redacted]");
  });

  it("redacts the full extent of an echo found through a later, different occurrence in the state", () => {
    // The probe at offset 16 ("abc…p") first occurs in the state after "Y", so
    // its range starts at 16. The next probe matches the occurrence after "X",
    // whose range reaches back to offset 15; merging must take the earliest start.
    const p = "abcdefghijklmnop";
    const q = "QRSTUVWXYZ!@#$%^";
    const state = `Y${p}.X${p}${q}Z`;
    const body = `0123456789-=+*&X${p}${q}`;
    const error = classifyHttpError(500, body, undefined, "P", state);
    expect(error.body).toBe("0123456789-=+*&[redacted]");
    expect(error.message).toBe("P request failed with HTTP 500 (server): 0123456789-=+*&[redacted]");
  });

  it("replaces adjacent echoes of different parts of the state with a single marker", () => {
    const body = `${STATE.slice(0, 60)}${STATE.slice(100, 160)}`;
    const error = classifyHttpError(500, body, undefined, "P", STATE);
    expect(error.body).toBe("[redacted]");
  });

  it("classifies on the full body even when the echo is redacted from what is kept", () => {
    const body = JSON.stringify({
      error: { code: 400, message: "Provider returned error", metadata: { raw: `context_length_exceeded for ${STATE}` } },
    });
    const error = classifyHttpError(400, body, undefined, "P", STATE);
    expect(error.kind).toBe("context_limit");
    expectNoState(error);
  });

  it("property: status mapping invariants hold for any status and body", () => {
    fc.assert(
      fc.property(fc.integer({ min: 100, max: 599 }), fc.string({ maxLength: 3000 }), (status, body) => {
        const error = classifyHttpError(status, body, undefined, "P");
        expect(error.status).toBe(status);
        expect(error.retryable).toBe(["rate_limit", "server", "overloaded", "timeout", "network"].includes(error.kind));
        if (status === 413) expect(error.kind).toBe("context_limit");
        if (status === 401 || status === 403) expect(error.kind).toBe("auth");
        if (status === 429) expect(error.kind).toBe("rate_limit");
        if (status >= 500 && status !== 503 && status !== 529) expect(["server", "overloaded"]).toContain(error.kind);
        if (error.kind === "context_limit") expect([400, 413, 422]).toContain(status);
        expect((error.body ?? "").length).toBeLessThanOrEqual(2000);
      }),
    );
  });
});

// ---------------------------------------------------------------------------
// parseNativeResponse
// ---------------------------------------------------------------------------

describe("parseNativeResponse", () => {
  it("normalizes the documented wire response (snake_case usage)", () => {
    expect(parseNativeResponse(WIRE_RESPONSE, "OpenRouter")).toStrictEqual(NORMALIZED_WIRE_RESPONSE);
  });

  it("accepts camelCase usage", () => {
    const camel = { ...WIRE_RESPONSE, usage: { inputTokens: 10, outputTokens: 2, costUsd: 0.5 } };
    expect(parseNativeResponse(camel, "P").usage).toStrictEqual({ inputTokens: 10, outputTokens: 2, costUsd: 0.5 });

    const sdkStyle = { ...WIRE_RESPONSE, usage: { inputTokens: 10, outputTokens: 2, cost: 0.25 } };
    expect(parseNativeResponse(sdkStyle, "P").usage).toStrictEqual({ inputTokens: 10, outputTokens: 2, costUsd: 0.25 });
  });

  it("keeps partial usage (TypeSafe direct reports no cost) and drops malformed usage fields", () => {
    const noCost = { ...WIRE_RESPONSE, usage: { input_tokens: 5, output_tokens: 1 } };
    expect(parseNativeResponse(noCost, "P").usage).toStrictEqual({ inputTokens: 5, outputTokens: 1 });

    const zeroCost = { ...WIRE_RESPONSE, usage: { input_tokens: 5, output_tokens: 1, cost: 0 } };
    expect(parseNativeResponse(zeroCost, "P").usage).toStrictEqual({ inputTokens: 5, outputTokens: 1, costUsd: 0 });

    const malformed = { ...WIRE_RESPONSE, usage: { input_tokens: "12", output_tokens: -1, cost: Number.NaN } };
    expect(parseNativeResponse(malformed, "P")).not.toHaveProperty("usage");

    for (const usage of [undefined, null, "12 tokens", [1, 2]]) {
      expect(parseNativeResponse({ ...WIRE_RESPONSE, usage }, "P")).not.toHaveProperty("usage");
    }
  });

  it("keeps id and provider only when they are strings", () => {
    const parsed = parseNativeResponse({ model: "m", answers: {}, id: 42, provider: { name: "x" } }, "P");
    expect(parsed).toStrictEqual({ model: "m", answers: {} });
  });

  it("treats null optional answer fields as absent and drops unknown fields", () => {
    const parsed = parseNativeResponse(
      {
        model: "m",
        answers: {
          c: { type: "choice", choice: "a", probabilities: null, confidence: null, extra: true },
          s: { type: "score", score: 1, probabilities: null, legend: null, confidence: null },
          n: { type: "noul", noul: 0.3, probabilities: { yes: 0.3 }, confidence: 0.4 },
        },
      },
      "P",
    );
    expect(parsed.answers).toStrictEqual({
      c: { type: "choice", choice: "a" },
      s: { type: "score", score: 1 },
      n: { type: "noul", noul: 0.3 },
    });
  });

  it("keeps structured legend values (they mirror the criteria sent)", () => {
    const legend = { "0": { what: "low", examples: ["a"] }, "1": ["high", "very high"] };
    const parsed = parseNativeResponse({ model: "m", answers: { s: { type: "score", score: 0.5, legend } } }, "P");
    expect(parsed.answers["s"]).toStrictEqual({ type: "score", score: 0.5, legend });
  });

  it('preserves "__proto__" and other special keys as own data properties', () => {
    const text =
      '{"model":"m","answers":{"__proto__":{"type":"choice","choice":"__proto__",' +
      '"probabilities":{"__proto__":0.7,"constructor":0.2,"toString":0.1},"confidence":0.5},' +
      '"s":{"type":"score","score":0,"legend":{"__proto__":"odd"}}}}';
    const parsed = parseNativeResponse(JSON.parse(text), "P");

    expect(Object.keys(parsed.answers)).toStrictEqual(["__proto__", "s"]);
    const answer: unknown = Object.getOwnPropertyDescriptor(parsed.answers, "__proto__")?.value;
    expect(answer).toMatchObject({ type: "choice", choice: "__proto__", confidence: 0.5 });
    const probabilities = (answer as { probabilities: Record<string, number> }).probabilities;
    expect(Object.getPrototypeOf(probabilities)).toBe(Object.prototype);
    expect(Object.entries(probabilities)).toStrictEqual([
      ["__proto__", 0.7],
      ["constructor", 0.2],
      ["toString", 0.1],
    ]);
    const legend = (parsed.answers["s"] as { legend: Record<string, unknown> }).legend;
    expect(Object.entries(legend)).toStrictEqual([["__proto__", "odd"]]);
    expect(({} as Record<string, unknown>)["polluted"]).toBeUndefined();
  });

  it("property: arbitrary choice keys survive with their values and order", () => {
    fc.assert(
      fc.property(
        fc.uniqueArray(
          fc.tuple(
            fc.oneof(fc.constantFrom("__proto__", "constructor", "hasOwnProperty", "0", ""), fc.string()),
            fc.double({ min: 0, max: 1, noNaN: true }),
          ),
          { selector: ([key]) => key, minLength: 1, maxLength: 12 },
        ),
        (entries) => {
          const probabilities = Object.fromEntries(entries);
          const text = JSON.stringify({
            model: "m",
            answers: { q: { type: "choice", choice: entries[0]![0], probabilities } },
          });
          const decoded: unknown = JSON.parse(text);
          const parsed = parseNativeResponse(decoded, "P");
          const answer = parsed.answers["q"];
          expect(answer?.type).toBe("choice");
          const got = (answer as { probabilities: Record<string, number> }).probabilities;
          const expected = (decoded as { answers: { q: { probabilities: Record<string, number> } } }).answers.q.probabilities;
          expect(Object.entries(got)).toStrictEqual(Object.entries(expected));
        },
      ),
    );
  });

  const INVALID: Array<[string, unknown]> = [
    ["null", null],
    ["an array", []],
    ["a string", "ok"],
    ["a number", 42],
    ["missing model", { answers: {} }],
    ["empty model", { model: "", answers: {} }],
    ["numeric model", { model: 7, answers: {} }],
    ["missing answers", { model: "m" }],
    ["array answers", { model: "m", answers: [] }],
    ["null answers", { model: "m", answers: null }],
    ["string answer", { model: "m", answers: { q: "choice" } }],
    ["null answer", { model: "m", answers: { q: null } }],
    ["answer without type", { model: "m", answers: { q: {} } }],
    ["unknown answer type", { model: "m", answers: { q: { type: "multi" } } }],
    ["numeric answer type", { model: "m", answers: { q: { type: 1 } } }],
    ["noul without value", { model: "m", answers: { q: { type: "noul" } } }],
    ["noul as string", { model: "m", answers: { q: { type: "noul", noul: "0.5" } } }],
    ["noul NaN", { model: "m", answers: { q: { type: "noul", noul: Number.NaN } } }],
    ["noul Infinity", { model: "m", answers: { q: { type: "noul", noul: Number.POSITIVE_INFINITY } } }],
    ["choice without value", { model: "m", answers: { q: { type: "choice" } } }],
    ["numeric choice", { model: "m", answers: { q: { type: "choice", choice: 1 } } }],
    ["array probabilities", { model: "m", answers: { q: { type: "choice", choice: "a", probabilities: [0.5] } } }],
    ["string probability", { model: "m", answers: { q: { type: "choice", choice: "a", probabilities: { a: "0.5" } } } }],
    ["NaN probability", { model: "m", answers: { q: { type: "choice", choice: "a", probabilities: { a: Number.NaN } } } }],
    ["string confidence", { model: "m", answers: { q: { type: "choice", choice: "a", confidence: "high" } } }],
    ["score without value", { model: "m", answers: { q: { type: "score" } } }],
    ["score as string", { model: "m", answers: { q: { type: "score", score: "2" } } }],
    ["string legend", { model: "m", answers: { q: { type: "score", score: 1, legend: "levels" } } }],
    ["numeric legend value", { model: "m", answers: { q: { type: "score", score: 1, legend: { "0": 5 } } } }],
    ["null legend value", { model: "m", answers: { q: { type: "score", score: 1, legend: { "0": null } } } }],
    ["infinite score confidence", { model: "m", answers: { q: { type: "score", score: 1, confidence: Number.POSITIVE_INFINITY } } }],
  ];

  it.each(INVALID)("rejects %s as invalid_response (not retryable)", (_label, raw) => {
    let caught: unknown;
    try {
      parseNativeResponse(raw, "OpenRouter");
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(JevProviderError);
    const error = caught as JevProviderError;
    expect(error.kind).toBe("invalid_response");
    expect(error.retryable).toBe(false);
    expect(error.message).toMatch(/^OpenRouter returned an invalid decision response: /);
  });

  it("names the offending field without echoing string contents", () => {
    const secretish = "The quarterly revenue for Nightingale";
    expect(() => parseNativeResponse({ model: "m", answers: { q: { type: "noul", noul: secretish } } }, "P")).toThrow(
      'answers["q"].noul must be a finite number, got a string of length 37',
    );
    expect(() => parseNativeResponse({ model: "m", answers: { q: { type: "multi" } } }, "P")).toThrow(
      'answers["q"].type must be "choice", "score", or "noul", got "multi"',
    );
  });
});

// ---------------------------------------------------------------------------
// postDecision
// ---------------------------------------------------------------------------

describe("postDecision", () => {
  it("POSTs the JSON body with the configured headers and returns the normalized response", async () => {
    const fetchMock = fetchAlways(() => jsonResponse(WIRE_RESPONSE));
    const body = wireBody();
    const result = await postDecision(httpConfig(fetchMock), body);

    expect(result).toStrictEqual(NORMALIZED_WIRE_RESPONSE);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const { url, init } = callAt(fetchMock);
    expect(url).toBe("https://jev.example.test/decide");
    expect(init.method).toBe("POST");
    expect(init.headers).toStrictEqual({ "content-type": "application/json", authorization: "Bearer test" });
    expect(JSON.parse(String(init.body))).toStrictEqual(body);
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });

  it.each([
    [400, "maximum context length is 32768 tokens", "context_limit"],
    [400, "bad question", "bad_request"],
    [401, "Invalid token", "auth"],
    [402, "no credits", "payment"],
    [403, "forbidden", "auth"],
    [404, "no such model", "not_found"],
    [408, "timeout", "timeout"],
    [413, "too big", "context_limit"],
    [422, "state too large", "context_limit"],
    [422, "invalid", "bad_request"],
    [429, "rate limited", "rate_limit"],
    [500, "oops", "server"],
    [502, "bad gateway", "server"],
    [503, "unavailable", "overloaded"],
    [504, "gateway timeout", "server"],
    [529, "overloaded", "overloaded"],
    [418, "teapot", "unknown"],
  ] as const)("maps HTTP %d (%j) to %s", async (status, message, kind) => {
    const fetchMock = fetchAlways(() => jsonResponse({ error: { code: status, message } }, { status }));
    const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody()));
    expect(error.kind).toBe(kind);
    expect(error.status).toBe(status);
    expect(error.message).toContain("TestProvider");
    expect(error.message).toContain(message);
  });

  it("carries Retry-After (seconds and HTTP-date) onto the error", async () => {
    const seconds = fetchAlways(() => textResponse("slow down", { status: 429, headers: { "Retry-After": "3" } }));
    const secondsError = await providerErrorOf(postDecision(httpConfig(seconds), wireBody()));
    expect(secondsError.kind).toBe("rate_limit");
    expect(secondsError.retryAfterMs).toBe(3_000);

    const date = new Date(Date.now() + 30_000).toUTCString();
    const dated = fetchAlways(() => textResponse("busy", { status: 503, headers: { "Retry-After": date } }));
    const datedError = await providerErrorOf(postDecision(httpConfig(dated), wireBody()));
    expect(datedError.kind).toBe("overloaded");
    expect(datedError.retryAfterMs).toBeGreaterThanOrEqual(28_000);
    expect(datedError.retryAfterMs).toBeLessThanOrEqual(30_000);
  });

  describe("HTTP 200 with a top-level error object", () => {
    it("classifies by error.code like an HTTP error, including Retry-After", async () => {
      const fetchMock = fetchAlways(() =>
        jsonResponse({ error: { code: 429, message: "Rate limit exceeded upstream" } }, { headers: { "retry-after": "2" } }),
      );
      const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody()));
      expect(error.kind).toBe("rate_limit");
      expect(error.status).toBe(429);
      expect(error.retryable).toBe(true);
      expect(error.retryAfterMs).toBe(2_000);
      expect(error.message).toContain("Rate limit exceeded upstream");
    });

    it.each([
      [{ code: 400, message: "This endpoint's maximum context length is 32768 tokens" }, "context_limit", 400],
      [{ code: 502, message: "Provider returned error" }, "server", 502],
      [{ code: 400, message: "Invalid token" }, "bad_request", 400],
      [{ message: "Upstream model overloaded" }, "overloaded", undefined],
      [{ code: "E42", message: "weird" }, "unknown", undefined],
      [{ code: 7, message: "weird" }, "unknown", undefined],
    ] as const)("error %j -> %s", async (errorObject, kind, status) => {
      const fetchMock = fetchAlways(() => jsonResponse({ error: errorObject }));
      const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody()));
      expect(error.kind).toBe(kind);
      expect(error.status).toBe(status);
    });

    it("treats an error object as a failure even when answers are also present", async () => {
      const fetchMock = fetchAlways(() =>
        jsonResponse({ ...WIRE_RESPONSE, error: { code: 503, message: "degraded" } }),
      );
      const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody()));
      expect(error.kind).toBe("overloaded");
    });
  });

  it.each([
    ["non-JSON text", () => textResponse("<html>Welcome</html>")],
    ["an empty body", () => textResponse("")],
    ["JSON without answers", () => jsonResponse({ model: "m" })],
    ["a JSON array", () => jsonResponse([WIRE_RESPONSE])],
  ])("rejects a 2xx with %s as invalid_response (not retryable)", async (_label, factory) => {
    const error = await providerErrorOf(postDecision(httpConfig(fetchAlways(factory)), wireBody()));
    expect(error.kind).toBe("invalid_response");
    expect(error.retryable).toBe(false);
  });

  it("keeps the redacted body of a non-JSON 2xx", async () => {
    const fetchMock = fetchAlways(() => textResponse(`Gateway page. You sent: ${STATE}`));
    const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody()));
    expect(error.kind).toBe("invalid_response");
    expect(error.status).toBe(200);
    expect(error.body).toContain("Gateway page.");
    expectNoState(error);
  });

  describe("deadlines, aborts, and network failures", () => {
    it("times out with a retryable timeout error when fetch does not answer", async () => {
      const fetchMock = fetchHonoringSignal();
      const error = await providerErrorOf(postDecision(httpConfig(fetchMock, { timeoutMs: 25 }), wireBody()));
      expect(error.kind).toBe("timeout");
      expect(error.retryable).toBe(true);
      expect(error.status).toBeUndefined();
      expect(error.message).toBe("TestProvider request timed out after 25 ms");
      expect(callAt(fetchMock).init.signal?.aborted).toBe(true);
    });

    it("enforces the deadline even when fetch ignores its signal", async () => {
      const error = await providerErrorOf(postDecision(httpConfig(fetchIgnoringSignal(), { timeoutMs: 25 }), wireBody()));
      expect(error.kind).toBe("timeout");
    });

    it("enforces the deadline while the response body is still streaming", async () => {
      const stalled = fetchAlways(() => new Response(new ReadableStream<Uint8Array>({ start: () => undefined })));
      const error = await providerErrorOf(postDecision(httpConfig(stalled, { timeoutMs: 25 }), wireBody()));
      expect(error.kind).toBe("timeout");
    });

    it("throws JevAbortError without calling fetch when the signal is already aborted", async () => {
      const fetchMock = fetchAlways(() => jsonResponse(WIRE_RESPONSE));
      const controller = new AbortController();
      controller.abort(new Error("caller gave up"));
      const error = await rejectionOf(postDecision(httpConfig(fetchMock), wireBody(), controller.signal));
      expect(error).toBeInstanceOf(JevAbortError);
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it.each([
      ["honors", fetchHonoringSignal],
      ["ignores", fetchIgnoringSignal],
    ])("throws JevAbortError when the caller aborts mid-request (fetch %s its signal)", async (_label, makeFetch) => {
      const fetchMock = makeFetch();
      const controller = new AbortController();
      const reason = new Error("caller gave up");
      const pending = postDecision(httpConfig(fetchMock, { timeoutMs: 5_000 }), wireBody(), controller.signal);
      setTimeout(() => controller.abort(reason), 5);
      const error = await rejectionOf(pending);
      expect(error).toBeInstanceOf(JevAbortError);
      expect(error).not.toBeInstanceOf(JevProviderError);
      expect((error as JevAbortError).name).toBe("AbortError");
      expect((error as JevAbortError).cause).toBe(reason);
      expect(callAt(fetchMock).init.signal?.aborted).toBe(true);
    });

    it("maps a fetch rejection to a retryable network error that quotes only the error name and code", async () => {
      const failure = new TypeError(`fetch failed while sending ${STATE}`, { cause: { code: "ECONNREFUSED" } });
      const fetchMock = vi.fn<typeof fetch>(async () => {
        throw failure;
      });
      const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody()));
      expect(error.kind).toBe("network");
      expect(error.retryable).toBe(true);
      expect(error.cause).toBe(failure);
      expect(error.message).toBe("TestProvider request failed: network error (TypeError ECONNREFUSED)");
      expectNoState(error);
    });

    it("maps a synchronous fetch throw to a network error", async () => {
      const fetchMock = vi.fn<typeof fetch>(() => {
        throw new TypeError("Invalid URL");
      });
      const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody()));
      expect(error.kind).toBe("network");
    });

    it("removes its listener from the caller's signal once settled", async () => {
      const controller = new AbortController();
      const removeSpy = vi.spyOn(controller.signal, "removeEventListener");
      await postDecision(httpConfig(fetchAlways(() => jsonResponse(WIRE_RESPONSE))), wireBody(), controller.signal);
      expect(removeSpy).toHaveBeenCalledWith("abort", expect.any(Function));
      expect(() => controller.abort()).not.toThrow();
    });

    it.each([0, -1, Number.NaN, Number.POSITIVE_INFINITY, 2 ** 31])(
      "rejects timeoutMs %d with JevValidationError before fetching",
      async (timeoutMs) => {
        const fetchMock = fetchAlways(() => jsonResponse(WIRE_RESPONSE));
        const error = await rejectionOf(postDecision(httpConfig(fetchMock, { timeoutMs }), wireBody()));
        expect(error).toBeInstanceOf(JevValidationError);
        expect(fetchMock).not.toHaveBeenCalled();
      },
    );
  });

  describe("never leaks request state into errors", () => {
    const echoes: Array<[string, () => Response]> = [
      ["a JSON message quoting the state", () => jsonResponse({ error: { message: `Invalid state: ${STATE}` } }, { status: 400 })],
      ["a plain-text echo", () => textResponse(`bad request: ${STATE}`, { status: 400 })],
      ["a partial echo", () => textResponse(`rejected input starting with ${STATE.slice(0, 120)}...`, { status: 422 })],
      [
        "an ASCII-escaped JSON echo (Python json.dumps style)",
        () =>
          textResponse(
            JSON.stringify({ detail: `bad state ${STATE}` }).replace(
              /[\u0080-\uffff]/g,
              (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`,
            ),
            { status: 400 },
          ),
      ],
      ["a 200 error object quoting the state", () => jsonResponse({ error: { code: 502, message: `upstream said ${STATE}` } })],
      ["a server error echoing it twice", () => textResponse(`${STATE} :: ${STATE}`, { status: 500 })],
    ];

    it.each(echoes)("%s", async (_label, factory) => {
      const error = await providerErrorOf(postDecision(httpConfig(fetchAlways(factory)), wireBody()));
      expectNoState(error);
      expect(`${error.message}${error.body ?? ""}`).toContain("[redacted]");
    });

    it("redacts a very large echoed state and still bounds the body", async () => {
      const bigState = Array.from({ length: 1500 }, (_, i) => `Sentence ${i} of the confidential record.`).join(" ");
      const fetchMock = fetchAlways(() => jsonResponse({ error: { message: `too large: ${bigState}` } }, { status: 400 }));
      const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody(bigState)));
      expect(error.kind).toBe("bad_request");
      expect((error.body ?? "").length).toBeLessThanOrEqual(2000);
      expectNoState(error, bigState);
    });

    it("withholds the body when the redaction marker itself would contain the state", async () => {
      const fetchMock = fetchAlways(() => jsonResponse({ error: { message: "bad value e" } }, { status: 400 }));
      const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody("e")));
      expect(error.body).toBeUndefined();
      expect(error.message).toBe("TestProvider request failed with HTTP 400 (bad_request)");
    });

    it("leaves bodies that do not echo the state untouched", async () => {
      const text = JSON.stringify({ error: { code: 400, message: "questions.decision.criteria must have 2 options" } });
      const fetchMock = fetchAlways(() => textResponse(text, { status: 400 }));
      const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody()));
      expect(error.body).toBe(text);
      expect(error.message).toBe(
        "TestProvider request failed with HTTP 400 (bad_request): questions.decision.criteria must have 2 options",
      );
    });

    it("property: an echoed state never survives in message or body", async () => {
      await fc.assert(
        fc.asyncProperty(
          fc.string({ minLength: 20, maxLength: 400, unit: "binary" }),
          fc.constantFrom("raw", "json", "ascii-json"),
          fc.string({ maxLength: 50 }),
          fc.constantFrom(400, 404, 422, 429, 500, 503),
          async (state, mode, prefix, status) => {
            const json = JSON.stringify({ error: { message: `${prefix}${state}` } });
            const bodyText =
              mode === "raw"
                ? `${prefix}${state}`
                : mode === "json"
                  ? json
                  : json.replace(/[\u0080-\uffff]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`);
            const fetchMock = fetchAlways(() => textResponse(bodyText, { status }));
            const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody(state)));
            const escaped = JSON.stringify(state).slice(1, -1);
            for (const text of [error.message, error.body ?? ""]) {
              expect(text.includes(state)).toBe(false);
              expect(text.includes(escaped)).toBe(false);
            }
          },
        ),
        { numRuns: 200 },
      );
    });
  });
});

// ---------------------------------------------------------------------------
// OpenRouterJevTransport
// ---------------------------------------------------------------------------

const CATALOG = {
  data: [
    {
      id: "typesafe/jev-1.13",
      canonical_slug: "typesafe/jev-1.13-20260917",
      context_length: 32_768,
      architecture: { output_modalities: ["decisions"] },
    },
    {
      id: "~typesafe/jev-latest",
      canonical_slug: "typesafe/jev-1.14-20261001",
      context_length: 65_536,
      alias_target: { slug: "typesafe/jev-1.14" },
    },
    { id: "acme/decider-2", canonical_slug: "acme/decider-2-20260101", context_length: 16_000.9 },
    { id: "typesafe/jev-broken", context_length: 0 },
    { id: "acme/broken", context_length: "big" },
    "garbage",
    { id: 5, context_length: 1 },
    { canonical_slug: "nameless", context_length: 1 },
  ],
};

describe("OpenRouterJevTransport", () => {
  beforeEach(() => {
    vi.stubEnv("OPENROUTER_API_KEY", undefined);
  });

  function transport(fetchMock: FetchMock, options: Record<string, unknown> = {}): OpenRouterJevTransport {
    return new OpenRouterJevTransport({ apiKey: "sk-or-test", fetch: fetchMock, ...options });
  }

  describe("construction", () => {
    it("throws JevValidationError when no API key is available", () => {
      const fetchMock = fetchAlways(() => jsonResponse(WIRE_RESPONSE));
      expect(() => new OpenRouterJevTransport({ fetch: fetchMock })).toThrow(JevValidationError);
      expect(() => new OpenRouterJevTransport({ fetch: fetchMock })).toThrow(/OPENROUTER_API_KEY/);
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it("throws for an empty or whitespace environment key", () => {
      vi.stubEnv("OPENROUTER_API_KEY", "   ");
      expect(() => new OpenRouterJevTransport({ fetch: fetchAlways(() => jsonResponse({})) })).toThrow(JevValidationError);
    });

    it("reads OPENROUTER_API_KEY from the environment; an explicit key wins", async () => {
      vi.stubEnv("OPENROUTER_API_KEY", "sk-from-env");
      const fromEnv = fetchAlways(() => jsonResponse(WIRE_RESPONSE));
      await new OpenRouterJevTransport({ fetch: fromEnv }).decide(request());
      expect(callAt(fromEnv).headers.get("authorization")).toBe("Bearer sk-from-env");

      const explicit = fetchAlways(() => jsonResponse(WIRE_RESPONSE));
      await new OpenRouterJevTransport({ fetch: explicit, apiKey: "sk-explicit" }).decide(request());
      expect(callAt(explicit).headers.get("authorization")).toBe("Bearer sk-explicit");
    });

    it("exposes its name and default model", () => {
      const fetchMock = fetchAlways(() => jsonResponse({}));
      expect(OPENROUTER_DEFAULT_MODEL).toBe("typesafe/jev-1.13");
      expect(transport(fetchMock).name).toBe("openrouter");
      expect(transport(fetchMock).defaultModel).toBe(OPENROUTER_DEFAULT_MODEL);
      expect(transport(fetchMock, { defaultModel: "typesafe/jev-2" }).defaultModel).toBe("typesafe/jev-2");
    });

    it.each<[string, Record<string, unknown>]>([
      ["empty apiKey", { apiKey: "" }],
      ["whitespace apiKey", { apiKey: "  " }],
      ["numeric apiKey", { apiKey: 123 }],
      ["unparseable baseUrl", { baseUrl: "not a url" }],
      ["non-http baseUrl", { baseUrl: "ftp://openrouter.ai" }],
      ["baseUrl with a query", { baseUrl: "https://openrouter.ai/?x=1" }],
      ["baseUrl with credentials", { baseUrl: "https://user:pass@openrouter.ai" }],
      ["zero timeoutMs", { timeoutMs: 0 }],
      ["negative timeoutMs", { timeoutMs: -1 }],
      ["NaN timeoutMs", { timeoutMs: Number.NaN }],
      ["infinite timeoutMs", { timeoutMs: Number.POSITIVE_INFINITY }],
      ["timeoutMs beyond the timer range", { timeoutMs: 2 ** 31 }],
      ["zero metadataTimeoutMs", { metadataTimeoutMs: 0 }],
      ["zero contextWindow", { contextWindow: 0 }],
      ["fractional contextWindow", { contextWindow: 1.5 }],
      ["negative contextWindow", { contextWindow: -32_000 }],
      ["invalid header name", { headers: { "bad name": "x" } }],
      ["non-string header value", { headers: { "x-count": 1 } }],
      ["array headers", { headers: [["x", "y"]] }],
      ["non-Latin-1 appName", { appName: "决策" }],
      ["non-boolean resolveContextWindow", { resolveContextWindow: "yes" }],
      ["empty defaultModel", { defaultModel: "" }],
      ["numeric sessionId", { sessionId: 5 }],
      ["non-function fetch", { fetch: "fetch" }],
    ])("rejects %s with JevValidationError", (_label, options) => {
      expect(() => new OpenRouterJevTransport({ apiKey: "sk-or-test", fetch: fetchAlways(() => jsonResponse({})), ...options })).toThrow(
        JevValidationError,
      );
    });

    it("does not quote an invalid API key in the validation error", () => {
      let caught: unknown;
      try {
        new OpenRouterJevTransport({ apiKey: "sk-or-secret\nvalue", fetch: fetchAlways(() => jsonResponse({})) });
      } catch (error) {
        caught = error;
      }
      expect(caught).toBeInstanceOf(JevValidationError);
      expect((caught as Error).message).not.toContain("sk-or-secret");
    });

    it("uses the global fetch when none is injected, looked up at request time", async () => {
      const fetchMock = fetchAlways(() => jsonResponse(WIRE_RESPONSE));
      const instance = new OpenRouterJevTransport({ apiKey: "sk-or-test" });
      vi.stubGlobal("fetch", fetchMock);
      try {
        await instance.decide(request());
      } finally {
        vi.unstubAllGlobals();
      }
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });
  });

  describe("decide", () => {
    it("POSTs { model, state, questions } to the Decisions API with auth and JSON headers", async () => {
      const fetchMock = fetchAlways(() => jsonResponse(WIRE_RESPONSE));
      const controller = new AbortController();
      const result = await transport(fetchMock).decide(request({ signal: controller.signal }));

      expect(result).toStrictEqual(NORMALIZED_WIRE_RESPONSE);
      const { url, init, headers } = callAt(fetchMock);
      expect(url).toBe("https://openrouter.ai/api/alpha/decisions");
      expect(init.method).toBe("POST");
      expect(headers.get("authorization")).toBe("Bearer sk-or-test");
      expect(headers.get("content-type")).toBe("application/json");
      expect(headers.get("accept")).toBe("application/json");
      expect(headers.has("x-title")).toBe(false);
      expect(headers.has("http-referer")).toBe(false);

      const body = sentBody(fetchMock);
      expect(body).toStrictEqual({ model: "typesafe/jev-1.13", state: STATE, questions: { decision: QUESTION } });
      expect(String(init.body)).not.toContain("signal");
    });

    it("sends attribution headers, session_id, user, and extra headers without letting them replace auth", async () => {
      const fetchMock = fetchAlways(() => jsonResponse(WIRE_RESPONSE));
      await transport(fetchMock, {
        appName: "Ticket Router",
        appUrl: "https://tickets.example.com",
        sessionId: "session-123",
        user: "user-456",
        headers: { "X-Custom": "1", authorization: "Bearer attacker", "Content-Type": "text/plain" },
      }).decide(request());

      const { headers } = callAt(fetchMock);
      expect(headers.get("x-openrouter-title")).toBe("Ticket Router");
      expect(headers.get("x-title")).toBe("Ticket Router");
      expect(headers.get("http-referer")).toBe("https://tickets.example.com");
      expect(headers.get("x-custom")).toBe("1");
      expect(headers.get("authorization")).toBe("Bearer sk-or-test");
      expect(headers.get("content-type")).toBe("application/json");

      const body = sentBody(fetchMock);
      expect(body["session_id"]).toBe("session-123");
      expect(body["user"]).toBe("user-456");
    });

    it("tolerates a trailing slash and a path prefix in baseUrl", async () => {
      const fetchMock = fetchAlways(() => jsonResponse(WIRE_RESPONSE));
      await transport(fetchMock, { baseUrl: "https://proxy.example.com/openrouter///" }).decide(request());
      expect(callAt(fetchMock).url).toBe("https://proxy.example.com/openrouter/api/alpha/decisions");
    });

    it("encodes caller choice keys such as __proto__ into the wire body", async () => {
      const criteria: Record<string, string> = Object.fromEntries([
        ["__proto__", "Prototype option."],
        ["constructor", "Constructor option."],
      ]);
      const question: ChoiceQuestion = { type: "choice", instructions: "Pick.", criteria };
      const fetchMock = fetchAlways(() => jsonResponse(WIRE_RESPONSE));
      await transport(fetchMock).decide(request({ questions: Object.fromEntries([["__proto__", question]]) }));

      const text = String(callAt(fetchMock).init.body);
      const questions = (JSON.parse(text) as { questions: Record<string, unknown> }).questions;
      expect(Object.keys(questions)).toStrictEqual(["__proto__"]);
      const sent = Object.getOwnPropertyDescriptor(questions, "__proto__")?.value as { criteria: Record<string, string> };
      expect(Object.entries(sent.criteria)).toStrictEqual([
        ["__proto__", "Prototype option."],
        ["constructor", "Constructor option."],
      ]);
    });

    it("sends only wire fields for each question type", async () => {
      const fetchMock = fetchAlways(() => jsonResponse(WIRE_RESPONSE));
      await transport(fetchMock).decide(
        request({
          questions: {
            n: { type: "noul", instructions: { goal: "Is it?" } },
            nc: { type: "noul", instructions: "Is it?", criteria: { true: "yes", false: "no" } },
            s: { type: "score", instructions: "Rate.", criteria: ["low", { what: "high" }] },
          },
        }),
      );
      expect(sentBody(fetchMock)["questions"]).toStrictEqual({
        n: { type: "noul", instructions: { goal: "Is it?" } },
        nc: { type: "noul", instructions: "Is it?", criteria: { true: "yes", false: "no" } },
        s: { type: "score", instructions: "Rate.", criteria: ["low", { what: "high" }] },
      });
    });

    it.each<[string, unknown]>([
      ["an empty model", { model: "", state: "x", questions: {} }],
      ["a non-string state", { model: "m", state: { text: "x" }, questions: {} }],
      ["non-object questions", { model: "m", state: "x", questions: "q" }],
      ["an unknown question type", { model: "m", state: "x", questions: { q: { type: "rank", instructions: "?" } } }],
      ["a null question", { model: "m", state: "x", questions: { q: null } }],
      ["a missing request", undefined],
    ])("rejects (asynchronously) %s with JevValidationError before fetching", async (_label, bad) => {
      const fetchMock = fetchAlways(() => jsonResponse(WIRE_RESPONSE));
      const pending = transport(fetchMock).decide(bad as NativeJevRequest);
      expect(pending).toBeInstanceOf(Promise);
      expect(await rejectionOf(pending)).toBeInstanceOf(JevValidationError);
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it("honors the request signal, before and during the request", async () => {
      const early = new AbortController();
      early.abort();
      const earlyFetch = fetchHonoringSignal();
      expect(await rejectionOf(transport(earlyFetch).decide(request({ signal: early.signal })))).toBeInstanceOf(JevAbortError);
      expect(earlyFetch).not.toHaveBeenCalled();

      const late = new AbortController();
      const lateFetch = fetchHonoringSignal();
      const pending = transport(lateFetch).decide(request({ signal: late.signal }));
      await vi.waitFor(() => expect(lateFetch).toHaveBeenCalledTimes(1));
      late.abort();
      expect(await rejectionOf(pending)).toBeInstanceOf(JevAbortError);
      expect(callAt(lateFetch).init.signal?.aborted).toBe(true);
    });

    it("honors timeoutMs", async () => {
      const error = await providerErrorOf(transport(fetchIgnoringSignal(), { timeoutMs: 20 }).decide(request()));
      expect(error.kind).toBe("timeout");
      expect(error.message).toBe("OpenRouter request timed out after 20 ms");
    });

    it("surfaces provider errors under the OpenRouter name without the state", async () => {
      const fetchMock = fetchAlways(() =>
        jsonResponse({ error: { code: 400, message: `maximum context length exceeded for: ${STATE}` } }, { status: 400 }),
      );
      const error = await providerErrorOf(transport(fetchMock).decide(request()));
      expect(error.kind).toBe("context_limit");
      expect(error.message).toMatch(/^OpenRouter request failed with HTTP 400 \(context_limit\): /);
      expectNoState(error);
    });
  });

  describe("contextWindow", () => {
    const MODELS_URL = "https://openrouter.ai/api/v1/models?output_modalities=decisions";

    it("returns an explicit contextWindow without fetching", async () => {
      const fetchMock = fetchAlways(() => jsonResponse(CATALOG));
      expect(await transport(fetchMock, { contextWindow: 12_345 }).contextWindow("typesafe/jev-1.13")).toBe(12_345);
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it("uses the static fallback without fetching when resolution is disabled", async () => {
      const fetchMock = fetchAlways(() => jsonResponse(CATALOG));
      const instance = transport(fetchMock, { resolveContextWindow: false });
      expect(await instance.contextWindow("typesafe/jev-1.13")).toBe(32_000);
      expect(await instance.contextWindow("openai/gpt-5")).toBeUndefined();
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it("GETs the decisions model catalog with the API key", async () => {
      const fetchMock = fetchAlways(() => jsonResponse(CATALOG));
      await transport(fetchMock).contextWindow("typesafe/jev-1.13");
      const { url, init, headers } = callAt(fetchMock);
      expect(url).toBe(MODELS_URL);
      expect(init.method).toBe("GET");
      expect(init.body).toBeUndefined();
      expect(headers.get("authorization")).toBe("Bearer sk-or-test");
    });

    it.each([
      ["an exact id", "typesafe/jev-1.13", 32_768],
      ["a canonical build slug", "typesafe/jev-1.13-20260917", 32_768],
      ["an alias id with its ~", "~typesafe/jev-latest", 65_536],
      ["an alias id without its ~", "typesafe/jev-latest", 65_536],
      ["a non-Jev model (floored)", "acme/decider-2", 16_000],
      ["a non-Jev canonical slug", "acme/decider-2-20260101", 16_000],
    ])("resolves %s", async (_label, model, expected) => {
      expect(await transport(fetchAlways(() => jsonResponse(CATALOG))).contextWindow(model)).toBe(expected);
    });

    it.each([
      ["an unknown Jev model", "typesafe/jev-9.9", 32_000],
      ["a Jev model in any case", "Vendor/JEV-Experimental", 32_000],
      ["a Jev entry with an invalid context_length", "typesafe/jev-broken", 32_000],
      ["an unknown non-Jev model", "openai/gpt-5", undefined],
      ["a non-Jev entry with an invalid context_length", "acme/broken", undefined],
    ])("falls back for %s", async (_label, model, expected) => {
      expect(await transport(fetchAlways(() => jsonResponse(CATALOG))).contextWindow(model)).toBe(expected);
    });

    it.each<[string, () => FetchMock]>([
      ["an HTTP error", () => fetchAlways(() => textResponse("down", { status: 500 }))],
      ["invalid JSON", () => fetchAlways(() => textResponse("<html>"))],
      ["a body without a data array", () => fetchAlways(() => jsonResponse({ models: [] }))],
      [
        "a rejected fetch",
        () =>
          vi.fn<typeof fetch>(async () => {
            throw new TypeError("fetch failed");
          }),
      ],
      [
        "a synchronously throwing fetch",
        () =>
          vi.fn<typeof fetch>(() => {
            throw new TypeError("boom");
          }),
      ],
      ["a hung request", fetchIgnoringSignal],
    ])("falls back (never throws) after %s", async (_label, makeFetch) => {
      const instance = transport(makeFetch(), { metadataTimeoutMs: 20 });
      await expect(instance.contextWindow("typesafe/jev-1.13")).resolves.toBe(32_000);
      await expect(instance.contextWindow("openai/gpt-5")).resolves.toBeUndefined();
    });

    it("fetches the catalog once for repeated and concurrent calls on one instance", async () => {
      const fetchMock = fetchAlways(() => jsonResponse(CATALOG));
      const instance = transport(fetchMock);
      const concurrent = await Promise.all([
        instance.contextWindow("typesafe/jev-1.13"),
        instance.contextWindow("~typesafe/jev-latest"),
      ]);
      expect(concurrent).toStrictEqual([32_768, 65_536]);
      expect(await instance.contextWindow("acme/decider-2")).toBe(16_000);
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });

    it("shares the catalog across instances with the same fetch and base URL only", async () => {
      const fetchMock = fetchAlways(() => jsonResponse(CATALOG));
      await transport(fetchMock).contextWindow("typesafe/jev-1.13");
      await transport(fetchMock, { apiKey: "sk-other" }).contextWindow("typesafe/jev-1.13");
      expect(fetchMock).toHaveBeenCalledTimes(1);

      await transport(fetchMock, { baseUrl: "https://proxy.example.com" }).contextWindow("typesafe/jev-1.13");
      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(callAt(fetchMock, 1).url).toBe("https://proxy.example.com/api/v1/models?output_modalities=decisions");

      const otherFetch = fetchAlways(() => jsonResponse(CATALOG));
      await transport(otherFetch).contextWindow("typesafe/jev-1.13");
      expect(otherFetch).toHaveBeenCalledTimes(1);
    });

    it("remembers a failure briefly, then retries the catalog", async () => {
      let now = 1_000_000;
      vi.spyOn(Date, "now").mockImplementation(() => now);
      const fetchMock = fetchSequence(
        () => textResponse("down", { status: 503 }),
        () => jsonResponse(CATALOG),
      );
      const instance = transport(fetchMock);

      expect(await instance.contextWindow("~typesafe/jev-latest")).toBe(32_000);
      expect(await instance.contextWindow("~typesafe/jev-latest")).toBe(32_000);
      expect(fetchMock).toHaveBeenCalledTimes(1);

      now += 61_000;
      expect(await instance.contextWindow("~typesafe/jev-latest")).toBe(65_536);
      expect(fetchMock).toHaveBeenCalledTimes(2);
    });

    it("refreshes a successful catalog after an hour", async () => {
      let now = 5_000_000;
      vi.spyOn(Date, "now").mockImplementation(() => now);
      const updated = { data: [{ id: "typesafe/jev-1.13", context_length: 128_000 }] };
      const fetchMock = fetchSequence(
        () => jsonResponse(CATALOG),
        () => jsonResponse(updated),
      );
      const instance = transport(fetchMock);

      expect(await instance.contextWindow("typesafe/jev-1.13")).toBe(32_768);
      now += 59 * 60_000;
      expect(await instance.contextWindow("typesafe/jev-1.13")).toBe(32_768);
      now += 2 * 60_000;
      expect(await instance.contextWindow("typesafe/jev-1.13")).toBe(128_000);
      expect(fetchMock).toHaveBeenCalledTimes(2);
    });
  });
});

// ---------------------------------------------------------------------------
// DirectJevTransport
// ---------------------------------------------------------------------------

describe("DirectJevTransport", () => {
  beforeEach(() => {
    vi.stubEnv("TYPESAFE_API_KEY", undefined);
  });

  function transport(fetchMock: FetchMock, options: Record<string, unknown> = {}): DirectJevTransport {
    return new DirectJevTransport({ apiKey: "ts-test", fetch: fetchMock, ...options });
  }

  it("throws JevValidationError when no API key is available", () => {
    const fetchMock = fetchAlways(() => jsonResponse(WIRE_RESPONSE));
    expect(() => new DirectJevTransport({ fetch: fetchMock })).toThrow(JevValidationError);
    expect(() => new DirectJevTransport({ fetch: fetchMock })).toThrow(/TYPESAFE_API_KEY/);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("reads TYPESAFE_API_KEY from the environment", async () => {
    vi.stubEnv("TYPESAFE_API_KEY", "ts-from-env");
    const fetchMock = fetchAlways(() => jsonResponse(WIRE_RESPONSE));
    await new DirectJevTransport({ fetch: fetchMock }).decide(request({ model: "jev-latest" }));
    expect(callAt(fetchMock).headers.get("authorization")).toBe("Bearer ts-from-env");
  });

  it("exposes its name, default model, and the documented 32K context window", () => {
    const instance = transport(fetchAlways(() => jsonResponse({})));
    expect(instance.name).toBe("direct");
    expect(DIRECT_DEFAULT_MODEL).toBe("jev-latest");
    expect(instance.defaultModel).toBe("jev-latest");
    expect(instance.contextWindow("jev-latest")).toBe(32_000);
    expect(transport(fetchAlways(() => jsonResponse({})), { contextWindow: 16_000 }).contextWindow("x")).toBe(16_000);
  });

  it("POSTs only { model, state, questions } to the TypeSafe endpoint", async () => {
    const fetchMock = fetchAlways(() => jsonResponse({ ...WIRE_RESPONSE, usage: { input_tokens: 476, output_tokens: 70 } }));
    const result = await transport(fetchMock).decide(request({ model: "jev-latest", signal: new AbortController().signal }));

    expect(result.usage).toStrictEqual({ inputTokens: 476, outputTokens: 70 });
    const { url, init, headers } = callAt(fetchMock);
    expect(url).toBe("https://api.typesafe.ai/v1/systemone");
    expect(init.method).toBe("POST");
    expect(headers.get("authorization")).toBe("Bearer ts-test");
    expect(headers.get("content-type")).toBe("application/json");
    expect(sentBody(fetchMock)).toStrictEqual({ model: "jev-latest", state: STATE, questions: { decision: QUESTION } });
  });

  it("joins a custom baseUrl and path, adding a missing leading slash", async () => {
    const fetchMock = fetchAlways(() => jsonResponse(WIRE_RESPONSE));
    await transport(fetchMock, { baseUrl: "https://jev.internal/", path: "v2/decide" }).decide(request());
    expect(callAt(fetchMock).url).toBe("https://jev.internal/v2/decide");
  });

  it.each<[string, Record<string, unknown>]>([
    ["a path with a query", { path: "/v1/systemone?x=1" }],
    ["a numeric path", { path: 1 }],
    ["an invalid baseUrl", { baseUrl: "api.typesafe.ai" }],
    ["a zero contextWindow", { contextWindow: 0 }],
    ["a bad timeoutMs", { timeoutMs: -5 }],
    ["an empty defaultModel", { defaultModel: " " }],
    ["an empty apiKey", { apiKey: "" }],
  ])("rejects %s with JevValidationError", (_label, options) => {
    expect(() => new DirectJevTransport({ apiKey: "ts-test", fetch: fetchAlways(() => jsonResponse({})), ...options })).toThrow(
      JevValidationError,
    );
  });

  it("classifies provider errors under the TypeSafe name", async () => {
    const fetchMock = fetchAlways(() =>
      jsonResponse({ detail: "Too many requests" }, { status: 429, headers: { "Retry-After": "4" } }),
    );
    const error = await providerErrorOf(transport(fetchMock).decide(request()));
    expect(error.kind).toBe("rate_limit");
    expect(error.retryAfterMs).toBe(4_000);
    expect(error.message).toBe("TypeSafe request failed with HTTP 429 (rate_limit): Too many requests");
  });

  it("detects a context-limit 422 and keeps the state out of the error", async () => {
    const fetchMock = fetchAlways(() =>
      jsonResponse({ detail: [{ msg: "state exceeds the maximum of 32000 tokens", input: STATE }] }, { status: 422 }),
    );
    const error = await providerErrorOf(transport(fetchMock).decide(request()));
    expect(error.kind).toBe("context_limit");
    expectNoState(error);
  });

  it("honors timeoutMs and the request signal", async () => {
    const timeout = await providerErrorOf(transport(fetchIgnoringSignal(), { timeoutMs: 20 }).decide(request()));
    expect(timeout.kind).toBe("timeout");

    const controller = new AbortController();
    const fetchMock = fetchHonoringSignal();
    const pending = transport(fetchMock).decide(request({ signal: controller.signal }));
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    controller.abort();
    expect(await rejectionOf(pending)).toBeInstanceOf(JevAbortError);
  });
});
