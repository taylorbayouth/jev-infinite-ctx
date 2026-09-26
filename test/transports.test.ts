import path from "node:path";
import { fileURLToPath } from "node:url";
import { inspect } from "node:util";
import ts from "typescript";
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import fc from "fast-check";
import {
  classifyHttpError,
  isContextLimitMessage,
  mergeHeaders,
  parseNativeResponse,
  parseRetryAfter,
  postDecision,
  type HttpDecisionConfig,
} from "../src/transports/http.js";
import { OPENROUTER_DEFAULT_MODEL, OpenRouterJevTransport } from "../src/transports/openrouter.js";
import { DIRECT_DEFAULT_MODEL, DirectJevTransport } from "../src/transports/direct.js";
import { decide } from "../src/decide.js";
import { MIN_STATE_TOKENS } from "../src/defaults.js";
import {
  JevAbortError,
  JevChunkFailedError,
  JevContextBudgetError,
  JevProviderError,
  JevValidationError,
  type ProviderErrorKind,
} from "../src/errors.js";
import type { ChoiceQuestion, JsonObject, NativeJevRequest, NoulQuestion } from "../src/types.js";

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

/** Shortest run shared with the state that withholds `body` (ECHO_RUN in http.ts). */
const GUARANTEED_RUN = 8;

/** What `body` holds in place of a provider body that echoes the request state. */
const WITHHELD = "[withheld: provider error body overlaps the request state]";

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
    // F17: a size stated as a count, without "maximum" or a context noun.
    "state exceeds 32000 tokens",
    "input exceeded 128,000 characters",
    "Request exceeds 40000 tokens",
    "String should have at most 128000 characters",
    "ensure this value has at most 128000 characters",
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
    // F17 patterns must not catch output-token limits, rate limits, or small field limits.
    "max_tokens exceeds 4096 tokens",
    "max_completion_tokens should be at most 4096 tokens",
    "Rate limit exceeded: 100000 tokens per minute",
    "Usage exceeded 40000 tokens/min",
    "String should have at most 64 characters",
    "criteria keys exceed 64 characters",
    "exceeds 12 tokens",
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
    // F17: over-length wordings that name a count rather than "maximum" or "context".
    [400, openRouterError("state exceeds 32000 tokens"), "context_limit", false],
    [
      422,
      JSON.stringify({
        detail: [
          {
            type: "string_too_long",
            loc: ["body", "state"],
            msg: "String should have at most 128000 characters",
            ctx: { max_length: 128000 },
          },
        ],
      }),
      "context_limit",
      false,
    ],
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

  it("withholds a body that echoes the state passed as `state`", () => {
    const body = JSON.stringify({ error: { message: `Invalid state: ${STATE}`, metadata: { raw: STATE } } });
    const error = classifyHttpError(400, body, undefined, "P", STATE);
    expectNoState(error);
    // With a state, the message never quotes provider text (F01). The kind
    // reads the body, echo included, so the echo's "context window" counts.
    expect(error.message).toBe("P request failed with HTTP 400 (context_limit)");
    expect(error.body).toBe(WITHHELD);
  });

  it("classifies on the full body even when the body is withheld", () => {
    const body = JSON.stringify({
      error: { code: 400, message: "Provider returned error", metadata: { raw: `context_length_exceeded for ${STATE}` } },
    });
    const error = classifyHttpError(400, body, undefined, "P", STATE);
    expect(error.kind).toBe("context_limit");
    expect(error.body).toBe(WITHHELD);
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
        if (error.kind === "overloaded") expect(status).toBeGreaterThanOrEqual(500);
        if (error.kind === "context_limit") expect([400, 413, 422]).toContain(status);
        expect((error.body ?? "").length).toBeLessThanOrEqual(2000);
      }),
    );
  });
});

// ---------------------------------------------------------------------------
// Error bodies for a request with state
// ---------------------------------------------------------------------------

describe("error bodies for a request with state", () => {
  it("withholds the whole body when an 8-character run of it occurs in the state, whitespace collapsed", () => {
    const memo = "Board memo. The Lisbon acquisition closes on 14 March,\tpending the regulator's review.";
    // Every 8-character run of the quote spans the tab, which the provider re-wrapped as two spaces.
    const body = JSON.stringify({ error: { message: "Invalid input near 'March,  pending' in state" } });
    const error = classifyHttpError(400, body, undefined, "P", memo);
    expect(error.body).toBe(WITHHELD);
    expect(error.message).toBe("P request failed with HTTP 400 (bad_request)");
  });

  it("keeps a body that shares no 8-character run with the state verbatim, truncated to 2000 characters", () => {
    // The echo starts 1,000 characters past the cut, beyond the 128 that are also compared.
    const long = classifyHttpError(500, `${"x".repeat(3000)}${STATE}`, undefined, "P", STATE);
    expect(long.body).toBe(`${"x".repeat(1999)}…`);
    expect(long.message).toBe("P request failed with HTTP 500 (server)");

    const short = JSON.stringify({ error: { code: 400, message: "questions.decision.criteria must have 2 options" } });
    expect(classifyHttpError(400, short, undefined, "P", STATE).body).toBe(short);
  });

  it("withholds a \\u-escaped CJK echo, decoding an escaped surrogate pair unit by unit", () => {
    const state = "診療記録：𠮷田花子さん、東京都港区在住。二型糖尿病の経過観察中。";
    // Python's json.dumps default: every non-ASCII UTF-16 unit as \uXXXX.
    const asciiJson = (value: unknown): string =>
      JSON.stringify(value).replace(/[\u0080-￿]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`);

    const body = asciiJson({ detail: `unexpected text: ${state.slice(10, 22)}` });
    expect(body).toMatch(/^[\x20-\x7e]*$/);
    expect(classifyHttpError(422, body, undefined, "P", state).body).toBe(WITHHELD);

    // Exactly 8 UTF-16 units, two of them the escaped surrogate pair of U+20BB7.
    const pair = asciiJson({ detail: "unknown name 𠮷田花子さん、" });
    expect(pair).toContain("\\ud842\\udfb7");
    expect(classifyHttpError(422, pair, undefined, "P", state).body).toBe(WITHHELD);
  });

  it("withholds a body that contains a state shorter than 8 characters", () => {
    const body = JSON.stringify({ error: { message: "unknown patient Zoë" } });
    expect(classifyHttpError(400, body, undefined, "P", "Zoë").body).toBe(WITHHELD);
    expect(classifyHttpError(400, '{"error":"unknown patient Zo\\u00EB"}', undefined, "P", "Zoë").body).toBe(WITHHELD);

    const other = JSON.stringify({ error: { message: "unknown patient Ana" } });
    expect(classifyHttpError(400, other, undefined, "P", "Zoë").body).toBe(other);
  });

  it("compares a non-string state as its JSON text", async () => {
    const fetchMock = fetchAlways(() => jsonResponse({ detail: "Unknown entity 'Maria Garcia'" }, { status: 422 }));
    const body: JsonObject = { ...wireBody(), state: { patient: "Maria Garcia", reason: "chest pain" } };
    const error = await providerErrorOf(postDecision(httpConfig(fetchMock), body));
    expect(error.body).toBe(WITHHELD);
    expect(error.message).toBe("TestProvider request failed with HTTP 422 (bad_request)");
  });

  it("classifies and checks a 1 MB body against a 200K-character state in under 100 ms", () => {
    const state = Array.from({ length: 5000 }, (_, i) => `Clause ${i}: the tenant pays the rent on the first of the month.`)
      .join("\n")
      .slice(0, 200_000);
    const raw = "upstream failure; ".repeat(60_000);
    const body = JSON.stringify({ error: { code: 400, message: "Provider returned error", metadata: { raw } } });
    expect(body.length).toBeGreaterThan(1_000_000);

    // No overlap, so the whole state is scanned: the slowest case. The best of
    // three runs keeps a busy machine from failing the bound.
    let bestMs = Number.POSITIVE_INFINITY;
    for (let run = 0; run < 3; run += 1) {
      const started = performance.now();
      const error = classifyHttpError(400, body, undefined, "P", state);
      bestMs = Math.min(bestMs, performance.now() - started);
      expect(error.kind).toBe("bad_request");
      expect(error.body).toBe(`${body.slice(0, 1999)}…`);
    }
    expect(bestMs).toBeLessThan(100);
  });

  it("bounds classification cost for a 50 MB body: only the first 1,000,000 characters are scanned", () => {
    const body = "upstream failure; ".repeat(Math.ceil(50_000_000 / 18));
    const state = "Clause 1: the tenant pays the rent on the first of the month.".repeat(1000);
    let bestMs = Number.POSITIVE_INFINITY;
    for (let run = 0; run < 2; run += 1) {
      const started = performance.now();
      const plain = classifyHttpError(400, body, undefined, "P");
      const withState = classifyHttpError(400, body, undefined, "P", state);
      bestMs = Math.min(bestMs, performance.now() - started);
      expect(plain.message).toBe(`P request failed with HTTP 400 (bad_request): ${body.slice(0, 299)}…`);
      expect(withState.body).toBe(`${body.slice(0, 1999)}…`);
    }
    expect(bestMs).toBeLessThan(250);
    // Wording past the scan window does not choose the kind.
    expect(classifyHttpError(400, `${"x".repeat(1_000_000)} maximum context length`, undefined, "P").kind).toBe(
      "bad_request",
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
    const known = new Set(["q"]);
    expect(() =>
      parseNativeResponse({ model: "m", answers: { q: { type: "noul", noul: secretish } } }, "P", known),
    ).toThrow('answers["q"].noul must be a finite number, got a string of length 37');
    // A type string that is not a protocol token came from the provider, so only its length is shown.
    expect(() => parseNativeResponse({ model: "m", answers: { q: { type: "multi" } } }, "P", known)).toThrow(
      'answers["q"].type must be "choice", "score", or "noul", got a string of length 5',
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
    // Provider text stays on `body`; the message never quotes it for a request with state (F01).
    expect(error.message).toBe(`TestProvider request failed with HTTP ${status} (${kind})`);
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
      expect(error.message).toBe("TestProvider request failed with HTTP 429 (rate_limit)");
      expect(error.body).toContain("Rate limit exceeded upstream");
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

  it("keeps the body of a non-JSON 2xx, or withholds it when it echoes the state", async () => {
    const page = "<html>Gateway page. Try again later.</html>";
    const plain = await providerErrorOf(postDecision(httpConfig(fetchAlways(() => textResponse(page))), wireBody()));
    expect(plain.kind).toBe("invalid_response");
    expect(plain.status).toBe(200);
    expect(plain.body).toBe(page);

    const fetchMock = fetchAlways(() => textResponse(`Gateway page. You sent: ${STATE}`));
    const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody()));
    expect(error.kind).toBe("invalid_response");
    expect(error.body).toBe(WITHHELD);
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
      expect(error.body).toBe(WITHHELD);
    });

    it("withholds the body for a very large echoed state", async () => {
      const bigState = Array.from({ length: 1500 }, (_, i) => `Sentence ${i} of the confidential record.`).join(" ");
      const fetchMock = fetchAlways(() => jsonResponse({ error: { message: `too large: ${bigState}` } }, { status: 400 }));
      const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody(bigState)));
      expect(error.kind).toBe("bad_request");
      expect(error.body).toBe(WITHHELD);
      expectNoState(error, bigState);
    });

    it("withholds the body when it contains a one-character state", async () => {
      const fetchMock = fetchAlways(() => jsonResponse({ error: { message: "bad value e" } }, { status: 400 }));
      const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody("e")));
      expect(error.body).toBe(WITHHELD);
      expect(error.message).toBe("TestProvider request failed with HTTP 400 (bad_request)");
    });

    it("leaves bodies that do not echo the state untouched", async () => {
      const text = JSON.stringify({ error: { code: 400, message: "questions.decision.criteria must have 2 options" } });
      const fetchMock = fetchAlways(() => textResponse(text, { status: 400 }));
      const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody()));
      expect(error.body).toBe(text);
      expect(error.message).toBe("TestProvider request failed with HTTP 400 (bad_request)");
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
      expect(error.message).toBe("OpenRouter request failed with HTTP 400 (context_limit)");
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
    expect(error.message).toBe("TypeSafe request failed with HTTP 429 (rate_limit)");
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

// ---------------------------------------------------------------------------
// Regression tests for review findings
// ---------------------------------------------------------------------------

const NOUL: NoulQuestion = { type: "noul", instructions: "Does the record mention a diagnosis?" };

describe("regression: short, nested, and re-escaped state echoes (F01)", () => {
  const PATIENT =
    "Patient record. SSN 123-45-6789, diagnosed with type 2 diabetes in March. Follow-up scheduled next quarter with endocrinology.";
  const FRAGMENT_BODY = JSON.stringify({ error: { message: "Invalid input near 'SSN 123-45-6789, diagnosed' in state" } });

  it("withholds a body quoting a short mid-state fragment and never quotes it in the message", async () => {
    const fetchMock = fetchAlways(() => textResponse(FRAGMENT_BODY, { status: 400 }));
    const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody(PATIENT)));
    expect(error.message).toBe("TestProvider request failed with HTTP 400 (bad_request)");
    expect(error.body).toBe(WITHHELD);
  });

  it("keeps the fragment out of the JevChunkFailedError that decide() throws", async () => {
    const transport = new DirectJevTransport({
      apiKey: "ts-test",
      fetch: fetchAlways(() => textResponse(FRAGMENT_BODY, { status: 400 })),
    });
    const error = await rejectionOf(decide({ input: PATIENT, question: NOUL, provider: { transport } }));
    expect(error).toBeInstanceOf(JevChunkFailedError);
    const message = (error as Error).message;
    expect(message).toContain("TypeSafe request failed with HTTP 400 (bad_request)");
    expect(message).not.toMatch(/SSN|123-45-6789|diagnosed|Invalid input/);
  });

  it("withholds a multi-line state echo double-escaped inside OpenRouter's metadata.raw", async () => {
    const state = [
      "Name: John Q. Doe",
      "SSN: 123-45-6789",
      "DOB: 1970-01-02",
      "Phone: +1 555 0100",
      'Notes: patient said "no known allergies" at intake.',
    ].join("\n");
    // The upstream provider echoes the state in its own JSON error, and
    // OpenRouter nests that raw body as a string.
    const nested = (message: string): unknown => ({
      error: {
        code: 400,
        message: "Provider returned error",
        metadata: { raw: JSON.stringify({ error: { message } }), provider_name: "TypeSafe" },
      },
    });
    const fetchMock = fetchAlways(() => jsonResponse(nested(`bad state: ${state}`), { status: 400 }));
    const error = await providerErrorOf(
      new OpenRouterJevTransport({ apiKey: "sk-or-test", fetch: fetchMock }).decide(request({ state })),
    );
    expect(error.message).toBe("OpenRouter request failed with HTTP 400 (bad_request)");
    expect(error.body).toBe(WITHHELD);
  });

  it("withholds an echo with Go's \\u003c, \\u003e, and \\u0026 escapes", () => {
    const state = "<b>Name:</b> John Q. Doe<br><b>SSN:</b> 123-45-6789<br><b>Dx:</b> HIV & hepatitis C<br>";
    const goEscaped = JSON.stringify({ error: `bad state ${state}` })
      .replace(/</g, "\\u003c")
      .replace(/>/g, "\\u003e")
      .replace(/&/g, "\\u0026");
    const error = classifyHttpError(400, goEscaped, undefined, "P", state);
    expect(error.body).toBe(WITHHELD);
    expect(error.message).toBe("P request failed with HTTP 400 (bad_request)");
  });

  it("withholds an echo with uppercase \\u escapes (.NET style)", () => {
    const state = "Zoë Müller, SSN 123-45-6789, née Schmidt, lives in Zürich.";
    const escaped = JSON.stringify({ error: `bad state ${state}` }).replace(
      /[\u0080-\uffff]/g,
      (c) => `\\u${c.charCodeAt(0).toString(16).toUpperCase().padStart(4, "0")}`,
    );
    expect(classifyHttpError(400, escaped, undefined, "P", state).body).toBe(WITHHELD);
  });

  it("withholds a 12-character name quoted on its own", () => {
    const state = "Referral letter.\nPatient: Maria Garcia\nReason: chest pain on exertion, two weeks.";
    const text = JSON.stringify({ detail: "Unknown entity 'Maria Garcia'" });
    expect(classifyHttpError(422, text, undefined, "P", state).body).toBe(WITHHELD);
  });

  it("checks past the cut, so the truncation cannot leave a short tail of an echo in body", () => {
    // Offset 1985 keeps 14 characters of the echo; 1992 keeps only "123-45-".
    for (const offset of [1985, 1992, 1998]) {
      const state = "123-45-6789 belongs to Maria Garcia, diagnosed with HIV in March. Follow-up next quarter.";
      const error = classifyHttpError(500, `${"x".repeat(offset)}${state} trailing`, undefined, "P", state);
      expect(error.body).toBe(WITHHELD);
    }

    // A JSON-escaped echo at 1990 keeps \"Maria\" (7 decoded characters).
    const quoted = '"Maria" SSN 123-45-6789 is the patient on record for this referral.';
    const escaped = `${"x".repeat(1990)}${JSON.stringify(quoted).slice(1, -1)}`;
    expect(classifyHttpError(500, escaped, undefined, "P", quoted).body).toBe(WITHHELD);

    // \u-escaped CJK, once (6 characters each) and twice (7 each), cut after a few characters.
    const cjk = "診療記録：田中花子さん、東京都港区在住。二型糖尿病の経過観察中。";
    const once = JSON.stringify(cjk).slice(1, -1).replace(/[\u0080-\uffff]/g, (c) => `\\u${c.charCodeAt(0).toString(16)}`);
    const twice = once.replace(/\\/g, "\\\\");
    for (const [echo, offset] of [
      [once, 1960],
      [twice, 1960],
      [twice, 1995],
    ] as const) {
      expect(classifyHttpError(500, `${"x".repeat(offset)}${echo}`, undefined, "P", cjk).body).toBe(WITHHELD);
    }
  });

  describe("an echo escaped twice, as in OpenRouter's metadata.raw", () => {
    /** Python's json.dumps default (ensure_ascii=True): every non-ASCII UTF-16 unit as \uXXXX. */
    const pythonJson = (value: unknown): string =>
      JSON.stringify(value).replace(/[\u0080-\uffff]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`);
    /** OpenRouter wraps the upstream body as a JSON string in error.metadata.raw. */
    const openRouterWrap = (upstreamBody: string): string =>
      JSON.stringify({
        error: { code: 400, message: "Provider returned error", metadata: { raw: upstreamBody, provider_name: "Upstream" } },
      });

    it("withholds a Cyrillic state echoed by a Python upstream", () => {
      const state = "Пациентка Иванова Мария, диагноз ВИЧ, наблюдается с марта, живёт в Казани.";
      const body = openRouterWrap(pythonJson({ detail: `invalid state: ${state}` }));
      // One decoded level still has \uXXXX between every character.
      expect(body).toMatch(/^[\x20-\x7e]*$/);
      expect(classifyHttpError(400, body, undefined, "OpenRouter", state).body).toBe(WITHHELD);
    });

    it("withholds a JSON-record state with no 8-character run free of quotes", () => {
      const state = '{"dx":"HIV+","age":42,"pin":"4821","zip":"94107","sex":"F"}';
      const body = openRouterWrap(JSON.stringify({ error: { message: `bad state: ${state}` } }));
      expect(classifyHttpError(400, body, undefined, "OpenRouter", state).body).toBe(WITHHELD);
    });

    it("withholds a short-line CSV state end to end through OpenRouterJevTransport.decide", async () => {
      const state = "id,dx\n1,HIV\n2,HCV\n3,TB\n4,HIV\n";
      const fetchMock = fetchAlways(() =>
        textResponse(openRouterWrap(JSON.stringify({ error: { message: `bad state: ${state}` } })), { status: 400 }),
      );
      const error = await providerErrorOf(
        new OpenRouterJevTransport({ apiKey: "sk-or-test", fetch: fetchMock }).decide(request({ state })),
      );
      expect(error.message).toBe("OpenRouter request failed with HTTP 400 (bad_request)");
      expect(error.body).toBe(WITHHELD);
    });
  });

  it("property: an 8+ character fragment, raw, JSON-escaped, nested, or \\u-escaped, withholds the body", async () => {
    const upperAscii = (text: string): string =>
      text.replace(/[\u0080-\uffff]/g, (c) => `\\u${c.charCodeAt(0).toString(16).toUpperCase().padStart(4, "0")}`);
    const nest = (raw: string): string =>
      JSON.stringify({ error: { message: "Provider returned error", metadata: { raw } } });
    await fc.assert(
      fc.asyncProperty(
        fc.string({ minLength: 40, maxLength: 300 }),
        fc.nat(),
        fc.integer({ min: 8, max: 40 }),
        fc.constantFrom("raw", "json", "json-in-json", "upper-ascii-json", "upper-ascii-json-in-json"),
        fc.string({ maxLength: 30 }),
        async (state, offset, length, mode, prefix) => {
          const start = offset % (state.length - length + 1);
          const fragment = state.slice(start, start + length);
          const message = `${prefix}${fragment} tail`;
          const json = JSON.stringify({ error: { message } });
          const bodyText =
            mode === "raw"
              ? `bad input: ${message}`
              : mode === "json"
                ? json
                : mode === "json-in-json"
                  ? nest(json)
                  : mode === "upper-ascii-json"
                    ? upperAscii(json)
                    : nest(upperAscii(json));
          const fetchMock = fetchAlways(() => textResponse(bodyText, { status: 400 }));
          const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody(state)));
          expect(error.message).toMatch(/^TestProvider request failed with HTTP 400 \((?:bad_request|context_limit)\)$/);
          // Runs are compared with whitespace collapsed, so a fragment that is
          // mostly whitespace can be too short to count.
          if (fragment.replace(/\s+/g, " ").length >= GUARANTEED_RUN) expect(error.body).toBe(WITHHELD);
        },
      ),
      { numRuns: 300 },
    );
  });
});

describe("regression: header validation never exposes the value (F15)", () => {
  const SECRET = "sk-F15-TOP-SECRET-KEY";

  function thrown(fn: () => unknown): Error {
    try {
      fn();
    } catch (error) {
      expect(error).toBeInstanceOf(JevValidationError);
      return error as Error;
    }
    throw new Error("expected a throw");
  }

  /** What console.error prints, plus every message and stack on the cause chain. */
  function everythingLogged(error: Error): string {
    const parts = [inspect(error, { depth: 10 })];
    for (let current: unknown = error; current instanceof Error; current = current.cause) {
      parts.push(current.message, String(current.stack));
    }
    return parts.join("\n");
  }

  it.each([
    ["LF", `${SECRET}\nX-Injected: 1`],
    ["CR", `${SECRET}\rrest`],
    ["NUL", `${SECRET}\u0000rest`],
  ])("DirectJevTransport rejects an API key containing %s without echoing it", (_label, apiKey) => {
    const error = thrown(() => new DirectJevTransport({ apiKey, fetch: fetchAlways(() => jsonResponse({})) }));
    expect(error.message).toBe("TypeSafe: the API key contains a character HTTP headers do not allow (such as CR, LF, or NUL)");
    expect(error.cause).toBeUndefined();
    expect(everythingLogged(error)).not.toContain(SECRET);
  });

  it("OpenRouterJevTransport rejects an API key containing LF without echoing it", () => {
    const error = thrown(
      () => new OpenRouterJevTransport({ apiKey: `${SECRET}\nX-Injected: 1`, fetch: fetchAlways(() => jsonResponse({})) }),
    );
    expect(error.message).toBe("OpenRouter: the API key contains a character HTTP headers do not allow (such as CR, LF, or NUL)");
    expect(everythingLogged(error)).not.toContain(SECRET);
  });

  it("names a caller header, but not its value", () => {
    const error = thrown(
      () =>
        new OpenRouterJevTransport({
          apiKey: "sk-or-test",
          fetch: fetchAlways(() => jsonResponse({})),
          headers: { "X-Custom-Auth": `${SECRET}\nabc` },
        }),
    );
    expect(error.message).toBe('OpenRouter: headers["X-Custom-Auth"] is not a valid HTTP header name and value');
    expect(everythingLogged(error)).not.toContain(SECRET);
  });

  it("names another fixed header, but not its value", () => {
    const error = thrown(
      () =>
        new OpenRouterJevTransport({
          apiKey: "sk-or-test",
          fetch: fetchAlways(() => jsonResponse({})),
          appName: `App ${SECRET}\nX`,
        }),
    );
    expect(error.message).toBe(
      "OpenRouter: the X-OpenRouter-Title header value contains a character HTTP headers do not allow (such as CR, LF, or NUL)",
    );
    expect(everythingLogged(error)).not.toContain(SECRET);
  });

  it("mergeHeaders carries nothing on the cause chain", () => {
    const error = thrown(() => mergeHeaders(undefined, { Authorization: `Bearer ${SECRET}\nX: y` }, "test"));
    expect(error.cause).toBeUndefined();
    expect(everythingLogged(error)).not.toContain(SECRET);
  });
});

describe("regression: echoed document text and the error kind (F16, superseded)", () => {
  // The kind reads the body, echoes of the state included. A document with
  // context-limit wording, echoed in a 400 or 422, can cause a spurious
  // re-chunk; it still fails closed and never leaks the state. "overload" is
  // read only for a 5xx or a code-less 2xx error, so an echo cannot make a
  // deterministic 4xx retryable.
  const sentence = "Section notes: the context window of the model was compared against the baseline run. ";
  const CPP_DOC =
    "Style guide. Prefer a free-function operator overload over a member when the left operand is not the class. " +
    "Keep overloaded operators consistent with the built-in semantics.";

  /** FastAPI/pydantic-style 422: the rejected object (the whole request body) is echoed as `input`. */
  function validationEcho(requestBody: unknown): string {
    return JSON.stringify({
      detail: [
        {
          type: "extra_forbidden",
          loc: ["body", "questions", "decision", "weight"],
          msg: "Extra inputs are not permitted",
          input: requestBody,
        },
      ],
    });
  }

  it("a 422 that echoes a state mentioning the context window is context_limit, with the body withheld", () => {
    const error = classifyHttpError(422, validationEcho(wireBody(sentence)), undefined, "P", sentence);
    expect(error.kind).toBe("context_limit");
    expect(error.body).toBe(WITHHELD);
    expect(error.message).toBe("P request failed with HTTP 422 (context_limit)");
  });

  it("decide() fails closed with JevContextBudgetError when every chunk's 422 echo mentions the context window", async () => {
    const input = sentence.repeat(250);
    const fetchMock = vi.fn<typeof fetch>(
      async (_input, init) => new Response(validationEcho(JSON.parse(String(init?.body))), { status: 422 }),
    );
    const transport = new DirectJevTransport({ apiKey: "ts-test", fetch: fetchMock });
    const error = await rejectionOf(decide({ input, question: NOUL, provider: { transport } }));
    expect(error).toBeInstanceOf(JevContextBudgetError);
    const cause = (error as JevContextBudgetError).cause as JevProviderError;
    expect(cause.kind).toBe("context_limit");
    expect(cause.body).toBe(WITHHELD);
    expect(`${(error as Error).message}${cause.message}`).not.toContain("Section notes");
  });

  it("a 409 that echoes a state mentioning operator overloads is unknown and not retried", () => {
    const body = JSON.stringify({ error: { message: "Conflict: idempotency key reused" }, request: wireBody(CPP_DOC) });
    const error = classifyHttpError(409, body, undefined, "P", CPP_DOC);
    expect(error.kind).toBe("unknown");
    expect(error.retryable).toBe(false);
    expect(error.body).toBe(WITHHELD);
  });

  it("decide() sends a deterministic 409 that echoes such a state once", async () => {
    const fetchMock = vi.fn<typeof fetch>(async (_input, init) => {
      const sent = JSON.parse(String(init?.body)) as { state: string };
      return new Response(JSON.stringify({ error: { message: "Conflict" }, request: { state: sent.state } }), {
        status: 409,
      });
    });
    const transport = new DirectJevTransport({ apiKey: "ts-test", fetch: fetchMock });
    const error = await rejectionOf(
      decide({ input: CPP_DOC, question: NOUL, provider: { transport }, execution: { retryBaseDelayMs: 0 } }),
    );
    expect(error).toBeInstanceOf(JevChunkFailedError);
    expect(((error as JevChunkFailedError).cause as JevProviderError).kind).toBe("unknown");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("still reads a provider's own \"overloaded\" in a 5xx or a code-less 2xx error", () => {
    const body = JSON.stringify({ error: { message: "Model is overloaded, try again" } });
    expect(classifyHttpError(500, body, undefined, "P", CPP_DOC).kind).toBe("overloaded");
    expect(classifyHttpError(undefined, body, undefined, "P", CPP_DOC).kind).toBe("overloaded");
    expect(classifyHttpError(409, body, undefined, "P").kind).toBe("unknown");
  });
});

describe("regression: count-based over-length wordings re-chunk (F17)", () => {
  const CHOICE: ChoiceQuestion = {
    type: "choice",
    instructions: "What is the text mostly about?",
    criteria: { tree: "Trees.", rock: "Rocks.", other: "Neither." },
  };

  /** A transport that rejects states longer than `maxStateChars` with a 400 carrying `message`. */
  function rejectingLongStates(message: string, maxStateChars: number): DirectJevTransport {
    return new DirectJevTransport({
      apiKey: "ts-test",
      fetch: async (_input, init) => {
        const body = JSON.parse(String(init?.body)) as { state: string };
        if (body.state.length > maxStateChars) return jsonResponse({ error: { message } }, { status: 400 });
        return jsonResponse({
          model: "jev-latest",
          answers: {
            decision: { type: "choice", choice: "tree", probabilities: { tree: 0.8, rock: 0.1, other: 0.1 }, confidence: 0.7 },
          },
        });
      },
    });
  }

  it.each(["state exceeds the model's context window", "state exceeds 32000 tokens"])(
    "re-chunks after %j and succeeds",
    async (message) => {
      const result = await decide({
        input: "oak tree grows in the forest. ".repeat(400),
        question: CHOICE,
        provider: { transport: rejectingLongStates(message, 3000) },
        chunking: { maxStateTokens: MIN_STATE_TOKENS * 8 },
        execution: { retryBaseDelayMs: 0 },
      });
      expect(result.usage.rechunks).toBeGreaterThan(0);
    },
  );
});

describe("regression: provider text that may echo the state stays out of the error chain", () => {
  const REFERRAL = "Referral. SSN 123-45-6789, Maria Garcia, HIV positive since March; follow-up with endocrinology.";
  const COPIED = "SSN 123-45-6789, Maria Garcia";

  it("drops the SyntaxError of a non-JSON 2xx, whose message quotes the body", async () => {
    const fetchMock = fetchAlways(() => textResponse(COPIED));
    const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody(REFERRAL)));
    expect(error.kind).toBe("invalid_response");
    expect(error.body).toBe(WITHHELD);
    expect(error.cause).toBeUndefined();
    expect(inspect(error)).not.toContain("123-45");
  });

  it("drops the SyntaxError of a malformed 2xx JSON, whose message quotes a window of it", async () => {
    const fetchMock = fetchAlways(() => textResponse('{"model":"m","note": SSN 123-45-6789 of Maria}'));
    const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody(REFERRAL)));
    expect(error.cause).toBeUndefined();
    expect(inspect(error)).not.toContain("123-45");
  });

  it("describes a provider `type` string by length only", async () => {
    const fetchMock = fetchAlways(() => jsonResponse({ model: "m", answers: { decision: { type: COPIED } } }));
    const error = await providerErrorOf(postDecision(httpConfig(fetchMock), wireBody(REFERRAL)));
    expect(error.message).toBe(
      'TestProvider returned an invalid decision response: answers["decision"].type must be "choice", "score", or "noul", got a string of length 29',
    );
  });

  it("quotes a response key only when the request sent it", async () => {
    const body: JsonObject = {
      ...wireBody(REFERRAL),
      questions: {
        decision: { type: "choice", instructions: "Which team?", criteria: { a: "team a", b: "team b" } },
        level: { type: "score", instructions: "How urgent?", criteria: ["low", "high"] },
      },
    };
    const messageFor = async (answers: unknown): Promise<string> =>
      (await providerErrorOf(postDecision(httpConfig(fetchAlways(() => jsonResponse({ model: "m", answers }))), body)))
        .message;

    expect(await messageFor({ [COPIED]: { type: "noul", noul: "x" } })).toContain(
      "answers[<unknown key of length 29>].noul must be a finite number",
    );
    expect(await messageFor({ decision: { type: "choice", choice: "a", probabilities: { a: 1, [COPIED]: "x" } } })).toContain(
      'answers["decision"].probabilities[<unknown key of length 29>] must be a finite number',
    );
    expect(await messageFor({ decision: { type: "choice", choice: "a", probabilities: { a: "x" } } })).toContain(
      'answers["decision"].probabilities["a"] must be a finite number',
    );
    expect(await messageFor({ level: { type: "score", score: 1, legend: { "1": 5 } } })).toContain(
      'answers["level"].legend["1"] must be a string, object, or array',
    );
    expect(await messageFor({ level: { type: "score", score: 1, legend: { [COPIED]: 5 } } })).toContain(
      'answers["level"].legend[<unknown key of length 29>] must be a string, object, or array',
    );
  });

  it("keeps a provider `choice` copied from the document out of JevChunkFailedError", async () => {
    const fetchMock = fetchAlways(() =>
      jsonResponse({ model: "m", answers: { decision: { type: "choice", choice: COPIED } } }),
    );
    const transport = new DirectJevTransport({ apiKey: "ts-test", fetch: fetchMock });
    const error = await rejectionOf(
      decide({
        input: REFERRAL,
        question: { type: "choice", instructions: "Which team?", criteria: { a: "team a", b: "team b" } },
        provider: { transport },
        execution: { retryBaseDelayMs: 0 },
      }),
    );
    expect(error).toBeInstanceOf(JevChunkFailedError);
    expect(((error as JevChunkFailedError).cause as Error).message).toBe(
      "Jev choice a string of length 29 is not one of the question's criteria keys.",
    );
    expect(inspect(error)).not.toContain("123-45");
  });
});

describe("regression: transport options accept an explicit undefined (F23)", () => {
  it("options built from possibly-unset values typecheck under exactOptionalPropertyTypes", () => {
    const consumerPath = path.join(path.dirname(fileURLToPath(import.meta.url)), "__transport_options_consumer__.ts");
    const consumerSource = `
      import { OpenRouterJevTransport } from "../src/transports/openrouter.js";
      import { DirectJevTransport } from "../src/transports/direct.js";

      declare function maybe<T>(): T | undefined;

      // README "OpenRouter Decisions API (default)" example, plus every other option.
      export const openRouter = new OpenRouterJevTransport({
        apiKey: process.env.OPENROUTER_API_KEY,
        baseUrl: process.env.OPENROUTER_BASE_URL,
        defaultModel: maybe<string>(),
        fetch: maybe<typeof fetch>(),
        timeoutMs: maybe<number>(),
        headers: maybe<Record<string, string>>(),
        appName: maybe<string>(),
        appUrl: maybe<string>(),
        sessionId: maybe<string>(),
        user: maybe<string>(),
        contextWindow: maybe<number>(),
        resolveContextWindow: maybe<boolean>(),
        metadataTimeoutMs: maybe<number>(),
      });

      export const direct = new DirectJevTransport({
        apiKey: process.env.TYPESAFE_API_KEY,
        baseUrl: maybe<string>(),
        path: maybe<string>(),
        defaultModel: maybe<string>(),
        fetch: maybe<typeof fetch>(),
        timeoutMs: maybe<number>(),
        headers: maybe<Record<string, string>>(),
        contextWindow: maybe<number>(),
      });
    `;
    const options: ts.CompilerOptions = {
      target: ts.ScriptTarget.ES2022,
      lib: ["lib.es2022.d.ts", "lib.dom.d.ts"],
      module: ts.ModuleKind.NodeNext,
      moduleResolution: ts.ModuleResolutionKind.NodeNext,
      strict: true,
      exactOptionalPropertyTypes: true,
      noEmit: true,
      skipLibCheck: true,
      types: ["node"],
    };
    const host = ts.createCompilerHost(options);
    const fileExists = host.fileExists.bind(host);
    const readFile = host.readFile.bind(host);
    const getSourceFile = host.getSourceFile.bind(host);
    host.fileExists = (file) => file === consumerPath || fileExists(file);
    host.readFile = (file) => (file === consumerPath ? consumerSource : readFile(file));
    host.getSourceFile = (file, language, onError, create) =>
      file === consumerPath
        ? ts.createSourceFile(file, consumerSource, language, true)
        : getSourceFile(file, language, onError, create);

    const program = ts.createProgram([consumerPath], options, host);
    const consumer = program.getSourceFile(consumerPath);
    expect(consumer).toBeDefined();
    // Only the consumer's diagnostics count: the package itself builds without the flag.
    const diagnostics = [...program.getSyntacticDiagnostics(consumer), ...program.getSemanticDiagnostics(consumer)].map(
      (diagnostic) => ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n"),
    );
    expect(diagnostics).toEqual([]);
  }, 60_000);
});
