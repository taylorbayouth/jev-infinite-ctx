/**
 * Shared HTTP plumbing for the built-in Jev transports (spec 13): request
 * encoding, deadline and abort handling, provider error classification, and
 * strict parsing of the Decisions wire format into `NativeJevResponse`.
 *
 * Invariant: no error created here carries request state in its message or
 * `body` (spec 16: never log source text). Provider bodies are truncated and
 * scrubbed of any echo of the state before they are attached to an error.
 */

import {
  JevAbortError,
  JevProviderError,
  JevValidationError,
  type ProviderErrorKind,
} from "../errors.js";
import type {
  JevCriterion,
  JevQuestion,
  JsonObject,
  NativeJevAnswer,
  NativeJevRequest,
  NativeJevResponse,
  NativeJevUsage,
} from "../types.js";

export interface HttpDecisionConfig {
  url: string;
  headers: Record<string, string>;
  fetch: typeof fetch;
  timeoutMs: number;
  /** Human-readable provider label used in error messages, e.g. "OpenRouter". */
  providerName: string;
}

/** Longest provider body kept on a `JevProviderError`. */
export const MAX_ERROR_BODY_CHARS = 2000;
/** Longest provider message quoted in an error's `message`. */
const MAX_ERROR_DETAIL_CHARS = 300;
/** setTimeout clamps larger delays to 1 ms, so they cannot be honored. */
const MAX_TIMER_MS = 2_147_483_647;

// ---------------------------------------------------------------------------
// Decision request
// ---------------------------------------------------------------------------

/**
 * POSTs one Decisions request and returns the normalized response.
 *
 * Throws `JevAbortError` when `signal` fires, and `JevProviderError` for
 * everything the provider or network does wrong: `timeout` when `timeoutMs`
 * elapses, `network` when fetch rejects, a status-derived kind for non-2xx
 * responses and for 2xx responses carrying an `error` object, and
 * `invalid_response` for bodies that are not a well-formed decision.
 */
export async function postDecision(
  config: HttpDecisionConfig,
  body: JsonObject,
  signal?: AbortSignal,
): Promise<NativeJevResponse> {
  const secret = stateSecret(body);
  const { response, text } = await fetchText(
    config.fetch,
    config.url,
    { method: "POST", headers: config.headers, body: JSON.stringify(body) },
    { timeoutMs: config.timeoutMs, providerName: config.providerName, signal },
  );

  if (!response.ok) {
    throw classifyHttpError(response.status, text, response.headers, config.providerName, secret);
  }

  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch (error) {
    throw new JevProviderError(
      `${config.providerName} returned a response body that is not valid JSON (HTTP ${response.status})`,
      {
        kind: "invalid_response",
        status: response.status,
        retryable: false,
        body: sanitize(text, secret, MAX_ERROR_BODY_CHARS),
        cause: error,
      },
    );
  }

  // OpenRouter can report a failure as HTTP 200 with a top-level `error`
  // object (for example when the upstream provider fails after the request
  // was accepted). Classify it exactly like the equivalent HTTP error.
  if (isRecord(json)) {
    const errorObject = own(json, "error");
    if (isRecord(errorObject)) {
      throw classifyHttpError(
        httpStatusOrUndefined(own(errorObject, "code")),
        text,
        response.headers,
        config.providerName,
        secret,
      );
    }
  }

  return parseNativeResponse(json, config.providerName);
}

/**
 * Encodes a `NativeJevRequest` as the JSON wire body `{ model, state,
 * questions }`. Questions are rebuilt field by field so nothing but the wire
 * fields is sent (the request's `signal` in particular never is), and the
 * question map is rebuilt with `Object.fromEntries` so caller keys such as
 * "__proto__" survive.
 */
export function encodeDecisionBody(request: NativeJevRequest): JsonObject {
  if (!isRecord(request)) throw new JevValidationError("request must be an object");
  if (typeof request.model !== "string" || request.model.trim() === "") {
    throw new JevValidationError("request.model must be a non-empty string");
  }
  if (typeof request.state !== "string") {
    throw new JevValidationError("request.state must be a string");
  }
  if (!isRecord(request.questions)) {
    throw new JevValidationError("request.questions must be an object");
  }
  const questions: JsonObject = Object.fromEntries(
    Object.entries(request.questions).map(([key, question]) => [key, encodeQuestion(key, question)]),
  );
  return { model: request.model, state: request.state, questions };
}

function encodeQuestion(key: string, question: JevQuestion): JsonObject {
  // Guards JavaScript callers of a public transport; typed callers cannot reach the throws.
  if (!isRecord(question)) {
    throw new JevValidationError(`request.questions[${quote(key)}] must be an object`);
  }
  switch (question.type) {
    case "noul":
      return question.criteria === undefined
        ? { type: "noul", instructions: question.instructions }
        : {
            type: "noul",
            instructions: question.instructions,
            criteria: { true: question.criteria.true, false: question.criteria.false },
          };
    case "choice":
      return { type: "choice", instructions: question.instructions, criteria: question.criteria };
    case "score":
      return { type: "score", instructions: question.instructions, criteria: question.criteria };
    default: {
      const unknownType: unknown = (question as { type?: unknown }).type;
      throw new JevValidationError(
        `request.questions[${quote(key)}].type must be "choice", "score", or "noul", got ${describeType(unknownType)}`,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Fetch with deadline
// ---------------------------------------------------------------------------

export interface FetchTextOptions {
  timeoutMs: number;
  providerName: string;
  signal?: AbortSignal | undefined;
}

/**
 * Runs `fetchFn` and reads the body as text under one deadline that also
 * covers the body read. The fetch and body promises are raced against the
 * internal abort signal, so the deadline holds even for a fetch
 * implementation (or a response stream) that ignores its signal.
 */
export async function fetchText(
  fetchFn: typeof fetch,
  url: string,
  init: Omit<RequestInit, "signal">,
  options: FetchTextOptions,
): Promise<{ response: Response; text: string }> {
  const { signal, providerName, timeoutMs } = options;
  if (!isValidTimerDelay(timeoutMs)) {
    throw new JevValidationError(`${providerName}: timeoutMs must be a number of milliseconds in (0, ${MAX_TIMER_MS}]`);
  }
  if (signal?.aborted) {
    throw new JevAbortError(`${providerName} request was aborted before it was sent`, {
      cause: signal.reason,
    });
  }

  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const forwardAbort = (): void => controller.abort(signal?.reason);
  signal?.addEventListener("abort", forwardAbort, { once: true });

  try {
    const response = await raceAbort(
      Promise.resolve(fetchFn(url, { ...init, signal: controller.signal })),
      controller.signal,
    );
    const text = await raceAbort(response.text(), controller.signal);
    return { response, text };
  } catch (error) {
    // Caller abort wins over a simultaneous timeout: the caller asked to stop.
    if (signal?.aborted) {
      throw new JevAbortError(`${providerName} request was aborted`, { cause: signal.reason });
    }
    if (timedOut) {
      throw new JevProviderError(`${providerName} request timed out after ${timeoutMs} ms`, {
        kind: "timeout",
        cause: error,
      });
    }
    // Only the error's name and system code are quoted; the full error stays
    // on `cause` for debugging.
    throw new JevProviderError(`${providerName} request failed: network error (${describeNetworkError(error)})`, {
      kind: "network",
      cause: error,
    });
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", forwardAbort);
  }
}

/** Settles with `promise`, or rejects as soon as `signal` aborts. Never leaves a rejection unhandled. */
function raceAbort<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    if (signal.aborted) {
      promise.catch(() => undefined);
      reject(signal.reason);
      return;
    }
    const onAbort = (): void => reject(signal.reason);
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener("abort", onAbort);
        resolve(value);
      },
      (error: unknown) => {
        signal.removeEventListener("abort", onAbort);
        reject(error);
      },
    );
  });
}

function describeNetworkError(error: unknown): string {
  const name = error instanceof Error ? error.name : typeof error;
  const cause: unknown = error instanceof Error ? error.cause : undefined;
  const code = isRecord(cause) ? own(cause, "code") : undefined;
  // Only system-style codes such as "ECONNREFUSED" or "UND_ERR_SOCKET".
  return typeof code === "string" && /^[A-Z][A-Z0-9_]{0,63}$/.test(code) ? `${name} ${code}` : name;
}

// ---------------------------------------------------------------------------
// Error classification
// ---------------------------------------------------------------------------

/**
 * Maps a failed provider response to a `JevProviderError` (spec 7 error
 * behavior). `status` is undefined for a 2xx body whose `error` object has no
 * numeric code. `redact` is the request state: any echo of it is removed from
 * the error's message and `body`.
 */
export function classifyHttpError(
  status: number | undefined,
  bodyText: string,
  headers: Headers | undefined,
  providerName: string,
  redact?: string,
): JevProviderError {
  const kind = kindForStatus(status, bodyText);
  const detail = sanitize(providerMessage(bodyText), redact, MAX_ERROR_DETAIL_CHARS, true);
  const head =
    status === undefined
      ? `${providerName} returned an error (${kind})`
      : `${providerName} request failed with HTTP ${status} (${kind})`;
  return new JevProviderError(detail === undefined ? head : `${head}: ${detail}`, {
    kind,
    status,
    retryAfterMs: parseRetryAfter(headers?.get("retry-after")),
    body: sanitize(bodyText, redact, MAX_ERROR_BODY_CHARS),
  });
}

const OVERLOAD_PATTERN = /overload/i;

function kindForStatus(status: number | undefined, text: string): ProviderErrorKind {
  if (status === undefined) {
    if (isContextLimitMessage(text)) return "context_limit";
    return OVERLOAD_PATTERN.test(text) ? "overloaded" : "unknown";
  }
  if (status === 413) return "context_limit";
  if (status === 400 || status === 422) {
    return isContextLimitMessage(text) ? "context_limit" : "bad_request";
  }
  if (status === 401 || status === 403) return "auth";
  if (status === 402) return "payment";
  if (status === 404) return "not_found";
  if (status === 408) return "timeout";
  if (status === 429) return "rate_limit";
  if (status === 503 || status === 529 || OVERLOAD_PATTERN.test(text)) return "overloaded";
  if (status >= 500 && status <= 599) return "server";
  return "unknown";
}

function httpStatusOrUndefined(code: unknown): number | undefined {
  return typeof code === "number" && Number.isInteger(code) && code >= 100 && code <= 599
    ? code
    : undefined;
}

/**
 * Converts a Retry-After header value (delay-seconds or HTTP-date) to a delay
 * in milliseconds. Returns undefined when absent or unparseable; a date in
 * the past yields 0.
 */
export function parseRetryAfter(value: string | null | undefined, nowMs: number = Date.now()): number | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  if (trimmed === "") return undefined;

  if (/^\d+(?:\.\d+)?$/.test(trimmed)) {
    const ms = Math.ceil(Number(trimmed) * 1000);
    return Number.isFinite(ms) ? ms : undefined;
  }

  // Require letters so bare or signed numbers ("-5") are not read as years by Date.parse.
  if (!/[a-z]/i.test(trimmed)) return undefined;
  // asctime() dates carry no zone but are defined as GMT (RFC 9110 5.6.7).
  const asctime = /^[a-z]{3} [a-z]{3} +\d{1,2} \d{2}:\d{2}:\d{2} \d{4}$/i.test(trimmed);
  const dateMs = Date.parse(asctime ? `${trimmed} GMT` : trimmed);
  if (!Number.isFinite(dateMs)) return undefined;
  return Math.max(0, dateMs - nowMs);
}

/**
 * Phrases providers use for "the request is larger than the model accepts".
 * Each pattern is tied to context, tokens, length, or size so that generic
 * token errors ("invalid token", "token expired") are not matched.
 */
const CONTEXT_LIMIT_PATTERNS: readonly RegExp[] = [
  // "context length", "context_length_exceeded", "context window", "context limit", "context size"
  /\bcontext[\s_-]*(?:length|window|limit|size)/i,
  /\bmaximum context\b/i,
  /\btoo many (?:input |prompt |state |context )?tokens\b/i,
  /\btokens?[\s_-]*limit\b/i,
  /\b(?:prompt|input|state|request|payload|message|messages|body|content|entity)\s+(?:is\s+|are\s+)?too\s+(?:long|large|big)\b/i,
  // "exceeds the maximum ... tokens/length/size/context", but not "temperature exceeds the maximum"
  /\bexceed(?:s|ed|ing)?\s+(?:the\s+)?(?:maximum|max)\b[^.;\n]{0,60}?\b(?:tokens?|length|context|size)\b/i,
  // "input tokens exceed", "prompt length exceeds", "state size exceeded"
  /\b(?:input|prompt|state|request|total)[\s_-]*(?:tokens?|length|size)\b[^.;\n]{0,30}?\bexceed/i,
  /\breduce the (?:length|size) of\b/i,
];

/** True when a provider message says the request exceeded the model's context. */
export function isContextLimitMessage(text: string): boolean {
  return CONTEXT_LIMIT_PATTERNS.some((pattern) => pattern.test(text));
}

/** Best human-readable message in a provider error body. */
function providerMessage(bodyText: string): string {
  let json: unknown;
  try {
    json = JSON.parse(bodyText);
  } catch {
    return bodyText;
  }
  if (!isRecord(json)) return bodyText;
  const error = own(json, "error");
  if (isRecord(error)) {
    const message = own(error, "message");
    if (typeof message === "string") return message;
  }
  if (typeof error === "string") return error;
  for (const key of ["message", "detail"]) {
    const value = own(json, key);
    if (typeof value === "string") return value;
  }
  return bodyText;
}

// ---------------------------------------------------------------------------
// State redaction (spec 16)
// ---------------------------------------------------------------------------

const REDACTION_MARKER = "[redacted]";
/**
 * Width of the aligned probes used to find echoed state. Any run of at least
 * 2 * ECHO_PROBE_CHARS - 1 characters shared with the state contains a
 * probe-aligned window, so every such run is found and removed.
 */
const ECHO_PROBE_CHARS = 16;

/** The text that must never appear in an error built for this body. */
function stateSecret(body: JsonObject): string | undefined {
  const state = own(body, "state");
  if (state === undefined || state === "") return undefined;
  return typeof state === "string" ? state : JSON.stringify(state);
}

/**
 * Truncates, redacts, optionally collapses whitespace, and truncates again;
 * returns undefined for an empty result. Redaction runs before collapsing
 * (collapsing would hide a multi-line echo from exact matching) and again
 * after it (collapsing can join fragments into a longer echo).
 */
function sanitize(
  text: string,
  secret: string | undefined,
  maxChars: number,
  collapseWhitespace = false,
): string | undefined {
  let cleaned = redactSecret(truncate(text, maxChars), secret);
  if (collapseWhitespace) cleaned = redactSecret(cleaned.replace(/\s+/g, " ").trim(), secret);
  cleaned = truncate(cleaned, maxChars);
  return cleaned === "" ? undefined : cleaned;
}

function truncate(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  let end = maxChars - 1;
  // Do not leave half of a surrogate pair before the ellipsis.
  const code = text.charCodeAt(end - 1);
  if (code >= 0xd800 && code <= 0xdbff) end -= 1;
  return `${text.slice(0, end)}…`;
}

/**
 * Removes every echo of `secret` from `text`: whole occurrences (raw or
 * JSON-escaped, as a provider may quote the state inside a JSON message) and
 * any long partial run. If a marker would itself recreate the secret (a
 * secret as short as "e"), the text is withheld entirely.
 */
function redactSecret(text: string, secret: string | undefined): string {
  if (secret === undefined || secret === "" || text === "") return text;
  const variants = secretVariants(secret);
  let out = text;
  for (const variant of variants) out = redactVariant(out, variant);
  return variants.some((variant) => out.includes(variant)) ? "" : out;
}

function secretVariants(secret: string): string[] {
  const json = JSON.stringify(secret).slice(1, -1);
  // ASCII-only JSON, as emitted by e.g. Python's json.dumps default.
  const asciiJson = json.replace(
    /[\u0080-\uffff]/g,
    (char) => `\\u${char.charCodeAt(0).toString(16).padStart(4, "0")}`,
  );
  return [...new Set([secret, json, asciiJson])];
}

function redactVariant(text: string, secret: string): string {
  const exact = text.split(secret).join(REDACTION_MARKER);
  if (secret.length < ECHO_PROBE_CHARS) return exact;

  const ranges: Array<[number, number]> = [];
  let pos = 0;
  while (pos + ECHO_PROBE_CHARS <= exact.length) {
    const at = secret.indexOf(exact.slice(pos, pos + ECHO_PROBE_CHARS));
    if (at < 0) {
      pos += ECHO_PROBE_CHARS;
      continue;
    }
    let start = pos;
    let s = at;
    while (start > 0 && s > 0 && exact.charCodeAt(start - 1) === secret.charCodeAt(s - 1)) {
      start -= 1;
      s -= 1;
    }
    let end = pos + ECHO_PROBE_CHARS;
    let e = at + ECHO_PROBE_CHARS;
    while (end < exact.length && e < secret.length && exact.charCodeAt(end) === secret.charCodeAt(e)) {
      end += 1;
      e += 1;
    }
    ranges.push([start, end]);
    pos = end;
  }
  if (ranges.length === 0) return exact;

  // A later probe can extend left past an earlier range (a different
  // occurrence in the secret), so sort, then merge overlapping or adjacent
  // ranges so each echo becomes a single marker.
  ranges.sort((a, b) => a[0] - b[0]);
  const merged: Array<[number, number]> = [];
  for (const [start, end] of ranges) {
    const last = merged[merged.length - 1];
    if (last !== undefined && start <= last[1]) last[1] = Math.max(last[1], end);
    else merged.push([start, end]);
  }
  let out = "";
  let cursor = 0;
  for (const [start, end] of merged) {
    out += exact.slice(cursor, start) + REDACTION_MARKER;
    cursor = end;
  }
  return out + exact.slice(cursor);
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

/**
 * Validates a decoded Decisions response and normalizes it to
 * `NativeJevResponse` (camelCase usage). Structural only: probability ranges
 * and option membership are checked later against the question (spec 8).
 * Throws `JevProviderError` kind "invalid_response" (not retryable).
 */
export function parseNativeResponse(raw: unknown, providerName: string): NativeJevResponse {
  const fail = (problem: string): JevProviderError =>
    new JevProviderError(`${providerName} returned an invalid decision response: ${problem}`, {
      kind: "invalid_response",
      retryable: false,
    });

  if (!isRecord(raw)) throw fail(`expected a JSON object, got ${describe(raw)}`);

  const model = own(raw, "model");
  if (typeof model !== "string" || model === "") throw fail("`model` must be a non-empty string");

  const answers = own(raw, "answers");
  if (!isRecord(answers)) throw fail(`\`answers\` must be an object, got ${describe(answers)}`);

  const parsedAnswers: Record<string, NativeJevAnswer> = Object.fromEntries(
    Object.entries(answers).map(([key, value]) => [key, parseAnswer(`answers[${quote(key)}]`, value, fail)]),
  );

  const id = own(raw, "id");
  const provider = own(raw, "provider");
  const usage = parseUsage(own(raw, "usage"));
  return {
    ...(typeof id === "string" ? { id } : {}),
    model,
    ...(typeof provider === "string" ? { provider } : {}),
    answers: parsedAnswers,
    ...(usage === undefined ? {} : { usage }),
  };
}

type Fail = (problem: string) => JevProviderError;

function parseAnswer(path: string, value: unknown, fail: Fail): NativeJevAnswer {
  if (!isRecord(value)) throw fail(`${path} must be an object, got ${describe(value)}`);
  const type = own(value, "type");
  switch (type) {
    case "noul": {
      // Extra fields some builds send for noul are dropped; the type has only `noul`.
      return { type: "noul", noul: requireFinite(`${path}.noul`, own(value, "noul"), fail) };
    }
    case "choice": {
      const choice = own(value, "choice");
      if (typeof choice !== "string") throw fail(`${path}.choice must be a string, got ${describe(choice)}`);
      const probabilities = optionalNumberMap(`${path}.probabilities`, own(value, "probabilities"), fail);
      const confidence = optionalFinite(`${path}.confidence`, own(value, "confidence"), fail);
      return {
        type: "choice",
        choice,
        ...(probabilities === undefined ? {} : { probabilities }),
        ...(confidence === undefined ? {} : { confidence }),
      };
    }
    case "score": {
      const score = requireFinite(`${path}.score`, own(value, "score"), fail);
      const probabilities = optionalNumberMap(`${path}.probabilities`, own(value, "probabilities"), fail);
      const legend = optionalLegend(`${path}.legend`, own(value, "legend"), fail);
      const confidence = optionalFinite(`${path}.confidence`, own(value, "confidence"), fail);
      return {
        type: "score",
        score,
        ...(probabilities === undefined ? {} : { probabilities }),
        ...(legend === undefined ? {} : { legend }),
        ...(confidence === undefined ? {} : { confidence }),
      };
    }
    default:
      throw fail(`${path}.type must be "choice", "score", or "noul", got ${describeType(type)}`);
  }
}

function requireFinite(path: string, value: unknown, fail: Fail): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw fail(`${path} must be a finite number, got ${describe(value)}`);
  }
  return value;
}

/** The wire schema marks these fields optional; null is read as absent. */
function optionalFinite(path: string, value: unknown, fail: Fail): number | undefined {
  return value === undefined || value === null ? undefined : requireFinite(path, value, fail);
}

function optionalNumberMap(path: string, value: unknown, fail: Fail): Record<string, number> | undefined {
  if (value === undefined || value === null) return undefined;
  if (!isRecord(value)) throw fail(`${path} must be an object, got ${describe(value)}`);
  return Object.fromEntries(
    Object.entries(value).map(([key, entry]) => [key, requireFinite(`${path}[${quote(key)}]`, entry, fail)]),
  );
}

function optionalLegend(path: string, value: unknown, fail: Fail): Record<string, JevCriterion> | undefined {
  if (value === undefined || value === null) return undefined;
  if (!isRecord(value)) throw fail(`${path} must be an object, got ${describe(value)}`);
  return Object.fromEntries(
    Object.entries(value).map(([key, entry]): [string, JevCriterion] => {
      if (!isCriterion(entry)) {
        throw fail(`${path}[${quote(key)}] must be a string, object, or array, got ${describe(entry)}`);
      }
      return [key, entry];
    }),
  );
}

/**
 * Accepts snake_case (`input_tokens`, `output_tokens`, `cost`) and camelCase
 * (`inputTokens`, `outputTokens`, `costUsd`) usage. Usage is metadata, so a
 * malformed field is dropped rather than failing an otherwise valid decision.
 */
function parseUsage(value: unknown): NativeJevUsage | undefined {
  if (!isRecord(value)) return undefined;
  const inputTokens = firstNonNegative(value, ["input_tokens", "inputTokens"]);
  const outputTokens = firstNonNegative(value, ["output_tokens", "outputTokens"]);
  const costUsd = firstNonNegative(value, ["cost", "costUsd", "cost_usd"]);
  if (inputTokens === undefined && outputTokens === undefined && costUsd === undefined) return undefined;
  return {
    ...(inputTokens === undefined ? {} : { inputTokens }),
    ...(outputTokens === undefined ? {} : { outputTokens }),
    ...(costUsd === undefined ? {} : { costUsd }),
  };
}

function firstNonNegative(record: Record<string, unknown>, keys: readonly string[]): number | undefined {
  for (const key of keys) {
    const value = own(record, key);
    if (typeof value === "number" && Number.isFinite(value) && value >= 0) return value;
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// Construction-time option helpers shared by the built-in transports
// ---------------------------------------------------------------------------

/** Reads an environment variable without assuming a Node runtime. */
function readEnv(name: string): string | undefined {
  return typeof process === "undefined" ? undefined : process.env[name];
}

/** Explicit key, else the environment variable; throws when neither is usable. */
export function resolveApiKey(explicit: unknown, envVar: string, providerLabel: string): string {
  if (explicit !== undefined) {
    if (typeof explicit !== "string" || explicit.trim() === "") {
      throw new JevValidationError(`${providerLabel}: apiKey must be a non-empty string`);
    }
    return explicit;
  }
  const fromEnv = readEnv(envVar);
  if (fromEnv === undefined || fromEnv.trim() === "") {
    throw new JevValidationError(
      `${providerLabel}: no API key. Pass provider.apiKey (or the transport's apiKey option) or set ${envVar}.`,
    );
  }
  return fromEnv;
}

/** Validates an http(s) base URL and strips trailing slashes so paths can be appended. */
export function normalizeBaseUrl(value: unknown, fallback: string, providerLabel: string): string {
  const raw = value ?? fallback;
  if (typeof raw !== "string") throw new JevValidationError(`${providerLabel}: baseUrl must be a string`);
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    throw new JevValidationError(`${providerLabel}: baseUrl is not a valid URL`);
  }
  if (url.protocol !== "https:" && url.protocol !== "http:") {
    throw new JevValidationError(`${providerLabel}: baseUrl must use http or https`);
  }
  if (url.search !== "" || url.hash !== "") {
    throw new JevValidationError(`${providerLabel}: baseUrl must not contain a query string or fragment`);
  }
  // fetch refuses URLs with embedded credentials; fail at construction instead.
  if (url.username !== "" || url.password !== "") {
    throw new JevValidationError(`${providerLabel}: baseUrl must not contain credentials`);
  }
  return url.href.replace(/\/+$/, "");
}

/** Calls the global fetch at request time, so a later-installed global is honored. */
const globalFetch: typeof fetch = (input, init) => globalThis.fetch(input, init);

export function resolveFetch(value: unknown, providerLabel: string): typeof fetch {
  if (value !== undefined) {
    if (typeof value !== "function") throw new JevValidationError(`${providerLabel}: fetch must be a function`);
    return value as typeof fetch;
  }
  if (typeof globalThis.fetch !== "function") {
    throw new JevValidationError(`${providerLabel}: no global fetch is available; pass the fetch option`);
  }
  return globalFetch;
}

/** A timeout in milliseconds that setTimeout can honor. */
export function resolveTimeoutMs(value: unknown, fallback: number, field: string, providerLabel: string): number {
  if (value === undefined) return fallback;
  if (!isValidTimerDelay(value)) {
    throw new JevValidationError(`${providerLabel}: ${field} must be a number of milliseconds in (0, ${MAX_TIMER_MS}]`);
  }
  return value;
}

function isValidTimerDelay(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 && value <= MAX_TIMER_MS;
}

/** An explicit context window: a positive safe integer, or undefined. */
export function resolveContextWindowOption(value: unknown, providerLabel: string): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0) {
    throw new JevValidationError(`${providerLabel}: contextWindow must be a positive integer`);
  }
  return value;
}

export function optionalString(value: unknown, field: string, providerLabel: string): string | undefined {
  if (value !== undefined && typeof value !== "string") {
    throw new JevValidationError(`${providerLabel}: ${field} must be a string`);
  }
  return value;
}

/**
 * Merges caller headers under the transport's fixed headers. Header names
 * are case-insensitive, so merging goes through `Headers`: a caller's
 * "authorization" can never duplicate or override the transport's key.
 * Invalid names or values fail at construction, not on the first request.
 */
export function mergeHeaders(
  extra: unknown,
  fixed: Readonly<Record<string, string | undefined>>,
  providerLabel: string,
): Record<string, string> {
  if (extra !== undefined && !isRecord(extra)) {
    throw new JevValidationError(`${providerLabel}: headers must be an object of strings`);
  }
  const headers = new Headers();
  try {
    for (const [name, value] of Object.entries(extra ?? {})) {
      if (typeof value !== "string") {
        throw new JevValidationError(`${providerLabel}: headers[${quote(name)}] must be a string`);
      }
      headers.set(name, value);
    }
    for (const [name, value] of Object.entries(fixed)) {
      if (value !== undefined) headers.set(name, value);
    }
  } catch (error) {
    if (error instanceof JevValidationError) throw error;
    // The platform message may quote the offending value (possibly the API
    // key), so it is kept only on `cause`.
    throw new JevValidationError(`${providerLabel}: a header name or value is not valid HTTP`, { cause: error });
  }
  // forEach rather than entries(): the tsconfig lib has DOM but not DOM.Iterable.
  const merged: Array<[string, string]> = [];
  headers.forEach((value, name) => merged.push([name, value]));
  return Object.fromEntries(merged);
}

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Own-property read, so a polluted Object.prototype cannot supply wire fields. */
export function own(record: Readonly<Record<string, unknown>>, key: string): unknown {
  return Object.hasOwn(record, key) ? record[key] : undefined;
}

function isCriterion(value: unknown): value is JevCriterion {
  return typeof value === "string" || isRecord(value) || Array.isArray(value);
}

function quote(key: string): string {
  return JSON.stringify(key.length > 64 ? `${key.slice(0, 64)}…` : key);
}

/** Short type description for error messages; never echoes string contents. */
function describe(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "an array";
  if (typeof value === "string") return `a string of length ${value.length}`;
  if (typeof value === "number") return Number.isFinite(value) ? "a number" : String(value);
  return typeof value;
}

/** A `type` discriminator is a short protocol token, so it is quoted when it is a string. */
function describeType(value: unknown): string {
  return typeof value === "string" ? quote(value) : describe(value);
}
