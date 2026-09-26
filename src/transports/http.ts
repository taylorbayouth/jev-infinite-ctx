/**
 * Shared HTTP plumbing for the built-in Jev transports (spec 13): request
 * encoding, deadline and abort handling, provider error classification, and
 * strict parsing of the Decisions wire format into `NativeJevResponse`.
 *
 * Invariant: no error message created here quotes provider text for a request
 * that carried state, so messages never contain it (spec 16: never log source
 * text). An error's `body` keeps provider text only when it shares no run of
 * ECHO_RUN characters with the state; otherwise the whole body is withheld.
 */

import {
  JevAbortError,
  JevProviderError,
  JevValidationError,
  type ProviderErrorKind,
} from "../errors.js";
import { isRecord, MAX_TIMER_DELAY_MS, own, quote, raceAbort } from "../internal.js";
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
  const state = requestState(body);
  const { response, text } = await fetchText(
    config.fetch,
    config.url,
    { method: "POST", headers: config.headers, body: JSON.stringify(body) },
    { timeoutMs: config.timeoutMs, providerName: config.providerName, signal },
  );

  if (!response.ok) {
    throw classifyHttpError(response.status, text, response.headers, config.providerName, state);
  }

  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch {
    // The SyntaxError is dropped rather than kept as `cause`: its message
    // quotes the body (all of a short one, a window of a long one), which can
    // echo the state, and loggers print the cause chain. The body itself is
    // on `body`, withheld when it echoes the state.
    throw new JevProviderError(
      `${config.providerName} returned a response body that is not valid JSON (HTTP ${response.status})`,
      {
        kind: "invalid_response",
        status: response.status,
        retryable: false,
        body: errorBody(text, state),
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
        state,
      );
    }
  }

  return parseNativeResponse(json, config.providerName, requestLabels(body));
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
    throw new JevValidationError(`${providerName}: timeoutMs must be a number of milliseconds in (0, ${MAX_TIMER_DELAY_MS}]`);
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
 * numeric code. `state` is the request state. When it is non-empty, the
 * message names only the provider, status, and kind, and `body` is withheld
 * if it echoes the state (see `errorBody`).
 */
export function classifyHttpError(
  status: number | undefined,
  bodyText: string,
  headers: Headers | undefined,
  providerName: string,
  state?: string,
): JevProviderError {
  const scanned = bodyText.length > MAX_SCAN_CHARS ? bodyText.slice(0, MAX_SCAN_CHARS) : bodyText;
  // The kind reads the scanned body, echoes of the state included. An echoed
  // document with context-limit wording can make a 400 or 422 a spurious
  // context_limit (a re-chunk that still fails closed, with
  // JevContextBudgetError at worst). One mentioning "overload" can make a 5xx
  // or a code-less 2xx error `overloaded` (retryable either way for a 5xx).
  const kind = kindForStatus(status, scanned);
  // Provider text can quote any fragment of the state, including ones too
  // short to find, and messages are what callers log (JevChunkFailedError
  // quotes this one).
  const detail = state === undefined || state === "" ? quotedDetail(providerMessage(scanned)) : undefined;
  const head =
    status === undefined
      ? `${providerName} returned an error (${kind})`
      : `${providerName} request failed with HTTP ${status} (${kind})`;
  return new JevProviderError(detail === undefined ? head : `${head}: ${detail}`, {
    kind,
    status,
    retryAfterMs: parseRetryAfter(headers?.get("retry-after")),
    body: errorBody(bodyText, state),
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
  if (status === 503 || status === 529) return "overloaded";
  // Only a 5xx is read for "overload": an unlisted 4xx is a client error, and
  // an echo of the state must not make it retryable.
  if (status >= 500 && status <= 599) return OVERLOAD_PATTERN.test(text) ? "overloaded" : "server";
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
 * Each pattern is tied to context, tokens, length, or size (or a large count
 * of tokens or characters) so that generic token errors ("invalid token",
 * "token expired") are not matched.
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
  // "state exceeds 32000 tokens", "input exceeded 128,000 characters". The size
  // must be at least 1,000 (so field limits such as "64 characters" are not
  // matched), and output-token limits and rate limits are excluded.
  /(?<!\b(?:max_(?:completion_)?tokens|rate[\s_-]*limit)\b[^.;\n]{0,40})\bexceed(?:s|ed|ing)?\b[^.;\n]{0,40}?(?<![\d,])(?:\d{1,3}(?:,\d{3})+|\d{4,})\s*(?:tokens?|characters?|chars)\b(?!\s*(?:per|\/)\s*(?:min|minute|sec|second|hour|day)\b)/i,
  // Max-length validators on the state: pydantic v2 "String should have at
  // most 128000 characters", pydantic v1 "ensure this value has at most ...".
  /(?<!\bmax_(?:completion_)?tokens\b[^.;\n]{0,40})\bat most\s+(?:\d{1,3}(?:,\d{3})+|\d{4,})\s*(?:tokens?|characters?|chars)\b/i,
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
// Error bodies and state echoes (spec 16)
// ---------------------------------------------------------------------------

/** Kept on `body` in place of a provider body that echoes the request state. */
const WITHHELD_BODY = "[withheld: provider error body overlaps the request state]";
/**
 * Shortest run of characters shared with the state that withholds a body.
 * Shorter fragments cannot be told apart from ordinary words, which is why
 * messages never quote provider text for a request that carried state.
 */
const ECHO_RUN = 8;
/**
 * Characters past the cut that are also checked for echoes, so an echo cut
 * short by the truncation is still found. ECHO_RUN characters escaped twice
 * as \uXXXX (7 characters each) fit, and so do three levels (9 each).
 */
const ECHO_CUT_SLACK = 128;
/**
 * Levels of JSON string escaping undone when looking for echoes: one for a
 * state quoted in a JSON string, two for a JSON body nested in a JSON string
 * (as in OpenRouter's `metadata.raw`), and one spare.
 */
const MAX_UNESCAPE_LEVELS = 3;
/** Longest prefix of a provider body that is classified and searched for a message to quote. */
const MAX_SCAN_CHARS = 1_000_000;
/** One JSON string escape: \uXXXX (hex in either case) or a one-character escape. */
const JSON_ESCAPE = /\\(?:u[0-9a-fA-F]{4}|["\\/bfnrt])/g;

/** The text an error built for this body must not echo; undefined when the request carried no state. */
function requestState(body: JsonObject): string | undefined {
  const state = own(body, "state");
  if (state === undefined || state === "") return undefined;
  return typeof state === "string" ? state : JSON.stringify(state);
}

/**
 * Collapses whitespace and truncates a provider message; undefined when
 * empty. The message is cut before whitespace is collapsed, so the cost does
 * not grow with the body.
 */
function quotedDetail(text: string): string | undefined {
  const head = truncate(text.trimStart(), 4 * MAX_ERROR_DETAIL_CHARS);
  const cleaned = truncate(collapseWhitespace(head).trim(), MAX_ERROR_DETAIL_CHARS);
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
 * The provider body kept on an error: truncated to MAX_ERROR_BODY_CHARS and,
 * for a request that carried state, replaced as a whole by WITHHELD_BODY when
 * it echoes the state (fail-closed: nothing is partially redacted). The check
 * also reads ECHO_CUT_SLACK characters past the cut, so the truncation cannot
 * leave an echo too short to find. Undefined for an empty body.
 */
function errorBody(bodyText: string, state: string | undefined): string | undefined {
  const kept = truncate(bodyText, MAX_ERROR_BODY_CHARS);
  if (kept === "") return undefined;
  if (state === undefined || state === "") return kept;
  return echoesState(bodyText.slice(0, MAX_ERROR_BODY_CHARS + ECHO_CUT_SLACK), state) ? WITHHELD_BODY : kept;
}

/**
 * True when a run of ECHO_RUN characters of `text`, as sent or with up to
 * MAX_UNESCAPE_LEVELS levels of JSON string escapes decoded, also occurs in
 * `state`, with whitespace collapsed to one space in all of them. A state
 * shorter than ECHO_RUN matches when it occurs whole. The text's runs go in a
 * set and the state is scanned once, so the cost is
 * O(MAX_UNESCAPE_LEVELS * |text| + |state|).
 */
function echoesState(text: string, state: string): boolean {
  const target = collapseWhitespace(state);
  const decoded = [text];
  for (let level = 1; level <= MAX_UNESCAPE_LEVELS; level += 1) {
    const previous = decoded[decoded.length - 1]!;
    const next = decodeJsonEscapes(previous);
    if (next === previous) break;
    decoded.push(next);
  }
  const views = decoded.map(collapseWhitespace);
  if (target.length < ECHO_RUN) return views.some((view) => view.includes(target));
  const runs = new Set<string>();
  for (const view of views) {
    for (let i = 0; i + ECHO_RUN <= view.length; i += 1) runs.add(view.slice(i, i + ECHO_RUN));
  }
  for (let i = 0; i + ECHO_RUN <= target.length; i += 1) {
    if (runs.has(target.slice(i, i + ECHO_RUN))) return true;
  }
  return false;
}

function collapseWhitespace(text: string): string {
  return text.replace(/\s+/g, " ");
}

/**
 * Undoes one level of JSON string escaping, as applied when a provider quotes
 * the state in a JSON message. Each \u unit is decoded on its own, so an
 * escaped surrogate pair becomes the original character again.
 */
function decodeJsonEscapes(text: string): string {
  return text.replace(JSON_ESCAPE, (escape) => JSON.parse(`"${escape}"`) as string);
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

/**
 * Validates a decoded Decisions response and normalizes it to
 * `NativeJevResponse` (camelCase usage). Structural only: probability ranges
 * and option membership are checked later against the question (spec 8).
 * Throws `JevProviderError` kind "invalid_response" (not retryable).
 *
 * `known` holds the strings the request itself sent as keys (see
 * `requestLabels`). A message quotes a response key only when it is one of
 * them: the provider may have copied any other string from the state.
 */
export function parseNativeResponse(
  raw: unknown,
  providerName: string,
  known: ReadonlySet<string> = new Set(),
): NativeJevResponse {
  const fail = (problem: string): JevProviderError =>
    new JevProviderError(`${providerName} returned an invalid decision response: ${problem}`, {
      kind: "invalid_response",
      retryable: false,
    });
  const keyName: KeyName = (key) => (known.has(key) ? quote(key) : `<unknown key of length ${key.length}>`);

  if (!isRecord(raw)) throw fail(`expected a JSON object, got ${describe(raw)}`);

  const model = own(raw, "model");
  if (typeof model !== "string" || model === "") throw fail("`model` must be a non-empty string");

  const answers = own(raw, "answers");
  if (!isRecord(answers)) throw fail(`\`answers\` must be an object, got ${describe(answers)}`);

  const parsedAnswers: Record<string, NativeJevAnswer> = Object.fromEntries(
    Object.entries(answers).map(([key, value]) => [key, parseAnswer(`answers[${keyName(key)}]`, value, fail, keyName)]),
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
/** Renders a response key for an error path. */
type KeyName = (key: string) => string;

/**
 * The strings a request sent as keys: its question keys, the keys of object
 * criteria (choice options, and "true" and "false" for noul), and the level
 * indices of array criteria (score). A response key among them is the
 * caller's own text, so an error message may quote it.
 */
function requestLabels(body: JsonObject): Set<string> {
  const labels = new Set<string>();
  const questions = own(body, "questions");
  if (!isRecord(questions)) return labels;
  for (const [key, question] of Object.entries(questions)) {
    labels.add(key);
    const criteria = isRecord(question) ? own(question, "criteria") : undefined;
    if (isRecord(criteria)) {
      for (const label of Object.keys(criteria)) labels.add(label);
    } else if (Array.isArray(criteria)) {
      for (let level = 0; level < criteria.length; level += 1) labels.add(String(level));
    }
  }
  return labels;
}

function parseAnswer(path: string, value: unknown, fail: Fail, keyName: KeyName): NativeJevAnswer {
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
      const probabilities = optionalNumberMap(`${path}.probabilities`, own(value, "probabilities"), fail, keyName);
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
      const probabilities = optionalNumberMap(`${path}.probabilities`, own(value, "probabilities"), fail, keyName);
      const legend = optionalLegend(`${path}.legend`, own(value, "legend"), fail, keyName);
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
      // Any other type string came from the provider, so only its length is shown.
      throw fail(`${path}.type must be "choice", "score", or "noul", got ${describe(type)}`);
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

function optionalNumberMap(
  path: string,
  value: unknown,
  fail: Fail,
  keyName: KeyName,
): Record<string, number> | undefined {
  if (value === undefined || value === null) return undefined;
  if (!isRecord(value)) throw fail(`${path} must be an object, got ${describe(value)}`);
  return Object.fromEntries(
    Object.entries(value).map(([key, entry]) => [key, requireFinite(`${path}[${keyName(key)}]`, entry, fail)]),
  );
}

function optionalLegend(
  path: string,
  value: unknown,
  fail: Fail,
  keyName: KeyName,
): Record<string, JevCriterion> | undefined {
  if (value === undefined || value === null) return undefined;
  if (!isRecord(value)) throw fail(`${path} must be an object, got ${describe(value)}`);
  return Object.fromEntries(
    Object.entries(value).map(([key, entry]): [string, JevCriterion] => {
      if (!isCriterion(entry)) {
        throw fail(`${path}[${keyName(key)}] must be a string, object, or array, got ${describe(entry)}`);
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
    throw new JevValidationError(`${providerLabel}: ${field} must be a number of milliseconds in (0, ${MAX_TIMER_DELAY_MS}]`);
  }
  return value;
}

function isValidTimerDelay(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 && value <= MAX_TIMER_DELAY_MS;
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
  for (const [name, value] of Object.entries(extra ?? {})) {
    if (typeof value !== "string") {
      throw new JevValidationError(`${providerLabel}: headers[${quote(name)}] must be a string`);
    }
    if (!trySetHeader(headers, name, value)) {
      throw new JevValidationError(`${providerLabel}: headers[${quote(name)}] is not a valid HTTP header name and value`);
    }
  }
  for (const [name, value] of Object.entries(fixed)) {
    if (value !== undefined && !trySetHeader(headers, name, value)) {
      const what = name.toLowerCase() === "authorization" ? "the API key" : `the ${name} header value`;
      throw new JevValidationError(
        `${providerLabel}: ${what} contains a character HTTP headers do not allow (such as CR, LF, or NUL)`,
      );
    }
  }
  // forEach rather than entries(): the tsconfig lib has DOM but not DOM.Iterable.
  const merged: Array<[string, string]> = [];
  headers.forEach((value, name) => merged.push([name, value]));
  return Object.fromEntries(merged);
}

/**
 * Sets one header; false when the platform rejects the name or value. The
 * platform error is dropped rather than kept as `cause`: its message quotes
 * the value, which may be the API key, and loggers print the cause chain.
 */
function trySetHeader(headers: Headers, name: string, value: string): boolean {
  try {
    headers.set(name, value);
    return true;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

function isCriterion(value: unknown): value is JevCriterion {
  return typeof value === "string" || isRecord(value) || Array.isArray(value);
}

/** Short type description for error messages; never echoes string contents. */
function describe(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "an array";
  if (typeof value === "string") return `a string of length ${value.length}`;
  if (typeof value === "number") return Number.isFinite(value) ? "a number" : String(value);
  return typeof value;
}

/** The caller's question `type` (a short protocol token), quoted when it is a string. */
function describeType(value: unknown): string {
  return typeof value === "string" ? quote(value) : describe(value);
}
