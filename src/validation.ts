/**
 * Request validation and option resolution (spec 5, 14, 15).
 *
 * Everything is checked before any provider call so a bad request fails
 * loudly with `JevValidationError` instead of spending Jev calls or being
 * silently coerced. Every message names the offending field path (for
 * example `chunking.overlap` or `question.criteria.tree`). Messages never
 * echo `input`: it is source text (spec 16).
 */

import {
  DEFAULTS,
  MAX_OVERLAP,
  MAX_SCORE_LEVELS,
  MIN_SCORE_LEVELS,
  MIN_STATE_TOKENS,
} from "./defaults.js";
import { JevValidationError } from "./errors.js";
import type {
  AggregationMethod,
  BuiltInTransportName,
  ChunkingOptions,
  ConfidenceOptions,
  ExecutionOptions,
  JevInfiniteCTXRequest,
  JevQuestion,
  JevQuestionType,
  ProviderOptions,
  ResolvedOptions,
} from "./types.js";

type UnknownRecord = Record<string, unknown>;

/**
 * Allowed field names per object. Typed as `Record<keyof X, true>` so the
 * compiler flags any drift from src/types.ts in either direction.
 */
const REQUEST_FIELDS: Record<keyof JevInfiniteCTXRequest, true> = {
  input: true,
  question: true,
  aggregation: true,
  chunking: true,
  confidence: true,
  execution: true,
  provider: true,
  tokenizer: true,
  signal: true,
  onEvent: true,
};
const QUESTION_FIELDS: Record<"type" | "instructions" | "criteria", true> = {
  type: true,
  instructions: true,
  criteria: true,
};
const CHUNKING_FIELDS: Record<keyof ChunkingOptions, true> = {
  overlap: true,
  maxStateTokens: true,
  contextSafetyReserve: true,
  protocolReserve: true,
  preferNaturalBoundaries: true,
  maxChunks: true,
};
const EXECUTION_FIELDS: Record<keyof ExecutionOptions, true> = {
  maxConcurrency: true,
  retries: true,
  retryBaseDelayMs: true,
  retryMaxDelayMs: true,
  maxRechunks: true,
  rechunkShrinkFactor: true,
};
const CONFIDENCE_FIELDS: Record<keyof ConfidenceOptions, true> = {
  method: true,
  cap: true,
  lambda: true,
  agreementFloor: true,
  agreementExponent: true,
};
const PROVIDER_FIELDS: Record<keyof ProviderOptions, true> = {
  transport: true,
  model: true,
  apiKey: true,
};

const QUESTION_TYPES: readonly JevQuestionType[] = ["choice", "score", "noul"];
const AGGREGATION_METHODS: readonly AggregationMethod[] = [
  "weighted_mean",
  "mean",
  "median",
  "min",
  "max",
];
const CONFIDENCE_METHODS: readonly Required<ConfidenceOptions>["method"][] = ["c3", "none"];
const BUILT_IN_TRANSPORTS: readonly BuiltInTransportName[] = ["openrouter", "direct"];

/** Upper bound for chunking.contextSafetyReserve (types.ts: range [0, 0.5]). */
const MAX_CONTEXT_SAFETY_RESERVE = 0.5;

/** Longest string echoed back in an error message. */
const MAX_ECHO_LENGTH = 64;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Validates the top-level request: `input`, `question`, `tokenizer`,
 * `onEvent`, `signal`, `provider`, and every option section (via
 * `resolveOptions`). Unknown fields are rejected so typos never silently
 * fall back to defaults. Returns the same object, typed.
 */
export function validateRequest(request: unknown): JevInfiniteCTXRequest {
  return validateRequestAndResolve(request).request;
}

/**
 * Same checks as `validateRequest`, and also returns the resolved options.
 * Option sections are read exactly once, so callers that need both (decide)
 * neither repeat the work nor observe a caller's option getters twice.
 */
export function validateRequestAndResolve(request: unknown): {
  request: JevInfiniteCTXRequest;
  options: ResolvedOptions;
} {
  const req = requireObject(request, "request");
  rejectUnknownFields(req, REQUEST_FIELDS, "");

  const input = req["input"];
  if (typeof input !== "string") {
    fail(`input must be a string, got ${describeValue(input)}.`);
  }
  // spec 15: empty input is rejected before calling Jev. The input itself is
  // never echoed; it is source text.
  if (!hasNonWhitespace(input)) {
    fail("input must contain at least one non-whitespace character; empty input is rejected.");
  }

  validateQuestion(req["question"]);
  checkTokenizer(req["tokenizer"]);
  checkOnEvent(req["onEvent"]);
  checkSignal(req["signal"]);
  checkProvider(req["provider"]);

  const typed = req as unknown as JevInfiniteCTXRequest;
  return { request: typed, options: resolveOptions(typed) };
}

/**
 * Validates a question (spec 4, 5.2) and returns it, typed. The question is
 * sent to Jev with every chunk, so it must survive JSON serialization; fields
 * are read as own enumerable properties because those are exactly what
 * `JSON.stringify` sends.
 */
export function validateQuestion(question: unknown): JevQuestion {
  const q = requireObject(question, "question");
  const type = checkEnum(ownValue(q, "type"), "question.type", QUESTION_TYPES);
  rejectUnknownFields(q, QUESTION_FIELDS, "question");
  checkInstructions(ownValue(q, "instructions"));

  const criteria = ownValue(q, "criteria");
  switch (type) {
    case "choice":
      checkChoiceCriteria(criteria);
      break;
    case "score":
      checkScoreCriteria(criteria);
      break;
    case "noul":
      // Noul criteria are optional; when given, both descriptions are required.
      if (criteria !== undefined) {
        checkNoulCriteria(criteria);
      }
      break;
  }
  return q as unknown as JevQuestion;
}

/**
 * Validates the option sections of a request and fills in DEFAULTS
 * (spec 14). Returns fresh objects; the request is not modified.
 */
export function resolveOptions(request: JevInfiniteCTXRequest): ResolvedOptions {
  const req = requireObject(request, "request");
  const aggregation = withDefault(req["aggregation"], DEFAULTS.aggregation, (v) =>
    checkEnum(v, "aggregation", AGGREGATION_METHODS),
  );

  const chunking = section(req["chunking"], "chunking", CHUNKING_FIELDS);
  const execution = section(req["execution"], "execution", EXECUTION_FIELDS);
  const confidence = section(req["confidence"], "confidence", CONFIDENCE_FIELDS);

  return {
    aggregation,
    chunking: {
      overlap: withDefault(chunking["overlap"], DEFAULTS.chunking.overlap, (v) =>
        checkNumber(v, "chunking.overlap", { min: 0, max: MAX_OVERLAP }),
      ),
      maxStateTokens: withDefault(
        chunking["maxStateTokens"],
        DEFAULTS.chunking.maxStateTokens,
        checkMaxStateTokens,
      ),
      contextSafetyReserve: withDefault(
        chunking["contextSafetyReserve"],
        DEFAULTS.chunking.contextSafetyReserve,
        (v) =>
          checkNumber(v, "chunking.contextSafetyReserve", {
            min: 0,
            max: MAX_CONTEXT_SAFETY_RESERVE,
          }),
      ),
      protocolReserve: withDefault(
        chunking["protocolReserve"],
        DEFAULTS.chunking.protocolReserve,
        (v) => checkNumber(v, "chunking.protocolReserve", { min: 0, integer: true }),
      ),
      preferNaturalBoundaries: withDefault(
        chunking["preferNaturalBoundaries"],
        DEFAULTS.chunking.preferNaturalBoundaries,
        (v) => checkBoolean(v, "chunking.preferNaturalBoundaries"),
      ),
      maxChunks: withDefault<number | undefined>(
        chunking["maxChunks"],
        DEFAULTS.chunking.maxChunks,
        (v) => checkNumber(v, "chunking.maxChunks", { min: 1, integer: true }),
      ),
    },
    execution: {
      maxConcurrency: withDefault(
        execution["maxConcurrency"],
        DEFAULTS.execution.maxConcurrency,
        (v) => checkNumber(v, "execution.maxConcurrency", { min: 1, integer: true }),
      ),
      retries: withDefault(execution["retries"], DEFAULTS.execution.retries, (v) =>
        checkNumber(v, "execution.retries", { min: 0, integer: true }),
      ),
      retryBaseDelayMs: withDefault(
        execution["retryBaseDelayMs"],
        DEFAULTS.execution.retryBaseDelayMs,
        (v) => checkNumber(v, "execution.retryBaseDelayMs", { min: 0 }),
      ),
      retryMaxDelayMs: withDefault(
        execution["retryMaxDelayMs"],
        DEFAULTS.execution.retryMaxDelayMs,
        (v) => checkNumber(v, "execution.retryMaxDelayMs", { min: 0 }),
      ),
      maxRechunks: withDefault(execution["maxRechunks"], DEFAULTS.execution.maxRechunks, (v) =>
        checkNumber(v, "execution.maxRechunks", { min: 0, integer: true }),
      ),
      rechunkShrinkFactor: withDefault(
        execution["rechunkShrinkFactor"],
        DEFAULTS.execution.rechunkShrinkFactor,
        (v) =>
          checkNumber(v, "execution.rechunkShrinkFactor", {
            min: 0,
            minExclusive: true,
            max: 1,
            maxExclusive: true,
          }),
      ),
    },
    confidence: {
      method: withDefault(confidence["method"], DEFAULTS.confidence.method, (v) =>
        checkEnum(v, "confidence.method", CONFIDENCE_METHODS),
      ),
      cap: withDefault(confidence["cap"], DEFAULTS.confidence.cap, (v) =>
        checkNumber(v, "confidence.cap", { min: 0, minExclusive: true, max: 1 }),
      ),
      lambda: withDefault(confidence["lambda"], DEFAULTS.confidence.lambda, (v) =>
        checkNumber(v, "confidence.lambda", { min: 0, minExclusive: true }),
      ),
      agreementFloor: withDefault(
        confidence["agreementFloor"],
        DEFAULTS.confidence.agreementFloor,
        (v) =>
          checkNumber(v, "confidence.agreementFloor", { min: 0, max: 1, maxExclusive: true }),
      ),
      agreementExponent: withDefault(
        confidence["agreementExponent"],
        DEFAULTS.confidence.agreementExponent,
        (v) => checkNumber(v, "confidence.agreementExponent", { min: 0, minExclusive: true }),
      ),
    },
  };
}

// ---------------------------------------------------------------------------
// Question parts
// ---------------------------------------------------------------------------

function checkInstructions(value: unknown): void {
  const path = "question.instructions";
  if (typeof value === "string") {
    if (!hasNonWhitespace(value)) {
      fail(`${path} must be a non-empty string, got an empty or whitespace-only string.`);
    }
    return;
  }
  if (typeof value !== "object" || value === null) {
    fail(`${path} must be a non-empty string, an object, or an array, got ${describeValue(value)}.`);
  }
  checkSerializable(value, path);
}

function checkChoiceCriteria(criteria: unknown): void {
  if (!isPlainObject(criteria)) {
    fail(
      `question.criteria must be a plain object mapping option keys to criteria for a "choice" question, got ${describeValue(criteria)}.`,
    );
  }
  const keys = Object.keys(criteria);
  if (keys.length < 2) {
    fail(`question.criteria must define at least 2 options for a "choice" question, got ${keys.length}.`);
  }
  for (const key of keys) {
    // `key` comes from Object.keys, so this reads the own property even for "__proto__".
    checkCriterion(criteria[key], `question.criteria${pathSegment(key)}`);
  }
}

function checkScoreCriteria(criteria: unknown): void {
  if (!Array.isArray(criteria)) {
    fail(
      `question.criteria must be an array of rubric levels (lowest first) for a "score" question, got ${describeValue(criteria)}.`,
    );
  }
  if (criteria.length < MIN_SCORE_LEVELS || criteria.length > MAX_SCORE_LEVELS) {
    fail(
      `question.criteria must have between ${MIN_SCORE_LEVELS} and ${MAX_SCORE_LEVELS} levels for a "score" question, got ${criteria.length}.`,
    );
  }
  // Index loop (not forEach) so holes in a sparse array are visited and rejected.
  for (let level = 0; level < criteria.length; level++) {
    checkCriterion(criteria[level], `question.criteria[${level}]`);
  }
}

function checkNoulCriteria(criteria: unknown): void {
  if (!isPlainObject(criteria)) {
    fail(
      `question.criteria must be an object with "true" and "false" criteria for a "noul" question, got ${describeValue(criteria)}.`,
    );
  }
  const keys = Object.keys(criteria);
  if (keys.length !== 2 || !Object.hasOwn(criteria, "true") || !Object.hasOwn(criteria, "false")) {
    fail(
      `question.criteria must have exactly the keys "true" and "false" for a "noul" question, got ${keys.map((k) => JSON.stringify(truncate(k))).join(", ") || "no keys"}.`,
    );
  }
  checkCriterion(criteria["true"], "question.criteria.true");
  checkCriterion(criteria["false"], "question.criteria.false");
}

/**
 * A criterion is a non-empty string or a structured JSON object/array
 * (types.ts JevCriterion). A blank string would send Jev a meaningless
 * option or rubric level, so it is rejected like blank instructions.
 */
function checkCriterion(value: unknown, path: string): void {
  if (typeof value === "string") {
    if (!hasNonWhitespace(value)) {
      fail(`${path} must be a non-empty string, got an empty or whitespace-only string.`);
    }
    return;
  }
  if (typeof value !== "object" || value === null) {
    fail(`${path} must be a string, an object, or an array, got ${describeValue(value)}.`);
  }
  checkSerializable(value, path);
}

/**
 * The question travels to Jev as JSON, so every part must serialize. This
 * catches cycles, BigInt values, and throwing `toJSON` methods, and names
 * the field that failed.
 */
function checkSerializable(value: unknown, path: string): void {
  let json: string | undefined;
  try {
    json = JSON.stringify(value);
  } catch (error) {
    const reason = error instanceof Error ? `: ${error.message}` : "";
    fail(`${path} must be JSON-serializable${reason}.`, error);
  }
  // A toJSON() returning undefined would make the field vanish from the request.
  if (json === undefined) {
    fail(`${path} must be JSON-serializable, but it serializes to nothing.`);
  }
}

// ---------------------------------------------------------------------------
// Request parts
// ---------------------------------------------------------------------------

function checkTokenizer(value: unknown): void {
  if (value === undefined) {
    return;
  }
  // Property access (not own-key reads): tokenizers are usually class
  // instances whose `count` lives on the prototype.
  const tokenizer = requireObject(value, "tokenizer");
  if (typeof tokenizer["count"] !== "function") {
    fail(`tokenizer.count must be a function, got ${describeValue(tokenizer["count"])}.`);
  }
  if (typeof tokenizer["name"] !== "string") {
    fail(`tokenizer.name must be a string, got ${describeValue(tokenizer["name"])}.`);
  }
}

function checkOnEvent(value: unknown): void {
  if (value !== undefined && typeof value !== "function") {
    fail(`onEvent must be a function, got ${describeValue(value)}.`);
  }
}

/**
 * Duck-typed so signals from other realms or polyfills work.
 * removeEventListener is required too: listeners are always detached.
 */
function checkSignal(value: unknown): void {
  if (value === undefined) {
    return;
  }
  const signal = requireObject(value, "signal");
  if (
    typeof signal["aborted"] !== "boolean" ||
    typeof signal["addEventListener"] !== "function" ||
    typeof signal["removeEventListener"] !== "function"
  ) {
    fail(
      "signal must be an AbortSignal (a boolean `aborted` plus addEventListener and removeEventListener).",
    );
  }
}

function checkProvider(value: unknown): void {
  const provider = section(value, "provider", PROVIDER_FIELDS);

  const model = provider["model"];
  if (model !== undefined && (typeof model !== "string" || !hasNonWhitespace(model))) {
    fail(`provider.model must be a non-empty string, got ${describeValue(model)}.`);
  }
  const apiKey = provider["apiKey"];
  if (apiKey !== undefined && typeof apiKey !== "string") {
    // Never echo the value: it may be a credential.
    fail("provider.apiKey must be a string.");
  }

  const transport = provider["transport"];
  if (transport === undefined) {
    return;
  }
  if (typeof transport === "string") {
    checkEnum(transport, "provider.transport", BUILT_IN_TRANSPORTS);
    return;
  }
  if (typeof transport !== "object" || transport === null || Array.isArray(transport)) {
    fail(
      `provider.transport must be "openrouter", "direct", or a JevTransport object, got ${describeValue(transport)}.`,
    );
  }
  const t = transport as UnknownRecord;
  for (const method of ["decide", "contextWindow"] as const) {
    if (typeof t[method] !== "function") {
      fail(`provider.transport.${method} must be a function, got ${describeValue(t[method])}.`);
    }
  }
  for (const field of ["name", "defaultModel"] as const) {
    if (typeof t[field] !== "string") {
      fail(`provider.transport.${field} must be a string, got ${describeValue(t[field])}.`);
    }
  }
  // The model sent to Jev is provider.model ?? transport.defaultModel; it must not be blank.
  if (model === undefined && !hasNonWhitespace(t["defaultModel"] as string)) {
    fail("provider.model is required because provider.transport.defaultModel is empty.");
  }
}

// ---------------------------------------------------------------------------
// Option helpers
// ---------------------------------------------------------------------------

function checkMaxStateTokens(value: unknown): "auto" | number {
  if (value === "auto") {
    return value;
  }
  if (typeof value !== "number" || !Number.isInteger(value) || value < MIN_STATE_TOKENS) {
    fail(
      `chunking.maxStateTokens must be "auto" or an integer >= ${MIN_STATE_TOKENS}, got ${describeValue(value)}.`,
    );
  }
  return value;
}

/** Returns `fallback` for an omitted (undefined) option, else the checked value. `null` is not "omitted". */
function withDefault<T>(value: unknown, fallback: T, check: (value: unknown) => T): T {
  return value === undefined ? fallback : check(value);
}

/** An optional option section: undefined → {}, otherwise a non-array object with only known fields. */
function section(value: unknown, path: string, fields: Record<string, true>): UnknownRecord {
  if (value === undefined) {
    return {};
  }
  const obj = requireObject(value, path);
  rejectUnknownFields(obj, fields, path);
  return obj;
}

interface NumberRule {
  min: number;
  minExclusive?: boolean;
  max?: number;
  maxExclusive?: boolean;
  integer?: boolean;
}

function checkNumber(value: unknown, path: string, rule: NumberRule): number {
  if (typeof value !== "number" || !satisfiesRule(value, rule)) {
    fail(`${path} must be ${describeRule(rule)}, got ${describeValue(value)}.`);
  }
  return value;
}

function satisfiesRule(value: number, rule: NumberRule): boolean {
  if (!Number.isFinite(value) || (rule.integer === true && !Number.isInteger(value))) {
    return false;
  }
  const aboveMin = rule.minExclusive === true ? value > rule.min : value >= rule.min;
  const belowMax =
    rule.max === undefined || (rule.maxExclusive === true ? value < rule.max : value <= rule.max);
  return aboveMin && belowMax;
}

/** e.g. "an integer >= 1", "a finite number in [0, 0.5]", "a finite number in (0, 1)". */
function describeRule(rule: NumberRule): string {
  const kind = rule.integer === true ? "an integer" : "a finite number";
  if (rule.max === undefined) {
    return `${kind} ${rule.minExclusive === true ? ">" : ">="} ${rule.min}`;
  }
  const open = rule.minExclusive === true ? "(" : "[";
  const close = rule.maxExclusive === true ? ")" : "]";
  return `${kind} in ${open}${rule.min}, ${rule.max}${close}`;
}

function checkBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") {
    fail(`${path} must be a boolean, got ${describeValue(value)}.`);
  }
  return value;
}

function checkEnum<T extends string>(value: unknown, path: string, allowed: readonly T[]): T {
  if (typeof value !== "string" || !(allowed as readonly string[]).includes(value)) {
    const options = allowed.map((a) => JSON.stringify(a)).join(", ");
    fail(`${path} must be one of ${options}, got ${describeValue(value)}.`);
  }
  return value as T;
}

// ---------------------------------------------------------------------------
// Generic helpers
// ---------------------------------------------------------------------------

function fail(message: string, cause?: unknown): never {
  throw new JevValidationError(message, cause === undefined ? undefined : { cause });
}

function requireObject(value: unknown, path: string): UnknownRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail(`${path} must be an object, got ${describeValue(value)}.`);
  }
  return value as UnknownRecord;
}

function isPlainObject(value: unknown): value is UnknownRecord {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const proto: unknown = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

/** Reads an own enumerable property: exactly what JSON.stringify would send. */
function ownValue(obj: UnknownRecord, key: string): unknown {
  return Object.prototype.propertyIsEnumerable.call(obj, key) ? obj[key] : undefined;
}

function rejectUnknownFields(obj: UnknownRecord, fields: Record<string, true>, path: string): void {
  for (const key of Object.keys(obj)) {
    if (!Object.hasOwn(fields, key)) {
      const expected = Object.keys(fields).join(", ");
      fail(`${path}${pathSegment(key, path === "")} is not a recognized field (expected: ${expected}).`);
    }
  }
}

/** `.key` for identifier-like keys, `["odd key"]` otherwise. `root` omits the leading dot. */
function pathSegment(key: string, root = false): string {
  if (/^[A-Za-z_$][\w$]*$/.test(key)) {
    return root ? key : `.${key}`;
  }
  return `[${JSON.stringify(truncate(key))}]`;
}

/** Uses a regex instead of trim() so a multi-megabyte input is not copied. */
function hasNonWhitespace(text: string): boolean {
  return /\S/.test(text);
}

function truncate(text: string): string {
  return text.length > MAX_ECHO_LENGTH ? `${text.slice(0, MAX_ECHO_LENGTH)}...` : text;
}

/** Short, safe description of an invalid value for error messages. */
function describeValue(value: unknown): string {
  if (value === null) {
    return "null";
  }
  if (Array.isArray(value)) {
    return "an array";
  }
  switch (typeof value) {
    case "string":
      return JSON.stringify(truncate(value));
    case "number":
    case "boolean":
    case "undefined":
      return String(value);
    case "bigint":
      return `${value}n`;
    case "function":
      return "a function";
    case "symbol":
      return "a symbol";
    default:
      return "an object";
  }
}
