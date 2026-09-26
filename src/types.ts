/**
 * Public and cross-module types for JevInfiniteCTX.
 *
 * This file is the contract between modules. Implementation modules import
 * from here; they do not redefine these shapes.
 */

// ---------------------------------------------------------------------------
// JSON-safe values (spec 5.2: `unknown` narrowed to Jev-supported JSON values)
// ---------------------------------------------------------------------------

export type JsonPrimitive = string | number | boolean | null;
/**
 * Arrays are readonly so `as const` questions and shared rubric constants
 * type-check; the package never mutates a caller's question. An object value
 * must be a type alias or literal: an `interface` has no index signature, so
 * TypeScript does not treat it as a JsonObject.
 */
export type JsonValue = JsonPrimitive | readonly JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };

/** Jev accepts a plain string or a structured JSON object/array for instructions. */
export type JevInstructions = string | JsonObject | readonly JsonValue[];

/**
 * A single criterion (choice option description, score level, noul true/false
 * description). Jev accepts a string or a structured object such as
 * `{ what, not_for, examples }`.
 */
export type JevCriterion = string | JsonObject | readonly JsonValue[];

// ---------------------------------------------------------------------------
// Questions (spec 4, 5.2)
// ---------------------------------------------------------------------------

export type JevQuestionType = "choice" | "score" | "noul";

export interface NoulQuestion {
  type: "noul";
  instructions: JevInstructions;
  /** Optional. When present, both `true` and `false` are required. */
  criteria?: { true: JevCriterion; false: JevCriterion } | undefined;
}

export interface ChoiceQuestion<K extends string = string> {
  type: "choice";
  instructions: JevInstructions;
  /** Caller-defined option map. Keys are preserved exactly. At least 2 options. */
  criteria: Record<K, JevCriterion>;
}

export interface ScoreQuestion {
  type: "score";
  instructions: JevInstructions;
  /** Ordered rubric, lowest level first. 2 to 10 levels. */
  criteria: readonly JevCriterion[];
}

export type JevQuestion = NoulQuestion | ChoiceQuestion<string> | ScoreQuestion;

// ---------------------------------------------------------------------------
// Transport contract (spec 13)
// ---------------------------------------------------------------------------

/** The wire-level request a transport sends to Jev for a single chunk. */
export interface NativeJevRequest {
  model: string;
  /** JevInfiniteCTX always sends the chunk text as a string state. */
  state: string;
  /** Keyed questions, exactly as Jev's Decisions API expects them. */
  questions: Record<string, JevQuestion>;
  /** Aborts the in-flight request. Transports must honor it. */
  signal?: AbortSignal | undefined;
}

export interface NativeNoulAnswer {
  type: "noul";
  /** Probability of yes, 0..1. Noul has no native confidence. */
  noul: number;
}

export interface NativeChoiceAnswer {
  type: "choice";
  choice: string;
  probabilities?: Record<string, number>;
  confidence?: number;
}

export interface NativeScoreAnswer {
  type: "score";
  /** Probability-weighted level index, 0..(levels - 1). */
  score: number;
  probabilities?: Record<string, number>;
  legend?: Record<string, JevCriterion>;
  confidence?: number;
}

export type NativeJevAnswer = NativeNoulAnswer | NativeChoiceAnswer | NativeScoreAnswer;

export interface NativeJevUsage {
  inputTokens?: number | undefined;
  outputTokens?: number | undefined;
  /** USD, when the provider reports it (OpenRouter does, TypeSafe direct does not). */
  costUsd?: number | undefined;
}

/** Normalized (camelCase) response every transport returns. */
export interface NativeJevResponse {
  id?: string;
  /** The exact model build that answered. */
  model: string;
  provider?: string;
  answers: Record<string, NativeJevAnswer>;
  usage?: NativeJevUsage;
}

/**
 * Provider abstraction. Chunking, aggregation, and C3 never depend on a
 * concrete provider. Transports should throw `JevProviderError` (see
 * errors.ts) so the orchestrator can classify retryable and context-limit
 * failures; any other thrown error is treated as a permanent failure.
 */
export interface JevTransport {
  /** Short provider identifier used in metadata and events, e.g. "openrouter". */
  readonly name: string;
  /** Model used when the caller does not specify one. */
  readonly defaultModel: string;
  decide(request: NativeJevRequest): Promise<NativeJevResponse>;
  /**
   * Total context window in tokens for the model. Return `undefined` when
   * reliable metadata is unavailable; the orchestrator then falls back to a
   * conservative state budget (spec 6.1).
   */
  contextWindow(model: string): Promise<number | undefined> | number | undefined;
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/**
 * Token counter used for budgeting and chunk boundaries. Jev's tokenizer is
 * not public, so the default is a conservative heuristic; callers can plug
 * in any real tokenizer (for example js-tiktoken) via `createTokenizer`.
 *
 * Contract: `count` must be deterministic, return a non-negative integer,
 * return 0 for "", and be (approximately) monotone non-decreasing as text is
 * extended. It must not retain references to the text.
 */
export interface Tokenizer {
  readonly name: string;
  count(text: string): number;
}

// ---------------------------------------------------------------------------
// Options (spec 5, 14)
//
// Every optional input field also accepts an explicit `undefined`, read as
// omitted, so options built from possibly-unset values (`process.env.X`)
// compile under `exactOptionalPropertyTypes`.
// ---------------------------------------------------------------------------

export type AggregationMethod = "weighted_mean" | "mean" | "median" | "min" | "max";

export interface ChunkingOptions {
  /** Overlap ratio of the usable chunk size. Default 0.05. Allowed range [0, 0.5]. */
  overlap?: number | undefined;
  /**
   * "auto" (default) derives the state budget from the transport's context
   * window. A number caps the state budget; the effective budget is
   * min(number, auto budget).
   */
  maxStateTokens?: "auto" | number | undefined;
  /** Fraction of the remaining context held back as safety reserve. Default 0.08. Range [0, 0.5]. */
  contextSafetyReserve?: number | undefined;
  /** Fixed token reserve for Jev's request scaffolding. Default 1024. */
  protocolReserve?: number | undefined;
  /** Prefer paragraph, sentence, whitespace boundaries before hard cuts. Default true. */
  preferNaturalBoundaries?: boolean | undefined;
  /**
   * Optional guard on the number of chunks. A first plan that exceeds it
   * throws JevValidationError before any request. A re-chunk pass (after
   * context-limit errors) that would exceed it throws JevContextBudgetError,
   * and the requests of earlier passes have already been made and billed.
   */
  maxChunks?: number | undefined;
}

export interface ExecutionOptions {
  /** Maximum concurrent Jev requests. Default 4. */
  maxConcurrency?: number | undefined;
  /** Retries per chunk for retryable failures (429, 5xx, overload, network). Default 3. */
  retries?: number | undefined;
  /** Base delay for exponential backoff. Default 500ms. */
  retryBaseDelayMs?: number | undefined;
  /** Maximum single backoff delay. Default 8000ms. */
  retryMaxDelayMs?: number | undefined;
  /** How many times to shrink the budget and re-chunk after a context-limit error. Default 4. */
  maxRechunks?: number | undefined;
  /**
   * Multiplier applied on each context-limit re-chunk: the new budget is
   * floor(min(current budget, failing chunk's estimated tokens) × factor).
   * Default 0.75.
   */
  rechunkShrinkFactor?: number | undefined;
}

export interface ConfidenceOptions {
  /** "c3" (default) applies the C3 correction. "none" reports adjusted = base. */
  method?: "c3" | "none" | undefined;
  /** C_cap. Default 0.98. Range (0, 1]. */
  cap?: number | undefined;
  /** λ saturation rate. Default 0.25. Must be > 0. */
  lambda?: number | undefined;
  /** A_floor. Default 0.5. Range [0, 1). */
  agreementFloor?: number | undefined;
  /** γ gate exponent. Default 2.0. Must be > 0. */
  agreementExponent?: number | undefined;
}

export type BuiltInTransportName = "openrouter" | "direct";

export interface ProviderOptions {
  /** Built-in transport name or any `JevTransport` implementation. Default "openrouter". */
  transport?: BuiltInTransportName | JevTransport | undefined;
  /** Model id. Defaults to the transport's `defaultModel`. */
  model?: string | undefined;
  /**
   * API key for a built-in transport. Defaults to OPENROUTER_API_KEY
   * ("openrouter") or TYPESAFE_API_KEY ("direct") from the environment.
   * Ignored when `transport` is an object.
   */
  apiKey?: string | undefined;
}

export interface JevInfiniteCTXRequest<Q extends JevQuestion = JevQuestion> {
  input: string;
  question: Q;
  aggregation?: AggregationMethod | undefined;
  chunking?: ChunkingOptions | undefined;
  confidence?: ConfidenceOptions | undefined;
  execution?: ExecutionOptions | undefined;
  provider?: ProviderOptions | undefined;
  /** Token counter for budgeting. Defaults to the built-in heuristic tokenizer. */
  tokenizer?: Tokenizer | undefined;
  /** Cancels the whole operation, including in-flight requests. */
  signal?: AbortSignal | undefined;
  /** Observability hook. Never receives source text. Exceptions thrown by the hook are swallowed. */
  onEvent?: ((event: JevInfiniteCTXEvent) => void) | undefined;
}

/** Fully resolved options after defaults are applied. */
export interface ResolvedOptions {
  aggregation: AggregationMethod;
  chunking: Required<Omit<ChunkingOptions, "maxChunks">> & { maxChunks: number | undefined };
  execution: Required<ExecutionOptions>;
  confidence: Required<ConfidenceOptions>;
}

// ---------------------------------------------------------------------------
// Chunking (spec 6)
// ---------------------------------------------------------------------------

export interface StateBudget {
  /** Context window used for the computation, or undefined when unknown. */
  contextWindow: number | undefined;
  /** Estimated tokens consumed by the question (instructions + criteria). */
  questionTokens: number;
  protocolReserve: number;
  safetyReserve: number;
  /** Final per-chunk state budget in estimated tokens. */
  usableStateTokens: number;
  /** "metadata" when derived from contextWindow, "fallback" when the window was unknown, "caller" when capped by maxStateTokens. */
  source: "metadata" | "fallback" | "caller";
}

export type ChunkBoundary = "paragraph" | "sentence" | "whitespace" | "hard" | "end";

/**
 * One planned chunk. Offsets are UTF-16 code unit indices into the input,
 * `[start, end)`. Chunk text is `input.slice(start, end)`; it is not stored.
 */
export interface ChunkSpan {
  index: number;
  start: number;
  end: number;
  /** Estimated tokens of the whole chunk. */
  tokens: number;
  /** Estimated tokens of the region shared with the previous chunk, `[start, previous.end)`. 0 for the first chunk. */
  overlapTokens: number;
  /** tokens - overlapTokens, floored at 1 (spec 6.4). */
  uniqueTokens: number;
  /** uniqueTokens / sum(uniqueTokens). Sums to 1 across the plan. */
  weight: number;
  /** How the end offset was chosen. The final chunk is "end". */
  boundary: ChunkBoundary;
}

export interface ChunkPlan {
  chunks: ChunkSpan[];
  /** The state budget each chunk respects. */
  stateTokenBudget: number;
  /** floor(stateTokenBudget * overlap). */
  overlapTokens: number;
  overlapRatio: number;
  /** Estimated tokens of the full input. */
  totalTokens: number;
  /** N_eff = Σ_i uniqueTokens_i / max_i tokens_i, clamped to [1, N] (spec 11.3). */
  effectiveCount: number;
}

// ---------------------------------------------------------------------------
// Common probability representation (spec 8)
// ---------------------------------------------------------------------------

/**
 * Labels are ordered and fixed per question:
 * - choice: Object.keys(criteria) in insertion order
 * - score: "0", "1", ..., String(levels - 1)
 * - noul: ["no", "yes"]  (P = [1 - p, p])
 * `probs[j]` is the probability of `labels[j]`; probs are finite, >= 0, and sum to 1.
 */
export interface Distribution {
  labels: readonly string[];
  probs: number[];
}

// ---------------------------------------------------------------------------
// Results (spec 12)
// ---------------------------------------------------------------------------

export type ConfidenceSource = "jev" | "derived";

export interface C3Components {
  /** N_eff from unique-token weights. */
  effectiveChunkCount: number;
  /** S = 1 - exp(-λ (N_eff - 1)). */
  saturation: number;
  /** G = clamp((A - A_floor) / (1 - A_floor), 0, 1)^γ. */
  gate: number;
  cap: number;
  lambda: number;
  agreementFloor: number;
  agreementExponent: number;
}

export interface ConfidenceInfo {
  /**
   * C_base. For choice/score: the weighted mean of Jev's raw chunk
   * confidences when every chunk reports one (source "jev"); otherwise
   * 1 − H(P_agg)/ln K from the aggregate (source "derived"). For noul:
   * |2·noul − 1| (source "derived").
   */
  base: number;
  /** C_adjusted. Equals base for one chunk or when method is "none". Never below base, never raised above cap. */
  adjusted: number;
  /** adjusted - base. */
  adjustment: number;
  /** "jev" when base comes from Jev's confidence field, "derived" when computed by this package. */
  source: ConfidenceSource;
  /** Cross-chunk agreement A = 1 − Σ w_i · TV_i, in [0, 1]. */
  agreement: number;
  method: "c3-v1" | "none";
  components: C3Components;
}

export interface ChunkUsage {
  inputTokens?: number;
  outputTokens?: number;
  costUsd?: number;
}

/** Raw, inspectable per-chunk result, in original chunk order. */
export interface ChunkResult {
  index: number;
  start: number;
  end: number;
  estimatedTokens: number;
  uniqueTokens: number;
  weight: number;
  boundary: ChunkBoundary;
  /** Jev's raw answer for this chunk, unmodified. */
  answer: NativeJevAnswer;
  /** Common probability representation, keyed by label. */
  probabilities: Record<string, number>;
  /** Jev's raw confidence for choice/score. Undefined for noul or when Jev omitted it. */
  jevConfidence?: number;
  /** Total variation distance between this chunk's distribution and the aggregate. */
  totalVariation: number;
  usage?: ChunkUsage;
  /** Number of attempts made (1 = no retries). */
  attempts: number;
  elapsedMs: number;
  model: string;
  responseId?: string;
}

export interface ChunksInfo {
  count: number;
  effectiveCount: number;
  overlap: number;
  overlapTokens: number;
  stateTokenBudget: number;
  /** Per-chunk results in original order. */
  results: ChunkResult[];
}

export interface UsageInfo {
  /**
   * Sum of provider-reported input tokens across every response received,
   * including discarded re-chunk passes. A request that fails, times out, or
   * is aborted in flight reports no usage, so it is counted only in `requests`.
   */
  inputTokens: number;
  outputTokens: number;
  /** Sum of provider-reported cost, or undefined if no response reported cost. */
  costUsd: number | undefined;
  elapsedMs: number;
  /** Estimated tokens of the full input (package tokenizer). */
  inputTokensEstimated: number;
  /** Every request sent, including failed, retried, and aborted ones. */
  requests: number;
  retries: number;
  rechunks: number;
}

export interface AggregationInfo {
  method: AggregationMethod;
  /**
   * True when the per-dimension reduction summed to zero (for example `min`
   * over disjoint one-hot chunks) and the aggregate fell back to uniform.
   */
  degenerate: boolean;
}

interface BaseResult {
  confidence: ConfidenceInfo;
  aggregation: AggregationInfo;
  chunks: ChunksInfo;
  usage: UsageInfo;
  model: string;
  provider: string;
}

export interface ChoiceResult<K extends string = string> extends BaseResult {
  type: "choice";
  choice: K;
  probabilities: Record<K, number>;
}

export interface ScoreResult extends BaseResult {
  type: "score";
  /** Native Jev score semantics: 0..(levels − 1). */
  score: number;
  /** score / (levels − 1). Metadata only. */
  normalizedScore: number;
  probabilities: Record<string, number>;
  /** Caller-provided rubric keyed by level index. */
  legend: Record<string, JevCriterion>;
}

export interface NoulResult extends BaseResult {
  type: "noul";
  /** Aggregated probability of yes. */
  noul: number;
}

export type JevInfiniteCTXResult = ChoiceResult<string> | ScoreResult | NoulResult;

/** Maps a question type to its result type; choice keys flow through. */
export type ResultFor<Q extends JevQuestion> = Q extends ChoiceQuestion<infer K>
  ? ChoiceResult<K>
  : Q extends ScoreQuestion
    ? ScoreResult
    : Q extends NoulQuestion
      ? NoulResult
      : JevInfiniteCTXResult;

// ---------------------------------------------------------------------------
// Observability (spec 16). Events never contain source text.
// ---------------------------------------------------------------------------

export interface DecisionCompletedEvent {
  type: "decision.completed";
  model: string;
  provider: string;
  questionType: JevQuestionType;
  inputTokensEstimated: number;
  inputTokensActual: number;
  chunkCount: number;
  effectiveChunkCount: number;
  overlap: number;
  stateTokenBudget: number;
  aggregationMethod: AggregationMethod;
  agreement: number;
  baseConfidence: number;
  adjustedConfidence: number;
  confidenceSource: ConfidenceSource;
  /** choice key, score, or noul probability. */
  finalAnswer: string | number;
  perChunkProbabilities: Array<Record<string, number>>;
  costUsd: number | undefined;
  latencyMs: number;
  retryCount: number;
  rechunkCount: number;
}

export interface ChunkRetryEvent {
  type: "chunk.retry";
  chunkIndex: number;
  /** The attempt that failed (1-based). */
  attempt: number;
  delayMs: number;
  errorKind: string;
  status?: number;
}

export interface RechunkEvent {
  type: "rechunk";
  previousBudget: number;
  newBudget: number;
  chunkIndex: number;
  status?: number;
}

export interface DecisionFailedEvent {
  type: "decision.failed";
  model: string;
  provider: string;
  errorName: string;
  errorKind?: string;
  chunkIndex?: number;
  latencyMs: number;
  retryCount: number;
  rechunkCount: number;
}

export type JevInfiniteCTXEvent =
  | DecisionCompletedEvent
  | ChunkRetryEvent
  | RechunkEvent
  | DecisionFailedEvent;
