/** Public types. */

export type JsonValue = string | number | boolean | null | readonly JsonValue[] | { readonly [key: string]: JsonValue };

/** Instructions or a criterion: a string, or structured JSON such as `{ what, not_for, examples }`. */
export type Criterion = string | { readonly [key: string]: JsonValue } | readonly JsonValue[];

// ---------------------------------------------------------------------------
// Questions: exactly Jev's three native types.
// ---------------------------------------------------------------------------

export interface ChoiceQuestion<K extends string = string> {
  type: "choice";
  instructions: Criterion;
  /** Options, keyed by the value you want back. At least 2. */
  criteria: Readonly<Record<K, Criterion>>;
}

export interface ScoreQuestion {
  type: "score";
  instructions: Criterion;
  /** Ordered rubric, lowest level first. 2 to 10 levels. */
  criteria: readonly Criterion[];
}

export interface NoulQuestion {
  type: "noul";
  instructions: Criterion;
  /** Optional descriptions of what counts as yes and no. */
  criteria?: { readonly true: Criterion; readonly false: Criterion } | undefined;
}

export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export interface DecideOptions<Q extends Question = Question> {
  /** The text to decide about, up to `maxInputTokens`. */
  input: string;
  question: Q;
  /**
   * How chunk answers become one answer. "average" (default) suits questions
   * about the input as a whole. "max" takes the chunk with the highest score
   * or yes-probability, for noul and score questions about whether something
   * appears anywhere; choice answers are always averaged.
   */
  combine?: (Q extends ChoiceQuestion ? "average" : "average" | "max") | undefined;
  /** Largest accepted input, in estimated tokens (see `estimateTokens`). Default 250,000. */
  maxInputTokens?: number | undefined;
  /** Jev model id. Default "typesafe/jev-1.13". */
  model?: string | undefined;
  /** OpenRouter API key. Default `process.env.OPENROUTER_API_KEY`. */
  apiKey?: string | undefined;
  /** Decisions API endpoint. Default OpenRouter's. */
  url?: string | undefined;
  /** Replaces the HTTP call entirely, for tests or other providers. */
  transport?: Transport | undefined;
  signal?: AbortSignal | undefined;
}

/**
 * One Decisions API call: takes the request body, returns the response body.
 * To get retries or smaller chunks, throw an error with a numeric `status`:
 * 408, 429, and 5xx are retried (honoring an optional `retryAfterMs`), 413
 * means the chunk was too long, anything else fails the call.
 */
export type Transport = (request: DecisionsRequest, signal: AbortSignal) => Promise<DecisionsResponse>;

export interface DecisionsRequest {
  model: string;
  state: string;
  questions: { decision: Question };
}

export interface DecisionsResponse {
  model?: string;
  answers: { readonly [key: string]: unknown };
  usage?: { input_tokens?: number; cost?: number };
}

// ---------------------------------------------------------------------------
// Results. Each chunk carries the same answer fields as the result itself.
// ---------------------------------------------------------------------------

/** Where a chunk sits in the input (`input.slice(start, end)`) and its share of the answer. */
export interface ChunkSpan {
  start: number;
  end: number;
  /** Share of the input that only this chunk covers. Weights sum to 1. */
  weight: number;
}

export interface ChoiceAnswer<K extends string = string> {
  choice: K;
  probabilities: Record<K, number>;
  /** Jev's own confidence. Absent only if Jev omitted it. */
  confidence?: number;
}

export interface ScoreAnswer {
  /** Probability-weighted level, from 0 to levels - 1. */
  score: number;
  /** Probability of each level, keyed "0", "1", ... */
  probabilities: Record<string, number>;
  /** Jev's own confidence. Absent only if Jev omitted it. */
  confidence?: number;
}

export interface NoulAnswer {
  /** Probability that the answer is yes. */
  noul: number;
}

export interface Usage {
  /** Requests sent, including retries. */
  requests: number;
  /** Input tokens reported by the provider. */
  inputTokens: number;
  /** Cost reported by the provider, in USD, when it reports one. */
  costUsd?: number;
}

export interface ResultDetails<A> {
  /** How consistently the chunks answered: 1 means identical answers, 0 completely different. */
  agreement: number;
  chunks: Array<ChunkSpan & A>;
  /** The model build that answered. */
  model: string;
  usage: Usage;
}

export interface ChoiceResult<K extends string = string> extends ChoiceAnswer<K>, ResultDetails<ChoiceAnswer<K>> {
  type: "choice";
}

export interface ScoreResult extends ScoreAnswer, ResultDetails<ScoreAnswer> {
  type: "score";
}

export interface NoulResult extends NoulAnswer, ResultDetails<NoulAnswer> {
  type: "noul";
}

export type Result = ChoiceResult | ScoreResult | NoulResult;

/** The result type for a question; choice keys flow through. */
export type ResultFor<Q extends Question> = Q extends ChoiceQuestion<infer K>
  ? ChoiceResult<K>
  : Q extends ScoreQuestion
    ? ScoreResult
    : Q extends NoulQuestion
      ? NoulResult
      : Result;
