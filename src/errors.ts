/**
 * Error types. Every error thrown by JevInfiniteCTX extends JevInfiniteCTXError.
 */

export class JevInfiniteCTXError extends Error {
  override name = "JevInfiniteCTXError";
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
  }
}

/** Invalid request or configuration. Thrown before any provider call is made. */
export class JevValidationError extends JevInfiniteCTXError {
  override name = "JevValidationError";
}

export type ProviderErrorKind =
  | "rate_limit"
  | "server"
  | "overloaded"
  | "timeout"
  | "network"
  | "context_limit"
  | "auth"
  | "payment"
  | "not_found"
  | "bad_request"
  | "invalid_response"
  | "unknown";

const RETRYABLE_KINDS: ReadonlySet<ProviderErrorKind> = new Set([
  "rate_limit",
  "server",
  "overloaded",
  "timeout",
  "network",
]);

export interface JevProviderErrorOptions {
  kind: ProviderErrorKind;
  status?: number | undefined;
  /** Server-requested delay (for example from Retry-After), in milliseconds. */
  retryAfterMs?: number | undefined;
  /** Overrides the default retryability derived from `kind`. */
  retryable?: boolean | undefined;
  /**
   * Provider response body, truncated to 2,000 characters. For a request with
   * a non-empty state, built-in transports replace the whole body with
   * "[withheld: provider error body overlaps the request state]" when any run
   * of 8 characters in that truncated body or in the 128 characters after the
   * cut (as sent, or with up to three levels of JSON string escapes decoded)
   * also occurs in the state, comparing with whitespace runs collapsed to one
   * space. A state shorter than 8 characters withholds the body when the body
   * contains it.
   */
  body?: string | undefined;
  cause?: unknown;
}

/**
 * Error thrown by transports. `retryable` drives backoff; `kind ===
 * "context_limit"` triggers a shrink-and-rechunk of the whole document.
 */
export class JevProviderError extends JevInfiniteCTXError {
  override name = "JevProviderError";
  readonly kind: ProviderErrorKind;
  readonly status: number | undefined;
  readonly retryAfterMs: number | undefined;
  readonly retryable: boolean;
  readonly body: string | undefined;

  constructor(message: string, options: JevProviderErrorOptions) {
    super(message, { cause: options.cause });
    this.kind = options.kind;
    this.status = options.status;
    this.retryAfterMs = options.retryAfterMs;
    this.retryable = options.retryable ?? RETRYABLE_KINDS.has(options.kind);
    this.body = options.body;
  }
}

/** A Jev answer did not match the question (wrong type, unknown option, bad probabilities). */
export class JevResponseError extends JevInfiniteCTXError {
  override name = "JevResponseError";
}

/**
 * A chunk failed permanently (non-retryable error, or retries exhausted).
 * The whole operation fails closed: no partial aggregate is returned.
 */
export class JevChunkFailedError extends JevInfiniteCTXError {
  override name = "JevChunkFailedError";
  readonly chunkIndex: number;
  readonly attempts: number;
  constructor(message: string, options: { chunkIndex: number; attempts: number; cause?: unknown }) {
    super(message, { cause: options.cause });
    this.chunkIndex = options.chunkIndex;
    this.attempts = options.attempts;
  }
}

/** Context-limit errors persisted after the maximum number of shrink-and-rechunk passes. */
export class JevContextBudgetError extends JevInfiniteCTXError {
  override name = "JevContextBudgetError";
  readonly lastBudget: number;
  readonly rechunks: number;
  constructor(message: string, options: { lastBudget: number; rechunks: number; cause?: unknown }) {
    super(message, { cause: options.cause });
    this.lastBudget = options.lastBudget;
    this.rechunks = options.rechunks;
  }
}

/** The caller's AbortSignal fired. */
export class JevAbortError extends JevInfiniteCTXError {
  override name = "AbortError";
}

export function isJevProviderError(error: unknown): error is JevProviderError {
  return error instanceof JevProviderError;
}
