/**
 * Retry with exponential backoff for transient provider failures (spec 7:
 * retry 429, transient 5xx, and overload responses).
 *
 * Only `JevProviderError`s marked retryable are retried. Context-limit
 * errors are never retried here: resending the same oversized state cannot
 * succeed, so they propagate to the orchestrator, which shrinks the budget
 * and re-chunks the whole document instead (spec 7, 15).
 */

import { JevInfiniteCTXError, JevProviderError } from "./errors.js";
import { abortError, MAX_TIMER_DELAY_MS, throwIfAborted } from "./internal.js";

export interface RetryInfo {
  /** The attempt that just failed (1-based). */
  attempt: number;
  /** Delay before the next attempt, in milliseconds. */
  delayMs: number;
  error: JevProviderError;
}

export interface RetryOptions {
  /** Retries after the first attempt, so at most `retries + 1` attempts are made. */
  retries: number;
  baseDelayMs: number;
  maxDelayMs: number;
  signal?: AbortSignal;
  /** Called before each backoff sleep. Exceptions propagate and end the retry loop. */
  onRetry?: (info: RetryInfo) => void;
  /** Injectable for tests. Defaults to `sleep` below. */
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
  /** Injectable for tests. Must return a value in [0, 1). Defaults to `Math.random`. */
  random?: () => number;
}

/**
 * Upper bound for a server-requested delay (Retry-After). Honoring the
 * server matters more than our own backoff ceiling, but a hostile or broken
 * header must not stall a chunk indefinitely.
 */
const MIN_RETRY_AFTER_CAP_MS = 60_000;

/**
 * Runs `fn` (called with the 1-based attempt number) until it succeeds, the
 * error is not retryable, or `retries` retries have been used.
 *
 * - Non-retryable errors (anything that is not a retryable
 *   `JevProviderError`, and every `context_limit` error) are rethrown
 *   immediately and unchanged.
 * - When retries are exhausted the last error is rethrown unchanged.
 * - A signal that is already aborted, or aborts between attempts or during
 *   the backoff sleep, ends the loop with `JevAbortError`.
 */
export async function withRetry<T>(
  fn: (attempt: number) => Promise<T>,
  options: RetryOptions,
): Promise<{ value: T; attempts: number }> {
  const { retries, baseDelayMs, maxDelayMs, signal, onRetry } = options;
  if (!(retries >= 0)) {
    throw new JevInfiniteCTXError(`retries must be >= 0, got ${retries}.`);
  }
  const sleepFn = options.sleep ?? sleep;
  const random = options.random ?? Math.random;

  for (let attempt = 1; ; attempt++) {
    // Also covers injected sleep functions that ignore the signal.
    throwIfAborted(signal);
    try {
      return { value: await fn(attempt), attempts: attempt };
    } catch (error) {
      if (!isRetryable(error) || attempt > retries) {
        throw error;
      }
      // A transport that ignored the abort must not trigger a retry (or a retry event).
      throwIfAborted(signal);
      const delayMs = retryDelayMs(error, attempt, baseDelayMs, maxDelayMs, random);
      onRetry?.({ attempt, delayMs, error });
      await sleepFn(delayMs, signal);
    }
  }
}

/**
 * Exponential backoff with "equal jitter" for the given failed attempt
 * (1-based): min(maxDelayMs, baseDelayMs · 2^(attempt − 1)) · (0.5 + 0.5 · random()).
 * Half the delay is guaranteed so retries from concurrent chunks back off
 * together; the other half is randomized so they do not retry in lockstep.
 */
export function computeBackoffMs(
  attempt: number,
  baseDelayMs: number,
  maxDelayMs: number,
  random: () => number,
): number {
  // Guarding base <= 0 avoids 0 · Infinity = NaN once 2^(attempt − 1) overflows.
  const ceiling =
    baseDelayMs > 0 ? Math.min(maxDelayMs, baseDelayMs * 2 ** Math.max(0, attempt - 1)) : 0;
  const jitter = Math.min(1, Math.max(0, random()));
  return ceiling * (0.5 + 0.5 * jitter);
}

/**
 * Resolves after `ms` milliseconds. Rejects with `JevAbortError` (cause: the
 * signal's reason) if `signal` is already aborted or aborts while waiting;
 * the timer and listener are always cleaned up.
 */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError(signal));
      return;
    }
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(abortError(signal));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, clampTimerDelay(ms));
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function isRetryable(error: unknown): error is JevProviderError {
  // context_limit is excluded even if a transport marks it retryable: only
  // re-chunking can fix it (spec 7).
  return error instanceof JevProviderError && error.retryable && error.kind !== "context_limit";
}

/** Retry-After takes precedence over computed backoff, within a generous cap. */
function retryDelayMs(
  error: JevProviderError,
  attempt: number,
  baseDelayMs: number,
  maxDelayMs: number,
  random: () => number,
): number {
  const retryAfter = error.retryAfterMs;
  if (retryAfter !== undefined && !Number.isNaN(retryAfter)) {
    return Math.min(Math.max(0, retryAfter), Math.max(maxDelayMs, MIN_RETRY_AFTER_CAP_MS));
  }
  return computeBackoffMs(attempt, baseDelayMs, maxDelayMs, random);
}

function clampTimerDelay(ms: number): number {
  if (Number.isNaN(ms) || ms <= 0) {
    return 0;
  }
  return Math.min(ms, MAX_TIMER_DELAY_MS);
}
