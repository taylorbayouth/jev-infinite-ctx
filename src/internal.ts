/**
 * Small helpers shared by several modules. Internal: src/index.ts does not
 * re-export this module.
 */

import { JevAbortError } from "./errors.js";

/** Longest delay setTimeout honors; a larger one fires after 1 ms instead. */
export const MAX_TIMER_DELAY_MS = 2_147_483_647;

/** Longest caller string (a key, label, or answer type) echoed back in an error message. */
const MAX_ECHO_LENGTH = 64;

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Own-property read, so a polluted Object.prototype cannot supply wire fields. */
export function own(record: Readonly<Record<string, unknown>>, key: string): unknown {
  return Object.hasOwn(record, key) ? record[key] : undefined;
}

/** A caller string, JSON-quoted for an error message and cut to MAX_ECHO_LENGTH characters. */
export function quote(text: string): string {
  return JSON.stringify(text.length > MAX_ECHO_LENGTH ? `${text.slice(0, MAX_ECHO_LENGTH)}…` : text);
}

/** The error a cancelled operation rejects with; its cause is the signal's reason. */
export function abortError(signal: AbortSignal | undefined): JevAbortError {
  return new JevAbortError("The operation was aborted.", { cause: signal?.reason });
}

export function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw abortError(signal);
}

/**
 * Settles with `promise`, or rejects with `reason()` (by default the signal's
 * reason) as soon as `signal` aborts. Never leaves a rejection unhandled.
 */
export function raceAbort<T>(
  promise: Promise<T>,
  signal: AbortSignal,
  reason: () => unknown = () => signal.reason,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    if (signal.aborted) {
      promise.catch(() => undefined);
      reject(reason());
      return;
    }
    const onAbort = (): void => reject(reason());
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
