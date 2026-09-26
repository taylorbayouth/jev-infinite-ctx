/**
 * Bounded-concurrency map used to fan chunk requests out to Jev (spec 7).
 *
 * Results come back in input order regardless of completion order, so chunk
 * metadata keeps the original document order. The operation fails closed:
 * the first failure cancels the remaining work instead of letting a partial
 * set of chunk results reach aggregation (spec 7, 15).
 */

import { JevAbortError, JevInfiniteCTXError } from "./errors.js";
import { abortError } from "./internal.js";

export interface MapWithConcurrencyOptions {
  /** Cancels the whole map: no new items start and in-flight items see an aborted signal. */
  signal?: AbortSignal;
}

/**
 * Maps `items` through `fn` with at most `limit` calls in flight.
 *
 * - Resolves with results in input order.
 * - On the first rejection: stops scheduling new items, aborts the signal
 *   passed to in-flight calls, waits for every in-flight call to settle, then
 *   rejects with that first error. Sibling failures caused by the abort are
 *   ignored so the root cause is what surfaces.
 * - When `options.signal` is already aborted or aborts mid-run: same
 *   cancellation, then rejects with `JevAbortError` (cause: the signal's
 *   reason), even if the in-flight calls themselves succeed.
 * - Whichever of the two happens first decides the rejection.
 *
 * Every promise returned by `fn` is awaited, so no rejection goes unhandled,
 * and the function never throws synchronously.
 */
export async function mapWithConcurrency<T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T, index: number, signal: AbortSignal) => Promise<R>,
  options: MapWithConcurrencyOptions = {},
): Promise<R[]> {
  // Infinity is accepted and means "everything at once".
  if (!(limit >= 1) || !(Number.isInteger(limit) || limit === Infinity)) {
    throw new JevInfiniteCTXError(`Concurrency limit must be an integer >= 1, got ${limit}.`);
  }
  const external = options.signal;
  if (external?.aborted) {
    throw abortError(external);
  }
  const count = items.length;
  if (count === 0) {
    return [];
  }

  const controller = new AbortController();
  const results = new Array<R>(count);
  let nextIndex = 0;
  // The first terminal event (task failure or caller abort) wins; later
  // failures are usually just consequences of the abort we triggered.
  let failure: { error: unknown } | undefined;

  const onExternalAbort = (): void => {
    if (failure === undefined) {
      failure = { error: abortError(external) };
    }
    controller.abort(external?.reason);
  };
  external?.addEventListener("abort", onExternalAbort, { once: true });

  // Each worker pulls the next unclaimed index until the list is exhausted
  // or the run is cancelled. Workers never reject; failures are recorded.
  const worker = async (): Promise<void> => {
    while (failure === undefined && nextIndex < count) {
      const index = nextIndex++;
      try {
        // Index is in bounds; the cast only drops noUncheckedIndexedAccess's `undefined`.
        results[index] = await fn(items[index] as T, index, controller.signal);
      } catch (error) {
        if (failure === undefined) {
          failure = { error };
          controller.abort(
            new JevAbortError(`Cancelled because item ${index} failed.`, { cause: error }),
          );
        }
        return;
      }
    }
  };

  try {
    const workers: Promise<void>[] = [];
    for (let w = 0; w < Math.min(limit, count); w++) {
      workers.push(worker());
    }
    await Promise.all(workers);
  } finally {
    external?.removeEventListener("abort", onExternalAbort);
  }

  if (failure !== undefined) {
    throw failure.error;
  }
  return results;
}
