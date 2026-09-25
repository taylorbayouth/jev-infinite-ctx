import fc from "fast-check";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mapWithConcurrency } from "../src/concurrency.js";
import { JevAbortError, JevInfiniteCTXError } from "../src/errors.js";

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason: unknown) => void;
}

function deferred<T = void>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** Lets pending timers and microtasks run. */
const tick = (ms = 0): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/** Resolves when `signal` aborts (immediately if it already has). */
function whenAborted(signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal.aborted) {
      resolve();
    } else {
      signal.addEventListener("abort", () => resolve(), { once: true });
    }
  });
}

/** Captures the settled outcome of a promise without leaving it unhandled. */
async function outcome<T>(promise: Promise<T>): Promise<{ value?: T; error?: unknown }> {
  try {
    return { value: await promise };
  } catch (error) {
    return { error };
  }
}

// Every test proves the "no unhandled rejections" requirement explicitly.
let unhandled: unknown[] = [];
const onUnhandled = (reason: unknown): void => {
  unhandled.push(reason);
};
beforeEach(() => {
  unhandled = [];
  process.on("unhandledRejection", onUnhandled);
});
afterEach(async () => {
  await tick(5);
  process.off("unhandledRejection", onUnhandled);
  expect(unhandled).toEqual([]);
});

describe("mapWithConcurrency: ordering and limits", () => {
  it("returns [] for no items without calling fn", async () => {
    const fn = vi.fn(async () => 1);
    await expect(mapWithConcurrency([], 4, fn)).resolves.toEqual([]);
    expect(fn).not.toHaveBeenCalled();
  });

  it("preserves input order and fills the pool under any completion order (fast-check scheduler)", async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.scheduler(),
        fc.array(fc.integer(), { maxLength: 30 }),
        fc.integer({ min: 1, max: 8 }),
        async (s, items, limit) => {
          let inFlight = 0;
          let maxInFlight = 0;
          const work = s.scheduleFunction(async (x: number) => x * 2);
          const run = mapWithConcurrency(items, limit, async (item, index) => {
            expect(item).toBe(items[index]);
            inFlight++;
            maxInFlight = Math.max(maxInFlight, inFlight);
            try {
              return await work(item);
            } finally {
              inFlight--;
            }
          });
          const result = await s.waitFor(run);
          expect(result).toEqual(items.map((x) => x * 2));
          expect(maxInFlight).toBe(Math.min(limit, items.length));
        },
      ),
      { numRuns: 200 },
    );
  });

  it("preserves input order with randomized real timer delays and never exceeds the limit", async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.array(fc.integer({ min: 0, max: 4 }), { minLength: 1, maxLength: 16 }),
        fc.integer({ min: 1, max: 6 }),
        async (delays, limit) => {
          let inFlight = 0;
          let maxInFlight = 0;
          const completionOrder: number[] = [];
          const result = await mapWithConcurrency(delays, limit, async (delay, index) => {
            inFlight++;
            maxInFlight = Math.max(maxInFlight, inFlight);
            await tick(delay);
            inFlight--;
            completionOrder.push(index);
            return `r${index}`;
          });
          expect(result).toEqual(delays.map((_, i) => `r${i}`));
          expect(maxInFlight).toBeLessThanOrEqual(limit);
          expect(maxInFlight).toBe(Math.min(limit, delays.length));
          expect([...completionOrder].sort((a, b) => a - b)).toEqual(delays.map((_, i) => i));
        },
      ),
      { numRuns: 30 },
    );
  });

  it("runs strictly sequentially with limit 1", async () => {
    const log: string[] = [];
    const result = await mapWithConcurrency(["a", "b", "c"], 1, async (item) => {
      log.push(`start ${item}`);
      await tick();
      log.push(`end ${item}`);
      return item.toUpperCase();
    });
    expect(result).toEqual(["A", "B", "C"]);
    expect(log).toEqual(["start a", "end a", "start b", "end b", "start c", "end c"]);
  });

  it("starts everything at once when the limit exceeds the item count (including Infinity)", async () => {
    for (const limit of [10, Infinity]) {
      const gate = deferred();
      let started = 0;
      const run = mapWithConcurrency([1, 2, 3], limit, async (x) => {
        started++;
        await gate.promise;
        return x;
      });
      expect(started).toBe(3);
      gate.resolve();
      await expect(run).resolves.toEqual([1, 2, 3]);
    }
  });

  it("passes each item's index and one shared, initially unaborted signal", async () => {
    const seen: Array<[string, number, AbortSignal]> = [];
    await mapWithConcurrency(["x", "y", "z"], 2, async (item, index, signal) => {
      expect(signal.aborted).toBe(false);
      seen.push([item, index, signal]);
    });
    expect(seen.map(([item, index]) => [item, index])).toEqual([
      ["x", 0],
      ["y", 1],
      ["z", 2],
    ]);
    expect(new Set(seen.map(([, , signal]) => signal)).size).toBe(1);
  });

  it.each([0, -1, 1.5, Number.NaN, -Infinity])(
    "rejects (never throws synchronously) for invalid limit %s",
    async (limit) => {
      const fn = vi.fn(async () => 1);
      let run: Promise<number[]> | undefined;
      expect(() => {
        run = mapWithConcurrency([1], limit, fn);
      }).not.toThrow();
      await expect(run).rejects.toBeInstanceOf(JevInfiniteCTXError);
      await expect(run).rejects.toThrow(/Concurrency limit must be an integer >= 1/);
      expect(fn).not.toHaveBeenCalled();
    },
  );
});

describe("mapWithConcurrency: first-error semantics", () => {
  it("stops scheduling, aborts siblings, waits for them to settle, then rejects with the first error", async () => {
    const boom = new Error("boom");
    const failNow = deferred();
    const started: number[] = [];
    const settled: number[] = [];
    const signals: AbortSignal[] = [];

    const run = mapWithConcurrency([0, 1, 2, 3, 4, 5], 3, async (item, _index, signal) => {
      started.push(item);
      signals.push(signal);
      try {
        if (item === 1) {
          await failNow.promise;
          throw boom;
        }
        await whenAborted(signal);
        // Slow cleanup: rejection must wait for this.
        await tick(10);
        throw new Error(`sibling ${item} aborted`);
      } finally {
        settled.push(item);
      }
    });
    let settledAtRejection: number[] | undefined;
    const result = outcome(
      run.catch((error: unknown) => {
        settledAtRejection = [...settled];
        throw error;
      }),
    );

    await tick();
    expect(started).toEqual([0, 1, 2]);
    failNow.resolve();

    const { error } = await result;
    expect(error).toBe(boom);
    expect(started).toEqual([0, 1, 2]); // nothing new was scheduled after the failure
    expect([...(settledAtRejection ?? [])].sort()).toEqual([0, 1, 2]);
    expect(signals.every((s) => s.aborted)).toBe(true);
  });

  it("aborts the shared signal with a JevAbortError whose cause is the first error", async () => {
    const boom = new Error("boom");
    let sharedSignal: AbortSignal | undefined;
    const run = mapWithConcurrency([0, 1], 2, async (item, _index, signal) => {
      sharedSignal = signal;
      if (item === 0) {
        throw boom;
      }
      await whenAborted(signal);
    });
    await expect(run).rejects.toBe(boom);
    expect(sharedSignal?.aborted).toBe(true);
    const reason: unknown = sharedSignal?.reason;
    expect(reason).toBeInstanceOf(JevAbortError);
    expect((reason as JevAbortError).cause).toBe(boom);
  });

  it("reports the earliest failure, not a later genuine failure", async () => {
    const first = new Error("first");
    const second = new Error("second");
    const gates = [deferred(), deferred()];
    const run = mapWithConcurrency([0, 1], 2, async (item) => {
      await gates[item]?.promise;
      throw item === 1 ? first : second;
    });
    gates[1]?.resolve(); // item 1 fails first
    await tick();
    gates[0]?.resolve();
    await expect(run).rejects.toBe(first);
  });

  it("waits for a sibling that ignores the abort and resolves late", async () => {
    const boom = new Error("boom");
    let siblingDone = false;
    const run = mapWithConcurrency([0, 1], 2, async (item) => {
      if (item === 0) {
        throw boom;
      }
      await tick(20);
      siblingDone = true;
      return item;
    });
    const { error } = await outcome(run);
    expect(error).toBe(boom);
    expect(siblingDone).toBe(true);
  });

  it("does not schedule new items from a worker whose task succeeds after a sibling failed", async () => {
    const boom = new Error("boom");
    const started: number[] = [];
    const run = mapWithConcurrency([0, 1, 2, 3], 2, async (item) => {
      started.push(item);
      if (item === 0) {
        throw boom;
      }
      await tick(10); // ignores the abort and succeeds; its worker must still stop
      return item;
    });
    await expect(run).rejects.toBe(boom);
    expect(started).toEqual([0, 1]);
  });

  it("turns a synchronous throw from fn into a rejection", async () => {
    const boom = new Error("sync boom");
    const fn = (item: number): Promise<number> => {
      if (item === 2) {
        throw boom;
      }
      return Promise.resolve(item);
    };
    let run: Promise<number[]> | undefined;
    expect(() => {
      run = mapWithConcurrency([1, 2, 3], 1, fn);
    }).not.toThrow();
    await expect(run).rejects.toBe(boom);
  });

  it("does not start any further items after a failure with limit 1", async () => {
    const fn = vi.fn(async (item: number) => {
      if (item === 1) {
        throw new Error("fail at 1");
      }
      return item;
    });
    await expect(mapWithConcurrency([0, 1, 2, 3], 1, fn)).rejects.toThrow("fail at 1");
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("rejects with the failing value even when it is not an Error", async () => {
    const run = mapWithConcurrency([0], 1, () => Promise.reject("plain string"));
    await expect(run).rejects.toBe("plain string");
  });
});

describe("mapWithConcurrency: external abort", () => {
  it("rejects with JevAbortError without calling fn when the signal is already aborted", async () => {
    const controller = new AbortController();
    const reason = new Error("caller gave up");
    controller.abort(reason);
    const fn = vi.fn(async () => 1);

    const { error } = await outcome(mapWithConcurrency([1, 2], 2, fn, { signal: controller.signal }));
    expect(error).toBeInstanceOf(JevAbortError);
    expect((error as JevAbortError).cause).toBe(reason);
    expect(fn).not.toHaveBeenCalled();

    // Cancellation takes precedence over the empty-input fast path.
    await expect(mapWithConcurrency([], 2, fn, { signal: controller.signal })).rejects.toBeInstanceOf(
      JevAbortError,
    );
  });

  it("stops scheduling, aborts in-flight work, waits for it to settle, then rejects with JevAbortError", async () => {
    const controller = new AbortController();
    const reason = new Error("stop");
    const started: number[] = [];
    const settled: number[] = [];
    let innerSignal: AbortSignal | undefined;

    const run = mapWithConcurrency(
      [0, 1, 2, 3, 4],
      2,
      async (item, _index, signal) => {
        innerSignal = signal;
        started.push(item);
        try {
          await whenAborted(signal);
          await tick(10);
          throw new Error(`task ${item} saw abort`);
        } finally {
          settled.push(item);
        }
      },
      { signal: controller.signal },
    );
    let settledAtRejection: number[] | undefined;
    const result = outcome(
      run.catch((error: unknown) => {
        settledAtRejection = [...settled];
        throw error;
      }),
    );

    await tick();
    controller.abort(reason);
    expect(innerSignal?.aborted).toBe(true);
    expect(innerSignal?.reason).toBe(reason);

    const { error } = await result;
    expect(error).toBeInstanceOf(JevAbortError);
    expect((error as JevAbortError).cause).toBe(reason);
    expect(started).toEqual([0, 1]);
    expect([...(settledAtRejection ?? [])].sort()).toEqual([0, 1]);
  });

  it("rejects with JevAbortError even if every in-flight task still succeeds, and starts nothing new", async () => {
    const controller = new AbortController();
    const gate = deferred();
    const started: number[] = [];
    const run = mapWithConcurrency(
      [0, 1, 2, 3],
      2,
      async (item) => {
        started.push(item);
        await gate.promise;
        return item;
      },
      { signal: controller.signal },
    );
    await tick();
    controller.abort();
    gate.resolve();
    await expect(run).rejects.toBeInstanceOf(JevAbortError);
    expect(started).toEqual([0, 1]);
  });

  it("keeps the task error when a task failed before the caller aborted", async () => {
    const controller = new AbortController();
    const boom = new Error("boom");
    const run = mapWithConcurrency(
      [0, 1],
      2,
      async (item) => {
        if (item === 0) {
          throw boom;
        }
        await tick(20);
      },
      { signal: controller.signal },
    );
    await tick();
    controller.abort(); // arrives while item 1 is still settling
    await expect(run).rejects.toBe(boom);
  });

  it("keeps JevAbortError when the abort came first and tasks then fail with other errors", async () => {
    const controller = new AbortController();
    const run = mapWithConcurrency(
      [0, 1],
      2,
      async (_item, _index, signal) => {
        await whenAborted(signal);
        throw new Error("transport failed after abort");
      },
      { signal: controller.signal },
    );
    await tick();
    controller.abort();
    await expect(run).rejects.toBeInstanceOf(JevAbortError);
  });

  it("detaches its listener from the caller's signal on success and on failure", async () => {
    for (const shouldFail of [false, true]) {
      const controller = new AbortController();
      const add = vi.spyOn(controller.signal, "addEventListener");
      const remove = vi.spyOn(controller.signal, "removeEventListener");
      await outcome(
        mapWithConcurrency(
          [1, 2],
          2,
          async (x) => {
            if (shouldFail) {
              throw new Error("x");
            }
            return x;
          },
          { signal: controller.signal },
        ),
      );
      expect(add).toHaveBeenCalledTimes(1);
      expect(remove).toHaveBeenCalledTimes(1);
      expect(remove.mock.calls[0]?.[1]).toBe(add.mock.calls[0]?.[1]);
    }
  });

  it("resolves normally with a signal that never aborts", async () => {
    const controller = new AbortController();
    await expect(
      mapWithConcurrency([1, 2, 3], 2, async (x) => x + 1, { signal: controller.signal }),
    ).resolves.toEqual([2, 3, 4]);
  });
});
