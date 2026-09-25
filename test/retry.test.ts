import fc from "fast-check";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  JevAbortError,
  JevInfiniteCTXError,
  JevProviderError,
  JevResponseError,
  type ProviderErrorKind,
} from "../src/errors.js";
import { computeBackoffMs, sleep, withRetry, type RetryInfo } from "../src/retry.js";

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

function providerError(
  kind: ProviderErrorKind,
  extra: { retryAfterMs?: number; retryable?: boolean; status?: number } = {},
): JevProviderError {
  return new JevProviderError(`provider failed: ${kind}`, { kind, ...extra });
}

/** Injected sleep that records requested delays and resolves immediately. */
function fakeSleep() {
  return vi.fn(async (_ms: number, _signal?: AbortSignal): Promise<void> => {});
}

/** fn that throws each error in `errors` in turn, then resolves with `value`. */
function failThen<T>(errors: unknown[], value: T) {
  let call = 0;
  return vi.fn(async (_attempt: number): Promise<T> => {
    const error = errors[call++];
    if (call <= errors.length) {
      throw error;
    }
    return value;
  });
}

const tick = (ms = 0): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

describe("computeBackoffMs", () => {
  it("doubles from the base with equal jitter in [0.5, 1] of the ceiling", () => {
    const at = (r: number) => (attempt: number) => computeBackoffMs(attempt, 100, 10_000, () => r);
    expect([1, 2, 3, 4].map(at(0))).toEqual([50, 100, 200, 400]);
    expect([1, 2, 3, 4].map(at(0.5))).toEqual([75, 150, 300, 600]);
    expect([1, 2, 3, 4].map(at(1))).toEqual([100, 200, 400, 800]);
  });

  it("caps the ceiling at maxDelayMs before applying jitter", () => {
    // 500 · 2^5 = 16000 > 8000
    expect(computeBackoffMs(6, 500, 8000, () => 0.5)).toBe(6000);
    expect(computeBackoffMs(6, 500, 8000, () => 0)).toBe(4000);
    // maxDelayMs below the base caps even the first retry
    expect(computeBackoffMs(1, 500, 200, () => 1)).toBe(200);
  });

  it("stays finite for huge attempts and returns 0 for a zero base (no 0 · Infinity = NaN)", () => {
    expect(computeBackoffMs(5000, 500, 8000, () => 1)).toBe(8000);
    expect(computeBackoffMs(5000, 0, 8000, () => 1)).toBe(0);
    expect(computeBackoffMs(1, 0, 8000, () => 1)).toBe(0);
  });

  it("clamps an out-of-range random source into [0, 1]", () => {
    expect(computeBackoffMs(1, 100, 1000, () => 7)).toBe(100);
    expect(computeBackoffMs(1, 100, 1000, () => -3)).toBe(50);
  });

  it("property: within [ceiling/2, ceiling] and non-decreasing in the attempt", () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 80 }),
        fc.double({ min: 0, max: 10_000, noNaN: true }),
        fc.double({ min: 0, max: 100_000, noNaN: true }),
        fc.double({ min: 0, max: 1, maxExcluded: true, noNaN: true }),
        (attempt, base, max, r) => {
          const random = () => r;
          const ceiling = base > 0 ? Math.min(max, base * 2 ** (attempt - 1)) : 0;
          const delay = computeBackoffMs(attempt, base, max, random);
          expect(Number.isFinite(delay)).toBe(true);
          expect(delay).toBeGreaterThanOrEqual(ceiling / 2);
          expect(delay).toBeLessThanOrEqual(ceiling);
          expect(computeBackoffMs(attempt + 1, base, max, random)).toBeGreaterThanOrEqual(delay);
        },
      ),
    );
  });
});

describe("withRetry: success and retry schedule", () => {
  it("returns the value with attempts = 1 when the first attempt succeeds", async () => {
    const sleepFn = fakeSleep();
    const onRetry = vi.fn();
    const fn = vi.fn(async () => "ok");
    await expect(
      withRetry(fn, { retries: 3, baseDelayMs: 100, maxDelayMs: 1000, sleep: sleepFn, onRetry }),
    ).resolves.toEqual({ value: "ok", attempts: 1 });
    expect(fn).toHaveBeenCalledWith(1);
    expect(sleepFn).not.toHaveBeenCalled();
    expect(onRetry).not.toHaveBeenCalled();
  });

  it("retries retryable errors with exact exponential delays and reports each retry", async () => {
    const errors = [1, 2, 3, 4].map(() => providerError("rate_limit", { status: 429 }));
    const fn = failThen(errors, "done");
    const sleepFn = fakeSleep();
    const infos: RetryInfo[] = [];
    const signal = new AbortController().signal;

    const result = await withRetry(fn, {
      retries: 4,
      baseDelayMs: 100,
      maxDelayMs: 1000,
      signal,
      sleep: sleepFn,
      random: () => 0.5,
      onRetry: (info) => infos.push(info),
    });

    expect(result).toEqual({ value: "done", attempts: 5 });
    expect(fn.mock.calls.map(([attempt]) => attempt)).toEqual([1, 2, 3, 4, 5]);
    expect(sleepFn.mock.calls.map(([ms]) => ms)).toEqual([75, 150, 300, 600]);
    expect(sleepFn.mock.calls.every(([, s]) => s === signal)).toBe(true);
    expect(infos.map(({ attempt, delayMs }) => [attempt, delayMs])).toEqual([
      [1, 75],
      [2, 150],
      [3, 300],
      [4, 600],
    ]);
    infos.forEach((info, i) => expect(info.error).toBe(errors[i]));
  });

  it("calls onRetry before the corresponding sleep", async () => {
    const log: string[] = [];
    await withRetry(failThen([providerError("server")], 1), {
      retries: 1,
      baseDelayMs: 10,
      maxDelayMs: 10,
      random: () => 1,
      onRetry: ({ delayMs }) => log.push(`onRetry ${delayMs}`),
      sleep: async (ms) => {
        log.push(`sleep ${ms}`);
      },
    });
    expect(log).toEqual(["onRetry 10", "sleep 10"]);
  });

  it.each<ProviderErrorKind>(["rate_limit", "server", "overloaded", "timeout", "network"])(
    "retries the retryable kind %s",
    async (kind) => {
      const fn = failThen([providerError(kind)], "ok");
      const result = await withRetry(fn, { retries: 1, baseDelayMs: 0, maxDelayMs: 0, sleep: fakeSleep() });
      expect(result).toEqual({ value: "ok", attempts: 2 });
    },
  );

  it("follows the error's retryable flag rather than its kind", async () => {
    const overriddenOn = failThen([providerError("unknown", { retryable: true })], "ok");
    await expect(
      withRetry(overriddenOn, { retries: 1, baseDelayMs: 0, maxDelayMs: 0, sleep: fakeSleep() }),
    ).resolves.toEqual({ value: "ok", attempts: 2 });

    const overriddenOff = providerError("rate_limit", { retryable: false });
    const fn = failThen([overriddenOff], "never");
    await expect(
      withRetry(fn, { retries: 3, baseDelayMs: 0, maxDelayMs: 0, sleep: fakeSleep() }),
    ).rejects.toBe(overriddenOff);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("uses Math.random for jitter when no random source is injected", async () => {
    vi.spyOn(Math, "random").mockReturnValue(0);
    const sleepFn = fakeSleep();
    await withRetry(failThen([providerError("server")], 1), {
      retries: 1,
      baseDelayMs: 400,
      maxDelayMs: 8000,
      sleep: sleepFn,
    });
    expect(sleepFn).toHaveBeenCalledWith(200, undefined);
  });

  it("uses the real sleep by default", async () => {
    vi.useFakeTimers();
    const fn = failThen([providerError("server")], "late");
    const run = withRetry(fn, { retries: 1, baseDelayMs: 1000, maxDelayMs: 8000, random: () => 0 });
    await vi.advanceTimersByTimeAsync(499);
    expect(fn).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    await expect(run).resolves.toEqual({ value: "late", attempts: 2 });
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("property: succeeds iff failures <= retries, with attempts and sleeps accounted exactly", async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.integer({ min: 0, max: 5 }),
        fc.integer({ min: 0, max: 8 }),
        async (retries, failures) => {
          const errors = Array.from({ length: failures }, () => providerError("overloaded"));
          const fn = failThen(errors, "value");
          const sleepFn = fakeSleep();
          const run = withRetry(fn, { retries, baseDelayMs: 1, maxDelayMs: 1, sleep: sleepFn });
          if (failures <= retries) {
            await expect(run).resolves.toEqual({ value: "value", attempts: failures + 1 });
            expect(sleepFn).toHaveBeenCalledTimes(failures);
          } else {
            await expect(run).rejects.toBe(errors[retries]);
            expect(fn).toHaveBeenCalledTimes(retries + 1);
            expect(sleepFn).toHaveBeenCalledTimes(retries);
          }
        },
      ),
    );
  });
});

describe("withRetry: Retry-After precedence", () => {
  const run = async (retryAfterMs: number, maxDelayMs = 8000) => {
    const sleepFn = fakeSleep();
    const random = vi.fn(() => 0.5);
    await withRetry(failThen([providerError("rate_limit", { retryAfterMs })], 1), {
      retries: 1,
      baseDelayMs: 500,
      maxDelayMs,
      sleep: sleepFn,
      random,
    });
    return { delay: sleepFn.mock.calls[0]?.[0], random };
  };

  it("uses retryAfterMs exactly instead of computed backoff (no jitter)", async () => {
    const { delay, random } = await run(2500);
    expect(delay).toBe(2500);
    expect(random).not.toHaveBeenCalled();
  });

  it("honors a Retry-After shorter than the backoff, including 0", async () => {
    expect((await run(10)).delay).toBe(10);
    expect((await run(0)).delay).toBe(0);
  });

  it("caps retryAfterMs at max(maxDelayMs, 60 000)", async () => {
    expect((await run(120_000)).delay).toBe(60_000);
    expect((await run(120_000, 90_000)).delay).toBe(90_000);
    expect((await run(Infinity)).delay).toBe(60_000);
    expect((await run(45_000)).delay).toBe(45_000); // above maxDelayMs but under the floor cap
  });

  it("treats a negative retryAfterMs as 0 and a NaN one as absent", async () => {
    expect((await run(-5)).delay).toBe(0);
    expect((await run(Number.NaN)).delay).toBe(375); // 500 · 2^0 · (0.5 + 0.5 · 0.5)
  });

  it("applies per error: Retry-After on one attempt, backoff on the next", async () => {
    const sleepFn = fakeSleep();
    const errors = [providerError("rate_limit", { retryAfterMs: 1234 }), providerError("server")];
    await withRetry(failThen(errors, 1), {
      retries: 2,
      baseDelayMs: 100,
      maxDelayMs: 1000,
      sleep: sleepFn,
      random: () => 1,
    });
    expect(sleepFn.mock.calls.map(([ms]) => ms)).toEqual([1234, 200]);
  });
});

describe("withRetry: non-retryable errors pass through unchanged", () => {
  const cases: Array<[string, () => unknown]> = [
    ["context_limit", () => providerError("context_limit", { status: 413 })],
    ["context_limit marked retryable", () => providerError("context_limit", { retryable: true })],
    ["auth", () => providerError("auth", { status: 401 })],
    ["bad_request", () => providerError("bad_request", { status: 400 })],
    ["invalid_response", () => providerError("invalid_response")],
    ["JevResponseError", () => new JevResponseError("wrong answer type")],
    ["JevAbortError", () => new JevAbortError("aborted")],
    ["plain Error", () => new Error("bug")],
    ["TypeError", () => new TypeError("not a function")],
    ["non-Error value", () => "a string"],
  ];

  it.each(cases)("%s is rethrown immediately", async (_name, make) => {
    const error = make();
    const fn = failThen([error], "never");
    const sleepFn = fakeSleep();
    const onRetry = vi.fn();
    await expect(
      withRetry(fn, { retries: 5, baseDelayMs: 10, maxDelayMs: 100, sleep: sleepFn, onRetry }),
    ).rejects.toBe(error);
    expect(fn).toHaveBeenCalledTimes(1);
    expect(sleepFn).not.toHaveBeenCalled();
    expect(onRetry).not.toHaveBeenCalled();
  });

  it("stops at a non-retryable error that follows retryable ones", async () => {
    const fatal = providerError("context_limit");
    const fn = failThen([providerError("server"), providerError("rate_limit"), fatal], "never");
    const sleepFn = fakeSleep();
    await expect(
      withRetry(fn, { retries: 5, baseDelayMs: 1, maxDelayMs: 1, sleep: sleepFn }),
    ).rejects.toBe(fatal);
    expect(fn).toHaveBeenCalledTimes(3);
    expect(sleepFn).toHaveBeenCalledTimes(2);
  });
});

describe("withRetry: exhaustion", () => {
  it("rethrows the last error unchanged after retries + 1 attempts", async () => {
    const errors = [providerError("server"), providerError("overloaded"), providerError("network")];
    const fn = failThen(errors, "never");
    const onRetry = vi.fn();
    await expect(
      withRetry(fn, { retries: 2, baseDelayMs: 1, maxDelayMs: 1, sleep: fakeSleep(), onRetry }),
    ).rejects.toBe(errors[2]);
    expect(fn).toHaveBeenCalledTimes(3);
    expect(onRetry.mock.calls.map(([info]) => (info as RetryInfo).attempt)).toEqual([1, 2]);
  });

  it("makes exactly one attempt with retries = 0", async () => {
    const error = providerError("rate_limit");
    const fn = failThen([error], "never");
    await expect(
      withRetry(fn, { retries: 0, baseDelayMs: 1, maxDelayMs: 1, sleep: fakeSleep() }),
    ).rejects.toBe(error);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it.each([-1, Number.NaN])("rejects an invalid retries value %s", async (retries) => {
    const fn = vi.fn(async () => 1);
    await expect(withRetry(fn, { retries, baseDelayMs: 1, maxDelayMs: 1 })).rejects.toBeInstanceOf(
      JevInfiniteCTXError,
    );
    expect(fn).not.toHaveBeenCalled();
  });

  it("propagates an exception thrown by onRetry", async () => {
    const hookError = new Error("hook failed");
    const run = withRetry(failThen([providerError("server")], 1), {
      retries: 1,
      baseDelayMs: 1,
      maxDelayMs: 1,
      sleep: fakeSleep(),
      onRetry: () => {
        throw hookError;
      },
    });
    await expect(run).rejects.toBe(hookError);
  });
});

describe("withRetry: abort", () => {
  it("rejects with JevAbortError without calling fn when the signal is already aborted", async () => {
    const controller = new AbortController();
    const reason = new Error("cancelled");
    controller.abort(reason);
    const fn = vi.fn(async () => 1);
    const error: unknown = await withRetry(fn, {
      retries: 3,
      baseDelayMs: 1,
      maxDelayMs: 1,
      signal: controller.signal,
    }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(JevAbortError);
    expect((error as JevAbortError).cause).toBe(reason);
    expect(fn).not.toHaveBeenCalled();
  });

  it("rejects with JevAbortError when aborted during the backoff sleep (real timers)", async () => {
    const controller = new AbortController();
    const fn = failThen([providerError("rate_limit")], "never");
    const started = Date.now();
    const run = withRetry(fn, {
      retries: 3,
      baseDelayMs: 60_000, // would stall the test if the abort were ignored
      maxDelayMs: 60_000,
      signal: controller.signal,
      onRetry: () => {
        setTimeout(() => controller.abort(new Error("user cancelled")), 5);
      },
    });
    await expect(run).rejects.toBeInstanceOf(JevAbortError);
    expect(Date.now() - started).toBeLessThan(5_000);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("does not start another attempt when an injected sleep ignores the abort", async () => {
    const controller = new AbortController();
    const fn = failThen([providerError("server")], "never");
    const run = withRetry(fn, {
      retries: 3,
      baseDelayMs: 1,
      maxDelayMs: 1,
      signal: controller.signal,
      sleep: async () => {
        controller.abort();
      },
    });
    await expect(run).rejects.toBeInstanceOf(JevAbortError);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("does not report a retry when the signal aborted while the attempt was failing", async () => {
    const controller = new AbortController();
    const onRetry = vi.fn();
    const sleepFn = fakeSleep();
    const run = withRetry(
      async () => {
        controller.abort();
        throw providerError("network"); // a transport that ignored the abort
      },
      { retries: 3, baseDelayMs: 1, maxDelayMs: 1, signal: controller.signal, onRetry, sleep: sleepFn },
    );
    await expect(run).rejects.toBeInstanceOf(JevAbortError);
    expect(onRetry).not.toHaveBeenCalled();
    expect(sleepFn).not.toHaveBeenCalled();
  });

  it("rethrows a JevAbortError raised by fn itself unchanged", async () => {
    const controller = new AbortController();
    const transportAbort = new JevAbortError("request aborted");
    const run = withRetry(
      async () => {
        controller.abort();
        throw transportAbort;
      },
      { retries: 3, baseDelayMs: 1, maxDelayMs: 1, signal: controller.signal },
    );
    await expect(run).rejects.toBe(transportAbort);
  });
});

describe("sleep", () => {
  it("resolves after the requested delay", async () => {
    vi.useFakeTimers();
    let done = false;
    const run = sleep(1000).then(() => {
      done = true;
    });
    await vi.advanceTimersByTimeAsync(999);
    expect(done).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    await run;
    expect(done).toBe(true);
  });

  it.each([0, -50, Number.NaN])("resolves promptly for %s ms", async (ms) => {
    await expect(sleep(ms)).resolves.toBeUndefined();
  });

  it("rejects immediately with JevAbortError when the signal is already aborted", async () => {
    vi.useFakeTimers();
    const controller = new AbortController();
    const reason = new Error("gone");
    controller.abort(reason);
    const error: unknown = await sleep(1000, controller.signal).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(JevAbortError);
    expect((error as JevAbortError).cause).toBe(reason);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("rejects with JevAbortError on abort and clears its timer", async () => {
    vi.useFakeTimers();
    const controller = new AbortController();
    const run = sleep(10_000, controller.signal);
    expect(vi.getTimerCount()).toBe(1);
    controller.abort("stop");
    const error: unknown = await run.catch((e: unknown) => e);
    expect(error).toBeInstanceOf(JevAbortError);
    expect((error as JevAbortError).cause).toBe("stop");
    expect(vi.getTimerCount()).toBe(0);
  });

  it("detaches its abort listener after resolving", async () => {
    const controller = new AbortController();
    const add = vi.spyOn(controller.signal, "addEventListener");
    const remove = vi.spyOn(controller.signal, "removeEventListener");
    await sleep(1, controller.signal);
    expect(remove).toHaveBeenCalledTimes(1);
    expect(remove.mock.calls[0]?.[1]).toBe(add.mock.calls[0]?.[1]);
  });

  it("clamps delays beyond the timer limit instead of firing immediately (real timers)", async () => {
    const controller = new AbortController();
    let settled = false;
    const run = sleep(1e12, controller.signal).then(
      () => {
        settled = true;
      },
      () => {},
    );
    await tick(30);
    expect(settled).toBe(false);
    controller.abort();
    await run;
    expect(settled).toBe(false);
  });
});
